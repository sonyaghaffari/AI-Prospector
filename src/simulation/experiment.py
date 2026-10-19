"""Experiment: the sequential learning loop orchestrator.

INITIALIZE -> FREEZE TRUTH -> REVEAL SHALLOW DATA -> TRAIN -> PREDICT -> RANK CANDIDATES
-> (user selects a hole) -> ORACLE REVEALS OBSERVATIONS -> UPDATE OBSERVED DATA -> RETRAIN
-> UPDATE PREDICTIONS -> RANK AGAIN -> repeat until budget exhausted.

This class is the only place that wires the AI model, the acquisition engine and the
oracle together. It never imports anything that lets prediction/acquisition code reach
raw hidden-truth arrays; the oracle is the sole channel, and only via ``drill()`` with
``authorized=True`` from the caller (UI button, benchmark driver, or Phase-7 script).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.observations import ObservationStore
from src.drilling.candidates import candidate_points
from src.drilling.oracle import DrillResult, SyntheticDrillingOracle
from src.models.ensemble import GeoModel
from src.optimization.acquisition import score_candidates
from src.optimization.hypotheses import HypothesisEngine
from src.simulation.truth_store import HiddenTruthStore
from src.simulation.world_builder import ExperimentPaths, build_world
from src.uncertainty.variogram import estimate_ranges
from src.utils.config import ExperimentConfig
from src.utils.gpu import PerfLog, configure_reproducibility, inspect_gpu, select_device


@dataclass
class IterationRecord:
    index: int
    n_samples: int
    n_holes: int
    train_seconds: float
    drilled_hole_id: str | None = None
    drill_summary: dict | None = None
    budget_remaining: float = 0.0
    timestamp: float = field(default_factory=time.time)


class Experiment:
    """Live experiment state: model + observed data + candidate rankings + budget."""

    def __init__(self, cfg: ExperimentConfig, exp_id: str, allow_cpu: bool | None = None) -> None:
        self.cfg = cfg
        self.exp_id = exp_id
        self.paths = ExperimentPaths(exp_id)
        self.gpu_info = inspect_gpu()
        self.device = select_device(allow_cpu)
        configure_reproducibility(cfg.world.seed, cfg.simulation.deterministic)
        self.perf = PerfLog()
        self.obs = ObservationStore(self.paths.observed)
        self.truth = HiddenTruthStore(self.paths.hidden)
        self.oracle: SyntheticDrillingOracle | None = None
        self.model: GeoModel | None = None
        self.hyp_engine: HypothesisEngine | None = None
        self.rankings: pd.DataFrame | None = None
        self.history: list[IterationRecord] = []
        self.budget_spent = 0.0
        self._ranges: tuple[float, float] | None = None

    # ------------------------------------------------------------------ setup
    @classmethod
    def create(cls, cfg: ExperimentConfig, exp_id: str, allow_cpu: bool | None = None) -> "Experiment":
        exp = cls(cfg, exp_id, allow_cpu)
        with exp.perf.track("world_generation"):
            build_world(cfg, exp.paths)
        exp._attach_oracle()
        return exp

    @classmethod
    def load(cls, exp_id: str, allow_cpu: bool | None = None) -> "Experiment":
        paths = ExperimentPaths(exp_id)
        with open(paths.world_record, encoding="utf-8") as fh:
            rec = json.load(fh)
        from src.utils.config import config_from_dict

        full_cfg = config_from_dict(rec["config"])
        full_cfg.world.scenario = "random"  # never read the hidden value back into the live config
        exp = cls(full_cfg, exp_id, allow_cpu)
        exp._attach_oracle()
        ckpt = exp.paths.artifact("checkpoints") / "model.pt"
        if ckpt.exists():
            exp.model = GeoModel.load(ckpt, exp.device)
        hist_path = exp.paths.artifact("reports") / "history.json"
        if hist_path.exists():
            with open(hist_path, encoding="utf-8") as fh:
                exp.history = [IterationRecord(**r) for r in json.load(fh)]
            exp.budget_spent = sum(h.drill_summary["cost_usd"] for h in exp.history if h.drill_summary)
        return exp

    def _attach_oracle(self) -> None:
        self.oracle = SyntheticDrillingOracle(
            self.truth, self.obs.candidates(), self.paths.artifact("reports") / "drilling_log.json",
            self.cfg.candidates, self.cfg.world.cutoff_zn_pb, self.cfg.historical.sample_spacing,
        )

    @property
    def budget_remaining(self) -> float:
        return self.cfg.simulation.budget_usd - self.budget_spent

    # ------------------------------------------------------------------ train / predict / rank
    def train(self, warm_start: bool = False, progress=None) -> dict:
        samples = self.obs.samples()
        if self.model is None:
            self.model = GeoModel(self._extent(), self.cfg.model, self.device, self.cfg.world.seed)
        epochs = self.cfg.simulation.finetune_epochs if (warm_start and self.cfg.simulation.retrain_mode ==
                                                         "finetune") else None
        with self.perf.track("training"):
            res = self.model.fit(samples, self.obs.fault_traces(), self.obs.geophysics(), epochs=epochs,
                                 warm_start=warm_start and self.cfg.simulation.retrain_mode == "finetune",
                                 progress=progress)
        self._fit_hypotheses(samples)
        self.model.save(self.paths.artifact("checkpoints") / "model.pt")
        return {"seconds": res.seconds, "n_samples": res.n_samples, "n_holes": res.n_holes,
                "final_val_loss": res.final_val_loss, "history": res.history, "best_epoch": res.best_epoch}

    def _extent(self) -> tuple[float, float, float]:
        return tuple(json.loads(self.paths.world_record.read_text())["config"]["world"]["extent"]) \
            if self.paths.world_record.exists() else self.cfg.world.extent

    def _fit_hypotheses(self, samples: pd.DataFrame) -> None:
        if not self.cfg.hypotheses.enabled:
            self.hyp_engine = None
            return
        strike = self.model.features.planes["main"].strike_vector()
        self.hyp_engine = HypothesisEngine(self.cfg.hypotheses, self.device, self.cfg.world.seed).fit(samples, strike)

    def _variogram_ranges(self, samples: pd.DataFrame) -> tuple[float, float]:
        if not self.cfg.acquisition.estimate_variogram:
            return self.cfg.acquisition.corr_range_h, self.cfg.acquisition.corr_range_v
        with self.perf.track("variogram_estimation"):
            est = estimate_ranges(samples, self.device, self.cfg.acquisition.target_points, self.cfg.world.seed)
        rh = est["range_h"] or self.cfg.acquisition.corr_range_h
        rv = est["range_v"] or self.cfg.acquisition.corr_range_v
        self._last_variogram = est
        return rh, rv

    def rank_candidates(self, strategy: str | None = None) -> pd.DataFrame:
        assert self.model is not None, "call train() first"
        samples = self.obs.samples()
        cands = self.obs.candidates()
        affordable = cands[cands.cost_usd <= self.budget_remaining]
        ranges = self._ranges or self._variogram_ranges(samples)
        self._ranges = ranges
        extent = self._extent()
        rng = np.random.default_rng(self.cfg.world.seed)
        n_t = min(self.cfg.acquisition.target_points, 20000)
        targets = rng.uniform([0, 0, 100.0], [extent[0], extent[1], extent[2]], size=(n_t, 3)).astype(np.float32)
        cfg = self.cfg.acquisition
        if strategy:
            cfg = type(cfg)(**{**cfg.__dict__, "strategy": strategy})
        pts, mask = candidate_points(affordable, cfg.trajectory_step)
        with self.perf.track("candidate_scoring"):
            self.rankings = score_candidates(self.model, samples, affordable, pts, mask, targets, ranges, cfg,
                                             self.hyp_engine, self.oracle.executed, self.cfg.historical.observed_max_depth,
                                             cfg.trajectory_step)
        return self.rankings

    def hypothesis_posterior(self) -> dict:
        if self.hyp_engine is None:
            return {"available": False, "reason": "disabled"}
        return self.hyp_engine.posterior(self.obs.samples())

    # ------------------------------------------------------------------ drilling
    def drill(self, hole_id: str, authorized: bool) -> DrillResult:
        if not authorized:
            raise PermissionError("Virtual drilling requires explicit user authorization.")
        result = self.oracle.execute(hole_id, authorized=True)
        self.obs.add_drilled(result.samples, result.collar)
        self.budget_spent += result.cost_usd
        return result

    def record_iteration(self, drill_result: DrillResult | None, train_info: dict) -> IterationRecord:
        rec = IterationRecord(index=len(self.history), n_samples=len(self.obs.samples()),
                              n_holes=self.obs.samples().hole_id.nunique(), train_seconds=train_info["seconds"],
                              drilled_hole_id=drill_result.hole_id if drill_result else None,
                              drill_summary=drill_result.summary() if drill_result else None,
                              budget_remaining=self.budget_remaining)
        self.history.append(rec)
        p = self.paths.artifact("reports") / "history.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump([r.__dict__ for r in self.history], fh, indent=1)
        return rec

    # ------------------------------------------------------------------ full automated step
    def step(self, strategy: str | None = None, hole_id: str | None = None, authorized: bool = False) -> dict:
        """One loop iteration: (re)train if needed, rank, optionally drill the chosen/top hole."""
        warm = len(self.history) > 0
        train_info = self.train(warm_start=warm)
        ranking = self.rank_candidates(strategy)
        result = None
        if authorized and len(ranking):
            hid = hole_id or ranking.iloc[0].hole_id
            row = ranking.loc[ranking.hole_id == hid]
            if row.empty:
                raise ValueError(f"{hid} is not a valid, affordable, unexecuted candidate.")
            if float(row.iloc[0].cost_usd) > self.budget_remaining:
                raise ValueError("Selected hole exceeds remaining budget.")
            result = self.drill(hid, authorized=True)
        rec = self.record_iteration(result, train_info)
        return {"train": train_info, "ranking": ranking, "drill_result": result, "iteration": rec}
