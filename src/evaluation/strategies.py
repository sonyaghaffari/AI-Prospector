"""Drilling-strategy comparison under equal budgets on the same frozen world.

Strategies: random, regular_grid, max_uncertainty, eivr_per_cost, multi_objective.
The first two are non-adaptive baselines; the latter three retrain the GPU ensemble after
every hole exactly like the interactive loop (same ``Experiment.step``), only driven
automatically instead of by a human choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model, validation_samples
from src.optimization.acquisition import regular_grid_order
from src.simulation.experiment import Experiment
from src.utils.config import ExperimentConfig

NON_ADAPTIVE = ("random", "regular_grid")


@dataclass
class StrategyRun:
    strategy: str
    exp_id: str
    iterations: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.iterations)


def run_strategy(base_exp_id: str, strategy: str, cfg: ExperimentConfig, n_holes: int, seed: int,
                 run_id: str) -> StrategyRun:
    """Run ``n_holes`` sequential iterations of ``strategy`` on the SAME frozen world
    (``base_exp_id``) via an independent ``Experiment`` instance / observed-data copy."""
    import shutil

    from src.simulation.world_builder import ExperimentPaths

    exp_id = f"{base_exp_id}__{run_id}"
    paths = ExperimentPaths(exp_id)
    base_paths = ExperimentPaths(base_exp_id)
    if not paths.observed.exists():
        shutil.copytree(base_paths.observed, paths.observed)
    exp = Experiment(cfg, exp_id)
    exp.truth = exp.truth.__class__(base_paths.hidden)  # same immutable hidden truth, read-only
    exp._attach_oracle()
    exp.oracle.log_path = paths.artifact("reports") / "drilling_log.json"
    exp.oracle.events = []

    rng = np.random.default_rng(seed)
    fixed_order: list[str] | None = None
    if strategy == "regular_grid":
        fixed_order = regular_grid_order(exp.obs.candidates(), exp.cfg.world.extent, n_holes)
    elif strategy == "random":
        ids = exp.obs.candidates().hole_id.to_numpy()
        fixed_order = list(rng.permutation(ids))

    val = validation_samples(exp.truth, exp.cfg.historical.observed_max_depth)
    run = StrategyRun(strategy, exp_id)
    for i in range(n_holes):
        if strategy in NON_ADAPTIVE:
            train_info = exp.train(warm_start=i > 0)
            cands = exp.obs.candidates()
            affordable = set(cands.loc[cands.cost_usd <= exp.budget_remaining, "hole_id"]) - exp.oracle.executed
            hid = next(h for h in fixed_order if h in affordable)
            result = exp.drill(hid, authorized=True)
            exp.record_iteration(result, train_info)
            metrics = evaluate_model(exp.model, val)
        else:
            out = exp.step(strategy=strategy, authorized=True)
            metrics = evaluate_model(exp.model, val)
            result = out["drill_result"]
        run.iterations.append({"iteration": i, "hole_id": result.hole_id, "cost_usd": result.cost_usd,
                               "cumulative_cost": exp.budget_spent, "deep_mineralized_m": result.summary()[
                                   "deep_mineralized_m"], **metrics})
        if exp.budget_remaining < cfg.candidates.mobilization_cost:
            break
    return run
