"""Create a new experiment: generate the complete world ONCE, freeze it, and publish
only the legitimately observable subset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.observations import ObservationStore
from src.drilling.candidates import generate_candidates
from src.drilling.historical import generate_historical
from src.drilling.logging_ import log_hole
from src.geology.generator import generate_world
from src.simulation.truth_store import HiddenTruthStore
from src.utils.config import PROJECT_ROOT, ExperimentConfig


@dataclass
class ExperimentPaths:
    exp_id: str
    root: Path = PROJECT_ROOT

    @property
    def hidden(self) -> Path:
        return self.root / "data" / "hidden_truth" / self.exp_id

    @property
    def observed(self) -> Path:
        return self.root / "data" / "observed" / self.exp_id

    @property
    def world_record(self) -> Path:
        return self.root / "data" / "synthetic_worlds" / f"{self.exp_id}.json"

    def artifact(self, kind: str) -> Path:
        p = self.root / "artifacts" / kind / self.exp_id
        p.mkdir(parents=True, exist_ok=True)
        return p


def build_world(cfg: ExperimentConfig, paths: ExperimentPaths) -> dict:
    """Generate, freeze and store the hidden truth; initialise the observed database.

    Raises FileExistsError if this experiment already has a truth (never regenerated)."""
    truth_store = HiddenTruthStore(paths.hidden)
    if truth_store.exists:
        raise FileExistsError(f"Experiment {paths.exp_id} already exists; geological truth is never regenerated.")
    w = cfg.world
    world = generate_world(w).freeze()
    collars, full, surveys = generate_historical(world, cfg.historical, w.seed, w.cutoff_zn_pb, w.collar_datum_masl)
    cands = generate_candidates(world.grid, cfg.candidates)
    noise_rng = np.random.default_rng(w.seed + 303)
    outcomes = pd.concat([
        log_hole(world, r.hole_id, np.array([r.x, r.y, r.z]), r.azimuth, r.dip, r.length,
                 cfg.historical.sample_spacing, w.cutoff_zn_pb, noise_rng, cfg.historical.assay_noise_rel)
        for r in cands.itertuples()], ignore_index=True)

    full_cfg = cfg.to_dict()
    full_cfg["world"]["scenario"] = world.meta["scenario"]
    manifest = truth_store.write(world, full_cfg, collars, full, outcomes)

    # ---- the observable subset: shallow portions of training holes + surface surveys -------
    train_ids = set(collars.loc[collars.split == "train", "hole_id"])
    shallow = full[full.hole_id.isin(train_ids) & (full.z <= cfg.historical.observed_max_depth)]
    ObservationStore.initialise(
        paths.observed, samples=shallow.reset_index(drop=True),
        collars=collars[collars.split == "train"].reset_index(drop=True),
        gravity=world.gravity_map, ip=world.ip_map, extent=w.extent, fault_traces=world.fault_traces,
        candidates=cands, public_config=cfg.public_dict(),
    )
    paths.world_record.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(paths.world_record, "w", encoding="utf-8") as fh:
        json.dump({"exp_id": paths.exp_id, "seed": w.seed, "config": cfg.public_dict(),
                   "truth_manifest_sha256": manifest["files"]}, fh, indent=1)
    surveys.to_csv(paths.observed / "surveys_historical.csv", index=False)
    return {"n_historical": len(collars), "n_train_samples": len(shallow), "n_candidates": len(cands),
            "manifest": manifest}
