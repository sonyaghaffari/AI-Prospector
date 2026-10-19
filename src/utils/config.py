"""Typed experiment configuration.

Configs are plain JSON files in ``configs/``. Every field has a default so a
config file only needs to override what differs from the demo profile.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIOS = ("A", "B", "C", "D")
SCENARIO_DESCRIPTIONS = {
    "A": "Mineralization continues down-plunge into the deep zone",
    "B": "Mineralization changes direction at depth",
    "C": "Mineralization weakens and terminates before 200 m",
    "D": "Main shoot weakens; a separate concealed body exists at depth",
}


@dataclass
class WorldConfig:
    extent: tuple[float, float, float] = (1000.0, 1000.0, 200.0)  # X, Y, depth (m)
    shape: tuple[int, int, int] = (50, 50, 40)  # voxels nx, ny, nz
    scenario: str = "random"  # "A".."D" or "random" (drawn from the seed, kept hidden)
    seed: int = 20261019
    cutoff_zn_pb: float = 2.0  # Zn+Pb % defining "mineralized"
    collar_datum_masl: float = 2000.0  # synthetic flat topography elevation


@dataclass
class HistoricalConfig:
    n_train_holes: int = 85
    n_validation_holes: int = 15
    sample_spacing: float = 2.5  # m along hole
    observed_max_depth: float = 100.0  # vertical depth visible at t=0
    cluster_fraction: float = 0.7
    assay_noise_rel: float = 0.05  # frozen multiplicative assay noise (1 sigma)


@dataclass
class CandidateConfig:
    spacing: float = 100.0
    inclined_dip: float = 65.0
    min_separation: float = 40.0  # min distance between deep-segment midpoints
    max_length: float = 250.0
    boundary_margin: float = 30.0
    mobilization_cost: float = 20000.0  # USD per hole
    cost_per_m: float = 180.0  # USD/m at surface
    depth_cost_factor: float = 0.004  # +0.4 % per metre of depth
    inclined_premium: float = 0.25


@dataclass
class ModelConfig:
    ensemble_size: int = 5
    hidden: int = 128
    layers: int = 3
    fourier_features: int = 32
    fourier_sigma: float = 1.0
    lr: float = 3e-3
    weight_decay: float = 1e-4
    max_epochs: int = 400
    min_epochs: int = 60
    patience: int = 40
    batch_size: int = 512
    val_hole_fraction: float = 0.15
    amp: bool = False
    use_geophysics: bool = True
    lith_loss_weight: float = 0.5


@dataclass
class AcquisitionConfig:
    strategy: str = "multi_objective"
    target_points: int = 4000
    corr_range_h: float = 150.0  # used if variogram estimation fails / disabled
    corr_range_v: float = 40.0
    estimate_variogram: bool = True
    trajectory_step: float = 5.0
    w_eivr: float = 0.5
    w_mineral: float = 0.3
    w_hypothesis: float = 0.2


@dataclass
class HypothesisConfig:
    enabled: bool = True
    particles_per_hypothesis: int = 256
    simulated_outcomes: int = 128
    composite_length: float = 10.0
    background_prob: float = 0.03


@dataclass
class SimulationConfig:
    budget_usd: float = 800000.0
    inference_batch: int = 65536
    deterministic: bool = True
    retrain_mode: str = "retrain"  # "retrain" (from scratch) or "finetune"
    finetune_epochs: int = 120


@dataclass
class BenchmarkConfig:
    n_worlds_per_scenario: int = 1
    strategies: tuple[str, ...] = ("random", "regular_grid", "max_uncertainty", "eivr_per_cost", "multi_objective")
    n_holes: int = 5
    base_seed: int = 1000


@dataclass
class ExperimentConfig:
    profile: str = "demo"
    world: WorldConfig = field(default_factory=WorldConfig)
    historical: HistoricalConfig = field(default_factory=HistoricalConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    hypotheses: HypothesisConfig = field(default_factory=HypothesisConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_dict(self) -> dict[str, Any]:
        """Config with the hidden scenario redacted (safe for observed/ and the UI)."""
        d = copy.deepcopy(self.to_dict())
        d["world"]["scenario"] = "<hidden>"
        return d


def _build(cls, data: dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        sub = getattr(cls(), f.name)
        if is_dataclass(sub):
            kwargs[f.name] = _build(type(sub), val)
        elif isinstance(sub, tuple):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = val
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, data)


def load_config(path_or_profile: str | Path) -> ExperimentConfig:
    p = Path(path_or_profile)
    if not p.suffix:
        p = PROJECT_ROOT / "configs" / f"{path_or_profile}.json"
    with open(p, encoding="utf-8") as fh:
        return config_from_dict(json.load(fh))
