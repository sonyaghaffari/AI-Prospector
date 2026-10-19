"""SyntheticDrillingOracle: the only gate through which hidden drilling outcomes are revealed.

It never generates geology. Catalogue holes return outcomes that were logged (including
frozen assay noise) when the world was created. Off-catalogue holes are read from the frozen
truth volume by deterministic trilinear / nearest interpolation with no added randomness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.drilling.candidates import drilling_cost
from src.drilling.logging_ import lithology_contacts, log_hole, mineralized_intervals, structural_zones
from src.simulation.truth_store import HiddenTruthStore
from src.utils.config import CandidateConfig


class DrillingAuthorizationError(RuntimeError):
    pass


@dataclass
class DrillResult:
    hole_id: str
    collar: dict
    samples: pd.DataFrame
    mineralized_intervals: list[dict]
    contacts: list[dict]
    structures: list[dict]
    cost_usd: float
    metres: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> dict:
        deep = self.samples[self.samples.z > 100]
        spacing = float((self.samples.to_m - self.samples.from_m).iloc[0]) if len(self.samples) else 0.0
        return {"hole_id": self.hole_id, "metres": self.metres, "cost_usd": self.cost_usd,
                "n_intervals": len(self.mineralized_intervals),
                "mineralized_m": float(self.samples.mineralized.sum() * spacing),
                "deep_mineralized_m": float(deep.mineralized.sum() * spacing),
                "max_zn": float(self.samples.zn.max()) if len(self.samples) else 0.0,
                "timestamp": self.timestamp}


class SyntheticDrillingOracle:
    def __init__(self, truth: HiddenTruthStore, catalogue: pd.DataFrame, log_path: str | Path,
                 cost_cfg: CandidateConfig, cutoff: float, sample_spacing: float) -> None:
        self._truth = truth
        self.__outcomes = truth.table("candidate_outcomes")  # name-mangled: not part of the public API
        self.catalogue = catalogue.set_index("hole_id", drop=False)
        self.log_path = Path(log_path)
        self.cost_cfg = cost_cfg
        self.cutoff = cutoff
        self.sample_spacing = sample_spacing
        self.events: list[dict] = []
        if self.log_path.exists():
            with open(self.log_path, encoding="utf-8") as fh:
                self.events = json.load(fh)

    @property
    def executed(self) -> set[str]:
        return {e["hole_id"] for e in self.events}

    def execute(self, hole_id: str, authorized: bool = False, allow_repeat: bool = False) -> DrillResult:
        """Reveal the frozen outcome of a catalogue hole. ``authorized`` must be set explicitly
        by the caller (UI button / benchmark driver) — the oracle never drills on its own."""
        if not authorized:
            raise DrillingAuthorizationError("Drilling requires explicit authorization.")
        if hole_id not in self.catalogue.index:
            raise DrillingAuthorizationError(f"Unknown candidate hole: {hole_id}")
        if hole_id in self.executed and not allow_repeat:
            raise DrillingAuthorizationError(f"{hole_id} has already been drilled.")
        self._truth.verify()
        row = self.catalogue.loc[hole_id]
        samples = self.__outcomes[self.__outcomes.hole_id == hole_id].reset_index(drop=True).copy()
        return self._finish(hole_id, row.to_dict(), samples, float(row.cost_usd), float(row.length))

    def execute_custom(self, collar: np.ndarray, azimuth: float, dip: float, length: float,
                       authorized: bool = False) -> DrillResult:
        """Deterministic interpolation from the immutable truth volume (debug / what-if holes)."""
        if not authorized:
            raise DrillingAuthorizationError("Drilling requires explicit authorization.")
        self._truth.verify()
        world = self._truth.load_world()
        hid = f"CUSTOM-{len([e for e in self.events if e['hole_id'].startswith('CUSTOM')]) + 1:03d}"
        samples = log_hole(world, hid, np.asarray(collar, float), azimuth, dip, length, self.sample_spacing,
                           self.cutoff)
        collar_d = {"hole_id": hid, "x": collar[0], "y": collar[1], "z": collar[2], "azimuth": azimuth,
                    "dip": dip, "length": length}
        return self._finish(hid, collar_d, samples, drilling_cost(self.cost_cfg, azimuth, dip, length), length)

    def _finish(self, hole_id, collar, samples, cost, metres) -> DrillResult:
        res = DrillResult(hole_id=hole_id, collar={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                   for k, v in collar.items()},
                          samples=samples, mineralized_intervals=mineralized_intervals(samples),
                          contacts=lithology_contacts(samples), structures=structural_zones(samples),
                          cost_usd=cost, metres=metres)
        self.events.append(res.summary())
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(self.events, fh, indent=1)
        return res
