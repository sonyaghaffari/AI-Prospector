"""ObservationStore: everything the AI is legitimately allowed to know.

Contents of ``data/observed/<experiment>/``:
  samples.parquet        drill samples (historical shallow portions + virtually drilled holes)
  collars.parquet        collars of holes whose samples are in samples.parquet
  geophysics.npz         surface gravity / IP survey maps (2-D, measured at surface)
  surface_faults.json    mapped fault traces at surface
  candidates.parquet     public candidate geometry + cost (no outcomes)
  experiment.json        public config (scenario redacted)
  drilling_log.json      oracle event log

This module is source-agnostic: a real drill-hole database converted to the same schema
(see ``src/data/importers.py``) can be loaded identically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_SAMPLE_COLUMNS = ("hole_id", "x", "y", "z", "lithology", "zn", "pb", "fault_main", "fault_sec",
                           "mineralized")


class ObservationStore:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self._samples: pd.DataFrame | None = None

    # ---------------------------------------------------------------- creation
    @classmethod
    def initialise(cls, directory: str | Path, samples: pd.DataFrame, collars: pd.DataFrame,
                   gravity: np.ndarray, ip: np.ndarray, extent: tuple, fault_traces: dict[str, np.ndarray],
                   candidates: pd.DataFrame, public_config: dict) -> "ObservationStore":
        d = Path(directory)
        if (d / "samples.parquet").exists():
            raise FileExistsError(f"Observed data already initialised at {d}")
        d.mkdir(parents=True, exist_ok=True)
        missing = set(REQUIRED_SAMPLE_COLUMNS) - set(samples.columns)
        if missing:
            raise ValueError(f"samples missing columns {missing}")
        samples.assign(source="historical").to_parquet(d / "samples.parquet", index=False)
        collars.assign(source="historical").to_parquet(d / "collars.parquet", index=False)
        np.savez(d / "geophysics.npz", gravity=gravity, ip=ip, extent=np.asarray(extent, dtype=np.float64))
        with open(d / "surface_faults.json", "w", encoding="utf-8") as fh:
            json.dump({k: v.tolist() for k, v in fault_traces.items()}, fh)
        candidates.to_parquet(d / "candidates.parquet", index=False)
        with open(d / "experiment.json", "w", encoding="utf-8") as fh:
            json.dump(public_config, fh, indent=1)
        return cls(d)

    # ---------------------------------------------------------------- reads
    def samples(self) -> pd.DataFrame:
        if self._samples is None:
            self._samples = pd.read_parquet(self.dir / "samples.parquet")
        return self._samples

    def collars(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "collars.parquet")

    def candidates(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "candidates.parquet")

    def geophysics(self) -> dict[str, np.ndarray]:
        g = np.load(self.dir / "geophysics.npz")
        return {k: g[k] for k in g.files}

    def fault_traces(self) -> dict[str, np.ndarray]:
        with open(self.dir / "surface_faults.json", encoding="utf-8") as fh:
            return {k: np.asarray(v) for k, v in json.load(fh).items()}

    def public_config(self) -> dict:
        with open(self.dir / "experiment.json", encoding="utf-8") as fh:
            return json.load(fh)

    def drilled_ids(self) -> list[str]:
        c = self.collars()
        return c.loc[c.source == "drilled", "hole_id"].tolist()

    # ---------------------------------------------------------------- updates
    def add_drilled(self, samples: pd.DataFrame, collar: dict) -> None:
        """Append legitimately revealed observations (from the oracle) to the training database."""
        s = self.samples()
        if collar["hole_id"] in set(s.hole_id):
            raise ValueError(f"{collar['hole_id']} already in observed database")
        new = pd.concat([s, samples.assign(source="drilled")], ignore_index=True)
        new.to_parquet(self.dir / "samples.parquet", index=False)
        self._samples = new
        c = self.collars()
        crow = {k: collar.get(k) for k in ("hole_id", "x", "y", "z", "azimuth", "dip", "length")}
        c = pd.concat([c, pd.DataFrame([{**crow, "source": "drilled", "split": "drilled"}])], ignore_index=True)
        c.to_parquet(self.dir / "collars.parquet", index=False)
