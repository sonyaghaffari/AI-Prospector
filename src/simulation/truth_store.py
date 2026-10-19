"""Write-once storage of the hidden geological truth with SHA-256 integrity manifest.

Only three components may read from here: the world builder (writes once), the
SyntheticDrillingOracle, and the evaluation module. Model, uncertainty and optimisation
code never import this module (enforced by tests/test_leakage.py).
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd

from src.geology.generator import WorldTruth
from src.geology.grid import Grid
from src.utils.hashing import array_sha256, file_sha256

TRUTH_FILES = ("truth.npz", "world_meta.json", "historical_full.parquet", "historical_collars.parquet",
               "candidate_outcomes.parquet")


class TruthIntegrityError(RuntimeError):
    pass


def _make_readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


class HiddenTruthStore:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self._world: WorldTruth | None = None

    @property
    def exists(self) -> bool:
        return (self.dir / "manifest.json").exists()

    # ------------------------------------------------------------------ write once
    def write(self, world: WorldTruth, config: dict, historical_collars: pd.DataFrame,
              historical_full: pd.DataFrame, candidate_outcomes: pd.DataFrame) -> dict:
        if self.exists:
            raise FileExistsError(f"Hidden truth already exists at {self.dir}; truth is generated only once.")
        self.dir.mkdir(parents=True, exist_ok=True)
        arrays = world.arrays()
        np.savez_compressed(
            self.dir / "truth.npz", **arrays, gravity_map=world.gravity_map, ip_map=world.ip_map,
            **{f"trace_{k}": v for k, v in world.fault_traces.items()},
        )
        with open(self.dir / "world_meta.json", "w", encoding="utf-8") as fh:
            json.dump({"meta": world.meta, "config": config,
                       "grid": {"extent": list(world.grid.extent), "shape": list(world.grid.shape)}}, fh, indent=1)
        historical_full.to_parquet(self.dir / "historical_full.parquet", index=False)
        historical_collars.to_parquet(self.dir / "historical_collars.parquet", index=False)
        candidate_outcomes.to_parquet(self.dir / "candidate_outcomes.parquet", index=False)
        manifest = {
            "files": {f: file_sha256(self.dir / f) for f in TRUTH_FILES},
            "arrays": {k: array_sha256(v) for k, v in arrays.items()},
        }
        with open(self.dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        for f in (*TRUTH_FILES, "manifest.json"):
            _make_readonly(self.dir / f)
        return manifest

    # ------------------------------------------------------------------ read
    @property
    def manifest(self) -> dict:
        with open(self.dir / "manifest.json", encoding="utf-8") as fh:
            return json.load(fh)

    def meta(self) -> dict:
        with open(self.dir / "world_meta.json", encoding="utf-8") as fh:
            return json.load(fh)

    def load_world(self) -> WorldTruth:
        if self._world is None:
            self.verify_files()
            m = self.meta()
            data = np.load(self.dir / "truth.npz")
            grid = Grid(tuple(m["grid"]["extent"]), tuple(m["grid"]["shape"]))
            traces = {k[6:]: data[k] for k in data.files if k.startswith("trace_")}
            self._world = WorldTruth(grid=grid, **{k: data[k] for k in WorldTruth.ARRAY_FIELDS},
                                     gravity_map=data["gravity_map"], ip_map=data["ip_map"],
                                     fault_traces=traces, meta=m["meta"]).freeze()
            self.verify_arrays(self._world)
        return self._world

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.dir / f"{name}.parquet")

    # ------------------------------------------------------------------ integrity
    def verify_files(self) -> None:
        for f, h in self.manifest["files"].items():
            if file_sha256(self.dir / f) != h:
                raise TruthIntegrityError(f"Hidden truth file modified: {f}")

    def verify_arrays(self, world: WorldTruth) -> None:
        for k, h in self.manifest["arrays"].items():
            if array_sha256(getattr(world, k)) != h:
                raise TruthIntegrityError(f"In-memory hidden truth array modified: {k}")

    def verify(self) -> bool:
        self.verify_files()
        if self._world is not None:
            self.verify_arrays(self._world)
        return True
