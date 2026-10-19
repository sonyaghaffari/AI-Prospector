"""Historical drilling database (generated once at world creation).

Holes are clustered around the known near-surface prospect (as real historical drilling is)
plus scattered regional holes. Full logs are stored as hidden truth; only samples at
vertical depth <= ``observed_max_depth`` from *training* holes enter the observed database.
Validation holes are drilled to 200 m and held out permanently (evaluation only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.drilling.logging_ import log_hole
from src.geology.generator import WorldTruth
from src.geology.trajectory import direction, survey_table
from src.utils.config import HistoricalConfig


def _plan_hole(rng: np.random.Generator, world: WorldTruth, clustered: bool, target_xy: np.ndarray,
               depth: float) -> tuple[np.ndarray, float, float, float]:
    ext = np.array(world.grid.extent[:2])
    for _ in range(200):
        vertical = rng.random() < 0.55
        dip = 90.0 if vertical else float(rng.uniform(55, 75))
        az = float(rng.uniform(0, 360)) if not vertical else 0.0
        if clustered:
            tgt = target_xy + rng.normal(0, 90, 2)
        else:
            tgt = rng.uniform(50, ext - 50)
        d = direction(az, dip)
        hz = 60.0 / np.tan(np.radians(dip)) if not vertical else 0.0
        collar_xy = tgt - hz * d[:2] / (np.linalg.norm(d[:2]) + 1e-12)
        length = depth / np.sin(np.radians(dip))
        end = collar_xy + length * d[:2]
        if np.all((collar_xy > 20) & (collar_xy < ext - 20) & (end > 0) & (end < ext)):
            return np.array([*collar_xy, 0.0]), az, dip, float(length)
    raise RuntimeError("could not place a historical hole")


def generate_historical(world: WorldTruth, cfg: HistoricalConfig, seed: int, cutoff: float,
                        datum: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (collars, full_samples, surveys). ``collars.split`` is 'train' or 'validation'."""
    rng = np.random.default_rng(seed + 101)
    noise_rng = np.random.default_rng(seed + 202)
    shoot = np.array(world.meta["shoot"]["axis"])
    z = world.grid.z
    target = shoot[int(np.argmin(np.abs(z - 40)))]
    # second "known prospect": the strongest shallow potential away from the main shoot
    shallow = world.potential[:, :, z < 60].max(axis=2)
    gx, gy = np.meshgrid(world.grid.x, world.grid.y, indexing="ij")
    far = np.hypot(gx - target[0], gy - target[1]) > 250
    i2 = np.unravel_index(np.argmax(np.where(far, shallow, -1)), shallow.shape)
    target2 = np.array([world.grid.x[i2[0]], world.grid.y[i2[1]]])

    collars, logs, surveys = [], [], []
    n_total = cfg.n_train_holes + cfg.n_validation_holes
    split = np.array(["train"] * cfg.n_train_holes + ["validation"] * cfg.n_validation_holes)
    split = split[rng.permutation(n_total)]
    for i in range(n_total):
        hid = f"HIST-{i + 1:03d}"
        r = rng.random()
        clustered = r < cfg.cluster_fraction
        tgt = target if rng.random() < 0.75 else target2
        if split[i] == "validation":
            depth = 199.0
        else:
            depth = float(rng.uniform(100, 160))  # deep portions exist but stay hidden
        collar, az, dip, length = _plan_hole(rng, world, clustered, tgt, depth)
        log = log_hole(world, hid, collar, az, dip, length, cfg.sample_spacing, cutoff, noise_rng,
                       cfg.assay_noise_rel)
        collars.append({"hole_id": hid, "x": collar[0], "y": collar[1], "z": collar[2],
                        "elevation_masl": datum - collar[2], "azimuth": az, "dip": dip, "length": length,
                        "split": split[i], "year": int(rng.integers(1975, 2015))})
        logs.append(log)
        surveys.extend(survey_table(hid, az, dip, length))
    return pd.DataFrame(collars), pd.concat(logs, ignore_index=True), pd.DataFrame(surveys)
