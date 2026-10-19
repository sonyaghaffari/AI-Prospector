"""Fixed catalogue of candidate exploration holes targeting the 100-200 m interval.

Geometry and cost are public (the optimiser may see them). Outcomes are logged from the
frozen truth at creation time and stored only under data/hidden_truth/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.geology.grid import Grid
from src.geology.trajectory import direction, trajectory_points
from src.utils.config import CandidateConfig


def drilling_cost(cfg: CandidateConfig, azimuth: float, dip: float, length: float, step: float = 1.0) -> float:
    pts, _ = trajectory_points(np.zeros(3), azimuth, dip, length, step)
    per_m = cfg.cost_per_m * (1 + cfg.depth_cost_factor * pts[:, 2])
    premium = 1 + (cfg.inclined_premium if dip < 89.9 else 0.0)
    return float(cfg.mobilization_cost + per_m.sum() * step * premium)


def generate_candidates(grid: Grid, cfg: CandidateConfig, target_depth: float = 200.0) -> pd.DataFrame:
    ext = np.array(grid.extent)
    xs = np.arange(cfg.spacing, ext[0] - cfg.spacing / 2, cfg.spacing)
    ys = np.arange(cfg.spacing, ext[1] - cfg.spacing / 2, cfg.spacing)
    azimuths = (45.0, 135.0, 225.0, 315.0)
    proposals = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            collar = np.array([x, y, 0.0])
            proposals.append((collar, 0.0, 90.0))
            for k in range(4):
                az = azimuths[(i + j + k) % 4]
                if _valid(collar, az, cfg.inclined_dip, grid, cfg, target_depth):
                    proposals.append((collar, az, cfg.inclined_dip))
                    break
    rows, mids = [], []
    for collar, az, dip in proposals:
        length = (target_depth - 1.0) / np.sin(np.radians(dip))
        if length > cfg.max_length or not _valid(collar, az, dip, grid, cfg, target_depth):
            continue
        mid = collar + (150.0 / np.sin(np.radians(dip))) * direction(az, dip)
        if mids and np.min(np.linalg.norm(np.array(mids) - mid, axis=1)) < cfg.min_separation:
            continue  # redundant with an accepted candidate
        mids.append(mid)
        rows.append({"hole_id": f"CAND-{len(rows) + 1:03d}", "x": collar[0], "y": collar[1], "z": collar[2],
                     "azimuth": az, "dip": dip, "length": float(length),
                     "type": "vertical" if dip >= 89.9 else "inclined",
                     "cost_usd": drilling_cost(cfg, az, dip, length)})
    return pd.DataFrame(rows)


def _valid(collar, az, dip, grid: Grid, cfg: CandidateConfig, target_depth: float) -> bool:
    length = (target_depth - 1.0) / np.sin(np.radians(dip))
    end = collar + length * direction(az, dip)
    m = cfg.boundary_margin
    ok_xy = lambda p: m <= p[0] <= grid.extent[0] - m and m <= p[1] <= grid.extent[1] - m  # noqa: E731
    return ok_xy(collar) and ok_xy(end) and length <= cfg.max_length


def candidate_points(cands: pd.DataFrame, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Padded trajectory points (C, P, 3) and validity mask (C, P) for batched GPU scoring."""
    trajs = [trajectory_points(np.array([r.x, r.y, r.z]), r.azimuth, r.dip, r.length, step)[0]
             for r in cands.itertuples()]
    p = max(len(t) for t in trajs)
    pts = np.zeros((len(trajs), p, 3), dtype=np.float32)
    mask = np.zeros((len(trajs), p), dtype=bool)
    for i, t in enumerate(trajs):
        pts[i, :len(t)] = t
        mask[i, :len(t)] = True
    return pts, mask
