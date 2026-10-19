"""Model input features, derived ONLY from observed data.

Features per point:
  x, y, z normalised to [-1, 1]
  signed distance to the *interpreted* main and secondary fault planes (+ proximity kernels)
  surface gravity and IP survey values at (x, y)  [optional, ``use_geophysics``]

Interpreted fault planes are least-squares fits to the mapped surface trace plus fault-zone
intersections logged in revealed drill holes. The true faults are listric / undulating, so
the interpretation diverges from truth at depth until deeper holes intersect them.

Note: surface geophysics is a legitimate observation that integrates signal from all
depths (as in real exploration). Disable it with ``use_geophysics=False`` for a strict
"shallow drilling only" ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import map_coordinates


@dataclass
class FaultPlane:
    normal: np.ndarray
    centroid: np.ndarray
    n_drill_points: int

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.centroid) @ self.normal

    def strike_vector(self) -> np.ndarray:
        h = np.array([-self.normal[1], self.normal[0]])
        return h / (np.linalg.norm(h) + 1e-12)


def fit_fault_plane(trace_xy: np.ndarray, intersections: np.ndarray) -> FaultPlane:
    """Weighted plane fit. With < 3 drill intersections the dip is unconstrained, so a
    vertical plane through the surface trace is assumed (documented prior)."""
    trace = np.column_stack([trace_xy, np.zeros(len(trace_xy))])
    if len(intersections) < 3 or np.ptp(intersections[:, 2]) < 15:
        d = trace[-1, :2] - trace[0, :2]
        d /= np.linalg.norm(d) + 1e-12
        normal = np.array([d[1], -d[0], 0.0])
        return FaultPlane(normal, trace.mean(0), len(intersections))
    pts = np.vstack([trace, intersections])
    w = np.concatenate([np.full(len(trace), 0.5 / len(trace)), np.full(len(intersections), 0.5 / len(intersections))])
    c = (w[:, None] * pts).sum(0) / w.sum()
    _, _, vt = np.linalg.svd((pts - c) * np.sqrt(w)[:, None], full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    if abs(n[2]) > 0.9:  # degenerate (near-horizontal) fit — fall back to vertical
        return fit_fault_plane(trace_xy, np.empty((0, 3)))
    return FaultPlane(n / np.linalg.norm(n), c, len(intersections))


def fault_intersections(samples: pd.DataFrame, col: str) -> np.ndarray:
    hit = samples[samples[col].astype(bool)]
    if hit.empty:
        return np.empty((0, 3))
    return hit.groupby("hole_id")[["x", "y", "z"]].mean().to_numpy()


class FeatureBuilder:
    def __init__(self, extent: tuple[float, float, float], use_geophysics: bool = True) -> None:
        self.extent = np.asarray(extent, dtype=np.float64)
        self.use_geophysics = use_geophysics
        self.planes: dict[str, FaultPlane] = {}
        self.geo: dict[str, np.ndarray] = {}
        self.geo_stats: dict[str, tuple[float, float]] = {}

    @property
    def n_features(self) -> int:
        return 7 + (2 if self.use_geophysics else 0)

    def fit(self, samples: pd.DataFrame, fault_traces: dict[str, np.ndarray],
            geophysics: dict[str, np.ndarray]) -> "FeatureBuilder":
        self.planes = {
            "main": fit_fault_plane(fault_traces["main"], fault_intersections(samples, "fault_main")),
            "secondary": fit_fault_plane(fault_traces["secondary"], fault_intersections(samples, "fault_sec")),
        }
        if self.use_geophysics:
            self.geo = {"gravity": geophysics["gravity"], "ip": geophysics["ip"]}
            self.geo_stats = {k: (float(v.mean()), float(v.std() + 1e-9)) for k, v in self.geo.items()}
        return self

    def _geo_at(self, name: str, pts: np.ndarray) -> np.ndarray:
        m = self.geo[name]
        sp = self.extent[:2] / np.array(m.shape)
        idx = (pts[:, :2] / sp - 0.5).T
        mu, sd = self.geo_stats[name]
        return (map_coordinates(m.astype(np.float64), idx, order=1, mode="nearest") - mu) / sd

    def transform_np(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        coords = 2 * pts / self.extent - 1
        cols = [coords]
        for name in ("main", "secondary"):
            d = self.planes[name].signed_distance(pts)
            cols.append(np.clip(d / 100.0, -3, 3)[:, None])
            cols.append(np.exp(-np.abs(d) / 50.0)[:, None])
        if self.use_geophysics:
            cols += [self._geo_at("gravity", pts)[:, None], self._geo_at("ip", pts)[:, None]]
        return np.concatenate(cols, axis=1).astype(np.float32)

    def transform(self, pts: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.transform_np(pts)).to(device, non_blocking=True)

    def state(self) -> dict:
        return {"extent": self.extent.tolist(), "use_geophysics": self.use_geophysics,
                "planes": {k: {"normal": p.normal.tolist(), "centroid": p.centroid.tolist(),
                               "n_drill_points": p.n_drill_points} for k, p in self.planes.items()}}
