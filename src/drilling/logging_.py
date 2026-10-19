"""Build drill-hole sample logs from the frozen truth. Used ONLY at world creation time
(historical / validation / candidate outcomes) and by the oracle for off-catalog holes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.geology.generator import WorldTruth
from src.geology.grid import LITHOLOGY
from src.geology.trajectory import trajectory_points

SAMPLE_COLUMNS = ["hole_id", "sample_idx", "from_m", "to_m", "depth_along", "x", "y", "z", "lithology",
                  "lith_name", "zn", "pb", "density", "fault_main", "fault_sec", "breccia", "mineralized"]


def log_hole(world: WorldTruth, hole_id: str, collar: np.ndarray, azimuth: float, dip: float, length: float,
             spacing: float, cutoff: float, noise_rng: np.random.Generator | None = None,
             assay_noise_rel: float = 0.0) -> pd.DataFrame:
    """Read frozen truth along a trajectory. ``noise_rng`` is only passed at world creation,
    so assay noise is drawn once and frozen with the outcome; the oracle never adds noise."""
    pts, s = trajectory_points(collar, azimuth, dip, length, spacing)
    inside = world.grid.contains(pts)
    pts, s = pts[inside], s[inside]
    if len(pts) == 0:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    v = world.sample(pts)
    zn, pb, rho = v["zn"], v["pb"], v["density"]
    if noise_rng is not None and assay_noise_rel > 0:
        zn = zn * np.exp(noise_rng.normal(0, assay_noise_rel, len(zn)))
        pb = pb * np.exp(noise_rng.normal(0, assay_noise_rel, len(pb)))
        rho = rho + noise_rng.normal(0, 0.01, len(rho))
    return pd.DataFrame({
        "hole_id": hole_id,
        "sample_idx": np.arange(len(s)),
        "from_m": s - spacing / 2,
        "to_m": s + spacing / 2,
        "depth_along": s,
        "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
        "lithology": v["lithology"],
        "lith_name": [LITHOLOGY[i] for i in v["lithology"]],
        "zn": zn.astype(np.float32), "pb": pb.astype(np.float32), "density": rho.astype(np.float32),
        "fault_main": v["fault_main"], "fault_sec": v["fault_sec"], "breccia": v["breccia"],
        "mineralized": (zn + pb) >= cutoff,
    })


def mineralized_intervals(log: pd.DataFrame) -> list[dict]:
    """Merge consecutive mineralized samples into intervals with length-weighted grades."""
    out: list[dict] = []
    if log.empty:
        return out
    m = log["mineralized"].to_numpy()
    start = None
    for i in range(len(m) + 1):
        if i < len(m) and m[i] and start is None:
            start = i
        elif (i == len(m) or not m[i]) and start is not None:
            seg = log.iloc[start:i]
            out.append({"from_m": float(seg.from_m.iloc[0]), "to_m": float(seg.to_m.iloc[-1]),
                        "length_m": float(seg.to_m.iloc[-1] - seg.from_m.iloc[0]),
                        "z_top": float(seg.z.iloc[0]), "z_bottom": float(seg.z.iloc[-1]),
                        "zn_pct": float(seg.zn.mean()), "pb_pct": float(seg.pb.mean())})
            start = None
    return out


def lithology_contacts(log: pd.DataFrame) -> list[dict]:
    out = []
    lith = log["lith_name"].to_numpy()
    for i in range(1, len(lith)):
        if lith[i] != lith[i - 1]:
            out.append({"depth_along": float(log.from_m.iloc[i]), "z": float(log.z.iloc[i]),
                        "from": lith[i - 1], "to": lith[i]})
    return out


def structural_zones(log: pd.DataFrame) -> list[dict]:
    out = []
    for col, name in (("fault_main", "Main fault zone"), ("fault_sec", "Secondary fault zone")):
        hit = log[log[col]]
        if len(hit):
            out.append({"structure": name, "from_m": float(hit.from_m.min()), "to_m": float(hit.to_m.max()),
                        "z_mid": float(hit.z.mean())})
    return out
