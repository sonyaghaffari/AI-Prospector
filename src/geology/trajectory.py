"""Drill-hole geometry (no geology). Straight holes: azimuth clockwise from north (+Y),
dip measured downward from horizontal (90 = vertical)."""
from __future__ import annotations

import numpy as np


def direction(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    az, dip = np.radians(azimuth_deg), np.radians(dip_deg)
    return np.array([np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), np.sin(dip)])


def trajectory_points(collar: np.ndarray, azimuth: float, dip: float, length: float,
                      step: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample midpoints along a straight hole. Returns (points (n,3), along-hole depth (n,))."""
    n = max(1, int(np.floor(length / step)))
    s = (np.arange(n) + 0.5) * step
    pts = np.asarray(collar, dtype=np.float64)[None, :] + s[:, None] * direction(azimuth, dip)[None, :]
    return pts, s


def length_to_depth(collar_z: float, dip: float, target_depth: float) -> float:
    return (target_depth - collar_z) / max(np.sin(np.radians(dip)), 1e-6)


def survey_table(hole_id: str, azimuth: float, dip: float, length: float) -> list[dict]:
    """Straight-hole survey (collar and end of hole) in a desurvey-friendly format."""
    return [
        {"hole_id": hole_id, "depth": 0.0, "azimuth": azimuth, "dip": dip},
        {"hole_id": hole_id, "depth": float(length), "azimuth": azimuth, "dip": dip},
    ]
