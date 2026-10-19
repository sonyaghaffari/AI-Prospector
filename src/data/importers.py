"""Import real drill-hole databases (collar / survey / assay / lithology CSVs) into the
ObservationStore sample schema. Intended for future Angouran data; requires geological QA/QC.

Expected inputs (column names configurable via ``colmap``):
  collars:  hole_id, x, y, z (collar RL converted to depth-below-datum), azimuth, dip, length
  assays:   hole_id, from_m, to_m, zn, pb [, density]
  litho:    hole_id, from_m, to_m, lith_code
Straight-hole desurvey is used; replace with minimum-curvature for deviated holes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.geology.grid import LITHOLOGY
from src.geology.trajectory import direction


def import_drillholes(collars: pd.DataFrame, assays: pd.DataFrame, litho: pd.DataFrame,
                      lith_map: dict[str, int], cutoff: float = 2.0, colmap: dict | None = None) -> pd.DataFrame:
    colmap = colmap or {}
    collars = collars.rename(columns=colmap)
    assays = assays.rename(columns=colmap)
    litho = litho.rename(columns=colmap)
    rows = []
    for c in collars.itertuples():
        a = assays[assays.hole_id == c.hole_id].sort_values("from_m")
        lt = litho[litho.hole_id == c.hole_id]
        d = direction(c.azimuth, c.dip)
        for r in a.itertuples():
            mid = 0.5 * (r.from_m + r.to_m)
            p = np.array([c.x, c.y, c.z]) + mid * d
            lrow = lt[(lt.from_m <= mid) & (lt.to_m > mid)]
            code = lith_map.get(str(lrow.lith_code.iloc[0]), 4) if len(lrow) else 4
            rows.append({"hole_id": c.hole_id, "from_m": r.from_m, "to_m": r.to_m, "depth_along": mid,
                         "x": p[0], "y": p[1], "z": p[2], "lithology": code, "lith_name": LITHOLOGY[code],
                         "zn": r.zn, "pb": r.pb, "density": getattr(r, "density", np.nan),
                         "fault_main": False, "fault_sec": False, "breccia": code == 2,
                         "mineralized": (r.zn + r.pb) >= cutoff})
    return pd.DataFrame(rows)
