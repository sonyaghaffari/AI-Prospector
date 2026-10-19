"""Experimental variogram of log1p(Zn) from observed samples (GPU pair computation).

Provides the spatial correlation ranges used by the acquisition surrogate. Horizontal
pairs: |dz| < 5 m between different holes; vertical pairs: horizontal offset < 10 m.
An exponential model gamma(d) = c0 + c (1 - exp(-d / a)) is fitted by weighted least squares
over a grid of scale parameters ``a``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _fit_exponential(lags: np.ndarray, gam: np.ndarray, cnt: np.ndarray, a_grid: np.ndarray) -> float | None:
    ok = cnt > 30
    if ok.sum() < 4:
        return None
    lags, gam, w = lags[ok], gam[ok], np.sqrt(cnt[ok])
    best, best_a = np.inf, None
    for a in a_grid:
        A = np.column_stack([np.ones_like(lags), 1 - np.exp(-lags / a)])
        coef, *_ = np.linalg.lstsq(A * w[:, None], gam * w, rcond=None)
        if coef[1] <= 0:
            continue
        r = np.sum(w * (A @ coef - gam) ** 2)
        if r < best:
            best, best_a = r, float(a)
    return best_a


@torch.no_grad()
def estimate_ranges(samples: pd.DataFrame, device: torch.device, max_points: int = 4000,
                    seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    s = samples if len(samples) <= max_points else samples.iloc[rng.choice(len(samples), max_points, replace=False)]
    xyz = torch.as_tensor(s[["x", "y", "z"]].to_numpy(np.float32), device=device)
    v = torch.as_tensor(np.log1p(s.zn.clip(lower=0).to_numpy(np.float32)), device=device)
    hole = torch.as_tensor(pd.factorize(s.hole_id)[0], device=device)
    dh = torch.cdist(xyz[:, :2], xyz[:, :2])
    dz = (xyz[:, 2:3] - xyz[:, 2:3].T).abs()
    sq = 0.5 * (v[:, None] - v[None, :]) ** 2
    iu = torch.triu(torch.ones_like(dh, dtype=torch.bool), 1)

    def binned(dist, mask, edges):
        dist, val = dist[mask & iu], sq[mask & iu]
        b = torch.bucketize(dist, torch.as_tensor(edges, device=device, dtype=dist.dtype)) - 1
        nb = len(edges) - 1
        keep = (b >= 0) & (b < nb)
        cnt = torch.bincount(b[keep], minlength=nb).double()
        tot = torch.bincount(b[keep], weights=val[keep].double(), minlength=nb)
        return (0.5 * (edges[1:] + edges[:-1]), (tot / cnt.clamp(min=1)).cpu().numpy(), cnt.cpu().numpy())

    h_edges = np.arange(0, 525, 25.0)
    v_edges = np.arange(0, 102.5, 5.0)
    lag_h, g_h, c_h = binned(dh, (dz < 5) & (hole[:, None] != hole[None, :]), h_edges)
    lag_v, g_v, c_v = binned(dz, dh < 10, v_edges)
    a_h = _fit_exponential(lag_h, g_h, c_h, np.linspace(15, 300, 58))
    a_v = _fit_exponential(lag_v, g_v, c_v, np.linspace(3, 80, 78))
    return {"range_h": a_h, "range_v": a_v,
            "horizontal": {"lag": lag_h.tolist(), "gamma": g_h.tolist(), "count": c_h.tolist()},
            "vertical": {"lag": lag_v.tolist(), "gamma": g_v.tolist(), "count": c_v.tolist()}}
