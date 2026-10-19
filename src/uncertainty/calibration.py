"""Calibration diagnostics (pure functions: predictions + labels in, statistics out)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, np.ndarray]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    conf, acc, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b]:
            conf[b], acc[b] = p[m].mean(), y[m].mean()
    return {"confidence": conf, "observed_freq": acc, "count": cnt, "edges": edges}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    r = reliability_curve(p, y, n_bins)
    w = r["count"] / max(r["count"].sum(), 1)
    return float(np.sum(w * np.abs(r["confidence"] - r["observed_freq"])))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def interval_coverage(lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((truth >= lo) & (truth <= hi)))


def error_uncertainty_correlation(abs_err: np.ndarray, sigma: np.ndarray) -> float:
    if np.std(sigma) == 0 or np.std(abs_err) == 0:
        return float("nan")
    return float(spearmanr(abs_err, sigma).statistic)
