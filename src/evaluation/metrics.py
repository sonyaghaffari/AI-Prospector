"""Geological prediction metrics against the permanently held-out validation holes.

The validation holes are drilled to ~200 m in the hidden truth but their deep samples are
never written into ``data/observed/``; this module reads them straight from the hidden
truth store (evaluation-only access, never touched by training or acquisition code).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.models.ensemble import GeoModel
from src.uncertainty.calibration import brier, error_uncertainty_correlation, expected_calibration_error, \
    interval_coverage


def validation_samples(truth_store, observed_max_depth: float | None = None) -> pd.DataFrame:
    full = truth_store.table("historical_full")
    collars = truth_store.table("historical_collars")
    val_ids = set(collars.loc[collars.split == "validation", "hole_id"])
    df = full[full.hole_id.isin(val_ids)]
    if observed_max_depth is not None:
        df = df[df.z > observed_max_depth]  # the part that was never, and could never be, trained on
    return df.reset_index(drop=True)


def evaluate_model(model: GeoModel, val: pd.DataFrame, extrapolation_depth: float = 100.0) -> dict:
    pts = val[["x", "y", "z"]].to_numpy(np.float32)
    pred = model.predict_points(pts)
    c = lambda t: t.detach().cpu().numpy()  # noqa: E731
    p_mean, zn_mean, pb_mean = c(pred["p_mean"]), c(pred["zn_mean"]), c(pred["pb_mean"])
    zn_sd = np.sqrt(c(pred["zn_epi_var"]) + c(pred["zn_alea_var"]))
    y_zn, y_pb, y_min = val.zn.to_numpy(), val.pb.to_numpy(), val.mineralized.to_numpy().astype(float)
    deep = val.z.to_numpy() > extrapolation_depth

    def block(mask, label):
        if mask.sum() < 2:
            return {}
        zn_err = np.abs(zn_mean[mask] - y_zn[mask])
        out = {
            "zn_mae": float(np.mean(zn_err)), "zn_rmse": float(np.sqrt(np.mean((zn_mean[mask] - y_zn[mask]) ** 2))),
            "pb_mae": float(np.mean(np.abs(pb_mean[mask] - y_pb[mask]))),
            "pb_rmse": float(np.sqrt(np.mean((pb_mean[mask] - y_pb[mask]) ** 2))),
            "ece": expected_calibration_error(p_mean[mask], y_min[mask]),
            "brier": brier(p_mean[mask], y_min[mask]),
            "interval_coverage_90": interval_coverage(c(pred["zn_p05"])[mask], c(pred["zn_p95"])[mask], y_zn[mask]),
            "error_uncertainty_spearman": error_uncertainty_correlation(zn_err, zn_sd[mask]),
            "n": int(mask.sum()),
        }
        if len(np.unique(y_min[mask])) > 1:
            out["roc_auc"] = float(roc_auc_score(y_min[mask], p_mean[mask]))
            out["average_precision"] = float(average_precision_score(y_min[mask], p_mean[mask]))
            pr, rc, _ = precision_recall_curve(y_min[mask], p_mean[mask])
            out["precision_at_50pct_recall"] = float(pr[np.searchsorted(rc[::-1], 0.5)]) if (rc >= 0.5).any() else None
        return {f"{label}_{k}": v for k, v in out.items()}

    return {**block(np.ones(len(val), bool), "all"), **block(~deep, "interpolation_shallow"),
            **block(deep, "extrapolation_deep")}


def classification_at_threshold(p: np.ndarray, y: np.ndarray, thresh: float = 0.5) -> dict:
    pred = p >= thresh
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "accuracy": (tp + tn) / max(len(y), 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
