#!/usr/bin/env python
"""Compare the GPU deep ensemble against IDW and Gaussian-Process baselines on the SAME
observed data and the SAME held-out validation holes (Section 9/19 requirement).

Does not assume the neural ensemble wins -- reports measured numbers only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import classification_at_threshold, validation_samples  # noqa: E402
from src.models.baselines import GPBaseline, IDWBaseline  # noqa: E402
from src.simulation.experiment import Experiment  # noqa: E402
from src.uncertainty.calibration import brier, expected_calibration_error  # noqa: E402


def score(name: str, pred: dict, val, seconds: float) -> dict:
    y_zn, y_pb, y_min = val.zn.to_numpy(), val.pb.to_numpy(), val.mineralized.to_numpy().astype(float)
    p = np.clip(pred["p_mean"], 0, 1)
    out = {
        "method": name, "fit_predict_seconds": seconds,
        "zn_mae": float(np.mean(np.abs(pred["zn_mean"] - y_zn))),
        "zn_rmse": float(np.sqrt(np.mean((pred["zn_mean"] - y_zn) ** 2))),
        "pb_mae": float(np.mean(np.abs(pred["pb_mean"] - y_pb))),
        "ece": expected_calibration_error(p, y_min), "brier": brier(p, y_min),
        **{f"cls_{k}": v for k, v in classification_at_threshold(p, y_min).items()
          if k in ("precision", "recall", "f1", "accuracy")},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    args = ap.parse_args()

    exp = Experiment.load(args.exp_id)
    if exp.model is None:
        exp.train()
    samples = exp.obs.samples()
    val = validation_samples(exp.truth, exp.cfg.historical.observed_max_depth)
    pts = val[["x", "y", "z"]].to_numpy(np.float32)

    results = []
    t0 = time.perf_counter()
    pred = exp.model.predict_points(pts)
    c = lambda t: t.detach().cpu().numpy()  # noqa: E731
    results.append(score("GPU Deep Ensemble", {"p_mean": c(pred["p_mean"]), "zn_mean": c(pred["zn_mean"]),
                                                "pb_mean": c(pred["pb_mean"])}, val, time.perf_counter() - t0))

    t0 = time.perf_counter()
    idw = IDWBaseline(exp.device, exp._ranges[0] if exp._ranges else 150.0,
                      exp._ranges[1] if exp._ranges else 40.0).fit(samples)
    results.append(score("IDW (GPU, k=16)", idw.predict(pts), val, time.perf_counter() - t0))

    t0 = time.perf_counter()
    gp = GPBaseline(seed=exp.cfg.world.seed).fit(samples)
    results.append(score("Gaussian Process", gp.predict(pts), val, time.perf_counter() - t0))

    print(f"{'method':<22}{'fit+pred(s)':>12}{'zn_mae':>10}{'zn_rmse':>10}{'pb_mae':>10}{'ece':>8}"
         f"{'brier':>8}{'f1':>8}")
    for r in results:
        print(f"{r['method']:<22}{r['fit_predict_seconds']:>12.2f}{r['zn_mae']:>10.3f}{r['zn_rmse']:>10.3f}"
             f"{r['pb_mae']:>10.3f}{r['ece']:>8.3f}{r['brier']:>8.3f}{r['cls_f1']:>8.3f}")

    out = exp.paths.artifact("reports") / "baseline_comparison.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
