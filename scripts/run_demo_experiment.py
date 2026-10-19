#!/usr/bin/env python
"""Phase 7 / Section 27 required demonstration: one real, measured end-to-end run.

Generates a frozen world, trains the GPU ensemble on shallow data only, predicts to
200 m, ranks candidates, drills one real virtual hole, retrains, re-ranks, evaluates,
and writes a JSON report with ACTUAL measured numbers (no fabricated results).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate_model, validation_samples  # noqa: E402
from src.simulation.experiment import Experiment  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="demo_end_to_end")
    ap.add_argument("--profile", default="demo")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--n-holes", type=int, default=3)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    cfg.world.seed = args.seed
    report: dict = {"exp_id": args.exp_id, "profile": args.profile, "seed": args.seed, "steps": []}

    t0 = time.perf_counter()
    exp = Experiment.create(cfg, args.exp_id, allow_cpu=args.allow_cpu)
    report["gpu"] = exp.gpu_info.as_dict()
    report["world_build_seconds"] = time.perf_counter() - t0
    report["n_historical_holes"] = int(exp.truth.table("historical_collars").shape[0])
    report["n_candidates"] = int(len(exp.obs.candidates()))
    report["n_initial_observed_samples"] = int(len(exp.obs.samples()))
    print(f"[1] World generated and frozen. {report['n_historical_holes']} historical holes, "
         f"{report['n_candidates']} candidates, {report['n_initial_observed_samples']} shallow samples. "
         f"({report['world_build_seconds']:.1f}s)")
    print(f"    GPU: {exp.gpu_info.summary()}")

    val = validation_samples(exp.truth, exp.cfg.historical.observed_max_depth)
    print(f"[2] Held-out validation set (deep, never used for training): {len(val)} samples.")

    for i in range(args.n_holes):
        warm = i > 0
        t1 = time.perf_counter()
        train_info = exp.train(warm_start=warm)
        metrics = evaluate_model(exp.model, val)
        rank = exp.rank_candidates()
        top = rank.iloc[0]
        t2 = time.perf_counter()
        result = exp.drill(top.hole_id, authorized=True)
        rec = exp.record_iteration(result, train_info)
        hyp = exp.hypothesis_posterior()
        step_report = {
            "iteration": i, "train_seconds": train_info["seconds"], "n_samples": train_info["n_samples"],
            "n_holes": train_info["n_holes"], "ranking_seconds": t2 - t1 - train_info["seconds"],
            "top_candidate": {"hole_id": top.hole_id, "score": float(top.score),
                              "exp_mineralized_m": float(top.exp_mineralized_m),
                              "eivr_pct": float(top.eivr_pct), "cost_usd": float(top.cost_usd)},
            "drill_result": result.summary(), "metrics_before_this_drill": metrics,
            "hypothesis_posterior": hyp, "budget_remaining": rec.budget_remaining,
        }
        report["steps"].append(step_report)
        print(f"[{3 + i}] Iter {i}: trained {train_info['seconds']:.2f}s on {train_info['n_samples']} samples "
             f"({train_info['n_holes']} holes). Extrapolation Zn MAE: "
             f"{metrics.get('extrapolation_deep_zn_mae', float('nan')):.3f}. "
             f"Top candidate {top.hole_id} (score {top.score:.3f}) -> drilled: "
             f"{result.summary()['deep_mineralized_m']:.1f} m mineralized below 100m, "
             f"cost ${result.cost_usd:,.0f}. Budget remaining ${rec.budget_remaining:,.0f}.")
        if exp.budget_remaining < cfg.candidates.mobilization_cost:
            print("    Budget exhausted — stopping.")
            break

    # final evaluation after all drilling
    final_metrics = evaluate_model(exp.model, val)
    report["final_metrics"] = final_metrics
    report["total_seconds"] = time.perf_counter() - t0
    report["perf_log"] = exp.perf.as_rows()
    report["truth_manifest_verified"] = exp.truth.verify()

    out = exp.paths.artifact("reports") / "end_to_end_demo_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n[DONE] Total wall time {report['total_seconds']:.1f}s. Report saved to {out}")
    print(f"Final extrapolation (100-200m) Zn MAE: {final_metrics.get('extrapolation_deep_zn_mae')}, "
         f"ROC-AUC: {final_metrics.get('extrapolation_deep_roc_auc')}")
    print("Hidden-truth integrity verified:", report["truth_manifest_verified"])


if __name__ == "__main__":
    main()
