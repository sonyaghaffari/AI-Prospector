#!/usr/bin/env python
"""Angouran Intelligent Exploration Simulator — CLI entry point.

Examples:
  python main.py gpu-check
  python main.py new --exp-id demo01 --profile demo
  python main.py train --exp-id demo01
  python main.py rank --exp-id demo01 --strategy multi_objective
  python main.py drill --exp-id demo01 --hole-id CAND-012 --yes
  python main.py step --exp-id demo01 --yes                 # train + rank + drill top candidate
  python main.py evaluate --exp-id demo01
  python main.py benchmark --profile benchmark
  python main.py dashboard                                   # launches Streamlit
"""
from __future__ import annotations

import argparse
import json
import sys

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.gpu import inspect_gpu


def cmd_gpu_check(args):
    info = inspect_gpu()
    print(info.summary())
    print(json.dumps(info.as_dict(), indent=2))
    if not info.cuda_available:
        sys.exit(1)


def cmd_new(args):
    from src.simulation.experiment import Experiment

    cfg = load_config(args.profile)
    if args.seed is not None:
        cfg.world.seed = args.seed
    exp = Experiment.create(cfg, args.exp_id, allow_cpu=args.allow_cpu)
    print(f"Created experiment '{args.exp_id}' (GPU: {exp.gpu_info.summary()})")
    print(f"Historical holes: {exp.truth.table('historical_collars').shape[0]} | "
         f"candidates: {len(exp.obs.candidates())} | observed samples: {len(exp.obs.samples())}")


def cmd_train(args):
    from src.simulation.experiment import Experiment

    exp = Experiment.load(args.exp_id, allow_cpu=args.allow_cpu)
    info = exp.train(warm_start=len(exp.history) > 0)
    print(f"Trained ensemble in {info['seconds']:.2f}s on {info['n_samples']} samples / {info['n_holes']} holes.")
    print("Per-member best validation loss:", info["final_val_loss"])


def cmd_rank(args):
    from src.simulation.experiment import Experiment

    exp = Experiment.load(args.exp_id, allow_cpu=args.allow_cpu)
    if exp.model is None:
        exp.train()
    rank = exp.rank_candidates(args.strategy)
    cols = ["rank", "hole_id", "type", "cost_usd", "exp_mineralized_m", "eivr_pct", "hypothesis_eig_nats", "score"]
    print(rank[cols].head(args.top).to_string(index=False))


def cmd_drill(args):
    from src.simulation.experiment import Experiment

    if not args.yes:
        print("Refusing to drill without --yes (explicit authorization required).")
        sys.exit(1)
    exp = Experiment.load(args.exp_id, allow_cpu=args.allow_cpu)
    result = exp.drill(args.hole_id, authorized=True)
    print(json.dumps(result.summary(), indent=2))
    exp.record_iteration(result, {"seconds": 0.0})


def cmd_step(args):
    from src.simulation.experiment import Experiment

    exp = Experiment.load(args.exp_id, allow_cpu=args.allow_cpu)
    out = exp.step(strategy=args.strategy, hole_id=args.hole_id, authorized=args.yes)
    print(f"Trained in {out['train']['seconds']:.2f}s.")
    if out["drill_result"]:
        print("Drilled:", json.dumps(out["drill_result"].summary(), indent=2))
    else:
        print("No hole drilled (pass --yes to authorize drilling the top-ranked candidate).")
    print(out["ranking"][["rank", "hole_id", "score"]].head(5).to_string(index=False))


def cmd_evaluate(args):
    from src.evaluation.metrics import evaluate_model, validation_samples
    from src.simulation.experiment import Experiment

    exp = Experiment.load(args.exp_id, allow_cpu=args.allow_cpu)
    if exp.model is None:
        exp.train()
    val = validation_samples(exp.truth, exp.cfg.historical.observed_max_depth)
    metrics = evaluate_model(exp.model, val)
    print(json.dumps(metrics, indent=2))


def cmd_benchmark(args):
    from src.evaluation.benchmark import run_benchmark, save, summarize

    cfg = load_config(args.profile)
    df = run_benchmark(cfg, base_exp_prefix=args.prefix)
    out = PROJECT_ROOT / "artifacts" / "reports" / f"benchmark_{args.prefix}.csv"
    save(df, out)
    summary = summarize(df)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Saved detailed results to {out}")


def cmd_dashboard(args):
    import subprocess

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "dashboard.py")])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--exp-id", required=True)
        sp.add_argument("--allow-cpu", action="store_true", help="Run on CPU if CUDA is unavailable (slow).")

    s = sub.add_parser("gpu-check", help="Inspect GPU / CUDA / PyTorch configuration.")
    s.set_defaults(func=cmd_gpu_check)

    s = sub.add_parser("new", help="Create a new experiment: generate and freeze the world.")
    s.add_argument("--exp-id", required=True)
    s.add_argument("--profile", default="demo")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--allow-cpu", action="store_true")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("train", help="Train/retrain the GPU ensemble on current observed data.")
    common(s)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("rank", help="Rank candidate drill holes.")
    common(s)
    s.add_argument("--strategy", default=None)
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(func=cmd_rank)

    s = sub.add_parser("drill", help="Execute a virtual drill hole (requires --yes).")
    common(s)
    s.add_argument("--hole-id", required=True)
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_drill)

    s = sub.add_parser("step", help="Train + rank + (optionally) drill in one call.")
    common(s)
    s.add_argument("--strategy", default=None)
    s.add_argument("--hole-id", default=None)
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_step)

    s = sub.add_parser("evaluate", help="Evaluate the current model against held-out validation holes.")
    common(s)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("benchmark", help="Run the multi-world, multi-strategy benchmark.")
    s.add_argument("--profile", default="benchmark")
    s.add_argument("--prefix", default="bench")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("dashboard", help="Launch the Streamlit dashboard.")
    s.set_defaults(func=cmd_dashboard)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
