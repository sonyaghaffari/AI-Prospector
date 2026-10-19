"""Benchmark: run every strategy on every independent frozen world, aggregate results.

Each world has its own immutable seed (``base_seed + world_index``) and is built once via
``Experiment.create``. The SAME world + SAME initial observed data + SAME candidate
catalogue is reused for every strategy (``run_strategy`` copies the observed/ snapshot
before each strategy's independent sequence of drilling decisions), so comparisons are
apples-to-apples. Scenario is drawn from the world seed and never leaks into any config.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.strategies import run_strategy
from src.simulation.experiment import Experiment
from src.utils.config import SCENARIOS, BenchmarkConfig, ExperimentConfig, config_from_dict


def run_benchmark(cfg: ExperimentConfig, base_exp_prefix: str = "bench") -> pd.DataFrame:
    bcfg: BenchmarkConfig = cfg.benchmark
    rows = []
    world_idx = 0
    for scenario in SCENARIOS:
        for rep in range(bcfg.n_worlds_per_scenario):
            seed = bcfg.base_seed + world_idx
            world_idx += 1
            wcfg = config_from_dict(cfg.to_dict())
            wcfg.world.scenario = scenario
            wcfg.world.seed = seed
            base_id = f"{base_exp_prefix}_{scenario}_{rep}_{seed}"
            t0 = time.perf_counter()
            Experiment.create(wcfg, base_id)
            build_seconds = time.perf_counter() - t0
            for strategy in bcfg.strategies:
                t1 = time.perf_counter()
                run = run_strategy(base_id, strategy, wcfg, bcfg.n_holes, seed, f"{strategy}")
                for it in run.iterations:
                    rows.append({"scenario": scenario, "world_seed": seed, "world_rep": rep,
                                "strategy": strategy, "build_seconds": build_seconds,
                                "strategy_seconds": time.perf_counter() - t1, **it})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across worlds, at the FINAL iteration of each (scenario, strategy) run."""
    last = df.sort_values("iteration").groupby(["scenario", "strategy", "world_seed"]).tail(1)
    metric_cols = [c for c in last.columns if c.startswith(("all_", "extrapolation_", "interpolation_"))
                  or c in ("cumulative_cost", "deep_mineralized_m")]
    agg = last.groupby(["strategy"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def save(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
