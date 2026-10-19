import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.world_builder import ExperimentPaths  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _force_rmtree(path: Path) -> None:
    """shutil.rmtree fails on Windows for read-only files (hidden-truth is deliberately
    made read-only); clear the attribute before removal."""
    def onerror(func, p, exc_info):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onerror=onerror)


def fast_cfg(seed: int = 4242):
    cfg = load_config("demo")
    cfg.world.seed = seed
    cfg.world.shape = (24, 24, 20)
    cfg.historical.n_train_holes = 30
    cfg.historical.n_validation_holes = 8
    cfg.model.max_epochs = 25
    cfg.model.min_epochs = 10
    cfg.model.patience = 8
    cfg.model.ensemble_size = 3
    cfg.acquisition.target_points = 500
    cfg.hypotheses.particles_per_hypothesis = 32
    cfg.hypotheses.simulated_outcomes = 16
    return cfg


@pytest.fixture
def clean_exp_id():
    exp_id = "pytest_tmp"
    yield exp_id
    p = ExperimentPaths(exp_id)
    for d in (p.hidden, p.observed):
        if d.exists():
            _force_rmtree(d)
    if p.world_record.exists():
        p.world_record.unlink()
    for kind in ("checkpoints", "reports", "predictions", "figures"):
        d = ROOT / "artifacts" / kind / exp_id
        if d.exists():
            _force_rmtree(d)
