"""Seed reproducibility and frozen-world reuse across strategies."""
import numpy as np

from src.geology.generator import generate_world
from src.simulation.experiment import Experiment
from tests.conftest import fast_cfg


def test_same_seed_same_world():
    w1 = generate_world(fast_cfg().world)
    w2 = generate_world(fast_cfg().world)
    assert np.array_equal(w1.zn, w2.zn)
    assert np.array_equal(w1.lithology, w2.lithology)
    assert w1.meta["scenario"] == w2.meta["scenario"]


def test_different_seed_different_world():
    c1, c2 = fast_cfg(seed=1), fast_cfg(seed=2)
    w1, w2 = generate_world(c1.world), generate_world(c2.world)
    assert not np.array_equal(w1.zn, w2.zn)


def test_deterministic_training_within_run(clean_exp_id):
    """With deterministic=True and a fixed seed, two fresh ensembles trained on the same
    data converge to the same validation loss (documented limit: exact bitwise equality of
    CUDA kernels across driver versions is not guaranteed; we check close agreement)."""
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    info1 = exp.train(warm_start=False)
    exp.model = None
    info2 = exp.train(warm_start=False)
    np.testing.assert_allclose(info1["final_val_loss"], info2["final_val_loss"], rtol=0.05, atol=0.05)


def test_frozen_world_reused_across_strategies(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    manifest1 = exp.truth.manifest
    exp.train()
    exp.rank_candidates()
    manifest2 = exp.truth.manifest
    assert manifest1 == manifest2  # untouched by training/ranking
