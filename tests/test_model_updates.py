"""Predictions must change appropriately after new observations; oracle determinism."""
import numpy as np

from src.simulation.experiment import Experiment
from tests.conftest import fast_cfg


def test_predictions_change_after_drilling(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    exp.train()
    extent = exp.cfg.world.extent
    rng = np.random.default_rng(0)
    probe = rng.uniform([0, 0, 100.0], [extent[0], extent[1], extent[2]], size=(200, 3)).astype(np.float32)
    pred_before = exp.model.predict_points(probe)["p_mean"].detach().cpu().numpy().copy()

    rank = exp.rank_candidates()
    exp.drill(rank.iloc[0].hole_id, authorized=True)
    exp.train(warm_start=True)
    pred_after = exp.model.predict_points(probe)["p_mean"].detach().cpu().numpy()

    assert not np.allclose(pred_before, pred_after, atol=1e-6)


def test_oracle_outcome_is_deterministic_not_resampled(clean_exp_id):
    """Executing the same candidate via execute() twice (allow_repeat) must return identical
    samples -- the oracle reads frozen outcomes, it never re-draws randomness."""
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    hid = exp.obs.candidates().hole_id.iloc[0]
    r1 = exp.oracle.execute(hid, authorized=True)
    r2 = exp.oracle.execute(hid, authorized=True, allow_repeat=True)
    np.testing.assert_array_equal(r1.samples.zn.to_numpy(), r2.samples.zn.to_numpy())
    np.testing.assert_array_equal(r1.samples.lithology.to_numpy(), r2.samples.lithology.to_numpy())


def test_custom_trajectory_interpolates_truth_without_new_randomness(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    collar = np.array([500.0, 500.0, 0.0])
    r1 = exp.oracle.execute_custom(collar, 0.0, 90.0, 190.0, authorized=True)
    r2 = exp.oracle.execute_custom(collar, 0.0, 90.0, 190.0, authorized=True)
    np.testing.assert_allclose(r1.samples.zn.to_numpy(), r2.samples.zn.to_numpy())
