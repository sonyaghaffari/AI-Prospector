"""Geological world sanity checks and shallow-depth information-leakage check (section 23)."""
import numpy as np

from src.drilling.candidates import generate_candidates
from src.drilling.historical import generate_historical
from src.geology.generator import generate_world
from src.utils.config import SCENARIOS
from tests.conftest import fast_cfg


def test_all_scenarios_generate_valid_worlds():
    for sc in SCENARIOS:
        cfg = fast_cfg().world
        cfg.scenario = sc
        w = generate_world(cfg)
        assert w.zn.shape == tuple(cfg.shape)
        assert np.isfinite(w.zn).all() and (w.zn >= 0).all()
        assert np.isfinite(w.pb).all() and (w.pb >= 0).all()
        assert set(np.unique(w.lithology)) <= {0, 1, 2, 3, 4}
        assert w.meta["scenario"] == sc


def test_mineralization_is_spatially_continuous_not_iid_noise():
    """Neighbouring voxels must be strongly correlated (continuity requirement, section 4)."""
    w = generate_world(fast_cfg().world)
    zn = w.zn
    corr = np.corrcoef(zn[:-1, :, :].ravel(), zn[1:, :, :].ravel())[0, 1]
    assert corr > 0.3  # i.i.d. noise would give ~0; spatial fields give strong positive correlation


def test_initial_observed_data_has_no_deep_information(clean_exp_id):
    """The public/observed dataset must not let the model see truth below observed_max_depth,
    directly OR indirectly (e.g. no deep assay values embedded anywhere in observed files)."""
    from src.simulation.experiment import Experiment

    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    samples = exp.obs.samples()
    assert samples.z.max() <= cfg.historical.observed_max_depth + 1e-6
    # candidates carry geometry/cost only -- no grade or lithology info pre-drilling
    assert not ({"zn", "pb", "lithology", "mineralized"} & set(exp.obs.candidates().columns))


def test_historical_holes_are_clustered_not_uniform():
    w = generate_world(fast_cfg().world)
    collars, _, _ = generate_historical(w, fast_cfg().historical, 4242, 2.0, 2000.0)
    xy = collars[["x", "y"]].to_numpy()
    center = xy.mean(0)
    d = np.linalg.norm(xy - center, axis=1)
    # if clustered, most holes should be much closer to the cluster centroid than a uniform
    # distribution over the 1000x1000 domain would predict (uniform expected mean dist ~ 383 m)
    assert np.median(d) < 300


def test_world_centres_are_well_spread():
    """Regression test: independently seeded worlds must NOT all place their mineralized centre
    near the domain's geometric middle (an earlier bug restricted it to +/-100 m of centre,
    which made independent worlds spatially correlated and inflated apparent cross-world model
    transfer -- see src/geology/generator.py's `centre` comment)."""
    centres = []
    for seed in (1, 2, 3, 4, 5, 6):
        cfg = fast_cfg(seed=seed).world
        w = generate_world(cfg)
        centres.append(w.meta["centre"])
    centres = np.array(centres)
    domain_mid = np.array(cfg.extent[:2]) / 2
    dist_from_mid = np.linalg.norm(centres - domain_mid, axis=1)
    # with a well-spread placement, at least some centres should land well away from the middle
    assert dist_from_mid.max() > 150.0, f"all centres suspiciously close to domain middle: {dist_from_mid}"
    assert dist_from_mid.std() > 30.0, "centres have implausibly little spread for independent seeds"


def test_candidate_catalogue_within_bounds_and_nonredundant():
    w = generate_world(fast_cfg().world)
    cands = generate_candidates(w.grid, fast_cfg().candidates)
    assert len(cands) > 5
    assert (cands.cost_usd > 0).all()
    xy = cands[["x", "y"]].to_numpy()
    assert (xy >= 0).all() and (xy[:, 0] <= w.grid.extent[0]).all() and (xy[:, 1] <= w.grid.extent[1]).all()
