"""Anti-data-leakage and isolation tests (requirements doc, section 23)."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.simulation.experiment import Experiment
from src.simulation.truth_store import HiddenTruthStore, TruthIntegrityError
from src.simulation.world_builder import ExperimentPaths
from tests.conftest import ROOT, fast_cfg


def test_truth_generated_only_once(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    assert exp.truth.exists
    with pytest.raises(FileExistsError):
        Experiment.create(cfg, clean_exp_id)


def test_hidden_truth_files_are_readonly(clean_exp_id):
    cfg = fast_cfg()
    Experiment.create(cfg, clean_exp_id)
    truth_dir = ExperimentPaths(clean_exp_id).hidden
    with pytest.raises(PermissionError):
        with open(truth_dir / "truth.npz", "wb") as fh:
            fh.write(b"corrupt")


def test_hidden_truth_integrity_detected(clean_exp_id, tmp_path):
    cfg = fast_cfg()
    Experiment.create(cfg, clean_exp_id)
    truth_dir = ExperimentPaths(clean_exp_id).hidden
    store = HiddenTruthStore(truth_dir)
    world = store.load_world()
    # swap in a modified copy of the frozen array to simulate tampering, then verify it is caught
    tampered = world.zn.copy()
    tampered[0, 0, 0] += 1000.0
    object.__setattr__(world, "zn", tampered)
    with pytest.raises(TruthIntegrityError):
        store.verify_arrays(world)


def test_model_cannot_access_hidden_deep_observations(clean_exp_id):
    """Shallow-only training samples must never contain z > observed_max_depth."""
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    samples = exp.obs.samples()
    assert (samples.z <= cfg.historical.observed_max_depth + 1e-6).all()


def test_validation_holes_never_in_observed_data(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    collars = exp.truth.table("historical_collars")
    val_ids = set(collars.loc[collars.split == "validation", "hole_id"])
    observed_ids = set(exp.obs.samples().hole_id)
    assert observed_ids.isdisjoint(val_ids)


def test_candidate_outcomes_not_in_observed_or_public_files(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    cand_cols = set(exp.obs.candidates().columns)
    assert "zn" not in cand_cols and "pb" not in cand_cols and "lithology" not in cand_cols
    for f in Path(exp.paths.observed).glob("*"):
        if f.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(f)
            assert not {"concealed_centre", "scenario"} & set(df.columns)


def test_executing_hole_reveals_only_that_hole(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    exp.train()
    rank = exp.rank_candidates()
    before = set(exp.obs.samples().hole_id)
    hid = rank.iloc[0].hole_id
    result = exp.drill(hid, authorized=True)
    after = set(exp.obs.samples().hole_id)
    assert after - before == {hid}
    assert set(result.samples.hole_id) == {hid}


def test_drilling_requires_explicit_authorization(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    exp.train()
    rank = exp.rank_candidates()
    from src.drilling.oracle import DrillingAuthorizationError

    with pytest.raises(DrillingAuthorizationError):
        exp.oracle.execute(rank.iloc[0].hole_id, authorized=False)
    with pytest.raises(PermissionError):
        exp.drill(rank.iloc[0].hole_id, authorized=False)


def test_cannot_drill_same_hole_twice(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    exp.train()
    hid = exp.rank_candidates().iloc[0].hole_id
    exp.drill(hid, authorized=True)
    from src.drilling.oracle import DrillingAuthorizationError

    with pytest.raises((DrillingAuthorizationError, ValueError)):
        exp.oracle.execute(hid, authorized=True)


def test_world_scenario_not_in_public_config_or_observed(clean_exp_id):
    cfg = fast_cfg()
    exp = Experiment.create(cfg, clean_exp_id)
    pub = exp.obs.public_config()
    assert pub["world"]["scenario"] == "<hidden>"
    rec = json.loads(exp.paths.world_record.read_text())
    assert rec["config"]["world"]["scenario"] == "<hidden>"


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("pkg", ["models", "optimization"])
def test_no_static_import_of_hidden_truth_modules(pkg):
    """Static check: prediction/acquisition code must never import the truth store,
    generator, or oracle modules directly (the only sanctioned path is Experiment)."""
    forbidden = {"src.simulation.truth_store", "src.geology.generator", "src.drilling.oracle",
                "src.drilling.historical"}
    for f in (ROOT / "src" / pkg).rglob("*.py"):
        imports = _module_imports(f)
        assert not (imports & forbidden), f"{f} imports hidden-truth module(s): {imports & forbidden}"
