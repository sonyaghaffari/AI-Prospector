"""The importer must produce ObservationStore-compatible samples from generic tabular input
(future real-data integration path), independent of the synthetic generator."""
import pandas as pd

from src.data.importers import import_drillholes
from src.data.observations import REQUIRED_SAMPLE_COLUMNS


def test_import_drillholes_produces_valid_schema():
    collars = pd.DataFrame([{"hole_id": "DDH-1", "x": 100.0, "y": 200.0, "z": 0.0, "azimuth": 0.0, "dip": 90.0,
                            "length": 50.0}])
    assays = pd.DataFrame([
        {"hole_id": "DDH-1", "from_m": 0.0, "to_m": 5.0, "zn": 0.5, "pb": 0.1},
        {"hole_id": "DDH-1", "from_m": 5.0, "to_m": 10.0, "zn": 8.0, "pb": 2.0},
    ])
    litho = pd.DataFrame([
        {"hole_id": "DDH-1", "from_m": 0.0, "to_m": 5.0, "lith_code": "SCHIST"},
        {"hole_id": "DDH-1", "from_m": 5.0, "to_m": 10.0, "lith_code": "MARBLE"},
    ])
    lith_map = {"SCHIST": 1, "MARBLE": 0}
    samples = import_drillholes(collars, assays, litho, lith_map, cutoff=2.0)
    assert set(REQUIRED_SAMPLE_COLUMNS) <= set(samples.columns)
    assert len(samples) == 2
    assert bool(samples.iloc[1].mineralized) is True  # 8.0 + 2.0 >= 2.0 cutoff
    assert bool(samples.iloc[0].mineralized) is False
    assert samples.iloc[1].lithology == 0  # MARBLE mapped correctly
    # straight vertical hole: second interval should be deeper (larger z) than the first
    assert samples.iloc[1].z > samples.iloc[0].z


def test_import_handles_column_remapping():
    collars = pd.DataFrame([{"HoleID": "A1", "x": 0.0, "y": 0.0, "z": 0.0, "azimuth": 0.0, "dip": 90.0,
                            "length": 20.0}])
    assays = pd.DataFrame([{"HoleID": "A1", "from_m": 0.0, "to_m": 10.0, "zn": 1.0, "pb": 1.0}])
    litho = pd.DataFrame([{"HoleID": "A1", "from_m": 0.0, "to_m": 10.0, "lith_code": "X"}])
    samples = import_drillholes(collars, assays, litho, {"X": 4}, colmap={"HoleID": "hole_id"})
    assert len(samples) == 1
    assert samples.iloc[0].hole_id == "A1"
