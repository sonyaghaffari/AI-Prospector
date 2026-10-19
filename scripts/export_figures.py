#!/usr/bin/env python
"""Export PNG/HTML figures and CSV/Parquet/GeoJSON results for a completed experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.experiment import Experiment  # noqa: E402
from src.visualization import theme  # noqa: E402,F401  (registers the dark template)
from src.visualization.scene3d import drill_holes_figure  # noqa: E402


def to_geojson(df, x="x", y="y") -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r[x]), float(r[y])]},
         "properties": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                        for k, v in r.items() if k not in (x, y)}}
        for _, r in df.iterrows()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    args = ap.parse_args()

    exp = Experiment.load(args.exp_id)
    fig_dir = exp.paths.artifact("figures")
    pred_dir = exp.paths.artifact("predictions")

    collars, samples, cands = exp.obs.collars(), exp.obs.samples(), exp.obs.candidates()
    fig3d = drill_holes_figure(collars, samples, cands, executed=exp.oracle.executed if exp.oracle else set())
    fig3d.write_html(fig_dir / "drill_holes_3d.html")

    samples.to_csv(pred_dir / "observed_samples.csv", index=False)
    samples.to_parquet(pred_dir / "observed_samples.parquet", index=False)
    cands.to_csv(pred_dir / "candidates.csv", index=False)
    with open(pred_dir / "collars.geojson", "w", encoding="utf-8") as fh:
        json.dump(to_geojson(collars), fh)
    if exp.rankings is not None:
        exp.rankings.to_csv(pred_dir / "latest_ranking.csv", index=False)

    print(f"Exported figures to {fig_dir} and tabular results to {pred_dir}")


if __name__ == "__main__":
    main()
