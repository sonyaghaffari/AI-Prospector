"""3D Plotly figures: drill holes, candidates, prediction volumes, uncertainty, debug truth."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.theme import ACCENT, GOOD, LITH_COLORS, LITH_NAMES, WARN

Z_SCENE = dict(scene=dict(zaxis=dict(autorange="reversed", title="Depth (m)"),
                          xaxis=dict(title="X (m)"), yaxis=dict(title="Y (m)"), aspectmode="data"))


def _hole_trace(samples: pd.DataFrame, hole_id: str, color: str, width: int, name: str, legendgroup=None,
               showlegend=True) -> go.Scatter3d:
    h = samples[samples.hole_id == hole_id].sort_values("depth_along")
    return go.Scatter3d(x=h.x, y=h.y, z=h.z, mode="lines", line=dict(color=color, width=width), name=name,
                        legendgroup=legendgroup or name, showlegend=showlegend,
                        hovertext=[f"{hole_id}<br>z={z:.0f}m" for z in h.z], hoverinfo="text")


def drill_holes_figure(collars: pd.DataFrame, samples: pd.DataFrame, candidates: pd.DataFrame | None = None,
                       selected_candidates: list[str] | None = None, executed: set[str] | None = None,
                       show_assays: bool = True) -> go.Figure:
    fig = go.Figure()
    executed = executed or set()
    for grp, color in (("train", "#8899aa"), ("drilled", GOOD)):
        ids = collars.loc[collars.source == grp, "hole_id"] if "source" in collars else []
        for i, hid in enumerate(ids):
            fig.add_trace(_hole_trace(samples, hid, color, 3 if grp == "drilled" else 2,
                                      "Historical" if grp == "train" else "Drilled (virtual)",
                                      legendgroup=grp, showlegend=i == 0))
    if show_assays:
        m = samples[samples.mineralized.astype(bool)]
        fig.add_trace(go.Scatter3d(x=m.x, y=m.y, z=m.z, mode="markers",
                                   marker=dict(size=3, color=m.zn + m.pb, colorscale="Inferno",
                                              colorbar=dict(title="Zn+Pb %"), cmin=0),
                                   name="Mineralized samples", hovertext=[f"Zn {a:.1f}% Pb {b:.1f}%"
                                                                          for a, b in zip(m.zn, m.pb)]))
    if candidates is not None and len(candidates):
        sel = set(selected_candidates or [])
        for grp, cands, color, width, dash in (("exec", candidates[candidates.hole_id.isin(executed)], "#555",
                                                 1, "dot"),
                                                ("sel", candidates[candidates.hole_id.isin(sel) & ~candidates
                                                                   .hole_id.isin(executed)], ACCENT, 5, "solid"),
                                                ("other", candidates[~candidates.hole_id.isin(sel | executed)],
                                                 "#3d4452", 1, "dot")):
            first = True
            for r in cands.itertuples():
                end = np.array([r.x, r.y, r.z]) + r.length * np.array([
                    np.sin(np.radians(r.azimuth)) * np.cos(np.radians(r.dip)),
                    np.cos(np.radians(r.azimuth)) * np.cos(np.radians(r.dip)), np.sin(np.radians(r.dip))])
                fig.add_trace(go.Scatter3d(x=[r.x, end[0]], y=[r.y, end[1]], z=[r.z, end[2]], mode="lines",
                                           line=dict(color=color, width=width, dash=dash),
                                           name={"exec": "Executed candidates", "sel": "Selected candidate",
                                                "other": "Candidate holes"}[grp],
                                           legendgroup=grp, showlegend=first, hovertext=r.hole_id, hoverinfo="text"))
                first = False
    fig.update_layout(**Z_SCENE, height=650, title="3D Exploration Model")
    return fig


def lithology_volume_figure(grid, lithology: np.ndarray, opacity: float = 0.12) -> go.Figure:
    fig = go.Figure()
    X, Y, Z = grid.mesh()
    for code, name in enumerate(LITH_NAMES):
        if code == 4:
            continue  # background rock omitted for clarity
        mask = lithology == code
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(x=X[mask][::7], y=Y[mask][::7], z=Z[mask][::7], mode="markers",
                                   marker=dict(size=2, color=LITH_COLORS[code], opacity=opacity), name=name))
    fig.update_layout(**Z_SCENE, height=600, title="Lithology (debug — hidden truth)")
    return fig


def isosurface_figure(grid, values: np.ndarray, title: str, colorscale: str = "Inferno", isomin=None,
                      isomax=None, surface_count: int = 6, opacity: float = 0.25) -> go.Figure:
    X, Y, Z = grid.mesh()
    v = values
    fig = go.Figure(go.Volume(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=v.ravel(),
                              isomin=isomin if isomin is not None else float(np.quantile(v, 0.6)),
                              isomax=isomax if isomax is not None else float(v.max()),
                              opacity=opacity, surface_count=surface_count, colorscale=colorscale,
                              caps=dict(x_show=False, y_show=False, z_show=False)))
    fig.update_layout(**Z_SCENE, height=600, title=title)
    return fig


def depth_slice_figure(grid, values: np.ndarray, depths: list[float], title: str, colorscale="Inferno",
                       zmin=None, zmax=None) -> go.Figure:
    import plotly.subplots as sp

    z = grid.z
    idxs = [int(np.argmin(np.abs(z - d))) for d in depths]
    fig = sp.make_subplots(rows=2, cols=4, subplot_titles=[f"{d:.0f} m" for d in depths])
    zmin = zmin if zmin is not None else float(values.min())
    zmax = zmax if zmax is not None else float(values.max())
    for i, (d, k) in enumerate(zip(depths, idxs)):
        r, c = divmod(i, 4)
        fig.add_trace(go.Heatmap(z=values[:, :, k].T, colorscale=colorscale, zmin=zmin, zmax=zmax,
                                 showscale=(i == 0)), row=r + 1, col=c + 1)
    fig.update_layout(height=520, title=title)
    return fig
