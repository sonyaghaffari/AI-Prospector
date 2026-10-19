"""2D diagnostic figures: training curves, calibration, variograms, uncertainty maps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from src.visualization.theme import ACCENT, GOOD, WARN


def training_curves(history: list[dict]) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.epoch, y=df.train_loss, name="Train loss", line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=df.epoch, y=df.val_loss, name="Val loss (hole-grouped)", line=dict(color=WARN)))
    fig.update_layout(title="Training / validation loss", xaxis_title="Epoch", yaxis_title="Loss", height=320)
    return fig


def reliability_figure(curve: dict, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#555"), name="Perfect calibration"))
    fig.add_trace(go.Scatter(x=curve["confidence"], y=curve["observed_freq"], mode="markers+lines",
                             marker=dict(size=8 + 20 * curve["count"] / max(curve["count"].max(), 1), color=ACCENT),
                             name=label))
    fig.update_layout(title=f"Reliability curve — {label}", xaxis_title="Predicted probability",
                      yaxis_title="Observed frequency", height=360, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig


def variogram_figure(est: dict) -> go.Figure:
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=["Horizontal", "Vertical"])
    for i, (key, rng) in enumerate((("horizontal", est.get("range_h")), ("vertical", est.get("range_v")))):
        d = est[key]
        fig.add_trace(go.Scatter(x=d["lag"], y=d["gamma"], mode="markers",
                                 marker=dict(size=6 + 20 * np.array(d["count"]) / max(max(d["count"]), 1)),
                                 name=key), row=1, col=i + 1)
        if rng:
            fig.add_vline(x=rng, line_dash="dash", line_color=WARN, row=1, col=i + 1)
    fig.update_layout(height=320, title="Experimental variogram of log1p(Zn)",
                      xaxis_title="Lag (m)", xaxis2_title="Lag (m)", yaxis_title="semivariance")
    return fig


def metric_evolution_figure(history_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df[metric], mode="lines+markers",
                             line=dict(color=ACCENT), marker=dict(size=9)))
    fig.update_layout(title=title, xaxis_title="Drilling iteration", height=320)
    return fig


def strategy_comparison_figure(summary: pd.DataFrame, metric_mean: str, metric_std: str, ylabel: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=summary.strategy, y=summary[metric_mean],
                           error_y=dict(type="data", array=summary[metric_std].fillna(0)),
                           marker_color=[ACCENT if s not in ("random", "regular_grid") else "#555"
                                        for s in summary.strategy]))
    fig.update_layout(title=ylabel, yaxis_title=ylabel, height=380)
    return fig


def gauge_uncertainty_map(gx: np.ndarray, gy: np.ndarray, values: np.ndarray, title: str, colorscale="Viridis") -> go.Figure:
    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=values.T, colorscale=colorscale))
    fig.update_layout(title=title, xaxis_title="X (m)", yaxis_title="Y (m)", height=450,
                      yaxis=dict(scaleanchor="x"))
    return fig
