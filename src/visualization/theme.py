"""Shared dark scientific theme for Plotly figures."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

BG = "#0e1117"
PANEL = "#161b22"
GRID = "#2a2f3a"
TEXT = "#e6e6e6"
ACCENT = "#4dd0e1"
WARN = "#ff8a65"
GOOD = "#81c784"

LITH_COLORS = {0: "#c9b458", 1: "#6d8ea8", 2: "#b06a5a", 3: "#a65ca6", 4: "#4a4f5a"}
LITH_NAMES = ("Marble", "Schist", "Breccia", "Altered host", "Background")

TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif", size=12),
        colorway=[ACCENT, WARN, GOOD, "#ba68c8", "#fff176"],
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        scene=dict(xaxis=dict(backgroundcolor=PANEL, gridcolor=GRID, color=TEXT),
                   yaxis=dict(backgroundcolor=PANEL, gridcolor=GRID, color=TEXT),
                   zaxis=dict(backgroundcolor=PANEL, gridcolor=GRID, color=TEXT)),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=40, b=10),
    )
)
pio.templates["angouran_dark"] = TEMPLATE
pio.templates.default = "angouran_dark"
