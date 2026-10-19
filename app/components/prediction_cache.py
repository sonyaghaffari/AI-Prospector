"""Cache full-volume predictions / targets so slice and 3D views don't recompute the
ensemble forward pass on every Streamlit rerun."""
from __future__ import annotations

import numpy as np
import streamlit as st
import torch

from src.geology.grid import Grid
from src.simulation.experiment import Experiment


def volume_predictions(exp: Experiment, exp_id: str, iteration: int) -> dict[str, np.ndarray]:
    key = f"pred::{exp_id}::{iteration}"
    if key in st.session_state:
        return st.session_state[key]
    extent = tuple(exp.cfg.world.extent)
    shape = tuple(exp.cfg.world.shape)
    grid = Grid(extent, shape)
    pts = grid.points()
    with torch.no_grad():
        pred = exp.model.predict_points(pts, batch=exp.cfg.simulation.inference_batch)
    c = lambda t: t.detach().cpu().numpy().reshape(shape)  # noqa: E731
    out = {"grid": grid, "p_mean": c(pred["p_mean"]), "p_std": c(pred["p_std"]),
          "zn_mean": c(pred["zn_mean"]), "pb_mean": c(pred["pb_mean"]),
          "zn_epi_sd": c(torch.sqrt(pred["zn_epi_var"])), "zn_alea_sd": c(torch.sqrt(pred["zn_alea_var"])),
          "lith_prob": pred["lith_prob"].detach().cpu().numpy().reshape(*shape, -1)}
    st.session_state[key] = out
    return out
