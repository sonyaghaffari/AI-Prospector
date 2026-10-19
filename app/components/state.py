"""Streamlit session-state helpers: list/create/load experiments, cached predictions."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.simulation.experiment import Experiment
from src.simulation.world_builder import ExperimentPaths
from src.utils.config import PROJECT_ROOT, load_config


def list_experiments() -> list[str]:
    d = PROJECT_ROOT / "data" / "synthetic_worlds"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json") if "__" not in p.stem)


def get_experiment(exp_id: str) -> Experiment:
    key = f"exp::{exp_id}"
    if key not in st.session_state:
        st.session_state[key] = Experiment.load(exp_id)
    return st.session_state[key]


def create_experiment(exp_id: str, profile: str, seed: int | None) -> Experiment:
    cfg = load_config(profile)
    if seed is not None:
        cfg.world.seed = seed
    exp = Experiment.create(cfg, exp_id)
    st.session_state[f"exp::{exp_id}"] = exp
    return exp


def invalidate_predictions(exp_id: str) -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(f"pred::{exp_id}") or k.startswith(f"rank::{exp_id}"):
            del st.session_state[k]
