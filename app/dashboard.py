"""Angouran Intelligent Exploration Simulator — interactive Streamlit dashboard.

Run with:  streamlit run app/dashboard.py   (or  python main.py dashboard)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.components.prediction_cache import volume_predictions  # noqa: E402
from app.components.state import create_experiment, get_experiment, invalidate_predictions, \
    list_experiments  # noqa: E402
from src.evaluation.metrics import evaluate_model, validation_samples  # noqa: E402
from src.optimization.acquisition import STRATEGIES  # noqa: E402
from src.uncertainty.calibration import reliability_curve  # noqa: E402
from src.utils.config import SCENARIO_DESCRIPTIONS  # noqa: E402
from src.visualization import theme  # noqa: E402
from src.visualization.plots2d import reliability_figure, training_curves, variogram_figure  # noqa: E402
from src.visualization.scene3d import depth_slice_figure, drill_holes_figure, isosurface_figure, \
    lithology_volume_figure  # noqa: E402
from src.visualization.theme import BG  # noqa: E402

st.set_page_config(page_title="Angouran Intelligent Exploration Simulator", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>.stApp {{background-color: {BG};}}</style>", unsafe_allow_html=True)

DEPTH_SLICES = [25, 50, 75, 100, 125, 150, 175, 200]

# ---------------------------------------------------------------- sidebar: experiment control
st.sidebar.title("🪨 Angouran Simulator")
st.sidebar.caption("Synthetic research prototype — not real Angouran geology.")

experiments = list_experiments()
mode = st.sidebar.radio("Experiment", ["Load existing", "Create new"], horizontal=True)
if mode == "Create new":
    new_id = st.sidebar.text_input("Experiment ID", value="demo01")
    profile = st.sidebar.selectbox("Profile", ["demo", "research", "benchmark"])
    seed = st.sidebar.number_input("Seed", value=20261019, step=1)
    if st.sidebar.button("Initialize world (generates + freezes truth)", type="primary"):
        try:
            create_experiment(new_id, profile, int(seed))
            st.sidebar.success(f"Created '{new_id}'.")
            st.rerun()
        except FileExistsError as e:
            st.sidebar.error(str(e))
    st.stop()

if not experiments:
    st.info("No experiments yet. Use 'Create new' in the sidebar to generate a synthetic world.")
    st.stop()

exp_id = st.sidebar.selectbox("Select experiment", experiments,
                              index=experiments.index(st.session_state.get("active_exp_id", experiments[0]))
                              if st.session_state.get("active_exp_id") in experiments else 0)
st.session_state["active_exp_id"] = exp_id
exp = get_experiment(exp_id)
debug_mode = st.sidebar.toggle("🔓 Scientific diagnostics (debug — hidden truth)", value=False)

st.sidebar.markdown("---")
st.sidebar.metric("Budget remaining", f"${exp.budget_remaining:,.0f}", delta=f"-${exp.budget_spent:,.0f} spent")
st.sidebar.metric("Virtual holes drilled", len(exp.oracle.executed) if exp.oracle else 0)
st.sidebar.caption(exp.gpu_info.summary())

st.title(f"Experiment: {exp_id}")
tabs = st.tabs(["A · Overview", "B · 3D Model", "C · AI Training", "D · Deep Prediction", "E · Drill Planning",
               "F · Virtual Drilling", "G · Model Evolution", "H · Evaluation", "I · Diagnostics"])

# ---------------------------------------------------------------- A. Overview
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid", f"{exp.cfg.world.shape[0]}×{exp.cfg.world.shape[1]}×{exp.cfg.world.shape[2]}")
    c2.metric("Historical holes", exp.truth.table("historical_collars").shape[0])
    c3.metric("Executed virtual holes", len(exp.oracle.executed))
    c4.metric("Candidates remaining", len(exp.obs.candidates()) - len(exp.oracle.executed))
    st.write(f"**Extent:** {exp.cfg.world.extent[0]:.0f} × {exp.cfg.world.extent[1]:.0f} × "
            f"{exp.cfg.world.extent[2]:.0f} m  |  **Mineralization cutoff:** Zn+Pb ≥ {exp.cfg.world.cutoff_zn_pb}%")
    if debug_mode:
        scen = exp.truth.meta()["meta"]["scenario"]
        st.warning(f"🔓 DEBUG: hidden scenario = **{scen}** — {SCENARIO_DESCRIPTIONS[scen]}. "
                  "This is never visible to the model, acquisition engine, or a non-debug user.")
    st.caption("Synthetic research prototype inspired by general Zn-Pb carbonate-hosted deposit geology; "
              "NOT the real geology of Angouran Mine. No performance claim here transfers to real exploration.")
    if exp.history:
        st.subheader("Drilling history")
        st.dataframe(pd.DataFrame([{"iteration": h.index, "hole_id": h.drilled_hole_id,
                                    "cost_usd": h.drill_summary["cost_usd"] if h.drill_summary else None,
                                    "budget_remaining": h.budget_remaining} for h in exp.history]),
                    use_container_width=True)

# ---------------------------------------------------------------- B. 3D Model
with tabs[1]:
    st.subheader("Interactive 3D exploration model")
    collars, samples, cands = exp.obs.collars(), exp.obs.samples(), exp.obs.candidates()
    show_assays = st.checkbox("Show mineralized sample points", True)
    fig = drill_holes_figure(collars, samples, cands, executed=exp.oracle.executed, show_assays=show_assays)
    st.plotly_chart(fig, use_container_width=True)
    if debug_mode:
        st.warning("🔓 DEBUG — hidden lithology volume (never used by training/acquisition)")
        world = exp.truth.load_world()
        st.plotly_chart(lithology_volume_figure(world.grid, world.lithology), use_container_width=True)

# ---------------------------------------------------------------- C. AI Training
with tabs[2]:
    st.subheader("GPU model training")
    st.write(f"**Device:** {exp.device} — {exp.gpu_info.device_name or 'CPU'}")
    colA, colB = st.columns([1, 2])
    with colA:
        warm = st.checkbox("Warm-start (finetune) instead of retrain from scratch", value=False)
        if st.button("Train / Retrain ensemble", type="primary"):
            with st.spinner("Training ensemble on GPU..."):
                info = exp.train(warm_start=warm)
            st.session_state["last_train_info"] = info
            invalidate_predictions(exp_id)
            st.success(f"Trained in {info['seconds']:.2f}s on {info['n_samples']} samples "
                      f"({info['n_holes']} holes).")
    info = st.session_state.get("last_train_info")
    if info:
        st.plotly_chart(training_curves(info["history"]), use_container_width=True)
        st.write("Per-member best validation loss:", [round(v, 3) for v in info["final_val_loss"]])
    elif exp.model is None:
        st.info("Model not yet trained for this session. Click 'Train / Retrain ensemble'.")

# ---------------------------------------------------------------- D. Deep Prediction
with tabs[3]:
    st.subheader("Prediction for the 100-200 m blind interval")
    if exp.model is None:
        st.info("Train the model first (tab C).")
    else:
        vol = volume_predictions(exp, exp_id, len(exp.history))
        field = st.selectbox("Field", ["p_mean (mineralization probability)", "zn_mean", "pb_mean",
                                       "zn_epi_sd (epistemic uncertainty)", "zn_alea_sd (aleatoric uncertainty)"])
        key = field.split(" ")[0]
        depths = st.multiselect("Depth slices (m)", DEPTH_SLICES, default=[100, 125, 150, 175, 200])
        if depths:
            st.plotly_chart(depth_slice_figure(vol["grid"], vol[key], sorted(depths), field), use_container_width=True)
        st.plotly_chart(isosurface_figure(vol["grid"], vol[key], f"3D volume — {field}"), use_container_width=True)
        st.caption("Interpolation (0-100 m, constrained by direct observations) vs. extrapolation "
                  "(100-200 m, inferred from spatial structure + surface geophysics only) — treat the "
                  "deep interval with appropriately higher caution.")

# ---------------------------------------------------------------- E. Drill Planning
with tabs[4]:
    st.subheader("Candidate drill-hole ranking")
    if exp.model is None:
        st.info("Train the model first (tab C).")
    else:
        strategy = st.selectbox("Acquisition strategy", STRATEGIES,
                                index=STRATEGIES.index(exp.cfg.acquisition.strategy))
        if st.button("Rank candidates"):
            with st.spinner("Scoring candidates on GPU..."):
                exp.rank_candidates(strategy)
        if exp.rankings is not None:
            cols = ["rank", "hole_id", "type", "x", "y", "azimuth", "dip", "length", "cost_usd",
                   "exp_mineralized_m", "p_hit_deep", "eivr_pct", "hypothesis_eig_nats", "score"]
            st.dataframe(exp.rankings[cols].head(20), use_container_width=True)
            top = exp.rankings.iloc[0]
            with st.expander(f"Reasoning for top candidate: {top.hole_id}", expanded=True):
                st.write(top.reasoning)
            hyp = exp.hypothesis_posterior()
            if hyp.get("available"):
                st.write("**Geological hypothesis posterior:**", hyp["posterior"])
        else:
            st.info("Click 'Rank candidates' to score the catalogue.")

# ---------------------------------------------------------------- F. Virtual Drilling
with tabs[5]:
    st.subheader("Execute virtual drilling")
    if exp.rankings is None or exp.model is None:
        st.info("Train and rank candidates first (tabs C, E).")
    else:
        choices = exp.rankings.hole_id.tolist()
        chosen = st.selectbox("Select a candidate to drill (override allowed)", choices)
        row = exp.rankings.loc[exp.rankings.hole_id == chosen].iloc[0]
        st.write(f"Cost: ${row.cost_usd:,.0f} | Length: {row.length:.0f} m | Type: {row.type} | "
                f"Score: {row.score:.3f}")
        confirm = st.checkbox(f"I confirm I want to drill {chosen} now (irreversible in this experiment)")
        if st.button("🔨 Execute Virtual Drill Hole", type="primary", disabled=not confirm):
            result = exp.drill(chosen, authorized=True)
            train_info = st.session_state.get("last_train_info", {"seconds": 0.0})
            exp.record_iteration(result, train_info)
            invalidate_predictions(exp_id)
            st.session_state["last_drill_result"] = result
            st.success(f"Drilled {chosen}. {result.summary()['deep_mineralized_m']:.1f} m mineralized "
                      f"below 100 m.")
        last = st.session_state.get("last_drill_result")
        if last is not None:
            st.write("**Lithology / assay log (frozen truth revealed by the oracle):**")
            st.dataframe(last.samples[["from_m", "to_m", "z", "lith_name", "zn", "pb", "density", "mineralized"]],
                        use_container_width=True)
            if last.mineralized_intervals:
                st.write("**Mineralized intervals:**", pd.DataFrame(last.mineralized_intervals))
            if last.structures:
                st.write("**Structural zones encountered:**", pd.DataFrame(last.structures))

# ---------------------------------------------------------------- G. Model Evolution
with tabs[6]:
    st.subheader("Before / after comparison across drilling iterations")
    if len(exp.history) < 1:
        st.info("Drill at least one hole to see model evolution.")
    else:
        val = validation_samples(exp.truth, exp.cfg.historical.observed_max_depth)
        rows = []
        for h in exp.history:
            rows.append({"iteration": h.index, "n_samples": h.n_samples, "n_holes": h.n_holes,
                        "budget_remaining": h.budget_remaining})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Re-run tab D after each drill to see the updated 3D prediction and uncertainty volumes; "
                  "re-run tab E to see updated candidate rankings. The cached volumes are invalidated "
                  "automatically on each drill + retrain.")

# ---------------------------------------------------------------- H. Evaluation
with tabs[7]:
    st.subheader("Model evaluation against permanently held-out validation holes")
    if exp.model is None:
        st.info("Train the model first (tab C).")
    else:
        val = validation_samples(exp.truth, exp.cfg.historical.observed_max_depth)
        metrics = evaluate_model(exp.model, val)
        c1, c2, c3 = st.columns(3)
        c1.metric("Extrapolation Zn MAE (100-200m)", f"{metrics.get('extrapolation_deep_zn_mae', float('nan')):.2f}")
        c2.metric("Extrapolation ROC-AUC", f"{metrics.get('extrapolation_deep_roc_auc', float('nan')):.3f}")
        c3.metric("Interval coverage (90%)", f"{metrics.get('extrapolation_deep_interval_coverage_90', float('nan')):.2f}")
        st.json(metrics)
        pts = val[["x", "y", "z"]].to_numpy(np.float32)
        pred = exp.model.predict_points(pts)
        p_mean = pred["p_mean"].detach().cpu().numpy()
        y = val.mineralized.to_numpy().astype(float)
        st.plotly_chart(reliability_figure(reliability_curve(p_mean, y), "Deep validation holes"),
                        use_container_width=True)
        if hasattr(exp, "_last_variogram"):
            st.plotly_chart(variogram_figure(exp._last_variogram), use_container_width=True)
        st.caption("Calibration and spatial robustness are measured on held-out drill holes that have never "
                  "been used for training or acquisition scoring. A result here is an empirical statistic "
                  "about this synthetic world, not a guarantee for other worlds or for the real deposit.")

# ---------------------------------------------------------------- I. Scientific Diagnostics
with tabs[8]:
    st.subheader("🔓 Evaluation-only hidden-truth diagnostics")
    if not debug_mode:
        st.info("Enable 'Scientific diagnostics' in the sidebar to view this tab. "
               "This view is for scientific evaluation only — it is never available to the training, "
               "prediction, or acquisition pipelines.")
    else:
        world = exp.truth.load_world()
        st.success(f"Hidden-truth integrity check: {'OK' if exp.truth.verify() else 'FAILED'}")
        meta = world.meta
        st.write(f"**Scenario:** {meta['scenario']} — {SCENARIO_DESCRIPTIONS[meta['scenario']]}")
        st.json({k: v for k, v in meta.items() if k not in ("lithology_names",)})
        field = st.selectbox("Hidden field", ["zn", "pb", "potential", "density"], key="debug_field")
        st.plotly_chart(isosurface_figure(world.grid, getattr(world, field), f"HIDDEN TRUTH — {field}",
                                          colorscale="Plasma"), use_container_width=True)
        st.plotly_chart(depth_slice_figure(world.grid, getattr(world, field), DEPTH_SLICES,
                                           f"HIDDEN TRUTH depth slices — {field}"), use_container_width=True)
