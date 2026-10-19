# Angouran Intelligent Exploration Simulator

A GPU-accelerated, synthetic-data mineral exploration simulator: 3D geological world
generation, a PyTorch deep-ensemble geological model, uncertainty-aware acquisition, a
simplified Bayesian hypothesis engine, a virtual drilling oracle, and an interactive
Streamlit dashboard — demonstrating the end-to-end sequential exploration workflow
conceptually inspired by KoBold Metals' *Intelligent Prospector* research.

> **This is a research prototype built entirely on synthetic data.** It is NOT the real
> geology of Angouran Mine, does not reproduce KoBold's proprietary software, and its
> performance on synthetic worlds is not evidence of real-world discovery capability. See
> [Limitations](#known-limitations-and-scientific-caveats).

## Quick start

```powershell
# 1. GPU / environment check (never silently falls back to CPU)
python main.py gpu-check

# 2. Create an experiment: generates and permanently freezes a synthetic world
python main.py new --exp-id demo01 --profile demo

# 3. Train the GPU ensemble on shallow (0-100 m) data only
python main.py train --exp-id demo01

# 4. Rank candidate drill holes targeting 100-200 m
python main.py rank --exp-id demo01 --strategy multi_objective

# 5. Execute a virtual drill hole (explicit authorization required)
python main.py drill --exp-id demo01 --hole-id CAND-012 --yes

# 6. Launch the interactive dashboard
python main.py dashboard
#  or: streamlit run app/dashboard.py

# 7. Run the required end-to-end demonstration (Section 27)
python scripts/run_demo_experiment.py --exp-id demo_end_to_end --profile demo --n-holes 3

# 8. Run the multi-world, multi-strategy benchmark
python main.py benchmark --profile benchmark
```

## GPU setup

This project needs CUDA PyTorch. It does **not** install or replace your existing PyTorch
without being told to — `python main.py gpu-check` only inspects and reports. If CUDA is
unavailable, it prints the exact diagnostic steps and refuses to silently run on CPU; pass
`--allow-cpu` (or set `ANGOURAN_ALLOW_CPU=1`) to opt in to slow CPU execution explicitly.

Verified in this environment: **NVIDIA GeForce RTX 2080 Ti**, driver 560.94 (CUDA 12.6),
PyTorch **2.14.1+cu126**, `torch.cuda.is_available() == True`. Install command used (project
venv, does not touch system Python):
```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\pip install -r requirements.txt
```

## Repository layout

```
app/                    Streamlit dashboard (dashboard.py) + session/cache helpers
src/geology/            Grid, spatially-correlated random fields, trajectories, world generator
src/drilling/           Historical holes, candidate catalogue, sample logging, drilling oracle
src/data/               ObservationStore (everything the AI is allowed to see) + real-data importer
src/models/             Feature builder, GPU deep-ensemble model, IDW/GP baselines
src/uncertainty/        Calibration diagnostics, experimental variogram estimation
src/optimization/       Hypothesis engine (Bayesian model comparison), acquisition scoring
src/simulation/         Hidden-truth store (hash-verified, write-once), world builder, Experiment loop
src/evaluation/         Metrics, strategy comparison, multi-world benchmark
src/visualization/      Shared dark theme, 3D scene figures, 2D diagnostic figures
configs/                demo.json / research.json / benchmark.json profiles
tests/                  pytest suite (leakage, reproducibility, GPU/OOM, geology, model updates)
scripts/                run_demo_experiment.py (Section 27 demo), export_figures.py
data/hidden_truth/      Frozen ground truth, read-only, SHA-256 manifest (never used by AI code)
data/observed/          Legitimately revealed observations (what the AI is trained on)
artifacts/              checkpoints / predictions / reports / figures, per experiment
```

## The sequential exploration loop

```
INITIALIZE WORLD → FREEZE TRUTH → REVEAL SHALLOW DATA (0-100 m) → TRAIN GPU ENSEMBLE
→ PREDICT TO 200 m → ESTIMATE UNCERTAINTY → RANK CANDIDATES → USER SELECTS A HOLE
→ ORACLE REVEALS FROZEN OBSERVATIONS → APPEND TO OBSERVED DATA → RETRAIN → REPEAT
```

`src/simulation/experiment.py:Experiment` is the only class that wires the model, the
acquisition engine, and the `SyntheticDrillingOracle` together. The hidden truth is
generated exactly once (`Experiment.create`, backed by `build_world`), written to
`data/hidden_truth/<exp_id>/` with every file and array SHA-256-hashed into a manifest and
made read-only, and verified before every oracle read. Drilling only happens through
`oracle.execute(hole_id, authorized=True)`, called by a human button-click in the dashboard,
the CLI (`--yes` required), or a benchmark driver — never automatically.

## What the model is allowed to see

Only `data/observed/<exp_id>/`: the shallow (≤100 m) portions of **training**-split
historical holes, surface gravity/IP surveys, mapped (not true) fault traces, and public
candidate geometry/cost. The permanently held-out **validation**-split historical holes and
every candidate's frozen outcome live only in `data/hidden_truth/`, read only by the oracle
(on authorized execution) and by `src/evaluation/metrics.py` (evaluation-only). This
separation is enforced by `tests/test_leakage.py`, including a static-AST check that
`src/models` and `src/optimization` never import the truth-store/generator/oracle modules.

## Model and acquisition design (brief)

- **Geological model**: a batched deep ensemble (default 5 members) of small coordinate
  networks — low-bandwidth random Fourier features of (x, y, z) + interpreted fault
  distances + surface geophysics → 3×128 SiLU MLP → multi-task heads (mineralization logit,
  heteroscedastic log-grade Gaussian for Zn/Pb, lithology). All members are trained in one
  batched-matmul pass on the GPU; validation splits are hole-grouped (no leakage between
  samples of the same hole) and independent per member.
- **Uncertainty**: ensemble variance = epistemic; per-member predicted log-variance =
  aleatoric. Calibration (ECE, Brier, reliability curves, interval coverage,
  error-vs-uncertainty Spearman correlation) is measured on held-out validation holes —
  never assumed.
- **Baselines**: GPU anisotropic IDW and an sklearn Gaussian Process (indicator + log-grade
  kriging analogue), both fit on the same observed samples, for honest comparison.
- **Acquisition**: three-plus strategies (`max_uncertainty`, `mineral_probability`, `eivr`,
  `eivr_per_cost`, `multi_objective`). EIVR is a labelled Gaussian-process surrogate for
  expected deep-variance reduction (exact conditioning on a per-candidate covariance built
  from ensemble epistemic std + an experimental-variogram correlation range), batched with
  Cholesky solves on the GPU — not exact information gain, and documented as such.
- **Hypothesis engine**: four geological hypotheses (continuation / orientation change /
  structural offset / termination) as an explicit, clearly-labelled *approximation* of
  hierarchical Bayesian reasoning — particle-filter forward model fit to shallow data,
  Monte-Carlo marginal-likelihood posterior, Monte-Carlo expected information gain about the
  hypothesis variable. With no deep data the posterior is exactly the uniform prior; nothing
  is fabricated.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
30 tests covering: truth generated/frozen once, read-only + SHA-256 tamper detection, no
deep/validation/candidate-outcome leakage into observed data, authorization required to
drill, no double-drilling, static leakage check, seed reproducibility, frozen-world reuse
across strategies, CUDA availability, OOM-safe batched inference, world/candidate geology
sanity, "predictions actually change after new data," and real-data importer schema
compliance. All 30 currently pass on this machine's RTX 2080 Ti.

## GPU verification (this run)

```
NVIDIA GeForce RTX 2080 Ti (sm_75) | VRAM 10.6/11.8 GB free | PyTorch 2.14.1+cu126
CUDA build 12.6 | driver 560.94 (CUDA 12.6) | cuDNN 91002
```

## Measured end-to-end demonstration (`demo_end_to_end`, seed 20261019, `demo` profile)

Produced by `scripts/run_demo_experiment.py`; full numbers in
`artifacts/reports/demo_end_to_end/end_to_end_demo_report.json`. These are **actual measured
values from one real run**, not illustrative numbers:

- World: 100 historical holes, 126 candidate holes, 3,596 initial shallow (≤100 m) samples,
  609 permanently held-out deep validation samples. Built + frozen in 2.6 s.
- 3 sequential drilling iterations, retraining the 5-member ensemble from scratch each time
  (1.6–2.1 s/iteration on the RTX 2080 Ti):
  - Iter 0: top-ranked `CAND-061` (multi-objective score 0.83) → **97.5 m** mineralized
    below 100 m, cost $70,076.
  - Iter 1: top-ranked `CAND-076` (score 0.77) → **90.0 m** mineralized below 100 m, cost
    $88,835.
  - Iter 2: top-ranked `CAND-072` (score 0.61) → **0.0 m** mineralized below 100 m (a
    genuine miss — not cherry-picked), cost $70,076.
- Final extrapolation (100–200 m) metrics against the untouched validation holes:
  Zn MAE ≈ 1.26, ROC-AUC ≈ 0.96 for mineralization classification.
- Hidden-truth manifest integrity: **verified** after all three drills.
- Total wall time for the full demonstration: 8.9 s.

Two of three holes were real hits and one was a real miss on this seed — reported as
measured, including the miss, consistent with Section 27's "do not fabricate results."

### Baseline comparison (`scripts/compare_baselines.py`, same experiment/validation set)

| method | fit+predict (s) | Zn MAE | Zn RMSE | Pb MAE | ECE | Brier | F1 |
|---|---|---|---|---|---|---|---|
| GPU Deep Ensemble | 0.12 | **1.54** | 3.11 | **0.41** | **0.080** | **0.074** | **0.83** |
| IDW (GPU, k=16) | 0.08 | 3.57 | 5.34 | 0.78 | 0.36 | 0.33 | 0.53 |
| Gaussian Process | 22.78 | 1.74 | **2.69** | 0.53 | 0.096 | 0.17 | 0.16 |

The ensemble wins on grade MAE, calibration, and classification F1; the GP baseline wins
on Zn RMSE (its smoother predictions are less penalized by the squared-error metric), at
~190× the compute cost. Neither baseline is uniformly worse, consistent with Section 9's
instruction not to assume the neural model automatically wins.

### Multi-strategy, multi-world benchmark

`configs/benchmark.json` trimmed to 2 worlds × 4 scenarios (A–D) × 5 strategies × 5 holes
= 200 drilling decisions, equal budget per world, same frozen world + same initial
observed data reused across all 5 strategies within a world (`src/evaluation/benchmark.py`).
Full detail: `artifacts/reports/benchmark_realbench.csv`; per-world cumulative table:
`artifacts/reports/benchmark_realbench_cumulative.csv`. Total wall time: 335 s (≈5.6 min)
on the RTX 2080 Ti, including building all 8 worlds and retraining the ensemble from
scratch at every one of the 200 iterations.

**Cumulative deep (100–200 m) mineralized metres discovered per 5-hole campaign**, mean ±
std over the 8 independent worlds:

| strategy | mean (m) | std (m) | Zn MAE (final) | ROC-AUC (final) | m per $1,000 |
|---|---|---|---|---|---|
| **multi_objective** | **205.3** | 131.7 | **0.777** | **0.982** | **0.536** |
| max_uncertainty | 147.5 | 61.8 | 0.825 | 0.975 | 0.383 |
| eivr_per_cost | 130.3 | 68.8 | 0.897 | 0.975 | 0.339 |
| regular_grid | 37.8 | 41.6 | 1.264 | 0.942 | 0.108 |
| random | 4.4 | 8.5 | 1.753 | 0.911 | 0.011 |

All three adaptive, model-driven strategies (`multi_objective`, `max_uncertainty`,
`eivr_per_cost`) clearly outperform the non-adaptive `regular_grid` and `random` baselines
on this synthetic benchmark — both in metres of mineralization actually intersected and in
final extrapolation accuracy — with `multi_objective` ahead on every column, including
cost efficiency. This is a real, unedited measurement from one benchmark run on one
machine; it is not evidence about real exploration economics (see Limitations), and
variance across worlds is substantial (note the std columns), so treat it as one
data point, not a guarantee.

## Known limitations and scientific caveats

- Synthetic geology only; not Angouran's actual subsurface. No claim of transferable
  real-world discovery performance.
- EIVR is a documented Gaussian-process surrogate for information gain, not an exact
  computation; the hypothesis engine is an explicit, simplified approximation of
  hierarchical Bayesian reasoning (one ore-shoot model; scenario D's separate concealed body
  is outside its model space).
- Ensemble disagreement is not a calibrated geological probability by assumption — it is
  checked against held-out holes, and calibration can and does vary by world/seed.
- Deterministic CUDA kernels reduce (not eliminate) run-to-run variation; exact bitwise
  reproducibility across different GPU/driver/PyTorch versions is not guaranteed.
- No claim that the deep ensemble outperforms IDW/GP in general — compare them yourself per
  world via `src/evaluation/metrics.py` and the benchmark.

## Next steps for real Angouran data integration

1. Convert real collar/survey/assay/lithology tables to the `ObservationStore` schema via
   `src/data/importers.py` (replace straight-hole desurvey with minimum-curvature for
   deviated holes).
2. Have a geologist validate the lithology code mapping and the Zn+Pb mineralization cutoff
   against real core logging conventions before trusting any prediction.
3. Replace the interpreted-fault-plane fit in `src/models/features.py` with real structural
   interpretation (seismic/mapping) once available.
4. Re-fit variogram ranges (`src/uncertainty/variogram.py`) and re-validate calibration on a
   real held-out drilling campaign before using acquisition scores operationally.
5. Treat every synthetic benchmark number in this repository as a sanity check on the
   *pipeline*, not as a predictor of real exploration economics.
