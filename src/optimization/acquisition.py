"""Candidate drill-hole scoring using ONLY the current model, observed data and public
candidate geometry/cost.

Expected uncertainty reduction (EIVR) — a labelled Gaussian SURROGATE, not exact EIG:
  The latent field is f(x) = log1p(Zn). Its prior covariance is modelled as
      C(a, b) = sigma_epi(a) * sigma_epi(b) * rho(a, b),
  where sigma_epi^2 is the current ensemble (epistemic) variance and rho is an
  anisotropic exponential correlation with ranges from the experimental variogram of the
  observed data. Drilling candidate c observes f at its trajectory points P with
  heteroscedastic noise tau^2 = ensemble-mean aleatoric variance. The posterior variance
  reduction at a target point t is the exact Gaussian conditioning result
      delta(t) = k_tP (K_PP + diag tau^2)^-1 k_Pt.
  EIVR_c = sum_t delta(t) / sum_t sigma_epi^2(t) over a fixed random subset of deep
  (100-200 m) voxels: the expected fractional reduction of deep epistemic variance.
  MI_c = 0.5 log det(I + diag(tau)^-1 K_PP diag(tau)^-1) is the matching mutual information.
  Batched over candidates with Cholesky solves on the GPU.

Strategies:
  max_uncertainty      mean epistemic std along the deep segment (classic, ignores
                       correlation, redundancy and cost)
  mineral_probability  expected mineralized metres in the deep segment (exploitation)
  eivr                 expected deep-variance reduction
  eivr_per_cost        EIVR / cost
  multi_objective      (w_e EIVR' + w_m minerals' + w_h hypothesis EIG') / relative cost,
                       primes = normalised by the best candidate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.ensemble import GeoModel
from src.optimization.hypotheses import HYPOTHESES, HypothesisEngine
from src.utils.config import AcquisitionConfig

STRATEGIES = ("max_uncertainty", "mineral_probability", "eivr", "eivr_per_cost", "multi_objective")


def _corr(a: torch.Tensor, b: torch.Tensor, rh: float, rv: float) -> torch.Tensor:
    """Exponential correlation between point sets a (..., n, 3) and b (..., m, 3)."""
    s = torch.tensor([1 / rh, 1 / rh, 1 / rv], device=a.device)
    return torch.exp(-torch.cdist(a * s, b * s))


@torch.no_grad()
def expected_variance_reduction(sig_p: torch.Tensor, tau2_p: torch.Tensor, pts: torch.Tensor, mask: torch.Tensor,
                                sig_t: torch.Tensor, tgt: torch.Tensor, rh: float, rv: float,
                                max_chunk_mb: float = 512.0) -> tuple[torch.Tensor, torch.Tensor]:
    """sig_p, tau2_p, mask: (C, P); pts (C, P, 3); sig_t (T,), tgt (T, 3). Returns (EIVR, MI) per candidate."""
    C, P = sig_p.shape
    T = tgt.shape[0]
    sig_p = torch.where(mask, sig_p, torch.zeros_like(sig_p))
    tau2_p = torch.where(mask, tau2_p.clamp(min=1e-4), torch.ones_like(tau2_p))
    chunk = max(1, int(max_chunk_mb * 1e6 / (4 * T * P * 3)))
    total = (sig_t**2).sum().clamp(min=1e-12)
    eivr, mi = [], []
    for i in range(0, C, chunk):
        sp, tp, pp = sig_p[i:i + chunk].double(), tau2_p[i:i + chunk].double(), pts[i:i + chunk]
        K = sp[:, :, None] * sp[:, None, :] * _corr(pp, pp, rh, rv).double()
        K = K + torch.diag_embed(tp)
        L = torch.linalg.cholesky(K)
        kPT = (sp[:, :, None] * _corr(pp, tgt.expand(pp.shape[0], -1, -1), rh, rv).double()
               * sig_t.double()[None, None, :])
        A = torch.linalg.solve_triangular(L, kPT, upper=False)
        eivr.append(((A**2).sum(dim=(1, 2)) / total).float())
        mi.append((torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1) - 0.5 * torch.log(tp).sum(1)).float())
    return torch.cat(eivr), torch.cat(mi)


@torch.no_grad()
def score_candidates(model: GeoModel, samples: pd.DataFrame, cands: pd.DataFrame, cand_pts: np.ndarray,
                     cand_mask: np.ndarray, target_pts: np.ndarray, ranges: tuple[float, float],
                     cfg: AcquisitionConfig, hyp: HypothesisEngine | None, executed: set[str],
                     split_depth: float = 100.0, step: float = 5.0) -> pd.DataFrame:
    dev = model.device
    C, P, _ = cand_pts.shape
    pts_t = torch.as_tensor(cand_pts, device=dev)
    mask_t = torch.as_tensor(cand_mask, device=dev)
    pr = model.predict_points(cand_pts.reshape(-1, 3))
    g = lambda k: pr[k].view(C, P)  # noqa: E731
    p_mean, p_std = g("p_mean"), g("p_std")
    deep = mask_t & (pts_t[..., 2] > split_depth)
    dsum = deep.sum(1).clamp(min=1)
    exp_min_m = (p_mean * deep).sum(1) * step
    p_hit = (pr["p_members"].view(-1, C, P) * deep).amax(2).mean(0)  # ensemble mean of max-point probability
    zn_d = (g("zn_mean") * deep).sum(1) / dsum
    pb_d = (g("pb_mean") * deep).sum(1) / dsum
    epi_sd = torch.sqrt(g("zn_epi_var"))
    max_unc = ((epi_sd + p_std) * deep).sum(1) / dsum

    tp = model.predict_points(target_pts)
    sig_t = torch.sqrt(tp["zn_epi_var"])
    eivr, mi = expected_variance_reduction(epi_sd, g("zn_alea_var"), pts_t, mask_t, sig_t,
                                           torch.as_tensor(target_pts, device=dev), *ranges)
    hyp_eig = (hyp.expected_information_gain(samples, pts_t, mask_t) if hyp is not None and hyp.ok
               else torch.zeros(C, device=dev))

    cost = torch.as_tensor(cands.cost_usd.to_numpy(np.float32), device=dev)
    nrm = lambda v: v / v.max().clamp(min=1e-12)  # noqa: E731
    rel_cost = cost / cost.mean()
    scores = {
        "max_uncertainty": max_unc,
        "mineral_probability": exp_min_m,
        "eivr": eivr,
        "eivr_per_cost": eivr / rel_cost,
        "multi_objective": (cfg.w_eivr * nrm(eivr) + cfg.w_mineral * nrm(exp_min_m)
                            + cfg.w_hypothesis * nrm(hyp_eig)) / rel_cost,
    }
    c = lambda v: v.detach().cpu().numpy()  # noqa: E731
    df = cands[["hole_id", "x", "y", "azimuth", "dip", "length", "type", "cost_usd"]].copy()
    df["p_hit_deep"] = c(p_hit)
    df["exp_mineralized_m"] = c(exp_min_m)
    df["exp_zn_pct"] = c(zn_d)
    df["exp_pb_pct"] = c(pb_d)
    df["mean_uncertainty"] = c(max_unc)
    df["eivr_pct"] = 100 * c(eivr)
    df["mutual_info_nats"] = c(mi)
    df["hypothesis_eig_nats"] = c(hyp_eig)
    for k, v in scores.items():
        df[f"score_{k}"] = c(v)
    df = df[~df.hole_id.isin(executed)].copy()
    key = f"score_{cfg.strategy}"
    df["score"] = df[key]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["reasoning"] = [explain(r, cfg.strategy) for r in df.itertuples()]
    return df


def explain(r, strategy: str) -> str:
    parts = [f"Strategy '{strategy}'.",
             f"Expected {r.exp_mineralized_m:.0f} m mineralized below 100 m (max-point P(hit) {r.p_hit_deep:.2f}).",
             f"Expected to remove {r.eivr_pct:.1f}% of deep epistemic grade variance (surrogate MI {r.mutual_info_nats:.1f} nats).",
             f"Mean deep uncertainty {r.mean_uncertainty:.2f}."]
    if r.hypothesis_eig_nats > 1e-3:
        parts.append(f"Hypothesis discrimination: {r.hypothesis_eig_nats:.3f} nats.")
    parts.append(f"Cost ${r.cost_usd:,.0f} ({r.type}, {r.length:.0f} m).")
    return " ".join(parts)


def regular_grid_order(cands: pd.DataFrame, extent, n: int) -> list[str]:
    """Non-adaptive baseline: vertical holes nearest to a regular pattern, fixed in advance."""
    v = cands[cands.type == "vertical"]
    k = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(k) + 0.5) / k * extent[0]
    ys = (np.arange(k) + 0.5) / k * extent[1]
    nodes = np.array([(x, y) for x in xs for y in ys])
    # farthest-point ordering so any prefix of the sequence is spatially spread
    seq = [int(np.argmin(np.hypot(*(nodes - np.asarray(extent[:2]) / 2).T)))]
    while len(seq) < len(nodes):
        dmin = np.min(np.linalg.norm(nodes[:, None] - nodes[seq][None], axis=-1), axis=1)
        dmin[seq] = -1
        seq.append(int(np.argmax(dmin)))
    order, used = [], set()
    for x, y in nodes[seq]:
        if True:
            d = np.hypot(v.x - x, v.y - y)
            for idx in np.argsort(d.to_numpy()):
                hid = v.hole_id.iloc[idx]
                if hid not in used:
                    used.add(hid)
                    order.append(hid)
                    break
    rest = [h for h in v.hole_id if h not in used]
    return order + rest


__all__ = ["STRATEGIES", "HYPOTHESES", "score_candidates", "regular_grid_order", "expected_variance_reduction"]
