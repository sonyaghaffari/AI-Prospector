"""Geological hypothesis engine — a clearly labelled APPROXIMATION of the hierarchical
Bayesian reasoning in Intelligent Prospector v2 (Mern & Caers et al.), not a reproduction.

Forward model (shared by all hypotheses)
  An ore shoot is a plunging tube whose axis x(z), y(z) is fitted by weighted least squares
  to the depth-binned centroids of mineralized samples in the shallow (<= 100 m) data. Its
  cross-section is an elliptical Gaussian whose radii / orientation come from the shallow
  spread. The probability that a sample at (x, y, z) is mineralized is
      p = eps + (p0 * I(z) - eps) * exp(-0.5 * [(u / r_u)^2 + (v / r_v)^2])
  with background rate eps.

Hypotheses differ only below 100 m (so the shallow data cannot discriminate them):
  H1 continuation      axis follows the shallow trend, I(z) = 1 (0 extra free parameters)
  H2 orientation change  below z_b ~ U(100,140) the axis acquires an extra horizontal
                         drift Delta (|Delta| ~ U(0.4,1.2) m/m, random azimuth)
                         (3 extra free parameters: z_b, azimuth, |Delta|)
  H3 compartmentalized   below z_c ~ U(100,140) the shoot is offset 60-200 m along the
                         interpreted main-fault strike (either sense)
                         (2 extra free parameters: z_c, signed offset)
  H4 termination         I(z) = sigmoid((z_t - z) / 6), z_t ~ U(100,170)
                         (1 extra free parameter: z_t)

Complexity correction: a 20-world benchmark found H1 losing systematically (0/5) even in
worlds generated as pure continuation, because real axes always wander a little and H2/H3's
extra free parameters let them fit that incidental wander better than H1's perfectly rigid
line, in a small-sample (10-31 composited observations) regime -- correct grid-quadrature
Bayesian averaging *should* penalize unneeded flexibility (that is the point of integrating
over the full prior rather than just the best fit), but with this little data the effect was
too weak. We tried giving H1 a small matched flexibility budget first; measured result: no
reliable improvement, because the same dilution effect that should penalize H2/H3 also dilutes
H1's own average once it has extra freedom. The fix actually used is an explicit BIC-style
penalty of 0.5 * (extra DOF) * log(n_composites) subtracted from each hypothesis's log-evidence
before normalizing -- a standard, well-established device for exactly this small-sample
regime, applied uniformly by DOF count rather than tuned per hypothesis. See EXTRA_DOF below.

Posterior: P(H | D) is proportional to P(H) * mean over prior particles of P(D_deep | theta),
a Monte-Carlo estimate of the marginal likelihood. D_deep are mineralized/barren indicators
of revealed samples below 100 m, composited to 10 m to reduce (not remove) the effect of
the independence assumption. With no deep data the posterior equals the uniform prior —
no probabilities are invented.

Limitations: one shoot only; a concealed separate body (scenario D) is outside the model
space; parameters of the shallow fit are treated as known (empirical Bayes).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from src.utils.config import HypothesisConfig

HYPOTHESES = {
    "H1": "Continued mineralization at depth",
    "H2": "Change in mineralization orientation",
    "H3": "Structural compartmentalization (fault offset)",
    "H4": "Termination of mineralization",
}


class HypothesisEngine:
    def __init__(self, cfg: HypothesisConfig, device: torch.device, seed: int, split_depth: float = 100.0) -> None:
        self.cfg, self.device, self.seed, self.split = cfg, device, seed, split_depth
        self.ok = False
        self.reason = "not fitted"
        self.fit_params: dict = {}

    # ------------------------------------------------------------------ shallow fit
    def fit(self, samples: pd.DataFrame, fault_strike_xy: np.ndarray) -> "HypothesisEngine":
        sh = samples[samples.z <= self.split]
        mn = sh[sh.mineralized.astype(bool)]
        if len(mn) < 15:
            self.ok, self.reason = False, f"only {len(mn)} shallow mineralized samples"
            return self
        xy = mn[["x", "y"]].to_numpy()
        # dominant cluster: kernel-density mode of mineralized sample locations
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        mode = xy[np.argmax(np.exp(-d2 / (2 * 80.0**2)).sum(1))]
        mn = mn[np.hypot(*(xy - mode).T) < 200]
        bins = np.arange(0, self.split + 1e-6, 10.0)
        zc, cx, cy, w = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = mn[(mn.z > lo) & (mn.z <= hi)]
            if len(b) >= 3:
                zc.append(b.z.mean()); cx.append(b.x.mean()); cy.append(b.y.mean()); w.append(len(b))  # noqa: E702
        if len(zc) < 3:
            self.ok, self.reason = False, "shallow mineralization does not define a depth trend"
            return self
        A = np.column_stack([np.ones(len(zc)), zc]) * np.sqrt(w)[:, None]
        bx = np.linalg.lstsq(A, np.array(cx) * np.sqrt(w), rcond=None)[0]
        by = np.linalg.lstsq(A, np.array(cy) * np.sqrt(w), rcond=None)[0]
        res = mn[["x", "y"]].to_numpy() - np.column_stack([bx[0] + bx[1] * mn.z, by[0] + by[1] * mn.z])
        evals, evecs = np.linalg.eigh(np.cov(res.T))
        r_major, r_minor = np.sqrt(np.clip(evals[::-1], 25.0**2, None))
        phi = math.atan2(evecs[1, -1], evecs[0, -1])
        # p0: mineralized fraction inside the 1-sigma ellipse at 50-100 m
        mid = sh[sh.z > 50]
        d = mid[["x", "y"]].to_numpy() - np.column_stack([bx[0] + bx[1] * mid.z, by[0] + by[1] * mid.z])
        u = d @ np.array([math.cos(phi), math.sin(phi)])
        v = d @ np.array([-math.sin(phi), math.cos(phi)])
        inside = (u / r_major) ** 2 + (v / r_minor) ** 2 < 1
        p0 = float(np.clip(mid.mineralized[inside].mean() if inside.sum() >= 5 else 0.5, 0.2, 0.95))
        strike = np.asarray(fault_strike_xy, dtype=np.float64)
        self.fit_params = dict(ax=bx.tolist(), ay=by.tolist(), r_major=float(r_major), r_minor=float(r_minor),
                               phi=phi, p0=p0, strike=strike.tolist(), n_mineralized=int(len(mn)))
        self._make_particles()
        self.ok, self.reason = True, "ok"
        return self

    def _make_particles(self) -> None:
        """Build each hypothesis's evaluation points by FULL deterministic grid quadrature
        (every parameter, not random/quasi-random sampling.

        Diagnosed empirically, in order: (1) plain i.i.d. sampling gave a run-to-run posterior
        std of ~0.10-0.12 even at tens of thousands of particles; (2) replacing only the
        hypothesis-specific parameter with a grid did not fix it; (3) switching all sampling to
        low-discrepancy Sobol sequences did not fix it either. The actual cause, measured
        directly: the effective sample size of the importance weights was ~2-4 out of 8192
        particles per hypothesis (one single particle routinely carried 50-65% of the total
        weight) -- the likelihood is sharply peaked relative to how wide the prior is, so no
        amount of i.i.d. or quasi-random sampling from that wide prior reliably lands particles
        near the peak; which particle "got lucky" is what was driving the instability, for every
        sampling method tried.

        Full grid quadrature has no sampling variance at all, regardless of how peaked the
        likelihood is: every cell contributes with its fixed, known quadrature weight, so two
        runs with the same configuration are now numerically identical. Its only failure mode is
        bias from a grid too coarse to resolve a very narrow peak -- a controllable, inspectable
        property of the chosen resolution, not randomness.
        """
        f = self.fit_params
        strike = torch.tensor(f["strike"], dtype=torch.float64)

        def grid_u(lo: float, hi: float, k: int) -> torch.Tensor:
            return lo + (torch.arange(k, dtype=torch.float64) + 0.5) / k * (hi - lo)  # cell midpoints

        def grid_n(mu: float, sd: float, k: int) -> torch.Tensor:
            q = (torch.arange(k, dtype=torch.float64) + 0.5) / k  # midpoint quantiles of N(mu, sd)
            return mu + sd * math.sqrt(2) * torch.erfinv(2 * q - 1)

        # shared nuisance grid: slope_x, slope_y (tight Gaussian around the shallow fit), scale,
        # p0 (both uniform) -- common to all 4 hypotheses, 4*4*3*2 = 96 combinations
        sx = grid_n(f["ax"][1], 0.15, 4)
        sy = grid_n(f["ay"][1], 0.15, 4)
        sc = grid_u(0.8, 1.25, 3)
        p0g = (f["p0"] * grid_u(0.7, 1.1, 2)).clamp(0.05, 0.98)
        shared_idx = torch.cartesian_prod(sx, sy, sc, p0g)  # (300, 4)
        n_shared = shared_idx.shape[0]

        def expand_shared(extra: torch.Tensor | None) -> dict[str, torch.Tensor]:
            """Cross the shared grid with an optional (n_extra, k) grid of extra columns."""
            n_extra = 1 if extra is None else extra.shape[0]
            sh = shared_idx.repeat_interleave(n_extra, dim=0)
            ex = extra.repeat(n_shared, 1) if extra is not None else None
            return sh, ex, n_shared * n_extra

        zeros = lambda n: torch.zeros(n, dtype=torch.float64)  # noqa: E731
        const = lambda n, v: torch.full((n,), float(v), dtype=torch.float64)  # noqa: E731
        groups: dict[int, dict[str, torch.Tensor]] = {}

        # H1 (continuation): no extra free parameter -- see _hyp_logpost / EXTRA_DOF for how its
        # rigidity is now made fair against H2/H3's flexibility (an explicit complexity penalty,
        # not extra parameters for H1 itself: giving H1 a matched flexible-bend parameterization
        # was tried and measured to be counterproductive, since the SAME Occam's-razor dilution
        # that should penalize H2/H3's extra freedom also dilutes H1's own average once it has
        # extra freedom, for no net gain -- see git history / the module docstring for the full
        # diagnosis).
        sh, _, n1 = expand_shared(None)
        groups[0] = dict(slope_x=sh[:, 0], slope_y=sh[:, 1], scale=sh[:, 2], p0=sh[:, 3], z_break=const(n1, 100),
                         dx=zeros(n1), dy=zeros(n1), ox=zeros(n1), oy=zeros(n1), z_t=const(n1, 1e9))

        # H2 (direction change): extra grid over (z_break x angle x magnitude), 5*8*3 = 120
        zb_g = grid_u(100, 140, 5)
        ang_g = grid_u(0, 2 * math.pi, 8)
        mag_g = grid_u(0.4, 1.2, 3)
        extra2 = torch.cartesian_prod(zb_g, ang_g, mag_g)
        sh, ex, n2 = expand_shared(extra2)
        groups[1] = dict(slope_x=sh[:, 0], slope_y=sh[:, 1], scale=sh[:, 2], p0=sh[:, 3], z_break=ex[:, 0],
                         dx=ex[:, 2] * torch.cos(ex[:, 1]), dy=ex[:, 2] * torch.sin(ex[:, 1]),
                         ox=zeros(n2), oy=zeros(n2), z_t=const(n2, 1e9))

        # H3 (fault offset): extra grid over (z_break x signed offset), 5*10 = 50
        off_g = torch.cat([-grid_u(60, 200, 5).flip(0), grid_u(60, 200, 5)])  # 10 points, both signs
        extra3 = torch.cartesian_prod(zb_g, off_g)
        sh, ex, n3 = expand_shared(extra3)
        groups[2] = dict(slope_x=sh[:, 0], slope_y=sh[:, 1], scale=sh[:, 2], p0=sh[:, 3], z_break=ex[:, 0],
                         dx=zeros(n3), dy=zeros(n3), ox=ex[:, 1] * strike[0], oy=ex[:, 1] * strike[1],
                         z_t=const(n3, 1e9))

        # H4 (termination): extra grid over termination depth, 12 points
        extra4 = grid_u(100, 170, 12).unsqueeze(1)
        sh, ex, n4 = expand_shared(extra4)
        groups[3] = dict(slope_x=sh[:, 0], slope_y=sh[:, 1], scale=sh[:, 2], p0=sh[:, 3], z_break=const(n4, 120),
                         dx=zeros(n4), dy=zeros(n4), ox=zeros(n4), oy=zeros(n4), z_t=ex[:, 0])

        self.group_sizes = [n1, n2, n3, n4]
        keys = ("slope_x", "slope_y", "scale", "p0", "z_break", "dx", "dy", "ox", "oy", "z_t")
        cat = {k: torch.cat([groups[h][k] for h in range(4)]) for k in keys}
        hyp = torch.cat([torch.full((self.group_sizes[h],), h, dtype=torch.long) for h in range(4)])
        self.particles = {k: v.to(self.device, torch.float32) for k, v in cat.items()}
        self.particles["hyp"] = hyp.to(self.device)

    # ------------------------------------------------------------------ forward model
    def prob(self, pts: torch.Tensor) -> torch.Tensor:
        """pts (..., 3) -> mineralization probability (S, ...)."""
        P, f = self.particles, self.fit_params
        shp = (-1,) + (1,) * (pts.dim() - 1)
        x, y, z = pts[..., 0][None], pts[..., 1][None], pts[..., 2][None]
        g = lambda k: P[k].view(shp)  # noqa: E731
        deep = (z - g("z_break")).clamp(min=0)
        below = (z > g("z_break")).float()
        axx = f["ax"][0] + g("slope_x") * z + deep * g("dx") + below * g("ox")
        axy = f["ay"][0] + g("slope_y") * z + deep * g("dy") + below * g("oy")
        dx, dy = x - axx, y - axy
        c, s = math.cos(f["phi"]), math.sin(f["phi"])
        uu, vv = dx * c + dy * s, -dx * s + dy * c
        sc = g("scale")
        shape = torch.exp(-0.5 * ((uu / (f["r_major"] * sc)) ** 2 + (vv / (f["r_minor"] * sc)) ** 2))
        inten = torch.sigmoid((g("z_t") - z) / 6.0)
        eps = self.cfg.background_prob
        return (eps + (g("p0") * inten - eps).clamp(min=0) * shape).clamp(1e-4, 1 - 1e-4)

    # ------------------------------------------------------------------ inference
    def composite_deep(self, samples: pd.DataFrame) -> pd.DataFrame:
        d = samples[samples.z > self.split]
        if d.empty:
            return d
        key = (d.depth_along // self.cfg.composite_length).astype(int)
        return d.groupby([d.hole_id, key]).agg(
            x=("x", "mean"), y=("y", "mean"), z=("z", "mean"), y_obs=("mineralized", "mean"),
            n=("mineralized", "size"), k=("mineralized", "sum")).reset_index(drop=True)

    def particle_log_weights(self, samples: pd.DataFrame) -> torch.Tensor:
        """Binomial likelihood per composite: k of n samples mineralized, shared probability p.
        Using the full (k, n) count rather than thresholding each composite to a single 0/1
        observation keeps the per-sample evidence (a 9-of-10 composite is much stronger evidence
        than a 6-of-10 one) instead of discarding it -- compositing still bounds how much
        correlated, non-independent evidence any one short interval can contribute."""
        logw = torch.cat([torch.full((sz,), math.log(0.25 / sz), device=self.device)
                         for sz in self.group_sizes])
        comp = self.composite_deep(samples)
        if len(comp):
            pts = torch.as_tensor(comp[["x", "y", "z"]].to_numpy(np.float32), device=self.device)
            k = torch.as_tensor(comp.k.to_numpy(np.float32), device=self.device)
            cnt = torch.as_tensor(comp.n.to_numpy(np.float32), device=self.device)
            p = self.prob(pts).clamp(1e-4, 1 - 1e-4)
            logw = logw + (k * torch.log(p) + (cnt - k) * torch.log1p(-p)).sum(-1)
        return logw

    # Extra free parameters per hypothesis beyond the 4 shared ones (slope_x, slope_y, scale,
    # p0), in HYPOTHESES order (H1..H4): H1 has none; H2 searches bend-depth, azimuth and
    # magnitude; H3 searches offset-depth and signed offset; H4 searches termination depth.
    # Used by the BIC-style complexity correction in _hyp_logpost -- see the module docstring.
    EXTRA_DOF = (0, 3, 2, 1)

    def _hyp_logpost(self, logw: torch.Tensor, n_comp: int) -> torch.Tensor:
        """Posterior over the 4 hypotheses, with a BIC-style penalty of 0.5 * dof * log(n_comp)
        subtracted from each hypothesis's raw grid-quadrature log-evidence before normalizing.

        Grid quadrature already applies *some* Occam's-razor correction on its own (extra free
        parameters mean averaging over more, mostly-mediocre-fit grid cells), but a 20-world
        benchmark found that alone was not enough with only 10-31 composited observations: H2/H3's
        extra flexibility let them out-fit H1's rigid line even in worlds where H1 was the true
        story, because a few metres of ordinary geological wander (present in every world, not
        just "direction change" ones) gave the flexible hypotheses real, if accidental, extra
        fit quality. The BIC term is a standard, independent correction for exactly this
        small-sample regime, scaled by how many extra parameters each hypothesis actually has."""
        h = self.particles["hyp"]
        lp = torch.stack([torch.logsumexp(logw[..., h == k], dim=-1) for k in range(4)], dim=-1)
        if n_comp > 0:
            penalty = lp.new_tensor([0.5 * d * math.log(n_comp) for d in self.EXTRA_DOF])
            lp = lp - penalty
        return lp - torch.logsumexp(lp, dim=-1, keepdim=True)

    def posterior(self, samples: pd.DataFrame) -> dict:
        if not self.ok:
            return {"available": False, "reason": self.reason}
        logw = self.particle_log_weights(samples)
        n_comp = len(self.composite_deep(samples))
        post = torch.exp(self._hyp_logpost(logw, n_comp)).cpu().numpy()
        return {"available": True, "posterior": {k: float(p) for k, p in zip(HYPOTHESES, post)},
                "n_deep_composites": int(n_comp), "fit": self.fit_params,
                "p_continuation": float(post[0] + post[1] + post[2])}

    @torch.no_grad()
    def expected_information_gain(self, samples: pd.DataFrame, cand_pts: torch.Tensor, cand_mask: torch.Tensor,
                                  chunk: int = 32) -> torch.Tensor:
        """EIG (nats) about the hypothesis variable for each candidate, by Monte Carlo
        simulation of deep outcomes from the current posterior predictive."""
        if not self.ok:
            return torch.zeros(cand_pts.shape[0], device=self.device)
        stride = max(1, int(round(self.cfg.composite_length / max(float(
            (cand_pts[0, 1] - cand_pts[0, 0]).norm()), 1e-6))))
        pts, mask = cand_pts[:, ::stride], cand_mask[:, ::stride] & (cand_pts[:, ::stride, 2] > self.split)
        logw = self.particle_log_weights(samples)
        n_comp = len(self.composite_deep(samples))
        w = torch.softmax(logw, 0)
        h0 = -(torch.exp(self._hyp_logpost(logw, n_comp)) * self._hyp_logpost(logw, n_comp)).sum()
        g = torch.Generator(device=self.device).manual_seed(self.seed + 7)
        out = []
        K = self.cfg.simulated_outcomes
        for i in range(0, pts.shape[0], chunk):
            p = self.prob(pts[i:i + chunk])  # (S, c, Q)
            m = mask[i:i + chunk].float()
            c = p.shape[1]
            sidx = torch.multinomial(w, c * K, replacement=True, generator=g).view(c, K)
            ps = p.permute(1, 0, 2)[torch.arange(c, device=self.device)[:, None], sidx]  # (c, K, Q)
            yv = torch.bernoulli(ps, generator=g) * m[:, None, :]
            lp = (torch.log(p) * m[None]).permute(1, 2, 0)  # (c, Q, S)
            l1 = (torch.log1p(-p) * m[None]).permute(1, 2, 0)
            ll = torch.bmm(yv, lp) + torch.bmm(m[:, None, :].expand(-1, K, -1) - yv, l1)  # (c, K, S)
            # the simulated composites add roughly stride/composite_length more observations;
            # approximated here with the CURRENT n_comp (a fixed, candidate-independent shift
            # that does not change the relative EIG ranking between candidates)
            hp = self._hyp_logpost(logw[None, None, :] + ll, n_comp)
            out.append(h0 - (-(torch.exp(hp) * hp).sum(-1)).mean(-1))
        return torch.cat(out).clamp(min=0)
