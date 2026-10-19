"""Baseline spatial predictors, fitted on the same observed samples as the ensemble.

1. Anisotropic inverse-distance weighting (k nearest neighbours, GPU).
2. Gaussian-process regression (sklearn; anisotropic RBF + nugget, hyper-parameters by
   marginal likelihood) — a simple-kriging analogue. The mineralization probability is
   obtained by GP regression on the 0/1 indicator, clipped to [0, 1] (indicator-kriging style).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


class IDWBaseline:
    name = "IDW"

    def __init__(self, device: torch.device, range_h: float = 150.0, range_v: float = 40.0, k: int = 16,
                 power: float = 2.0) -> None:
        self.device, self.k, self.power = device, k, power
        self.scale = torch.tensor([1.0, 1.0, range_h / range_v], device=device)

    def fit(self, samples: pd.DataFrame) -> "IDWBaseline":
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)  # noqa: E731
        self.xyz = t(samples[["x", "y", "z"]].to_numpy()) * self.scale
        self.vals = torch.stack([t(np.log1p(samples.zn.clip(lower=0))), t(np.log1p(samples.pb.clip(lower=0))),
                                 t(samples.mineralized.astype(np.float32))], dim=1)
        return self

    @torch.no_grad()
    def predict(self, pts: np.ndarray, chunk: int = 8192) -> dict[str, np.ndarray]:
        q = torch.as_tensor(pts, dtype=torch.float32, device=self.device) * self.scale
        outs, stds = [], []
        k = min(self.k, len(self.xyz))
        for i in range(0, len(q), chunk):
            d = torch.cdist(q[i:i + chunk], self.xyz)
            dk, ik = torch.topk(d, k, largest=False)
            w = 1.0 / (dk.clamp(min=1.0) ** self.power)
            w = w / w.sum(1, keepdim=True)
            v = self.vals[ik]  # (n, k, 3)
            mu = (w.unsqueeze(-1) * v).sum(1)
            outs.append(mu)
            stds.append(torch.sqrt((w.unsqueeze(-1) * (v - mu.unsqueeze(1)) ** 2).sum(1)))
        mu = torch.cat(outs).cpu().numpy()
        sd = torch.cat(stds).cpu().numpy()
        return {"zn_mean": np.expm1(mu[:, 0]), "pb_mean": np.expm1(mu[:, 1]), "p_mean": mu[:, 2],
                "p_std": sd[:, 2], "zn_p05": None, "zn_p95": None}


class GPBaseline:
    name = "GP"

    def __init__(self, max_points: int = 1500, seed: int = 0, range_h: float = 150.0, range_v: float = 40.0) -> None:
        self.max_points, self.seed = max_points, seed
        self.init_ls = [range_h, range_h, range_v]
        self.models: dict[str, GaussianProcessRegressor] = {}

    def fit(self, samples: pd.DataFrame) -> "GPBaseline":
        rng = np.random.default_rng(self.seed)
        s = samples if len(samples) <= self.max_points else samples.iloc[
            np.sort(rng.choice(len(samples), self.max_points, replace=False))]
        X = s[["x", "y", "z"]].to_numpy(dtype=np.float64)
        targets = {"zn": np.log1p(s.zn.clip(lower=0).to_numpy()), "pb": np.log1p(s.pb.clip(lower=0).to_numpy()),
                   "ind": s.mineralized.to_numpy(dtype=np.float64)}
        for name, y in targets.items():
            kernel = ConstantKernel(1.0, (1e-3, 1e2)) * RBF(self.init_ls, (5.0, 3000.0)) + WhiteKernel(0.1, (1e-4, 2.0))
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0, random_state=self.seed)
            self.models[name] = gp.fit(X, y)
        return self

    def predict(self, pts: np.ndarray, chunk: int = 20000) -> dict[str, np.ndarray]:
        res: dict[str, list] = {k: [] for k in ("zn", "zn_sd", "pb", "ind", "ind_sd")}
        for i in range(0, len(pts), chunk):
            p = np.asarray(pts[i:i + chunk], dtype=np.float64)
            mu, sd = self.models["zn"].predict(p, return_std=True)
            res["zn"].append(mu)
            res["zn_sd"].append(sd)
            res["pb"].append(self.models["pb"].predict(p))
            mu, sd = self.models["ind"].predict(p, return_std=True)
            res["ind"].append(mu)
            res["ind_sd"].append(sd)
        r = {k: np.concatenate(v) for k, v in res.items()}
        return {"zn_mean": np.expm1(r["zn"]).clip(min=0), "pb_mean": np.expm1(r["pb"]).clip(min=0),
                "p_mean": np.clip(r["ind"], 0, 1), "p_std": r["ind_sd"],
                "zn_p05": np.expm1(r["zn"] - 1.645 * r["zn_sd"]).clip(min=0),
                "zn_p95": np.expm1(r["zn"] + 1.645 * r["zn_sd"]).clip(min=0)}

    def kernels(self) -> dict[str, str]:
        return {k: str(m.kernel_) for k, m in self.models.items()}
