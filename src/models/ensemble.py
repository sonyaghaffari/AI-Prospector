"""Multi-task coordinate network trained as a vectorised deep ensemble on the GPU.

Architecture (per member):
  input features (coords + interpreted-fault distances + surface geophysics)
  -> random Fourier encoding of (x, y, z) with a LOW bandwidth (sigma ~ 1 on [-1, 1]
     coordinates) so the network represents smooth geology and does not invent
     high-frequency structure when extrapolating below 100 m
  -> 3 x 128 SiLU MLP
  -> heads: mineralization logit | log1p(Zn) mean, log-variance | log1p(Pb) mean,
     log-variance | 5 lithology logits

Multi-task learning is used because lithology, grades and the mineralization indicator
share the same geological controls. Grades use a heteroscedastic Gaussian likelihood on
log1p(grade) (aleatoric noise); disagreement between independently initialised members
(different seeds, Fourier bases, minibatch orders and hole-grouped validation splits) is
the epistemic component (Lakshminarayanan et al., 2017). Ensemble spread is NOT a
calibrated probability; calibration is measured separately on held-out data.

All M members are stored as batched weight tensors (M, in, out) and trained in one
batched-matmul pass, which keeps the GPU busy despite the tiny per-member networks.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.features import FeatureBuilder
from src.utils.config import ModelConfig
from src.utils.gpu import oom_safe_batched

N_LITH = 5
OUT_DIM = 1 + 2 + 2 + N_LITH


class EnsembleLinear(nn.Module):
    def __init__(self, m: int, d_in: int, d_out: int, gen: torch.Generator) -> None:
        super().__init__()
        bound = 1 / math.sqrt(d_in)
        self.weight = nn.Parameter((torch.rand(m, d_in, d_out, generator=gen) * 2 - 1) * bound * math.sqrt(3))
        self.bias = nn.Parameter((torch.rand(m, 1, d_out, generator=gen) * 2 - 1) * bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (M, B, d_in)
        return torch.baddbmm(self.bias, x, self.weight)


class GeoEnsemble(nn.Module):
    def __init__(self, n_features: int, cfg: ModelConfig, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        m = cfg.ensemble_size
        self.m = m
        self.register_buffer("fourier_b", torch.randn(m, 3, cfg.fourier_features, generator=gen) * cfg.fourier_sigma)
        d_in = 2 * cfg.fourier_features + n_features
        dims = [d_in] + [cfg.hidden] * cfg.layers
        self.layers = nn.ModuleList(EnsembleLinear(m, a, b, gen) for a, b in zip(dims[:-1], dims[1:]))
        # non-affine LayerNorm: no parameters shared between ensemble members
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.hidden, elementwise_affine=False) for _ in range(cfg.layers))
        self.head = EnsembleLinear(m, cfg.hidden, OUT_DIM, gen)
        with torch.no_grad():  # start grade log-variances near a moderate value
            self.head.bias[..., 2].fill_(-1.0)
            self.head.bias[..., 4].fill_(-1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (M, B, F) -> (M, B, OUT_DIM)
        proj = 2 * math.pi * torch.bmm(x[..., :3], self.fourier_b)
        h = torch.cat([torch.sin(proj), torch.cos(proj), x], dim=-1)
        for lin, norm in zip(self.layers, self.norms):
            h = F.silu(norm(lin(h)))
        return self.head(h)


def _split_outputs(out: torch.Tensor) -> dict[str, torch.Tensor]:
    out = out.float()
    return {"logit": out[..., 0], "zn_mu": out[..., 1], "zn_logvar": out[..., 2].clamp(-7, 3),
            "pb_mu": out[..., 3], "pb_logvar": out[..., 4].clamp(-7, 3), "lith": out[..., 5:]}


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    best_epoch: list[int] = field(default_factory=list)
    seconds: float = 0.0
    n_samples: int = 0
    n_holes: int = 0
    device: str = ""
    final_val_loss: list[float] = field(default_factory=list)


class GeoModel:
    """Feature builder + ensemble. ``fit`` consumes only an observed-samples DataFrame."""

    def __init__(self, extent, cfg: ModelConfig, device: torch.device, seed: int) -> None:
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.features = FeatureBuilder(extent, cfg.use_geophysics)
        self.net: GeoEnsemble | None = None

    # ------------------------------------------------------------------ training
    def _targets(self, s: pd.DataFrame) -> dict[str, torch.Tensor]:
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)  # noqa: E731
        return {"y_min": t(s.mineralized.astype(np.float32)), "zn": t(np.log1p(s.zn.clip(lower=0))),
                "pb": t(np.log1p(s.pb.clip(lower=0))), "lith": t(s.lithology, torch.long)}

    def _loss(self, o: dict[str, torch.Tensor], y: dict[str, torch.Tensor], pos_weight: torch.Tensor) -> torch.Tensor:
        """Per-element multi-task loss, shape (M, B)."""
        bce = F.binary_cross_entropy_with_logits(o["logit"], y["y_min"], pos_weight=pos_weight, reduction="none")
        nll = lambda mu, lv, t: 0.5 * (lv + (t - mu) ** 2 * torch.exp(-lv))  # noqa: E731
        ce = F.cross_entropy(o["lith"].reshape(-1, N_LITH), y["lith"].reshape(-1), reduction="none").view_as(bce)
        return bce + nll(o["zn_mu"], o["zn_logvar"], y["zn"]) + nll(o["pb_mu"], o["pb_logvar"], y["pb"]) \
            + self.cfg.lith_loss_weight * ce

    def fit(self, samples: pd.DataFrame, fault_traces, geophysics, epochs: int | None = None,
            warm_start: bool = False, progress=None) -> TrainResult:
        cfg, dev = self.cfg, self.device
        t0 = time.perf_counter()
        self.features.fit(samples, fault_traces, geophysics)
        X = self.features.transform(samples[["x", "y", "z"]].to_numpy(), dev)
        Y = self._targets(samples)
        n, m = len(samples), cfg.ensemble_size
        if self.net is None or not warm_start:
            self.net = GeoEnsemble(self.features.n_features, cfg, self.seed).to(dev)
        net = self.net

        # hole-grouped validation split, different for each member (no hole straddles train/val)
        rng = np.random.default_rng(self.seed)
        holes = samples.hole_id.to_numpy()
        uniq = np.unique(holes)
        n_val = max(1, int(round(cfg.val_hole_fraction * len(uniq))))
        val_mask = np.zeros((m, n), dtype=bool)
        for k in range(m):
            val_mask[k] = np.isin(holes, rng.permutation(uniq)[:n_val])
        train_idx = [np.flatnonzero(~val_mask[k]) for k in range(m)]
        val_mask_t = torch.as_tensor(val_mask, device=dev)
        n_pos = float(Y["y_min"].sum())
        pos_weight = torch.tensor(np.clip((n - n_pos) / max(n_pos, 1.0), 1.0, 20.0), device=dev)

        max_epochs = epochs or cfg.max_epochs
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr if not warm_start else cfg.lr * 0.3,
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=cfg.lr * 0.02)
        use_amp = cfg.amp and dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        bs = min(cfg.batch_size, max(len(t) for t in train_idx))
        steps = math.ceil(max(len(t) for t in train_idx) / bs)
        best = torch.full((m,), float("inf"), device=dev)
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        since = np.zeros(m, dtype=int)
        best_epoch = [0] * m
        res = TrainResult(n_samples=n, n_holes=len(uniq), device=str(dev))
        Xm = X.unsqueeze(0).expand(m, -1, -1)
        Ym = {k: v.unsqueeze(0).expand(m, *v.shape) for k, v in Y.items()}

        for epoch in range(max_epochs):
            net.train()
            perms = []
            for k in range(m):
                p = train_idx[k][rng.permutation(len(train_idx[k]))]
                perms.append(np.resize(p, steps * bs))
            idx = torch.as_tensor(np.stack(perms), device=dev).view(m, steps, bs)
            tr_loss = 0.0
            for s in range(steps):
                bi = idx[:, s]  # (M, bs)
                xb = torch.gather(Xm, 1, bi.unsqueeze(-1).expand(-1, -1, X.shape[1]))
                yb = {k: torch.gather(v, 1, bi) for k, v in Ym.items()}
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    out = net(xb)
                loss = self._loss(_split_outputs(out), yb, pos_weight).mean(dim=1).sum()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tr_loss += loss.item() / m
            sched.step()

            net.eval()
            with torch.no_grad():
                per = self._loss(_split_outputs(net(Xm)), Ym, pos_weight)  # (M, N)
                vl = (per * val_mask_t).sum(1) / val_mask_t.sum(1).clamp(min=1)
            improved = (vl < best - 1e-4)
            sd = net.state_dict()
            for k in torch.nonzero(improved).flatten().tolist():
                best[k] = vl[k]
                since[k] = 0
                best_epoch[k] = epoch
                for name, v in sd.items():
                    if v.dim() > 0 and v.shape[0] == m:
                        best_state[name][k] = v[k].detach()
            since[~improved.cpu().numpy()] += 1
            res.history.append({"epoch": epoch, "train_loss": tr_loss / steps, "val_loss": float(vl.mean()),
                                "lr": sched.get_last_lr()[0]})
            if progress is not None:
                progress(epoch, max_epochs, res.history[-1])
            if epoch >= cfg.min_epochs and np.all(since >= cfg.patience):
                break
        net.load_state_dict(best_state)
        res.best_epoch = best_epoch
        res.final_val_loss = best.tolist()
        res.seconds = time.perf_counter() - t0
        return res

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def member_outputs(self, X: torch.Tensor, batch: int = 65536) -> dict[str, torch.Tensor]:
        """Raw per-member outputs for feature tensor X (N, F) -> tensors (M, N[, ...])."""
        assert self.net is not None, "model not trained"
        self.net.eval()
        m = self.cfg.ensemble_size
        use_amp = self.cfg.amp and self.device.type == "cuda"

        def run(xc: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                return self.net(xc.unsqueeze(0).expand(m, -1, -1)).float().transpose(0, 1)  # (n, M, OUT)

        out = oom_safe_batched(run, X, batch).transpose(0, 1)
        return _split_outputs(out)

    @torch.no_grad()
    def predict_features(self, X: torch.Tensor, batch: int = 65536) -> dict[str, torch.Tensor]:
        o = self.member_outputs(X, batch)
        p = torch.sigmoid(o["logit"])
        res = {"p_members": p, "p_mean": p.mean(0), "p_std": p.std(0, unbiased=False)}
        for g in ("zn", "pb"):
            mu, var = o[f"{g}_mu"], torch.exp(o[f"{g}_logvar"])
            mu_mean = mu.mean(0)
            epi, alea = mu.var(0, unbiased=False), var.mean(0)
            sd = torch.sqrt(epi + alea)
            res.update({f"{g}_log_mu": mu_mean, f"{g}_log_mu_members": mu, f"{g}_epi_var": epi,
                        f"{g}_alea_var": alea, f"{g}_mean": torch.expm1(mu_mean).clamp(min=0),
                        f"{g}_p05": torch.expm1(mu_mean - 1.645 * sd).clamp(min=0),
                        f"{g}_p95": torch.expm1(mu_mean + 1.645 * sd).clamp(min=0)})
        res["lith_prob"] = torch.softmax(o["lith"], -1).mean(0)
        return res

    def predict_points(self, pts: np.ndarray, batch: int = 65536) -> dict[str, torch.Tensor]:
        return self.predict_features(self.features.transform(pts, self.device), batch)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "features": self.features.state(),
                    "geo": self.features.geo, "geo_stats": self.features.geo_stats,
                    "cfg": self.cfg.__dict__, "seed": self.seed}, path)

    @classmethod
    def load(cls, path, device: torch.device) -> "GeoModel":
        from src.models.features import FaultPlane

        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = ModelConfig(**ck["cfg"])
        fs = ck["features"]
        model = cls(tuple(fs["extent"]), cfg, device, ck["seed"])
        model.features.planes = {k: FaultPlane(np.array(v["normal"]), np.array(v["centroid"]), v["n_drill_points"])
                                 for k, v in fs["planes"].items()}
        model.features.geo, model.features.geo_stats = ck["geo"], ck["geo_stats"]
        model.net = GeoEnsemble(model.features.n_features, cfg, ck["seed"]).to(device)
        model.net.load_state_dict(ck["state_dict"])
        return model
