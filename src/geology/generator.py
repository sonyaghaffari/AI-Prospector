"""Synthetic Angouran-*inspired* Zn-Pb world generator.

This is NOT the geology of Angouran Mine. It is a plausible, structurally controlled
carbonate-hosted Zn-Pb system: a marble unit between schists, a listric main fault, a
secondary cross fault, breccia along faults and their intersection, a plunging ore shoot
along the structural intersection, a contact-controlled "manto", and an alteration halo.

All randomness comes from a single ``numpy.random.Generator`` seeded once. The function is
called exactly once per experiment (``Experiment.create``); its output is then frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from src.geology.fields import gaussian_blur_2d, gaussian_random_field
from src.geology.grid import (LITH_ALTERED, LITH_BACKGROUND, LITH_BRECCIA, LITH_MARBLE, LITH_SCHIST,
                              LITHOLOGY, Grid)
from src.utils.config import SCENARIOS, WorldConfig

FAULT_ZONE_HALFWIDTH = 4.0  # m; a sample this close to a fault logs a fault-zone observation


@dataclass
class WorldTruth:
    """Complete hidden truth. Arrays are (nx, ny, nz); maps are (nx, ny)."""

    grid: Grid
    lithology: np.ndarray
    zn: np.ndarray
    pb: np.ndarray
    density: np.ndarray
    dist_main: np.ndarray
    dist_sec: np.ndarray
    potential: np.ndarray
    gravity_map: np.ndarray  # surface survey (observable)
    ip_map: np.ndarray  # surface survey (observable)
    fault_traces: dict[str, np.ndarray]  # mapped surface traces (observable)
    meta: dict = field(default_factory=dict)  # scenario + parameters (hidden)

    ARRAY_FIELDS = ("lithology", "zn", "pb", "density", "dist_main", "dist_sec", "potential")

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.ARRAY_FIELDS}

    def mineralized(self, cutoff: float) -> np.ndarray:
        return (self.zn + self.pb) >= cutoff

    def freeze(self) -> "WorldTruth":
        for k in (*self.ARRAY_FIELDS, "gravity_map", "ip_map"):
            getattr(self, k).setflags(write=False)
        for v in self.fault_traces.values():
            v.setflags(write=False)
        return self

    # ---- deterministic point sampling from the frozen volume (no randomness) ----
    def sample(self, pts: np.ndarray) -> dict[str, np.ndarray]:
        idx = self.grid.to_index(pts)
        lin = lambda a: map_coordinates(a.astype(np.float64), idx, order=1, mode="nearest")  # noqa: E731
        lith = map_coordinates(self.lithology, idx, order=0, mode="nearest").astype(np.int64)
        d_main, d_sec = lin(self.dist_main), lin(self.dist_sec)
        return {
            "lithology": lith,
            "zn": np.clip(lin(self.zn), 0, None),
            "pb": np.clip(lin(self.pb), 0, None),
            "density": lin(self.density),
            "fault_main": np.abs(d_main) < FAULT_ZONE_HALFWIDTH,
            "fault_sec": np.abs(d_sec) < FAULT_ZONE_HALFWIDTH,
            "breccia": lith == LITH_BRECCIA,
        }


def _unit(az_deg: float) -> np.ndarray:
    a = np.radians(az_deg)
    return np.array([np.sin(a), np.cos(a)])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_world(cfg: WorldConfig) -> WorldTruth:
    rng = np.random.default_rng(cfg.seed)
    scenario = cfg.scenario if cfg.scenario in SCENARIOS else str(rng.choice(SCENARIOS))
    grid = Grid(tuple(cfg.extent), tuple(cfg.shape))
    dx, dy, dz = grid.spacing
    X, Y, Z = grid.mesh()
    z = grid.z
    P = np.stack([X, Y], axis=-1)
    ext_xy = np.array(cfg.extent[:2])
    grf = lambda L: gaussian_random_field(rng, grid.shape, grid.spacing, L)  # noqa: E731

    # The mineralizing centre is placed anywhere across the bulk of the domain (margin keeps the
    # shoot, its ~70-110 m radius and breccia halo away from the grid edge). An earlier version
    # restricted this to +/-100 m of the exact domain centre, which made every independently-seeded
    # world's main target land in roughly the same place -- harmless within one experiment, but it
    # silently inflated apparent cross-world model transfer (two "independent" worlds were actually
    # spatially correlated by construction). See tests/test_geology.py::test_world_centres_are_well_spread.
    margin = 260.0
    centre = rng.uniform(margin, ext_xy - margin, 2)

    # ---------------- main fault: NE strike, steep, listric (flattens with depth) ----------
    strike_m = rng.uniform(35, 55)
    s_m, n_m = _unit(strike_m), _unit(strike_m + 90)  # n_m = dip direction
    dip0_m = rng.uniform(72, 82)
    listric = rng.uniform(10, 20)  # degrees of flattening by 200 m
    dip_m_z = dip0_m - listric * z / cfg.extent[2]
    cot = 1 / np.tan(np.radians(dip_m_z))
    h_m = np.concatenate([[0.0], np.cumsum(cot[:-1] * dz)]) + cot[0] * dz / 2  # horizontal shift
    und_m = 8.0 * grf((300, 300, 80))
    dh_main = (P - centre) @ n_m - h_m[None, None, :] - und_m
    dist_main = dh_main * np.sin(np.radians(dip_m_z))[None, None, :]

    # ---------------- secondary fault: cross-cutting, planar --------------------------------
    strike_s = strike_m + rng.uniform(55, 85)
    s_s, n_s = _unit(strike_s), _unit(strike_s + 90)
    centre_s = centre + s_m * rng.uniform(-50, 50)
    dip_s = rng.uniform(65, 85)
    h_s = z / np.tan(np.radians(dip_s))
    dist_sec = ((P - centre_s) @ n_s - h_s[None, None, :]) * np.sin(np.radians(dip_s))

    # structural intersection line q(z)
    A = np.stack([n_m, n_s])
    q = np.array([np.linalg.solve(A, [h_m[k] + n_m @ centre, h_s[k] + n_s @ centre_s]) for k in range(len(z))])

    # ---------------- stratigraphy: dipping marble between schists -----------------------
    bed_dip = rng.uniform(20, 35)
    bed_dir = _unit(rng.uniform(0, 360))
    throw = rng.uniform(15, 40)
    strat = Z - np.tan(np.radians(bed_dip)) * ((P - centre) @ bed_dir) + 8.0 * grf((250, 250, 40))
    strat = strat + throw * (dist_main > 0)
    k50 = int(np.argmin(np.abs(z - 50)))
    s_ref = 50 - np.tan(np.radians(bed_dip)) * ((q[k50] - centre) @ bed_dir)
    c1 = s_ref - rng.uniform(50, 80)
    c2 = s_ref + rng.uniform(140, 180)
    c3 = c2 + rng.uniform(40, 70)
    host = np.full(grid.shape, LITH_SCHIST, dtype=np.int8)
    host[(strat >= c1) & (strat < c2)] = LITH_MARBLE
    host[strat >= c3] = LITH_BACKGROUND

    # ---------------- breccia: fault damage zones + intersection pipe ----------------------
    g_b = grf((80, 80, 20))
    r_int = np.linalg.norm(P - q[None, None, :, :], axis=-1)
    brec = np.maximum.reduce([
        np.exp(-0.5 * (dist_main / (10 * (1 + 0.3 * g_b))) ** 2),
        np.exp(-0.5 * (r_int / 35) ** 2),
        0.6 * np.exp(-0.5 * (dist_sec / 7) ** 2),
    ])
    is_breccia = brec > 0.5

    # ---------------- ore shoot axis by scenario -------------------------------------------
    axis = q.copy()
    intensity = np.ones_like(z)
    params: dict = {}
    if scenario == "A":
        intensity = 1.0 + 0.1 * z / cfg.extent[2]
    elif scenario == "B":
        z_b = rng.uniform(105, 130)
        kb = int(np.searchsorted(z, z_b))
        d_q = (q[-1] - q[0]) / (z[-1] - z[0])
        base_dir = d_q / (np.linalg.norm(d_q) + 1e-9)
        ang = np.radians(rng.choice([-1, 1]) * rng.uniform(70, 120))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        v = rot @ base_dir * rng.uniform(0.8, 1.4)
        axis[kb:] = q[kb - 1] + (z[kb:, None] - z[kb - 1]) * v[None, :]
        params.update(bend_depth=z_b, bend_vector=v.tolist())
    elif scenario == "C":
        z_t = rng.uniform(110, 150)
        intensity = _sigmoid((z_t - z) / 8)
        params.update(termination_depth=z_t)
    elif scenario == "D":
        z_t = rng.uniform(125, 160)
        intensity = _sigmoid((z_t - z) / 10)
        params.update(termination_depth=z_t)

    d_ax = P - axis[None, None, :, :]
    u, v_ = d_ax @ s_m, d_ax @ n_m
    r_u, r_v = rng.uniform(70, 110), rng.uniform(30, 45)
    shoot = np.exp(-0.5 * ((u / r_u) ** 2 + (v_ / r_v) ** 2))
    manto = 0.7 * np.exp(-0.5 * ((strat - c1 - 8) / 10) ** 2) * np.exp(
        -0.5 * (np.linalg.norm(d_ax, axis=-1) / 180) ** 2)
    struct = 0.55 + 0.45 * np.exp(-np.abs(dist_main) / 60)
    p_main = np.maximum(shoot, manto) * intensity[None, None, :] * struct

    body = np.zeros(grid.shape)
    if scenario == "D":
        zc = rng.uniform(165, 182)
        for _ in range(100):
            ang = rng.uniform(0, 2 * np.pi)
            kc = int(np.searchsorted(z, zc))
            cxy = q[min(kc, len(z) - 1)] + rng.uniform(250, 350) * np.array([np.cos(ang), np.sin(ang)])
            if np.all((cxy > 150) & (cxy < ext_xy - 150)):
                break
        ra, rb = rng.uniform(80, 110), rng.uniform(40, 60)
        rc = min(rng.uniform(18, 24), (zc - 110) / 2.4)
        dd = P - cxy
        body = 0.95 * np.exp(-0.5 * (((dd @ s_s) / ra) ** 2 + ((dd @ n_s) / rb) ** 2 + ((Z - zc) / rc) ** 2))
        params.update(concealed_centre=[*cxy.tolist(), zc], concealed_radii=[ra, rb, rc])

    host_factor = np.select([host == LITH_MARBLE, host == LITH_SCHIST], [1.0, 0.3], 0.15)
    host_factor = host_factor * np.where(is_breccia, 1.3, 1.0)
    potential = np.maximum(p_main, body) * host_factor * np.exp(0.3 * grf((60, 60, 15)))

    zn = 32 * np.tanh(0.8 * potential) ** 2 * np.exp(0.2 * grf((50, 50, 12))) + 0.05 * np.exp(
        0.5 * grf((100, 100, 30)))
    pb = zn * 0.22 * np.exp(0.3 * grf((80, 80, 20)))

    # ---------------- final lithology with alteration halo ---------------------------------
    alt = gaussian_filter(potential, sigma=1.5) > (0.12 + 0.03 * grf((60, 60, 15)))
    lith = host.copy()
    lith[alt & np.isin(host, (LITH_MARBLE, LITH_SCHIST))] = LITH_ALTERED
    lith[is_breccia] = LITH_BRECCIA

    base_rho = np.array([2.71, 2.76, 2.55, 2.62, 2.66])
    density = base_rho[lith] + 0.03 * (zn + pb) + 0.015 * grf((60, 60, 15))

    # ---------------- surface geophysical surveys (observable, computed once) ---------------
    drho = density - density.mean()
    grav = np.zeros(grid.shape[:2])
    ip = np.zeros(grid.shape[:2])
    for k, zk in enumerate(z):
        sig = max(1.0, 0.5 * zk / dx)  # deeper sources -> broader, weaker anomalies
        grav += gaussian_blur_2d(0.0419 * drho[:, :, k] * dz, sig) / (1 + zk / 100)
        ip += gaussian_blur_2d((zn + pb)[:, :, k] * dz, sig) * np.exp(-zk / 60)
    grav = gaussian_blur_2d(grav, 1.0)
    ip = 5.0 + 0.05 * gaussian_blur_2d(ip, 1.5)
    grav += rng.normal(0, 0.03 * grav.std(), grav.shape)
    ip += rng.normal(0, 0.05 * ip.std(), ip.shape)

    # ---------------- mapped fault traces at surface (with mapping error) -------------------
    traces = {}
    for name, c0, sdir, ndir in (("main", centre, s_m, n_m), ("secondary", centre_s, s_s, n_s)):
        t = np.arange(-800, 801, 20.0)
        pts = c0[None, :] + t[:, None] * sdir[None, :] + rng.normal(0, 4.0, (len(t), 1)) * ndir[None, :]
        keep = np.all((pts >= 0) & (pts <= ext_xy), axis=1)
        traces[name] = pts[keep].astype(np.float32)

    meta = dict(
        scenario=scenario, seed=cfg.seed, centre=centre.tolist(),
        main_fault=dict(strike=strike_m, dip_surface=dip0_m, listric_flattening=listric),
        secondary_fault=dict(strike=strike_s, dip=dip_s, centre=centre_s.tolist()),
        bedding=dict(dip=bed_dip, throw=throw, contacts=[c1, c2, c3]),
        shoot=dict(r_along=r_u, r_across=r_v, axis=axis.tolist(), intersection=q.tolist()),
        lithology_names=list(LITHOLOGY), **params,
    )
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    return WorldTruth(
        grid=grid, lithology=np.ascontiguousarray(lith, dtype=np.int8), zn=f32(zn), pb=f32(pb),
        density=f32(density), dist_main=f32(dist_main), dist_sec=f32(dist_sec), potential=f32(potential),
        gravity_map=f32(grav), ip_map=f32(ip), fault_traces=traces, meta=meta,
    )
