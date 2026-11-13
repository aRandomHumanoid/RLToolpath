"""v1.5 interlayer term: interface statics + bond quality.

Interface j joins layer j (below) and layer j+1 (above). Its shared surface is
the set of upper-layer cells whose centres fall on in-part cells of the lower
layer. Section properties and the load resultants on the cut depend only on
geometry and loads, so the stress maps (sigma, tau per cell) are fixed per part.
Only bond quality changes during optimisation:

    q_i = max(phi(dt_i) * s_i, q_min),  phi(dt) = 1 / (1 + (dt / tau)^n)
    FI_i = (max(sigma_i, 0) / (q_i Zt))^2 + (tau_i / (q_i Sz))^2

dt_i = (T_lower + layer change) + t_upper(i) - t_lower(i): the absolute-clock gap,
which depends only on the two layers of the interface.
(v2 would replace sigma/tau with samples from a 3D FEA; not implemented.)

Sparse layers (ZInterface): only cells filled in both layers bond, so the section
that carries the cut's loads is the bonded set, recomputed for every edit. Less
overlap between adjacent layers' toolpaths means a smaller, weaker section. Besides
any point loads above the cut, three generic cases stand for Z strength in general:
a pull and bending about each principal axis of the full shared surface, each sized
so that a fully bonded interface peaks at sigma0 = Zt / 2 (FI 0.25 at q = 1). Bending
counts both signs. Per bonded cell, FI^p is averaged over the cases (a soft max);
unbonded cells count as 0, and N stays the shared-surface cell count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rltp.config import BondModel, Material
from rltp.model import Layer


@dataclass
class Interface:
    j: int
    lo: np.ndarray           # cell indices in the lower layer
    up: np.ndarray           # matching cell indices in the upper layer
    sigma: np.ndarray        # normal stress per shared cell (MPa), fixed
    tau: np.ndarray          # shear stress per shared cell (MPa), fixed
    A: float
    centroid: tuple
    Ix: float
    Iy: float
    angle: float             # principal axes angle (rad)
    loaded: bool             # False -> r^2 weighting fallback

    @property
    def N(self) -> int:
        return int(self.lo.size)


@dataclass
class IfaceResult:
    S_p: float
    N: int
    fi: np.ndarray           # per shared cell
    q: np.ndarray


def section(x, y, a, p):
    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy
    Ixx = float((dy ** 2).sum() * a)
    Iyy = float((dx ** 2).sum() * a)
    Ixy = float((dx * dy).sum() * a)
    phi = 0.5 * np.arctan2(-2 * Ixy, Ixx - Iyy)
    c, s = np.cos(phi), np.sin(phi)
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    own = x.size * a * p * p / 12
    Ix = float((yb ** 2).sum() * a + own)
    Iy = float((xb ** 2).sum() * a + own)
    return cx, cy, phi, xb, yb, Ix, Iy


def build_interface(j: int, lo: Layer, up: Layer, z_cut: float, point_loads, mat: Material) -> Interface | None:
    upc = np.flatnonzero(up.flat_mask)
    x, y = up.frame.cell_xy(upc, up.W)
    loc = lo.frame.world_to_cell(x, y, lo.H, lo.W)
    ok = loc >= 0
    ok[ok] &= lo.flat_mask[loc[ok]]
    if ok.sum() < 3:
        return None
    upc, loc, x, y = upc[ok], loc[ok], x[ok], y[ok]
    p = up.frame.p
    a = p * p
    A = a * upc.size
    cx, cy, phi, xb, yb, Ix, Iy = section(x, y, a, p)
    J = Ix + Iy
    above = [pl for pl in (point_loads or []) if pl.pos[2] > z_cut]
    if above:
        F = np.array([pl.force for pl in above], float)
        R = np.array([pl.pos for pl in above], float) - np.array([cx, cy, z_cut])
        M = np.cross(R, F).sum(0)
        Fs = F.sum(0)
        c, s = np.cos(phi), np.sin(phi)
        Vx, Vy = c * Fs[0] + s * Fs[1], -s * Fs[0] + c * Fs[1]
        Mx, My = c * M[0] + s * M[1], -s * M[0] + c * M[1]
        Nz, T = Fs[2], M[2]
        sigma = Nz / A + Mx * yb / Ix - My * xb / Iy
        tau = np.sqrt((Vx / A - T * yb / J) ** 2 + (Vy / A + T * xb / J) ** 2)
        loaded = True
    else:
        r = np.hypot(xb, yb)
        rmax = max(float(r.max()), p)
        sigma = 0.5 * mat.Zt * r / rmax if not point_loads else np.zeros_like(r)
        tau = np.zeros_like(r)
        loaded = False
    return Interface(j, loc.astype(np.int64), upc.astype(np.int64), sigma, tau, A, (cx, cy), Ix, Iy,
                     float(phi), loaded)


@dataclass
class ZInterface:
    j: int
    lo: np.ndarray           # lower-layer cells of the shared surface (in-part in both layers)
    up: np.ndarray           # matching upper-layer cells
    x: np.ndarray            # world centres of the shared cells
    y: np.ndarray
    phi: float               # principal axes of the full shared surface
    A0: float
    Ix0: float
    Iy0: float
    ymax0: float
    xmax0: float
    loads: list              # point loads above the cut: (pos relative to z_cut, force)
    p: float
    sigma: np.ndarray = None  # fully bonded generic bending envelope (observations only)
    tau: np.ndarray = None

    @property
    def N(self) -> int:
        return int(self.lo.size)


def build_zinterface(j: int, lo: Layer, up: Layer, z_cut: float, point_loads,
                     mat: Material) -> ZInterface | None:
    upc = np.flatnonzero(up.part_mask.reshape(-1))
    x, y = up.material_frame.cell_xy(upc, up.W)
    loc = lo.material_frame.world_to_cell(x, y, lo.H, lo.W)
    ok = loc >= 0
    ok[ok] &= lo.part_mask.reshape(-1)[loc[ok]]
    if ok.sum() < 3:
        return None
    upc, loc, x, y = upc[ok], loc[ok], x[ok], y[ok]
    p = up.frame.p
    a = p * p
    cx, cy, phi, xb, yb, Ix, Iy = section(x, y, a, p)
    loads = [(np.array(pl.pos, float) - np.array([0.0, 0.0, z_cut]), np.array(pl.force, float))
             for pl in (point_loads or []) if pl.pos[2] > z_cut]
    ymax, xmax = max(float(np.abs(yb).max()), p / 2), max(float(np.abs(xb).max()), p / 2)
    sigma = 0.5 * mat.Zt * np.maximum(np.abs(yb) / ymax, np.abs(xb) / xmax)
    return ZInterface(j, loc.astype(np.int64), upc.astype(np.int64), x, y, float(phi), a * upc.size,
                      Ix, Iy, ymax, xmax, loads, p, sigma, np.zeros_like(sigma))


def eval_zinterface(itf: ZInterface, fill_lo, fill_up, t_lo, ends_lo, T_lo: float, t_up, ends_up,
                    bond: BondModel, mat: Material, t_change: float, p_norm: float) -> IfaceResult:
    """Bonded-area interface statics with generic Z cases (see the module docstring)."""
    b = np.asarray(fill_lo)[itf.lo] & np.asarray(fill_up)[itf.up]
    n = int(b.sum())
    fi_all = np.zeros(itf.N)
    q_all = np.zeros(itf.N)
    if n < 3:                                    # (nearly) detached: as bad as it gets
        fi_all[:] = 1e3
        return IfaceResult(float(itf.N * 1e3 ** p_norm), itf.N, fi_all, q_all)
    lo, up = itf.lo[b], itf.up[b]
    a = itf.p * itf.p
    xs, ys = itf.x[b], itf.y[b]
    cxb, cyb = float(xs.mean()), float(ys.mean())
    c, s = np.cos(itf.phi), np.sin(itf.phi)
    dx, dy = xs - cxb, ys - cyb
    xb, yb = c * dx + s * dy, -s * dx + c * dy
    A = a * n
    own = n * a * itf.p * itf.p / 12
    Ix = float((yb ** 2).sum() * a + own)
    Iy = float((xb ** 2).sum() * a + own)
    s0 = 0.5 * mat.Zt
    sig = [np.full(n, s0 * itf.A0 / A),                                  # pull
           (s0 * itf.Ix0 / itf.ymax0) * np.abs(yb) / Ix,                  # bending about x
           (s0 * itf.Iy0 / itf.xmax0) * np.abs(xb) / Iy]                  # bending about y
    taus = [0.0, 0.0, 0.0]
    if itf.loads:                                # the point loads above this cut
        F = np.array([f for _, f in itf.loads])
        R = np.array([r for r, _ in itf.loads]) - np.array([cxb, cyb, 0.0])
        M = np.cross(R, F).sum(0)
        Fs = F.sum(0)
        Vx, Vy = c * Fs[0] + s * Fs[1], -s * Fs[0] + c * Fs[1]
        Mx, My = c * M[0] + s * M[1], -s * M[0] + c * M[1]
        J = Ix + Iy
        sig.append(Fs[2] / A + Mx * yb / Ix - My * xb / Iy)
        taus.append(np.sqrt((Vx / A - M[2] * yb / J) ** 2 + (Vy / A + M[2] * xb / J) ** 2))
    dt = T_lo + t_change + t_up[up] - t_lo[lo]
    dt = np.maximum(np.nan_to_num(dt, nan=0.0), 0.0)
    phi_b = 1.0 / (1.0 + (dt / bond.tau) ** bond.n)
    sc = np.where(ends_lo[lo] & ends_up[up], bond.s_end, 1.0)
    q = np.maximum(phi_b * sc, bond.q_min)
    m = np.zeros(n)
    for sg, tu in zip(sig, taus):
        f = (np.maximum(sg, 0) / (q * mat.Zt)) ** 2 + (tu / (q * mat.Sz)) ** 2
        m += f ** p_norm
    m /= len(sig)
    fi_all[b] = m ** (1 / p_norm)
    q_all[b] = q
    return IfaceResult(float(m.sum()), itf.N, fi_all, q_all)


def eval_interface(itf: Interface, t_lo, ends_lo, T_lo: float, t_up, ends_up,
                   bond: BondModel, mat: Material, t_change: float, p_norm: float) -> IfaceResult:
    dt = T_lo + t_change + t_up[itf.up] - t_lo[itf.lo]
    dt = np.maximum(np.nan_to_num(dt, nan=0.0), 0.0)
    phi = 1.0 / (1.0 + (dt / bond.tau) ** bond.n)
    s = np.where(ends_lo[itf.lo] & ends_up[itf.up], bond.s_end, 1.0)
    q = np.maximum(phi * s, bond.q_min)
    fi = (np.maximum(itf.sigma, 0) / (q * mat.Zt)) ** 2 + (itf.tau / (q * mat.Sz)) ** 2
    return IfaceResult(float(np.sum(fi ** p_norm)), itf.N, fi, q)
