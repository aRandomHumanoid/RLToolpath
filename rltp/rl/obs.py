"""Observation channels for one layer (no torch import: used by env workers and the loop)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from rltp.cover.cover import bead_weights, end_mask, seam_mask
from rltp.eval.fea2d import principal

CHANNELS = (
    "mask", "dist", "bead_cos2", "bead_sin2", "order", "ends", "seams",
    "fi", "pdir_cos2", "pdir_sin2", "pmag", "support", "load_x", "load_y",
    "w_strength", "w_time", "w_cont", "w_inter",
    "below_t", "below_ends", "above_t", "above_ends",
    "below_sigma", "below_tau", "above_sigma", "above_tau",
)
N_CH = len(CHANNELS)


def _norm_max(x):
    m = np.max(np.abs(x)) if x.size else 0.0
    return x / m if m > 0 else x


def build_obs(part, k: int, w, pad_to: tuple | None = None) -> np.ndarray:
    L = part.layers[k]
    ev = part.evals[k]
    H, W, n = L.H, L.W, L.n
    m = L.flat_mask
    obs = np.zeros((N_CH, n), np.float32)
    obs[0] = m
    d = ndimage.distance_transform_edt(L.mask).reshape(-1)
    obs[1] = np.minimum(d, 16.0) / 16.0
    wh = bead_weights(L)
    obs[2] = np.where(m, 2 * wh - 1, 0)
    obs[3] = 0.0  # grid-frame beads are 0 or 90 deg; kept for lattices with other angles
    T = ev.T if ev.T > 0 else 1.0
    obs[4] = np.nan_to_num(ev.timing.t_cell / T, nan=0.0)
    obs[5] = end_mask(L)
    obs[6] = seam_mask(L)
    fr = ev.fea
    if fr.N:
        obs[7] = _norm_max(fr.fi)
        th, mag = principal(fr.sx, fr.sy, fr.txy)
        obs[8] = np.where(m, np.cos(2 * th), 0)
        obs[9] = np.where(m, np.sin(2 * th), 0)
        obs[10] = _norm_max(mag)
    fp = part.fea(k)
    if fp is not None and fp.active:
        obs[11] = fp.support_map()
        lm = fp.load_map()
        dvec = fp.load_dir / (np.linalg.norm(fp.load_dir) + 1e-12)
        obs[12] = lm * dvec[0]
        obs[13] = lm * dvec[1]
    w = np.asarray(w, np.float32)
    obs[14:18] = w[:4, None]
    if part.use_inter:
        mat = part.spec.material
        if k - 1 >= 0 and k - 1 < len(part.ifaces) and part.ifaces[k - 1] is not None:
            itf = part.ifaces[k - 1]
            lo = part.evals[k - 1]
            obs[18, itf.up] = np.nan_to_num(lo.timing.t_cell[itf.lo] / max(lo.T, 1e-9))
            obs[19, itf.up] = lo.ends[itf.lo]
            obs[22, itf.up] = itf.sigma / mat.Zt
            obs[23, itf.up] = itf.tau / mat.Sz
        if k < len(part.ifaces) and part.ifaces[k] is not None:
            itf = part.ifaces[k]
            up = part.evals[k + 1]
            obs[20, itf.lo] = np.nan_to_num(up.timing.t_cell[itf.up] / max(up.T, 1e-9))
            obs[21, itf.lo] = up.ends[itf.up]
            obs[24, itf.lo] = itf.sigma / mat.Zt
            obs[25, itf.lo] = itf.tau / mat.Sz
    obs = obs.reshape(N_CH, H, W)
    if pad_to is not None:
        Hp, Wp = pad_to
        out = np.zeros((N_CH, Hp, Wp), np.float32)
        out[:, :H, :W] = obs
        obs = out
    return obs


def pad_mask(M: np.ndarray, H: int, W: int, Hp: int, Wp: int) -> np.ndarray:
    """(n_ops, H*W) -> (n_ops, Hp*Wp) with False padding."""
    n_ops = M.shape[0]
    out = np.zeros((n_ops, Hp, Wp), bool)
    out[:, :H, :W] = M.reshape(n_ops, H, W)
    return out.reshape(n_ops, Hp * Wp)


def unpad_action(a: int, H: int, W: int, Hp: int, Wp: int) -> int:
    op, cell = divmod(int(a), Hp * Wp)
    r, c = divmod(cell, Wp)
    return op * H * W + r * W + c
