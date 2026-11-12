"""Print-time model: firmware-style trapezoidal motion with junction-deviation corners.

Each stroke starts and ends at rest. Straight runs of cells are collapsed into
single moves (as in the G-code), so corners are where the planner slows down.
Between strokes: a travel move (trapezoid at travel speed), plus retract and
unretract when the travel is longer than 2p, plus optional z-hop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rltp.cover.cover import Pieces
from rltp.model import Layer


@dataclass
class Motion:
    v: float
    a: float
    jd: float
    v_travel: float
    a_travel: float
    retract_len: float
    retract_speed: float
    z_hop: float
    z_speed: float
    retract_min: float

    @staticmethod
    def from_printer(pr, first: bool = False) -> "Motion":
        return Motion(
            v=pr.first_layer_speed if first else pr.print_speed,
            a=pr.accel, jd=pr.junction_deviation,
            v_travel=pr.travel_speed, a_travel=pr.travel_accel,
            retract_len=pr.retract_length, retract_speed=pr.retract_speed,
            z_hop=pr.z_hop, z_speed=pr.z_speed, retract_min=2 * pr.bead_width,
        )


@dataclass
class Timing:
    T: float                 # layer duration (s)
    t_cell: np.ndarray       # (n,) time each cell centre is printed, nan outside
    order: np.ndarray        # piece ids in print order
    stroke_start: np.ndarray # per stroke in order: time printing starts
    stroke_time: np.ndarray  # per stroke in order: printing duration
    travel_time: np.ndarray  # per stroke in order: travel before it
    travel_dist: np.ndarray  # per stroke in order: travel length before it (mm)
    retract: np.ndarray      # per stroke in order: retracted before it
    end_xy: tuple            # local (u, v) where the layer ends


def trapezoid_time(v0, v1, vmax, a, L):
    """Duration of a move of length L from v0 to v1 with cruise cap vmax (vectorised)."""
    v0, v1, L = np.asarray(v0, float), np.asarray(v1, float), np.asarray(L, float)
    vp = np.sqrt(np.maximum((2 * a * L + v0 ** 2 + v1 ** 2) / 2, 0))
    cruise = vp > vmax
    da = (vmax ** 2 - v0 ** 2) / (2 * a)
    dd = (vmax ** 2 - v1 ** 2) / (2 * a)
    t_cr = (vmax - v0) / a + (vmax - v1) / a + np.maximum(L - da - dd, 0) / vmax
    t_tr = (vp - v0) / a + (vp - v1) / a
    return np.where(cruise, t_cr, t_tr)


def time_at(v0, v1, vmax, a, L, s):
    """Time to reach distance s along a trapezoid move (vectorised)."""
    vp = np.sqrt(np.maximum((2 * a * L + v0 ** 2 + v1 ** 2) / 2, 0))
    vc = np.minimum(vp, vmax)
    da = (vc ** 2 - v0 ** 2) / (2 * a)
    dd = (vc ** 2 - v1 ** 2) / (2 * a)
    t_acc = (vc - v0) / a
    T = trapezoid_time(v0, v1, vmax, a, L)
    t1 = (np.sqrt(v0 ** 2 + 2 * a * np.maximum(s, 0)) - v0) / a
    t2 = t_acc + (s - da) / np.maximum(vc, 1e-12)
    r = np.maximum(L - s, 0)
    t3 = T - (np.sqrt(v1 ** 2 + 2 * a * r) - v1) / a
    return np.where(s <= da, t1, np.where(s <= L - dd, t2, t3))


def junction_v2(d_prev, d_next, a, jd):
    """Squared max junction speed (Marlin/Klipper junction deviation)."""
    cos_t = -np.sum(d_prev * d_next, axis=-1)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    sin_h = np.sqrt(0.5 * (1 - cos_t))
    out = a * jd * sin_h / np.maximum(1 - sin_h, 1e-12)
    out = np.where(cos_t < -0.999999, np.inf, out)
    return np.where(cos_t > 0.999999, 0.0, out)


def plan(lim2, seg_len, a):
    """Node speeds^2 given node limits^2 and segment lengths (forward + backward pass).

    Min-plus closed form: w_f[k] = C[k] + cummin(lim2 - C)[k], C = cumsum(2 a L)."""
    C = np.concatenate([[0.0], np.cumsum(2 * a * seg_len)])
    wf = C + np.minimum.accumulate(lim2 - C)
    D = C[-1] - C
    wb = D + np.minimum.accumulate((wf - D)[::-1])[::-1]
    return np.maximum(wb, 0)


def order_strokes(pc: Pieces, W: int, p: float, start_uv=(0.0, 0.0)) -> np.ndarray:
    """Greedy nearest-neighbour stroke order from a fixed start point.

    The start point is fixed per layer (the grid origin) so a layer's order never
    depends on other layers."""
    S = pc.count
    if S == 0:
        return np.empty(0, np.int64)
    starts = np.array([c[0] for c in pc.cells])
    ends = np.array([c[0] if lp else c[-1] for c, lp in zip(pc.cells, pc.is_loop)])
    sr, sc = np.divmod(starts, W)
    er, ec = np.divmod(ends, W)
    su, sv = (sc + 0.5) * p, (sr + 0.5) * p
    eu, ev = (ec + 0.5) * p, (er + 0.5) * p
    left = np.ones(S, bool)
    order = np.empty(S, np.int64)
    cu, cv = start_uv
    for k in range(S):
        d = (su - cu) ** 2 + (sv - cv) ** 2
        d[~left] = np.inf
        j = int(np.argmin(d))
        order[k] = j
        left[j] = False
        cu, cv = eu[j], ev[j]
    return order


def time_layer(L: Layer, pc: Pieces, mot: Motion) -> Timing:
    p, W, n = L.frame.p, L.W, L.n
    order = order_strokes(pc, W, p)
    S = len(order)
    t_cell = np.full(n, np.nan)
    if S == 0:
        z = np.zeros(0)
        return Timing(0.0, t_cell, order, z, z, z, z, z.astype(bool), (0.0, 0.0))

    seqs = []
    for pid in order:
        cells = pc.cells[pid]
        seqs.append(np.append(cells, cells[0]) if pc.is_loop[pid] else cells)
    lens = np.array([len(s) for s in seqs])
    allc = np.concatenate(seqs)
    sid = np.repeat(np.arange(S), lens)
    r, c = np.divmod(allc, W)
    u, v = (c + 0.5) * p, (r + 0.5) * p

    first_idx = np.concatenate([[0], np.cumsum(lens)[:-1]])
    last_idx = np.cumsum(lens) - 1

    lk = np.nonzero(sid[1:] == sid[:-1])[0]
    stroke_time = np.zeros(S)
    local_end = np.empty(0)
    if lk.size:
        du = np.rint((u[lk + 1] - u[lk]) / p).astype(np.int8)
        dv = np.rint((v[lk + 1] - v[lk]) / p).astype(np.int8)
        lsid = sid[lk]
        newseg = np.ones(lk.size, bool)
        newseg[1:] = (lsid[1:] != lsid[:-1]) | (du[1:] != du[:-1]) | (dv[1:] != dv[:-1])
        seg_id = np.cumsum(newseg) - 1
        nseg = int(seg_id[-1]) + 1
        seg_first = np.nonzero(newseg)[0]
        seg_len = np.bincount(seg_id, minlength=nseg) * p
        seg_dir = np.stack([du[newseg], dv[newseg]], 1).astype(float)
        seg_sid = lsid[newseg]

        lim2 = np.zeros(nseg + 1)
        if nseg > 1:
            jv = junction_v2(seg_dir[:-1], seg_dir[1:], mot.a, mot.jd)
            same = seg_sid[1:] == seg_sid[:-1]
            lim2[1:-1] = np.where(same, np.minimum(jv, mot.v ** 2), 0.0)
        w = plan(lim2, seg_len, mot.a)
        vn = np.sqrt(w)
        v0, v1 = vn[:-1], vn[1:]
        seg_T = trapezoid_time(v0, v1, mot.v, mot.a, seg_len)
        stroke_time = np.bincount(seg_sid, weights=seg_T, minlength=S)
        cum = np.cumsum(seg_T) - seg_T
        stroke_first_seg = np.searchsorted(seg_sid, np.arange(S), "left")  # seg_sid is sorted
        seg_local_start = cum - cum[stroke_first_seg[seg_sid]]
        k_in_seg = np.arange(lk.size) - seg_first[seg_id]
        s_end = (k_in_seg + 1) * p
        local_end = seg_local_start[seg_id] + time_at(
            v0[seg_id], v1[seg_id], mot.v, mot.a, seg_len[seg_id], s_end)

    single = lens == 1
    if single.any():
        stroke_time[single] = trapezoid_time(0.0, 0.0, mot.v, mot.a, p / 2)

    # travels
    pu, pv = 0.0, 0.0
    su, sv = u[first_idx], v[first_idx]
    eu, ev = u[last_idx], v[last_idx]
    prev_u = np.concatenate([[pu], eu[:-1]])
    prev_v = np.concatenate([[pv], ev[:-1]])
    dist = np.hypot(su - prev_u, sv - prev_v)
    retract = dist > mot.retract_min + 1e-9
    retract[0] = True
    t_tr = trapezoid_time(0.0, 0.0, mot.v_travel, mot.a_travel, dist)
    t_tr = t_tr + retract * (2 * mot.retract_len / mot.retract_speed + 2 * mot.z_hop / mot.z_speed)
    stroke_start = np.cumsum(t_tr + np.concatenate([[0.0], stroke_time[:-1]]))
    T = float(stroke_start[-1] + stroke_time[-1])

    if lk.size:
        t_cell[allc[lk + 1]] = stroke_start[sid[lk]] + local_end
    t_cell[allc[first_idx]] = stroke_start
    return Timing(T, t_cell, order, stroke_start, stroke_time, t_tr, dist, retract,
                  (float(eu[-1]), float(ev[-1])))
