"""Continuity, per-layer evaluation and score assembly.

Raw part-level terms (higher is better):
  strength = -log FI_agg over all elements of the part   (p-norm, p ~ 8)
  time     = -sum of layer times (s)
  cont     = -sum of layer continuity costs
  inter    = -log FI_agg over all interface cells (v1.5)
Each term's change is divided by the running standard deviation of its per-edit
change, then weighted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rltp.config import TERMS, ScoreConfig
from rltp.cover.cover import Pieces, end_mask, get_pieces, seam_mask
from rltp.eval.fea2d import FEAProblem, FEAResult
from rltp.eval.timing import Motion, Timing, time_layer
from rltp.model import Layer


@dataclass
class Continuity:
    C: float
    n_strokes: int
    travel_cells: float
    n_boundary_ends: int


def continuity(L: Layer, pc: Pieces, tm: Timing, cfg: ScoreConfig, bnd: np.ndarray) -> Continuity:
    travel = float(tm.travel_dist[1:].sum() / L.frame.p) if tm.travel_dist.size > 1 else 0.0
    ends = end_mask(L) | seam_mask(L)
    nb = int((ends & bnd).sum())
    S = pc.count
    C = cfg.c_stroke * S + cfg.c_travel * travel + cfg.c_boundary_end * nb
    return Continuity(C, S, travel, nb)


@dataclass
class LayerEval:
    timing: Timing
    cont: Continuity
    fea: FEAResult
    ends: np.ndarray          # bool (n,): path ends or seams

    @property
    def T(self) -> float:
        return self.timing.T


def evaluate_layer(L: Layer, motion: Motion, fea: FEAProblem | None, cfg: ScoreConfig,
                   bnd: np.ndarray, warm: FEAResult | None = None, direct: bool = False,
                   keep_factor: bool = True, wh: np.ndarray | None = None) -> LayerEval:
    from rltp.cover.cover import bead_weights
    pc = get_pieces(L)
    tm = time_layer(L, pc, motion)
    ct = continuity(L, pc, tm, cfg, bnd)
    if fea is not None:
        if wh is None:
            wh = bead_weights(L)
        fr = fea.solve(wh, warm=None if warm is None else warm.u, direct=direct, keep=keep_factor)
    else:
        z = np.zeros(L.n)
        fr = FEAResult(np.zeros(0), z, 0.0, 0, z, z, z)
    ends = end_mask(L) | seam_mask(L)
    return LayerEval(tm, ct, fr, ends)


class Normalizer:
    """Running standard deviation of each term's per-edit change (Welford)."""

    def __init__(self, floor: float = 1e-6, prior: dict | None = None, prior_count: float = 0):
        self.floor = floor
        self.n = {t: 0.0 for t in TERMS}
        self.mean = {t: 0.0 for t in TERMS}
        self.m2 = {t: 0.0 for t in TERMS}
        self.frozen = False
        if prior:
            for t, s in prior.items():
                self.n[t] = prior_count
                self.m2[t] = (s ** 2) * prior_count

    def update(self, d: dict) -> None:
        if self.frozen:
            return
        for t, x in d.items():
            if x is None or not np.isfinite(x):
                continue
            self.n[t] += 1
            delta = x - self.mean[t]
            self.mean[t] += delta / self.n[t]
            self.m2[t] += delta * (x - self.mean[t])

    def std(self, t: str) -> float:
        if self.n[t] < 2:
            return 1.0
        s = float(np.sqrt(self.m2[t] / (self.n[t] - 1)))
        return s if s > self.floor else 1.0 if self.m2[t] == 0 else self.floor

    def scales(self) -> dict:
        return {t: self.std(t) for t in TERMS}

    def total(self, d: dict, w) -> float:
        return float(sum(w[i] * d.get(t, 0.0) / self.std(t) for i, t in enumerate(TERMS)))

    def state(self) -> dict:
        return {"n": dict(self.n), "mean": dict(self.mean), "m2": dict(self.m2), "frozen": self.frozen}

    def load(self, s: dict) -> "Normalizer":
        self.n, self.mean, self.m2 = dict(s["n"]), dict(s["mean"]), dict(s["m2"])
        self.frozen = s.get("frozen", False)
        return self


def strength_raw(S_p: float, N: int, p: float) -> float:
    if N == 0 or S_p <= 0:
        return 0.0
    return float(-np.log((S_p / N) ** (1 / p)))
