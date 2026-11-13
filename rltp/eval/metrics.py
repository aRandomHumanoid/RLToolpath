"""Continuity, per-layer evaluation and score assembly.

Raw part-level terms (higher is better):
  strength = -log FI_agg over all elements of the part   (p-norm, p ~ 8)
  time     = -sum of layer times (s)
  cont     = -sum of layer continuity costs
  inter    = -log FI_agg over all interface cells (v1.5)
Each term's change is divided by a scale (Normalizer), then weighted. By default
the scale is the part's baseline value of the term, so the score is a weighted
sum of fractional improvements.
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
    travel = float(tm.travel_dist[1:].sum() / (L.frame.p * L.kb)) if tm.travel_dist.size > 1 else 0.0
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
    filled: np.ndarray | None = None   # bool (n,): filled cells (sparse layers change them)
    t_mat: np.ndarray | None = None    # fine layers: print time per material cell (ends: per cell too)

    @property
    def T(self) -> float:
        return self.timing.T


def evaluate_layer(L: Layer, motion: Motion, fea: FEAProblem | None, cfg: ScoreConfig,
                   bnd: np.ndarray, warm: FEAResult | None = None, direct: bool = False,
                   keep_factor: bool = True, wh: np.ndarray | None = None) -> LayerEval:
    from rltp.cover.cover import bead_weights
    pc = get_pieces(L)
    if L.kb > 1:                      # fine bead grid: strokes print as polylines (diagonals straight)
        from rltp.cover.beads import runs
        from rltp.eval.timing import time_polylines
        tm = time_polylines(L, pc, motion, runs(L)["poly"])
    else:
        tm = time_layer(L, pc, motion)
    ct = continuity(L, pc, tm, cfg, bnd)
    if L.kb > 1:                      # fine bead grid: material and directions from footprints
        from rltp.cover.beads import cell_fields, cell_shares
        from rltp.eval.fea2d import VOID
        cf = cell_fields(L, tm)
        D = cf["D"].reshape(-1)
        if fea is not None:
            fr = fea.solve(cf["wh"], warm=None if warm is None else warm.u, direct=direct, keep=keep_factor,
                           filled=D >= 0.5, dens=np.clip(D, VOID, 1.0), shares=cell_shares(L))
        else:
            z = np.zeros(L.n)
            fr = FEAResult(np.zeros(0), z, 0.0, 0, z, z, z)
        return LayerEval(tm, ct, fr, cf["ends"], D >= 0.5, cf["t"])
    if fea is not None:
        if wh is None:
            wh = bead_weights(L)
        fr = fea.solve(wh, warm=None if warm is None else warm.u, direct=direct, keep=keep_factor,
                       filled=L.flat_mask if L.sparse else None)
    else:
        z = np.zeros(L.n)
        fr = FEAResult(np.zeros(0), z, 0.0, 0, z, z, z)
    ends = end_mask(L) | seam_mask(L)
    return LayerEval(tm, ct, fr, ends, L.flat_mask)


LOG_TERMS = ("strength", "inter")   # -log FI_agg: a change is already a fraction of the current value
ABS_TERMS = ("infill", "cover")    # already fractions (infill off target; mis-covered share of the part)
NORM_MODES = ("relative", "std")


def full_weights(w) -> np.ndarray:
    """A weight per term; weights missing at the end (older 4-term vectors) are 0."""
    w = np.asarray(w, float).reshape(-1)
    return np.pad(w, (0, max(0, len(TERMS) - w.size)))


class Normalizer:
    """Per-term scales for the weighted score, plus per-edit statistics (Welford).

    mode "relative" (default): each term's change is divided by the part's baseline
    magnitude of that term, so every term is a fraction of where the part started.
    strength and inter are -log FI_agg, already fractional, so their scale is 1;
    time and cont are divided by the baseline print time and continuity cost. A
    weight then means "importance of a 1% change", whatever the part.

    mode "std" (the design doc's original): each term's change is divided by the
    standard deviation of its change over random edits. On real parts this lets
    a term with tiny per-edit changes but a large total (strength) dominate.

    Both modes track the per-edit spread; edit_std() gives the temperature unit.
    """

    def __init__(self, floor: float = 1e-6, prior: dict | None = None, prior_count: float = 0,
                 mode: str = "relative"):
        if mode not in NORM_MODES:
            raise ValueError(f"unknown normalisation {mode!r}; expected one of {NORM_MODES}")
        self.floor = floor
        self.mode = mode
        self.ref: dict | None = None
        self.n = {t: 0.0 for t in TERMS}
        self.mean = {t: 0.0 for t in TERMS}
        self.m2 = {t: 0.0 for t in TERMS}
        self.frozen = False
        if prior:
            for t, s in prior.items():
                self.n[t] = prior_count
                self.m2[t] = (s ** 2) * prior_count

    def set_reference(self, raw: dict) -> None:
        """Relative mode: take each term's scale from the part's baseline raw terms."""
        self.ref = {}
        for t in TERMS:
            v = abs(float(raw.get(t, 0.0)))
            self.ref[t] = 1.0 if t in LOG_TERMS or t in ABS_TERMS or v < 1e-9 else v

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

    def sigma(self, t: str) -> float:
        """Standard deviation of the term's raw change per edit."""
        if self.n[t] < 2:
            return 1.0
        s = float(np.sqrt(self.m2[t] / (self.n[t] - 1)))
        return s if s > self.floor else 1.0 if self.m2[t] == 0 else self.floor

    def scale(self, t: str) -> float:
        """What a term's raw change is divided by before weighting."""
        if self.mode == "relative":
            return self.ref[t] if self.ref is not None else 1.0
        return self.sigma(t)

    def scales(self) -> dict:
        return {t: self.scale(t) for t in TERMS}

    def total(self, d: dict, w) -> float:
        w = full_weights(w)
        return float(sum(w[i] * d.get(t, 0.0) / self.scale(t) for i, t in enumerate(TERMS)))

    def edit_std(self, w) -> float:
        """Typical size of one random edit's change in the weighted score: the unit
        that annealing temperatures are given in. In std mode the terms are already in
        these units (1.0); in relative mode it comes from the tracked per-edit spread."""
        if self.mode == "std":
            return 1.0
        w = full_weights(w)
        v = sum((w[i] * self.sigma(t) / self.scale(t)) ** 2
                for i, t in enumerate(TERMS) if w[i] and self.n[t] >= 2)
        return float(np.sqrt(v)) if v > 0 else 1.0

    def state(self) -> dict:
        return {"n": dict(self.n), "mean": dict(self.mean), "m2": dict(self.m2), "frozen": self.frozen,
                "mode": self.mode, "ref": None if self.ref is None else dict(self.ref)}

    def load(self, s: dict) -> "Normalizer":
        self.n, self.mean, self.m2 = dict(s["n"]), dict(s["mean"]), dict(s["m2"])
        for t in TERMS:                     # states saved before a term existed
            self.n.setdefault(t, 0.0)
            self.mean.setdefault(t, 0.0)
            self.m2.setdefault(t, 0.0)
        self.frozen = s.get("frozen", False)
        self.mode = s.get("mode", "std")    # states saved before modes existed were std
        self.ref = dict(s["ref"]) if s.get("ref") else None
        if self.ref is not None:
            for t in TERMS:
                self.ref.setdefault(t, 1.0)
        return self


def strength_raw(S_p: float, N: int, p: float) -> float:
    if N == 0 or S_p <= 0:
        return 0.0
    return float(-np.log((S_p / N) ** (1 / p)))
