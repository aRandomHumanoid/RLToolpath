"""PartState: layers, spec, cached evaluations, interfaces and the score.

Incremental scoring: an edit to layer k re-evaluates only layer k (warm-started
FEA) and, with the interlayer term on, interfaces (k-1, k) and (k, k+1).
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from rltp.config import TERMS, PartSpec
from rltp.cover.baselines import fill_baseline
from rltp.cover.cover import boundary_mask
from rltp.eval.fea2d import FEAProblem, dominant_direction, isotropic
from rltp.eval.interlayer import IfaceResult, Interface, build_interface, eval_interface
from rltp.eval.metrics import LayerEval, Normalizer, evaluate_layer, strength_raw
from rltp.eval.timing import Motion
from rltp.model import Layer
from rltp.slicing.raster import make_frame, rasterize
from rltp.slicing.slice import footprint, slice_mesh

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    k: int
    layer: Layer
    ev: LayerEval
    ifaces: dict             # j -> IfaceResult
    d: dict                  # raw term changes


class PartState:
    def __init__(self, layers: list[Layer], spec: PartSpec, z0: float = 0.0, fea_cache: int = 6,
                 fea_layers=None):
        self.layers = layers
        self.fea_layers = None if fea_layers is None else set(fea_layers)
        self.spec = spec
        self.z0 = z0
        self.terms = tuple(spec.terms)
        self.p = spec.score.p_norm
        self.norm = Normalizer(spec.score.sigma_floor)
        self._fea: OrderedDict = OrderedDict()
        self._fea_cache = fea_cache
        self.bnd = [boundary_mask(L.mask) for L in layers]
        self.motions = [Motion.from_printer(spec.printer, first=L.first) for L in layers]
        self.evals: list[LayerEval] = [None] * len(layers)
        self.ifaces: list[Interface | None] = []
        self.iface_res: list[IfaceResult | None] = []
        self.n_evals = 0
        self.best_total = -np.inf
        self.best_snap = None
        self.best_raw = None
        self._build_interfaces()
        self.full_recheck()

    # -------------------------------------------------------------- setup

    @property
    def use_inter(self) -> bool:
        return "inter" in self.terms and len(self.layers) > 1

    def fea(self, k: int) -> FEAProblem | None:
        if not self.spec.loads.has_2d:
            return None
        if self.fea_layers is not None and k not in self.fea_layers:
            return None
        if k in self._fea:
            self._fea.move_to_end(k)
            return self._fea[k]
        L = self.layers[k]
        scale = L.h / self.spec.part_height if self.spec.loads.split_by_height else 1.0
        fp = FEAProblem(L.mask, L.frame, L.h, self.spec.material, self.spec.loads, scale,
                        z=L.z - L.h / 2 + self.z0, p_norm=self.p)
        self._fea[k] = fp
        while len(self._fea) > self._fea_cache:
            self._fea.popitem(last=False)
        return fp

    def _build_interfaces(self) -> None:
        self.ifaces = []
        if not self.use_inter:
            return
        pls = []
        for pl in self.spec.loads.point_loads:
            x, y, z = pl.pos
            pls.append(type(pl)((x, y, z - self.z0), pl.force))
        for j in range(len(self.layers) - 1):
            lo, up = self.layers[j], self.layers[j + 1]
            self.ifaces.append(build_interface(j, lo, up, lo.z, pls, self.spec.material))

    # -------------------------------------------------------------- evaluation

    def _eval_layer(self, k: int, L: Layer, warm=None, direct=False) -> LayerEval:
        return evaluate_layer(L, self.motions[k], self.fea(k), self.spec.score, self.bnd[k],
                              warm=warm, direct=direct)

    def _eval_iface(self, j: int, ev_lo: LayerEval, ev_up: LayerEval) -> IfaceResult | None:
        itf = self.ifaces[j]
        if itf is None:
            return None
        return eval_interface(itf, ev_lo.timing.t_cell, ev_lo.ends, ev_lo.T, ev_up.timing.t_cell,
                              ev_up.ends, self.spec.bond, self.spec.material,
                              self.spec.printer.layer_change_time, self.p)

    def full_recheck(self) -> float:
        """Fresh evaluation of every layer (direct solves). Returns the drift in the
        raw strength term versus the incremental bookkeeping (0 on first call)."""
        before = self.raw() if self.evals[0] is not None else None
        for k, L in enumerate(self.layers):
            fp = self.fea(k)
            if fp is not None:
                fp.drop_factorization()
            self.evals[k] = self._eval_layer(k, L, direct=True)
        self.iface_res = [self._eval_iface(j, self.evals[j], self.evals[j + 1])
                          for j in range(len(self.ifaces))]
        self._recount()
        if before is None:
            return 0.0
        after = self.raw()
        return max(abs(after[t] - before[t]) for t in TERMS)

    def _recount(self) -> None:
        self.Sp = np.array([e.fea.S_p for e in self.evals])
        self.Np = np.array([e.fea.N for e in self.evals])
        self.Tk = np.array([e.T for e in self.evals])
        self.Ck = np.array([e.cont.C for e in self.evals])
        self.ISp = np.array([r.S_p if r else 0.0 for r in self.iface_res])
        self.INp = np.array([r.N if r else 0 for r in self.iface_res])

    def raw(self) -> dict:
        out = {
            "strength": strength_raw(self.Sp.sum(), int(self.Np.sum()), self.p),
            "time": -float(self.Tk.sum()),
            "cont": -float(self.Ck.sum()),
            "inter": strength_raw(self.ISp.sum(), int(self.INp.sum()), self.p) if self.use_inter else 0.0,
        }
        for t in TERMS:
            if t not in self.terms:
                out[t] = 0.0
        return out

    def total(self, w) -> float:
        return self.norm.total(self.raw(), w)

    def propose(self, k: int, cand_layer: Layer) -> Candidate:
        """Score a candidate replacement for layer k without committing it."""
        self.n_evals += 1
        old = self.evals[k]
        ev = self._eval_layer(k, cand_layer, warm=old.fea)
        d = {t: 0.0 for t in TERMS}
        if "strength" in self.terms:
            Sp = self.Sp.sum() - old.fea.S_p + ev.fea.S_p
            d["strength"] = strength_raw(Sp, int(self.Np.sum()), self.p) - \
                strength_raw(self.Sp.sum(), int(self.Np.sum()), self.p)
        if "time" in self.terms:
            d["time"] = -(ev.T - old.T)
        if "cont" in self.terms:
            d["cont"] = -(ev.cont.C - old.cont.C)
        ifs = {}
        if self.use_inter:
            for j in (k - 1, k):
                if 0 <= j < len(self.ifaces) and self.ifaces[j] is not None:
                    lo = ev if j == k else self.evals[j]
                    up = ev if j + 1 == k else self.evals[j + 1]
                    ifs[j] = self._eval_iface(j, lo, up)
            ISp = self.ISp.sum() + sum(ifs[j].S_p - self.ISp[j] for j in ifs)
            N = int(self.INp.sum())
            d["inter"] = strength_raw(ISp, N, self.p) - strength_raw(self.ISp.sum(), N, self.p)
        return Candidate(k, cand_layer, ev, ifs, d)

    def commit(self, c: Candidate) -> None:
        k = c.k
        self.layers[k] = c.layer
        self.evals[k] = c.ev
        self.Sp[k], self.Np[k] = c.ev.fea.S_p, c.ev.fea.N
        self.Tk[k], self.Ck[k] = c.ev.T, c.ev.cont.C
        for j, r in c.ifaces.items():
            self.iface_res[j] = r
            self.ISp[j], self.INp[j] = r.S_p, r.N

    # -------------------------------------------------------------- best tracking

    def snapshot(self):
        return [(L.nxt.copy(), L.prv.copy(), set(L.seam)) for L in self.layers]

    def restore(self, snap) -> None:
        for L, (nxt, prv, seam) in zip(self.layers, snap):
            L.nxt[:] = nxt
            L.prv[:] = prv
            L.seam = set(seam)
            L.invalidate()
        self.full_recheck()

    def keep_best(self, w) -> bool:
        tot = self.total(w)
        if tot > self.best_total:
            self.best_total = tot
            self.best_snap = self.snapshot()
            self.best_raw = self.raw()
            return True
        return False

    def fi_share(self) -> np.ndarray:
        s = self.Sp.sum()
        return self.Sp / s if s > 0 else np.full(len(self.layers), 1 / max(len(self.layers), 1))

    def breakdown(self) -> dict:
        p = self.p
        Nt = int(self.Np.sum())
        return {
            "raw": self.raw(),
            "fi_agg": float((self.Sp.sum() / Nt) ** (1 / p)) if Nt else 0.0,
            "inter_fi_agg": float((self.ISp.sum() / self.INp.sum()) ** (1 / p)) if self.INp.sum() else 0.0,
            "time_s": float(self.Tk.sum()),
            "strokes": int(sum(e.cont.n_strokes for e in self.evals)),
            "travel_mm": float(sum(e.timing.travel_dist[1:].sum() for e in self.evals)),
            "retractions": int(sum(e.timing.retract[1:].sum() for e in self.evals)),
            "boundary_ends": int(sum(e.cont.n_boundary_ends for e in self.evals)),
            "layers": len(self.layers),
            "cells": int(sum(L.flat_mask.sum() for L in self.layers)),
        }


# ------------------------------------------------------------------ building parts

def _layer_theta(geom, p, h, spec: PartSpec, frame0, H0, W0, z) -> float:
    """v0.5: dominant principal stress direction from one isotropic FEA of the layer."""
    if not spec.loads.has_2d:
        return 0.0
    mask, _ = rasterize(geom, frame0, H0, W0)
    if mask.sum() < 4:
        return 0.0
    m = spec.material
    iso = isotropic(0.5 * (m.E1 + m.E2), m.nu12)
    fp = FEAProblem(mask, frame0, h, iso, spec.loads, 1.0, z=z)
    if not fp.active:
        return 0.0
    r = fp.solve(np.full(mask.size, 0.5), direct=True, keep=False)
    return dominant_direction(r.sx[fp.cells], r.sy[fp.cells], r.txy[fp.cells]) + frame0.theta


def build_part(mesh, spec: PartSpec, baseline: str = "zigzag", rotate: bool = False,
               min_cells: int = 3) -> PartState:
    pr = spec.printer
    slices = slice_mesh(mesh, pr.layer_height, pr.first_layer_height)
    z0 = float(mesh.bounds[0, 2])
    fp = footprint(slices)
    spec.loads.resolve(fp.bounds)
    spec.part_height = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])
    p = pr.bead_width
    frame0, H0, W0 = make_frame(fp, 0.0, p)
    layers = []
    for s in slices:
        if s.geom.is_empty:
            continue
        if rotate:
            th = _layer_theta(s.geom, p, s.h, spec, frame0, H0, W0, s.z_cut + z0)
            frame, H, W = make_frame(fp, th, p)
        else:
            frame, H, W = frame0, H0, W0
        mask, dropped = rasterize(s.geom, frame, H, W, min_cells)
        for cnt, xy in dropped:
            log.info("layer %d: dropped %d-cell piece near (%.2f, %.2f)", s.k, cnt, *xy)
        if not mask.any():
            log.warning("layer %d rasterised to nothing; skipped", s.k)
            continue
        L = Layer.empty(len(layers), s.z_top, s.h, mask, frame, poly=s.geom, first=(s.k == 0))
        fill_baseline(L, baseline)
        layers.append(L)
    if not layers:
        raise ValueError("no printable layers")
    return PartState(layers, spec, z0=z0)
