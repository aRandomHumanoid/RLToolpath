"""PartState: layers, spec, cached evaluations, interfaces and the score.

Incremental scoring: an edit to layer k re-evaluates only layer k (warm-started
FEA) and, with the interlayer term on, interfaces (k-1, k) and (k, k+1).

Stack models for in-plane strength:
  split     each layer is its own plate carrying h / height of the load (the original v0).
  laminate  the layers are bonded and stretch together (equal in-plane strain, as in
            classical laminate theory): one plate whose stiffness is the sum of every
            layer's, under the whole load; each layer's stresses then follow from the
            shared strain and its own bead directions and fill. Stiff layers carry more
            load, so sparse middle layers aren't charged a solid layer's share. Needs one
            shared grid. An edit to layer k still costs one solve, but changes every
            layer's stresses.
"auto" (the default) is laminate for sparse parts and split otherwise.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from rltp.config import TERMS, PartSpec
from rltp.cover.baselines import fill_baseline
from rltp.cover.cover import bead_weights, boundary_mask
from rltp.cover.shell import (disk, fill_sparse, line_density, solid_masks, support_ctx, support_radius,
                              unsupported)
from rltp.eval.fea2d import VOID, FEAProblem, dominant_direction, isotropic
from rltp.eval.interlayer import (IfaceResult, Interface, build_interface, build_zinterface, eval_interface,
                                  eval_zinterface)
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
    lam: tuple | None = None  # laminate: (u, per-layer results, c0, c90, wh, fill) after the edit


class PartState:
    def __init__(self, layers: list[Layer], spec: PartSpec, z0: float = 0.0, fea_cache: int = 6,
                 fea_layers=None):
        self.layers = layers
        self.fea_layers = None if fea_layers is None else set(fea_layers)
        self.spec = spec
        self.z0 = z0
        self.terms = tuple(spec.terms)
        self.p = spec.score.p_norm
        self.norm = Normalizer(spec.score.sigma_floor, mode=spec.score.norm)
        self._fea: OrderedDict = OrderedDict()
        self._fea_cache = fea_cache
        self.sparse = any(L.sparse for L in layers)
        self.support = None            # disk of the support rule (sparse parts)
        if self.sparse:
            if len({(L.frame, L.H, L.W) for L in layers}) > 1:
                raise ValueError("sparse layers need one shared grid (no per-layer rotation)")
            self.support = disk(support_radius(spec.printer.max_bridge, layers[0].material_frame.p))
            pr = spec.printer
            self.infill_target = float(pr.infill)
            self.opt_total = int(sum((L.part & ~L.req).sum() for L in layers))
            self.Fk = np.array([self._opt_filled(L) for L in layers])
            if pr.infill_rule not in ("both", "band", "penalty", "free"):
                raise ValueError(f"unknown infill rule {pr.infill_rule!r}; expected both, band, penalty or free")
            self.budget = None         # band rule: shared by every layer's support context
            if pr.infill_rule in ("band", "both") and self.opt_total:
                self.budget = {"lo": (pr.infill - pr.infill_band) * self.opt_total,
                               "hi": (pr.infill + pr.infill_band) * self.opt_total,
                               "total": int(self.Fk.sum())}
            self.refresh_support()
        # path ends near the outline: in-part cells on the edge, or (fine grid) the edge of
        # the region where a bead's footprint fits
        self.bnd = [boundary_mask(L.aux["region"] if L.kb > 1 else L.part_mask) for L in layers]
        self.fine = any(L.kb > 1 for L in layers)
        stack = getattr(spec.score, "stack", "auto")
        if stack not in ("auto", "split", "laminate"):
            raise ValueError(f"unknown stack model {stack!r}; expected auto, split or laminate")
        self.laminate = stack == "laminate" or (stack == "auto" and self.sparse)
        if self.laminate and len({(L.frame, L.H, L.W) for L in layers}) > 1:
            raise ValueError("the laminate stack model needs one shared grid (no per-layer rotation)")
        self.lam: FEAProblem | None = None     # laminate: one plate over the union of the layers
        self.motions = [Motion.from_printer(spec.printer, first=L.first) for L in layers]
        self.evals: list[LayerEval] = [None] * len(layers)
        self.ifaces: list[Interface | None] = []
        self.iface_res: list[IfaceResult | None] = []
        self.n_evals = 0
        self.best_total = -np.inf
        self.best_snap = None
        self.best_raw = None
        self.case_scale = None        # 1 / baseline FI_agg per load case (several cases only)
        self.case_fi0 = None
        self._build_interfaces()
        self.full_recheck()
        if spec.loads.n_cases > 1 and spec.loads.has_2d:
            # score every case against its own starting FI_agg, so force sizes don't matter
            # and the p-norm over cases favours whichever case is furthest behind
            fi0 = self.case_fi_agg()
            self.case_fi0 = fi0
            self.case_scale = np.where(fi0 > 0, 1.0 / np.where(fi0 > 0, fi0, 1.0), 0.0)
            self._fea.clear()
            self.lam = None
            self.full_recheck()
        if self.norm.mode == "relative":
            self.norm.set_reference(self.raw())   # scores are fractions of this starting state

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
        fp = FEAProblem(L.part_mask, L.material_frame, L.h, self.spec.material, self.spec.loads, scale,
                        z=L.z - L.h / 2 + self.z0, p_norm=self.p, case_scale=self.case_scale,
                        refine=getattr(self.spec.score, "fea_refine", 1),
                        solver=getattr(self.spec.score, "solver", "fast"))
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
            if self.sparse:      # bonded area changes with the fill: see ZInterface
                self.ifaces.append(build_zinterface(j, lo, up, lo.z, pls, self.spec.material))
            else:
                self.ifaces.append(build_interface(j, lo, up, lo.z, pls, self.spec.material))

    # -------------------------------------------------------------- evaluation

    def _eval_layer(self, k: int, L: Layer, warm=None, direct=False) -> LayerEval:
        fea = None if self.laminate else self.fea(k)     # laminate: strength comes from _lam_*
        return evaluate_layer(L, self.motions[k], fea, self.spec.score, self.bnd[k],
                              warm=warm, direct=direct)

    # -------------------------------------------------------------- laminate

    def _lam_problem(self) -> FEAProblem | None:
        if self.lam is None and self.spec.loads.has_2d:
            union = np.zeros_like(self.layers[0].part_mask)
            for L in self.layers:
                union |= L.part_mask
            L0 = self.layers[0]
            # unit thickness: each layer's h enters through its stiffness multipliers;
            # the whole load acts on the stack (no split by height), z limits are ignored
            self.lam = FEAProblem(union, L0.material_frame, 1.0, self.spec.material, self.spec.loads, 1.0,
                                  z=None, p_norm=self.p, case_scale=self.case_scale,
                                  refine=getattr(self.spec.score, "fea_refine", 1),
                                  solver=getattr(self.spec.score, "solver", "fast"))
        return self.lam if self.lam is not None and self.lam.active else None

    def _lam_layer(self, L: Layer):
        """(c0, c90, wh, fill, diag) of one layer over the laminate's cells: its stiffness
        multipliers (thickness x fill x bead fraction), bead weights, filled flags, and for a
        layer with diagonal beads diag = (c45, c135, shares) (else None)."""
        cells = self.lam.cells
        inpart = L.part_mask.reshape(-1)[cells]
        if L.kb > 1:                      # fine bead grid: stiffness follows the material
            from rltp.cover.beads import cell_shares, cell_wh
            D = L.density.reshape(-1)[cells]
            fill = D >= 0.5
            dens = np.where(inpart, np.clip(D, VOID, 1.0), 0.0) * L.h
            sh = cell_shares(L)
            if sh is not None:
                sh = sh[cells]
                wh = sh[:, 0] + 0.5 * (sh[:, 1] + sh[:, 3])
                return dens * sh[:, 0], dens * sh[:, 2], wh, fill, (dens * sh[:, 1], dens * sh[:, 3], sh)
            wh = cell_wh(L)[cells]
            return dens * wh, dens * (1 - wh), wh, fill, None
        wh = bead_weights(L)[cells]
        fill = L.flat_mask[cells]
        dens = np.where(fill, 1.0, np.where(inpart, VOID, 0.0)) * L.h
        return dens * wh, dens * (1 - wh), wh, fill, None

    def _lam_K(self, a0, a90, diags):
        """Laminate K; the 45/135-degree terms only when some layer has diagonal beads."""
        if all(d is None for d in diags):
            return self.lam.assemble_coeffs(a0, a90)
        z = np.zeros_like(a0)
        a45 = sum((d[0] for d in diags if d is not None), z)
        a135 = sum((d[1] for d in diags if d is not None), z)
        return self.lam.assemble_coeffs(a0, a90, a45, a135)

    def _lam_shares(self, diags, whs):
        """Per-layer (0, 45, 90, 135) shares for post_layers, or None if no diagonals."""
        if all(d is None for d in diags):
            return None
        return [d[2] if d is not None else np.stack([w, np.zeros_like(w), 1 - w, np.zeros_like(w)], -1)
                for d, w in zip(diags, whs)]

    def _lam_full(self) -> None:
        lam = self._lam_problem()
        if lam is None:
            return
        cols = [self._lam_layer(L) for L in self.layers]
        self.lam_c0 = [c[0] for c in cols]
        self.lam_c90 = [c[1] for c in cols]
        self.lam_wh = [c[2] for c in cols]
        self.lam_fill = [c[3] for c in cols]
        self.lam_diag = [c[4] for c in cols]
        self.lam_N = [int(L.part_mask.sum()) for L in self.layers]
        self.lam_a0 = np.sum(self.lam_c0, 0)
        self.lam_a90 = np.sum(self.lam_c90, 0)
        lam.drop_factorization()
        u, iters, direct = lam.solve_K(self._lam_K(self.lam_a0, self.lam_a90, self.lam_diag), direct=True)
        self.lam_u = u
        res = lam.post_layers(u, self.lam_wh, self.lam_fill, self.lam_N, iters, direct,
                              shares=self._lam_shares(self.lam_diag, self.lam_wh))
        for ev, r in zip(self.evals, res):
            ev.fea = r

    def _lam_propose(self, k: int, cand: Layer):
        lam = self._lam_problem()
        if lam is None:
            return None
        c0, c90, wh, fill, dg = self._lam_layer(cand)
        a0 = self.lam_a0 - self.lam_c0[k] + c0
        a90 = self.lam_a90 - self.lam_c90[k] + c90
        diags = list(self.lam_diag)
        diags[k] = dg
        u, iters, direct = lam.solve_K(self._lam_K(a0, a90, diags), warm=self.lam_u)
        whs = list(self.lam_wh)
        fills = list(self.lam_fill)
        whs[k], fills[k] = wh, fill
        res = lam.post_layers(u, whs, fills, self.lam_N, iters, direct, shares=self._lam_shares(diags, whs))
        return u, res, c0, c90, wh, fill, dg

    def _eval_iface(self, j: int, ev_lo: LayerEval, ev_up: LayerEval) -> IfaceResult | None:
        itf = self.ifaces[j]
        if itf is None:
            return None
        if self.sparse:
            t_lo = ev_lo.t_mat if ev_lo.t_mat is not None else ev_lo.timing.t_cell
            t_up = ev_up.t_mat if ev_up.t_mat is not None else ev_up.timing.t_cell
            return eval_zinterface(itf, ev_lo.filled, ev_up.filled, t_lo, ev_lo.ends, ev_lo.T,
                                   t_up, ev_up.ends, self.spec.bond, self.spec.material,
                                   self.spec.printer.layer_change_time, self.p)
        return eval_interface(itf, ev_lo.timing.t_cell, ev_lo.ends, ev_lo.T, ev_up.timing.t_cell,
                              ev_up.ends, self.spec.bond, self.spec.material,
                              self.spec.printer.layer_change_time, self.p)

    def full_recheck(self) -> float:
        """Fresh evaluation of every layer (direct solves). Returns the drift in the
        raw strength term versus the incremental bookkeeping (0 on first call)."""
        before = self.raw() if self.evals[0] is not None else None
        for k, L in enumerate(self.layers):
            fp = None if self.laminate else self.fea(k)
            if fp is not None:
                fp.drop_factorization()
            self.evals[k] = self._eval_layer(k, L, direct=True)
        if self.laminate:
            self._lam_full()
        self._share_fi()
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
        if self.sparse:
            self.Fk = np.array([self._opt_filled(L) for L in self.layers])
            if self.budget is not None:
                self.budget["total"] = int(self.Fk.sum())
        if self.fine:
            gk = [self._miscover(L) for L in self.layers]
            self.Gk = np.array([g[0] for g in gk])
            self.Ok = np.array([g[1] for g in gk])
            self.cover_N = int(sum(L.part_mask.sum() for L in self.layers))

    @staticmethod
    def _miscover(L: Layer) -> tuple[float, float]:
        """Fine grid: (gap, over-extrusion) in cells: required cells' shortfall below 1,
        and every cell's excess above 1."""
        D = L.density
        return float(np.maximum(1.0 - D, 0.0)[L.req].sum()), float(np.maximum(D - 1.0, 0.0).sum())

    def _cover_raw(self, gaps: float, over: float) -> float:
        return -(gaps + over) / self.cover_N if self.cover_N else 0.0

    def raw(self) -> dict:
        out = {
            "strength": strength_raw(self.Sp.sum(), int(self.Np.sum()), self.p),
            "time": -float(self.Tk.sum()),
            "cont": -float(self.Ck.sum()),
            "inter": strength_raw(self.ISp.sum(), int(self.INp.sum()), self.p) if self.use_inter else 0.0,
            "infill": self._infill_raw(self.Fk.sum()) if self.sparse else 0.0,
            "cover": self._cover_raw(self.Gk.sum(), self.Ok.sum()) if self.fine else 0.0,
        }
        for t in TERMS:
            if t not in self.terms:
                out[t] = 0.0
        return out

    def _infill_raw(self, filled: float) -> float:
        """Minus the distance of the part's infill fraction from the target."""
        return -abs(filled / self.opt_total - self.infill_target) if self.opt_total else 0.0

    def total(self, w) -> float:
        return self.norm.total(self.raw(), w)

    def propose(self, k: int, cand_layer: Layer) -> Candidate:
        """Score a candidate replacement for layer k without committing it."""
        self.n_evals += 1
        old = self.evals[k]
        ev = self._eval_layer(k, cand_layer, warm=old.fea)
        d = {t: 0.0 for t in TERMS}
        lam = self._lam_propose(k, cand_layer) if self.laminate else None
        if lam is not None:
            ev.fea = lam[1][k]
        if "strength" in self.terms and lam is not None:
            Sp = sum(r.S_p for r in lam[1])
            d["strength"] = strength_raw(Sp, int(self.Np.sum()), self.p) - \
                strength_raw(self.Sp.sum(), int(self.Np.sum()), self.p)
        elif "strength" in self.terms:
            Sp = self.Sp.sum() - old.fea.S_p + ev.fea.S_p
            d["strength"] = strength_raw(Sp, int(self.Np.sum()), self.p) - \
                strength_raw(self.Sp.sum(), int(self.Np.sum()), self.p)
        if "time" in self.terms:
            d["time"] = -(ev.T - old.T)
        if "cont" in self.terms:
            d["cont"] = -(ev.cont.C - old.cont.C)
        if "infill" in self.terms and self.sparse:
            F = self.Fk.sum()
            d["infill"] = self._infill_raw(F - self.Fk[k] + self._opt_filled(cand_layer)) - self._infill_raw(F)
        if "cover" in self.terms and self.fine:
            g, o = self._miscover(cand_layer)
            G, O = self.Gk.sum(), self.Ok.sum()
            d["cover"] = self._cover_raw(G - self.Gk[k] + g, O - self.Ok[k] + o) - self._cover_raw(G, O)
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
        return Candidate(k, cand_layer, ev, ifs, d, lam)

    def _share_fi(self) -> None:
        if not self.sparse:
            return
        for L, ev in zip(self.layers, self.evals):
            if L.ctx is not None and ev is not None:
                L.ctx["fi"] = ev.fea.fi

    def refresh_support(self, ks=None) -> None:
        """Recompute the support context of layers `ks` (all by default) after a change."""
        if not self.sparse:
            return
        for k in (range(len(self.layers)) if ks is None else ks):
            if 0 <= k < len(self.layers):
                self.layers[k].ctx = support_ctx(self.layers, k, self.support)
                self.layers[k].ctx["budget"] = self.budget

    @staticmethod
    def _opt_filled(L: Layer) -> int:
        return int((L.material & ~L.req).sum())

    def proposal_weights(self, k: int) -> np.ndarray:
        """Per cell of layer k: (local FI / layer max)^2 + 0.02, FI smoothed over about a bead
        and a half, so stress-biased proposals land near the stressed material."""
        from scipy import ndimage
        L = self.layers[k]
        fi = np.asarray(self.evals[k].fea.fi, float)
        if fi.size != L.n or not np.isfinite(fi).any() or fi.max() <= 0:
            return np.ones(L.n)
        size = max(3, int(round(1.5 * L.kb)) | 1) if L.kb > 1 else 3
        fs = ndimage.uniform_filter(fi.reshape(L.H, L.W), size=size, mode="constant").reshape(-1)
        mx = fs.max()
        return (fs / mx) ** 2 + 0.02 if mx > 0 else np.ones(L.n)

    def infill_fraction(self) -> float:
        return float(self.Fk.sum() / self.opt_total) if self.sparse and self.opt_total else 1.0

    def commit(self, c: Candidate) -> None:
        k = c.k
        self.layers[k] = c.layer
        if self.fine:
            self.Gk[k], self.Ok[k] = self._miscover(c.layer)
        if self.sparse:
            self.Fk[k] = self._opt_filled(c.layer)
            if self.budget is not None:
                self.budget["total"] = int(self.Fk.sum())
            self.refresh_support((k - 1, k, k + 1))
        self.evals[k] = c.ev
        self.Sp[k], self.Np[k] = c.ev.fea.S_p, c.ev.fea.N
        if c.lam is not None:            # every layer's stresses moved with the shared strain
            u, res, c0, c90, wh, fill, dg = c.lam
            self.lam_a0 = self.lam_a0 - self.lam_c0[k] + c0
            self.lam_a90 = self.lam_a90 - self.lam_c90[k] + c90
            self.lam_c0[k], self.lam_c90[k], self.lam_wh[k], self.lam_fill[k] = c0, c90, wh, fill
            self.lam_diag[k] = dg
            self.lam_u = u
            for j, r in enumerate(res):
                self.evals[j].fea = r
                self.Sp[j] = r.S_p
        self._share_fi()
        self.Tk[k], self.Ck[k] = c.ev.T, c.ev.cont.C
        for j, r in c.ifaces.items():
            self.iface_res[j] = r
            self.ISp[j], self.INp[j] = r.S_p, r.N

    # -------------------------------------------------------------- best tracking

    def snapshot(self):
        """Per layer (nxt, prv, seam), plus the filled mask for a sparse layer."""
        return [(L.nxt.copy(), L.prv.copy(), set(L.seam)) + ((L.mask.copy(),) if L.sparse else ())
                for L in self.layers]

    def restore(self, snap) -> None:
        for L, (nxt, prv, seam, *rest) in zip(self.layers, snap):
            L.nxt[:] = nxt
            L.prv[:] = prv
            L.seam = set(seam)
            if rest:
                L.mask = np.asarray(rest[0], bool).reshape(L.H, L.W).copy()
            L.invalidate()
        self.refresh_support()
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

    def case_fi_agg(self) -> np.ndarray | None:
        """Raw FI_agg of each load case over the part (None with a single case)."""
        spc = [e.fea.S_pc for e in self.evals if e.fea.S_pc is not None]
        if not spc:
            return None
        N = int(self.Np.sum())
        if not N:
            return np.zeros(len(spc[0]))
        return (np.sum(spc, 0) / N) ** (1 / self.p)

    def breakdown(self) -> dict:
        p = self.p
        Nt = int(self.Np.sum())
        cases = self.case_fi_agg()
        extra = {} if cases is None else {"case_fi_agg": [float(x) for x in cases]}
        if self.sparse:
            opt = sum(int((L.part & ~L.req).sum()) for L in self.layers)
            opt_filled = sum(int((L.material & ~L.req).sum()) for L in self.layers)
            shared = bonded = 0
            weakest = 100.0
            for itf in self.ifaces:
                if itf is None:
                    continue
                b = int((self.layers[itf.j].material.reshape(-1)[itf.lo]
                         & self.layers[itf.j + 1].material.reshape(-1)[itf.up]).sum())
                shared += itf.N
                bonded += b
                weakest = min(weakest, 100.0 * b / itf.N)
            if shared:
                extra.update({"bonded_pct": 100.0 * bonded / shared, "weakest_bond_pct": weakest})
            if self.fine:
                extra.update({"bead_nodes": int(sum(L.mask.sum() for L in self.layers)),
                              "gap_pct": 100.0 * float(self.Gk.sum()) / max(self.cover_N, 1),
                              "over_pct": 100.0 * float(self.Ok.sum()) / max(self.cover_N, 1)})
            extra.update({"part_cells": int(sum(L.part.sum() for L in self.layers)),
                          "required_cells": int(sum(L.req.sum() for L in self.layers)),
                          "infill_pct": 100.0 * opt_filled / opt if opt else 0.0,
                          "unsupported": int(sum(unsupported(self.layers, self.support)))})
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
            "cells": int(sum(L.material.sum() for L in self.layers)),
            **extra,
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
               min_cells: int = 3, fields=None) -> PartState:
    """Slice, rasterise and lay out the starting toolpaths. fields: FieldOptions for the
    gradient starts (rltp.search.fields; defaults when None)."""
    pr = spec.printer
    slices = slice_mesh(mesh, pr.layer_height, pr.first_layer_height)
    z0 = float(mesh.bounds[0, 2])
    fp = footprint(slices)
    spec.loads.resolve(fp.bounds)
    spec.loads.add_random_cases(fp, pr.bead_width, fp.bounds)
    spec.part_height = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])
    kb = pr.kb
    p = pr.pitch                      # grid cell: the bead width, or finer (fine bead grid)
    frame0, H0, W0 = make_frame(fp, 0.0, p)
    if pr.sparse:
        if rotate:
            raise ValueError("--rotate isn't supported with walls/infill yet (sparse layers share one grid)")
        if baseline not in ("zigzag", "zigzag0", "zigzag90"):
            raise ValueError("walls/infill start from a zigzag; use --fill zigzag or zigzag90")
    layers = []
    for s in slices:
        if s.geom.is_empty:
            continue
        if rotate:
            th = _layer_theta(s.geom, p, s.h, spec, frame0, H0, W0, s.z_cut + z0)
            frame, H, W = make_frame(fp, th, p)
        else:
            frame, H, W = frame0, H0, W0
        mask, dropped = rasterize(s.geom, frame, H, W, min_cells * kb * kb)
        for cnt, xy in dropped:
            log.info("layer %d: dropped %d-cell piece near (%.2f, %.2f)", s.k, cnt, *xy)
        if not mask.any():
            log.warning("layer %d rasterised to nothing; skipped", s.k)
            continue
        L = Layer.empty(len(layers), s.z_top, s.h, mask, frame, poly=s.geom, first=(s.k == 0))
        if not pr.sparse:
            fill_baseline(L, baseline)
        layers.append(L)
    if not layers:
        raise ValueError("no printable layers")
    if pr.sparse:
        radius = support_radius(pr.max_bridge, p)
        density = line_density(pr.infill, radius)
        if density > pr.infill + 1e-9:
            log.warning("infill %.2f is below what %.1f mm bridges allow; starting at %.2f",
                        pr.infill, pr.max_bridge, density)
        base = 90 if baseline == "zigzag90" else 0
        parts = [L.mask.copy() for L in layers]
        solids = solid_masks(parts, pr.solid_layers)
        orient = sweight = None
        field = None
        if pr.start not in ("slicer", "stress", "stress4", "gradient", "gradient4"):
            raise ValueError(f"unknown start {pr.start!r}; expected slicer, stress, stress4, gradient "
                             f"or gradient4")
        if pr.start in ("stress", "stress4", "gradient", "gradient4"):
            from rltp.eval.stressmap import orientation, orientation4, stress_field
            union = np.zeros_like(parts[0])
            for m_ in parts:
                union |= m_
            field = stress_field(union, layers[0].frame, spec)
            if field is not None:          # directions only: material stays evenly spread, so
                win = max(int(round(2.0 / p)), 1)                           # (~2 mm smoothing)
                # a fine grid can lay 45/135-degree beads too; the 0.4 mm grid only 0/90
                big = int(round((3.0 / p) ** 2))                           # diagonal areas >= 3 x 3 mm
                # stress4 (fine grid only) also starts 45/135-degree areas; on the bracket that
                # tripled the stroke count for no strength gain, so "stress" keeps 0/90 and
                # leaves diagonals to the edits
                if pr.start == "stress4" and kb > 1:
                    orient = orientation4(field, win, big)
                else:
                    orient = orientation(field, win)
                # extra infill where stress is high is left to the optimizer to find
                # (stressmap.weight + the layouts' sweight would pack it in from the start)
        if pr.infill_pattern not in ("alternate", "stacked"):
            raise ValueError(f"unknown infill pattern {pr.infill_pattern!r}; expected alternate or stacked")
        spacing = max(1, (2 * radius) // kb) if kb > 1 else 2 * radius   # sparsest lines, in beads

        def lay_out(orient_, fill_map=None):
            for k, L in enumerate(layers):
                skin = (base + 90 * k) % 180
                line = skin if pr.infill_pattern == "alternate" else base
                if kb > 1:            # fine bead grid: centerline nodes, walls on the true outline
                    from rltp.cover.beads import fine_layout, node_frame
                    if L.mframe is None:
                        L.mframe = L.frame
                        L.frame = node_frame(L.mframe, kb)
                    fine_layout(L, parts[k], solids[k], pr.walls, pr.infill, skin, line,
                                spacing, kb, orient_, sweight, fill_map)
                else:
                    fill_sparse(L, parts[k], solids[k], pr.walls, pr.infill, skin, spacing, line,
                                orient_, sweight, fill_map)

        lay_out(orient)
        if pr.start in ("gradient", "gradient4") and field is not None:
            return _gradient_start(layers, spec, z0, field, lay_out, density, 1.0 / spacing,
                                   diagonals=(pr.start == "gradient4" and kb > 1), opts=fields)
    return PartState(layers, spec, z0=z0)


def _snapshot(layers) -> list:
    """PartState.snapshot of bare layers (sparse)."""
    return [(L.nxt.copy(), L.prv.copy(), set(L.seam), L.mask.copy()) for L in layers]


def _gradient_start(layers, spec, z0, field, lay_out, target, rho_min, diagonals, opts=None):
    """Optimise the bead-direction and infill-density fields on the stress start (the
    gradient layout step, rltp.search.fields) from several starting direction fields, lay
    every layer out from the best, and return the part. The best `opts.keep` layouts go to
    part.start_snaps (annealing chains start from them), the step's numbers to
    part.start_info."""
    from rltp.search.fields import FieldOptions, multi_start
    opts = opts or FieldOptions()
    t = time.time()
    part = PartState(layers, spec, z0=z0)
    fi_stress = part.breakdown()["fi_agg"]
    p = layers[0].material_frame.p
    end_cost, w_strength = 0.0, 1.0
    if opts.weights is not None:
        # line ends cost continuity and print time, priced in the score's own units (the
        # stress start's term scales) so the field step trades them against strength
        from rltp.eval.metrics import full_weights
        wf = full_weights(opts.weights)
        sc = part.norm.scale
        w_strength = wf[0] / sc("strength") if wf[0] > 0 else 1.0
        end_cost = opts.end_scale * (wf[2] * opts.end_strokes / sc("cont") + wf[1] * opts.end_seconds / sc("time"))
    results = multi_start(part, field["theta"], target, rho_min, opts, diagonals,
                          min_axis=int(round((2.0 / p) ** 2)), min_diag=int(round((3.0 / p) ** 2)),
                          progress=lambda m: log.info(m), end_cost=end_cost, w_strength=w_strength)

    def place(res):
        """Lay out one result, shifting its density field until the laid-out infill share lands
        on the target (lines are whole beads, so it drifts at edges and in thin areas)."""
        def laid(shift):
            lay_out(res.classes, np.where(res.rho_cells, np.clip(res.rho + shift, rho_min, 1.0), 0.0))
            opt = sum(int((L.part & ~L.req).sum()) for L in layers)
            return sum(int((L.material & L.part & ~L.req).sum()) for L in layers) / max(opt, 1)

        xs, fs = [0.0], [laid(0.0)]
        while abs(fs[-1] - target) > 0.002 and len(xs) < 6:          # secant steps, a layout each
            if len(xs) == 1:
                x = target - fs[0]
            else:
                slope = (fs[-1] - fs[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 1.0
                x = xs[-1] + (target - fs[-1]) / (slope if abs(slope) > 0.05 else 1.0)
            x = float(np.clip(x, -1.0, 1.0))
            xs.append(x)
            fs.append(laid(x))
        i = int(np.argmin([abs(f - target) for f in fs]))
        if i != len(xs) - 1:
            laid(xs[i])
        return xs[i], len(xs)

    totals = {}
    if opts.weights is not None and len(results) > 1:
        # rank by the whole score, each laid-out start scored against the stress start's
        # references (the same normaliser and load-case scales for every one)
        w = np.asarray(opts.weights, float)
        for res in results:
            place(res)
            part.refresh_support()
            part.full_recheck()
            totals[res.start] = part.norm.total(part.raw(), w)
        results = sorted(results, key=lambda r: -totals[r.start])
    keep = results[:max(int(opts.keep), 1)]
    others = []
    for res in keep[1:]:
        place(res)
        others.append(_snapshot(layers))
    shift, n_layouts = place(keep[0])
    out = PartState(layers, spec, z0=z0)
    best = keep[0]
    mask = best.theta_cells
    rounded = [r.fi_rounded for r in results]
    info = {"start": best.start, "field_fi_start": best.fi_start, "field_fi_end": best.fi_end,
            "field_fi_rounded": best.fi_rounded, "stress_start_fi": fi_stress,
            "fi_agg": out.breakdown()["fi_agg"], "iterations": len(best.history) - 1, "solves": best.evals,
            "seconds": time.time() - t, "diagonals": diagonals, "continuation": opts.continuation,
            "density_shift": shift, "layouts": n_layouts, "end_cost": end_cost, "ends": best.ends,
            "rounded_spread": float(max(rounded) / min(rounded) - 1) if rounded else 0.0,
            "ranked_by": "score" if totals else "fi_rounded",
            "starts": [{"start": r.start, "fi_start": r.fi_start, "fi_end": r.fi_end, "fi_rounded": r.fi_rounded,
                        "ends": r.ends, "total": totals.get(r.start), "iterations": len(r.history) - 1,
                        "seconds": r.seconds}
                       for r in results],
            "class_share": {str(c): float((best.classes[mask] == c).mean()) for c in (0, 45, 90, 135)}}
    out.start_info = info
    out.start_fields = {"theta": best.theta, "rho": best.rho, "classes": best.classes}
    out.start_snaps = [out.snapshot()] + others
    out.start_names = [r.start for r in keep]
    log.info("gradient start: FI_agg %.4f (stress start %.4f) from %s; field model %.4f -> %.4f, rounded "
             "%.4f; %d starts within %.1f%% after rounding; %.0fs", info["fi_agg"], fi_stress, best.start,
             best.fi_start, best.fi_end, best.fi_rounded, len(results), 100 * info["rounded_spread"],
             info["seconds"])
    return out
