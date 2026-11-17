"""Direction guidance for the annealer: where turning beads would lower the failure index
most, and which way to turn them.

From the strains the current layout produces, every cell gets the p-mean over load cases
of FI^p for beads at 0, 45, 90 and 135 degrees (the evaluator's own Tsai-Wu check, with
each case's scale). For a layer's bead cells (filled, not walls), `regret` is how much its
FI^p would drop if the cell alone turned to its best direction. A guided proposal picks a
window by smoothed regret and re-lays it (the refill operators) in the direction that
minimises the window's summed FI^p, instead of a random operator at a random angle.

The strains of the whole part change a little with every accepted edit; the map is
refreshed every `every` evaluator calls, which is plenty for placing proposals (the
evaluator still scores every one exactly).

With 16 line bins (true directions, --line-bins 16), DirectionMap.layer16 scores beads every
11.25 degrees the same way, regret against each cell's own bin, and for each lay window size
the bin with the least summed FI^p around every cell; attach16 puts those maps in the layer's
context, where the lay and turn16 edits (rltp.cover.beads) read them, and guided proposals
become lay ops (_lay_action).

WorthMap does the same for material: where one more unit of infill would gain the score
most (the field model's density gradient on the current layout), for the pack operator,
which re-lays a window at its own infill share with the lines packed onto the most valuable
cells (pack_action places it; guided_action takes a share `pack` of its proposals for it).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from rltp.cover.ops import OP_INDEX
from rltp.eval.adapt import ANGLES, _cell_strains, layer_dirs
from rltp.eval.fea2d import to_material, tsai_wu_fi

# window side (cells) -> operator name prefix; diagonals only come in 16 and 32
SIZES = ((16, "refill"), (32, "refill32"), (64, "refill64"))
DEG = (0, 45, 90, 135)


def _dir_fi(fea, u, case_scale, p, case_scale2=None, a8: float = 1.0, a2: float = 0.0):
    """(n_cells, 4) over fea.cells: a8 x the p-mean over cases of FI^p for beads at
    0/45/90/135, plus a2 x the same at p = 2 (the bulk term's), each case scaled as the
    score scales it."""
    eps = _cell_strains(fea, u)
    C = eps.shape[1]
    Qs = (fea.Q0, fea.Q45, fea.Q90, fea.Q135)
    fi = np.stack([tsai_wu_fi(*to_material(*np.moveaxis(eps @ Q.T, -1, 0), a), fea.mat)
                   for Q, a in zip(Qs, ANGLES)], -1)                 # (nc, C, 4)
    if getattr(fea, "keep", None) is not None:                      # clamp band: as the score sees it
        fi = fi * fea.keep[:, :, None]
    cs = case_scale if case_scale is not None and C > 1 else np.ones(C)
    out = a8 * np.mean((fi * cs[None, :, None]) ** p, 1)
    if a2 > 0:
        cs2 = case_scale2 if case_scale2 is not None and C > 1 else np.ones(C)
        out = out + a2 * np.mean((fi * cs2[None, :, None]) ** 2, 1)
    return out


NBIN16 = 16          # line bins (rltp.cover.beads.LINE_BINS): every 11.25 degrees
LAY = ((16, "lay16"), (32, "lay32"), (64, "lay64"))


def _dir_fi16(fea, u, case_scale, p, case_scale2=None, a8: float = 1.0, a2: float = 0.0):
    """_dir_fi for beads every 11.25 degrees: (n_cells, 16), column 4j the class 45j."""
    from rltp.eval.fea2d import q_rotate
    eps = _cell_strains(fea, u)
    C = eps.shape[1]
    cs = case_scale if case_scale is not None and C > 1 else np.ones(C)
    cs2 = case_scale2 if case_scale2 is not None and C > 1 else np.ones(C)
    keep = getattr(fea, "keep", None)
    out = np.zeros((eps.shape[0], NBIN16))
    for j in range(NBIN16):
        a = j * np.pi / NBIN16
        fi = tsai_wu_fi(*to_material(*np.moveaxis(eps @ q_rotate(fea.Q0, a).T, -1, 0), a), fea.mat)
        if keep is not None:
            fi = fi * keep
        out[:, j] = a8 * np.mean((fi * cs[None, :]) ** p, 1)
        if a2 > 0:
            out[:, j] += a2 * np.mean((fi * cs2[None, :]) ** 2, 1)
    return out


class DirectionMap:
    """Per cell and bead direction, the weighted score's cost: strength's FI^p over the
    part's sum (raw strength is -ln(S/N)/p, so a cell's share of S moves it by that over p),
    plus the bulk term's FI^2 likewise, by the run's weights. Without a bulk weight this is
    strength's FI^p alone (up to a constant factor)."""

    def __init__(self, part, every: int = 50, w=None):
        from rltp.eval.metrics import full_weights
        wf = full_weights(w if w is not None else [1.0])
        self.ws = float(wf[0]) if wf[0] > 0 else 1.0
        self.wb = float(wf[7]) if len(wf) > 7 and "bulk" in getattr(part, "terms", ()) else 0.0
        self.part = part
        self.every = every
        self._at = None
        self._lam = None
        self._layers: dict = {}
        self._lam16 = None
        self._layers16: dict = {}

    def _stale(self) -> bool:
        return self._at is None or self.part.n_evals - self._at >= self.every

    def _refresh(self) -> None:
        if self._stale():
            self._layers.clear()
            self._layers16.clear()
            self._lam = self._lam16 = None
            self._at = self.part.n_evals

    def layer16(self, k: int):
        """Fit mode, 16 line bins (laminate): (best, regret, mdir, lay) over layer k's grid:
        best bin 0..15 (-1 off the bead cells), regret against each cell's own bin
        (rltp.cover.beads.bead_bins), FI^p per bin (n, 16), and per window size of the lay
        ops (16, 32, 64) the bin with the least summed FI^p over the bead cells of a window
        centred there (H x W, -1 where it holds none)."""
        part = self.part
        self._refresh()
        if k in self._layers16:
            return self._layers16[k]
        lam = part._lam_problem() if part.laminate else None
        if lam is None:
            return None
        from rltp.cover.beads import bead_bins
        L = part.layers[k]
        H, W = L.H, L.W
        n = H * W
        S8, S2 = max(float(part.Sp.sum()), 1e-300), max(float(part.Sp2.sum()), 1e-300)
        a8, a2 = self.ws / (part.p * S8), self.wb / (2.0 * S2)
        if self._lam16 is None:
            self._lam16 = _dir_fi16(lam, part.lam_u, lam.case_scale, part.p, lam.case_scale2, a8, a2)
        mdir = np.zeros((n, NBIN16))
        mdir[lam.cells] = self._lam16
        _, _, wall = layer_dirs(L)
        b = bead_bins(L).reshape(-1)
        bead = (b >= 0) & ~wall
        cur = mdir[np.arange(n), np.maximum(b, 0)]
        best = np.where(bead, np.argmin(mdir, 1), -1)
        regret = np.where(bead, np.maximum(cur - mdir.min(1), 0.0), 0.0)
        bm = bead.reshape(H, W).astype(float)
        md = (mdir * bead[:, None]).reshape(H, W, NBIN16)
        lay = {}
        for size, _ in LAY:
            cnt = ndimage.uniform_filter(bm, size=size, mode="constant")
            sums = np.stack([ndimage.uniform_filter(md[..., j], size=size, mode="constant")
                             for j in range(NBIN16)])
            lay[size] = np.where(cnt > 1e-9, np.argmin(sums, 0), -1)
        out = (best, regret, mdir, lay)
        self._layers16[k] = out
        return out

    def attach16(self, k: int) -> None:
        """Put layer k's lay maps in its context, which turns on the 16-bin edits (lay,
        turn16; rltp.cover.beads.fine_masks)."""
        L = self.part.layers[k]
        if L.ctx is None:
            return
        got = self.layer16(k)
        if got is not None:
            L.ctx["lay"] = got[3]

    def layer(self, k: int):
        """(best, regret, mdir) over layer k's grid (H*W): best direction index 0..3 (-1 off
        the bead cells), regret >= 0, and the per-direction FI^p (n, 4)."""
        part = self.part
        self._refresh()
        if k in self._layers:
            return self._layers[k]
        L = part.layers[k]
        n = L.H * L.W
        mdir = np.zeros((n, 4))
        S8, S2 = max(float(part.Sp.sum()), 1e-300), max(float(part.Sp2.sum()), 1e-300)
        a8, a2 = self.ws / (part.p * S8), self.wb / (2.0 * S2)
        if part.laminate:
            lam = part._lam_problem()
            if lam is None:
                return None
            if self._lam is None:
                self._lam = _dir_fi(lam, part.lam_u, lam.case_scale, part.p, lam.case_scale2, a8, a2)
            mdir[lam.cells] = self._lam
        else:
            fea = part.fea(k)
            ev = part.evals[k]
            if fea is None or not fea.active or ev is None:
                return None
            mdir[fea.cells] = _dir_fi(fea, ev.fea.u, fea.case_scale, part.p, fea.case_scale2, a8, a2)
        filled, sh, wall = layer_dirs(L)
        bead = filled & ~wall
        cur = np.max(np.where(sh > 1e-9, mdir, 0.0), 1)
        best = np.where(bead, np.argmin(mdir, 1), -1)
        regret = np.where(bead, np.maximum(cur - mdir.min(1), 0.0), 0.0)
        out = (best, regret, mdir)
        self._layers[k] = out
        return out


class WorthMap:
    """Where one more unit of infill is worth most, for the pack operator (rltp.cover.beads).

    The gradient layout step's field model (rltp.search.fields.FieldModel) on the fields the
    current layout holds (layout_fields): its density gradient, weighted as the gradient start
    weights it (strength, and the bulk and stiffness terms when the score has them, in the
    score's own units), per unit of material. Worth = max(-d score / d material, 0) per cell
    of the material grid, 0 off the optional cells; the same map serves every sparse layer
    (the laminate's layers share their strain). Refreshed every `every` evaluator calls
    (about one factorisation and two adjoint solves each)."""

    def __init__(self, part, w=None, every: int = 200):
        self.part = part
        self.w = w
        self.every = every
        self.model = None
        self._at = None
        self._grid = None

    def grid(self) -> np.ndarray | None:
        part = self.part
        if self._grid is not None and part.n_evals - self._at < self.every:
            return self._grid
        if not getattr(part, "fine", False) or not part.laminate or part._lam_problem() is None:
            return None
        from rltp.eval.metrics import full_weights
        from rltp.search.fields import FieldModel, layout_fields
        wf = full_weights(self.w if self.w is not None else [1.0])
        sc = part.norm.scale
        ws = (wf[0] / sc("strength") if wf[0] > 0 else 1.0)
        wst = wf[6] / sc("stiff") if len(wf) > 6 and wf[6] > 0 and "stiff" in part.terms else 0.0
        wb = wf[7] / sc("bulk") if len(wf) > 7 and wf[7] > 0 and "bulk" in part.terms else 0.0
        if self.model is None:
            self.model = FieldModel(part, bulk=wb > 0)
        m = self.model
        th, rho = layout_fields(part, m)
        ev = m.evaluate(th, rho, grad=True)
        g = ws / m.p * ev.g_rho / ev.J
        if wb > 0 and ev.g_rho_b is not None:
            g = g + 0.5 * wb * ev.g_rho_b / ev.J2
        if wst > 0 and ev.g_rho_s is not None:
            g = g + wst * ev.g_rho_s / ev.stiff
        L0 = part.layers[0]
        out = np.zeros(L0.H * L0.W)
        out[m.cells] = np.where(m.rho_cells, np.maximum(-g, 0.0) / np.maximum(m.npn, 1), 0.0)
        self._grid, self._at = out.reshape(L0.H, L0.W), part.n_evals
        return self._grid

    def attach(self, L) -> None:
        """Put the current map in a sparse layer's context, where pack (and its action mask)
        finds it."""
        if L.ctx is not None and L.part is not None:
            g = self.grid()
            if g is not None:
                L.ctx["worth"] = g


PACK_SIZES = ((32, "pack32", (0, 1, 2, 3)), (64, "pack64", (0, 2)))


def pack_action(part, k: int, M: np.ndarray, rng, mdir: np.ndarray, best: np.ndarray, tries: int = 4):
    """A pack action for layer k, or None: a window drawn by the worth of its optional cells
    (where material matters most), re-laid in the window's best direction (as the guided
    refills choose it) with its lines packed onto the most valuable cells."""
    L = part.layers[k]
    worth = L.ctx.get("worth") if L.ctx is not None else None
    if worth is None or L.part is None:
        return None
    H, W, n = L.H, L.W, L.H * L.W
    val = worth * (L.part & ~L.req if L.req is not None else L.part)
    for _ in range(tries):
        size, head, dirs = PACK_SIZES[int(rng.choice(2, p=(0.6, 0.4)))]
        sm = np.sqrt(np.maximum(ndimage.uniform_filter(val, size=size, mode="constant"), 0.0)).reshape(-1)
        tot = sm.sum()
        if tot <= 0:
            return None
        cell = int(rng.choice(n, p=sm / tot))
        r, c = divmod(cell, W)
        h = size // 2
        win = np.zeros((H, W), bool)
        win[max(r - h, 0):min(r + h, H), max(c - h, 0):min(c + h, W)] = True
        wb = win.reshape(-1) & (best >= 0)
        tot_d = mdir[wb].sum(0) if wb.any() else np.zeros(4)
        d = min(dirs, key=lambda j: tot_d[j])
        a = OP_INDEX[f"{head}_{DEG[d]}"] * n + cell
        if M[a]:
            return a
    return None


def _lay_action(gmap: DirectionMap, k: int, M: np.ndarray, rng, n: int, W: int, tries: int):
    """16 line bins: a lay op (window 16, 32 or 64) at a cell drawn by its window's smoothed
    regret against the best of 16 directions; the op lays the window's best bin there."""
    got = gmap.layer16(k)
    if got is None:
        return None
    _, regret, _, _ = got
    if regret.max() <= 0:
        return None
    H = n // W
    for _ in range(tries):
        size, name = LAY[int(rng.choice(3, p=(0.5, 0.3, 0.2)))]
        sm = ndimage.uniform_filter(regret.reshape(H, W), size=size, mode="constant").reshape(-1)
        sm = np.sqrt(np.maximum(sm, 0.0))
        tot = sm.sum()
        if tot <= 0:
            return None
        a = OP_INDEX[name] * n + int(rng.choice(n, p=sm / tot))
        if M[a]:
            return a
    return None


def guided_action(part, k: int, M: np.ndarray, rng, gmap: DirectionMap, tries: int = 4,
                  align: float = 0.0, pack: float = 0.0):
    """A guided action for layer k, or None (no regret, or no valid window found). With
    probability `pack`, a pack placed by the worth map (pack_action); with 16 line bins (the
    layer holds lay maps, DirectionMap.attach16), a lay op (_lay_action); with probability
    `align`, align_region at a cell drawn by regret (the connected area sharing its best
    direction, re-laid that way); otherwise a refill placed by smoothed regret and turned to
    the window's best direction."""
    got = gmap.layer(k)
    if got is None:
        return None
    best, regret, mdir = got
    L = part.layers[k]
    H, W, n = L.H, L.W, L.H * L.W
    if pack > 0 and rng.random() < pack:
        a = pack_action(part, k, M, rng, mdir, best)
        if a is not None:
            return a
    if L.ctx is not None and L.ctx.get("lay") is not None:     # 16 line bins: lay ops
        return _lay_action(gmap, k, M, rng, n, W, tries)
    if regret.max() <= 0:
        return None
    if align > 0 and rng.random() < align and "align_region" in OP_INDEX and L.ctx is not None:
        L.ctx["best"] = best.reshape(H, W)
        sm = np.sqrt(ndimage.uniform_filter(regret.reshape(H, W), size=16, mode="constant").reshape(-1).clip(0))
        cand = sm * (best >= 0)
        if cand.sum() > 0:
            return OP_INDEX["align_region"] * n + int(rng.choice(n, p=cand / cand.sum()))
    for _ in range(tries):
        size, head = SIZES[int(rng.choice(3, p=(0.5, 0.3, 0.2)))]
        sm = ndimage.uniform_filter(regret.reshape(H, W), size=size, mode="constant").reshape(-1)
        sm = np.sqrt(np.maximum(sm, 0.0))           # spread proposals beyond the very worst windows
        tot = sm.sum()
        if tot <= 0:
            return None
        cell = int(rng.choice(n, p=sm / tot))
        r, c = divmod(cell, W)
        h = size // 2
        win = np.zeros((H, W), bool)
        win[max(r - h, 0):min(r + h, H), max(c - h, 0):min(c + h, W)] = True
        wb = win.reshape(-1) & (best >= 0)
        if not wb.any():
            continue
        dirs = (0, 1, 2, 3) if size <= 32 else (0, 2)
        tot_d = mdir[wb].sum(0)
        d = min(dirs, key=lambda j: tot_d[j])
        name = f"{head}_{DEG[d]}"
        if name not in OP_INDEX:
            continue
        a = OP_INDEX[name] * n + cell
        if M[a]:
            return a
    return None
