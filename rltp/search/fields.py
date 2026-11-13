"""Gradient layout step: a bead-direction field and an infill-density field optimised for
in-plane strength before any toolpath exists (design doc: Search against the evaluator).

A continuous model of the strength term on the laminate stack (PartState, stack laminate):

  * walls keep the start layout's stiffness, bead directions and fill;
  * every other in-part cell holds beads at the field angle theta (one field shared by
    every layer); solid layers are full there, sparse layers' optional cells hold
    infill lines at density rho (one field shared by the sparse layers);
  * a cell's stiffness is sum over layers of h x fill x Qbar(theta). Infill lines share
    the laminate's strain with the skins, so their stiffness adds with their share of
    the cell (the rule of mixtures; the evaluator resolves the lines and gets the same);
  * a cell's share of sum FI^p is its filled fraction times FI^p, FI the Tsai-Wu index
    of a bead at theta under the cell's strain (empty cells count 0, as in the score).

FI_agg = (sum FI^p / N)^(1/p) over every in-part cell of every layer, as in the score.
One adjoint solve per load case, reusing the factorisation, gives its gradient over every
cell. theta is parametrised by the doubled-angle vector (cos 2 phi, sin 2 phi), filtered
over about `sigma` mm so that areas stay printable; rho by a density filter, with the
sparse layers' mean held at the infill target and each cell within [1 / max_spacing, 1]
(the sparsest lines the support rule allows, and solid).

Given the run's weights, the objective also prices the infill line ends the fields cause
(line_ends) in the score's units, so strength is traded against continuity and print
time. Local optima: optimize_fields can start wide and smooth (continuation: a wide filter
and a low aggregate exponent, narrowing to the final ones), and multi_start runs it from
several starting direction fields in parallel (rltp.search.parallel).

Fields to a layout: theta rounds to each area's line direction (0/90, or 45/135 too), and
rho lays infill lines that start and stop with the field: each line is laid wherever rho
exceeds its dither threshold (fine_layout / fill_sparse `fill_map`, shell.dither_ranks).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from rltp.eval.fea2d import _G, _bhat, q_matrix

log = logging.getLogger(__name__)

DIRS = np.radians([0.0, 45.0, 90.0, 135.0])     # bead classes, the order of cell_shares


def strain_rot(theta: np.ndarray) -> np.ndarray:
    """(n, 3, 3) T with (eps1, eps2, gamma12) = T @ (eps_x, eps_y, gamma_xy) for a bead
    at theta (engineering shear). The bead's stiffness in the grid frame is T' Q T."""
    C, S = np.cos(2 * theta), np.sin(2 * theta)
    T = np.empty(np.shape(theta) + (3, 3))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 2] = (1 + C) / 2, (1 - C) / 2, S / 2
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2] = (1 - C) / 2, (1 + C) / 2, -S / 2
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2] = -S, S, C
    return T


def strain_rot_d(theta: np.ndarray) -> np.ndarray:
    """d strain_rot / d theta."""
    C, S = np.cos(2 * theta), np.sin(2 * theta)
    T = np.empty(np.shape(theta) + (3, 3))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 2] = -S, S, C
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2] = S, -S, -C
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2] = -2 * C, 2 * C, -2 * S
    return T


def tsai_wu_grad(s: np.ndarray, mat):
    """Tsai-Wu inverse strength ratio of material-frame stresses s (..., 3), as
    fea2d.tsai_wu_fi, and its gradient with respect to s."""
    F1 = 1 / mat.Xt - 1 / mat.Xc
    F2 = 1 / mat.Yt - 1 / mat.Yc
    F11 = 1 / (mat.Xt * mat.Xc)
    F22 = 1 / (mat.Yt * mat.Yc)
    F66 = 1 / mat.S ** 2
    F12 = -0.5 * np.sqrt(F11 * F22)
    Fm = np.array([[F11, F12, 0.0], [F12, F22, 0.0], [0.0, 0.0, F66]])
    f = np.array([F1, F2, 0.0])
    Fs = s @ Fm
    a = np.maximum(np.sum(Fs * s, -1), 0.0)
    b = s @ f
    q = np.sqrt(b * b + 4 * a)
    fi = 0.5 * (b + q)
    qs = np.maximum(q, 1e-30)[..., None]
    g = 0.5 * (f + (b[..., None] * f + 4 * Fs) / qs)
    return fi, g


def wall_cells(L) -> np.ndarray:
    """Cells covered by the layer's wall beads (bool H x W)."""
    if L.lock is None:
        return np.zeros_like(L.part_mask)
    if L.kb > 1:
        from rltp.cover.beads import cover_sum
        return (cover_sum(L.lock, L.kb) > 0.5) & L.part_mask
    return L.lock & L.part_mask


@dataclass
class FieldEval:
    J: float                  # sum of FI^q at the aggregate exponent q asked for (the objective)
    fi_agg: float             # FI_agg at the score's own p
    g_theta: np.ndarray | None = None   # dJ / d theta per union cell
    g_rho: np.ndarray | None = None     # dJ / d rho per union cell
    seconds: float = 0.0


class FieldModel:
    """The laminate strength term as a function of per-cell theta and rho (over the
    laminate problem's cells, `lam.cells`)."""

    def __init__(self, part):
        if not part.laminate:
            raise ValueError("the gradient layout step needs the laminate stack model")
        lam = part._lam_problem()
        if lam is None:
            raise ValueError("the gradient layout step needs in-plane supports and loads")
        self.part, self.lam = part, lam
        self.p = float(part.p)
        self.mat = part.spec.material
        self.Q = q_matrix(self.mat)
        self.r = lam.r
        self.cells = lam.cells
        self.ec = lam.elem_cell
        nc = self.cells.size
        self.N = int(sum(L.part_mask.sum() for L in part.layers))
        C = lam.n_cases
        self.C = C
        self.cs = (np.ones(C) if lam.case_scale is None or C == 1
                   else np.asarray(lam.case_scale, float))
        G = np.zeros((3, 3, 8, 8))
        for xi in (-_G, _G):
            for eta in (-_G, _G):
                B = _bhat(xi, eta)
                G += np.einsum("ia,jb->ijab", B, B)
        self.Gb = G
        self.B0 = lam.B0
        # per cell: layers holding field beads (solid, or sparse optional) and their thickness
        ns, npn, hs, hp = (np.zeros(nc) for _ in range(4))
        aw = np.zeros((nc, 4))                    # walls' stiffness multipliers (0, 45, 90, 135)
        keys = []
        for L in part.layers:
            inpart = L.part_mask.reshape(-1)[self.cells]
            wall = wall_cells(L).reshape(-1)[self.cells]
            req = L.req.reshape(-1)[self.cells] if L.req is not None else inpart
            fld = inpart & ~wall
            solid, opt = fld & req, fld & ~req
            ns += solid
            npn += opt
            hs += L.h * solid
            hp += L.h * opt
            c0, c90, wh, fill, dg = part._lam_layer(L)
            if dg is None:
                sh = np.stack([wh, np.zeros_like(wh), 1 - wh, np.zeros_like(wh)], -1)
                c45 = c135 = np.zeros_like(c0)
            else:
                c45, c135, sh = dg
            aw += np.stack([c0, c45, c90, c135], -1) * wall[:, None]
            pres = sh > 1e-9
            bits = pres @ np.array([1, 2, 4, 8])
            sel = np.flatnonzero(wall & fill)
            keys.append(sel * 16 + bits[sel])
        self.ns, self.npn, self.hs, self.hp = ns, npn, hs, hp
        self.theta_cells = (ns + npn) > 0
        self.rho_cells = npn > 0
        uk, cnt = np.unique(np.concatenate(keys), return_counts=True)
        wcell, wbits = np.divmod(uk, 16)
        # wall groups per element: (element, present directions, count of layers / r^2)
        if self.r == 1:
            self.w_elem, self.w_bits, self.w_cnt = wcell, wbits, cnt.astype(float)
        else:
            order = np.argsort(self.ec, kind="stable")
            starts = np.searchsorted(self.ec[order], np.arange(nc))
            k = self.r * self.r
            el = order[(starts[wcell][:, None] + np.arange(k)[None, :])].reshape(-1)
            self.w_elem = el
            self.w_bits = np.repeat(wbits, k)
            self.w_cnt = np.repeat(cnt.astype(float), k) / k
        self.w_pres = ((self.w_bits[:, None] >> np.arange(4)) & 1).astype(bool)
        Td = strain_rot(DIRS)
        self.Md = self.Q @ Td                                   # (4, 3, 3) bead stress from strain
        Qd = np.swapaxes(Td, -1, -2) @ self.Q @ Td              # (4, 3, 3) grid-frame stiffness
        Dw = np.einsum("cd,dij->cij", aw, Qd)                   # walls' D per cell
        self.Dw = Dw[self.ec] if self.r > 1 else Dw

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, theta: np.ndarray, rho: np.ndarray, grad: bool = True, q: float | None = None) -> FieldEval:
        """theta, rho: per union cell (lam.cells order). rho matters only on cells where
        some sparse layer is optional, theta where any layer holds field beads. q: the
        aggregate's exponent for J and its gradient (default the score's p; continuation
        starts lower); fi_agg is always at the score's p."""
        t0 = time.time()
        lam, ec, C = self.lam, self.ec, self.C
        p = self.p if q is None else float(q)
        r2 = float(self.r * self.r)
        S = self.hs + self.hp * rho
        wgt = (self.ns + self.npn * rho) / r2
        th_e, S_e, w_e = (theta[ec], S[ec], wgt[ec]) if self.r > 1 else (theta, S, wgt)
        T = strain_rot(th_e)
        QT = self.Q @ T                                          # (ne, 3, 3) = M
        Qb = np.swapaxes(T, -1, -2) @ QT
        D = self.Dw + S_e[:, None, None] * Qb
        Ke = np.einsum("eij,ijab->eab", D, self.Gb)
        K = lam._to_csc(Ke)
        lu = lam._factor(K)
        ff = lam.ff.reshape(lam.ff.shape[0], -1)                 # (nf, C)
        uf = lu.solve(ff)
        u = np.zeros((lam.ndof, C))
        u[lam.free] = uf
        ue = u[lam.edof]                                         # (ne, 8, C)
        eps = np.einsum("ja,eac->ecj", self.B0, ue)             # (ne, C, 3)
        cs = self.cs
        # field beads
        sm = np.einsum("eij,ecj->eci", QT, eps)
        fi, gs = tsai_wu_grad(sm, self.mat)                     # (ne, C), (ne, C, 3)
        fis = fi * cs[None, :]
        Jf_e = w_e * np.mean(fis ** p, 1)
        # walls: the worst of the directions each holds
        ew = self.w_elem
        epw = eps[ew]                                            # (nw, C, 3)
        smd = np.einsum("dij,wcj->wdci", self.Md, epw)          # (nw, 4, C, 3)
        fid, gsd = tsai_wu_grad(smd, self.mat)                  # (nw, 4, C), (nw, 4, C, 3)
        fid = np.where(self.w_pres[:, :, None], fid, -1.0)
        arg = np.argmax(fid, 1)                                 # (nw, C)
        fiw = np.take_along_axis(fid, arg[:, None, :], 1)[:, 0, :]
        fiw = np.maximum(fiw, 0.0)
        Jw = float(np.sum(self.w_cnt[:, None] * (fiw * cs) ** p) / C)
        J = float(Jf_e.sum()) + Jw
        if p == self.p:
            Js = J
        else:
            Js = float(np.sum(w_e * np.mean(fis ** self.p, 1))
                       + np.sum(self.w_cnt[:, None] * (fiw * cs) ** self.p) / C)
        fa = (Js / self.N) ** (1 / self.p) if self.N else 0.0
        if not grad:
            return FieldEval(J, fa, seconds=time.time() - t0)
        # dJ/d eps
        cf = (w_e[:, None] * p * cs[None, :] ** p * np.maximum(fi, 0.0) ** (p - 1) / C)   # (ne, C)
        dsig = cf[..., None] * gs                                                          # (ne, C, 3)
        deps = np.einsum("eij,eci->ecj", QT, dsig)                                         # M' dsig
        gsel = np.take_along_axis(gsd, arg[:, None, :, None], 1)[:, 0]                     # (nw, C, 3)
        Msel = self.Md[arg]                                                                # (nw, C, 3, 3)
        cw = self.w_cnt[:, None] * p * cs[None, :] ** p * fiw ** (p - 1) / C
        depw = np.einsum("wcij,wci->wcj", Msel, cw[..., None] * gsel)
        np.add.at(deps, ew, depw)
        due = np.einsum("ja,ecj->eac", self.B0, deps)                                       # (ne, 8, C)
        rhs = np.zeros((lam.ndof, C))
        for c in range(C):
            rhs[:, c] = np.bincount(lam.edof.reshape(-1), weights=due[:, :, c].reshape(-1),
                                    minlength=lam.ndof)
        lf = lu.solve(rhs[lam.free])
        lm = np.zeros((lam.ndof, C))
        lm[lam.free] = lf
        le = lm[lam.edof]
        # E_ij = lambda_e' G_ij u_e, summed over cases
        Gu = np.einsum("ijab,ebc->eijac", self.Gb, ue)
        E = np.einsum("eac,eijac->eij", le, Gu)
        dQb = np.swapaxes(strain_rot_d(th_e), -1, -2) @ QT
        dQb = dQb + np.swapaxes(dQb, -1, -2)
        gS_e = -np.einsum("eij,eij->e", Qb, E)                  # implicit, per unit S
        gth_e = -S_e * np.einsum("eij,eij->e", dQb, E)
        dM = self.Q @ strain_rot_d(th_e)
        gth_e += np.einsum("eci,eij,ecj->e", dsig, dM, eps)      # explicit: the bead frame turns
        grho_e = self.hp[ec] * gS_e if self.r > 1 else self.hp * gS_e
        grho_e = grho_e + (self.npn[ec] if self.r > 1 else self.npn) / r2 * np.mean(fis ** p, 1)
        if self.r > 1:
            nc = self.cells.size
            g_th = np.bincount(ec, gth_e, minlength=nc)
            g_rho = np.bincount(ec, grho_e, minlength=nc)
        else:
            g_th, g_rho = gth_e, grho_e
        return FieldEval(J, fa, g_th * self.theta_cells, g_rho * self.rho_cells, time.time() - t0)


# ---------------------------------------------------------------------- filters

class Smooth:
    """Normalised Gaussian filter over a mask: (G * (x m)) / (G * m) on the mask."""

    def __init__(self, mask: np.ndarray, sigma: float):
        self.m = mask.astype(float)
        self.sigma = float(sigma)
        self.den = self._g(self.m)
        self.den = np.where(mask, np.maximum(self.den, 1e-12), 1.0)

    def _g(self, x):
        if self.sigma <= 0:
            return x.copy()
        return ndimage.gaussian_filter(x, self.sigma, mode="constant", truncate=3.0)

    def __call__(self, x):
        return self._g(x * self.m) / self.den * self.m

    def T(self, y):
        return self._g(y * self.m / self.den) * self.m


# ---------------------------------------------------------------------- optimisation

@dataclass
class FieldResult:
    theta: np.ndarray            # (H, W) bead angle (rad), filtered; 0 outside the field cells
    rho: np.ndarray              # (H, W) infill density of sparse layers' optional cells
    theta_cells: np.ndarray      # (H, W) bool, where theta matters
    rho_cells: np.ndarray        # (H, W) bool, where rho matters
    fi_start: float              # field-model FI_agg at the start fields (final filter)
    fi_end: float
    history: list = field(default_factory=list)
    seconds: float = 0.0
    evals: int = 0
    model: object = None         # the FieldModel (for re-scoring rounded fields); None from a worker
    start: str = ""              # name of the starting direction field
    fi_rounded: float | None = None       # field-model FI_agg with theta rounded to the layout's classes
    classes: np.ndarray | None = None     # (H, W) the layout's bead class per cell
    ends: float | None = None             # predicted infill line ends (line_ends) of the rounded layout


@dataclass
class FieldOptions:
    """Settings of the gradient layout step."""
    starts: int = 8              # starting direction fields: the stress map, turned 90, 0/90/45/135, random
    workers: int = 0             # processes for the starts (0: one per start, up to the CPU count)
    continuation: bool = True    # wide filter and low exponent first (stages_for)
    sigma_mm: float = 0.75       # final filter
    iters: int = 60              # iterations of the final stage (earlier stages get a third each)
    keep: int = 1                # best starts laid out; annealing chains start from them in turn
    weights: tuple | None = None  # score weights: rank the laid-out starts by the whole score
                                  # (starts as strong can differ a lot on continuity), and charge
                                  # line ends in the field step; None: by FI_agg after rounding, no
                                  # line-end cost
    end_strokes: float = 0.057   # per unit of line_ends: strokes and print time (s) in the laid-out
    end_seconds: float = 0.025   # start (fitted over 17 bracket starts: r 0.57 strokes, 0.64 time)
    end_scale: float = 1.0       # x the line-end cost (0: ignore line ends). Bracket starts: +23.8
                                 # without, +47.3 at 0.5x and 1x, +28.2 at 2x


def line_ends(theta: np.ndarray, rho: np.ndarray, mask: np.ndarray, solid: np.ndarray, sparse: np.ndarray,
              kb: int, eps: float = 1e-3):
    """Expected bead-line ends of the layout the fields lay out, and their gradients.

    Lines along x are laid at density Lx = L cos^2 theta and lines along y at Ly = L sin^2
    theta, with L = solid layers + sparse layers x rho the cell's filled layers. A line
    ends wherever its own density changes along it: Lx across x, Ly across y. That counts
    both the infill lines that stop where rho drops (dither thresholds are spread evenly, so
    a change d in rho ends a share d of the lines) and every line that runs into an area of
    the other direction, in skins and infill alike. On the grid: faces between field cells
    (`mask`), |change| smoothed by eps, over kb cells per bead. Returns (ends, dE/dtheta,
    dE/drho), grids H x W.

    (A first version counted only rho's changes, weighted by the line direction; it missed
    direction-area boundaries and didn't track the laid-out strokes, r = 0.18 over 17 bracket
    starts, against 0.57 for this one.)"""
    c2 = np.cos(theta) ** 2
    s2t = np.sin(2 * theta)
    Lf = solid + sparse * rho
    E = 0.0
    g_t = np.zeros_like(theta)
    g_r = np.zeros_like(rho)
    # Dx = Lf c2: dDx/drho = sparse c2, dDx/dtheta = -Lf sin 2th; Dy = Lf (1 - c2): sparse (1 - c2), +Lf sin 2th
    for axis, D, dr, dt in ((1, Lf * c2, sparse * c2, -Lf * s2t), (0, Lf * (1.0 - c2), sparse * (1.0 - c2), Lf * s2t)):
        lo = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
        hi = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
        both = mask[lo] & mask[hi]
        d = np.where(both, D[hi] - D[lo], 0.0)
        a = np.sqrt(d * d + eps * eps)
        E += float(np.where(both, a - eps, 0.0).sum())
        s = np.where(both, d / a, 0.0)
        gD = np.zeros_like(D)
        gD[hi] += s
        gD[lo] -= s
        g_r += gD * dr
        g_t += gD * dt
    k = float(max(kb, 1))
    return E / k, g_t / k, g_r / k


def stages_for(sigma_mm: float, p: float, iters: int, continuation: bool = True) -> list:
    """(filter sigma in mm, aggregate exponent, iterations) per stage. Continuation starts
    with a wide filter, which leaves only coarse direction choices, and a low exponent,
    close to an average and much smoother than the score's p-norm; then both narrow to
    the final ones, each stage starting where the last ended."""
    if not continuation:
        return [(sigma_mm, p, iters)]
    return [(2.5 * sigma_mm, min(2.0, p), max(iters // 3, 5)),
            (1.6 * sigma_mm, min(4.0, p), max(iters // 3, 5)),
            (sigma_mm, p, iters)]


def optimize_fields(part, theta0: np.ndarray, target: float, rho_min: float, sigma_mm: float = 0.75,
                    iters: int = 60, move: float = 0.05, turn: float = np.radians(6.0),
                    tol: float = 1e-4, continuation: bool = False, model=None, name: str = "",
                    end_cost: float = 0.0, w_strength: float = 1.0, progress=None) -> FieldResult:
    """theta0 (H x W, rad): starting directions (e.g. the stress map's). target: the sparse
    layers' mean infill density; rho_min: sparsest allowed lines. Each iteration takes one
    gradient step on both fields (optimality criteria for rho, a scaled step for theta),
    halving the steps whenever the objective would rise; with `continuation`, through the
    stages of stages_for.

    Objective, in the score's units: w_strength x ln FI_agg (at the stage's exponent) plus
    end_cost x the predicted infill line ends (line_ends); end_cost 0 leaves strength alone."""
    t0 = time.time()
    model = model or FieldModel(part)
    L0 = part.layers[0]
    H, W = L0.H, L0.W
    pc = L0.material_frame.p
    cells = model.cells
    tmask = np.zeros(H * W, bool)
    tmask[cells[model.theta_cells]] = True
    rmask = np.zeros(H * W, bool)
    rmask[cells[model.rho_cells]] = True
    tmask, rmask = tmask.reshape(H, W), rmask.reshape(H, W)
    vol_w = np.zeros(H * W)
    vol_w[cells] = model.npn
    vol_w = vol_w.reshape(H, W)
    sol_w = np.zeros(H * W)
    sol_w[cells] = model.ns
    sol_w = sol_w.reshape(H, W)
    rho_min = float(np.clip(rho_min, 1e-3, target))
    tag = f"fields {name}: " if name else "fields: "

    def vol(rt):
        return float((vol_w * rt).sum() / max(vol_w.sum(), 1e-12))

    kb = L0.kb

    def make(sig_mm, q):
        Ft, Fr = Smooth(tmask, sig_mm / pc), Smooth(rmask, sig_mm / pc)
        ws = w_strength / q                     # FI_agg = (J / N)^(1/q)

        def run(phi, rho, grad=True):
            vx, vy = Ft(np.cos(2 * phi)), Ft(np.sin(2 * phi))
            th = 0.5 * np.arctan2(vy, vx)
            rt = Fr(rho)
            ev = model.evaluate(th.reshape(-1)[cells], rt.reshape(-1)[cells], grad=grad, q=q)
            ends = line_ends(th, rt, tmask, sol_w, vol_w, kb) if end_cost > 0 else (0.0, 0.0, 0.0)
            ev.F = ws * np.log(max(ev.J, 1e-300)) + end_cost * ends[0]
            ev.ends = ends
            return ev, th, vx, vy, rt

        def chain(ev, vx, vy, phi):
            gth = np.zeros(H * W)
            gth[cells] = ws * ev.g_theta / ev.J
            grh = np.zeros(H * W)
            grh[cells] = ws * ev.g_rho / ev.J
            gth, grh = gth.reshape(H, W), grh.reshape(H, W)
            if end_cost > 0:
                gth = gth + end_cost * ev.ends[1] * tmask
                grh = grh + end_cost * ev.ends[2] * rmask
            n2 = np.maximum(vx * vx + vy * vy, 1e-12)
            gx, gy = Ft.T(gth * (-0.5 * vy / n2)), Ft.T(gth * (0.5 * vx / n2))
            gphi = (gx * (-2 * np.sin(2 * phi)) + gy * (2 * np.cos(2 * phi))) * tmask
            return gphi, Fr.T(grh)

        return Fr, run, chain

    phi = np.where(tmask, theta0, 0.0)
    rho = np.where(rmask, target, 0.0)
    fi_start = make(sigma_mm, model.p)[1](phi, rho, grad=False)[0].fi_agg
    n_ev = 1
    hist = [{"it": 0, "stage": -1, "fi_agg": fi_start}]
    if progress:
        progress(f"{tag}start FI_agg {fi_start:.4f} (field model)")
    it = 0
    for si, (sig_mm, q, n_it) in enumerate(stages_for(sigma_mm, model.p, iters, continuation)):
        Fr, run, chain = make(sig_mm, q)
        ev, th, vx, vy, rt = run(phi, rho)
        n_ev += 1
        dv = Fr.T(vol_w)
        mv, tv = move, turn
        stall = 0
        for _ in range(n_it):
            it += 1
            gphi, grho = chain(ev, vx, vy, phi)
            ref = np.percentile(np.abs(gphi[tmask]), 95) if tmask.any() else 1.0
            accepted = False
            for _try in range(4):
                dphi = -np.clip(gphi / max(ref, 1e-30), -1.0, 1.0) * tv
                phi_n = np.where(tmask, phi + dphi, 0.0)
                lo = np.maximum(rho - mv, rho_min)
                hi = np.minimum(rho + mv, 1.0)
                B = np.maximum(-grho, 0.0) / np.maximum(dv, 1e-30)
                l1, l2 = 1e-12, 1e12
                rho_n = rho
                for _b in range(80):                       # bisection on the volume multiplier
                    lmid = np.sqrt(l1 * l2)
                    rho_n = np.where(rmask, np.clip(rho * np.sqrt(B / lmid), lo, hi), 0.0)
                    if vol(Fr(rho_n)) > target:
                        l1 = lmid
                    else:
                        l2 = lmid
                    if l2 / l1 < 1 + 1e-9:
                        break
                ev_n, th_n, vx_n, vy_n, rt_n = run(phi_n, rho_n)
                n_ev += 1
                if ev_n.F <= ev.F:
                    accepted = True
                    break
                mv *= 0.5
                tv *= 0.5
            if not accepted:
                break
            gain = (ev.F - ev_n.F) / max(w_strength / q, 1e-12)   # in units of ln J
            phi, rho, ev, th, vx, vy, rt = phi_n, rho_n, ev_n, th_n, vx_n, vy_n, rt_n
            mv, tv = min(mv * 1.25, move), min(tv * 1.25, turn)
            hist.append({"it": it, "stage": si, "fi_agg": ev.fi_agg, "ends": ev.ends[0], "vol": vol(rt),
                         "s": ev.seconds})
            stall = stall + 1 if gain < tol else 0
            if stall >= 5:
                break
        if progress:
            progress(f"{tag}stage {si} (filter {sig_mm:.2f} mm, p {q:g}): FI_agg {ev.fi_agg:.4f}, "
                     f"line ends {ev.ends[0]:.0f}, after {it} iterations, infill {vol(rt):.3f}")
    return FieldResult(np.where(tmask, th, 0.0), np.where(rmask, rt, 0.0), tmask, rmask,
                       fi_start, ev.fi_agg, hist, time.time() - t0, n_ev, model, name)


# ---------------------------------------------------------------------- several starts

START_KINDS = ("stress", "stress90", "0", "90", "45", "135")


def start_names(n: int) -> list:
    """The first n starts: the stress map, the stress map turned 90 degrees, uniform 0, 90,
    45, 135 degrees, then random smooth fields."""
    names = list(START_KINDS[:n])
    i = 0
    while len(names) < n:
        names.append(f"random{i}")
        i += 1
    return names


def start_field(name: str, theta_stress: np.ndarray, p_cell: float) -> np.ndarray:
    """Starting directions (H x W, rad) by name."""
    if name == "stress":
        return theta_stress.copy()
    if name == "stress90":
        return theta_stress + np.pi / 2
    if name.startswith("random"):
        rng = np.random.default_rng(1000 + int(name[6:] or 0))
        a = rng.uniform(0, np.pi, theta_stress.shape)
        s = 3.0 / p_cell                                   # smooth over ~3 mm
        cx = ndimage.gaussian_filter(np.cos(2 * a), s, mode="nearest")
        cy = ndimage.gaussian_filter(np.sin(2 * a), s, mode="nearest")
        return 0.5 * np.arctan2(cy, cx)
    return np.full(theta_stress.shape, np.radians(float(name)))


def _start_job(name):
    from rltp.search.parallel import SHARED
    J = SHARED
    part, opts = J["part"], J["opts"]
    model = FieldModel(part)
    res = optimize_fields(part, start_field(name, J["theta_stress"], part.layers[0].material_frame.p),
                          J["target"], J["rho_min"], sigma_mm=opts.sigma_mm, iters=opts.iters,
                          continuation=opts.continuation, model=model, name=name, end_cost=J["end_cost"],
                          w_strength=J["w_strength"], progress=J["progress"])
    cls = classes(res.theta, res.theta_cells, J["diagonals"], J["min_axis"], J["min_diag"])
    cells = model.cells
    thr = np.radians(cls.astype(float))
    res.fi_rounded = model.evaluate(thr.reshape(-1)[cells], res.rho.reshape(-1)[cells], grad=False).fi_agg
    sp, so = np.zeros(res.rho.size), np.zeros(res.rho.size)
    sp[cells], so[cells] = model.npn, model.ns
    res.ends = line_ends(thr, res.rho, res.theta_cells, so.reshape(res.rho.shape), sp.reshape(res.rho.shape),
                         part.layers[0].kb)[0]
    res.classes = cls
    res.model = None                       # holds the part; not sent back from a worker
    return res


def multi_start(part, theta_stress: np.ndarray, target: float, rho_min: float, opts: FieldOptions,
                diagonals: bool = False, min_axis: int = 0, min_diag: int = 0, progress=None,
                end_cost: float = 0.0, w_strength: float = 1.0) -> list:
    """optimize_fields from opts.starts starting direction fields (in parallel processes
    where the platform can fork), best first by FI_agg after rounding to the layout's classes.
    end_cost, w_strength: the line-end cost and strength weight (optimize_fields)."""
    import os
    from rltp.search.parallel import fork_map
    names = start_names(max(int(opts.starts), 1))
    workers = opts.workers or min(len(names), os.cpu_count() or 1)
    shared = {"part": part, "opts": opts, "theta_stress": theta_stress, "target": target, "rho_min": rho_min,
              "diagonals": diagonals, "min_axis": min_axis, "min_diag": min_diag, "progress": progress,
              "end_cost": end_cost, "w_strength": w_strength}
    res = fork_map(_start_job, names, workers, shared)
    return sorted(res, key=lambda r: r.fi_rounded)


# ---------------------------------------------------------------------- fields to a layout

def classes(theta: np.ndarray, mask: np.ndarray, diagonals: bool, min_axis: int = 0,
            min_diag: int = 0) -> np.ndarray:
    """Bead class per cell (0, 45, 90, 135 degrees) nearest to theta; 0/90 only unless
    `diagonals`. Diagonal patches under `min_diag` cells fall back to the nearer axis, and
    axis patches under `min_axis` cells take the other axis (fewer short strokes)."""
    deg = np.mod(np.degrees(theta), 180.0)
    axis = np.where(np.abs(deg - 90.0) < 45.0, 90, 0)
    if diagonals:
        cls = (np.round(deg / 45.0).astype(int) % 4) * 45
        if min_diag > 0:
            for d in (45, 135):
                lab, n = ndimage.label((cls == d) & mask)
                if n:
                    size = np.bincount(lab.reshape(-1))
                    small = (size < min_diag)[lab] & (lab > 0)
                    cls = np.where(small, axis, cls)
    else:
        cls = axis
    if min_axis > 0:
        for d, other in ((0, 90), (90, 0)):
            lab, n = ndimage.label((cls == d) & mask)
            if n:
                size = np.bincount(lab.reshape(-1))
                small = (size < min_axis)[lab] & (lab > 0)
                cls = np.where(small, other, cls)
    return np.where(mask, cls, 0)
