"""Plane-stress FEA on the layer grid.

Elements are the in-part cells: bilinear 4-node quads with nodes at cell corners
(as in the 88-line topology optimisation code). Each element is an orthotropic
lamina with its fibre (bead) along local x or y; a corner cell averages the two
stiffnesses and is checked in both material frames (iso-strain). Failure uses
the Tsai-Wu strength ratio: FI = 1 / R where the stress state times R just
reaches the Tsai-Wu surface, so FI >= 0 and scales linearly with load.

After a small edit, the solve warm-starts conjugate gradient from the previous
displacements, preconditioned by the last sparse factorisation (which is close
to the new matrix because only a few elements changed).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from rltp.config import LoadSpec, Material
from rltp.model import Frame

log = logging.getLogger(__name__)

_G = 1 / np.sqrt(3)
_NODES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)


VOID = 1e-3   # stiffness fraction of an empty cell in a sparse layer (keeps the mesh fixed)


def q_matrix(mat: Material) -> np.ndarray:
    nu21 = mat.nu12 * mat.E2 / mat.E1
    den = 1 - mat.nu12 * nu21
    return np.array([
        [mat.E1 / den, mat.nu12 * mat.E2 / den, 0],
        [mat.nu12 * mat.E2 / den, mat.E2 / den, 0],
        [0, 0, mat.G12],
    ])


def q_rot90(Q: np.ndarray) -> np.ndarray:
    return np.array([[Q[1, 1], Q[0, 1], 0], [Q[0, 1], Q[0, 0], 0], [0, 0, Q[2, 2]]])


def q_rotate(Q: np.ndarray, theta: float) -> np.ndarray:
    """Plane-stress stiffness of a lamina whose fibre (bead) runs at theta, in the grid frame
    (the classical Q-bar transformation)."""
    c, s = np.cos(theta), np.sin(theta)
    c2, s2, cs = c * c, s * s, c * s
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    q11 = Q11 * c2 * c2 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * s2 * s2
    q22 = Q11 * s2 * s2 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * c2 * c2
    q12 = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (s2 * s2 + c2 * c2)
    q66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (s2 * s2 + c2 * c2)
    q16 = (Q11 - Q12 - 2 * Q66) * c2 * cs - (Q22 - Q12 - 2 * Q66) * s2 * cs
    q26 = (Q11 - Q12 - 2 * Q66) * s2 * cs - (Q22 - Q12 - 2 * Q66) * c2 * cs
    return np.array([[q11, q12, q16], [q12, q22, q26], [q16, q26, q66]])


def to_material(sx, sy, txy, theta: float):
    """Grid-frame stresses -> (along the bead, across it, shear) for a bead at theta."""
    c, s = np.cos(theta), np.sin(theta)
    s1 = c * c * sx + s * s * sy + 2 * c * s * txy
    s2 = s * s * sx + c * c * sy - 2 * c * s * txy
    t12 = -c * s * sx + c * s * sy + (c * c - s * s) * txy
    return s1, s2, t12


def isotropic(E: float, nu: float, strength: float = 50.0) -> Material:
    return Material(E1=E, E2=E, G12=E / (2 * (1 + nu)), nu12=nu,
                    Xt=strength, Xc=strength, Yt=strength, Yc=strength, S=strength / np.sqrt(3))


def _bhat(xi: float, eta: float) -> np.ndarray:
    """Strain-displacement matrix in natural coordinates for a square element.

    The true B is (2/p) * bhat; with dA = (p/2)^2 dxi deta the element stiffness
    is independent of p."""
    dxi = 0.25 * _NODES[:, 0] * (1 + _NODES[:, 1] * eta)
    deta = 0.25 * _NODES[:, 1] * (1 + _NODES[:, 0] * xi)
    B = np.zeros((3, 8))
    B[0, 0::2] = dxi
    B[1, 1::2] = deta
    B[2, 0::2] = deta
    B[2, 1::2] = dxi
    return B


def ke_q4(D: np.ndarray, t: float = 1.0) -> np.ndarray:
    K = np.zeros((8, 8))
    for xi in (-_G, _G):
        for eta in (-_G, _G):
            B = _bhat(xi, eta)
            K += B.T @ D @ B
    return K * t


def tsai_wu_fi(s1, s2, t12, mat: Material) -> np.ndarray:
    """Tsai-Wu failure index as the inverse strength ratio (>= 0, linear in load)."""
    F1 = 1 / mat.Xt - 1 / mat.Xc
    F2 = 1 / mat.Yt - 1 / mat.Yc
    F11 = 1 / (mat.Xt * mat.Xc)
    F22 = 1 / (mat.Yt * mat.Yc)
    F66 = 1 / mat.S ** 2
    F12 = -0.5 * np.sqrt(F11 * F22)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    return 0.5 * (b + np.sqrt(b * b + 4 * np.maximum(a, 0)))


@dataclass
class FEAResult:
    u: np.ndarray            # free+fixed dof vector over used nodes (2 per node)
    fi: np.ndarray           # (n_cells,) failure index per cell, 0 outside / inactive
    S_p: float               # sum FI^p over elements
    N: int                   # element count (0 when the layer carries no load)
    sx: np.ndarray           # (n_cells,) mean stresses in the grid frame
    sy: np.ndarray
    txy: np.ndarray
    iters: int = 0
    direct: bool = False
    S_pc: np.ndarray | None = None   # (n_cases,) raw sum FI^p per load case; None for one case

    def fi_agg(self, p: float) -> float:
        return float((self.S_p / self.N) ** (1 / p)) if self.N else 0.0


class FEAProblem:
    """Fixed per layer during optimisation: mesh, supports and loads. Only the
    bead-direction weights change between solves.

    Several load cases share the stiffness matrix and are solved as separate
    right-hand sides. With more than one case, each cell's FI is the p-mean over
    cases of FI_c * case_scale[c], where case_scale (set by PartState) is 1 over the
    case's baseline FI_agg. Stresses (sx, sy, txy) are those of case 0, the primary.

    solver "fast" (default) factorises K as symmetric (minimum-degree ordering of A'+A,
    no pivoting, which K's positive definiteness allows) and runs the warm-started
    conjugate gradient to a relative residual of 1e-7; on the fine-grid bracket that is
    2.7x faster per solve than "legacy" (SuperLU's general LU, 1e-9) with the same
    answer to 2e-10. "legacy" reproduces earlier runs bit for bit.

    refine r splits each cell into r x r elements (toolpath cells stay one bead wide);
    every element takes its cell's bead weights and fill. Inputs and outputs stay per
    cell: a cell's FI is the p-mean of its elements' (so FI_agg keeps its meaning), and
    its stresses are their mean. Loads and supports act on the refined nodes. r = 1 is
    the original one-element-per-cell mesh, bit for bit."""

    def __init__(self, mask: np.ndarray, frame: Frame, h: float, mat: Material,
                 loads: LoadSpec | None, scale: float = 1.0, z: float | None = None,
                 p_norm: float = 8.0, case_scale: np.ndarray | None = None, refine: int = 1,
                 solver: str = "fast"):
        H, W = mask.shape
        self.H, self.W, self.n = H, W, H * W
        if solver not in ("fast", "legacy"):
            raise ValueError(f"unknown solver {solver!r}; expected fast or legacy")
        self.solver = solver
        self.rtol = 1e-7 if solver == "fast" else 1e-9
        self.mat = mat
        self.p_norm = p_norm
        self.cells = np.flatnonzero(mask.reshape(-1))
        self.r = rf = max(int(refine), 1)
        Hr, Wr = H * rf, W * rf                       # the element grid
        self.Hr, self.Wr = Hr, Wr
        if rf == 1:
            elems = self.cells
            self.elem_cell = np.arange(self.cells.size)
        else:
            elems = np.flatnonzero(np.kron(mask, np.ones((rf, rf), bool)).reshape(-1))
            er_, ec_ = np.divmod(elems, Wr)
            self.elem_cell = np.searchsorted(self.cells, (er_ // rf) * W + ec_ // rf)
            frame = Frame(frame.ox, frame.oy, frame.theta, frame.p / rf)
        ne = elems.size
        r, c = np.divmod(elems, Wr)
        n0 = r * (Wr + 1) + c
        enodes = np.stack([n0, n0 + 1, n0 + Wr + 2, n0 + Wr + 1], 1)
        used = np.unique(enodes)
        remap = np.full((Hr + 1) * (Wr + 1), -1, np.int64)
        remap[used] = np.arange(used.size)
        en = remap[enodes]
        self.edof = np.stack([2 * en, 2 * en + 1], 2).reshape(ne, 8)
        ndof = 2 * used.size
        self.ndof = ndof

        Q = q_matrix(mat)
        self.Q0, self.Q90 = Q, q_rot90(Q)
        self.K0 = ke_q4(self.Q0, h)
        self.K90 = ke_q4(self.Q90, h)
        # diagonal beads (fine bead grid): fibres at 45 and 135 degrees
        self.Q45, self.Q135 = q_rotate(Q, np.pi / 4), q_rotate(Q, 3 * np.pi / 4)
        self.K45 = ke_q4(self.Q45, h)
        self.K135 = ke_q4(self.Q135, h)
        self.B0 = _bhat(0.0, 0.0) * (2 / frame.p)

        nx, ny = frame.node_xy(Hr, Wr)
        nx, ny = nx[used], ny[used]
        fixed = np.zeros(ndof, bool)
        cases = loads.all_cases if loads is not None else [[]]
        self.n_cases = len(cases)
        fs = [np.zeros(ndof) for _ in cases]
        self.support_nodes = np.zeros(used.size, bool)
        self.load_nodes = np.zeros(used.size, bool)     # primary case only (observations)
        self.load_dir = np.zeros(2)
        if loads is not None:
            for reg in loads.supports:
                inside = reg.contains(nx, ny, z)
                self.support_nodes |= inside
            fixed[0::2] = self.support_nodes
            fixed[1::2] = self.support_nodes
            Rt = frame.R.T  # world -> local
            for c, (case, f) in enumerate(zip(cases, fs)):
                for ld in case:
                    inside = ld.region.contains(nx, ny, z) & ~self.support_nodes
                    cnt = int(inside.sum())
                    if cnt == 0:
                        continue
                    F = Rt @ np.asarray(ld.force[:2], float) * scale
                    f[0::2][inside] += F[0] / cnt
                    f[1::2][inside] += F[1] / cnt
                    if c == 0:
                        self.load_nodes |= inside
                        self.load_dir += F
        self.used_nodes = used
        self.active = bool(fixed.any() and any(np.abs(f).sum() > 0 for f in fs) and ne > 0)
        self.fixed = fixed
        self.free = np.flatnonzero(~fixed)
        # one case keeps 1D vectors, exactly as before multi-case support
        f = fs[0] if self.n_cases == 1 else np.stack(fs, 1)
        self.f = f
        self.ff = f[self.free]
        self.case_scale = None if case_scale is None else np.asarray(case_scale, float)
        self._lu = None
        self._u_last = None

        if self.active:
            # sparsity pattern of K_ff and the scatter map from element entries
            iK = np.repeat(self.edof, 8, axis=1).reshape(-1)
            jK = np.tile(self.edof, (1, 8)).reshape(-1)
            fmap = np.full(ndof, -1, np.int64)
            fmap[self.free] = np.arange(self.free.size)
            fi_, fj_ = fmap[iK], fmap[jK]
            keep = (fi_ >= 0) & (fj_ >= 0)
            self._keep = keep
            nf = self.free.size
            key = fj_[keep] * nf + fi_[keep]         # column-major for CSC
            ukey, inv = np.unique(key, return_inverse=True)
            self._scatter = inv
            self._nnz = ukey.size
            cols, rows = np.divmod(ukey, nf)
            self._indices = rows.astype(np.int32)
            self._indptr = np.searchsorted(cols, np.arange(nf + 1)).astype(np.int32)
            self._diag_pos = np.flatnonzero(rows == cols)
            self._nf = nf

    # -------------------------------------------------------------- assembly & solve

    def assemble(self, wh_cells: np.ndarray, dens: np.ndarray | None = None) -> sp.csc_matrix:
        if self.r > 1:
            wh_cells = wh_cells[self.elem_cell]
            dens = None if dens is None else dens[self.elem_cell]
        wh = wh_cells[:, None, None]
        Ke = wh * self.K0 + (1 - wh) * self.K90
        if dens is not None:
            Ke = Ke * dens[:, None, None]
        return self._to_csc(Ke)

    def assemble_coeffs(self, a0: np.ndarray, a90: np.ndarray, a45: np.ndarray | None = None,
                        a135: np.ndarray | None = None) -> sp.csc_matrix:
        """K from per-element multipliers of the bead-along-x and bead-along-y stiffness
        (a laminate sums thickness x fill x bead fraction over its layers), plus the 45 and
        135 degree ones when diagonal beads are present."""
        if self.r > 1:
            a0, a90 = a0[self.elem_cell], a90[self.elem_cell]
            if a45 is not None:
                a45, a135 = a45[self.elem_cell], a135[self.elem_cell]
        Ke = a0[:, None, None] * self.K0 + a90[:, None, None] * self.K90
        if a45 is not None:
            Ke = Ke + a45[:, None, None] * self.K45 + a135[:, None, None] * self.K135
        return self._to_csc(Ke)

    def _to_csc(self, Ke: np.ndarray) -> sp.csc_matrix:
        data = np.bincount(self._scatter, weights=Ke.reshape(-1)[self._keep], minlength=self._nnz)
        # keeps floating islands solvable; per node, so scaled by 1/r^2 to keep the total
        # (and its slight stiffening) the same however finely the cells are split
        reg = 1e-9 * data[self._diag_pos].mean() / (self.r * self.r)
        data[self._diag_pos] += reg
        return sp.csc_matrix((data, self._indices, self._indptr), shape=(self._nf, self._nf))

    def _factor(self, K):
        if self.solver == "fast":
            return splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                        options={"SymmetricMode": True})
        return splu(K)

    def _pcg(self, K, b, x0, M, rtol=None, maxiter=40):
        rtol = self.rtol if rtol is None else rtol
        x = x0.copy()
        r = b - K @ x
        bn = np.linalg.norm(b)
        if bn == 0:
            return np.zeros_like(b), 0, True
        z = M(r)
        d = z.copy()
        rz = r @ z
        for it in range(1, maxiter + 1):
            Kd = K @ d
            alpha = rz / (d @ Kd)
            x += alpha * d
            r -= alpha * Kd
            if np.linalg.norm(r) <= rtol * bn:
                return x, it, True
            z = M(r)
            rz_new = r @ z
            d = z + (rz_new / rz) * d
            rz = rz_new
        return x, maxiter, False

    def solve(self, wh: np.ndarray, warm: np.ndarray | None = None, direct: bool = False,
              keep: bool = True, filled: np.ndarray | None = None,
              dens: np.ndarray | None = None, shares: np.ndarray | None = None) -> FEAResult:
        """wh: (n,) fraction of each cell's bead along local x. filled: (n,) bool for a
        sparse layer; empty cells keep a VOID fraction of the stiffness (so the mesh never
        changes) and count as FI 0. N stays the in-part cell count."""
        n = self.n
        zeros = np.zeros(n)
        multi = self.n_cases > 1
        ushape = (self.ndof, self.n_cases) if multi else (self.ndof,)
        if not self.active:
            return FEAResult(np.zeros(ushape), zeros, 0.0, 0, zeros, zeros, zeros,
                             S_pc=np.zeros(self.n_cases) if multi else None)
        whc = wh[self.cells]
        filc = None if filled is None else np.asarray(filled, bool)[self.cells]
        if shares is not None:                   # four bead directions (0, 45, 90, 135 degrees)
            sh = np.asarray(shares, float)[self.cells]
            dc = np.asarray(dens, float)[self.cells] if dens is not None else np.ones(len(self.cells))
            K = self.assemble_coeffs(dc * sh[:, 0], dc * sh[:, 2], dc * sh[:, 1], dc * sh[:, 3])
            u, iters, used_direct = self.solve_K(K, warm, direct, keep)
            f = filc if filc is not None else np.ones(len(self.cells), bool)
            return self.post_layers(u, [None], [f], [len(self.cells)], iters, used_direct, shares=[sh])[0]
        if dens is not None:                     # fine bead grid: stiffness follows the material
            K = self.assemble(whc, np.asarray(dens, float)[self.cells])
        else:
            K = self.assemble(whc, None if filc is None else np.where(filc, 1.0, VOID))
        u, iters, used_direct = self.solve_K(K, warm, direct, keep)
        return self._post(u, whc, iters, used_direct, filc)

    def solve_K(self, K, warm: np.ndarray | None = None, direct: bool = False, keep: bool = True):
        """Displacements for an assembled K: a direct solve, or conjugate gradient warm-started
        from `warm` and preconditioned by the last factorisation. Returns (u, iters, direct)."""
        multi = self.n_cases > 1
        ushape = (self.ndof, self.n_cases) if multi else (self.ndof,)
        iters, used_direct = 0, False
        if direct or self._lu is None:
            lu = self._factor(K)
            uf = lu.solve(self.ff)
            used_direct = True
            if keep:
                self._lu = lu
        else:
            x0 = warm[self.free] if warm is not None else self._lu.solve(self.ff)
            if not multi:
                uf, iters, ok = self._pcg(K, self.ff, x0, self._lu.solve)
            else:
                uf, ok = np.empty_like(self.ff), True
                for c in range(self.n_cases):
                    uf[:, c], it, ok = self._pcg(K, self.ff[:, c], x0[:, c], self._lu.solve)
                    iters += it
                    if not ok:
                        break
            if not ok:
                lu = self._factor(K)
                uf = lu.solve(self.ff)
                used_direct = True
                if keep:
                    self._lu = lu
        u = np.zeros(ushape)
        u[self.free] = uf
        return u, iters, used_direct

    def drop_factorization(self) -> None:
        self._lu = None

    def _to_cells(self, per_elem: np.ndarray) -> np.ndarray:
        """Mean over each cell's elements (rows = elements; 1D or 2D)."""
        nc, k = self.cells.size, self.r * self.r
        if per_elem.ndim == 1:
            return np.bincount(self.elem_cell, per_elem, minlength=nc) / k
        return np.stack([np.bincount(self.elem_cell, per_elem[:, j], minlength=nc) / k
                         for j in range(per_elem.shape[1])], 1)

    def _post(self, u, whc, iters, direct, filc=None) -> FEAResult:
        if self.r > 1:
            return self.post_layers(u, [whc], [filc if filc is not None else np.ones(whc.size, bool)],
                                    [self.cells.size], iters, direct)[0]
        if self.n_cases > 1:
            return self._post_cases(u, whc, iters, direct, filc)
        ue = u[self.edof]                       # (ne, 8)
        eps = ue @ self.B0.T                    # (ne, 3)
        s0 = eps @ self.Q0.T                    # stresses if bead along x
        s90 = eps @ self.Q90.T                  # stresses if bead along y
        fi0 = tsai_wu_fi(s0[:, 0], s0[:, 1], s0[:, 2], self.mat)
        fi90 = tsai_wu_fi(s90[:, 1], s90[:, 0], s90[:, 2], self.mat)
        fie = np.where(whc >= 1, fi0, np.where(whc <= 0, fi90, np.maximum(fi0, fi90)))
        sm = whc[:, None] * s0 + (1 - whc[:, None]) * s90
        if filc is not None:
            fie = np.where(filc, fie, 0.0)
            sm = sm * filc[:, None]
        n = self.n
        fi = np.zeros(n)
        fi[self.cells] = fie
        sx, sy, txy = np.zeros(n), np.zeros(n), np.zeros(n)
        sx[self.cells], sy[self.cells], txy[self.cells] = sm[:, 0], sm[:, 1], sm[:, 2]
        S_p = float(np.sum(fie ** self.p_norm))
        return FEAResult(u, fi, S_p, int(self.cells.size), sx, sy, txy, iters, direct)

    def _post_cases(self, u, whc, iters, direct, filc=None) -> FEAResult:
        ue = u[self.edof]                                   # (ne, 8, C)
        eps = np.einsum("jk,ekc->ecj", self.B0, ue)         # (ne, C, 3)
        s0 = eps @ self.Q0.T
        s90 = eps @ self.Q90.T
        fi0 = tsai_wu_fi(s0[..., 0], s0[..., 1], s0[..., 2], self.mat)      # (ne, C)
        fi90 = tsai_wu_fi(s90[..., 1], s90[..., 0], s90[..., 2], self.mat)
        w = whc[:, None]
        fic = np.where(w >= 1, fi0, np.where(w <= 0, fi90, np.maximum(fi0, fi90)))
        if filc is not None:
            fic = fic * filc[:, None]
        p = self.p_norm
        S_pc = np.sum(fic ** p, 0)
        cs = self.case_scale if self.case_scale is not None else np.ones(self.n_cases)
        m = np.mean((fic * cs) ** p, 1)                     # per-cell p-mean over cases, to the p
        sm = w * s0[:, 0] + (1 - w) * s90[:, 0]             # primary case's stresses
        if filc is not None:
            sm = sm * filc[:, None]
        n = self.n
        fi = np.zeros(n)
        fi[self.cells] = m ** (1 / p)
        sx, sy, txy = np.zeros(n), np.zeros(n), np.zeros(n)
        sx[self.cells], sy[self.cells], txy[self.cells] = sm[:, 0], sm[:, 1], sm[:, 2]
        return FEAResult(u, fi, float(m.sum()), int(self.cells.size), sx, sy, txy, iters, direct,
                         S_pc=S_pc)

    def post_layers(self, u, whcs, filcs, Ns, iters: int = 0, direct: bool = False, shares=None) -> list:
        """Laminate: one displacement field shared by every layer (equal in-plane strain);
        one FEAResult per layer from its own bead weights and fill (over this mesh's cells).
        N of each result is that layer's in-part cell count. `shares` (per layer, or None):
        cells' (0, 45, 90, 135 degree) bead shares, replacing the 0/90 weights `whcs`; a
        cell's FI is then the worst over the directions it holds."""
        multi = self.n_cases > 1
        refined = self.r > 1
        ue = u[self.edof]
        eps = np.einsum("jk,ekc->ecj", self.B0, ue) if multi else ue @ self.B0.T
        s0 = eps @ self.Q0.T
        s90 = eps @ self.Q90.T
        fi0 = tsai_wu_fi(s0[..., 0], s0[..., 1], s0[..., 2], self.mat)
        fi90 = tsai_wu_fi(s90[..., 1], s90[..., 0], s90[..., 2], self.mat)
        diag = shares is not None and any(sh is not None for sh in shares)
        if diag:
            s45 = eps @ self.Q45.T
            s135 = eps @ self.Q135.T
            fi45 = tsai_wu_fi(*to_material(s45[..., 0], s45[..., 1], s45[..., 2], np.pi / 4), self.mat)
            fi135 = tsai_wu_fi(*to_material(s135[..., 0], s135[..., 1], s135[..., 2], 3 * np.pi / 4), self.mat)
        p = self.p_norm
        n = self.n
        cs = self.case_scale if self.case_scale is not None else np.ones(self.n_cases)
        out = []
        for j, (whc, filc, N) in enumerate(zip(whcs, filcs, Ns)):
            sh = shares[j] if diag else None
            if sh is not None:
                if refined:
                    sh, filc = sh[self.elem_cell], filc[self.elem_cell]
                f = filc[:, None] if multi else filc
                pres = sh > 1e-9
                parts_ = [fi0, fi45, fi90, fi135]
                fic = np.zeros_like(fi0)
                for d in range(4):
                    pd = pres[:, d][:, None] if multi else pres[:, d]
                    fic = np.maximum(fic, np.where(pd, parts_[d], 0.0))
                fic = fic * f
                ax = sh[:, 0] + 0.5 * (sh[:, 1] + sh[:, 3])        # for the stress outputs only
                whc = ax
                w = whc[:, None] if multi else whc
            else:
                if refined:                        # per-cell inputs -> elements
                    whc, filc = whc[self.elem_cell], filc[self.elem_cell]
                w = whc[:, None] if multi else whc
                f = filc[:, None] if multi else filc
                fic = np.where(w >= 1, fi0, np.where(w <= 0, fi90, np.maximum(fi0, fi90))) * f
            if multi:
                S_pc = np.sum(fic ** p, 0) / (self.r * self.r)
                m = np.mean((fic * cs) ** p, 1)
                sm = (w * s0[:, 0] + (1 - w) * s90[:, 0]) * filc[:, None]
                if refined:
                    m, sm = self._to_cells(m), self._to_cells(sm)
                fie, S_p = m ** (1 / p), float(m.sum())
            else:
                S_pc = None
                sm = (whc[:, None] * s0 + (1 - whc[:, None]) * s90) * filc[:, None]
                if refined:
                    m = self._to_cells(fic ** p)
                    fie, S_p, sm = m ** (1 / p), float(m.sum()), self._to_cells(sm)
                else:
                    fie, S_p = fic, float(np.sum(fic ** p))
            fi = np.zeros(n)
            fi[self.cells] = fie
            sx, sy, txy = np.zeros(n), np.zeros(n), np.zeros(n)
            sx[self.cells], sy[self.cells], txy[self.cells] = sm[:, 0], sm[:, 1], sm[:, 2]
            out.append(FEAResult(u, fi, S_p, int(N), sx, sy, txy, iters, direct, S_pc=S_pc))
        return out

    # -------------------------------------------------------------- maps for observations

    def support_map(self) -> np.ndarray:
        return self._node_to_cell(self.support_nodes)

    def load_map(self) -> np.ndarray:
        return self._node_to_cell(self.load_nodes)

    def _node_to_cell(self, node_flag) -> np.ndarray:
        Hr, Wr, rf = self.Hr, self.Wr, self.r
        full = np.zeros((Hr + 1) * (Wr + 1), bool)
        full[self.used_nodes[node_flag]] = True
        g = full.reshape(Hr + 1, Wr + 1)
        cell = g[:-1, :-1] | g[:-1, 1:] | g[1:, :-1] | g[1:, 1:]
        if rf > 1:
            cell = cell.reshape(self.H, rf, self.W, rf).any(axis=(1, 3))
        return cell.reshape(-1).astype(np.float32)


def principal(sx, sy, txy):
    """Principal direction theta1 (rad) of the larger principal stress, and its magnitude."""
    th = 0.5 * np.arctan2(2 * txy, sx - sy)
    c = 0.5 * (sx + sy)
    rr = np.sqrt((0.5 * (sx - sy)) ** 2 + txy ** 2)
    s1, s2 = c + rr, c - rr
    mag = np.where(np.abs(s1) >= np.abs(s2), np.abs(s1), np.abs(s2))
    th = np.where(np.abs(s1) >= np.abs(s2), th, th + np.pi / 2)
    return th, mag


def dominant_direction(sx, sy, txy) -> float:
    """Magnitude-weighted mean of the doubled principal angle."""
    th, mag = principal(sx, sy, txy)
    w = mag ** 2
    if w.sum() <= 0:
        return 0.0
    return float(0.5 * np.arctan2((w * np.sin(2 * th)).sum(), (w * np.cos(2 * th)).sum()))
