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

    def fi_agg(self, p: float) -> float:
        return float((self.S_p / self.N) ** (1 / p)) if self.N else 0.0


class FEAProblem:
    """Fixed per layer during optimisation: mesh, supports and loads. Only the
    bead-direction weights change between solves."""

    def __init__(self, mask: np.ndarray, frame: Frame, h: float, mat: Material,
                 loads: LoadSpec | None, scale: float = 1.0, z: float | None = None,
                 p_norm: float = 8.0):
        H, W = mask.shape
        self.H, self.W, self.n = H, W, H * W
        self.mat = mat
        self.p_norm = p_norm
        self.cells = np.flatnonzero(mask.reshape(-1))
        ne = self.cells.size
        r, c = np.divmod(self.cells, W)
        n0 = r * (W + 1) + c
        enodes = np.stack([n0, n0 + 1, n0 + W + 2, n0 + W + 1], 1)
        used = np.unique(enodes)
        remap = np.full((H + 1) * (W + 1), -1, np.int64)
        remap[used] = np.arange(used.size)
        en = remap[enodes]
        self.edof = np.stack([2 * en, 2 * en + 1], 2).reshape(ne, 8)
        ndof = 2 * used.size
        self.ndof = ndof

        Q = q_matrix(mat)
        self.Q0, self.Q90 = Q, q_rot90(Q)
        self.K0 = ke_q4(self.Q0, h)
        self.K90 = ke_q4(self.Q90, h)
        self.B0 = _bhat(0.0, 0.0) * (2 / frame.p)

        nx, ny = frame.node_xy(H, W)
        nx, ny = nx[used], ny[used]
        fixed = np.zeros(ndof, bool)
        f = np.zeros(ndof)
        self.support_nodes = np.zeros(used.size, bool)
        self.load_nodes = np.zeros(used.size, bool)
        self.load_dir = np.zeros(2)
        if loads is not None:
            for reg in loads.supports:
                inside = reg.contains(nx, ny, z)
                self.support_nodes |= inside
            fixed[0::2] = self.support_nodes
            fixed[1::2] = self.support_nodes
            Rt = frame.R.T  # world -> local
            for ld in loads.loads:
                inside = ld.region.contains(nx, ny, z) & ~self.support_nodes
                cnt = int(inside.sum())
                if cnt == 0:
                    continue
                F = Rt @ np.asarray(ld.force[:2], float) * scale
                f[0::2][inside] += F[0] / cnt
                f[1::2][inside] += F[1] / cnt
                self.load_nodes |= inside
                self.load_dir += F
        self.used_nodes = used
        self.active = bool(fixed.any() and np.abs(f).sum() > 0 and ne > 0)
        self.fixed = fixed
        self.free = np.flatnonzero(~fixed)
        self.f = f
        self.ff = f[self.free]
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

    def assemble(self, wh_cells: np.ndarray) -> sp.csc_matrix:
        wh = wh_cells[:, None, None]
        Ke = wh * self.K0 + (1 - wh) * self.K90
        data = np.bincount(self._scatter, weights=Ke.reshape(-1)[self._keep], minlength=self._nnz)
        reg = 1e-9 * data[self._diag_pos].mean()
        data[self._diag_pos] += reg      # keeps floating islands solvable
        return sp.csc_matrix((data, self._indices, self._indptr), shape=(self._nf, self._nf))

    def _pcg(self, K, b, x0, M, rtol=1e-9, maxiter=40):
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
              keep: bool = True) -> FEAResult:
        """wh: (n,) fraction of each cell's bead along local x."""
        n = self.n
        zeros = np.zeros(n)
        if not self.active:
            return FEAResult(np.zeros(self.ndof), zeros, 0.0, 0, zeros, zeros, zeros)
        whc = wh[self.cells]
        K = self.assemble(whc)
        iters, used_direct = 0, False
        if direct or self._lu is None:
            lu = splu(K)
            uf = lu.solve(self.ff)
            used_direct = True
            if keep:
                self._lu = lu
        else:
            x0 = warm[self.free] if warm is not None else self._lu.solve(self.ff)
            uf, iters, ok = self._pcg(K, self.ff, x0, self._lu.solve)
            if not ok:
                lu = splu(K)
                uf = lu.solve(self.ff)
                used_direct = True
                if keep:
                    self._lu = lu
        u = np.zeros(self.ndof)
        u[self.free] = uf
        return self._post(u, whc, iters, used_direct)

    def drop_factorization(self) -> None:
        self._lu = None

    def _post(self, u, whc, iters, direct) -> FEAResult:
        ue = u[self.edof]                       # (ne, 8)
        eps = ue @ self.B0.T                    # (ne, 3)
        s0 = eps @ self.Q0.T                    # stresses if bead along x
        s90 = eps @ self.Q90.T                  # stresses if bead along y
        fi0 = tsai_wu_fi(s0[:, 0], s0[:, 1], s0[:, 2], self.mat)
        fi90 = tsai_wu_fi(s90[:, 1], s90[:, 0], s90[:, 2], self.mat)
        fie = np.where(whc >= 1, fi0, np.where(whc <= 0, fi90, np.maximum(fi0, fi90)))
        sm = whc[:, None] * s0 + (1 - whc[:, None]) * s90
        n = self.n
        fi = np.zeros(n)
        fi[self.cells] = fie
        sx, sy, txy = np.zeros(n), np.zeros(n), np.zeros(n)
        sx[self.cells], sy[self.cells], txy[self.cells] = sm[:, 0], sm[:, 1], sm[:, 2]
        S_p = float(np.sum(fie ** self.p_norm))
        return FEAResult(u, fi, S_p, int(self.cells.size), sx, sy, txy, iters, direct)

    # -------------------------------------------------------------- maps for observations

    def support_map(self) -> np.ndarray:
        return self._node_to_cell(self.support_nodes)

    def load_map(self) -> np.ndarray:
        return self._node_to_cell(self.load_nodes)

    def _node_to_cell(self, node_flag) -> np.ndarray:
        H, W = self.H, self.W
        full = np.zeros((H + 1) * (W + 1), bool)
        full[self.used_nodes[node_flag]] = True
        g = full.reshape(H + 1, W + 1)
        cell = g[:-1, :-1] | g[:-1, 1:] | g[1:, :-1] | g[1:, 1:]
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
