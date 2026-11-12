"""Data model: layer frames and layers (grid + cover).

Grid convention: cell (r, c) has flat index i = r * W + c and local centre
((c + 0.5) p, (r + 0.5) p). Local x runs along columns, local y along rows.
World = origin + R(theta) @ local.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Frame:
    ox: float
    oy: float
    theta: float
    p: float

    @property
    def R(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def to_world(self, u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        c, s = np.cos(self.theta), np.sin(self.theta)
        return self.ox + c * u - s * v, self.oy + s * u + c * v

    def to_local(self, x, y):
        x = np.asarray(x, float) - self.ox
        y = np.asarray(y, float) - self.oy
        c, s = np.cos(self.theta), np.sin(self.theta)
        return c * x + s * y, -s * x + c * y

    def cell_centers(self, H: int, W: int):
        """World (x, y) of every cell centre, flat arrays of length H*W."""
        r, c = np.divmod(np.arange(H * W), W)
        return self.to_world((c + 0.5) * self.p, (r + 0.5) * self.p)

    def cell_xy(self, cells, W: int):
        r, c = np.divmod(np.asarray(cells), W)
        return self.to_world((c + 0.5) * self.p, (r + 0.5) * self.p)

    def node_xy(self, H: int, W: int):
        """World (x, y) of the (H+1) x (W+1) grid nodes, flat, row-major."""
        i, j = np.divmod(np.arange((H + 1) * (W + 1)), W + 1)
        return self.to_world(j * self.p, i * self.p)

    def world_to_cell(self, x, y, H: int, W: int) -> np.ndarray:
        u, v = self.to_local(x, y)
        c = np.floor(u / self.p).astype(np.int64)
        r = np.floor(v / self.p).astype(np.int64)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        return np.where(ok, r * W + c, -1)

    def to_dict(self) -> dict:
        return {"ox": self.ox, "oy": self.oy, "theta": self.theta, "p": self.p}


@dataclass
class Layer:
    k: int
    z: float                 # top of the layer (nozzle height while printing it)
    h: float                 # layer thickness
    mask: np.ndarray         # bool (H, W)
    frame: Frame
    nxt: np.ndarray          # int32 (H*W,), -1 for none
    prv: np.ndarray
    seam: set = field(default_factory=set)   # one cell per closed loop: where it starts and ends
    poly: Any = None         # shapely geometry of the exact slice outline (world)
    first: bool = False      # first layer (slower speed)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def H(self) -> int:
        return self.mask.shape[0]

    @property
    def W(self) -> int:
        return self.mask.shape[1]

    @property
    def n(self) -> int:
        return self.mask.size

    @property
    def flat_mask(self) -> np.ndarray:
        return self.mask.reshape(-1)

    def copy(self) -> "Layer":
        return Layer(self.k, self.z, self.h, self.mask, self.frame,
                     self.nxt.copy(), self.prv.copy(), set(self.seam), self.poly, self.first)

    def invalidate(self) -> None:
        self._cache.clear()

    def seam_dict(self) -> dict:
        """{loop id: seam cell}, loop id = smallest cell index in the loop."""
        from rltp.cover.cover import get_pieces
        pc = get_pieces(self)
        out = {}
        for lab, cells in enumerate(pc.cells):
            if pc.is_loop[lab]:
                s = [c for c in cells if c in self.seam]
                out[int(min(cells))] = int(s[0]) if s else None
        return out

    @staticmethod
    def empty(k, z, h, mask, frame, poly=None, first=False) -> "Layer":
        n = mask.size
        return Layer(k, z, h, mask.astype(bool), frame,
                     np.full(n, -1, np.int32), np.full(n, -1, np.int32), set(), poly, first)
