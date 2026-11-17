"""Ribbons: the surfaces infill is built from, their starting families, and the smooth warp the
search deforms them with.

A ribbon is a polyline in world XY (mm), open or closed. Stage 1 ribbons are vertical, so the
same curve is laid in every infill layer (rltp.surfaces.layout). `fixed` ribbons are laid first
and are never warped: offset rings beside the walls, which other ribbons end against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Ribbon:
    xy: np.ndarray          # (M, 2) world mm; closed: the last point joins back to the first
    closed: bool = False
    fixed: bool = False

    def points(self) -> np.ndarray:
        """The polyline to lay, with a closed ribbon's first point repeated at the end."""
        return np.vstack([self.xy, self.xy[:1]]) if self.closed else self.xy


def line_family(bounds, angle_deg: float, spacing: float, phase: float = 0.0,
                step: float = 1.0) -> list[Ribbon]:
    """Parallel straight ribbons at `angle_deg` (0 = along x), `spacing` mm apart, covering
    `bounds` (x0, y0, x1, y1) and running past it (the layout clips them to the infill region).
    `phase` shifts the family across the lines; `step` is the point spacing along each line, so
    a warp can bend it."""
    x0, y0, x1, y1 = bounds
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    R = 0.5 * float(np.hypot(x1 - x0, y1 - y0)) + spacing
    a = np.radians(angle_deg)
    t = np.array([np.cos(a), np.sin(a)])
    n = np.array([-np.sin(a), np.cos(a)])
    m = int(np.ceil(2 * R / step)) + 1
    s = np.linspace(-R, R, m)[:, None]
    out = []
    for o in np.arange(-R + (phase % spacing), R, spacing):
        out.append(Ribbon(np.array([cx, cy]) + o * n + s * t))
    return out


def offset_rings(poly, first: float, spacing: float, n: int, step: float = 0.25) -> list[Ribbon]:
    """`n` closed, fixed ribbons following every outline of `poly` (shapely, world mm: the outer
    edge and each hole) at `first`, `first + spacing`, ... mm inside it: edge-following bands
    beside the walls. Bead centres one bead apart touch, so spacing = bead width makes a
    solid band."""
    out = []
    for i in range(n):
        g = poly.buffer(-(first + i * spacing), quad_segs=16)
        if g.is_empty:
            break
        for pg in getattr(g, "geoms", [g]):
            for ring in [pg.exterior, *pg.interiors]:
                L = ring.length
                if L < 4 * step:
                    continue
                d = np.linspace(0.0, L, int(np.ceil(L / step)), endpoint=False)
                xy = np.array([ring.interpolate(x).coords[0] for x in d])
                out.append(Ribbon(xy, closed=True, fixed=True))
    return out


class Warp:
    """A smooth displacement field over `bounds`: a grid of 2D vectors every `spacing` mm,
    interpolated by cubic splines, each component amp * tanh(parameter), so no point moves
    more than `amp` mm. Parameters all 0 leave every ribbon where it is."""

    def __init__(self, bounds, spacing: float, amp: float):
        x0, y0, x1, y1 = bounds
        self.g = float(spacing)
        self.amp = float(amp)
        self.x0, self.y0 = x0 - spacing, y0 - spacing
        self.nx = int(np.ceil((x1 - x0) / spacing)) + 3
        self.ny = int(np.ceil((y1 - y0) / spacing)) + 3

    @property
    def n(self) -> int:
        return 2 * self.nx * self.ny

    def __call__(self, params, xy: np.ndarray) -> np.ndarray:
        P = self.amp * np.tanh(np.asarray(params, float).reshape(2, self.ny, self.nx))
        gx = (xy[:, 0] - self.x0) / self.g
        gy = (xy[:, 1] - self.y0) / self.g
        dx = ndimage.map_coordinates(P[0], [gy, gx], order=3, mode="nearest")
        dy = ndimage.map_coordinates(P[1], [gy, gx], order=3, mode="nearest")
        return xy + np.c_[dx, dy]

    def apply(self, params, ribbons: list[Ribbon]) -> list[Ribbon]:
        """The ribbons with every non-fixed one warped."""
        return [rb if rb.fixed else Ribbon(self(params, rb.xy), rb.closed) for rb in ribbons]


class Warp3D:
    """A Warp that also varies with height, so vertical ribbons become 3D surfaces: the same XY
    grid at `levels` heights spread evenly over [z0, z1], linear in z between them.

    The middle level is a 2D Warp (its parameters come first, so a finished vertical search
    warm-starts it), interpolated by cubic splines. Every other level adds, per grid node and
    component, a step from the level next to it toward the middle, bounded to
    dz * tan(max_lean) / sqrt(2). The height-varying part is interpolated bilinearly in XY (a
    weighted average of grid values, so it never overshoots them), which makes the bound exact:
    no surface leans more than `max_lean` degrees from vertical anywhere. All steps 0: the 2D
    warp."""

    def __init__(self, bounds, spacing: float, amp: float, z0: float, z1: float, levels: int = 3,
                 max_lean: float = 45.0):
        if levels < 2 or z1 <= z0:
            raise ValueError("Warp3D needs at least 2 levels over a positive height")
        self.base = Warp(bounds, spacing, amp)
        self.levels = int(levels)
        self.z = np.linspace(z0, z1, self.levels)
        dz = (z1 - z0) / (self.levels - 1)
        self.step = dz * np.tan(np.radians(max_lean)) / np.sqrt(2.0)
        self.mid = self.levels // 2
        self.others = [i for i in range(self.levels) if i != self.mid]

    @property
    def n(self) -> int:
        return self.base.n * self.levels

    def grids(self, params) -> np.ndarray:
        """Displacement grids per level, (levels, 2, ny, nx) mm."""
        b = self.base
        p = np.asarray(params, float).reshape(self.levels, 2, b.ny, b.nx)
        G = np.empty_like(p)
        G[self.mid] = b.amp * np.tanh(p[0])
        steps = dict(zip(self.others, p[1:]))
        for i in range(self.mid + 1, self.levels):
            G[i] = G[i - 1] + self.step * np.tanh(steps[i])
        for i in range(self.mid - 1, -1, -1):
            G[i] = G[i + 1] + self.step * np.tanh(steps[i])
        return G

    def _at(self, G: np.ndarray, z: float) -> np.ndarray:
        """Level grids G interpolated to height z (clamped to [z0, z1]), (2, ny, nx)."""
        t = float(np.clip((z - self.z[0]) / (self.z[-1] - self.z[0]), 0.0, 1.0)) * (self.levels - 1)
        i = min(int(np.floor(t)), self.levels - 2)
        f = t - i
        return (1 - f) * G[i] + f * G[i + 1]

    def __call__(self, params, xy: np.ndarray, z: float) -> np.ndarray:
        b = self.base
        G = self.grids(params)
        Pz = self._at(G, z) - G[self.mid]
        gx = (xy[:, 0] - b.x0) / b.g
        gy = (xy[:, 1] - b.y0) / b.g
        out = np.array(xy, float, copy=True)
        for c in range(2):
            out[:, c] += ndimage.map_coordinates(G[self.mid][c], [gy, gx], order=3, mode="nearest")
            out[:, c] += ndimage.map_coordinates(Pz[c], [gy, gx], order=1, mode="nearest")
        return out

    def apply(self, params, ribbons: list[Ribbon], z: float) -> list[Ribbon]:
        """The ribbons' cut at height z: every non-fixed one warped by the field at z."""
        return [rb if rb.fixed else Ribbon(self(params, rb.xy, z), rb.closed) for rb in ribbons]

    def lean(self, params, ribbons: list[Ribbon], zs) -> float:
        """The largest lean from vertical (degrees) of any ribbon point between consecutive
        heights `zs` (the infill layers' mid-heights)."""
        zs = sorted(zs)
        worst = 0.0
        prev = None
        for z in zs:
            cur = [self(params, rb.xy, z) for rb in ribbons if not rb.fixed]
            if prev is not None and cur:
                d = max(float(np.hypot(*(a - b).T).max()) for a, b in zip(cur, prev[1]))
                worst = max(worst, float(np.degrees(np.arctan2(d, z - prev[0]))))
            prev = (z, cur)
        return worst
