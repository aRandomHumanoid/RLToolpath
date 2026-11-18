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


def density_family(bounds, angle_deg: float, rho, bead: float, phase: float = 0.0, step: float = 0.5,
                   dv: float = 0.05, inside=None, level: float | None = None,
                   norm: str = "station", smooth_along: float = 0.0) -> list[Ribbon]:
    """Lines along `angle_deg` whose spacing follows a density field: at every station along the
    lines, line i sits where the cumulative density across the family, integral(rho dv) / bead,
    reaches i + phase. Where rho changes along a line the lines curve, so they bunch where rho is
    high and spread where it is low, and stay continuous. rho(x, y) -> density in (0, 1] takes
    world mm arrays; rho = const gives straight lines bead / rho apart. Covers `bounds` and runs
    past it, like line_family.

    The cumulative runs from one edge of the bounds, so a station crossing more material than
    another shifts every line beyond it: lines far from a dense spot tilt to make room for it.
    With `inside(x, y) -> bool` (the part's infill region) and `level` (the uniform density),
    the field is rescaled per station so the part's cross-section holds as many lines as
    uniform lines at `level` would: `norm` "station" over the whole cross-section, "segment" over
    each stretch of it inside the part separately (lines never move across a hole). Lines then
    only gather within a cross-section, toward its high-density side, and lines past the part
    stay straight.

    `smooth_along` (mm, Gaussian sigma) smooths the field along the lines first: lines tilt where
    the density across them changes from station to station, so this bounds the tilt (short
    lines across a thin bar otherwise zigzag through its density) and keeps the contrast across
    the family."""
    x0, y0, x1, y1 = bounds
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    R = 0.5 * float(np.hypot(x1 - x0, y1 - y0)) + 2.0
    a = np.radians(angle_deg)
    t = np.array([np.cos(a), np.sin(a)])
    n = np.array([-np.sin(a), np.cos(a)])
    us = np.arange(-R, R + step, step)
    vs = np.arange(-R, R + dv, dv)
    U, V = np.meshgrid(us, vs, indexing="ij")
    X = cx + U * t[0] + V * n[0]
    Y = cy + U * t[1] + V * n[1]
    D = np.clip(np.asarray(rho(X.ravel(), Y.ravel()), float).reshape(X.shape), 1e-3, 1.0)
    if smooth_along > 0:
        D = ndimage.gaussian_filter1d(D, smooth_along / step, axis=0, mode="nearest")
    if inside is not None and level is not None:
        D = _normalise(D, np.asarray(inside(X.ravel(), Y.ravel()), bool).reshape(X.shape), level, norm)
    C = np.concatenate([np.zeros((len(us), 1)), np.cumsum(0.5 * (D[:, 1:] + D[:, :-1]) * dv, axis=1) / bead], axis=1)
    m = int(np.floor(C[:, -1].min() - phase))
    lines = np.empty((m, len(us), 2))
    for j in range(len(us)):
        v = np.interp(np.arange(m) + phase, C[j], vs)
        lines[:, j, 0] = cx + us[j] * t[0] + v * n[0]
        lines[:, j, 1] = cy + us[j] * t[1] + v * n[1]
    return [Ribbon(lines[i]) for i in range(m)]


def _normalise(D: np.ndarray, I: np.ndarray, level: float, norm: str) -> np.ndarray:
    """D (stations x samples) rescaled so each station's inside samples (or each run of them,
    norm "segment") average `level`; outside samples become `level`. Capped at 1: a run that
    would need more keeps the cap (fewer lines than uniform there)."""
    if norm not in ("station", "segment"):
        raise ValueError(f"norm: station or segment, not {norm!r}")
    out = np.full_like(D, level)
    for j in range(D.shape[0]):
        idx = np.flatnonzero(I[j])
        if not idx.size:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if norm == "segment" else [idx]
        for r in runs:
            d = D[j, r]
            out[j, r] = np.minimum(d * (level / max(float(d.mean()), 1e-9)), 1.0)
    return out


def region_inside(region: np.ndarray, frame):
    """inside(x, y) -> bool for a cell mask on `frame`'s grid (world mm in), for density_family."""
    H, W = region.shape

    def inside(x, y):
        u, v = frame.to_local(x, y)
        c = np.floor(np.asarray(u, float) / frame.p).astype(np.int64)
        r = np.floor(np.asarray(v, float) / frame.p).astype(np.int64)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        out = np.zeros(r.shape, bool)
        out[ok] = region[r[ok], c[ok]]
        return out
    return inside


def stress_density(mag: np.ndarray, frame, region: np.ndarray, target: float, gamma: float,
                   smooth_cells: float = 15.0, lo: float = 0.15, hi: float = 0.9):
    """A density field for density_family: target x (stress magnitude / its mean)^gamma over the
    cells of `region` (on `frame`'s grid), smoothed by `smooth_cells`, clipped to [lo, hi] and
    rescaled until its mean over the region is `target` (outside the region it is `target`).
    gamma 0: uniform. Returns rho(x, y)."""
    from scipy import ndimage as nd
    w = region.astype(float)
    ms = nd.gaussian_filter(np.where(region, mag, 0.0), smooth_cells) / np.maximum(nd.gaussian_filter(w, smooth_cells), 1e-9)
    ms = np.maximum(ms, 1e-9)
    base = (ms / ms[region].mean()) ** gamma
    s = 1.0
    for _ in range(30):
        r = np.clip(target * s * base, lo, hi)
        mean = r[region].mean()
        if abs(mean - target) < 1e-4:
            break
        s *= target / mean
    grid = np.where(region, np.clip(target * s * base, lo, hi), target)
    H, W = grid.shape

    def rho(x, y):
        u, v = frame.to_local(x, y)
        cc = [np.asarray(v, float) / frame.p - 0.5, np.asarray(u, float) / frame.p - 0.5]
        return ndimage.map_coordinates(grid, cc, order=1, mode="nearest")
    return rho


def streamline_family(theta: np.ndarray, tframe, region: np.ndarray, rframe, spacing: float,
                      step: float = 0.25, test: float = 0.5, min_len: float = 2.0, max_len: float = 400.0,
                      seed_xy=None) -> list[Ribbon]:
    """Evenly spaced streamlines of a line field (Jobard & Lefer 1997): ribbons that follow the
    bead direction `theta` (radians mod pi, per cell of `tframe`'s grid, e.g. the stress
    consensus), `spacing` mm apart, inside `region` (bool, per cell of `rframe`'s grid).

    Each line is integrated both ways from its seed (midpoint steps of `step` mm, the field's
    sign kept consistent with the last step) until it leaves the region, comes within
    test x spacing of another line or of itself further back, or reaches `max_len`. New seeds
    are tried spacing mm to either side of every accepted line (every spacing along it), and on
    a 2-spacing grid over the region for pockets no line reaches; a seed closer than spacing to a
    line is skipped. Lines shorter than `min_len` are dropped."""
    from scipy.spatial import cKDTree
    th = np.asarray(theta, float)
    Ht, Wt = th.shape
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    Hr, Wr = region.shape
    ct, st = np.cos(tframe.theta), np.sin(tframe.theta)

    def direction(x: float, y: float):
        u, v = tframe.to_local(np.array([x]), np.array([y]))
        cc = [np.asarray(v, float) / tframe.p - 0.5, np.asarray(u, float) / tframe.p - 0.5]
        a = 0.5 * np.arctan2(ndimage.map_coordinates(s2, cc, order=1, mode="nearest")[0],
                             ndimage.map_coordinates(c2, cc, order=1, mode="nearest")[0])
        lx, ly = np.cos(a), np.sin(a)
        return np.array([ct * lx - st * ly, st * lx + ct * ly])

    def inside(p) -> bool:
        cell = int(rframe.world_to_cell(np.array([p[0]]), np.array([p[1]]), Hr, Wr)[0])
        return cell >= 0 and bool(region.reshape(-1)[cell])

    lines: list[np.ndarray] = []
    pts = np.zeros((0, 2))
    tree = None
    dt = test * spacing
    gap = int(np.ceil(2 * spacing / step))           # own points this far back may stop the line

    def near(p, r) -> bool:
        return tree is not None and tree.query(p, distance_upper_bound=r)[0] < r

    def trace(s):
        halves = []
        for sign in (1.0, -1.0):
            p = np.asarray(s, float)
            d_prev = sign * direction(*p)
            out = []
            n = int(max_len / step / 2)
            for _ in range(n):
                d = direction(*p)
                if d @ d_prev < 0:
                    d = -d
                mid = p + 0.5 * step * d
                dm = direction(*mid)
                if dm @ d < 0:
                    dm = -dm
                q = p + step * dm
                if not inside(q) or near(q, dt):
                    break
                if len(out) > gap:
                    own = np.asarray(out[:-gap])
                    if (np.hypot(*(own - q).T) < dt).any():
                        break
                out.append(q)
                d_prev, p = dm, q
            halves.append(out)
        return np.array(halves[1][::-1] + [list(s)] + halves[0], float)

    x0, y0 = rframe.cell_xy(np.flatnonzero(region.reshape(-1)), Wr)
    if seed_xy is None:
        i = int(np.argmin((x0 - x0.mean()) ** 2 + (y0 - y0.mean()) ** 2))
        seed_xy = (x0[i], y0[i])
    grid = []
    for gx in np.arange(x0.min(), x0.max(), 2 * spacing):
        for gy in np.arange(y0.min(), y0.max(), 2 * spacing):
            grid.append((gx, gy))
    queue = [tuple(seed_xy)]
    gi = 0
    while queue or gi < len(grid):
        if queue:
            s = np.asarray(queue.pop(0), float)
        else:
            s = np.asarray(grid[gi], float)
            gi += 1
        if not inside(s) or near(s, spacing * 0.999):
            continue
        line = trace(s)
        if len(line) < 2 or float(np.hypot(*np.diff(line, axis=0).T).sum()) < min_len:
            continue
        lines.append(line)
        pts = np.vstack([pts, line])
        tree = cKDTree(pts)
        k = max(1, int(round(spacing / step)))
        for i in range(0, len(line), k):
            j = min(i + 1, len(line) - 1)
            t = line[j] - line[max(j - 1, 0)]
            nt = np.hypot(*t)
            if nt == 0:
                continue
            nrm = np.array([-t[1], t[0]]) / nt
            queue.append(tuple(line[i] + spacing * nrm))
            queue.append(tuple(line[i] - spacing * nrm))
    return [Ribbon(line) for line in lines]


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
