"""Procedural training shapes, supports and loads.

Families: random star polygons, unions/differences of circles and rectangles,
plates with holes, thin-rib frames, brackets. Whole families can be held out.
Optionally, layer slices of meshes from a directory (e.g. Thingi10K) with
synthetic loads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from rltp.config import Load2D, LoadSpec, PointLoad, Region

FAMILIES = ("star", "blobs", "plate_holes", "ribs", "bracket")
TRAIN_FAMILIES = ("star", "blobs", "plate_holes", "bracket")
HOLDOUT_FAMILIES = ("ribs",)


def _star(rng, S):
    n = int(rng.integers(6, 14))
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(0.45, 1.0, n) * S * 0.45
    pts = np.stack([S / 2 + rad * np.cos(ang), S / 2 + rad * np.sin(ang)], 1)
    return Polygon(pts).buffer(0)


def _blobs(rng, S):
    parts = []
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = rng.uniform(0.25, 0.75, 2) * S
        if rng.random() < 0.5:
            parts.append(Point(cx, cy).buffer(rng.uniform(0.1, 0.25) * S))
        else:
            w, h = rng.uniform(0.15, 0.6, 2) * S
            parts.append(affinity.rotate(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                                         rng.uniform(0, 90)))
    g = unary_union(parts)
    if rng.random() < 0.4:
        cx, cy = rng.uniform(0.3, 0.7, 2) * S
        g = g.difference(Point(cx, cy).buffer(rng.uniform(0.05, 0.12) * S))
    return g


def _plate_holes(rng, S):
    w, h = rng.uniform(0.55, 0.95, 2) * S
    x0, y0 = (S - w) / 2, (S - h) / 2
    g = box(x0, y0, x0 + w, y0 + h)
    for _ in range(int(rng.integers(1, 4))):
        r = rng.uniform(0.05, 0.16) * S
        cx = rng.uniform(x0 + r + 1, x0 + w - r - 1) if w > 2 * r + 2 else x0 + w / 2
        cy = rng.uniform(y0 + r + 1, y0 + h - r - 1) if h > 2 * r + 2 else y0 + h / 2
        g = g.difference(Point(cx, cy).buffer(r))
    return g


def _ribs(rng, S, p):
    w, h = rng.uniform(0.6, 0.95, 2) * S
    x0, y0 = (S - w) / 2, (S - h) / 2
    t = p * int(rng.integers(2, 4))
    g = box(x0, y0, x0 + w, y0 + h)
    nx, ny = int(rng.integers(1, 4)), int(rng.integers(1, 3))
    cw = (w - (nx + 1) * t) / nx
    ch = (h - (ny + 1) * t) / ny
    if cw > 2 * p and ch > 2 * p:
        for i in range(nx):
            for j in range(ny):
                px = x0 + t + i * (cw + t)
                py = y0 + t + j * (ch + t)
                g = g.difference(box(px, py, px + cw, py + ch))
    return g


def _bracket(rng, S):
    t1, t2 = rng.uniform(0.18, 0.4, 2) * S
    a, b = rng.uniform(0.6, 0.9, 2) * S
    x0 = y0 = (S - max(a, b)) / 2
    g = box(x0, y0, x0 + a, y0 + t1).union(box(x0, y0, x0 + t2, y0 + b))
    if rng.random() < 0.5:
        g = g.union(Polygon([(x0 + t2, y0 + t1), (x0 + t2 + 0.2 * S, y0 + t1), (x0 + t2, y0 + t1 + 0.2 * S)]))
    if rng.random() < 0.5:
        r = 0.3 * min(t1, t2)
        g = g.difference(Point(x0 + a - 1.5 * r - 0.5, y0 + t1 / 2).buffer(r))
    return g


def random_shape(rng, family: str, size_mm: float, p: float):
    S = size_mm
    fn = {"star": lambda: _star(rng, S), "blobs": lambda: _blobs(rng, S),
          "plate_holes": lambda: _plate_holes(rng, S), "ribs": lambda: _ribs(rng, S, p),
          "bracket": lambda: _bracket(rng, S)}[family]
    g = fn()
    # keep inside the grid with a one-cell margin
    g = g.intersection(box(p, p, S - p, S - p))
    if rng.random() < 0.5 and family != "ribs":
        g = affinity.rotate(g, float(rng.choice([0, 90, 180, 270])), origin=(S / 2, S / 2))
    return g.buffer(0)


def mesh_slice_shapes(directory: str | Path, size_mm: float, rng, max_files: int = 200):
    """Yield random planar sections of meshes in `directory`, scaled to fit size_mm."""
    import trimesh
    files = sorted(Path(directory).glob("*.stl"))[:max_files]
    if not files:
        return None
    f = files[int(rng.integers(len(files)))]
    try:
        m = trimesh.load(str(f), force="mesh")
        z = rng.uniform(m.bounds[0, 2], m.bounds[1, 2])
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return None
        pl, _ = sec.to_2D()
        g = unary_union(list(pl.polygons_full))
    except Exception:
        return None
    if g.is_empty:
        return None
    x0, y0, x1, y1 = g.bounds
    s = (size_mm * 0.85) / max(x1 - x0, y1 - y0, 1e-9)
    g = affinity.scale(g, s, s, origin=(x0, y0))
    g = affinity.translate(g, -g.bounds[0] + size_mm * 0.07, -g.bounds[1] + size_mm * 0.07)
    return g.buffer(0)


def random_loads(rng, geom, p: float, neighbors_z: float | None = None) -> LoadSpec:
    """A fixed region, a load region and a force vector; optional 3D point loads above."""
    x0, y0, x1, y1 = geom.bounds
    sides = ["xmin", "xmax", "ymin", "ymax"]
    opp = {"xmin": "xmax", "xmax": "xmin", "ymin": "ymax", "ymax": "ymin"}
    s = sides[int(rng.integers(4))]
    w = p * float(rng.uniform(1.0, 3.0))
    sup = Region(side=s, width=w)
    mode = rng.random()
    if mode < 0.6:
        ld = Region(side=opp[s], width=p * float(rng.uniform(1.0, 3.0)))
    else:
        # a disc on the boundary away from the support
        b = geom.boundary
        for _ in range(20):
            pt = b.interpolate(float(rng.uniform(0, b.length)))
            if not Region(side=s, width=w + 3 * p).resolve(geom.bounds).contains([pt.x], [pt.y])[0]:
                break
        ld = Region(circle=[pt.x, pt.y, p * float(rng.uniform(1.5, 3.5))])
    ang = rng.uniform(0, 2 * np.pi)
    mag = rng.uniform(10, 100)
    spec = LoadSpec([sup], [Load2D(ld, (mag * np.cos(ang), mag * np.sin(ang)))], split_by_height=False)
    if neighbors_z is not None:
        cx, cy = rng.uniform([x0, y0], [x1, y1])
        f = rng.normal(size=3)
        f = f / np.linalg.norm(f) * rng.uniform(5, 50)
        spec.point_loads = [PointLoad((cx, cy, neighbors_z + rng.uniform(1, 20)), tuple(f))]
    return spec.resolve(geom.bounds)
