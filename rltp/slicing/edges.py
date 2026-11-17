"""Edges of a part's outline: the sides a support or a load can act on.

Each contour (the outline and each hole) is cut at its corners, where it turns by more
than `corner_deg`, and where a straight run meets a curved one. The edges are then what
a CAD model would call them: straight sides, arcs (a fillet, the round end of a link)
and whole round holes. Slicing a mesh gives polygons, so an arc arrives as many short
segments; a run of points is one edge while it stays within `tol` of one line or one
circle. `edge_kind` names each one (hole, side, arc) for the random load cases.
"""
from __future__ import annotations

import numpy as np


def _rings(geom):
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty:
            continue
        yield poly.exterior
        yield from poly.interiors


def _turns(xy: np.ndarray) -> np.ndarray:
    """Signed turn (degrees) at each vertex of a closed ring given without its repeated point."""
    d = np.roll(xy, -1, 0) - xy
    a = np.arctan2(d[:, 1], d[:, 0])
    return np.degrees((a - np.roll(a, 1) + np.pi) % (2 * np.pi) - np.pi)


def _fits_line(P: np.ndarray, tol: float) -> bool:
    t = P[-1] - P[0]
    L = float(np.hypot(*t))
    if L < 1e-12:
        return False
    n = np.array([-t[1], t[0]]) / L
    return bool(np.abs((P - P[0]) @ n).max() <= tol)


def _fits_circle(P: np.ndarray, tol: float) -> bool:
    if len(P) < 4:              # any 3 points fit a circle (a side plus one point of the next arc)
        return False
    c0 = P.mean(0)
    Q = P - c0
    A = np.column_stack([Q, np.ones(len(Q))])
    sol = np.linalg.lstsq(A, (Q * Q).sum(1), rcond=None)[0]       # x^2 + y^2 = 2 a x + 2 b y + c
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return False
    return bool(np.abs(np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) - np.sqrt(r2)).max() <= tol)


def _fits(P: np.ndarray, tol: float) -> bool:
    return _fits_line(P, tol) or _fits_circle(P, tol)


def _run_ok(P: np.ndarray, a: int, b: int, tol: float) -> bool:
    """Whether points a..b form one edge. Three points that aren't a line are an arc only
    if the next point stays on their circle."""
    Q = P[a:b + 1]
    if _fits_line(Q, tol):
        return True
    if len(Q) >= 4:
        return _fits_circle(Q, tol)
    return b + 1 < len(P) and _fits_circle(P[a:b + 2], tol)


def _runs(P: np.ndarray, tol: float) -> list[int]:
    """Cut indices into a chain of points so each run between cuts fits a line or a circle
    (greedy: each run as long as it can be)."""
    cuts, i, m = [0], 0, len(P) - 1
    while i < m:
        j = i + 1
        while j < m and _run_ok(P, i, j + 1, tol):
            j += 1
        cuts.append(j)
        i = j
    return cuts


EDGE_KINDS = ("hole", "side", "arc")


def edge_kind(edge: np.ndarray, geom, tol: float = 0.002) -> str:
    """What an outline edge is: "hole" (a whole round hole: a closed edge around empty space),
    "side" (straight) or "arc" (any other curve, a round outer contour included)."""
    if len(edge) > 3 and np.allclose(edge[0], edge[-1]):
        from shapely.geometry import Polygon
        inner = Polygon(edge)
        if inner.is_valid and inner.area > 0 and geom.intersection(inner).area < 0.5 * inner.area:
            return "hole"
        return "arc"
    return "side" if _fits_line(edge, tol) else "arc"


def outline_edges(geom, tol: float = 0.002, corner_deg: float = 40.0) -> list[np.ndarray]:
    """The edges of a (Multi)Polygon's contours, each an (n, 2) polyline of the contour's
    own points. A contour with no corner and one fit (a round hole) is a single closed edge,
    its first point repeated at the end."""
    edges = []
    for ring in _rings(geom):
        xy = np.asarray(ring.simplify(1e-4).coords)[:-1]   # drop duplicate and collinear points
        if len(xy) < 3:
            continue
        corner = np.flatnonzero(np.abs(_turns(xy)) > corner_deg)
        if corner.size:
            xy = np.roll(xy, -int(corner[0]), 0)            # start at a corner
            cut = np.append((corner - corner[0]) % len(xy), len(xy))
            cut = np.unique(cut)
            P = np.vstack([xy, xy[:1]])
            for a, b in zip(cut[:-1], cut[1:]):
                chain = P[a:b + 1]
                c = _runs(chain, tol)
                edges += [chain[s:e + 1] for s, e in zip(c[:-1], c[1:])]
            continue
        P = np.vstack([xy, xy[:1]])                         # smooth closed contour
        c = _runs(P, tol)
        if len(c) == 2:
            edges.append(P)
            continue
        pieces = [P[s:e + 1] for s, e in zip(c[:-1], c[1:])]
        wrap = np.vstack([pieces[-1], pieces[0][1:]])       # the start point split one run in two
        if _fits(wrap, tol):
            pieces = [wrap] + pieces[1:-1]
        edges += pieces
    return edges
