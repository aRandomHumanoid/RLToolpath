"""Planar slicing: mesh -> shapely polygons per layer."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

log = logging.getLogger(__name__)


@dataclass
class Slice:
    k: int
    z_cut: float     # where the mesh was cut (middle of the layer)
    z_top: float     # nozzle height while printing the layer
    h: float
    geom: object     # shapely (Multi)Polygon in world XY


def layer_heights(zmin: float, zmax: float, h: float, h0: float):
    """Mid-layer cut heights and layer tops. First layer h0, then h."""
    tops, cuts, hs = [], [], []
    top = zmin + h0
    cuts.append(zmin + h0 / 2)
    tops.append(top)
    hs.append(h0)
    while top + h / 2 < zmax - 1e-9:
        cuts.append(top + h / 2)
        top += h
        tops.append(top)
        hs.append(h)
    return np.array(cuts), np.array(tops), np.array(hs)


def _path_to_geom(path) -> object:
    polys = list(path.polygons_full)
    if not polys:
        return Polygon()
    g = unary_union(polys)
    T = path.metadata.get("to_3D") if hasattr(path, "metadata") else None
    if T is not None:
        T = np.asarray(T)
        if not np.allclose(T[:2, :2], np.eye(2)) or not np.allclose(T[:2, 3], 0):
            g = affinity.affine_transform(g, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 3], T[1, 3]])
    return g.buffer(0)


def slice_mesh(mesh: trimesh.Trimesh, h: float, h0: float | None = None) -> list[Slice]:
    h0 = h if h0 is None else h0
    zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    cuts, tops, hs = layer_heights(zmin, zmax, h, h0)
    secs = mesh.section_multiplane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], heights=cuts.tolist())
    out = []
    for k, (zc, zt, hh, sec) in enumerate(zip(cuts, tops, hs, secs)):
        if sec is None:
            g = Polygon()
        else:
            g = _path_to_geom(sec)
        if isinstance(g, Polygon) or isinstance(g, MultiPolygon) or g.is_empty:
            pass
        else:
            g = unary_union([x for x in getattr(g, "geoms", [g]) if x.area > 0])
        out.append(Slice(k, float(zc - zmin), float(zt - zmin), float(hh), g))
    empties = sum(1 for s in out if s.geom.is_empty)
    if empties:
        log.warning("%d of %d slices are empty", empties, len(out))
    return out


def footprint(slices: list[Slice]):
    """Union of all slice outlines (used to size a shared grid)."""
    gs = [s.geom for s in slices if not s.geom.is_empty]
    if not gs:
        raise ValueError("mesh produced no slices")
    return shapely.union_all(gs)
