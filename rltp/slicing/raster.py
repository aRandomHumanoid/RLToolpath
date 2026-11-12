"""Polygon -> cell mask, and layer frames."""
from __future__ import annotations

import logging

import numpy as np
import shapely
from scipy import ndimage

from rltp.model import Frame

log = logging.getLogger(__name__)


def make_frame(geom, theta: float, p: float, margin_cells: int = 1):
    """A frame whose grid covers `geom` (any shapely geometry, world XY) when rotated by theta.

    Returns (frame, H, W). With theta = 0 and a shared `geom` (the part's
    footprint), every layer gets the identical grid so cells stack exactly."""
    c, s = np.cos(theta), np.sin(theta)
    xy = np.asarray(shapely.get_coordinates(geom))
    if xy.size == 0:
        raise ValueError("empty geometry")
    u = c * xy[:, 0] + s * xy[:, 1]
    v = -s * xy[:, 0] + c * xy[:, 1]
    u0 = np.floor(u.min() / p) * p - margin_cells * p
    v0 = np.floor(v.min() / p) * p - margin_cells * p
    W = int(np.ceil((u.max() - u0) / p)) + margin_cells
    H = int(np.ceil((v.max() - v0) / p)) + margin_cells
    ox, oy = c * u0 - s * v0, s * u0 + c * v0
    return Frame(float(ox), float(oy), float(theta), float(p)), H, W


def rasterize(geom, frame: Frame, H: int, W: int, min_cells: int = 3):
    """A cell is in the part if its centre is inside the polygon.

    Drops 4-connected pieces smaller than `min_cells` and returns them as a list
    of (cell count, world centroid) for logging."""
    if geom is None or geom.is_empty:
        return np.zeros((H, W), bool), []
    x, y = frame.cell_centers(H, W)
    shapely.prepare(geom)
    mask = shapely.contains_xy(geom, x, y).reshape(H, W)
    dropped = []
    if min_cells > 1 and mask.any():
        lab, nlab = ndimage.label(mask)
        sizes = np.bincount(lab.ravel(), minlength=nlab + 1)
        small = np.nonzero((sizes < min_cells) & (np.arange(nlab + 1) > 0))[0]
        for s in small:
            cells = np.nonzero(lab.ravel() == s)[0]
            cx, cy = frame.cell_xy(cells, W)
            dropped.append((int(cells.size), (float(cx.mean()), float(cy.mean()))))
            mask[lab == s] = False
    return mask, dropped
