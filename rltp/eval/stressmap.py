"""A part's stress map for laying out and searching toolpaths.

One plane-stress solve with an isotropic material over the union of the layers'
cells under the primary load case (whole load, z limits ignored) gives, per cell,
the direction and size of the larger principal stress. It is computed once, before
any toolpath exists, so it describes where and which way the part is loaded rather
than how a particular toolpath carries it.

orientation(): which cells want beads along y (True) rather than x, by the sign of
the stress-weighted, smoothed cos(2 theta), so small low-stress patches follow
their surroundings.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from rltp.eval.fea2d import FEAProblem, isotropic, principal


def stress_field(mask: np.ndarray, frame, spec) -> dict | None:
    """{"theta", "mag"} per cell (H x W), or None without supports and loads."""
    if not spec.loads.has_2d or not mask.any():
        return None
    m = spec.material
    iso = isotropic(0.5 * (m.E1 + m.E2), m.nu12)
    fp = FEAProblem(mask, frame, 1.0, iso, spec.loads, 1.0, z=None)
    if not fp.active:
        return None
    r = fp.solve(np.full(mask.size, 0.5), direct=True, keep=False)
    th, mag = principal(r.sx, r.sy, r.txy)
    H, W = mask.shape
    return {"theta": th.reshape(H, W), "mag": np.where(mask, mag.reshape(H, W), 0.0)}


def orientation(field: dict, window: int) -> np.ndarray:
    """True where beads should run along y (90 degrees): smoothed sign of mag^2 cos(2 theta)."""
    v = field["mag"] ** 2 * np.cos(2 * field["theta"])
    v = ndimage.uniform_filter(v, size=max(int(window), 1), mode="constant")
    return v < 0


def weight(field: dict, mask: np.ndarray) -> np.ndarray:
    """Relative stress per cell (median of the loaded cells = 1)."""
    mag = field["mag"]
    ref = float(np.median(mag[mask & (mag > 0)])) if (mask & (mag > 0)).any() else 1.0
    return np.where(mask, mag / max(ref, 1e-12), 0.0)


def orientation4(field: dict, window: int, min_cells: int = 0) -> np.ndarray:
    """Bead class per cell, the nearest of 0, 45, 90, 135 degrees to the smoothed
    stress-weighted principal direction (doubled-angle average). Diagonal patches smaller
    than `min_cells` fall back to the nearer of 0 and 90 degrees: small diagonal slivers at
    transitions cost far more in short strokes and seams than they gain."""
    m2 = field["mag"] ** 2
    C = ndimage.uniform_filter(m2 * np.cos(2 * field["theta"]), size=max(int(window), 1), mode="constant")
    S = ndimage.uniform_filter(m2 * np.sin(2 * field["theta"]), size=max(int(window), 1), mode="constant")
    phi = 0.5 * np.arctan2(S, C)                    # -90..90 degrees
    deg = np.mod(np.degrees(phi), 180.0)
    cls = (np.round(deg / 45.0).astype(int) % 4) * 45
    if min_cells > 0:
        axis = np.where(C < 0, 90, 0)
        for d in (45, 135):
            lab, n = ndimage.label(cls == d)
            if n:
                size = np.bincount(lab.reshape(-1))
                small = (size < min_cells)[lab] & (lab > 0)
                cls = np.where(small, axis, cls)
    return cls
