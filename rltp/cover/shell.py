"""Sparse layers: fixed walls, solid top/bottom skins, optional infill, and the support rule.

Every layer gets `walls` rings of cells along its outline (outer contour and holes),
traced into loops that edits never re-route (only their seams move). A cell is solid
when it lies within `solid` layers of an upward- or downward-facing surface, as
slicers do it: any of the `solid` layers above or below lacks that cell. Walls and
solid cells must stay filled. Every other in-part cell is optional; the starting
cover fills rectilinear lines at the requested density, alternating 0/90 degrees
layer by layer like the skins.

Support rule: a filled cell must lie within `radius` cells (Euclidean) of a filled
cell in the layer below, unless the layer below has no part there at all (a
geometric overhang, which only supports could fix). With radius = max_bridge / 2,
that allows bridges up to max_bridge between infill lines. Edits keep the rule both
ways: a new cell must be supported, and removing cells must not strand any filled
cell in the layer above. Sparse layers share one grid (no per-layer rotation), so
"the cell below" is the same grid cell.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from rltp.cover.baselines import link_ring, repair, rings, zigzag_links
from rltp.cover.cover import adjacent, is_end, join_ends, link, linked, neighbors4, normalize_seams, stroke_cells
from rltp.model import Layer

log = logging.getLogger(__name__)


def disk(radius: int) -> np.ndarray:
    r = max(int(radius), 0)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1e-9


def support_radius(max_bridge: float, p: float) -> int:
    """Cells a filled cell may sit from its nearest support below (half the bridge span)."""
    return max(int(np.floor(max_bridge / 2 / p + 1e-9)), 1)


def line_density(infill: float, radius: int) -> float:
    """Starting line density: the target, but no lower than one line per 2 * radius cells,
    so every cell above an infill layer is within `radius` of a line."""
    return float(np.clip(infill, 1.0 / (2 * radius), 1.0))


def line_picks(counts: np.ndarray, density, max_spacing: int) -> np.ndarray:
    """Which rows (or columns) carry an infill line, given each one's count of optional
    cells: error diffusion, so the filled share of optional cells tracks `density` (one
    value, or one per row) on any shape, with lines never more than `max_spacing` apart
    (the support rule)."""
    counts = np.asarray(counts, float)
    dens = np.broadcast_to(np.asarray(density, float), counts.shape)
    out = np.zeros(len(counts), bool)
    owed, gap = 0.0, max_spacing
    for r, c in enumerate(counts):
        gap += 1
        if c <= 0:
            continue
        owed += dens[r] * c
        if owed >= c / 2 or gap >= max_spacing:
            out[r] = True
            owed -= c
            gap = 0
    return out


def solid_masks(parts: list[np.ndarray], n_solid: int) -> list[np.ndarray]:
    """Cells within n_solid layers of an up- or down-facing surface (layers past the
    ends of the part count as empty, so the first and last n_solid layers are solid)."""
    K = len(parts)
    out = []
    for k in range(K):
        covered_above = np.ones_like(parts[k])
        covered_below = np.ones_like(parts[k])
        for j in range(1, n_solid + 1):
            covered_above &= parts[k + j] if k + j < K else False
            covered_below &= parts[k - j] if k - j >= 0 else False
        out.append(parts[k] & ~(covered_above & covered_below))
    return out


def row_density(sel: np.ndarray, density: float, sweight: np.ndarray | None, max_spacing: int,
                axis: int) -> np.ndarray | float:
    """Per-row (axis 0) or per-column infill density: `density` scaled by the rows' mean
    stress (square-rooted, so quiet rows aren't starved), keeping the overall share."""
    if sweight is None or density >= 1.0:
        return density
    c = sel.sum(1 - axis) if axis == 0 else sel.sum(0)
    wsum = (np.where(sel, np.sqrt(np.maximum(sweight, 0.0)), 0.0)).sum(1 - axis if axis == 0 else 0)
    w = np.where(c > 0, wsum / np.maximum(c, 1), 0.0)
    norm = (c * w).sum() / max(c.sum(), 1)
    if norm <= 0:
        return density
    return np.clip(density * w / norm, 1.0 / max(max_spacing, 1), 1.0)


def dither_ranks(n: int, spacing: int, start: int = 0) -> np.ndarray:
    """Thresholds for laying a density field out as lines (rows, columns or diagonals,
    in order): a line is laid wherever the field exceeds its threshold, so lines start
    and stop with the density. Every `spacing`-th line has threshold -1 and is always laid
    (the support rule's sparsest lines); the others get (i + 0.5) / spacing in an order
    that keeps them spread out, so the laid lines nest as the density rises and a line
    runs on wherever its surroundings are dense enough. A field value d lays a share of
    lines within 1 / (2 spacing) of d, never below 1 / spacing. `start`: index of the
    first line (lines are counted from a fixed origin, so neighbouring areas agree)."""
    S = max(int(spacing), 1)
    order = [0]
    while len(order) < S:
        rest = [o for o in range(S) if o not in order]
        order.append(max(rest, key=lambda o: min(min(abs(o - q), S - abs(o - q)) for q in order)))
    rank = np.empty(S)
    rank[0] = -1.0
    for i, o in enumerate(order[1:], start=1):
        rank[o] = (i + 0.5) / S
    return rank[(start + np.arange(n)) % S]


def fill_sparse(L: Layer, part: np.ndarray, solid: np.ndarray, walls: int, density: float,
                angle: int, max_spacing: int = 1 << 30, line_angle: int | None = None,
                orient: np.ndarray | None = None, sweight: np.ndarray | None = None,
                fill_map: np.ndarray | None = None) -> Layer:
    """Set up a sparse layer: walls as locked loops, solid cells, and infill lines along rows
    (angle 0) or columns (90) holding a share `density` of the optional cells, at most
    `max_spacing` apart; the filled cells are linked as a zigzag, then repaired. Infill lines
    run at `line_angle` (default `angle`); solid cells zigzag at `angle`. `fill_map` (per
    cell, the gradient layout's density field) lays lines where it exceeds each line's
    dither threshold instead (dither_ranks)."""
    H, W = part.shape
    wall = np.zeros_like(part)
    L.mask = part.copy()
    L.nxt[:] = -1
    L.prv[:] = -1
    L.seam.clear()
    for ring in rings(part, walls):
        link_ring(L, ring.reshape(-1))
        close_ring(L, ring.reshape(-1))
        wall |= ring
    req = wall | (solid & part)
    opt = part & ~req
    line_angle = angle if line_angle is None else line_angle
    def pick(sel, axis):
        """Cells of `sel` on infill lines: rows (axis 0) or columns (axis 1)."""
        if fill_map is not None and density < 1.0:
            if axis == 0:
                return sel & (fill_map > dither_ranks(H, max_spacing)[:, None])
            return sel & (fill_map > dither_ranks(W, max_spacing)[None, :])
        d = row_density(sel, density, sweight, max_spacing, axis)
        if axis == 0:
            return np.repeat(line_picks(sel.sum(1), d, max_spacing)[:, None], W, 1) & sel
        return np.repeat(line_picks(sel.sum(0), d, max_spacing)[None, :], H, 0) & sel

    if orient is not None:                  # stress-aligned: rows where x-loaded, columns where y
        orient = orient == 90 if orient.dtype != bool else orient
        lines = pick(opt & ~orient, 0) | pick(opt & orient, 1)
    else:
        lines = pick(opt, 0 if line_angle == 0 else 1)
    fill = req | (part & lines)
    L.mask = fill
    L.part, L.req, L.lock = part, req, wall
    if orient is not None:
        zigzag_links(L, fill & ~wall & ~orient, 0)
        zigzag_links(L, fill & ~wall & orient, 90)
    elif line_angle == angle:
        zigzag_links(L, fill & ~wall, angle)
    else:                                   # skins and infill lines run different ways
        zigzag_links(L, fill & req & ~wall, angle)
        zigzag_links(L, fill & ~req & ~wall, line_angle)
    repair(L)
    normalize_seams(L)
    L.invalidate()
    return L


def close_ring(L: Layer, ring: np.ndarray) -> None:
    """Join the open walks a ring was traced into wherever their ends touch (only
    within the ring), then close each path whose head and tail are adjacent."""
    H, W = L.H, L.W
    cells = np.flatnonzero(ring).tolist()
    changed = True
    while changed:
        changed = False
        for x in cells:
            if not is_end(L, x):
                continue
            for y in neighbors4(x, H, W):
                if ring[y] and is_end(L, y) and not linked(L, x, y):
                    sx, loop = stroke_cells(L, x)
                    if loop or y in set(sx):
                        continue
                    if join_ends(L, x, y):
                        changed = True
                        break
    for x in cells:
        if L.prv[x] == -1 and L.nxt[x] != -1:          # a path head
            sx, _ = stroke_cells(L, x)
            if len(sx) >= 4 and adjacent(sx[-1], sx[0], W):
                link(L, sx[-1], sx[0])
                L.seam.add(int(sx[0]))
    L.invalidate()


# ------------------------------------------------------------------ support rule

def dilate(m: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Binary dilation by a disk; large disks (fine grids) via a distance transform."""
    if structure.shape[0] <= 13:
        return ndimage.binary_dilation(m, structure)
    if not m.any():
        return np.zeros_like(m)
    R = (structure.shape[0] - 1) / 2
    return ndimage.distance_transform_edt(~m) <= R + 1e-9


def support_ctx(layers: list[Layer], k: int, structure: np.ndarray) -> dict:
    """What layer k's edits must respect from its neighbours.

    supp:  cells layer k may fill (supported from below, or no part below to bridge from).
    above: filled cells of layer k+1 that layer k can support (they need it to stay so)."""
    L = layers[k]
    if k == 0:
        supp = np.ones_like(L.part_mask)
    else:
        below = layers[k - 1]
        supp = dilate(below.material, structure) | ~dilate(below.part_mask, structure)
    above = None
    if k + 1 < len(layers):
        above = layers[k + 1].material & dilate(L.part_mask, structure)
    below = layers[k - 1].material if k > 0 else None
    return {"supp": supp, "above": above, "below": below, "structure": structure,
            "below_layer": layers[k - 1] if k > 0 else None}


def critical(L: Layer) -> np.ndarray:
    """Filled cells of L that are the only support of some filled cell above."""
    ctx = L.ctx
    if ctx is None or ctx["above"] is None or not ctx["above"].any():
        return np.zeros_like(L.mask)
    st = ctx["structure"]
    count = ndimage.convolve(L.mask.astype(np.int32), st.astype(np.int32), mode="constant")
    lone = ctx["above"] & (count == 1)
    if not lone.any():
        return np.zeros_like(L.mask)
    return ndimage.binary_dilation(lone, st) & L.mask


def support_violation(old: Layer, new: Layer) -> str | None:
    """Why `new` (an edit of `old`) breaks the support rule, or None. Only changes are
    judged: cells that were already unsupported before the edit don't block it."""
    ctx = new.ctx
    if ctx is None:
        return None
    added = new.material & ~old.material
    if added.any() and (added & ~ctx["supp"]).any():
        return "new cell has no support below"
    lost = old.material & ~new.material
    if lost.any() and ctx["above"] is not None and ctx["above"].any():
        st = ctx["structure"]
        before = ctx["above"] & ~dilate(old.material, st)
        after = ctx["above"] & ~dilate(new.material, st)
        if (after & ~before).any():
            return "removal strands a cell in the layer above"
    return None


def unsupported(layers: list[Layer], structure: np.ndarray) -> list[int]:
    """Count of filled cells breaking the support rule, per layer (0 for a valid part)."""
    out = []
    for k, L in enumerate(layers):
        if k == 0:
            out.append(0)
            continue
        ctx = support_ctx(layers, k, structure)
        out.append(int((L.material & ~ctx["supp"]).sum()))
    return out
