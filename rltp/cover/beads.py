"""Fine bead grid: cells smaller than a bead, beads placed anywhere on the cell grid.

With `cell` = bead_width / k (e.g. 0.1 mm cells, 0.4 mm beads, k = 4), a layer's
`mask` holds bead *centerline* nodes and its links are 0.1 mm steps, so a bead's line
can sit at any 0.1 mm position and jog sideways. The material is derived:

    D[cell] = (number of centerline nodes whose k x k footprint covers the cell) / k

as laid by the nozzle: every 0.1 mm step (link) deposits a strip one cell long and k
cells wide, every path end a half-bead cap (k/2 cells), every 90-degree corner its
outer k/2 x k/2 quarter, and a lone node a k x k dab. A cell inside a straight bead
gets exactly 1 (ends included), an empty cell 0, a cell under two beads 2; inner
corners count twice over a small square. The total matches the G-code's extrusion
(length x width per step plus half-width caps). k must be even: nodes sit on cell
corners, so the layer's path frame is shifted by half a cell and path coordinates
(G-code, timing) are exactly the bead centres; `mframe` is the material grid's frame,
used by the FEA, loads and interfaces.

A cell counts as filled when D >= 0.5. The FEA takes stiffness min(D, 1) per cell and
the bead direction averaged over the covering nodes. Required cells (walls, solid
skin) with D < 1 are gaps and any D > 1 is over-extrusion; the "cover" score term
charges both, since an exact cover by k-wide beads is usually impossible.

Edits (see ops.py): shift a straight run sideways by one cell, extend or retract a
path end, join two nearby ends with an L-shaped connector, refill a window with beads
k cells apart on the row (column) the edit picks, clear a window's optional beads, or
copy the layer below into a window. Reverse, split and move_seam work as before.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from rltp.cover.baselines import link_ring, zigzag_links
from rltp.cover.cover import (
    adjacent, bead_weights, fix_seams_local, is_end, join_ends, link, neighbors4, normalize_seams, stroke_cells,
    unlink_cell,
)
from rltp.model import Frame, Layer

WINDOW = 16          # fine-edit window side, cells


class FineEditError(Exception):
    pass


def offsets(k: int) -> tuple[int, int]:
    """Footprint rows (and columns) relative to its node: r + lo .. r + hi."""
    return -((k - 1) // 2), k // 2


def box_sum(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """S[r, c] = sum of a over rows r+lo..r+hi and columns c+lo..c+hi (zero outside)."""
    H, W = a.shape
    ii = np.zeros((H + 1, W + 1))
    ii[1:, 1:] = np.asarray(a, float).cumsum(0).cumsum(1)
    r = np.arange(H)[:, None]
    c = np.arange(W)[None, :]
    r0, r1 = np.clip(r + lo, 0, H), np.clip(r + hi + 1, 0, H)
    c0, c1 = np.clip(c + lo, 0, W), np.clip(c + hi + 1, 0, W)
    return ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]


def cover_sum(nodes: np.ndarray, k: int) -> np.ndarray:
    """Per cell: sum of node values whose footprint covers it."""
    lo, hi = offsets(k)
    return box_sum(nodes, -hi, -lo)


DIAG_MIN = 6         # alternating one-cell steps (same signs) that make a diagonal run


def runs(L: Layer) -> dict:
    """Diagonal runs and polylines of a fine layer (cached).

    A diagonal run is a stretch of at least DIAG_MIN links alternating one horizontal
    and one vertical step with constant signs: a 45 (x and y rising together) or 135
    degree bead drawn as a staircase. Returns {"diag": [(a, b, cls, a_end, b_end)],
    "xr": node has its link to the right inside a diagonal run, "xd": same for its link
    down, "dnode": node touches a diagonal link, "poly": {piece: (vertex nodes, segment
    of each link)}} where a polyline merges straight axis runs and each diagonal run into
    single segments (what the G-code and the time model print)."""
    R = L._cache.get("runs")
    if R is not None:
        return R
    from rltp.cover.cover import get_pieces
    W, n = L.W, L.n
    pc = get_pieces(L)
    xr = np.zeros(n, bool)
    xd = np.zeros(n, bool)
    dnode = np.zeros(n, bool)
    diag, poly = [], {}
    for pid, cells in enumerate(pc.cells):
        seq = [int(x) for x in cells]
        loop = bool(pc.is_loop[pid])
        full = seq + [seq[0]] if loop and len(seq) > 1 else seq
        st = np.diff(full) if len(full) > 1 else np.zeros(0, np.int64)
        typ = np.where(np.abs(st) == 1, 0, 1)
        sg = np.sign(st)
        in_run = np.full(len(st), -1)
        i = 0
        while i < len(st):
            j, hs, vs = i, 0, 0
            while j < len(st):
                if j > i and typ[j] == typ[j - 1]:
                    break
                if typ[j] == 0:
                    if hs == 0:
                        hs = sg[j]
                    elif sg[j] != hs:
                        break
                else:
                    if vs == 0:
                        vs = sg[j]
                    elif sg[j] != vs:
                        break
                j += 1
            if j - i >= DIAG_MIN and hs != 0 and vs != 0:
                in_run[i:j] = len(diag)
                a, b = full[i], full[j]
                ends = (not loop) and (i == 0), (not loop) and (j == len(st))
                diag.append((a, b, 45 if hs * vs > 0 else 135, ends[0], ends[1]))
                for q in range(i, j):
                    x, y = full[q], full[q + 1]
                    lo_, d = min(x, y), abs(y - x)
                    (xr if d == 1 else xd)[lo_] = True
                    dnode[x] = dnode[y] = True
                i = j
            else:
                i += 1
        # polyline: a vertex wherever the segment (axis direction or diagonal run) changes
        key = [("d", int(in_run[q])) if in_run[q] >= 0 else ("a", int(st[q])) for q in range(len(st))]
        verts = [full[0]]
        link_seg = np.zeros(len(st), np.int64)
        sid = 0
        for q in range(len(st)):
            if q > 0 and key[q] != key[q - 1]:
                verts.append(full[q])
                sid += 1
            link_seg[q] = sid
        if len(st):
            verts.append(full[-1])
        poly[pid] = (np.asarray(verts, np.int64), link_seg)
    R = {"diag": diag, "xr": xr, "xd": xd, "dnode": dnode, "poly": poly}
    L._cache["runs"] = R
    return R


def _spread(mark: np.ndarray, lo: int, hi: int, axis: int) -> np.ndarray:
    """Spread each mark across the bead: a mark in row r (axis 0) lands on rows r+lo..r+hi."""
    H, W = mark.shape
    if axis == 0:
        cs = np.zeros((H + 1, W))
        cs[1:] = mark.cumsum(0)
        r = np.arange(H)
        r0, r1 = np.clip(r - hi, 0, H), np.clip(r - lo + 1, 0, H)
        return cs[r1] - cs[r0]
    cs = np.zeros((H, W + 1))
    cs[:, 1:] = mark.cumsum(1)
    c = np.arange(W)
    c0, c1 = np.clip(c - hi, 0, W), np.clip(c - lo + 1, 0, W)
    return cs[:, c1] - cs[:, c0]


def _deposit(L: Layer) -> tuple[np.ndarray, np.ndarray]:
    """(material per cell, the part of it laid along x) from links, end caps, corners, dabs,
    and rotated rectangles for diagonal runs; the 4 direction shares go to the cache."""
    k, H, W = L.kb, L.H, L.W
    if k % 2:
        raise ValueError("the fine bead grid needs an even number of cells per bead")
    lo, hi = offsets(k)
    h = k // 2
    m = L.flat_mask
    i = np.arange(L.n)
    r, c = np.divmod(i, W)
    nxt, prv = L.nxt, L.prv
    R = runs(L)
    xr, xd, dnode = R["xr"], R["xd"], R["dnode"]
    right = m & (c < W - 1) & ((nxt == i + 1) | (prv == i + 1)) & ~xr
    down = m & (r < H - 1) & ((nxt == i + W) | (prv == i + W)) & ~xd
    left = m & (c > 0) & ((nxt == i - 1) | (prv == i - 1)) & ~np.roll(xr, 1)
    up = m & (r > 0) & ((nxt == i - W) | (prv == i - W)) & ~np.roll(xd, W)
    hmark = np.zeros((H, W))
    vmark = np.zeros((H, W))
    # links: node (r, c) sits on the corner after cell (r, c); the step to (r, c+1) lays
    # cell column c+1, k cells across (rows r+lo..r+hi)
    rr, cc = np.divmod(np.flatnonzero(right), W)
    np.add.at(hmark, (rr, cc + 1), 1.0)
    rr, cc = np.divmod(np.flatnonzero(down), W)
    np.add.at(vmark, (rr + 1, cc), 1.0)
    deg = right.astype(int) + down + left + up
    # end caps: k/2 cells beyond the end node, away from its one neighbour
    for side, horizontal, outward in ((left, True, +1), (right, True, -1), (up, False, +1), (down, False, -1)):
        e = np.flatnonzero(m & (deg == 1) & side & ~dnode)
        er, ec = np.divmod(e, W)
        for j in range(h):
            off = j + 1 if outward > 0 else -j       # cells c+1..c+h, or c-h+1..c
            tr = er if horizontal else er + off
            tc = ec + off if horizontal else ec
            ok = (tr >= 0) & (tr < H) & (tc >= 0) & (tc < W)
            np.add.at(hmark if horizontal else vmark, (tr[ok], tc[ok]), 1.0)
    D_h = _spread(hmark, lo, hi, axis=0)
    D_v = _spread(vmark, lo, hi, axis=1)
    # corners: the outer k/2 x k/2 quarter, opposite both links
    sq = np.zeros((H, W))
    corner = m & (deg == 2) & (left | right) & (up | down) & ~dnode
    for e in np.flatnonzero(corner).tolist():
        er, ec = divmod(e, W)
        r0 = er + 1 if up[e] else er - h + 1
        c0 = ec + 1 if left[e] else ec - h + 1
        sq[max(r0, 0):max(min(r0 + h, H), 0), max(c0, 0):max(min(c0 + h, W), 0)] += 1.0
    # lone nodes: a k x k dab
    lone = (m & (deg == 0) & ~dnode).reshape(H, W)
    dab = cover_sum(lone, k)
    # diagonal runs: a rectangle one bead wide along the straight line between the run's
    # end nodes (nodes sit on cell corners: node (r, c) at (c + 1, r + 1) in cell units),
    # extended by half a bead at path ends and a quarter at joints with other segments
    d45 = np.zeros((H, W))
    d135 = np.zeros((H, W))
    for a, b, cls, a_end, b_end in R["diag"]:
        ra, ca = divmod(a, W)
        rb, cb = divmod(b, W)
        P = np.array([ca + 1.0, ra + 1.0])
        Q = np.array([cb + 1.0, rb + 1.0])
        v = Q - P
        ln = float(np.hypot(*v))
        if ln <= 0:
            continue
        u = v / ln
        ea = h if a_end else h / 2
        eb = h if b_end else h / 2
        pad = h + 1
        x0, x1 = int(max(min(P[0], Q[0]) - pad - 1, 0)), int(min(max(P[0], Q[0]) + pad + 1, W))
        y0, y1 = int(max(min(P[1], Q[1]) - pad - 1, 0)), int(min(max(P[1], Q[1]) + pad + 1, H))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        frac = np.zeros(yy.shape)
        for oy in (1 / 6, 0.5, 5 / 6):          # 3 x 3 samples per cell: a 45 degree band
            for ox in (1 / 6, 0.5, 5 / 6):      # would otherwise alias to ~88% of its area
                dx, dy = xx + ox - P[0], yy + oy - P[1]
                along = dx * u[0] + dy * u[1]
                perp = np.abs(dx * u[1] - dy * u[0])
                frac += (along >= -ea) & (along <= ln + eb) & (perp <= h)
        (d45 if cls == 45 else d135)[y0:y1, x0:x1] += frac / 9.0
    D = D_h + D_v + sq + dab + d45 + d135
    ax = D_h + 0.5 * (sq + dab)
    L._cache["shares_raw"] = (ax, D_v + 0.5 * (sq + dab), d45, d135)
    return D, ax


def density(L: Layer) -> np.ndarray:
    D = L._cache.get("density")
    if D is None:
        D, ax = _deposit(L)
        L._cache["density"] = D
        L._cache["along_x"] = ax
    return D


def spread_max(vals: np.ndarray, nodes: np.ndarray, k: int, fill: float = -np.inf) -> np.ndarray:
    """Per cell: max of `vals` over the nodes whose footprint covers it."""
    lo, hi = offsets(k)
    H, W = nodes.shape
    v = np.where(nodes, vals, fill)
    out = np.full((H, W), fill)
    for dr in range(lo, hi + 1):
        for dc in range(lo, hi + 1):
            rs = slice(max(dr, 0), H + min(dr, 0))
            rd = slice(max(-dr, 0), H + min(-dr, 0))
            cs = slice(max(dc, 0), W + min(dc, 0))
            cd = slice(max(-dc, 0), W + min(-dc, 0))
            out[rs, cs] = np.maximum(out[rs, cs], v[rd, cd])
    return out


def node_frame(mframe: Frame, k: int) -> Frame:
    """Path frame: for even k, bead centres sit on cell corners, half a cell off."""
    if k % 2:
        return mframe
    p = mframe.p
    x, y = mframe.to_world(0.5 * p, 0.5 * p)
    return Frame(float(x), float(y), mframe.theta, p)


def cell_shares(L: Layer) -> np.ndarray | None:
    """(n, 4) shares of each cell's material laid at 0, 45, 90, 135 degrees, or None when
    the layer has no diagonal runs (then cell_wh says it all)."""
    D = density(L)
    if not runs(L)["diag"]:
        return None
    s0, s90, s45, s135 = L._cache["shares_raw"]
    Dz = np.where(D > 0, D, 1.0)
    out = np.stack([s0 / Dz, s45 / Dz, s90 / Dz, s135 / Dz], -1).reshape(-1, 4)
    out[D.reshape(-1) <= 0] = (0.5, 0.0, 0.5, 0.0)
    return out


def cell_wh(L: Layer) -> np.ndarray:
    """Per material cell: share of its material laid along x (0.5 where empty)."""
    D = density(L)
    ax = L._cache["along_x"]
    return np.where(D > 0, ax / np.where(D > 0, D, 1.0), 0.5).reshape(-1)


def cell_fields(L: Layer, timing) -> dict:
    """Per material cell: density, bead direction (fraction along x), latest print time,
    and whether a path end or seam lies over it."""
    k = L.kb
    D = density(L)
    nodes = L.mask
    whc = cell_wh(L)
    t = np.nan_to_num(timing.t_cell.reshape(L.H, L.W), nan=0.0)
    tm = spread_max(t, nodes, k, 0.0)
    from rltp.cover.cover import end_mask, seam_mask
    ends = (end_mask(L) | seam_mask(L)).reshape(L.H, L.W)
    em = cover_sum(ends & nodes, k) > 0
    return {"D": D, "wh": whc, "t": tm.reshape(-1), "ends": em.reshape(-1)}


# ------------------------------------------------------------------ lines at 0, 45, 90, 135 degrees

def line_set(sel: np.ndarray, cls: int, k: int, anchor: tuple, dens: float, spacing: int,
             fill_map: np.ndarray | None = None) -> np.ndarray:
    """Nodes of `sel` on parallel bead lines of class `cls` degrees, one bead apart
    (perpendicular), through `anchor` (r, c); with dens < 1 a share of the lines picked by
    error diffusion (never more than `spacing` lines apart), or with `fill_map` each line
    laid wherever the field exceeds its dither threshold (shell.dither_ranks). A 45-degree
    line is the staircase c - r in {d, d + 1}; lines sit k*sqrt(2) apart in d (alternating
    6 and 5 for k = 4), so their perpendicular spacing averages exactly one bead."""
    from rltp.cover.shell import dither_ranks, line_picks
    H, W = sel.shape
    if not sel.any():
        return np.zeros_like(sel)
    rr, cc = np.mgrid[0:H, 0:W]
    ar, ac = anchor
    if cls in (0, 90):
        key = rr if cls == 0 else cc
        a0 = ar if cls == 0 else ac
        P = np.arange(a0 % k, H if cls == 0 else W, k)
        width = 0
        j0 = 0
    else:
        key = cc - rr if cls == 45 else cc + rr
        d0 = (ac - ar) if cls == 45 else (ac + ar)
        step = k * np.sqrt(2.0)
        lo_, hi_ = int(key[sel].min()), int(key[sel].max())
        j0 = int(np.floor((lo_ - d0) / step)) - 1
        j1 = int(np.ceil((hi_ - d0) / step)) + 1
        P = np.array([d0 + int(round(j * step)) for j in range(j0, j1 + 1)])
        width = 1
    idx = np.searchsorted(P, key, "right") - 1
    ok = (idx >= 0) & (key - P[np.clip(idx, 0, len(P) - 1)] <= width)
    lid = np.where(ok, idx, -1)
    on = sel & ok
    counts = np.bincount(lid[on], minlength=len(P))
    if dens >= 1.0:
        pick = counts > 0
    elif fill_map is not None:
        rank = dither_ranks(len(P), spacing, j0)
        return on & (fill_map > rank[np.clip(lid, 0, len(P) - 1)])
    else:
        pick = line_picks(counts, dens, spacing)
    return on & pick[np.clip(lid, 0, len(P) - 1)]


def diag_links(L: Layer, nodes: np.ndarray, cls: int) -> None:
    """Link the staircase runs of 45 (cls 45) or 135 degree lines in `nodes` (all unlinked).
    A line holds cells with c - r (45) or c + r (135) in {d, d + 1}; in row order (columns
    rising for 45, falling for 135) consecutive cells are 4-adjacent wherever unbroken."""
    W = L.W
    idx = np.flatnonzero(nodes.reshape(-1))
    if idx.size == 0:
        return
    r, c = np.divmod(idx, W)
    key = c - r if cls == 45 else c + r
    by: dict = {}
    for x, kk in zip(idx.tolist(), key.tolist()):
        by.setdefault(kk, []).append(x)
    used: set = set()
    for kk in sorted(by):
        if kk in used:
            continue
        group = by[kk] + (by[kk + 1] if (kk + 1) in by else [])
        used.update((kk, kk + 1))
        g = np.asarray(group)
        gr, gc = np.divmod(g, W)
        g = g[np.lexsort((gc if cls == 45 else -gc, gr))].tolist()
        for x, y in zip(g[:-1], g[1:]):
            if adjacent(x, y, W) and L.nxt[x] == -1 and L.prv[y] == -1:
                link(L, x, y)


def link_lines(L: Layer, nodes: np.ndarray, cls: int) -> None:
    if cls in (0, 90):
        zigzag_links(L, nodes, cls)
    else:
        diag_links(L, nodes, cls)


# ------------------------------------------------------------------ starting layout

def _erode(m: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return m.copy()
    return ndimage.binary_erosion(m, np.ones((3, 3), bool), iterations=n, border_value=0)


def fine_layout(L: Layer, part: np.ndarray, solid: np.ndarray, walls: int, density_: float, angle: int,
                line_angle: int, max_spacing: int, k: int, orient: np.ndarray | None = None,
                sweight: np.ndarray | None = None, fill_map: np.ndarray | None = None) -> Layer:
    """Walls as locked loops k cells apart along the true outline, solid skin as beads k
    cells apart, infill lines picked by error diffusion over bead rows (max_spacing in
    beads apart), connected where ends meet. With `orient` (True where beads should run
    along y, or a class per cell: 0, 45, 90, 135 degrees) skin and infill follow it instead
    of `angle`/`line_angle`, and `sweight` (relative stress per cell) packs infill lines
    where stress is high. `fill_map` (per cell, the gradient layout's density field) sets
    each line's density instead of `density_`."""
    from rltp.cover.shell import close_ring, dither_ranks, line_picks, row_density
    H, W = part.shape
    lo, hi = offsets(k)
    region = box_sum(part, lo, hi) >= k * k - 0.5          # footprint inside the part
    L.kb = k
    L.part = part
    L.mask = np.zeros_like(part)
    L.nxt[:] = -1
    L.prv[:] = -1
    L.seam.clear()
    lock = np.zeros_like(part)
    E = region.copy()
    for _ in range(walls):
        if not E.any():
            break
        ring = E & ~_erode(E, 1)
        L.mask |= ring
        link_ring(L, ring.reshape(-1))
        close_ring(L, ring.reshape(-1))
        lock |= ring
        E = _erode(E, k)
    inner = E
    walls_band = cover_sum(lock, k) > 0
    req = walls_band | (solid & part)
    solid_n = inner & solid
    opt_n = inner & ~solid
    rows = np.flatnonzero(inner.any(1))
    cols = np.flatnonzero(inner.any(0))

    def lines(sel: np.ndarray, ang: int, dens: float) -> np.ndarray:
        if not sel.any():
            return np.zeros_like(sel)
        if ang == 0:
            ph = int(rows[0]) % k
            bead_rows = np.arange(ph, H, k)
            sub = sel[bead_rows]
            out = np.zeros_like(sel)
            if fill_map is not None and dens < 1.0:     # lines start and stop with the field
                out[bead_rows] = sub & (fill_map[bead_rows] > dither_ranks(len(bead_rows), max_spacing)[:, None])
                return out
            d = row_density(sub, dens, None if sweight is None else sweight[bead_rows], max_spacing, 0)
            pick = bead_rows[line_picks(sub.sum(1), d, max_spacing)]
            out[pick] = sel[pick]
        else:
            ph = int(cols[0]) % k
            bead_cols = np.arange(ph, W, k)
            sub = sel[:, bead_cols]
            out = np.zeros_like(sel)
            if fill_map is not None and dens < 1.0:
                out[:, bead_cols] = sub & (fill_map[:, bead_cols] > dither_ranks(len(bead_cols), max_spacing)[None, :])
                return out
            d = row_density(sub, dens, None if sweight is None else sweight[:, bead_cols], max_spacing, 1)
            pick = bead_cols[line_picks(sub.sum(0), d, max_spacing)]
            out[:, pick] = sel[:, pick]
        return out

    if orient is not None:                 # stress-aligned: each area runs its own way
        cls_map = np.where(orient, 90, 0) if orient.dtype == bool else orient
        anchor = (int(rows[0]), int(cols[0])) if rows.size else (0, 0)
        taken = np.zeros_like(part)
        for cls in (0, 90, 45, 135):
            area = cls_map == cls
            if not area.any():
                continue
            if cls in (0, 90):
                got = lines(solid_n & area, cls, 1.0) | lines(opt_n & area, cls, density_)
            else:
                got = line_set(solid_n & area, cls, k, anchor, 1.0, max_spacing) | \
                    line_set(opt_n & area, cls, k, anchor, density_, max_spacing, fill_map)
            got &= ~taken
            taken |= got
            L.mask |= got
            link_lines(L, got, cls)
    else:
        skin = lines(solid_n, angle, 1.0)
        infill = lines(opt_n, line_angle, density_)
        L.mask |= skin | infill
        zigzag_links(L, skin, angle)
        zigzag_links(L, infill & ~skin, line_angle)
    L.req, L.lock = req, lock
    optfoot = box_sum(req, lo, hi) == 0                    # node footprint touches no required cell
    L.aux = {"region": region, "inner": inner, "optfoot": optfoot, "max_spacing": max_spacing,
             "phase": (int(rows[0]) % k if rows.size else 0, int(cols[0]) % k if cols.size else 0)}
    L.invalidate()
    repair_fine(L)
    normalize_seams(L)
    L.invalidate()
    return L


# ------------------------------------------------------------------ connecting ends

def _line(a: int, b: int, W: int) -> list[int]:
    """Cells strictly between a and b along a row or column (inclusive of b)."""
    ra, ca = divmod(a, W)
    rb, cb = divmod(b, W)
    out = []
    if ra == rb:
        s = 1 if cb > ca else -1
        out = [ra * W + c for c in range(ca + s, cb + s, s)]
    elif ca == cb:
        s = 1 if rb > ra else -1
        out = [r * W + ca for r in range(ra + s, rb + s, s)]
    return out


def connect(L: Layer, a: int, b: int) -> list[int] | None:
    """Join path ends a and b (different strokes) through free region nodes on an L-shaped
    connector (either corner first). Returns the added nodes, or None."""
    W = L.W
    region = L.aux["region"].reshape(-1)
    m = L.flat_mask
    ra, ca = divmod(a, W)
    rb, cb = divmod(b, W)
    if adjacent(a, b, W):
        return [] if join_ends(L, a, b) else None
    for corner in ((ra, cb), (rb, ca)):
        cn = corner[0] * W + corner[1]
        path = _line(a, cn, W) if cn != a else []
        path = path + (_line(cn, b, W) if cn != b else [])
        if path and path[-1] == b:
            path = path[:-1]
        if not path or len(set(path)) != len(path):
            continue
        if any(m[x] or not region[x] for x in path):
            continue
        for x in path:
            L.mask.reshape(-1)[x] = True
        for x, y in zip(path[:-1], path[1:]):
            link(L, x, y)
        L.invalidate()
        if join_ends(L, a, path[0]):
            if join_ends(L, path[-1], b):
                L.invalidate()
                return path
            # undo the first join
            if L.nxt[a] == path[0]:
                L.nxt[a] = -1
                L.prv[path[0]] = -1
            elif L.prv[a] == path[0]:
                L.prv[a] = -1
                L.nxt[path[0]] = -1
        for x in path:
            unlink_cell(L, x)
            L.mask.reshape(-1)[x] = False
        L.invalidate()
    return None


def _ends(L: Layer) -> np.ndarray:
    m = L.flat_mask
    e = m & ((L.nxt == -1) | (L.prv == -1))
    if L.lock is not None:
        e &= ~L.lock.reshape(-1)
    return e


def repair_fine(L: Layer, cells=None, reach: int | None = None) -> int:
    """Greedily connect free path ends to the nearest end of another stroke within
    `reach` cells (default 2k) in rows and columns. Returns the number of joins."""
    k, W = L.kb, L.W
    reach = 2 * k if reach is None else reach
    joins = 0
    ef = _ends(L)
    ends = np.flatnonzero(ef)
    cand = ends if cells is None else np.array(sorted(int(c) for c in cells if ef[c]), np.int64)
    er, ec = np.divmod(ends, W)
    for x in cand.tolist():
        if not is_end(L, x) or not L.flat_mask[x]:
            continue
        rx, cx = divmod(x, W)
        near = (np.abs(er - rx) <= reach) & (np.abs(ec - cx) <= reach) & (ends != x)
        if not near.any():
            continue
        own = set(stroke_cells(L, x)[0])
        cands = ends[near]
        d = np.abs(er[near] - rx) + np.abs(ec[near] - cx)
        for y in cands[np.argsort(d, kind="stable")].tolist():
            if y in own or not L.flat_mask[y] or not is_end(L, y):
                continue
            if connect(L, x, y) is not None:
                joins += 1
                break
    L.invalidate()
    return joins


# ------------------------------------------------------------------ sequence edits

def _rebuild(N: Layer, old: list[int], new: list[int], loop: bool, prefer=()) -> None:
    """Replace stroke `old` by the node sequence `new` (validated)."""
    W = N.W
    m = N.flat_mask
    region = N.aux["region"].reshape(-1)
    olds = set(old)
    if len(set(new)) != len(new):
        raise FineEditError("path crosses itself")
    for x in new:
        if not (0 <= x < N.n) or not region[x] or (m[x] and x not in olds):
            raise FineEditError("node outside the region or on another bead")
    for a, b in zip(new[:-1], new[1:]):
        if not adjacent(a, b, W):
            raise FineEditError("path not 4-connected")
    if loop and (len(new) < 4 or not adjacent(new[-1], new[0], W)):
        raise FineEditError("loop can't close")
    for x in old:
        N.nxt[x] = -1
        N.prv[x] = -1
        N.mask.reshape(-1)[x] = False
        N.seam.discard(x)
    for x in new:
        N.mask.reshape(-1)[x] = True
    for a, b in zip(new[:-1], new[1:]):
        link(N, a, b)
    if loop:
        link(N, new[-1], new[0])
        keep = next((s for s in prefer if s in set(new)), new[0])
        N.seam.add(int(keep))
    N.invalidate()


def _cleanup(seq: list[int], W: int) -> list[int]:
    """Drop repeated nodes and back-tracks, fill straight gaps of two cells."""
    out: list[int] = []
    for x in seq:
        if out and out[-1] == x:
            continue
        if len(out) >= 2 and out[-2] == x:
            out.pop()
            continue
        if out and not adjacent(out[-1], x, W):
            ra, ca = divmod(out[-1], W)
            rb, cb = divmod(x, W)
            if (ra == rb and abs(ca - cb) == 2) or (ca == cb and abs(ra - rb) == 2):
                out.append(((ra + rb) // 2) * W + (ca + cb) // 2)
        out.append(x)
    return out


def _run(seq: list[int], i: int, step: int) -> tuple[int, int]:
    """Indices j0..j1 of the maximal straight run through seq[i] with link step |step|."""
    j0 = i
    while j0 > 0 and abs(seq[j0] - seq[j0 - 1]) == step:
        j0 -= 1
    j1 = i
    while j1 < len(seq) - 1 and abs(seq[j1 + 1] - seq[j1]) == step:
        j1 += 1
    return j0, j1


def shift(L: Layer, N: Layer, cell: int, d: int) -> dict:
    """Move the straight run through `cell` one cell sideways (d = +-1); the legs at its
    ends lengthen or shorten to follow."""
    W, H = L.W, L.H
    old, loop = stroke_cells(L, cell)
    seq = list(old)
    n = len(seq)
    i = seq.index(cell)
    if loop:
        nb = [seq[i - 1], seq[(i + 1) % n]]
    else:
        nb = ([seq[i - 1]] if i > 0 else []) + ([seq[i + 1]] if i < n - 1 else [])
    horiz = any(abs(x - cell) == 1 for x in nb) or not nb
    step = 1 if horiz else W
    if loop:                    # open the loop just after the run so the run doesn't wrap
        j = i
        while j - i < n - 1 and abs(seq[(j + 1) % n] - seq[j % n]) == step:
            j += 1
        start = (j + 1) % n
        seq = seq[start:] + seq[:start]
        i = seq.index(cell)
    j0, j1 = _run(seq, i, step)
    run = seq[j0:j1 + 1]
    if horiz:
        if not (0 <= run[0] // W + d < H):
            raise FineEditError("off the grid")
        new_run = [x + d * W for x in run]
    else:
        cs = [x % W + d for x in run]
        if min(cs) < 0 or max(cs) >= W:
            raise FineEditError("off the grid")
        new_run = [x + d for x in run]
    new = _cleanup(seq[:j0] + new_run + seq[j1 + 1:], W)
    if loop and len(new) > 2 and not adjacent(new[-1], new[0], W):
        new = _cleanup(new + [new[0]], W)[:-1]      # fill a 2-cell gap across the wrap
    _rebuild(N, old, new, loop, prefer=sorted(L.seam))
    return {"run": [int(run[0]), int(run[-1])], "d": d}


def extend(L: Layer, N: Layer, cell: int) -> dict:
    W, H = L.W, L.H
    seq, loop = stroke_cells(N, cell)
    if loop or not is_end(L, cell):
        raise FineEditError("not a path end")
    at_head = seq[0] == cell
    if len(seq) == 1:
        nxt = cell + 1 if (cell % W) + 1 < W else cell - 1
    else:
        nb = seq[1] if at_head else seq[-2]
        dd = cell - nb
        if abs(dd) == 1 and not (0 <= cell % W + dd < W):
            raise FineEditError("off the grid")
        nxt = cell + dd
    new = [nxt] + seq if at_head else seq + [nxt]
    _rebuild(N, seq, new, False)
    return {"added": int(nxt)}


def retract(L: Layer, N: Layer, cell: int) -> dict:
    seq, loop = stroke_cells(N, cell)
    if loop or not is_end(L, cell):
        raise FineEditError("not a path end")
    new = seq[1:] if seq[0] == cell else seq[:-1]
    if new:
        _rebuild(N, seq, new, False)
    else:
        N.mask.reshape(-1)[cell] = False
        N.invalidate()
    return {"removed": int(cell)}


def join(L: Layer, N: Layer, cell: int) -> dict:
    if not is_end(L, cell) or (L.lock is not None and L.lock.reshape(-1)[cell]):
        raise FineEditError("not a free path end")
    if repair_fine(N, cells=[cell]) == 0:
        raise FineEditError("no end within reach")
    return {}


def _window(cell: int, H: int, W: int, s: int = WINDOW):
    r, c = divmod(cell, W)
    h = s // 2
    return max(r - h, 0), min(r + h, H), max(c - h, 0), min(c + h, W)


def _clear_nodes(N: Layer, gone: np.ndarray) -> set:
    touched: set = set()
    for x in np.flatnonzero(gone.reshape(-1)).tolist():
        touched.update(unlink_cell(N, x))
        N.seam.discard(x)
    N.mask[gone] = False
    N.invalidate()
    return touched - set(np.flatnonzero(gone.reshape(-1)).tolist())


def _relay(L: Layer, N: Layer, area: np.ndarray, angle: int, cell: int) -> None:
    """Re-lay the free beads over `area` (node positions) k cells apart, through `cell`'s row
    (angle 0) or column (90): every line over required cells, and over optional cells the
    share of lines that keeps the area's current infill share."""
    from rltp.cover.shell import line_picks
    H, W, k = L.H, L.W, L.kb
    lock = L.lock if L.lock is not None else np.zeros_like(area)
    opt_cells = area & L.part & ~L.req
    dens = float((L.material & opt_cells).sum()) / max(int(opt_cells.sum()), 1)
    touched = _clear_nodes(N, area & N.mask & ~lock)
    r, c = divmod(cell, W)
    if angle in (45, 135):                 # diagonal lines through the cell, keeping the share
        cand = area & N.aux["inner"] & ~N.mask
        spacing = int(N.aux.get("max_spacing", 1 << 30))
        new = line_set(cand & L.req, angle, k, (r, c), 1.0, spacing) | \
            line_set(cand & ~L.req, angle, k, (r, c), dens, spacing)
        if not new.any():
            raise FineEditError("nothing to lay")
        N.mask |= new
        N.invalidate()
        diag_links(N, new, angle)
        ends = set(np.flatnonzero(new.reshape(-1) & _ends(N)).tolist()) | {x for x in touched if N.flat_mask[x]}
        repair_fine(N, cells=ends)
        fix_seams_local(N, ends | touched, prefer=sorted(L.seam))
        return
    rr, cc = np.mgrid[0:H, 0:W]
    phase = (rr % k == r % k) if angle == 0 else (cc % k == c % k)
    cand = area & N.aux["inner"] & phase & ~N.mask
    req_n = cand & L.req
    opt_n = cand & ~L.req
    if opt_n.any() and dens < 1.0:
        spacing = int(N.aux.get("max_spacing", 1 << 30))
        if angle == 0:
            bead = np.arange(r % k, H, k)
            pick = bead[line_picks(opt_n[bead].sum(1), dens, spacing)]
            keep = np.zeros_like(opt_n)
            keep[pick] = True
        else:
            bead = np.arange(c % k, W, k)
            pick = bead[line_picks(opt_n[:, bead].sum(0), dens, spacing)]
            keep = np.zeros_like(opt_n)
            keep[:, pick] = True
        opt_n &= keep
    new = req_n | opt_n
    if not new.any():
        raise FineEditError("nothing to lay")
    N.mask |= new
    N.invalidate()
    zigzag_links(N, new, angle)
    ends = set(np.flatnonzero(new.reshape(-1) & _ends(N)).tolist()) | {x for x in touched if N.flat_mask[x]}
    repair_fine(N, cells=ends)
    fix_seams_local(N, ends | touched, prefer=sorted(L.seam))


def refill(L: Layer, N: Layer, cell: int, angle: int, size: int = WINDOW) -> dict:
    """Re-lay a size x size window at 0 or 90 degrees, keeping its infill share."""
    H, W = L.H, L.W
    r0, r1, c0, c1 = _window(cell, H, W, size)
    win = np.zeros((H, W), bool)
    win[r0:r1, c0:c1] = True
    _relay(L, N, win, angle, cell)
    return {"window": [r0, c0, r1, c1]}


def bead_class(L: Layer) -> np.ndarray:
    """Per cell: dominant bead direction (0, 45, 90, 135; -1 where empty or tied)."""
    D = density(L)
    sh = cell_shares(L)
    if sh is None:
        wh = np.where(D > 0, L._cache["along_x"] / np.where(D > 0, D, 1.0), 0.5)
        out = np.where(wh > 0.5, 0, np.where(wh < 0.5, 90, -1))
    else:
        sh = sh.reshape(L.H, L.W, 4)
        top = np.argmax(sh, -1)
        tie = np.sort(sh, -1)[..., -1] - np.sort(sh, -1)[..., -2] < 1e-9
        out = np.where(tie, -1, np.array([0, 45, 90, 135])[top])
    return np.where(D >= 0.5, out, -1)


def flip_region(L: Layer, N: Layer, cell: int, turn: int = 90) -> dict:
    """Turn the connected area of same-direction beads around `cell` (walls excluded) by
    `turn` degrees (90: 0 <-> 90, 45 <-> 135; 45: one class onward) in one move, keeping
    its infill share."""
    H, W, k = L.H, L.W, L.kb
    cls = bead_class(L)
    lock = L.lock if L.lock is not None else np.zeros_like(L.mask)
    walls = cover_sum(lock, k) > 0
    r, c = divmod(cell, W)
    if walls[r, c] or cls[r, c] < 0:
        raise FineEditError("no single-direction bead here")
    same = (cls == cls[r, c]) & ~walls
    lab, _ = ndimage.label(same)
    area = lab == lab[r, c]
    to = int((cls[r, c] + turn) % 180)
    _relay(L, N, area, to, cell)
    return {"cells": int(area.sum()), "to": to}


def clear(L: Layer, N: Layer, cell: int) -> dict:
    """Remove the window's beads that only cover optional cells."""
    H, W = L.H, L.W
    r0, r1, c0, c1 = _window(cell, H, W)
    win = np.zeros((H, W), bool)
    win[r0:r1, c0:c1] = True
    lock = L.lock if L.lock is not None else np.zeros_like(win)
    gone = win & N.mask & ~lock & N.aux["optfoot"]
    if not gone.any():
        raise FineEditError("nothing to clear")
    touched = _clear_nodes(N, gone)
    touched = {x for x in touched if N.flat_mask[x]}
    repair_fine(N, cells=touched)
    fix_seams_local(N, touched, prefer=sorted(L.seam))
    return {"window": [r0, c0, r1, c1]}


def stack(L: Layer, N: Layer, cell: int) -> dict:
    """Copy the layer below's free beads (and their links) into the window."""
    below = L.ctx.get("below_layer") if L.ctx is not None else None
    if below is None:
        raise FineEditError("no layer below")
    H, W = L.H, L.W
    r0, r1, c0, c1 = _window(cell, H, W)
    win = np.zeros((H, W), bool)
    win[r0:r1, c0:c1] = True
    lock = L.lock if L.lock is not None else np.zeros_like(win)
    blk = below.lock if below.lock is not None else np.zeros_like(win)
    want = win & below.mask & ~blk & N.aux["inner"]
    if (want == (win & N.mask & ~lock & N.aux["inner"])).all():
        raise FineEditError("window already stacked")
    touched = _clear_nodes(N, win & N.mask & ~lock)
    want &= ~N.mask
    N.mask |= want
    N.invalidate()
    wf = want.reshape(-1)
    for x in np.flatnonzero(wf).tolist():
        y = int(below.nxt[x])
        if y >= 0 and wf[y] and N.nxt[x] == -1 and N.prv[y] == -1:
            link(N, x, y)
    N.invalidate()
    ends = set(np.flatnonzero(wf & _ends(N)).tolist()) | {x for x in touched if N.flat_mask[x]}
    repair_fine(N, cells=ends)
    fix_seams_local(N, ends | touched, prefer=sorted(L.seam))
    return {"window": [r0, c0, r1, c1]}


# ------------------------------------------------------------------ masks

def fine_masks(L: Layer) -> dict:
    """Valid cells for the fine edits (cheap, slightly permissive; apply() is exact)."""
    H, W, k = L.H, L.W, L.kb
    m = L.mask
    lock = L.lock if L.lock is not None else np.zeros_like(m)
    free = m & ~lock
    ends = _ends(L).reshape(H, W)
    reach = 2 * k
    near = box_sum(ends, -reach, reach) >= 2
    h = WINDOW // 2
    inner = L.aux["inner"]
    out = {
        "join": ends & near,
        "shift_pos": free, "shift_neg": free,
        "extend": ends, "retract": ends,
        "refill_0": box_sum(inner, -h, h - 1) > 0,
        "clear_w": box_sum(free & L.aux["optfoot"], -h, h - 1) > 0,
    }
    out["refill_90"] = out["refill_0"]
    out["refill32_0"] = out["refill32_90"] = box_sum(inner, -16, 15) > 0
    out["refill64_0"] = out["refill64_90"] = box_sum(inner, -32, 31) > 0
    out["flip_region"] = out["turn_region"] = free
    out["refill_45"] = out["refill_135"] = out["refill_0"]
    out["refill32_45"] = out["refill32_135"] = out["refill32_0"]
    below = L.ctx.get("below_layer") if L.ctx is not None else None
    if below is not None:
        diff = (below.mask ^ m) & inner
        out["stack_w"] = box_sum(diff, -h, h - 1) > 0
    else:
        out["stack_w"] = np.zeros_like(m)
    return out


FINE_OPS = ("join", "shift_pos", "shift_neg", "extend", "retract", "refill_0", "refill_90", "clear_w", "stack_w",
            "refill32_0", "refill32_90", "refill64_0", "refill64_90", "flip_region",
            "refill_45", "refill_135", "refill32_45", "refill32_135", "turn_region")


def apply_fine(L: Layer, N: Layer, name: str, cell: int) -> dict:
    if name == "join":
        return join(L, N, cell)
    if name in ("shift_pos", "shift_neg"):
        if L.lock is not None and L.lock.reshape(-1)[cell]:
            raise FineEditError("wall")
        if not L.flat_mask[cell]:
            raise FineEditError("not a bead node")
        return shift(L, N, cell, 1 if name == "shift_pos" else -1)
    if name == "extend":
        return extend(L, N, cell)
    if name == "retract":
        return retract(L, N, cell)
    if name.startswith("refill"):
        head, ang = name.split("_")
        size = {"refill": WINDOW, "refill32": 32, "refill64": 64}[head]
        return refill(L, N, cell, int(ang), size)
    if name == "flip_region":
        return flip_region(L, N, cell, 90)
    if name == "turn_region":
        return flip_region(L, N, cell, 45)
    if name == "clear_w":
        return clear(L, N, cell)
    if name == "stack_w":
        return stack(L, N, cell)
    raise FineEditError(f"unknown fine edit {name}")
