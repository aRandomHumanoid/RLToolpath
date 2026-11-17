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
copy the layer below into a window. Reverse, split and move_seam work as before. Refills
keep a window's infill share and space its lines evenly; `pack` keeps the share but packs
the lines where one more unit of infill is worth most (a per-cell map the annealing loop
puts in L.ctx["worth"], rltp.search.guide.WorthMap), so material can gather where the score
wants it, e.g. into an edge-following band beside a wall. Without that map it is never valid.

Fit mode (fit_mode; a PartState scoring true bead directions sets it on its layers): strokes
print as straight segments fitted within FIT_TOL_MM of their nodes (_fit_runs) instead of
axis runs and 45-degree staircases, and material is deposited along those segments
(_deposit_fit: one bead wide, half-bead end caps, mitred joints), so a staircase at any angle
is one straight bead in the G-code, the time model and the material alike.

Anchoring: a path end is anchored when another bead touches it (is_anchored), so the
bead can pass load in-plane to the rest of the layer; the continuity term charges each
free end. `anchor` extends a free end along its line until it touches a bead or wall;
the starting layouts and every window re-lay anchor the ends they make.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from rltp.cover.baselines import link_ring, zigzag_links
from rltp.cover.cover import (
    adjacent, bead_weights, fix_seams_local, get_pieces, is_end, join_ends, link, neighbors4, normalize_seams,
    stroke_cells, unlink_cell,
)
from rltp.model import Frame, Layer

WINDOW = 16          # fine-edit window side, cells
ANCHOR_REACH = 40    # cells: how far anchoring extends a free path end (4 mm on the 0.1 mm grid)
ANCHOR_TURNS = True  # anchoring may also turn 90 degrees to the nearest bead beside the line (a T-joint)
INFILL_LINES = ("cut", "whole")   # gradient layouts: lines start and stop with the field, or each run whole


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


FIT_TOL_MM = 0.1     # fit mode: a printed segment passes within this of every node it replaces


def fit_mode(L: Layer) -> bool:
    """Whether the layer prints (and deposits) along fitted straight segments: set by a
    PartState scoring true bead directions (ScoreConfig.directions = "path")."""
    return bool(L.aux) and bool(L.aux.get("fit"))


def _ls_line(S: np.ndarray, s: int, e: int, x: np.ndarray, y: np.ndarray):
    """Least-squares line through points s..e (prefix sums S of 1, x, y, xx, yy, xy):
    (centroid (2,), unit direction (2,) pointing from s to e)."""
    t = S[e + 1] - S[s]
    m = t[0]
    mx, my = t[1] / m, t[2] / m
    cxx, cyy, cxy = t[3] / m - mx * mx, t[4] / m - my * my, t[5] / m - mx * my
    phi = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    u = np.array([np.cos(phi), np.sin(phi)])
    if u[0] * (x[e] - x[s]) + u[1] * (y[e] - y[s]) < 0:
        u = -u
    return np.array([mx, my]), u


def _fit_ok(S, s, e, x, y, pts, tol) -> bool:
    """Every point of `pts` (positions s..e) within tol of the least-squares line s..e."""
    c, u = _ls_line(S, s, e, x, y)
    return bool(np.abs((x[pts] - c[0]) * u[1] - (y[pts] - c[1]) * u[0]).max() <= tol)


def _fit_piece(x: np.ndarray, y: np.ndarray, cand: np.ndarray, fixed: np.ndarray, loop: bool, tol: float):
    """Segment bounds (positions) and vertex coordinates of one stroke (points x, y; a loop's
    last point repeats its first). Between fixed vertices, each segment grows greedily over
    the candidate vertices (galloping search) while the least-squares line through its
    points stays within tol of every candidate (the points where the path turns: the
    farthest from any line between them). Vertices: where consecutive lines meet, or the
    shared point's projection onto both, averaged, where they run nearly parallel or would
    meet far from it; a path end's point projected onto its line."""
    n = len(x)
    S = np.zeros((n + 1, 6))
    S[1:] = np.cumsum(np.stack([np.ones(n), x, y, x * x, y * y, x * y], 1), 0)
    bounds = [0]
    for a, b in zip(fixed[:-1], fixed[1:]):
        s = a
        while s < b:
            i0 = int(np.searchsorted(cand, s))             # cand[i0] == s
            cs = cand[i0 + 1:int(np.searchsorted(cand, b, "right"))]
            lo, hi = 0, 1                      # cs[0] always fits: the path is straight up to it
            while hi < len(cs):
                if not _fit_ok(S, s, int(cs[hi]), x, y, cand[i0:i0 + hi + 2], tol):
                    break
                lo, hi = hi, min(2 * hi, len(cs))
                if lo == len(cs) - 1:
                    break
            hi = min(hi, len(cs))
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _fit_ok(S, s, int(cs[mid]), x, y, cand[i0:i0 + mid + 2], tol):
                    lo = mid
                else:
                    hi = mid
            s = int(cs[lo])
            bounds.append(s)
    lines = [_ls_line(S, s, e, x, y) for s, e in zip(bounds[:-1], bounds[1:])]
    m = len(lines)
    P = np.stack([x, y], 1)
    seg_len = np.hypot(*(P[bounds[1:]] - P[bounds[:-1]]).T)

    def proj(i, j):
        c, u = lines[i]
        return c + ((P[j] - c) @ u) * u

    V = np.zeros((m + 1, 2))
    for j in range(m + 1):
        if j in (0, m) and not loop:
            V[j] = proj(0 if j == 0 else m - 1, bounds[j])
            continue
        i0, i1 = (j - 1) % m, j % m
        b = bounds[j]
        (c0, u0), (c1, u1) = lines[i0], lines[i1]
        det = u0[0] * u1[1] - u0[1] * u1[0]
        X = None
        if abs(det) > 0.09:                    # lines more than ~5 degrees apart: their crossing,
            d = c1 - c0                        # unless far off (a jog between near-parallel lines)
            X = c0 + (d[0] * u1[1] - d[1] * u1[0]) / det * u0
            if np.hypot(*(X - P[b])) > max(2 * tol, 0.3 * min(seg_len[i0], seg_len[i1])):
                X = None
        V[j] = X if X is not None else 0.5 * (proj(i0, b) + proj(i1, b))
    if loop:
        V[m] = V[0]
    return np.asarray(bounds, np.int64), V


def _fit_runs(L: Layer) -> dict:
    """runs() in fit mode: each stroke's polyline is straight segments, each the
    least-squares line through its nodes within FIT_TOL_MM of every one (_fit_piece), so a
    staircase at any angle or a 0.1 mm jog prints as one move along its true direction. The
    corners the direction fit finds (rltp.cover.paths.node_corners) end segments. Polylines
    are (vertex nodes: each segment's first and last node, segment of each link, vertex
    coordinates (m + 1, 2) as (column, row) in node units, floats). No diagonal runs:
    _deposit_fit lays every segment itself."""
    from rltp.cover.cover import get_pieces
    from rltp.cover.paths import node_corners
    W, n = L.W, L.n
    tol = FIT_TOL_MM / L.frame.p
    pc = get_pieces(L)
    corner = node_corners(L)
    poly = {}
    for pid, cells in enumerate(pc.cells):
        seq = np.asarray(cells, np.int64)
        loop = bool(pc.is_loop[pid]) and len(seq) > 1
        full = np.append(seq, seq[0]) if loop else seq
        nl = len(full) - 1
        r, c = np.divmod(full, W)
        if nl <= 0:
            poly[pid] = (full.copy(), np.zeros(0, np.int64), np.array([[c[0], r[0]]], float))
            continue
        st = np.diff(full)
        forced = np.flatnonzero(corner[full[1:-1]]) + 1
        cand = np.union1d(np.r_[0, np.flatnonzero(st[1:] != st[:-1]) + 1, nl], forced)
        fixed = np.union1d([0, nl], forced)
        bounds, V = _fit_piece(c.astype(float), r.astype(float), cand, fixed, loop, tol)
        poly[pid] = (full[bounds], np.repeat(np.arange(len(bounds) - 1), np.diff(bounds)), V)
    z = np.zeros(n, bool)
    return {"diag": [], "xr": z, "xd": z, "dnode": z, "poly": poly}


def runs(L: Layer) -> dict:
    """Diagonal runs and polylines of a fine layer (cached).

    A diagonal run is a stretch of at least DIAG_MIN links alternating one horizontal
    and one vertical step with constant signs: a 45 (x and y rising together) or 135
    degree bead drawn as a staircase. Returns {"diag": [(a, b, cls, a_end, b_end)],
    "xr": node has its link to the right inside a diagonal run, "xd": same for its link
    down, "dnode": node touches a diagonal link, "poly": {piece: (vertex nodes, segment
    of each link)}} where a polyline merges straight axis runs and each diagonal run into
    single segments (what the G-code and the time model print). In fit mode the polylines
    are fitted segments at any angle instead (_fit_runs)."""
    R = L._cache.get("runs")
    if R is not None:
        return R
    if fit_mode(L):
        R = L._cache["runs"] = _fit_runs(L)
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


SAMPLES = (1 / 6, 0.5, 5 / 6)   # 3 x 3 sample points per cell for diagonal runs (class mode)
# fit mode: 16 points per cell from the R2 low-discrepancy sequence, which has no rows of
# samples for a band edge to line up with (a 3 x 3 grid's points sit on 45-degree lines 0.24
# cells apart: a 45-degree band 4 cells wide caught 16 or 17 of them, +-6%; lattices have such
# rows too, e.g. 13 points with generator 5 along 33.7 and 123.7 degrees, on two line bins)
FIT_SAMPLES = tuple(((0.5 + i * 0.7548776662466927) % 1.0, (0.5 + i * 0.5698402909980532) % 1.0)
                    for i in range(1, 17))


def _deposit_fit(L: Layer) -> tuple[np.ndarray, np.ndarray]:
    """_deposit in fit mode: each segment of runs(L)["poly"] lays a band one bead wide along
    its straight line, half a bead past a path end; at a joint turning 90 degrees or less
    between segments long enough for it, the two bands meet at the bisector (a mitre, so
    the joint is covered once with nothing missing), else both stop at the vertex. Every
    cell under one straight bead gets exactly 1 and a stroke deposits its length x width
    plus half-bead caps, as its G-code extrudes. A lone node lays a k x k dab. Direction
    shares: each segment's material split between the two nearest of 0/45/90/135 degrees
    by angle; along x: cos^2 of its angle."""
    from rltp.cover.cover import get_pieces
    k, H, W = L.kb, L.H, L.W
    h = k // 2
    pc = get_pieces(L)
    poly = runs(L)["poly"]
    segs = []                 # per piece: P (m, 2), u, length, start/end normals and flags
    lone = np.zeros(L.n, bool)
    for pid, cells in enumerate(pc.cells):
        if len(cells) == 1:
            lone[cells[0]] = True
            continue
        xy = poly[pid][2] + 1.0                   # node (r, c) sits at (c + 1, r + 1), cell units
        d = np.diff(xy, axis=0)
        ln = np.hypot(d[:, 0], d[:, 1])
        tiny = ln < 1e-6                          # coinciding fitted vertices: lays nothing
        u = d / np.where(tiny, 1.0, ln)[:, None]
        u[tiny] = (1.0, 0.0)
        m = len(u)
        loop = bool(pc.is_loop[pid])
        uin = np.vstack([u[-1:] if loop else u[:1], u])          # into vertex j: u[j-1]
        uout = np.vstack([u, u[:1] if loop else u[-1:]])         # out of vertex j: u[j]
        s = uin + uout
        sn = np.hypot(s[:, 0], s[:, 1])
        cos = (uin * uout).sum(1)
        t = h * np.sqrt(np.clip((1 - cos) / np.maximum(1 + cos, 1e-12), 0, None))   # h tan(turn / 2)
        lin = np.r_[ln[-1:] if loop else [np.inf], ln]
        lout = np.r_[ln, ln[:1] if loop else [np.inf]]
        mitre = (cos >= -1e-12) & (sn > 1e-9) & (lin >= 2 * t) & (lout >= 2 * t)
        mitre &= ~np.r_[tiny[-1:] if loop else [False], tiny] & ~np.r_[tiny, tiny[:1] if loop else [False]]
        if not loop:
            mitre[0] = mitre[-1] = False
        nrm = s / np.maximum(sn, 1e-12)[:, None]
        end = np.zeros(m + 1, bool)
        if not loop:
            end[0] = end[-1] = True
        segs.append((xy[:-1], xy[1:], u, ln, nrm[:-1], mitre[:-1], end[:-1], nrm[1:], mitre[1:], end[1:], ~tiny))
    D = np.zeros(L.n)
    ax = np.zeros(L.n)
    sh = np.zeros((4, L.n))           # 0, 45, 90, 135 degrees
    if segs:
        *arrs, real = (np.concatenate(x) for x in zip(*segs))
        P, Q, u, ln, nA, mA, eA, nB, mB, eB = (a[real] for a in arrs)
        S = len(ln)
        a0 = np.where(eA | mA, -h, 0.0)                     # a mitre reaches h tan(turn / 2) <= h past
        a1 = ln + np.where(eB | mB, h, 0.0)                 # its vertex; the bisector cuts it there
        sw = np.abs(u[:, 1]) > np.abs(u[:, 0])               # y-major: work with x and y swapped

        def swp(a):
            return np.where(sw[:, None], a[:, ::-1], a)
        Ps, Qs, us, nAs, nBs = swp(P), swp(Q), swp(u), swp(nA), swp(nB)
        vs = np.stack([-us[:, 1], us[:, 0]], 1)
        cx = np.stack([Ps[:, 0] + a * us[:, 0] + sg * h * vs[:, 0] for a in (a0, a1) for sg in (-1, 1)], 1)
        c0 = np.floor(cx.min(1)).astype(np.int64)
        ncol = np.ceil(cx.max(1)).astype(np.int64) - c0
        seg = np.repeat(np.arange(S), ncol)
        col = c0[seg] + np.arange(seg.size) - np.repeat(np.cumsum(ncol) - ncol, ncol)
        slope = us[seg, 1] / us[seg, 0]
        y0 = Ps[seg, 1] + (col - Ps[seg, 0]) * slope
        half = h / np.abs(us[seg, 0])
        rlo = np.floor(np.minimum(y0, y0 + slope) - half).astype(np.int64)
        M = int(np.ceil(2 * h * np.sqrt(2) + 2)) + 1
        row = rlo[:, None] + np.arange(M)[None, :]
        frac = np.zeros(row.shape)
        px, py = Ps[seg, 0][:, None], Ps[seg, 1][:, None]
        ux, uy = us[seg, 0][:, None], us[seg, 1][:, None]
        lo_, hi_ = a0[seg][:, None], a1[seg][:, None]
        ma, mb = mA[seg][:, None], mB[seg][:, None]
        qx, qy = Qs[seg, 0][:, None], Qs[seg, 1][:, None]
        nax, nay = nAs[seg, 0][:, None], nAs[seg, 1][:, None]
        nbx, nby = nBs[seg, 0][:, None], nBs[seg, 1][:, None]
        for o1, o2 in FIT_SAMPLES:
            # (o1, o2) along (x, y); a swapped segment's frame has them the other way round
            X = col[:, None] + np.where(sw[seg], o2, o1)[:, None]
            Y = row + np.where(sw[seg], o1, o2)[:, None]
            dx, dy = X - px, Y - py
            along = dx * ux + dy * uy
            ok = (np.abs(dx * uy - dy * ux) <= h) & (along >= lo_) & (along <= hi_)
            ok &= ~ma | ((X - px) * nax + (Y - py) * nay >= 0)
            ok &= ~mb | ((X - qx) * nbx + (Y - qy) * nby < 0)
            frac += ok
        frac /= len(FIT_SAMPLES)
        cc = np.where(sw[seg][:, None], row, col[:, None])
        rr = np.where(sw[seg][:, None], col[:, None], row)
        keep = (frac > 0) & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        idx = (rr * W + cc)[keep]
        f = frac[keep]
        sid = np.broadcast_to(seg[:, None], frac.shape)[keep]
        ang = np.arctan2(u[:, 1], u[:, 0]) % np.pi
        tq = ang / (np.pi / 4)
        i0 = np.floor(tq).astype(np.int64) % 4
        w1 = tq - np.floor(tq)
        D = np.bincount(idx, f, L.n)
        ax = np.bincount(idx, f * np.cos(ang)[sid] ** 2, L.n)
        for j in range(4):
            wj = np.where(i0 == j, 1 - w1, 0.0) + np.where((i0 + 1) % 4 == j, w1, 0.0)
            sh[j] = np.bincount(idx, f * wj[sid], L.n)
    D, ax, sh = D.reshape(H, W), ax.reshape(H, W), sh.reshape(4, H, W)
    dab = cover_sum(lone.reshape(H, W), k)
    D = D + dab
    ax = ax + 0.5 * dab
    L._cache["shares_raw"] = (sh[0] + 0.5 * dab, sh[2] + 0.5 * dab, sh[1], sh[3])
    return D, ax


def density(L: Layer) -> np.ndarray:
    D = L._cache.get("density")
    if D is None:
        D, ax = _deposit_fit(L) if fit_mode(L) else _deposit(L)
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
    the layer has no diagonal runs (then cell_wh says it all). Fit mode always gives them."""
    D = density(L)
    if not runs(L)["diag"] and not fit_mode(L):
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

def line_set(sel: np.ndarray, cls, k: int, anchor: tuple, dens: float, spacing: int,
             fill_map: np.ndarray | None = None, lines: str = "cut") -> np.ndarray:
    """_line_set; at the in-between angles (not 0/45/90/135), without the line pieces shorter
    than a bead width (k nodes), which a line clipping an area's corner leaves as dabs."""
    out = _line_set(sel, cls, k, anchor, dens, spacing, fill_map, lines)
    if cls not in (0, 45, 90, 135) and out.any():
        lab, n = ndimage.label(out)
        size = np.bincount(lab.reshape(-1), minlength=n + 1)
        size[0] = k
        out &= (size >= k)[lab]
    return out


def _line_set(sel: np.ndarray, cls, k: int, anchor: tuple, dens: float, spacing: int,
              fill_map: np.ndarray | None = None, lines: str = "cut") -> np.ndarray:
    """Nodes of `sel` on parallel bead lines of class `cls` degrees, one bead apart
    (perpendicular), through `anchor` (r, c); with dens < 1 a share of the lines picked by
    error diffusion (never more than `spacing` lines apart), or with `fill_map` each line
    laid wherever the field exceeds its dither threshold (shell.dither_ranks; `lines`
    "whole": each run of a line laid whole where the field's mean over it does, whole_runs).
    A 45-degree
    line is the staircase c - r in {d, d + 1}; lines sit k*sqrt(2) apart in d (alternating
    6 and 5 for k = 4), so their perpendicular spacing averages exactly one bead. Any other
    angle (fit mode: multiples of 11.25 degrees, LINE_BINS) lays standard digital lines: the
    nodes whose coordinate across the line lies within |cos| + |sin| of it (4-connected,
    never 4-adjacent to the next line), lines exactly one bead apart across."""
    from rltp.cover.shell import dither_ranks, line_picks
    H, W = sel.shape
    if not sel.any():
        return np.zeros_like(sel)
    rr, cc = np.mgrid[0:H, 0:W]
    ar, ac = anchor
    if cls not in (0, 45, 90, 135):
        t = np.radians(float(cls))
        ux, uy = np.cos(t), np.sin(t)
        om = abs(ux) + abs(uy)
        q = (rr - ar) * ux - (cc - ac) * uy + om / 2         # across the line, from the anchor's
        m = np.floor(q / k).astype(np.int64)
        within = q - m * k < om
        if not (sel & within).any():
            return np.zeros_like(sel)
        j0, j1 = int(m[sel & within].min()), int(m[sel & within].max())
        P = np.arange(j0, j1 + 1)
        lid = np.where(within & (m >= j0) & (m <= j1), m - j0, -1)
        ok = lid >= 0
    elif cls in (0, 90):
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
    if cls in (0, 45, 90, 135):
        idx = np.searchsorted(P, key, "right") - 1
        ok = (idx >= 0) & (key - P[np.clip(idx, 0, len(P) - 1)] <= width)
        lid = np.where(ok, idx, -1)
    on = sel & ok
    counts = np.bincount(lid[on], minlength=len(P))
    if dens >= 1.0:
        pick = counts > 0
    elif fill_map is not None:
        rank = dither_ranks(len(P), spacing, j0)[np.clip(lid, 0, len(P) - 1)]
        return whole_runs(on, fill_map, rank) if lines == "whole" else on & (fill_map > rank)
    else:
        pick = line_picks(counts, dens, spacing)
    return on & pick[np.clip(lid, 0, len(P) - 1)]


def whole_runs(cand: np.ndarray, fill_map: np.ndarray, thr: np.ndarray,
               structure: np.ndarray | None = None) -> np.ndarray:
    """The runs of `cand` (bead-line nodes) whose mean `fill_map` exceeds their mean
    threshold `thr` (per node, the line's dither rank). Each infill line is laid whole
    between walls and other areas, or not at all, instead of starting and stopping wherever
    the field crosses its threshold, which leaves pieces that touch nothing in their layer.
    A run is a 4-connected piece of `cand` (fine grid: lines are never 4-adjacent), or a
    piece under `structure` (e.g. along rows only, where lines are adjacent)."""
    lab, n = ndimage.label(cand, structure)
    if n == 0:
        return np.zeros_like(cand)
    idx = lab[cand]
    cnt = np.maximum(np.bincount(idx, minlength=n + 1), 1)
    keep = np.bincount(idx, fill_map[cand], minlength=n + 1) / cnt >         np.bincount(idx, thr[cand], minlength=n + 1) / cnt
    keep[0] = False
    return keep[lab]


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


LINE_BINS = 16       # fit mode: bead lines at multiples of 180 / 16 = 11.25 degrees
BIN_DEG = 180.0 / LINE_BINS


def angle_links(L: Layer, nodes: np.ndarray, deg: float) -> None:
    """Link the lines at `deg` degrees in `nodes` (all unlinked, line_set's standard digital
    lines): each 4-connected piece in order along the line (lines are never 4-adjacent)."""
    W = L.W
    lab, n = ndimage.label(nodes)
    if n == 0:
        return
    idx = np.flatnonzero(nodes.reshape(-1))
    r, c = np.divmod(idx, W)
    t = np.radians(float(deg))
    li = lab.reshape(-1)[idx]
    order = np.lexsort((c * np.cos(t) + r * np.sin(t), li))
    idx, li = idx[order].tolist(), li[order].tolist()
    for x, y, a, b in zip(idx[:-1], idx[1:], li[:-1], li[1:]):
        if a == b and adjacent(x, y, W) and L.nxt[x] == -1 and L.prv[y] == -1:
            link(L, x, y)


def link_lines(L: Layer, nodes: np.ndarray, cls) -> None:
    if cls in (0, 90):
        zigzag_links(L, nodes, int(cls))
    elif cls in (45, 135):
        diag_links(L, nodes, int(cls))
    else:
        angle_links(L, nodes, cls)


# ------------------------------------------------------------------ starting layout

def _erode(m: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return m.copy()
    return ndimage.binary_erosion(m, np.ones((3, 3), bool), iterations=n, border_value=0)


def fine_layout(L: Layer, part: np.ndarray, solid: np.ndarray, walls: int, density_: float, angle: int,
                line_angle: int, max_spacing: int, k: int, orient: np.ndarray | None = None,
                sweight: np.ndarray | None = None, fill_map: np.ndarray | None = None,
                lines: str = "cut") -> Layer:
    """Walls as locked loops k cells apart along the true outline, solid skin as beads k
    cells apart, infill lines picked by error diffusion over bead rows (max_spacing in
    beads apart), connected where ends meet. With `orient` (True where beads should run
    along y, or a class per cell: 0, 45, 90, 135 degrees) skin and infill follow it instead
    of `angle`/`line_angle`, and `sweight` (relative stress per cell) packs infill lines
    where stress is high. `fill_map` (per cell, the gradient layout's density field) sets
    each line's density instead of `density_`: lines start and stop with the field, or with
    `lines` "whole" each run of a line is laid whole or not at all (whole_runs). Free path
    ends are then anchored (anchor_ends): straight on, or a T-joint to the next line."""
    from rltp.cover.shell import close_ring, dither_ranks, line_picks, row_density
    H, W = part.shape
    lo, hi = offsets(k)
    fit = fit_mode(L)                                      # kept: set by the layer's PartState
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
            if fill_map is not None and dens < 1.0:     # each run whole, by the field's mean along it
                out[bead_rows] = sub
                thr = np.zeros(sel.shape)
                thr[bead_rows] = dither_ranks(len(bead_rows), max_spacing)[:, None]
                return whole_runs(out, fill_map, thr) if lines == "whole" else out & (fill_map > thr)
            d = row_density(sub, dens, None if sweight is None else sweight[bead_rows], max_spacing, 0)
            pick = bead_rows[line_picks(sub.sum(1), d, max_spacing)]
            out[pick] = sel[pick]
        else:
            ph = int(cols[0]) % k
            bead_cols = np.arange(ph, W, k)
            sub = sel[:, bead_cols]
            out = np.zeros_like(sel)
            if fill_map is not None and dens < 1.0:
                out[:, bead_cols] = sub
                thr = np.zeros(sel.shape)
                thr[:, bead_cols] = dither_ranks(len(bead_cols), max_spacing)[None, :]
                return whole_runs(out, fill_map, thr) if lines == "whole" else out & (fill_map > thr)
            d = row_density(sub, dens, None if sweight is None else sweight[:, bead_cols], max_spacing, 1)
            pick = bead_cols[line_picks(sub.sum(0), d, max_spacing)]
            out[:, pick] = sel[:, pick]
        return out

    if orient is not None:                 # stress-aligned: each area runs its own way
        cls_map = np.where(orient, 90, 0) if orient.dtype == bool else orient
        anchor = (int(rows[0]), int(cols[0])) if rows.size else (0, 0)
        taken = np.zeros_like(part)
        more = sorted(set(np.unique(cls_map[inner]).tolist()) - {0, 45, 90, 135}) if inner.any() else []
        for cls in [0, 90, 45, 135] + more:            # more: fit mode's other 11.25-degree bins
            area = cls_map == cls
            if not area.any():
                continue
            if cls in (0, 90):
                got = lines(solid_n & area, cls, 1.0) | lines(opt_n & area, cls, density_)
            else:
                got = line_set(solid_n & area, cls, k, anchor, 1.0, max_spacing) | \
                    line_set(opt_n & area, cls, k, anchor, density_, max_spacing, fill_map, lines)
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
             "phase": (int(rows[0]) % k if rows.size else 0, int(cols[0]) % k if cols.size else 0),
             "fit": fit}
    L.invalidate()
    repair_fine(L)
    ext = anchor_ends(L, free_ends(L))
    if ext:
        repair_fine(L, cells=ext)
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


# ------------------------------------------------------------------ anchoring

def _near_end(seq, at_head: bool, m: int) -> set:
    """The m nodes of a stroke nearest its head (or tail): the end's own bead, which never
    anchors it."""
    return set(seq[:m]) if at_head else set(seq[-m:])


def is_anchored(L: Layer, e: int, own: set) -> bool:
    """True when another bead's centreline node (walls included) lies within one bead width
    (kb cells, Chebyshev) of path end e: the footprints touch, so the bead can pass load
    in-plane to the rest of the layer. `own`: the end's own nodes to ignore."""
    k = max(L.kb, 1)
    W = L.W
    r, c = divmod(int(e), W)
    r0, c0 = max(r - k, 0), max(c - k, 0)
    rr, cc = np.nonzero(L.mask[r0:r + k + 1, c0:c + k + 1])
    return any(x not in own for x in ((rr + r0) * W + cc + c0).tolist())


def free_ends(L: Layer, pc=None) -> list[int]:
    """Path ends (walls excluded) that no other bead touches: see is_anchored."""
    pc = get_pieces(L) if pc is None else pc
    lock = L.lock.reshape(-1) if L.lock is not None else None
    m = 2 * max(L.kb, 1) + 1
    out = []
    for seq, loop in zip(pc.cells, pc.is_loop):
        if loop or (lock is not None and lock[seq[0]]):
            continue
        for at_head in ((True, False) if len(seq) > 1 else (True,)):
            e = seq[0] if at_head else seq[-1]
            if not is_anchored(L, e, _near_end(seq, at_head, m)):
                out.append(int(e))
    return out


def _end_run(L: Layer, e: int, m: int) -> tuple[list[int], bool] | None:
    """The m nodes of e's stroke nearest e, starting at e, and whether e is the head; None
    unless e ends an open path."""
    at_head = L.prv[e] == -1
    if not at_head and L.nxt[e] != -1:
        return None
    step = L.nxt if at_head else L.prv
    run = [int(e)]
    while len(run) < m and step[run[-1]] != -1:
        run.append(int(step[run[-1]]))
    return run, at_head


def _anchor_path(N: Layer, e: int, steps: tuple, run: list, reach: int, m: int) -> list[int] | None:
    """Nodes from e repeating `steps` through empty region nodes until another bead touches
    the last one, or None (blocked, or nothing within reach)."""
    W = N.W
    region = N.aux["region"].reshape(-1) if N.aux is not None else N.part_mask.reshape(-1)
    flat = N.flat_mask
    path: list[int] = []
    x = e
    for i in range(reach):
        y = x + steps[i % len(steps)]
        if not (0 <= y < N.n) or not adjacent(x, y, W) or not region[y] or flat[y]:
            return None
        path.append(y)
        x = y
        if is_anchored(N, y, set(path[-m:]) | (set(run) if len(path) < m else set())):
            return path
    return None


def anchor_end(N: Layer, e: int, reach: int = ANCHOR_REACH, turns: bool | None = None) -> list[int] | None:
    """Extend path end e until another bead touches it, the shortest way of: along its line
    (straight, or a 45-degree staircase: the last two steps repeat), or (`turns`, default
    ANCHOR_TURNS; straight ends only) turning 90 degrees to the nearest bead beside it, a
    T-joint that keeps the line's length. Returns the added nodes ([] if e is already
    anchored), or None, with N unchanged, when nothing is in reach or every way is blocked
    (the region's edge, another bead's node)."""
    m = 2 * max(N.kb, 1) + 1
    got = _end_run(N, e, m)
    if got is None:
        return None
    run, at_head = got
    if is_anchored(N, e, set(run)):
        return []
    if len(run) < 2:
        return None
    s1 = run[0] - run[1]
    s0 = run[1] - run[2] if len(run) >= 3 else s1
    ways = [(s0, s1)]
    if (ANCHOR_TURNS if turns is None else turns) and s0 == s1:
        side = N.W if abs(s1) == 1 else 1
        ways += [(side,), (-side,)]
    best = None
    for st in ways:
        path = _anchor_path(N, e, st, run, reach if best is None else len(best) - 1, m)
        if path is not None and (best is None or len(path) < len(best)):
            best = path
    if best is None:
        return None
    path = best
    prev = e
    for y in path:
        N.mask.reshape(-1)[y] = True
        if at_head:
            link(N, y, prev)
        else:
            link(N, prev, y)
        prev = y
    N.invalidate()
    return path


def anchor_ends(N: Layer, cells, reach: int = ANCHOR_REACH) -> list[int]:
    """Anchor the free path ends among `cells` (walls excluded) that can be; returns the new
    end nodes."""
    lock = N.lock.reshape(-1) if N.lock is not None else None
    out = []
    for e in sorted(int(c) for c in cells):
        if not N.flat_mask[e] or not is_end(N, e) or (lock is not None and lock[e]):
            continue
        try:
            path = anchor_end(N, e, reach)
        except FineEditError:
            continue
        if path:
            out.append(path[-1])
    return out


def anchor(L: Layer, N: Layer, cell: int) -> dict:
    """Edit: extend the free path end at `cell` until it touches a bead or a wall."""
    if not is_end(L, cell) or (L.lock is not None and L.lock.reshape(-1)[cell]):
        raise FineEditError("not a free path end")
    path = anchor_end(N, cell)
    if path is None:
        raise FineEditError("nothing to anchor to within reach")
    if not path:
        raise FineEditError("end already anchored")
    repair_fine(N, cells=[path[-1]])
    return {"added": len(path)}


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


PACK_SHARP = 3.0     # pack: the laid density follows worth ** PACK_SHARP (then clipped to 0..1)


def packed_field(worth: np.ndarray, sel: np.ndarray, dens: float) -> np.ndarray:
    """A density field over `sel` with mean `dens` that follows worth ** PACK_SHARP, each cell
    clipped to [0, 1] (so the most valuable lines fill solid first); `dens` everywhere where
    worth is flat or missing."""
    out = np.where(sel, dens, 0.0)
    if not sel.any() or dens <= 0.0 or dens >= 1.0:
        return out
    w = np.maximum(np.asarray(worth, float)[sel], 0.0) ** PACK_SHARP
    if w.max() <= 0.0 or w.max() - w.min() <= 1e-12 * w.max():
        return out
    lo, hi = 0.0, dens / w.mean()
    while np.minimum(hi * w, 1.0).mean() < dens and hi < 1e12:   # mean(min(s w, 1)) rises with s
        hi *= 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.minimum(mid * w, 1.0).mean() < dens else (lo, mid)
    out[sel] = np.minimum(hi * w, 1.0)
    return out


def _relay(L: Layer, N: Layer, area: np.ndarray, angle: int, cell: int, worth: np.ndarray | None = None) -> None:
    """Re-lay the free beads over `area` (node positions) k cells apart, through `cell`'s row
    (angle 0) or column (90): every line over required cells, and over optional cells the
    share of lines that keeps the area's current infill share. With `worth` (pack), the
    optional lines are laid from packed_field instead of evenly: the same share, packed onto
    the lines where worth is highest (each run of a line laid whole, shell.dither_ranks)."""
    from rltp.cover.shell import line_picks
    H, W, k = L.H, L.W, L.kb
    lock = L.lock if L.lock is not None else np.zeros_like(area)
    opt_cells = area & L.part & ~L.req
    dens = float((L.material & opt_cells).sum()) / max(int(opt_cells.sum()), 1)
    touched = _clear_nodes(N, area & N.mask & ~lock)
    r, c = divmod(cell, W)
    if worth is not None:
        cand = area & N.aux["inner"] & ~N.mask
        spacing = int(N.aux.get("max_spacing", 1 << 30))
        opt = cand & ~L.req
        new = line_set(cand & L.req, angle, k, (r, c), 1.0, spacing) | \
            line_set(opt, angle, k, (r, c), dens, spacing, fill_map=packed_field(worth, opt, dens), lines="whole")
        if not new.any():
            raise FineEditError("nothing to lay")
        N.mask |= new
        N.invalidate()
        link_lines(N, new, angle)
        ends = set(np.flatnonzero(new.reshape(-1) & _ends(N)).tolist()) | {x for x in touched if N.flat_mask[x]}
        _join_and_anchor(L, N, ends, touched)
        return
    if angle not in (0, 90):               # diagonal lines through the cell, keeping the share
        cand = area & N.aux["inner"] & ~N.mask
        spacing = int(N.aux.get("max_spacing", 1 << 30))
        new = line_set(cand & L.req, angle, k, (r, c), 1.0, spacing) | \
            line_set(cand & ~L.req, angle, k, (r, c), dens, spacing)
        if not new.any():
            raise FineEditError("nothing to lay")
        N.mask |= new
        N.invalidate()
        link_lines(N, new, angle)
        ends = set(np.flatnonzero(new.reshape(-1) & _ends(N)).tolist()) | {x for x in touched if N.flat_mask[x]}
        _join_and_anchor(L, N, ends, touched)
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
    _join_and_anchor(L, N, ends, touched)


def _join_and_anchor(L: Layer, N: Layer, ends: set, touched: set) -> None:
    """After laying beads: join the new and cut ends to nearby ends, anchor the rest that can
    be (anchor_ends), and fix the seams of the loops touched."""
    repair_fine(N, cells=ends)
    ext = anchor_ends(N, ends)
    if ext:
        repair_fine(N, cells=ext)
    fix_seams_local(N, ends | touched | set(ext), prefer=sorted(L.seam))


def refill(L: Layer, N: Layer, cell: int, angle: int, size: int = WINDOW) -> dict:
    """Re-lay a size x size window at 0 or 90 degrees, keeping its infill share."""
    H, W = L.H, L.W
    r0, r1, c0, c1 = _window(cell, H, W, size)
    win = np.zeros((H, W), bool)
    win[r0:r1, c0:c1] = True
    _relay(L, N, win, angle, cell)
    return {"window": [r0, c0, r1, c1]}


def pack(L: Layer, N: Layer, cell: int, angle: int, size: int = 32) -> dict:
    """Re-lay a size x size window like refill, keeping its infill share, with its lines packed
    where one more unit of infill is worth most: L.ctx["worth"], per cell (set by the annealing
    loop from rltp.search.guide.WorthMap)."""
    worth = L.ctx.get("worth") if L.ctx is not None else None
    if worth is None:
        raise FineEditError("no worth map")
    H, W = L.H, L.W
    r0, r1, c0, c1 = _window(cell, H, W, size)
    win = np.zeros((H, W), bool)
    win[r0:r1, c0:c1] = True
    _relay(L, N, win, angle, cell, worth=worth)
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


def bin_deg(b: int):
    """Degrees of an 11.25-degree bin: an int for the four classes (the 0/45/90/135 code
    paths), a float otherwise."""
    d = (int(b) % LINE_BINS) * BIN_DEG
    return int(d) if d in (0, 45, 90, 135) else d


def bead_bins(L: Layer) -> np.ndarray:
    """Per cell (H x W): the 11.25-degree bin (0..15) most of its covering nodes' path
    directions fall in (rltp.cover.paths.node_angles), -1 where empty (cached)."""
    out = L._cache.get("bead_bins")
    if out is not None:
        return out
    from rltp.cover.paths import node_angles
    H, W, k = L.H, L.W, L.kb
    a = node_angles(L)
    ok = ~np.isnan(a)
    b = np.full(L.n, -1, np.int64)
    b[ok] = np.round(np.degrees(a[ok]) / BIN_DEG).astype(np.int64) % LINE_BINS
    cnt = np.stack([cover_sum((b == j).reshape(H, W), k) for j in range(LINE_BINS)])
    out = np.where((cnt.sum(0) > 0) & (density(L) >= 0.5), np.argmax(cnt, 0), -1)
    L._cache["bead_bins"] = out
    return out


def turn16(L: Layer, N: Layer, cell: int, step: int) -> dict:
    """Fit mode: turn the connected area around `cell` (walls excluded, within ALIGN_WINDOW)
    whose beads run in the cell's 11.25-degree bin by `step` bins, keeping its infill share."""
    if not fit_mode(L):
        raise FineEditError("11.25-degree lines need fit mode")
    H, W, k = L.H, L.W, L.kb
    b = bead_bins(L)
    r, c = divmod(cell, W)
    lock = L.lock if L.lock is not None else np.zeros_like(L.mask)
    walls = cover_sum(lock, k) > 0
    if walls[r, c] or b[r, c] < 0:
        raise FineEditError("no bead here")
    r0, r1, c0, c1 = _window(cell, H, W, ALIGN_WINDOW)
    same = np.zeros((H, W), bool)
    same[r0:r1, c0:c1] = b[r0:r1, c0:c1] == b[r, c]
    lab, _ = ndimage.label(same & ~walls)
    area = lab == lab[r, c]
    to = bin_deg(b[r, c] + step)
    _relay(L, N, area, to, cell)
    return {"cells": int(area.sum()), "to": to}


LAY_SIZES = {"lay16": 16, "lay32": 32, "lay64": 64}


def lay(L: Layer, N: Layer, cell: int, size: int) -> dict:
    """Fit mode: re-lay a size x size window, keeping its infill share, at the 11.25-degree
    bin the direction guide picks for a window of that size there: L.ctx["lay"][size] (H x W,
    0..15, -1 for none; rltp.search.guide.DirectionMap)."""
    maps = L.ctx.get("lay") if L.ctx is not None else None
    if not fit_mode(L) or maps is None or size not in maps:
        raise FineEditError("no direction map")
    H, W = L.H, L.W
    r, c = divmod(cell, W)
    d = int(maps[size][r, c])
    if d < 0:
        raise FineEditError("no best direction here")
    r0, r1, c0, c1 = _window(cell, H, W, size)
    win = np.zeros((H, W), bool)
    win[r0:r1, c0:c1] = True
    _relay(L, N, win, bin_deg(d), cell)
    return {"window": [r0, c0, r1, c1], "to": bin_deg(d)}


ALIGN_WINDOW = 64      # align_region: the area stays within this window around the cell


def align_region(L: Layer, N: Layer, cell: int) -> dict:
    """Re-lay the connected area around `cell` (walls excluded, within ALIGN_WINDOW) whose
    best bead direction is the cell's, in that direction, keeping its infill share. The map
    is L.ctx["best"] (H x W: 0..3 for 0/45/90/135 degrees, -1 for none), set by the
    direction guide (rltp.search.guide) from the current strains."""
    best = L.ctx.get("best") if L.ctx is not None else None
    if best is None:
        raise FineEditError("no direction map")
    H, W, k = L.H, L.W, L.kb
    r, c = divmod(cell, W)
    d = int(best[r, c])
    if d < 0:
        raise FineEditError("no best direction here")
    lock = L.lock if L.lock is not None else np.zeros_like(L.mask)
    r0, r1, c0, c1 = _window(cell, H, W, ALIGN_WINDOW)
    same = np.zeros((H, W), bool)
    same[r0:r1, c0:c1] = (best[r0:r1, c0:c1] == d)
    same &= ~(cover_sum(lock, k) > 0)
    lab, _ = ndimage.label(same)
    if lab[r, c] == 0:
        raise FineEditError("cell under a wall")
    area = lab == lab[r, c]
    angle = (0, 45, 90, 135)[d]
    cls = bead_class(L)
    beads = area & (cls >= 0)
    if beads.any() and (cls[beads] == angle).mean() > 0.9:
        raise FineEditError("already aligned")
    _relay(L, N, area, angle, cell)
    return {"cells": int(area.sum()), "to": angle}


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
    _join_and_anchor(L, N, ends, touched)
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
    best = L.ctx.get("best") if L.ctx is not None else None
    out["align_region"] = free & (best >= 0) if best is not None else np.zeros_like(m)
    fe = np.zeros(L.n, bool)
    fe[free_ends(L)] = True
    out["anchor"] = fe.reshape(H, W)
    worth = L.ctx.get("worth") if L.ctx is not None else None
    none = np.zeros_like(m)
    for head, size in (("pack32", 32), ("pack64", 64)):
        valid = out[f"refill{size}_0"] & (L.part is not None) if worth is not None else none
        for ang in PACK_ANGLES[head]:
            out[f"{head}_{ang}"] = valid
    # 16 line bins: on in fit mode once the direction guide has put its lay maps in the
    # layer's context (the annealing loop with line_bins = 16)
    maps = L.ctx.get("lay") if (fit_mode(L) and L.ctx is not None) else None
    out["turn16_pos"] = out["turn16_neg"] = free if maps is not None else none
    for name, size in LAY_SIZES.items():
        win = out["refill_0"] if size == 16 else out[f"refill{size}_0"]
        out[name] = win & (maps[size] >= 0) if maps is not None and size in maps else none
    below = L.ctx.get("below_layer") if L.ctx is not None else None
    if below is not None:
        diff = (below.mask ^ m) & inner
        out["stack_w"] = box_sum(diff, -h, h - 1) > 0
    else:
        out["stack_w"] = np.zeros_like(m)
    return out


PACK_ANGLES = {"pack32": (0, 45, 90, 135), "pack64": (0, 90)}
FINE_OPS = ("join", "shift_pos", "shift_neg", "extend", "retract", "refill_0", "refill_90", "clear_w", "stack_w",
            "refill32_0", "refill32_90", "refill64_0", "refill64_90", "flip_region",
            "refill_45", "refill_135", "refill32_45", "refill32_135", "turn_region", "align_region", "anchor",
            "pack32_0", "pack32_45", "pack32_90", "pack32_135", "pack64_0", "pack64_90",
            "turn16_pos", "turn16_neg", "lay16", "lay32", "lay64")


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
    if name.startswith("pack"):
        head, ang = name.split("_")
        return pack(L, N, cell, int(ang), {"pack32": 32, "pack64": 64}[head])
    if name in ("turn16_pos", "turn16_neg"):
        return turn16(L, N, cell, 1 if name == "turn16_pos" else -1)
    if name in LAY_SIZES:
        return lay(L, N, cell, LAY_SIZES[name])
    if name == "flip_region":
        return flip_region(L, N, cell, 90)
    if name == "turn_region":
        return flip_region(L, N, cell, 45)
    if name == "align_region":
        return align_region(L, N, cell)
    if name == "anchor":
        return anchor(L, N, cell)
    if name == "clear_w":
        return clear(L, N, cell)
    if name == "stack_w":
        return stack(L, N, cell)
    raise FineEditError(f"unknown fine edit {name}")
