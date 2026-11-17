"""Bead directions read off the toolpath itself (fine bead grids), for scoring beads along
their true direction instead of the 0/45/90/135 class their grid steps fall into.

On the 0.1 mm grid a bead is a 4-connected chain of nodes, so a bead at 20 degrees is a
staircase of x-steps with a y-jog every few cells. Scored by its steps, every cell near a
jog holds some 90-degree material and takes the worse of 0 and 90 degrees, as if the bead
were loaded across. The extruded strand runs at 20 degrees: a 0.1 mm jog is a quarter of
the bead's width. Here each node gets the direction of its path:

  * tangent: a least-squares line through the nodes within TAN_MM of path either side
    (centred, so exact on circles; grid jogs average out over ~16 nodes);
  * corners: the fit never crosses a corner, where the path turns more than CORNER_DEG
    between least-squares directions over CORNER_MM before and after a node, and the turn
    is concentrated there (at least CONC as large over half that path, compared up to 90
    degrees: a curve turns about half as much, two corners close together keep turning).
    Candidates are the sharpest node at the 0.2 mm scale; one per 0.3 mm; none within
    half a bead (CORNER_MM / 2) of a path end, where a last grid jog looks like a turn.

Checked on rasterised paths (scripts/analysis/tangent_fit.py, Nov 16, 2026): straight lines at
every angle 0.8 deg mean error (95% within 2.0), circles of 1.6-10 mm radius 0.6-1.3 deg
mean with no false corners, zigzag connectors (90-degree turns 0.4 mm apart) exact, bends
of 45 degrees and more found as corners. 1 mm radii are the weak spot (~9 deg mean).

Per material cell (cells under a node's k x k footprint): the mean over its covering nodes
of (cos 2a, sin 2a, cos 4a, sin 4a), which with the cell's material gives its stiffness
exactly (a bead's plane stiffness is linear in those), and the set of directions present,
quantised to NBINS over 180 degrees (a cache key for the failure index, finer than the
fit). A lone node (no direction) counts as holding both axes, as the class scoring has it.
"""
from __future__ import annotations

import numpy as np

TAN_MM = 0.8
CORNER_DEG = 30.0
CORNER_MM = 0.8
CONC = 0.7
NBINS = 128
MERGE = 2        # bins: directions this close in one cell are one (cell_bins)


def _concat(L):
    """The layer's strokes as one coordinate array: loops padded by wrapping round, so
    windows near a seam see the path on both sides. Returns (node, x, y, lo, hi, out)
    per position: node index, column, row, the stroke's bounds [lo, hi) and whether the
    position is the stroke's own (not padding)."""
    from rltp.cover.cover import get_pieces
    W = L.W
    p = L.material_frame.p
    pad = int(round(max(TAN_MM, CORNER_MM) / p)) + 4
    pc = get_pieces(L)
    nodes, los, his, own = [], [], [], []
    o = 0
    for cells, loop in zip(pc.cells, pc.is_loop):
        s = np.asarray(cells, np.int64)
        n = len(s)
        if loop and n > 2:
            ext = s[np.arange(-pad, n + pad) % n]
            ow = np.zeros(len(ext), bool)
            ow[pad:pad + n] = True
        else:
            ext, ow = s, np.ones(n, bool)
        m = len(ext)
        nodes.append(ext)
        los.append(np.full(m, o))
        his.append(np.full(m, o + m))
        own.append(ow)
        o += m
    if not nodes:
        z = np.zeros(0, np.int64)
        return z, z.astype(float), z.astype(float), z, z, z.astype(bool)
    node = np.concatenate(nodes)
    r, c = np.divmod(node, W)
    return node, c.astype(float), r.astype(float), np.concatenate(los), np.concatenate(his), np.concatenate(own)


def _ls_dirs(S, a, b):
    """Principal direction (rad, mod pi) of positions a..b (inclusive) from prefix sums S of
    (1, x, y, xx, yy, xy); nan where fewer than 2 points."""
    T = S[b + 1] - S[a]
    n = T[:, 0]
    ok = n >= 2
    nz = np.where(ok, n, 1.0)
    mx, my = T[:, 1] / nz, T[:, 2] / nz
    cxx = T[:, 3] / nz - mx * mx
    cyy = T[:, 4] / nz - my * my
    cxy = T[:, 5] / nz - mx * my
    return np.where(ok, 0.5 * np.arctan2(2 * cxy, cxx - cyy), np.nan)


def _turn(u, v):
    """Angle (rad, 0..pi) between direction vectors (rows)."""
    nu = np.hypot(u[:, 0], u[:, 1])
    nv = np.hypot(v[:, 0], v[:, 1])
    ok = (nu > 0) & (nv > 0)
    cos = np.where(ok, (u * v).sum(1) / np.where(ok, nu * nv, 1.0), 1.0)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def node_angles(L) -> np.ndarray:
    """Bead direction (rad, 0..pi) of every centerline node of a fine layer, nan elsewhere
    and at lone nodes (cached)."""
    out = L._cache.get("node_angle")
    if out is not None:
        return out
    p = L.material_frame.p
    h = max(int(round(TAN_MM / p)), 1)
    m = max(int(round(CORNER_MM / p)), 2)
    node, x, y, lo, hi, own = _concat(L)
    N = len(node)
    ang = np.full(L.H * L.W, np.nan)
    L._cache["node_corner"] = np.zeros(L.H * L.W, bool)
    if N == 0:
        L._cache["node_angle"] = ang
        return ang
    i = np.arange(N)
    P = np.stack([x, y], 1)
    S = np.zeros((N + 1, 6))
    S[1:] = np.cumsum(np.stack([np.ones(N), x, y, x * x, y * y, x * y], 1), 0)

    def clip(j, base):
        """Positions j kept within the strokes of positions `base`."""
        return np.clip(j, lo[base], hi[base] - 1)

    # candidates: sharp turns at the 0.2 mm scale, local maxima within 2 nodes
    small = _turn(P[clip(i + 2, i)] - P, P - P[clip(i - 2, i)])
    small[(i + 2 >= hi) | (i - 2 < lo)] = 0.0
    nb = np.max(np.stack([small[clip(i + s, i)] for s in (-2, -1, 1, 2)]), 0)
    cand = np.flatnonzero((small >= np.radians(45.0)) & (small >= nb))
    cuts = np.zeros(0, np.int64)
    if cand.size:
        def side(j, s, mm):
            a = clip(j - mm, j) if s < 0 else j
            b = j if s < 0 else clip(j + mm, j)
            phi = _ls_dirs(S, a, b)
            u = np.stack([np.cos(phi), np.sin(phi)], 1)
            ch = P[b] - P[a]
            flip = (u * ch).sum(1) < 0
            u[flip] *= -1
            return np.where(np.isnan(phi)[:, None], 0.0, u)

        t8 = _turn(side(cand, -1, m), side(cand, 1, m))
        t4 = _turn(side(cand, -1, m // 2), side(cand, 1, m // 2))
        ok = (t8 > np.radians(CORNER_DEG)) & (t4 >= CONC * np.minimum(t8, np.pi / 2))
        ok &= (cand - m >= lo[cand]) | (cand + m < hi[cand])       # some context on one side at least
        # and half a bead's path on both: a jog 1-3 nodes from a path end isn't a corner
        ok &= (cand - m // 2 >= lo[cand]) & (cand + m // 2 < hi[cand])
        conf, sharp = cand[ok], t8[ok]
        keep = []
        for q, t in zip(conf.tolist(), sharp.tolist()):           # one corner per 3 nodes: the sharper
            if keep and q - keep[-1][0] <= 3 and lo[q] == lo[keep[-1][0]]:
                if t > keep[-1][1]:
                    keep[-1] = (q, t)
                continue
            keep.append((q, t))
        cuts = np.array([q for q, _ in keep], np.int64)
        L._cache["node_corner"][node[cuts]] = True
    # each node's piece: from the last cut at or before it to the next cut after it (a cut
    # node starts the next piece and also closes the previous one)
    mark = np.full(N, -1, np.int64)
    mark[cuts] = cuts
    prev = np.maximum(np.maximum.accumulate(mark), lo)
    mark2 = np.full(N, N, np.int64)
    mark2[cuts] = cuts
    nxt = np.minimum.accumulate(mark2[::-1])[::-1]
    # the next cut strictly after a cut node
    after = np.full(N, N, np.int64)
    after[:-1] = nxt[1:]
    nxt = np.where(mark >= 0, after, nxt)
    nxt = np.minimum(nxt, hi - 1)
    a = np.maximum(i - h, prev)
    b = np.minimum(i + h, nxt)
    phi = _ls_dirs(S, a, b) % np.pi
    sel = own & ~np.isnan(phi)
    ang[node[sel]] = phi[sel]
    L._cache["node_angle"] = ang
    return ang


def node_corners(L) -> np.ndarray:
    """Per grid node (bool, H*W): a corner node_angles found (its fits stop there)."""
    if "node_corner" not in L._cache or "node_angle" not in L._cache:
        L._cache.pop("node_angle", None)
        node_angles(L)
    return L._cache["node_corner"]


def cell_moments(L) -> np.ndarray:
    """(H*W, 5) per material cell: number of covering nodes, and the mean over them of
    cos 2a, sin 2a, cos 4a, sin 4a (lone nodes count as 0 and 90 degrees half each:
    0 for the 2a terms, 1 for cos 4a) (cached)."""
    out = L._cache.get("cell_moments")
    if out is not None:
        return out
    from rltp.cover.beads import cover_sum
    ang = node_angles(L).reshape(L.H, L.W)
    m = L.mask
    lone = m & np.isnan(ang)
    a = np.where(m & ~lone, ang, 0.0)
    on = (m & ~lone).astype(float)
    k = L.kb
    cnt = cover_sum(m.astype(float), k)
    sums = [cover_sum(on * f, k) for f in (np.cos(2 * a), np.sin(2 * a), np.cos(4 * a), np.sin(4 * a))]
    sums[2] = sums[2] + cover_sum(lone.astype(float), k)
    nz = np.where(cnt > 0, cnt, 1.0)
    out = np.stack([cnt] + [s / nz for s in sums], -1).reshape(-1, 5)
    L._cache["cell_moments"] = out
    return out


def cell_bins(L) -> tuple[np.ndarray, np.ndarray]:
    """(cells, bins): every (material cell, quantised direction) pair a covering node puts
    there, unique, sorted by cell (cached). A lone node puts both axes."""
    out = L._cache.get("cell_bins")
    if out is not None:
        return out
    from rltp.cover.beads import offsets
    H, W, k = L.H, L.W, L.kb
    ang = node_angles(L)
    nodes = np.flatnonzero(L.flat_mask)
    a = ang[nodes]
    lone = np.isnan(a)
    b = np.where(lone, 0, np.rint(np.nan_to_num(a) / np.pi * NBINS).astype(np.int64) % NBINS)
    nodes = np.concatenate([nodes, nodes[lone]])
    b = np.concatenate([b, np.full(int(lone.sum()), NBINS // 2, np.int64)])
    r, c = np.divmod(nodes, W)
    lo_, hi_ = offsets(k)
    keys = []
    for dr in range(lo_, hi_ + 1):
        for dc in range(lo_, hi_ + 1):
            rr, cc = r + dr, c + dc
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            keys.append((rr[ok] * W + cc[ok]) * NBINS + b[ok])
    u = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    cell, b = u // NBINS, u % NBINS
    # one direction per cluster: neighbouring nodes of one bead fall in adjacent bins (fit
    # noise), which would multiply the pairs; a cell keeps a bin only when it lies more than
    # MERGE bins past the previous one (sorted), so a chain of close bins counts once, and
    # drops its last bin when that wraps round to within MERGE of its first (0 = 180 degrees)
    first = np.r_[True, cell[1:] != cell[:-1]]
    keep = first.copy()
    if len(b) > 1:
        keep[1:] |= (b[1:] - b[:-1]) > MERGE
        start = np.flatnonzero(first)
        end = np.r_[start[1:], len(b)] - 1
        multi = end > start
        wrap = multi & (b[start] + NBINS - b[end] <= MERGE) & keep[end]
        keep[end[wrap]] = False
    out = (cell[keep], b[keep])
    L._cache["cell_bins"] = out
    return out


def bin_angle(b) -> np.ndarray:
    return np.asarray(b, float) * np.pi / NBINS
