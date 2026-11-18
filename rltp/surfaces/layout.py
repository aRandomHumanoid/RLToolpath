"""Lay ribbons into a part's infill layers as bead paths on the fine bead grid.

Per infill layer (`lay`):
1. **Clear** the layer's infill beads; walls and solid skin stay.
2. **Rasterize** each ribbon into a 4-connected chain of centreline nodes (diagonal steps get
   the corner cell nearer the curve) and keep the nodes that are
   - inside the infill region (a bead clear of the walls and skin, off the solid skin), so
     open ribbons end against the walls;
   - at least `clearance` (kb - 1 cells: beads may overlap by a quarter, which the coverage
     term charges) from every bead laid before them, and from their own ribbon further
     along (a fold back onto itself). Where a ribbon crosses or crowds an
     earlier one it is cut, and each cut end touches that bead: a T-joint, anchored, never
     a free end. Crossings are therefore printed as two T-joints; a later layer can lay the
     ribbons in the other order (`layout_snapshot(alternate=True)`) so the cut alternates
     and crossings don't stack.
   Runs shorter than `min_run` nodes are dropped.
3. **Merge** path ends during toolpath generation: every open end is first pulled back a bead
   (a run meeting the wall at a slant otherwise walls off one side of its own end), then ends
   are joined by connectors, cheapest first (Dijkstra within `link_reach`), never closing a
   path on itself and never crossing another end or running within a bead of another
   connector. A connector runs along the free
   region's edge (the band just inside the walls, or beside a fixed ribbon; cost 1 per step)
   or through open space a bead clear of every other bead (INTERIOR_COST per step), so ends on
   a hole's edge can merge with ends on the outline's. Beside parallel ribbons this gives the
   familiar zigzag; with curved ribbons, whatever pairs are nearest.
4. Anchor any end that still touches nothing (rltp.cover.beads.anchor_ends), fix seams.

`layout_snapshot` lays one template per distinct infill-layer geometry (two with `alternate`)
and copies it to every layer sharing that geometry: vertical ribbons. Given a function of
height instead of a list (3D ribbons, rltp.surfaces.ribbons.Warp3D), every infill layer is laid
from the ribbons' cut at its own mid-height.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from rltp.cover.beads import anchor_ends, cover_sum, free_ends
from rltp.cover.cover import join_ends, normalize_seams
from rltp.model import Layer
from rltp.surfaces.ribbons import Ribbon

LINK_REACH_MM = 4.0      # longest connector, mm (interior steps count INTERIOR_COST)
LINK_PASSES = 1          # merge passes: later ones route still-open ends around laid connectors


def clearance(L: Layer) -> float:
    """How close (cells, Euclidean) a ribbon node may come to another bead's node: kb - 1, so
    neighbouring beads overlap by at most a quarter of a width."""
    return L.kb - 1.0


@dataclass
class LayStats:
    pieces: int = 0          # bead runs laid from ribbons
    loops: int = 0           # closed ribbons laid whole
    links: int = 0           # connectors that merged two runs
    anchored: int = 0        # ends extended to touch a bead
    nodes: int = 0           # centreline nodes laid (connectors included)
    extra: dict = field(default_factory=dict)


def is_infill_layer(L: Layer) -> bool:
    return L.part is not None and L.kb > 1 and bool((L.part & ~L.req).any())


def _solid(L: Layer) -> np.ndarray:
    """Solid-skin cells of a sparse layer (required, not the walls' band)."""
    walls_band = cover_sum(L.lock, L.kb) > 0 if L.lock is not None else np.zeros_like(L.part)
    return L.req & ~walls_band


def kept_nodes(L: Layer) -> np.ndarray:
    """The nodes laying ribbons keeps (H x W bool): walls and solid-skin lines."""
    lock = L.lock if L.lock is not None else np.zeros_like(L.mask)
    return L.mask & (lock | (L.aux["inner"] & _solid(L)))


def infill_region(L: Layer) -> np.ndarray:
    """Node cells infill beads may use (H x W bool): the bead's footprint inside the part, off
    the solid skin, and at least kb - 1 cells (Euclidean) from the wall and
    skin nodes, the same clearance ribbons keep from each other. A ribbon cut there touches
    the wall: a T-joint."""
    clear = ndimage.distance_transform_edt(~kept_nodes(L)) >= clearance(L)
    return L.aux["region"] & ~_solid(L) & clear


def _clear(L: Layer) -> None:
    """Remove every node but the walls and skin lines, cutting the links into them."""
    m = L.flat_mask
    gone = m & ~kept_nodes(L).reshape(-1)
    for arr in (L.nxt, L.prv):
        tgt = arr >= 0
        into = np.zeros_like(tgt)
        into[tgt] = gone[arr[tgt]]
        arr[into & ~gone] = -1
    L.nxt[gone] = -1
    L.prv[gone] = -1
    L.mask.reshape(-1)[gone] = False
    L.seam = {s for s in L.seam if not gone[s]}
    L.invalidate()


def rasterize(xy: np.ndarray, frame, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the 4-connected chain of node cells along polyline `xy` (world mm), in
    the layer's node frame. Cells may fall outside the grid (callers check)."""
    u, v = frame.to_local(xy[:, 0], xy[:, 1])
    u = np.asarray(u, float) / frame.p
    v = np.asarray(v, float) / frame.p
    if len(u) == 1:
        return np.floor(v).astype(np.int64), np.floor(u).astype(np.int64)
    du, dv = np.diff(u), np.diff(v)
    n = np.maximum(np.ceil(np.hypot(du, dv) * 3).astype(np.int64), 1)     # 3 samples per cell
    seg = np.repeat(np.arange(len(n)), n)
    j = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = j / n[seg]
    su = np.r_[u[seg] + t * du[seg], u[-1]]
    sv = np.r_[v[seg] + t * dv[seg], v[-1]]
    r = np.floor(sv).astype(np.int64)
    c = np.floor(su).astype(np.int64)
    keep = np.r_[True, (np.diff(r) != 0) | (np.diff(c) != 0)]
    r, c, su, sv = r[keep], c[keep], su[keep], sv[keep]
    dr, dc = np.diff(r), np.diff(c)
    i = np.flatnonzero((dr != 0) & (dc != 0))           # a diagonal step: add the nearer corner
    if i.size:
        mu, mv = 0.5 * (su[i] + su[i + 1]), 0.5 * (sv[i] + sv[i + 1])
        da = (c[i + 1] + 0.5 - mu) ** 2 + (r[i] + 0.5 - mv) ** 2      # corner (r_i, c_i+1)
        db = (c[i] + 0.5 - mu) ** 2 + (r[i + 1] + 0.5 - mv) ** 2      # corner (r_i+1, c_i)
        a = da <= db
        r = np.insert(r, i + 1, np.where(a, r[i], r[i + 1]))
        c = np.insert(c, i + 1, np.where(a, c[i + 1], c[i]))
    return r, c


def _runs(keep: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(keep)
    if not idx.size:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


def _chain(rb: Ribbon, L: Layer, region: np.ndarray, placed: np.ndarray, thr: float,
           min_run: int) -> list[tuple[np.ndarray, bool]]:
    """The runs (flat node cells, is_loop) ribbon `rb` lays in L, given the nodes already laid."""
    H, W, k = L.H, L.W, L.kb
    r, c = rasterize(rb.points(), L.frame, H, W)
    if rb.closed and len(r) > 1 and r[-1] == r[0] and c[-1] == c[0]:
        r, c = r[:-1], c[:-1]
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    cell = np.where(ok, r * W + c, 0)
    keep = ok & region.reshape(-1)[cell]
    first = np.zeros(len(cell), bool)
    first[np.unique(cell, return_index=True)[1]] = True
    keep &= first                                          # a cell visited twice: the first visit
    pts = np.c_[r, c].astype(float)
    if len(placed):
        d, _ = cKDTree(placed).query(pts, distance_upper_bound=thr)
        keep &= ~(d < thr)
    pairs = cKDTree(pts).query_pairs(thr, output_type="ndarray")
    if len(pairs):
        gap = np.abs(pairs[:, 0] - pairs[:, 1])
        if rb.closed:
            gap = np.minimum(gap, len(pts) - gap)
        far = pairs[gap > 3 * k]
        keep[np.maximum(far[:, 0], far[:, 1])] = False     # a fold back onto itself: cut it
    if rb.closed and keep.all() and len(keep) >= 4:
        return [(cell, True)]
    if rb.closed and not keep.all():                       # start at a gap, so no run wraps
        s = int(np.flatnonzero(~keep)[0])
        cell, keep = np.roll(cell, -s), np.roll(keep, -s)
    return [(cell[run], False) for run in _runs(keep) if len(run) >= min_run]


def _join(L: Layer, a: int, b: int, inner: list[int]) -> bool:
    """Join path ends a and b through connector nodes `inner` (a 4-connected path between them)."""
    if not inner:
        return join_ends(L, a, b)
    m = L.mask.reshape(-1)
    path = np.asarray(inner, np.int64)
    m[path] = True
    L.nxt[path[:-1]] = path[1:]
    L.prv[path[1:]] = path[:-1]
    L.invalidate()
    if not join_ends(L, a, int(path[0])):
        L.nxt[path] = -1
        L.prv[path] = -1
        m[path] = False
        L.invalidate()
        return False
    L.invalidate()
    ok = join_ends(L, int(path[-1]), b)
    L.invalidate()
    return ok


INTERIOR_COST = 1.5      # a connector step through open infill space, against 1 along the edge band


def _merge(L: Layer, pieces: list, band: np.ndarray, region: np.ndarray, thr: float, reach: int,
           passes: int = 1) -> int:
    """Merge open path ends with connectors (see the module docstring). Connectors run along
    `band` (cost 1 per step) or through open space inside `region` a bead clear of every other
    bead (INTERIOR_COST per step), so ends on different edges (a hole's and the outline's) can
    merge too; each accepted connector keeps later ones a bead away. With `passes` > 1, ends
    left open are routed again around the connectors already laid (until a pass adds none)."""
    W, k = L.W, L.kb
    m = L.flat_mask
    bflat = band.reshape(-1)
    # pull every open end back a bead (+1 node) first: a run that meets the edge at a slant
    # crosses the thin band beside its own end and would wall off one side of it; the gap
    # gives connectors a way past. Ends left unmerged are re-extended by anchoring afterwards.
    cut = k + 1
    for i, (cells, loop) in enumerate(pieces):
        if loop or len(cells) < 2 * cut + 2 * k:
            continue
        L.prv[cells[cut]] = -1
        L.nxt[cells[-cut - 1]] = -1
        gone = np.r_[cells[:cut], cells[-cut:]]
        L.nxt[gone] = -1
        L.prv[gone] = -1
        m[gone] = False
        pieces[i] = (cells[cut:-cut], False)
    L.invalidate()
    end_piece: dict[int, int] = {}
    zone = np.zeros(m.size, bool)                 # each end's own last nodes don't block it
    for i, (cells, loop) in enumerate(pieces):
        if loop:
            continue
        for e in {int(cells[0]), int(cells[-1])}:
            end_piece[e] = i
        zone[cells[: 2 * k + 1]] = True
        zone[cells[-(2 * k + 1):]] = True
    if len(end_piece) < 2:
        return 0
    ends = np.array(sorted(end_piece), np.int64)
    endset = set(ends.tolist())
    parent = list(range(len(pieces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used: set = set()
    blocked = np.zeros(m.size, bool)              # within a bead of an accepted connector
    rad = int(np.ceil(thr))
    links = 0
    for _ in range(max(int(passes), 1)):
        # pass 1 routes every pair at once and skips a pair whose route an earlier connector
        # took or crowds; later passes route the ends still open around the connectors laid
        live = np.array([e for e in ends.tolist() if e not in used], np.int64)
        if len(live) < 2:
            break
        clear = ndimage.distance_transform_edt(~(L.mask & ~zone.reshape(L.mask.shape))) >= thr
        interior = (clear & region).reshape(-1) & ~bflat
        nodes = (bflat | interior) & ~m & ~blocked
        nodes[live] = True
        cells = np.flatnonzero(nodes)
        nid = np.full(m.size, -1, np.int64)
        nid[cells] = np.arange(len(cells))
        cost = np.where(interior[cells], INTERIOR_COST, 1.0)
        r, c = np.divmod(cells, W)
        right = cells[(c + 1 < W)]
        right = right[nodes[right + 1]]
        down = cells[(cells + W) < m.size]
        down = down[nodes[down + W]]
        src = np.r_[nid[right], nid[down]]
        dst = np.r_[nid[right + 1], nid[down + W]]
        wgt = 0.5 * (cost[src] + cost[dst])
        G = coo_matrix((wgt, (src, dst)), shape=(len(cells), len(cells))).tocsr()
        D, P = dijkstra(G, directed=False, indices=nid[live], limit=reach, return_predecessors=True)
        De = D[:, nid[live]]
        ia, ib = np.triu_indices(len(live), 1)
        d = De[ia, ib]
        fin = np.isfinite(d)
        order = np.argsort(d[fin], kind="stable")
        ia, ib = ia[fin][order], ib[fin][order]
        added = 0
        for a, b in zip(ia.tolist(), ib.tolist()):
            ea, eb = int(live[a]), int(live[b])
            if ea in used or eb in used:
                continue
            pa, pb = find(end_piece[ea]), find(end_piece[eb])
            if pa == pb:
                continue
            path = []
            x = int(nid[eb])
            src_ = int(nid[ea])
            while x != src_ and x >= 0:
                path.append(int(cells[x]))
                x = int(P[a, x])
            if x < 0:
                continue
            inner = path[1:][::-1]                       # strictly between ea and eb, from ea's side
            if any(y in endset or y in used for y in inner) or (inner and blocked[inner].any()):
                continue
            if _join(L, ea, eb, inner):
                links += 1
                added += 1
                parent[pa] = pb
                used.update((ea, eb))
                used.update(inner)
                for y in inner:
                    ry, cy = divmod(y, W)
                    r0, r1 = max(ry - rad, 0), min(ry + rad + 1, L.H)
                    c0, c1 = max(cy - rad, 0), min(cy + rad + 1, W)
                    blocked.reshape(L.H, W)[r0:r1, c0:c1] = True
                blocked[[ea, eb]] = False
        if not added:
            break
    return links


def lay(L: Layer, ribbons: list[Ribbon], min_run: int | None = None,
        link_reach: int | None = None, link_passes: int | None = None) -> LayStats:
    """Replace L's infill with `ribbons` (fixed ones first, then the rest in order); see the
    module docstring. Modifies L in place."""
    k = L.kb
    _clear(L)
    region = infill_region(L)
    thr = clearance(L)
    min_run = 2 * k if min_run is None else min_run
    reach = int(round(LINK_REACH_MM / L.frame.p)) if link_reach is None else link_reach
    placed = np.zeros((0, 2))
    pieces: list = []
    fixed = [rb for rb in ribbons if rb.fixed]
    free = [rb for rb in ribbons if not rb.fixed]
    band = None
    for group in (fixed, free):
        if group is free:                             # where free ribbons end: the region's edge,
            avail = region.copy()                     # or a bead clear of a fixed ribbon
            if len(placed):
                fm = np.zeros(region.shape, bool)
                fm[placed[:, 0].astype(int), placed[:, 1].astype(int)] = True
                avail &= ndimage.distance_transform_edt(~fm) >= thr
            band = avail & ~ndimage.binary_erosion(avail, np.ones((3, 3), bool), border_value=0)
        for rb in group:
            got = _chain(rb, L, region, placed, thr, min_run)
            for cells, loop in got:
                pieces.append((cells, loop))
                placed = np.vstack([placed, np.c_[np.divmod(cells, L.W)]])
    m = L.mask.reshape(-1)
    for cells, loop in pieces:
        m[cells] = True
        L.nxt[cells[:-1]] = cells[1:]
        L.prv[cells[1:]] = cells[:-1]
        if loop:
            L.nxt[cells[-1]] = cells[0]
            L.prv[cells[0]] = cells[-1]
            L.seam.add(int(cells[0]))
    L.invalidate()
    st = LayStats(pieces=len(pieces), loops=sum(1 for _, lp in pieces if lp))
    if band is not None and reach > 0:
        st.links = _merge(L, pieces, band, avail, thr, reach,
                          LINK_PASSES if link_passes is None else link_passes)
    L.invalidate()
    st.anchored = len(anchor_ends(L, free_ends(L)))
    normalize_seams(L)
    L.invalidate()
    st.nodes = int(L.mask.sum() - (L.lock.sum() if L.lock is not None else 0))
    return st


def _groups(part) -> list[list[int]]:
    """Infill layers grouped by identical geometry (part, required cells)."""
    groups: dict = {}
    for k, L in enumerate(part.layers):
        if is_infill_layer(L):
            key = hash((L.part.tobytes(), L.req.tobytes()))
            groups.setdefault(key, []).append(k)
    return list(groups.values())


def _reverse_free(ribbons: list[Ribbon]) -> list[Ribbon]:
    return [rb for rb in ribbons if rb.fixed] + [rb for rb in ribbons if not rb.fixed][::-1]


def _templates(part, ribbons, alternate: bool, **kw):
    """[(template layer, LayStats, layer indices it applies to)]: one per infill geometry (two
    with `alternate`: odd layers of a group lay the free ribbons in reverse order, so crossings
    are cut the other way). `ribbons` may instead be a pair (even layers' ribbons, odd layers'
    ribbons: two families alternating by layer, e.g. lines at 0 and 90 degrees), or a function
    of height (3D surfaces, e.g. Warp3D.apply): every infill layer is then laid on its own, from
    the ribbons' cut at its mid-height."""
    out = []
    if isinstance(ribbons, tuple):            # families rotating through each group's layers: layer i
        nf = len(ribbons)                     # gets family i mod len (a pair alternates)
        for ks in _groups(part):
            for parity in range(nf):
                sel = [k for i, k in enumerate(ks) if i % nf == parity]
                if sel:
                    T = part.layers[sel[0]].copy()
                    out.append((T, lay(T, ribbons[parity], **kw), sel))
        return out
    if callable(ribbons):
        ks = [k for k, L in enumerate(part.layers) if is_infill_layer(L)]
        for i, k in enumerate(ks):
            L = part.layers[k]
            T = L.copy()
            rbs = ribbons(L.z - 0.5 * L.h)
            out.append((T, lay(T, _reverse_free(rbs) if alternate and i % 2 else rbs, **kw), [k]))
        return out
    for ks in _groups(part):
        for parity in ((0, 1) if alternate else (0,)):
            sel = [k for i, k in enumerate(ks) if not alternate or i % 2 == parity]
            if not sel:
                continue
            T = part.layers[sel[0]].copy()
            out.append((T, lay(T, _reverse_free(ribbons) if parity else ribbons, **kw), sel))
    return out


def layout_snapshot(part, ribbons: list[Ribbon], alternate: bool = False, **kw):
    """A PartState snapshot with every infill layer laid from `ribbons` (see _templates), and
    the templates' LayStats. `part.restore(snap)` scores it."""
    snap = part.snapshot()
    stats = []
    for T, st, sel in _templates(part, ribbons, alternate, **kw):
        stats.append(st)
        for k in sel:
            snap[k] = (T.nxt.copy(), T.prv.copy(), set(T.seam), T.mask.copy())
    return snap, stats


def apply_layout(part, ribbons: list[Ribbon], alternate: bool = False, **kw) -> list[LayStats]:
    """Lay `ribbons` into every infill layer and re-score the part, as layout_snapshot +
    part.restore do, but faster: layers that get the same template share one derived-data cache
    (deposited material, fitted segments, direction mixes and stroke walks depend only on a
    layer's own paths and outline), layers the ribbons don't touch keep theirs, and PartState
    computes each shared layer's laminate columns and failure indices once. The support-rule
    context, which only edits read, is not rebuilt. About 3x faster than restore on the 5 mm
    parts (21 identical infill layers). Call unshare_caches(part) before editing any layer
    afterwards: it also rebuilds the support context."""
    stats = []
    for T, st, sel in _templates(part, ribbons, alternate, **kw):
        stats.append(st)
        shared: dict = {}
        for k in sel:
            L = part.layers[k]
            L.nxt[:] = T.nxt
            L.prv[:] = T.prv
            L.seam = set(T.seam)
            L.mask = T.mask.copy()
            L._cache = shared
    part.full_recheck()
    return stats


def unshare_caches(part) -> None:
    """Give every layer its own derived-data cache again and rebuild the support context (after
    apply_layout, before any edit)."""
    for L in part.layers:
        L._cache = dict(L._cache)
    part.refresh_support()
