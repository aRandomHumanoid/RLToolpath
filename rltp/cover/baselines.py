"""Baseline covers (zigzag, concentric) and the greedy repair pass."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from rltp.cover.cover import is_end, join_ends, link, neighbors4, normalize_seams, walk_pieces
from rltp.model import Layer


def _runs(line: np.ndarray):
    """(start, stop) index pairs of True runs in a 1D bool array."""
    d = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def zigzag_links(L: Layer, cells_mask: np.ndarray, angle: int = 0) -> None:
    """Link back-and-forth runs (rows for 0 deg, columns for 90 deg) inside
    `cells_mask` (bool H x W, a subset of the part, all cells currently unlinked)."""
    H, W = L.H, L.W
    if angle == 0:
        lines = range(H)
        for k, r in enumerate(lines):
            for a, b in _runs(cells_mask[r]):
                seq = [r * W + c for c in range(a, b)]
                if k % 2:
                    seq.reverse()
                for x, y in zip(seq[:-1], seq[1:]):
                    link(L, x, y)
    else:
        for k, c in enumerate(range(W)):
            for a, b in _runs(cells_mask[:, c]):
                seq = [r * W + c for r in range(a, b)]
                if k % 2:
                    seq.reverse()
                for x, y in zip(seq[:-1], seq[1:]):
                    link(L, x, y)


def repair(L: Layer, cells=None) -> int:
    """Greedily join 4-adjacent ends of different strokes until no join is possible.

    `cells` limits which ends start a join (their neighbours may lie anywhere).
    Never closes a stroke on itself, and never touches a sparse layer's wall cells
    (`L.lock`). Returns the number of joins."""
    pc = walk_pieces(L)
    parent = list(range(pc.count))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    lab = pc.label
    m = L.flat_mask
    if L.lock is not None:
        m = m & ~L.lock.reshape(-1)
    H, W = L.H, L.W
    if cells is None:
        cand = np.nonzero(m & ((L.nxt == -1) | (L.prv == -1)))[0].tolist()
    else:
        cand = sorted(int(c) for c in cells if m[c])
    joins = 0
    changed = True
    while changed:
        changed = False
        for x in cand:
            if not is_end(L, x):
                continue
            for y in neighbors4(x, H, W):
                if not m[y] or not is_end(L, y):
                    continue
                rx, ry = find(int(lab[x])), find(int(lab[y]))
                if rx == ry:
                    continue
                if join_ends(L, x, y):
                    parent[ry] = rx
                    joins += 1
                    changed = True
                    if not is_end(L, x):
                        break
    L.invalidate()
    return joins


def _trace_ring(ring: np.ndarray, W: int, H: int):
    """Split a ring of cells (flat bool) into 4-connected walks.

    Each walk is greedy with Warnsdorff's rule (go to the neighbour with fewest
    unvisited ring neighbours), which follows one-cell-wide bands cleanly."""
    left = set(np.nonzero(ring)[0].tolist())
    walks = []
    while left:
        start = min(left)
        left.discard(start)
        seq = [start]
        cur = start
        while True:
            cands = [y for y in neighbors4(cur, H, W) if y in left]
            if not cands:
                break
            nxt = min(cands, key=lambda y: (sum(1 for z in neighbors4(y, H, W) if z in left and z != y), y))
            left.discard(nxt)
            seq.append(nxt)
            cur = nxt
        walks.append(seq)
    return walks


def link_ring(L: Layer, ring: np.ndarray) -> None:
    """Trace a ring of cells (flat bool, all unlinked) into a loop where it closes
    cleanly, otherwise into short strokes."""
    H, W = L.H, L.W
    for seq in _trace_ring(ring, W, H):
        for x, y in zip(seq[:-1], seq[1:]):
            link(L, x, y)
        if len(seq) >= 4 and (seq[0] - seq[-1]) in (1, -1, W, -W):
            a, b = seq[-1], seq[0]
            if abs(a - b) == W or (a // W == b // W):
                link(L, a, b)
                L.seam.add(seq[0])


def rings(mask: np.ndarray, n: int | None = None) -> list[np.ndarray]:
    """Peel `mask` (bool H x W) ring by ring (8-neighbour erosion); the first n rings."""
    out = []
    remaining = mask.copy()
    st = np.ones((3, 3), bool)
    while remaining.any() and (n is None or len(out) < n):
        inner = ndimage.binary_erosion(remaining, structure=st, border_value=0)
        out.append(remaining & ~inner)
        remaining = inner
    return out


def concentric(L: Layer) -> None:
    """Peel the mask's outer ring of cells repeatedly; trace each ring into a
    loop where it closes cleanly, otherwise into short strokes."""
    for ring in rings(L.mask):
        link_ring(L, ring.reshape(-1))
    L.invalidate()


def fill_baseline(L: Layer, kind: str = "zigzag") -> Layer:
    """Fill an empty layer with a valid cover, then run the repair pass."""
    L.nxt[:] = -1
    L.prv[:] = -1
    L.seam.clear()
    if kind in ("zigzag", "zigzag0"):
        zigzag_links(L, L.mask, 0)
    elif kind == "zigzag90":
        zigzag_links(L, L.mask, 90)
    elif kind == "concentric":
        concentric(L)
    else:
        raise ValueError(f"unknown baseline {kind!r}")
    repair(L)
    normalize_seams(L)
    L.invalidate()
    return L


BASELINES = ("zigzag", "zigzag90", "concentric")
