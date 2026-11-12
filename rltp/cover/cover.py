"""Covers: nxt/prv link arrays, invariants, stroke walking and the low-level edits.

A cover links each in-part cell to at most one successor and one predecessor,
so every connected piece is an open path or a closed loop (a stroke).
Invariants:
  * out-of-part cells have no links;
  * links join only 4-adjacent in-part cells, and nxt[prv[i]] == i;
  * no 2-cycles (a -> b -> a);
  * each closed loop holds exactly one seam cell; paths hold none.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rltp.model import Layer


@dataclass
class Pieces:
    label: np.ndarray        # int32 (n,), -1 outside the part
    cells: list              # list[np.ndarray] in print order: head..tail, or seam..prv[seam]
    is_loop: np.ndarray      # bool per piece
    length: np.ndarray       # int per piece

    @property
    def count(self) -> int:
        return len(self.cells)


def neighbors4(i: int, H: int, W: int):
    r, c = divmod(i, W)
    if c + 1 < W:
        yield i + 1
    if r + 1 < H:
        yield i + W
    if c > 0:
        yield i - 1
    if r > 0:
        yield i - W


def adjacent(i: int, j: int, W: int) -> bool:
    d = j - i
    if d == W or d == -W:
        return True
    if d == 1:
        return j % W != 0
    if d == -1:
        return i % W != 0
    return False


# ---------------------------------------------------------------- walking

def walk_pieces(L: Layer) -> Pieces:
    n = L.n
    m = L.flat_mask
    nxt = L.nxt.tolist()
    prv = L.prv.tolist()
    label = [-1] * n
    cells_out, loops = [], []

    heads = np.nonzero(m & (L.prv == -1))[0].tolist()
    for h in heads:
        lab = len(cells_out)
        seq = []
        c = h
        while c != -1:
            if label[c] != -1:
                raise ValueError(f"cover corrupt: cell {c} reached twice")
            label[c] = lab
            seq.append(c)
            c = nxt[c]
        cells_out.append(seq)
        loops.append(False)

    def walk_loop(s):
        lab = len(cells_out)
        seq = []
        c = s
        while True:
            if c == -1:
                raise ValueError(f"cover corrupt: loop through {s} is open")
            label[c] = lab
            seq.append(c)
            c = nxt[c]
            if c == s:
                break
            if label[c] != -1:
                raise ValueError(f"cover corrupt: cell {c} reached twice")
        cells_out.append(seq)
        loops.append(True)

    for s in sorted(L.seam):
        if 0 <= s < n and m[s] and label[s] == -1 and prv[s] != -1:
            walk_loop(s)
    rest = np.nonzero(m)[0].tolist()
    for s in rest:
        if label[s] == -1:
            walk_loop(s)

    return Pieces(
        np.asarray(label, np.int32),
        [np.asarray(s, np.int64) for s in cells_out],
        np.asarray(loops, bool),
        np.asarray([len(s) for s in cells_out], np.int64),
    )


def get_pieces(L: Layer) -> Pieces:
    pc = L._cache.get("pieces")
    if pc is None:
        pc = walk_pieces(L)
        L._cache["pieces"] = pc
    return pc


def stroke_cells(L: Layer, cell: int):
    """Cells of the stroke through `cell` in print order, and whether it is a loop.

    Loops start at their seam when they have one, else at `cell`.
    """
    nxt, prv = L.nxt, L.prv
    c = cell
    loop = False
    while True:
        p = int(prv[c])
        if p == -1:
            head = c
            break
        if p == cell:
            head = cell
            loop = True
            break
        c = p
    out = [head]
    c = int(nxt[head])
    while c != -1 and c != head:
        out.append(c)
        c = int(nxt[c])
    if loop:
        for k, x in enumerate(out):
            if x in L.seam:
                out = out[k:] + out[:k]
                break
    return out, loop


# ---------------------------------------------------------------- invariants

def check_invariants(L: Layer) -> list[str]:
    errs = []
    n, W = L.n, L.W
    m = L.flat_mask
    nxt, prv = L.nxt, L.prv
    if nxt.shape != (n,) or prv.shape != (n,):
        return ["nxt/prv have the wrong shape"]
    if (nxt < -1).any() or (nxt >= n).any() or (prv < -1).any() or (prv >= n).any():
        return ["link index out of range"]
    if (nxt[~m] != -1).any() or (prv[~m] != -1).any():
        errs.append("out-of-part cell has a link")
    for name, a, b in (("nxt", nxt, prv), ("prv", prv, nxt)):
        i = np.nonzero(m & (a >= 0))[0]
        j = a[i]
        if not m[j].all():
            errs.append(f"{name} link leaves the part")
        d = j - i
        ok = (np.abs(d) == W) | ((d == 1) & (j % W != 0)) | ((d == -1) & (i % W != 0))
        if not ok.all():
            errs.append(f"{name} link joins non-adjacent cells, e.g. {int(i[~ok][0])}->{int(j[~ok][0])}")
        if (b[j] != i).any():
            errs.append(f"{name}/{'prv' if name == 'nxt' else 'nxt'} not symmetric")
        if name == "nxt" and (nxt[j] == i).any():
            errs.append("2-cycle")
    if errs:
        return errs
    for s in L.seam:
        if not (0 <= s < n) or not m[s]:
            errs.append(f"seam {s} outside part")
    if errs:
        return errs
    try:
        pc = walk_pieces(L)
    except ValueError as e:
        return [str(e)]
    if (pc.label[m] < 0).any():
        errs.append("in-part cell not on any stroke")
    seam_arr = np.zeros(n, bool)
    seam_arr[list(L.seam)] = True
    for lab, cells in enumerate(pc.cells):
        ns = int(seam_arr[cells].sum())
        if pc.is_loop[lab] and ns != 1:
            errs.append(f"loop with {ns} seams")
        if not pc.is_loop[lab] and ns != 0:
            errs.append("seam on an open path")
    return errs


def assert_valid(L: Layer) -> None:
    errs = check_invariants(L)
    if errs:
        raise AssertionError("invalid cover: " + "; ".join(errs[:5]))


# ---------------------------------------------------------------- edits (in place)

def link(L: Layer, a: int, b: int) -> None:
    assert L.nxt[a] == -1 and L.prv[b] == -1, (a, b)
    L.nxt[a] = b
    L.prv[b] = a


def unlink_cell(L: Layer, a: int) -> list[int]:
    """Cut both links at `a`; returns the former neighbours."""
    out = []
    b = int(L.nxt[a])
    if b != -1:
        L.nxt[a] = -1
        L.prv[b] = -1
        out.append(b)
    p = int(L.prv[a])
    if p != -1:
        L.prv[a] = -1
        L.nxt[p] = -1
        out.append(p)
    return out


def reverse_cells(L: Layer, cells) -> None:
    idx = np.asarray(cells, np.int64)
    a = L.nxt[idx].copy()
    L.nxt[idx] = L.prv[idx]
    L.prv[idx] = a


def reverse_stroke(L: Layer, cell: int) -> list[int]:
    cells, _ = stroke_cells(L, cell)
    reverse_cells(L, cells)
    return cells


def linked(L: Layer, a: int, b: int) -> bool:
    return L.nxt[a] == b or L.prv[a] == b


def is_end(L: Layer, a: int) -> bool:
    return L.nxt[a] == -1 or L.prv[a] == -1


def join_ends(L: Layer, x: int, y: int) -> bool:
    """Link adjacent stroke ends x and y, reversing the shorter stroke if their
    directions clash. Joining a path's own head and tail closes it into a loop
    (only when it has >= 3 cells, so no 2-cycles). Returns False if impossible."""
    nxt, prv = L.nxt, L.prv
    if x == y or linked(L, x, y) or not (is_end(L, x) and is_end(L, y)):
        return False
    if not (nxt[x] == -1 and prv[y] == -1) and not (nxt[y] == -1 and prv[x] == -1):
        sx, _ = stroke_cells(L, x)
        if y in set(sx):
            return False
        sy, _ = stroke_cells(L, y)
        reverse_cells(L, sy if len(sy) <= len(sx) else sx)
    if nxt[x] == -1 and prv[y] == -1:
        a, b = x, y
    elif nxt[y] == -1 and prv[x] == -1:
        a, b = y, x
    else:  # pragma: no cover - unreachable for valid covers
        return False
    link(L, a, b)
    return True


def fix_seams_local(L: Layer, cells, prefer=()) -> None:
    """Restore the seam invariant on every stroke through `cells`."""
    m = L.flat_mask
    done: set = set()
    for c in cells:
        c = int(c)
        if c in done or not m[c]:
            continue
        sc, loop = stroke_cells(L, c)
        ss = set(sc)
        done |= ss
        seams_in = [x for x in sc if x in L.seam]
        if loop:
            if not seams_in:
                pick = next((p for p in prefer if p in ss), min(sc))
                L.seam.add(int(pick))
            elif len(seams_in) > 1:
                keep = next((p for p in prefer if p in seams_in), min(seams_in))
                for s in seams_in:
                    if s != keep:
                        L.seam.discard(s)
        else:
            for s in seams_in:
                L.seam.discard(s)


def normalize_seams(L: Layer) -> None:
    fix_seams_local(L, np.nonzero(L.flat_mask)[0])


# ---------------------------------------------------------------- derived maps

def bead_weights(L: Layer) -> np.ndarray:
    """Fraction of each cell's bead running along local x (0 deg); the rest runs along y.

    Straight cells are 1 or 0, corner cells 0.5 (they carry both directions),
    single-cell strokes 0.5."""
    n, W = L.n, L.W
    i = np.arange(n)
    nxt, prv = L.nxt, L.prv

    def v_of(j):
        return (j >= 0) & (np.abs(j - i) == W)

    def h_of(j):
        return (j >= 0) & (np.abs(j - i) == 1)

    hz = h_of(nxt) | h_of(prv)
    vt = v_of(nxt) | v_of(prv)
    return np.where(hz & ~vt, 1.0, np.where(vt & ~hz, 0.0, 0.5))


def end_mask(L: Layer) -> np.ndarray:
    """Path ends (heads, tails, single cells)."""
    return L.flat_mask & ((L.nxt == -1) | (L.prv == -1))


def seam_mask(L: Layer) -> np.ndarray:
    s = np.zeros(L.n, bool)
    if L.seam:
        s[list(L.seam)] = True
    return s


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """In-part cells with a 4-neighbour outside the part (or at the grid edge)."""
    p = np.pad(mask, 1, constant_values=False)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return (mask & ~inner).reshape(-1)
