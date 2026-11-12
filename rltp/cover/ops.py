"""Edit operators. Every operator maps a valid cover to another valid cover.

Actions are flattened operator-major: a = op * (H * W) + cell.
"""
from __future__ import annotations

import numpy as np

from rltp.cover.baselines import repair, zigzag_links
from rltp.cover.cover import (
    fix_seams_local, get_pieces, is_end, join_ends, linked, neighbors4, reverse_stroke,
    seam_mask, stroke_cells, unlink_cell,
)
from rltp.model import Layer

OPS = ("reverse", "merge", "split", "flip_right", "flip_down", "move_seam",
       "patch5_0", "patch5_90", "patch9_0", "patch9_90")
N_OPS = len(OPS)
OP_INDEX = {name: i for i, name in enumerate(OPS)}
PATCHES = {6: (5, 0), 7: (5, 90), 8: (9, 0), 9: (9, 90)}


class InvalidEdit(Exception):
    pass


def decode(a: int, n: int) -> tuple[int, int]:
    return int(a) // n, int(a) % n


def encode(op: int, cell: int, n: int) -> int:
    return int(op) * n + int(cell)


def ops_to_indices(ops) -> list[int]:
    if ops is None or ops == "all":
        return list(range(N_OPS))
    out = []
    for o in ops:
        if isinstance(o, int):
            out.append(o)
        elif o in ("flip", "patch"):
            out += [i for i, nm in enumerate(OPS) if nm.startswith(o)]
        else:
            out.append(OP_INDEX[o])
    return sorted(set(out))


# ---------------------------------------------------------------- masks

def _box_any(mask: np.ndarray, s: int) -> np.ndarray:
    """For each centre cell: does the s x s window around it hold any part cell?"""
    H, W = mask.shape
    hlf = s // 2
    ii = np.zeros((H + 1, W + 1), np.int64)
    ii[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    r = np.arange(H)[:, None]
    c = np.arange(W)[None, :]
    r0, r1 = np.clip(r - hlf, 0, H), np.clip(r + hlf + 1, 0, H)
    c0, c1 = np.clip(c - hlf, 0, W), np.clip(c + hlf + 1, 0, W)
    tot = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
    return (tot > 0).reshape(-1)


def link_grids(L: Layer):
    """hl[r, c]: (r,c)-(r,c+1) linked; vl[r, c]: (r,c)-(r+1,c) linked."""
    H, W = L.H, L.W
    nxt = L.nxt.reshape(H, W)
    prv = L.prv.reshape(H, W)
    idx = np.arange(H * W).reshape(H, W)
    hl = (nxt[:, :-1] == idx[:, 1:]) | (prv[:, :-1] == idx[:, 1:])
    vl = (nxt[:-1, :] == idx[1:, :]) | (prv[:-1, :] == idx[1:, :])
    return hl, vl


def action_mask(L: Layer, ops=None) -> np.ndarray:
    """Bool (N_OPS, H*W): which operator is valid at which cell."""
    H, W, n = L.H, L.W, L.n
    m = L.flat_mask
    pc = get_pieces(L)
    lab = pc.label
    safe = np.where(lab >= 0, lab, 0)
    plen = np.where(lab >= 0, pc.length[safe], 0)
    on_loop = np.where(lab >= 0, pc.is_loop[safe], False)
    nxt, prv = L.nxt, L.prv
    M = np.zeros((N_OPS, n), bool)

    M[0] = m & (plen >= 2)                                  # reverse

    end = m & ((nxt == -1) | (prv == -1))                   # merge
    e2, l2, p2 = end.reshape(H, W), lab.reshape(H, W), plen.reshape(H, W)
    mg = np.zeros((H, W), bool)

    def pair(a, b):
        return e2[a] & e2[b] & ((l2[a] != l2[b]) | (p2[a] >= 3))
    mg[:, :-1] |= pair((slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    mg[:, 1:] |= pair((slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    mg[:-1, :] |= pair((slice(None, -1), slice(None)), (slice(1, None), slice(None)))
    mg[1:, :] |= pair((slice(1, None), slice(None)), (slice(None, -1), slice(None)))
    M[1] = mg.reshape(-1)

    M[2] = m & (nxt >= 0)                                   # split

    hl, vl = link_grids(L)                                  # flips
    fr = np.zeros((H, W), bool)
    fd = np.zeros((H, W), bool)
    if H > 1 and W > 1:
        fr[:-1, :-1] = hl[:-1, :] & hl[1:, :] & ~vl[:, :-1] & ~vl[:, 1:]
        fd[:-1, :-1] = vl[:, :-1] & vl[:, 1:] & ~hl[:-1, :] & ~hl[1:, :]
    M[3] = fr.reshape(-1)
    M[4] = fd.reshape(-1)

    M[5] = m & on_loop & ~seam_mask(L)                      # move seam

    ov5, ov9 = _box_any(L.mask, 5), _box_any(L.mask, 9)     # patch refill
    M[6] = M[7] = ov5
    M[8] = M[9] = ov9

    if ops is not None:
        keep = np.zeros(N_OPS, bool)
        keep[ops_to_indices(ops)] = True
        M[~keep] = False
    return M


# ---------------------------------------------------------------- apply

def _cut(L: Layer, a: int, b: int) -> None:
    if L.nxt[a] == b:
        L.nxt[a] = -1
        L.prv[b] = -1
    elif L.nxt[b] == a:
        L.nxt[b] = -1
        L.prv[a] = -1
    else:
        raise InvalidEdit("cells not linked")


def _flip(N: Layer, a: int, b: int, c: int, d: int) -> None:
    """Replace links a-b and c-d with a-c and b-d."""
    _cut(N, a, b)
    _cut(N, c, d)
    if not join_ends(N, a, c) or not join_ends(N, b, d):
        raise InvalidEdit("flip could not relink")


def apply(L: Layer, op: int, cell: int) -> tuple[Layer, dict]:
    """Return (new layer, params). Raises InvalidEdit when op is invalid at cell."""
    H, W = L.H, L.W
    m = L.flat_mask
    if not (0 <= cell < L.n):
        raise InvalidEdit("cell out of range")
    N = L.copy()
    params: dict = {}
    old_seams = sorted(L.seam)
    r, c = divmod(cell, W)

    if op == 0:                                                    # reverse
        if not m[cell]:
            raise InvalidEdit("reverse outside part")
        sc = reverse_stroke(N, cell)
        if len(sc) < 2:
            raise InvalidEdit("single-cell stroke")
    elif op == 1:                                                  # merge
        if not m[cell] or not is_end(L, cell):
            raise InvalidEdit("not a stroke end")
        pc = get_pieces(L)
        lab = pc.label
        cands = [j for j in neighbors4(cell, H, W) if m[j] and is_end(L, j)]
        diff_ = [j for j in cands if lab[j] != lab[cell]]
        same = [j for j in cands if lab[j] == lab[cell] and pc.length[lab[cell]] >= 3]
        heads = [x for x in [cell] + same if L.prv[x] == -1]
        for j in diff_ + same:
            if join_ends(N, cell, j):
                params["with"] = int(j)
                fix_seams_local(N, [cell, j], prefer=heads + old_seams)
                break
        else:
            raise InvalidEdit("no adjacent end to merge with")
    elif op == 2:                                                  # split
        j = int(L.nxt[cell])
        if j < 0:
            raise InvalidEdit("no successor")
        N.nxt[cell] = -1
        N.prv[j] = -1
        fix_seams_local(N, [cell, j])
    elif op in (3, 4):                                             # flips
        if r >= H - 1 or c >= W - 1:
            raise InvalidEdit("flip block outside grid")
        a, b, cc, d = cell, cell + 1, cell + W, cell + W + 1
        if op == 3:   # horizontal pair a-b, cc-d -> vertical a-cc, b-d
            if not (linked(L, a, b) and linked(L, cc, d)) or linked(L, a, cc) or linked(L, b, d):
                raise InvalidEdit("links not parallel")
            _flip(N, a, b, cc, d)
        else:         # vertical pair a-cc, b-d -> horizontal a-b, cc-d
            if not (linked(L, a, cc) and linked(L, b, d)) or linked(L, a, b) or linked(L, cc, d):
                raise InvalidEdit("links not parallel")
            _flip(N, a, cc, b, d)
        fix_seams_local(N, [a, b, cc, d], prefer=old_seams)
    elif op == 5:                                                  # move seam
        if not m[cell] or cell in L.seam:
            raise InvalidEdit("cell already a seam or outside")
        sc, loop = stroke_cells(N, cell)
        if not loop:
            raise InvalidEdit("not on a loop")
        for s in sc:
            N.seam.discard(s)
        N.seam.add(cell)
    elif op in PATCHES:                                            # patch refill
        s, ang = PATCHES[op]
        hlf = s // 2
        r0, r1 = max(r - hlf, 0), min(r + hlf + 1, H)
        c0, c1 = max(c - hlf, 0), min(c + hlf + 1, W)
        wm = np.zeros((H, W), bool)
        wm[r0:r1, c0:c1] = L.mask[r0:r1, c0:c1]
        win = np.nonzero(wm.reshape(-1))[0]
        if win.size == 0:
            raise InvalidEdit("window misses the part")
        touched = set(win.tolist())
        for x in win.tolist():
            touched.update(unlink_cell(N, x))
        zigzag_links(N, wm, ang)
        repair(N, cells=touched)
        fix_seams_local(N, touched, prefer=old_seams)
        params["window"] = [int(r0), int(c0), int(r1), int(c1)]
    else:
        raise InvalidEdit(f"unknown op {op}")
    N.invalidate()
    return N, params


def apply_action(L: Layer, a: int) -> tuple[Layer, dict]:
    op, cell = decode(a, L.n)
    return apply(L, op, cell)


def layer_diff(old: Layer, new: Layer) -> dict:
    ch = np.nonzero((old.nxt != new.nxt) | (old.prv != new.prv))[0]
    return {
        "cells": ch.tolist(),
        "nxt": new.nxt[ch].tolist(),
        "prv": new.prv[ch].tolist(),
        "seam_add": sorted(new.seam - old.seam),
        "seam_del": sorted(old.seam - new.seam),
    }


def sample_valid(M: np.ndarray, rng: np.random.Generator, k: int = 1, by_op: bool = True):
    """Sample k valid flat actions. by_op: pick the operator uniformly first, then a cell,
    so rare operators (merge, flips) are not drowned out by always-valid ones."""
    M2 = M.reshape(N_OPS, -1)
    n = M2.shape[1]
    if not by_op:
        idx = np.flatnonzero(M2)
        if idx.size == 0:
            return np.empty(0, np.int64)
        return rng.choice(idx, size=k, replace=idx.size < k)
    per_op = [np.flatnonzero(M2[o]) for o in range(N_OPS)]
    ops = [o for o in range(N_OPS) if per_op[o].size]
    if not ops:
        return np.empty(0, np.int64)
    out = []
    for _ in range(k):
        o = ops[rng.integers(len(ops))]
        out.append(o * n + per_op[o][rng.integers(per_op[o].size)])
    return np.asarray(out, np.int64)
