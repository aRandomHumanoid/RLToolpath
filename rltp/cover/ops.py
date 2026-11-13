"""Edit operators. Every operator maps a valid cover to another valid cover.

Actions are flattened operator-major: a = op * (H * W) + cell.

The first N_PATH_OPS operators re-route paths over a fixed set of filled cells; they
are the trained policy's action space. The rest change which cells are filled and
only apply to sparse layers (see rltp.cover.shell): add or remove one optional cell,
fill the optional cells of a 5x5 window and zigzag it, or clear a 5x5 window. The four
moves keep the layer's infill count: they add (remove) material at the cell, then trim
the least stressed path ends (grow the most stressed path ends) elsewhere in the layer
by as many cells, using the layer's current FI map. stack9 copies the layer below's
infill into a 9x9 window (so it bonds fully there), zigzags it along the copied lines,
and trims or grows elsewhere to keep the count. On a
sparse layer no operator may change a wall cell's links (only move_seam touches
walls), empty a required cell, or break the support rule; apply() checks all three.
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
       "patch5_0", "patch5_90", "patch9_0", "patch9_90",
       "add", "remove", "fill5_0", "fill5_90", "clear5",
       "add_trim", "remove_grow", "fill5_trim", "clear5_grow", "stack9",
       "join", "shift_pos", "shift_neg", "extend", "retract", "refill_0", "refill_90", "clear_w", "stack_w",
       "refill32_0", "refill32_90", "refill64_0", "refill64_90", "flip_region",
       "refill_45", "refill_135", "refill32_45", "refill32_135", "turn_region")
FINE_FIRST = 20          # fine bead-grid edits (rltp.cover.beads) start here
N_OPS = len(OPS)
N_PATH_OPS = 10          # operators on a fixed set of filled cells: the policy's action space
OP_INDEX = {name: i for i, name in enumerate(OPS)}
PATCHES = {6: (5, 0), 7: (5, 90), 8: (9, 0), 9: (9, 90)}
FILLS = {12: 0, 13: 90}


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
        elif o in ("flip", "patch", "fill"):
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

    lock = L.lock.reshape(-1) if L.lock is not None else None
    free = m if lock is None else m & ~lock                 # cells whose links may change

    M[0] = free & (plen >= 2)                               # reverse

    end = free & ((nxt == -1) | (prv == -1))                # merge
    e2, l2, p2 = end.reshape(H, W), lab.reshape(H, W), plen.reshape(H, W)
    mg = np.zeros((H, W), bool)

    def pair(a, b):
        return e2[a] & e2[b] & ((l2[a] != l2[b]) | (p2[a] >= 3))
    mg[:, :-1] |= pair((slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    mg[:, 1:] |= pair((slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    mg[:-1, :] |= pair((slice(None, -1), slice(None)), (slice(1, None), slice(None)))
    mg[1:, :] |= pair((slice(1, None), slice(None)), (slice(None, -1), slice(None)))
    M[1] = mg.reshape(-1)

    M[2] = free & (nxt >= 0)                                # split

    hl, vl = link_grids(L)                                  # flips
    fr = np.zeros((H, W), bool)
    fd = np.zeros((H, W), bool)
    if H > 1 and W > 1:
        fr[:-1, :-1] = hl[:-1, :] & hl[1:, :] & ~vl[:, :-1] & ~vl[:, 1:]
        fd[:-1, :-1] = vl[:, :-1] & vl[:, 1:] & ~hl[:-1, :] & ~hl[1:, :]
        if lock is not None:                                # no wall cell in the 2x2 block
            lk = L.lock
            blk = np.zeros((H, W), bool)
            blk[:-1, :-1] = lk[:-1, :-1] | lk[:-1, 1:] | lk[1:, :-1] | lk[1:, 1:]
            fr &= ~blk
            fd &= ~blk
    M[3] = fr.reshape(-1)
    M[4] = fd.reshape(-1)

    M[5] = m & on_loop & ~seam_mask(L)                      # move seam

    pm = L.mask if lock is None else L.mask & ~L.lock       # patch refill
    ov5, ov9 = _box_any(pm, 5), _box_any(pm, 9)
    M[6] = M[7] = ov5
    M[8] = M[9] = ov9

    if L.kb > 1:                                            # fine bead grid
        from rltp.cover.beads import FINE_OPS, fine_masks
        for o in (1, 3, 4, 6, 7, 8, 9):                     # unit-cell path edits don't apply
            M[o] = False
        fm = fine_masks(L)
        for j, name in enumerate(FINE_OPS):
            M[FINE_FIRST + j] = fm[name].reshape(-1)
    elif L.part is not None:                                # fill edits (sparse layers)
        from rltp.cover.shell import critical
        supp = L.ctx["supp"] if L.ctx is not None else np.ones_like(L.part)
        can_add = L.part & ~L.mask & supp
        can_del = L.mask & ~L.req
        room_up = room_down = True
        bud = L.ctx.get("budget") if L.ctx is not None else None
        if bud is not None:                                 # band rule on the part's infill
            room_up = bud["total"] + 1 <= bud["hi"] or bud["total"] < bud["lo"]
            room_down = bud["total"] - 1 >= bud["lo"] or bud["total"] > bud["hi"]
        crit = critical(L)
        M[10] = can_add.reshape(-1) & room_up
        M[11] = (can_del & ~crit).reshape(-1) & room_down
        M[12] = M[13] = _box_any(can_add, 5) & room_up
        M[14] = _box_any(can_del, 5) & room_down
        can_trim = bool(_trimmable(L, crit).any())          # moves: the count stays put
        can_grow = bool(_growable(L).any())
        M[15] = can_add.reshape(-1) & can_trim
        M[16] = (can_del & ~crit).reshape(-1) & can_grow
        M[17] = _box_any(can_add, 5) & can_trim
        M[18] = _box_any(can_del, 5) & can_grow
        below = L.ctx.get("below") if L.ctx is not None else None
        if below is not None:                               # stack9: windows that differ from below
            opt = L.part & ~L.req
            M[19] = _box_any(opt & ((below & supp) != L.mask), 9)

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
    N, params = _apply(L, op, cell)
    if L.part is not None:
        _check_sparse(L, N)
    return N, params


def _apply(L: Layer, op: int, cell: int) -> tuple[Layer, dict]:
    """apply() without the sparse-layer checks (moves combine two unchecked halves)."""
    if op >= FINE_FIRST or (L.kb > 1 and op not in (0, 2, 5)):
        if L.kb == 1:
            raise InvalidEdit("fine edits need a fine bead grid")
        if op < FINE_FIRST:
            raise InvalidEdit("unit-cell edit on a fine bead grid")
        from rltp.cover.beads import FINE_OPS, FineEditError, apply_fine
        if not (0 <= cell < L.n):
            raise InvalidEdit("cell out of range")
        N = L.copy()
        try:
            params = apply_fine(L, N, FINE_OPS[op - FINE_FIRST], cell)
        except FineEditError as e:
            raise InvalidEdit(str(e))
        N.invalidate()
        return N, params
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
        params["window"] = _patch(N, r, c, s, ang, old_seams)
    elif op >= N_PATH_OPS and L.part is None:
        raise InvalidEdit("fill edits need a sparse layer")
    elif op == 10:                                                 # add one cell
        if not L.part.reshape(-1)[cell] or m[cell]:
            raise InvalidEdit("cell not empty or outside the part")
        N.mask.reshape(-1)[cell] = True
        N.invalidate()
        lk = L.lock.reshape(-1)
        joined = []
        for j in neighbors4(cell, H, W):        # extend a neighbouring stroke, then bridge to a second
            if len(joined) == 2 or not is_end(N, cell):
                break
            if N.mask.reshape(-1)[j] and not lk[j] and is_end(N, j) and join_ends(N, cell, j):
                joined.append(int(j))
        params["with"] = joined
        fix_seams_local(N, [cell] + joined, prefer=old_seams)
    elif op == 11:                                                 # remove one cell
        if not m[cell] or L.req.reshape(-1)[cell]:
            raise InvalidEdit("cell empty or required")
        touched = unlink_cell(N, cell)
        N.seam.discard(cell)
        N.mask.reshape(-1)[cell] = False
        N.invalidate()
        fix_seams_local(N, touched, prefer=old_seams)
    elif op in FILLS:                                              # fill a 5x5 window, zigzag it
        r0, r1, c0, c1 = _window(r, c, 5, H, W)
        supp = L.ctx["supp"] if L.ctx is not None else np.ones_like(L.part)
        new_cells = np.zeros((H, W), bool)
        new_cells[r0:r1, c0:c1] = (L.part & ~L.mask & supp)[r0:r1, c0:c1]
        if not new_cells.any():
            raise InvalidEdit("nothing to fill")
        N.mask |= new_cells
        N.invalidate()
        params["window"] = _patch(N, r, c, 5, FILLS[op], old_seams)
    elif op in (15, 17):                                           # add here, trim elsewhere
        before = int(N.mask.sum())
        if op == 15:
            N, params = _apply(L, 10, cell)
            keep = {cell}
        else:
            N, params = _apply(L, 12, cell)
            r0, c0, r1, c1 = params["window"]
            keep = {r * W + cc for r in range(r0, r1) for cc in range(c0, c1)}
        params["trimmed"] = _trim(N, int(N.mask.sum()) - before, keep, old_seams)
    elif op in (16, 18):                                           # remove here, grow elsewhere
        before = int(N.mask.sum())
        if op == 16:
            N, params = _apply(L, 11, cell)
            keep = {cell}
        else:
            N, params = _apply(L, 14, cell)
            r0, c0, r1, c1 = params["window"]
            keep = {r * W + cc for r in range(r0, r1) for cc in range(c0, c1)}
        params["grown"] = _grow(N, before - int(N.mask.sum()), keep, old_seams)
    elif op == 19:                                                 # stack on the layer below
        below = L.ctx.get("below") if L.ctx is not None else None
        if below is None:
            raise InvalidEdit("no layer below")
        r0, r1, c0, c1 = _window(r, c, 9, H, W)
        win = np.zeros((H, W), bool)
        win[r0:r1, c0:c1] = True
        opt = win & L.part & ~L.req
        want = opt & below & L.ctx["supp"]
        if (want == (opt & L.mask)).all():
            raise InvalidEdit("window already stacked")
        before = int(N.mask.sum())
        gone = np.flatnonzero(opt & L.mask & ~want)
        touched: set = set()
        for x in gone.tolist():
            touched.update(unlink_cell(N, x))
            N.seam.discard(x)
        N.mask[opt & ~want] = False
        N.mask[want] = True
        N.invalidate()
        # re-link the window's filled unlocked cells along the copied lines' direction
        sub = want[r0:r1, c0:c1]
        horiz = int((sub[:, 1:] & sub[:, :-1]).sum())
        vert = int((sub[1:, :] & sub[:-1, :]).sum())
        params["window"] = _patch(N, r, c, 9, 0 if horiz >= vert else 90, old_seams)
        outside = touched - set(gone.tolist())
        if outside:
            repair(N, cells=outside)
            fix_seams_local(N, outside, prefer=old_seams)
        keep = {int(x) for x in np.flatnonzero(win)}
        diff = int(N.mask.sum()) - before
        if diff > 0:
            params["trimmed"] = _trim(N, diff, keep, old_seams)
        elif diff < 0:
            params["grown"] = _grow(N, -diff, keep, old_seams)
    elif op == 14:                                                 # clear a 5x5 window
        r0, r1, c0, c1 = _window(r, c, 5, H, W)
        gone = np.zeros((H, W), bool)
        gone[r0:r1, c0:c1] = (L.mask & ~L.req)[r0:r1, c0:c1]
        cells = np.flatnonzero(gone)
        if cells.size == 0:
            raise InvalidEdit("nothing to clear")
        touched: set = set()
        for x in cells.tolist():
            touched.update(unlink_cell(N, x))
            N.seam.discard(x)
        N.mask[gone] = False
        N.invalidate()
        touched -= set(cells.tolist())
        repair(N, cells=touched)
        fix_seams_local(N, touched, prefer=old_seams)
        params["window"] = [int(r0), int(c0), int(r1), int(c1)]
    else:
        raise InvalidEdit(f"unknown op {op}")
    N.invalidate()
    return N, params


def _fi(L: Layer) -> np.ndarray:
    fi = L.ctx.get("fi") if L.ctx is not None else None
    return np.zeros(L.n) if fi is None or np.size(fi) != L.n else np.asarray(fi, float)


def _bonded(L: Layer) -> np.ndarray:
    """Cells with a filled cell directly below or above (they carry the Z bond)."""
    out = np.zeros(L.n, bool)
    if L.ctx is not None:
        for key in ("below", "above"):
            if L.ctx.get(key) is not None:
                out |= np.asarray(L.ctx[key]).reshape(-1)
    return out


def _trimmable(L: Layer, crit=None) -> np.ndarray:
    """Optional filled cells at a path end whose removal can't strand the layer above."""
    from rltp.cover.shell import critical
    end = L.flat_mask & ((L.nxt == -1) | (L.prv == -1))
    crit = critical(L) if crit is None else crit
    return end & ~L.req.reshape(-1) & ~crit.reshape(-1)


def _growable(L: Layer) -> np.ndarray:
    """Empty, supported optional cells next to the end of an unlocked path."""
    H, W = L.H, L.W
    end = (L.flat_mask & ((L.nxt == -1) | (L.prv == -1)) & ~L.lock.reshape(-1)).reshape(H, W)
    near = np.zeros((H, W), bool)
    near[:, 1:] |= end[:, :-1]
    near[:, :-1] |= end[:, 1:]
    near[1:, :] |= end[:-1, :]
    near[:-1, :] |= end[1:, :]
    supp = L.ctx["supp"] if L.ctx is not None else np.ones_like(L.part)
    return (near & L.part & ~L.mask & supp).reshape(-1)


def _trim(N: Layer, n: int, keep: set, old_seams) -> list[int]:
    """Remove n optional path-end cells (outside `keep`): unbonded ones first, then the
    least stressed."""
    fi = _fi(N)
    rank = fi + np.where(_bonded(N), 1e6, 0.0)
    out = []
    for _ in range(n):
        cand = _trimmable(N)
        if keep:
            cand[list(keep)] = False
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            raise InvalidEdit("nothing to trim")
        x = int(idx[np.argmin(rank[idx])])
        touched = unlink_cell(N, x)
        N.seam.discard(x)
        N.mask.reshape(-1)[x] = False
        N.invalidate()
        fix_seams_local(N, touched, prefer=old_seams)
        out.append(x)
    return out


def _grow(N: Layer, n: int, keep: set, old_seams) -> list[int]:
    """Add n cells (outside `keep`), each extending a path end: bonded cells first, then
    the end with the highest FI."""
    H, W = N.H, N.W
    fi = _fi(N)
    bonus = np.where(_bonded(N), 1e6, 0.0)
    lk = N.lock.reshape(-1)
    out = []
    for _ in range(n):
        cand = _growable(N)
        if keep:
            cand[list(keep)] = False
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            raise InvalidEdit("nothing to grow")
        best, best_end, best_fi = -1, -1, -np.inf
        for x in idx.tolist():          # the stressed end it would extend
            for e in neighbors4(x, H, W):
                score = fi[e] + bonus[x]
                if N.mask.reshape(-1)[e] and not lk[e] and is_end(N, e) and score > best_fi:
                    best, best_end, best_fi = x, e, score
        N.mask.reshape(-1)[best] = True
        N.invalidate()
        if not join_ends(N, best, best_end):
            raise InvalidEdit("could not extend a path")
        fix_seams_local(N, [best, best_end], prefer=old_seams)
        out.append(best)
    return out


def _window(r: int, c: int, s: int, H: int, W: int):
    hlf = s // 2
    return max(r - hlf, 0), min(r + hlf + 1, H), max(c - hlf, 0), min(c + hlf + 1, W)


def _patch(N: Layer, r: int, c: int, s: int, ang: int, old_seams) -> list[int]:
    """Re-cover the filled, unlocked cells of the s x s window around (r, c) with a zigzag."""
    H, W = N.H, N.W
    r0, r1, c0, c1 = _window(r, c, s, H, W)
    wm = np.zeros((H, W), bool)
    wm[r0:r1, c0:c1] = N.mask[r0:r1, c0:c1]
    if N.lock is not None:
        wm &= ~N.lock
    win = np.nonzero(wm.reshape(-1))[0]
    if win.size == 0:
        raise InvalidEdit("window misses the part")
    touched = set(win.tolist())
    for x in win.tolist():
        touched.update(unlink_cell(N, x))
    zigzag_links(N, wm, ang)
    repair(N, cells=touched)
    fix_seams_local(N, touched, prefer=old_seams)
    return [int(r0), int(c0), int(r1), int(c1)]


def _check_sparse(L: Layer, N: Layer) -> None:
    """Walls keep their links, required cells stay filled, the support rule holds."""
    lk = L.lock.reshape(-1)
    if lk.any() and ((N.nxt[lk] != L.nxt[lk]).any() or (N.prv[lk] != L.prv[lk]).any()):
        raise InvalidEdit("edit changes a wall")
    if L.kb == 1 and (L.req & ~N.mask).any():           # fine grids: gaps cost, not forbidden
        raise InvalidEdit("edit empties a required cell")
    from rltp.cover.shell import support_violation
    why = support_violation(L, N)
    if why:
        raise InvalidEdit(why)
    bud = L.ctx.get("budget") if L.ctx is not None else None
    if bud is not None:                  # band rule: stay in the band, or move toward it
        delta = int((N.material & ~N.req).sum()) - int((L.material & ~L.req).sum())
        old, new = bud["total"], bud["total"] + delta

        def out(x):
            return max(bud["lo"] - x, x - bud["hi"], 0.0)
        if out(new) > 0 and out(new) >= out(old) and delta != 0:
            raise InvalidEdit("infill would leave its band")


def apply_action(L: Layer, a: int) -> tuple[Layer, dict]:
    op, cell = decode(a, L.n)
    return apply(L, op, cell)


def layer_diff(old: Layer, new: Layer) -> dict:
    """Cells whose links (or, on a sparse layer, filled state) changed; an emptied cell
    gets nxt -2, as in history snapshots."""
    changed = (old.nxt != new.nxt) | (old.prv != new.prv)
    nxt = new.nxt.astype(np.int64)
    if new.sparse:
        changed |= old.flat_mask != new.flat_mask
        nxt[~new.flat_mask] = -2
    ch = np.nonzero(changed)[0]
    return {
        "cells": ch.tolist(),
        "nxt": nxt[ch].tolist(),
        "prv": new.prv[ch].tolist(),
        "seam_add": sorted(new.seam - old.seam),
        "seam_del": sorted(old.seam - new.seam),
    }


def sample_valid(M: np.ndarray, rng: np.random.Generator, k: int = 1, by_op: bool = True,
                 weights: np.ndarray | None = None, bias: float = 0.0):
    """Sample k valid flat actions. by_op: pick the operator uniformly first, then a cell,
    so rare operators (merge, flips) are not drowned out by always-valid ones.
    M is flat over all N_OPS operators, or 2D (n_ops, cells) for a smaller action space.
    With `weights` (per cell) and bias > 0, that share of draws picks the cell in
    proportion to its weight instead of uniformly."""
    M2 = M if M.ndim == 2 else M.reshape(N_OPS, -1)
    n = M2.shape[1]
    if not by_op:
        idx = np.flatnonzero(M2)
        if idx.size == 0:
            return np.empty(0, np.int64)
        return rng.choice(idx, size=k, replace=idx.size < k)
    per_op = [np.flatnonzero(M2[o]) for o in range(M2.shape[0])]
    ops = [o for o in range(M2.shape[0]) if per_op[o].size]
    if not ops:
        return np.empty(0, np.int64)
    out = []
    for _ in range(k):
        o = ops[rng.integers(len(ops))]
        cells = per_op[o]
        if weights is not None and bias > 0 and rng.random() < bias:
            pw = weights[cells]
            tot = pw.sum()
            j = int(rng.choice(cells.size, p=pw / tot)) if tot > 0 else int(rng.integers(cells.size))
            out.append(o * n + cells[j])
        else:
            out.append(o * n + cells[rng.integers(cells.size)])
    return np.asarray(out, np.int64)
