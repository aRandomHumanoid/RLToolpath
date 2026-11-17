"""How far a part's toolpaths have adapted to its loads: diagnostics, not score terms.

- layout_change: between two layouts of a part, the share of bead cells whose direction
  changed and of optional cells whose fill changed.
- orientation: with the strains the current layout produces (laminate parts), how much of
  sum FI^p each cell would lose if its beads alone turned to the best of 0/45/90/135
  degrees. eff = sum(best) / sum(current) is 1 when no single cell can do better by
  turning; `aligned` is the FI^2-weighted share of bead cells already in their best
  direction. p = 2 sees the bulk of the part, p = 8 its peaks (as FI_agg does).
  `wall` is the share of sum FI^p (current directions) sitting in the locked wall rings,
  which no edit can turn.
"""
from __future__ import annotations

import numpy as np

from rltp.cover.cover import bead_weights
from rltp.eval.fea2d import to_material, tsai_wu_fi

ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def layer_dirs(L):
    """(filled, shares, wall) per material cell: filled bool (n,), shares (n, 4) of the
    material laid at 0, 45, 90 and 135 degrees, wall bool (n,)."""
    from rltp.search.fields import wall_cells
    if L.kb > 1:
        from rltp.cover.beads import cell_shares, cell_wh, density
        filled = density(L).reshape(-1) >= 0.5
        sh = cell_shares(L)
        if sh is None:
            wh = cell_wh(L)
            sh = np.stack([wh, 0 * wh, 1 - wh, 0 * wh], -1)
    else:
        filled = L.flat_mask.copy()
        wh = bead_weights(L)
        sh = np.stack([wh, 0 * wh, 1 - wh, 0 * wh], -1)
    return filled, sh, wall_cells(L).reshape(-1)


def layout_change(start_layers, layers) -> dict:
    """Share of field cells (filled in both, not walls) whose main bead direction changed,
    and share of the part's non-wall cells whose fill changed."""
    both = chg = nonwall = fill = 0
    per_layer = []
    for S, F in zip(start_layers, layers):
        fs, ss, ws = layer_dirs(S)
        ff, sf, wf = layer_dirs(F)
        inpart = F.part_mask.reshape(-1) & ~ws & ~wf
        b = fs & ff & inpart
        c = int((np.argmax(ss, 1)[b] != np.argmax(sf, 1)[b]).sum())
        both += int(b.sum())
        chg += c
        nonwall += int(inpart.sum())
        fc = int((fs != ff)[inpart].sum())
        fill += fc
        per_layer.append(round(c / max(int(b.sum()), 1), 4))
    return {"dir_changed": chg / max(both, 1), "fill_changed": fill / max(nonwall, 1),
            "dir_changed_per_layer": per_layer}


def _cell_strains(lam, u):
    U = u.reshape(lam.ndof, -1)
    eps = np.einsum("jk,ekc->ecj", lam.B0, U[lam.edof])            # (ne, C, 3)
    if lam.cf > 1:
        return eps[lam.cell_elem]
    if lam.r > 1:                                                   # mean over each cell's elements
        nc, k = lam.cells.size, lam.r * lam.r
        flat = eps.reshape(eps.shape[0], -1)
        out = np.stack([np.bincount(lam.elem_cell, flat[:, j], minlength=nc) / k
                        for j in range(flat.shape[1])], 1)
        return out.reshape(nc, *eps.shape[1:])
    return eps


def orientation(part, ps=(2, 8)) -> dict | None:
    """Laminate parts only (sparse parts, or --stack laminate): see the module docstring."""
    if not part.laminate or part.lam is None or not part.lam.active:
        return None
    lam = part.lam
    eps = _cell_strains(lam, part.lam_u)
    C = eps.shape[1]
    Qs = (lam.Q0, lam.Q45, lam.Q90, lam.Q135)
    fi = np.stack([tsai_wu_fi(*to_material(*np.moveaxis(eps @ Q.T, -1, 0), a), lam.mat)
                   for Q, a in zip(Qs, ANGLES)], -1)                # (nc, C, 4)
    cs = lam.case_scale if lam.case_scale is not None and C > 1 else np.ones(C)
    if getattr(lam, "keep", None) is not None:                     # clamp band: as the score sees it
        fi = fi * lam.keep[:, :, None]
    out = {}
    for p in ps:
        m = np.mean((fi * cs[None, :, None]) ** p, 1)                # (nc, 4)
        cur_t = best_t = w_al = w_t = wall_t = 0.0
        for L in part.layers:
            filled, sh, wall = layer_dirs(L)
            wr = (filled & wall)[lam.cells]
            wall_t += float(np.max(np.where(sh[lam.cells][wr] > 1e-9, m[wr], 0.0), 1).sum()) if wr.any() else 0.0
            rows = (filled & ~wall)[lam.cells]
            shr = sh[lam.cells][rows]
            mr = m[rows]
            cur = np.max(np.where(shr > 1e-9, mr, 0.0), 1)
            best = mr.min(1)
            cur_t += float(cur.sum())
            best_t += float(best.sum())
            al = np.argmax(shr, 1) == np.argmin(mr, 1)
            w_al += float(best[al].sum())
            w_t += float(best.sum())
        out[f"eff_p{p}"] = best_t / cur_t if cur_t > 0 else 1.0
        out[f"aligned_p{p}"] = w_al / w_t if w_t > 0 else 1.0
        out[f"wall_p{p}"] = wall_t / (wall_t + cur_t) if wall_t + cur_t > 0 else 0.0
    return out
