"""How much does scoring a cell at the worst bead direction it holds cost curved beads?

The evaluator (fea2d.post_layers) gives a cell holding material in several of the 0/45/90/
135 classes the worst of their failure indices. A bead following a curve on the 0.1 mm grid
is a staircase whose one-cell jogs lay a little 90 (or 0) degree material, so its cells
score as if loaded across the bead. Here every cell is also scored at its bead's own path
direction (the node path smoothed over +-SMOOTH nodes), from the same laminate strains.

For one suite run, layouts: slicer start (1 wall), final, and the equal-material second-ring
layout of probe_walls.py (2 walls, infill from its JSON). Per layout and rule (eval / path):
per-case sum of fill x FI^8 (raw), split into wall ring and the rest, and per-case peaks.

usage: python scripts/analysis/probe_dirs.py RUN_DIR PROBE_WALLS_JSON OUT_JSON   (from the repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path(__file__).parent))
from direction_maps import build_for, restore  # noqa: E402
from probe_walls import build  # noqa: E402

from rltp.cover.beads import cell_shares, cell_wh, cover_sum, density  # noqa: E402
from rltp.cover.cover import get_pieces  # noqa: E402
from rltp.eval.adapt import _cell_strains  # noqa: E402
from rltp.eval.fea2d import q_matrix, to_material, tsai_wu_fi  # noqa: E402
from rltp.export.state import load_state  # noqa: E402
from rltp.search.fields import strain_rot, wall_cells  # noqa: E402

SMOOTH = 6


def node_dirs(L):
    """Doubled-angle sums (cos, sin) and counts of the bead path direction per node."""
    H, W = L.H, L.W
    vx, vy = np.zeros(H * W), np.zeros(H * W)
    pc = get_pieces(L)
    for cells, loop in zip(pc.cells, pc.is_loop):
        s = np.asarray(cells)
        n = len(s)
        if n < 2:
            continue
        r, c = np.divmod(s, W)
        i = np.arange(n)
        k = min(SMOOTH, n - 1)
        a, b = ((i + k) % n, (i - k) % n) if loop else (np.minimum(i + k, n - 1), np.maximum(i - k, 0))
        phi = np.arctan2(r[a] - r[b], c[a] - c[b])
        vx[s] += np.cos(2 * phi)
        vy[s] += np.sin(2 * phi)
    return vx.reshape(H, W), vy.reshape(H, W)


def score(part):
    lam = part.lam
    eps = _cell_strains(lam, part.lam_u)                     # (nc, C, 3), shared by the layers
    mat = lam.mat
    C = eps.shape[1]
    Qs = (lam.Q0, lam.Q45, lam.Q90, lam.Q135)
    fic = np.stack([tsai_wu_fi(*to_material(*np.moveaxis(eps @ Q.T, -1, 0), a), mat)
                    for Q, a in zip(Qs, (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4))], -1)   # (nc, C, 4)
    Qm = q_matrix(mat)
    # check the path rule's rotation against the evaluator's classes
    for d, a in enumerate((0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
        s = np.einsum("ij,ecj->eci", Qm @ strain_rot(np.array(a)), eps[:200])
        assert np.allclose(tsai_wu_fi(s[..., 0], s[..., 1], s[..., 2], mat), fic[:200, :, d], rtol=1e-6, atol=1e-9), d
    keep = getattr(lam, "keep", None)
    kp = np.ones((len(eps), C)) if keep is None else np.broadcast_to(np.asarray(keep, float).reshape(len(eps), -1), (len(eps), C))
    out = {k: np.zeros(C) for k in ("eval_wall", "eval_rest", "path_wall", "path_rest")}
    peak = {"eval": np.zeros(C), "path": np.zeros(C)}
    mixed = [0, 0]
    for L in part.layers:
        D = np.minimum(density(L), 1.0).reshape(-1)[lam.cells]
        on = D > 1e-9
        sh = cell_shares(L)
        if sh is None:
            wh = cell_wh(L)
            sh = np.stack([wh, 0 * wh, 1 - wh, 0 * wh], -1)
        sh = sh[lam.cells]
        pres = (sh > 1e-9) & on[:, None]
        f_eval = np.max(np.where(pres[:, None, :], fic, 0.0), -1)                     # (nc, C)
        vx, vy = node_dirs(L)
        k = L.kb
        cx = cover_sum(vx, k).reshape(-1)[lam.cells]
        cy = cover_sum(vy, k).reshape(-1)[lam.cells]
        phi = 0.5 * np.arctan2(cy, cx)
        has = on & (np.hypot(cx, cy) > 1e-9)
        f_path = f_eval.copy()
        idx = np.flatnonzero(has)
        s = np.einsum("eij,ecj->eci", Qm @ strain_rot(phi[idx]), eps[idx])
        f_path[idx] = tsai_wu_fi(s[..., 0], s[..., 1], s[..., 2], mat)
        f_eval, f_path = f_eval * kp, f_path * kp
        wall = wall_cells(L).reshape(-1)[lam.cells] & on
        for rule, f in (("eval", f_eval), ("path", f_path)):
            f8 = D[:, None] * f ** 8
            out[f"{rule}_wall"] += f8[wall].sum(0)
            out[f"{rule}_rest"] += f8[~wall].sum(0)
            full = D > 0.5
            if full.any():
                peak[rule] = np.maximum(peak[rule], f[full].max(0))
        mixed[0] += int((pres.sum(1) > 1)[wall].sum())
        mixed[1] += int(wall.sum())
    res = {k: v.tolist() for k, v in out.items()} | {f"peak_{k}": v.tolist() for k, v in peak.items()}
    res["wall_cells_mixed"] = mixed[0] / max(mixed[1], 1)
    return res


def main():
    run, pw, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    res, part, _ = build_for(run)
    part.raw()
    rows = {"slicer": score(part)}
    restore(part, load_state(run / "state.npz"))
    part.raw()
    rows["final"] = score(part)
    x = json.loads(pw.read_text())["infill_B"]
    rows["ring2"] = score(build(res, run, 2, x))
    outp.write_text(json.dumps({"part": res["part"], "rows": rows}))


if __name__ == "__main__":
    main()
