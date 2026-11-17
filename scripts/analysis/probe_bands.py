"""Where does the second ring (probe_walls.py) raise or lower the score's sum FI^8?

Per load case c, the score's FI_agg^8 ratio of layout B to the slicer start A is
S_c(B) / S_c(A), S_c = sum over layers and cells of fill x FI_c^8 (the evaluator's rule:
a cell at the worst direction class it holds). Split by distance from the outline, the
terms S_c,band / S_c(A) of A sum to 1 and those of B to the ratio; the mean over cases of
the ratio is FI_agg(B)^8 / FI_agg(A)^8.

usage: python scripts/analysis/probe_bands.py RUN_DIR PROBE_WALLS_JSON OUT_JSON [old|new]
  old: the run's own loads (its loads.yaml + random cases drawn alike);
  new: load-agnostic (no loads.yaml, 10 + 10 random cases, edges drawn hole 3 : side 2 : arc 1)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path(__file__).parent))
from probe_walls import build  # noqa: E402

from rltp.cover.beads import cell_shares, cell_wh, density  # noqa: E402
from rltp.eval.adapt import _cell_strains  # noqa: E402
from rltp.eval.fea2d import to_material, tsai_wu_fi  # noqa: E402

BANDS = ((0.0, 0.4, "wall"), (0.4, 0.8, "ring2"), (0.8, 2.4, "0.8-2.4"), (2.4, 4.0, "2.4-4"), (4.0, 1e9, ">4"))


def sums(part):
    lam = part.lam
    eps = _cell_strains(lam, part.lam_u)
    C = eps.shape[1]
    fic = np.stack([tsai_wu_fi(*to_material(*np.moveaxis(eps @ Q.T, -1, 0), a), lam.mat)
                    for Q, a in zip((lam.Q0, lam.Q45, lam.Q90, lam.Q135),
                                    (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4))], -1)
    keep = getattr(lam, "keep", None)
    kp = 1.0 if keep is None else np.asarray(keep, float).reshape(len(eps), -1)
    S = np.zeros((len(BANDS), C))
    for L in part.layers:
        D = np.minimum(density(L), 1.0).reshape(-1)[lam.cells]
        sh = cell_shares(L)
        if sh is None:
            wh = cell_wh(L)
            sh = np.stack([wh, 0 * wh, 1 - wh, 0 * wh], -1)
        pres = (sh[lam.cells] > 1e-9) & (D > 1e-9)[:, None]
        f = np.max(np.where(pres[:, None, :], fic, 0.0), -1) * kp
        f8 = D[:, None] * f ** 8
        pm = L.part_mask
        p = L.material_frame.p
        dist = (ndimage.distance_transform_edt(np.pad(pm, 1))[1:-1, 1:-1] * p - p / 2).reshape(-1)[lam.cells]
        for b, (lo, hi, _) in enumerate(BANDS):
            S[b] += f8[(dist >= lo) & (dist < hi)].sum(0)
    return S


def main():
    run, pw, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "old"
    res = json.loads((run / "result.json").read_text())
    if mode == "new":
        res = dict(res, yaml_loads=False, load_weights=[3.0, 2.0, 1.0])
    x = json.loads(pw.read_text())["infill_B"]
    SA = sums(build(res, run, 1, 0.4))
    SB = sums(build(res, run, 2, x))
    tA = SA.sum(0)
    ratio = SB.sum(0) / tA
    row = {"part": res["part"], "mode": mode, "n_cases": int(tA.size),
           "fi_pct": round(100 * (ratio.mean() ** (1 / 8) - 1), 1),
           "worst_pct": round(100 * (ratio.max() ** (1 / 8) - 1), 1),
           "cases_better": int((ratio < 1).sum()),
           "bands": {name: [round(float((SA[b] / tA).mean()), 3), round(float((SB[b] / tA).mean()), 3)]
                     for b, (_, _, name) in enumerate(BANDS)},
           "case_ratio_pct": [round(100 * (r ** (1 / 8) - 1), 1) for r in ratio]}
    outp.write_text(json.dumps(row))
    print(json.dumps(row))


if __name__ == "__main__":
    main()
