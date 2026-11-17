"""What is one more unit of infill worth at each distance from the outline?

The gradient layout step's field model (rltp.search.fields.FieldModel) on fields read off
a layout: rho = the sparse layers' mean fill of each optional cell, theta = the doubled-angle
mean of the bead directions in each cell. Its gradient, weighted as the gradient start
weights it (strength, bulk and stiffness in the score's units), is averaged per distance
band and given as score points for moving 1% of the part's optional material into the band
(the part-wide mean subtracted, so a band above 0 is where material is worth more than on
average). Layouts: slicer start, gradient start (state0), final (state).

usage: python scripts/analysis/probe_grad.py RUN_DIR OUT_JSON [old|new]   (from the repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from direction_maps import build_for, restore, run_argv  # noqa: E402

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.cover.beads import cell_shares, cell_wh, density  # noqa: E402
from rltp.eval.adapt import _cell_strains  # noqa: E402
from rltp.eval.fea2d import q_matrix, tsai_wu_fi  # noqa: E402
from rltp.search.fields import strain_rot  # noqa: E402
from rltp.eval.metrics import full_weights  # noqa: E402
from rltp.export.state import load_state  # noqa: E402
from rltp.search.fields import FieldModel, wall_cells  # noqa: E402

BANDS = ((0.4, 1.2, "0.4-1.2"), (1.2, 2.4, "1.2-2.4"), (2.4, 4.0, "2.4-4"), (4.0, 1e9, ">4"))


def fields(part, model):
    cells = model.cells
    nc = cells.size
    rn, vx, vy = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    for L in part.layers:
        inpart = L.part_mask.reshape(-1)[cells]
        wall = wall_cells(L).reshape(-1)[cells]
        req = L.req.reshape(-1)[cells] if L.req is not None else inpart
        fld = inpart & ~wall
        opt = fld & ~req
        D = np.minimum(density(L), 1.0).reshape(-1)[cells]
        sh = cell_shares(L)
        if sh is None:
            wh = cell_wh(L)
            sh = np.stack([wh, 0 * wh, 1 - wh, 0 * wh], -1)
        sh = sh[cells]
        rn += D * opt
        w = D * fld
        vx += w * (sh[:, 0] - sh[:, 2])
        vy += w * (sh[:, 1] - sh[:, 3])
    rho = np.clip(rn / np.maximum(model.npn, 1), 0.0, 1.0)
    return 0.5 * np.arctan2(vy, vx), rho


def probe(part, w):
    model = FieldModel(part, bulk=True)
    th, rho = fields(part, model)
    ev = model.evaluate(th, rho, grad=True)
    wf = full_weights(w)
    sc = part.norm.scale
    ws = wf[0] / sc("strength") / model.p
    wst = wf[6] / sc("stiff") if "stiff" in part.terms else 0.0
    wb = wf[7] / sc("bulk") if "bulk" in part.terms else 0.0
    comps = {"strength": ws * ev.g_rho / ev.J,
             "bulk": 0.5 * wb * ev.g_rho_b / ev.J2 if ev.g_rho_b is not None else 0 * ev.g_rho,
             "stiff": wst * ev.g_rho_s / ev.stiff if ev.g_rho_s is not None else 0 * ev.g_rho}
    comps["total"] = comps["strength"] + comps["bulk"] + comps["stiff"]
    # strength's explicit part: the added material's own FI^p terms (the rest is stiffening,
    # which lowers every cell's strain); from the evaluator's strains at the field directions
    lam = part.lam
    eps = _cell_strains(lam, part.lam_u)
    s = np.einsum("eij,ecj->eci", q_matrix(lam.mat) @ strain_rot(th), eps)
    fi = tsai_wu_fi(s[..., 0], s[..., 1], s[..., 2], lam.mat)
    kp = getattr(lam, "keep", None)
    if kp is not None:
        fi = fi * np.asarray(kp, float).reshape(len(eps), -1)
    expl = model.npn * np.mean((fi * model.cs[None, :]) ** model.p, 1)
    comps["str_explicit"] = ws * expl / ev.J * model.rho_cells
    comps["str_stiffening"] = comps["strength"] - comps["str_explicit"]
    L = part.layers[len(part.layers) // 2]
    p = L.material_frame.p
    dist = (ndimage.distance_transform_edt(np.pad(L.part_mask, 1))[1:-1, 1:-1] * p - p / 2).reshape(-1)[model.cells]
    on = model.rho_cells
    npn = model.npn
    move = 0.01 * float(npn[on].sum())                 # 1% of the optional cell-layers
    out = {"fi_model": round(float(ev.fi_agg), 4), "fi_eval": round(float(part.breakdown()["fi_agg"]), 4),
           "rho_by_band": {}, "points": {}}
    for name, g in comps.items():
        per = g / np.maximum(npn, 1)                   # d F / d (one cell-layer of material)
        mean = float((g[on]).sum() / npn[on].sum())
        out["points"][name] = {b: round(float(-100 * move * ((g[sel]).sum() / npn[sel].sum() - mean)), 2)
                               for lo, hi, b in BANDS if (sel := on & (dist >= lo) & (dist < hi)).any()}
        out["points"][name]["mean"] = round(-100 * move * mean, 2)
        del per
    for lo, hi, b in BANDS:
        sel = on & (dist >= lo) & (dist < hi)
        if sel.any():
            out["rho_by_band"][b] = round(float((rho[sel] * npn[sel]).sum() / npn[sel].sum()), 3)
    return out


def main():
    run, outp = Path(sys.argv[1]), Path(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "old"
    res, part, w = build_for(run)
    if mode == "new":
        res = dict(res, yaml_loads=False, load_weights=[3.0, 2.0, 1.0])
        args = build_parser().parse_args(run_argv(res, run))
        w = parse_weights(args.weights)
        _apply_adapt(args, w)
        args.start = "slicer"
        part = _build(args, _spec(args, w), w)
    part.raw()
    rows = {"slicer": probe(part, w)}
    for label, f in (("grad", "state0.npz"), ("final", "state.npz")):
        restore(part, load_state(run / f))
        part.raw()
        rows[label] = probe(part, w)
    outp.write_text(json.dumps({"part": res["part"], "mode": mode, "rows": rows}))
    print(json.dumps({"part": res["part"], "mode": mode, "rows": rows}))


if __name__ == "__main__":
    main()
