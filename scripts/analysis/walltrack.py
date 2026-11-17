"""Does the infill change how hard the (locked) walls work? Per region, the p-mean FI
((sum FI^8 / cells)^(1/8), like FI_agg but over that region only) for the slicer start,
the gradient start (state0) and the final layout (state).

usage: python scripts/analysis/walltrack.py RUN_DIR OUT_JSON   (from the repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from direction_maps import restore, run_argv  # noqa: E402

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.export.state import load_state  # noqa: E402

P = 8.0
REG = ["solid", "wall", "near", "interior"]
run, outp = Path(sys.argv[1]), Path(sys.argv[2])
res = json.loads((run / "result.json").read_text())
args = build_parser().parse_args(run_argv(res, run))
w = parse_weights(args.weights)
_apply_adapt(args, w)
args.start = "slicer"
part = _build(args, _spec(args, w), w)


def regions(part):
    acc = {r: [0.0, 0] for r in REG}
    for L, ev in zip(part.layers, part.evals):
        m = L.material.reshape(-1)
        fi = np.asarray(ev.fea.fi, float).reshape(-1)
        if not (L.part & ~L.req).any():
            acc["solid"][0] += float(np.sum(fi[m] ** P))
            acc["solid"][1] += int(m.sum())
            continue
        p = L.frame.p
        d = (ndimage.distance_transform_edt(np.pad(L.part, 1))[1:-1, 1:-1] * p - p / 2).reshape(-1)
        for r, sel in (("wall", d < 0.4), ("near", (d >= 0.4) & (d < 1.2)), ("interior", d >= 1.2)):
            acc[r][0] += float(np.sum(fi[m & sel] ** P))
            acc[r][1] += int((m & sel).sum())
    b = part.breakdown()
    S = sum(v[0] for v in acc.values())
    return {"fi_agg": b["fi_agg"], "time_s": b["time_s"],
            "pmean": {r: (v[0] / v[1]) ** (1 / P) if v[1] else None for r, v in acc.items()},
            "share": {r: v[0] / S for r, v in acc.items()},
            "cells": {r: v[1] for r, v in acc.items()}}


out = {"part": res["part"], "arm": res["arm"]}
part.raw()
out["slicer"] = regions(part)
for label, f in (("start", "state0.npz"), ("final", "state.npz")):
    restore(part, load_state(run / f))
    part.raw()
    out[label] = regions(part)
outp.write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)
