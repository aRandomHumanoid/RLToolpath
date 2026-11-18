"""Where a ribbon layout's strength sits, against the reference: per region (walls: the sparse
layers' outer 0.4 mm; solid top/bottom layers; infill 0.4-1.2 mm from a wall; interior infill),
the p-mean FI ((sum FI^8 / cells)^(1/8), like FI_agg over that region) and its share of sum FI^8,
plus the bulk (p = 2) mean. Rebuilds the part as scripts/ribbons.py does.

usage: python scripts/analysis/ribbon_regions.py RUN_DIR [RUN_DIR ...] [--ref PART=STATE.npz ...]
  RUN_DIR: a scripts/ribbons.py output (result.json + state.npz)
  --ref: any saved layout of the same part to compare, e.g. the c20 reference final, recovered
         with `git show 8efb311:runs/selfimprove/c20_bins16/frame/G4/state.npz > ref_frame.npz`
(from the repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ribbons as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from direction_maps import restore  # noqa: E402

from rltp.export.state import load_state  # noqa: E402

P = 8.0
REG = ("walls", "solid", "near", "interior")


def regions(part) -> dict:
    acc = {r: [0.0, 0.0, 0] for r in REG}        # sum FI^8, sum FI^2, cells
    for L, ev in zip(part.layers, part.evals):
        m = L.material.reshape(-1)
        fi = np.asarray(ev.fea.fi, float).reshape(-1)
        if not (L.part & ~L.req).any():
            sel = {"solid": m}
        else:
            p = L.material_frame.p
            d = (ndimage.distance_transform_edt(np.pad(L.part, 1))[1:-1, 1:-1] * p - p / 2).reshape(-1)
            sel = {"walls": m & (d < 0.4), "near": m & (d >= 0.4) & (d < 1.2), "interior": m & (d >= 1.2)}
        for r, s in sel.items():
            acc[r][0] += float(np.sum(fi[s] ** P))
            acc[r][1] += float(np.sum(fi[s] ** 2))
            acc[r][2] += int(s.sum())
    S = sum(v[0] for v in acc.values())
    b = part.breakdown()
    return {"fi_agg": b["fi_agg"], "strokes": b["strokes"], "time_s": b["time_s"],
            **{r: {"p8": (v[0] / v[2]) ** (1 / P) if v[2] else None, "p2": (v[1] / v[2]) ** 0.5 if v[2] else None,
                   "share8": v[0] / S if S else None, "cells": v[2]} for r, v in acc.items()}}


def show(label: str, x: dict) -> None:
    cols = "  ".join(f"{r} {x[r]['p8']:.3f}/{x[r]['p2']:.3f} ({100 * x[r]['share8']:.0f}%)" for r in REG if x[r]["p8"])
    print(f"{label:28s} FI_agg {x['fi_agg']:.3f}  {cols}", flush=True)


def main() -> None:
    args = sys.argv[1:]
    refs = {}
    if "--ref" in args:
        i = args.index("--ref")
        refs = dict(a.split("=", 1) for a in args[i + 1:])
        args = args[:i]
    built = {}
    print("region: p8 FI / p2 FI (share of sum FI^8)")
    for run in map(Path, args):
        res = json.loads((run / "result.json").read_text())
        name = res["part"]
        if name not in built:
            built[name] = R.build(name)
            show(f"{name} slicer", regions(built[name]))
            if name in refs:
                restore(built[name], load_state(refs[name]))
                show(f"{name} reference", regions(built[name]))
        restore(built[name], load_state(run / "state.npz"))
        show(run.name, regions(built[name]))


if __name__ == "__main__":
    main()
