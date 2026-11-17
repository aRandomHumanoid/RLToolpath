"""Where does FI_agg come from? For a run's final layout, split sum FI^8 (FI_agg = (sum FI^8 / N)^(1/8))
by region: solid layers; sparse layers' wall ring (< 0.4 mm from the outline), near-wall infill
(0.4-1.2 mm), interior infill (>= 1.2 mm). Also: how concentrated FI^8 is, the floor FI_agg would hit
if a region's FI went to zero, and (path mode) the share carried by off-class (in-between) beads.

usage: python scripts/analysis/decomp.py RUN_DIR OUT_JSON   (from the repo root)
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
from rltp.cover import paths  # noqa: E402
from rltp.export.state import load_state  # noqa: E402

P = 8.0
run, outp = Path(sys.argv[1]), Path(sys.argv[2])
res = json.loads((run / "result.json").read_text())
args = build_parser().parse_args(run_argv(res, run))
w = parse_weights(args.weights)
_apply_adapt(args, w)
args.start = "slicer"
part = _build(args, _spec(args, w), w)
restore(part, load_state(run / "state.npz"))
part.raw()
b = part.breakdown()

REG = ["solid", "wall", "near", "interior"]
fis = {r: [] for r in REG}
off_fi, on_fi = [], []
chk = []
for L, ev in zip(part.layers, part.evals):
    m = L.material.reshape(-1)
    fi = np.asarray(ev.fea.fi, float).reshape(-1)
    chk.append((float(np.sum(fi[m] ** P)), int(m.sum()), float(ev.fea.S_p), int(ev.fea.N)))
    if not (L.part & ~L.req).any():
        fis["solid"].append(fi[m])
        continue
    H, Wd = L.mask.shape
    p = L.frame.p
    d = (ndimage.distance_transform_edt(np.pad(L.part, 1))[1:-1, 1:-1] * p - p / 2).reshape(-1)
    for r, sel in (("wall", d < 0.4), ("near", (d >= 0.4) & (d < 1.2)), ("interior", d >= 1.2)):
        fis[r].append(fi[m & sel])
    if L.kb > 1:
        a = paths.node_angles(L)
        lock = L.lock.reshape(-1) if L.lock is not None else np.zeros(L.n, bool)
        node = ~np.isnan(a)
        bb = np.where(node, np.round(np.degrees(np.nan_to_num(a)) / 11.25).astype(int) % 16, -1)
        bb = np.where(lock, -2, bb)                                  # walls: neither class
        # each filled cell takes the bin of its nearest bead node
        _, idx = ndimage.distance_transform_edt(~node.reshape(H, Wd), return_indices=True)
        cb = bb.reshape(H, Wd)[idx[0], idx[1]].reshape(-1)
        free = m & (cb >= 0)
        off_fi.append(fi[free & (cb % 4 != 0)])
        on_fi.append(fi[free & (cb % 4 == 0)])

allfi = np.concatenate([np.concatenate(v) for v in fis.values() if v])
N = allfi.size
S = float(np.sum(allfi ** P))
peak = float(allfi.max())
agg = (S / N) ** (1 / P)
out = {"part": res["part"], "arm": res["arm"], "fi_agg_breakdown": b["fi_agg"], "fi_agg_check": agg,
       "peak": peak, "cells": int(N), "regions": {},
       "check_layers": [[round(a / max(c, 1e-30), 4), b == e] for a, b, c, e in chk]}
for r in REG:
    if not fis[r]:
        continue
    x = np.concatenate(fis[r])
    s8 = float(np.sum(x ** P))
    out["regions"][r] = {
        "area": round(x.size / N, 4), "fi8": round(s8 / S, 4),
        "max_rel_peak": round(float(x.max()) / peak, 3),
        "median_rel_peak": round(float(np.median(x)) / peak, 3),
        "below_half_peak": round(float(np.mean(x < 0.5 * peak)), 3),
        "floor_if_zero": round(((S - s8) / N) ** (1 / P) / agg, 4),     # FI_agg / now if this region's FI -> 0
        "floor_if_10pct": round(((S - s8 + s8 * 0.9 ** P) / N) ** (1 / P) / agg, 4),   # region 10% stronger
    }
srt = np.sort(allfi ** P)[::-1]
cum = np.cumsum(srt) / S
out["cells_for_50pct"] = round(float((np.searchsorted(cum, 0.5) + 1) / N), 5)
out["cells_for_90pct"] = round(float((np.searchsorted(cum, 0.9) + 1) / N), 5)
if off_fi:
    xo, xn = np.concatenate(off_fi), np.concatenate(on_fi)
    s_off = float(np.sum(xo ** P))
    out["off_class"] = {"area": round(xo.size / N, 4), "fi8": round(s_off / S, 4),
                        "median_rel_peak": round(float(np.median(xo)) / peak, 3) if xo.size else None,
                        "floor_if_7pct_weaker": round(((S + s_off * (1 / 0.93 ** P - 1)) / N) ** (1 / P) / agg, 4)}
outp.write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)
