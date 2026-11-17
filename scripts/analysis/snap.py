"""What are a path-mode layout's in-between bead angles worth? Score its start (state0) and
final (state) layouts twice: as printed (true directions), and with every bead's direction
rounded to the nearest of 0/45/90/135 (same beads, same material; only the evaluator's angle
changes). Also: the share of free (non-wall) bead nodes per 11.25-degree bin.

usage: python scripts/analysis/snap.py RUN_DIR OUT_JSON   (from the repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from direction_maps import restore, run_argv  # noqa: E402

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.cover import paths  # noqa: E402
from rltp.export.state import load_state  # noqa: E402

run, outp = Path(sys.argv[1]), Path(sys.argv[2])
res = json.loads((run / "result.json").read_text())
args = build_parser().parse_args(run_argv(res, run))
w = parse_weights(args.weights)
_apply_adapt(args, w)
args.start = "slicer"
part = _build(args, _spec(args, w), w)
orig = paths.node_angles
Q = np.pi / 4


FREE = len(sys.argv) > 3 and sys.argv[3] == "free"     # walls keep their true directions


def snapped(L):
    a = orig(L)
    keep = np.isnan(a)
    if FREE and L.lock is not None:
        keep = keep | L.lock.reshape(-1)
    return np.where(keep, a, (np.round(a / Q) * Q) % np.pi)


def bins(part):
    cnt = np.zeros(16)
    for L in part.layers:
        if L.kb <= 1:
            continue
        a = orig(L)
        lock = L.lock.reshape(-1) if L.lock is not None else np.zeros(L.n, bool)
        ok = ~np.isnan(a) & ~lock
        b = np.round(np.degrees(a[ok]) / 11.25).astype(int) % 16
        cnt += np.bincount(b, minlength=16)
    return cnt / max(cnt.sum(), 1)


out = {"part": res["part"], "arm": res["arm"]}
for when, f in (("start", "state0.npz"), ("final", "state.npz")):
    st = load_state(run / f)
    row = {}
    for mode in ("true", "snapped"):
        paths.node_angles = snapped if mode == "snapped" else orig
        restore(part, st)
        part.raw()
        b = part.breakdown()
        row[mode] = {"fi_agg": b["fi_agg"], "bulk": b["fi_agg2"], "cases": list(map(float, b["case_fi_agg"]))}
    paths.node_angles = orig
    restore(part, st)
    sh = bins(part)
    row["bins"] = [round(float(x), 4) for x in sh]
    row["off_class"] = round(float(sh[[i for i in range(16) if i % 4]].sum()), 4)
    t, s = row["true"], row["snapped"]
    row["snap_fi_pct"] = round(100 * (s["fi_agg"] / t["fi_agg"] - 1), 2)
    row["snap_bulk_pct"] = round(100 * (s["bulk"] / t["bulk"] - 1), 2)
    row["snap_worst_pct"] = round(float(100 * np.max(np.asarray(s["cases"]) / np.asarray(t["cases"]) - 1)), 2)
    out[when] = row
    print(when, {k: v for k, v in row.items() if k not in ("true", "snapped")}, flush=True)
outp.write_text(json.dumps(out))
