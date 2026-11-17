"""Free (non-wall) strokes of a path-mode run's final layout by direction: class angles
(0/45/90/135, within 5.6 deg) vs in-between ones, from the fitted segments of each stroke
(its length-weighted dominant 11.25-degree bin). usage: python scripts/analysis/strokes.py RUN_DIR [state.npz]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from direction_maps import restore, run_argv  # noqa: E402

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.cover import beads  # noqa: E402
from rltp.export.state import load_state  # noqa: E402

run = Path(sys.argv[1])
fname = sys.argv[2] if len(sys.argv) > 2 else "state.npz"
res = json.loads((run / "result.json").read_text())
args = build_parser().parse_args(run_argv(res, run))
w = parse_weights(args.weights)
_apply_adapt(args, w)
args.start = "slicer"
part = _build(args, _spec(args, w), w)
restore(part, load_state(run / fname))
st = {"class": [0, 0.0], "off": [0, 0.0], "dot": [0, 0.0]}
bylen = {b: [0, 0.0] for b in range(16)}
for L in part.layers:
    if L.kb <= 1:
        continue
    p = L.material_frame.p
    lock = L.lock.reshape(-1) if L.lock is not None else np.zeros(L.n, bool)
    poly = beads.runs(L)["poly"]
    for pid, ent in poly.items():
        nodes, V = ent[0], ent[2]
        if lock[np.asarray(nodes)].any():
            continue
        d = np.diff(np.asarray(V, float), axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        if seg.sum() <= 0:
            st["dot"][0] += 1
            continue
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180
        b = np.round(ang / 11.25).astype(int) % 16
        wb = np.bincount(b, seg, minlength=16)
        bb = int(np.argmax(wb))
        key = "class" if bb % 4 == 0 else "off"
        st[key][0] += 1
        st[key][1] += seg.sum() * p
        bylen[bb][0] += 1
        bylen[bb][1] += seg.sum() * p
out = {k: {"n": v[0], "mm": round(v[1]), "mean_mm": round(v[1] / max(v[0], 1), 1)} for k, v in st.items()}
out["by_bin"] = {beads.bin_deg(b): (v[0], round(v[1] / max(v[0], 1), 1)) for b, v in bylen.items() if v[0]}
print(res["part"], res["arm"], fname, json.dumps(out))
