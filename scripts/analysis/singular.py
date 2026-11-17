"""Are the top hot spots stress singularities (sharp re-entrant corners)? Around each spot's
peak cell, the layer-averaged FI (p-mean over layers) at the edge, binned by distance from the
peak: a singular corner keeps rising as r -> 0 like r^-0.46 (90-degree re-entrant corner),
a smooth notch or hole edge flattens out within a fraction of its radius.

usage: python scripts/analysis/singular.py RUN_DIR SPOTS_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root
from rltp.export.state import load_state  # noqa: E402

P = 8.0
run, spots = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())["q50"]["spots"][:4]
st = load_state(run / "state.npz")
L0 = st["layers"][0]
H, W = L0["mask"].shape
p = L0["frame"]["p"]
s8 = np.zeros((H, W))
cnt = np.zeros((H, W))
union = np.zeros((H, W), bool)
for L in st["layers"]:
    fi = L["fi"].reshape(H, W)
    m = L["mask"] | L["part"]
    s8 += np.where(m, fi ** P, 0)
    cnt += m
    union |= L["part"]
fim = np.where(cnt > 0, (s8 / np.maximum(cnt, 1)) ** (1 / P), 0)
d = ndimage.distance_transform_edt(np.pad(union, 1))[1:-1, 1:-1] * p - p / 2
edge = union & (d < 0.25)                       # the outer two cell rows (the wall's cells)
rr, cc = np.mgrid[0:H, 0:W]
radii = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
for s in spots:
    r0, c0 = s["y_mm"] / p, s["x_mm"] / p
    win = (np.abs(rr - r0) < 30) & (np.abs(cc - c0) < 30) & edge
    i = np.argmax(np.where(win, fim, 0))
    pr, pc = divmod(int(i), W)
    dist = np.hypot(rr - pr, cc - pc) * p
    row = []
    lo = 0.0
    for r in radii:
        ring = edge & (dist > lo) & (dist <= r)
        row.append(round(float(fim[ring].max()) if ring.any() else float("nan"), 3))
        lo = r
    f = np.array(row)
    rad = np.array(radii)
    ok = np.isfinite(f) & (rad <= 1.2)
    slope = float(np.polyfit(np.log(rad[ok]), np.log(f[ok]), 1)[0])
    print(f"spot share {s['share']:.3f} at ({s['x_mm']}, {s['y_mm']}) mm: peak FI {fim[pr, pc]:.3f}; "
          f"max edge FI by ring {dict(zip(radii, row))}; log-log slope 0.1-1.2 mm {slope:+.2f}", flush=True)
