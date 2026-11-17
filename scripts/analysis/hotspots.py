"""Where are a final layout's hot spots? Sum FI^8 over layers per (row, col), take the cells
holding half (and 90%) of the total, group them into spots, and describe each: share of the
score, centre (mm), distance to the outline, and the part fraction in a 1 mm disk (0.5 = straight
edge, < 0.5 convex corner, > 0.5 concave: re-entrant corner or hole edge). Writes a PNG.

usage: python scripts/analysis/hotspots.py RUN_DIR OUT_PREFIX
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root
from rltp.export.state import load_state  # noqa: E402

P = 8.0
run, pre = Path(sys.argv[1]), sys.argv[2]
st = load_state(run / "state.npz")
L0 = st["layers"][0]
H, W = L0["mask"].shape
p = L0["frame"]["p"]
s8 = np.zeros((H, W))
s8_wall = np.zeros((H, W))
union = np.zeros((H, W), bool)
for L in st["layers"]:
    fi = L["fi"].reshape(H, W)
    s8 += fi ** P
    union |= L["part"]
S = s8.sum()
d = ndimage.distance_transform_edt(np.pad(union, 1))[1:-1, 1:-1] * p - p / 2
r = int(round(1.0 / p))
yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
disk = (xx ** 2 + yy ** 2 <= r * r).astype(float)
frac = ndimage.convolve(union.astype(float), disk, mode="constant") / disk.sum()

flat = np.sort(s8.ravel())[::-1]
cum = np.cumsum(flat) / S
out = {"part": run.parent.name, "arm": run.name}
for q in (0.5, 0.9):
    thr = flat[np.searchsorted(cum, q)]
    hot = s8 >= thr
    lab, n = ndimage.label(ndimage.binary_dilation(hot, iterations=int(round(0.5 / p))))
    spots = []
    for k in range(1, n + 1):
        m = (lab == k) & hot
        sh = s8[m].sum() / S
        if sh < 0.02:
            continue
        rr, cc = np.nonzero(m)
        w = s8[m]
        i = int(np.argmax(w))
        spots.append({"share": round(float(sh), 3), "cells_xy": int(m.sum()),
                      "x_mm": round(float(np.average(cc, weights=w) * p), 1),
                      "y_mm": round(float(np.average(rr, weights=w) * p), 1),
                      "dist_edge_mm": round(float(d[rr[i], cc[i]]), 2),
                      "disk_frac": round(float(frac[rr[i], cc[i]]), 2)})
    spots.sort(key=lambda s: -s["share"])
    out[f"q{int(q * 100)}"] = {"cells_xy_share_of_part": round(float(hot.sum() / union.sum()), 4),
                               "n_spots_over_2pct": len(spots), "spots": spots[:8]}

# layers: which layers hold the hot cells (share of S per layer)
out["per_layer"] = [round(float((L["fi"] ** P).sum() / S), 3) for L in st["layers"]]
Path(pre + ".json").write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)

fig, ax = plt.subplots(figsize=(W / 40, H / 40), dpi=100)
img = np.where(union, np.log10(s8 / s8.max() + 1e-12), np.nan)
ax.imshow(img, origin="lower", cmap="inferno", vmin=-4, vmax=0, interpolation="nearest")
thr = flat[np.searchsorted(cum, 0.5)]
ax.contour(s8 >= thr, levels=[0.5], colors="cyan", linewidths=0.8, origin="lower")
ax.set_title(f"{out['part']} {out['arm']}: log10 sum-over-layers FI^8 (cyan: cells holding half)")
ax.set_xticks([]); ax.set_yticks([])
fig.savefig(pre + ".png", bbox_inches="tight")
