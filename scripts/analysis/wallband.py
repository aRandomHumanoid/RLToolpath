"""Do the optimized sparse layers grow wall-like structure next to the walls?

For one suite run: per distance band from the outline (wall ring, then bands inside it),
over every layer, report
  cover : share of the band's area under beads (centerline cells x bead width / area)
  par   : share of bead length in the band running along the local edge (within 22.5 deg)
  fi8   : share of the layers' sum FI^8 sitting in the band
for the slicer start, the gradient start (state0) and the final layout.

Note: the solid top/bottom layers are counted too (load_state gives every layer of a
sparse part a "part" mask, so the skip below never fires). The c18-c20 numbers in
runs/selfimprove/CYCLES.md were made this way; decomp.py and walltrack.py split the solid
layers out.

usage: python scripts/analysis/wallband.py RUN_DIR OUT_JSON   (from the repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, "scripts")
from direction_maps import build_for  # noqa: E402

from rltp.export.state import load_state, save_state  # noqa: E402

BEAD = 0.4
BANDS = [(0.0, 0.4, "wall"), (0.4, 1.2, "0.4-1.2"), (1.2, 2.4, "1.2-2.4"), (2.4, 4.0, "2.4-4"),
         (4.0, 1e9, ">4")]


def bead_angles(L, Wd):
    """Bead direction (radians mod pi, grid x/y) per centerline cell, nan elsewhere."""
    ang = np.full(L["mask"].size, np.nan)
    for s, loop in zip(L["strokes"], L["loops"]):
        n = len(s)
        if n < 3:
            continue
        r, c = np.divmod(np.asarray(s), Wd)
        i = np.arange(n)
        k = min(3, n - 1)
        a, b = ((i + k) % n, (i - k) % n) if loop else (np.minimum(i + k, n - 1), np.maximum(i - k, 0))
        ang[s] = np.arctan2(r[a] - r[b], c[a] - c[b]) % np.pi
    return ang


def stats(st):
    acc = {name: np.zeros(4) for *_, name in BANDS}     # area, centerline, parallel, fi8
    for L in st["layers"]:
        part = L.get("part")
        if part is None:
            continue                                     # solid layer
        H, Wd = part.shape
        p = L["frame"]["p"]
        d = ndimage.distance_transform_edt(np.pad(part, 1))[1:-1, 1:-1] * p - p / 2
        gy, gx = np.gradient(ndimage.gaussian_filter(d, 2.0))
        tang = (np.arctan2(gy, gx) + np.pi / 2) % np.pi
        ang = bead_angles(L, Wd).reshape(H, Wd)
        cl = ~np.isnan(ang) & part
        dev = np.abs((ang - tang + np.pi / 2) % np.pi - np.pi / 2)
        fi8 = np.asarray(L["fi"], float).reshape(H, Wd) ** 8
        for lo, hi, name in BANDS:
            band = part & (d >= lo) & (d < hi)
            acc[name] += [band.sum(), (cl & band).sum(), (cl & band & (dev < np.pi / 8)).sum(),
                          fi8[band].sum()]
    tot8 = sum(v[3] for v in acc.values())
    k = BEAD / p
    return {name: {"cover": round(float(min(v[1] * k / max(v[0], 1), 1.0)), 3),
                   "par": round(float(v[2] / max(v[1], 1)), 3),
                   "fi8": round(float(v[3] / tot8), 3) if tot8 > 0 else 0.0,
                   "area_share": None}
            for name, v in acc.items()} | {"_area": {n: int(v[0]) for n, v in acc.items()}}


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    res, part, _ = build_for(run)
    part.raw()
    slicer = out.with_suffix(".slicer.npz")
    save_state(slicer, part)
    rows = {}
    for label, path in (("slicer", slicer), ("grad", run / "state0.npz"), ("final", run / "state.npz")):
        rows[label] = stats(load_state(path))
    out.write_text(json.dumps({"part": res["part"], "rows": rows}))


if __name__ == "__main__":
    main()
