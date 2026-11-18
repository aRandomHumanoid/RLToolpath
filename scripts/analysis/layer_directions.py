"""Bead directions and fill per layer of a saved layout: for every layer, the share of its bead
material along 0 / 45 / 90 / 135 degrees (rltp.cover.beads.bead_class, path mode's nearest class),
its filled share of the part, and the layer's p = 8 FI. For comparing what a layout does in its
skins and infill (e.g. the c20 reference against a ribbon result).

usage: python scripts/analysis/layer_directions.py PART STATE.npz [STATE.npz ...]   (from the repo root)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ribbons as R  # noqa: E402
from direction_maps import restore  # noqa: E402

from rltp.cover.beads import bead_class  # noqa: E402
from rltp.export.state import load_state  # noqa: E402


def main() -> None:
    name, states = sys.argv[1], sys.argv[2:]
    part = R.build(name)
    for st in states:
        restore(part, load_state(st))
        print(f"== {st}: FI_agg {part.breakdown()['fi_agg']:.3f}")
        for k, (L, ev) in enumerate(zip(part.layers, part.evals)):
            if k not in (0, 1, 2, 3, 12, len(part.layers) - 3, len(part.layers) - 2, len(part.layers) - 1):
                continue
            cls = bead_class(L)
            m = L.material
            sh = [float((cls[m] == c).mean()) if m.any() else 0.0 for c in (0, 45, 90, 135)]
            fi = np.asarray(ev.fea.fi, float).reshape(L.H, L.W)[m]
            p8 = float(np.mean(fi ** 8) ** 0.125) if fi.size else 0.0
            kind = "skin" if not (L.part & ~L.req).any() else "infill"
            print(f"  layer {k:2d} {kind:6s} fill {m.sum() / L.part.sum():.2f}  0/45/90/135 "
                  + " ".join(f"{100 * s:3.0f}%" for s in sh) + f"  p8 FI {p8:.3f}", flush=True)


if __name__ == "__main__":
    main()
