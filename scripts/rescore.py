"""Score saved suite layouts (state.npz) under one run's settings, against that run's slicer
start: the objective (README convention), FI_agg, the bulk aggregate, the worst load case,
the aligned share, the share of sum FI^8 in the walls, print time, strokes and free ends.
Lets runs made under different settings (e.g. before the clamp band) be compared on one
footing.

usage: python scripts/rescore.py SETTINGS_RUN LABEL=STATE_DIR [LABEL=STATE_DIR ...]
  SETTINGS_RUN: a suite run dir (OUT/<part>/<arm>) whose settings score everything;
  STATE_DIR: a run dir holding state.npz (its final layout), a state .npz file, or `start`
  for the slicer start.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_maps import build_for, restore  # noqa: E402
from suite import objective  # noqa: E402

from rltp.eval.adapt import orientation  # noqa: E402
from rltp.export.state import load_state  # noqa: E402


def main():
    settings = Path(sys.argv[1])
    res, part, w = build_for(settings)
    raw0, b0 = part.raw(), part.breakdown()
    cf0 = np.asarray(b0["case_fi_agg"])
    rows = []
    for spec_ in sys.argv[2:]:
        label, _, d = spec_.partition("=")
        if d != "start":
            restore(part, load_state(Path(d) if d.endswith(".npz") else Path(d) / "state.npz"))
        raw, b = part.raw(), part.breakdown()
        o = objective(raw0, raw, w, raw0)
        rows.append({"label": label, "objective": round(o["total"], 1),
                     "terms": {k: round(float(v), 1) for k, v in o.items() if k != "total"},
                     "fi_pct": round(100 * (b["fi_agg"] / b0["fi_agg"] - 1), 1),
                     "bulk_pct": round(100 * (b["fi_agg2"] / b0["fi_agg2"] - 1), 1),
                     "worst_pct": round(float(100 * np.max(np.asarray(b["case_fi_agg"]) / cf0 - 1)), 1),
                     "aligned": round((o := orientation(part))["aligned_p2"], 3),
                     "wall_p8": round(o["wall_p8"], 3), "eff_p8": round(o["eff_p8"], 4),
                     "time_s": round(b["time_s"], 1), "strokes": b["strokes"],
                     "free_ends": b.get("free_ends")})
    print(json.dumps({"part": res["part"], "rows": rows}))


if __name__ == "__main__":
    main()
