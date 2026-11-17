"""Does the score reward contour-following material next to the wall, at equal material?

Builds a suite run's part as the slicer lays it (1 wall, 40% infill: A) and again with a
second ring and less infill (B), the infill chosen so both deposit the same material.
Scores B against A's per-case references, as the run's score would: FI_agg, the bulk
aggregate, the worst case, compliance, print time. The second ring stands in for a
wall-like band the optimizer could grow; it is a probe of the score, not a proposal.

usage: python scripts/analysis/probe_walls.py RUN_DIR OUT_JSON   (from the repo root)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from direction_maps import run_argv  # noqa: E402

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.cover.beads import density  # noqa: E402


def build(res, run, walls, infill):
    argv = run_argv(res, run)
    i = argv.index("--walls")
    argv[i + 1] = str(walls)
    argv += ["--infill", f"{infill:.4f}"]
    args = build_parser().parse_args(argv)
    w = parse_weights(args.weights)
    _apply_adapt(args, w)
    args.start = "slicer"
    part = _build(args, _spec(args, w), w)
    part.raw()
    return part


def measure(part):
    b = part.breakdown()
    mat = sum(float(np.minimum(density(L), 1.0).sum()) for L in part.layers)
    comp = np.asarray(part.lam.compliance(part.lam_u)) if part.lam is not None else None
    return {"case8": np.asarray(b["case_fi_agg"]), "case2": np.asarray(part.case_fi_agg(2)),
            "material": mat, "time": b["time_s"], "comp": comp}


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = json.loads((run / "result.json").read_text())
    A = measure(build(res, run, 1, 0.4))
    # B's material is linear in its infill: two builds fix the line, a third hits A's
    lo, hi = measure(build(res, run, 2, 0.25)), measure(build(res, run, 2, 0.40))
    x = 0.25 + 0.15 * (A["material"] - lo["material"]) / (hi["material"] - lo["material"])
    B = measure(build(res, run, 2, x))
    r8, r2 = B["case8"] / A["case8"], B["case2"] / A["case2"]
    row = {"part": res["part"], "infill_B": round(x, 4),
           "material_pct": round(100 * (B["material"] / A["material"] - 1), 2),
           "fi_pct": round(100 * (np.mean(r8 ** 8) ** (1 / 8) - 1), 1),
           "bulk_pct": round(100 * (np.mean(r2 ** 2) ** 0.5 - 1), 1),
           "worst_pct": round(100 * (r8.max() - 1), 1), "best_pct": round(100 * (r8.min() - 1), 1),
           "cases_better": int((r8 < 1).sum()), "n_cases": int(r8.size),
           "time_pct": round(100 * (B["time"] / A["time"] - 1), 1)}
    if A["comp"] is not None and B["comp"] is not None:
        row["comp_pct"] = round(100 * (np.mean(B["comp"] / A["comp"]) - 1), 1)
    out.write_text(json.dumps(row))
    print(json.dumps(row))


if __name__ == "__main__":
    main()
