"""Score saved layouts on held-out random load cases (other --load-seed values) against the
slicer start under those same cases: FI_agg, bulk and worst-case change, per seed.

usage: python scripts/analysis/heldout.py SETTINGS_RUN OUT_JSON SEEDS LABEL=STATE ...   (from the repo root)
  SEEDS: comma list, e.g. 1,2,3; STATE: a run dir (its state.npz) or an .npz file
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from direction_maps import restore, run_argv  # noqa: E402

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.export.state import load_state  # noqa: E402


def main():
    run, outp, seeds = Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")]
    labels = [a.partition("=") for a in sys.argv[4:]]
    res = json.loads((run / "result.json").read_text())
    out = {"part": res["part"], "seeds": {}}
    states = {lab: load_state(Path(p) if p.endswith(".npz") else Path(p) / "state.npz") for lab, _, p in labels}
    for seed in seeds:
        args = build_parser().parse_args(run_argv(res, run) + ["--load-seed", str(seed)])
        w = parse_weights(args.weights)
        _apply_adapt(args, w)
        args.start = "slicer"
        part = _build(args, _spec(args, w), w)
        part.raw()
        b0 = part.breakdown()
        c0 = np.asarray(b0["case_fi_agg"])
        rows = {}
        for lab, st in states.items():
            restore(part, st)
            part.raw()
            b = part.breakdown()
            c = np.asarray(b["case_fi_agg"])
            rows[lab] = {"fi_pct": round(100 * (b["fi_agg"] / b0["fi_agg"] - 1), 1),
                         "bulk_pct": round(100 * (b["fi_agg2"] / b0["fi_agg2"] - 1), 1),
                         "worst_pct": round(float(100 * np.max(c / c0 - 1)), 1),
                         "cases_better": int((c < c0).sum()), "n_cases": int(c.size)}
        out["seeds"][seed] = rows
    outp.write_text(json.dumps(out))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
