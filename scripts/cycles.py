"""Roll up scripts/suite.py summaries: per cycle and arm, the mean over parts of the
objective, FI_agg change, worst case change, share of bead directions changed, and the
orientation measures (rltp.eval.adapt) at the end.

usage: python scripts/cycles.py runs/selfimprove [--parts]   (--parts: one row per part too)
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
per_part = "--parts" in sys.argv
print(f"{'cycle':6s} {'arm':5s} {'n':>2s} {'objective':>9s} {'FI%':>6s} {'worst%':>7s} {'dirchg%':>8s} "
      f"{'fillchg%':>8s} {'eff2':>6s} {'eff8':>6s} {'align2':>6s} {'strokes%':>8s}")
for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()):
    rows = json.loads((d / "summary.json").read_text())
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]

        def m(f):
            return float(np.mean([f(r) for r in rs]))
        print(f"{d.name:6s} {arm:5s} {len(rs):2d} {m(lambda r: r['final_obj']):9.1f} {m(lambda r: r['fi_change_pct']):6.1f} "
              f"{m(lambda r: r['case_change_pct_worst']):7.1f} {100 * m(lambda r: r['dir_changed']):8.1f} "
              f"{100 * m(lambda r: r['fill_changed']):8.1f} {m(lambda r: r['eff_p2'][1]):6.3f} {m(lambda r: r['eff_p8'][1]):6.3f} "
              f"{m(lambda r: r['aligned_p2'][1]):6.3f} {m(lambda r: 100 * (r['strokes'][1] / r['strokes'][0] - 1)):8.1f}")
        if per_part:
            for r in rs:
                print(f"{'':6s} {'':5s}   {r['final_obj']:9.1f} {r['fi_change_pct']:6.1f} {r['case_change_pct_worst']:7.1f} "
                      f"{100 * r['dir_changed']:8.1f} {100 * r['fill_changed']:8.1f} {r['eff_p2'][1]:6.3f} "
                      f"{r['eff_p8'][1]:6.3f} {r['aligned_p2'][1]:6.3f} {100 * (r['strokes'][1] / r['strokes'][0] - 1):8.1f}  {r['part']}")
