"""c20: raw finals per arm, op acceptance, start_info. usage: python scripts/analysis/c20_raw.py RUN_ROOT"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
for p in ["frame", "plate", "wrench"]:
    rs = {a: json.loads((root / p / a / "result.json").read_text()) for a in ["G4", "G16"]}
    print(f"== {p}")
    for a, r in rs.items():
        cf0, cf1 = np.asarray(r["case_fi0"]), np.asarray(r["case_fi1"])
        print(f"  {a:4s} fi {r['fi0']:.4f}->{r['fi1']:.4f}  cases mean {cf1.mean():.4f} max {cf1.max():.4f}  "
              f"time {r['time0']:.0f}->{r['time1']:.0f}  strokes {r['strokes0']}->{r['strokes1']}  "
              f"evals {r['evals']}  weights {r['weights']}")
        print("       raw0", {k: round(v, 4) for k, v in r["raw0"].items()})
        print("       raw1", {k: round(v, 4) for k, v in r["raw1"].items()})
        print("       start_info", r["start_info"])
        print("       ops", r["ops"])
    g4, g16 = rs["G4"], rs["G16"]
    c4, c16 = np.asarray(g4["case_fi1"]), np.asarray(g16["case_fi1"])
    print(f"  G16/G4 final fi {g16['fi1'] / g4['fi1'] - 1:+.2%}  per-case {np.min(c16 / c4 - 1):+.1%}..{np.max(c16 / c4 - 1):+.1%}"
          f"  start fi {g16['fi0'] / g4['fi0'] - 1:+.2%}")
