"""Compare suite runs of one part on the same footing: each run's final layout scored with one
weight vector against one slicer start (the reference run's), plus FI_agg, the bulk aggregate
and the worst load case against that start, the aligned share, print time and strokes.

usage: python scripts/compare.py ROOT REF_RUN RUN [RUN ...] [--parts a,b,c]
  RUN = cycle/arm (e.g. c08/GB2); REF_RUN gives the slicer start and the weights.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite import objective  # noqa: E402


def main():
    argv = sys.argv[1:]
    parts = ["bracket", "link", "hook", "frame", "wrench"]
    if "--parts" in argv:
        i = argv.index("--parts")
        parts = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    root, ref_run, runs = Path(argv[0]), argv[1], argv[2:]
    print(f"{'part':8s} {'run':10s} {'objective':>9s} {'FI_agg':>7s} {'bulk':>6s} {'worst':>6s} {'aligned':>7s} "
          f"{'time s':>7s} {'strokes':>7s}")
    means: dict = {}
    for p in parts:
        cyc, arm = ref_run.split("/")
        rf = root / cyc / p / arm / "result.json"
        if not rf.exists():
            continue
        ref = json.loads(rf.read_text())
        w = ref["weights"]
        for run in runs:
            c, a = run.split("/")
            f = root / c / p / a / "result.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            o = objective(ref["raw0"], r["raw1"], w, ref["raw0"])["total"]
            fi = math.exp(-r["raw1"]["strength"]) - 1
            bulk = math.exp(-r["raw1"]["bulk"]) - 1 if "bulk" in r["raw1"] else float("nan")
            worst = max(x / y - 1 for x, y in zip(r["case_fi1"], ref["case_fi0"]))
            al = r["orient1"]["aligned_p2"]
            print(f"{p:8s} {run:10s} {o:+9.1f} {100 * fi:+6.1f}% {100 * bulk:+5.1f}% {100 * worst:+5.1f}% "
                  f"{al:7.3f} {r['time1']:7.0f} {r['strokes1']:7d}")
            m = means.setdefault(run, [])
            m.append((o, fi, bulk, worst, al))
    print("mean over the parts each run has:")
    for run, m in means.items():
        n = len(m)
        print(f"  {run:10s} n={n} objective {sum(x[0] for x in m) / n:+.1f}, FI_agg {100 * sum(x[1] for x in m) / n:+.1f}%, "
              f"bulk {100 * sum(x[2] for x in m) / n:+.1f}%, worst {100 * sum(x[3] for x in m) / n:+.1f}%, "
              f"aligned {sum(x[4] for x in m) / n:.3f}")


if __name__ == "__main__":
    main()
