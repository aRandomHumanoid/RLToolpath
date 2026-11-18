"""Run ribbon arms (scripts/ribbons.py settings) over the test parts and tabulate them against the
reference (examples/c20_reference.json) and against each other.

usage:
  python scripts/ribbon_suite.py OUT_DIR --arm NAME="ribbons.py args" [--arm ...]
      [--parts frame,plate,wrench] [--jobs 1] [--angles frame=0,plate=90,wrench=0]
  python scripts/ribbon_suite.py OUT_DIR --table          (only rebuild the summary)

In an arm's args, {part} is the part's name, {angle} its angle from --angles (the best
straight start per part) and {alt} that angle + 90, e.g.
  --arm 3D="--angle {angle} --z-levels 3 --warm runs/ribbons/c00_baseline/{part}_A/result.json"
Runs go to OUT_DIR/<part>_<arm>/ with a log each; --jobs runs that many at once (each run's
workers come from its own args, --workers; mind memory: ~1.4 GB per process plus ~1.5 GB per
worker). Writes OUT_DIR/summary.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "examples/c20_reference.json").read_text())
TERMS = ("strength", "bulk", "stiff", "cont", "time", "inter", "cover", "infill")


def run(job) -> tuple[str, int, float]:
    out, part, arm, args = job
    d = out / f"{part}_{arm}"
    log = out / f"{part}_{arm}.log"
    cmd = [sys.executable, str(ROOT / "scripts/ribbons.py"), part, str(d)] + shlex.split(args)
    t = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1"))
    return f"{part}_{arm}", rc, time.time() - t


def table(out: Path) -> str:
    rows = []
    for f in sorted(out.glob("*/result.json")):
        r = json.loads(f.read_text())
        for lab in ("start", "best"):
            if lab == "best" and r["settings"]["evals"] == 0:
                continue
            s = r[lab]
            rows.append({"run": f.parent.name, "part": r["part"], "which": lab, "objective": s["objective"]["total"],
                         "fi_agg": s["fi_agg"], "time_s": s["time_s"], "strokes": s["strokes"],
                         "infill_pct": s["infill_pct"], "lean_deg": s.get("lean_deg"),
                         **{t: s["objective"].get(t) for t in TERMS}})
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    lines = ["| run | | points | FI_agg | print time | strokes | infill % | " + " | ".join(TERMS) + " |",
             "| --- | --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in TERMS) + " |"]
    for part in sorted({r["part"] for r in rows}):
        ref = REF[part]["final"]
        lines.append(f"| {part}: c20 final (reference) | | {ref['objective']['total']:+.1f} | {ref['fi_agg']:.3f} | "
                     f"{ref['time_s']:.0f} | {ref['strokes']} | {ref['infill_pct']:.1f} | "
                     + " | ".join(f"{ref['objective'].get(t, 0):+.1f}" for t in TERMS) + " |")
        for r in [r for r in rows if r["part"] == part]:
            lean = f" (lean {r['lean_deg']:.0f}°)" if r.get("lean_deg") is not None else ""
            lines.append(f"| {r['run']}{lean} | {r['which']} | {r['objective']:+.1f} | {r['fi_agg']:.3f} | "
                         f"{r['time_s']:.0f} | {r['strokes']} | {r['infill_pct']:.1f} | "
                         + " | ".join(f"{r[t]:+.1f}" if r[t] is not None else "–" for t in TERMS) + " |")
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--arm", action="append", default=[], help='NAME="ribbons.py args"')
    ap.add_argument("--parts", default="frame,plate,wrench")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--angles", default="frame=0,plate=90,wrench=0")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.table:
        angles = dict(kv.split("=") for kv in a.angles.split(","))
        jobs = []
        for arm in a.arm:
            name, args = arm.split("=", 1)
            for part in a.parts.split(","):
                ang = angles.get(part, "0")
                jobs.append((out, part, name, args.format(part=part, angle=ang, alt=f"{(float(ang) + 90) % 180:g}")))
        with ThreadPoolExecutor(max(1, a.jobs)) as ex:
            for name, rc, dt in ex.map(run, jobs):
                print(f"[suite] {name}: exit {rc}, {dt / 60:.1f} min", flush=True)
    print(table(out), flush=True)


if __name__ == "__main__":
    main()
