"""Write viewer.html for finished suite runs (scripts/suite.py without --viewer).

Rebuilds each run's part from its result.json, restores its saved start (state0.npz) for the
baseline numbers and its final layout (state.npz), replays history.jsonl(.gz) into the History
tab, and writes RUN_DIR/viewer.html. Runs scored with --directions path draw their fitted
printed moves (the "printed path" toggle).

usage: python scripts/run_viewer.py RUN_DIR [RUN_DIR ...]   (from the repo root)
"""
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_maps import restore, run_argv  # noqa: E402

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.export.state import load_state  # noqa: E402
from rltp.export.viewer import write_viewer  # noqa: E402


def read_events(run: Path) -> list[dict]:
    for name, opener in (("history.jsonl", open), ("history.jsonl.gz", gzip.open)):
        p = run / name
        if p.exists():
            with opener(p, "rt") as f:
                return [json.loads(x) for x in f if x.strip()]
    return []


def main(run: Path) -> None:
    t = time.time()
    res = json.loads((run / "result.json").read_text())
    args = build_parser().parse_args(run_argv(res, run))
    w = parse_weights(args.weights)
    _apply_adapt(args, w)
    args.start = "slicer"
    spec = _spec(args, w)
    part = _build(args, spec, w)
    restore(part, load_state(run / "state0.npz"))
    b0 = part.breakdown()
    restore(part, load_state(run / "state.npz"))
    b1 = part.breakdown()
    meta = {"command": "suite", "arm": res["arm"], "args": res.get("args"), "evals": res.get("evals"),
            "directions": res.get("directions", "classes"), "baseline": b0, "final": b1,
            "load_cases": spec.loads.describe_cases(), "start": (res.get("start_info") or {}).get("start")}
    write_viewer(run / "viewer.html", part, read_events(run), meta)
    mb = (run / "viewer.html").stat().st_size / 1e6
    print(f"{run}/viewer.html: {mb:.1f} MB ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(Path(a))
