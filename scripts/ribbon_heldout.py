"""Score saved layouts on held-out load cases: rebuild each part with other random-case seeds
(--load-seed; the search only ever saw seed 0), and score every layout against that seed's own
slicer layout, so gains fitted to seed 0's twenty cases show up as a drop here.

usage: python scripts/ribbon_heldout.py RUN_DIR|PART=STATE.npz [...] [--seeds 1,2]
  RUN_DIR: a scripts/ribbons.py output (its result.json names the part; its state.npz is scored)
  PART=STATE.npz: any saved layout of a test part, e.g. the c20 reference recovered from git
(from the repo root)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ribbons as R  # noqa: E402
from direction_maps import restore  # noqa: E402

from rltp.export.state import load_state  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    seeds = [0, 1, 2]
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = [0] + [int(s) for s in args[i + 1].split(",")]
        args = args[:i] + args[i + 2:]
    items = []
    for a in args:
        if "=" in a:
            name, st = a.split("=", 1)
            items.append((name, Path(st).name, Path(st)))
        else:
            run = Path(a)
            items.append((json.loads((run / "result.json").read_text())["part"], run.name, run / "state.npz"))
    by_part: dict = {}
    for name, label, st in items:
        by_part.setdefault(name, []).append((label, st))
    print("points against each seed's slicer layout (FI_agg; strength / bulk / stiffness points)")
    for name, runs in by_part.items():
        w = R.REF[name]["weights"]
        rows = {label: [] for label, _ in runs}
        for s in seeds:
            part = R.build(name, ["--load-seed", str(s)])
            raw_s = part.raw()
            for label, st in runs:
                restore(part, load_state(st))
                o = R.objective(raw_s, part.raw(), w, raw_s)
                rows[label].append(f"{o['total']:+6.1f} ({part.breakdown()['fi_agg']:.3f}; s/b/k "
                                   f"{o['strength']:+.0f}/{o['bulk']:+.0f}/{o['stiff']:+.0f})")
        print(f"{name}: seeds {', '.join(map(str, seeds))}")
        for label, vals in rows.items():
            print(f"  {label:24s} " + "   ".join(vals), flush=True)


if __name__ == "__main__":
    main()
