"""One gradient-start field optimisation per part, laid out under several line rules:
FI_agg, bulk, objective (vs the slicer start), free ends, strokes, print time, infill.
usage: python scripts/analysis/probe_layouts.py PART [WORKERS]"""
import json
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]      # repo root
sys.path.insert(0, str(R / "scripts"))
from suite import COMMON, ROOT, objective  # noqa: E402

import rltp.cover.beads as B  # noqa: E402
import rltp.part as P  # noqa: E402
import rltp.search.fields as F  # noqa: E402
from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.eval.adapt import orientation  # noqa: E402

name = sys.argv[1]
workers = sys.argv[2] if len(sys.argv) > 2 else "3"
cache, cap = {}, {}
orig_ms, orig_gs, orig_anchor = F.multi_start, P._gradient_start, B.anchor_ends


def ms(*a, **k):
    if "r" not in cache:
        cache["r"] = orig_ms(*a, **k)
    return cache["r"]


def gs(*a, **k):
    cap["a"], cap["k"] = a, k
    return orig_gs(*a, **k)


F.multi_start, P._gradient_start = ms, gs
args = build_parser().parse_args(["optimize", str(ROOT / f"examples/{name}.stl"), "--loads",
                                  str(ROOT / f"examples/{name}_loads.yaml"), "--out", "/tmp/x"] + COMMON
                                 + ["--adapt", "--workers", workers])
w = parse_weights(args.weights)
_apply_adapt(args, w)
spec = _spec(args, w)
t = time.time()
_build(args, spec, w)
print(f"{name}: field step {time.time() - t:.0f}s", flush=True)
layers, spec_, z0, field, lay_out = cap["a"][:5]
refs = cap["k"].get("refs")
lay_out(None)
slicer = P.PartState(layers, spec_, z0=z0, refs=refs)
raw0 = slicer.raw()
rows = []
for label, runs, turns, anchor in (("cut lines, no anchoring (c16)", False, False, False),
                                   ("whole runs, straight anchoring (c17)", True, False, True),
                                   ("cut lines, straight anchoring", False, False, True),
                                   ("cut lines, straight or T anchoring", False, True, True),
                                   ("whole runs, straight or T anchoring", True, True, True)):
    B.FILL_RUNS, B.ANCHOR_TURNS = runs, turns
    B.anchor_ends = orig_anchor if anchor else (lambda *a, **k: [])
    t = time.time()
    part = orig_gs(*cap["a"], **cap["k"])
    b = part.breakdown()
    o = objective(raw0, part.raw(), w, raw0)
    r = {"part": name, "rule": label, "fi_agg": round(b["fi_agg"], 4), "bulk": round(b["fi_agg2"], 4),
         "objective": round(o["total"], 2), "terms": {k: round(v, 2) for k, v in o.items() if k != "total"},
         "free_ends": b["free_ends"], "strokes": b["strokes"], "time_s": round(b["time_s"], 1),
         "infill_pct": round(b["infill_pct"], 2), "aligned": round(orientation(part)["aligned_p2"], 3),
         "seconds": round(time.time() - t)}
    rows.append(r)
    print(json.dumps(r), flush=True)
B.anchor_ends = orig_anchor
