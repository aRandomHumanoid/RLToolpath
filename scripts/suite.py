"""Run the test parts through `rltp optimize` settings in parallel and measure how far the
toolpaths adapt to the loads (rltp.eval.adapt), start against final.

usage:
  python scripts/suite.py OUT_DIR [--parts bracket,link,hook,frame,wrench]
      [--arm NAME="--start slicer" ...] [--evals 3000] [--workers 10] [--viewer] [--yaml-loads]
      [--start-state 'RUNS/{part}/ARM/state0.npz'] [-- extra rltp optimize args for every arm]

--start-state: every run starts from another run's saved start layout ({part} filled in)
instead of computing its own (the slicer layout is built, then the layout restored), so arms
compared on one start skip repeating the gradient start.

Every arm runs on every part with configs/printer.yaml, 0.4 mm FEA elements and 10 random
force + 10 random moment cases (seed 0) unless the extra args say otherwise. The cases are
load-agnostic: no loads.yaml, so none is special. --yaml-loads adds the part's
examples/<part>_loads.yaml as the primary case (runs before Nov 15, 2026). Writes
OUT_DIR/<part>/<arm>/result.json (and viewer.html with --viewer), and OUT_DIR/summary.json.
The objective is the README's: weighted % changes against the part's slicer start.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import shlex
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMON = ["--printer", str(ROOT / "configs/printer.yaml"), "--fea-element", "0.4",
          "--load-cases", "10", "--load-moments", "10", "--weights", "0.5,1,1"]
TERMS_PCT = ("strength", "time", "cont", "inter", "cover", "infill")


def objective(raw0: dict, raw1: dict, w, ref: dict) -> dict:
    """Weighted % change per term against raw0 (relative scales from ref, the slicer start)."""
    from rltp.config import TERMS
    from rltp.eval.metrics import Normalizer, full_weights
    n = Normalizer()
    n.set_reference(ref)
    w = full_weights(w)
    pts = {t: 100 * w[i] * (raw1.get(t, 0) - raw0.get(t, 0)) / n.scale(t) for i, t in enumerate(TERMS)}
    return {"total": sum(pts.values()), **{t: round(v, 3) for t, v in pts.items() if w[TERMS.index(t)]}}


def part_argv(part_name, yaml_loads, out):
    """`rltp optimize` mesh and loads arguments for a suite part."""
    loads = ["--loads", str(ROOT / f"examples/{part_name}_loads.yaml")] if yaml_loads else []
    return ["optimize", str(ROOT / f"examples/{part_name}.stl")] + loads + ["--out", str(out)]


def job(spec_):
    part_name, arm, arm_args, extra, evals, out_dir, viewer, yaml_loads, start_state = spec_
    from rltp.cli import _apply_adapt, _build, _spec, _write_outputs, build_parser
    from rltp.config import parse_weights
    from rltp.eval.adapt import layout_change, orientation
    from rltp.export.state import save_state
    from rltp.loop.optimize import History, LoopConfig, optimize_part
    from rltp.search.sa import make_proposer
    out = Path(out_dir) / part_name / arm
    out.mkdir(parents=True, exist_ok=True)
    argv = (part_argv(part_name, yaml_loads, out) + ["--max-evals", str(evals)]
            + COMMON + extra + shlex.split(arm_args))
    args = build_parser().parse_args(argv)
    w = parse_weights(args.weights)
    _apply_adapt(args, w)
    t0 = time.time()
    src = Path(start_state.format(part=part_name)) if start_state else None
    if src is not None:              # another run's start layout: build the slicer's, then restore it
        args.start = "slicer"
    spec = _spec(args, w)
    part = _build(args, spec, w)
    if src is not None:
        from rltp.export.state import load_state
        part.restore([(d["nxt"], d["prv"], d["seam"], d["mask"]) for d in load_state(src)["layers"]])
    tb = time.time() - t0
    start_layers = [L.copy() for L in part.layers]
    save_state(out / "state0.npz", part, {"arm": arm, "when": "start"})
    raw0, b0 = part.raw(), part.breakdown()
    o0 = orientation(part)
    proposer = make_proposer(args.proposer or "sa", None, args.temp0, args.greedy_n, args.sample)
    cfg = LoopConfig(burst=args.burst, temp0=args.temp0, anneal_passes=args.anneal_passes, eps=args.eps,
                     max_passes=args.max_passes, max_evals=args.max_evals, warmup=args.warmup,
                     snapshot_every=args.snapshot_every, seed=args.seed, persist_bans=not args.burst_bans,
                     strength_boost=args.strength_first, stress_bias=args.stress_bias, reheat=args.reheat,
                     guide=args.guide, align=args.align, pack=args.pack, line_bins=args.line_bins)
    hist = History(out / "history.jsonl", snapshot_every=args.snapshot_every, keep=viewer)
    t1 = time.time()
    res = optimize_part(part, proposer, w, cfg, hist)
    hist.close()
    raw1, b1 = part.raw(), part.breakdown()
    save_state(out / "state.npz", part, {"arm": arm, "when": "final"})
    ops: dict = {}
    events = hist.events if viewer else (json.loads(x) for x in open(out / "history.jsonl"))
    for e in events:
        if e.get("type") == "edit":
            a = ops.setdefault(e["op"] + ("*" if e.get("guided") else ""), [0, 0])
            a[0] += 1
            a[1] += bool(e["accepted"])
    r = {"part": part_name, "arm": arm, "args": arm_args, "extra": extra, "weights": list(map(float, w)),
         "walls": spec.printer.walls, "solid_layers": spec.printer.solid_layers, "yaml_loads": yaml_loads,
         "load_weights": list(spec.loads.random_weights) if spec.loads.random_weights else "uniform",
         "start_state": str(src) if src is not None else None,
         "directions": getattr(spec.score, "directions", "classes"),
         "build_s": tb, "run_s": time.time() - t1, "evals": res["evals"], "raw0": raw0, "raw1": raw1,
         "fi0": b0["fi_agg"], "fi1": b1["fi_agg"], "case_fi0": b0.get("case_fi_agg"), "case_fi1": b1.get("case_fi_agg"),
         "time0": b0["time_s"], "time1": b1["time_s"], "strokes0": b0["strokes"], "strokes1": b1["strokes"],
         "orient0": o0, "orient1": orientation(part), "change": layout_change(start_layers, part.layers),
         "ops": ops, "start_info": getattr(part, "start_info", None) and
         {k: v for k, v in part.start_info.items() if k in ("fi_agg", "stress_start_fi", "start", "seconds")}}
    if viewer:
        meta = {"command": "suite", "arm": arm, "args": arm_args, "loop": res, "baseline": b0, "final": b1,
                "load_cases": spec.loads.describe_cases(), "start": spec.printer.start}
        _write_outputs(out, part, spec, hist.events, meta)
    (out / "result.json").write_text(json.dumps(r, indent=1, default=float))
    return r


def summarize(results, out_dir):
    by_part: dict = {}
    for r in results:
        by_part.setdefault(r["part"], {})[r["arm"]] = r
    rows = []
    for pn, arms in by_part.items():
        ref = next((a["raw0"] for a in arms.values() if "--start slicer" in a["args"] or a["arm"] == "A"),
                   next(iter(arms.values()))["raw0"])
        for an, r in arms.items():
            o_start = objective(ref, r["raw0"], r["weights"], ref)
            o_end = objective(ref, r["raw1"], r["weights"], ref)
            cf0, cf1 = np.asarray(r["case_fi0"] or [r["fi0"]]), np.asarray(r["case_fi1"] or [r["fi1"]])
            rows.append({"part": pn, "arm": an, "start_obj": round(o_start["total"], 2),
                         "final_obj": round(o_end["total"], 2), "terms": o_end,
                         "fi_change_pct": round(100 * (r["fi1"] / r["fi0"] - 1), 2),
                         "case_change_pct_mean": round(float(100 * np.mean(cf1 / cf0 - 1)), 2),
                         "case_change_pct_worst": round(float(100 * np.max(cf1 / cf0 - 1)), 2),
                         "dir_changed": round(r["change"]["dir_changed"], 4),
                         "fill_changed": round(r["change"]["fill_changed"], 4),
                         "eff_p2": [round(r["orient0"]["eff_p2"], 4), round(r["orient1"]["eff_p2"], 4)] if r["orient0"] else None,
                         "eff_p8": [round(r["orient0"]["eff_p8"], 4), round(r["orient1"]["eff_p8"], 4)] if r["orient0"] else None,
                         "aligned_p2": [round(r["orient0"]["aligned_p2"], 4), round(r["orient1"]["aligned_p2"], 4)] if r["orient0"] else None,
                         "wall_p8": [round(r["orient0"].get("wall_p8", float("nan")), 4),
                                     round(r["orient1"].get("wall_p8", float("nan")), 4)] if r["orient0"] else None,
                         "time": [round(r["time0"], 1), round(r["time1"], 1)],
                         "strokes": [r["strokes0"], r["strokes1"]],
                         "run_min": round((r["build_s"] + r["run_s"]) / 60, 1)})
    (Path(out_dir) / "summary.json").write_text(json.dumps(rows, indent=1))
    hdr = f"{'part':8s} {'arm':6s} {'start':>7s} {'final':>7s} {'FI%':>6s} {'case%':>6s} {'worst%':>6s} " \
          f"{'dirchg':>6s} {'fillchg':>7s} {'eff2 0->1':>13s} {'eff8 0->1':>13s} {'align2':>13s} {'wall8':>13s} "           f"{'strokes':>9s} {'min':>5s}"
    print(hdr)
    for x in rows:
        e2 = f"{x['eff_p2'][0]:.3f}->{x['eff_p2'][1]:.3f}" if x["eff_p2"] else "-"
        e8 = f"{x['eff_p8'][0]:.3f}->{x['eff_p8'][1]:.3f}" if x["eff_p8"] else "-"
        al = f"{x['aligned_p2'][0]:.3f}->{x['aligned_p2'][1]:.3f}" if x["aligned_p2"] else "-"
        wl = f"{x['wall_p8'][0]:.3f}->{x['wall_p8'][1]:.3f}" if x["wall_p8"] else "-"
        print(f"{x['part']:8s} {x['arm']:6s} {x['start_obj']:7.1f} {x['final_obj']:7.1f} {x['fi_change_pct']:6.1f} "
              f"{x['case_change_pct_mean']:6.1f} {x['case_change_pct_worst']:6.1f} {x['dir_changed']:6.3f} "
              f"{x['fill_changed']:7.3f} {e2:>13s} {e8:>13s} {al:>13s} {wl:>13s} {x['strokes'][0]:>4d}->{x['strokes'][1]:<4d} "
              f"{x['run_min']:5.1f}")
    return rows


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    out_dir = Path(argv[0])
    if argv[1:2] == ["--summarize"]:          # tabulate the runs finished so far
        summarize([json.loads(p.read_text()) for p in sorted(out_dir.glob("*/*/result.json"))], out_dir)
        return
    parts = ["bracket", "link", "hook", "frame", "wrench"]
    arms, evals, workers, viewer, yaml_loads, start_state = [], 3000, 10, False, False, None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--parts":
            parts = argv[i + 1].split(",")
            i += 2
        elif a == "--arm":
            name, _, args = argv[i + 1].partition("=")
            arms.append((name, args))
            i += 2
        elif a == "--evals":
            evals = int(argv[i + 1])
            i += 2
        elif a == "--workers":
            workers = int(argv[i + 1])
            i += 2
        elif a == "--viewer":
            viewer = True
            i += 1
        elif a == "--yaml-loads":
            yaml_loads = True
            i += 1
        elif a == "--start-state":
            start_state = argv[i + 1]
            i += 2
        else:
            raise SystemExit(f"unknown argument {a}")
    arms = arms or [("A", "--start slicer")]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, n, a, extra, evals, str(out_dir), viewer, yaml_loads, start_state) for p in parts for n, a in arms]
    with mp.get_context("fork").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        results = pool.map(job, jobs, chunksize=1)
    summarize(results, out_dir)


if __name__ == "__main__":
    main()
