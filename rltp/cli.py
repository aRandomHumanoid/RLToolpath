"""rltp command line.

  rltp baseline part.stl --printer printer.yaml --out out/
  rltp optimize part.stl --loads loads.yaml --printer printer.yaml \
                --policy policy.pt --weights 1.0,0.3,0.3 --out out/
  rltp train --config train.yaml --out runs/exp01/
  rltp render out/ --mode playback --layer 3 --format gif
  rltp bench --policy policy.pt --out bench/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rltp.config import TERMS, PartSpec, load_loads, load_printer, parse_weights, _from_dict, \
    Material, BondModel, LoadSpec, ScoreConfig

log = logging.getLogger("rltp")


def _say(msg: str) -> None:
    print(f"[rltp] {msg}", file=sys.stderr, flush=True)


def _spec(args, weights) -> PartSpec:
    pr = load_printer(args.printer)
    loads = LoadSpec()
    if getattr(args, "loads", None):
        loads, extra = load_loads(args.loads)
        if "material" in extra:
            pr.material = _from_dict(Material, {**asdict(pr.material), **extra["material"]})
        if "bond" in extra:
            pr.bond = _from_dict(BondModel, {**asdict(pr.bond), **extra["bond"]})
    for key in ("walls", "solid_layers", "infill", "max_bridge", "infill_rule", "infill_band", "infill_weight",
                "infill_pattern", "fea_element", "cell", "cover_weight", "start"):
        if getattr(args, key, None) is not None:
            setattr(pr, key, type(getattr(pr, key))(getattr(args, key)))
    n_given = len(args.weights.split(",")) if getattr(args, "weights", None) else 0
    if pr.sparse and n_given < 4 and weights[3] == 0:
        weights[3] = 0.3                       # sparse parts: Z bonding counts unless a 4th weight says no
    if pr.kb > 1 and n_given < 6 and weights[5] == 0:
        weights[5] = pr.cover_weight          # fine bead grid: gaps and over-extrusion cost
    if pr.kb == 1:
        weights[5] = 0.0
    penalty = pr.sparse and pr.infill_rule in ("penalty", "both")
    if penalty and weights[4] == 0:
        weights[4] = pr.infill_weight          # the infill term's weight (a 5th --weights entry wins)
    if not penalty:
        weights[4] = 0.0
    if getattr(args, "load_cases", None) is not None:
        loads.random_cases = args.load_cases
    if getattr(args, "load_seed", None) is not None:
        loads.random_seed = args.load_seed
    terms = [t for t, wt in zip(TERMS, weights) if wt > 0]
    if "strength" in terms and not loads.has_2d:
        _say("no supports/loads in the load spec: strength term disabled")
        terms.remove("strength")
    if "inter" in terms and pr.sparse and not loads.point_loads:
        _say("interlayer term on: bonded area under a generic pull and bending (no point_loads)")
    elif "inter" in terms and not loads.point_loads:
        _say("interlayer term on without point_loads: cells weighted by squared distance from the centroid")
    return PartSpec(printer=pr, loads=loads, terms=tuple(terms),
                    score=ScoreConfig(norm=getattr(args, "norm", "relative"),
                                      stack=getattr(args, "stack", None) or "auto",
                                      fea_refine=_refine(pr),
                                      solver=getattr(args, "solver", None) or "fast"))


def _refine(pr) -> int:
    """FEA elements per cell side from printer.fea_element (mm; 0 = one per cell)."""
    if pr.fea_element <= 0:
        return 1
    r = pr.pitch / pr.fea_element
    if abs(r - round(r)) > 1e-6:
        _say(f"fea_element {pr.fea_element} mm doesn't divide the {pr.pitch} mm cells; "
             f"using {pr.pitch / max(round(r), 1):.3f} mm")
    return max(int(round(r)), 1)


def _write_outputs(out: Path, part, spec, events, meta: dict) -> dict:
    from rltp.export.gcode import write_gcode
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    gs = write_gcode(out / "part.gcode", part, spec.printer,
                     header=f"rltp {meta.get('command')} {meta.get('mesh')}")
    meta["gcode"] = asdict(gs)
    try:
        from rltp.export.state import save_state
        save_state(out / "state.npz", part)
    except ImportError as e:  # pragma: no cover
        _say(f"state.npz skipped: {e}")
    try:
        from rltp.export.viewer import write_viewer
        vmeta = dict(meta)
        if "loop" in vmeta:  # the score curve is already charted from the history
            vmeta["loop"] = {k: v for k, v in vmeta["loop"].items() if k != "curve"}
        write_viewer(out / "viewer.html", part, events, vmeta)
    except ImportError as e:  # pragma: no cover
        _say(f"viewer skipped: {e}")
    (out / "metrics.json").write_text(json.dumps(meta, indent=1, default=_json_default))
    _say(f"wrote {out}/part.gcode, viewer.html, history.jsonl, metrics.json ({time.time() - t:.1f}s)")
    return meta


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _build(args, spec, w=None):
    from rltp.io.mesh import load_mesh
    from rltp.part import build_part
    from rltp.search.fields import FieldOptions
    t = time.time()
    mesh = load_mesh(args.mesh)
    fields = FieldOptions(starts=args.field_starts, workers=args.workers,
                          continuation=not args.no_field_continuation,
                          keep=max(1, min(getattr(args, "chains", 1), getattr(args, "chain_starts", 1),
                                          args.field_starts)),
                          weights=None if w is None else tuple(float(x) for x in w),
                          end_scale=args.field_end_cost)
    part = build_part(mesh, spec, baseline=args.fill, rotate=args.rotate, fields=fields)
    L0 = part.layers[0]
    b = part.breakdown()
    _say(f"sliced {len(part.layers)} layers, grid {L0.H}x{L0.W} at p={L0.frame.p} mm, "
         f"{b['cells']} cells ({time.time() - t:.1f}s)")
    si = getattr(part, "start_info", None)
    if si:
        _say(f"gradient start: FI_agg {si['fi_agg']:.4f} against {si['stress_start_fi']:.4f} for the stress "
             f"start; best of {len(si['starts'])} starts by {si['ranked_by']}: {si['start']} (field model "
             f"{si['field_fi_start']:.4f} -> {si['field_fi_end']:.4f}, rounded {si['field_fi_rounded']:.4f}); "
             f"starts within {100 * si['rounded_spread']:.1f}% after rounding; {si['seconds']:.0f}s")
    if part.sparse:
        pr = spec.printer
        _say(f"sparse: {pr.walls} walls, {pr.solid_layers} solid layers, bridges <= {pr.max_bridge} mm; "
             f"{b['cells']} of {b['part_cells']} cells filled, infill {b['infill_pct']:.1f}% of the "
             f"optional cells (target {100 * pr.infill:.0f}%, rule {pr.infill_rule}), "
             f"{b['unsupported']} unsupported")
    return part


def cmd_baseline(args) -> int:
    from rltp.loop.optimize import History
    w = parse_weights(args.weights)
    spec = _spec(args, w)
    part = _build(args, spec, w)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = History(out / "history.jsonl")
    for k, L in enumerate(part.layers):
        hist.snapshot(0, k, L)
    hist.close()
    meta = {"command": "baseline", "mesh": str(args.mesh), "fill": args.fill, "rotate": args.rotate,
            "weights": w.tolist(), "terms": list(spec.terms), "printer": spec.printer.name,
            "load_cases": spec.loads.describe_cases(), "start": spec.printer.start,
            "start_info": getattr(part, "start_info", None),
            "baseline": part.breakdown(), "final": part.breakdown()}
    _write_outputs(out, part, spec, hist.events, meta)
    b = meta["final"]
    _say(f"time {b['time_s']:.1f}s  strokes {b['strokes']}  FI_agg {b['fi_agg']:.4f}")
    return 0


def cmd_optimize(args) -> int:
    from rltp.loop.optimize import History, LoopConfig, optimize_part
    from rltp.search.sa import make_proposer
    w = parse_weights(args.weights)
    if args.interlayer and w[3] == 0:
        w[3] = 0.3
    spec = _spec(args, w)
    part = _build(args, spec, w)
    proposer_name = args.proposer or ("policy" if args.policy else "sa")
    if proposer_name.startswith("policy") and spec.printer.kb > 1:
        raise SystemExit("[rltp] the trained policy works on one cell per bead; use --proposer sa, "
                         "greedy or random on a fine bead grid (or --cell 0)")
    proposer = make_proposer(proposer_name, args.policy, args.temp0, args.greedy_n, args.sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = part.breakdown()
    cfg = LoopConfig(burst=args.burst, temp0=args.temp0, anneal_passes=args.anneal_passes, eps=args.eps,
                     max_passes=args.max_passes, max_evals=args.max_evals, warmup=args.warmup,
                     snapshot_every=args.snapshot_every, seed=args.seed,
                     persist_bans=not args.burst_bans, strength_boost=args.strength_first,
                     stress_bias=args.stress_bias, reheat=args.reheat)
    chains = None
    if args.chains > 1:
        from rltp.search.chains import load_events, run_chains
        cr = run_chains(part, proposer, w, cfg, args.chains, out, workers=args.workers,
                        starts=getattr(part, "start_snaps", None), start_names=getattr(part, "start_names", None),
                        progress=_say)
        chains = cr["chains"]
        win = next(c for c in chains if c["chain"] == cr["best"])
        res = dict(win["loop"], chain=cr["best"], chains=len(chains))
        events = load_events(out / "history.jsonl")
        for c in sorted(chains, key=lambda c: -c["total"]):
            _say(f"  chain {c['chain']} (seed {c['seed']}, start {c['start'] or 'as built'}): total "
                 f"{c['start_total']:+.4f} -> {c['total']:+.4f}, FI_agg {c['final']['fi_agg']:.4f}, "
                 f"strokes {c['final']['strokes']}, reheats {c['loop'].get('reheats', 0)}")
    else:
        hist = History(out / "history.jsonl", snapshot_every=args.snapshot_every)
        res = optimize_part(part, proposer, w, cfg, hist, progress=_say)
        hist.close()
        events = hist.events
    after = part.breakdown()
    meta = {"command": "optimize", "mesh": str(args.mesh), "fill": args.fill, "rotate": args.rotate,
            "weights": w.tolist(), "terms": list(spec.terms), "printer": spec.printer.name,
            "proposer": proposer_name, "policy": args.policy, "loop": res,
            "load_cases": spec.loads.describe_cases(), "start": spec.printer.start,
            "start_info": getattr(part, "start_info", None),
            "baseline": before, "final": after}
    if chains is not None:
        meta["chains"] = chains
    _write_outputs(out, part, spec, events, meta)
    _say(f"{proposer_name}: {res['evals']} evals, {res['passes']} passes, stop={res['stop']}, "
         f"{res['seconds']:.1f}s" + (f" (best of {len(chains)} chains: chain {res['chain']})" if chains else ""))
    for key in ("fi_agg", "inter_fi_agg", "time_s", "strokes", "travel_mm", "boundary_ends", "cells",
                "infill_pct"):
        if key in after:
            _say(f"  {key:14s} {before[key]:>10.4g} -> {after[key]:>10.4g}")
    if "case_fi_agg" in after:
        pct = [100 * (a / b - 1) if b > 0 else 0.0 for a, b in zip(after["case_fi_agg"], before["case_fi_agg"])]
        _say("  FI_agg change per load case (0 = primary): " +
             ", ".join(f"{i}: {x:+.1f}%" for i, x in enumerate(pct)))
    return 0


def cmd_train(args) -> int:
    from rltp.rl.train import train
    train(args.config, args.out, progress=_say)
    return 0


def cmd_render(args) -> int:
    from rltp.export.render import render
    files = render(args.out_dir, mode=args.mode, layer=args.layer, frames=args.frames, fps=args.fps,
                   fmt=args.format, color=args.color, dest=args.dest)
    _say(f"wrote {len(files)} file(s)" + (f", e.g. {files[0]}" if files else ""))
    return 0


def cmd_bench(args) -> int:
    from rltp.bench import run_bench
    run_bench(args, progress=_say)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rltp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def part_args(p, need_loads=False):
        p.add_argument("mesh", help="STL, OBJ or 3MF")
        p.add_argument("--printer", required=True, help="printer.yaml")
        p.add_argument("--loads", required=need_loads, help="loads.yaml")
        p.add_argument("--out", required=True)
        p.add_argument("--weights", default=None,
                       help="strength,time,continuity[,interlayer] (default 1.0,0.3,0.3)")
        p.add_argument("--fill", default="zigzag", choices=["zigzag", "zigzag90", "concentric"])
        p.add_argument("--rotate", action="store_true",
                       help="v0.5: rotate each layer's grid to its dominant principal stress")
        p.add_argument("--norm", default="relative", choices=["relative", "std"],
                       help="term scales: fraction of the part's baseline (default), or the doc's "
                            "original per-edit std")
        p.add_argument("--walls", type=int, default=None,
                       help="wall rings along every outline, printed as fixed loops (printer.yaml: walls)")
        p.add_argument("--solid-layers", type=int, default=None,
                       help="solid layers at every top and bottom surface (printer.yaml: solid_layers)")
        p.add_argument("--infill", type=float, default=None,
                       help="starting infill density of the optional cells; the optimizer then adds or "
                            "removes material (printer.yaml: infill). --walls 0 --solid-layers 0 "
                            "--infill 1 fills every cell, the original mode")
        p.add_argument("--infill-rule", default=None, choices=["both", "band", "penalty", "free"],
                       help="how the optimizer holds the infill target: penalty (a score term), band "
                            "(stay within +-infill_band), both (the penalty inside a hard band; "
                            "printer.yaml default), free (the target only sets the start)")
        p.add_argument("--infill-band", type=float, default=None,
                       help="band rule: allowed deviation of the infill fraction (printer.yaml: infill_band)")
        p.add_argument("--infill-weight", type=float, default=None,
                       help="penalty rule: cost of 1 percentage point off target, as a 1%% change "
                            "of a term with weight 1 (printer.yaml: infill_weight)")
        p.add_argument("--infill-pattern", default=None, choices=["alternate", "stacked"],
                       help="starting infill lines: alternate 0/90 by layer, or stacked in one direction "
                            "so adjacent layers overlap (printer.yaml: infill_pattern)")
        p.add_argument("--max-bridge", type=float, default=None,
                       help="mm, longest span a filled cell may bridge over the layer below "
                            "(printer.yaml: max_bridge)")
        p.add_argument("--stack", default=None, choices=["auto", "split", "laminate"],
                       help="in-plane model of the layer stack: split (each layer carries h/height "
                            "of the load), laminate (bonded layers share the strain), auto (default: "
                            "laminate with walls/infill, split otherwise)")
        p.add_argument("--start", default=None, choices=["slicer", "stress", "stress4", "gradient", "gradient4"],
                       help="starting layout: slicer (fixed directions), stress (beads along each "
                            "area's main stress direction, 0/90), stress4 (fine grid: 45/135 too), "
                            "gradient (bead directions and infill density optimised for in-plane "
                            "strength as fields, then laid out 0/90) or gradient4 (fine grid: 45/135 too)")
        p.add_argument("--solver", default=None, choices=["fast", "legacy"],
                       help="FEA linear solver: fast (default; symmetric LU, CG to 1e-7) or legacy "
                            "(the original general LU, CG to 1e-9; reproduces earlier runs bit for bit)")
        p.add_argument("--cell", type=float, default=None, metavar="MM",
                       help="grid cell size; below the bead width, beads sit anywhere on the finer grid "
                            "(printer.yaml: cell; 0 = one cell per bead)")
        p.add_argument("--fea-element", type=float, default=None, metavar="MM",
                       help="strength-analysis element size in mm, dividing the cell size "
                            "(printer.yaml: fea_element; 0 = one element per cell). Toolpaths are unchanged")
        p.add_argument("--load-cases", type=int, default=None, metavar="N",
                       help="add N random load cases (overrides random_cases in loads.yaml); "
                            "strength is then scored on every case against its own baseline")
        p.add_argument("--load-seed", type=int, default=None,
                       help="seed for the random load cases (default 0, or random_seed in loads.yaml)")
        p.add_argument("--field-starts", type=int, default=8, metavar="N",
                       help="gradient start: optimise the fields from N starting direction fields (the stress "
                            "map, turned 90 degrees, uniform 0/90/45/135, then random) and keep the best by "
                            "the whole score (default 8)")
        p.add_argument("--no-field-continuation", action="store_true",
                       help="gradient start: optimise at the final filter and p-norm from the first step, "
                            "without starting wide and smooth")
        p.add_argument("--field-end-cost", type=float, default=1.0, metavar="X",
                       help="gradient start: X times the fitted cost of bead-line ends (density changes and "
                            "direction-area boundaries, priced in strokes and print time with the run's "
                            "weights) traded against strength; 0 ignores them (default 1)")
        p.add_argument("--workers", type=int, default=0,
                       help="processes for the field starts and annealing chains (default: one per job, up to "
                            "the CPU count; set OMP_NUM_THREADS=1)")

    b = sub.add_parser("baseline", help="slice + baseline covers, no optimisation")
    part_args(b)
    b.set_defaults(fn=cmd_baseline)

    o = sub.add_parser("optimize", help="run the part loop")
    part_args(o, need_loads=False)
    o.add_argument("--policy", help="policy.pt from rltp train")
    o.add_argument("--proposer", choices=["policy", "policy-sample", "sa", "greedy", "random"])
    o.add_argument("--interlayer", action="store_true", help="v1.5 term (weight 0.3 unless given)")
    o.add_argument("--max-evals", type=int, default=20000)
    o.add_argument("--max-passes", type=int, default=50)
    o.add_argument("--burst", type=int, default=32)
    o.add_argument("--temp0", type=float, default=None)
    o.add_argument("--anneal-passes", type=int, default=4)
    o.add_argument("--eps", type=float, default=1e-3)
    o.add_argument("--warmup", type=int, default=128)
    o.add_argument("--greedy-n", type=int, default=16)
    o.add_argument("--sample", action="store_true", help="sample from the policy instead of argmax")
    o.add_argument("--snapshot-every", type=int, default=50)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--strength-first", type=float, default=1.0, metavar="X",
                   help="multiply the strength weight by X at the start, easing back to 1 as "
                        "annealing ends, so directions settle before continuity takes over")
    o.add_argument("--stress-bias", type=float, default=0.0, metavar="F",
                   help="share of proposals placed in proportion to local stress (0..1)")
    o.add_argument("--burst-bans", action="store_true",
                   help="forget rejected edits after each burst (the original loop) instead of "
                        "keeping them banned until their layer changes")
    o.add_argument("--chains", type=int, default=1, metavar="N",
                   help="run N independent annealing chains (seeds seed..seed+N-1; with a gradient start, "
                        "from the best field starts in turn) and keep the best final layout")
    o.add_argument("--chain-starts", type=int, default=1, metavar="K",
                   help="gradient start: chains start from the K best field starts in turn (default 1: on "
                        "the bracket, chains from the second-best start ended 5-10 points behind)")
    o.add_argument("--reheat", type=float, default=0.0, metavar="F",
                   help="once greedy passes stall (two in a row gaining under 0.005), anneal again from "
                        "F x the starting temperature (0 = off)")
    o.set_defaults(fn=cmd_optimize)

    t = sub.add_parser("train", help="curriculum: imitation + PPO")
    t.add_argument("--config", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("render", help="PNG / GIF / MP4 frames of a run")
    r.add_argument("out_dir")
    r.add_argument("--mode", default="layer", choices=["layer", "playback", "history"])
    r.add_argument("--layer", type=int, default=None)
    r.add_argument("--frames", type=int, default=60)
    r.add_argument("--fps", type=int, default=20)
    r.add_argument("--format", default="png", choices=["png", "gif", "mp4"])
    r.add_argument("--color", default="direction", choices=["direction", "order", "fi", "stroke"])
    r.add_argument("--dest", default=None)
    r.set_defaults(fn=cmd_render)

    be = sub.add_parser("bench", help="score vs evaluator calls on held-out shapes")
    be.add_argument("--out", required=True)
    be.add_argument("--policy", default=None)
    be.add_argument("--proposers", default="policy,policy-sample,sa,greedy,random")
    be.add_argument("--families", default="ribs")
    be.add_argument("--shapes", type=int, default=8)
    be.add_argument("--grid", type=int, default=32)
    be.add_argument("--terms", default="strength")
    be.add_argument("--weights", default=None)
    be.add_argument("--budget", type=int, default=1024)
    be.add_argument("--seeds", type=int, default=1)
    be.add_argument("--workers", type=int, default=4)
    be.add_argument("--norm", default="relative", choices=["relative", "std"])
    be.add_argument("--burst-bans", action="store_true", help="the original loop's per-burst bans")
    be.set_defaults(fn=cmd_bench)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
