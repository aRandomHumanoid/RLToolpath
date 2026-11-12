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
    Material, BondModel, LoadSpec

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
    terms = [t for t, wt in zip(TERMS, weights) if wt > 0]
    if "strength" in terms and not loads.has_2d:
        _say("no supports/loads in the load spec: strength term disabled")
        terms.remove("strength")
    if "inter" in terms and not loads.point_loads:
        _say("interlayer term on without point_loads: cells weighted by squared distance from the centroid")
    return PartSpec(printer=pr, loads=loads, terms=tuple(terms))


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


def _build(args, spec):
    from rltp.io.mesh import load_mesh
    from rltp.part import build_part
    t = time.time()
    mesh = load_mesh(args.mesh)
    part = build_part(mesh, spec, baseline=args.fill, rotate=args.rotate)
    L0 = part.layers[0]
    _say(f"sliced {len(part.layers)} layers, grid {L0.H}x{L0.W} at p={L0.frame.p} mm, "
         f"{part.breakdown()['cells']} cells ({time.time() - t:.1f}s)")
    return part


def cmd_baseline(args) -> int:
    from rltp.loop.optimize import History
    w = parse_weights(args.weights)
    spec = _spec(args, w)
    part = _build(args, spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = History(out / "history.jsonl")
    for k, L in enumerate(part.layers):
        hist.snapshot(0, k, L)
    hist.close()
    meta = {"command": "baseline", "mesh": str(args.mesh), "fill": args.fill, "rotate": args.rotate,
            "weights": w.tolist(), "terms": list(spec.terms), "printer": spec.printer.name,
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
    part = _build(args, spec)
    proposer_name = args.proposer or ("policy" if args.policy else "sa")
    proposer = make_proposer(proposer_name, args.policy, args.temp0, args.greedy_n, args.sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = History(out / "history.jsonl", snapshot_every=args.snapshot_every)
    before = part.breakdown()
    cfg = LoopConfig(burst=args.burst, temp0=args.temp0, anneal_passes=args.anneal_passes, eps=args.eps,
                     max_passes=args.max_passes, max_evals=args.max_evals, warmup=args.warmup,
                     snapshot_every=args.snapshot_every, seed=args.seed)
    res = optimize_part(part, proposer, w, cfg, hist, progress=_say)
    hist.close()
    after = part.breakdown()
    meta = {"command": "optimize", "mesh": str(args.mesh), "fill": args.fill, "rotate": args.rotate,
            "weights": w.tolist(), "terms": list(spec.terms), "printer": spec.printer.name,
            "proposer": proposer_name, "policy": args.policy, "loop": res,
            "baseline": before, "final": after}
    _write_outputs(out, part, spec, hist.events, meta)
    _say(f"{proposer_name}: {res['evals']} evals, {res['passes']} passes, stop={res['stop']}, "
         f"{res['seconds']:.1f}s")
    for key in ("fi_agg", "inter_fi_agg", "time_s", "strokes", "travel_mm", "boundary_ends"):
        _say(f"  {key:14s} {before[key]:>10.4g} -> {after[key]:>10.4g}")
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
    be.set_defaults(fn=cmd_bench)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
