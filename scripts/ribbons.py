"""Ribbon (surface) infill on a test part: lay a starting family of ribbons, then search a smooth
warp of them with CMA-ES against the usual score (rltp.surfaces; design doc "Surface infill").

The part is built exactly as the c20 four-class runs were (examples/c20_reference.json: path
mode, 10 + 10 random edge load cases, --adapt weights), from its slicer layout, and every
layout is scored against that slicer layout, so the numbers compare with the reference's
`final` (the best edit-by-edit result on these settings).

usage: python scripts/ribbons.py PART OUT_DIR [--angle 0] [--spacing 1.0] [--cross]
           [--rings 0] [--ring-gap 0.4] [--alternate] [--grid 4] [--amp 1.5] [--sigma 0.3]
           [--z-levels 0] [--max-lean 45] [--warm RESULT_JSON]
           [--evals 1500] [--popsize 12] [--workers 6] [--seed 0]
  (from the repo root; WSL/Linux for --workers > 1)

--z-levels N (N >= 2) makes the ribbons 3D: the warp varies with height over N levels, each
layer laid from the ribbons' cut at its own height, no surface leaning more than --max-lean.
--warm starts from a finished vertical run (its best warp becomes the middle level).

Writes OUT_DIR/result.json (scores of the slicer layout, the starting ribbons, the best warp and
the reference), state.npz, viewer.html and layers.png (an infill layer: start and best).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import TERMS, parse_weights  # noqa: E402
from rltp.eval.metrics import Normalizer, full_weights  # noqa: E402
from rltp.surfaces.cmaes import CMAES, ElitistES  # noqa: E402
from rltp.surfaces.layout import apply_layout, is_infill_layer, unshare_caches  # noqa: E402
from rltp.surfaces.ribbons import Warp, Warp3D, line_family, offset_rings  # noqa: E402

REF = json.loads((ROOT / "examples/c20_reference.json").read_text())
BREAKDOWN = ("fi_agg", "time_s", "strokes", "free_ends", "infill_pct", "travel_mm", "retractions")


def objective(raw0: dict, raw1: dict, w, ref: dict) -> dict:
    """Weighted % change per term against raw0, scales from ref (scripts/suite.py's)."""
    n = Normalizer()
    n.set_reference(ref)
    w = full_weights(w)
    pts = {t: 100 * w[i] * (raw1.get(t, 0) - raw0.get(t, 0)) / n.scale(t) for i, t in enumerate(TERMS)}
    return {"total": sum(pts.values()), **{t: round(v, 3) for t, v in pts.items() if w[TERMS.index(t)]}}


def build(name: str):
    ref = REF[name]
    argv = (["optimize", str(ROOT / f"examples/{name}.stl"), "--out", "/tmp/ribbons"]
            + [str(ROOT / a) if a == "configs/printer.yaml" else a for a in ref["argv"]])
    args = build_parser().parse_args(argv)
    w = parse_weights(args.weights)
    _apply_adapt(args, w)
    args.start = "slicer"
    return _build(args, _spec(args, w), w)


G: dict = {}          # what evaluate() reads; set before the worker pool forks


def evaluate(x):
    part = G["part"]
    w3 = G.get("warp3")
    if w3 is not None:
        def rbs(z):
            return w3.apply(x, G["seed"], z)
    else:
        rbs = G["warp"].apply(x, G["seed"]) if G["warp"] is not None else G["seed"]
    stats = apply_layout(part, rbs, alternate=G["alternate"])
    if w3 is not None:                       # one LayStats per layer: keep the totals
        stats = [type(stats[0])(**{f: sum(getattr(s, f) for s in stats) for f in
                                   ("pieces", "loops", "links", "anchored", "nodes")})]
    raw = part.raw()
    obj = objective(G["raw_s"], raw, G["w"], G["raw_s"])
    return obj["total"], obj, raw, [vars(s) for s in stats]


def summary(part) -> dict:
    b = part.breakdown()
    return {k: b.get(k) for k in BREAKDOWN}


def draw(path: Path, panels: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rltp.cover.cover import get_pieces
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (title, L) in zip(axes[0], panels):
        pc = get_pieces(L)
        lock = L.lock.reshape(-1) if L.lock is not None else np.zeros(L.n, bool)
        for cells, loop in zip(pc.cells, pc.is_loop):
            cells = list(cells) + ([cells[0]] if loop else [])
            x, y = L.frame.cell_xy(cells, L.W)
            wall = bool(lock[cells[0]])
            ax.plot(x, y, color="0.6" if wall else rng.uniform(0.1, 0.85, 3), lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(REF))
    ap.add_argument("out")
    ap.add_argument("--angle", type=float, default=0.0, help="starting lines' angle, degrees")
    ap.add_argument("--spacing", type=float, default=1.0, help="starting lines' spacing, mm (1.0 = 40%% infill)")
    ap.add_argument("--cross", action="store_true", help="a second family at angle + 90 (each at 2 x spacing)")
    ap.add_argument("--rings", type=int, default=0, help="fixed rings following every outline, beside the wall")
    ap.add_argument("--ring-gap", type=float, default=0.4, help="ring spacing, mm (0.4: touching, a solid band)")
    ap.add_argument("--alternate", action="store_true", help="odd layers lay free ribbons in reverse order")
    ap.add_argument("--grid", type=float, default=4.0, help="warp grid spacing, mm")
    ap.add_argument("--amp", type=float, default=1.5, help="largest warp displacement, mm")
    ap.add_argument("--z-levels", type=int, default=0,
                    help="3D ribbons: warp levels over the infill height (0: vertical ribbons)")
    ap.add_argument("--max-lean", type=float, default=45.0, help="3D ribbons: steepest lean from vertical, degrees")
    ap.add_argument("--warm", default=None,
                    help="a finished vertical run's result.json: its best warp starts the middle level "
                         "(same part, start family, --grid and --amp)")
    ap.add_argument("--sigma", type=float, default=0.3, help="CMA-ES starting step (warp parameters)")
    ap.add_argument("--optimizer", choices=("cma", "elitist"), default="cma",
                    help="cma: CMA-ES; elitist: (1+lambda)-ES around the best point so far")
    ap.add_argument("--evals", type=int, default=1500, help="search budget (0: score the start only)")
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    part = build(a.part)
    raw_s, sum_s = part.raw(), summary(part)
    w = REF[a.part]["weights"]
    k0 = next(k for k, L in enumerate(part.layers) if is_infill_layer(L))
    poly = part.layers[k0].poly
    bounds = poly.bounds
    bw = part.spec.printer.bead_width
    seed = offset_rings(poly, 1.5 * bw + 0.05, a.ring_gap, a.rings) if a.rings else []
    if a.cross:
        seed += line_family(bounds, a.angle, 2 * a.spacing) + line_family(bounds, a.angle + 90, 2 * a.spacing)
    else:
        seed += line_family(bounds, a.angle, a.spacing)
    warp = Warp(bounds, a.grid, a.amp) if a.evals > 0 or a.warm else None
    zs = [L.z - 0.5 * L.h for L in part.layers if is_infill_layer(L)]
    warp3 = Warp3D(bounds, a.grid, a.amp, min(zs), max(zs), a.z_levels, a.max_lean) if a.z_levels > 1 else None
    G.update(part=part, seed=seed, warp=warp, warp3=warp3, alternate=a.alternate, w=w, raw_s=raw_s)
    x0 = np.zeros(warp3.n if warp3 else (warp.n if warp else 0))
    if a.warm:
        xw = np.asarray(json.loads(Path(a.warm).read_text())["best"]["x"], float)
        if warp is None or xw.size != warp.n:
            raise SystemExit(f"--warm: {xw.size} parameters, but this grid has {warp.n if warp else 0}")
        x0[:xw.size] = xw
    f0, obj0, raw0, st0 = evaluate(x0)
    sum0 = summary(part)
    L0 = part.layers[k0].copy()
    L0._cache = {}
    print(f"[ribbons] {a.part}: {len(seed)} ribbons, {len(x0)} warp parameters"
          f"{f' ({a.z_levels} levels, lean <= {a.max_lean:g} deg)' if warp3 else ''}; start {f0:+.1f} "
          f"(FI_agg {sum0['fi_agg']:.3f}, {sum0['time_s']:.0f} s, {sum0['strokes']} strokes); "
          f"reference final {REF[a.part]['final']['objective']['total']:+.1f}; {time.time() - t0:.0f} s", flush=True)
    best = (f0, x0, obj0)
    hist = [{"evals": 1, "best": f0, "gen_best": f0, "gen_mean": f0, "sigma": a.sigma}]
    if warp is not None:
        es = (ElitistES(x0, a.sigma, a.popsize, a.seed, f0=-f0) if a.optimizer == "elitist"
              else CMAES(x0, a.sigma, a.popsize, a.seed))
        n_ev = 1
        ctx = mp.get_context("fork") if a.workers > 1 and "fork" in mp.get_all_start_methods() else None
        pool = ctx.Pool(a.workers) if ctx else None
        try:
            while n_ev < a.evals:
                X = es.ask()
                res = pool.map(evaluate, list(X), chunksize=1) if pool else [evaluate(x) for x in X]
                fs = np.array([r[0] for r in res])
                es.tell(X, -fs)
                n_ev += len(X)
                i = int(np.argmax(fs))
                if fs[i] > best[0]:
                    best = (float(fs[i]), X[i].copy(), res[i][1])
                hist.append({"evals": n_ev, "best": best[0], "gen_best": float(fs[i]),
                             "gen_mean": float(fs.mean()), "sigma": es.sigma})
                if es.gen % 5 == 0 or n_ev >= a.evals:
                    print(f"[ribbons] gen {es.gen} evals {n_ev}: best {best[0]:+.1f}, this gen {fs[i]:+.1f} "
                          f"(mean {fs.mean():+.1f}), sigma {es.sigma:.3f}, {time.time() - t0:.0f} s", flush=True)
        finally:
            if pool:
                pool.close()
                pool.join()
    fb, objb, rawb, stb = evaluate(best[1])
    unshare_caches(part)
    sumb = summary(part)
    from rltp.export.state import save_state
    from rltp.export.viewer import write_viewer
    save_state(out / "state.npz", part)
    draw(out / "layers.png", [(f"start: {f0:+.1f}", L0), (f"best: {fb:+.1f}", part.layers[k0])])
    write_viewer(out / "viewer.html", part, [], {
        "command": "ribbons", "arm": out.name, "args": " ".join(sys.argv[1:]), "evals": len(hist),
        "directions": "path", "baseline": part.breakdown(), "final": part.breakdown(),
        "load_cases": part.spec.loads.describe_cases(), "start": "ribbons"})
    ref = REF[a.part]
    result = {
        "part": a.part, "settings": vars(a), "ribbons": len(seed), "warp_parameters": len(x0),
        "run_s": round(time.time() - t0, 1),
        "slicer": {"objective": 0.0, **sum_s, "raw": raw_s},
        "start": {"objective": obj0, **sum0, "raw": raw0, "layout": st0},
        "best": {"objective": objb, **sumb, "raw": rawb, "layout": stb, "x": best[1].tolist(),
                 **({"lean_deg": warp3.lean(best[1], seed, zs)} if warp3 else {})},
        "reference": {k: {"objective": ref[k]["objective"], "fi_agg": ref[k]["fi_agg"], "time_s": ref[k]["time_s"],
                          "strokes": ref[k]["strokes"]} for k in ("gradient_start", "final")},
        "history": hist,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[ribbons] {a.part}: best {fb:+.1f} (FI_agg {sumb['fi_agg']:.3f}, {sumb['time_s']:.0f} s, "
          f"{sumb['strokes']} strokes, {sumb['free_ends']} free ends) vs reference final "
          f"{ref['final']['objective']['total']:+.1f} (FI_agg {ref['final']['fi_agg']:.3f}, "
          f"{ref['final']['time_s']:.0f} s, {ref['final']['strokes']} strokes); {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
