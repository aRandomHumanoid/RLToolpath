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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rltp.cli import _apply_adapt, _build, _spec, build_parser  # noqa: E402
from rltp.config import TERMS, parse_weights  # noqa: E402
from rltp.eval.metrics import Normalizer, full_weights  # noqa: E402
from rltp.surfaces.cmaes import CMAES, ElitistES, LocalSearch  # noqa: E402
from rltp.surfaces.layout import apply_layout, is_infill_layer, unshare_caches  # noqa: E402
from rltp.surfaces.ribbons import (Warp, Warp3D, density_family, line_family, offset_rings,  # noqa: E402
                                   region_inside)

REF = json.loads((ROOT / "examples/c20_reference.json").read_text())
BREAKDOWN = ("fi_agg", "time_s", "strokes", "free_ends", "infill_pct", "travel_mm", "retractions")


def objective(raw0: dict, raw1: dict, w, ref: dict) -> dict:
    """Weighted % change per term against raw0, scales from ref (scripts/suite.py's)."""
    n = Normalizer()
    n.set_reference(ref)
    w = full_weights(w)
    pts = {t: 100 * w[i] * (raw1.get(t, 0) - raw0.get(t, 0)) / n.scale(t) for i, t in enumerate(TERMS)}
    return {"total": sum(pts.values()), **{t: round(v, 3) for t, v in pts.items() if w[TERMS.index(t)]}}


def build(name: str, extra=()):
    """The part, built as the reference runs were, from its slicer layout; `extra` adds rltp
    optimize arguments after the reference's own (e.g. --load-seed 1 for held-out cases)."""
    ref = REF[name]
    argv = (["optimize", str(ROOT / f"examples/{name}.stl"), "--out", "/tmp/ribbons"]
            + [str(ROOT / a) if a == "configs/printer.yaml" else a for a in ref["argv"]] + list(extra))
    args = build_parser().parse_args(argv)
    w = parse_weights(args.weights)
    _apply_adapt(args, w)
    args.start = "slicer"
    return _build(args, _spec(args, w), w)


def stream_start(part, k0: int, spacing: float, smooth: float):
    """Streamlines of the load-agnostic stress consensus (one isotropic solve over the part's
    load cases, as the stress start uses), smoothed over `smooth` mm, inside layer k0's infill
    region."""
    from scipy import ndimage

    from rltp.eval.stressmap import stress_field
    from rltp.surfaces.layout import infill_region
    from rltp.surfaces.ribbons import streamline_family
    L = part.layers[k0]
    union = np.zeros_like(L.part)
    for M in part.layers:
        union |= M.part_mask
    f = stress_field(union, L.material_frame, part.spec)
    if f is None:
        raise SystemExit("--start stream needs load cases")
    s = smooth / L.material_frame.p
    w = union.astype(float)
    num_c = ndimage.gaussian_filter(np.where(union, np.cos(2 * f["theta"]), 0.0), s)
    num_s = ndimage.gaussian_filter(np.where(union, np.sin(2 * f["theta"]), 0.0), s)
    den = np.maximum(ndimage.gaussian_filter(w, s), 1e-9)
    th = 0.5 * np.arctan2(num_s / den, num_c / den)
    return streamline_family(th, L.material_frame, infill_region(L), L.frame, spacing)


def density_start(part, k0: int, target: float, gamma: float, hi: float | None = None):
    """The density field for --density-gamma: the load-agnostic stress magnitude (one isotropic
    solve over the part's load cases) on the material grid, normalised over layer k0's optional
    cells (rltp.surfaces.ribbons.stress_density), at most `hi` (lines >= bead / hi apart)."""
    from rltp.eval.stressmap import stress_field
    from rltp.surfaces.ribbons import stress_density
    L = part.layers[k0]
    if "mag" not in G:
        union = np.zeros_like(L.part)
        for M in part.layers:
            union |= M.part_mask
        f = stress_field(union, L.material_frame, part.spec)
        if f is None:
            raise SystemExit("--density-gamma needs load cases")
        G["mag"] = f["mag"]
    return stress_density(G["mag"], L.material_frame, L.part & ~L.req, target, gamma, hi=hi or G.get("density_hi", 0.8))


def field_density(part, k0: int, target: float, cache: Path, lo: float = 0.15, hi: float | None = None):
    """The density field for --density-from fields: the gradient layout step's optimised infill
    density (rltp.search.fields.optimize_fields: one start, the stress map's directions, with
    continuation), weighted like the reference runs (strength, stiffness, bulk at the run's weights
    over their term scales; no line-end cost: ribbons have no free ends), cached in `cache`.
    Returns rho(x, y) scaled by `target / 0.4`-style calibration through G["field_scale"]."""
    from rltp.search.fields import optimize_fields
    from rltp.eval.stressmap import stress_field
    L = part.layers[k0]
    if "field_rho" not in G:
        if cache.exists():
            z = np.load(cache)
            G["field_rho"] = z["rho"]
        else:
            union = np.zeros_like(L.part)
            for M in part.layers:
                union |= M.part_mask
            th = stress_field(union, L.material_frame, part.spec)["theta"]
            w = REF[G["part_name"]]["weights"]
            sc = part.norm.scale
            res = optimize_fields(part, th, 100.0 * part.spec.printer.infill / 100.0,
                                  1.0 / max(L.aux["max_spacing"], 1), sigma_mm=0.75, iters=60, continuation=True,
                                  w_strength=w[0] / sc("strength"), w_stiff=w[6] / sc("stiff"),
                                  w_bulk=w[7] / sc("bulk"), end_cost=0.0)
            rho = np.where(res.rho_cells, res.rho, 100.0 * part.spec.printer.infill / 100.0)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, rho=rho, theta=res.theta, fi_start=res.fi_start, fi_end=res.fi_end)
            G["field_rho"] = rho
    lam = G.get("density_mix", 1.0)          # blend toward uniform: (1 - lam) x target + lam x field
    grid = (1.0 - lam) * float(part.spec.printer.infill) + lam * G["field_rho"]
    s = target / float(part.spec.printer.infill)
    fr = L.material_frame

    def rho(x, y):
        u, v = fr.to_local(x, y)
        cc = [np.asarray(v, float) / fr.p - 0.5, np.asarray(u, float) / fr.p - 0.5]
        return np.clip(s * ndimage.map_coordinates(grid, cc, order=1, mode="nearest"), lo, hi or G.get("density_hi", 0.8))
    return rho


def regrid(xw: np.ndarray, old: dict, warp: Warp, bounds) -> np.ndarray:
    """A warm start's warp parameters on another grid: the old warp's displacement sampled at the
    new grid's nodes and inverted through the new bound (amp x tanh), so a coarse search can seed
    a finer one; parameters after the warp (spacing, skins) are carried over."""
    ow = Warp(bounds, old["grid"], old["amp"])
    P = ow.amp * np.tanh(np.asarray(xw[:ow.n], float).reshape(2, ow.ny, ow.nx))
    gy, gx = np.mgrid[0:warp.ny, 0:warp.nx]
    X = warp.x0 + gx * warp.g
    Y = warp.y0 + gy * warp.g
    cc = [(Y - ow.y0) / ow.g, (X - ow.x0) / ow.g]
    D = np.stack([ndimage.map_coordinates(P[c], cc, order=3, mode="nearest") for c in range(2)])
    xn = np.arctanh(np.clip(D / warp.amp, -0.995, 0.995)).reshape(-1)
    return np.r_[xn, np.asarray(xw[ow.n:], float)]


def param_blocks(n: int, warp, warp3, split: bool) -> list:
    """--optimizer local: the parameter blocks a move perturbs together: each warp grid node's
    two components (per level for a 3D warp, per family with --warp-per-family), then every
    remaining parameter (spacing, skin rings, skin directions) on its own."""
    blocks = []
    if warp3 is not None:
        m = warp3.base.ny * warp3.base.nx
        for lev in range(warp3.levels):
            off = lev * 2 * m
            blocks += [np.array([off + j, off + m + j]) for j in range(m)]
        used = warp3.n
    elif warp is not None:
        m = warp.ny * warp.nx
        for fam in range(2 if split else 1):
            off = fam * 2 * m
            blocks += [np.array([off + j, off + m + j]) for j in range(m)]
        used = warp.n * (2 if split else 1)
    else:
        used = 0
    blocks += [np.array([i]) for i in range(used, n)]
    return blocks


def skin_layers(part) -> list[int]:
    """Fully solid sparse layers (top and bottom skins), bottom to top."""
    return [k for k, L in enumerate(part.layers)
            if L.part is not None and L.kb > 1 and not (L.part & ~L.req).any()]


def relay_skins(part, mode: str, rings: int = 0) -> list:
    """Re-lay the solid skin layers: `mode` "slicer" keeps them; a comma list of classes
    (0/45/90/135, one per skin layer bottom to top, the last repeated) lays each layer's skin
    one way; "stress4" lays areas of 0/45/90/135 following the stress consensus (the stress4
    start's map: ~2 mm smoothing, diagonal areas of at least 3 x 3 mm). Walls are rebuilt
    identically. Returns the classes used per skin layer (or the mode)."""
    if mode == "slicer" and not rings:
        return ["slicer"]
    ks = skin_layers(part)
    if mode == "slicer":
        mode = ",".join(str(c) for c in slicer_skin_classes(part))
    if mode == "stress4":
        from rltp.eval.stressmap import orientation4, stress_field
        L = part.layers[ks[0]]
        union = np.zeros_like(L.part)
        for M in part.layers:
            union |= M.part_mask
        f = stress_field(union, L.material_frame, part.spec)
        p = L.material_frame.p
        cmap = orientation4(f, max(int(round(2.0 / p)), 1), int(round((3.0 / p) ** 2)))
        maps = [cmap] * len(ks)
    else:
        cls = [int(c) for c in mode.split(",")]
        maps = [np.full(part.layers[k].part.shape, cls[min(i, len(cls) - 1)]) for i, k in enumerate(ks)]
    lay_skins(part, ks, maps, [rings] * len(ks))
    part.refresh_support()
    part.full_recheck()
    return [mode]


def lay_skins(part, ks, maps, rings=None) -> None:
    """Re-lay skin layers `ks` with orientation maps `maps` (a class per cell, or an int for the
    whole layer) and, per layer, `rings` extra concentric loops following every outline inside
    the wall (0: walls rebuilt identically, the raster filling the rest). Not re-scored."""
    from rltp.cover.beads import fine_layout
    pr = part.spec.printer
    rings = rings or [0] * len(ks)
    for k, cm, r in zip(ks, maps, rings):
        L = part.layers[k]
        if np.isscalar(cm):
            cm = np.full(L.part.shape, int(cm))
        fine_layout(L, L.part.copy(), L.part.copy(), pr.walls + int(r), pr.infill, 0, 0, L.aux["max_spacing"],
                    L.kb, orient=cm, lines=pr.infill_lines)
        L._cache = {}


def slicer_skin_classes(part) -> list[int]:
    """The slicer's skin direction per skin layer: 0 on even layers, 90 on odd (build_part's
    `(base + 90 k) % 180` with the zigzag base 0)."""
    return [0 if k % 2 == 0 else 90 for k in skin_layers(part)]


G: dict = {}          # what evaluate() reads; set before the worker pool forks


def spacing_of(v: float) -> float:
    """--spacing-search: the line spacing for parameter v, within +-20% of --spacing."""
    return G["spacing0"] * float(np.exp(0.2 * np.tanh(v)))


def evaluate(x):
    part = G["part"]
    ks = G.get("skin_ks")
    if ks:                                   # the last len(ks) parameters: each skin layer 0 (< 0) or 90
        x = np.asarray(x, float)
        cls = tuple(0 if v < 0 else 90 for v in x[-len(ks):])
        x = x[:-len(ks)]
        rings = (0,) * len(ks)
        if G.get("skin_rings"):              # before them: each skin layer's edge rings, 0..3
            rings = tuple(int(np.clip(np.round(v), 0, 3)) for v in x[-len(ks):])
            x = x[:-len(ks)]
        if (cls, rings) != G.get("skin_now"):
            lay_skins(part, ks, cls, rings)
            G["skin_now"] = (cls, rings)
    if G.get("spacing_search"):              # then the one before them: the line spacing
        x = np.asarray(x, float)
        sp = round(spacing_of(x[-1]), 3)
        x = x[:-1]
        if sp != G.get("spacing_now"):
            fixed = [rb for rb in G["seed"] if rb.fixed]
            G["seed"] = fixed + line_family(G["bounds"], G["angle"], sp)
            if G.get("seed_b") is not None:
                G["seed_b"] = fixed + line_family(G["bounds"], G["alt_angle"], sp)
            G["spacing_now"] = sp
    w3 = G.get("warp3")
    sb = G.get("seed_b")                     # a second family on odd infill layers (--alt-angle)
    fams = [G["seed"]] + ([sb] if sb is not None else []) + list(G.get("seed_more") or [])
    if w3 is not None:
        z0, h = G["zs"][0], G["dz"]

        def rbs(z):
            return w3.apply(x, fams[int(round((z - z0) / h)) % len(fams)], z)
    elif len(fams) > 2:                      # --families K >= 3: rotate through K families
        rbs = tuple(G["warp"].apply(x, f) if G["warp"] is not None else f for f in fams)
    elif sb is not None:
        if G["warp"] is None:
            rbs = (G["seed"], sb)
        elif G.get("split"):                 # --warp-per-family: each family its own warp
            n = G["warp"].n
            rbs = (G["warp"].apply(x[:n], G["seed"]), G["warp"].apply(x[n:2 * n], sb))
        else:
            rbs = (G["warp"].apply(x, G["seed"]), G["warp"].apply(x, sb))
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
    ap.add_argument("--start", choices=("lines", "stream"), default="lines",
                    help="lines: parallel straight ribbons at --angle; stream: evenly spaced streamlines "
                         "of the load-agnostic stress consensus direction")
    ap.add_argument("--field-smooth", type=float, default=1.0, help="stream: smoothing of the direction field, mm")
    ap.add_argument("--skin", default="slicer",
                    help="solid skin layers: slicer (keep), a comma list of classes 0/45/90/135 per skin "
                         "layer bottom to top, or stress4 (areas following the stress consensus)")
    ap.add_argument("--skin-search", action="store_true",
                    help="search each skin layer's direction (0 or 90 degrees) with the warp, starting "
                         "from the slicer's skins")
    ap.add_argument("--skin-from", default=None,
                    help="take the skin layers from a saved layout of the same part (state.npz), e.g. a "
                         "gradient start's (screening: what better skins are worth)")
    ap.add_argument("--skin-ring-count", type=int, default=0,
                    help="fixed edge-following rings in every skin layer (without --skin-rings)")
    ap.add_argument("--skin-rings", action="store_true",
                    help="with --skin-search: also search each skin layer's number of edge-following "
                         "rings inside the wall (0-3), starting at 0")
    ap.add_argument("--angle", type=float, default=0.0, help="starting lines' angle, degrees")
    ap.add_argument("--density-from", choices=("stress", "fields"), default="stress",
                    help="the density field: stress (the load-agnostic stress magnitude, --density-gamma) or "
                         "fields (the gradient layout step's optimised density, computed once per part and cached)")
    ap.add_argument("--density-hi", type=float, default=0.8,
                    help="density cap (lines at least bead / cap apart across the family; tilted lines come "
                         "closer: 0.6 keeps them a clearance apart up to 45 degrees of tilt)")
    ap.add_argument("--density-norm", choices=("none", "station", "segment"), default="none",
                    help="rescale the density per cross-section of each family so it holds as many lines "
                         "as uniform ones (station: the whole cross-section; segment: each stretch inside "
                         "the part): lines gather across their family, not along it, and don't drift")
    ap.add_argument("--density-smooth", type=float, default=0.0,
                    help="smooth the density along each family's lines first (Gaussian sigma, mm): "
                         "bounds how far lines tilt to follow it, keeps its contrast across them")
    ap.add_argument("--density-on", choices=("both", "first", "second"), default="both",
                    help="with --alt-angle: which family the density spaces (first: --angle's); the "
                         "other stays straight at the same mean spacing")
    ap.add_argument("--density-cases", type=int, default=0,
                    help="fit the density field (stress or fields) to N random force + N moment cases "
                         "instead of the part's own 10 + 10 (the load distribution, not one sample of "
                         "it; the search still scores the part's own cases); fields cached per N")
    ap.add_argument("--density-mix", type=float, default=1.0,
                    help="fields: blend the optimised density toward uniform, (1 - mix) x target + mix x field")
    ap.add_argument("--density-gamma", type=float, default=None,
                    help="lines start: space each family by a density field, target x (load-agnostic stress "
                         "magnitude / mean)^gamma (0: uniform), clipped to 0.15-0.9, mean held at the target")
    ap.add_argument("--alt-angle", type=float, default=None,
                    help="a second family of lines at this angle on odd infill layers (the warp bends both)")
    ap.add_argument("--families", type=int, default=0,
                    help="K families of straight lines rotating through the infill layers, 180/K degrees apart "
                         "from --angle (3: a 0/60/120 quasi-isotropic layup); 2 is --alt-angle angle+90")
    ap.add_argument("--warp-per-family", action="store_true",
                    help="with --alt-angle: each family gets its own warp (twice the parameters)")
    ap.add_argument("--spacing-search", action="store_true",
                    help="search the straight families' line spacing too (within +-20%% of --spacing)")
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
    ap.add_argument("--then-local", type=int, default=0,
                    help="after the search, refine its best with this many evaluations of local block moves")
    ap.add_argument("--local-sigma", type=float, default=0.5, help="step of the local block moves")
    ap.add_argument("--optimizer", choices=("cma", "elitist", "local"), default="cma",
                    help="cma: CMA-ES; elitist: (1+lambda)-ES around the best point so far")
    ap.add_argument("--evals", type=int, default=1500, help="search budget (0: score the start only)")
    ap.add_argument("--cases", type=int, default=0,
                    help="search over N random force + N random moment cases instead of the reference's 10 + 10 "
                         "(less fitting to one set; check on held-out seeds, scripts/ribbon_heldout.py)")
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link-passes", type=int, default=1,
                    help="merge passes: later passes route the ends still open around the connectors "
                         "already laid (rltp.surfaces.layout.LINK_PASSES)")
    ap.add_argument("--link-reach", type=float, default=None,
                    help="longest connector merging two ribbon ends, mm (rltp.surfaces.layout.LINK_REACH_MM, "
                         "default 4; steps through open infill count 1.5)")
    a = ap.parse_args()
    import rltp.surfaces.layout as _layout   # set before the worker pool forks: workers inherit it
    if a.link_reach is not None:
        _layout.LINK_REACH_MM = a.link_reach
    _layout.LINK_PASSES = a.link_passes
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    part = build(a.part, ["--load-cases", str(a.cases), "--load-moments", str(a.cases)] if a.cases else [])
    raw_s, sum_s = part.raw(), summary(part)
    relay_skins(part, a.skin, a.skin_ring_count)   # after the slicer reference: skins are part of the layout
    if a.skin_from:
        from rltp.export.state import load_state
        st = load_state(a.skin_from)["layers"]
        for k in skin_layers(part):
            L = part.layers[k]
            L.nxt[:] = st[k]["nxt"]
            L.prv[:] = st[k]["prv"]
            L.seam = set(st[k]["seam"])
            L.mask = np.asarray(st[k]["mask"], bool).reshape(L.H, L.W).copy()
            L._cache = {}
        part.refresh_support()
        part.full_recheck()
    w = REF[a.part]["weights"]
    k0 = next(k for k, L in enumerate(part.layers) if is_infill_layer(L))
    poly = part.layers[k0].poly
    bounds = poly.bounds
    bw = part.spec.printer.bead_width
    seed = offset_rings(poly, 1.5 * bw + 0.05, a.ring_gap, a.rings) if a.rings else []
    rho = None
    G["part_name"] = a.part
    G["density_mix"] = a.density_mix
    G["density_hi"] = a.density_hi
    fcache = ROOT / "runs/ribbons/cache" / (f"{a.part}_fields_c{a.density_cases}.npz" if a.density_cases
                                            else f"{a.part}_fields.npz")
    dpart = part                             # the part the density field is fitted on
    if a.density_cases and ((a.density_from == "fields" and not fcache.exists())
                            or (a.density_from == "stress" and a.density_gamma is not None)):
        dpart = build(a.part, ["--load-cases", str(a.density_cases), "--load-moments", str(a.density_cases)])
        relay_skins(dpart, a.skin, a.skin_ring_count)   # (a cached field needs only the geometry)

    def density(target):
        if a.density_from == "fields":
            return field_density(dpart, k0, target, fcache)
        return density_start(dpart, k0, target, a.density_gamma)
    inside = None
    if a.density_norm != "none":             # each cross-section keeps the uniform line count
        Lk = part.layers[k0]
        inside = region_inside(Lk.part & ~Lk.req, Lk.material_frame)

    def dfam(angle, r, t, which):
        if a.density_on not in ("both", which):    # this family stays uniform, at the same mean
            return line_family(bounds, angle, bw / t)
        kw = dict(inside=inside, level=t, norm=a.density_norm) if inside is not None else {}
        return density_family(bounds, angle, r, bw, smooth_along=a.density_smooth, **kw)
    if a.density_gamma is not None or a.density_from == "fields":
        rho = density(bw / a.spacing)
    if a.start == "stream":
        seed += stream_start(part, k0, a.spacing, a.field_smooth)
    elif rho is not None:
        seed += dfam(a.angle, rho, bw / a.spacing, "first")
    elif a.cross:
        seed += line_family(bounds, a.angle, 2 * a.spacing) + line_family(bounds, a.angle + 90, 2 * a.spacing)
    else:
        seed += line_family(bounds, a.angle, a.spacing)
    warp = Warp(bounds, a.grid, a.amp) if a.evals > 0 or a.warm else None
    zs = [L.z - 0.5 * L.h for L in part.layers if is_infill_layer(L)]
    warp3 = Warp3D(bounds, a.grid, a.amp, min(zs), max(zs), a.z_levels, a.max_lean) if a.z_levels > 1 else None
    G.update(part=part, seed=seed, warp=warp, warp3=warp3, alternate=a.alternate, w=w, raw_s=raw_s,
             zs=sorted(zs), dz=part.layers[k0].h)
    if a.families >= 2:                      # K families rotating by layer, 180/K degrees apart
        if rho is not None or a.spacing_search or a.warp_per_family:
            raise SystemExit("--families works with uniform lines only (no density, spacing search or split warps)")
        a.alt_angle = a.angle + 180.0 / a.families
        fixed = [rb for rb in seed if rb.fixed]
        G["seed_more"] = [fixed + line_family(bounds, a.angle + j * 180.0 / a.families, a.spacing)
                          for j in range(2, a.families)]
    if a.alt_angle is not None:
        G["seed_b"] = [rb for rb in seed if rb.fixed] + (dfam(a.alt_angle, rho, bw / a.spacing, "second")
                                                         if rho is not None else line_family(bounds, a.alt_angle, a.spacing))
    if rho is not None:                      # equal material: rescale the field until the laid-out
        want = 100.0 * part.spec.printer.infill          # infill share is the printer's target
        t = bw / a.spacing
        for _ in range(5):
            fams = G["seed"] if G.get("seed_b") is None else (G["seed"], G["seed_b"])
            apply_layout(part, fams, alternate=a.alternate)
            got = float(part.breakdown()["infill_pct"])
            if abs(got - want) < 0.5:
                break
            t *= want / max(got, 1.0)
            rho = density(t)
            fixed = [rb for rb in G["seed"] if rb.fixed]
            G["seed"] = fixed + dfam(a.angle, rho, t, "first")
            if a.alt_angle is not None:
                G["seed_b"] = fixed + dfam(a.alt_angle, rho, t, "second")
        seed = G["seed"]
        src = "fields" if a.density_from == "fields" else f"gamma {a.density_gamma:g}"
        print(f"[ribbons] density {src}: target {t:.3f} lays {got:.1f}% infill", flush=True)
    split = bool(a.warp_per_family and a.alt_angle is not None and warp is not None and warp3 is None)
    G["split"] = split
    x0 = np.zeros(warp3.n if warp3 else (warp.n * (2 if split else 1) if warp else 0))
    if a.warm:
        wres = json.loads(Path(a.warm).read_text())
        xw = np.asarray(wres["best"]["x"], float)
        ws = wres["settings"]
        nw = warp.n if warp else 0
        if (warp is not None and not ws.get("z_levels") and not ws.get("warp_per_family")
                and (ws["grid"], ws["amp"]) != (a.grid, a.amp)):
            xw = regrid(xw, ws, warp, bounds)          # coarse to fine: resample the warm warp
        if warp is None or xw.size < nw:
            raise SystemExit(f"--warm: {xw.size} parameters, but this grid has {nw}")
        x0[:nw] = xw[:nw]                    # a warm start's own skin parameters, if any, are dropped
        if split:                            # the second family: its own warp if the start had one
            x0[nw:2 * nw] = xw[nw:2 * nw] if xw.size >= 2 * nw and json.loads(
                Path(a.warm).read_text())["settings"].get("warp_per_family") else xw[:nw]
    if a.spacing_search:
        if a.start != "lines":
            raise SystemExit("--spacing-search needs --start lines")
        G.update(spacing_search=True, spacing0=a.spacing, spacing_now=round(a.spacing, 3), bounds=bounds,
                 angle=a.angle, alt_angle=a.alt_angle)
        x0 = np.r_[x0, 0.0]
    if a.skin_search:
        ks = skin_layers(part)
        G["skin_ks"] = ks
        G["skin_now"] = None
        if a.skin_rings:
            G["skin_rings"] = True
            x0 = np.r_[x0, np.zeros(len(ks))]
        x0 = np.r_[x0, [-0.5 if c == 0 else 0.5 for c in slicer_skin_classes(part)]]
    if a.warm:                               # same settings: continue from the whole best point
        wres = json.loads(Path(a.warm).read_text())
        xw = np.asarray(wres["best"]["x"], float)
        ws = wres["settings"]
        if (warp is not None and warp3 is None and not split and not ws.get("z_levels")
                and not ws.get("warp_per_family") and (ws["grid"], ws["amp"]) != (a.grid, a.amp)):
            xw = regrid(xw, ws, warp, bounds)          # coarse to fine: resample the warm warp
        if xw.size == x0.size:
            x0 = xw.copy()
        elif warp is not None and not ws.get("z_levels") and 0 < xw.size - warp.n < x0.size:
            x0[-(xw.size - warp.n):] = xw[warp.n:]    # vertical into 3D: keep its skins/spacing too
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
        if a.optimizer == "local":
            es = LocalSearch(x0, a.sigma, a.popsize, a.seed, -f0, param_blocks(x0.size, warp, warp3, split))
        elif a.optimizer == "elitist":
            es = ElitistES(x0, a.sigma, a.popsize, a.seed, f0=-f0)
        else:
            es = CMAES(x0, a.sigma, a.popsize, a.seed)
        n_ev = 1
        ctx = mp.get_context("fork") if a.workers > 1 and "fork" in mp.get_all_start_methods() else None
        pool = ctx.Pool(a.workers) if ctx else None
        phases = [(es, a.evals)]
        if a.then_local > 0:                 # then refine the best with local block moves
            phases.append((None, a.evals + a.then_local))
        try:
            for es, limit in phases:
                if es is None:
                    es = LocalSearch(best[1], a.local_sigma, a.popsize, a.seed + 1, -best[0],
                                     param_blocks(best[1].size, warp, warp3, split))
                    print(f"[ribbons] local refinement from {best[0]:+.1f}", flush=True)
                while n_ev < limit:
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
                    if es.gen % 5 == 0 or n_ev >= limit:
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
                 **({"skin": [0 if v < 0 else 90 for v in best[1][-len(G["skin_ks"]):]]}
                    if G.get("skin_ks") else {}),
                 **({"skin_rings": [int(np.clip(np.round(v), 0, 3))
                                    for v in best[1][-2 * len(G["skin_ks"]):-len(G["skin_ks"])]]}
                    if G.get("skin_rings") else {}),
                 **({"spacing_mm": spacing_of(best[1][-1 - len(G.get("skin_ks") or [])
                                                     * (2 if G.get("skin_rings") else 1)])}
                    if G.get("spacing_search") else {}),
                 **({"lean_deg": warp3.lean(best[1][:warp3.n], seed, zs)} if warp3 else {})},
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
