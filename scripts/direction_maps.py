"""Bead directions before and after a suite run (scripts/suite.py), next to the direction
each cell would suit best under the final strains (rltp.search.guide).

usage: python scripts/direction_maps.py OUT.png RUN_DIR [RUN_DIR ...] [--layers 1,4]
       python scripts/direction_maps.py OUT.png --compare BASE_RUN FINAL_RUN [--layers 1,4]

Each RUN_DIR is OUT_DIR/<part>/<arm> of a suite run (state0.npz, state.npz, result.json).
One row per run and layer: start, final, best direction; colours 0 / 45 / 90 / 135 degrees,
walls grey, empty white.
"""
import json
import shlex
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite import COMMON, ROOT  # noqa: E402

from rltp.cli import _build, _spec, build_parser  # noqa: E402
from rltp.config import parse_weights  # noqa: E402
from rltp.cover.beads import bead_class  # noqa: E402
from rltp.export.state import load_state  # noqa: E402
from rltp.search.fields import wall_cells  # noqa: E402
from rltp.search.guide import DirectionMap  # noqa: E402

CMAP = ListedColormap(["white", "0.6", "#1c7ed6", "#2f9e44", "#e8590c", "#ae3ec9"])   # empty, wall, 0, 45, 90, 135


def code(L):
    cls = bead_class(L)
    out = np.zeros(cls.shape, int)
    for i, d in enumerate((0, 45, 90, 135)):
        out[cls == d] = 2 + i
    out[wall_cells(L)] = 1
    return out


def restore(part, st):
    snap = [(d["nxt"], d["prv"], d["seam"], d["mask"]) for d in st["layers"]]
    part.restore(snap)


def part_dir(name, run) -> Path:
    """Where the run's part files are: examples/ (5 mm since Nov 15, 2026) or, for a run whose
    saved start has the 2 mm parts' 10 layers, examples/2mm/."""
    n = len(load_state(run / "state0.npz")["layers"])
    for d in (ROOT / "examples", ROOT / "examples/2mm"):
        if (d / f"{name}.stl").exists() and round(trimesh.load(d / f"{name}.stl").extents[2] / 0.2) == n:
            return d
    raise FileNotFoundError(f"no examples/ or examples/2mm/ {name}.stl with {n} layers for {run}")


def run_argv(res, run):
    """`rltp optimize` arguments that rebuild a suite run's part. Runs made before the switch
    to one wall (Nov 14, 2026) don't record `walls` and used 2. Runs before Nov 15, 2026 don't
    record `yaml_loads`, `load_weights` or `solid_layers`: they used the part's loads.yaml,
    edges drawn alike, and 3 solid layers on the 2 mm parts (2 on the 5 mm ones). Runs that
    don't record `directions` scored the 0/45/90/135 classes. An arm's own --directions
    (in its args) overrides."""
    name, d = res["part"], part_dir(res["part"], run)
    loads = ["--loads", str(d / f"{name}_loads.yaml")] if res.get("yaml_loads", True) else []
    lw = res.get("load_weights", "uniform")
    solid = res.get("solid_layers", 3 if d.name == "2mm" else 2)
    return (["optimize", str(d / f"{name}.stl")] + loads + ["--out", "/tmp/x"] + COMMON
            + ["--walls", str(res.get("walls", 2)), "--solid-layers", str(solid),
               "--load-weights", lw if isinstance(lw, str) else ",".join(map(str, lw)),
               "--directions", res.get("directions", "classes")]
            + res["extra"] + shlex.split(res["args"]))


def build_for(run):
    """The run's part (slicer layout) and weights, rebuilt from its result.json."""
    res = json.loads((run / "result.json").read_text())
    args = build_parser().parse_args(run_argv(res, run))
    w = parse_weights(args.weights)
    from rltp.cli import _apply_adapt
    _apply_adapt(args, w)
    args.start = "slicer"                      # the layouts come from the saved states
    spec = _spec(args, w)
    return res, _build(args, spec, w), w


def compare(out, base, final, layers):
    """Rows (part, layer): slicer start, BASE final, FINAL start, FINAL final, best direction
    under FINAL's final strains; each panel's share of bead cells in that best direction."""
    res, part, w = build_for(final)
    slicer = [code(L) for L in part.layers]
    restore(part, load_state(base / "state.npz"))
    based = [code(L) for L in part.layers]
    restore(part, load_state(final / "state0.npz"))
    f0 = [code(L) for L in part.layers]
    restore(part, load_state(final / "state.npz"))
    f1 = [code(L) for L in part.layers]
    gm = DirectionMap(part, w=w)
    fig, axs = plt.subplots(len(layers), 5, figsize=(17, 3.4 * len(layers)), squeeze=False)
    for ax, k in zip(axs, layers):
        L = part.layers[k]
        best, _, _ = gm.layer(k)
        cb = np.where(best.reshape(L.H, L.W) >= 0, best.reshape(L.H, L.W) + 2, 0)
        cb[f1[k] == 1] = 1
        panels = [("slicer start", slicer[k]), ("plain annealing", based[k]), ("--adapt start", f0[k]),
                  ("--adapt final", f1[k]), ("best direction (final strains)", cb)]
        for a, (title, img) in zip(ax, panels):
            a.imshow(img, origin="lower", cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
            if title.startswith("best"):
                a.set_title(f"{res['part']} layer {k}: {title}", fontsize=9)
            else:
                both = (img >= 2) & (cb >= 2)
                al = ((img == cb) & both).sum() / max(both.sum(), 1)
                a.set_title(f"{title}: {100 * al:.0f}% aligned", fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
    fig.text(0.5, 0.005, "blue 0°, green 45°, orange 90°, purple 135°, grey walls", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=80)
    print(f"wrote {out}")


def main():
    argv = sys.argv[1:]
    layers = [1, 4]
    if "--layers" in argv:
        i = argv.index("--layers")
        layers = [int(x) for x in argv[i + 1].split(",")]
        argv = argv[:i] + argv[i + 2:]
    if "--compare" in argv:                    # OUT.png --compare BASE_RUN FINAL_RUN
        i = argv.index("--compare")
        compare(Path(argv[0]), Path(argv[i + 1]), Path(argv[i + 2]), layers)
        return
    out, runs = Path(argv[0]), [Path(a) for a in argv[1:]]
    rows = [(r, k) for r in runs for k in layers]
    fig, axs = plt.subplots(len(rows), 3, figsize=(13, 3.2 * len(rows)), squeeze=False)
    for (run, k), ax in zip(rows, axs):
        res = json.loads((run / "result.json").read_text())
        name = res["part"]
        args = build_parser().parse_args(run_argv(res, run))
        w = parse_weights(args.weights)
        args.start = "slicer"                      # the layouts come from the saved states
        spec = _spec(args, w)
        part = _build(args, spec, w)
        restore(part, load_state(run / "state0.npz"))
        c0 = code(part.layers[k])
        restore(part, load_state(run / "state.npz"))
        c1 = code(part.layers[k])
        best, _, _ = DirectionMap(part, w=w).layer(k)
        L = part.layers[k]
        cb = np.where(best.reshape(L.H, L.W) >= 0, best.reshape(L.H, L.W) + 2, 0)
        cb[c1 == 1] = 1
        same = (c1 >= 2) & (cb >= 2)
        al0 = ((c0 == cb) & (c0 >= 2) & (cb >= 2)).sum() / max(((c0 >= 2) & (cb >= 2)).sum(), 1)
        al1 = ((c1 == cb) & same).sum() / max(same.sum(), 1)
        for a, img, title in zip(ax, (c0, c1, cb), (f"{name} layer {k}: start", f"{res['arm']} final",
                                                    "best direction (final strains)")):
            a.imshow(img, origin="lower", cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
            a.set_title(title, fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
        ax[1].set_xlabel(f"cells in their best direction: {100 * al0:.0f}% -> {100 * al1:.0f}%", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
