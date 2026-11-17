"""Fragmentation of a suite run's layouts: slicer start, the run's start (state0), final.
Per layout and layer kind: non-wall strokes, short ones, and what each path end runs into
(a wall, another bead, or open space)."""
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]      # repo root
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(R))
from direction_maps import build_for, restore  # noqa: E402

from rltp.cover.beads import bead_class  # noqa: E402
from rltp.cover.cover import get_pieces  # noqa: E402
from rltp.export.state import load_state  # noqa: E402

run = Path(sys.argv[1])
png = sys.argv[2] if len(sys.argv) > 2 else None
res, part, w = build_for(run)


def kind(L):
    inner = L.aux["inner"]
    return "skin" if (L.req & inner).sum() > 0.5 * max(inner.sum(), 1) else "infill"


def end_kind(L, e, prev, look):
    W = L.W
    d = e - prev
    m, lock, inner = L.flat_mask, L.lock.reshape(-1), L.aux["inner"].reshape(-1)
    x = e
    for _ in range(look):
        r0, c0 = divmod(x, W)
        x = x + d
        r1, c1 = divmod(x, W) if 0 <= x < L.n else (-9, -9)
        if x < 0 or x >= L.n or abs(r1 - r0) + abs(c1 - c0) != 1:
            return "wall"
        if lock[x] or not inner[x]:
            return "wall"
        if m[x]:
            return "bead"
    return "open"


def stats(label):
    agg = {}
    for k, L in enumerate(part.layers):
        kd = kind(L)
        a = agg.setdefault(kd, {"layers": 0, "strokes": 0, "lt1": 0, "lt3": 0, "len": 0.0, "short_len": 0.0,
                                "wall": 0, "bead": 0, "open": 0})
        a["layers"] += 1
        pc = get_pieces(L)
        lock = L.lock.reshape(-1)
        p = L.frame.p
        look = 2 * L.kb
        for cells, loop in zip(pc.cells, pc.is_loop):
            cells = np.asarray(cells)
            if lock[cells].all():
                continue
            ln = len(cells) * p
            a["strokes"] += 1
            a["len"] += ln
            a["lt1"] += ln < 1.0
            a["lt3"] += ln < 3.0
            if ln < 3.0:
                a["short_len"] += ln
            if loop or len(cells) < 2:
                continue
            for e, pv in ((cells[0], cells[1]), (cells[-1], cells[-2])):
                a[end_kind(L, int(e), int(pv), look)] += 1
    for kd, a in agg.items():
        n_end = a["wall"] + a["bead"] + a["open"]
        print(f"{label:9s} {kd:6s} layers {a['layers']:2d}  strokes/layer {a['strokes'] / a['layers']:6.1f}  "
              f"<1mm {a['lt1'] / a['layers']:5.1f}  <3mm {a['lt3'] / a['layers']:5.1f}  "
              f"mean len {a['len'] / max(a['strokes'], 1):5.1f} mm  share of length in <3mm {100 * a['short_len'] / a['len']:4.1f}%  "
              f"ends: wall {100 * a['wall'] / n_end:3.0f}% bead {100 * a['bead'] / n_end:3.0f}% open {100 * a['open'] / n_end:3.0f}%")


stats("slicer")
restore(part, load_state(run / "state0.npz"))
stats("start")
restore(part, load_state(run / "state.npz"))
stats("final")

if png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "0.6", "#e03131", "#2f9e44", "#1c7ed6", "#ae3ec9"])
    n = len(part.layers)
    cols = 6
    fig, axs = plt.subplots((n + cols - 1) // cols, cols, figsize=(3 * cols, 2.4 * ((n + cols - 1) // cols)))
    for k, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if k >= n:
            ax.axis("off")
            continue
        L = part.layers[k]
        cls = bead_class(L)
        img = np.zeros(cls.shape, int)
        for i, d in enumerate((0, 45, 90, 135)):
            img[cls == d] = 2 + i
        img[L.lock & L.mask] = 1
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
        ax.set_title(f"layer {k} ({kind(L)})", fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=70)
