"""Infill-layer bead paths before and after anchoring: cycle 16 (no anchoring) and cycle 17
(anchored starts + free-end cost), --adapt start and final; free path ends in red.
usage: python scripts/analysis/ends_fig.py OUT.png LAYER PART [PART ...]"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[2]      # repo root
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(R))
from direction_maps import build_for, restore  # noqa: E402

from rltp.cover.beads import free_ends  # noqa: E402
from rltp.cover.cover import get_pieces  # noqa: E402
from rltp.export.state import load_state  # noqa: E402

out, k, parts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
S = R / "runs/selfimprove"
panels = [("before: gradient start", "c16", "state0.npz"), ("before: after annealing", "c16", "state.npz"),
          ("anchored: gradient start", "c17", "state0.npz"), ("anchored: after annealing", "c17", "state.npz")]
fig, axs = plt.subplots(len(parts), 4, figsize=(16, 4.3 * len(parts)), squeeze=False)
for row, name in zip(axs, parts):
    res, part, w = build_for(S / "c17" / name / "ADAPT")
    for ax, (title, cyc, f) in zip(row, panels):
        restore(part, load_state(S / cyc / name / "ADAPT" / f))
        L = part.layers[k]
        p = L.frame.p
        lock = L.lock.reshape(-1)
        pc = get_pieces(L)
        for cells, loop in zip(pc.cells, pc.is_loop):
            cells = list(cells) + ([cells[0]] if loop else [])
            r, c = np.divmod(np.asarray(cells), L.W)
            wall = bool(lock[cells[0]])
            ax.plot(c * p, r * p, color="#a9b1ba" if wall else "#3b6fb6", lw=1, alpha=0.9,
                    solid_capstyle="round", solid_joinstyle="round", gid="bead")
        fe = free_ends(L, pc)
        if fe:
            r, c = np.divmod(np.asarray(fe), L.W)
            ax.plot(c * p, r * p, "o", ms=9, mfc="none", mec="#d62728", mew=1.8, zorder=5)
        ax.set_title(f"{name}, {title}\n{len(fe)} free ends in layer {k}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
fig.tight_layout()
fig.canvas.draw()
for ax in axs.flat:                    # draw each bead at its true width (0.4 mm)
    x0, x1 = ax.get_xlim()
    pts_per_mm = ax.get_window_extent().width / fig.dpi * 72 / (x1 - x0)
    for ln in ax.lines:
        if ln.get_gid() == "bead":
            ln.set_linewidth(0.4 * pts_per_mm)
fig.savefig(out, dpi=110)
print("wrote", out)
