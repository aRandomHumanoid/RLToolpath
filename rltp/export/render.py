"""Render the viewer's views as PNG frames, GIF or MP4 with matplotlib (for papers and slides).

    render(out_dir, mode="layer" | "playback" | "history", layer=None, frames=60, fps=20,
           fmt="png" | "gif" | "mp4", color="direction" | "order" | "fi" | "stroke", dest=None)

Reads out_dir/state.npz (and out_dir/history.jsonl for mode="history").
`draw_layer` is also used to save toolpath images during training.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from rltp.cover.cover import walk_pieces  # noqa: E402
from rltp.eval.timing import order_strokes  # noqa: E402
from rltp.export.state import load_state  # noqa: E402
from rltp.model import Frame, Layer  # noqa: E402

log = logging.getLogger(__name__)

TRAVEL_C = "#7c7f88"
SEAM_C = "#e8590c"
END_C = "#3d3f45"
HL_C = (0.88, 0.16, 0.37, 0.45)


# ------------------------------------------------------------------ helpers

def _frame(frame) -> Frame:
    if isinstance(frame, Frame):
        return frame
    if isinstance(frame, dict):
        return Frame(float(frame["ox"]), float(frame["oy"]), float(frame["theta"]), float(frame["p"]))
    ox, oy, th, p = frame
    return Frame(float(ox), float(oy), float(th), float(p))


def _prv_from_nxt(nxt: np.ndarray) -> np.ndarray:
    prv = np.full(nxt.size, -1, np.int64)
    i = np.flatnonzero(nxt >= 0)
    prv[nxt[i]] = i
    return prv


def walk_cover(mask, frame, nxt, prv=None, seams=()):
    """Strokes as (cells, is_loop) in nearest-neighbour print order (the time model's rule)."""
    fr = _frame(frame)
    nxt = np.asarray(nxt, np.int32)
    prv = _prv_from_nxt(nxt).astype(np.int32) if prv is None else np.asarray(prv, np.int32)
    L = Layer(0, 0.0, 0.0, np.asarray(mask, bool), fr, nxt.copy(), prv.copy(), set(int(s) for s in seams))
    pc = walk_pieces(L)
    order = order_strokes(pc, L.W, fr.p)
    return [(pc.cells[i], bool(pc.is_loop[i])) for i in order]


def _cell_quads(fr: Frame, cells, W):
    r, c = np.divmod(np.asarray(cells, np.int64), W)
    p = fr.p
    us = np.stack([c * p, (c + 1) * p, (c + 1) * p, c * p], 1)
    vs = np.stack([r * p, r * p, (r + 1) * p, (r + 1) * p], 1)
    x, y = fr.to_world(us, vs)
    return np.stack([x, y], -1)


def _link_colors(fr, a, b, W, mode, t_cell, stroke_idx, n_strokes, T):
    if mode == "direction":
        ang = fr.theta + np.where(np.abs(b - a) == 1, 0.0, np.pi / 2)
        h = np.mod(np.degrees(ang), 180.0) / 180.0
        return plt.get_cmap("hsv")(h)
    if mode == "order":
        if t_cell is not None and T > 0:
            f = np.nan_to_num(t_cell[a] / T)
        else:
            f = stroke_idx / max(n_strokes - 1, 1)
        return plt.get_cmap("viridis")(np.clip(f, 0, 1))
    if mode == "stroke":
        return plt.get_cmap("tab20")(stroke_idx % 20)
    return np.tile(np.array([[0.19, 0.2, 0.22, 1.0]]), (len(a), 1))


def _links(strokes, upto=None, t_cell=None, ts=None, te=None):
    """Links printed by time `upto` (all when None).

    Returns (a, b, stroke index, fraction drawn) per link, the number of strokes
    started, and the nozzle as (from cell, to cell or None, fraction)."""
    A, B, I, F = [], [], [], []

    def out(n_drawn, nozzle):
        return (np.array(A, np.int64), np.array(B, np.int64), np.array(I, np.int64),
                np.array(F, float), n_drawn, nozzle)

    nozzle = None
    for i, (c, loop) in enumerate(strokes):
        if upto is not None and ts[i] > upto:
            if i > 0 and upto > te[i - 1]:
                prev_c, prev_loop = strokes[i - 1]
                end = int(prev_c[0] if prev_loop else prev_c[-1])
                f = (upto - te[i - 1]) / max(ts[i] - te[i - 1], 1e-9)
                nozzle = (end, int(c[0]), f)
            return out(i, nozzle)
        L = len(c)
        for j in range(L if loop else L - 1):
            a = int(c[j])
            b = int(c[j + 1]) if j + 1 < L else int(c[0])
            f = 1.0
            if upto is not None:
                ta = t_cell[a]
                tb = t_cell[b] if j + 1 < L else te[i]
                if tb > upto:
                    f = (upto - ta) / max(tb - ta, 1e-9) if upto > ta else 0.0
                    A.append(a); B.append(b); I.append(i); F.append(f)
                    return out(i + 1, (a, b, f))
            A.append(a); B.append(b); I.append(i); F.append(f)
        nozzle = (int(c[0] if loop else c[-1]), None, 1.0)
    return out(len(strokes), nozzle)


# ------------------------------------------------------------------ public drawing API

def draw_layer(ax, mask, frame, nxt, prv=None, seams=(), color="direction", fi=None, t_cell=None,
               strokes=None, highlight=None, travels=True, markers=True, upto=None, ts=None, te=None,
               title=None, lw=None):
    """Draw one layer's cover on a matplotlib axes.

    mask (H, W) bool; frame a Frame or dict(ox, oy, theta, p); nxt/prv flat int arrays
    (prv derived if None); seams a set of cells. color: direction | order | fi | stroke.
    strokes: optional [(cells, is_loop)] in print order (else nearest-neighbour order).
    highlight: cells to shade. upto/ts/te/t_cell: draw only what is printed by time `upto`."""
    mask = np.asarray(mask, bool)
    fr = _frame(frame)
    H, W = mask.shape
    if strokes is None:
        strokes = walk_cover(mask, fr, nxt, prv, seams)
    cells = np.flatnonzero(mask.reshape(-1))
    ax.set_aspect("equal")
    ax.set_axis_off()
    if cells.size == 0:
        return ax
    quads = _cell_quads(fr, cells, W)
    xy_all = quads.reshape(-1, 2)
    pad = fr.p
    ax.set_xlim(xy_all[:, 0].min() - pad, xy_all[:, 0].max() + pad)
    ax.set_ylim(xy_all[:, 1].min() - pad, xy_all[:, 1].max() + pad)
    if color == "fi" and fi is not None:
        f = np.asarray(fi, float)[cells]
        pcoll = PolyCollection(quads, array=f, cmap="YlOrRd", edgecolors="none")
        pcoll.set_clim(0, max(float(f.max()), 1e-12))
        ax.add_collection(pcoll)
    else:
        ax.add_collection(PolyCollection(quads, facecolors=(0, 0, 0, 0.05), edgecolors="none"))
    if lw is None:
        fig = ax.figure
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        xr = ax.get_xlim()
        pts_per_unit = bbox.width * 72 / max(xr[1] - xr[0], 1e-9)
        lw = float(np.clip(fr.p * pts_per_unit * 0.62, 0.3, 12))

    T = float(np.nanmax(t_cell)) if t_cell is not None and np.isfinite(t_cell).any() else 0.0
    if te is not None and len(te):
        T = max(T, float(np.max(te)))
    a, b, si, frac, n_drawn, noz = _links(strokes, upto, t_cell, ts, te)
    if a.size:
        xa, ya = fr.cell_xy(a, W)
        xb, yb = fr.cell_xy(b, W)
        xb = xa + (xb - xa) * frac
        yb = ya + (yb - ya) * frac
        segs = np.stack([np.stack([xa, ya], 1), np.stack([xb, yb], 1)], 1)
        cols = _link_colors(fr, a, b, W, "path" if color == "fi" else color, t_cell, si, len(strokes), T)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw * (0.4 if color == "fi" else 1.0),
                                         capstyle="round", joinstyle="round"))
    singles = [c[0] for c, lp in strokes[:n_drawn] if len(c) == 1]
    if singles:
        x, y = fr.cell_xy(np.array(singles), W)
        ax.scatter(x, y, s=(lw * 0.9) ** 2, c="#555", zorder=3)
    if travels and n_drawn > 1:
        ends = [c[0] if lp else c[-1] for c, lp in strokes[:n_drawn - 1]]
        starts = [c[0] for c, _ in strokes[1:n_drawn]]
        xe, ye = fr.cell_xy(np.array(ends), W)
        xs, ys = fr.cell_xy(np.array(starts), W)
        segs = np.stack([np.stack([xe, ye], 1), np.stack([xs, ys], 1)], 1)
        ax.add_collection(LineCollection(segs, colors=TRAVEL_C, linewidths=0.6, linestyles=(0, (3, 3))))
    if markers and upto is None:
        seam_c = [c[0] for c, lp in strokes if lp]
        end_c = [x for c, lp in strokes if not lp for x in ((c[0], c[-1]) if len(c) > 1 else (c[0],))]
        if end_c:
            x, y = fr.cell_xy(np.array(end_c), W)
            ax.scatter(x, y, s=max(lw * 0.5, 1.0) ** 2, c=END_C, zorder=4, linewidths=0)
        if seam_c:
            x, y = fr.cell_xy(np.array(seam_c), W)
            ax.scatter(x, y, s=max(lw * 0.8, 2.0) ** 2, c=SEAM_C, zorder=5, linewidths=0)
    if highlight is not None and len(highlight):
        ax.add_collection(PolyCollection(_cell_quads(fr, np.asarray(list(highlight)), W),
                                         facecolors=HL_C, edgecolors="none", zorder=6))
    if upto is not None and noz is not None:
        ca, cb, f = noz
        x0, y0 = fr.cell_xy(np.array([ca]), W)
        if cb is not None:
            x1, y1 = fr.cell_xy(np.array([cb]), W)
            x0, y0 = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        ax.scatter(x0, y0, s=max(lw * 2.2, 6) ** 2, facecolors=(0.18, 0.43, 0.87, 0.3),
                   edgecolors=(0.18, 0.43, 0.87, 1.0), linewidths=1.5, zorder=7)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


# ------------------------------------------------------------------ history replay (from jsonl events)

def read_history(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def replay_history(events: list[dict], k: int, n_cells: int):
    """Yield (iter, nxt, prv, seams, changed_cells) for layer k: first the initial snapshot
    (iter -1), then the state after each accepted edit on that layer."""
    nxt = prv = None
    seams: set = set()
    for e in events:
        typ = e.get("type")
        if typ == "snapshot" and e["layer"] == k and nxt is None:
            nxt = np.asarray(e["nxt"], np.int64)
            prv = _prv_from_nxt(nxt)
            seams = set(e.get("seam", []))
            yield -1, nxt.copy(), prv.copy(), set(seams), []
        elif typ == "edit" and e["layer"] == k and e.get("accepted") and "diff" in e and nxt is not None:
            d = e["diff"]
            cells = np.asarray(d["cells"], np.int64)
            nxt[cells] = d["nxt"]
            prv[cells] = d["prv"]
            seams -= set(d.get("seam_del", []))
            seams |= set(d.get("seam_add", []))
            yield int(e["iter"]), nxt.copy(), prv.copy(), set(seams), cells.tolist()


# ------------------------------------------------------------------ writers

class _Sink:
    """Collects frames as numbered PNGs, a GIF, or an MP4."""

    def __init__(self, fig, dest: Path, stem: str, fmt: str, fps: int, dpi: int):
        self.fig, self.dest, self.stem, self.dpi = fig, dest, stem, dpi
        self.paths: list[Path] = []
        self.i = 0
        if fmt == "mp4" and shutil.which("ffmpeg") is None:
            log.warning("ffmpeg not found on PATH; writing a GIF instead of MP4")
            fmt = "gif"
        self.fmt = fmt
        self.writer = None
        if fmt in ("gif", "mp4"):
            from matplotlib import animation
            path = dest / f"{stem}.{fmt}"
            self.writer = (animation.PillowWriter(fps=fps) if fmt == "gif"
                           else animation.FFMpegWriter(fps=fps))
            self.writer.setup(fig, str(path), dpi=dpi)
            self.paths.append(path)

    def add(self):
        if self.writer is not None:
            self.writer.grab_frame()
        else:
            p = self.dest / f"{self.stem}_{self.i:04d}.png"
            self.fig.savefig(p, dpi=self.dpi)
            self.paths.append(p)
        self.i += 1

    def close(self) -> list[Path]:
        if self.writer is not None:
            self.writer.finish()
        plt.close(self.fig)
        return self.paths


def _fig(lay, size=6.0):
    m = lay["mask"]
    H, W = m.shape
    aspect = H / max(W, 1)
    fig, ax = plt.subplots(figsize=(size, max(2.0, size * min(aspect, 2.0))), dpi=100)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.93)
    return fig, ax


def render(out_dir, mode: str = "layer", layer: int | None = None, frames: int = 60, fps: int = 20,
           fmt: str = "png", color: str = "direction", dest=None, dpi: int = 120) -> list[Path]:
    out_dir = Path(out_dir)
    st = load_state(out_dir / "state.npz")
    layers = st["layers"]
    dest = Path(dest) if dest else out_dir / "render"
    dest.mkdir(parents=True, exist_ok=True)
    if fmt not in ("png", "gif", "mp4"):
        raise ValueError(f"unknown format {fmt!r}")

    def lay_kwargs(lay):
        return dict(mask=lay["mask"], frame=lay["frame"], nxt=lay["nxt"], prv=lay["prv"], seams=lay["seam"],
                    fi=lay["fi"], t_cell=lay["t_cell"], strokes=list(zip(lay["strokes"], lay["loops"])))

    if mode == "layer":
        ks = range(len(layers)) if layer is None else [layer]
        if fmt == "png":
            paths = []
            for k in ks:
                lay = layers[k]
                fig, ax = _fig(lay)
                draw_layer(ax, color=color, title=f"layer {k}  z={lay['z']:.2f} mm  T={lay['T']:.1f} s",
                           **lay_kwargs(lay))
                p = dest / f"layer_{k:03d}.png"
                fig.savefig(p, dpi=dpi)
                plt.close(fig)
                paths.append(p)
            return paths
        fig, ax = _fig(layers[0])
        sink = _Sink(fig, dest, "layers", fmt, fps, dpi)
        for k in ks:
            ax.clear()
            lay = layers[k]
            draw_layer(ax, color=color, title=f"layer {k}  z={lay['z']:.2f} mm", **lay_kwargs(lay))
            sink.add()
        return sink.close()

    if mode == "playback":
        k = 0 if layer is None else layer
        lay = layers[k]
        T = float(lay["T"])
        fig, ax = _fig(lay)
        sink = _Sink(fig, dest, f"playback_L{k:03d}", fmt, fps, dpi)
        for t in np.linspace(0, T, max(frames, 2)):
            ax.clear()
            draw_layer(ax, color=color, upto=float(t), ts=lay["stroke_start"], te=lay["stroke_end"],
                       title=f"layer {k}  t={t:.1f} / {T:.1f} s", **lay_kwargs(lay))
            sink.add()
        return sink.close()

    if mode == "history":
        hp = out_dir / "history.jsonl"
        if not hp.exists():
            raise FileNotFoundError(f"{hp} not found (baseline runs have no history)")
        events = read_history(hp)
        if layer is None:
            counts = np.zeros(len(layers), int)
            for e in events:
                if e.get("type") == "edit" and e.get("accepted"):
                    counts[e["layer"]] += 1
            k = int(np.argmax(counts))
        else:
            k = layer
        lay = layers[k]
        states = list(replay_history(events, k, lay["mask"].size))
        if not states:
            raise ValueError(f"no history for layer {k}")
        pick = np.unique(np.rint(np.linspace(0, len(states) - 1, min(frames, len(states)))).astype(int))
        fig, ax = _fig(lay)
        sink = _Sink(fig, dest, f"history_L{k:03d}", fmt, fps, dpi)
        hcolor = "direction" if color in ("fi", "order") else color
        for i in pick:
            it, nxt, prv, seams, changed = states[i]
            part = lay.get("part", lay["mask"])
            filled = (part.reshape(-1) & (nxt != -2)).reshape(part.shape)   # -2: empty (sparse layers)
            ax.clear()
            draw_layer(ax, filled, lay["frame"], np.where(nxt == -2, -1, nxt), prv, seams, color=hcolor,
                       highlight=changed,
                       title=f"layer {k}  " + ("baseline" if it < 0 else f"after iteration {it}"))
            sink.add()
        return sink.close()

    raise ValueError(f"unknown mode {mode!r}")
