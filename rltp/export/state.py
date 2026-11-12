"""Save / load the final part state (state.npz) so render and analysis tools work
without rebuilding a PartState.

Layout (np.savez_compressed), one group of keys per layer k:
  L{k}_mask (bool H x W), L{k}_frame [ox, oy, theta, p], L{k}_zh [z, h, T],
  L{k}_nxt, L{k}_prv (int32 H*W), L{k}_seam (int32),
  L{k}_t (float32 H*W, nan outside), L{k}_fi (float32 H*W),
  L{k}_seq (int32: stroke cells concatenated in print order, loops not closed),
  L{k}_slen (int32 per stroke), L{k}_sloop (bool per stroke),
  L{k}_ts / L{k}_te (float64 per stroke: print start / end time),
  L{k}_travel (float64 per stroke: travel length before it), L{k}_retract (bool per stroke)
plus `meta` (JSON string) holding {"n_layers", "version", ...user metadata}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rltp.cover.cover import get_pieces

VERSION = 1


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_state(path, part, meta: dict | None = None) -> None:
    arrs: dict = {}
    for k, (L, ev) in enumerate(zip(part.layers, part.evals)):
        pc = get_pieces(L)
        order = ev.timing.order
        seqs = [pc.cells[pid] for pid in order]
        arrs[f"L{k}_mask"] = L.mask.astype(bool)
        arrs[f"L{k}_frame"] = np.array([L.frame.ox, L.frame.oy, L.frame.theta, L.frame.p])
        arrs[f"L{k}_zh"] = np.array([L.z, L.h, ev.T])
        arrs[f"L{k}_nxt"] = L.nxt.astype(np.int32)
        arrs[f"L{k}_prv"] = L.prv.astype(np.int32)
        arrs[f"L{k}_seam"] = np.array(sorted(L.seam), np.int32)
        arrs[f"L{k}_t"] = ev.timing.t_cell.astype(np.float32)
        arrs[f"L{k}_fi"] = np.asarray(ev.fea.fi, np.float32)
        arrs[f"L{k}_seq"] = (np.concatenate(seqs) if seqs else np.empty(0)).astype(np.int32)
        arrs[f"L{k}_slen"] = np.array([len(s) for s in seqs], np.int32)
        arrs[f"L{k}_sloop"] = np.array([bool(pc.is_loop[pid]) for pid in order], bool)
        arrs[f"L{k}_ts"] = np.asarray(ev.timing.stroke_start, float)
        arrs[f"L{k}_te"] = np.asarray(ev.timing.stroke_start + ev.timing.stroke_time, float)
        arrs[f"L{k}_travel"] = np.asarray(ev.timing.travel_dist, float)
        arrs[f"L{k}_retract"] = np.asarray(ev.timing.retract, bool)
    m = {"version": VERSION, "n_layers": len(part.layers)}
    if meta:
        m.update(_jsonable(meta))
    arrs["meta"] = np.array(json.dumps(m))
    np.savez_compressed(path, **arrs)


def load_state(path) -> dict:
    """{"meta": dict, "layers": [dict per layer]}; each layer dict has k, mask, frame
    (dict ox/oy/theta/p), z, h, T, nxt, prv, seam (set), t_cell, fi, strokes (list of
    int arrays in print order, loops not closed), loops, stroke_start, stroke_end,
    travel_dist, retract."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        layers = []
        for k in range(int(meta["n_layers"])):
            g = lambda name: z[f"L{k}_{name}"]  # noqa: E731
            fr = g("frame")
            zh = g("zh")
            seq, slen = g("seq"), g("slen")
            bounds = np.concatenate([[0], np.cumsum(slen)])
            strokes = [seq[bounds[i]:bounds[i + 1]].astype(np.int64) for i in range(len(slen))]
            layers.append({
                "k": k,
                "mask": g("mask").astype(bool),
                "frame": {"ox": float(fr[0]), "oy": float(fr[1]), "theta": float(fr[2]), "p": float(fr[3])},
                "z": float(zh[0]), "h": float(zh[1]), "T": float(zh[2]),
                "nxt": g("nxt").astype(np.int64),
                "prv": g("prv").astype(np.int64),
                "seam": set(int(s) for s in g("seam")),
                "t_cell": g("t").astype(float),
                "fi": g("fi").astype(float),
                "strokes": strokes,
                "loops": g("sloop").astype(bool),
                "stroke_start": g("ts").astype(float),
                "stroke_end": g("te").astype(float),
                "travel_dist": g("travel").astype(float),
                "retract": g("retract").astype(bool),
            })
    return {"meta": meta, "layers": layers}
