"""Part-level loop: pick a layer, run a short burst of proposals, keep accepted edits.

Stops once acceptance has turned greedy (temperature 0) and a full pass gains
less than eps, or when the evaluator budget runs out. Returns the best part seen.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from rltp.config import TERMS
from rltp.cover.ops import OPS, action_mask, apply_action, layer_diff, ops_to_indices, sample_valid
from rltp.part import PartState

log = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    burst: int = 32
    temp0: float | None = None       # None -> proposer default
    anneal_passes: int = 4           # temperature reaches 0 after this many passes
    eps: float = 1e-3
    max_passes: int = 50
    max_evals: int = 20000
    warmup: int = 128                # random edits used to set the term scales
    snapshot_every: int = 50
    ops: list | None = None
    seed: int = 0


class History:
    """history.jsonl: one line per proposed edit, plus snapshot and pass lines."""

    def __init__(self, path=None, snapshot_every: int = 50, terms=TERMS):
        self.path = path
        self.f = open(path, "w") if path else None
        self.events: list = []
        self.snapshot_every = snapshot_every
        self.accepted_per_layer: dict = {}

    def _write(self, rec: dict) -> None:
        self.events.append(rec)
        if self.f:
            self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def snapshot(self, it: int, k: int, L) -> None:
        self._write({"type": "snapshot", "iter": it, "layer": k,
                     "nxt": L.nxt.tolist(), "seam": sorted(int(s) for s in L.seam)})

    def edit(self, rec: dict, L_after=None) -> None:
        rec = {"type": "edit", **rec}
        self._write(rec)
        if rec["accepted"] and L_after is not None:
            k = rec["layer"]
            c = self.accepted_per_layer.get(k, 0) + 1
            self.accepted_per_layer[k] = c
            if c % self.snapshot_every == 0:
                self.snapshot(rec["iter"], k, L_after)

    def note(self, rec: dict) -> None:
        self._write(rec)

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None


def warmup_normalizer(part: PartState, n: int, rng, ops=None) -> None:
    """Set each term's scale from the changes of n random valid edits."""
    L_count = len(part.layers)
    for i in range(n):
        k = int(rng.integers(L_count))
        M = action_mask(part.layers[k], ops).reshape(-1)
        if not M.any():
            continue
        a = int(sample_valid(M, rng)[0])
        newL, _ = apply_action(part.layers[k], a)
        c = part.propose(k, newL)
        part.norm.update(c.d)
    part.norm.frozen = True


def schedule(part: PartState, recent_gain: np.ndarray) -> list[int]:
    """Highest priority first: share of the part's failure index plus recent gain.
    Every layer is still visited once per pass."""
    share = part.fi_share()
    g = recent_gain / recent_gain.sum() if recent_gain.sum() > 0 else np.zeros_like(recent_gain)
    pr = share + g
    return [int(k) for k in np.argsort(-pr, kind="stable")]


def accept(d: float, temp: float, rng) -> bool:
    if d > 1e-12:
        return True
    if temp > 0:
        return bool(rng.random() < math.exp(max(d / temp, -50.0)))
    return False


def optimize_part(part: PartState, proposer, w, cfg: LoopConfig, history: History | None = None,
                  progress=None) -> dict:
    rng = np.random.default_rng(cfg.seed)
    w = np.asarray(w, float)
    history = history or History(None)
    t_start = time.time()
    ops = ops_to_indices(cfg.ops) if cfg.ops else None

    for k, L in enumerate(part.layers):
        history.snapshot(0, k, L)

    if not part.norm.frozen:
        warmup_normalizer(part, cfg.warmup, rng, ops)
    warm_evals = part.n_evals
    part.n_evals = 0
    part.keep_best(w)
    history.note({"type": "start", "proposer": proposer.name, "weights": w.tolist(),
                  "scales": part.norm.scales(), "raw": part.raw(), "total": part.total(w),
                  "warmup_evals": warm_evals})

    temp0 = proposer.temp0 if cfg.temp0 is None else cfg.temp0
    temp = temp0
    it = 0
    recent = np.zeros(len(part.layers))
    curve = [(0, part.total(w))]
    pass_idx = 0
    stop_reason = "max_passes"
    while pass_idx < cfg.max_passes:
        pass_gain = 0.0
        layer_gain = np.zeros(len(part.layers))
        n_acc = n_prop = 0
        for k in schedule(part, recent):
            banned: set = set()
            for _ in range(cfg.burst):
                if part.n_evals >= cfg.max_evals:
                    break
                L = part.layers[k]
                M = action_mask(L, ops).reshape(-1)
                if banned:
                    M[list(banned)] = False
                if not M.any():
                    break
                a, cand = proposer.propose(part, k, M, w, rng)
                if cand is None:
                    newL, params = apply_action(L, a)
                    cand = part.propose(k, newL)
                else:
                    params = {}
                d = part.norm.total(cand.d, w)
                acc = accept(d, temp, rng)
                op, cell = divmod(a, L.n)
                rec = {"iter": it, "layer": k, "op": OPS[op], "cell": [cell // L.W, cell % L.W],
                       "params": params, "accepted": acc,
                       "d": {t: cand.d[t] for t in TERMS}, "d_total": d, "evals": part.n_evals,
                       "pass": pass_idx}
                if acc:
                    rec["diff"] = layer_diff(L, cand.layer)
                    part.commit(cand)
                    pass_gain += d
                    layer_gain[k] += max(d, 0)
                    banned.clear()
                    n_acc += 1
                else:
                    banned.add(a)
                raw = part.raw()
                tot = part.norm.total(raw, w)
                rec["total"] = tot
                rec["s"] = {t: raw[t] / part.norm.std(t) for t in TERMS}
                history.edit(rec, part.layers[k] if acc else None)
                n_prop += 1
                it += 1
                if it % 50 == 0:
                    curve.append((part.n_evals, tot))
                if progress and it % 200 == 0:
                    progress(f"iter {it} evals {part.n_evals} total {tot:.4f} temp {temp:.3f}")
            if part.n_evals >= cfg.max_evals:
                break
        drift = part.full_recheck()
        part.keep_best(w)
        recent = layer_gain
        tot = part.total(w)
        curve.append((part.n_evals, tot))
        history.note({"type": "pass", "pass": pass_idx, "iter": it, "evals": part.n_evals,
                      "gain": pass_gain, "accepted": n_acc, "proposed": n_prop, "temp": temp,
                      "total": tot, "best": part.best_total, "drift": drift, "raw": part.raw()})
        if progress:
            progress(f"pass {pass_idx}: gain {pass_gain:.4f} accepted {n_acc}/{n_prop} "
                     f"total {tot:.4f} best {part.best_total:.4f} temp {temp:.3f} drift {drift:.2e}")
        pass_idx += 1
        if part.n_evals >= cfg.max_evals:
            stop_reason = "max_evals"
            break
        if temp == 0 and pass_gain < cfg.eps:
            stop_reason = "converged"
            break
        temp = temp0 * max(0.0, 1.0 - pass_idx / max(cfg.anneal_passes, 1))
        if temp < 1e-9:
            temp = 0.0

    if part.best_snap is not None:
        part.restore(part.best_snap)
    final = part.total(w)
    history.note({"type": "end", "iter": it, "evals": part.n_evals, "total": final,
                  "raw": part.raw(), "stop": stop_reason})
    return {
        "proposer": proposer.name, "iterations": it, "evals": part.n_evals, "passes": pass_idx,
        "stop": stop_reason, "seconds": time.time() - t_start, "total": final,
        "curve": curve, "warmup_evals": warm_evals, "scales": part.norm.scales(),
    }
