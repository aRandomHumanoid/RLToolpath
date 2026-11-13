"""Several independent annealing chains on one part; the best final layout wins.

Each chain runs the ordinary part loop (optimize_part) with its own seed, from one of
the starting layouts it is given (chain i takes start i mod n: the gradient start's best
few), in its own forked process. One warmup on the part sets the term scales and the
temperature unit for every chain, so their final scores are on one scale and the best
can be picked. Chains that land in different local optima show up as a spread in their
finals; best-of-N is the simplest guard against a bad one.

Each chain writes its history to out/chain<i>/history.jsonl; the winner's is copied to
out/history.jsonl for the viewer.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path

import numpy as np

from rltp.loop.optimize import History, LoopConfig, optimize_part, warmup_normalizer
from rltp.search.parallel import can_fork, fork_map


def _chain_job(i: int) -> dict:
    from rltp.search.parallel import SHARED
    J = SHARED
    part, cfg, out = J["part"], J["cfg"], J["out"]
    starts = J["starts"]
    if starts:
        part.restore(starts[i % len(starts)])          # also the fallback's reset between chains
    part.best_total, part.best_snap, part.best_raw = -np.inf, None, None
    part.n_evals = 0
    start_raw = part.raw()
    d = out / f"chain{i}"
    d.mkdir(parents=True, exist_ok=True)
    hist = History(d / "history.jsonl", snapshot_every=cfg.snapshot_every)
    say = J["progress"]
    prog = (lambda m: say(f"chain {i}: {m}")) if say else None
    res = optimize_part(part, J["proposer"], J["w"], dataclasses.replace(cfg, seed=cfg.seed + i), hist,
                        progress=prog, warm=J["warm"])
    hist.close()
    res = {k: v for k, v in res.items() if k != "curve"} | {"curve": res["curve"][::4]}
    out_i = {"chain": i, "seed": cfg.seed + i, "start": J["names"][i % len(J["names"])] if J["names"] else "",
             "loop": res, "total": part.total(J["w"]), "start_total": part.norm.total(start_raw, J["w"]),
             "final": part.breakdown(), "snap": part.snapshot()}
    (d / "metrics.json").write_text(json.dumps({k: v for k, v in out_i.items() if k != "snap"}, indent=1,
                                               default=_json_default))
    return out_i


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_chains(part, proposer, w, cfg: LoopConfig, n: int, out, workers: int = 0, starts=None,
               start_names=None, progress=None) -> dict:
    """n chains (seeds cfg.seed .. cfg.seed + n - 1). starts: snapshots to start from
    (default: the part as it is). Leaves `part` holding the best chain's final layout and
    returns {"best": i, "chains": [per chain: seed, start, loop, total, start_total, final]},
    totals on the part's own scale (its references and one shared warmup)."""
    out = Path(out)
    w = np.asarray(w, float)
    snap0 = part.snapshot()
    starts = list(starts) if starts else [snap0]
    rng = np.random.default_rng(cfg.seed)
    warm_d = warmup_normalizer(part, cfg.warmup, rng) if not part.norm.frozen else []
    warm = (warm_d, part.n_evals)
    workers = workers or min(n, os.cpu_count() or 1)
    if progress:
        how = f"{min(workers, n)} processes" if (workers > 1 and n > 1 and can_fork()) else "one after another"
        progress(f"{n} annealing chains from {len(starts)} start(s), {how}")
    shared = {"part": part, "proposer": proposer, "w": w, "cfg": cfg, "out": out, "starts": starts,
              "names": list(start_names or []), "warm": warm, "progress": progress}
    res = fork_map(_chain_job, range(n), workers, shared)
    best = max(range(len(res)), key=lambda j: res[j]["total"])
    part.restore(res[best]["snap"])
    part.best_total, part.best_snap, part.best_raw = res[best]["total"], res[best]["snap"], part.raw()
    src = out / f"chain{res[best]['chain']}" / "history.jsonl"
    if src.exists():
        shutil.copyfile(src, out / "history.jsonl")
    for r in res:
        r.pop("snap", None)
    return {"best": res[best]["chain"], "chains": res}


def load_events(path) -> list:
    """History events from a history.jsonl (for the viewer)."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]
