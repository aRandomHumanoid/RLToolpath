"""Run independent jobs in forked worker processes.

Workers inherit the parent's memory (the part, its factorisations), so nothing large is
pickled on the way in; only each job's name goes over, and its result comes back. Where
the platform can't fork (native Windows) the jobs run one after another in this process.
Set OMP_NUM_THREADS=1 for many workers, so each one stays on one core.
"""
from __future__ import annotations

import multiprocessing as mp

SHARED: dict = {}         # what the jobs read: set for the duration of fork_map


def can_fork() -> bool:
    return "fork" in mp.get_all_start_methods()


def fork_map(fn, items, workers: int, shared: dict | None = None) -> list:
    """[fn(x) for x in items], in up to `workers` forked processes. `fn` must be a module-level
    function; it reads its inputs from SHARED."""
    global SHARED
    items = list(items)
    SHARED = dict(shared or {})
    try:
        if workers > 1 and len(items) > 1 and can_fork():
            with mp.get_context("fork").Pool(min(workers, len(items))) as pool:
                return pool.map(fn, items, chunksize=1)
        return [fn(x) for x in items]
    finally:
        SHARED = {}
