"""Annealing safeguards against local optima: reheating and parallel chains."""
import json

import numpy as np
import trimesh
from shapely.geometry import Point, box

from rltp.config import Load2D, LoadSpec, PartSpec, Printer, Region, ScoreConfig
from rltp.loop.optimize import History, LoopConfig, optimize_part
from rltp.part import build_part
from rltp.search.chains import load_events, run_chains
from rltp.search.sa import make_proposer

W = np.array([1.0, 0.3, 0.3, 0, 0, 0])


def _part():
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    m = trimesh.creation.extrude_polygon(g, height=0.6)
    loads = LoadSpec([Region(circle=[2.5, 10, 2.0])], [Load2D(Region(circle=[13, 2.5, 2.0]), (0, -60))])
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, infill_rule="free")
    return build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont"), score=ScoreConfig()),
                      baseline="zigzag")


def test_reheat_anneals_again_once_greedy_stalls():
    part = _part()
    start = part.total(W)
    hist = History(None)
    cfg = LoopConfig(max_evals=600, burst=8, warmup=32, anneal_passes=1, reheat=0.5,
                     reheat_below=1e9, reheat_after=1, seed=1)     # every greedy pass counts as stalled
    res = optimize_part(part, make_proposer("sa"), W, cfg, hist)
    assert res["reheats"] >= 1
    re = [e for e in hist.events if e["type"] == "reheat"]
    assert len(re) == res["reheats"] and all(e["temp"] > 0 for e in re)
    temps = [e["temp"] for e in hist.events if e["type"] == "pass"]
    assert 0.0 in temps and any(t > 0 for t in temps[temps.index(0.0):])   # warm again after cold
    assert part.total(W) >= start - 1e-9                                      # the best part is kept


def test_no_reheat_by_default():
    part = _part()
    res = optimize_part(part, make_proposer("sa"), W, LoopConfig(max_evals=200, burst=8, warmup=32,
                                                                 anneal_passes=1, seed=0))
    assert res["reheats"] == 0


def test_chains_keep_the_best_and_its_history(tmp_path):
    part = _part()
    cfg = LoopConfig(max_evals=120, burst=8, warmup=32, anneal_passes=2, seed=3)
    cr = run_chains(part, make_proposer("sa"), W, cfg, 3, tmp_path, workers=1)
    ch = cr["chains"]
    assert len(ch) == 3 and sorted(c["seed"] for c in ch) == [3, 4, 5]
    best = max(ch, key=lambda c: c["total"])
    assert cr["best"] == best["chain"]
    assert part.total(W) == np.float64(best["total"]) or abs(part.total(W) - best["total"]) < 1e-9
    assert all(c["total"] >= c["start_total"] - 1e-9 for c in ch)
    ev = load_events(tmp_path / "history.jsonl")
    assert ev and ev[0]["type"] == "snapshot"
    assert json.loads((tmp_path / f"chain{best['chain']}" / "metrics.json").read_text())["total"] == best["total"]


def test_chains_start_from_the_given_layouts(tmp_path):
    part = _part()
    a = part.snapshot()
    L = part.layers[1]
    # a second start: the same part with one layer's infill removed where allowed
    opt = (L.part & ~L.req)
    L.mask = L.mask & ~opt
    L.nxt[opt.reshape(-1)] = -1
    L.prv[opt.reshape(-1)] = -1
    L.nxt[(L.nxt >= 0) & ~L.mask.reshape(-1)[np.maximum(L.nxt, 0)]] = -1
    L.prv[(L.prv >= 0) & ~L.mask.reshape(-1)[np.maximum(L.prv, 0)]] = -1
    L.invalidate()
    b = part.snapshot()
    part.restore(a)
    cfg = LoopConfig(max_evals=40, burst=8, warmup=16, anneal_passes=1, seed=0)
    cr = run_chains(part, make_proposer("sa"), W, cfg, 2, tmp_path, workers=1, starts=[a, b],
                    start_names=["full", "emptied"])
    assert [c["start"] for c in sorted(cr["chains"], key=lambda c: c["chain"])] == ["full", "emptied"]
    st = {c["start"]: c["start_total"] for c in cr["chains"]}
    assert st["full"] != st["emptied"]
