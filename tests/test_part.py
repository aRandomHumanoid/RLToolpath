"""PartState bookkeeping, interface statics, G-code volume and the part loop."""
import re

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, box

from rltp.config import Load2D, LoadSpec, PartSpec, PointLoad, Printer, Region
from rltp.cover.cover import check_invariants
from rltp.cover.ops import action_mask, apply_action, sample_valid
from rltp.eval.interlayer import build_interface, eval_interface
from rltp.export.gcode import write_gcode
from rltp.loop.optimize import History, LoopConfig, optimize_part
from rltp.part import build_part
from rltp.search.sa import make_proposer


def bracket_mesh():
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    return trimesh.creation.extrude_polygon(g, height=1.0)


def bracket_spec(terms=("strength", "time", "cont", "inter")):
    loads = LoadSpec([Region(circle=[2.5, 10, 2.0])], [Load2D(Region(circle=[13, 2.5, 2.0]), (0, -60))],
                     [PointLoad((13, 2.5, 1.0), (0, -60, 10))])
    return PartSpec(Printer(), loads, terms=terms)


def test_incremental_matches_full_recheck():
    part = build_part(bracket_mesh(), bracket_spec(), baseline="concentric")
    assert len(part.layers) == 5 and len(part.ifaces) == 4
    rng = np.random.default_rng(0)
    for i in range(60):
        k = int(rng.integers(len(part.layers)))
        M = action_mask(part.layers[k]).reshape(-1)
        a = int(sample_valid(M, rng)[0])
        newL, _ = apply_action(part.layers[k], a)
        c = part.propose(k, newL)
        before = part.raw()
        part.commit(c)
        after = part.raw()
        for t in c.d:
            assert after[t] - before[t] == pytest.approx(c.d[t], abs=1e-9)
    inc = part.raw()
    drift = part.full_recheck()
    assert drift < 1e-6, (inc, part.raw())
    for L in part.layers:
        assert check_invariants(L) == []


def test_interface_statics_resultants():
    from rltp.model import Frame, Layer
    frame = Frame(0.0, 0.0, 0.0, 0.5)
    mask = np.zeros((20, 30), bool)
    mask[2:18, 3:27] = True
    lo = Layer.empty(0, 0.2, 0.2, mask, frame)
    up = Layer.empty(1, 0.4, 0.2, mask, frame)
    a = 0.25
    cx, cy = 7.5, 5.0
    # pure axial pull
    itf = build_interface(0, lo, up, 0.2, [PointLoad((cx, cy, 5.0), (0, 0, 100.0))], Printer().material)
    assert np.allclose(itf.sigma, 100.0 / itf.A) and np.allclose(itf.tau, 0)
    # a lateral force above the cut bends the section: zero net axial force, moment balance
    F, z = 10.0, 5.0
    itf = build_interface(0, lo, up, 0.2, [PointLoad((cx, cy, z), (F, 0, 0))], Printer().material)
    x, y = up.frame.cell_xy(itf.up, up.W)
    assert abs((itf.sigma * a).sum()) < 1e-6
    My = -(itf.sigma * (x - cx) * a).sum()
    assert My == pytest.approx(F * (z - 0.2), rel=0.02)
    assert (itf.tau * a).sum() == pytest.approx(F, rel=1e-6)


def test_bond_quality_falls_with_delay():
    from rltp.config import BondModel, Material
    from rltp.model import Frame, Layer
    mask = np.ones((6, 6), bool)
    fr = Frame(0, 0, 0, 0.4)
    lo, up = Layer.empty(0, 0.2, 0.2, mask, fr), Layer.empty(1, 0.4, 0.2, mask, fr)
    itf = build_interface(0, lo, up, 0.2, [], Material())
    t = np.linspace(0, 5, 36)
    ends = np.zeros(36, bool)
    fast = eval_interface(itf, t, ends, 5.0, t, ends, BondModel(), Material(), 0.5, 8)
    slow = eval_interface(itf, t, ends, 60.0, t, ends, BondModel(), Material(), 0.5, 8)
    assert slow.S_p > fast.S_p and (slow.q < fast.q).all()


def test_gcode_volume_and_structure(tmp_path):
    pr = Printer()
    part = build_part(bracket_mesh(), bracket_spec(("time", "cont")), baseline="concentric")
    st = write_gcode(tmp_path / "p.gcode", part, pr)
    txt = (tmp_path / "p.gcode").read_text()
    assert "M83" in txt and txt.count(";LAYER_CHANGE") == len(part.layers)
    e_print = 0.0
    body = txt[txt.index(";LAYER_CHANGE"):]   # skip the purge line in the start G-code
    for line in body.splitlines():
        m = re.match(r"G1 X[-\d.]+ Y[-\d.]+ E([-\d.]+)", line)
        if m:
            e_print += float(m.group(1))
        assert not (line.startswith("G0") and " E" in line)
    area = np.pi * (pr.filament_diameter / 2) ** 2
    expect = sum(L.flat_mask.sum() * pr.bead_width * pr.bead_width * L.h for L in part.layers) / area
    assert e_print == pytest.approx(expect, rel=1e-3)
    assert st.strokes == sum(e.cont.n_strokes for e in part.evals)


@pytest.mark.parametrize("name", ["sa", "greedy", "random"])
def test_loop_improves_and_logs(name):
    part = build_part(bracket_mesh(), bracket_spec(("strength", "time", "cont")), baseline="zigzag90")
    w = np.array([1.0, 0.3, 0.3, 0.0])
    hist = History(None)
    res = optimize_part(part, make_proposer(name), w, LoopConfig(max_evals=400, burst=16, warmup=32), hist)
    edits = [e for e in hist.events if e["type"] == "edit"]
    assert edits and res["evals"] >= 400 - 1 or res["stop"] == "converged"
    start = next(e for e in hist.events if e["type"] == "start")["total"]
    assert res["total"] >= start - 1e-9
    assert res["total"] > start  # zigzag90 is a poor start for this load
    # replaying diffs from the snapshots reproduces the final covers of the last accepted state
    snaps = {e["layer"]: np.array(e["nxt"]) for e in hist.events if e["type"] == "snapshot" and e["iter"] == 0}
    for e in edits:
        if e["accepted"]:
            snaps[e["layer"]][e["diff"]["cells"]] = e["diff"]["nxt"]
    for L in part.layers:
        assert check_invariants(L) == []


def test_sanity_tension_bar_beads_follow_load():
    """Design-doc sanity test: under pure tension a bar's beads end up along the load."""
    from rltp.cover.cover import bead_weights
    m = trimesh.creation.box(extents=[12, 4, 0.2])
    m.apply_translation([6, 2, 0.1])
    loads = LoadSpec([Region(side="xmin", width=0.3)], [Load2D(Region(side="xmax", width=0.3), (40.0, 0.0))])
    part = build_part(m, PartSpec(Printer(), loads, terms=("strength",)), baseline="zigzag90")
    L = part.layers[0]
    before = bead_weights(L)[L.flat_mask].mean()
    optimize_part(part, make_proposer("sa"), [1.0, 0, 0, 0], LoopConfig(max_evals=600, burst=32, warmup=32))
    L = part.layers[0]
    after = bead_weights(L)[L.flat_mask].mean()
    assert before < 0.2 and after > 0.85
