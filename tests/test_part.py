"""PartState bookkeeping, interface statics, G-code volume and the part loop."""
import re

import numpy as np
import pytest
import trimesh
from shapely.geometry import LineString, Point, box

from rltp.config import Load2D, LoadSpec, PartSpec, PointLoad, Printer, Region
from rltp.cover.cover import check_invariants, get_pieces
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
    # fi "ratio": the inverse strength ratio, the square root of the quadratic index
    ratio = eval_interface(itf, t, ends, 60.0, t, ends, BondModel(fi="ratio"), Material(), 0.5, 8)
    assert np.allclose(ratio.fi, np.sqrt(slow.fi)) and np.allclose(ratio.q, slow.q)


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


class _FirstValid:
    """Deterministic proposer: always the lowest-index allowed action."""
    name = "first-valid"
    temp0 = 0.0

    def propose(self, part, k, M, w, rng):
        return int(np.flatnonzero(M)[0]), None


def _small_bar_part():
    m = trimesh.creation.box(extents=[4, 2, 0.2])
    m.apply_translation([2, 1, 0.1])
    loads = LoadSpec([Region(side="xmin", width=0.3)], [Load2D(Region(side="xmax", width=0.3), (20.0, 0.0))])
    return build_part(m, PartSpec(Printer(), loads, terms=("strength",)), baseline="zigzag90")


def test_loop_bans_persist_until_the_layer_changes():
    """A deterministic proposer never re-proposes a rejected edit while its layer is
    unchanged, and the loop stops once no untried edit is left: a local optimum."""
    part = _small_bar_part()
    w = np.array([1.0, 0, 0, 0])
    hist = History(None)
    cfg = LoopConfig(burst=8, eps=-np.inf, max_passes=10 ** 6, max_evals=20000, warmup=16)
    res = optimize_part(part, _FirstValid(), w, cfg, hist)
    assert res["stop"] == "exhausted"
    since_accept: set = set()
    for e in (e for e in hist.events if e["type"] == "edit"):
        key = (e["op"], tuple(e["cell"]))
        assert key not in since_accept
        since_accept = set() if e["accepted"] else since_accept | {key}
    # nothing single-step improves the final cover
    L = part.layers[0]
    M = action_mask(L).reshape(-1)
    for a in np.flatnonzero(M):
        newL, _ = apply_action(L, int(a))
        assert part.norm.total(part.propose(0, newL).d, w) <= 1e-12


def test_loop_burst_bans_repeat_rejected_edits():
    """The original per-burst bans: a deterministic proposer stuck at a local optimum
    replays the same rejected edits every burst until the budget runs out."""
    part = _small_bar_part()
    w = np.array([1.0, 0, 0, 0])
    hist = History(None)
    cfg = LoopConfig(burst=8, eps=-np.inf, max_passes=10 ** 6, max_evals=3000, warmup=16, persist_bans=False)
    res = optimize_part(part, _FirstValid(), w, cfg, hist)
    assert res["stop"] == "max_evals"
    rejected = [(e["op"], tuple(e["cell"])) for e in hist.events if e["type"] == "edit" and not e["accepted"]]
    assert len(rejected) > len(set(rejected))


# ------------------------------------------------------------------ load cases

def _case_spec(**kw):
    s = bracket_spec(("strength", "time", "cont"))
    for k, v in kw.items():
        setattr(s.loads, k, v)
    return s


def _random_edits(parts, n, seed=0):
    """Apply the same random edits to identical parts; yield each part's candidate."""
    from rltp.cover.ops import InvalidEdit
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = int(rng.integers(len(parts[0].layers)))
        a = int(sample_valid(action_mask(parts[0].layers[k]).reshape(-1), rng)[0])
        try:
            news = [apply_action(p.layers[k], a)[0] for p in parts]
        except InvalidEdit:
            continue
        cands = [p.propose(k, nl) for p, nl in zip(parts, news)]
        for p, c in zip(parts, cands):
            p.commit(c)
        yield cands


def test_load_cases_start_at_one_and_keep_the_primary_case():
    """Each case is scored against its own baseline, so the combined FI_agg starts at 1;
    case 0 is the loads.yaml case, unchanged."""
    single = build_part(bracket_mesh(), _case_spec(), baseline="zigzag")
    multi = build_part(bracket_mesh(), _case_spec(random_cases=3), baseline="zigzag")
    assert multi.spec.loads.n_cases == 4 and "case_fi_agg" not in single.breakdown()
    b = multi.breakdown()
    assert b["fi_agg"] == pytest.approx(1.0, abs=1e-12)
    assert b["raw"]["strength"] == pytest.approx(0.0, abs=1e-12)
    assert len(b["case_fi_agg"]) == 4
    assert b["case_fi_agg"][0] == pytest.approx(single.breakdown()["fi_agg"], rel=1e-9)


def test_load_cases_incremental_matches_full_recheck():
    part = build_part(bracket_mesh(), _case_spec(random_cases=3), baseline="concentric")
    for _ in _random_edits([part], 40):
        pass
    drift = part.full_recheck()
    assert drift < 1e-6


def test_a_scaled_copy_of_a_case_changes_nothing():
    """Force sizes don't matter: adding a 10x copy of the primary load as a second case
    leaves every edit's strength change as it was with the primary case alone."""
    single = build_part(bracket_mesh(), _case_spec(), baseline="zigzag")
    big = [Load2D(Region(circle=[13, 2.5, 2.0]), (0, -600))]
    double = build_part(bracket_mesh(), _case_spec(cases=[big]), baseline="zigzag")
    assert double.spec.loads.n_cases == 2
    n = 0
    for c1, c2 in _random_edits([single, double], 40):
        assert c2.d["strength"] == pytest.approx(c1.d["strength"], rel=1e-6, abs=1e-12)
        n += abs(c1.d["strength"]) > 0
    assert n > 10


def test_random_cases_are_seeded_and_clear_of_supports():
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))

    def make(seed):
        loads = LoadSpec.from_dict({"supports": [{"circle": [2.5, 10, 2.0]}],
                                    "loads": [{"region": {"circle": [13, 2.5, 2.0]}, "force": [0, -60]}],
                                    "load_cases": [{"region": {"side": "xmax", "width": 0.5}, "force": [5, 0]}],
                                    "random_cases": 4, "random_seed": seed, "random_mode": "disc"})
        loads.resolve(g.bounds)
        loads.add_random_cases(g, 0.4, g.bounds)
        return loads

    a, b, c = make(3), make(3), make(4)
    assert a.n_cases == 6 and a.has_2d
    assert a.describe_cases() == b.describe_cases() != c.describe_cases()
    sup = a.supports[0]._geom
    for case in a.generated:
        x, y, r = case[0].region.circle
        assert g.boundary.distance(Point(x, y)) < 1e-6           # on the outline
        assert sup.distance(Point(x, y)) > r                      # off the support
        assert np.hypot(*case[0].force) == pytest.approx(60.0)    # primary's total force


def test_outline_edges_split_at_corners_and_tangent_arcs():
    """Sides, arcs and round holes come out as CAD would have them, even when an arc meets a
    side tangentially (a stadium) and the slicer's union leaves extra collinear points."""
    from shapely.geometry import LineString
    from rltp.slicing.edges import _fits_line, outline_edges
    L = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    E = outline_edges(L.union(box(0, 0, 3, 3)))             # the union adds collinear points
    closed = [e for e in E if np.allclose(e[0], e[-1])]
    sides = [e for e in E if not np.allclose(e[0], e[-1])]
    assert len(E) == 7 and len(closed) == 1
    assert sorted(round(LineString(e).length, 6) for e in sides) == [5, 5, 7, 11, 12, 16]
    assert LineString(closed[0]).length == pytest.approx(2 * np.pi * 1.2, rel=1e-2)
    stadium = LineString([(0, 0), (10, 0)]).buffer(3).difference(Point(8, 0).buffer(1))
    E = outline_edges(stadium)
    lines = [e for e in E if _fits_line(e, 1e-6)]
    assert len(E) == 5 and len(lines) == 2
    assert all(LineString(e).length == pytest.approx(10) for e in lines)


def _edge_loads(n, seed=0, supports=True, moments=0):
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    d = {"loads": [{"region": {"circle": [13, 2.5, 2.0]}, "force": [0, -60]}],
         "random_cases": n, "random_moments": moments, "random_seed": seed}
    if supports:
        d["supports"] = [{"circle": [2.5, 10, 2.0]}]
    loads = LoadSpec.from_dict(d)
    loads.resolve(g.bounds)
    loads.add_random_cases(g, 0.4, g.bounds)
    return g, loads


def test_random_edge_cases_fix_one_edge_and_load_another():
    from shapely.geometry import LineString
    from rltp.slicing.edges import outline_edges
    g, a = _edge_loads(8, seed=3)
    assert a.random_mode == "edge" and a.n_cases == 9 and a.has_2d
    edges = [np.round(e, 4).tolist() for e in outline_edges(g)]
    sups = a.case_supports
    assert sups[0] is a.supports                                   # the primary keeps loads.yaml's
    for case, sup in zip(a.all_cases[1:], sups[1:]):
        (ld,) = case
        (s,) = sup
        assert ld.region.edge in edges and s.edge in edges and ld.region.edge != s.edge
        assert LineString(ld.region.edge).length >= 0.8 and ld.region.width == 0.4
        assert np.hypot(*ld.force) == pytest.approx(60.0)
    d = a.describe_cases()
    assert "supports" not in d[0][0] and all("edge" in c[0]["supports"][0] for c in d[1:])
    assert d == _edge_loads(8, seed=3)[1].describe_cases() != _edge_loads(8, seed=4)[1].describe_cases()
    # no loads.yaml supports needed: every random case brings its own
    assert _edge_loads(2, supports=False)[1].has_2d
    assert _edge_loads(0, supports=False, moments=1)[1].has_2d
    with pytest.raises(ValueError, match="two outline edges"):
        LoadSpec(random_cases=1).add_random_cases(Point(0, 0).buffer(3), 0.4, (-3, -3, 3, 3))


def test_without_a_primary_every_case_is_random():
    """Load-agnostic: no loads.yaml loads, so the cases are only the random ones, each with its
    own supports; the combined FI_agg still starts at 1."""
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    loads = LoadSpec.from_dict({"random_cases": 3, "random_moments": 2, "random_seed": 1})
    assert loads.has_primary and loads.n_cases == 1                # nothing drawn yet: the old empty case 0
    loads.resolve(g.bounds)
    loads.add_random_cases(g, 0.4, g.bounds)
    assert not loads.has_primary and loads.n_cases == 5
    assert len(loads.case_supports) == 5 and all(s is not loads.supports for s in loads.case_supports)
    assert all("supports" in c[0] for c in loads.describe_cases())
    loads = LoadSpec.from_dict({"random_cases": 3, "random_moments": 2, "random_seed": 1})
    part = build_part(bracket_mesh(), PartSpec(Printer(), loads, terms=("strength", "time", "cont")),
                      baseline="zigzag")
    b = part.breakdown()
    assert len(b["case_fi_agg"]) == 5 and b["fi_agg"] == pytest.approx(1.0, abs=1e-12)
    for _ in _random_edits([part], 20):
        pass
    assert part.full_recheck() < 1e-6


def test_edge_kinds():
    from rltp.slicing.edges import edge_kind, outline_edges
    link = (LineString([(0, 0), (30, 0)]).buffer(5).difference(Point(0, 0).buffer(2))
            .difference(Point(30, 0).buffer(2)))
    kinds = sorted(edge_kind(e, link) for e in outline_edges(link) if LineString(e).length >= 0.8)
    assert kinds == ["arc", "arc", "hole", "hole", "side", "side"]       # (as the random cases see them)
    disc = Point(0, 0).buffer(3)
    assert [edge_kind(e, disc) for e in outline_edges(disc)] == ["arc"]      # a round outline isn't a hole


def _drawn(g, weights, n=400, seed=0):
    from shapely.geometry import LineString
    from rltp.slicing.edges import edge_kind
    loads = LoadSpec(random_cases=n, random_seed=seed, random_weights=weights)
    loads.resolve(g.bounds)
    loads.add_random_cases(g, 0.4, g.bounds)
    out = []
    for case, (sup,) in zip(loads.generated, loads.generated_supports):
        e_s, e_l = np.asarray(sup.edge), np.asarray(case[0].region.edge)
        out.append((edge_kind(e_s, g), LineString(e_s).length, edge_kind(e_l, g), LineString(e_l).length))
    return out


def test_random_edges_follow_kind_odds_then_length():
    """An edge's kind is drawn with odds random_weights over the kinds the part has, then an
    edge of that kind by length; the load's edge the same way from the other edges."""
    link = (LineString([(0, 0), (30, 0)]).buffer(5).difference(Point(0, 0).buffer(2))
            .difference(Point(30, 0).buffer(2)))
    d = _drawn(link, (3.0, 2.0, 1.0), n=1500)
    share = {k: np.mean([x[0] == k for x in d]) for k in ("hole", "side", "arc")}
    assert share == pytest.approx({"hole": 0.5, "side": 1 / 3, "arc": 1 / 6}, abs=0.04)
    assert all(x[2] != "arc" for x in _drawn(link, (1.0, 1.0, 0.0)))
    assert all(x[0] == x[2] == "arc" for x in _drawn(link, (0.0, 0.0, 1.0), n=50))
    plate = box(0, 0, 20, 5).difference(Point(10, 2.5).buffer(1.0))   # sides 20, 5, 20, 5
    d = _drawn(plate, (0.0, 1.0, 0.0), n=1500)
    assert np.mean([x[1] > 10 for x in d]) == pytest.approx(0.8, abs=0.04)
    with pytest.raises(ValueError, match="leave no edge"):
        _drawn(plate, (1.0, 0.0, 0.0), n=1)                          # one hole: nothing left to load


def test_uniform_load_weights_draw_as_before():
    """random_weights None draws every edge alike, with the same random numbers as runs before
    the kind weighting (Nov 15, 2026)."""
    from rltp.slicing.edges import outline_edges
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    loads = LoadSpec.from_dict({"random_cases": 5, "random_moments": 3, "random_seed": 7,
                                "random_weights": "uniform"})
    assert loads.random_weights is None
    loads.resolve(g.bounds)
    loads.add_random_cases(g, 0.4, g.bounds)
    edges = [np.round(e, 4).tolist() for e in outline_edges(g) if LineString(e).length >= 0.8]
    rng = np.random.default_rng(7)
    want = []
    for i in range(8):
        s = int(rng.integers(len(edges)))
        ld = int(rng.integers(len(edges) - 1))
        want.append((edges[s], edges[ld + (ld >= s)]))
        rng.uniform(0, 2 * np.pi) if i < 5 else rng.random()
    got = [(sup[0].edge, case[0].region.edge) for case, sup in zip(loads.generated, loads.generated_supports)]
    assert got == want


def test_consensus_stress_map_follows_free_edges():
    """Without a primary case the stress map is the cases' consensus: along a straight side that
    no case holds or loads, every case's stress runs along the edge, and so does the map."""
    from rltp.eval.stressmap import stress_field
    from rltp.slicing.raster import make_frame, rasterize
    link = (LineString([(0, 0), (30, 0)]).buffer(5).difference(Point(0, 0).buffer(2))
            .difference(Point(30, 0).buffer(2)))
    loads = LoadSpec(random_cases=6, random_moments=4, random_weights=(1.0, 0.0, 0.0))   # hole to hole
    loads.resolve(link.bounds)
    loads.add_random_cases(link, 0.4, link.bounds)
    frame, H, W = make_frame(link, 0.0, 0.4)
    mask = rasterize(link, frame, H, W, 3)[0]
    f = stress_field(mask, frame, PartSpec(Printer(), loads))
    th = []
    for j in range(W // 3, 2 * W // 3):
        rows = np.flatnonzero(mask[:, j])
        th += [f["theta"][rows[0], j], f["theta"][rows[-1], j]]
    off = np.degrees(np.abs((np.asarray(th) + np.pi / 2) % np.pi - np.pi / 2))     # from the x axis
    assert np.median(off) < 10
    assert f["mag"][mask].min() >= 0


def test_random_moment_cases_are_pure_couples():
    """A moment case puts no net force on its edge, and its forces turn about the loaded
    nodes' centroid by the case's moment."""
    from rltp.config import Material
    from rltp.eval.fea2d import FEAProblem
    from rltp.slicing.raster import make_frame, rasterize
    g, loads = _edge_loads(2, seed=5, moments=4)
    assert loads.n_cases == 7
    mom = [c[0] for c in loads.all_cases[3:]]
    assert all(ld.force == (0.0, 0.0) and abs(ld.moment) == pytest.approx(600.0) for ld in mom)
    assert all("moment" in c[0] for c in loads.describe_cases()[3:])
    frame, H, W = make_frame(g, 0.0, 0.4)
    fp = FEAProblem(rasterize(g, frame, H, W, 3)[0], frame, 0.2, Material(), loads, 0.5)
    nx, ny = frame.node_xy(H, W)
    nx, ny = nx[fp.used_nodes], ny[fp.used_nodes]
    for c in range(3, 7):
        fx, fy = fp.f[0::2, c], fp.f[1::2, c]
        on = (fx != 0) | (fy != 0)
        assert on.sum() > 2 and abs(fx.sum()) < 1e-9 and abs(fy.sum()) < 1e-9
        cx, cy = nx[on].mean(), ny[on].mean()
        assert ((nx - cx) * fy - (ny - cy) * fx).sum() == pytest.approx(0.5 * mom[c - 3].moment, rel=1e-9)


def test_edge_cases_solve_with_their_own_supports():
    """Each random edge case gets exactly the answer of a one-case problem with its own
    supports and load, while the primary case keeps the shared supports."""
    from rltp.config import Material
    from rltp.eval.fea2d import FEAProblem
    from rltp.slicing.raster import make_frame, rasterize
    g, loads = _edge_loads(4, seed=1)
    frame, H, W = make_frame(g, 0.0, 0.4)
    mask = rasterize(g, frame, H, W, 3)[0]
    wh = np.random.default_rng(0).random(H * W)
    multi = FEAProblem(mask, frame, 0.2, Material(), loads, 1.0)
    assert len(multi.systems) == len({tuple(map(id, s)) for s in loads.case_supports})
    rm = multi.solve(wh, direct=True)
    nfix = []
    for c, (case, sups) in enumerate(zip(loads.all_cases, loads.case_supports)):
        one = FEAProblem(mask, frame, 0.2, Material(), LoadSpec(sups, case), 1.0)
        r1 = one.solve(wh, direct=True)
        assert np.allclose(rm.u[:, c], r1.u, rtol=1e-9, atol=1e-12 * np.abs(r1.u).max())
        assert rm.S_pc[c] == pytest.approx(r1.S_p, rel=1e-9)
        fixed = np.repeat(one.support_nodes, 2)
        assert np.abs(rm.u[fixed, c]).max() == 0 and np.abs(rm.u[~fixed, c]).max() > 0
        nfix.append(fixed)
    assert all((f != nfix[0]).any() for f in nfix[1:])              # every case is held elsewhere


def test_edge_cases_incremental_matches_full_recheck():
    """Dense (one system per layer and set of supports) and sparse (one laminate)."""
    part = build_part(bracket_mesh(), _case_spec(random_cases=3), baseline="concentric")
    assert len(part.fea(0).systems) > 1
    for _ in _random_edits([part], 30):
        pass
    assert part.full_recheck() < 1e-6
    sparse = _sparse_part(loads=_edge_loads(3, moments=2)[1])
    assert sparse.laminate and len(sparse.lam.systems) > 1
    for _ in _random_edits([sparse], 30):
        pass
    assert sparse.full_recheck() < 1e-6


def test_clamp_band_leaves_each_cases_own_supports_and_loads_out():
    """Cells near a case's own support and load regions drop out of that case's FI only;
    edits still score incrementally."""
    from rltp.config import Material, ScoreConfig
    from rltp.eval.fea2d import FEAProblem
    from rltp.slicing.raster import make_frame, rasterize
    g, loads = _edge_loads(3, seed=2, moments=1)
    frame, H, W = make_frame(g, 0.0, 0.4)
    mask = rasterize(g, frame, H, W, 3)[0]
    wh = np.full(H * W, 0.5)
    a = FEAProblem(mask, frame, 0.2, Material(), loads, 1.0)
    b = FEAProblem(mask, frame, 0.2, Material(), loads, 1.0, clamp_band=1.0)
    assert a.keep is None and b.keep.shape == (a.cells.size, loads.n_cases)
    ra, rb = a.solve(wh, direct=True), b.solve(wh, direct=True)
    assert (rb.S_pc < ra.S_pc).all() and not b.keep.all(0).any() and b.keep.any(0).all()
    # a case's kept cells are exactly those beyond 1 mm of its own regions (to a cell)
    nx, ny = frame.to_world((a.cells % W + 0.5) * 0.4, (a.cells // W + 0.5) * 0.4)
    from shapely import distance, points, union_all
    for c, (case, sups) in enumerate(zip(loads.all_cases, loads.case_supports)):
        d = distance(union_all([r._geom for r in sups] + [ld.region._geom for ld in case]), points(nx, ny))
        assert (b.keep[d > 1.0 + 0.6, c]).all() and not b.keep[d < 1.0 - 0.6, c].any()
    part = build_part(bracket_mesh(), _case_spec(random_cases=2, random_moments=1), baseline="concentric")
    part.spec.score.clamp_band = 1.0
    part._fea.clear()
    part.full_recheck()
    for _ in _random_edits([part], 25):
        pass
    assert part.full_recheck() < 1e-6


def test_stiffness_term_rewards_beads_along_the_load():
    """stiff = -log(mean over cases of compliance / start compliance): 0 at the start, scored
    incrementally, higher with the beads along a pull than across it."""
    from rltp.eval.fea2d import FEAProblem
    from rltp.model import Frame
    terms = ("strength", "time", "cont", "stiff")
    part = build_part(bracket_mesh(), _case_spec(random_cases=2, random_moments=1), baseline="concentric")
    part.terms = terms
    assert part.raw()["stiff"] == pytest.approx(0.0, abs=1e-12) and part.comp0.shape == (4,)
    for cands in _random_edits([part], 30):
        pass
    after = part.raw()["stiff"]
    part.full_recheck()
    assert part.raw()["stiff"] == pytest.approx(after, abs=1e-9)
    from rltp.config import Material
    H, W = 10, 30
    g = box(0, 0, 3.0, 1.0)
    loads = LoadSpec([Region(side="xmin", width=1e-3)], [Load2D(Region(side="xmax", width=1e-3), (5.0, 0))]).resolve(g.bounds)
    fp = FEAProblem(np.ones((H, W), bool), Frame(0, 0, 0, 0.1), 0.2, Material(), loads, 1.0)
    along, across = (fp.solve(np.full(H * W, w_), direct=True).comp[0] for w_ in (1.0, 0.0))
    assert along < across * 0.9


def test_starts_share_the_slicer_layouts_case_scales():
    """With several load cases, a stress start is scored with the slicer layout's case
    scales and stiffness reference, so its FI_agg compares with the slicer start's (not 1)."""
    from rltp.config import ScoreConfig
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    m = trimesh.creation.extrude_polygon(g, height=1.2)
    parts = {}
    for start in ("slicer", "stress"):
        pr = Printer(walls=2, solid_layers=1, infill=0.3, max_bridge=2.0, infill_rule="free", start=start)
        loads = _edge_loads(2, moments=1)[1]
        parts[start] = build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "stiff"),
                                              score=ScoreConfig()), baseline="zigzag")
    a, b = parts["slicer"], parts["stress"]
    assert np.allclose(a.case_scale, b.case_scale) and np.allclose(a.comp0, b.comp0)
    assert a.breakdown()["fi_agg"] == pytest.approx(1.0) and b.breakdown()["fi_agg"] != pytest.approx(1.0, abs=1e-6)


def test_bulk_term_scores_the_p2_aggregate_incrementally():
    """bulk = -ln of the p = 2 aggregate (each case scaled to its start): 0 at the start with
    several cases, scored incrementally, and lower with the beads across a pull."""
    part = build_part(bracket_mesh(), _case_spec(random_cases=2, random_moments=1), baseline="concentric")
    part.terms = ("strength", "time", "cont", "bulk")
    assert part.raw()["bulk"] == pytest.approx(0.0, abs=1e-9)
    ds = 0.0
    for (c,) in _random_edits([part], 30):
        ds += c.d["bulk"] if c is not None else 0
    after = part.raw()["bulk"]
    part.full_recheck()
    assert part.raw()["bulk"] == pytest.approx(after, abs=1e-9) and after != 0
    from rltp.config import Material
    from rltp.eval.fea2d import FEAProblem
    from rltp.model import Frame
    H, W = 10, 30
    g = box(0, 0, 3.0, 1.0)
    loads = LoadSpec([Region(side="xmin", width=1e-3)], [Load2D(Region(side="xmax", width=1e-3), (5.0, 0))]).resolve(g.bounds)
    fp = FEAProblem(np.ones((H, W), bool), Frame(0, 0, 0, 0.1), 0.2, Material(), loads, 1.0)
    along, across = (fp.solve(np.full(H * W, w_), direct=True).S_p2 for w_ in (1.0, 0.0))
    assert along < 0.6 * across


def test_direction_guide_turns_windows_toward_their_best_direction():
    """The direction map's regret is zero off the bead cells and where a cell already lies
    in its best direction; a guided action is a refill turned to its window's best direction,
    and a guided run keeps scoring incrementally."""
    from rltp.config import ScoreConfig
    from rltp.cover.ops import OPS, action_mask
    from rltp.search.guide import DEG, DirectionMap, guided_action
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    g = box(0, 0, 8, 3).union(box(0, 0, 3, 7))
    m = trimesh.creation.extrude_polygon(g, height=0.6)
    loads = LoadSpec([Region(box=[0, 6, 3, 7])], [Load2D(Region(box=[7, 0, 8, 3]), (0, -20))])
    part = build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover"),
                                  score=ScoreConfig(fea_coarsen=2)), baseline="zigzag")
    gm = DirectionMap(part)
    best, regret, mdir = gm.layer(1)
    assert (regret >= 0).all() and (regret[best < 0] == 0).all() and regret.max() > 0
    rng = np.random.default_rng(0)
    M = action_mask(part.layers[1]).reshape(-1)
    seen = 0
    for _ in range(20):
        a = guided_action(part, 1, M, rng, gm)
        if a is None:
            continue
        op, cell = divmod(a, part.layers[1].n)
        assert OPS[op].startswith("refill") and int(OPS[op].split("_")[1]) in DEG
        seen += 1
    assert seen >= 10
    # align_region: the connected area sharing the cell's best direction, re-laid that way
    from rltp.cover.ops import InvalidEdit, OP_INDEX, apply
    L = part.layers[1]
    done = 0
    for _ in range(12):
        a = guided_action(part, 1, M, rng, gm, align=1.0)
        op, cell = divmod(a, L.n)
        assert OPS[op] == "align_region"
        try:
            N, params = apply(L, op, cell)
        except InvalidEdit:
            continue
        assert check_invariants(N) == [] and params["to"] == DEG[best[cell]]
        done += 1
    assert done >= 3
    cfg = LoopConfig(max_evals=60, burst=8, warmup=8, guide=1.0, align=0.5)
    optimize_part(part, make_proposer("sa"), np.array([1.0, 0.3, 0.3, 0, 0, 1.0]), cfg)
    assert part.full_recheck() < 1e-6


def test_packed_field_keeps_the_share_and_follows_worth():
    from rltp.cover.beads import packed_field
    sel = np.ones((10, 20), bool)
    sel[:, :2] = False
    worth = np.tile(np.linspace(0.0, 1.0, 20), (10, 1))
    f = packed_field(worth, sel, 0.4)
    assert f[sel].mean() == pytest.approx(0.4, abs=1e-6)
    assert (f[~sel] == 0).all() and f.max() <= 1.0
    assert (np.diff(f[0, 2:]) >= -1e-12).all() and f[0, -1] > 2 * f[0, 3]
    assert (packed_field(np.ones_like(worth), sel, 0.4)[sel] == 0.4).all()


def test_pack_gathers_a_windows_infill_where_it_is_worth_most():
    """No pack edit is valid without a worth map (runs without --pack are unchanged). With
    one, pack re-lays a window at about its infill share with the material moved toward the
    valuable side, where a refill spaces it evenly; the worth map is >= 0 and only on optional
    cells, and a run with pack keeps scoring incrementally."""
    from rltp.config import ScoreConfig
    from rltp.cover.ops import OP_INDEX, OPS, apply
    from rltp.search.fields import FieldModel, layout_fields
    from rltp.search.guide import WorthMap
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    m = trimesh.creation.extrude_polygon(box(0, 0, 12, 6), height=0.6)
    loads = LoadSpec([Region(box=[0, 0, 1, 6])], [Load2D(Region(box=[11, 0, 12, 6]), (0, -20))])
    part = build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover"),
                                  score=ScoreConfig(fea_coarsen=2)), baseline="zigzag")
    L = part.layers[1]
    assert L.part is not None and L.kb > 1
    packs = [i for i, nm in enumerate(OPS) if nm.startswith("pack")]
    assert len(packs) == 6 and not action_mask(L)[packs].any()
    # fields read off the layout: the infill share on the optional cells, the model near the evaluator
    fm = FieldModel(part)
    th, rho = layout_fields(part, fm)
    on = fm.rho_cells
    assert rho[on].mean() == pytest.approx(0.4, abs=0.06) and (rho >= 0).all() and (rho <= 1).all()
    assert fm.evaluate(th, rho, grad=False).fi_agg == pytest.approx(part.breakdown()["fi_agg"], rel=0.25)
    # worth rising with the row: pack moves the window's lines toward high rows, refill doesn't
    rows = np.mgrid[0:L.H, 0:L.W][0].astype(float)
    L.ctx["worth"] = rows
    assert action_mask(L)[packs].any()
    cell = (L.H // 2) * L.W + L.W // 2
    r0, c0 = divmod(cell, L.W)
    win = np.zeros((L.H, L.W), bool)
    win[max(r0 - 32, 0):r0 + 32, max(c0 - 32, 0):c0 + 32] = True
    opt = win & L.part & ~L.req

    def share_and_row(N):
        mat = N.material & opt
        return mat.sum() / opt.sum(), rows[mat].mean()

    s0, _ = share_and_row(L)
    Np, _ = apply(L, OP_INDEX["pack64_0"], cell)
    Nr, _ = apply(L, OP_INDEX["refill64_0"], cell)
    assert check_invariants(Np) == []
    (sp, rp), (sr, rr_) = share_and_row(Np), share_and_row(Nr)
    assert abs(sp - s0) < 0.08 and abs(sr - s0) < 0.08
    assert rp > rr_ + 3
    del L.ctx["worth"]
    # the worth map, and a run with pack
    wm = WorthMap(part, w=np.array([1.0, 0.3, 0.3, 0, 0, 1.0]))
    gw = wm.grid()
    optc = np.zeros(gw.size, bool)
    for Lk in part.layers:
        if Lk.part is not None:
            optc |= (Lk.part & ~Lk.req).reshape(-1)
    assert (gw >= 0).all() and gw.max() > 0 and (gw.reshape(-1)[~optc] == 0).all()
    hist = History(None)
    cfg = LoopConfig(max_evals=80, burst=8, warmup=8, guide=1.0, pack=0.5)
    optimize_part(part, make_proposer("sa"), np.array([1.0, 0.3, 0.3, 0, 0, 1.0]), cfg, hist)
    assert part.full_recheck() < 1e-6
    assert any(e.get("op", "").startswith("pack") for e in hist.events if e.get("type") == "edit")


def test_q_basis_reproduces_qbar_at_any_angle():
    from rltp.config import Material
    from rltp.eval.fea2d import q_basis, q_matrix, q_rotate
    Q = q_matrix(Material())
    M = q_basis(Q)
    for a in np.linspace(0, np.pi, 13):
        rec = M[0] + np.cos(2 * a) * M[1] + np.sin(2 * a) * M[2] + np.cos(4 * a) * M[3] + np.sin(4 * a) * M[4]
        assert np.allclose(rec, q_rotate(Q, a), atol=1e-8 * np.abs(Q).max())


def _ring_part(directions):
    from rltp.config import ScoreConfig
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    g = Point(0, 0).buffer(5.0, quad_segs=64).difference(Point(0, 0).buffer(2.5, quad_segs=64))
    m = trimesh.creation.extrude_polygon(g, height=0.6)
    m.apply_translation([5.0, 5.0, 0])
    loads = LoadSpec([Region(box=[0, 4.0, 2.6, 6.0])], [Load2D(Region(box=[7.4, 4.0, 10.0, 6.0]), (0, -20))])
    return build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover"),
                                  score=ScoreConfig(fea_coarsen=2, directions=directions)), baseline="zigzag")


def test_path_directions_follow_curves_and_score_incrementally():
    """--directions path: wall nodes on a ring read the circle's tangent (a few degrees), a
    straight run reads its axis exactly; the ring's walls (staircases) score lower FI along
    their paths than as the worst class their steps hold; straight zigzag infill scores
    about the same either way; edits score incrementally."""
    from rltp.cover.paths import node_angles
    from rltp.search.fields import wall_cells
    parts = {d: _ring_part(d) for d in ("classes", "path")}
    L = parts["path"].layers[1]
    a = node_angles(L)
    nodes = np.flatnonzero(L.flat_mask)
    wall = wall_cells(L).reshape(-1)[nodes] & ~np.isnan(a[nodes])
    r, c = np.divmod(nodes[wall], L.W)
    fr = L.material_frame
    x, y = fr.to_world((c + 1.0) * fr.p, (r + 1.0) * fr.p)        # nodes sit on cell corners
    true = (np.arctan2(y - 5.0, x - 5.0) + np.pi / 2) % np.pi
    err = np.degrees(np.abs((a[nodes[wall]] - true + np.pi / 2) % np.pi - np.pi / 2))
    assert wall.sum() > 100 and np.median(err) < 3.0 and np.percentile(err, 90) < 8.0
    # straight runs along x read 0 exactly
    sx = (np.abs(L.nxt[nodes] - nodes) == 1) & (np.abs(L.prv[nodes] - nodes) == 1)
    deep = sx.copy()
    for s in (2, 4, 8):
        deep &= np.isin(nodes + s, nodes) & np.isin(nodes - s, nodes)
    if deep.any():
        assert np.allclose(np.minimum(a[nodes[deep]], np.pi - a[nodes[deep]]), 0.0, atol=1e-9)
    # the walls' share of sum FI^8 falls when they're read along their paths
    shares = {}
    for d, part in parts.items():
        tot = wsum = 0.0
        for Lk, ev in zip(part.layers, part.evals):
            f8 = np.asarray(ev.fea.fi, float) ** 8
            wsum += f8[wall_cells(Lk).reshape(-1)].sum()
            tot += f8.sum()
        shares[d] = wsum / tot
    assert shares["path"] < shares["classes"]
    # edits score incrementally in path mode
    part = parts["path"]
    for _ in _random_edits([part], 20):
        pass
    assert part.full_recheck() < 1e-6


def test_path_and_class_scoring_agree_on_straight_infill():
    """A box of straight zigzag infill (axis lines, rectangular walls) scores within a few
    percent either way (they differ only where several directions meet in a cell)."""
    from rltp.config import ScoreConfig
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    m = trimesh.creation.extrude_polygon(box(0, 0, 12, 6), height=0.6)
    loads = LoadSpec([Region(box=[0, 0, 1, 6])], [Load2D(Region(box=[11, 0, 12, 6]), (0, -20))])
    fi = {}
    for d in ("classes", "path"):
        part = build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover"),
                                      score=ScoreConfig(fea_coarsen=2, directions=d)), baseline="zigzag")
        fi[d] = sum(e.fea.S_p for e in part.evals)
    assert fi["path"] == pytest.approx(fi["classes"], rel=0.15)


def _fit_layer(paths):
    from rltp.cover.cover import link
    from rltp.model import Frame, Layer
    H = W = 200
    L = Layer.empty(0, 0.2, 0.2, np.zeros((H, W), bool), Frame(0, 0, 0, 0.1))
    L.kb = 4
    L.aux = {"fit": True}
    for path in paths:
        ids = [r * W + c for r, c in path]
        L.mask.reshape(-1)[ids] = True
        for a, b in zip(ids[:-1], ids[1:]):
            link(L, a, b)
    L.invalidate()
    return L


@pytest.mark.parametrize("deg", [11.25, 22.5, 33.75, 61.0, 101.25])
def test_fit_mode_prints_a_staircase_as_one_straight_bead(deg):
    """Fit mode (true directions): a 4-connected staircase along any angle prints as one
    straight move within 0.1 mm of every node and deposits a straight bead: at most 1 per
    cell, (length + two half-bead caps) x width in all, shared between the two nearest
    direction classes by angle."""
    from rltp.cover.beads import cell_shares, density, runs
    t = np.radians(deg)
    ux, uy = np.cos(t), np.sin(t)
    sx, sy = (1 if ux >= 0 else -1), (1 if uy >= 0 else -1)
    x = y = 0
    pts = [(40, 40 if deg < 90 else 160)]
    for _ in range(150):
        x, y = min([(x + sx, y), (x, y + sy)], key=lambda q: abs(q[0] * uy - q[1] * ux))
        pts.append((pts[0][0] + y, pts[0][1] + x))
    L = _fit_layer([pts])
    verts, seg, V = runs(L)["poly"][0]
    assert len(V) == 2 and seg.size == len(pts) - 1
    d = V[1] - V[0]
    assert np.degrees(np.arctan2(d[1], d[0])) % 180 == pytest.approx(deg, abs=0.1)   # its true angle
    for r, c in pts:
        assert abs(d[0] * (r - V[0, 1]) - d[1] * (c - V[0, 0])) / np.hypot(*d) <= 1.0 + 1e-9
    D = density(L)
    assert D.max() <= 1.0 + 1e-9
    assert D.sum() == pytest.approx((np.hypot(*d) + 4) * 4, rel=0.01)
    sh = cell_shares(L).reshape(L.H, L.W, 4)[D > 0.99].mean(0)
    j = int(deg // 45) % 4
    f = deg / 45 - deg // 45
    assert sh[j] == pytest.approx(1 - f, abs=0.01) and sh[(j + 1) % 4] == pytest.approx(f, abs=0.01)


@pytest.mark.parametrize("deg", [11.25, 33.75, 101.25, 157.5])
def test_line_set_lays_lines_every_11_25_degrees(deg):
    """line_set at a 16-bin angle: 4-connected digital lines one bead apart across (the
    nodes of different lines never 4-adjacent), linked along each line into strokes whose
    path direction reads the angle (node_angles) and that print as single moves."""
    from scipy import ndimage

    from rltp.cover.beads import angle_links, line_set, runs
    from rltp.cover.paths import node_angles
    sel = np.zeros((200, 200), bool)
    sel[30:170, 30:170] = True
    got = line_set(sel, deg, 4, (100, 100), 1.0, 4)
    lab, n = ndimage.label(got)
    assert n >= 20
    L = _fit_layer([])
    L.mask |= got
    angle_links(L, got, deg)
    L.invalidate()
    pc = get_pieces(L)
    assert len(pc.cells) == n                     # each line one stroke
    t = np.radians(deg)
    r, c = np.divmod(np.flatnonzero(got.reshape(-1)), 200)
    across = r * np.cos(t) - c * np.sin(t)
    size = np.bincount(lab[r, c], minlength=n + 1)[1:]
    per = ndimage.mean(across, lab[r, c], np.arange(1, n + 1))[size >= 60]
    gaps = np.diff(np.sort(per))
    assert gaps.size >= 10 and abs(gaps.mean() - 4.0) < 0.02 and np.allclose(gaps, 4.0, atol=0.25)
    a = node_angles(L)
    long = [pid for pid, cells in enumerate(pc.cells) if len(cells) > 60]
    mid = [pc.cells[pid][len(pc.cells[pid]) // 2] for pid in long]
    err = np.degrees(np.abs((a[mid] - t + np.pi / 2) % np.pi - np.pi / 2))
    assert long and err.max() < 2.0
    for pid in long:                              # printed as one move along the line
        V = runs(L)["poly"][pid][2]
        assert len(V) == 2
        seg = np.arctan2(V[1, 1] - V[0, 1], V[1, 0] - V[0, 0])
        assert np.degrees(abs((seg - t + np.pi / 2) % np.pi - np.pi / 2)) < 0.5


def test_dir_fi16_matches_the_four_classes():
    """The guide's 16-bin failure indices agree with its four-class ones at 0/45/90/135."""
    from rltp.search.guide import _dir_fi, _dir_fi16
    part = _ring_part("path")
    lam = part._lam_problem()
    f4 = _dir_fi(lam, part.lam_u, lam.case_scale, part.p)
    f16 = _dir_fi16(lam, part.lam_u, lam.case_scale, part.p)
    assert np.allclose(f16[:, ::4], f4, rtol=1e-9, atol=1e-12 * f4.max())


def test_classes_round_to_16_bins():
    """classes(bins=16): nearest multiple of 11.25 degrees; small in-between patches fall back
    to the nearest of the four classes."""
    from rltp.search.fields import classes
    th = np.radians(np.full((40, 40), 20.0))
    th[:3, :3] = np.radians(80.0)                 # a 9-cell patch
    mask = np.ones((40, 40), bool)
    cls = classes(th, mask, True, 0, 0, bins=16)
    assert cls[20, 20] == 22.5 and cls[0, 0] == 78.75
    cls = classes(th, mask, True, 0, 20, bins=16)
    assert cls[20, 20] == 22.5 and cls[0, 0] == 90.0


def test_line_bins_16_guided_lay_and_turn():
    """--line-bins 16 (true directions): the guide's lay maps turn on the lay and turn16 edits;
    guided proposals are lay ops that re-lay a window at its best 11.25-degree bin; turn16
    turns an area by one bin; edits keep the layer valid and score incrementally."""
    from rltp.cover.beads import bead_bins, bin_deg
    from rltp.cover.ops import OPS, InvalidEdit, action_mask, apply
    from rltp.search.guide import DirectionMap, guided_action
    part = _ring_part("path")
    gm = DirectionMap(part)
    k = 1
    L = part.layers[k]
    from rltp.cover.ops import OP_INDEX
    M0 = action_mask(L)
    assert not M0[OP_INDEX["lay32"]].any() and not M0[OP_INDEX["turn16_pos"]].any()   # off until attached
    gm.attach16(k)
    M = action_mask(L).reshape(-1)
    rng = np.random.default_rng(0)
    done = turned = 0
    for _ in range(24):
        a = guided_action(part, k, M, rng, gm)
        if a is None:
            continue
        op, cell = divmod(a, L.n)
        assert OPS[op].startswith("lay")
        try:
            N, params = apply(L, op, cell)
        except InvalidEdit:
            continue
        size = int(OPS[op][3:])
        assert check_invariants(N) == [] and params["to"] == bin_deg(L.ctx["lay"][size].reshape(-1)[cell])
        done += 1
    assert done >= 5
    cells = np.flatnonzero(M.reshape(len(OPS), -1)[OP_INDEX["turn16_pos"]])
    for cell in rng.choice(cells, 12, replace=False):
        try:
            N, params = apply(L, OP_INDEX["turn16_pos"], int(cell))
        except InvalidEdit:
            continue
        assert check_invariants(N) == []
        assert params["to"] == bin_deg(bead_bins(L).reshape(-1)[cell] + 1)
        turned += 1
    assert turned >= 3
    cfg = LoopConfig(max_evals=60, burst=8, warmup=8, guide=1.0, line_bins=16)
    optimize_part(part, make_proposer("sa"), np.array([1.0, 0.3, 0.3, 0, 0, 1.0]), cfg)
    assert part.full_recheck() < 1e-6
    with pytest.raises(ValueError):
        optimize_part(_ring_part("classes"), make_proposer("sa"), [1.0], LoopConfig(max_evals=4, line_bins=16))


def test_fit_mode_zigzag_and_jog():
    """Fit mode: a one-cell jog in a straight line is one move; a zigzag of axis lines one
    bead apart covers its block exactly once (mitred 90-degree turns), as much material as
    its G-code extrudes; a ring part scored along true directions prints its curved walls
    in fewer, longer moves and faster than the class scoring's staircases."""
    from rltp.cover.beads import density, fit_mode, runs
    jog = [(100, c) for c in range(20, 101)] + [(101, c) for c in range(100, 181)]
    L = _fit_layer([jog])
    assert len(runs(L)["poly"][0][0]) == 2
    assert density(L).sum() == pytest.approx((np.hypot(1, 160) + 4) * 4, rel=1e-3)
    zz = []
    for i in range(6):
        cs = range(20, 181) if i % 2 == 0 else range(180, 19, -1)
        if i:
            zz += [(20 + 4 * i - 4 + j, zz[-1][1]) for j in range(1, 4)]
        zz += [(20 + 4 * i, c) for c in cs]
    D = density(_fit_layer([zz]))
    assert D.max() == pytest.approx(1.0) and D[20:40, 20:180].min() == pytest.approx(1.0)
    assert D.sum() == pytest.approx((6 * 160 + 5 * 4 + 4) * 4)
    parts = {d: _ring_part(d) for d in ("classes", "path")}
    assert all(fit_mode(L) for L in parts["path"].layers) and not any(fit_mode(L) for L in parts["classes"].layers)
    nseg = {d: sum(len(v[0]) - 1 for L in p.layers for v in runs(L)["poly"].values()) for d, p in parts.items()}
    assert nseg["path"] < 0.7 * nseg["classes"]
    assert parts["path"].breakdown()["time_s"] < parts["classes"].breakdown()["time_s"]


def test_coarse_fea_scores_incrementally_on_a_fine_grid():
    """0.1 mm cells, 0.4 mm elements: edits still score incrementally, and the start's
    FI_agg is near the one-element-per-cell analysis (stress peaks are blunter)."""
    from rltp.config import ScoreConfig
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    g = box(0, 0, 8, 3).union(box(0, 0, 3, 7))
    m = trimesh.creation.extrude_polygon(g, height=0.6)
    loads = LoadSpec([Region(box=[0, 6, 3, 7])], [Load2D(Region(box=[7, 0, 8, 3]), (0, -20))])
    fi = {}
    for cf in (1, 4):
        part = build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover"),
                                      score=ScoreConfig(fea_coarsen=cf)), baseline="zigzag")
        assert part.lam.cf == cf
        fi[cf] = part.breakdown()["fi_agg"]
        if cf == 4:
            for _ in _random_edits([part], 25):
                pass
            assert part.full_recheck() < 1e-6
    assert fi[4] == pytest.approx(fi[1], rel=0.35)


# ------------------------------------------------------------------ sparse layers

def _sparse_part(infill=0.2, walls=2, solid=1, stack="auto", height=1.2, loads=None, terms=("strength", "time", "cont"),
                 rule="free", band=0.02):
    from rltp.config import ScoreConfig
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    m = trimesh.creation.extrude_polygon(g, height=height)
    pr = Printer(walls=walls, solid_layers=solid, infill=infill, max_bridge=2.0, infill_rule=rule, infill_band=band)
    spec = PartSpec(pr, loads or bracket_spec().loads, terms=terms, score=ScoreConfig(stack=stack))
    spec.loads.point_loads = []
    return build_part(m, spec, baseline="zigzag")


def _sparse_rules(part):
    from rltp.cover.shell import unsupported
    errs = []
    for k, L in enumerate(part.layers):
        errs += [f"layer {k}: {e}" for e in check_invariants(L)]
        if (L.mask & ~L.part).any():
            errs.append(f"layer {k}: filled cell outside the part")
        if (L.req & ~L.mask).any():
            errs.append(f"layer {k}: required cell empty")
        lk = L.lock.reshape(-1)
        nb = np.concatenate([L.nxt[lk], L.prv[lk]])
        if (nb >= 0).any() and not lk[nb[nb >= 0]].all():
            errs.append(f"layer {k}: a wall is linked to infill")
    bad = unsupported(part.layers, part.support)
    if any(bad):
        errs.append(f"unsupported cells per layer: {bad}")
    return errs


def test_sparse_baseline_layout():
    """Walls on every layer, solid first/last layers, infill lines in between, all supported."""
    part = _sparse_part(infill=0.25, walls=2, solid=1)
    assert len(part.layers) == 6 and part.laminate
    assert _sparse_rules(part) == []
    for k, L in enumerate(part.layers):
        assert L.lock.sum() > 0 and (L.lock <= L.req).all()
        if k in (0, len(part.layers) - 1):
            assert L.mask.sum() == L.part.sum()                  # solid skin
        else:
            opt = L.part & ~L.req
            assert 0.15 < (L.mask & opt).sum() / opt.sum() < 0.35  # about the requested 25%
    # the outermost wall ring of every outline is a closed loop (the second ring can't
    # close where the hole sits 1.3 mm from the edge: too thin for 2 rings each side)
    from rltp.cover.baselines import rings
    from rltp.cover.cover import get_pieces
    for L in part.layers:
        pc = get_pieces(L)
        outer = rings(L.part, 1)[0].reshape(-1)
        assert all(pc.is_loop[lab] for lab in set(pc.label[outer].tolist()))
    b = part.breakdown()
    assert b["unsupported"] == 0 and 0 < b["infill_pct"] < 100


def test_sparse_edits_keep_every_rule():
    """Random edits of every kind, committed regardless of score: walls never change,
    required cells stay filled, the support rule holds, scores stay incremental."""
    from rltp.cover.ops import OPS, InvalidEdit
    part = _sparse_part(infill=0.25)
    walls0 = [(L.nxt[L.lock.reshape(-1)].copy(), L.prv[L.lock.reshape(-1)].copy()) for L in part.layers]
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(300):
        k = int(rng.integers(len(part.layers)))
        L = part.layers[k]
        a = int(sample_valid(action_mask(L).reshape(-1), rng)[0])
        try:
            newL, _ = apply_action(L, a)
        except InvalidEdit:
            continue
        seen.add(OPS[a // L.n])
        c = part.propose(k, newL)
        before = part.raw()
        part.commit(c)
        after = part.raw()
        for t in c.d:
            assert after[t] - before[t] == pytest.approx(c.d[t], abs=1e-9)
    assert {"add", "remove", "fill5_0", "fill5_90", "clear5"} <= seen
    assert _sparse_rules(part) == []
    for L, (nx, pv) in zip(part.layers, walls0):
        lk = L.lock.reshape(-1)
        assert (L.nxt[lk] == nx).all() and (L.prv[lk] == pv).all()
    assert part.full_recheck() < 1e-6


def test_removal_cannot_strand_the_layer_above():
    """A cell that is the only support of a filled cell above can't be removed."""
    from rltp.cover.ops import InvalidEdit, OP_INDEX, apply
    from rltp.cover.shell import critical
    part = _sparse_part(infill=0.25)
    k = 1                                   # a sparse layer under another sparse layer
    rng = np.random.default_rng(0)
    for _ in range(400):                    # thin out its infill until some cell above hangs on one support
        L = part.layers[k]
        crit = critical(L) & ~L.req
        if crit.any():
            break
        M = action_mask(L)[OP_INDEX["remove"]]
        cell = int(rng.choice(np.flatnonzero(M)))
        part.commit(part.propose(k, apply(L, OP_INDEX["remove"], cell)[0]))
    assert crit.any()
    cell = int(np.flatnonzero(crit)[0])
    assert not action_mask(L)[OP_INDEX["remove"], cell]
    with pytest.raises(InvalidEdit):
        apply(L, OP_INDEX["remove"], cell)


def test_fill_edits_need_a_sparse_layer():
    from rltp.cover.ops import N_PATH_OPS
    part = build_part(bracket_mesh(), bracket_spec(("strength",)), baseline="zigzag")
    M = action_mask(part.layers[0])
    assert not M[N_PATH_OPS:].any() and M[:N_PATH_OPS].any()


def test_empty_cells_carry_no_failure_index():
    part = _sparse_part(infill=0.25)
    for L, ev in zip(part.layers, part.evals):
        fi = ev.fea.fi
        assert (fi[~L.flat_mask] == 0).all()
        assert ev.fea.N == L.part.sum()


def test_laminate_infill_matters_less_than_split():
    """Bonded layers share the strain, so emptying the middle layers raises FI_agg far
    less than when each layer carries an equal share of the load on its own."""
    rel = {}
    for stack in ("split", "laminate"):
        full = _sparse_part(infill=1.0, walls=2, solid=1, stack=stack)
        sparse = _sparse_part(infill=0.2, walls=2, solid=1, stack=stack)
        rel[stack] = sparse.breakdown()["fi_agg"] / full.breakdown()["fi_agg"]
    assert rel["laminate"] < rel["split"]
    assert rel["split"] > 1.5


def test_sparse_loop_runs_and_keeps_rules():
    part = _sparse_part(infill=0.25)
    res = optimize_part(part, make_proposer("sa"), np.array([0.5, 1.0, 1.0, 0.0]),
                        LoopConfig(max_evals=400, burst=16, warmup=32))
    assert res["evals"] <= 400
    assert _sparse_rules(part) == []


# ------------------------------------------------------------------ infill target

def test_line_picks_track_density_and_spacing():
    from rltp.cover.shell import line_picks
    even = np.full(20, 10)
    assert line_picks(even, 0.2, 99).sum() == 4
    assert line_picks(even, 0.4, 99).sum() == 8
    assert line_picks(even, 1.0, 99).all()
    sparse = np.flatnonzero(line_picks(even, 0.05, 4))
    assert np.diff(sparse).max() <= 4                       # the spacing cap beats the density
    uneven = np.array([2, 30, 2, 30, 2, 30, 2, 30, 0, 0])
    got = line_picks(uneven, 0.5, 99)
    assert abs(uneven[got].sum() / uneven.sum() - 0.5) < 0.15


def test_baseline_starts_at_the_infill_target():
    for target in (0.25, 0.4, 0.6):
        part = _sparse_part(infill=target)
        assert abs(part.infill_fraction() - target) < 0.02, (target, part.infill_fraction())


def test_band_rule_keeps_infill_in_its_band():
    from rltp.cover.ops import OPS, InvalidEdit
    part = _sparse_part(infill=0.4, rule="band", band=0.03)
    bud = part.budget
    f0 = part.infill_fraction()
    inside0 = bud["lo"] <= bud["total"] <= bud["hi"]
    rng = np.random.default_rng(3)
    fills = 0
    for _ in range(400):
        k = int(rng.integers(len(part.layers)))
        L = part.layers[k]
        a = int(sample_valid(action_mask(L).reshape(-1), rng)[0])
        try:
            newL, _ = apply_action(L, a)
        except InvalidEdit:
            continue
        fills += OPS[a // L.n] in ("add", "remove", "fill5_0", "fill5_90", "clear5")
        part.commit(part.propose(k, newL))
        if inside0:
            assert bud["lo"] <= bud["total"] <= bud["hi"]
    assert fills > 20
    assert bud["total"] == sum(int((L.mask & ~L.req).sum()) for L in part.layers)
    assert abs(part.infill_fraction() - 0.4) <= max(0.03, abs(f0 - 0.4)) + 1e-9


def test_penalty_rule_scores_distance_from_target():
    part = _sparse_part(infill=0.4, rule="penalty", terms=("strength", "time", "cont", "infill"))
    assert part.raw()["infill"] == pytest.approx(-abs(part.infill_fraction() - 0.4))
    assert part.budget is None
    for _ in _random_edits([part], 120, seed=5):    # asserts nothing; incremental check below
        pass
    assert part.raw()["infill"] == pytest.approx(-abs(part.infill_fraction() - 0.4))
    assert part.full_recheck() < 1e-6


def test_normalizer_loads_states_from_before_the_infill_term():
    from rltp.eval.metrics import Normalizer
    old = {"n": {t: 3.0 for t in ("strength", "time", "cont", "inter")},
           "mean": {t: 0.0 for t in ("strength", "time", "cont", "inter")},
           "m2": {t: 1.0 for t in ("strength", "time", "cont", "inter")}, "frozen": True}
    nz = Normalizer().load(old)
    assert nz.mode == "std" and nz.total({"strength": 1.0}, [1.0, 0, 0, 0]) == pytest.approx(1 / np.sqrt(0.5))


# ------------------------------------------------------------------ moves and Z bonding

def test_moves_keep_the_infill_count_and_every_rule():
    from rltp.cover.ops import OPS, InvalidEdit, OP_INDEX
    part = _sparse_part(infill=0.4, rule="band", band=0.0)        # a zero-width band: only moves change fill
    rng = np.random.default_rng(7)
    total0 = int(part.Fk.sum())
    moved = 0
    for _ in range(400):
        k = int(rng.integers(len(part.layers)))
        L = part.layers[k]
        M = action_mask(L)
        M[:OP_INDEX["add_trim"]] = False                          # only the moves (and stack9)
        if not M.any():
            continue
        a = int(sample_valid(M.reshape(-1), rng)[0])
        try:
            newL, params = apply_action(L, a)
        except InvalidEdit:
            continue
        assert int((newL.mask & ~newL.req).sum()) == int((L.mask & ~L.req).sum())
        assert params.get("trimmed") or params.get("grown") or OPS[a // L.n] == "stack9"
        part.commit(part.propose(k, newL))
        moved += 1
    assert moved > 50 and int(part.Fk.sum()) == total0
    assert _sparse_rules(part) == []
    assert part.full_recheck() < 1e-6


def test_trim_takes_unbonded_then_least_stressed_ends():
    from rltp.cover.ops import OP_INDEX, _bonded, _trimmable, apply
    part = _sparse_part(infill=0.4)
    k, L = 1, part.layers[1]
    fi, bonded = L.ctx["fi"], _bonded(L)
    cell = int(np.flatnonzero(action_mask(L)[OP_INDEX["add_trim"]])[0])
    cand = _trimmable(L)
    cand[cell] = False
    N, params = apply(L, OP_INDEX["add_trim"], cell)
    x = params["trimmed"][0]
    idx = np.flatnonzero(cand)
    pool = idx[~bonded[idx]] if (~bonded[idx]).any() else idx
    assert not bonded[x] or bonded[idx].all()
    assert fi[x] == pytest.approx(fi[pool].min())


def _zpart(infill, stacked):
    """Sparse test part; `stacked` turns every sparse layer's infill to the same direction."""
    part = _sparse_part(infill=infill, solid=1, terms=("strength", "time", "cont", "inter"), height=1.6)
    if stacked:
        from rltp.cover.shell import fill_sparse, solid_masks
        parts = [L.part for L in part.layers]
        solids = solid_masks(parts, 1)
        for k, L in enumerate(part.layers):
            fill_sparse(L, parts[k], solids[k], 2, infill, 0, 4)
        part.refresh_support()
        part.full_recheck()
    return part


def test_zbond_rewards_overlap_between_layers():
    """Lines stacked in one direction overlap far more than alternating 0/90 lines, and
    the interlayer term scores that as stronger."""
    cross, stack = _zpart(0.3, False), _zpart(0.3, True)
    bc, bs = cross.breakdown(), stack.breakdown()
    assert bs["weakest_bond_pct"] > bc["weakest_bond_pct"] + 5
    assert bs["inter_fi_agg"] < bc["inter_fi_agg"]
    full = _sparse_part(infill=1.0, solid=1, terms=("strength", "time", "cont", "inter"), height=1.6)
    assert full.breakdown()["bonded_pct"] == pytest.approx(100.0)
    assert full.breakdown()["inter_fi_agg"] < bs["inter_fi_agg"]


def test_zbond_is_incremental():
    part = _zpart(0.3, False)
    assert part.use_inter and all(itf is not None for itf in part.ifaces)
    for _ in _random_edits([part], 150, seed=11):
        pass
    assert part.full_recheck() < 1e-6


def test_stack9_copies_the_layer_below_and_keeps_the_count():
    from rltp.cover.ops import OP_INDEX, InvalidEdit, apply
    part = _zpart(0.3, False)                    # alternating 0/90 infill: little overlap
    k = 2
    L, below = part.layers[k], part.layers[k - 1]
    M = action_mask(L)[OP_INDEX["stack9"]]
    assert M.any()
    opt = (L.part & ~L.req).reshape(-1)
    done = raised = 0
    for cell in np.flatnonzero(M)[::7]:
        try:
            N, params = apply(L, OP_INDEX["stack9"], int(cell))
        except InvalidEdit:
            continue
        over0 = int((L.flat_mask & below.flat_mask & opt).sum())
        over1 = int((N.flat_mask & below.flat_mask & opt).sum())
        assert over1 >= over0
        raised += over1 > over0
        assert int((N.mask & ~N.req).sum()) == int((L.mask & ~L.req).sum())
        assert check_invariants(N) == []
        done += 1
    assert done >= 3 and raised >= done // 2


@pytest.mark.parametrize("sparse", [False, True])
def test_refined_fea_scores_incrementally(sparse):
    from rltp.config import ScoreConfig
    if sparse:
        part = _sparse_part(infill=0.3)
        part.spec.score.fea_refine = 2
        part._fea.clear()
        part.lam = None
        part.full_recheck()
        assert part.lam.r == 2
    else:
        spec = bracket_spec(("strength", "time", "cont"))
        spec.score = ScoreConfig(fea_refine=2)
        part = build_part(bracket_mesh(), spec, baseline="zigzag")
        assert part.fea(0).r == 2
    for _ in _random_edits([part], 40, seed=2):
        pass
    assert part.full_recheck() < 1e-6


# ------------------------------------------------------------------ fine bead grid

def _fine_part(infill=0.4, walls=2, solid=1, height=1.2, terms=("strength", "time", "cont", "inter", "cover")):
    from rltp.config import ScoreConfig
    g = box(0, 0, 8, 3).union(box(0, 0, 3, 6)).difference(Point(6.5, 1.5).buffer(0.6))
    m = trimesh.creation.extrude_polygon(g, height=height)
    pr = Printer(walls=walls, solid_layers=solid, infill=infill, max_bridge=2.0, cell=0.1,
                 infill_rule="free", infill_pattern="stacked")
    loads = LoadSpec([Region(circle=[1.5, 5.0, 1.0])], [Load2D(Region(circle=[6.5, 1.5, 1.0]), (0, -20))])
    return build_part(m, PartSpec(pr, loads, terms=terms), baseline="zigzag")


def test_fine_layout_beads_four_cells_wide():
    """0.1 mm cells, 0.4 mm beads: a straight bead deposits exactly 1 per cell, walls are
    locked loops along the true outline, solid layers are nearly fully covered."""
    from rltp.cover.beads import density
    part = _fine_part()
    L0 = part.layers[0]
    assert L0.kb == 4 and L0.frame.p == pytest.approx(0.1) and part.fine
    D = density(L0)
    filled = D > 0
    assert np.isclose(D[filled], 1.0).mean() > 0.85                     # mostly exactly one bead
    assert (D[L0.req] >= 0.5).mean() > 0.95                             # solid skin covered
    pc = get_pieces(L0)
    outer = set(pc.label[L0.lock.reshape(-1)].tolist())
    assert outer and all(pc.is_loop[l] for l in outer)
    b = part.breakdown()
    assert b["unsupported"] == 0 and 0 <= b["gap_pct"] < 5 and 0 < b["over_pct"] < 10
    assert abs(b["infill_pct"] - 40) < 8
    import shapely
    for L in part.layers:
        assert check_invariants(L) == []
        # bead centres sit at least half a bead inside the outline (0.1 mm rasterisation)
        x, y = L.frame.cell_xy(np.flatnonzero(L.flat_mask), L.W)
        assert shapely.contains_xy(L.poly.buffer(-0.2 + 0.1 + 1e-6), x, y).all()


def test_fine_edits_keep_rules_and_score_incrementally():
    from rltp.cover.ops import FINE_FIRST, OPS, InvalidEdit
    part = _fine_part()
    walls0 = [(L.nxt[L.lock.reshape(-1)].copy(), L.prv[L.lock.reshape(-1)].copy()) for L in part.layers]
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(160):
        k = int(rng.integers(len(part.layers)))
        L = part.layers[k]
        a = int(sample_valid(action_mask(L).reshape(-1), rng)[0])
        op = a // L.n
        assert op in (0, 2, 5) or op >= FINE_FIRST                     # no unit-cell edits on a fine grid
        try:
            newL, _ = apply_action(L, a)
        except InvalidEdit:
            continue
        seen.add(OPS[op])
        c = part.propose(k, newL)
        before = part.raw()
        part.commit(c)
        after = part.raw()
        for t in c.d:
            assert after[t] - before[t] == pytest.approx(c.d[t], abs=1e-9)
        assert check_invariants(part.layers[k]) == []
    assert {"shift_pos", "shift_neg", "extend", "retract", "refill_0", "clear_w"} <= seen
    for L, (nx, pv) in zip(part.layers, walls0):
        lk = L.lock.reshape(-1)
        assert (L.nxt[lk] == nx).all() and (L.prv[lk] == pv).all()
    assert part.breakdown()["unsupported"] == 0
    assert part.full_recheck() < 1e-6


def test_fine_shift_moves_a_bead_by_one_cell():
    from rltp.cover.ops import OP_INDEX, apply
    part = _fine_part(solid=0, infill=0.3)
    L = part.layers[1]
    free = np.flatnonzero(L.flat_mask & ~L.lock.reshape(-1))
    # a node in the middle of a horizontal run
    cell = next(int(x) for x in free if L.nxt[x] in (x + 1, x - 1) and L.prv[x] in (x + 1, x - 1))
    N, params = apply(L, OP_INDEX["shift_pos"], cell)
    a, b = params["run"]
    assert N.flat_mask[cell + L.W] and not N.flat_mask[cell]
    assert int(N.mask.sum()) - int(L.mask.sum()) in (-2, -1, 0, 1, 2)    # legs grow or shrink by one
    assert check_invariants(N) == []


def test_fine_gcode_extrudes_bead_width():
    """Fine-grid G-code: every 0.1 mm step of bead extrudes w * step * h of material."""
    import tempfile
    from rltp.export.gcode import write_gcode
    part = _fine_part(solid=1, infill=1.0)
    pr = part.spec.printer
    st = write_gcode(tempfile.mktemp(suffix=".gcode"), part, pr)
    links = sum(int(((L.nxt >= 0) & L.flat_mask).sum()) for L in part.layers)
    assert st.print_mm == pytest.approx(links * 0.1, rel=1e-6)          # 0.1 mm steps
    vol = st.filament_mm * np.pi * (pr.filament_diameter / 2) ** 2
    material = sum(float(L.density.sum()) * L.material_frame.p ** 2 * L.h for L in part.layers)
    assert vol == pytest.approx(material, rel=0.1)                       # G-code lays what the model counts


# ------------------------------------------------------------------ anchoring

def test_whole_runs_lays_each_line_whole_or_not_at_all():
    """Gradient layouts: a line's run is laid whole when the field's mean along it clears
    the line's threshold, instead of starting and stopping with the field."""
    from rltp.cover.beads import whole_runs
    cand = np.zeros((9, 20), bool)
    cand[[0, 4, 8], 2:18] = True
    fill = np.zeros((9, 20))
    fill[:, 2:10], fill[:, 10:18] = 0.9, 0.1           # mean 0.5 along every run
    thr = np.zeros((9, 20))
    thr[0], thr[4], thr[8] = 0.2, 0.45, 0.7
    out = whole_runs(cand, fill, thr)
    assert out[0, 2:18].all() and out[4, 2:18].all() and not out[8].any()
    assert not (out & ~cand).any()


def test_cut_line_is_a_free_end_and_anchor_reattaches_it():
    """Layouts leave no free ends; a line cut short of the wall leaves one, which costs
    continuity (c_free_end), and the anchor edit extends it back until it touches a bead."""
    from rltp.cover.beads import anchor_end, free_ends
    from rltp.cover.ops import OP_INDEX, InvalidEdit, apply
    part = _fine_part(solid=0, infill=0.4, walls=1)
    assert part.breakdown()["free_ends"] == 0
    L = part.layers[1]
    lock = L.lock.reshape(-1)
    pc = get_pieces(L)
    done = 0
    for seq in [s for s, lp in zip(pc.cells, pc.is_loop) if not lp and not lock[s[0]] and len(s) > 40]:
        N = L
        for i in range(15):                              # cut 1.5 mm off the line's end
            N, _ = apply(N, OP_INDEX["retract"], seq[-1 - i])
        e = seq[-16]
        if e not in free_ends(N):
            continue
        c = part.propose(1, N)
        assert c.ev.cont.n_free_ends == part.evals[1].cont.n_free_ends + 1
        cfg = part.spec.score
        assert c.ev.cont.C - part.evals[1].cont.C > cfg.c_free_end - 15 * cfg.c_travel
        N2, prm = apply(N, OP_INDEX["anchor"], e)             # the shorter of straight on or a T-joint
        assert 0 < prm["added"] <= 15 and free_ends(N2) == [] and check_invariants(N2) == []
        N3 = N.copy()
        assert len(anchor_end(N3, e, turns=False)) == 15       # straight on: back to where the line ended
        assert free_ends(N3) == [] and check_invariants(N3) == []
        with pytest.raises(InvalidEdit):
            apply(L, OP_INDEX["anchor"], seq[-1])        # anchored ends don't move
        done += 1
    assert done >= 1


def test_relays_anchor_the_ends_they_make():
    from rltp.cover.beads import anchor_end, free_ends
    from rltp.cover.ops import OP_INDEX, InvalidEdit, apply
    part = _fine_part(solid=0, infill=0.4, walls=1)
    L = part.layers[1]
    rng = np.random.default_rng(0)
    n = 0
    for name in ("refill32_90", "refill32_0", "refill_45", "refill64_90"):
        for _ in range(12):
            try:
                N, _ = apply(L, OP_INDEX[name], int(rng.integers(L.n)))
            except InvalidEdit:
                continue
            assert free_ends(N) == [] and check_invariants(N) == []
            n += 1
    assert n >= 20


# ------------------------------------------------------------------ stress-guided start and search

def _fine_bracket(start="stress", **kw):
    pr = Printer(walls=2, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free",
                 infill_pattern="stacked", start=start, **kw)
    g = box(0, 0, 10, 2.4).union(box(0, 0, 2.4, 10))
    m = trimesh.creation.extrude_polygon(g, height=0.6)
    loads = LoadSpec([Region(box=[0, 8.5, 2.4, 10])], [Load2D(Region(box=[8.5, 0, 10, 2.4]), (0, -20))])
    return build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover")), baseline="zigzag")


def test_stress_start_runs_beads_along_each_leg():
    """An L clamped at the top of its upright and pulled down at the tip of its foot: the
    foot bends along x, the upright along y, and the stress start lays beads that way."""
    from rltp.cover.beads import cell_wh
    part = _fine_bracket("stress")
    L = part.layers[0]
    wh = cell_wh(L).reshape(L.H, L.W)
    mat = L.material & ~(__import__("rltp.cover.beads", fromlist=["cover_sum"]).cover_sum(L.lock, 4) > 0)
    x, y = L.material_frame.cell_centers(L.H, L.W)
    x, y = x.reshape(L.H, L.W), y.reshape(L.H, L.W)
    foot = mat & (x > 5)
    upright = mat & (y > 5)
    assert wh[foot].mean() > 0.7 and wh[upright].mean() < 0.3
    slicer = _fine_bracket("slicer")
    assert part.breakdown()["fi_agg"] < slicer.breakdown()["fi_agg"]


def test_stress_bias_puts_proposals_where_stress_is():
    from rltp.cover.ops import N_OPS, sample_valid
    part = _fine_bracket("stress")
    w = part.proposal_weights(0)
    L = part.layers[0]
    M = np.zeros((N_OPS, L.n), bool)
    M[0, :] = L.flat_mask            # one op, every node valid
    rng = np.random.default_rng(0)
    picks = sample_valid(M.reshape(-1), rng, 400, weights=w, bias=1.0) % L.n
    uni = sample_valid(M.reshape(-1), np.random.default_rng(0), 400) % L.n
    assert w[picks].mean() > 2 * w[uni].mean()


def test_strength_first_eases_back():
    part = _fine_bracket("slicer")
    w = np.array([0.5, 1, 1, 0, 0, 1.0])
    seen = []

    class Spy:
        name, temp0, bias = "spy", 0.1, 0.0

        def propose(self, part_, k, M, w_, rng):
            seen.append(float(w_[0]))
            return int(np.flatnonzero(M)[0]), None

    optimize_part(part, Spy(), w, LoopConfig(max_evals=60, burst=4, warmup=8, anneal_passes=4, strength_boost=3.0))
    assert seen[0] == pytest.approx(1.5) and min(seen) >= 0.5 - 1e-12


def test_big_refill_keeps_infill_and_flip_turns_a_region():
    from rltp.cover.beads import cell_wh
    from rltp.cover.ops import OP_INDEX, InvalidEdit, apply
    part = _fine_bracket("slicer")
    L = part.layers[1] if len(part.layers) > 2 else part.layers[0]
    opt = L.part & ~L.req
    share = lambda N: (N.material & opt).sum() / max(opt.sum(), 1)   # noqa: E731
    cell = int(np.flatnonzero(L.aux["inner"].reshape(-1))[len(np.flatnonzero(L.aux["inner"])) // 2])
    N, _ = apply(L, OP_INDEX["refill64_90"], cell)
    assert abs(share(N) - share(L)) < 0.08 and check_invariants(N) == []
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    plate = trimesh.creation.extrude_polygon(box(0, 0, 5, 4), height=0.4)
    loads = LoadSpec([Region(side="xmin", width=0.3)], [Load2D(Region(side="xmax", width=0.3), (10.0, 0))])
    S = build_part(plate, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover")),
                   baseline="zigzag").layers[0]                  # a solid skin: flips act on areas
    free = np.flatnonzero(S.flat_mask & ~S.lock.reshape(-1))
    r, c = np.divmod(free, S.W)
    node = int(free[np.argmin((r - S.H / 2) ** 2 + (c - S.W / 2) ** 2)])   # the middle
    wh0 = cell_wh(S)
    N2, params = apply(S, OP_INDEX["flip_region"], node)
    wh1 = cell_wh(N2)
    area = params["cells"]
    assert area > 50 and check_invariants(N2) == []
    changed = np.abs(wh1 - wh0) > 0.5
    assert changed.sum() > 0.5 * area


# ------------------------------------------------------------------ diagonal beads

def _staircase_layer(n_steps=20, cls=45):
    from rltp.cover.cover import link
    from rltp.model import Frame, Layer
    H = W = 40
    L = Layer.empty(0, 0.2, 0.2, np.zeros((H, W), bool), Frame(0, 0, 0, 0.1))
    L.kb = 4
    path = [10 * W + (10 if cls == 45 else 30)]
    for _ in range(n_steps):
        path.append(path[-1] + (1 if cls == 45 else -1))
        path.append(path[-1] + W)
    for a, b in zip(path[:-1], path[1:]):
        L.mask.reshape(-1)[a] = L.mask.reshape(-1)[b] = True
        link(L, a, b)
    L.invalidate()
    return L


@pytest.mark.parametrize("cls", [45, 135])
def test_staircase_is_a_straight_diagonal_bead(cls):
    """A staircase of one-cell steps is one diagonal run: deposited as a rotated bead of the
    right area (D = 1 inside), all at the right angle, printed as one straight move."""
    from rltp.cover.beads import cell_shares, density, runs
    L = _staircase_layer(20, cls)
    R = runs(L)
    assert [d[2] for d in R["diag"]] == [cls] and len(R["poly"][0][0]) == 2
    D = density(L)
    ln = np.hypot(20, 20)
    assert D.sum() == pytest.approx((ln + 4) * 4, rel=0.02)          # (length + 2 half caps) x width
    assert D.max() <= 1.0 + 1e-9
    sh = cell_shares(L)
    j = 1 if cls == 45 else 3
    assert sh[D.reshape(-1) > 0][:, j].min() == pytest.approx(1.0)


def test_diagonal_beads_change_the_stress_model():
    """Tension along x on a 45-degree-bead plate: stiffer than across (90) and weaker than
    along (0); its failure index falls between them."""
    from rltp.config import Material
    from rltp.eval.fea2d import FEAProblem
    from rltp.model import Frame
    mat = Material()
    H, W = 10, 30
    mask = np.ones((H, W), bool)
    g = box(0, 0, W * 0.1, H * 0.1)
    loads = LoadSpec([Region(side="xmin", width=1e-3)], [Load2D(Region(side="xmax", width=1e-3), (5.0, 0))]).resolve(g.bounds)
    fp = FEAProblem(mask, Frame(0, 0, 0, 0.1), 0.2, mat, loads, 1.0)
    n = H * W
    out = {}
    for name, sh in (("0", (1, 0, 0, 0)), ("45", (0, 1, 0, 0)), ("90", (0, 0, 1, 0))):
        r = fp.solve(np.full(n, 0.5), direct=True, filled=np.ones(n, bool), dens=np.ones(n),
                     shares=np.tile(sh, (n, 1)).astype(float))
        out[name] = (r.fi_agg(8), np.abs(r.u[0::2]).max())
    assert out["0"][0] < out["45"][0] < out["90"][0]
    assert out["0"][1] < out["45"][1] and out["90"][1] < out["45"][1] * 1.5


def test_diagonal_refill_and_turn_keep_rules():
    from rltp.cover.beads import bead_class
    from rltp.cover.ops import OP_INDEX, apply
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    plate = trimesh.creation.extrude_polygon(box(0, 0, 5, 4), height=0.4)
    loads = LoadSpec([Region(side="xmin", width=0.3)], [Load2D(Region(side="xmax", width=0.3), (10.0, 0))])
    part = build_part(plate, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover")), baseline="zigzag")
    L = part.layers[0]
    mid = int(np.flatnonzero(L.aux["inner"].reshape(-1))[L.aux["inner"].sum() // 2])
    N, _ = apply(L, OP_INDEX["refill32_45"], mid)
    assert check_invariants(N) == [] and (bead_class(N) == 45).sum() > 200
    free = np.flatnonzero(L.flat_mask & ~L.lock.reshape(-1))
    r, c = np.divmod(free, L.W)
    node = int(free[np.argmin((r - L.H / 2) ** 2 + (c - L.W / 2) ** 2)])
    N2, params = apply(L, OP_INDEX["turn_region"], node)
    assert params["to"] in (45, 135) and check_invariants(N2) == []
    c = part.propose(0, N)
    assert np.isfinite(c.d["strength"]) and np.isfinite(c.d["time"])
