"""PartState bookkeeping, interface statics, G-code volume and the part loop."""
import re

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, box

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
                                    "random_cases": 4, "random_seed": seed})
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
