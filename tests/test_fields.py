"""Gradient layout step: the field model, its adjoint gradient, the field optimiser, and
laying fields out as lines (rltp.search.fields)."""
import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, box

from rltp.config import Load2D, LoadSpec, PartSpec, Printer, Region, ScoreConfig
from rltp.cover.cover import check_invariants
from rltp.cover.shell import dither_ranks, unsupported
from rltp.eval.fea2d import q_matrix, q_rotate, to_material, tsai_wu_fi
from rltp.eval.stressmap import orientation, stress_field
from rltp.part import build_part
from rltp.search.fields import (FieldModel, FieldOptions, classes, line_ends, multi_start, optimize_fields,
                                start_field, start_names, stages_for, strain_rot, tsai_wu_grad)


def _loads():
    return LoadSpec([Region(circle=[2.5, 10, 2.0])], [Load2D(Region(circle=[13, 2.5, 2.0]), (0, -60))])


def _coarse_part(start="stress", infill=0.3):
    g = box(0, 0, 16, 5).union(box(0, 0, 5, 12)).difference(Point(13, 2.5).buffer(1.2))
    m = trimesh.creation.extrude_polygon(g, height=1.2)
    pr = Printer(walls=2, solid_layers=1, infill=infill, max_bridge=2.0, infill_rule="free",
                 infill_pattern="stacked", start=start)
    return build_part(m, PartSpec(pr, _loads(), terms=("strength", "time", "cont"), score=ScoreConfig()),
                      baseline="zigzag")


def _fine_bracket(start, fields=None):
    pr = Printer(walls=2, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free",
                 infill_pattern="stacked", start=start)
    g = box(0, 0, 10, 2.4).union(box(0, 0, 2.4, 10))
    m = trimesh.creation.extrude_polygon(g, height=0.6)
    loads = LoadSpec([Region(box=[0, 8.5, 2.4, 10])], [Load2D(Region(box=[8.5, 0, 10, 2.4]), (0, -20))])
    return build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont", "cover")), baseline="zigzag",
                      fields=fields or FieldOptions(starts=3, keep=2, workers=1, iters=20))


def _start_fields(part):
    """The stress start's directions per union cell, and a uniform rho at its infill."""
    m = FieldModel(part)
    L0 = part.layers[0]
    union = np.zeros_like(L0.part_mask)
    for L in part.layers:
        union |= L.part_mask
    fld = stress_field(union, L0.material_frame, part.spec)
    o = orientation(fld, max(int(round(2.0 / L0.material_frame.p)), 1))
    th = np.where(o, np.pi / 2, 0.0).reshape(-1)[m.cells]
    return m, th, np.full(m.cells.size, part.infill_fraction()), fld


def test_rotation_and_tsai_wu_match_the_evaluator():
    from rltp.config import Material
    mat = Material()
    Q = q_matrix(mat)
    th = np.array([0.0, 0.3, np.pi / 4, 1.2, 2.5])
    T = strain_rot(th)
    for i, t in enumerate(th):
        assert np.allclose(T[i].T @ Q @ T[i], q_rotate(Q, t), atol=1e-9 * Q[0, 0])
    rng = np.random.default_rng(0)
    eps = rng.normal(size=(5, 3)) * 1e-3
    for i, t in enumerate(th):
        sg = q_rotate(Q, t) @ eps[i]
        assert tsai_wu_grad(Q @ T[i] @ eps[i], mat)[0] == pytest.approx(
            tsai_wu_fi(*to_material(*sg, t), mat), rel=1e-12)
    s = rng.normal(size=(4, 3)) * 20
    _, g = tsai_wu_grad(s, mat)
    for j in range(3):
        d = np.zeros(3)
        d[j] = 1e-6
        fd = (tsai_wu_grad(s + d, mat)[0] - tsai_wu_grad(s - d, mat)[0]) / 2e-6
        assert np.allclose(g[:, j], fd, rtol=1e-6, atol=1e-9)


def test_field_model_matches_the_evaluator_on_a_fine_grid():
    """Same directions, uniform infill: the field model and the evaluator agree closely on
    0.1 mm cells (on 0.4 mm cells the zigzag's turn cells, checked both ways, add a few %)."""
    part = _fine_bracket("stress")
    m, th, rho, _ = _start_fields(part)
    assert m.evaluate(th, rho, False).fi_agg == pytest.approx(part.breakdown()["fi_agg"], rel=0.03)


def test_field_gradient_matches_finite_differences():
    part = _coarse_part()
    m, th, rho, _ = _start_fields(part)
    ev = m.evaluate(th, rho)
    for name, g in (("theta", ev.g_theta), ("rho", ev.g_rho)):
        for i in np.argsort(-np.abs(g))[:3]:
            h = 1e-5
            a, b = (th.copy(), th.copy()) if name == "theta" else (rho.copy(), rho.copy())
            a[i] += h
            b[i] -= h
            Jp = (m.evaluate(a, rho, False) if name == "theta" else m.evaluate(th, a, False)).J
            Jm = (m.evaluate(b, rho, False) if name == "theta" else m.evaluate(th, b, False)).J
            assert g[i] == pytest.approx((Jp - Jm) / (2 * h), rel=1e-3)


def test_optimize_fields_lowers_fi_and_holds_the_infill():
    part = _coarse_part()
    _, _, _, fld = _start_fields(part)
    res = optimize_fields(part, fld["theta"], 0.3, 0.1, sigma_mm=0.8, iters=12)
    assert res.fi_end < 0.97 * res.fi_start
    r = res.rho[res.rho_cells]
    w = np.zeros(res.rho.shape)
    w.reshape(-1)[res.model.cells] = res.model.npn
    assert (w * res.rho).sum() / w.sum() == pytest.approx(0.3, abs=2e-3)
    assert r.min() >= 0.1 - 1e-9 and r.max() <= 1 + 1e-9


def test_dither_ranks_nest_and_keep_the_sparsest_lines():
    S = 10
    rk = dither_ranks(200, S)
    base = np.flatnonzero(rk < 0)
    assert (np.diff(base) == S).all() and base[0] == 0
    prev = None
    for d in np.linspace(0.1, 1.0, 19):
        on = rk < d
        assert abs(on.mean() - d) <= 0.5 / S + 1e-9
        if prev is not None:
            assert (on >= prev).all()                    # lines laid at d stay laid at more
        prev = on
        gaps = np.diff(np.flatnonzero(on))
        assert gaps.max() <= S


def test_fill_map_puts_lines_where_the_field_is_dense():
    """A fill map dense on the left and sparse on the right lays more lines on the left,
    keeps the support rule, and averages to the map's mean."""
    part = _coarse_part("slicer", infill=0.3)
    pr = part.spec.printer
    from rltp.cover.shell import fill_sparse, solid_masks, support_radius
    L = part.layers[2]
    part_m = L.part.copy()
    fmap = np.where(np.arange(L.W)[None, :] < L.W // 3, 0.8, 0.1) * np.ones((L.H, 1))
    radius = support_radius(pr.max_bridge, L.frame.p)
    solids = solid_masks([X.part for X in part.layers], pr.solid_layers)
    fill_sparse(L, part_m, solids[2], pr.walls, 0.3, 0, 2 * radius, 0, None, None, fmap)
    opt = L.part & ~L.req
    left = opt & (np.arange(L.W)[None, :] < L.W // 3)
    right = opt & ~left
    assert (L.mask & left).sum() / left.sum() > 0.6
    assert (L.mask & right).sum() / right.sum() < 0.35          # the sparsest lines: every 4th row
    assert check_invariants(L) == []


def test_gradient_start_is_stronger_than_the_stress_start():
    stress = _fine_bracket("stress")
    grad = _fine_bracket("gradient")
    si = grad.start_info
    assert si["fi_agg"] < stress.breakdown()["fi_agg"]
    assert si["field_fi_end"] < si["field_fi_start"]
    assert si["stress_start_fi"] == pytest.approx(stress.breakdown()["fi_agg"], rel=1e-9)
    b = grad.breakdown()
    assert b["unsupported"] == 0 and abs(b["infill_pct"] - 40) < 8
    for L in grad.layers:
        assert check_invariants(L) == []
    assert unsupported(grad.layers, grad.support) == [0] * len(grad.layers)
    # several starts, best first by rounded FI_agg; the best two laid out for chains
    assert [s["start"] for s in si["starts"]][0] == si["start"] and len(si["starts"]) == 3
    assert si["ranked_by"] == "fi_rounded"
    r = [s["fi_rounded"] for s in si["starts"]]
    assert r == sorted(r)
    assert len(grad.start_snaps) == 2 and grad.start_names[0] == si["start"]


def test_gradient_start_ranks_by_the_whole_score_when_given_weights():
    grad = _fine_bracket("gradient", FieldOptions(starts=3, keep=3, workers=1, iters=15,
                                                  weights=(0.5, 1, 1, 0, 0, 1)))
    si = grad.start_info
    t = [s["total"] for s in si["starts"]]
    assert si["ranked_by"] == "score" and t == sorted(t, reverse=True)
    # the kept snapshots restore to layouts that score as ranked
    tot = []
    for snap in grad.start_snaps:
        grad.restore(snap)
        tot.append(grad.raw())
    assert len(tot) == 3


def test_continuation_stages_end_at_the_final_filter_and_p():
    st = stages_for(0.75, 8.0, 60, True)
    assert st[-1] == (0.75, 8.0, 60) and st[0][0] > st[1][0] > 0.75 and st[0][1] < st[1][1] < 8
    assert stages_for(0.75, 8.0, 60, False) == [(0.75, 8.0, 60)]
    part = _coarse_part()
    _, _, _, fld = _start_fields(part)
    res = optimize_fields(part, fld["theta"], 0.3, 0.1, sigma_mm=0.8, iters=9, continuation=True)
    assert {h["stage"] for h in res.history} >= {0, 2}
    assert res.fi_end < res.fi_start


def test_starts_cover_the_stress_map_axes_and_random_fields():
    names = start_names(8)
    assert names[:6] == ["stress", "stress90", "0", "90", "45", "135"] and names[6:] == ["random0", "random1"]
    th = np.random.default_rng(0).uniform(-1, 1, (30, 40))
    assert np.allclose(start_field("stress90", th, 0.1), th + np.pi / 2)
    assert np.allclose(start_field("45", th, 0.1), np.pi / 4)
    r0, r1 = start_field("random0", th, 0.1), start_field("random1", th, 0.1)
    assert r0.shape == th.shape and not np.allclose(r0, r1)
    assert np.allclose(r0, start_field("random0", th, 0.1))          # seeded


def test_line_ends_count_density_steps_and_direction_boundaries():
    H, W = 40, 80
    mask = np.ones((H, W), bool)
    none, two = np.zeros((H, W)), np.full((H, W), 2.0)
    rho = np.full((H, W), 0.1)
    rho[:, 40:] = 1.0                                   # a density step across x
    e0 = line_ends(np.zeros((H, W)), rho, mask, none, two, 4)[0]            # lines along x cross it
    e90 = line_ends(np.full((H, W), np.pi / 2), rho, mask, none, two, 4)[0]  # lines along y run beside it
    assert e0 == pytest.approx(2 * (H / 4) * 0.9, rel=0.01) and e90 < 1e-6
    # direction boundaries, uniform fill (1 solid layer + 2 sparse at 0.4 = 1.8 layers of beads)
    flat, one = np.full((H, W), 0.4), np.ones((H, W))
    th = np.zeros((H, W))
    th[:, 40:] = np.pi / 2                              # x-lines run into a y area: every one ends
    assert line_ends(th, flat, mask, one, two, 4)[0] == pytest.approx((H / 4) * 1.8, rel=0.01)
    th = np.zeros((H, W))
    th[20:, :] = np.pi / 2                              # boundary along x: only the y-lines end
    assert line_ends(th, flat, mask, one, two, 4)[0] == pytest.approx((W / 4) * 1.8, rel=0.01)
    rng = np.random.default_rng(0)
    from scipy import ndimage
    th = ndimage.gaussian_filter(rng.uniform(0, np.pi, (H, W)), 4)
    rr = np.clip(ndimage.gaussian_filter(rng.uniform(0, 1, (H, W)), 3) * 3 - 1, 0.1, 1)
    E, gt, gr = line_ends(th, rr, mask, one, two, 4)
    for g, x, which in ((gt, th, 0), (gr, rr, 1)):
        for i in (rng.integers(H * W) for _ in range(4)):
            a, b = x.copy(), x.copy()
            a.reshape(-1)[i] += 1e-6
            b.reshape(-1)[i] -= 1e-6
            args_a = (a, rr) if which == 0 else (th, a)
            args_b = (b, rr) if which == 0 else (th, b)
            fd = (line_ends(*args_a, mask, one, two, 4)[0] - line_ends(*args_b, mask, one, two, 4)[0]) / 2e-6
            assert g.reshape(-1)[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_line_end_cost_trades_strength_for_fewer_ends():
    part = _coarse_part()
    m, _, _, fld = _start_fields(part)
    H, W = part.layers[0].H, part.layers[0].W
    sp, so = np.zeros(H * W), np.zeros(H * W)
    sp[m.cells], so[m.cells] = m.npn, m.ns
    sp, so = sp.reshape(H, W), so.reshape(H, W)
    out = []
    for cost in (0.0, 0.02):
        r = optimize_fields(part, fld["theta"], 0.3, 0.25, sigma_mm=0.8, iters=12, model=m, end_cost=cost)
        out.append((line_ends(r.theta, r.rho, r.theta_cells, so, sp, 1)[0], r.fi_end))
    (e_free, fi_free), (e_cost, fi_cost) = out
    assert e_cost < 0.8 * e_free and fi_cost >= fi_free - 1e-9


def test_multi_start_returns_best_first():
    part = _coarse_part()
    _, _, _, fld = _start_fields(part)
    res = multi_start(part, fld["theta"], 0.3, 0.25, FieldOptions(starts=3, workers=1, iters=6))
    assert [r.start for r in res] != [] and len(res) == 3
    assert [r.fi_rounded for r in res] == sorted(r.fi_rounded for r in res)
    assert all(r.classes is not None and r.model is None for r in res)


def test_classes_round_and_drop_small_patches():
    th = np.zeros((40, 40))
    th[:, 20:] = np.pi / 2
    th[5:7, 5:7] = np.pi / 2                     # a 4-cell patch inside the 0 degree area
    th[30:40, 0:10] = np.pi / 4                  # a 100-cell diagonal patch
    mask = np.ones_like(th, bool)
    c = classes(th, mask, diagonals=False, min_axis=10)
    assert (c[:, 25:] == 90).all() and c[6, 6] == 0
    c4 = classes(th, mask, diagonals=True, min_axis=10, min_diag=50)
    assert (c4[32:38, 2:8] == 45).all()
    c4b = classes(th, mask, diagonals=True, min_axis=10, min_diag=200)
    assert not (c4b == 45).any()
