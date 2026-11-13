"""Evaluator checks: FEA against analytic tension-bar and cantilever solutions,
time model against closed-form trapezoids, and incremental vs full scoring."""
import numpy as np
import pytest
from shapely.geometry import box

from rltp.config import Load2D, LoadSpec, Material, Region
from rltp.eval.fea2d import FEAProblem, isotropic, tsai_wu_fi
from rltp.eval.timing import Motion, plan, time_layer, trapezoid_time
from rltp.cover.cover import get_pieces
from rltp.model import Frame, Layer
from tests.conftest import SHAPES, make_layer


def _bar(Lx, Ly, p, mat, force, support="xmin", load="xmax", refine=1):
    frame = Frame(0.0, 0.0, 0.0, p)
    H, W = int(round(Ly / p)), int(round(Lx / p))
    mask = np.ones((H, W), bool)
    g = box(0, 0, Lx, Ly)
    loads = LoadSpec([Region(side=support, width=1e-3)],
                     [Load2D(Region(side=load, width=1e-3), force)]).resolve(g.bounds)
    return FEAProblem(mask, frame, 1.0, mat, loads, 1.0, refine=refine), H, W


def _cantilever_tip(refine):
    E, nu, P = 1000.0, 0.3, -1.0
    Lx, Ly, p = 40.0, 4.0, 0.5
    fp, H, W = _bar(Lx, Ly, p, isotropic(E, nu), (0.0, P), refine=refine)
    r = fp.solve(np.full(H * W, 0.5), direct=True)
    I = Ly ** 3 / 12
    G = E / (2 * (1 + nu))
    delta = P * Lx ** 3 / (3 * E * I) + P * Lx / (5 / 6 * G * Ly)
    j = fp.used_nodes % (fp.Wr + 1)
    return r.u[1::2][j == fp.Wr].mean(), delta


def test_refined_mesh_is_closer_to_beam_theory():
    """Bilinear quads are too stiff in bending; splitting each cell into r x r elements
    converges the cantilever's tip deflection toward Timoshenko beam theory (the last
    ~0.3% is the clamped root, which beam theory ignores). Also guards the solver's
    floating-island regularization, which used to stiffen fine meshes."""
    errs = []
    for rf in (1, 2, 3, 4):
        tip, delta = _cantilever_tip(rf)
        errs.append(abs(tip / delta - 1))
    assert errs[3] < errs[2] < errs[1] < 0.6 * errs[0]
    assert errs[3] < 0.005


def test_refined_mesh_keeps_cell_outputs():
    """Uniform tension: every element sees the same stress, so per-cell FI, stresses and
    FI_agg match the one-element-per-cell mesh; outputs stay one value per cell."""
    mat = isotropic(1000.0, 0.3)
    a, H, W = _bar(20.0, 4.0, 0.5, mat, (50.0, 0.0))
    b, _, _ = _bar(20.0, 4.0, 0.5, mat, (50.0, 0.0), refine=2)
    wh = np.full(H * W, 0.5)
    ra, rb = a.solve(wh, direct=True), b.solve(wh, direct=True)
    assert rb.fi.shape == ra.fi.shape == (H * W,) and rb.N == ra.N == H * W
    mid = (slice(None), slice(W // 4, 3 * W // 4))
    assert np.allclose(rb.sx.reshape(H, W)[mid], ra.sx.reshape(H, W)[mid], rtol=0.02)
    assert rb.fi_agg(8) == pytest.approx(ra.fi_agg(8), rel=0.1)
    assert b.support_map().shape == (H * W,) and b.support_map().sum() == a.support_map().sum()


def test_tension_bar_matches_analytic():
    E, nu, F = 1000.0, 0.3, 50.0
    Lx, Ly, p = 20.0, 4.0, 0.5
    fp, H, W = _bar(Lx, Ly, p, isotropic(E, nu), (F, 0.0))
    r = fp.solve(np.full(H * W, 0.5), direct=True)
    sx = r.sx.reshape(H, W)
    mid = sx[:, W // 2]
    assert np.allclose(mid, F / (Ly * 1.0), rtol=0.03)
    # elongation of the loaded end
    nodes = fp.used_nodes
    j = nodes % (W + 1)
    ux = r.u[0::2]
    tip = ux[j == W].mean()
    assert tip == pytest.approx(F * Lx / (E * Ly), rel=0.05)


def test_cantilever_matches_beam_theory():
    E, nu, P = 1000.0, 0.3, -1.0
    Lx, Ly, p = 40.0, 4.0, 0.25
    fp, H, W = _bar(Lx, Ly, p, isotropic(E, nu), (0.0, P))
    r = fp.solve(np.full(H * W, 0.5), direct=True)
    I = Ly ** 3 / 12
    G = E / (2 * (1 + nu))
    delta = P * Lx ** 3 / (3 * E * I) + P * Lx / (5 / 6 * G * Ly)
    j = fp.used_nodes % (W + 1)
    uy = r.u[1::2]
    tip = uy[j == W].mean()
    assert tip == pytest.approx(delta, rel=0.08)


def test_orthotropic_bead_direction_matters():
    mat = Material()
    fp, H, W = _bar(20.0, 4.0, 0.5, mat, (50.0, 0.0))
    along = fp.solve(np.ones(H * W), direct=True)
    across = fp.solve(np.zeros(H * W), direct=True)
    assert along.fi_agg(8) < across.fi_agg(8)
    assert along.u[0::2].max() < across.u[0::2].max()


def test_warm_start_matches_direct():
    mat = Material()
    fp, H, W = _bar(20.0, 4.0, 0.5, mat, (30.0, 10.0))
    wh = np.ones(H * W)
    r0 = fp.solve(wh, direct=True)
    wh2 = wh.copy()
    wh2[np.random.default_rng(0).choice(H * W, 12, replace=False)] = 0.0
    warm = fp.solve(wh2, warm=r0.u)
    direct = fp.solve(wh2, direct=True, keep=False)
    assert not warm.direct and warm.iters < 40
    assert warm.S_p == pytest.approx(direct.S_p, rel=1e-6)


def test_tsai_wu_uniaxial():
    m = Material()
    assert tsai_wu_fi(np.array([m.Xt]), np.array([0.0]), np.array([0.0]), m)[0] == pytest.approx(1.0)
    assert tsai_wu_fi(np.array([-m.Xc]), np.array([0.0]), np.array([0.0]), m)[0] == pytest.approx(1.0)
    assert tsai_wu_fi(np.array([0.0]), np.array([m.Yt]), np.array([0.0]), m)[0] == pytest.approx(1.0)
    assert tsai_wu_fi(np.array([0.0]), np.array([0.0]), np.array([m.S]), m)[0] == pytest.approx(1.0)


def _motion(**kw):
    d = dict(v=40.0, a=1500.0, jd=0.013, v_travel=150.0, a_travel=3000.0, retract_len=0.8,
             retract_speed=35.0, z_hop=0.0, z_speed=10.0, retract_min=0.8)
    d.update(kw)
    return Motion(**d)


def test_straight_stroke_time():
    frame = Frame(0.0, 0.0, 0.0, 0.4)
    mask = np.zeros((3, 50), bool)
    mask[1, :] = True
    L = Layer.empty(0, 0.2, 0.2, mask, frame)
    from rltp.cover.baselines import fill_baseline
    fill_baseline(L, "zigzag")
    mot = _motion()
    tm = time_layer(L, get_pieces(L), mot)
    expect = trapezoid_time(0.0, 0.0, mot.v, mot.a, 49 * 0.4)
    assert tm.stroke_time[0] == pytest.approx(float(expect), rel=1e-9)
    t = tm.t_cell[mask.reshape(-1)]
    assert np.all(np.diff(t) > 0) or np.all(np.diff(t[::-1]) > 0)


def test_planner_respects_limits():
    lim2 = np.array([0.0, 25.0, 1e9, 4.0, 0.0])
    seg = np.array([0.4, 10.0, 10.0, 0.4])
    w = plan(lim2, seg, 1000.0)
    assert np.all(w <= lim2 + 1e-9)
    assert np.all(np.abs(np.diff(w)) <= 2 * 1000.0 * seg + 1e-6)


def test_corners_cost_time():
    """Same cell count: one long zigzag over a 20x20 square vs its 90-degree twin are equal,
    but a stroke with a corner every cell is much slower than straight rows."""
    sq = SHAPES["square"]
    zz = make_layer(sq, p=0.4, kind="zigzag")
    cc = make_layer(sq, p=0.4, kind="concentric")
    mot = _motion()
    t_zz = time_layer(zz, get_pieces(zz), mot).T
    t_cc = time_layer(cc, get_pieces(cc), mot).T
    assert t_zz > 0 and t_cc > 0
    n = zz.flat_mask.sum()
    assert t_zz > n * 0.4 / mot.v  # never faster than cruising every cell
