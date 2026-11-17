"""Ribbon (surface) infill: rasterising, laying, merging, crossings, rings, the warp, CMA-ES."""
import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, box

from rltp.config import Load2D, LoadSpec, PartSpec, Printer, Region, ScoreConfig
from rltp.cover.beads import free_ends
from rltp.cover.cover import check_invariants, get_pieces
from rltp.part import build_part
from rltp.surfaces.cmaes import CMAES
from rltp.surfaces.layout import infill_region, is_infill_layer, lay, layout_snapshot, rasterize
from rltp.surfaces.ribbons import Ribbon, Warp, line_family, offset_rings


@pytest.fixture(scope="module")
def part():
    pr = Printer(walls=1, solid_layers=1, infill=0.4, max_bridge=2.0, cell=0.1, infill_rule="free")
    g = box(0, 0, 12, 8).difference(Point(4, 4).buffer(1.5, quad_segs=32))
    m = trimesh.creation.extrude_polygon(g, height=1.0)
    loads = LoadSpec([Region(box=[0, 0, 0.6, 8])], [Load2D(Region(box=[11.4, 0, 12, 8]), (0, -20))])
    return build_part(m, PartSpec(pr, loads, terms=("strength", "time", "cont"),
                                  score=ScoreConfig(fea_coarsen=2, directions="path")), baseline="zigzag")


def _infill(part):
    k = next(i for i, L in enumerate(part.layers) if is_infill_layer(L))
    return k, part.layers[k]


def _strokes(L):
    return len(get_pieces(L).cells)


def test_rasterize_gives_a_4_connected_chain_along_the_curve(part):
    _, L = _infill(part)
    th = np.linspace(0, 2 * np.pi, 200)
    for xy in (np.array([[1.0, 1.0], [9.0, 6.3]]), np.c_[6 + 2.5 * np.cos(th), 4 + 2.5 * np.sin(th)]):
        r, c = rasterize(xy, L.frame, L.H, L.W)
        steps = np.abs(np.diff(r)) + np.abs(np.diff(c))
        assert (steps == 1).all()
        x, y = L.frame.cell_xy(r * L.W + c, L.W)
        if len(xy) == 2:                       # every cell centre within a cell of the segment
            t, q = xy[1] - xy[0], np.c_[x, y] - xy[0]
            d = (t[0] * q[:, 1] - t[1] * q[:, 0]) / np.linalg.norm(t)
        else:
            d = np.hypot(x - 6, y - 4) - 2.5
        assert np.abs(d).max() <= L.frame.p


def test_lay_lines_is_valid_anchored_and_merges_ends(part):
    _, L = _infill(part)
    rbs = line_family(L.poly.bounds, 0.0, 1.0)
    A, B = L.copy(), L.copy()
    st = lay(A, rbs)
    lay(B, rbs, link_reach=0)
    assert check_invariants(A) == [] and check_invariants(B) == []
    assert free_ends(A) == []
    assert st.links > 0 and _strokes(A) < _strokes(B)
    infill = A.mask & ~A.lock
    assert infill.any() and not (infill & ~infill_region(A)).any()   # infill stays inside the walls
    assert (A.mask & A.lock).sum() == (L.mask & L.lock).sum()       # walls untouched


def test_crossing_ribbons_are_cut_into_t_joints(part):
    _, L = _infill(part)
    b = L.poly.bounds
    fam0, fam90 = line_family(b, 0.0, 2.0), line_family(b, 90.0, 2.0)
    alone = [lay(L.copy(), f, link_reach=0).pieces for f in (fam0, fam90)]
    A = L.copy()
    st = lay(A, fam0 + fam90, link_reach=0)
    assert check_invariants(A) == []
    assert free_ends(A) == []                         # every cut end touches the bead it met
    assert st.pieces > sum(alone)                     # the later family is cut at each crossing
    # no two strokes' nodes closer than a bead, except where a cut end meets the bead it touches
    pc = get_pieces(A)
    lab = pc.label.reshape(A.H, A.W)
    lock = A.lock
    rr, cc = np.nonzero(A.mask & ~lock)
    from scipy.spatial import cKDTree
    pts = np.c_[rr, cc].astype(float)
    pairs = cKDTree(pts).query_pairs(A.kb - 1.5, output_type="ndarray")
    assert all(lab[rr[i], cc[i]] == lab[rr[j], cc[j]] for i, j in pairs)


def test_rings_are_laid_as_loops_and_lines_end_against_them(part):
    _, L = _infill(part)
    rbs = offset_rings(L.poly, 0.65, 0.4, 2) + line_family(L.poly.bounds, 0.0, 1.0)
    A = L.copy()
    st = lay(A, rbs)
    assert check_invariants(A) == []
    assert st.loops >= 2                              # outline and hole, at least the first ring
    assert free_ends(A) == []


def test_layout_snapshot_lays_every_infill_layer_and_scores(part):
    rbs = line_family(_infill(part)[1].poly.bounds, 45.0, 1.0)
    snap, stats = layout_snapshot(part, rbs, alternate=True)
    ks = [k for k, L in enumerate(part.layers) if is_infill_layer(L)]
    assert len(stats) == (2 if len(ks) > 1 else 1)
    part.restore(snap)
    raw = part.raw()
    assert all(np.isfinite(v) for v in raw.values())
    for k in ks:
        assert check_invariants(part.layers[k]) == []


def test_apply_layout_scores_like_snapshot_and_restore(part):
    """The shared-cache fast path gives exactly the plain restore's score, and unsharing leaves
    every layer with its own cache."""
    from rltp.surfaces.layout import apply_layout, unshare_caches
    rbs = line_family(_infill(part)[1].poly.bounds, 30.0, 1.0)
    for alternate in (False, True):
        snap, _ = layout_snapshot(part, rbs, alternate=alternate)
        part.restore(snap)
        want = part.raw()
        apply_layout(part, rbs, alternate=alternate)
        got = part.raw()
        assert all(np.isclose(got[t], want[t], rtol=1e-9, atol=1e-12) for t in want)
        unshare_caches(part)
        ids = [id(L._cache) for L in part.layers]
        assert len(set(ids)) == len(ids)


def test_warp_is_identity_at_zero_and_bounded():
    xy = np.random.default_rng(0).uniform(0, 10, (50, 2))
    w = Warp((0, 0, 10, 10), 3.0, 0.8)
    assert np.allclose(w(np.zeros(w.n), xy), xy)
    big = w(np.full(w.n, 50.0), xy) - xy
    assert np.abs(big).max() <= 0.8 + 1e-9
    rb = [Ribbon(xy, fixed=True), Ribbon(xy)]
    out = w.apply(np.ones(w.n), rb)
    assert out[0] is rb[0] and not np.allclose(out[1].xy, xy)


def test_warp3d_starts_as_the_2d_warp_and_never_leans_past_its_limit():
    """Steps all 0: the middle level's 2D warp at every height. Saturated random steps: every
    ribbon point moves at most tan(max_lean) per unit height between any two heights."""
    from rltp.surfaces.ribbons import Warp3D
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 10, (80, 2))
    w3 = Warp3D((0, 0, 10, 10), 3.0, 1.0, 0.3, 2.3, levels=4, max_lean=30.0)
    w2 = w3.base
    base = rng.normal(0, 1, w2.n)
    x = np.r_[base, np.zeros(w3.n - w2.n)]
    for z in (0.0, 0.3, 1.1, 2.3, 3.0):
        assert np.allclose(w3(x, xy, z), w2(base, xy))
    x = np.r_[base, rng.choice([-50.0, 50.0], w3.n - w2.n)]
    zs = np.linspace(0.3, 2.3, 11)
    pts = [w3(x, xy, z) for z in zs]
    for (za, a), (zb, b) in zip(zip(zs, pts), zip(zs[1:], pts[1:])):
        assert np.hypot(*(b - a).T).max() <= np.tan(np.radians(30.0)) * (zb - za) + 1e-9
    rb = [Ribbon(xy)]
    assert w3.lean(x, rb, zs) <= 30.0 + 1e-6
    assert w3.lean(x, rb, zs) > 5.0                       # the steps do lean the surface


def test_3d_ribbons_lay_each_layer_from_its_own_height(part):
    """Ribbons given as a function of height: each infill layer is laid from its own cut, the
    layouts differ where the cuts do, and the fast path still scores like a plain restore."""
    from rltp.surfaces.layout import apply_layout, unshare_caches
    ks = [k for k, L in enumerate(part.layers) if is_infill_layer(L)]
    assert len(ks) >= 2
    b = part.layers[ks[0]].poly.bounds

    def rbs(z):
        return line_family(b, 20.0 * z, 1.0)          # turns with height

    snap, stats = layout_snapshot(part, rbs)
    assert len(stats) == len(ks)
    masks = [snap[k][3] for k in ks]
    assert any((m != masks[0]).any() for m in masks[1:])
    part.restore(snap)
    want = part.raw()
    apply_layout(part, rbs)
    got = part.raw()
    assert all(np.isclose(got[t], want[t], rtol=1e-9, atol=1e-12) for t in want)
    unshare_caches(part)
    for k in ks:
        assert check_invariants(part.layers[k]) == []


def test_cmaes_minimises_a_quadratic():
    es = CMAES(np.full(6, 3.0), 1.0, seed=1)
    for _ in range(250):
        X = es.ask()
        es.tell(X, (X ** 2).sum(1))
    assert (es.mean ** 2).sum() < 1e-6


def test_elitist_es_never_loses_its_best_and_converges():
    from rltp.surfaces.cmaes import ElitistES
    x0 = np.full(6, 3.0)
    es = ElitistES(x0, 1.0, popsize=8, seed=1, f0=float((x0 ** 2).sum()))
    best = es.f
    for _ in range(400):
        X = es.ask()
        es.tell(X, (X ** 2).sum(1))
        assert es.f <= best and np.isclose(es.f, (es.mean ** 2).sum())
        best = es.f
    assert es.f < 1e-4
