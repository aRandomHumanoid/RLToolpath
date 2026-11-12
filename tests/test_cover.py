"""Cover invariants under baselines and random edit sequences (hypothesis fuzzing)."""
import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from rltp.cover.baselines import BASELINES
from rltp.cover.cover import check_invariants, get_pieces, reverse_stroke
from rltp.cover.ops import N_OPS, OPS, action_mask, apply, apply_action, layer_diff, sample_valid
from tests.conftest import SHAPES, make_layer


def test_baselines_valid(shape):
    for kind in BASELINES:
        L = make_layer(shape, kind=kind)
        assert check_invariants(L) == [], kind
        pc = get_pieces(L)
        assert (pc.label[L.flat_mask] >= 0).all()


def test_zigzag_square_is_one_stroke():
    L = make_layer(SHAPES["square"], kind="zigzag")
    assert get_pieces(L).count == 1


def test_concentric_square_is_loops():
    L = make_layer(SHAPES["square"], kind="concentric")
    pc = get_pieces(L)
    assert pc.is_loop.sum() >= 5


def test_every_masked_action_applies(shape):
    for kind in BASELINES:
        L = make_layer(shape, kind=kind)
        M = action_mask(L)
        rng = np.random.default_rng(0)
        for op in range(N_OPS):
            idx = np.flatnonzero(M[op])
            for cell in rng.choice(idx, size=min(15, idx.size), replace=False) if idx.size else []:
                N, _ = apply(L, op, int(cell))
                errs = check_invariants(N)
                assert errs == [], (kind, OPS[op], int(cell), errs)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    shape=st.sampled_from(sorted(SHAPES)),
    kind=st.sampled_from(BASELINES),
    seed=st.integers(0, 2**31 - 1),
    steps=st.integers(1, 40),
)
def test_random_edit_sequences_keep_invariants(shape, kind, seed, steps):
    L = make_layer(SHAPES[shape], kind=kind)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        M = action_mask(L)
        a = sample_valid(M, rng)[0]
        L2, _ = apply_action(L, int(a))
        errs = check_invariants(L2)
        assert errs == [], (OPS[int(a) // L.n], errs)
        # diffs replay exactly
        d = layer_diff(L, L2)
        nxt = L.nxt.copy()
        nxt[d["cells"]] = d["nxt"]
        assert (nxt == L2.nxt).all()
        assert (L.seam - set(d["seam_del"])) | set(d["seam_add"]) == L2.seam
        L = L2


@settings(max_examples=40, deadline=None)
@given(shape=st.sampled_from(sorted(SHAPES)), kind=st.sampled_from(BASELINES), seed=st.integers(0, 10**6))
def test_reverse_twice_is_identity(shape, kind, seed):
    L = make_layer(SHAPES[shape], kind=kind)
    rng = np.random.default_rng(seed)
    cell = int(rng.choice(np.flatnonzero(L.flat_mask)))
    N = L.copy()
    reverse_stroke(N, cell)
    reverse_stroke(N, cell)
    assert (N.nxt == L.nxt).all() and (N.prv == L.prv).all() and N.seam == L.seam


def test_masked_invalid_actions_raise():
    from rltp.cover.ops import InvalidEdit
    L = make_layer(SHAPES["plate_hole"], kind="zigzag")
    M = action_mask(L)
    rng = np.random.default_rng(1)
    bad = np.flatnonzero(~M.reshape(-1))
    raised = 0
    for a in rng.choice(bad, 200, replace=False):
        op, cell = divmod(int(a), L.n)
        if op in (6, 7, 8, 9):
            continue
        try:
            N, _ = apply(L, op, cell)
        except InvalidEdit:
            raised += 1
        else:
            # some ops (reverse on single cells etc.) are masked as no-ops but still valid
            assert check_invariants(N) == []
    assert raised > 0
