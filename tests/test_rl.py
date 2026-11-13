"""LayerEnv, observation, network and policy proposer."""
import numpy as np
import pytest
import torch

from rltp.cover.cover import check_invariants
from rltp.cover.ops import N_PATH_OPS as N_OPS, sample_valid
from rltp.rl.env import LayerEnv
from rltp.rl.net import PolicyProposer, UNet, masked_logits
from rltp.rl.obs import N_CH, build_obs


@pytest.mark.parametrize("terms,grid", [(("strength",), 32), (("strength", "time", "cont", "inter"), 32)])
def test_env_episode(terms, grid):
    env = LayerEnv(grid=grid, terms=terms, episode_len=12)
    obs, info = env.reset(seed=1)
    assert obs.shape == (N_CH, grid, grid) and obs.dtype == np.float32
    assert info["action_mask"].shape == (N_OPS * grid * grid,)
    assert np.array_equal(info["action_mask"], env.action_masks())
    rng = np.random.default_rng(0)
    for t in range(12):
        a = sample_valid(info["action_mask"].reshape(N_OPS, -1), rng)[0]
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r) and not info["invalid"]
    assert trunc and "episode" in info
    assert check_invariants(env.part.layers[env.k]) == []


def test_env_invalid_action_penalised():
    env = LayerEnv(grid=32, terms=("strength",), ops=["flip_right"])
    _, info = env.reset(seed=2)
    bad = int(np.flatnonzero(~info["action_mask"])[0])
    _, r, _, _, info = env.step(bad)
    assert r == -1.0 and info["invalid"]


def test_unet_any_size_and_masking():
    m = UNet()
    for H, W in [(32, 32), (19, 45)]:
        lg, v = m(torch.zeros(2, N_CH, H, W))
        assert lg.shape == (2, N_OPS, H, W) and v.shape == (2,)
    lg, _ = m(torch.randn(1, N_CH, 8, 8))
    mask = torch.zeros(1, N_OPS * 64, dtype=torch.bool)
    mask[0, 5] = True
    assert int(masked_logits(lg, mask).argmax()) == 5


def test_policy_proposer_in_loop():
    from rltp.loop.optimize import LoopConfig, optimize_part
    env = LayerEnv(grid=32, terms=("strength", "time", "cont"))
    env.reset(seed=3)
    prop = PolicyProposer(UNet())
    res = optimize_part(env.part, prop, env.w, LoopConfig(max_evals=64, burst=16, warmup=16))
    assert res["evals"] <= 64 + 1
    assert check_invariants(env.part.layers[0]) == []


def test_obs_unpadded_matches_layer_size():
    env = LayerEnv(grid=32, terms=("strength",))
    env.reset(seed=4)
    o = build_obs(env.part, 0, env.w)
    L = env.part.layers[0]
    assert o.shape == (N_CH, L.H, L.W)


def test_env_rejects_worsening_edits_like_the_loop():
    env = LayerEnv(grid=32, terms=("strength",))
    _, info = env.reset(seed=5)
    rng = np.random.default_rng(0)
    seen = {True: 0, False: 0}
    for _ in range(60):
        a = int(sample_valid(info["action_mask"].reshape(N_OPS, -1), rng)[0])
        before = env.part.layers[0].nxt.copy()
        expected = env.score_actions([a])[0]
        _, r, _, trunc, info = env.step(a)
        seen[info["accepted"]] += 1
        if info["accepted"]:
            assert r > 0 and r == pytest.approx(expected, rel=1e-3, abs=1e-9)
            assert not env.banned
        else:
            assert r == 0.0 and expected < 1e-6
            assert np.array_equal(env.part.layers[0].nxt, before)
            assert not info["action_mask"][a]          # banned until the next accepted edit
        if trunc:
            break
    assert seen[True] and seen[False]


def test_env_accept_all_applies_worsening_edits():
    env = LayerEnv(grid=32, terms=("strength",), accept="all")
    _, info = env.reset(seed=5)
    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(30):
        a = int(sample_valid(info["action_mask"].reshape(N_OPS, -1), rng)[0])
        _, r, _, _, info = env.step(a)
        assert info["accepted"]
        rewards.append(r)
    assert min(rewards) < 0


def test_relative_scores_are_fractions_of_baseline():
    env = LayerEnv(grid=32, terms=("strength", "time", "cont"))
    env.reset(seed=6)
    part = env.part
    raw0, sc = part.raw(), part.norm.scales()
    assert sc["strength"] == 1.0 and sc["inter"] == 1.0
    assert sc["time"] == pytest.approx(-raw0["time"]) and sc["cont"] == pytest.approx(-raw0["cont"])
    # 1% less FI (log), 2% more time, 3% less continuity cost
    d = {"strength": 0.01, "time": -0.02 * sc["time"], "cont": 0.03 * sc["cont"], "inter": 0.0}
    assert part.norm.total(d, [1.0, 1.0, 1.0, 0.0]) == pytest.approx(0.01 - 0.02 + 0.03)
    # the scales survive a save/load (bench ships them to workers)
    from rltp.eval.metrics import Normalizer
    n2 = Normalizer(mode="std").load(part.norm.state())
    assert n2.mode == "relative" and n2.scales() == sc
    legacy = {k: v for k, v in part.norm.state().items() if k not in ("mode", "ref")}
    assert Normalizer().load(legacy).mode == "std"


def test_std_norm_env_keeps_running_scales():
    env = LayerEnv(grid=32, terms=("strength",), norm="std")
    env.reset(seed=7)
    env.warmup_norm(16)
    assert env.part.norm.mode == "std" and env.scorer is env.norm
    assert env.norm.n["strength"] >= 2 and env.edit_scale() == 1.0


@pytest.mark.parametrize("norm", ["relative", "std"])
def test_teacher_smoke(norm):
    from rltp.rl.imitate import collect_teacher
    ek = {"grid": 32, "terms": ["strength"], "ops": ["flip_right", "flip_down", "patch5_0"], "norm": norm}
    data = collect_teacher(ek, n_states=6, n_cand=8, sa_steps=12, workers=1, seed=3)
    assert len(data["label"]) == 6
    for lab, cand, gain in zip(data["label"], data["cand"], data["gain"]):
        ok = cand >= 0
        assert lab in cand[ok] and np.isfinite(gain[ok]).all()
        assert gain[ok].max() == gain[ok][list(cand[ok]).index(lab)]
