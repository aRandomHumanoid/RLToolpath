"""LayerEnv, observation, network and policy proposer."""
import numpy as np
import pytest
import torch

from rltp.cover.cover import check_invariants
from rltp.cover.ops import N_OPS, sample_valid
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
