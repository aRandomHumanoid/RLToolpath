"""U-Net policy and value heads.

4 levels (32 -> 64 -> 128 -> 256 channels), two 3x3 convs per level, GroupNorm,
SiLU. A global-pooling branch at the bottleneck is broadcast back to every cell
so context stays part-wide on grids larger than any seen in training. The
network is fully convolutional: any grid size works (padded to a multiple of 8).
Policy head: 1x1 conv to one logit per operator per cell. Value head: global
average pool of the bottleneck, then a small MLP.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rltp.cover.ops import N_OPS as N_ALL_OPS, N_PATH_OPS as N_OPS
from rltp.rl.obs import N_CH, build_obs
from rltp.search.sa import Proposer

NEG = -1e9


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _gn(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = N_CH, n_ops: int = N_OPS, chs=(32, 64, 128, 256)):
        super().__init__()
        self.cfg = {"in_ch": in_ch, "n_ops": n_ops, "chs": list(chs)}
        c1, c2, c3, c4 = chs
        self.e1, self.e2, self.e3, self.e4 = Block(in_ch, c1), Block(c1, c2), Block(c2, c3), Block(c3, c4)
        self.glob = nn.Sequential(nn.Linear(c4, c4), nn.SiLU(), nn.Linear(c4, c4))
        self.d3, self.d2, self.d1 = Block(c4 + c3, c3), Block(c3 + c2, c2), Block(c2 + c1, c1)
        self.pi = nn.Conv2d(c1, n_ops, 1)
        self.v = nn.Sequential(nn.Linear(c4, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        Hp, Wp = -(-H // 8) * 8, -(-W // 8) * 8
        if (Hp, Wp) != (H, W):
            x = F.pad(x, (0, Wp - W, 0, Hp - H))
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.e4(F.max_pool2d(e3, 2))
        g = b.mean((2, 3))
        b = b + self.glob(g)[:, :, None, None]
        value = self.v(b.mean((2, 3))).squeeze(-1)
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        logits = self.pi(d1)[:, :, :H, :W]
        return logits, value


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, n_ops, H, W) logits + (B, n_ops*H*W) bool mask -> (B, n_ops*H*W)."""
    flat = logits.reshape(logits.shape[0], -1)
    return flat.masked_fill(~mask, NEG)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_policy(path, model: UNet, meta: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "config": model.cfg, "meta": meta or {}}, path)


def load_policy(path, device="cpu") -> tuple[UNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = UNet(cfg["in_ch"], cfg["n_ops"], tuple(cfg["chs"]))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck.get("meta", {})


class PolicyProposer(Proposer):
    """Frozen policy inside the part loop: masked argmax (or a sample) over operator x cell."""
    name = "policy"
    temp0 = 0.0

    def __init__(self, model: UNet, sample: bool = False, device: str = "cpu", ops=None):
        from rltp.cover.ops import ops_to_indices
        self.model = model
        self.sample = sample
        self.device = device
        self.op_keep = np.zeros(N_ALL_OPS, bool)
        self.op_keep[[i for i in ops_to_indices(ops) if i < N_OPS]] = True   # only what it was trained on
        if sample:
            self.name = "policy-sample"

    @staticmethod
    def load(path, sample=False, device="cpu") -> "PolicyProposer":
        model, meta = load_policy(path, device)
        ops = meta.get("ops") or meta.get("env", {}).get("ops")
        return PolicyProposer(model, sample, device, ops)

    @torch.no_grad()
    def propose(self, part, k, M, w, rng):
        obs = build_obs(part, k, w)
        x = torch.from_numpy(obs[None]).to(self.device)
        logits, _ = self.model(x)
        lg_path = logits.reshape(N_OPS, -1).cpu().numpy().astype(np.float64)
        lg = np.full((N_ALL_OPS, lg_path.shape[1]), -np.inf)   # fill edits: not in its action space
        lg[:N_OPS] = lg_path
        lg[~self.op_keep] = -np.inf
        lg = lg.reshape(-1)
        lg[~M] = -np.inf
        if not np.isfinite(lg).any():   # nothing the policy knows is valid: fall back to any valid edit
            lg = np.where(M, 0.0, -np.inf)
        if self.sample:
            z = lg - lg[np.isfinite(lg)].max()
            pr = np.exp(z)
            pr /= pr.sum()
            a = int(rng.choice(lg.size, p=pr))
        else:
            a = int(np.argmax(lg))
        return a, None
