"""Proposers: random edits, simulated annealing, best-of-N greedy, and the policy.

Every proposer plugs into the same part loop and evaluator, so comparisons are
at equal evaluator-call budgets. SA differs from random only in the loop's
temperature (it accepts some worsening edits early on).
"""
from __future__ import annotations

import numpy as np

from rltp.cover.ops import apply_action, sample_valid


class Proposer:
    name = "base"
    temp0 = 0.0

    def propose(self, part, k: int, M: np.ndarray, w, rng):
        """Return (action, Candidate or None). M: flat bool mask (n_ops * H * W)."""
        raise NotImplementedError


class RandomProposer(Proposer):
    name = "random"
    temp0 = 0.0

    def propose(self, part, k, M, w, rng):
        a = int(sample_valid(M, rng)[0])
        return a, None


class SAProposer(RandomProposer):
    name = "sa"

    def __init__(self, temp0: float = 0.1):
        self.temp0 = temp0


class GreedyProposer(Proposer):
    """Best of N random valid edits (each one costs an evaluator call)."""
    name = "greedy"
    temp0 = 0.0

    def __init__(self, n: int = 16):
        self.n = n

    def propose(self, part, k, M, w, rng):
        acts = np.unique(sample_valid(M, rng, self.n))
        best = None
        for a in acts:
            newL, _ = apply_action(part.layers[k], int(a))
            c = part.propose(k, newL)
            s = part.norm.total(c.d, w)
            if best is None or s > best[0]:
                best = (s, int(a), c)
        return best[1], best[2]


def make_proposer(name: str, policy_path=None, temp0=None, greedy_n: int = 16, sample: bool = False,
                  device: str = "cpu") -> Proposer:
    if name == "random":
        p = RandomProposer()
    elif name == "sa":
        p = SAProposer()
    elif name == "greedy":
        p = GreedyProposer(greedy_n)
    elif name in ("policy", "policy-sample"):
        sample = sample or name == "policy-sample"
        if policy_path is None:
            raise ValueError("--proposer policy needs --policy policy.pt")
        from rltp.rl.net import PolicyProposer
        p = PolicyProposer.load(policy_path, sample=sample, device=device)
    else:
        raise ValueError(f"unknown proposer {name!r}")
    if temp0 is not None:
        p.temp0 = float(temp0)
    return p
