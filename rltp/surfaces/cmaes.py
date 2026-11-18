"""A compact CMA-ES (Hansen's (mu/mu_w, lambda) with rank-one and rank-mu covariance updates and
cumulative step-size adaptation), ask/tell, minimising. Enough for the ribbon search's few hundred
parameters; no restarts or boundary handling (the warp bounds itself with tanh)."""
from __future__ import annotations

import numpy as np


class CMAES:
    def __init__(self, x0, sigma: float, popsize: int | None = None, seed: int = 0):
        x0 = np.asarray(x0, float)
        n = x0.size
        self.n = n
        self.lam = int(popsize or 4 + int(3 * np.log(n)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        me = self.mueff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.damps = 1 + 2 * max(0.0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean = x0.copy()
        self.sigma = float(sigma)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.gen = 0
        self._eig_gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        """lam candidate points (lam x n)."""
        z = self.rng.standard_normal((self.lam, self.n))
        self._y = z @ (self.B * self.D).T
        return self.mean + self.sigma * self._y

    def tell(self, xs: np.ndarray, fs) -> None:
        """Update from the asked points `xs` and their values `fs` (lower is better)."""
        n = self.n
        fs = np.asarray(fs, float)
        idx = np.argsort(fs, kind="stable")[: self.mu]
        y = (np.asarray(xs)[idx] - self.mean) / self.sigma
        yw = self.w @ y
        self.mean = self.mean + self.sigma * yw
        Cinv_half = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinv_half @ yw)
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN
                < 1.4 + 2 / (n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        if self.gen - self._eig_gen > self.lam / (self.c1 + self.cmu) / n / 10:
            self._eig_gen = self.gen
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            d2, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(d2, 1e-20))


class LocalSearch:
    """Sparse moves around the best point so far, for rugged scores where moving every parameter
    at once flips discrete decisions everywhere: each candidate perturbs 1-3 random `blocks`
    (index arrays: e.g. one warp node's two components, one skin parameter) by sigma x N(0, 1) and
    the best candidate replaces the incumbent only if it is better. sigma follows the generation's
    success rate (target 1/5, time constant ~10 generations), within [sigma0 / 8, sigma0 x 4].
    ask/tell, minimising; `f0` is the starting point's value."""

    def __init__(self, x0, sigma: float, popsize: int, seed: int, f0: float, blocks: list):
        self.mean = np.asarray(x0, float).copy()
        self.n = self.mean.size
        self.lam = int(popsize)
        self.f = float(f0)
        self.sigma = self.sigma0 = float(sigma)
        self.blocks = [np.asarray(b, np.int64) for b in blocks]
        self.gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        X = np.repeat(self.mean[None, :], self.lam, axis=0)
        for i in range(self.lam):
            for j in self.rng.choice(len(self.blocks), size=int(self.rng.integers(1, 4)), replace=False):
                b = self.blocks[j]
                X[i, b] += self.sigma * self.rng.standard_normal(b.size)
        return X

    def tell(self, xs: np.ndarray, fs) -> None:
        fs = np.asarray(fs, float)
        i = int(np.argmin(fs))
        rate = float(np.mean(fs < self.f))
        if fs[i] < self.f:
            self.mean = np.asarray(xs)[i].copy()
            self.f = float(fs[i])
        self.gen += 1
        self.sigma = float(np.clip(self.sigma * np.exp(0.1 * (rate - 0.2) / 0.2), self.sigma0 / 8, self.sigma0 * 4))


class ElitistES:
    """A (1+lambda) evolution strategy: every generation samples lam points around the best point
    so far (the incumbent) and moves there only if one is better, so a lucky find is never lost
    (CMA-ES's mean can drift off it on a rugged score). Step size by the smoothed success rule of
    the (1+lambda)-CMA-ES (Igel, Suttorp & Hansen 2006); in a few hundred dimensions its covariance
    learning is negligible, so the search is isotropic. After `stall` generations without
    improvement the step resets to its starting value, a restart around the incumbent. ask/tell,
    minimising, like CMAES; `f0` is the starting point's value."""

    def __init__(self, x0, sigma: float, popsize: int = 12, seed: int = 0, f0: float = np.inf,
                 stall: int = 12):
        x0 = np.asarray(x0, float)
        n = x0.size
        self.n = n
        self.lam = int(popsize)
        self.mean = x0.copy()
        self.f = float(f0)
        self.sigma = self.sigma0 = float(sigma)
        self.d = 1 + n / (2 * self.lam)
        self.ptarget = 1 / (5 + np.sqrt(self.lam) / 2)
        self.cp = self.ptarget * self.lam / (2 + self.ptarget * self.lam)
        self.psucc = self.ptarget
        self.stall = stall
        self.since = 0
        self.gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        return self.mean + self.sigma * self.rng.standard_normal((self.lam, self.n))

    def tell(self, xs: np.ndarray, fs) -> None:
        fs = np.asarray(fs, float)
        i = int(np.argmin(fs))
        wins = int(np.sum(fs < self.f))
        if fs[i] < self.f:
            self.mean = np.asarray(xs)[i].copy()
            self.f = float(fs[i])
            self.since = 0
        else:
            self.since += 1
        self.gen += 1
        self.psucc = (1 - self.cp) * self.psucc + self.cp * wins / self.lam
        self.sigma *= np.exp((self.psucc - self.ptarget) / (self.d * (1 - self.ptarget)))
        if self.since >= self.stall:
            self.sigma = self.sigma0
            self.since = 0
