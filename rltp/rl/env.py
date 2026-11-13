"""Gymnasium LayerEnv: one layer, one proposed edit per step, reward = accepted gain.

reset samples a shape, loads, a weight vector w and a baseline cover. The
observation is padded to the curriculum stage's grid size; the action space is
Discrete(10 * G * G), flattened operator x cell, with action_masks() giving the
valid entries. With neighbors=True the editable layer sits between two layers
of the same outline (own baselines) and the v1.5 interlayer term is scored.

accept="improve" (default) is the part loop at temperature 0: a proposal that
does not raise the score is scored but not applied, earns 0, and is masked out
until the next accepted edit. A step is one evaluator call either way, so an
episode's return is the gain the loop would get from the policy in that many
calls. accept="all" applies every edit, worsening ones included (the original
env; PPO trained on it learned sequences the loop then rejected).

norm="relative" (default) scores each term as a fraction of this episode's
baseline, the same scales the loop uses. norm="std" divides by a running
per-edit std pooled over every episode this env has seen.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rltp.config import TERMS, PartSpec, Printer, ScoreConfig
from rltp.cover.baselines import BASELINES, fill_baseline
from rltp.cover.ops import N_PATH_OPS as N_OPS, InvalidEdit, action_mask, apply, ops_to_indices
from rltp.eval.metrics import NORM_MODES, Normalizer
from rltp.loop.optimize import accept as loop_accept
from rltp.model import Frame, Layer
from rltp.part import PartState
from rltp.rl.obs import N_CH, build_obs
from rltp.rl.shapes import TRAIN_FAMILIES, mesh_slice_shapes, random_loads, random_shape
from rltp.slicing.raster import rasterize


class LayerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid: int = 32, terms=("strength",), ops=None, episode_len: int = 64,
                 families=TRAIN_FAMILIES, neighbors: bool = False, p: float = 0.4,
                 min_cells: int = 40, mesh_dir: str | None = None, mesh_frac: float = 0.0,
                 fixed_w=None, baselines=BASELINES, printer: Printer | None = None, seed: int | None = None,
                 accept: str = "improve", norm: str = "relative"):
        super().__init__()
        if accept not in ("improve", "all"):
            raise ValueError(f"accept must be 'improve' or 'all', not {accept!r}")
        if norm not in NORM_MODES:
            raise ValueError(f"norm must be one of {NORM_MODES}, not {norm!r}")
        self.accept = accept
        self.norm_mode = norm
        self.banned: set = set()
        self.G = grid
        self.terms = tuple(terms)
        self.ops = ops_to_indices(ops)
        self.episode_len = episode_len
        self.families = tuple(families)
        self.neighbors = neighbors or ("inter" in self.terms)
        self.p = p
        self.min_cells = min_cells
        self.mesh_dir = mesh_dir
        self.mesh_frac = mesh_frac
        self.fixed_w = None if fixed_w is None else np.asarray(fixed_w, float)
        self.baselines = tuple(baselines)
        self.printer = printer or Printer()
        self.norm = Normalizer(mode="std")   # running scales across episodes, used when norm="std"
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_CH, grid, grid), np.float32)
        self.action_space = spaces.Discrete(N_OPS * grid * grid)
        self.part: PartState | None = None
        self.k = 0
        self.t = 0
        self.w = np.zeros(len(TERMS))
        self.episode_gain = 0.0
        self.shape_family = None
        if seed is not None:
            self.reset(seed=seed)

    # -------------------------------------------------------------- sampling

    def _sample_w(self, rng) -> np.ndarray:
        if self.fixed_w is not None:
            return self.fixed_w.copy()
        w = np.zeros(len(TERMS))
        for i, t in enumerate(TERMS):
            if t in self.terms:
                w[i] = 1.0 if len(self.terms) == 1 else rng.uniform(0.1, 1.0)
        if "strength" in self.terms and len(self.terms) > 1:
            w[0] = rng.uniform(0.5, 1.0)
        return w

    def _sample_part(self, rng) -> PartState:
        G, p = self.G, self.p
        size = G * p
        frame = Frame(0.0, 0.0, 0.0, p)
        for _ in range(100):
            fam = self.families[int(rng.integers(len(self.families)))]
            geom = None
            if self.mesh_dir and rng.random() < self.mesh_frac:
                geom = mesh_slice_shapes(self.mesh_dir, size, rng)
                fam = "mesh"
            if geom is None:
                geom = random_shape(rng, fam, size, p)
            if geom.is_empty:
                continue
            mask, _ = rasterize(geom, frame, G, G, min_cells=3)
            if mask.sum() < self.min_cells:
                continue
            h = self.printer.layer_height
            nz = 3 if self.neighbors else 1
            loads = random_loads(rng, geom, p, neighbors_z=nz * h if self.neighbors else None)
            spec = PartSpec(self.printer, loads, ScoreConfig(norm=self.norm_mode), part_height=nz * h,
                            terms=self.terms)
            layers = []
            for k in range(nz):
                L = Layer.empty(k, (k + 1) * h, h, mask, frame, poly=geom, first=False)
                fill_baseline(L, self.baselines[int(rng.integers(len(self.baselines)))])
                layers.append(L)
            self.k = 1 if self.neighbors else 0
            part = PartState(layers, spec, fea_layers={self.k})
            if "strength" in self.terms and not part.evals[self.k].fea.N:
                continue
            self.shape_family = fam
            return part
        raise RuntimeError("could not sample a usable shape")

    # -------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.part = self._sample_part(rng)
        self.w = self._sample_w(rng)
        self.t = 0
        self.episode_gain = 0.0
        self.banned = set()
        return self._obs(), {"action_mask": self.action_masks()}

    @property
    def scorer(self) -> Normalizer:
        """Turns raw term changes into the weighted score (the loop uses part.norm too)."""
        return self.part.norm if self.norm_mode == "relative" else self.norm

    def _cell(self, action: int):
        L = self.part.layers[self.k]
        op, cell = divmod(int(action), self.G * self.G)
        r, c = divmod(cell, self.G)
        return L, op, r, c

    def step(self, action):
        part, k = self.part, self.k
        action = int(action)
        L, op, r, c = self._cell(action)
        reward = 0.0
        raw = {t: 0.0 for t in TERMS}
        invalid = False
        accepted = False
        if r < L.H and c < L.W and op in self.ops and action not in self.banned:
            try:
                newL, _ = apply(L, op, r * L.W + c)
            except InvalidEdit:
                invalid = True
            else:
                cand = part.propose(k, newL)
                if self.norm_mode == "std":
                    self.norm.update(cand.d)
                gain = self.scorer.total(cand.d, self.w)
                raw = cand.d
                if self.accept == "all" or loop_accept(gain, 0.0, None):
                    part.commit(cand)
                    self.banned.clear()
                    reward = gain
                    accepted = True
                else:
                    self.banned.add(action)
        else:
            invalid = True
        if invalid:
            reward = -1.0
        self.t += 1
        self.episode_gain += reward
        mask = self.action_masks()
        truncated = self.t >= self.episode_len or not mask.any()   # or: no untried edit left
        info = {"action_mask": mask, "raw": raw, "invalid": invalid, "accepted": accepted}
        if truncated:
            info["episode"] = {"r": self.episode_gain, "l": self.t, "family": self.shape_family}
        return self._obs(), float(reward), False, truncated, info

    def action_masks(self) -> np.ndarray:
        L = self.part.layers[self.k]
        M = action_mask(L, self.ops)[:N_OPS]
        G = self.G
        out = np.zeros((N_OPS, G, G), bool)
        out[:, :L.H, :L.W] = M.reshape(N_OPS, L.H, L.W)
        out = out.reshape(-1)
        if self.banned:
            out[list(self.banned)] = False
        return out

    def _obs(self) -> np.ndarray:
        return build_obs(self.part, self.k, self.w, pad_to=(self.G, self.G))

    # -------------------------------------------------------------- helpers for teachers

    def commit_action(self, action) -> float:
        """Apply an edit whatever its gain (a teacher walking an annealing run). Returns the gain."""
        part, k = self.part, self.k
        L, op, r, c = self._cell(action)
        newL, _ = apply(L, op, r * L.W + c)
        cand = part.propose(k, newL)
        if self.norm_mode == "std":
            self.norm.update(cand.d)
        part.commit(cand)
        self.banned.clear()
        return self.scorer.total(cand.d, self.w)

    def edit_scale(self, n: int = 16, rng=None) -> float:
        """Typical change of the weighted score per random edit on this episode's part:
        the unit for a teacher's annealing temperature. In std mode scores are already
        in those units (1.0, nothing evaluated)."""
        if self.norm_mode == "std":
            return 1.0
        from rltp.cover.ops import sample_valid
        rng = rng if rng is not None else np.random.default_rng(int(self.np_random.integers(2**31)))
        L = self.part.layers[self.k]
        M = action_mask(L, self.ops)
        if M.any():
            for a in sample_valid(M, rng, n):
                op, cell = divmod(int(a), L.n)
                newL, _ = apply(L, op, cell)
                self.part.norm.update(self.part.propose(self.k, newL).d)
        return self.part.norm.edit_std(self.w)

    def warmup_norm(self, n: int = 64, episodes: int = 4) -> None:
        """std mode: seed the running term scales with changes of random valid edits (not
        committed). Relative mode takes its scales from each episode's baseline instead."""
        if self.norm_mode != "std":
            return
        from rltp.cover.ops import sample_valid
        rng = np.random.default_rng(int(self.np_random.integers(2**31)))
        for _ in range(episodes):
            self.reset()
            L = self.part.layers[self.k]
            M = action_mask(L, self.ops)
            for a in sample_valid(M, rng, max(1, n // episodes)):
                op, cell = divmod(int(a), L.n)
                newL, _ = apply(L, op, cell)
                self.norm.update(self.part.propose(self.k, newL).d)

    def score_actions(self, actions) -> np.ndarray:
        """Normalised gain of each (padded) action from the current state, without committing."""
        part, k = self.part, self.k
        out = np.full(len(actions), -np.inf)
        for i, a in enumerate(actions):
            L, op, r, c = self._cell(a)
            try:
                newL, _ = apply(L, op, r * L.W + c)
            except InvalidEdit:
                continue
            cand = part.propose(k, newL)
            out[i] = self.scorer.total(cand.d, self.w)
        return out
