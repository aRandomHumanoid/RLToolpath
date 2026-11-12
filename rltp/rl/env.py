"""Gymnasium LayerEnv: one layer, one edit per step, reward = change in score.

reset samples a shape, loads, a weight vector w and a baseline cover. The
observation is padded to the curriculum stage's grid size; the action space is
Discrete(10 * G * G), flattened operator x cell, with action_masks() giving the
valid entries. With neighbors=True the editable layer sits between two layers
of the same outline (own baselines) and the v1.5 interlayer term is scored.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rltp.config import TERMS, PartSpec, Printer, ScoreConfig
from rltp.cover.baselines import BASELINES, fill_baseline
from rltp.cover.ops import N_OPS, InvalidEdit, action_mask, apply, ops_to_indices
from rltp.eval.metrics import Normalizer
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
                 fixed_w=None, baselines=BASELINES, printer: Printer | None = None, seed: int | None = None):
        super().__init__()
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
        self.norm = Normalizer()
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_CH, grid, grid), np.float32)
        self.action_space = spaces.Discrete(N_OPS * grid * grid)
        self.part: PartState | None = None
        self.k = 0
        self.t = 0
        self.w = np.zeros(4)
        self.episode_gain = 0.0
        self.shape_family = None
        if seed is not None:
            self.reset(seed=seed)

    # -------------------------------------------------------------- sampling

    def _sample_w(self, rng) -> np.ndarray:
        if self.fixed_w is not None:
            return self.fixed_w.copy()
        w = np.zeros(4)
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
            spec = PartSpec(self.printer, loads, ScoreConfig(), part_height=nz * h, terms=self.terms)
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
        return self._obs(), {"action_mask": self.action_masks()}

    def step(self, action):
        part, k = self.part, self.k
        L = part.layers[k]
        G = self.G
        op, cell = divmod(int(action), G * G)
        r, c = divmod(cell, G)
        reward = 0.0
        raw = {t: 0.0 for t in TERMS}
        invalid = False
        if r < L.H and c < L.W and op in self.ops:
            try:
                newL, _ = apply(L, op, r * L.W + c)
            except InvalidEdit:
                invalid = True
            else:
                cand = part.propose(k, newL)
                self.norm.update(cand.d)
                reward = self.norm.total(cand.d, self.w)
                raw = cand.d
                part.commit(cand)
        else:
            invalid = True
        if invalid:
            reward = -1.0
        self.t += 1
        self.episode_gain += reward
        truncated = self.t >= self.episode_len
        info = {"action_mask": self.action_masks(), "raw": raw, "invalid": invalid}
        if truncated:
            info["episode"] = {"r": self.episode_gain, "l": self.t, "family": self.shape_family}
        return self._obs(), float(reward), False, truncated, info

    def action_masks(self) -> np.ndarray:
        L = self.part.layers[self.k]
        M = action_mask(L, self.ops)
        G = self.G
        out = np.zeros((N_OPS, G, G), bool)
        out[:, :L.H, :L.W] = M.reshape(N_OPS, L.H, L.W)
        return out.reshape(-1)

    def _obs(self) -> np.ndarray:
        return build_obs(self.part, self.k, self.w, pad_to=(self.G, self.G))

    # -------------------------------------------------------------- helpers for teachers

    def warmup_norm(self, n: int = 64, episodes: int = 4) -> None:
        """Seed the running term scales with changes of random valid edits (not committed)."""
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
        part, k, G = self.part, self.k, self.G
        L = part.layers[k]
        out = np.full(len(actions), -np.inf)
        for i, a in enumerate(actions):
            op, cell = divmod(int(a), G * G)
            r, c = divmod(cell, G)
            try:
                newL, _ = apply(L, op, r * L.W + c)
            except InvalidEdit:
                continue
            cand = part.propose(k, newL)
            out[i] = self.norm.total(cand.d, self.w)
        return out
