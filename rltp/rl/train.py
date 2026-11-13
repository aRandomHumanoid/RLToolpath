"""Curriculum training: imitation warm-start, then PPO, stage by stage.

Default curriculum (configs/train.yaml):
  1. 32x32 grids, strength only, Flip and Patch refill.
  2. time and continuity added, 64x64 grids, every operator.
  3. above/below-layer channels and the v1.5 interlayer term.
  4. (runtime) the frozen policy runs inside the part loop.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from rltp.rl.ppo import PPOConfig, train_ppo
from rltp.rl.shapes import FAMILIES, HOLDOUT_FAMILIES

log = logging.getLogger(__name__)

ENV_KEYS = ("grid", "terms", "ops", "episode_len", "families", "neighbors", "p", "min_cells",
            "mesh_dir", "mesh_frac", "fixed_w", "baselines", "accept", "norm")


def env_kwargs_for(stage: dict, defaults: dict) -> dict:
    d = {**defaults.get("env", {}), **stage.get("env", {})}
    for k in ENV_KEYS:
        if k in stage:
            d[k] = stage[k]
    if "families" not in d:
        hold = set(defaults.get("holdout_families", HOLDOUT_FAMILIES))
        d["families"] = [f for f in FAMILIES if f not in hold]
    if "terms" in d:
        d["terms"] = list(d["terms"])
    return {k: v for k, v in d.items() if k in ENV_KEYS}


def train(config_path, out_dir, progress=print) -> Path:
    import torch
    from rltp.rl.imitate import collect_teacher, train_bc
    from rltp.rl.net import UNet, load_policy, n_params, save_policy

    cfg = yaml.safe_load(open(config_path)) or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    if cfg.get("init"):
        model, _ = load_policy(cfg["init"])
        model.train()
    else:
        net = cfg.get("net", {})
        model = UNet(chs=tuple(net.get("chs", (32, 64, 128, 256))))
    progress(f"policy: {n_params(model) / 1e6:.2f}M weights")
    summary = {"stages": []}
    ppo_defaults = cfg.get("ppo", {})
    ppo_keys = {f.name for f in fields(PPOConfig)}
    for i, stage in enumerate(cfg.get("stages", [])):
        name = stage.get("name", f"stage{i + 1}")
        sdir = out / name
        sdir.mkdir(exist_ok=True)
        ek = env_kwargs_for(stage, cfg)
        rec = {"name": name, "env": ek}
        t0 = time.time()
        progress(f"== {name}: {ek}")
        im = stage.get("imitation")
        if im:
            data = collect_teacher(ek, int(im.get("states", 1000)), int(im.get("candidates", 32)),
                                   int(im.get("sa_steps", 128)), float(im.get("temp0", 0.1)),
                                   int(im.get("workers", 4)), seed + i)
            bc = train_bc(model, data, int(im.get("epochs", 10)), float(im.get("lr", 1e-3)),
                          int(im.get("batch", 64)), float(im.get("val_frac", 0.1)), seed + i,
                          progress=progress)
            rec["imitation"] = bc
            save_policy(sdir / "policy_bc.pt", model, {"stage": name, "phase": "bc", "env": ek})
        pp = stage.get("ppo")
        if pp:
            pc = PPOConfig(**{k: v for k, v in {**ppo_defaults, **pp}.items() if k in ppo_keys})
            pc.seed = seed + i
            rec["ppo"] = train_ppo(model, ek, pc, sdir, progress=progress)
        save_policy(sdir / "policy.pt", model, {"stage": name, "env": ek})
        rec["seconds"] = time.time() - t0
        summary["stages"].append(rec)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    final = out / "policy.pt"
    from rltp.cover.ops import OPS, ops_to_indices
    trained = sorted({i for s in summary["stages"] for i in ops_to_indices(s["env"].get("ops"))})
    save_policy(final, model, {"config": cfg, "stages": [s["name"] for s in summary["stages"]],
                               "ops": [OPS[i] for i in trained]})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    progress(f"saved {final}")
    return final
