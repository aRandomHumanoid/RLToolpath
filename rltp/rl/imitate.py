"""Imitation warm-start: best-of-N teacher labels -> behaviour cloning.

Teacher: at states sampled from annealing runs, score N random valid edits and
label the one with the largest gain. Train with masked cross-entropy. Held-out
imitation accuracy (does the policy rank the teacher's pick first among the N
candidates?) compares architectures before any RL run.
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time

import numpy as np

from rltp.cover.ops import N_PATH_OPS as N_OPS, sample_valid

log = logging.getLogger(__name__)


def _teacher_worker(args):
    env_kwargs, n_states, n_cand, sa_steps, temp0, seed = args
    from rltp.rl.env import LayerEnv
    env = LayerEnv(**env_kwargs)
    rng = np.random.default_rng(seed)
    if env.norm_mode == "std":
        # without this the first states are scored against near-zero running scales
        # (one such state had a best-of-N gain of ~1e4 edit stds)
        env.reset(seed=seed)
        env.warmup_norm(64)
    obs_l, mask_l, lab_l, cand_l, gain_l = [], [], [], [], []
    ep = 0
    while len(lab_l) < n_states:
        env.reset(seed=int(rng.integers(2**31)))
        unit = env.edit_scale(rng=rng)      # temperatures are in typical-edit units
        ep += 1
        # sample a few states along an annealing run
        stops = sorted(rng.choice(np.arange(sa_steps + 1), size=min(4, sa_steps + 1), replace=False).tolist())
        step = 0
        for s_at in stops:
            while step < s_at:
                M = env.action_masks()
                a = int(sample_valid(M.reshape(N_OPS, -1), rng)[0])
                g = env.score_actions([a])[0]
                temp = temp0 * unit * (1 - step / max(sa_steps, 1))
                if g > 0 or (temp > 0 and rng.random() < math.exp(max(g / temp, -50))):
                    env.commit_action(a)
                step += 1
            M = env.action_masks()
            cands = np.unique(sample_valid(M.reshape(N_OPS, -1), rng, n_cand))
            gains = env.score_actions(cands)
            if not np.isfinite(gains).any():
                continue
            obs_l.append(env._obs().astype(np.float16))
            mask_l.append(np.packbits(M))
            lab_l.append(int(cands[int(np.argmax(gains))]))
            pad = n_cand - cands.size
            cand_l.append(np.concatenate([cands, np.full(pad, -1)]))
            gain_l.append(np.concatenate([gains, np.full(pad, -np.inf)]))
            if len(lab_l) >= n_states:
                break
    return (np.stack(obs_l), np.stack(mask_l), np.array(lab_l), np.stack(cand_l), np.stack(gain_l))


def collect_teacher(env_kwargs: dict, n_states: int, n_cand: int = 32, sa_steps: int = 128,
                    temp0: float = 0.1, workers: int = 4, seed: int = 0) -> dict:
    per = [n_states // workers + (1 if i < n_states % workers else 0) for i in range(workers)]
    jobs = [(env_kwargs, n, n_cand, sa_steps, temp0, seed * 1000 + i) for i, n in enumerate(per) if n > 0]
    t0 = time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(len(jobs)) as pool:
            res = pool.map(_teacher_worker, jobs)
    else:
        res = [_teacher_worker(j) for j in jobs]
    out = {
        "obs": np.concatenate([r[0] for r in res]),
        "mask": np.concatenate([r[1] for r in res]),
        "label": np.concatenate([r[2] for r in res]),
        "cand": np.concatenate([r[3] for r in res]),
        "gain": np.concatenate([r[4] for r in res]),
        "n_actions": N_OPS * env_kwargs.get("grid", 32) ** 2,
    }
    log.info("teacher: %d states in %.1fs", len(out["label"]), time.time() - t0)
    return out


def train_bc(model, data: dict, epochs: int = 10, lr: float = 1e-3, batch: int = 64,
             val_frac: float = 0.1, seed: int = 0, device: str = "cpu", progress=None) -> dict:
    import torch
    import torch.nn.functional as F
    from rltp.rl.net import masked_logits

    n = len(data["label"])
    A = int(data["n_actions"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nv = max(1, int(n * val_frac)) if n > 10 else 0
    val, tr = idx[:nv], idx[nv:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * math.ceil(len(tr) / batch)))
    model.to(device)

    def batch_tensors(ii):
        x = torch.from_numpy(data["obs"][ii].astype(np.float32)).to(device)
        m = torch.from_numpy(np.unpackbits(data["mask"][ii], axis=1, count=A).astype(bool)).to(device)
        y = torch.from_numpy(data["label"][ii]).long().to(device)
        return x, m, y

    def evaluate(ii):
        model.eval()
        hits = hits_full = 0
        regret = []
        with torch.no_grad():
            for s in range(0, len(ii), batch):
                b = ii[s:s + batch]
                x, m, y = batch_tensors(b)
                lg, _ = model(x)
                ml = masked_logits(lg, m)
                hits_full += int((ml.argmax(1) == y).sum())
                cand = data["cand"][b]
                gain = data["gain"][b]
                mlc = ml.cpu().numpy()
                for r in range(len(b)):
                    ok = cand[r] >= 0
                    cs = cand[r][ok]
                    pick = cs[int(np.argmax(mlc[r][cs]))]
                    hits += int(pick == data["label"][b[r]])
                    g = gain[r][ok]
                    regret.append(float(g.max() - g[int(np.argmax(mlc[r][cs]))]))
        model.train()
        k = max(len(ii), 1)
        return {"acc_cand": hits / k, "acc_full": hits_full / k, "regret": float(np.mean(regret)) if regret else 0.0}

    hist = []
    model.train()
    for ep in range(epochs):
        perm = rng.permutation(tr)
        losses = []
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            x, m, y = batch_tensors(b)
            lg, _ = model(x)
            loss = F.cross_entropy(masked_logits(lg, m), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": ep, "loss": float(np.mean(losses)) if losses else 0.0}
        if nv:
            rec.update({f"val_{k}": v for k, v in evaluate(val).items()})
        hist.append(rec)
        if progress:
            progress(f"bc epoch {ep}: " + ", ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"))
    return {"history": hist, "n_train": int(len(tr)), "n_val": int(nv)}
