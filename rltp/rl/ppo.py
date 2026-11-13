"""Single-file masked PPO in the CleanRL style.

Environments run in subprocesses (they never import torch). Action masks travel
bit-packed. Invalid logits are set to -1e9 and one softmax runs over all
operator x cell entries. Rewards are normalised by the running std of the
discounted return.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    num_envs: int = 32
    num_steps: int = 64
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.97
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatch_size: int = 128
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.05
    norm_reward: bool = True
    save_every: int = 10
    image_every: int = 10
    threads: int = 0
    seed: int = 0


# ------------------------------------------------------------------ vector env

def _worker(remote, parent, env_kwargs, seed):
    parent.close()
    from rltp.rl.env import LayerEnv
    env = LayerEnv(**env_kwargs)
    env.reset(seed=seed)
    env.warmup_norm(64)
    obs, info = env.reset(seed=seed + 1)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, r, term, trunc, info = env.step(int(data))
                ep = info.get("episode")
                if term or trunc:
                    obs, info = env.reset()
                remote.send((obs, np.packbits(info["action_mask"]), r, term or trunc, ep))
            elif cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, np.packbits(info["action_mask"])))
            elif cmd == "snapshot":
                L = env.part.layers[env.k]
                remote.send({"mask": L.mask, "frame": L.frame.to_dict(), "nxt": L.nxt, "prv": L.prv,
                             "seam": set(L.seam), "fi": env.part.evals[env.k].fea.fi,
                             "family": env.shape_family})
            elif cmd == "close":
                break
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, env_kwargs: dict, n: int, seed: int = 0):
        ctx = mp.get_context("fork")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, a, env_kwargs, seed + 7919 * i), daemon=True)
            p.start()
            b.close()
            self.remotes.append(a)
            self.procs.append(p)
        self.n = n

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        res = [r.recv() for r in self.remotes]
        return np.stack([x[0] for x in res]), np.stack([x[1] for x in res])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        res = [r.recv() for r in self.remotes]
        obs = np.stack([x[0] for x in res])
        masks = np.stack([x[1] for x in res])
        rew = np.array([x[2] for x in res], np.float32)
        done = np.array([x[3] for x in res], np.float32)
        eps = [x[4] for x in res if x[4] is not None]
        return obs, masks, rew, done, eps

    def snapshot(self, i: int = 0):
        self.remotes[i].send(("snapshot", None))
        return self.remotes[i].recv()

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(), x.var(), x.size
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot


# ------------------------------------------------------------------ training

def save_toolpath_image(snap: dict, path: Path, title: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rltp.export.render import draw_layer
    except Exception as e:  # pragma: no cover
        log.debug("no image: %s", e)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    draw_layer(axes[0], snap["mask"], snap["frame"], snap["nxt"], snap["prv"], snap["seam"], color="direction")
    draw_layer(axes[1], snap["mask"], snap["frame"], snap["nxt"], snap["prv"], snap["seam"], color="fi",
               fi=snap["fi"])
    fig.suptitle(f"{title} ({snap.get('family')})")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def train_ppo(model, env_kwargs: dict, cfg: PPOConfig, out_dir, progress=None) -> dict:
    import torch
    from torch.distributions import Categorical
    from rltp.cover.ops import N_PATH_OPS as N_OPS
    from rltp.rl.net import masked_logits, save_policy

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    G = env_kwargs.get("grid", 32)
    A = N_OPS * G * G
    T, N = cfg.num_steps, cfg.num_envs
    envs = SubprocVecEnv(env_kwargs, N, seed=cfg.seed * 100003)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    rms = RunningMeanStd()
    ret_acc = np.zeros(N)
    logf = open(out / "ppo_log.jsonl", "a")

    obs0, mk0 = envs.reset()
    C = obs0.shape[1]
    obs_buf = np.zeros((T, N, C, G, G), np.float16)
    mask_buf = np.zeros((T, N, mk0.shape[1]), np.uint8)
    act_buf = np.zeros((T, N), np.int64)
    logp_buf = np.zeros((T, N), np.float32)
    rew_buf = np.zeros((T, N), np.float32)
    done_buf = np.zeros((T, N), np.float32)
    val_buf = np.zeros((T, N), np.float32)
    next_obs, next_mask, next_done = obs0, mk0, np.zeros(N, np.float32)

    def unpack(mk):
        return torch.from_numpy(np.unpackbits(mk, axis=-1, count=A).astype(bool))

    num_updates = max(1, cfg.total_steps // (T * N))
    gstep = 0
    t0 = time.time()
    ep_returns: list = []
    summary = {}
    try:
        for upd in range(1, num_updates + 1):
            if cfg.anneal_lr:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * (1.0 - (upd - 1) / num_updates)
            model.eval()
            for t in range(T):
                obs_buf[t] = next_obs
                mask_buf[t] = next_mask
                done_buf[t] = next_done
                with torch.no_grad():
                    x = torch.from_numpy(next_obs.astype(np.float32))
                    lg, v = model(x)
                    dist = Categorical(logits=masked_logits(lg, unpack(next_mask)))
                    a = dist.sample()
                    logp_buf[t] = dist.log_prob(a).numpy()
                    val_buf[t] = v.numpy()
                act_buf[t] = a.numpy()
                next_obs, next_mask, r, d, eps = envs.step(act_buf[t])
                if cfg.norm_reward:
                    ret_acc = ret_acc * cfg.gamma + r
                    rms.update(ret_acc)
                    ret_acc[d > 0] = 0.0
                    r = r / np.sqrt(rms.var + 1e-8)
                rew_buf[t] = r
                next_done = d
                ep_returns += [e["r"] for e in eps]
                gstep += N

            with torch.no_grad():
                _, nv = model(torch.from_numpy(next_obs.astype(np.float32)))
                nv = nv.numpy()
            adv = np.zeros((T, N), np.float32)
            last = np.zeros(N, np.float32)
            for t in reversed(range(T)):
                if t == T - 1:
                    nnt, nval = 1.0 - next_done, nv
                else:
                    nnt, nval = 1.0 - done_buf[t + 1], val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * nval * nnt - val_buf[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nnt * last
                adv[t] = last
            ret = adv + val_buf

            b_obs = obs_buf.reshape(T * N, C, G, G)
            b_mask = mask_buf.reshape(T * N, -1)
            b_act = torch.from_numpy(act_buf.reshape(-1))
            b_logp = torch.from_numpy(logp_buf.reshape(-1))
            b_adv = torch.from_numpy(adv.reshape(-1))
            b_ret = torch.from_numpy(ret.reshape(-1))
            model.train()
            stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": []}
            B = T * N
            mb = min(cfg.minibatch_size, B)
            stop = False
            for ep in range(cfg.epochs):
                perm = np.random.permutation(B)
                for s in range(0, B, mb):
                    ii = perm[s:s + mb]
                    x = torch.from_numpy(b_obs[ii].astype(np.float32))
                    lg, v = model(x)
                    dist = Categorical(logits=masked_logits(lg, unpack(b_mask[ii])))
                    ti = torch.from_numpy(ii)
                    newlogp = dist.log_prob(b_act[ti])
                    ent = dist.entropy().mean()
                    logratio = newlogp - b_logp[ti]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        kl = ((ratio - 1) - logratio).mean().item()
                        stats["kl"].append(kl)
                        stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    ma = b_adv[ti]
                    ma = (ma - ma.mean()) / (ma.std() + 1e-8)
                    pg = torch.max(-ma * ratio, -ma * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    vf = 0.5 * ((v - b_ret[ti]) ** 2).mean()
                    loss = pg - cfg.ent_coef * ent + cfg.vf_coef * vf
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    stats["pg"].append(pg.item())
                    stats["vf"].append(vf.item())
                    stats["ent"].append(ent.item())
                if cfg.target_kl is not None and np.mean(stats["kl"][-max(1, B // mb):]) > cfg.target_kl:
                    stop = True
                    break
            sps = gstep / (time.time() - t0)
            rec = {"update": upd, "step": gstep, "sps": round(sps, 1),
                   "ep_return": float(np.mean(ep_returns[-64:])) if ep_returns else None,
                   "episodes": len(ep_returns), "lr": opt.param_groups[0]["lr"], "early_stop": stop,
                   **{k: float(np.mean(v)) for k, v in stats.items() if v}}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            summary = rec
            if progress:
                progress(f"ppo upd {upd}/{num_updates} step {gstep} ep_return {rec['ep_return']} "
                         f"ent {rec.get('ent', 0):.3f} kl {rec.get('kl', 0):.4f} sps {sps:.0f}")
            if upd % cfg.save_every == 0 or upd == num_updates:
                save_policy(out / "policy.pt", model, {"update": upd, "step": gstep, "env": env_kwargs})
            if cfg.image_every and (upd % cfg.image_every == 0 or upd == num_updates):
                save_toolpath_image(envs.snapshot(0), out / "images" / f"upd{upd:05d}.png",
                                    f"update {upd}, step {gstep}")
    finally:
        envs.close()
        logf.close()
    return {"updates": num_updates, "steps": gstep, "last": summary, "config": asdict(cfg)}
