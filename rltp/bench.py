"""Main evaluation plot: score against evaluator calls on held-out shapes, for the
policy, simulated annealing, greedy search and random edits.

Every proposer runs in the same loop with the same evaluator, from the same
start state, with the same frozen term scales, at the same evaluator budget.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np


def _shape_part(env_kwargs, seed):
    from rltp.rl.env import LayerEnv
    env = LayerEnv(**env_kwargs)
    env.reset(seed=seed)
    return env


def pct_change(raw0: dict, raw1: dict, terms) -> dict:
    """Physical change of each enabled term, in percent (negative = smaller FI / time / cost).
    strength and inter are -log FI_agg; time and cont are minus a time and a cost."""
    out = {}
    for t in terms:
        a, b = float(raw0[t]), float(raw1[t])
        if t in ("strength", "inter"):
            out[t] = 100.0 * (np.exp(-(b - a)) - 1.0)
        elif a != 0:
            out[t] = 100.0 * (b / a - 1.0)
    return out


TERM_LABEL = {"strength": "FI_agg", "time": "time", "cont": "cont cost", "inter": "inter FI"}


def _run_one(job):
    env_kwargs, seed, proposer_name, policy, budget, w, norm_state, run_seed, persist = job
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass
    from rltp.loop.optimize import History, LoopConfig, optimize_part
    from rltp.search.sa import make_proposer
    env = _shape_part(env_kwargs, seed)
    part = env.part
    part.norm.load(norm_state)
    part.norm.frozen = True
    if w is None:
        w = env.w
    w = np.asarray(w, float)
    prop = make_proposer(proposer_name, policy)
    burst = 32
    cfg = LoopConfig(burst=burst, anneal_passes=max(1, int(0.6 * budget / burst)), eps=-np.inf,
                     max_passes=10 ** 6, max_evals=budget, warmup=0, seed=run_seed, persist_bans=persist)
    start = part.total(w)
    raw0 = part.raw()
    hist = History(None)
    optimize_part(part, prop, w, cfg, hist)
    raw1 = part.raw()                    # the loop leaves the part at the best state seen
    xs = np.arange(0, budget + 1)
    best = np.full(budget + 1, np.nan)
    best[0] = 0.0
    cur = 0.0
    for e in hist.events:
        if e.get("type") != "edit":
            continue
        cur = max(cur, e["total"] - start)
        i = min(int(e["evals"]), budget)
        best[i] = cur if np.isnan(best[i]) else max(best[i], cur)
    best = np.fmax.accumulate(np.nan_to_num(best, nan=-np.inf))
    best[np.isinf(best)] = 0.0
    return {"seed": seed, "proposer": proposer_name, "run_seed": run_seed, "family": env.shape_family,
            "start": start, "final_gain": float(best[-1]), "curve": best[xs].tolist(),
            "raw0": raw0, "raw1": raw1, "pct": pct_change(raw0, raw1, env.terms)}


def run_bench(args, progress=print) -> dict:
    from rltp.config import parse_weights
    from rltp.loop.optimize import warmup_normalizer
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    terms = tuple(t.strip() for t in args.terms.split(","))
    fams = tuple(f.strip() for f in args.families.split(","))
    env_kwargs = {"grid": args.grid, "terms": terms, "families": fams,
                  "norm": getattr(args, "norm", "relative")}
    persist = not getattr(args, "burst_bans", False)
    w = None if args.weights is None else parse_weights(args.weights).tolist()
    props = [p.strip() for p in args.proposers.split(",") if p.strip()]
    if not args.policy:
        for pn in ("policy", "policy-sample"):
            if pn in props:
                progress(f"no --policy given: skipping {pn}")
                props.remove(pn)
    jobs = []
    rng = np.random.default_rng(12345)
    for s in range(args.shapes):
        seed = int(rng.integers(2**31))
        env = _shape_part(env_kwargs, seed)
        warmup_normalizer(env.part, 128, np.random.default_rng(seed))
        ns = env.part.norm.state()
        for pn in props:
            for rs in range(args.seeds):
                jobs.append((env_kwargs, seed, pn, args.policy, args.budget, w, ns, rs, persist))
    progress(f"bench: {args.shapes} shapes x {len(props)} proposers x {args.seeds} seeds, "
             f"budget {args.budget} evals, families {fams}, terms {terms}")
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(_run_one, jobs)
    else:
        res = [_run_one(j) for j in jobs]
    summary = {}
    for pn in props:
        mine = [r for r in res if r["proposer"] == pn]
        curves = np.array([r["curve"] for r in mine])
        pct = {t: float(np.mean([r["pct"][t] for r in mine if t in r["pct"]]))
               for t in terms if any(t in r["pct"] for r in mine)}
        summary[pn] = {"mean_final_gain": float(curves[:, -1].mean()),
                       "std_final_gain": float(curves[:, -1].std()),
                       "mean_pct_change": pct,
                       "mean_curve": curves.mean(0).tolist()}
        changes = "  ".join(f"{TERM_LABEL.get(t, t)} {v:+.1f}%" for t, v in pct.items())
        progress(f"  {pn:13s} mean gain {curves[:, -1].mean():.4f} +- {curves[:, -1].std():.4f}   {changes}")
    # evals to reach SA's final score
    if "sa" in summary:
        target = summary["sa"]["mean_final_gain"]
        for pn in props:
            mc = np.array(summary[pn]["mean_curve"])
            hit = np.flatnonzero(mc >= target)
            summary[pn]["evals_to_sa_final"] = int(hit[0]) if hit.size else None
    (out / "results.json").write_text(json.dumps({"args": vars(args), "summary": summary, "runs": res},
                                                 default=str))
    _plot(summary, out / "score_vs_evals.png", args)
    progress(f"wrote {out / 'score_vs_evals.png'}")
    return summary


def _plot(summary, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"policy": "#2a6fdb", "policy-sample": "#7b4fd6", "sa": "#d9822b", "greedy": "#3a9a5b",
              "random": "#8a8a8a"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for pn, s in summary.items():
        mc = np.array(s["mean_curve"])
        ax.plot(np.arange(mc.size), mc, label=pn, color=colors.get(pn), lw=2)
    ax.set_xlabel("evaluator calls")
    ax.set_ylabel("best score gain over baseline (normalised)")
    ax.set_title(f"held-out {args.families}, {args.shapes} shapes, grid {args.grid}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
