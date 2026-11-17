"""What blocks strength-improving edits? From a run's history: each edit's per-term change d
(higher is better for every term) and its weighted total d_total. Effective weights are fitted
(d_total = sum w_t d_t). For edits that would lower FI (d.strength > 0): share accepted, and for
the rejected ones, which term's weighted change outweighed the strength gain (the blocker).
Split by op family.

usage: python scripts/analysis/blockers.py RUN_DIR [RUN_DIR ...]
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

for run in map(Path, sys.argv[1:]):
    ev = [json.loads(x) for x in gzip.open(run / "history.jsonl.gz", "rt") if x.strip()]
    ed = [e for e in ev if e.get("type") == "edit" and e.get("d")]
    terms = sorted({t for e in ed for t in e["d"]})
    D = np.array([[e["d"].get(t, 0.0) for t in terms] for e in ed])
    y = np.array([e["d_total"] for e in ed])
    w, *_ = np.linalg.lstsq(D, y, rcond=None)
    fit = float(np.max(np.abs(D @ w - y)))
    acc = np.array([bool(e["accepted"]) for e in ed])
    fam = [re.sub(r"\d+.*$", "", e["op"]) or e["op"] for e in ed]
    si = terms.index("strength")
    WD = D * w
    print(f"\n=== {run.parent.name}/{run.name}: {len(ed)} edits, accepted {acc.mean():.1%}; "
          f"weights " + " ".join(f"{t}={x:.3g}" for t, x in zip(terms, w)) + f" (fit err {fit:.1e})")
    good = WD[:, si] > 0
    rej = good & ~acc
    print(f"  strength-improving edits: {good.sum()} ({good.mean():.0%}), accepted {acc[good].mean():.0%}")
    others = [i for i in range(len(terms)) if i != si]
    blk = collections.Counter()
    lost = collections.Counter()
    for i in np.nonzero(rej)[0]:
        neg = {terms[j]: WD[i, j] for j in others if WD[i, j] < 0}
        if not neg:
            blk["(none: SA draw)"] += 1
            continue
        t = min(neg, key=neg.get)
        blk[t] += 1
        lost[t] += WD[i, si]
    tot = sum(blk.values())
    print("  rejected strength gains, main blocker: " +
          ", ".join(f"{t} {n / tot:.0%} (strength forgone {lost[t]:.3g})" for t, n in blk.most_common()))
    print(f"  strength gained by accepted edits {WD[acc & good, si].sum():.3g}; "
          f"forgone in rejected {WD[rej, si].sum():.3g}")
    # rejected strength-improving edits: how big is the cont penalty vs the strength gain
    ci = terms.index("cont") if "cont" in terms else None
    ti = terms.index("time")
    for name, idx in (("cont", ci), ("time", ti)):
        if idx is None:
            continue
        m = rej & (WD[:, idx] < 0)
        if m.any():
            ratio = -WD[m, idx] / WD[m, si]
            print(f"  {name}: in {m.mean() / max(rej.mean(), 1e-9):.0%} of rejected gains; "
                  f"penalty / gain median {np.median(ratio):.1f}x, p25 {np.percentile(ratio, 25):.1f}x")
    print("  by op family (tried, strength-improving, accepted of those, top blocker of rejected):")
    for f in sorted(set(fam), key=lambda f: -fam.count(f)):
        m = np.array([x == f for x in fam])
        g = m & good
        r = g & ~acc
        b = collections.Counter()
        for i in np.nonzero(r)[0]:
            neg = {terms[j]: WD[i, j] for j in others if WD[i, j] < 0}
            b[min(neg, key=neg.get) if neg else "draw"] += 1
        bs = ", ".join(f"{t} {n}" for t, n in b.most_common(2))
        print(f"    {f:14s} {m.sum():5d}  {g.sum():5d}  {acc[g].mean() if g.any() else 0:4.0%}  {bs}")
