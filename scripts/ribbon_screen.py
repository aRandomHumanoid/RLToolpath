"""Screen ribbon starts on one part without search: build the part once, then score straight
families at every angle step, alone and alternating with the perpendicular family, optionally
density-spaced, under given skins. Prints a table sorted by score; the best row's arguments
start a search (scripts/ribbons.py).

usage: python scripts/ribbon_screen.py PART [--step 15] [--skins 0,0,0,0] [--gammas none,0.5]
(from the repo root)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ribbons as R  # noqa: E402

from rltp.surfaces.layout import apply_layout, is_infill_layer  # noqa: E402
from rltp.surfaces.ribbons import density_family, line_family  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(R.REF))
    ap.add_argument("--step", type=float, default=15.0)
    ap.add_argument("--skins", default="slicer", help="skin classes per skin layer, or slicer")
    ap.add_argument("--gammas", default="none,0.5", help="density gammas to try (none: uniform lines)")
    ap.add_argument("--spacing", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    part = R.build(a.part)
    raw_s = part.raw()
    w = R.REF[a.part]["weights"]
    R.relay_skins(part, a.skins)
    k0 = next(k for k, L in enumerate(part.layers) if is_infill_layer(L))
    bounds = part.layers[k0].poly.bounds
    bw = part.spec.printer.bead_width
    want = 100.0 * part.spec.printer.infill
    R.G["part_name"] = a.part
    rows = []
    for g in a.gammas.split(","):
        rho = None
        for ang in np.arange(0.0, 180.0, a.step):
            for alt in (False, True):
                if alt and ang >= 90.0:
                    continue                   # (a, a + 90) pairs: each pair once
                def fams(r):
                    f = (lambda th: density_family(bounds, th, r, bw)) if r is not None else \
                        (lambda th: line_family(bounds, th, a.spacing))
                    return (f(ang), f(ang + 90.0)) if alt else f(ang)
                if g == "none":
                    apply_layout(part, fams(None))
                else:
                    t = bw / a.spacing           # calibrate to the infill target, as the driver does
                    for _ in range(5):
                        rho = R.density_start(part, k0, t, float(g))
                        apply_layout(part, fams(rho))
                        got = float(part.breakdown()["infill_pct"])
                        if abs(got - want) < 0.5:
                            break
                        t *= want / max(got, 1.0)
                b = part.breakdown()
                obj = R.objective(raw_s, part.raw(), w, raw_s)
                rows.append((obj["total"], b["fi_agg"], b["strokes"], b["time_s"], b["infill_pct"],
                             f"--angle {ang:g}" + (f" --alt-angle {ang + 90:g}" if alt else "")
                             + ("" if g == "none" else f" --density-gamma {g}")))
    rows.sort(key=lambda r: -r[0])
    print(f"{a.part}, skins {a.skins}: {len(rows)} starts in {time.time() - t0:.0f} s")
    for obj, fi, st, ts, inf, args in rows:
        print(f"  {obj:+7.1f}  FI_agg {fi:.3f}  {st:4d} strokes  {ts:5.0f} s  {inf:4.1f}%  {args}", flush=True)


if __name__ == "__main__":
    main()
