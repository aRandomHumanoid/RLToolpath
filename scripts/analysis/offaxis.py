"""Strength of one bead loaded in uniaxial tension or compression at an angle to it, from the
evaluator's Tsai-Wu check and the material in configs/printer.yaml: how much a bead loses
when it is 5.6 (16 bins' worst case), 22.5 (the four classes' worst case) or 45-90 degrees
off the load.

usage: python scripts/analysis/offaxis.py   (from the repo root)
"""
import numpy as np

from rltp.config import load_printer
from rltp.eval.fea2d import to_material, tsai_wu_fi

mat = load_printer("configs/printer.yaml").material
print("load angle off bead   tension MPa   compression MPa   (relative to aligned)")
t0 = c0 = None
for a in [0, 5.625, 11.25, 22.5, 30, 45, 60, 90]:
    th = np.radians(a)
    ft = float(tsai_wu_fi(*to_material(np.array(1.0), np.array(0.0), np.array(0.0), th), mat))
    fc = float(tsai_wu_fi(*to_material(np.array(-1.0), np.array(0.0), np.array(0.0), th), mat))
    st, sc = 1 / ft, 1 / fc
    t0 = t0 or st
    c0 = c0 or sc
    print(f"{a:8.2f} deg   {st:7.1f} ({st / t0:5.1%})   {sc:7.1f} ({sc / c0:5.1%})")
