"""Pick the evaluator's path-tangent settings: how well does a local least-squares tangent
recover a bead's true direction from its 0.1 mm, 4-connected node path?

Cases: straight lines at every angle, circles (radius 1-10 mm), zigzag infill (90 deg turns
0.4 mm apart). Settings: half-window h (mm of path each side), corner split threshold.
"""
import numpy as np

P = 0.1          # grid cell, mm
LARGE_MM = 0.8   # corner test: one-sided least-squares directions over this much path
CONC = 0.7       # ... turning at least this share as much over half that path (a corner, not a curve)


def raster_line(p0, p1):
    """4-connected node path from p0 to p1 (cell units): each step in x or y, whichever
    keeps the next node closest to the ideal segment."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = d / max(np.hypot(*d), 1e-12)
    cur = np.round(p0).astype(int)
    end = np.round(p1).astype(int)
    out = [cur.copy()]
    sx, sy = int(np.sign(end[0] - cur[0])), int(np.sign(end[1] - cur[1]))
    while (cur != end).any():
        cands = []
        if cur[0] != end[0]:
            cands.append(cur + [sx, 0])
        if cur[1] != end[1]:
            cands.append(cur + [0, sy])
        dist = [abs((c - p0)[0] * n[1] - (c - p0)[1] * n[0]) for c in cands]
        cur = cands[int(np.argmin(dist))]
        out.append(cur.copy())
    return np.array(out)


def raster_poly(pts, loop=False):
    seq = []
    pts = list(pts) + ([pts[0]] if loop else [])
    for a, b in zip(pts[:-1], pts[1:]):
        s = raster_line(a, b)
        seq.append(s if not seq else s[1:])
    s = np.concatenate(seq)
    if loop and (s[0] == s[-1]).all():
        s = s[:-1]
    return s


def tangents(xy, loop, h_mm=0.8, corner_deg=30.0, corner_mm=0.4):
    """Angle (rad, mod pi) per node: least-squares direction over nodes within h of path length,
    never across a corner (direction turning > corner_deg within corner_mm)."""
    n = len(xy)
    idx = np.arange(n)
    h = max(int(round(h_mm / P)), 1)
    c = max(int(round(corner_mm / P)), 1)

    def at(i):
        return xy[i % n] if loop else xy[np.clip(i, 0, n - 1)]

    def turn_at(s):
        """Turn (rad) between the chords over s nodes before and after each node."""
        fwd = np.array([at(i + s) - at(i) for i in idx], float)
        bwd = np.array([at(i) - at(i - s) for i in idx], float)
        a = np.arctan2(fwd[:, 1], fwd[:, 0]) - np.arctan2(bwd[:, 1], bwd[:, 0])
        t = np.abs((a + np.pi) % (2 * np.pi) - np.pi)
        if not loop:
            t[:s] = 0.0
            t[n - s:] = 0.0
        return t

    # a corner: a sharp turn at the 0.2 mm scale (a local maximum there) that persists at the
    # corner_mm + 0.2 mm scale either side (grid jogs don't; real turns do)
    small = turn_at(2)

    def side_dir(i, sgn, m):
        """Least-squares direction (oriented along travel) of the m nodes on one side of i."""
        js = i + sgn * np.arange(0, m + 1)
        if not loop:
            js = js[(js >= 0) & (js < n)]
        pts = xy[js % n].astype(float)
        if len(pts) < 3:
            return None
        ch = (pts[-1] - pts[0]) * sgn
        pc = pts - pts.mean(0)
        v = np.linalg.svd(pc, full_matrices=False)[2][0]
        return v if v @ ch >= 0 else -v

    m_large = int(round(LARGE_MM / P))
    large = np.zeros(n)
    for i in np.flatnonzero(small >= np.radians(45.0)):
        a, b = side_dir(i, -1, m_large), side_dir(i, 1, m_large)
        if a is None or b is None:
            continue
        t8 = np.arccos(np.clip(a @ b, -1.0, 1.0))
        a4, b4 = side_dir(i, -1, m_large // 2), side_dir(i, 1, m_large // 2)
        t4 = np.arccos(np.clip(a4 @ b4, -1.0, 1.0)) if a4 is not None and b4 is not None else t8
        # a corner turns as much over half the windows; a curve about half as much (two
        # corners close together, a zigzag's U-turn, keep turning: compare up to 90 deg)
        if t4 >= CONC * min(t8, np.pi / 2):
            large[i] = t8
    if loop:
        nb = np.stack([np.roll(small, -s) for s in (-2, -1, 1, 2)])
    else:
        pad = np.pad(small, 2)
        nb = np.stack([pad[2 + s:2 + s + n] for s in (-2, -1, 1, 2)])
    peak = (small >= nb.max(0)) & (small >= np.radians(45.0))
    cand = np.flatnonzero(peak & (large > np.radians(corner_deg)))
    cuts = []
    for q in cand:                       # one corner within 3 nodes: the sharper
        if cuts and q - cuts[-1] <= 3:
            if large[q] > large[cuts[-1]]:
                cuts[-1] = int(q)
            continue
        cuts.append(int(q))
    piece = np.zeros(n, int)
    for q in cuts:
        piece[q:] += 1          # nodes after a cut belong to the next piece (the cut node to both)
    out = np.zeros(n)
    for i in idx:
        lo, hi = i - h, i + h
        js = np.arange(lo, hi + 1)
        if not loop:
            js = js[(js >= 0) & (js < n)]
        own = piece[i]
        bounds = [cuts[own - 1]] if own >= 1 else []
        if own < len(cuts):
            bounds.append(cuts[own])
        if loop and cuts:                   # the pieces wrap: the last piece continues the first
            pw = piece % len(cuts)
            same = (pw[js % n] == pw[i]) | np.isin(js % n, [cuts[own % len(cuts)], cuts[own - 1]])
        else:
            same = (piece[js % n] == own) | np.isin(js % n, bounds)
        if loop and cuts == []:
            same[:] = True
        pts = xy[js[same] % n].astype(float)
        if len(pts) < 2:
            continue
        pts -= pts.mean(0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        out[i] = np.arctan2(vt[0, 1], vt[0, 0]) % np.pi
    return out, cuts


def err_deg(a, b):
    return np.degrees(np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2))


def bends(h, corner_deg):
    """5 mm legs turning by 20..135 deg: corners found, error away from the bends."""
    out = []
    for turn in (20, 30, 45, 60, 90, 135):
        a1 = np.radians(10.0)
        a2 = a1 + np.radians(turn)
        p1 = np.array([50 * np.cos(a1), 50 * np.sin(a1)])
        p2 = p1 + [50 * np.cos(a2), 50 * np.sin(a2)]
        xy = raster_poly([(0, 0), tuple(p1), tuple(p2)])
        t, cuts = tangents(xy, False, h, corner_deg)
        k = int(np.argmin(np.hypot(*(xy - np.round(p1)).T)))
        true = np.where(np.arange(len(xy)) < k, a1, a2)
        far = np.abs(np.arange(len(xy)) - k) > int(0.4 / P)       # more than 0.4 mm from the bend
        far[:10] = far[-10:] = False
        out.append(f"{turn}:{len(cuts)}/{err_deg(t[far], true[far]).max():.0f}")
    return " ".join(out)


def main():
    for corner_deg in (30.0, 40.0):
        for h in (0.6, 0.8, 1.0):
            print(f"corner {corner_deg:.0f} deg, h {h:.1f}: bends (turn:cuts/max err > 0.4 mm away) {bends(h, corner_deg)}")
    for h in (0.6, 0.8, 1.0):
        # straight lines, every degree; interior nodes (1 mm from the ends)
        e_line = []
        for d in np.arange(0, 180, 1.0):
            a = np.radians(d)
            xy = raster_line((0, 0), (100 * np.cos(a), 100 * np.sin(a)))
            t, _ = tangents(xy, False, h)
            e_line.append(err_deg(t[10:-10], a))
        e_line = np.concatenate(e_line)
        # circles
        e_circ = {}
        for R in (1.0, 1.6, 2.0, 5.0, 10.0):
            r = R / P
            m = max(int(2 * np.pi * r / 2), 24)
            th = np.linspace(0, 2 * np.pi, m, endpoint=False)
            xy = raster_poly([(r * np.cos(x), r * np.sin(x)) for x in th], loop=True)
            t, cuts = tangents(xy, True, h)
            true = (np.arctan2(xy[:, 1], xy[:, 0]) + np.pi / 2) % np.pi
            e_circ[R] = (np.mean(err_deg(t, true)), np.percentile(err_deg(t, true), 95), len(cuts))
        # zigzag: 10 mm lines along x, 0.4 mm connectors, 6 lines
        pts = []
        for j in range(6):
            y = 4 * j
            pts += [(0, y), (100, y)] if j % 2 == 0 else [(100, y), (0, y)]
        xy = raster_poly(pts)
        t, cuts = tangents(xy, False, h)
        true = np.zeros(len(xy))
        steps = np.diff(xy, axis=0)
        vert = np.zeros(len(xy), bool)
        vert[1:] |= steps[:, 0] == 0
        vert[:-1] &= True
        # node truth: on a connector (x at 0 or 100 and between two lines) 90 deg, else 0
        on_conn = ((xy[:, 0] == 0) | (xy[:, 0] == 100)) & (xy[:, 1] % 4 != 0)
        true[on_conn] = np.pi / 2
        ez = err_deg(t, true)
        line_nodes = ~on_conn & ~np.isin(np.arange(len(xy)), cuts)
        print(f"h {h:.1f} mm | lines: mean {e_line.mean():.2f} deg, 95% {np.percentile(e_line, 95):.2f}, max {e_line.max():.2f}"
              f" | circles R1/1.6/2/5/10: " + ", ".join(f"{v[0]:.1f}/{v[1]:.1f} ({v[2]} cuts)" for v in e_circ.values())
              + f" | zigzag: {len(cuts)} cuts of 10, line nodes max {ez[line_nodes].max():.1f} deg, connectors max {ez[on_conn].max():.1f}")


if __name__ == "__main__":
    main()
