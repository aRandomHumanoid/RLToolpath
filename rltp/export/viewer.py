"""JSON + template -> self-contained viewer.html.

Embedded JSON (compact; only in-part cells are stored):
{
  "v": 1, "meta": {...summary metrics...}, "terms": [...],
  "layers": [{
     "k", "z", "h", "H", "W", "f": [ox, oy, theta, p], "T",
     "kb"                              cells per bead width (4 on the 0.1 mm bead grid, else 1;
                                       files written before it existed may lack it: then 1)
     "rle": [start, len, ...]          in-part cells as runs over the flat index r*W+c,
     "nx": "0123..."                   nxt per in-part cell (sorted): 0 none, 1 +1, 2 +W, 3 -1, 4 -W
     "seam": [cell, ...],
     "t": base64 uint16 LE             print time per in-part cell, t = q / 65535 * T
     "fi": base64 uint16 LE | null     failure index per in-part cell, fi = q / 65535 * fimax
     "fimax",
     "order": [first cell of each stroke in print order],  (loops start at their seam)
     "ts": [...], "te": [...]          stroke print start / end times
     "fp": {"n": [...], "v": [...]}    fit mode only (--directions path): the moves each stroke
                                       prints (rltp.cover.beads._fit_runs), in print order:
                                       vertex count per stroke, then (column, row) of every
                                       vertex in hundredths of a cell (node units, as the grid)
  }],
  "hist": null | {
     "n": number of proposals, "ops": [names],
     "layer": [...], "op": [...], "cell": [...], "acc": "0101...", "d": [...]   one entry per proposal
     "snaps": [[k, applied, nx, seam], ...]    cover of layer k after `applied` proposals
     "diffs": [[iter, runs, nx, seam_add, seam_del], ...]    accepted edits, sorted by iter;
                                      runs = [start, len, ...] over the layer's in-part cell
                                      index (position in the sorted in-part cell list)
     "chart": {"x": [...], "total": [...], "s": {term: [...]}}  change since start (normalised units)
     "passes": [[iter, pass, total, best], ...], "end": {...}
  },
  "cases": null | [{                   load cases, in LoadSpec.all_cases order (Load cases tab)
     "kind": "primary" | "case" | "force" | "moment",   loads.yaml's loads, its load_cases, random
     "sup": [[ring, ...], ...]          each support region clipped to the part: rings of [x, y]
                                        (world mm; outer and holes, filled even-odd)
     "loads": [{"poly": [ring, ...], "f": [fx, fy] (N), "m": moment (N mm), "at": [x, y]}]
  }]
}
State "after s proposals" = nearest snapshot of the layer with applied <= s, then every
diff of that layer with applied <= iter < s.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from rltp.config import TERMS
from rltp.cover.cover import get_pieces
from rltp.cover.ops import OPS

TEMPLATE = Path(__file__).with_name("viewer_template.html")
PLACEHOLDER = "__RLTP_DATA__"


# ------------------------------------------------------------------ encoders

def b64_u16(x: np.ndarray, vmax: float) -> str:
    x = np.nan_to_num(np.asarray(x, float), nan=0.0)
    q = np.zeros(x.shape, "<u2") if vmax <= 0 else np.clip(np.rint(x / vmax * 65535), 0, 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def decode_u16(s: str, vmax: float) -> np.ndarray:
    q = np.frombuffer(base64.b64decode(s), "<u2").astype(float)
    return q / 65535 * vmax


def rle(flat_mask: np.ndarray) -> list[int]:
    m = np.asarray(flat_mask, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    out = np.empty(2 * starts.size, np.int64)
    out[0::2] = starts
    out[1::2] = ends - starts
    return out.tolist()


def unrle(r: list[int]) -> np.ndarray:
    if not r:
        return np.empty(0, np.int64)
    return np.concatenate([np.arange(s, s + n) for s, n in zip(r[0::2], r[1::2])])


def runs_of(sorted_idx: np.ndarray) -> list[int]:
    """[start, len, ...] runs of consecutive integers in a sorted array."""
    x = np.asarray(sorted_idx, np.int64)
    if x.size == 0:
        return []
    brk = np.flatnonzero(np.diff(x) != 1) + 1
    starts = np.concatenate([[0], brk])
    stops = np.concatenate([brk, [x.size]])
    out = np.empty(2 * starts.size, np.int64)
    out[0::2] = x[starts]
    out[1::2] = stops - starts
    return out.tolist()


def nx_codes(cells: np.ndarray, nxt_vals: np.ndarray, W: int) -> str:
    """Direction code per cell for its successor."""
    cells = np.asarray(cells, np.int64)
    nv = np.asarray(nxt_vals, np.int64)
    d = nv - cells
    code = np.zeros(cells.size, np.uint8)
    code[d == 1] = 1
    code[d == -1] = 3
    code[d == W] = 2      # after +-1 so a 1-wide grid reads as vertical
    code[d == -W] = 4
    code[nv < 0] = 0
    code[nv == -2] = 5    # an empty cell of a sparse layer
    return (code + 48).tobytes().decode("ascii")


def nx_decode(codes: str, cells: np.ndarray, W: int, n: int) -> np.ndarray:
    off = np.array([0, 1, W, -1, -W, 0])
    c = np.frombuffer(codes.encode("ascii"), np.uint8).astype(np.int64) - 48
    nxt = np.full(n, -1, np.int64)
    has = (c > 0) & (c < 5)
    nxt[cells[has]] = cells[has] + off[c[has]]
    nxt[cells[c == 5]] = -2
    return nxt


def _r(x, nd=4):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# ------------------------------------------------------------------ builders

def _viewer_nxt(L) -> np.ndarray:
    """nxt with -2 at a sparse layer's empty in-part cells (they show as background)."""
    nxt = L.nxt.astype(np.int64)
    if L.sparse:
        nxt[L.part.reshape(-1) & ~L.flat_mask] = -2
    return nxt


def _fitted_paths(L, order) -> dict:
    """Fit mode: each stroke's printed moves (fitted vertices), in print order."""
    from rltp.cover.beads import runs
    poly = runs(L)["poly"]
    n, v = [], []
    for pid in order:
        V = np.asarray(poly[int(pid)][2], float)
        n.append(int(len(V)))
        v.append(np.rint(V * 100).astype(np.int64).reshape(-1))
    return {"n": n, "v": np.concatenate(v).tolist() if v else []}


def _layer_data(L, ev) -> dict:
    from rltp.cover.beads import fit_mode
    cells = np.flatnonzero(L.part_mask.reshape(-1))
    pc = get_pieces(L)
    tm = ev.timing
    T = float(tm.T)
    fi = np.asarray(ev.fea.fi, float)[cells] if ev.fea.fi.size else np.zeros(cells.size)
    fimax = float(fi.max()) if fi.size else 0.0
    extra = {"fp": _fitted_paths(L, tm.order)} if L.kb > 1 and fit_mode(L) else {}
    return extra | {
        "k": int(L.k), "z": _r(L.z), "h": _r(L.h), "H": int(L.H), "W": int(L.W),
        "f": [_r(L.frame.ox, 6), _r(L.frame.oy, 6), _r(L.frame.theta, 8), _r(L.frame.p, 6)],
        "T": _r(T, 4),
        "rle": rle(L.part_mask.reshape(-1)),
        "nx": nx_codes(cells, _viewer_nxt(L)[cells], L.W),
        "seam": sorted(int(s) for s in L.seam),
        "t": b64_u16(tm.t_cell[cells], T),
        "fi": b64_u16(fi, fimax) if fimax > 0 else None,
        "fimax": _r(fimax, 6),
        "order": [int(pc.cells[pid][0]) for pid in tm.order],
        "ts": [_r(x, 3) for x in tm.stroke_start],
        "te": [_r(x, 3) for x in tm.stroke_start + tm.stroke_time],
        "n_strokes": int(pc.count),
        "n_cells": int(L.material.sum()),
        "kb": int(L.kb),
    }


def _downsample(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    idx = np.unique(np.rint(np.linspace(0, n - 1, cap)).astype(np.int64))
    return idx


def _hist_data(events: list[dict], layers, cap_points: int = 4000) -> dict | None:
    edits = [e for e in events if e.get("type") == "edit"]
    if not edits:
        return None
    ops = list(OPS)
    op_idx = {o: i for i, o in enumerate(ops)}
    cells_of = {k: np.flatnonzero(L.part_mask.reshape(-1)) for k, L in enumerate(layers)}
    Ws = [L.W for L in layers]

    n = max(int(e["iter"]) for e in edits) + 1
    lay = np.zeros(n, np.int64)
    op = np.zeros(n, np.int64)
    cell = np.zeros(n, np.int64)
    acc = np.zeros(n, np.uint8)
    dd = np.zeros(n)
    tot = np.full(n, np.nan)
    s_terms = {t: np.full(n, np.nan) for t in TERMS}
    diffs, snaps, passes = [], [], []
    last_edit = -1
    start = next((e for e in events if e.get("type") == "start"), None)
    end = next((e for e in events if e.get("type") == "end"), None)
    for e in events:
        typ = e.get("type")
        if typ == "edit":
            i = int(e["iter"])
            k = int(e["layer"])
            last_edit = max(last_edit, i)
            lay[i] = k
            name = e["op"]
            if name not in op_idx:
                op_idx[name] = len(ops)
                ops.append(name)
            op[i] = op_idx[name]
            r, c = e["cell"]
            cell[i] = int(r) * Ws[k] + int(c)
            acc[i] = 1 if e["accepted"] else 0
            dd[i] = e.get("d_total", 0.0)
            tot[i] = e.get("total", np.nan)
            for t in TERMS:
                s_terms[t][i] = (e.get("s") or {}).get(t, np.nan)
            if e["accepted"] and "diff" in e:
                df = e["diff"]
                dc = np.asarray(df["cells"], np.int64)
                order = np.argsort(dc, kind="stable")
                dc = dc[order]
                dn = np.asarray(df["nxt"], np.int64)[order]
                local = np.searchsorted(cells_of[k], dc)
                diffs.append([i, runs_of(local), nx_codes(dc, dn, Ws[k]),
                              list(df.get("seam_add", [])), list(df.get("seam_del", []))])
        elif typ == "snapshot":
            k = int(e["layer"])
            cells = cells_of[k]
            nxt = np.asarray(e["nxt"], np.int64)
            snaps.append([k, last_edit + 1, nx_codes(cells, nxt[cells], Ws[k]),
                          sorted(int(s) for s in e.get("seam", []))])
        elif typ == "pass":
            passes.append([int(e["iter"]), int(e["pass"]), _r(e["total"], 5), _r(e.get("best", np.nan), 5)])
    diffs.sort(key=lambda x: x[0])

    # chart: change since start, at x = proposals applied (0 = start)
    if start is not None:
        tot0 = float(start.get("total", tot[0]))
        sc = start.get("scales", {})
        raw0 = start.get("raw", {})
        s0 = {t: float(raw0.get(t, 0.0)) / float(sc.get(t, 1.0) or 1.0) for t in TERMS}
    else:
        tot0 = float(np.nan_to_num(tot[0]))
        s0 = {t: float(np.nan_to_num(s_terms[t][0])) for t in TERMS}
    idx = _downsample(n, cap_points)
    x = [0] + (idx + 1).tolist()
    chart = {"x": x, "total": [0.0] + [_r(v - tot0, 5) for v in tot[idx]], "s": {}}
    for t in TERMS:
        ser = s_terms[t][idx] - s0[t]
        if np.all(~np.isfinite(ser)) or np.nanmax(np.abs(ser)) < 1e-12:
            continue
        chart["s"][t] = [0.0] + [_r(v, 5) for v in ser]

    return {
        "n": int(n), "ops": ops,
        "layer": lay.tolist(), "op": op.tolist(), "cell": cell.tolist(),
        "acc": (acc + 48).tobytes().decode("ascii"),
        "d": [_r(v, 4) for v in dd],
        "snaps": snaps, "diffs": diffs, "chart": chart, "passes": passes,
        "end": {k: v for k, v in (end or {}).items() if k in ("iter", "evals", "total", "stop")},
        "weights": (start or {}).get("weights"),
    }


def _rings(g, tol: float = 0.02) -> list:
    """Polygon rings (outer and holes) of a shapely area, simplified, as [[x, y], ...]."""
    out = []
    for poly in getattr(g.simplify(tol), "geoms", [g.simplify(tol)]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        for ring in [poly.exterior, *poly.interiors]:
            out.append([[round(float(x), 3), round(float(y), 3)] for x, y in ring.coords[:-1]])
    return out


def _case_data(part) -> list | None:
    """Each load case's supports and loads for the viewer: regions clipped to the part
    outline, forces (N), moments (N mm) and the loaded area's centroid."""
    import shapely
    loads = part.spec.loads
    polys = [L.poly for L in part.layers if getattr(L, "poly", None) is not None]
    if not loads.has_2d or not polys:
        return None
    outline = shapely.union_all(polys).buffer(0.05)

    def clip(r):
        if r._geom is None:
            r.resolve(outline.bounds)
        return r._geom.intersection(outline)

    n0 = 1 if loads.has_primary else 0          # load-agnostic runs have no primary case
    n_explicit = len([c for c in loads.cases if c])
    out = []
    for i, (case, sups) in enumerate(zip(loads.all_cases, loads.case_supports)):
        kind = "primary" if i < n0 else "case" if i < n0 + n_explicit else \
            "moment" if any(ld.moment for ld in case) else "force"
        lds = []
        for ld in case:
            g = clip(ld.region)
            c = g.centroid if not g.is_empty else ld.region._geom.centroid
            lds.append({"poly": _rings(g), "f": [_r(ld.force[0], 3), _r(ld.force[1], 3)],
                        "m": _r(ld.moment, 3), "at": [_r(c.x, 3), _r(c.y, 3)]})
        out.append({"kind": kind, "sup": [_rings(clip(r)) for r in sups], "loads": lds})
    return out


def build_viewer_data(part, events: list[dict] | None = None, meta: dict | None = None) -> dict:
    from rltp.export.state import _jsonable
    layers = [_layer_data(L, ev) for L, ev in zip(part.layers, part.evals)]
    if meta is None:
        meta = part.breakdown()
    return {
        "v": 1,
        "meta": _jsonable(meta),
        "terms": list(TERMS),
        "layers": layers,
        "hist": _hist_data(events, part.layers) if events else None,
        "cases": _case_data(part),
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise RuntimeError("viewer template is missing the data placeholder")
    return html.replace(PLACEHOLDER, payload, 1)


def write_viewer(path, part, events: list[dict] | None = None, meta: dict | None = None) -> None:
    Path(path).write_text(render_html(build_viewer_data(part, events, meta)), encoding="utf-8")


def extract_data(html: str) -> dict:
    """Inverse of render_html (for tests and tooling)."""
    a = html.index('<script type="application/json" id="rltp-data">') + len(
        '<script type="application/json" id="rltp-data">')
    b = html.index("</script>", a)
    return json.loads(html[a:b])


# ------------------------------------------------------------------ python mirror of the JS replay

def replay_viewer_layer(data: dict, k: int, s: int) -> tuple[np.ndarray, set]:
    """Cover (nxt over the full grid, seams) of layer k after s proposals, decoded from
    the viewer JSON exactly as the page does it."""
    d = data["layers"][k]
    cells = unrle(d["rle"])
    W, n = d["W"], d["H"] * d["W"]
    h = data["hist"]
    if h is None:
        return nx_decode(d["nx"], cells, W, n), set(d["seam"])
    best = None
    for sk, applied, nx, seam in h["snaps"]:
        if sk == k and applied <= s and (best is None or applied >= best[0]):
            best = (applied, nx, seam)
    if best is None:
        raise ValueError(f"no snapshot of layer {k} at or before {s}")
    applied, nx, seam = best
    nxt = nx_decode(nx, cells, W, n)
    seams = set(seam)
    off = np.array([0, 1, W, -1, -W, 0])
    for it, runs, codes, sadd, sdel in h["diffs"]:
        if it < applied or it >= s or h["layer"][it] != k:
            continue
        dc = cells[unrle(runs)]
        c = np.frombuffer(codes.encode("ascii"), np.uint8).astype(np.int64) - 48
        nxt[dc] = np.where(c == 5, -2, np.where(c > 0, dc + off[c], -1))
        seams -= set(sdel)
        seams |= set(sadd)
    return nxt, seams
