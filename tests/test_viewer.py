"""Viewer / state / render: encoders, round trips, history replay, file output."""

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, Polygon

from rltp.config import Load2D, LoadSpec, PartSpec, Printer, Region
from rltp.cover.cover import get_pieces
from rltp.export.render import draw_layer, read_history, render, replay_history
from rltp.export.state import load_state, save_state
from rltp.export.viewer import (
    b64_u16, build_viewer_data, decode_u16, extract_data, nx_codes, nx_decode, render_html,
    replay_viewer_layer, rle, unrle, write_viewer,
)
from rltp.loop.optimize import History, LoopConfig, optimize_part
from rltp.part import build_part
from rltp.search.sa import make_proposer


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("viewer")
    poly = Polygon([(0, 0), (9, 0), (9, 3), (3, 3), (3, 7), (0, 7)]).difference(Point(1.5, 1.5).buffer(0.6))
    mesh = trimesh.creation.extrude_polygon(poly, 0.4)
    loads = LoadSpec([Region(side="xmin", width=0.4)], [Load2D(Region(side="xmax", width=0.4), (10.0, -4.0))])
    spec = PartSpec(Printer(), loads, terms=("strength", "time", "cont"))
    part = build_part(mesh, spec, baseline="concentric")
    hist = History(str(out / "history.jsonl"))
    # temperature 0: every accepted edit improves the score, so the best part is the last one
    optimize_part(part, make_proposer("random"), np.array([1.0, 0.3, 0.3, 0.0]),
                  LoopConfig(max_evals=160, burst=16, warmup=16), hist)
    hist.close()
    write_viewer(out / "viewer.html", part, hist.events, {"proposer": "random", **part.breakdown()})
    save_state(out / "state.npz", part, {"note": "test"})
    return out, part, hist.events


def test_encoders_round_trip():
    rng = np.random.default_rng(0)
    m = rng.random(200) < 0.4
    assert (unrle(rle(m)) == np.flatnonzero(m)).all()
    assert unrle(rle(np.zeros(5, bool))).size == 0
    W, n = 7, 49
    cells = np.arange(n)
    nxt = np.full(n, -1)
    nxt[[0, 1, 8, 15, 20]] = [1, 8, 15, 14, 13]
    assert (nx_decode(nx_codes(cells, nxt, W), cells, W, n) == nxt).all()
    x = rng.random(300) * 42.0
    assert np.allclose(decode_u16(b64_u16(x, 42.0), 42.0), x, atol=42.0 / 65535)


def test_state_round_trip(run):
    out, part, _ = run
    st = load_state(out / "state.npz")
    assert st["meta"]["n_layers"] == len(part.layers) and st["meta"]["note"] == "test"
    for L, ev, lay in zip(part.layers, part.evals, st["layers"]):
        assert (lay["mask"] == L.mask).all()
        assert (lay["nxt"] == L.nxt).all() and (lay["prv"] == L.prv).all()
        assert lay["seam"] == L.seam
        assert np.allclose(np.nan_to_num(lay["t_cell"]), np.nan_to_num(ev.timing.t_cell), atol=1e-4)
        pc = get_pieces(L)
        want = [pc.cells[i].tolist() for i in ev.timing.order]
        assert [s.tolist() for s in lay["strokes"]] == want
        assert lay["T"] == pytest.approx(ev.T)


def test_viewer_html_embeds_data(run):
    out, part, _ = run
    html = (out / "viewer.html").read_text()
    assert "__RLTP_DATA__" not in html
    data = extract_data(html)
    assert len(data["layers"]) == len(part.layers)
    assert data["hist"]["n"] > 0
    for L, d in zip(part.layers, data["layers"]):
        cells = unrle(d["rle"])
        assert (cells == np.flatnonzero(L.flat_mask)).all()
        assert (nx_decode(d["nx"], cells, d["W"], d["H"] * d["W"]) == L.nxt).all()


def test_script_breakout_is_escaped(run):
    _, part, _ = run
    data = build_viewer_data(part, None, {"name": "</script><b>x"})
    html = render_html(data)
    assert "</script><b>" not in html
    assert extract_data(html)["meta"]["name"] == "</script><b>x"
    assert data["hist"] is None


def test_history_replay_matches_final(run):
    out, part, events = run
    data = extract_data((out / "viewer.html").read_text())
    n = data["hist"]["n"]
    jsonl = read_history(out / "history.jsonl")
    assert len(jsonl) == len(events)
    accepted = sum(1 for e in events if e.get("type") == "edit" and e["accepted"])
    assert accepted > 0
    for k, L in enumerate(part.layers):
        nxt, seams = replay_viewer_layer(data, k, n)
        assert (nxt == L.nxt).all() and seams == L.seam
        states = list(replay_history(jsonl, k, L.n))
        _, nxt2, prv2, seams2, _ = states[-1]
        assert (nxt2 == L.nxt).all() and (prv2 == L.prv).all() and seams2 == L.seam
        base, _ = replay_viewer_layer(data, k, 0)
        assert (base == states[0][1]).all()


def test_render_writes_files(run, tmp_path):
    out, part, _ = run
    ps = render(out, mode="layer", dest=tmp_path / "a")
    assert len(ps) == len(part.layers) and all(p.stat().st_size > 1000 for p in ps)
    ps = render(out, mode="playback", layer=0, frames=3, color="order", dest=tmp_path / "b")
    assert len(ps) == 3
    ps = render(out, mode="history", frames=3, dest=tmp_path / "c")
    assert 1 <= len(ps) <= 3
    ps = render(out, mode="playback", layer=0, frames=4, fmt="gif", dest=tmp_path / "d")
    assert ps[0].suffix == ".gif" and ps[0].stat().st_size > 0


def test_draw_layer_helper(run, tmp_path):
    import matplotlib.pyplot as plt
    _, part, _ = run
    L = part.layers[0]
    fig, ax = plt.subplots()
    draw_layer(ax, L.mask, L.frame.to_dict(), L.nxt, None, L.seam, color="fi", fi=part.evals[0].fea.fi)
    fig.savefig(tmp_path / "x.png")
    plt.close(fig)
    assert (tmp_path / "x.png").stat().st_size > 1000
