"""End to end through the command line on a small part."""
import json
from pathlib import Path

import pytest
import trimesh

from rltp.cli import main

ROOT = Path(__file__).resolve().parents[1]
DENSE = ["--walls", "0", "--solid-layers", "0", "--infill", "1", "--cell", "0"]   # every cell filled


@pytest.fixture
def small_part(tmp_path):
    m = trimesh.creation.box(extents=[8, 6, 0.6])
    m.apply_translation([4, 3, 0.3])
    p = tmp_path / "part.stl"
    m.export(p)
    loads = tmp_path / "loads.yaml"
    loads.write_text("supports:\n  - region: {side: xmin, width: 0.5}\n"
                     "loads:\n  - region: {side: xmax, width: 0.5}\n    force: [40, 0]\n")
    return p, loads


def test_baseline_cli(small_part, tmp_path):
    mesh, loads = small_part
    out = tmp_path / "base"
    assert main(["baseline", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--out", str(out)]) == 0
    for f in ("part.gcode", "viewer.html", "history.jsonl", "metrics.json"):
        assert (out / f).exists(), f
    meta = json.loads((out / "metrics.json").read_text())
    assert meta["final"]["layers"] == 3


def test_no_loads_scores_random_cases(small_part, tmp_path):
    """Without loads.yaml, strength is scored over 10 random force and 10 random moment cases,
    each with supports of its own and none special."""
    mesh, _ = small_part
    out = tmp_path / "noloads"
    assert main(["baseline", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--out", str(out), "--cell", "0"]) == 0
    meta = json.loads((out / "metrics.json").read_text())
    cases = meta["load_cases"]
    assert len(cases) == 20 and all("supports" in c[0] for c in cases)
    assert sum("moment" in c[0] for c in cases) == 10
    assert len(meta["final"]["case_fi_agg"]) == 20


def test_optimize_cli(small_part, tmp_path):
    mesh, loads = small_part
    out = tmp_path / "opt"
    assert main(["optimize", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--fill", "zigzag90", "--proposer", "sa",
                 "--weights", "1.0,0.3,0.3,0.3", "--max-evals", "300", "--warmup", "32",
                 "--out", str(out), *DENSE]) == 0
    meta = json.loads((out / "metrics.json").read_text())
    assert meta["final"]["fi_agg"] < meta["baseline"]["fi_agg"]
    lines = (out / "history.jsonl").read_text().splitlines()
    assert any('"type":"edit"' in ln for ln in lines)


def test_rotate_cli(small_part, tmp_path):
    mesh, loads = small_part
    out = tmp_path / "rot"
    assert main(["baseline", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--rotate", "--out", str(out), *DENSE]) == 0


def test_sparse_cli(small_part, tmp_path):
    """The printer.yaml defaults: walls, solid skins, optimizer-placed infill."""
    mesh, loads = small_part
    out = tmp_path / "sparse"
    assert main(["optimize", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--solid-layers", "1", "--max-evals", "200", "--warmup", "16",
                 "--weights", "0.5,1,1", "--out", str(out)]) == 0
    meta = json.loads((out / "metrics.json").read_text())
    b, f = meta["baseline"], meta["final"]
    assert b["cells"] < b["part_cells"] and f["unsupported"] == 0 and 0 < f["infill_pct"] < 100
    assert meta["weights"][4] == 1.0 and "infill" in meta["terms"]      # rule "both": the penalty is on
    assert abs(f["infill_pct"] - 40) <= 2 + abs(b["infill_pct"] - 40)   # and the band holds
    assert "inter" in meta["terms"] and meta["weights"][3] == 0.3      # Z bonding on by default
    assert 0 < f["bonded_pct"] <= 100 and f["weakest_bond_pct"] <= f["bonded_pct"]
    out2 = tmp_path / "noz"
    assert main(["optimize", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--max-evals", "50", "--warmup", "8",
                 "--weights", "0.5,1,1,0", "--out", str(out2)]) == 0           # a 4th weight turns it off
    assert "inter" not in json.loads((out2 / "metrics.json").read_text())["terms"]


def test_optimize_with_policy_cli(small_part, tmp_path):
    from rltp.rl.net import UNet, save_policy
    mesh, loads = small_part
    pol = tmp_path / "policy.pt"
    save_policy(pol, UNet())
    out = tmp_path / "pol"
    assert main(["optimize", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--policy", str(pol), "--max-evals", "100", "--warmup", "16",
                 "--cell", "0", "--out", str(out)]) == 0
    meta = json.loads((out / "metrics.json").read_text())
    assert meta["proposer"] == "policy" and meta["loop"]["evals"] <= 101


def test_adapt_preset_fills_only_unset_settings():
    """--adapt sets the self-improvement cycles' settings where the command line leaves the
    default, and leaves explicit ones alone."""
    import numpy as np
    from rltp.cli import _apply_adapt, build_parser
    from rltp.config import parse_weights
    base = ["optimize", "m.stl", "--printer", "p.yaml", "--out", "o"]
    args = build_parser().parse_args(base + ["--adapt", "--guide", "0.3", "--weights", "0.5,1,1"])
    w = parse_weights(args.weights)
    _apply_adapt(args, w)
    assert args.start == "gradient" and args.clamp_band == 1.0 and args.guide == 0.3
    assert args.stress_bias == 0.5 and args.strength_first == 3.0
    assert np.allclose(w, [0.5, 1, 1, 0, 0, 0, 0.5, 2.0])
    off = build_parser().parse_args(base)
    w0 = parse_weights(off.weights)
    _apply_adapt(off, w0)
    assert off.start is None and off.clamp_band is None and off.guide == 0 and w0[6] == w0[7] == 0
