"""End to end through the command line on a small part."""
import json
from pathlib import Path

import pytest
import trimesh

from rltp.cli import main

ROOT = Path(__file__).resolve().parents[1]


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


def test_optimize_cli(small_part, tmp_path):
    mesh, loads = small_part
    out = tmp_path / "opt"
    assert main(["optimize", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--fill", "zigzag90", "--proposer", "sa",
                 "--weights", "1.0,0.3,0.3,0.3", "--max-evals", "300", "--warmup", "32",
                 "--out", str(out)]) == 0
    meta = json.loads((out / "metrics.json").read_text())
    assert meta["final"]["fi_agg"] < meta["baseline"]["fi_agg"]
    lines = (out / "history.jsonl").read_text().splitlines()
    assert any('"type":"edit"' in ln for ln in lines)


def test_rotate_cli(small_part, tmp_path):
    mesh, loads = small_part
    out = tmp_path / "rot"
    assert main(["baseline", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--rotate", "--out", str(out)]) == 0


def test_optimize_with_policy_cli(small_part, tmp_path):
    from rltp.rl.net import UNet, save_policy
    mesh, loads = small_part
    pol = tmp_path / "policy.pt"
    save_policy(pol, UNet())
    out = tmp_path / "pol"
    assert main(["optimize", str(mesh), "--printer", str(ROOT / "configs/printer.yaml"),
                 "--loads", str(loads), "--policy", str(pol), "--max-evals", "100", "--warmup", "16",
                 "--out", str(out)]) == 0
    meta = json.loads((out / "metrics.json").read_text())
    assert meta["proposer"] == "policy" and meta["loop"]["evals"] <= 101
