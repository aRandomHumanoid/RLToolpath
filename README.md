# rltp — surface infill fork

`rltp` slices a mesh into layers with fixed walls, solid top and bottom skins and infill, and
scores toolpaths with one evaluator: in-plane strength over many load cases, print time,
continuity, interlayer bond and coverage. It writes G-code and a self-contained HTML viewer.

**This fork fills the inside with surfaces.** A part is hollowed to its shell (one wall, two
solid layers top and bottom) and the inside is filled with "ribbons": surfaces through the
infill layers whose cut in every layer is one continuous bead path. Ribbons end against the
walls, may cross or bond with their neighbours, and their path ends are merged into longer
strokes while the toolpaths are generated. A search deforms the ribbons, and every candidate is
laid out as bead paths and scored by the unchanged evaluator.

- Design and the decisions behind it: [RL Toolpath Optimizer — Design Doc.md](RL%20Toolpath%20Optimizer%20%E2%80%94%20Design%20Doc.md)
  ("Why surfaces", "Surface infill").
- The evaluator, term by term: [docs/evaluator.md](docs/evaluator.md).
- Experiment log: [docs/experiments.md](docs/experiments.md).

**Where it comes from.** The fork starts from the edit-by-edit optimizer: annealing over
single-window edits with a gradient start (cycles 0–20, Nov 12–16, 2026), and an RL policy
frozen on Nov 12. That line's runs, viewers and working notes (`runs/selfimprove/CYCLES.md`) are
on branch `selfimprove-cycles`; the run data is commit `8efb311`. Its code still runs
(`rltp optimize`), and its best results are the reference ribbons are compared against
(`examples/c20_reference.json`).

## Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[test]"
```

Searches start their workers with `fork`, so run them on Linux, or in WSL on Windows.

## Ribbon infill

```bash
# straight ribbons at 0 degrees, 1.0 mm apart (40% infill), scored without search
python scripts/ribbons.py frame out/frame_0 --angle 0 --evals 0

# then search a smooth warp of them with CMA-ES (6 forked workers)
python scripts/ribbons.py frame out/frame_0 --angle 0 --evals 1800 --workers 6

# 45-degree lines plus one ring beside every wall, lines spaced 1.2 mm to hold 40% infill
python scripts/ribbons.py plate out/plate_ring --angle 45 --rings 1 --spacing 1.2

# a crossing grid; odd layers cut the other family at each crossing
python scripts/ribbons.py wrench out/wrench_grid --cross --alternate

# 3D ribbons: the warp varies over 3 height levels, no surface leaning past 45 degrees,
# starting from the vertical run above
python scripts/ribbons.py frame out/frame_3d --angle 0 --z-levels 3 --max-lean 45 \
       --warm out/frame_0/result.json --sigma 0.2
```

The part is one of the test parts (`frame`, `plate`, `wrench`), built with the settings of the
reference runs: `--directions path`, 10 + 10 random edge load cases, the `--adapt` weights
(strength 0.5, time 1, continuity 1, interlayer 0.3, infill 1, coverage 1, stiffness 0.5,
bulk 2). Every layout is scored in points against the part's slicer layout, the same footing as
the reference's numbers.

| option | default | meaning |
| --- | --- | --- |
| `--angle` | 0 | starting lines' angle, degrees (0 runs along x) |
| `--spacing` | 1.0 | starting lines' spacing, mm (bead 0.4 mm: 1.0 is 40% infill) |
| `--cross` | off | a second family at angle + 90, each at twice the spacing |
| `--rings` | 0 | fixed closed ribbons following every outline, beside the wall |
| `--ring-gap` | 0.4 | ring spacing, mm (0.4: touching, a solid band) |
| `--alternate` | off | odd layers lay the free ribbons in reverse order, so crossings alternate |
| `--grid`, `--amp` | 4, 1.5 | warp grid spacing and largest displacement, mm |
| `--z-levels` | 0 | 3D ribbons: warp levels over the infill height (0: vertical ribbons) |
| `--max-lean` | 45 | 3D ribbons: steepest lean from vertical, degrees (held exactly) |
| `--warm` | none | a finished vertical run's `result.json`: its best warp starts the middle level (same part, start family, `--grid`, `--amp`) |
| `--sigma`, `--popsize`, `--evals` | 0.3, 12, 1500 | CMA-ES step, population, budget (0: score the start only) |
| `--workers`, `--seed` | 6, 0 | forked workers; CMA-ES seed |

Each run writes `result.json` (the slicer layout, the start, the best warp and the reference,
each with its score breakdown and layout statistics, plus the search history), `state.npz`,
`viewer.html` and `layers.png` (an infill layer, start and best).

**How a layer is laid** (`rltp/surfaces/layout.py`):
1. **Clear and draw.** Clear the infill (walls and skin stay), then draw each ribbon onto the
   0.1 mm node grid.
2. **Clip and cut.** Keep the nodes inside the infill region and at least 0.3 mm from every bead
   laid before. A ribbon that crosses or crowds an earlier one is cut, and each cut end touches
   the bead it met.
3. **Merge ends.** Pull each open end back a bead, then join ends cheapest first with
   connectors along the walls or through open space. Ends left over are extended back to the
   wall.
4. **Copy or cut.** Vertical ribbons: every infill layer of the same outline gets the same
   layout. 3D ribbons: each infill layer is laid from the ribbons' cut at its own height.

### Results

Points against each part's slicer layout (FI_agg, lower is stronger; print time; strokes).
The reference is the edit-by-edit optimizer's best on the same settings (c20 four-class run:
gradient start + 8,000 annealing edits). Details in [docs/experiments.md](docs/experiments.md).

| part | reference | straight ribbons, no search | warped (vertical) | warped in 3D |
| --- | --- | --- | --- | --- |
| frame | +61.0 (0.633, 890 s, 940) | +52.8 (1.033, 727 s, 297) | +81.2 (0.839, 756 s, 360) | **+89.6** (0.735, 751 s, 374) |
| plate | +19.5 (0.491, 1347 s, 1629) | +81.9 (1.027, 1034 s, 339) | +112.0 (0.786, 1074 s, 339) | **+120.9** (0.610, 1086 s, 437) |
| wrench | +73.1 (0.733, 724 s, 727) | +53.8 (1.002, 621 s, 301) | +81.0 (0.914, 607 s, 196) | **+91.5** (0.802, 613 s, 240) |

- **Ribbons win on continuity and print time:** 240–437 strokes against 727–1629, 15–19%
  faster prints.
- **They trail on strength:** FI_agg 0.61–0.80 against 0.49–0.73.
- **3D surfaces add strength on every part** (FI_agg 12–22% lower than vertical) at a small
  continuity cost.
- **Rings beside the walls lose** at this spacing: each is a loop per layer and cuts the lines
  it meets (616–822 strokes).

The improvement cycles that follow are logged in `docs/experiments.md`.

## The edit-by-edit optimizer

```bash
# slice + baseline covers (zigzag | zigzag90 | concentric), no optimisation
rltp baseline examples/frame.stl --printer configs/printer.yaml --out out/base

# optimise: gradient start, then annealing over window edits (the reference runs' settings)
rltp optimize examples/frame.stl --printer configs/printer.yaml --directions path --adapt \
              --start gradient4 --fea-element 0.4 --out out/opt
rltp optimize ... --loads examples/frame_loads.yaml   # one known load case plus random ones
rltp optimize ... --chains 4 --workers 4              # 4 annealing chains, keep the best
rltp optimize ... --walls 0 --solid-layers 0 --infill 1   # every cell filled (original mode)
```

Each run writes `part.gcode`, `viewer.html`, `history.jsonl`, `metrics.json` and `state.npz`.
`scripts/suite.py` runs settings over the test parts in parallel; `scripts/compare.py`,
`scripts/rescore.py`, `scripts/run_viewer.py` and `scripts/analysis/` score, compare, view and
analyse its runs.

**Viewer tabs:** Layer (bead direction, print order, failure index or strokes; with
`--directions path`, the printed path at true angles), Nozzle playback, Optimization history,
Load cases. `viewer.html#layer:12` opens layer 12, `#loads:3` load case 3.

**Weights** (`--weights`): strength, time, continuity, interlayer, infill target, coverage,
stiffness, bulk, in that order. Each term is scored as a fraction of the part's baseline value,
so a weight is the importance of a 1% change in that term.

### Inputs

- `configs/printer.yaml`: motion (speed, accel, junction deviation), retraction,
  temperatures, start/end G-code templates, **material constants** (E1, E2, G12, ν12, Xt, Xc,
  Yt, Yc, S, Zt, Sz) and the **bond model** (τ, n, s_end, q_min). **Shell and infill**:
  `walls` (1), `solid_layers` (2), the `infill` target (0.4), `max_bridge` (4 mm). **Grid**:
  `cell` (0.1 mm: beads placed to 0.1 mm), `fea_element` (0.1 mm). The material values are
  generic-PLA placeholders, enough to rank layouts of one part against each other.
- `loads.yaml` (optional, see `examples/`): `supports` and `loads` as 2D regions (`box`,
  `circle`, `polygon`, `side`, `edge`), each load a force `[Fx, Fy]` and optionally a
  `moment`; `load_cases`, `random_cases`, `random_moments` for more cases.
- **No `loads.yaml` (load-agnostic, the default for the test parts).** Strength is scored over
  10 random force and 10 random moment cases, each fixing one outline edge and pushing or
  twisting another (a hole, side or arc at odds 3 : 2 : 1), none special.

## Layout

```text
rltp/
  surfaces/ribbons.py  Ribbon, line_family, offset_rings, Warp
  surfaces/layout.py   ribbons -> bead paths: rasterize, clip, cut, merge ends; layout_snapshot
  surfaces/cmaes.py    CMA-ES (ask/tell)
  config.py            printer / material / bond / loads / score settings
  model.py             Frame, Layer (mask, frame, nxt/prv, seams)
  part.py              PartState: cached evaluation, incremental deltas, full re-check
  slicing/             slice.py, raster.py, edges.py (outline edges for the random load cases)
  cover/               cover.py (paths, invariants), shell.py (walls, skins, support),
                       beads.py (fine bead grid, fit mode, edits), paths.py (true directions),
                       baselines.py, ops.py
  eval/                fea2d.py (laminate FEA, Tsai-Wu, p-norm), timing.py, metrics.py,
                       interlayer.py, stressmap.py, adapt.py
  search/, loop/       the edit-by-edit optimizer (annealing, guide, gradient start, chains)
  export/              gcode.py, viewer.py, state.py, render.py
  rl/                  the frozen RL policy (not developed in this fork)
scripts/
  ribbons.py           ribbon infill on a test part
  suite.py, cycles.py, compare.py, rescore.py, direction_maps.py, run_viewer.py
  analysis/            one script per question asked during the edit-by-edit cycles
examples/
  {frame,plate,wrench,hook,link,bracket}.stl (+ _loads.yaml), make_parts.py (5 mm parts),
  2mm/ (the 2 mm parts), c20_reference.json (reference scores for ribbons)
```

## How experiments are run

1. **Question.** One thing to explain or improve, e.g. "do edge-following rings pay for their
   strokes?"
2. **Diagnose first.** Score starts with `--evals 0` before searching (about 10 s each), and
   read what each term contributes (`result.json`'s `objective`).
3. **Change the code behind an option**, with tests (`tests/test_ribbons.py`). Defaults change
   only when an arm wins.
4. **Run arms** with `scripts/ribbons.py`, one setting per part, on the same footing as the
   reference (`examples/c20_reference.json`: the slicer layout, the c20 gradient start and its
   final after 8,000 annealing edits, all scored under the same settings).
5. **Record** results and reasons in `docs/experiments.md` as they come in; summarize what was
   kept here and in the design doc.

**Files.** Put run directories under `runs/<experiment>/<part>_<arm>/`. Their `result.json`
and logs are committed; `state.npz`, `viewer.html` and G-code stay local (`.gitignore`), so keep
a run's directory to rebuild its viewer.

**Running.** WSL or Linux, `OMP_NUM_THREADS=1` with several workers. A ribbon run's process
holds about 1.4 GB, and each forked worker grows to about 1.5 GB as its copy of the part
diverges. On the 16-core machine these ran on, with WSL capped at 15 GB, one run with 4 workers
at a time fits; three runs at once with 3 workers each went 3 GB into swap. One evaluation
(lay + re-score) takes about 1.2 s on the frame. Start long batches detached
(`setsid nohup ... &`).

## Tests

```bash
.venv/bin/python -m pytest -q
```

Covers the cover invariants under fuzzed edit sequences, the FEA against analytic solutions,
the time model, incremental scoring against a full re-check, load cases, sparse layers and the
support rule, the viewer and G-code, and ribbon layouts: rasterized chains are 4-connected and
follow the curve, laid layers are valid covers with no free ends, merging joins ends, crossings
become T-joints with no overlapping strokes, rings lay as loops, every infill layer is laid and
scores (the shared-cache fast path exactly as a plain re-score), the warp is bounded, the 3D
warp starts as the 2D one and never leans past its cap, 3D ribbons lay each layer from its own
height, and CMA-ES converges on a quadratic.
