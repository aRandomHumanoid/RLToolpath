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
# recommended settings per part (cycles c01-c20, docs/experiments.md); about 15-35 min each
# with 3-4 workers
python scripts/ribbons.py frame out/frame --angle 0 --alt-angle 90 --skin-search \
       --grid 5 --amp 1.5 --evals 1500 --then-local 750 --workers 4
python scripts/ribbons.py plate out/plate --angle 90 --alt-angle 0 --density-from fields \
       --density-norm station --density-smooth 3 --density-cases 50 --skin-search \
       --grid 5 --amp 1.5 --evals 1500 --then-local 750 --workers 4
python scripts/ribbons.py wrench out/wrench --angle 0 --density-gamma 0.5 --density-cases 50 \
       --skin-search --grid 5 --amp 1.5 --evals 1500 --then-local 750 --workers 4

# score a start without search (about 10 s; a density field fitted to 100 cases is computed
# once per part and cached in runs/ribbons/cache/, 10-15 min)
python scripts/ribbons.py frame out/frame_0 --angle 0 --alt-angle 90 --evals 0

# rescore saved layouts on load cases the search never saw (seeds 1-3)
python scripts/ribbon_heldout.py out/plate out/wrench --seeds 1,2,3

# run arms over the three parts and tabulate them against the reference
python scripts/ribbon_suite.py runs/ribbons/my_test --arm A="--angle {angle} --evals 0"
```

The part is one of the test parts (`frame`, `plate`, `wrench`), built with the settings of the
reference runs: `--directions path`, 10 + 10 random edge load cases, the `--adapt` weights
(strength 0.5, time 1, continuity 1, interlayer 0.3, infill 1, coverage 1, stiffness 0.5,
bulk 2). Every layout is scored in points against the part's slicer layout, the same footing as
the reference's numbers.

| option | default | meaning |
| --- | --- | --- |
| **start** | | |
| `--angle` | 0 | lines' angle, degrees (0 runs along x) |
| `--alt-angle` | none | a second family on odd infill layers (a cross-ply with `angle + 90`) |
| `--families` | 0 | K families rotating by layer, 180/K degrees apart |
| `--spacing` | 1.0 | line spacing, mm (bead 0.4 mm: 1.0 is 40% infill) |
| `--cross`, `--rings`, `--ring-gap`, `--alternate` | off, 0, 0.4, off | a crossing grid; rings beside every wall; odd layers lay in reverse order |
| **density** | | |
| `--density-gamma` | none | space each family by the load-agnostic stress magnitude^gamma, calibrated to the infill target |
| `--density-from` | stress | `fields`: the gradient field step's optimised density instead |
| `--density-cases` | 0 | fit the density to N + N random load cases instead of the part's 10 + 10 |
| `--density-norm` | none | `station`: every cross-section of a family keeps the uniform line count |
| `--density-smooth` | 0 | smooth the density along the lines, mm (bounds how far they lean) |
| `--density-on` | both | with `--alt-angle`: `first` or `second` spaces only that family |
| `--density-hi`, `--density-mix` | 0.8, 1 | density cap; blend of the field toward uniform |
| **skins** | | |
| `--skin-search` | off | search each solid layer's direction (0 or 90 degrees) with the warp |
| `--skin`, `--skin-from`, `--skin-ring-count`, `--skin-rings` | slicer | fixed skin classes, a saved layout's skins, rings in the skins (screens) |
| **warp and search** | | |
| `--grid`, `--amp` | 4, 1.5 | warp grid spacing and largest displacement, mm |
| `--z-levels`, `--max-lean` | 0, 45 | 3D ribbons: warp levels over the height; steepest lean, degrees (held exactly) |
| `--optimizer` | cma | `cma`, `local` (block moves), `elitist` |
| `--sigma`, `--popsize`, `--evals` | 0.3, 12, 1500 | step, population, budget (0: score the start only) |
| `--then-local`, `--local-sigma` | 0, 0.5 | then refine the best with N evaluations of local block moves |
| `--warm` | none | continue from a finished run's `result.json`: the whole point when the settings match, its warp as the middle level of a 3D one |
| `--cases` | 0 | search over N + N load cases instead of 10 + 10 |
| `--warp-per-family`, `--spacing-search` | off | each family its own warp; search the line spacing (c04, not kept) |
| **layout** | | |
| `--link-reach`, `--link-passes` | 4, 1 | longest connector merging two ends, mm; merge passes |
| `--workers`, `--seed` | 6, 0 | forked workers; search seed |

Each run writes `result.json` (the slicer layout, the start, the best point and the reference,
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

**Density families** (`density_family` in `rltp/surfaces/ribbons.py`): line i of a family sits
where the cumulative density across the family reaches i, at every station along it. The
density is rescaled per cross-section so the part's cross-section holds the uniform line count
(lines gather across the family and don't drift), smoothed along the lines (they don't lean to
follow it), and calibrated until the laid-out infill is the 40% target.

### Results

Points against each part's slicer layout (FI_agg, lower is stronger; strokes). "Unseen loads"
is the mean over load seeds 1–3, which no search saw, each against that seed's slicer layout
(`scripts/ribbon_heldout.py`). The reference is the edit-by-edit optimizer's best on the same
settings (c20 four-class run: gradient start + 8,000 annealing edits).

| part | reference | ribbons, c00 (warped, vertical) | ribbons, final | final on unseen loads | reference on unseen loads |
| --- | --- | --- | --- | --- | --- |
| frame | +61.0 (0.633, 940) | +81.2 (0.839, 360) | **+112.3** (0.805, 322)¹ | **+118.2** (0.789) | +46.4 (0.696) |
| plate | +19.5 (0.491, 1629) | +112.0 (0.786, 339) | **+139.6** (0.617, 319)² | **+123.6** (0.701) | −18.3 (0.654) |
| wrench | +73.1 (0.733, 727) | +81.0 (0.914, 196) | **+124.9 / +128.3** (0.858 / 0.861, 217)³ | **+125.5 / +127.3** (0.854 / 0.860) | +71.2 (0.729) |

¹ Uniform cross-ply, CMA-ES 750 then local moves 1,500 (c13); the recommended 1,500 + 750 gave
+109.8 and +110.2 (+114.2 on unseen loads), within the ~5-point run-to-run noise.
² The 100-case field density (c18); the 20-case smoothed field gave +142.5 and +142.8 with two
search seeds (+122.9 and +121.1 on unseen loads).
³ Two search seeds (c18, c19).

- **Ribbons score 51–120 points above the reference on its own load cases and 54–142 on
  unseen ones**: a third to a fifth of the strokes, 15–23% shorter prints, better bonding on
  frame and plate, and less fitting to one set of cases (on unseen loads ribbons move −16 to
  +6 points, the reference −2 to −38).
- **The reference is still stronger**: FI_agg 7–15% lower on unseen loads. Its gradient start
  places material and directions per cell; ribbons place it through the line spacing and a
  smooth warp, and keep their continuity.
- **What added the 28–44 points since c00** (`docs/experiments.md`): searching the skins with
  the warp, alternating cross-ply families, local refinement after CMA-ES, and density-spaced
  families that keep each cross-section's line count, are smoothed along the lines and are
  fitted to 100 load cases rather than the search's 20.

The improvement cycles behind these, and the changes that didn't pay, are in
[docs/experiments.md](docs/experiments.md) and summarized in the design doc.

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
  surfaces/ribbons.py  Ribbon, line_family, density_family (normalised, smoothed), stress_density,
                       streamline_family, offset_rings, Warp, Warp3D
  surfaces/layout.py   ribbons -> bead paths: rasterize, clip, cut, merge ends; layout_snapshot
  surfaces/cmaes.py    CMA-ES, LocalSearch (block moves), ElitistES (ask/tell)
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
  ribbon_suite.py      arms over the test parts, tabulated against the reference
  ribbon_heldout.py    rescore saved layouts on held-out load seeds
  ribbon_screen.py     score many starts in one process (angles, density)
  suite.py, cycles.py, compare.py, rescore.py, direction_maps.py, run_viewer.py
  analysis/            one script per question asked (ribbon_regions.py: where Σ FI^8 sits;
                       layer_directions.py: bead directions per layer)
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
height, density families space lines by the field and, normalised, keep each cross-section's
line count with straight lines past the part, smoothing along the lines bounds their lean,
local search moves few blocks and never loses its best, and CMA-ES converges on a quadratic.
