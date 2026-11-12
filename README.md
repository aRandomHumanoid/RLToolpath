# rltp — RL toolpath optimizer

`rltp` slices a mesh into fully dense layers, fills each layer with a valid
cover (every cell visited exactly once by paths and loops), and improves each
layer's toolpath one edit at a time — with a trained policy, simulated
annealing, best-of-N greedy search, or random edits, all in the same loop
with the same evaluator. It writes G-code, a self-contained HTML viewer that
plays back nozzle motion and the optimizer's history, and a replayable log.

See `RL Toolpath Optimizer — Design Doc.md` for the design. This README covers
usage and what is (and is not) implemented.

## Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[test]"
```

## Use

```bash
# slice + baseline covers (zigzag | zigzag90 | concentric), no optimisation
rltp baseline examples/bracket.stl --printer configs/printer.yaml --loads examples/bracket_loads.yaml --out out/base

# optimise; proposer defaults to the policy when --policy is given, else simulated annealing
rltp optimize examples/bracket.stl --loads examples/bracket_loads.yaml --printer configs/printer.yaml \
              --policy runs/exp01/policy.pt --weights 1.0,0.3,0.3 --out out/opt
rltp optimize ... --proposer sa|greedy|random|policy|policy-sample --max-evals 20000 --interlayer --rotate

# train: imitation warm-start then PPO through the curriculum
rltp train --config configs/train.yaml --out runs/exp01/

# PNG / GIF frames for slides (MP4 when ffmpeg is installed)
rltp render out/opt --mode layer|playback|history --layer 3 --format gif

# the evaluation plot: score vs evaluator calls on held-out shapes, all proposers
rltp bench --policy runs/exp01/policy.pt --families ribs --shapes 16 --budget 1024 --out bench/
```

Each `baseline`/`optimize` run writes `part.gcode`, `viewer.html`,
`history.jsonl`, `metrics.json`, plus `state.npz` (used by `render`).

`--weights` is strength, time, continuity and (v1.5) interlayer, in that order.
A zero weight turns that term off. `--interlayer` turns on the v1.5 term with weight 0.3.
`--rotate` is v0.5: each layer's grid is rotated to the dominant principal stress
direction from one isotropic FEA of that layer.

### Inputs

- `configs/printer.yaml`: motion (speed, accel, junction deviation), retraction,
  temperatures, start/end G-code templates, **material constants** (E1, E2, G12,
  ν12, Xt, Xc, Yt, Yc, S, Zt, Sz) and the **bond model** (τ, n, s_end, q_min).
  The material values are generic-PLA placeholders until you have coupon data.
- `loads.yaml` (see `examples/`): `supports` and `loads` are 2D regions in world
  XY (`box`, `circle`, `polygon`, or `side: xmin|xmax|ymin|ymax` with `width`),
  each load with a total force `[Fx, Fy]` split across layers by `h / height`.
  `point_loads` (`pos: [x,y,z]`, `force: [Fx,Fy,Fz]`, mesh coordinates) drive
  the v1.5 interface statics. A `material:`/`bond:` block here overrides the
  printer's.

## Layout

```text
rltp/
  config.py           printer / material / bond / loads / score settings
  model.py            Frame, Layer (mask, frame, nxt/prv, seams)
  part.py             PartState: cached evals, incremental deltas, full re-check, best part
  io/mesh.py          load + repair (trimesh), fails loudly if not watertight
  slicing/slice.py    section_multiplane -> shapely polygons (mid-layer cuts)
  slicing/raster.py   polygon -> cell mask (contains_xy), layer frames
  cover/cover.py      nxt/prv arrays, invariants, stroke walking, joins, seams
  cover/baselines.py  zigzag, concentric (ring peeling), greedy repair pass
  cover/ops.py        the 10 edit operators + vectorised action masks
  eval/fea2d.py       plane-stress Q4, orthotropic beads, Tsai-Wu, p-norm, warm-started PCG
  eval/timing.py      trapezoidal motion + junction deviation, travel/retract
  eval/metrics.py     continuity, layer evaluation, running term normaliser
  eval/interlayer.py  v1.5 interface statics + bond quality
  rl/obs.py           26 observation channels (no torch import)
  rl/env.py           Gymnasium LayerEnv
  rl/shapes.py        procedural shape families, random supports/loads
  rl/net.py           U-Net policy + value heads (2.1M weights), PolicyProposer
  rl/imitate.py       best-of-N teacher -> behaviour cloning
  rl/ppo.py           masked PPO, subprocess envs
  rl/train.py         curriculum driver
  search/sa.py        random / SA / greedy proposers
  loop/optimize.py    part loop, scheduler, history log
  export/gcode.py     strokes -> G-code
  export/viewer.py    JSON + template -> viewer.html
  export/render.py    matplotlib PNG/GIF/MP4 frames
  bench.py            score-vs-evaluator-calls comparison
  cli.py
```

## Status against the design doc

Implemented: v0 (single nozzle, 4-neighbour grid; strength, time, continuity),
v0.5 (per-layer grid rotation), v1.5 (interlayer term from interface statics),
i.e. the software side of milestones M0–M5.

Not implemented, deliberately: v1 triangular lattice (the doc gates it on
0°/90° beads proving limiting), v2 3D FEA stress maps (gated on Z coupons
failing where v1.5 predicted they'd hold), v3 dual nozzles / rotating bed (M7).
Physical validation (test prints, coupons, PrusaSlicer time comparison,
Klipper testing) is outside what code can do.

Interpretation choices worth knowing:

- **Seams** are stored as a set with one cell per closed loop; `Layer.seam_dict()`
  gives the doc's loop-id → cell form.
- **Flip right/down** act on the 2×2 block whose top-left is the chosen cell:
  "right" swaps a horizontal link pair for a vertical one, "down" the reverse.
- **Merge** picks the first adjacent end on a different stroke (E, S, W, N),
  falling back to closing the stroke's own ends into a loop (only ≥ 3 cells, so no 2-cycles).
- **Stroke order** is greedy nearest-neighbour from the layer's grid origin, so
  a layer's timing never depends on other layers (keeps edits local).
- **Failure index** is the Tsai-Wu inverse strength ratio (≥ 0, linear in load).
  Corner cells average the two stiffnesses and take the larger FI of the two
  material frames.
- **Term scales**: the loop sets each term's scale from 128 random edits, then
  freezes it so every proposer is scored identically. The RL env keeps a running
  scale per worker.

## First results (CPU only, Nov 2026)

These come from short runs on a 16-core CPU with no GPU. Each is one seed. Treat them as
a sign the pipeline works, not as conclusions.

**Imitation (stage 1: 32×32, strength only, flips + patches; 8k teacher states).**
The held-out data has 32 candidates per state. The cloned policy's top pick is the
teacher's label 24% of the time, against ~3% by chance. Its regret is ~0.8 normalized
units, against ~15 for picking a candidate at random.

**Bench (`runs/bench_bc`: 12 shapes from the held-out `ribs` family, 512-eval budget).**
The columns are the mean fraction of each shape's best-found gain, after 64 and 512 evaluator calls.

| proposer | @64 | @512 | evals to reach SA's 512-eval mean score |
| --- | --- | --- | --- |
| policy (argmax) | 0.84 | 0.87 | 89 |
| policy-sample | 0.90 | 0.94 | 33 |
| SA | 0.41 | 0.91 | 512 |
| greedy (best of 16) | 0.48 | 0.93 | 512 |
| random | 0.46 | 0.97 | 455 |

The policy is far faster early on. The argmax policy plateaus: on a single layer it replays
the same top proposals every burst once it reaches a local optimum, because the ban list
clears between bursts. Sampling (`--proposer policy-sample`, or `--sample`) fixes this. With
a large enough budget, plain random edits still end slightly higher per shape.

**PPO fine-tuning (150k steps after imitation) made the loop worse.** Training return went
up (≈1.0 → 2.5 per episode). But on the same benchmark, `policy-sample` needed 270
evaluations to reach SA's 512-eval mean score, against 33 before fine-tuning. The argmax
policy fell from 0.87 to 0.75 of the best gain. It got worse on the training families too
(0.98 → 0.95 at 512), so this is not overfitting to shape families. The likely cause is that
the env and the loop optimize different things. `LayerEnv` applies every proposed edit,
including ones that lower the score. The loop rejects those (once the temperature is 0). PPO
learns sequences that pay off only if worsening steps are kept. Making the env apply the
loop's acceptance rule (and ban rejected actions) would align the two. Until then, use the
imitation checkpoint (`runs/s1_cpu/s1_strength/policy_bc.pt`).

**Term normalization.** The doc divides each term by the std of its per-edit change. On the
bracket this made strength dominate. Its per-edit std is tiny (4.7e-4), while its total
gain over a run is ~300 such units. So with weights 1.0/0.3/0.3/0.3, SA cut FI_agg by 13%
and travel by 34%, but print time rose 4%. If the weights should mean "relative importance
of the total change", the scales need another reference, such as baseline-to-optimum
ranges per term.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Covers: cover invariants under hypothesis-fuzzed edit sequences and reverse∘reverse = id;
FEA against analytic tension bar (0.2 %) and cantilever (≈1–2 %); Tsai-Wu
uniaxial limits; warm-started PCG = direct solve; planner limits; incremental
scoring = full re-check; interface statics resultants; G-code extrusion = one
cell volume per cell; the doc's tension-bar sanity test (beads end up along the
load); env/network/policy-in-loop; CLI end to end; viewer data and replay.
