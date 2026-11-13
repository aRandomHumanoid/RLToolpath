# rltp — RL toolpath optimizer

`rltp` slices a mesh into layers with fixed walls, solid top and bottom skins and
sparse infill (or fully dense layers), covers each layer's filled cells with paths
and loops (every filled cell visited exactly once), and improves the toolpath and
where the infill goes one edit at a time — with a trained policy, simulated
annealing, best-of-N greedy search, or random edits, all in the same loop
with the same evaluator. It writes G-code, a self-contained HTML viewer that
plays back nozzle motion and the optimizer's history, and a replayable log.

See `RL Toolpath Optimizer — Design Doc.md` for the design and
[docs/evaluator.md](docs/evaluator.md) for how a toolpath is scored, term by term. This
README covers usage and what is (and is not) implemented.

## Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[test]"
```

`rltp train` and `rltp bench` start their workers with `fork`, so run them on Linux, or in WSL on
Windows.

## Use

```bash
# slice + baseline covers (zigzag | zigzag90 | concentric), no optimisation
rltp baseline examples/bracket.stl --printer configs/printer.yaml --loads examples/bracket_loads.yaml --out out/base

# optimise with simulated annealing (the default proposer)
rltp optimize examples/bracket.stl --loads examples/bracket_loads.yaml --printer configs/printer.yaml \
              --weights 1.0,0.3,0.3 --out out/opt
rltp optimize ... --proposer sa|greedy|random|policy|policy-sample --max-evals 20000 --interlayer --rotate
rltp optimize ... --load-cases 5 --load-seed 0   # also score strength on 5 random load cases
rltp optimize ... --walls 2 --solid-layers 3 --infill 0.4 --max-bridge 4   # the printer.yaml defaults
rltp optimize ... --infill 0.3 --infill-rule both --infill-band 0.02 --infill-weight 1   # remove 70%
rltp optimize ... --cell 0.1          # 0.1 mm grid: 0.4 mm beads placed anywhere to 0.1 mm (SA/greedy/random)
rltp optimize ... --fea-element 0.4   # coarse stress mesh: much faster per edit on 0.4 mm cells
rltp optimize ... --cell 0            # 0.4 mm grid (one cell per bead; needed for --policy)
rltp optimize ... --solver legacy     # the original FEA solver, to reproduce older runs bit for bit
rltp optimize ... --start stress --stress-bias 0.5 --strength-first 3   # direction guidance
rltp optimize ... --start gradient     # directions and infill density optimised for strength first
rltp optimize ... --chains 4 --workers 4 --reheat 0.5   # 4 annealing chains, keep the best; reheat on stalls
rltp optimize ... --walls 0 --solid-layers 0 --infill 1                    # every cell filled (original mode)

# train (frozen, 0.4 mm grid only): imitation warm-start then PPO through the curriculum
rltp train --config configs/train.yaml --out runs/exp01/

# PNG / GIF frames for slides (MP4 when ffmpeg is installed)
rltp render out/opt --mode layer|playback|history --layer 3 --format gif

# the evaluation plot: score vs evaluator calls on held-out shapes, all proposers
rltp bench --policy runs/exp01/policy.pt --families ribs --shapes 16 --budget 1024 --out bench/
```

Each `baseline`/`optimize` run writes `part.gcode`, `viewer.html`,
`history.jsonl`, `metrics.json`, plus `state.npz` (used by `render`).

`--weights` is strength, time, continuity, (v1.5) interlayer and infill target, in that order.
A zero weight turns that term off. `--interlayer` turns on the v1.5 term with weight 0.3.
With walls/infill the interlayer term (Z bonding) is on at 0.3 unless you give a fourth
weight, the fifth comes from `infill_weight`, and on a fine bead grid the sixth (coverage)
from `cover_weight`.
Each term is scored as a fraction of the part's baseline value, so a weight is the importance
of a 1% change. With `1.0,0.3,0.3`, a 1% drop in FI_agg is worth a 3.3% rise in print time.
To favor continuity, raise the third weight. `--norm std` restores the original per-edit-std
scaling, and `--burst-bans` the original per-burst ban list. Both exist for comparison only.
`--rotate` is v0.5: each layer's grid is rotated to the dominant principal stress
direction from one isotropic FEA of that layer.

### Inputs

- `configs/printer.yaml`: motion (speed, accel, junction deviation), retraction,
  temperatures, start/end G-code templates, **material constants** (E1, E2, G12,
  ν12, Xt, Xc, Yt, Yc, S, Zt, Sz) and the **bond model** (τ, n, s_end, q_min).
  **Shell and infill**: `walls` (2), `solid_layers` (3), the `infill` target (0.4, i.e.
  remove 60% of the inside), how it's held (`infill_rule`, `infill_band`, `infill_weight`)
  and `max_bridge` (4 mm). **Grid and analysis**: `cell` (0.1 mm: beads placed to 0.1 mm;
  0 for the 0.4 mm grid), `cover_weight` (1) and `fea_element` (0.1 mm). Setting walls and solid layers to 0 and infill to 1 fills every
  cell, the original mode; the code's built-in default is that mode too.
  The material values are generic-PLA placeholders. That's enough to rank toolpaths for the
  same part against each other, which is how the strength term is used; coupons are only
  needed for absolute numbers.
- `loads.yaml` (see `examples/`): `supports` and `loads` are 2D regions in world
  XY (`box`, `circle`, `polygon`, or `side: xmin|xmax|ymin|ymax` with `width`),
  each load with a total force `[Fx, Fy]` split across layers by `h / height`.
  `point_loads` (`pos: [x,y,z]`, `force: [Fx,Fy,Fz]`, mesh coordinates) drive
  the v1.5 interface statics. A `material:`/`bond:` block here overrides the
  printer's.
- Extra load cases in `loads.yaml`: `load_cases` is a list of cases, each a list of
  loads written like `loads`. `random_cases: N` adds N random cases, seeded by `random_seed`.
  `--load-cases` and `--load-seed` override those two. All cases share the supports.

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
  cover/ops.py        the 20 edit operators (10 path, 5 fill, 5 moves) + vectorised action masks
  cover/shell.py      walls, solid skins, starting infill, the support rule
  cover/beads.py      fine bead grid: deposition model, layout, 0.1 mm edits
  eval/fea2d.py       plane-stress Q4, orthotropic beads, Tsai-Wu, p-norm, warm-started PCG
  eval/timing.py      trapezoidal motion + junction deviation, travel/retract
  eval/metrics.py     continuity, layer evaluation, running term normaliser
  eval/interlayer.py  v1.5 interface statics + bond quality
  eval/stressmap.py   one isotropic stress solve for the stress-aligned start
  rl/obs.py           26 observation channels (no torch import)
  rl/env.py           Gymnasium LayerEnv
  rl/shapes.py        procedural shape families, random supports/loads
  rl/net.py           U-Net policy + value heads (2.1M weights), PolicyProposer
  rl/imitate.py       best-of-N teacher -> behaviour cloning
  rl/ppo.py           masked PPO, subprocess envs
  rl/train.py         curriculum driver
  search/sa.py        random / SA / greedy proposers
  search/fields.py    gradient layout step: field model, adjoint, starts, line-end cost
  search/chains.py    parallel annealing chains, best final kept
  search/parallel.py  forked worker processes (the field starts and the chains)
  loop/optimize.py    part loop, scheduler, reheating, history log
  export/gcode.py     strokes -> G-code
  export/viewer.py    JSON + template -> viewer.html
  export/render.py    matplotlib PNG/GIF/MP4 frames
  bench.py            score-vs-evaluator-calls comparison
  cli.py
```

## Status against the design doc

**Direction (Nov 12, 2026).** Parts are optimized by searching directly against the
evaluator, with simulated annealing by default. The RL policy is frozen: it still runs on the
0.4 mm grid (`--proposer policy`, `rltp train`, `rltp bench`) but isn't retrained or
extended. Next come a gradient layout step for strength (built: `--start gradient`; see the
results below), parallel chains (independent chains built: `--chains`), cheap screening and
learning within the run; see the design doc's Search against the evaluator. On the 0.1 mm
bracket, `--start gradient --chains 4` with D's guidance reaches +56.5 against D's +43.3,
better on every term; one part so far.

Implemented: v0 (single nozzle, 4-neighbour grid; strength, time, continuity),
v0.5 (per-layer grid rotation), v1.5 (interlayer term from interface statics),
i.e. the software side of milestones M0–M5.

Not implemented, deliberately: v1 triangular lattice (the doc gates it on
0°/90° beads proving limiting), v2 3D FEA stress maps (gated on Z coupons
failing where v1.5 predicted they'd hold), v3 dual nozzles / rotating bed (M7).
Physical validation (test prints, coupons, PrusaSlicer time comparison,
Klipper testing) is outside what code can do. Strength is treated as a relative,
theoretical score with placeholder material constants. The goal is good continuity and
decent strength, not calibrated strength predictions (see the design doc's Priorities),
so coupon work is deferred.

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
- **Term scales**: each term's change is divided by its value on the part's baseline.
  Strength and interlayer are −log FI_agg, already fractional, so they use a scale of 1.
  The loop, the bench and the RL env all use the same rule, so training rewards and
  loop scores agree. The loop's 128 warmup edits now only measure a typical edit's
  score change, which is the unit for annealing temperatures.
- **Rejected edits** stay banned until their layer changes, or with the interlayer
  term, until a neighbor changes. The loop stops when no layer has an untried edit left.
- **Load cases.** With more than one case, each case's FI is divided by its own baseline
  FI_agg, and FI_agg becomes a p-norm over every cell of every case, so it starts at 1 and
  force sizes drop out. The p-norm acts as a soft worst case: no case can be traded away
  for another. A random case is a disc load (4 cells in radius) at a random point on the
  outline away from the supports, in a random direction. `metrics.json` lists every case and
  reports each one's raw FI_agg as `case_fi_agg`; `fi_agg` is then the combined value.
  Observations and `--rotate` take stress directions from case 0, the `loads` block.
- **Shell and infill.** Walls are the outermost rings of cells around every outline, holes
  included, traced into closed loops and locked: no edit may change a wall's links, only
  its seam. Where a section is too thin for two rings each side, the inner ring stays an
  open path. A cell is solid if any of the `solid_layers` layers above or below lacks it,
  as slicers do it, so steps and ledges get skins too. Walls and solid cells must stay
  filled; every other cell is optional. The starting infill is rectilinear lines placed so
  the filled share of optional cells matches the target. `infill_pattern: stacked` (the
  printer.yaml default) runs every layer's lines the same way, so adjacent layers overlap
  and bond; `alternate` switches 0°/90° by layer. Skins alternate either way.
- **Infill target.** `infill` is the share of optional cells to keep (0.4 removes 60% of
  the inside). `infill_rule: penalty` adds a fifth score term, minus the distance from
  target, with weight `infill_weight`: weight 1 makes one percentage point off target cost
  as much as a 1% change in a weight-1 term. `band` rejects any edit that would take the
  part's infill outside target ± `infill_band` (or, from outside, further from it). `both`
  (the default) is the penalty inside the band. `free` uses the target only for the
  starting layout. A fifth `--weights` entry overrides `infill_weight`.
- **Support rule.** A filled cell must be within `max_bridge`/2 of a filled cell in the
  layer below, unless that layer has no part there at all. Removing cells must not strand
  anything above. The starting line pitch is capped at `max_bridge`.
- **Fill edits.** `add` and `remove` one optional cell, `fill5_0`/`fill5_90` fill a 5×5
  window's optional cells and zigzag it, and `clear5` empties a 5×5 window. Time and
  continuity count every filled cell, which is what makes material cost something. The
  trained policy only proposes the 10 path edits; annealing, greedy and random use all 20.
- **Moves** keep the layer's infill count: `add_trim`/`fill5_trim` add a cell (fill a 5×5
  window) and then trim as many path-end cells elsewhere in the layer; `remove_grow`/
  `clear5_grow` remove a cell (clear a window) and extend as many path ends. Trimming takes
  unbonded cells (nothing filled directly below or above) first, then the least stressed;
  growing takes bonded cells first, then extends the most stressed ends. Working at path
  ends keeps strokes whole. `stack9` copies the layer below's infill into a 9×9 window,
  so it bonds fully there, zigzags it along the copied lines, and trims or grows elsewhere
  to keep the count: the one edit that lines infill up between layers.
- **Z bonding (sparse parts).** Only cells filled in both layers bond, so the interlayer
  term's section is the bonded set, recomputed for every edit: less overlap between
  adjacent layers' toolpaths gives a smaller section and higher interface stress. It
  scores a generic pull and bending about both axes (sized so a fully bonded interface
  peaks at half the interlayer strength), plus any `point_loads` above the cut, along
  with the existing bond quality (time gap, stacked ends). `metrics.json` reports
  `bonded_pct` and `weakest_bond_pct`.
- **Stack model.** Sparse parts default to a laminate: bonded layers stretch together, so
  one plate with every layer's stiffness carries the whole load, and each layer's stress
  follows from that shared strain and its own beads and fill. The original split (each
  layer carries h/height of the load alone) charged sparse middle layers a solid layer's
  share and put the bracket's FI_agg at 2.8× the dense part's. `--stack split|laminate`
  picks either. Empty cells keep 0.1% stiffness and count as FI 0.
- **Stress mesh.** `fea_element` (mm, printer.yaml default 0.1) splits each cell into
  elements of that size for the strength analysis only. Each cell's FI is the p-mean of its
  elements', so FI_agg keeps its meaning. On the bracket, 0.1 mm elements raise FI_agg about
  35% over 0.4 mm ones (stress peaks at the holes and the inside corner get resolved) at
  about 19× the cost per edit; layouts optimized on the coarse mesh still improve under the
  fine one, by less. The cantilever test converges to beam theory as elements shrink.
- **Fine bead grid (`cell: 0.1`).** Cells are 0.1 mm and beads stay 0.4 mm (4 cells). A
  layer's paths are bead centerlines in 0.1 mm steps, so beads can sit at any 0.1 mm position
  and jog sideways. Material per cell is what the nozzle lays: each step a strip 4 cells wide,
  half-bead caps at path ends, the outer quarter at corners. A straight bead gives exactly 1,
  two overlapping beads 2. The FEA uses that material (stiffness min(D, 1), bead direction
  from the strips), and support, infill share and Z bonding count cells with D ≥ 0.5. Walls
  are centerline loops a whole bead apart along the true outline (accurate to 0.1 mm). Exact
  cover by 0.4 mm beads is usually impossible, so a coverage term (`cover_weight`, the sixth
  weight) charges gaps in walls/skin and any over-extrusion. Edits: shift a straight run one
  cell sideways, extend or retract a path end, join nearby ends with an L-shaped connector,
  refill a 1.6 mm window with beads 0.4 mm apart on the row or column the edit picks, clear
  a window's optional beads, copy the layer below into a window; reverse, split and
  move_seam carry over. The trained policy only knows the 0.4 mm grid, so fine grids use
  `--proposer sa|greedy|random`. Cells per bead must be even.
- **Directions.** `--start stress` lays each area's skin and infill along its main stress
  direction (0 or 90 degrees) from one isotropic stress solve; `stress4` adds 45/135-degree
  areas (fine grid). `--stress-bias F` places that share of proposals by local stress;
  `--strength-first X` starts the strength weight X times higher, easing back over annealing.
  `--start gradient` first optimises a bead-direction field and an infill-density field
  for in-plane strength (`rltp/search/fields.py`: the laminate strength term as a function
  of both fields, its adjoint gradient, about 45 s on the 0.1 mm bracket), then lays them out:
  0/90 areas, and infill lines that start and stop with the density field (`gradient4`
  adds 45/135 areas).
- **Gradient start and chains.** The field step runs from `--field-starts` (default 8)
  starting direction fields in parallel, each with continuation (`--no-field-continuation`
  skips it), and charges the bead-line ends it causes (density changes and direction-area
  boundaries) at the run's continuity and time weights (`--field-end-cost X` scales the
  fitted price, default 1; 0 ignores line ends). It lays every start out and
  ranks them by the whole score. `--chains N` runs N annealing chains in parallel (seeds
  seed..seed+N−1, from the `--chain-starts` best field starts in turn, default 1) and keeps
  the best; each chain's history and metrics go to `out/chain<i>/`. `--reheat F` anneals
  again from F × the starting temperature once greedy passes stall. `--workers` caps the
  processes; set `OMP_NUM_THREADS=1`. Parallel work needs fork (Linux, WSL); elsewhere the
  starts and chains run one after another.
  Refills come in 1.6, 3.2 and 6.4 mm and keep the area's infill share; `flip_region` and
  `turn_region` turn a whole same-direction area by 90 or 45 degrees.
- **Diagonal beads (fine grid).** A staircase of one-cell steps (6 or more, constant signs) is
  a straight diagonal bead: deposited as a rotated bead, printed and timed as one straight
  move, and modelled with 45/135-degree stiffness and strength. `refill_45/135` lay them.
- **Not built on purpose:** extra walls or infill where stress is high as a rule, to see
  whether the optimizer gets there itself (see the design doc). Annealing doesn't; the
  gradient start does, by optimizing the strength term.
- **Not yet with walls/infill:** `--rotate`, and a policy trained with the fill edits.
- **LayerEnv** follows the loop at temperature 0. A proposal that doesn't raise the
  score is not applied, earns 0, and is masked until the next accepted edit.
  `accept: all` in a training config restores the original env.

## First results (CPU only, Nov 2026)

These results cover curriculum stage 1 only: 32×32 grids, strength only, flips and patch
refills, run on the CPU only (16 cores; the GPU wasn't used). Everything until the shell-and-infill results
used fully filled layers; to rerun those bracket commands now, add `--walls 0
--solid-layers 0 --infill 1 --cell 0 --fea-element 0 --solver legacy`. Training and
benching are deterministic.
Retraining `runs/s1_cpu` from scratch in WSL reproduced every log line and both
checkpoints bit for bit. Strength is reported as the percent drop in FI_agg. That is a
relative number; see Priorities in the design doc.

**Imitation (8k teacher states, 32 candidates each).** On validation states, the cloned
policy's top pick among the 32 candidates is the teacher's best 24% of the time, against
about 3% by chance. Its mean regret is 0.84 edit-std units, against 1.89 for picking a
random candidate. (An earlier version of this section said about 15; that was wrong.)
Validation states are split at random, so they share shapes with training states; the
bench below is the held-out test.

**Bench: imitation checkpoint, 48 held-out `ribs` shapes × 3 seeds, up to 512 evaluator
calls.** Each cell is the mean percent drop in FI_agg. 95% bootstrap intervals over shapes
are about ±3.5 points.

| proposer | @32 | @64 | @128 | @512 |
| --- | --- | --- | --- | --- |
| policy (argmax) | 17.0 | 17.1 | 17.4 | 17.5 |
| policy-sample | 16.5 | 17.3 | 17.5 | 17.7 |
| SA | 4.1 | 7.7 | 12.3 | 17.3 |
| greedy (best of 16) | 4.1 | 7.5 | 12.4 | 17.4 |
| random | 4.9 | 8.5 | 13.1 | 17.5 |

- **Speed.** The policy gets in about 32 evaluator calls what annealing needs about 500
  for. Shape by shape, `policy-sample` reaches SA's final score in a median of 142 calls,
  on 39 of 48 shapes. At 64 calls it beats SA on all 48 shapes. That meets M3's exit
  criterion.
- **Saturation.** By 512 calls every proposer ends at about the same place. This task is
  nearly saturated and can't separate methods at large budgets; stage 2 (64×64, time and
  continuity) is the real test.
- **Correction.** The first version of this section reported 33 calls, from 12 shapes and
  one seed. One shape with 4× the typical gain drove that; on the 48-shape set, with the
  old loop, it was 89.

```bash
rltp bench --policy runs/s1_cpu/s1_strength/policy_bc.pt --families ribs --shapes 48 --seeds 3 \
           --budget 512 --workers 12 --out bench/
```

**Persistent bans.** The loop used to clear its ban list at every burst, so on a single
layer the argmax policy replayed the same rejected proposals once it reached a local
optimum. Bans now last until the layer changes. Argmax now reaches SA's final score on 27
of 48 shapes (was 17), and `policy-sample`'s median falls from 181 to 142 calls.

**PPO does not beat imitation yet.** Each run fine-tunes the imitation checkpoint for 150k
steps with the same settings and seed, and is benched as above:

| checkpoint | policy-sample @32 | @64 | @512 | median calls to SA's final (shapes reached) | argmax @32 |
| --- | --- | --- | --- | --- | --- |
| imitation only | 16.5 | 17.3 | 17.7 | 142 (39/48) | 17.0 |
| PPO, original env | 15.7 | 16.5 | 17.5 | not reached on most shapes (24/48) | 16.2 |
| PPO, aligned env | 14.9 | 16.5 | 17.8 | 166 (39/48) | 14.0 |

- **Why the original env failed.** It applied every edit, worsening ones included. PPO
  raised its training return (about 1.0 → 2.5) by learning sequences the loop then
  rejects.
- **What aligning fixed.** The env now follows the loop's acceptance rule, which removed
  most of the damage.
- **What it didn't fix.** PPO's training return stayed flat (about 0.15 per episode), and
  its entropy drifted up, which hurts argmax.
- **Recommendation.** Use the imitation checkpoint, `runs/s1_cpu/s1_strength/policy_bc.pt`.
- **Next for PPO.** Warm up the value head before policy updates, and normalize advantages
  per shape, since returns vary about 100× between shapes. The aligned run is
  `configs/train_s1_ppo_from_bc.yaml`.

**Term scales (bracket, SA, 4,000 evals).** Changes against the zigzag baseline. Continuity
cost is strokes + 0.05 × travel cells + 0.5 × boundary ends.

| scaling, weights (strength/time/cont/inter) | FI_agg | interlayer FI | print time | continuity cost |
| --- | --- | --- | --- | --- |
| per-edit std (original), 1/.3/.3/.3 | −13.5% | +61% | +4.1% | −13.5% |
| relative, 1/.3/.3/.3 | −6.4% | −35% | +5.8% | −8.6% |
| per-edit std (original), .5/1/1/.3 | −10.0% | +57% | +0.6% | −28.4% |
| relative, .5/1/1/.3 | −3.5% | −29% | +4.2% | −20.3% |
| relative, .5/1/1/0 | −6.4% | off | +1.6% | −32.8% |

- **Old scaling.** Under per-edit std scaling, the interlayer term was effectively ignored
  and got about 60% worse whatever its weight.
- **Relative scaling.** Terms now trade as the weights say. The interlayer term has the
  most room to move (30% and more), so at equal weights it wins trades.
- **Continuity first.** `--weights 0.5,1,1,0` cut the continuity cost by a third (strokes
  120 → 98, travel −51%, boundary ends −34%) and FI_agg by 6%, for 1.6% more print time.

**Load cases (bracket, SA, 4,000 evals, weights 0.5/1/1/0).** Both covers are scored on
the `loads.yaml` case (0) and the 5 random cases from `--load-cases 5` (seed 0). Each cell is
the change in that case's FI_agg against the zigzag baseline.

| optimized for | case 0 | 1 | 2 | 3 | 4 | 5 | worst | print time | continuity cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case 0 only | −6.4% | −1.7% | −0.6% | −0.6% | −6.1% | −1.6% | −0.6% | +1.6% | −32.8% |
| all 6 cases | −2.7% | −2.5% | −6.3% | −5.0% | −2.7% | −2.4% | −2.4% | +1.3% | −31.8% |

- **No case got worse** when optimizing for case 0 alone, but cases 2 and 3 barely improved.
  Case 4 pulls near the tip like case 0, which is why it tracked case 0.
- **With all cases**, the gain is spread out: the worst case improves 4× as much (−2.4%
  against −0.6%), and case 0 gives up more than half its gain.
- **Continuity and print time are unaffected.**
- **Cost.** About 20% slower here (86 s against 71 s); the cases share the stiffness matrix.
- One part and one seed, so treat this as an illustration rather than a measured effect.
- `--load-cases 0` (the default) reproduces the single-case run bit for bit.

**Shell and infill (bracket, SA, 4,000 evals, weights 0.5/1/1/0).** 2 walls, 3 solid layers
top and bottom, infill starting at 20% with no target (this ran before the infill target
existed; `--infill 0.2 --infill-rule free` is the nearest setting now, with slightly
different starting lines), bridges up to 4 mm. The bracket has 10 layers, so layers 3–6
are sparse. Every row is scored with the laminate model and the `loads.yaml` case, so the
FI_agg values compare directly.

| cover | FI_agg | print time | filled cells | strokes | travel | continuity cost |
| --- | --- | --- | --- | --- | --- | --- |
| dense zigzag (old baseline) | 0.879 | 366 s | 29,420 | 120 | 704 mm | 328 |
| dense, optimized | 0.800 | 371 s | 29,420 | 98 | 342 mm | 220 |
| 2 walls, solid inside | 0.663 | 380 s | 29,420 | 160 | 712 mm | 254 |
| sparse baseline | 0.761 | 309 s | 22,724 | 208 | 905 mm | 326 |
| sparse, optimized | 0.749 | 310 s | 22,823 | 185 | 578 mm | 263 |

- **Walls do most of the strength work.** Beads that follow the outline carry the bending
  stress along the edges, so walls alone cut FI_agg by a quarter against the dense zigzag.
- **The sparse part is stronger than the dense zigzag** (FI_agg −15%) with 22% fewer cells and
  15% less print time, and 6% stronger than the optimized dense part.
- **The optimizer** cut continuity cost by 19% (travel −36%, 23 fewer strokes) and FI_agg by
  1.6%, and moved some infill (infill 19.1% → 20.3%, no unsupported cells). The middle
  layers' infill lines are each a separate stroke at the start, which is most of the
  remaining continuity cost; a longer run joins more of them.
- With `--load-cases 5` as well, the per-case FI_agg changes were −0.5% (case 0), −2.3%,
  −7.0%, −7.0%, −0.6% and −1.1%, with a similar continuity gain (−18.7%).
- 4,000 evals took 35 s (85 s with 6 load cases).

**Moves (bracket, same setup and seeds as below, alternating start, Z term off).** Edits that
keep the infill count let material shift without fighting the target. Objective as below.

| rule | without moves | with moves and `stack9` | paired gain |
| --- | --- | --- | --- |
| `both` (default) | 18.6 | 23.9 | +5.3, better on 6/6 |
| `band` ±0.5 | 18.2 | 24.2 | +6.0, 6/6 |
| `penalty`, weight 4 | 16.0 | 22.7 | +6.7, 6/6 |

All three still end at 40.0–40.1% infill. The strong penalty gains most: it had been
freezing material, and moves change no count.

**Z bonding (bracket, default rule, weights 0.5/1/1 + Z at 0.3, 3 seeds).** Each final layout
is re-scored with the Z term on, against the alternating start: 71% of the shared surface
bonded, 41% at the weakest interface. The objective adds 0.3 × the Z term to the one below.

| start, Z term | bonded | weakest interface | Z FI_agg | in-plane FI_agg | print time | continuity | objective |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alternating, off | 71.3% | 41.2% | +70.4% | −0.9% | +0.1% | −23.7% | +8.2 |
| alternating, on | 71.6% | 42.0% | −39.9% | −1.2% | +1.9% | −18.4% | +32.4 |
| stacked 90°, on | 75.4% | 52.9% | −41.0% | −0.3% | +1.9% | −18.2% | +32.3 |
| stacked 0°, on | 75.8% | 54.5% | −41.1% | −0.1% | +1.8% | −21.9% | +36.0 |

- **Ignoring Z costs Z.** Optimizing time and continuity alone made the interface failure
  index 70% worse. The bonded share didn't change, so that came from bond quality (time
  between a cell and the one above it, stacked ends). With the term on, it improved 40% for
  about 2% more print time.
- **Overlap is mostly set by the start.** Alternating 0°/90° infill bonds only where lines
  cross, and local edits barely change that (41% → 42% at the weakest interface, even with
  `stack9`). Stacked lines start at 58% and end around 54%, hence the new default.
- The Z FI_agg values are relative: the generic cases are sized to half the interlayer
  strength, and bond quality near 0.5 (about 30 s between layers) puts the absolute value
  above 1.

**0.1 mm bead grid (bracket, printer.yaml defaults, weights 0.5/1/1 + Z 0.3 + coverage 1,
SA, 4,000 evals, 2 seeds).** Against the 0.4 mm grid's stacked, Z-on runs from the table
above. FI values aren't comparable across grids (different meshes), so each is a change
from its own start.

| grid | in-plane FI_agg | Z FI_agg | continuity | strokes | print time | infill | weakest bond | gaps / over-extrusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 mm | −0.8%, −1.0% | −38%, −36% | −22%, −22% | 162 → 139 / 148 | +0.8%, +0.3% | 40.5%, 40.0% | 47%, 48% | 0.7% / 4.5 → 5.6% |
| 0.4 mm | −0.8%, −1.9% | −42%, −38% | −16%, −16% | 268 → 239 / 238 | +1.8%, +3.4% | 40.2%, 40.7% | 55% | n/a |

- **It works end to end**: every constraint holds (walls, support, infill band), G-code
  extrusion matches the material model, and the viewer replays it.
- **Continuity gains more** (−22% against −16%): connectors join lines more freely and the
  walls follow the outline exactly.
- **Over-extrusion rose about a point**, traded for continuity and Z bonding; `cover_weight`
  sets that trade. The weakest interface's bonded share fell from 58% as lines moved.
- **Cost**: 112 ms per edit with the default solver (190 ms with `--solver legacy`, which
  these runs used: about 17 minutes each), against 7 ms on the 0.4 mm grid with 0.4 mm
  elements. The trained policy can't run on it (`--cell 0` for that).

**Bead directions and search guidance (0.1 mm bracket, printer.yaml defaults, weights
0.5/1/1 + Z 0.3 + coverage 1, SA, 4,000 evals, 2 seeds).** Objective = 0.5 × strength + time +
continuity + 0.3 × Z + coverage + infill penalty, as a % change from the slicer start (the
stress start itself scores +11.2). Diagonal edits were available in every run.

| run | objective | in-plane FI_agg | Z FI_agg | print time | continuity cost | strokes |
| --- | --- | --- | --- | --- | --- | --- |
| A: slicer start | +43.8 [41.8, 45.8] | 0.978 (−0.9%) | 4.07 | 342.6 s | 200 | 127 |
| B: stress start | +38.7 [36.8, 40.5] | 0.961 (−2.7%) | 4.41 | 344.9 s | 205 | 140 |
| C: B + `--stress-bias 0.5` | +40.7 [35.4, 46.1] | 0.965 (−2.3%) | 4.30 | 346.4 s | 196 | 130 |
| D: C + `--strength-first 3` | +43.3 [42.0, 44.7] | 0.959 (−2.8%) | 4.16 | 342.8 s | 199 | 135 |

- **The optimizer doesn't find the stress directions on its own**: from the slicer start it
  improved in-plane FI_agg by 0.9%, against 2.8% for the stress start with guidance (D).
- **D ties A overall** and ends 2% stronger in-plane, at the same print time and continuity;
  A's edge is Z (4.07 against 4.16) and 8 fewer strokes. The stress start alone (B) loses
  ground on continuity and Z; strength-first recovers it.
- **Diagonals stayed rare**: under 0.5% of the interior material ended at 45/135 degrees. The
  bracket's legs are loaded along x and y.
- **No extra infill where stress is high**: the infill share by stress quarter hardly moved
  (slicer start: 39/38/42/42% → 41/37/42/41%, quietest to most stressed). With the walls and
  skins carrying the load, a little more infill doesn't pay for its print time here.
- **Annealing temperature**: on fine grids its unit is now the robust spread of a typical
  edit's change (1.4826 × median), not the std. Region turns and big refills change the score
  10–100× more than a typical edit, which inflated the std 18× and made annealing accept most
  bad edits: the first try of this table ended where it started.
- About 30 minutes per run with 8 in parallel.

**Gradient start (same setup and objective; the gradient step is the design doc's Search
against the evaluator, step 1).**

| run | objective | in-plane FI_agg | Z FI_agg | print time | continuity cost | strokes |
| --- | --- | --- | --- | --- | --- | --- |
| stress start, no search | +11.2 | 0.958 (−3.0%) | 5.09 | 341.7 s | 278 | 142 |
| gradient start, no search | +5.9 | 0.858 (−13.1%) | 4.44 | 345.9 s | 319 | 180 |
| D (from the table above) | +43.3 [42.0, 44.7] | 0.959 (−2.8%) | 4.16 | 342.8 s | 199 | 135 |
| E: gradient start + D's guidance | +33.6 [32.2, 35.1] | 0.879 (−11.0%) | 4.28 | 346.2 s | 236 | 150 |
| F: gradient start, plain annealing | +30.8 [28.8, 32.8] | 0.875 (−11.3%) | 4.15 | 349.8 s | 242 | 158 |

- **Four times D's strength gain, and it survives annealing**: 0.879 against 0.959 at
  about the same print time and Z. The field step takes 46 s.
- **It loses overall** (+33.6 against +43.3), mostly on continuity: against D, +4.4 points
  of strength, but −12.5 continuity and −1.0 print time. Dense infill where stress is high
  means lines that end where the density drops: 38 more strokes at the start. Annealing
  recovers as much continuity as it does from the stress start but not the head start; at
  these weights 10% more strength is worth less than 40 strokes.
- **Fields to lines**: giving each whole line its mean density lost most of the gain (field
  model 0.857, laid out 0.928); lines that start and stop with the field keep it (0.858).
- **Diagonals** (`gradient4`) add little strength (0.854) and double the strokes.
- E and F ran before the step shifted the density field to land on the infill target (they
  started at 40.8% infill); annealing brought both to 40.0%.

**Local optima: several starts, continuation, chains, reheating (same setup and objective,
D's guidance, 4 chains × 4,000 evals per run).** The field step now runs from 8 starting
direction fields (the stress map, turned 90°, uniform 0/90/45/135, 2 random smooth fields),
each through three stages (filter 1.9 → 1.2 → 0.75 mm, p 2 → 4 → 8), and ranks the laid-out
starts by the whole score. Here chain i started from the i-th best start (the default is now
the best start only, `--chain-starts 1`).

| run | start objective | chain finals | best | in-plane FI_agg of the best | continuity cost |
| --- | --- | --- | --- | --- | --- |
| S: stress start, reheat 0.5 | +11.2 | +40.8, +44.0, +42.6, +38.6 | +44.0 | 0.951 (−3.7%) | 194 |
| H: gradient start | +23.8 | +41.9, +42.7, +26.7, +33.9 | +42.7 | 0.888 (−10.1%) | 202 |
| G: H + reheat 0.5 | +23.8 | +41.9, +42.6, +26.7, +33.9 | +42.6 | 0.882 (−10.7%) | 201 |

- **Field step**: the 8 starts end within 2.2% of each other after rounding (1.3% with
  continuation); the stress map is within 0.1% of the best. Local optima cost little here.
  But starts as strong lay out very differently on continuity (266 against 323), so ranking
  them by the whole score lifted the gradient start from +5.9 to +23.8.
- **Annealing from one start** (S): the 4 seeds end 5.4 points apart; best of 4 is 2.5
  points over the mean chain. D's two seeds (+42.0, +44.7) fit the same spread.
- **From different starts** (H): the finals follow the starts' rank. The two best starts end
  at +41.9 and +42.7 with 3× the stress start's strength gain, about level with S overall;
  the 3rd and 4th (random0, stress) end at +26.7 and +33.9.
- **The remaining gap** (H's best against S's, −1.3 points): +3.4 strength and +1.0 Z, but
  −2.9 continuity and −2.4 print time. Both losses are in the 4 infill layers (15 more strokes
  and 7.5 s more there, with slightly less material), from lines that end at the edges of
  dense areas; the skin layers are as fast and have 7 fewer strokes.
- **Reheating** fired once in 16,000 chain evals (G chain 1, +0.1 points worse): at 4,000
  evals greedy passes are still gaining, so the budget, not a stall, ends the runs.
- About 53 minutes per chain with 12 processes running. The field step (8 starts) took 6
  minutes on 4 workers next to the other runs, about 2 minutes on 8 workers alone.
- **Line-end cost** (design doc, Search against the evaluator, step 1). A first version
  priced only density changes along the lines: it cut its predicted ends from about 120 to
  under 20 but made the laid-out starts worse (+14.2, +4.6, +8.2 at 1×, 2×, 4× against
  +23.8). Over 17 laid-out starts it didn't track strokes at all (r = 0.18). The corrected
  version also counts lines that run into an area of the other direction, in skins and
  infill (r = 0.57 with strokes, 0.64 with time). Starts with it, before any annealing:

  | line-end cost | FI_agg | Z FI | print time | continuity cost | strokes | start objective |
  | --- | --- | --- | --- | --- | --- | --- |
  | off | 0.857 | 4.45 | 347.2 s | 266.5 | 180 | +23.8 |
  | ½× | 0.879 | 4.16 | 327.6 s | 221.2 | 126 | **+47.3** |
  | 1× (default) | 0.888 | 4.03 | 328.6 s | 221.7 | 122 | **+47.3** |
  | 2× | 0.893 | 4.53 | 331.8 s | 263.7 | 146 | +28.2 |

  The start alone beats D's and S's finals (+43.3, +44.0): 10% stronger in-plane than the
  slicer start, faster than every other start, and fewer strokes than the stress start (142).

**Annealing from the corrected gradient start (same setup and objective, D's guidance, 4
chains × 4,000 evals; chains 0 and 2 from the best field start, 1 and 3 from the second).**

| run | start objective | chain finals | best | in-plane FI_agg | Z FI | print time | continuity cost | strokes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S: stress start | +11.2 | +40.8, +44.0, +42.6, +38.6 | +44.0 | 0.951 (−3.7%) | 4.31 | 343.9 s | 194 | 126 |
| K05: gradient, ½× line-end cost | +47.3 | +56.5, +45.1, +55.9, +49.3 | **+56.5** | 0.890 (−9.9%) | 3.89 | 338.9 s | 184 | 121 |
| K1: gradient, 1× (default) | +47.3 | +55.3, +48.2, +51.5, +49.3 | +55.3 | 0.902 (−8.7%) | 3.95 | 337.2 s | 185 | 113 |

(FI_agg, Z FI, time, continuity cost and strokes are the best chain's.)

- **The design doc's first exit is met on the bracket**: +56.5 against D's +43.3 with less
  continuity cost (184 against 199). Against S's best, the best part is better on every
  term at once: in-plane FI_agg 9.9% below the slicer start against 3.7% (D: 2.8%), lower
  Z FI, 5 s faster, fewer strokes, and fewer coverage gaps and less over-extrusion.
- **Every one of the 8 chains beats S's best** (+44.0).
- **Chains from the second-best field start end 5–10 points behind** those from the best,
  as in H, so all chains now start from the best (`--chain-starts 1`).
- ½× and 1× differ by less than the seed spread; 1× stays the default.
- About 32 minutes per chain with 8 processes running, plus 6 minutes for the field step.
- **One part so far**, and the line-end price was fitted on it; a second part comes next.

**Holding an infill target: rule or weight? (bracket, target 40% = remove 60%, SA, 4,000
evals, weights 0.5/1/1/0, 6 seeds.)** The objective is 0.5 × strength + time + continuity,
each as a % change from the shared start (40.3% infill); higher is better. The paired column
compares each rule with `band` ±2 on the same seed.

| rule | objective, mean [range] | final infill | paired vs `band` ±2 |
| --- | --- | --- | --- |
| `penalty`, weight 1 (= `both`, the default) | 18.6 [16.6, 19.9] | 40.1% | +0.2, better on 4/6 |
| `band` ±2 points | 18.4 [15.8, 22.1] | 41.2% | — |
| `band` ±0.5 points | 18.2 [16.2, 21.5] | 40.2% | −0.2, better on 3/6 |
| `penalty`, weight 4 | 16.0 [14.8, 17.8] | 40.0% | −2.4, better on 0/6 |
| `free` (3 seeds) | 18.4 [15.8, 22.1] | 41.4% | identical to `band` ±2 |

- **A moderate penalty and a band tie** within seed noise; the penalty lands closer to the
  target (40.1% against 41.2%) with less spread.
- **A strong penalty freezes material.** Every fill edit changes the cell count, and at
  weight 4 almost none pays for its own distance from target, so material stops moving,
  including the cells that join infill lines into longer strokes (continuity −15.7%
  against −18.3%).
- **The band ±2 never bound here.** Left alone the optimizer only adds about a point of
  infill in 4,000 evals, so `band` ±2 and `free` ran identically. A penalty only holds the
  target while the other terms pull weakly; a band holds it whatever the weights.
- **Default: `both`.** Weight 1 inside a ±2 band. Here it reproduces the weight-1 runs
  exactly, since the band never bound, and it keeps the guarantee if higher strength or
  continuity weights pull harder.

```bash
rltp optimize examples/bracket.stl --loads examples/bracket_loads.yaml --printer configs/printer.yaml \
              --weights 0.5,1,1,0 --max-evals 4000 --out out/bracket
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

Covers: cover invariants under hypothesis-fuzzed edit sequences and reverse∘reverse = id;
FEA against analytic tension bar (0.2 %) and cantilever (≈1–2 %); Tsai-Wu
uniaxial limits; warm-started PCG = direct solve; planner limits; incremental
scoring = full re-check; interface statics resultants; G-code extrusion = one
cell volume per cell; the doc's tension-bar sanity test (beads end up along the
load); env/network/policy-in-loop; CLI end to end; viewer data and replay; the loop's
persistent bans, with stopping on exhaustion certified as a single-edit local optimum; env
acceptance matching the loop; relative term scales; the imitation teacher in both scaling
modes; load cases (combined FI_agg starts at 1, a 10× copy of a case changes no edit's
score, incremental = full re-check, seeded random cases on the outline and off the supports);
sparse layers (baseline layout and support, random edits of all 15 kinds never change a wall,
empty a required cell or break the support rule, a sole support can't be removed, empty cells
carry no FI, the laminate charges sparse layers less than the split, the loop end to end).
