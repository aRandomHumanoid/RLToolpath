# The evaluator

The evaluator scores a whole part's toolpath. It computes six raw terms, divides each term's
change by a scale, and adds them up with the user's weights. Every proposer (annealing,
greedy, random and the trained policy), the RL environment and the bench call the same two
methods, `PartState.propose` and `PartState.total`. A score from one of them therefore means
the same as a score from any other.

This page describes what the code does now. The design doc's Evaluator section explains why
it is built this way and what was planned, including the v2 3D FEA, which isn't built.

- [Terms at a glance](#terms-at-a-glance)
- [The weighted score](#the-weighted-score)
- [Strength](#strength)
- [Print time](#print-time)
- [Continuity](#continuity)
- [Interlayer (Z) strength](#interlayer-z-strength)
- [Infill target](#infill-target)
- [Coverage (fine grid)](#coverage-fine-grid)
- [Hard rules outside the score](#hard-rules-outside-the-score)
- [Incremental evaluation](#incremental-evaluation)
- [Annealing temperature unit](#annealing-temperature-unit)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Code map](#code-map)
- [Tests](#tests)
- [Adding a score term](#adding-a-score-term)
- [Known limitations](#known-limitations)

## Terms at a glance

Every raw term is signed so that higher is better. In the default `relative` mode, the scale
is what a term's change is divided by.

| Term | Raw value | Scale | On when |
| --- | --- | --- | --- |
| `strength` | −log FI\_agg over every in-part cell of every layer (and every load case) | 1 | its weight is > 0 and `loads.yaml` has supports and loads |
| `time` | −(sum of layer print times), in s | baseline print time | its weight is > 0 |
| `cont` | −(sum of layer continuity costs) | baseline continuity cost | its weight is > 0 |
| `inter` | −log FI\_agg over every interface cell | 1 | its weight is > 0 and the part has more than one layer; sparse parts default to 0.3 |
| `infill` | −\|infill fraction − target\| | 1 | sparse parts with `infill_rule` `penalty` or `both` |
| `cover` | −(gaps + over-extrusion) / in-part cells | 1 | fine bead grid (`cell` < `bead_width`) |

A term whose weight is 0 is left out of `spec.terms`. It is not computed, and it reads 0 in
`raw()`.

## The weighted score

```math
S = \sum_t w_t \, \frac{\mathrm{raw}_t}{\mathrm{scale}_t}
```

Only differences of S matter. A proposal is accepted or rejected on the change it makes, and
the best layout is the one with the highest S.

**Relative mode** (`--norm relative`, the default). `time` and `cont` are divided by their
values on the part's starting layout. `strength` and `inter` are logarithms, so their changes
are already fractions (Δ(−log FI) ≈ −ΔFI/FI). `infill` and `cover` are fractions to begin
with. These four keep a scale of 1. A weight therefore prices a 1% change in its term. With
weights `0.5,1,1`, a 1% faster print is worth a 2% lower FI\_agg. The scales come from the
layout the run starts from, so two runs that start from different layouts have slightly
different units.

**Std mode** (`--norm std`). This was the original rule: each term's change is divided by the
standard deviation of its change over random edits. Strength changes by a tiny amount per
edit but by a lot over a run, so in this mode it swamped every other term. The mode is kept
only for comparison.

**Weights.** `--weights` takes up to six entries, in the order strength, time, continuity,
interlayer, infill, cover. Missing entries are filled in as follows (`cli._spec`):

| Entry | Without `--weights` | Filled in automatically |
| --- | --- | --- |
| strength, time, cont | 1.0, 0.3, 0.3 | — |
| inter | 0 | 0.3 on sparse parts unless a 4th entry is given; `--interlayer` also sets it to 0.3 |
| infill | 0 | `infill_weight` when the rule is `penalty` or `both`; forced to 0 for `band` and `free` |
| cover | 0 | `cover_weight` on a fine grid unless a 6th entry is given; forced to 0 on the 0.4 mm grid |

**Worked example.** The table below is the bracket run `runs/repro/phaseA/D_s0`, from its start
to its end. It used weights `0.5,1,1,0.3,1,1`, a scale of 341.67 s for time and 278.36 for
continuity.

| Term | Start | End | Raw change | × weight / scale |
| --- | --- | --- | --- | --- |
| strength | +0.0428 | +0.0333 | −0.0096 | −0.0048 |
| time | −341.67 | −344.68 | −3.01 | −0.0088 |
| cont | −278.36 | −192.39 | +85.97 | +0.3089 |
| inter | −1.627 | −1.420 | +0.207 | +0.0621 |
| infill | −0.0125 | −0.0000 | +0.0124 | +0.0124 |
| cover | −0.0519 | −0.0715 | −0.0196 | −0.0196 |
| **total** | | | | **+0.350** |

In this run the optimizer gave up about 1% of in-plane strength and 1% of print time in
exchange for 31% lower continuity cost and 19% lower Z FI\_agg.

## Strength

Plane-stress finite elements on the layer grid, the Tsai-Wu failure index per element, and a
p-norm over all of them. Code: [rltp/eval/fea2d.py](../rltp/eval/fea2d.py), with stacking and
bookkeeping in [rltp/part.py](../rltp/part.py).

### Mesh

- **Elements.** The elements are the in-part cells: bilinear 4-node quads with nodes at the
  cell corners and 2×2 Gauss integration. `fea_element` (mm) splits each cell into r×r
  elements, with r = cell / `fea_element`, for the analysis only. Every element takes its
  cell's bead directions and fill. A cell's FI is the p-mean of its elements' FI, and its
  stresses are their mean.
- **Supports.** Nodes inside a support region are fixed in x and y.
- **Loads.** A load's force is spread evenly over the unsupported nodes inside its region.
- **Z limits.** The `zmin`/`zmax` limits on support and load regions apply in the split model.
  The laminate model ignores them.
- **Floating islands.** Each free degree of freedom gets 1e-9 × the mean diagonal stiffness,
  divided by r², so that islands with no support can still be solved. The 1/r² keeps the
  total the same however finely the cells are split.

### Material and bead direction

Each element is an orthotropic lamina: E1 along the bead, E2 across it, G12 and ν12. Its
stiffness is a mix of the bead-direction stiffnesses, weighted by each direction's share of
the element:

- **0.4 mm grid** (`cover.bead_weights`). A straight cell runs 100% along x or 100% along y.
  Corner cells and single-cell strokes are 50/50.
- **Fine grid** (`beads.cell_wh` and `beads.cell_shares`). The shares come from the deposited
  bead footprints. Once a layer has diagonal runs, they become four shares: 0, 45, 90 and 135
  degrees. The diagonal stiffnesses use the standard Q-bar rotation (`q_rotate`).

How much stiffness a cell keeps depends on its fill:

| Layer | Filled cell | Empty cell |
| --- | --- | --- |
| dense | 1 | — (every in-part cell is filled) |
| sparse, 0.4 mm grid | 1 | `VOID` = 1e-3 |
| fine grid | min(D, 1), with D the deposited material | at least `VOID` |

Empty cells keep a little stiffness so that the mesh never changes during a run.

### Failure index

FI is the Tsai-Wu inverse strength ratio: the stress state times 1/FI just reaches the
failure surface. FI ≥ 0, it scales linearly with load, and FI = 1 means failure. It uses the
interaction term F12 = −½√(F11·F22).

- **Material frame.** Each element's stress is checked in the frame of each bead direction
  it holds. The element takes the largest of those FIs. A corner cell is therefore checked
  as both a 0° and a 90° bead.
- **Empty cells** have FI 0 but still count in N, the in-part cell count. Removing material
  lowers the sum directly, so the infill target and band are what keep the optimizer from
  emptying the part.

### Aggregate

```math
\mathrm{FI}_{\mathrm{agg}} = \Big(\frac{1}{N}\sum_{e} \mathrm{FI}_e^{\,p}\Big)^{1/p}, \qquad p = 8,
\qquad \mathrm{raw}_{\mathrm{strength}} = -\log \mathrm{FI}_{\mathrm{agg}}
```

The sum runs over every layer. With p = 8 the aggregate sits close to the worst cells but
still responds to edits elsewhere.

### Load cases

`loads.yaml` can hold extra cases (`load_cases`) and random ones (`random_cases: N` or
`--load-cases N`). A random case is a disc load, 4 bead widths in radius, on the outline away
from the supports, pushing in a random direction. It is seeded by `random_seed` or `--load-seed`.

All cases share one stiffness matrix and are solved as separate right-hand sides. When the
part is built, each case's FI\_agg on the starting layout is measured, and its FI is divided
by that value from then on (`case_scale`). A cell's value is then the p-mean over cases, so
FI\_agg starts at 1 and a case's force size drops out. The p-norm acts as a soft worst case:
edits are pushed toward whichever case has fallen furthest behind its start.

The stress outputs (`sx`, `sy`, `txy`), the observations and `--rotate` use case 0, the
`loads` block. `metrics.json` reports each case's raw FI\_agg as `case_fi_agg`.

### Stack models (`--stack`)

- **split** (the original): each layer is its own plate of thickness h. It carries
  h / part height of the load, which assumes a prismatic part.
- **laminate**: the layers are bonded and stretch together, with equal in-plane strain as in
  classical laminate theory.
  - There is one plate over the union of the layers' cells, under the whole load.
  - Its stiffness is the sum over layers of h × fill × direction share.
  - Each layer's stresses then follow from the shared displacement and its own beads and fill
    (`post_layers`). Stiff layers carry more of the load.
  - An edit to one layer still costs one solve, but it changes every layer's FI.
- **auto** (the default): laminate for sparse parts, split otherwise. The laminate model needs
  one shared grid, so it can't be combined with `--rotate`.

### Solver (`--solver`)

- **Full recheck.** The stiffness matrix is factorized directly with SuperLU:
  - `fast`: symmetric mode, minimum-degree ordering of Aᵀ + A, no pivoting. The matrix is
    positive definite, so pivoting isn't needed.
  - `legacy`: the general LU.
- **Proposal.** Conjugate gradient, warm-started from the current displacements and
  preconditioned by the last factorization. It stops at a relative residual of 1e-7
  (`legacy`: 1e-9) or after 40 iterations. If it hasn't converged by then, the matrix is
  factorized afresh.
- **Speed and reproducibility.** On the bracket, `fast` was 1.7–2.7× faster per solve than
  `legacy` and agreed with it to about 2e-10. `legacy` reproduces older runs bit for bit.
- **Cache.** In split mode, up to 6 layers keep their problem and factorization (least
  recently used are dropped).

## Print time

Code: [rltp/eval/timing.py](../rltp/eval/timing.py). The model follows what printer firmware
does: trapezoidal velocity per move, with corner speeds set by junction deviation.

1. **Stroke order.** Greedy nearest neighbour from the layer's grid origin. The order depends
   only on the layer itself, which keeps edits local. The G-code prints in the same order.
2. **Moves.** Straight runs of cells collapse into single moves. On the fine grid,
   `time_polylines` prints each diagonal staircase as one straight move.
3. **Corners.** Each corner's speed is capped by the Marlin/Klipper junction-deviation formula
   (`accel`, `junction_deviation`) and by the print speed. A forward and backward pass then
   sets the speed at each corner. Every stroke starts and ends at rest.
4. **Single-cell strokes** take the time of a move half a cell long.
5. **Travel.** Each stroke is preceded by a trapezoidal travel move from the previous stroke's
   end, or from the origin for the first stroke. A travel longer than 2 bead widths adds a
   retract and an unretract (2 × `retract_length` / `retract_speed`) and any z-hop. The first
   stroke always retracts.
6. **First layer.** It prints at `first_layer_speed`.

A layer's time T is its last stroke's end. `layer_change_time` doesn't count toward T; it
only enters the interlayer time gap. Besides T, timing records when each cell is printed
(`t_cell`) for the interlayer term, plus stroke order, travel distances and retractions.

## Continuity

Code: `continuity` in [rltp/eval/metrics.py](../rltp/eval/metrics.py).

```text
C = c_stroke × strokes + c_travel × travel (in bead widths) + c_boundary_end × boundary ends
  = 1.0 × strokes + 0.05 × travel + 0.5 × boundary ends      (ScoreConfig defaults)
```

- **Strokes**: each one costs a start, a stop and usually a retraction.
- **Travel** leaves out the first move from the origin.
- **Boundary ends** are path ends and loop seams on the boundary. On the 0.4 mm grid the
  boundary is the in-part cells next to the outside. On the fine grid it is the edge of the
  region where a bead's footprint fits.

## Interlayer (Z) strength

Code: [rltp/eval/interlayer.py](../rltp/eval/interlayer.py). Interface j joins layer j (below)
and layer j + 1 (above). This term uses beam statics on the shared surface. The v2 3D FEA
isn't built.

### Bond quality

```text
q_i  = max(φ(Δt_i) × s_i, q_min)      φ(Δt) = 1 / (1 + (Δt / τ)^n)
Δt_i = T_below + layer_change_time + t_above(i) − t_below(i)
s_i  = s_end where both layers have a path end or seam over cell i, else 1
FI_i = (max(σ_i, 0) / (q_i Zt))² + (τ_i / (q_i Sz))²
```

The bond model's defaults are τ = 30 s, n = 2, `s_end` = 0.7 and `q_min` = 0.05. Δt is
measured on the absolute clock, so it depends only on the interface's two layers. On the
fine grid, a cell's print time is the latest time any bead covering it was laid (`t_mat`).

The aggregate is the same p-norm as in-plane strength, taken over every interface cell, and
raw = −log FI\_agg.

### Stress on the interface

Dense and sparse parts use different models.

**Dense parts** (`Interface`). The shared surface is fixed for the whole run, and so are σ and
τ:

| Point loads in `loads.yaml` | σ and τ on an interface |
| --- | --- |
| some above this cut | from the loads above it: σ by Navier's formula and τ by the weld-group method, about the surface's principal axes |
| some, but none above this cut | 0 |
| none at all | σ = ½ Zt × r / r\_max, a fallback that weights cells by distance from the centroid |

During a run only q changes.

**Sparse parts** (`ZInterface`). Only cells with material in both layers bond. The section
that carries the load is that bonded set, and it is recomputed on every edit: less overlap
between adjacent layers means a smaller, more highly stressed section.

- **Generic cases.** Three cases stand in for Z loading in general: a pull, and bending about
  each principal axis of the full shared surface. Each is sized so that a fully bonded
  interface peaks at σ = Zt / 2, which gives FI 0.25 at q = 1. Bending counts both signs.
- **Point loads** above the cut add a fourth case.
- **Per bonded cell**, FI^p is averaged over the cases.
- **Unbonded cells** have FI 0, and N stays the full shared-surface cell count.
- **A nearly detached interface** (fewer than 3 bonded cells) gives every cell FI = 1000.

## Infill target

Sparse parts only. The infill fraction F is filled optional cells divided by optional cells,
over the whole part. Optional cells are in-part cells that aren't walls or solid skin. The
target is `infill`, for example 0.4 to remove 60% of the inside.

| `infill_rule` | Score term | Hard band |
| --- | --- | --- |
| `penalty` | raw = −\|F − target\|, weight `infill_weight` | no |
| `band` | none | rejects edits that leave target ± `infill_band`, or move further from it when outside |
| `both` (default) | as `penalty` | as `band` |
| `free` | none | none; the target only sets the starting layout |

With weight 1, one percentage point off target costs as much as a 1% change in a weight-1
term.

## Coverage (fine grid)

On the 0.1 mm grid, beads are 4 cells wide and can sit anywhere, so an exact cover is usually
impossible. D is the material deposited in a cell: 1 inside one bead, 2 under two overlapping
beads.

```text
gaps = Σ over walls and solid skin of max(1 − D, 0)
over = Σ over every cell of max(D − 1, 0)
raw  = −(gaps + over) / in-part cells
```

`metrics.json` reports the two parts as `gap_pct` and `over_pct`. Elsewhere on the fine grid,
a cell counts as filled when D ≥ 0.5; the support rule, the infill fraction and Z bonding all
use that test.

## Hard rules outside the score

Some layouts are never scored, because `ops._check_sparse` rejects the edit first with
`InvalidEdit`:

- **Walls are locked**: their links never change.
- **Walls and solid skin stay filled** (0.4 mm grid).
- **Support rule**: a filled cell must lie within `max_bridge` / 2 of material in the layer
  below, unless that layer has no part there at all. Removing material may not strand the
  layer above.
- **Infill band**, under the `band` and `both` rules.

## Incremental evaluation

Code: `PartState` in [rltp/part.py](../rltp/part.py).

- **`propose(k, layer)`** scores a candidate for layer k without committing it, and returns
  each term's raw change. It re-times layer k, recounts its continuity, and re-solves the
  FEA, warm-started:
  - split model: layer k's own plate;
  - laminate model: the shared plate, with layer k's stiffness swapped in, which updates
    every layer's FI.

  It also re-scores interfaces (k−1, k) and (k, k+1). Every call counts toward `n_evals`,
  which is the budget `--max-evals` caps.
- **`commit(candidate)`** updates the running per-layer sums: in-plane and interface FI^p
  sums, times, continuity costs, infill counts and miscoverage. It also updates the laminate
  caches. On sparse parts it refreshes the support context of layers k−1 to k+1 and passes
  each layer's FI to it, so that trim, grow and stress-biased edits can see the stress.
- **`full_recheck()`** re-evaluates everything with direct solves. The loop calls it at the
  end of every pass and logs the drift: the largest change in any raw term against the
  incremental sums. It should stay near the PCG tolerance; anything larger means the
  bookkeeping is wrong.

## Annealing temperature unit

`--temp0` is given in units of a typical random edit's change in S. That keeps a temperature
meaningful whatever the weights and the part.

- **Warmup.** Before the loop starts, it proposes up to 128 random edits (`--warmup`) without
  committing them; edits that break a hard rule are skipped. The `Normalizer` records each
  term's per-edit spread, then freezes.
- **Unit.** It is `edit_std`, √Σ(w × σ\_term / scale)². On fine grids it is instead
  1.4826 × median |ΔS| over the warmup edits. Region turns and large refills there change S
  10–100× more than a typical edit; using the standard deviation inflated the unit 18×, so
  annealing accepted most bad edits.
- **Acceptance.** Annealing always accepts an edit that raises S, and accepts one that lowers
  it with probability exp(ΔS / (temp × unit)).
- **Strength-first** (`--strength-first X`) multiplies the strength weight used for
  acceptance by up to X, easing back to 1 over the annealing passes. The best layout is
  always judged on the plain weights.

The RL environment's reward is the accepted gain in the same units.

## Outputs

`metrics.json` holds a breakdown of the starting layout (`baseline`) and of the final one
(`final`), from `PartState.breakdown()`:

| Field | Meaning |
| --- | --- |
| `raw` | the six raw terms |
| `fi_agg`, `inter_fi_agg` | in-plane and Z FI\_agg (lower is stronger) |
| `case_fi_agg` | raw FI\_agg per load case (several cases only) |
| `time_s`, `strokes`, `travel_mm`, `retractions`, `boundary_ends` | time and continuity details |
| `layers`, `cells` | layer count and filled cells |
| `part_cells`, `required_cells`, `infill_pct`, `unsupported` | sparse parts |
| `bonded_pct`, `weakest_bond_pct` | sparse parts: overlap with the layer below, overall and at the worst interface |
| `bead_nodes`, `gap_pct`, `over_pct` | fine grid |

`metrics.json` also records the `weights` and `terms` used. Its `loop` block records the
scales and the score curve.

In `history.jsonl`, each edit line carries the raw change per term (`d`), the weighted change
(`d_total`), the new `total`, and each scaled term (`s`). Each `pass` line carries the recheck
drift. The viewer's FI channel shows each cell's FI.

## Configuration

| Setting | Where | Default | Effect |
| --- | --- | --- | --- |
| weights | `--weights` | see above | trade-offs between terms |
| `norm` | `--norm` | relative | term scales |
| `stack` | `--stack` | auto | split or laminate stack model |
| `solver` | `--solver` | fast | FEA linear solver |
| `fea_element` | printer.yaml, `--fea-element` | 0.1 (printer.yaml) | element size for the strength analysis; 0 = one element per cell |
| `cell` | printer.yaml, `--cell` | 0.1 (printer.yaml) | fine bead grid |
| load cases | loads.yaml `load_cases`, `random_cases`; `--load-cases`, `--load-seed` | none | extra cases for the strength term |
| `material` | printer.yaml, or a loads.yaml override | generic PLA | E1, E2, G12, ν12, Xt, Xc, Yt, Yc, S, Zt, Sz |
| `bond` | printer.yaml, or a loads.yaml override | τ 30, n 2, s\_end 0.7, q\_min 0.05 | bond quality |
| motion | printer.yaml | see file | the time model |
| `layer_change_time` | printer.yaml | 0.5 s | interlayer time gap |
| `infill`, `infill_rule`, `infill_band`, `infill_weight` | printer.yaml, CLI | 0.4 (printer.yaml), both, 0.02, 1.0 | infill target |
| `cover_weight` | printer.yaml, CLI | 1.0 | coverage weight |
| `p_norm`, `c_stroke`, `c_travel`, `c_boundary_end` | `ScoreConfig` (code only) | 8, 1.0, 0.05, 0.5 | aggregate exponent and continuity costs |

## Code map

| File | What it holds |
| --- | --- |
| [rltp/config.py](../rltp/config.py) | `TERMS`, `ScoreConfig`, `Material`, `BondModel`, `LoadSpec` (cases, random cases), `parse_weights` |
| [rltp/eval/metrics.py](../rltp/eval/metrics.py) | `evaluate_layer` (time, continuity and FEA for one layer), `continuity`, `Normalizer`, `full_weights` |
| [rltp/eval/fea2d.py](../rltp/eval/fea2d.py) | `FEAProblem` (mesh, loads, assembly, solve, post-processing), `tsai_wu_fi`, `q_rotate`, `post_layers` for the laminate |
| [rltp/eval/timing.py](../rltp/eval/timing.py) | `time_layer`, `time_polylines`, planner, junction deviation, stroke order |
| [rltp/eval/interlayer.py](../rltp/eval/interlayer.py) | `Interface` (dense), `ZInterface` (sparse), bond quality, interface FI |
| [rltp/eval/stressmap.py](../rltp/eval/stressmap.py) | one isotropic solve for the stress-aligned start; not part of the score |
| [rltp/part.py](../rltp/part.py) | `PartState`: `raw`, `total`, `propose`, `commit`, `full_recheck`, `breakdown`, laminate bookkeeping, infill and coverage terms |
| [rltp/cover/beads.py](../rltp/cover/beads.py) | fine grid: deposited material and direction shares (`density`, `cell_wh`, `cell_shares`, `cell_fields`) |
| [rltp/loop/optimize.py](../rltp/loop/optimize.py) | warmup, temperature unit, acceptance, per-pass recheck |
| [rltp/search/fields.py](../rltp/search/fields.py) | `FieldModel`: a continuous copy of the laminate strength term (stiffness, Tsai-Wu FI, p-norm aggregate) for the gradient start; not part of the score, but it must follow any change to the strength model |

## Tests

| What | Tests |
| --- | --- |
| FEA against analytic solutions | `test_tension_bar_matches_analytic`, `test_cantilever_matches_beam_theory`, `test_refined_mesh_is_closer_to_beam_theory` (tests/test_eval.py) |
| Material model | `test_tsai_wu_uniaxial`, `test_orthotropic_bead_direction_matters`, `test_diagonal_beads_change_the_stress_model` |
| Solver | `test_warm_start_matches_direct`, `test_refined_mesh_keeps_cell_outputs` |
| Time model | `test_straight_stroke_time`, `test_planner_respects_limits`, `test_corners_cost_time` |
| Incremental = full recheck | `test_incremental_matches_full_recheck`, `test_load_cases_incremental_matches_full_recheck`, `test_zbond_is_incremental`, `test_refined_fea_scores_incrementally`, `test_fine_edits_keep_rules_and_score_incrementally` (tests/test_part.py) |
| Load cases | `test_load_cases_start_at_one_and_keep_the_primary_case`, `test_a_scaled_copy_of_a_case_changes_nothing`, `test_random_cases_are_seeded_and_clear_of_supports` |
| Interlayer | `test_interface_statics_resultants`, `test_bond_quality_falls_with_delay`, `test_zbond_rewards_overlap_between_layers` |
| Sparse scoring | `test_empty_cells_carry_no_failure_index`, `test_laminate_infill_matters_less_than_split`, `test_penalty_rule_scores_distance_from_target` |
| Gradient start's copy of the strength term | `test_rotation_and_tsai_wu_match_the_evaluator`, `test_field_model_matches_the_evaluator_on_a_fine_grid` (within 3%), `test_field_gradient_matches_finite_differences` (tests/test_fields.py) |

## Adding a score term

1. Add its name to the end of `TERMS` in `config.py`. Older weight vectors are padded with 0
   by `full_weights`, and `Normalizer.load` fills in missing terms for saved states.
2. Decide its scale. A term that is already a fraction goes in `ABS_TERMS`, a −log term in
   `LOG_TERMS`. Anything else is divided by its baseline value.
3. Compute its raw value in `PartState.raw()` from running per-layer sums. Set those up in
   `_recount()` and update them in `commit()`.
4. Compute its change in `PartState.propose()` from layer k's candidate alone, plus the
   interfaces next to it if the term involves neighbours. That is what keeps evaluation
   incremental.
5. Wire its weight in `cli._spec`, including any default, and add a field to `breakdown()`
   for `metrics.json`.
6. Add a test that applies random edits and checks that the incremental sum equals
   `full_recheck()`.

## Known limitations

- **Relative, not calibrated.** The material constants are generic-PLA placeholders. A change
  in FI\_agg ranks layouts of the same part against each other; it doesn't predict a failure
  load.
- **The two strength terms move at different rates.** Z FI is quadratic in stress, while
  in-plane FI is linear. The same relative stress change moves the `inter` term about twice
  as much as `strength`; keep that in mind when setting their weights.
- **Bonding is modelled two ways.** The laminate model treats layers as perfectly bonded
  in-plane, while the Z term models imperfect bonds. Each is a simplification on its own
  axis.
- **Split model** assumes a prismatic part with an equal share of the load per layer.
- **Interlayer statics** are beam theory on a plane section. They are rough near load points,
  holes and abrupt changes in cross-section.
- **Stroke order** is greedy, not optimized. Time and continuity are scored on that order.
- **Time model** covers motion only: no extrusion-rate limits, pressure advance or cooling
  waits. `layer_change_time` doesn't count toward `time_s`.
