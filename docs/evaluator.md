# The evaluator

The evaluator scores a whole part's toolpath. It computes eight raw terms, divides each term's
change by a scale, and adds them up with the user's weights. Every proposer (annealing,
greedy, random and the trained policy), the RL environment and the bench call the same two
methods, `PartState.propose` and `PartState.total`. A score from one of them therefore means
the same as a score from any other.

This page describes what the code does now. The design doc's Evaluator section explains why
it is built this way and what was planned, including the v2 3D FEA, which isn't built.

- [Terms at a glance](#terms-at-a-glance)
- [The weighted score](#the-weighted-score)
- [Strength](#strength)
- [Stiffness](#stiffness-stiff-7th-weight)
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
| `cover` | −(gaps + over-extrusion) / in-part cells | 1 | fine bead grid (`cover`'s weight > 0) |
| `stiff` | −log of the mean over load cases of compliance / the slicer layout's | 1 | its (7th) weight is > 0 |
| `bulk` | −log of the p = 2 aggregate of FI, each case scaled to its start | 1 | its (8th) weight is > 0 |

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
- **Coarser elements.** A `fea_element` that is a multiple c of the cell merges c×c cells
  into one element (`coarsen`). The element's stiffness is the mean of its cells' (cells
  outside the part count 0). Each cell's stress and FI come from its element's centre strain
  through the cell's own bead directions, so a cell's beads still set its FI. On the 0.1 mm
  bracket with 21 load cases, 0.4 mm elements cut a proposal from 2.1 s to 0.22 s. Stress
  peaks come out blunter, so FI\_agg values aren't comparable across element sizes.
- **Supports.** Nodes inside a support region are fixed in x and y.
- **Loads.** A load's force is spread evenly over the unsupported nodes inside its region.
  A load's `moment` (N·mm about +z) is applied as a couple on the same nodes: forces normal
  to each node's arm from the nodes' centroid, sized by the arm, with no net force.
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
- **Fine grid, true directions** (`--directions path`, `ScoreConfig.directions`; since Nov 16,
  2026; laminate only). The class shares come from the grid steps, so a bead following a
  curve, a staircase of x-steps with a y-jog every few cells, holds a little 90-degree
  material near every jog. A cell then scores at the worse of 0 and 90 degrees, as if the
  bead were loaded across it, though the strand runs along the curve. Here
  `cover.paths.node_angles` gives each node its path's direction instead:
  - **Tangent:** a least-squares line through the nodes within 0.8 mm of path either side.
  - **Corners:** the fit stops at corners: a turn of more than 30° between least-squares
    directions over 0.8 mm either side, concentrated at the node, and at least half a bead
    (0.4 mm of path) from either end: a last grid jog near an end only looks like a turn.
  - **Accuracy** on rasterised paths: straight lines 0.8° mean error, circles of 1.6–10 mm
    radius 0.6–1.3° mean, zigzag turns exact; 1 mm radii are the weak spot (~9°).

  A cell's stiffness is exact for any mix of directions. Q-bar is
  M0 + cos 2a M1 + sin 2a M2 + cos 4a M3 + sin 4a M4 (`fea2d.q_basis`), so each layer gives
  five per-cell coefficients: thickness × material × the mean of those terms over its
  covering nodes. Its FI is the worst over the directions its beads hold, quantised to
  1.4° (`paths.cell_bins`, directions within 2.8° merged). The laminate's layers share
  their strain, so every (cell, direction) pair any layer holds is scored once per solve
  (`fea2d.post_layers_path`, a reference-counted table). Cost: about +7% per proposal.
  The 0.4 mm grid, the dense split model, the field model's walls and the direction
  guide still use the classes.

  The same mode also prints and deposits along fitted straight segments (`beads.fit_mode`,
  set on each fine layer by its `PartState`; class mode is unchanged):
  - **Printed path** (`beads._fit_runs`, `_fit_piece`): each stroke is cut into straight
    segments.
    - Each segment is the least-squares line through its nodes, grown greedily while every
      node stays within 0.1 mm (one cell, a quarter bead) of it. Only the nodes where the
      path turns are checked, since they are the farthest from any line.
    - The corners `node_angles` finds always end a segment.
    - Vertices sit where consecutive lines cross. Near-parallel lines, or lines that would
      cross far from their shared node, instead get the average of that node's projections
      onto both. A path end is its node projected onto its line.
    - So a staircase at any angle, or a 0.1 mm jog, is one move along its true direction:
      lines laid at the 16 bin angles come out within 0.05°. On circles of 1–10 mm radius,
      every node stays within 0.1 mm of the printed polyline.
    - G-code (`export/gcode.py`) and the time model (`timing.time_polylines`) read these
      polylines from `runs(L)`. A polyline entry carries a third element, its vertices as
      (column, row) floats.
  - **Deposit** (`beads._deposit_fit`): each segment lays a band one bead wide, sampled at 16
    R2 low-discrepancy points per cell. A grid of samples has rows that a band edge at the
    wrong angle lines up with: ±6% for 3 × 3 at 45°, and 4% for a 13-point lattice at 33.75°.
    - A path end gets a half-bead cap.
    - A joint of 90° or less, between segments long enough for it, is mitred at the
      bisector: covered once, nothing missing. A sharper joint lets both bands stop at the
      vertex.
    - A single straight bead is at most 1 per cell. A stroke deposits its length × width
      plus caps, as its G-code extrudes: lines at any angle within ±0.6%; circles +0.3% at
      10 mm radius to +3.9% at 1 mm, since the polygon is a little larger.
    - Shares: each segment's material splits between the two nearest of 0/45/90/135 by
      angle, so `cell_shares` and the tools built on them keep working.
  - **Effect** on frame (slicer start / an optimised layout):
    - moves 11.9k → 4.3k / 17.6k → 8.7k;
    - print time −9% / −10%: curved walls print smooth, not as staircases of 90° corners;
    - material −6% / −8%: staircases deposited their Manhattan length plus corner squares;
    - over-extrusion 3.0% → 0.3% / 5.7% → 1.9% of part cells;
    - FI_agg on the same optimised layout +0.7% (less material at the walls).
  - **Wall gaps:** gaps rise 0.7% → 1.6% of part cells, all in the wall bands: a smooth wall
    half-covers the rasterised outline's staircase corner cells. Skin gaps are unchanged.
    The walls are locked, so this is a constant in the cover term.
  - **Cost:** about +15% per proposal against class mode (the true-direction FI table
    included). Per layer: deposit 20 ms, segment fit 10 ms, corners and tangents 6 ms.
    Proposals score incrementally (drift 4e-11).

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
still responds to edits elsewhere. In practice it sits very close: on the 0.1 mm bracket
half of Σ FI^p is in 2 cells at the inside corner (in the walls), and on the link
(`examples/link.stl`) in 5 cells where the clamped washer region ends. Corners like the
bracket's are real stress concentrations whose size depends on the mesh. A clamp edge is
an artifact of modeling a support as a rigid region, and it carries about 30% of the link's
FI\_agg (see Known limitations).

**Clamp band (`--clamp-band MM`, off by default).** For each load case, the cells within MM
of that case's own support and load regions don't count toward its FI (they still count in
N). Those are the singular ends of a rigid support or a concentrated load: no toolpath can
change them. With 21 random edge cases (bracket, start), 71% of Σ FI⁸ sat in walls within
1 mm of a case's own regions, 20% in other wall cells, and 9% in the bead cells edits can
turn (hook: 81%, 17%, 1%), so without the band the strength term barely saw bead directions.
The band is computed on the cell grid (a distance transform of each case's regions).

**Bulk strength (`bulk`, 8th weight).** The same per-cell FI, aggregated at p = 2 instead
of 8, with each case scaled to its own p = 2 aggregate on the slicer layout:

```math
\mathrm{raw}_{\mathrm{bulk}} = -\log \Big(\frac{1}{N}\sum_{e} \overline{(\mathrm{FI}_{e,c}\, s^{(2)}_c)^2}\Big)^{1/2}
```

With 2 walls and the band, the walls held 24–52% of Σ FI⁸ at the slicer start and 41–71%
after `--adapt` (5 test parts): they already run along the edges where stress peaks, and as
the interior improves the peak sits more and more in them. So under p = 8 the bead directions
inside matter less and less. With 1 wall (the printer.yaml default since then) the shares
are 17–41% and 26–63%. p = 2 sees how well every bead suits its loads.

## Stiffness (`stiff`, 7th weight)

Each load case's compliance (the work its loads do, f · u) relative to the slicer layout's,
averaged over the cases: raw = −log mean\_c(C\_c / C\_c,0). On the laminate it is the plate's;
in split mode it's summed over the layers. It costs nothing extra (f and u are at hand) and is
scored incrementally like strength. With the placeholder PLA constants (E1/E2 = 1.27) bead
direction changes stiffness much less than strength (Xt/Yt = 1.8): the self-improvement
cycles moved compliance by 0.3–1.9% where FI\_agg moved 2–6%.

### Load cases

`loads.yaml` can hold extra cases (`load_cases`) and random ones: `random_cases: N` force
cases and `random_moments: N` moment cases (`--load-cases N`, `--load-moments N`), seeded by
`random_seed` or `--load-seed`. `random_mode` (`--load-mode`) picks how they're made:

- **edge** (the default). The footprint's outline is split into edges
  ([rltp/slicing/edges.py](../rltp/slicing/edges.py)): each contour is cut at corners (a
  turn over 40°) and where a straight run meets an arc, so the edges are straight sides,
  arcs and whole round holes. Edges shorter than 2 bead widths are dropped. Each random case
  fixes one random edge and loads a different one, both as the points within one bead
  width of their edge. A force case pushes in a uniformly random direction with the primary
  case's total force (1 N without one); a moment case twists either way by that force × 10 mm.
  - **Which edges.** Parts are mostly held and loaded at bolt holes and flat faces, so an
    edge is drawn in two steps: its kind, with odds `random_weights` (`--load-weights`,
    default 3 : 2 : 1 for round hole : straight side : arc) over the kinds the part has;
    then an edge of that kind in proportion to its length, so short fillets are rarely
    picked. The load's edge is drawn the same way from the other edges. `uniform` draws
    every edge alike with the same random numbers as runs before Nov 15, 2026. On the 5 mm
    hook, 10 of its 13 edges are arcs, several under 2 mm; drawn alike, most cases held or
    loaded a fillet.
- **disc** (the cases of runs before edge mode). A disc load, 4 bead widths in radius, on the
  outline away from the loads.yaml supports, pushing in a random direction. Force cases only.

Cases that share supports (the primary, `load_cases`, disc cases, and edge cases that fix the
same edge) share one stiffness matrix and are solved as separate right-hand sides. Each
other set of supports gets a linear system of its own, with its own factorization, so edge
cases cost more: up to one system per edge. When the
part is built, each case's FI\_agg on the starting layout is measured, and its FI is divided
by that value from then on (`case_scale`). A cell's value is then the p-mean over cases, so
FI\_agg starts at 1 and a case's force size drops out. The p-norm acts as a soft worst case:
edits are pushed toward whichever case has fallen furthest behind its start.

The stress outputs (`sx`, `sy`, `txy`), the observations and `--rotate` use case 0, the
`loads` block. `metrics.json` reports each case's raw FI\_agg as `case_fi_agg`.

**Without a primary case (load-agnostic).** With no `loads` (no `--loads` file, or one with
only random cases), every case comes from the random draw and none is special. Running
without a loads file now does this by default, with 10 force and 10 moment cases
(`--load-cases`/`--load-moments` change it), and `scripts/suite.py` runs this way unless
given `--yaml-loads`. Dividing each case by its own start is then the right measure: if no
load is known, neither is its size, so a case that barely stresses the part now could be ten
times larger in use. The p-norm over cases reads as "no weak direction, and at least as good
as the slicer under any plausible load". Case 0 is then the first random case. The stress
start's map ([rltp/eval/stressmap.py](../rltp/eval/stressmap.py)) becomes the cases'
consensus: per cell, the doubled-angle sum of every case's principal direction, each case
scaled by its own p = 8 mean. Along a free edge only the edge-tangent stress survives,
whatever the load, so the map follows the outline there and is weak where the cases
disagree.

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
- **Cache.** In split mode, up to 6 layers keep their problem and factorizations (least
  recently used are dropped).
- **Several load cases.** All the cases in one system run their conjugate gradients on the
  same preconditioner. With several cases, the laminate's per-layer post-processing only
  touches each layer's filled cells and the diagonal directions where some layer lays them.
  That gives the same values up to rounding, and single-case runs are unchanged bit for bit.

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
    + c_free_end × free ends
  = 1.0 × strokes + 0.05 × travel + 0.5 × boundary ends + 1.0 × free ends   (ScoreConfig defaults)
```

- **Strokes**: each one costs a start, a stop and usually a retraction.
- **Travel** leaves out the first move from the origin.
- **Boundary ends** are path ends and loop seams on the boundary. On the 0.4 mm grid the
  boundary is the in-part cells next to the outside. On the fine grid it is the edge of the
  region where a bead's footprint fits.
- **Free ends** are path ends (walls aside) that no other bead touches: no other bead's
  centreline node within one bead width (Chebyshev, `kb` cells) of the end, the end's own
  last 2·kb + 1 nodes aside (`free_ends` in [rltp/cover/beads.py](../rltp/cover/beads.py)).
  Such a bead can't pass load in-plane to the rest of its layer, which the strength model
  can't see: the laminate bonds every layer perfectly, so a line that touches nothing in its
  own layer carries load as well as one tied to the wall. `--free-end-cost` sets the cost
  (0 turns it off). In a full layer every end touches a neighbour, so only sparse layers pay.

The layouts keep their ends anchored (`anchor_end`). A free end is extended the shortest
way that makes it touch a bead or wall, up to 4 mm:
- straight on along its line;
- or, for a straight end, a 90° turn to the nearest bead beside it (a T-joint to the
  neighbouring infill line).

The starting layouts anchor every free end. In the gradient start an infill line still starts
and stops with the density field, and its ends get T-joints. `--infill-lines whole` lays each
run of a line whole instead, by the field's mean along it. On one bracket and one link field
that cost 5–12% of the start's strength, against 0–1% for T-joints. Window re-lays and stack
copies anchor the ends they make, and the `anchor` edit anchors one free end. Under `--adapt`
on the five suite parts (README, cycle 17), free ends per part fell from 38 to 0.2 after
annealing, with FI_agg unchanged. Without the cost, annealing leaves about 4.

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

The bond model's defaults are τ = 30 s, n = 2, `s_end` = 0.7 and `q_min` = 0.05. `fi:
ratio` (`--z-fi ratio`) scores a cell by the square root of FI_i instead: the inverse strength
ratio, linear in load like the in-plane FI. The quadratic default acts like p = 16 under the
p = 8 aggregate: a cell where both layers have a path end weighs 0.7⁻¹⁶ ≈ 300 times as much.
The ratio form cut the Z changes of the same 60 skin re-lays to about a third, with almost
the same ranking (rank correlation 0.93). Δt is
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
| `time_s`, `strokes`, `travel_mm`, `retractions`, `boundary_ends`, `free_ends` | time and continuity details |
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
| `fea_element` | printer.yaml, `--fea-element` | 0.1 (printer.yaml) | element size for the strength analysis: a fraction of the cell splits it, a multiple merges cells; 0 = one element per cell |
| `cell` | printer.yaml, `--cell` | 0.1 (printer.yaml) | fine bead grid |
| load cases | loads.yaml `load_cases`, `random_cases`, `random_moments`, `random_mode`; `--load-cases`, `--load-moments`, `--load-mode`, `--load-seed` | none; edge mode | extra cases for the strength term |
| `material` | printer.yaml, or a loads.yaml override | generic PLA | E1, E2, G12, ν12, Xt, Xc, Yt, Yc, S, Zt, Sz |
| `bond` | printer.yaml, or a loads.yaml override | τ 30, n 2, s\_end 0.7, q\_min 0.05 | bond quality |
| motion | printer.yaml | see file | the time model |
| `layer_change_time` | printer.yaml | 0.5 s | interlayer time gap |
| `infill`, `infill_rule`, `infill_band`, `infill_weight` | printer.yaml, CLI | 0.4 (printer.yaml), both, 0.02, 1.0 | infill target |
| `cover_weight` | printer.yaml, CLI | 1.0 | coverage weight |
| `p_norm`, `c_stroke`, `c_travel`, `c_boundary_end` | `ScoreConfig` (code only) | 8, 1.0, 0.05, 0.5 | aggregate exponent and continuity costs |
| `c_free_end` | `ScoreConfig`, `--free-end-cost` | 1.0 | cost of a path end no other bead touches |

## Code map

| File | What it holds |
| --- | --- |
| [rltp/config.py](../rltp/config.py) | `TERMS`, `ScoreConfig`, `Material`, `BondModel`, `LoadSpec` (cases, random edge and disc cases, per-case supports), `parse_weights` |
| [rltp/slicing/edges.py](../rltp/slicing/edges.py) | `outline_edges`: an outline split into straight sides, arcs and round holes |
| [rltp/eval/metrics.py](../rltp/eval/metrics.py) | `evaluate_layer` (time, continuity and FEA for one layer), `continuity`, `Normalizer`, `full_weights` |
| [rltp/eval/fea2d.py](../rltp/eval/fea2d.py) | `FEAProblem` (mesh, loads and moments, one linear system per set of supports, coarsened or refined elements, assembly, solve, post-processing), `tsai_wu_fi`, `q_rotate`, `post_layers` for the laminate |
| [rltp/search/guide.py](../rltp/search/guide.py) | `DirectionMap`: per cell and bead direction, the weighted score's cost under the current strains; `guided_action` (`--guide`) |
| [rltp/eval/adapt.py](../rltp/eval/adapt.py) | diagnostics: how far bead directions changed, and how close each cell is to its best direction under the current strains |
| [rltp/eval/timing.py](../rltp/eval/timing.py) | `time_layer`, `time_polylines`, planner, junction deviation, stroke order |
| [rltp/eval/interlayer.py](../rltp/eval/interlayer.py) | `Interface` (dense), `ZInterface` (sparse), bond quality, interface FI |
| [rltp/eval/stressmap.py](../rltp/eval/stressmap.py) | one isotropic solve for the stress-aligned start; not part of the score |
| [rltp/part.py](../rltp/part.py) | `PartState`: `raw`, `total`, `propose`, `commit`, `full_recheck`, `breakdown`, laminate bookkeeping, infill and coverage terms |
| [rltp/cover/beads.py](../rltp/cover/beads.py) | fine grid: deposited material and direction shares (`density`, `cell_wh`, `cell_shares`, `cell_fields`) |
| [rltp/cover/paths.py](../rltp/cover/paths.py) | fine grid, `--directions path`: each node's path direction (`node_angles`), per-cell stiffness moments and direction pairs |
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
| Load cases | `test_load_cases_start_at_one_and_keep_the_primary_case`, `test_a_scaled_copy_of_a_case_changes_nothing`, `test_random_cases_are_seeded_and_clear_of_supports`, `test_outline_edges_split_at_corners_and_tangent_arcs`, `test_random_edge_cases_fix_one_edge_and_load_another`, `test_random_moment_cases_are_pure_couples`, `test_edge_cases_solve_with_their_own_supports`, `test_edge_cases_incremental_matches_full_recheck` (tests/test_part.py) |
| Coarse elements | `test_coarse_mesh_keeps_cell_outputs_and_each_cells_beads` (tests/test_eval.py), `test_coarse_fea_scores_incrementally_on_a_fine_grid` (tests/test_part.py), `test_field_gradient_matches_finite_differences` (tests/test_fields.py, with coarsening) |
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
- **Rigid support edges.** A support region fixes every node in it, so the stress is
  singular where the region ends. On the link about 30% of FI\_agg comes from the cells
  within 1 mm of the support and load regions; the optimizer can spend effort there that
  buys no real strength. Compare runs with those cells left out when a support sits in
  stressed material.
- **Interlayer statics** are beam theory on a plane section. They are rough near load points,
  holes and abrupt changes in cross-section.
- **Stroke order** is greedy, not optimized. Time and continuity are scored on that order.
- **Time model** covers motion only: no extrusion-rate limits, pressure advance or cooling
  waits. `layer_change_time` doesn't count toward `time_s`.
