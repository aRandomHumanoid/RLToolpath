# RL Toolpath Optimizer — Design Doc

Nov 11, 2026 · @Albert Lin

## Overview and goals

`rltp` is a Python tool that slices a mesh into layers (walls, solid skins and optimizer-placed infill, or fully dense) and improves each layer's toolpath one edit at a time, searching directly against the evaluator. Simulated annealing is the default proposer. An RL policy that proposes edits was built and is frozen (see Search against the evaluator). It outputs G-code, a self-contained HTML viewer that plays back nozzle motion, and a log that replays how the optimizer changed the part.

**Goals**

- Mesh in (STL, OBJ, 3MF), printable G-code out.
- No fixed perimeter/infill split: each layer is one coverage problem, and structure emerges from the score.
- Score toolpaths on strength (plane-stress FEA), print time, and continuity, then add interlayer adhesion.
- Show two kinds of motion over time: the nozzle printing a layer, and the toolpath changing across optimizer iterations.
- Run the same loop with a trained policy, simulated annealing, or random edits, so they can be compared at equal evaluation budgets.

**Priorities**

- Good continuity and decent strength come first. The strength term is a relative, theoretical score: it ranks toolpaths for the same part and load against each other, and it is not a calibrated prediction of failure load.
- Decent strength means holding up under plausible loads, not only the one in `loads.yaml`. Random load cases (see Strength) keep the optimizer from tuning to a single pull.
- Material constants stay at generic-PLA placeholders. A change in FI\_agg is read as a relative improvement, which holds whatever the exact constants, since bead direction matters through the E1/E2 and strength ratios. Coupon calibration is optional, and the physical milestones (coupons, Z tests, measured strength) are deferred until a part needs absolute numbers.

**Non-goals for v0**

- Support structures, non-planar layers, multiple materials. (Sparse infill is in: see Shell and infill.)
- Overhang handling. v0 assumes layers that support each other, like a 2D profile extruded upward.
- Clean code. Single-process Python first; speed up the evaluator only when profiling says to.

**Interface**

```bash
rltp baseline part.stl --printer printer.yaml --out out/
rltp optimize part.stl --loads loads.yaml --printer printer.yaml \
              --weights 1.0,0.3,0.3 --out out/
rltp train --config train.yaml --out runs/exp01/   # frozen: 0.4 mm grid only
```

Each run writes `part.gcode`, `viewer.html`, `history.jsonl` and `metrics.json` to `out/`. `--weights` sets the strength, time and continuity weights in that order, plus interlayer from v1.5 on.

**Versions**

| Version | Adds |
| --- | --- |
| v0 | single nozzle, square 4-neighbor grid; strength, time and continuity |
| v0.5 | per-layer grid rotation to the dominant principal stress |
| v1 | triangular lattice, if 0° and 90° beads prove limiting |
| v1.5 | interlayer term from the layers above and below, using interface statics |
| v2 | 3D FEA stress maps in place of the v1.5 statics |
| v3 | two nozzles on opposite arms, rotating bed, collision checking |

## Architecture

Training and the per-part loop share one set of edit operators and one evaluator. Offline training produces a frozen policy, and the loop uses it to propose every edit.

This was the original design. Since Nov 12, 2026, the loop proposes edits with simulated annealing by default and the training path is frozen (see Search against the evaluator). The diagram still shows the original design.

&#91;embedded content: system architecture · per-part loop and offline training\]

The trained weights are the only thing that crosses from training to runtime. Every proposal, accepted or rejected, loops back for the next edit, and the scheduler picks a new layer after each burst.

**Stack.** Python 3.11, numpy and scipy, trimesh, shapely 2, PyTorch, Gymnasium, and hypothesis for invariant fuzzing.

**Module layout**

```text
rltp/
  io/mesh.py          load + repair (trimesh)
  slicing/slice.py    section_multiplane -> shapely polygons
  slicing/raster.py   polygon -> cell mask, layer frames
  cover/cover.py      nxt/prv arrays, invariants, stroke walking
  cover/baselines.py  zigzag, concentric, repair pass
  cover/ops.py        edit operators + action masks
  eval/fea2d.py       plane-stress Q4 solver, Tsai-Wu, p-norm
  eval/timing.py      trapezoidal motion + junction deviation
  eval/metrics.py     continuity, seams, score assembly
  eval/interlayer.py  v1.5 statics + bond quality, v2 FEA maps
  rl/env.py           Gymnasium LayerEnv
  rl/net.py           U-Net policy + value heads
  rl/imitate.py       teacher labels -> behavior cloning
  rl/ppo.py           masked PPO
  search/sa.py        simulated-annealing proposer (baseline)
  eval/collision.py   v3: timelines, collision check, repair
  loop/optimize.py    part loop, scheduler, history log
  export/gcode.py     strokes -> G-code
  export/viewer.py    JSON + template -> viewer.html
  cli.py
```

## Data model

Each layer is a grid of cells one bead wide, and its toolpath is a set of paths and loops that visits every filled cell exactly once. A fully dense layer fills every in-part cell; a sparse layer (see Shell and infill) fills its walls, its solid skin and whichever optional cells the optimizer chooses.

- **Grid.** Cell pitch p equals the nominal bead width (0.4 mm for a 0.4 mm nozzle). Each layer has its own origin and rotation θ\_k, so its grid can later align with that layer's dominant principal stress.
- **Cover.** Two int32 arrays per layer, `nxt[i]` and `prv[i]`, with −1 for none. Each cell links to at most one successor and one predecessor, so every connected piece is an open path or a closed loop. Each piece is one stroke, printed without lifting the nozzle.
- **Invariants.** Every in-part cell is in exactly one stroke, links join only 4-adjacent in-part cells, and `nxt[prv[i]] == i`. Edits touch few cells, so checking them is cheap.
- **Why this shape.** Concentric loops, one long zigzag, spirals and any mix are all valid covers, so nothing favors perimeter plus infill. Fewer strokes means fewer travels and retractions.

| Field | Type | Meaning |
| --- | --- | --- |
| `mask` | bool H×W | cell is filled (dense layers: cell lies inside the slice polygon) |
| `part`, `req`, `lock` | bool H×W | sparse layers only: in-part cells, cells that must stay filled (walls and solid skin), wall cells whose links never change |
| `poly` | shapely MultiPolygon | exact slice outline, for display and fidelity checks |
| `frame` | origin, θ\_k, p | grid-to-world transform |
| `nxt`, `prv` | int32, length H·W | the cover |
| `seam` | dict, loop id → cell | where each closed loop starts and ends |
| `theta` (derived) | float H×W | bead direction per cell; a corner cell carries both of its directions |
| `t_print` (derived) | float H×W | when each cell is printed, from the time model |
| `fi` (derived) | float H×W | failure index per element from the last FEA |

`PartState` holds the list of layers, the part spec (material constants, loads, printer profile), the current score breakdown, the per-interface stress maps from v1.5, and in v2 the cached 3D FEA stress maps.

**History log.** `history.jsonl` gets one line per proposed edit: `iter`, `layer`, `op`, `cell`, `params`, `accepted`, the change in each score term, and the new total. Each accepted edit also stores which cells' links changed, with a full snapshot of the layer every 50 accepted edits. The viewer rebuilds any iteration by replaying diffs from the nearest snapshot.

## Slicing and baseline toolpaths

trimesh cuts the mesh into planar sections, each section becomes a cell mask, and a baseline fills every mask with a valid cover.

1. **Load.** `trimesh.load`, then `mesh.process()`. Fail loudly if the mesh is not watertight.
2. **Slice.** Cut through the middle of each layer (z\_k = h/2 + k·h when the first layer is also h thick) with `mesh.section_multiplane`, and convert each section to shapely polygons with holes.
3. **Pick the grid rotation.** v0 uses θ\_k = 0 for every layer. v0.5 sets θ\_k to the dominant principal stress direction from one isotropic FEA of that layer.
4. **Rasterize.** A cell is in the part if its center is inside the polygon (shapely 2 `contains_xy`, vectorized). Drop pieces smaller than a few cells and log them.
5. **Fill with a baseline.**
   - Zigzag: a back-and-forth run per row, broken at gaps in the mask. Trivial and always valid.
   - Concentric: repeatedly peel the mask's outer ring of cells, trace each ring into a loop, and link its cells in order. Cells that don't fit a clean loop become short strokes.
6. **Repair.** Greedily join 4-adjacent stroke ends until no join is possible.

The baselines don't have to be pretty. Leftover short strokes are valid covers, and the optimizer cleans them up. Connected Fermat spirals join the set as a stronger baseline once the loop works.

## Evaluator

The evaluator returns one weighted score built from four normalized terms. v0 implements strength, time and continuity. v1.5 adds an interlayer term built from the two neighboring layers and simple statics, and v2 swaps those statics for a 3D FEA.

The code now scores six terms: these four, plus an infill-target penalty (see Shell and infill) and, on the fine bead grid, a coverage term. `docs/evaluator.md` documents what the code does, term by term. This section keeps the design reasoning.

| Term | Computed from | Version |
| --- | --- | --- |
| Strength | plane-stress FEA on the layer grid, Tsai-Wu index per element, p-norm aggregate | v0 |
| Print time | trapezoidal motion per stroke with junction-deviation corner speeds, plus travel and retraction time | v0 |
| Continuity | stroke count (one retraction each), travel distance, stroke ends on the outer boundary | v0 |
| Interlayer | layers above and below only: bond quality per cell against beam statics on their shared surface | v1.5 |
| Interlayer | the v1.5 bond terms against 3D FEA stress maps | v2 |

```latex
S = w_s \hat{s}_{\mathrm{strength}} + w_t \hat{s}_{\mathrm{time}} + w_c \hat{s}_{\mathrm{cont}} + w_i \hat{s}_{\mathrm{inter}}
```

Each term is signed so higher is better, then divided by its value on the part's baseline cover, so every term is a fraction of where the part started. Strength and interlayer are −log FI\_agg, whose changes are already fractions, so they are not rescaled; print time and continuity are divided by the baseline time and continuity cost. A weight then means the importance of a 1% change in that term. With weights 1.0/0.3/0.3, a 1% drop in FI\_agg is worth a 3.3% rise in print time.

The first version divided each term by the standard deviation of its change over random edits. That made the weights meaningless on real parts. On the bracket, strength's per-edit std was tiny (4.7e-4) but its total gain over a run was about 300 of those units, so strength swamped everything: annealing cut FI\_agg by 13% while print time rose 4%. That mode is kept as `--norm std` for comparison.

Annealing temperatures are given in units of a typical random edit's change in the weighted score, measured during the loop's warmup, so T0 means the same thing under either scaling.

**Strength**

- Elements are the cells: 4-node quads with nodes at cell corners, as in the 88-line topology optimization code.
- Each element gets orthotropic plane-stress stiffness, rotated to its bead angle. A corner cell averages its two directions.
- Material constants: E1, E2, G12, ν12, and strengths Xt, Xc, Yt, Yc, S. Generic-PLA placeholders by default (see Priorities); printed coupons can replace them when absolute numbers matter.
- v0 applies the 2D load spec from `loads.yaml` to every layer, scaled by h over the part height. That assumes a prismatic part and equal load share (see risks).
- Solve with scipy sparse. After a small edit, warm-start conjugate gradient from the previous displacements.
- The strength term is −log of the aggregate below.

```latex
\mathrm{FI}_{\mathrm{agg}} = \Big( \frac{1}{N} \sum_{e=1}^{N} \mathrm{FI}_e^{\,p} \Big)^{1/p}, \qquad p \approx 8
```

**Load cases.** One load case lets the optimizer tune beads to exactly that pull, which can leave the part weak against anything else. `loads.yaml` can list extra cases (`load_cases`), and `random_cases: N` (or `--load-cases N`) adds N random ones. Each random case is a disc load, 4 cells in radius, at a random point on the part's outline away from the supports, pointing in a random in-plane direction. Every case keeps the same supports. The random cases are seeded, so a part always gets the same set.

All cases share one stiffness matrix and are solved as separate right-hand sides. Each case's FI is divided by that case's FI\_agg on the baseline cover, and the aggregate runs over every element of every case:

```latex
\mathrm{FI}_{\mathrm{agg}} = \Big( \frac{1}{NC} \sum_{c=1}^{C} \sum_{e=1}^{N} \big(\mathrm{FI}_{e,c} / \mathrm{FI}^{0}_{\mathrm{agg},c}\big)^{p} \Big)^{1/p}
```

It starts at 1. FI is linear in load, so a case's force size drops out, and only where and which way it pushes matters. The p-norm over cases acts as a soft worst case: edits are pushed toward whichever case is furthest behind its baseline, so one case can't be traded away for another. Observations and `--rotate` use case 0 (the `loads` block) for stress directions; the FI channel shows the combined per-cell value.

**Print time.** Use the motion model printer firmware uses: trapezoidal velocity per segment, with corner speeds capped by acceleration and junction deviation from `printer.yaml`. Cell-scale zigzags spend most of their time in corners, so this term matters.

**Interlayer (v1.5)**

v1.5 scores each interface from only the layers on either side of it, using beam statics on their shared surface instead of a 3D FEA.

- **Shared surface.** Cells present in both layer k and layer k+1. Its centroid, area A, principal axes and second moments I\_x and I\_y (with J = I\_x + I\_y) depend only on geometry, so they're computed once per part.
- **Loads on the cut.** Assume the part is supported at its base. The resultants at interface k sum the loads applied above it, with moments about the surface centroid: axial force N, shears V\_x and V\_y, bending moments M\_x and M\_y, torsion T. This needs point loads with 3D positions in `loads.yaml`.
- **Stress per cell.** Navier's formula for normal stress and the weld-group method for shear, with x̄ and ȳ a cell's position in principal axes (below). Bending stress grows with distance from the neutral axis, so the outer cells carry the most. These maps are also fixed per part.
- **Bond quality per cell.** q\_i = φ(Δt\_i)·s\_i, floored at a small value. Δt\_i is the time between printing cell i on layer k and on layer k+1, taken from absolute clock times so pauses between layers count. s\_i < 1 where both layers have a stroke end at that cell. Start with φ(Δt) = 1/(1 + (Δt/τ)ⁿ) and fit τ and n to Z-direction coupons.
- **Failure index.** Tension and shear against the bond-scaled interlayer strengths Z\_t and S\_z, since compression doesn't open an interface. Aggregate with the same p-norm as in-plane strength.

```latex
\begin{aligned}
\sigma_i &= \frac{N}{A} + \frac{M_x \bar{y}_i}{I_x} - \frac{M_y \bar{x}_i}{I_y} \\
\tau_i &= \sqrt{\Big(\frac{V_x}{A} - \frac{T \bar{y}_i}{J}\Big)^2 + \Big(\frac{V_y}{A} + \frac{T \bar{x}_i}{J}\Big)^2}
\end{aligned}
```

```latex
\mathrm{FI}_i = \Big(\frac{\max(\sigma_i, 0)}{q_i Z_t}\Big)^2 + \Big(\frac{\tau_i}{q_i S_z}\Big)^2
```

**Why it's cheap and exactly local.** The surface, the loads and the stress maps never change during optimization; only q does. An edit to layer k changes only its own print times and stroke ends, so only interfaces (k−1, k) and (k, k+1) change, and Δt on each depends on just its two layers. Each evaluation is one pass over cells, with no solve.

**What the agent should learn.** Put warm, seam-free bonds where the moment arm is largest, usually the outer edges of each interface.

**Limits.** Treating each interface as a plane section is a beam assumption. It breaks down near load points, holes and abrupt changes in cross-section, and when supports aren't all below the cut; v2 covers those cases. Without a load spec, weight each cell by its squared distance from the centroid instead.

**Interlayer (v2)**

- Replace the v1.5 statics with one isotropic 3D FEA of the whole part at load time (Gmsh tet mesh, scikit-fem solve). Sample interlayer normal and shear stress at each interface to get σ\_i and τ\_i per cell directly. Bond quality, the failure index and its aggregate stay as in v1.5.

**Checks before trusting it.** Compare the FEA with analytic tension-bar and cantilever solutions, and the time model with measured print times.

## Edit operators and environment

Every operator turns a valid cover into another valid cover, so the policy never has to learn coverage. There are 10 path operator variants, each applied at one cell, plus 5 fill operators for sparse layers (see Shell and infill).

| Operator | Effect | Valid when |
| --- | --- | --- |
| Reverse | flips the print direction of the stroke through the cell | always |
| Merge | links a stroke end at the cell to an adjacent stroke end, reversing one stroke if their directions clash | the cell is an end and a 4-neighbor is another end; joining a stroke's own two ends closes it into a loop |
| Split | cuts the link leaving the cell | the cell has a successor |
| Flip, right or down (2) | replaces parallel links a→b and c→d with a→c and b→d, re-orienting the segment between | the two links are parallel and adjacent |
| Move seam | makes the cell the start of its closed loop | the cell is on a closed loop |
| Patch refill (2 sizes × 2 angles) | unlinks a 5×5 or 9×9 window, refills it with zigzag at 0° or 90°, then runs the repair pass on the window's edge | the window overlaps the part |

Flip can turn one path into a path plus a loop, which is fine because loops are valid strokes. Patch refill is the large move that lets a layout drift far from its baseline.

## Shell and infill

Real prints aren't solid. The printer profile sets `walls` (rings of cells along every outline, holes included), `solid_layers` (solid skin near top and bottom surfaces), an `infill` target, and `max_bridge`. With walls 2, solid layers 3 and infill 0.4 (the default `configs/printer.yaml`):

- **Walls.** The outermost 2 rings of each layer's cells (8-neighbor erosion), traced into closed loops where the ring allows. They are locked: no edit may change a wall cell's links, so they print as slicer-style perimeters. Only their seams move. Where a section is too thin for two rings on each side (under about 4 beads between a hole and an edge), the inner ring stays an open path.
- **Solid skin, per surface.** A cell is solid if any of the 3 layers above or below lacks that cell, so the first and last 3 layers are solid and so is any ledge or step. Walls and solid cells are required: they stay filled.
- **Optional cells.** Everything else. The starting cover fills rectilinear lines and repairs adjacent ends into longer strokes. `infill_pattern: stacked` (the printer.yaml default) runs every layer's lines the same way, so adjacent layers overlap and bond; `alternate` switches 0°/90° by layer. Skins alternate either way. On the bracket, stacked lines bond 58% of the weakest interface against 41% for alternating, and optimizing from them gave the better result with in-plane FI\_agg unchanged. Lines are picked row by row (column by column at 90°) by error diffusion over each row's optional cells, so the filled share matches the target on any shape, and never more than max\_bridge apart. (The gradient start lays lines from a density field instead; see Search against the evaluator.) From there the optimizer decides where material goes.
- **Infill target.** `infill` is the share of optional cells to keep; the default 0.4 removes 60% of the inside. The `penalty` rule adds a fifth score term, −|infill − target| as a fraction, with weight `infill_weight`. The `band` rule holds the part's infill within ±`infill_band` of the target: fill edits that would leave the band are rejected, as are edits that move further away from it when outside. `both` (the default) is the penalty inside the band, and `free` uses the target only to lay out the start. On the bracket (6 seeds), weight 1 and a ±2-point band tied on the other terms, and the penalty landed closer to the target. Weight 4 was worse on every seed: each fill edit changes the cell count, so a strong penalty stops material moving. A penalty holds the target only while the other terms pull weakly, hence the band as a backstop.
- **Support rule.** A filled cell must lie within max\_bridge/2 of a filled cell in the layer below (Euclidean, in cells), unless the layer below has no part within that distance (a geometric overhang only support structures could fix). That allows bridges up to max\_bridge between infill lines. It cuts both ways: a new cell needs support, and removing cells must not strand a filled cell in the layer above. The starting line pitch is capped at max\_bridge so the baseline is always valid.
- **Fill operators.** Add one optional cell (it joins up to two adjacent stroke ends, so one add can bridge two strokes); remove one optional cell; fill the optional cells of a 5×5 window and re-zigzag it at 0° or 90°; clear the optional cells of a 5×5 window and repair the ends around the hole. Every operator, old and new, is checked on sparse layers: walls unchanged, required cells filled, support rule held. The masks are exact for the single-cell edits.
- **Moves.** Five operators keep the layer's infill count, so material can shift without a budget or penalty fighting it. Add a cell or fill a 5×5 window, then trim as many path-end cells elsewhere in the layer: unbonded ones first (nothing filled directly below or above), then the least stressed by the layer's current FI. Or remove a cell or clear a window, then extend as many path ends: into bonded cells first, then from the most stressed ends. Working at path ends keeps strokes whole, and the trim avoids cells that are the sole support of a cell above. `stack9` copies the layer below's infill into a 9×9 window, zigzags it along the copied lines and trims or grows to keep the count. Single-cell and 5×5 edits can't line infill up across layers, since that takes a coordinated change; on the bracket, without `stack9`, the Z term improved only through bond timing and the weakest interface's bonded share stayed at 41%.
- **Z bonding.** On sparse parts the interlayer term uses the bonded set (cells filled in both layers) as the interface section, recomputed for every edit: area, centroid and second moments about the full surface's principal axes. Alternating 0°/90° infill bonds only where lines cross; stacked lines bond along their length. Besides any point loads above the cut, three generic cases stand for Z strength in general: a pull, and bending of either sign about each principal axis, each sized so that a fully bonded interface peaks at Zt/2. FI^p is averaged over the cases per bonded cell; unbonded cells count 0 and N stays the shared-surface count. Bond quality (time gap, stacked ends) applies as before. It is on by default for sparse parts at weight 0.3.
- **Scoring.** Print time and continuity already count every filled cell, so material has a cost: emptying cells saves time but can split strokes. Empty cells keep 1e-3 of the stiffness, so the FEA mesh never changes, and they count as FI 0; FI\_agg still divides by every in-part cell.
- **Stack model.** With sparse layers, the default in-plane model is a laminate: the layers are bonded and stretch together (equal in-plane strain, as in classical laminate theory). One plate carries the whole load with stiffness summed over every layer's cells (thickness × fill × bead direction), and each layer's stresses follow from the shared strain and its own beads. The v0 split, where each layer carries h/height of the load on its own, charges a sparse middle layer the same load as a solid skin: on the bracket it put FI\_agg at 2.8× the dense part's, and the optimizer would mostly refill the middle. `--stack split|laminate` picks either for any part. The laminate needs one shared grid, so sparse parts can't use `--rotate` yet, and an edit to one layer moves every layer's stresses (still one solve per edit).
- **Stress mesh.** `fea_element` splits each cell into r × r elements for the strength analysis only (elements inherit their cell's bead weights and fill; a cell's FI is the p-mean of its elements'). The solver's floating-island regularization is scaled by 1/r², otherwise it stiffened fine meshes: before that fix the cantilever moved away from beam theory as elements shrank.
- **Not yet.** Per-layer rotation, and a policy trained with the fill operators. The current policy only proposes path operators; annealing, greedy and random use all 19.

**LayerEnv (Gymnasium)**

- `reset` samples a shape, loads, a weight vector w and a baseline cover.
- Observation: float32 channels × H × W, padded to the curriculum stage's grid size (32×32, then 64×64). The network is fully convolutional, so any size works at runtime.
- Action: `Discrete(10·H·W)`, flattened as operator × cell, over the 10 path operators. `action_masks()` returns the matching boolean array.
- Step: score the proposed edit incrementally. As in the part loop at temperature 0, keep it only if it raises the score: reward = the gain if kept, else 0, with the layer unchanged. A rejected action stays masked until the next kept edit.
- Episode: 64 proposals (64 evaluator calls), discount γ = 0.97. The return is the gain the loop would get from this policy in 64 calls, the same quantity the evaluation plot measures. The episode ends early if every valid edit has been tried and rejected.

The first env applied every edit, worsening ones included. PPO on it raised training return from about 1.0 to 2.5 per episode, but made the policy worse inside the loop. It learned sequences that only pay off if worsening steps are kept, and the loop rejects those. `accept: all` keeps that env for comparison.

Observation channels:

- part mask and distance to the boundary
- bead direction as cos 2θ and sin 2θ
- normalized print order, stroke ends, seams
- failure index map, principal stress direction (cos 2θ₁, sin 2θ₁) and magnitude
- support and load maps
- the weight vector w, one constant channel per weight
- v1.5: print times and stroke ends of the layers above and below, plus both interfaces' stress maps (v2 swaps in the FEA stress maps)

**Tests.** Fuzz the operators with hypothesis: random edit sequences must never break the invariants, and reversing a stroke twice must be the identity.

## Fine bead grid

With `cell` below the bead width (0.1 mm cells for 0.4 mm beads, k = 4 cells per bead), placement is no longer tied to one cell per bead.

- **Paths are centerlines.** A layer's `mask` holds bead centerline nodes and its links are 0.1 mm steps, reusing the cover machinery (strokes, ends, seams, timing, G-code). Nodes sit on cell corners (k even), so the path frame is the material grid's shifted by half a cell and the G-code puts beads exactly where the model does.
- **Material is deposited, not assigned.** Each step lays a strip one cell long and k cells wide; path ends add half-bead caps, corners the outer k/2 × k/2 quarter, lone nodes a k × k dab. A straight bead gives D = 1 per cell including its ends; D = 2 where beads overlap; inner corners count twice over a small square. The total matches the G-code's extrusion. Filled means D ≥ 0.5.
- **Evaluation.** The laminate FEA runs on the 0.1 mm material grid with stiffness min(D, 1) per cell and the bead direction of the strips over it. Support, infill share and Z bonding use filled cells; interface timing takes each cell's latest covering node.
- **Coverage.** Exact cover by k-wide beads is usually impossible (a 1.0 mm section takes 2 beads and a gap or 3 overlapping), so it is a cost, not a rule: a sixth score term, minus (shortfall below 1 in required cells + excess above 1 anywhere) as a share of the part's cells.
- **Walls** are centerline rings of the region where a bead's footprint fits, k cells apart, traced into locked loops: the outer wall follows the true outline to 0.1 mm. Skin and infill start as beads k cells apart (infill rows by error diffusion), joined at their ends by L-shaped connectors.
- **Edits.** Shift the straight run through a node one cell sideways (the legs at its ends follow), extend or retract a path end, join an end to another within 2k cells via an L-shaped connector, refill a 1.6 mm window with beads k apart on the edit's row or column phase (0° or 90°), clear a window's beads that cover only optional cells, copy the layer below's beads into a window. Reverse, split and move_seam carry over; the unit-cell edits (merge, flips, patches, the coarse fill edits) are masked off. Every edit is checked for locked walls, the support rule and the infill band.
- **Cost.** About 160 ms per edit on the bracket (12k nodes per layer, a 95k-unknown laminate solve dominating), against 7 ms on 0.4 mm cells with 0.4 mm elements. The trained policy's action space is the 0.4 mm grid's, so fine grids use annealing, greedy or random until a policy is trained for them.

## Bead directions and search guidance

On the bracket, the fine-grid optimizer kept the starting directions almost exactly (each layer's share of x-running steps moved by under a point in 4,000 evals): turning an area's beads takes a refill that splits and rejoins strokes, so continuity outweighs the strength gain, and the edits are local. Four changes address that.

- **Stress map** (`rltp/eval/stressmap.py`). One isotropic plane-stress solve of the union of the layers under the primary load gives each cell's principal direction and size. It describes where and which way the part is loaded, before any toolpath exists.
- **Stress-aligned start** (`start: stress`). Skin and infill in each area run along the nearer of 0 and 90 degrees to the smoothed principal direction (stress-weighted doubled-angle average over ~2 mm), in every layer. Material stays spread as in the slicer start. On the 0.1 mm bracket this alone took FI\_agg from 0.987 to 0.958 and Z FI\_agg from 6.9 to 5.5, with fewer strokes, before any optimization. `stress4` also starts 45/135-degree areas; on the bracket that tripled the stroke count (short diagonal lines in ~4 mm areas whose ends don't line up) for no strength gain, so it is opt-in.
- **Search guidance.** `--stress-bias F`: that share of proposals picks its cell in proportion to (local smoothed FI / layer max)² + 0.02, so edits land where stress is. `--strength-first X`: the strength weight starts X times higher and eases back to 1 over the annealing passes, so directions settle before continuity takes over; the best part is still judged on the real weights.
- **Bigger moves.** Refills at 3.2 and 6.4 mm besides 1.6, all keeping the area's current infill share (required cells get every line, optional cells the share of lines that preserves their fill), so turning an area no longer dumps material into it. `flip_region` turns the connected area of same-direction beads around the edit by 90 degrees; `turn_region` by 45.
- **Diagonal beads (fine grid).** A staircase of at least 6 alternating one-cell steps with constant signs is a diagonal run. It deposits as a rotated rectangle one bead wide (3 × 3 samples per cell; a 45-degree band would otherwise alias to ~88% of its area), prints as one straight G-code move, and is timed as one segment (`time_polylines`: strokes as polylines of axis and diagonal runs). The FEA carries per-cell shares at 0, 45, 90 and 135 degrees: stiffness is the share-weighted sum of four rotated element matrices (classical Q-bar), and a cell's FI is the worst over the directions it holds, each from that direction's stresses rotated into its bead axes. With no diagonals anywhere the original 0/90 path runs unchanged. Edits `refill_45/135` (1.6 and 3.2 mm) lay diagonal lines one bead apart (k√2 cells in c ± r, alternating 6 and 5 for k = 4), and region flips/turns cover all four classes. Hole walls traced as staircases now print as straight diagonals too, which made the bracket's slicer start 2% faster.
- **Results** (0.1 mm bracket, 2 seeds, 4,000 evals; README for the table). From the slicer start the optimizer improved in-plane FI\_agg by 0.9% and never found the stress directions; the stress start with stress-biased proposals and strength-first ended 2.8% better, tying the slicer start on the overall objective with the same print time and continuity (slightly worse Z, a few more strokes). Diagonals were barely used (under 0.5% of interior material). Infill share by stress quarter barely moved, so the optimizer did not add material where stress is high on its own.
- **Annealing temperature on fine grids** uses the robust spread of warmup edits (1.4826 × median |change|) as its unit. Region turns and large refills change the score 10–100× more than a typical edit; with the std as the unit annealing accepted most worsening edits and every run ended back at its start.
- **Deliberately not built: extra walls or infill where stress is high.** The layouts can pack infill lines by stress (`stressmap.weight` with the layouts' `sweight`), and contour-parallel edge beads were sketched, but neither is used: the question is whether the optimizer arrives at something similar on its own (it can add infill through the fill and move edits). Either can be switched on later. The gradient start (Search against the evaluator, step 1) now places infill by optimizing the strength term, and it does concentrate material where stress is high.

## Search against the evaluator

**Decision (Nov 12, 2026).** Each part is optimized by searching directly against the evaluator, with simulated annealing as the default proposer. The RL policy is frozen: it stays in the code and still runs on the 0.4 mm grid, but it is not retrained, extended, or used by default. RL can come back once the evaluator stops changing (see the last part of this section).

**Why**

- **The policy goes stale whenever the evaluator changes.** Since it was trained, the evaluator gained sparse layers, the laminate model, Z bonding, the fine bead grid and diagonal beads. Each change would need a retrain. Search always optimizes the evaluator as it is now.
- **The policy can't reach the current features.** It knows the 0.4 mm grid and the 10 path edits. The fill edits, the moves, the 0.1 mm grid and diagonal beads all run through annealing, and so has every bracket result since the 0.1 mm grid.
- **No gap between training shapes and real parts.** The policy learned on procedural shapes; search works on the part itself.
- **The policy's speedup was shown only on a small task.** On curriculum stage 1 (32×32 grids, strength only) it reached in about 32 evaluator calls what annealing needed about 500 for. By 512 calls every proposer ended at the same place. Stage 2 (time and continuity on larger grids) was never run.
- **The cost.** Every part pays the full search price: about 30 minutes for 4,000 evaluator calls on the 0.1 mm bracket. The next steps below aim at that cost and at strength, the term annealing hasn't improved on its own.

**What stays**

- `rltp train`, `rltp bench` and `--proposer policy|policy-sample` keep working on the 0.4 mm grid, and their tests keep running.
- The bench stays the way to compare proposers at equal evaluator budgets.
- The Policy and training section below is kept as the record of what was built.

**Next steps, in order**

1. **Gradient layout step for strength.** The strength term is a smooth function of each cell's fill and bead direction, so it can be optimized as a continuous field before any toolpath exists, the way topology optimization works. One edit costs one solve and gives one number; one adjoint solve gives the gradient over every cell.
   - **Status (Nov 13, 2026).** Built as `--start gradient` (`rltp/search/fields.py`), since extended with several starts, continuation, ranking by the whole score and a line-end cost. The strength half of the exit is met. The overall half is not yet: the best so far is +42.7 against D's +43.3, with continuity cost 202 against 199 (see Local optima and Where the gap is). A first line-end cost made starts worse. A corrected one, which also counts direction-area boundaries, lifts the start itself to +47.3, above D's final, and is on by default; annealing from it is running.
   - **Variables.** A bead angle θ for every in-part cell outside the walls, one field shared by every layer, skins included (as the stress start does). A fill ρ ∈ [1 / max\_spacing, 1] for every optional cell, one field shared by the sparse layers, which keeps infill stacked for Z bonding. Walls keep the start layout's stiffness, directions and fill.
   - **Model.** A cell's stiffness is Σ over layers of h × fill × Q̄(θ), with Q̄(θ) = Tᵀ Q T, the rotation `q_rotate` does. Stiffness is linear in ρ, not ρ³ as first planned: infill lines share the laminate's strain with the skins, so their stiffness adds with their share of the cell (the evaluator resolves each line and agrees), and a partial ρ is printable as line spacing, so it needs no penalty. A cell's share of Σ FI^p is its filled fraction times FI^p. The objective is in-plane FI\_agg on the laminate model over every load case, with the sparse layers' mean fill held at the infill target. On the stress start's layout the model gives 0.9596 against the evaluator's 0.9581.
   - **Gradient.** K is symmetric, so the adjoint solve reuses the factorization: one extra solve per load case gives ∂FI\_agg/∂ρ and ∂FI\_agg/∂θ for every cell. Checked against finite differences.
   - **Update.** Optimality criteria for ρ (move limit 0.05) and scaled gradient steps for θ (at most 6° per iteration); both steps halve whenever the objective would rise. θ is parametrized by the doubled-angle vector (cos 2θ, sin 2θ), and both fields pass a Gaussian filter (0.75 mm in the end), which keeps them at the scale of printable areas. ρ starts from the target. The objective is ln FI\_agg, or, given the run's weights, w\_strength × ln FI\_agg plus the line-end cost (below). The first version ran one start, the stress map's principal directions, at the final filter: 46 iterations, 63 solves, 46 s on the bracket. It now runs 8 starts with continuation (Local optima, below), about 100 iterations and 110 s per start, in parallel.
   - **From fields to a layout.** θ rounds to 0/90 per area; areas under 2 × 2 mm take the other direction. `gradient4` adds 45/135 areas of at least 3 × 3 mm. ρ lays infill lines that start and stop with the field: each line has a dither threshold and is laid wherever ρ exceeds it. Every max\_spacing-th line is always laid (the support rule), and the thresholds nest, so a line runs on wherever its surroundings are dense enough. A first version gave each whole line its mean ρ (the layouts' `sweight` route) and lost most of the gain: 0.857 in the field model became 0.928. The field is then shifted until the laid-out infill share hits the target. Annealing handles continuity, time, Z bonding and coverage.
   - **Infill where stress is high.** The step places material by optimizing the score, not by a rule, and it does concentrate it: 43% of the optional cells end at the sparsest lines and 12% solid.
   - **Exit.** On the 0.1 mm bracket at 40% infill, the gradient layout has a lower FI\_agg than the stress start. After annealing, its overall objective beats run D (+43.3 against the slicer start) with no worse continuity.
   - **Results** (0.1 mm bracket, printer.yaml defaults, weights 0.5/1/1 + Z 0.3 + coverage 1, 2 seeds, 4,000 evals; the README has the table).
     - The start alone: FI\_agg 0.858 against 0.958 for the stress start (−10.4%), Z FI 4.44 against 5.09, print time +4 s. But it has 38 more strokes (continuity cost 319 against 278), because lines end where the density drops. The field model went from 0.953 to 0.816 unrounded and 0.857 rounded to 0/90; the evaluator gives the laid-out part 0.858.
     - After annealing with D's settings (E): FI\_agg 0.879, 11.0% below the slicer start against D's 2.8%. But the continuity cost is 236 against D's 199, so the objective is +33.6 against D's +43.3. Plain annealing (F) reaches +30.8 at 11.3% below.
     - Annealing recovered as much continuity from the gradient start as from the stress start (about 28 points), but the gradient start began 14 points behind on continuity and only 5.5 ahead on strength. At these weights a part 10% stronger in-plane is worth less than 40 strokes. Rescored with a strength weight above about 1.6 instead of 0.5, E's finals would beat D's (the runs themselves used 0.5).
     - Diagonals (`gradient4`) gain little on top (0.854), double the strokes (380) and raise Z FI to 10.4.
   - **Local optima (Nov 13, 2026).** The step now runs from 8 starting direction fields in parallel (the stress map, turned 90°, uniform 0/90/45/135, two random smooth fields; `--field-starts`), each with continuation: three stages, filter 1.9 → 1.2 → 0.75 mm and the aggregate's p 2 → 4 → 8. Given the run's weights, it lays out every start and ranks them by the whole score, not by FI\_agg alone.
     - The starts end within 2.2% of each other after rounding, 1.3% with continuation, and the stress map is within 0.1% of the best. Local optima in the field step cost little.
     - Starts as strong lay out very differently on continuity (cost 266 against 323), so ranking by the whole score raised the start from +5.9 to +23.8 against the slicer start.
     - After annealing (D's guidance, 4 chains × 4,000 evals, chain i from the i-th best start): +41.9 and +42.7 from the two best starts, in-plane FI\_agg 10–11% below the slicer start, continuity cost 207 and 202. The stress start's best of 4 chains reached +44.0 at 3.7% and 194, and D's two seeds +43.3 on average. So the gradient start now about ties the stress start overall, at three times its strength gain; the exit's "beats D with no worse continuity" is still not met, narrowly.
     - The 3rd and 4th starts' chains ended at +26.7 and +33.9: finals follow the start's rank, so chains are better spent on the top starts.
   - **Where the gap is.** Against the stress start's best chain (+44.0), the gradient start's (+42.7) gains 3.4 points of strength and 1.0 of Z, but loses 2.9 on continuity and 2.4 on print time. Both losses sit in the 4 infill layers: 15 more strokes and 7.5 s more there, with slightly less material, while the skin layers are as fast and have 7 fewer strokes. Lines that start and stop with the density field end at the edges of dense areas; annealing joins about two-thirds of the extra strokes with connectors, but it can't make the lines longer, so the stops and turns keep costing time.
   - **Line-end cost (Nov 13, 2026).** First version: the field step charges the line ends it causes. Where rho changes by d along a line, a share d of the lines there end (the dither thresholds are spread evenly), so the expected ends are the field's change along the line direction, summed over the area, divided by the bead width (`line_ends`: cos²θ weights faces across x, sin²θ faces across y; times the sparse layers). Each predicted end is priced at 0.11 strokes and 0.055 s, what the bracket's ends left after annealing, in the score's own units: the run's continuity and time weights over their term scales. The field step then minimizes w\_strength × ln FI\_agg plus that cost, so it can also turn lines to run along dense bands rather than across them (`--field-end-cost X` scales the price; 0 ignores line ends). Chains now start from the 2 best field starts in turn (`--chain-starts`).
     - **It doesn't work as built.** On the bracket, at 1×, 2× and 4× the price, predicted ends fell from about 120 to 19, 4 and 0.6, but the laid-out starts got worse on the real score: objective +14.2, +4.6 and +8.2 against +23.8 without the cost, continuity cost 288–297 against 266.5, and FI\_agg 0.908–0.964 against 0.857. With density almost uniform (4×), the infill layers still had 76 strokes against the stress start's 64, so something the measure ignores makes the line ends. The likely cause is boundaries between direction areas: every line that crosses from a 0° area into a 90° area ends there, and the measure only counts density changes. The price came from two layouts, too few to show that it tracked the wrong thing. The cost is off by default (`--field-end-cost 0`) until a measure tracks the real strokes.
     - **Diagnosis over 17 laid-out bracket starts** (8 field starts with the cost off and at 1×, plus the stress start): the density-only measure doesn't track strokes, time or continuity cost (r = 0.18, −0.06, 0.14). Skin-layer strokes alone range from 60 to 102 between starts, so fragmented direction areas cost strokes as much as infill line ends do. Start-to-start travel (760–1,190 mm) adds noise to continuity, but annealing mostly removes it.
     - **Corrected measure.** Lines along x are laid at density L cos²θ and lines along y at L sin²θ, where L = solid layers + sparse layers × ρ, the cell's filled layers. A line ends wherever its own density changes along it. That counts infill lines stopping where ρ drops and every line running into an area of the other direction, in skins and infill alike. Over the same 17 starts it tracks strokes with r = 0.57, time with 0.64 and continuity cost with 0.41: better, but still only a moderate predictor. Fitted price per unit: 0.057 strokes and 0.025 s.
     - **It works.** With the corrected measure, the best laid-out start (the stress-map field start in every case) scores +47.3 at both ½× and 1× the fitted price, against +23.8 without the cost. That is above D's and S's finals after 4,000 annealing evals (+43.3, +44.0), before any annealing. At 1×: FI\_agg 0.888 (10% below the slicer start; the stress start is 3% below), Z FI 4.03, print time 328.6 s (the stress start: 341.7 s), continuity cost 221.7, 122 strokes (stress start 142). Priced boundaries make the step lay out a few large, aligned areas, so the part prints faster as well. At 2× the start falls back to +28.2: too high a price leaves strength on the table without finding a cheaper layout. The cost is on by default again (1×). Annealing from these starts is running.
   - **Next.** How much strength is worth against continuity and time is the user's call, through the weights.
   - **Risks.**
     - Rounding the fields to beads may lose much of the gain. Measured: whole-line densities lost most of it, while dithered lines keep it; the cost moved to continuity.
     - Angle optimization has local minima. Measured with 8 starts: they end within 2.2% of each other after rounding (1.3% with continuation), and the stress map is within 0.1% of the best, so they cost little on the bracket.
     - Z bonding depends on how adjacent layers' paths overlap, which the field can't see, so it stays with annealing.
2. **Parallel chains.** Runs are single-threaded, and most of the 16 cores sit idle.
   - First, run independent annealing chains in separate processes (different seeds, possibly different starts) and keep the best. The Phase A comparisons already did this by hand.
   - **Built (Nov 13, 2026):** `--chains N` (`rltp/search/chains.py`). One shared warmup sets the term scales and temperature unit, so finals compare on one scale; chain i takes seed + i and, with a gradient start, the best field starts in turn (the 2 best by default, `--chain-starts`; the first experiment gave each chain its own start, best first); the best final is kept, and every chain's history and metrics are saved. Workers are forked processes (`rltp/search/parallel.py`); without fork (native Windows) chains run one after another. From the stress start, 4 seeds ended 5.4 points apart, and best of 4 was 2.5 points above the mean chain.
   - **Reheating (`--reheat F`)**: once two greedy passes in a row gain under 0.005, anneal again from F × the starting temperature. It fired once in 16,000 chain evals at 4,000 evals per chain, for no gain: greedy passes are still improving when the budget runs out. It matters for longer runs. Replica exchange stays unbuilt until chains from one start disagree by more than a few points.
   - Then add batched candidate scoring within one chain (item 1 under Performance).
   - **Exit.** Compare 8 chains × 4,000 evals with one chain × 32,000 at equal total compute, to learn whether width or depth pays.
3. **Cheap screening.** Score each proposal on a coarse stress mesh, and re-score on the full mesh only the proposals the coarse score would accept.
   - On the 0.4 mm grid, coarse 0.4 mm elements cost about 1/19 as much per edit as 0.1 mm ones.
   - The fine grid has no coarse mesh yet: one would need elements of 4×4 cells, and the FEA can only split cells, not merge them.
   - First, measure how often coarse and full scores rank edits the same way, especially near holes.
4. **Learning within the run.** Track each edit type's recent gain per evaluator call and propose types in proportion to it, with a floor so that no type starves (a multi-armed bandit). This recovers part of what the policy offered, with no offline training.
   - **Exit.** An equal or better final objective at equal evals on the bracket.

**When to bring RL back.** Once the evaluator and the edit set stop changing, a policy could spread search costs across parts. It would need fine-grid observations and the full edit set, with 16× the cells per layer, so training would need the GPU. A lighter option is a learned prior that only biases which cells and edit types annealing proposes. The gradient step may cover much of what a policy would learn about bead directions.

## Policy and training

**Status: frozen since Nov 12, 2026** (see Search against the evaluator). This section records the design as built; it runs on the 0.4 mm grid with the 10 path edits.

A U-Net maps the observation to one logit per operator and cell. It is trained first by imitating a best-of-N search teacher, then fine-tuned with PPO.

**Network**

- U-Net with 4 levels at 64×64, channels 32 → 64 → 128 → 256, two 3×3 convolutions per level, GroupNorm, SiLU. About 2–3M weights. A global-pooling branch at the bottleneck is broadcast back to every cell, so context stays part-wide on grids larger than any seen in training.
- Policy head: a 1×1 convolution to 10 channels, one per operator variant.
- Value head: global average pool of the bottleneck, then a small MLP to one number.
- Masking: set invalid logits to −1e9, then take one softmax over all operator × cell entries.

**Imitation warm-start**

- Teacher: at states sampled from annealing runs, score N random valid edits and label the one with the largest gain. Annealing alone is a weak teacher, since it accepts any improving edit rather than the best one.
- Train with masked cross-entropy.
- Use held-out imitation accuracy to compare architectures before any RL run.

**PPO**

- Write a single-file masked PPO in the CleanRL style. Stable-Baselines3 expects a vector action head, so per-pixel logits would mean subclassing its policy internals.
- Run 32 environments in subprocesses, start from standard PPO settings, and normalize rewards.

**Training data**

- Procedural shapes from shapely: random polygons, unions and differences of circles and rectangles, holes, thin ribs.
- Random supports and loads: a fixed region, a load region and a force vector.
- Layer slices from Thingi10K with synthetic loads.
- Hold out whole shape families for testing.

**Curriculum**

1. 32×32 grids, strength only, Flip and Patch refill: the operators that change bead direction.
2. Add time and continuity, 64×64 grids, every operator.
3. Add above- and below-layer channels and the v1.5 interlayer term.
4. Run the frozen policy inside the part loop.

**Evaluation**

- Main plot: score against evaluator calls on held-out shapes, for policy, simulated annealing, greedy search and random edits.
- Report each term's physical change, such as % change in FI\_agg, print time and continuity cost, alongside the weighted score. Use enough shapes and seeds to show per-shape statistics: a median over shapes, paired wins and losses, and bootstrap intervals. With 12 shapes and one seed, one shape with 4× the typical gain made the policy appear to reach annealing's final score in 33 calls. On 48 shapes × 3 seeds it took 89.
- Sanity test: under pure tension, a bar's beads should end up along the load.
- Save toolpath images during training. A reward jump with odd-looking paths usually means the evaluator has a hole.

## Part-level optimization loop

The loop picks a layer, runs the proposer (simulated annealing by default) for a short burst of edits, keeps accepted ones, and stops once acceptance has turned greedy and a full pass gains less than ε. It returns the best part seen.

```python
part = baseline(mesh)                      # every layer starts as a valid cover
temp = T0
banned = {k: set() for k in layers}        # rejected actions, per layer
while True:
    pass_gain = 0.0
    for k in schedule(part):               # highest priority first; every layer each pass
        for _ in range(T_BURST):           # e.g. 32 edits
            a = proposer.act(obs(part, k), mask(part, k) - banned[k], w)
            cand = apply(part.layer(k), a)
            d = delta_score(part, k, cand) # layer k, plus interfaces k-1/k and k/k+1 from v1.5
            if accept(d, temp):            # always if d > 0; sometimes if d < 0 while temp > 0
                part.set_layer(k, cand); pass_gain += d
                banned[k].clear()          # v1.5: also banned[k-1], banned[k+1]
            else:
                banned[k].add(a)           # without this, a deterministic policy repeats it
            log(k, a, d)
    part.keep_best()                       # remember the best part seen so far
    if temp == 0 and pass_gain < EPS or no proposals this pass:
        break
    temp = cool(temp)                      # reaches 0 after a few passes
return part.best()
```

- **Rejected edits stay masked until their layer changes.** A frozen policy picking its most likely action proposes the same rejected edit forever unless it is banned. The first loop cleared the ban list at every burst, so an argmax policy at a local optimum replayed the same 32 rejected proposals every pass. It reached annealing's final score on only 17 of 48 held-out shapes. Keeping bans until the layer (or, with the interlayer term, a neighbor) changes makes it work down its ranking instead. When every layer has run out of untried edits, the part is at a local optimum for single edits and the loop stops.
- **Scheduling.** A layer's priority is its share of the part's failure index plus its recent gain. Every layer still gets visited once per pass.
- **Neighbors.** Once the interlayer term is on, an edit's score change includes both interfaces it touches.
- **Full re-check.** Each pass ends with a fresh evaluation of every layer, so small incremental-solver errors cannot pile up.
- **Proposers.** `--proposer policy|sa|greedy|random` swaps only the proposer, so comparisons use the same loop and evaluator.

## G-code export

Each stroke becomes one extrusion run. Straight runs of cells collapse into single moves, and extrusion is sized so every cell gets the same volume.

- **Coordinates.** Map cell centers to world XY through the layer frame. z\_k = first-layer height + k·h.
- **Collapse straight runs.** Emit one `G1` per run of collinear links, never one per cell. A move per 0.4 mm cell stutters on many printers.
- **Extrusion.** Use relative extrusion (`M83`). Each segment extrudes the formula below, with bead width w = p and filament diameter d\_f. Add half a cell's worth at each stroke end, since cell-center paths stop half a cell short.
- **Travel.** Between strokes: retract (length from `printer.yaml`) if the travel is longer than 2p, `G0` at travel speed, optional z-hop, then unretract.
- **Speeds.** One print speed with no perimeter/infill distinction, plus a slower first layer. Corners are left to the firmware planner.
- **Templates.** Start and end G-code (heating, homing, purge line) come from `printer.yaml`.

```latex
E_{\mathrm{seg}} = \frac{L \, w \, h}{\pi \, (d_f / 2)^2}
```

**Validation.** Open the output in PrusaSlicer's G-code viewer, compare its time estimate with our time model, and print a 20 mm square before anything else.

## Visualization and playback

One self-contained `viewer.html` shows a layer's toolpath, animates the nozzle printing it, and scrubs through the optimizer's history.

**Data.** The exporter embeds JSON in the page: per layer, each stroke as a flat coordinate array with a start time per segment from the time model, plus the history diffs and snapshots.

**Rendering.** Canvas 2D with inline JavaScript and no build step. Canvas stays fast with 100k+ segments, where SVG would bog down.

**Views**

- **Layer view.** A layer slider, with color by bead direction (hue = angle mod 180°), print order, failure index or stroke. Travels are thin dashed lines, seams are dots.
- **Nozzle playback.** Play, pause, a speed multiplier and a time slider. Draws every segment up to time t, with a marker at the nozzle's position and travels visible as they happen.
- **Optimization history.** An iteration slider replays the log, highlighting cells that changed at each step. A line chart beside it plots the total score and each term against iteration.
- **3D stack (later).** three.js lines extruded by layer, for checking how layers stack and where seams line up.

`rltp render` writes the same views as PNG frames or MP4 through matplotlib, for papers and slides.

## v3: two nozzles, one rotating bed

v3 drives two nozzles at once on cantilevered arms mounted on opposite sides, over a bed that rotates. The nozzles split each layer's cells between them.

&#91;embedded content: v3 machine concept · top view and clearance rules\]

At the same y the arms are in line, so A's nozzle must stay at least D\_x left of B's. Once the arms are at least D\_y apart in y, their x ranges may overlap.

**Machine model**

- Arm A mounts on the left rail and arm B on the right. Each arm moves in y along its rail, and its nozzle moves in x along the arm.
- Each arm slides through its mount like a ram, so inside the build area it occupies only the span from its rail to its nozzle carriage.
- Both arms move in the same plane, so neither can pass over the other.
- Shared Z: both nozzles print the same layer, and the build steps up only after both finish.
- The bed angle ψ\_k is set once per layer, before printing. Rotating during a layer couples both nozzles' paths, so that's left for later.

**What changes in the design**

- **Data model.** Each cell gets an owner, A or B, and each nozzle keeps its own `nxt`/`prv` cover over its own cells. ψ\_k joins the layer frame.
- **Baseline.** Pick ψ\_k so the part's area splits evenly about the bed's center line, give each nozzle the half on its side, and fill each half with the v0 baselines.
- **Operators.** A new Transfer moves a stroke or a patch to the other nozzle, cutting any links that would cross the new ownership boundary. Patch refill also picks an owner.
- **Observation.** An owner channel, a print-time channel per nozzle, and a near-miss channel from the checker.
- **Time term.** Layer makespan, a smoothed max(T\_A, T\_B) including dwells (see v3 evaluator changes), so splitting work evenly matters.
- **G-code.** One stream per nozzle in machine coordinates, transformed by ψ\_k. Each layer starts with a bed-rotation move and a sync point, and every dwell is a sync point too.
- **Viewer.** Plays both nozzles together, colored by owner, with the arms' footprints drawn and near misses marked.

## v3 legality checks

A two-nozzle layer is legal only if it passes all six checks below, run cheapest first. A failed edit leaves the state unchanged, is banned for the rest of the burst as in the part loop, and costs a small penalty during training.

1. **Joint cover.** Every in-part cell has exactly one owner and lies on exactly one of that owner's strokes. Links never join cells with different owners. These are the v0 invariants per nozzle, plus the ownership check.
2. **Reach.** Map every printed and travel point from the part frame to the machine frame by rotating ψ\_k about the bed center. Each point must fall inside its own arm's reach box.
3. **Timelines.** Build p\_A(t) and p\_B(t) with the same motion model as the time term, including travels, retractions and dwells. Both start at the layer's start time.
4. **Collision.** The arms' footprints must never intersect. Arm A spans from its rail to x\_A and arm B from x\_B to its rail, so a collision is the condition below. Sample t at a fixed step dt and add 2·v\_max·dt to D\_x and D\_y, so arms closing on each other at full speed can't slip through between samples. Skip any stretch where the nozzles' y-ranges are farther apart than D\_y plus that margin.
5. **Schedule repair.** If check 4 fails, add a dwell before the stroke that leads one nozzle into the conflict, then repeat check 4. Dwells go only at stroke starts, since pausing mid-bead leaves a blob. Try delaying each nozzle, keep the one with the shorter makespan, and stop after a fixed number of rounds; if conflicts remain, the edit is illegal.
6. **Layer barrier.** The next layer starts only after both timelines end. Shared Z enforces this anyway, but checking it explicitly lets a later version with independent Z swap in a per-cell rule: a cell on layer k+1 waits until the cells under it on layer k are printed.

```latex
\text{collision}(t) \iff \lvert y_A(t) - y_B(t) \rvert < D_y \;\wedge\; x_B(t) - x_A(t) < D_x
```

D\_x is the x clearance the two carriages need: their half-widths plus a safety margin. D\_y is the same for the arms in y.

**What the checker returns.** A legal flag, the dwell schedule, and the time and cell of every near miss. Near misses also become an observation channel, so the policy learns where conflicts cluster instead of only being told no.

## v3 evaluator changes

Two nozzles open six gaps in the evaluator. The first one affects predicted strength; the other five affect how well the score guides the optimizer.

| Gap | Why it matters with two nozzles | Change |
| --- | --- | --- |
| Cold weld line where A's cells meet B's | neighbors across the ownership boundary can be laid long apart and bond weakly, but the FEA uses one cross-bead strength everywhere; boundaries stacked across layers form a weak plane | scale each cell's E2 and Yt by an in-layer bond curve of the time gap to its cross-bead neighbors; penalize ownership boundaries that line up across layers; use ψ\_k to rotate them apart |
| Makespan is flat for the faster nozzle | max(T\_A, T\_B) ignores edits to the nozzle that finishes first, so about half of all edits get no time signal | a smooth maximum (below), plus a small idle-time term |
| Time depends on the repair heuristic | greedy repair can score equally good toolpath pairs differently, and a small edit can add or remove a dwell | deterministic repair; dwell time reported as its own term; periodically compare greedy repair with a search over which nozzle waits |
| Thermal proxy rewards too-fast layers | φ(Δt) only decreases, and two nozzles roughly halve layer time, so small layers can overheat and sag | a minimum layer time, or a φ that falls off again below a cooling threshold |
| Edits aren't local within a layer | changing A's path shifts when B must wait, so B's print times and bond terms move too | re-run the timelines and collision check from the first affected time onward; earlier times don't change |
| Dwells only cost time | a hot nozzle waiting in place oozes and can mark the print | retract and move clear before each dwell, plus a small penalty per dwell |

Fit the in-layer bond curve with coupons that deliberately delay neighboring beads, since in-layer bonding may not follow the interlayer curve φ. The same factor also helps single-nozzle layers wherever neighboring strokes are printed far apart in time.

```latex
T_{\mathrm{layer}} \approx \frac{1}{\beta} \log\big( e^{\beta T_A} + e^{\beta T_B} \big)
```

Larger β tracks the true maximum more closely. Pick it so that a gap of a few seconds between the nozzles still shows in the score.

## Milestones

Build in this order: each milestone has something printable or measurable before the next one starts.

1. **M0 — Pipeline, no optimization.** Slicing, rasterizing, zigzag and concentric baselines, G-code, viewer with playback.
   - Exit: a 20 mm square and one real part print cleanly from our G-code.
2. **M1 — Evaluator.** Plane-stress FEA, time model, continuity metrics, placeholder material constants (coupon-measured constants are optional; see Priorities).
   - Exit: FEA matches analytic tension-bar and cantilever results, and the time model tracks measured print times on the M0 parts.
3. **M2 — Operators and search.** All operators, invariant fuzzing, simulated annealing and random proposers, history log and scrubber.
   - Exit: annealing reliably improves the score on test shapes, and a long fuzz run finds no invariant failures.
4. **M3 — RL on single layers.** LayerEnv, U-Net, imitation of the best-of-N teacher, then PPO through the curriculum.
   - Exit: on held-out shapes, the policy reaches annealing's final score in fewer evaluator calls.
   - Status: met for curriculum stage 1 on the 0.4 mm grid. Frozen on Nov 12, 2026 in favor of M5b.
5. **M4 — Full-part loop.** Layer scheduling, neighbor-aware score changes, per-layer grid rotation (v0.5).
   - Exit: optimized G-code for multi-layer test parts that prints.
6. **M5 — Interlayer terms.** v1.5 first: adjacent-layer bond quality against interface statics. Then v2: 3D FEA stress maps in place of the statics.
   - Exit: on test parts, the v1.5 term moves seams and stroke ends off the most loaded interface regions without giving up the in-plane terms. Z-direction coupons that would confirm a real strength change are deferred (see Priorities). v2 waits on them: build it only if coupons fail where v1.5 predicted they'd hold.
7. **M5b — Search against the evaluator.** The gradient layout step for strength, parallel chains, cheap screening, and learning within the run, in that order (see Search against the evaluator).
   - Exit: on the 0.1 mm bracket, the gradient layout plus annealing beats run D's overall objective with no worse continuity, and parallel chains improve the best result reached per hour of wall-clock time.
   - Status (Nov 13, 2026): the gradient start (several starts, continuation, ranked by the whole score) plus annealing reaches +42.7 against D's +43.3, ending 10% stronger in-plane than the slicer start against D's 2.8%, with slightly worse continuity (202 against 199) and 9 s more print time, both from infill line ends. Independent parallel chains are built. A first line-end cost in the field step made starts worse (it ignored direction-area boundaries); the corrected one lifts the gradient start to +47.3 before annealing, above D's final. Annealing from it is running.
8. **M6 — Physical validation (deferred).** In-plane tensile and bending coupons, Z tensile and short-beam shear, against slicer defaults and the M2 annealing result. This isn't needed for the current goals (see Priorities); run it only once a part needs absolute strength numbers.
9. **M7 — v3 dual nozzles.** Owner map, Transfer operator, legality checker with schedule repair, two-stream G-code, dual playback.
   - Exit: the checker flags every collision in a hand-built set of crossing cases, and a dry run with extrusion off and the bed lowered finishes without contact.

## Performance

Profiled on the 0.1 mm bracket, about 80% of each evaluator call is the laminate FEA: one ~92k-unknown solve per edit, warm-started conjugate gradient preconditioned by the last factorisation (about 4 iterations, each a triangular solve). Masks, the support rule, deposition and timing are a few ms each.

**Done: a faster factorisation.** K is symmetric positive definite, so the default solver now factorises it as such (minimum-degree ordering of A'+A, no pivoting) and stops CG at a relative residual of 1e-7. Per solve: 162 → 61 ms; per edit: 156 → 59 ms on 0.4 mm cells with 0.1 mm elements, 190 → 112 ms on the 0.1 mm bead grid; strength changes agree with the old solver to 4e-10. `--solver legacy` keeps the original (general LU, 1e-9), which reproduces earlier runs bit for bit. Algebraic multigrid (pyamg) was tried and dropped: the 1000:1 stiffness contrast of empty cells needed ~150 iterations, 20× slower. CHOLMOD would likely be faster still but needs SuiteSparse installed system-wide.

**Later, once the fine-grid model and its edits settle** (each would lock in structure that is still changing). Item 1 is now step 2 of Search against the evaluator, starting with independent chains; items 2 and 3 stay later.

1. **Parallel candidate evaluation.** Persistent worker processes (Python threads won't help) each hold a copy of the part, score candidate edits concurrently, and apply every accepted edit to all copies. Greedy's best-of-N maps on directly; annealing becomes batched (score a batch from one state, accept from it). Expect 8–12× more evaluations per second on 16 cores. Mind: commits must reach every worker in order, and history logging must stay deterministic for a given seed.
2. **Matrix-free multigrid on the structured grid.** Every element is the same square, so K·u is a fixed stencil over the grid with per-cell coefficients (fill × bead fraction), and geometric multigrid on the regular grid preconditions it, as large-scale topology optimisation does. Cost grows linearly with cells, which finer grids and bigger parts need; it also removes the sparse assembly (~15 ms per edit now). The empty-cell contrast needs care in the smoother and coarse operators.
3. **GPU.** Only after (2). The dev machine has an RTX 5070 (12 GB), but the venv's PyTorch is the CPU build and the FEA is SciPy (CPU only), so nothing uses it yet: a stencil-plus-multigrid solve ports directly (PyTorch or CuPy), plain GPU CG with a simple preconditioner does not (same contrast problem). The other GPU use is batched candidate scoring and policy training, which a fine-grid policy (16× the cells per layer) will need.

## Risks and open questions

The biggest risk is the lattice: a 4-neighbor grid can only lay beads at 0° and 90° to its own axes.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Lattice limits bead angles | beads can't follow stress that runs at other angles | rotate each layer's grid to its dominant stress (v0.5); move to a triangular lattice with 6 equal-length neighbors (v1) |
| Diagonal links change bead width | equal volume per cell makes diagonal beads about 0.71p wide | prefer the triangular lattice over adding diagonals |
| Equal load share per layer | stiffer layers draw more load, which per-layer 2D can't see | periodic laminate-level check that re-splits load by layer stiffness |
| Evaluator exploitation | the optimizer finds and exploits FEA and proxy errors | periodic finer FEA on the best part, viewer inspection, coupons |
| Per-part search cost | without a policy, every part pays the full search price (about 30 minutes for 4,000 evaluator calls on the 0.1 mm bracket) | the gradient layout start, parallel chains, cheap screening; RL can come back once the evaluator settles |
| Fields don't survive layout | the gradient step's gain may be lost when its fields are rounded to beads | measured: dithered lines keep it (0.857 in the field, 0.858 laid out), but their line ends cost continuity and print time; the field step now prices line ends |
| FEA cost | per-edit solves dominate runtime | warm-started conjugate gradient, small training grids, a learned surrogate only if profiling demands it |
| Cell-scale motion | 0.4 mm zigzags mean many short moves and corners | straight-run collapsing, the time term's corner cost, testing on Klipper |
| Material constants | E2 and the strengths depend on the printer and settings | placeholders are enough to rank toolpaths against each other (see Priorities); a coupon set per material and profile only when absolute numbers are needed |
| Loads on 3D parts | a 2D load spec only fits prismatic parts | v2 derives per-layer loads from the 3D FEA |
| v3: a missed collision | the checker samples time and trusts the motion model to match the firmware; a miss can damage the machine | the 2·v\_max·dt sampling margin, a dry run with extrusion off and the bed lowered for every new part |

**Open questions**

- Which ring-peeling method gives a concentric baseline that handles thin features?
- What form of φ(Δt) fits our interlayer coupons?
- Gradient step: is one fill and direction field shared by every sparse layer enough, or does per-layer freedom pay for the Z bonding it costs?
- Parallel chains: at equal compute, do many short chains beat one long one?
- Gradient step: the line-end price (0.057 strokes and 0.025 s per unit of the measure) was fitted on 17 bracket starts, and ½× to 1× worked equally well there. Does it carry to other parts and printers, or should each part fit its own (lay out a few starts, regress)?
- How much is in-plane strength worth against continuity and print time? At strength weight 0.5, a part 10% stronger in-plane gains about 5 points, about what 15 strokes cost; the weights decide whether the gradient start's extra strength is worth its line ends.
- On hold while the policy is frozen: train on a fixed grid size, or on mixed sizes per batch? Should the policy also choose which layer to edit, replacing the hand-written scheduler?
- v3: how will the two G-code streams run in sync? That depends on the controller.
