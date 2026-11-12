# RL Toolpath Optimizer — Design Doc

Nov 11, 2026 · @Albert Lin

## Overview and goals

`rltp` is a Python tool that slices a mesh into fully dense layers and improves each layer's toolpath with an RL policy that proposes one edit at a time. It outputs G-code, a self-contained HTML viewer that plays back nozzle motion, and a log that replays how the optimizer changed the part.

**Goals**

- Mesh in (STL, OBJ, 3MF), printable G-code out.
- No fixed perimeter/infill split: each layer is one coverage problem, and structure emerges from the score.
- Score toolpaths on strength (plane-stress FEA), print time, and continuity, then add interlayer adhesion.
- Show two kinds of motion over time: the nozzle printing a layer, and the toolpath changing across optimizer iterations.
- Run the same loop with a trained policy, simulated annealing, or random edits, so they can be compared at equal evaluation budgets.

**Non-goals for v0**

- Sparse infill, supports, non-planar layers, multiple materials.
- Overhang handling. v0 assumes layers that support each other, like a 2D profile extruded upward.
- Clean code. Single-process Python first; speed up the evaluator only when profiling says to.

**Interface**

```bash
rltp baseline part.stl --printer printer.yaml --out out/
rltp optimize part.stl --loads loads.yaml --printer printer.yaml \
              --policy policy.pt --weights 1.0,0.3,0.3 --out out/
rltp train --config train.yaml --out runs/exp01/
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

Each layer is a grid of cells one bead wide, and its toolpath is a set of paths and loops that visits every cell exactly once.

- **Grid.** Cell pitch p equals the nominal bead width (0.4 mm for a 0.4 mm nozzle). Each layer has its own origin and rotation θ\_k, so its grid can later align with that layer's dominant principal stress.
- **Cover.** Two int32 arrays per layer, `nxt[i]` and `prv[i]`, with −1 for none. Each cell links to at most one successor and one predecessor, so every connected piece is an open path or a closed loop. Each piece is one stroke, printed without lifting the nozzle.
- **Invariants.** Every in-part cell is in exactly one stroke, links join only 4-adjacent in-part cells, and `nxt[prv[i]] == i`. Edits touch few cells, so checking them is cheap.
- **Why this shape.** Concentric loops, one long zigzag, spirals and any mix are all valid covers, so nothing favors perimeter plus infill. Fewer strokes means fewer travels and retractions.

| Field | Type | Meaning |
| --- | --- | --- |
| `mask` | bool H×W | cell lies inside the slice polygon |
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

Each term is signed so higher is better, then divided by the running standard deviation of its per-edit change, since rewards are changes. Without that, the term with the biggest units swamps the weights.

**Strength**

- Elements are the cells: 4-node quads with nodes at cell corners, as in the 88-line topology optimization code.
- Each element gets orthotropic plane-stress stiffness, rotated to its bead angle. A corner cell averages its two directions.
- Material constants come from printed coupons: E1, E2, G12, ν12, and strengths Xt, Xc, Yt, Yc, S.
- v0 applies the 2D load spec from `loads.yaml` to every layer, scaled by h over the part height. That assumes a prismatic part and equal load share (see risks).
- Solve with scipy sparse. After a small edit, warm-start conjugate gradient from the previous displacements.
- The strength term is −log of the aggregate below.

```latex
\mathrm{FI}_{\mathrm{agg}} = \Big( \frac{1}{N} \sum_{e=1}^{N} \mathrm{FI}_e^{\,p} \Big)^{1/p}, \qquad p \approx 8
```

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

Every operator turns a valid cover into another valid cover, so the policy never has to learn coverage. There are 10 operator variants, each applied at one cell.

| Operator | Effect | Valid when |
| --- | --- | --- |
| Reverse | flips the print direction of the stroke through the cell | always |
| Merge | links a stroke end at the cell to an adjacent stroke end, reversing one stroke if their directions clash | the cell is an end and a 4-neighbor is another end; joining a stroke's own two ends closes it into a loop |
| Split | cuts the link leaving the cell | the cell has a successor |
| Flip, right or down (2) | replaces parallel links a→b and c→d with a→c and b→d, re-orienting the segment between | the two links are parallel and adjacent |
| Move seam | makes the cell the start of its closed loop | the cell is on a closed loop |
| Patch refill (2 sizes × 2 angles) | unlinks a 5×5 or 9×9 window, refills it with zigzag at 0° or 90°, then runs the repair pass on the window's edge | the window overlaps the part |

Flip can turn one path into a path plus a loop, which is fine because loops are valid strokes. Patch refill is the large move that lets a layout drift far from its baseline.

**LayerEnv (Gymnasium)**

- `reset` samples a shape, loads, a weight vector w and a baseline cover.
- Observation: float32 channels × H × W, padded to the curriculum stage's grid size (32×32, then 64×64). The network is fully convolutional, so any size works at runtime.
- Action: `Discrete(10·H·W)`, flattened as operator × cell. `action_masks()` returns the matching boolean array.
- Step: apply the edit, re-evaluate incrementally, reward = score after − score before.
- Episode: 64 edits, discount γ = 0.97.

Observation channels:

- part mask and distance to the boundary
- bead direction as cos 2θ and sin 2θ
- normalized print order, stroke ends, seams
- failure index map, principal stress direction (cos 2θ₁, sin 2θ₁) and magnitude
- support and load maps
- the weight vector w, one constant channel per weight
- v1.5: print times and stroke ends of the layers above and below, plus both interfaces' stress maps (v2 swaps in the FEA stress maps)

**Tests.** Fuzz the operators with hypothesis: random edit sequences must never break the invariants, and reversing a stroke twice must be the identity.

## Policy and training

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
- Sanity test: under pure tension, a bar's beads should end up along the load.
- Save toolpath images during training. A reward jump with odd-looking paths usually means the evaluator has a hole.

## Part-level optimization loop

The loop picks a layer, runs the frozen policy for a short burst of edits, keeps accepted ones, and stops once acceptance has turned greedy and a full pass gains less than ε. It returns the best part seen.

```python
part = baseline(mesh)                      # every layer starts as a valid cover
temp = T0
while True:
    pass_gain = 0.0
    for k in schedule(part):               # highest priority first; every layer each pass
        banned = set()
        for _ in range(T_BURST):           # e.g. 32 edits
            a = proposer.act(obs(part, k), mask(part, k) - banned, w)
            cand = apply(part.layer(k), a)
            d = delta_score(part, k, cand) # layer k, plus interfaces k-1/k and k/k+1 from v1.5
            if accept(d, temp):            # always if d > 0; sometimes if d < 0 while temp > 0
                part.set_layer(k, cand); pass_gain += d; banned.clear()
            else:
                banned.add(a)              # without this, a deterministic policy repeats it
            log(k, a, d)
    part.keep_best()                       # remember the best part seen so far
    if temp == 0 and pass_gain < EPS:
        break
    temp = cool(temp)                      # reaches 0 after a few passes
return part.best()
```

- **Rejected edits are masked.** A frozen policy picking its most likely action proposes the same rejected edit forever unless it is banned for the burst.
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
2. **M1 — Evaluator.** Plane-stress FEA, time model, continuity metrics, coupon-measured material constants.
   - Exit: FEA matches analytic tension-bar and cantilever results, and the time model tracks measured print times on the M0 parts.
3. **M2 — Operators and search.** All operators, invariant fuzzing, simulated annealing and random proposers, history log and scrubber.
   - Exit: annealing reliably improves the score on test shapes, and a long fuzz run finds no invariant failures.
4. **M3 — RL on single layers.** LayerEnv, U-Net, imitation of the best-of-N teacher, then PPO through the curriculum.
   - Exit: on held-out shapes, the policy reaches annealing's final score in fewer evaluator calls.
5. **M4 — Full-part loop.** Layer scheduling, neighbor-aware score changes, per-layer grid rotation (v0.5).
   - Exit: optimized G-code for multi-layer test parts that prints.
6. **M5 — Interlayer terms.** v1.5 first: adjacent-layer bond quality against interface statics. Then v2: 3D FEA stress maps in place of the statics.
   - Exit: Z-direction coupons show the v1.5 term changes interlayer strength. Build v2 only if coupons fail where v1.5 predicted they'd hold.
7. **M6 — Physical validation.** In-plane tensile and bending coupons, Z tensile and short-beam shear, against slicer defaults and the M2 annealing result.
8. **M7 — v3 dual nozzles.** Owner map, Transfer operator, legality checker with schedule repair, two-stream G-code, dual playback.
   - Exit: the checker flags every collision in a hand-built set of crossing cases, and a dry run with extrusion off and the bed lowered finishes without contact.

## Risks and open questions

The biggest risk is the lattice: a 4-neighbor grid can only lay beads at 0° and 90° to its own axes.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Lattice limits bead angles | beads can't follow stress that runs at other angles | rotate each layer's grid to its dominant stress (v0.5); move to a triangular lattice with 6 equal-length neighbors (v1) |
| Diagonal links change bead width | equal volume per cell makes diagonal beads about 0.71p wide | prefer the triangular lattice over adding diagonals |
| Equal load share per layer | stiffer layers draw more load, which per-layer 2D can't see | periodic laminate-level check that re-splits load by layer stiffness |
| Evaluator exploitation | the optimizer finds and exploits FEA and proxy errors | periodic finer FEA on the best part, viewer inspection, coupons |
| FEA cost | per-edit solves dominate runtime | warm-started conjugate gradient, small training grids, a learned surrogate only if profiling demands it |
| Cell-scale motion | 0.4 mm zigzags mean many short moves and corners | straight-run collapsing, the time term's corner cost, testing on Klipper |
| Material constants | E2 and the strengths depend on the printer and settings | a coupon set per material and profile before M1 exits |
| Loads on 3D parts | a 2D load spec only fits prismatic parts | v2 derives per-layer loads from the 3D FEA |
| v3: a missed collision | the checker samples time and trusts the motion model to match the firmware; a miss can damage the machine | the 2·v\_max·dt sampling margin, a dry run with extrusion off and the bed lowered for every new part |

**Open questions**

- Which ring-peeling method gives a concentric baseline that handles thin features?
- What form of φ(Δt) fits our interlayer coupons?
- Train on a fixed grid size, or on mixed sizes per batch?
- Should the policy also choose which layer to edit, replacing the hand-written scheduler?
- v3: how will the two G-code streams run in sync? That depends on the controller.
