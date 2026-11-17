# RL Toolpath Optimizer — Design Doc

Nov 11, 2026; rewritten Nov 16, 2026 for the surface-infill fork · @Albert Lin

## Overview and goals

`rltp` slices a mesh into layers with fixed walls, solid top and bottom skins and infill, scores
toolpaths with one evaluator (in-plane strength over many load cases, print time, continuity,
interlayer bond, coverage), and writes G-code and a self-contained HTML viewer.

**This fork builds infill from surfaces.** A part is hollowed to its shell (one wall, two solid
layers top and bottom), and the inside is filled with a set of surfaces ("ribbons"). Each
surface cuts every infill layer in one continuous bead path, ends against the walls, and may
cross or bond with its neighbours. The search adjusts the surfaces; every candidate is laid out
as bead paths and scored by the unchanged evaluator.

The fork starts from the edit-by-edit optimizer (cycles 0–20, Nov 12–16, 2026): annealing over
single-window edits, a gradient start, and an RL policy that was frozen on Nov 12. That line, its
runs and its working notes are on branch `selfimprove-cycles` (run data in commit `8efb311`).
Its lessons are below under "Why surfaces"; its code still runs (`rltp optimize`).

**Goals**

- Mesh in, printable G-code out; slicing a real part needs no loads (load-agnostic strength).
- Infill as a few continuous surfaces: no loose ends, layers that stack and bond, and structure
  (edge-following bands, material around holes) that the optimizer forms itself rather than
  by adding walls.
- Continuity and print quality first, decent strength second.

**Priorities**

- The strength term is relative: it ranks layouts of the same part against each other, with
  generic-PLA constants (E1/E2 = 3300/2600 MPa, Xt/Yt = 50/28 MPa). Coupons only when a part
  needs absolute numbers.
- The walls stay in the score (see Evaluator): they can't be edited, but how hard they work
  depends on everything around them.

**Not in this fork:** RL policy training (the frozen policy's code stays in `rltp/rl`), the v3
two-nozzle rotating-bed plan, support structures, non-planar layers.

## Why surfaces: what the edit-by-edit optimizer showed

Measured on the 5 mm frame, plate and wrench under load-agnostic cases (cycles 18–20 and the
analyses after them; full notes on `selfimprove-cycles`, `runs/selfimprove/CYCLES.md`).

1. **Strength came from coordinated changes, slowly.** On the 2 mm parts (cycles 0–17) the
   gradient start, a field step that re-plans every direction and density at once, did most
   of the adapting; plain annealing moved FI_agg by −0.5%. On the 5 mm parts annealing from
   the gradient start cut FI_agg a further 9–37% over 8,000 edits, but most of the walls'
   relief came from the gradient start, and most strength-improving edits were rejected (2).
2. **Edits toward stronger layouts were rejected for the loose ends they leave.** Of 8,000
   edits per run, 48–56% would have lowered FI_agg; 20–25% of those were accepted. The main
   blocker of the rest: continuity (37–61%: a re-laid window's lines don't link to the lines
   around it) and interlayer bond (23–42%: the new lines no longer sit on the layers above and
   below). Print time blocked only 12–20%. Penalties were a median 7–39× the strength gain.
3. **The score sits in a few hot spots, largely in the walls.** FI_agg = (mean FI⁸)^(1/8), so
   half of Σ FI⁸ sat in 0.08–1.5% of the printed cells, at inside corners and hole edges, the
   same places in every layer. At the end the walls held 25–70% of it. They are not a fixed
   cost: their FI fell 18–28% from the slicer start without a wall bead moving, because the
   layers stretch together and stiffer surroundings strain them less.
4. **Exact bead angles are worth little with these constants.** A bead 22.5° off its load keeps
   93% of its tensile strength (5.6°: 99.6%); lines every 11.25° tied or lost against the four
   classes on every part (c20). Where angles don't matter the search drifts.
5. **Wall-like bands didn't grow.** No edit added lines beside a wall, and spreading material
   along the whole outline was worth at most 0.5 points (c18, c19).

Points 2 and 5 are representation problems: one window at a time, every step toward a coherent
structure first breaks continuity and stacking. A surface is continuous in every layer and
stacked through all of them by construction, and one change moves it everywhere at once.

## Architecture

```text
rltp/
  slicing/            slice.py (mesh -> polygons), raster.py (cell masks), edges.py (outline edges)
  cover/              cover.py (nxt/prv paths, invariants), shell.py (walls, skins, support),
                      beads.py (fine bead grid: deposition, fit mode, edits), paths.py (true bead
                      directions), baselines.py, ops.py (edit operators)
  surfaces/           ribbons.py (Ribbon, starting families, Warp, Warp3D), layout.py (ribbons -> bead
                      paths: clip, cut, merge ends), cmaes.py (the search)
  eval/               fea2d.py (laminate plane-stress FEA, Tsai-Wu, p-norm), timing.py, metrics.py,
                      interlayer.py, stressmap.py, adapt.py
  search/, loop/      the edit-by-edit optimizer: annealing, guide, gradient start, chains
  part.py             PartState: layers, cached evaluation, full re-check, snapshots
  export/             gcode.py, viewer.py (+ template), state.py, render.py
  rl/                 the frozen policy (not developed in this fork)
scripts/
  ribbons.py          ribbon infill on a test part: start family + warp search, vs the reference
  suite.py, ...       the edit-by-edit suite runner and its tools; analysis/ one script per question
examples/             test parts (5 mm; 2 mm in 2mm/), c20_reference.json (the reference scores)
```

## Data model

- **Fine bead grid.** Cells are 0.1 mm; beads are 0.4 mm (kb = 4 cells). A layer's `mask`
  holds bead centreline nodes, linked by `nxt`/`prv` into paths and loops (each a stroke).
  Material per cell is deposited from the paths (a strip kb cells wide per step, half-bead end
  caps); D ≥ 0.5 counts as filled, D > 1 is over-extrusion.
- **Sparse layers.** `part` (in-part cells), `req` (cells that must stay filled: walls and
  solid skin) and `lock` (wall nodes, whose links never change). Walls are one centreline loop
  per outline, a bead inside the true outline. A cell is solid skin if a layer within
  `solid_layers` above or below lacks it.
- **Fit mode** (`--directions path`): strokes print as straight segments fitted within 0.1 mm of
  their nodes, and each bead is scored along its own path direction (`rltp/cover/paths.py`).
  Every ribbon run uses it.

## Evaluator

`docs/evaluator.md` documents it term by term. In short:

- **Strength.** One laminate plate: the layers are bonded and stretch together, stiffness summed
  over every layer's beads (orthotropic, rotated to each bead's direction). Each cell's Tsai-Wu
  failure index, aggregated as FI_agg = (mean FI⁸)^(1/8) over cells and load cases, each case
  scaled to 1 on the slicer layout. Load-agnostic parts get 10 random force and 10 random
  moment cases, each fixing one outline edge and loading another.
- **Bulk, stiffness.** The p = 2 aggregate and the mean compliance against the slicer layout.
- **Print time** (trapezoidal motion, junction deviation, travel, retraction), **continuity**
  (strokes, travel, ends on the boundary, free ends), **interlayer** (bond quality against
  interface statics), **infill target**, **coverage** (gaps in required cells, over-extrusion).
- **The score** is the weighted sum of each term's change as a fraction of the slicer layout's
  value ("points"). The fork uses the `--adapt` weights: strength 0.5, time 1, continuity 1,
  interlayer 0.3, infill 1, coverage 1, stiffness 0.5, bulk 2.

**Decision: the walls stay in the score.** They were 25–70% of Σ FI⁸ after optimization, and
they can't be edited, but leaving them out would stop rewarding layouts that relieve them, and
with print time in the objective it would reward thinning the material that supports them. They
are the most stressed region in all three test parts, so the reported score would improve while
the part got weaker. To weight strength more, raise its weight.

## Load-agnostic strength

Slicing a real part should need no loads. Strength only exists relative to a load, so "no loads"
means robust over a spread of plausible ones: 20 random edge cases, none special, each scaled
to its slicer-layout value, combined as a p = 8 mean (a soft worst case). Random edges are drawn
as parts are used: round hole : straight side : arc at 3 : 2 : 1. What carries over between
loads: a free edge carries only stress along the edge, so edge-following beads suit every load
that leaves it free; holes, inside corners and narrow sections concentrate stress under most
loads; path ends, seams and gaps weaken a part under any load. Away from the edges no direction
is best for every load. Held-out load seeds (`scripts/analysis/heldout.py`) check that gains
carry over to cases the search never saw.

## Surface infill (ribbons)

Decisions made on Nov 16, 2026, with the reasons.

### Representation

- **Explicit ribbons, not level sets.** A ribbon is a polyline in XY (open or closed). An
  implicit family (surfaces as level sets of a few 3D scalar fields) can't intersect within a
  family and controls spacing through its gradient, but its contours can split, so one
  continuous path per layer isn't guaranteed, and crossings are wanted anyway (below). Explicit
  ribbons match "a number of surfaces, each one toolpath".
- **Vertical by default, 3D with the same control method.** A vertical ribbon is the same
  curve in every infill layer. `--z-levels N` makes them 3D: the warp that bends them gets N
  height levels (`Warp3D`), every layer is laid from the ribbons' cut at its own height, and
  each surface stays continuous through the stack. Under the in-plane laminate evaluator a
  lean costs stacking and bonding and buys nothing directly; what height variation can buy
  there is ends and crossings that don't stack, and a mix of directions through the thickness.
  Out-of-plane loads, where leaning surfaces pay, need a 3D evaluator (Stage 2).
- **Open ribbons end on the walls** by construction: they are laid long and clipped to the
  infill region, so every end is anchored (a T-joint into the wall).
- **Closed and fixed ribbons.** Closed ribbons print as loops. `fixed` ribbons are laid first and
  not deformed: offset rings following every outline, the wall-like bands the edit-by-edit line
  never grew.
- **Tight turns live in the connections, not in the surfaces.** A 180° hairpin between lines
  0.4 mm apart needs a 0.2 mm radius; smooth curves with control points millimetres apart can't
  make it, and adding control points for it makes the search ill-conditioned. So surfaces stay
  smooth and describe the long runs, and the turns are made when generating toolpaths (Merging).

### Laying ribbons into a layer (`rltp/surfaces/layout.py`)

1. Clear the layer's infill; walls and skin stay.
2. Rasterize each ribbon into a 4-connected chain of centreline nodes (a diagonal step takes the
   corner cell nearer the curve).
3. Keep the nodes inside the infill region and at least kb − 1 cells (0.3 mm) from every bead
   laid before, and from the ribbon's own nodes further along it (a fold). Where a ribbon
   crosses or crowds an earlier one it is cut; each cut end touches the bead it met.
4. Merge path ends (below), anchor any end that still touches nothing, fix seams.
5. Vertical ribbons: copy the layout to every infill layer of the same geometry (two layouts
   with `alternate`); those layers share one cache of derived data, which halves the re-score.
   3D ribbons: lay every infill layer from the ribbons' cut at its mid-height.

### Merging during toolpath generation

Per layer, after the ribbons are cut: every open end is pulled back a bead first (a run meeting
the wall at a slant otherwise walls off one side of its own end on the thin band beside the
wall; on the frame this took 45° lines from 19 to 66 merges). Then ends are joined cheapest first
(Dijkstra within 4 mm) by connectors that run along the band just inside the walls (cost 1 per
step) or through open space a bead clear of every other bead (cost 1.5 per step), never closing a
path on itself and never within a bead of another connector. Beside parallel ribbons this is the
familiar zigzag; ends on a hole's edge can join ends on the outline's. Ends left unjoined are
re-extended to the wall. Surfaces are never merged geometrically: a merged surface would have to
contain the hairpin.

### Crossings

**Crossing is allowed and costed by what it does, not penalized as such.** Grid, triangle and
cubic infill cross every layer. A crossing costs double material where the beads overlap (area
w²/sin θ: cheap at 90°, a 2.3 mm smear at 10°) and, if crossings stack in the same place layer
after layer, a lump the nozzle can hit. It gains load paths between surfaces (in-plane shear
stiffness, a connected network) and simpler routing.

As built, the node grid can't share a cell between two strokes, so a crossing is cut into two
T-joints: the later ribbon stops a bead short on each side. `alternate` lays odd layers in the
reverse order, so the cut alternates and the joints don't stack. A true crossing (one stroke
passing over another, its overlap charged by coverage) needs a layout representation where
strokes can share a node; not built. A cost for stacked crossings is not built either (with
vertical ribbons and `alternate`, crossings alternate rather than stack).

### Spacing and bonding

Adjacent ribbons closer than a bead overlap; the layout keeps nodes kb − 1 cells apart (a
quarter-bead overlap at most), and the coverage term charges the excess. Side-by-side beads
within a bead width touch and share load in the laminate FEA; across layers, stacked vertical
ribbons bond fully in the interlayer term.

### Overhangs and leaning surfaces

A single-bead surface can lean about 45° (each 0.2 mm layer shifts half a 0.4 mm bead), 30° to be
safe; 60° leaves 12% overlap with the bead below. So the lean is capped (`--max-lean`, default
45°), and exactly: `Warp3D` builds each level from its neighbour toward the middle by steps
bounded to dz·tan(max_lean)/√2 per component, interpolates linearly in height, and interpolates
the height-varying part bilinearly in XY (a weighted average of grid values, which can't
overshoot them). The middle level keeps the smooth cubic warp. Rasterizing onto the 0.1 mm grid
adds up to half a cell per layer on top. The out-of-plane loads that make leaning surfaces worth
having need a 3D evaluator: about 175k hex elements for the plate at 0.4 × 0.4 × 0.2 mm, seconds
per solve, shared across load cases (Stage 2).

### Search

- **Start families** (`rltp/surfaces/ribbons.py`): parallel lines at an angle and spacing
  (`line_family`; 1.0 mm is 40% infill), offset rings beside every outline (`offset_rings`),
  and a crossing grid (`--cross`).
- **Deformation:** a smooth warp field over the part, a grid of 2D displacements every 5 mm
  interpolated by cubic splines, each component bounded by tanh to ±1.5 mm (`Warp`). All free
  ribbons bend together; spacing can tighten or open locally, so material can gather.
- **Optimizer:** CMA-ES (`rltp/surfaces/cmaes.py`) over the warp parameters, forked workers
  scoring candidates in parallel (`scripts/ribbons.py`). Gradient-free because the score is
  piecewise constant below the 0.1 mm grid. One evaluation (lay and re-score) is about 1.2 s on
  the frame for vertical ribbons, about 3 s for 3D ones (every layer laid and scored on its own).
- **3D, warm-started.** A 3D warp has N times the parameters, so a 3D search starts from a
  finished vertical one (`--warm`): its warp becomes the middle level and every height step
  starts at 0, so the search begins exactly at the vertical result and adds height variation
  only where it scores.
- **Next:** per-ribbon moves (move a control point, add or remove a ribbon, split, close a
  loop), starts traced along the gradient start's direction field, a band primitive (one
  surface plus a bead count, printed as a zigzag band), and gradients through a differentiable
  rasterization into the field model's adjoint.

### Results (c00, Nov 16, 2026)

Against each part's slicer layout, under the reference settings (`docs/experiments.md`):

| part | edit-by-edit reference | ribbons, warped in 3D |
| --- | --- | --- |
| frame | +61.0: FI_agg 0.633, 890 s, 940 strokes | +89.6: 0.735, 751 s, 374 strokes |
| plate | +19.5: 0.491, 1347 s, 1629 strokes | +120.9: 0.610, 1086 s, 437 strokes |
| wrench | +73.1: 0.733, 724 s, 727 strokes | +91.5: 0.802, 613 s, 240 strokes |

- **The representation does what it was for:** a third to a quarter of the strokes, faster
  prints, no free ends, every layer stacked.
- **Strength is the gap.** The reference's FI_agg is 9–20% lower. Its gradient start concentrates
  material where the score wants it; uniform ribbons only bend.
- **3D surfaces add 8–11 points, all strength.** Shifting lines from layer to layer spreads
  material through the stack. The laminate model's perfect bonding may overstate this: checked
  in cycle 1.
- **The warp search is not converged:** a rugged score (cuts and merges flip discretely) leaves
  each generation's mean about 20 points below its best.

## G-code and viewer

Each stroke is one extrusion run; fit mode prints straight segments fitted within 0.1 mm, with
extrusion sized to the bead (E = L·w·h / filament area). Travel retracts above 2 bead widths.
`viewer.html` shows each layer (bead direction, print order, failure index, strokes; the printed
path at true angles), plays back the nozzle, and lists the load cases; `scripts/ribbons.py`
writes one per run.

## Next steps

1. Per-ribbon moves and a search that changes the number of ribbons.
2. Starts from the gradient start's direction and density fields (streamlines, spaced by
   density).
3. True crossings (strokes sharing a node) and a stacked-crossing cost.
4. A band primitive for edge-following ribs.
5. Stage 2: a 3D evaluator, so leaning and warped surfaces (built, `--z-levels`) are scored on
   what they are for.
6. Held-out load seeds for every ribbon result.

## Risks and open questions

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Rasterization noise | curves on a 0.1 mm grid swing ±0.5 cell, so clearances and merges flicker | kb − 1 clearance; connectors pulled back; the warp moves whole families smoothly |
| Warp is a weak lever | one smooth field bends every ribbon together; it can't add or remove ribbons | per-ribbon moves next |
| Crossings as T-joints | each crossing costs a stroke, which biases against crossing networks | true crossings in the layout representation |
| Stacked ends | identical layers stack every path end, which bond quality charges | alternate merges by layer |
| In-plane evaluator | it charges a lean (less overlap, weaker bonds) but can't reward one: out-of-plane loads aren't modelled | Stage 2, 3D evaluator |
| Placeholder constants | E1/E2 = 1.27 makes angles cheap; real filaments may be more directional | coupons when absolute numbers matter |

**Open questions**

- Does the search find edge-following bands (rings) on its own, once the number and placement of
  ribbons can change?
- How many surfaces does a part want, and does that depend on its size or only on the infill
  target?
- Do true crossings beat T-joints once overlap is charged honestly?
