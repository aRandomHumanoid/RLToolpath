# Experiments: surface (ribbon) infill

Working notes, newest last. Summary in the README ("Results") and the design doc ("Surface
infill"). The edit-by-edit optimizer's notes (cycles 0–20) are on branch `selfimprove-cycles`,
`runs/selfimprove/CYCLES.md`.

## Setup (Nov 16, 2026)

- **Parts.** frame, plate, wrench: the 5 mm test parts (25 layers: 2 solid top and bottom,
  21 infill layers sharing one outline), 1 wall, 40% infill target, 0.1 mm grid, 0.4 mm beads.
- **Score.** Path mode (beads scored along their own direction, printed as fitted segments),
  10 random force + 10 random moment edge cases (seed 0, none special), the `--adapt` weights
  0.5/1/1/0.3/1/1/0.5/2 (strength, time, continuity, interlayer, infill, coverage, stiffness,
  bulk). Points = weighted % change of each term against the part's slicer layout.
- **Reference** (`examples/c20_reference.json`): the c20 four-class run of the edit-by-edit
  optimizer on the same settings, rescored on this footing.

  | part | slicer: FI_agg, time, strokes | c20 gradient start | c20 final (8,000 annealing edits) |
  | --- | --- | --- | --- |
  | frame | 1.000, 772 s, 696 | −40.3 (0.906, 895 s, 1450) | **+61.0** (0.633, 890 s, 940) |
  | plate | 1.000, 1083 s, 801 | −123.3 (0.782, 1342 s, 2215) | **+19.5** (0.491, 1347 s, 1629) |
  | wrench | 1.000, 635 s, 700 | −21.0 (0.808, 713 s, 1146) | **+73.1** (0.733, 724 s, 727) |

## Layout fixes found while building it

- **Clearance kb − 0.5 cut rings to pieces.** Curves drawn on the 0.1 mm grid swing about ±0.5
  cell, so touching rings (4 cells apart nominally) and rings beside curved walls fell inside
  3.5 cells in places. Now kb − 1 (a quarter-bead overlap at most, charged by coverage), and
  the infill region is a Euclidean clearance from the wall and skin nodes rather than a square
  erosion, which cut deeper on diagonals and curves.
- **Ends on different edges couldn't merge.** Connectors ran only along the band just inside
  the walls; ends on a hole's edge and on the outline sit on separate bands. Connectors may now
  also cross open space (cost 1.5 per step against 1): frame at 0°, strokes 843 → 780.
- **Slanted ends walled themselves off.** A 45° line crosses the thin band beside its own end,
  so neighbouring ends never shared a reachable gap: 19 merges of 79 possible on the frame.
  Pulling every end back a bead before merging (and re-extending the unmerged ones): 66.
  At 0°, 34 → 39 merges.

## Starting families, no search

`scripts/ribbons.py PART DIR --angle A --rings R --evals 0`, lines 1.0 mm apart (40% infill),
rings 0.4 mm apart (a solid band) starting a bead inside the wall. Points against the slicer
layout; FI_agg, print time, strokes.

| family | frame | plate | wrench |
| --- | --- | --- | --- |
| lines 0° | **+52.8** (1.033, 727 s, 297) | +78.5 (0.997, 1019 s, 360) | **+53.8** (1.002, 621 s, 301) |
| lines 45° | +40.5 (0.944, 802 s, 465) | +76.0 (0.817, 1083 s, 486) | +33.1 (1.042, 654 s, 301) |
| lines 90° | +29.8 (1.256, 754 s, 339) | **+81.9** (1.027, 1034 s, 339) | +10.3 (1.466, 640 s, 238) |
| lines 135° | +34.1 (0.946, 804 s, 465) | +79.8 (0.831, 1078 s, 507) | +29.5 (1.006, 661 s, 364) |
| lines 0° + 1 ring | +35.8 (0.942, 823 s, 801) | +28.1 (0.984, 1129 s, 822) | +3.8 (1.110, 686 s, 658) |
| lines 45° + 2 rings | −6.1 (0.804, 975 s, 1095) | +46.5 (**0.644**, 1267 s, 1137) | −39.6 (0.869, 800 s, 994) |
| grid 0°/90°, alternating | −24.1 (1.037, 909 s, 780) | −11.8 (0.987, 1337 s, 674) | −23.6 (0.983, 763 s, 662) |
| grid 45°/135°, alternating | −174.8 (1.129, 958 s, 2271) | −291.1 (0.919, 1433 s, 4150) | −135.7 (1.088, 761 s, 1594) |

- **Straight ribbons are a strong start on continuity alone.** Merged at the walls, 0° lines
  print in 297–360 strokes against the slicer's 696–801 and the c20 finals' 727–1629, at equal
  or lower print time. On the plate every straight family beats the c20 final (+19.5) before
  any search; frame and wrench come within 8 and 19 points of theirs.
- **They are no stronger than the slicer** (FI_agg 0.94–1.03 at the best angle; 90° on the
  wrench, across its handle, 1.47). The c20 finals are 27–51% stronger.
- **Rings buy strength and cost strokes.** Every ring level adds a loop per outline per layer,
  cuts the lines it meets, and adds material (infill 43–59% at 1.0 mm line spacing, which the
  infill and time terms charge). Bulk and stiffness rise with each ring (plate 45°: bulk
  +10.8 → +29.1 points with 0 → 2 rings, FI_agg 0.817 → 0.644): edge-following bands are worth
  strength, as the load-agnostic reasoning says, but not their continuity at this spacing.
- **Grids lose** because a crossing is cut into two T-joints, a stroke each (design doc,
  Crossings). At 45° the cuts multiply: thousands of short pieces.

## Warp search (c00, `runs/ribbons/c00_baseline`)

Each part's best straight start (frame 0°, plate 90°, wrench 0°) with its warp searched by
CMA-ES: 5 mm warp grid, ±1.5 mm, σ 0.3, population 12, 1,500 evaluations (A). Then 3D ribbons
warm-started from A: 3 height levels, lean ≤ 45°, σ 0.2, 1,200 evaluations (3D). And 45° lines
1.2 mm apart plus one ring beside every wall, searched like A (B).

| part | c20 final | A: vertical warp | 3D | B: 45° + 1 ring |
| --- | --- | --- | --- | --- |
| frame | +61.0 (0.633, 890 s, 940) | +81.2 (0.839, 756 s, 360) | **+89.6** (0.735, 751 s, 374) | +39.1 (0.896, 834 s, 759) |
| plate | +19.5 (0.491, 1347 s, 1629) | +112.0 (0.786, 1074 s, 339) | **+120.9** (0.610, 1086 s, 437) | +61.7 (0.834, 1107 s, 822) |
| wrench | +73.1 (0.733, 724 s, 727) | +81.0 (0.914, 607 s, 196) | **+91.5** (0.802, 613 s, 240) | +34.6 (0.903, 680 s, 616) |

(points; FI_agg, print time, strokes.)

- **Ribbons beat the reference on the composite score on every part, through continuity.**
  The 3D results print in 240–437 strokes against 727–1629 and are 15–19% faster to print.
  Continuity is worth +47 to +59 points against the reference's −49 to +10.
- **They trail it on strength.** FI_agg 0.61–0.80 against 0.49–0.73: strength, bulk and
  stiffness together are worth +30 to +65 points for ribbons against +52 to +92 for the
  reference. Closing that gap without giving back continuity is the cycles' main target.
- **The searches are far from converged.** Each generation's mean sat about 20 points below the
  best and σ never shrank (0.23–0.26 at the end): small warps flip discrete cut and merge
  decisions, so the score is rugged, and the best points were lucky samples more than the
  product of a converging mean. Plate A found its best by 745 evaluations and never improved.
- **3D added 8.4–10.5 points on every part, all through strength** (FI_agg −12% to −22% from
  the vertical result; continuity −3 to −15, interlayer unchanged). The leans went near the
  cap (41–42°). Likely mechanism: lines shifting from layer to layer spread material more evenly
  through the stack, which the laminate model rewards. To check: the laminate bonds layers
  perfectly, so part of this could be the model's optimism (cycle 1).
- **Rings lost 40–50 points**: 616–822 strokes. They raise bulk and stiffness a little but each
  ring is a loop per layer and cuts every line it meets.

## Improvement cycles

Each cycle: read the previous results, name what is lacking (strength, continuity, or the
process itself), propose a change, review it adversarially (how could it make toolpaths worse,
or the numbers mislead? how to guard against that), implement it with tests, run it, record.
Runs: `scripts/ribbon_suite.py runs/ribbons/cNN_topic --arm ...`, three parts unless noted;
points against the slicer layout as above. A change is kept when it wins on the parts without
a guard tripping.

| cycle | change | frame | plate | wrench | kept |
| --- | --- | --- | --- | --- | --- |
| c00 | baseline: straight start + CMA-ES warp, vertical (A) / then 3D (3D) | +81.2 / +89.6 | +112.0 / +120.9 | +81.0 / +91.5 | — |
