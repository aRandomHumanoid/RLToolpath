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
| c01 | elitist (1+λ) search instead of CMA-ES (stopped early) | +64.8 at 661 ev. | +91.2 at 481 ev. | – | no |
| c02 | skin directions (0/90 per skin layer) searched with the warp | **+92.3** | +110.3 | **+108.8** | yes |
| c03 | a second line family at +90° on alternate infill layers | **+100.8** | **+117.0** | +97.4 | frame, plate |
| c04 | each family its own warp / line spacing searched | +94.0 / +90.4* | +113.5 / – | – | no |
| c05 | density-adaptive families (stress^0.5) | +86.7* | +105.8* | **+114.6** | wrench |
| c06 | refine the best with small CMA-ES steps (σ 0.1, +750 ev.) | +102.3 | +117.0 | – | no |
| c07 | refine the best with local block moves (+750 ev.) | **+109.8** | **+133.7** | **+121.5** | yes |
| c08 | screened start angles / local moves only (2,250 ev.) | +97.2 / +91.1 | +127.1 / +124.1 | +120.5 / +119.3 | no |
| c09 | 3D local refinement (+600 ev.) / vertical control | +110.3 / **+110.2** | +134.2 / **+135.5** | +124.5 / **+122.0** | no (3D) |
| c10 | held-out check of local refinement (process) | holds | holds (−20 on unseen loads) | – | — |
| c11 | search the plate over 40 load cases (judged on held-out seeds) | – | worse held-out | – | no |
| c12 | warp amplitude ±3 mm (+600 local ev.) | +109.9 | +134.5 | +122.9 | no |
| c13 | budget: CMA-ES 750 + local 1,500 / two CMA-ES 750 runs + local 750 | +112.3 / +111.0 | +125.1 / +127.2 | +119.9 / +126.1 | no (noise) |
| c14 | three line families rotating 0/60/120° by layer | +96.8 | +124.5 | +110.6 | no |
| c15 | density families normalised per cross-section (gradient field density) | +99.5 | +127.5 | – | no |
| c16 | connector reach 6–12 mm / re-routed merge passes (re-laid bests, no search) | +111.7 | +130.9 | +110.8 | no |
| c17 | density smoothed 3 mm along the lines (plate); on one family (frame); field density (wrench) | +100.3 | **+142.5** | +117.1 | plate |
| c18 | density fitted to 100 load cases (held-out means in brackets) | +112.3 (+113.8) | +139.6 (+123.6) | **+124.9** (+125.5) | plate, wrench |
| c19 | replicates of the kept settings, search seed 1 (held-out means) | – | +142.8 (+121.1) | **+128.3** (+127.3) | (both hold) |
| c20 | final settings and the reference on one footing (held-out means) | +112.3 (+118.2) | +139.6 (+123.6) | +124.9 (+125.5) | — |

### c01: elitist (1+λ) search instead of CMA-ES — not kept (`runs/ribbons/c01_elitist`)

- **Lacking.** c00's CMA-ES mean ran about 20 points below its best and σ never shrank: on a
  rugged score the mean drifts away from lucky finds.
- **Change.** `--optimizer elitist`: sample λ = 12 points around the best so far, move only on
  improvement, success-rule step size ((1+λ)-CMA-ES constants), and reset σ after 12 generations
  without improvement.
- **Adversarial review.** Elitism can stick on a narrow lucky spike; isotropic steps crawl in 168
  dimensions; one seed is noisy. Guards: the stall reset, c00's start and budget, a seed
  replicate.
- **Result (stopped at 661 / 481 evaluations).** Frame +64.8 (CMA-ES had +75.6 by 745), stuck
  since generation 25 with σ back at 0.25; plate +91.2 (CMA-ES +101.2 by 373).
- **Why.** The guard was the bug. In 168 dimensions the success rule shrinks σ about 2% per
  generation (damping d = 1 + n/2λ = 8), so halving it takes about 35 generations, but the stall
  reset restored σ every 12: every sample moved all parameters by about 0.25 and almost none
  improved. Not kept; CMA-ES stays. A restart interval must be longer than the step-size
  controller's time constant.
- **Kept from this cycle (process).** Layers sharing a cache compute their laminate columns and
  failure indices once (PartState, post-processing), and `apply_layout` skips the support
  context, which only edits read: 1.20 → 0.89 s per evaluation on the frame, identical scores
  (test). Tried without a search: streamlines of the load-agnostic stress consensus as the start
  (`--start stream`): frame +24.7 (FI_agg 1.049, 528 strokes) against +52.8 for straight lines —
  the spacing opens where the field spreads and lines break up.
- **Where the strength gap is** (`scripts/analysis/ribbon_regions.py`, frame; p = 8 FI, its
  share of Σ FI⁸):

  | | walls | top/bottom solid layers | infill 0.4–1.2 mm | interior |
  | --- | --- | --- | --- | --- |
  | reference | 0.740 (39%) | 0.670 (25%) | 0.659 (15%) | 0.638 (21%) |
  | ribbons, vertical | 0.830 (10%) | **1.026 (78%)** | 0.759 (4%) | 0.738 (7%) |
  | ribbons, 3D | 0.827 (29%) | **0.840 (46%)** | 0.734 (10%) | 0.712 (15%) |

  Ribbons never touch the solid layers: they keep the slicer's skin, one direction per layer,
  which holds most of the score. The reference's gradient start re-oriented the skins too
  (1.247 → 0.670).

### c02: skin directions searched with the warp — kept (`runs/ribbons/c02_skin`, `c02_skin_screen`)

- **Lacking.** The top/bottom solid layers keep the slicer's skin and hold 46–78% of Σ FI⁸
  (c01's region split).
- **Screen first** (c00's best warp, skins re-laid with `fine_layout`, no search; points):

  | skin, bottom→top | frame | plate | wrench |
  | --- | --- | --- | --- |
  | slicer (0/90/90/0) | +81.2 | +112.0 | +81.0 |
  | all 0° | +94.8 | **+115.9** | +104.2 |
  | all 90° | +87.2 | +107.6 | **+106.7** |
  | all 45° / all 135° / 45-135-135-45 | −9.8 / −14.6 / −17.7 | +9.4 / +8.0 / +14.5 | +24.9 / +39.8 / +23.4 |
  | stress4 areas | +69.4 | +31.0 | +29.4 |

  Diagonal skins and consensus areas lose to strokes (diagonal skin lines aren't zigzag-linked:
  918–930 strokes on the frame). The 0/90 gains are almost all the **interlayer** term (wrench
  +28.1, frame +16.4): a skin printed in the same order as the infill beside it shortens the
  time between depositing a cell and the layer above it, which the bond model rewards. In-plane
  strength barely moves (frame 0°: FI_agg 0.839 → 0.864, plate 0.786 → 0.732, wrench 0.914 →
  0.988).
- **Change.** `--skin-search`: one parameter per skin layer (< 0: 0°, else 90°), searched with
  the warp, starting at the slicer's skins.
- **Adversarial review.** The gain rides on the bond-timing model (placeholder τ = 30 s), and
  uniform skins are anisotropic in-plane. Guards: report in-plane FI_agg next to the score;
  load-agnostic FI_agg charges anisotropy; a noise control (c00's setup at seed 1).
- **Result** (1,500 evaluations from the straight start; FI_agg in brackets):

  | | frame | plate | wrench |
  | --- | --- | --- | --- |
  | c00 A | +81.2 (0.839) | +112.0 (0.786) | +81.0 (0.914) |
  | c02 skin search | **+92.3** (0.895) | +110.3 (0.754) | **+108.8** (0.878) |
  | skins chosen | 90/90/90/90 | 0/0/90/0 | 0/0/0/90 |
  | c00 setup, seed 1 | +79.2 (0.937) | | |

  Seed noise on the frame: 2.0 points (one replicate), so the plate's −1.7 is noise. Frame and
  wrench gain 11 and 28 points, through the interlayer term (+22.1, +29.4); the frame's in-plane
  strength fell (0.839 → 0.895) while its bonds improved. Kept, with in-plane FI_agg tracked
  from now on.

### c03: two line families alternating by layer — kept for frame and plate (`runs/ribbons/c03_alternate`)

- **Lacking.** In-plane strength: every infill layer ran the same family, a one-directional
  laminate, weak across its lines under random loads.
- **Change.** `--alt-angle`: odd infill layers lay a second family at angle + 90° (the warp
  bends both); no crossings within a layer, so no T-joint strokes. Screened without search:
  frame 0/90 +67.0 (FI_agg 0.876) against one family +52.8 (1.033); plate 90/0 +110.9 (0.744)
  against +81.9 (1.027); wrench 0/90 +59.9 (0.940) against +53.8 (1.002). Interlayer barely
  changed (frame +1.9 → +1.5, plate +12.6 → +10.0): consecutive layers bond only at crossings,
  but these parts' Z term is small.
- **Adversarial review.** Less overlap between layers weakens real Z bonds more than the model
  may show; a crossing layup may suit some shapes and not others. Guards: the interlayer term,
  each part judged separately.
- **Result** (with skin search, 1,500 evaluations; against c02):

  | | frame | plate | wrench |
  | --- | --- | --- | --- |
  | c02 (one family) | +92.3 (0.895) | +110.3 (0.754) | **+108.8** (0.878) |
  | c03 (alternating) | **+100.8** (0.810) | **+117.0** (0.644) | +97.4 (0.914) |

  Frame and plate gain 7–9 points and 10–15% in FI_agg. The wrench loses 11.4: its long, narrow
  handle makes the 90° family short strokes (268 against 220) that add little stiffness along
  it (bulk +5.3 against +10.8). Kept per part: alternating for frame and plate, one family for
  the wrench.

### c04: per-family warps; line spacing searched — neither kept (`runs/ribbons/c04_split_spacing`)

- **Lacking.** With two families one warp bends both alike, so a bend that suits one family's
  load path drags the other along; and density was fixed at the start's spacing.
- **Changes.** SPLIT: `--warp-per-family`, each family its own warp (336 warp parameters on the
  frame). SPACE: `--spacing-search`, one parameter scaling the line spacing within ±20%.
- **Adversarial review.** Twice the parameters converge more slowly at equal budget; regenerating
  every line whenever the spacing changes makes the score even more rugged. Guards: c03's
  settings and budget otherwise.
- **Result.** SPLIT: frame +94.0 (FI_agg 0.818), plate +113.5 (0.666), against c03's +100.8 and
  +117.0. SPACE (stopped): frame +90.4 at 1,141 evaluations where c03 had +100.8 by 961; plate
  level with c03 at 301. Both risks materialised; neither kept.

### Screens between c04 and c05 (no search)

- **Edge rings in the skins** (`--skin-ring-count`, c03's best layouts): every ring made it
  weaker — frame FI_agg 0.810 → 0.813 / 0.855 / 0.846 with 1 / 2 / 3 rings, plate 0.644 → 0.651
  / 0.696 / 0.712. The layers stretch together, so stiffening the skins' edge band draws load
  into it, where the hot spots are.
- **The reference's skins on our infill** (`--skin-from` the c20 gradient start's and final
  states): FI_agg 0.814 / 0.832 (frame), 0.641 / 0.642 (plate), 0.861 / 0.863 (wrench) against
  0.810, 0.644, 0.878 with our skins, at more strokes. The reference's low skin FI (0.67, the same
  as its infill's) is not about skin directions: its infill relieves the skins.
- **Where the reference puts its infill** (frame, fill by quartile of the load-agnostic stress
  magnitude, low → high): reference gradient start 0.33 / 0.31 / 0.43 / 0.54, final 0.39 / 0.35 /
  0.43 / 0.51; ribbons (c03) 0.42 / 0.40 / 0.39 / 0.38. It concentrates material where stress is
  high; ribbons spread it evenly and the warp never learned otherwise.

### c05: density-adaptive line families — kept for the wrench (`runs/ribbons/c05_density`)

- **Lacking.** Material placement: ribbons spread infill evenly where the reference concentrates
  it at high stress.
- **Change.** `--density-gamma γ`: each family spaced by a density field, the 40% target times
  (load-agnostic stress magnitude / mean)^γ, clipped to 0.15–0.8, placed by equal steps of
  cumulative density across the family (`density_family`: lines curve and bunch, staying
  continuous), and calibrated until the laid-out infill is the target.
- **Adversarial review.** The isotropic consensus may not be where the laminate model wants
  material; dense bands could be cut by the clearance rule; and an uncalibrated first screen
  confounded placement with quantity (infill 44–46%, so most of its "gain" was extra material).
  Guards: γ = 0 as the control, calibration to equal material, the cap.
- **Screen at equal material** (no search): γ = 0.5 frame +78.2 (FI_agg 0.825, 417 strokes)
  against +80.0 (0.837, 311) uniform; plate +85.7 (0.646, 474) against +109.7 (0.744, 349);
  wrench +91.5 (0.965, 325) against +70.9 (1.043, 304). Placement alone strengthens plate and
  wrench, but curved families cost strokes: where density changes along a line the lines tilt,
  so their perpendicular gap shrinks below the spacing intended across the family and the
  clearance rule cuts them (18 cuts per frame layer; uniform lines: none). A lower cap doesn't
  help (the density never reaches it). The gradient field step's own optimised density
  (`--density-from fields`, cached per part) is stronger still (frame FI_agg 0.772) but far more
  contrasty: 822 strokes.
- **Result** (with skin search; frame and plate alternating families; 1,500 evaluations):
  frame and plate stuck at +86.7 and +105.8, far behind c03, stopped. **Wrench +114.6**
  (FI_agg 0.852, 286 strokes) against c02's +108.8 (0.878). Kept for the wrench.

### Held-out load seeds (process; `scripts/ribbon_heldout.py`)

Every layout so far was scored on the twenty random cases of seed 0, which the search optimised
against. Rescored on seeds 1 and 2 (each against that seed's own slicer layout):

| layout | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| frame, reference | +61.0 (0.633) | +44.9 (0.732) | +47.0 (0.667) |
| frame, c03 | +100.8 (0.810) | +103.9 (0.826) | +104.3 (0.805) |
| plate, reference | +19.5 (0.491) | −16.0 (0.629) | −23.9 (0.717) |
| plate, c03 | +117.0 (0.644) | +100.7 (0.730) | +100.9 (0.730) |
| wrench, reference | +73.1 (0.733) | +70.7 (0.715) | +62.8 (0.796) |
| wrench, c02 | +108.8 (0.878) | +112.6 (0.821) | +109.0 (0.885) |

The reference was partly fitted to seed 0: on unseen cases its FI_agg rises 5–46% and its score
falls up to 43 points. Ribbons hold on frame and wrench and drop 16 points on the plate. On
held-out loads the strength gap shrinks to 3–13% (and reverses on the wrench's seed 1).
From here on held-out seeds guard every kept change.

### c06: refine with small steps (σ 0.1) — not kept (`runs/ribbons/c06_exploit`)

- **Lacking.** Every search ended with σ 0.21–0.26 and generation means 15–25 points below the
  best: the neighbourhood of the best was never searched finely.
- **Change.** Continue from c03's best for 750 evaluations at σ 0.1 (`--warm` now restores the
  whole point, skins included, when the settings match).
- **Adversarial review.** More evaluations alone would show "improvement"; small steps may
  stall on a rugged score. Guard: a σ 0.3 control with the same extra budget from the same point.
- **Result.** σ 0.1: frame +100.8 → +102.3, plate +117.0 → +117.0; even at σ 0.1 the generation
  mean sat 13 points below the best. The control found nothing in its first 181 evaluations and
  was stopped once c07 answered the question. A whole-vector step of any size moves every warp
  node at once, and every node move flips cut and merge decisions somewhere: the best point is a
  narrow spike. Not kept.

### c07: local block moves — kept as the refinement stage (`runs/ribbons/c07_local`)

- **Lacking.** c06: around the best, any whole-vector step flips decisions everywhere.
- **Change.** `--optimizer local` (`LocalSearch`): each candidate perturbs 1–3 random blocks — one
  warp node's two components, or one skin parameter — by σ·N(0, 1) (σ 0.5); the best of the
  12 replaces the incumbent only if better; σ follows the success rate (target 1/5, time
  constant about 10 generations, within σ0/8..4σ0 — c01's lesson). A move bends the ribbons
  near one node only.
- **Adversarial review.** Greedy local moves stall in local optima; a narrow spike may not be a
  robust layout. Guards: equal budget from the same start as c06 (frame, plate) and a CMA-ES
  continuation control (wrench); held-out seeds before relying on it.
- **Result** (750 evaluations from each part's best):

  | | frame | plate | wrench |
  | --- | --- | --- | --- |
  | start | +100.8 (0.810) | +117.0 (0.644) | +114.6 (0.852) |
  | CMA-ES (c06 σ 0.1 / wrench σ 0.3) | +102.3 | +117.0 | +118.6 |
  | local block moves | **+109.8** (0.797, 345 strokes) | **+133.7** (0.632, 340) | **+121.5** (0.833, 265) |

  Local moves win on every part, on strength and strokes together. Kept: CMA-ES to explore,
  local moves to refine (`--then-local N`).
- **Coarse to fine doesn't carry over.** Resampling the frame's 5 mm best onto a 3 mm grid
  (`regrid`) keeps its strength (FI_agg 0.803) but loses 27 points to strokes (416 against 345):
  the resampled field differs slightly between nodes and that flips merges. The tuned optimum is
  narrow.

### c08: start selection; local moves only — neither kept (`runs/ribbons/c08_start_local`)

- **Lacking.** Each part's start angle came from a coarse four-angle screen.
- **Change.** `scripts/ribbon_screen.py`: one build, every 15° (one family, and alternating with
  the perpendicular one), uniform and density-spaced (~1 s per start). Best starts: frame
  30°/120° +89.4 (FI_agg 0.782) against +80.0 for 0°/90°; plate 0°/90° stays first (30°/120°:
  +104.7 but FI 0.674 against 0.743); wrench 45°/135° with density +90.2 (0.857). Searched from
  those starts at 2,250 evaluations: CMA-ES 1,500 then local 750 (HYB), and local only (LOC).
- **Adversarial review.** A start's rank need not survive the search (the search adds 20–35
  points, and headroom differs). Guard: equal total budget against the current path
  (c03/c05 + c07: CMA-ES 1,500 + local 750).
- **Result.** HYB: frame +97.2, plate +127.1, wrench +120.5; LOC: +91.1, +124.1, +119.3; current
  path +109.8, +133.7, +121.5. The better-looking starts lose after search on every part, and
  local moves alone trail the hybrid: CMA-ES exploration is worth keeping before refinement.

### c09: 3D surfaces with local refinement — not kept (`runs/ribbons/c09_3d_local`)

- **Lacking.** Every best was vertical; in c00 3D (under CMA-ES) added 8–10 points.
- **Change.** Warm-start a 3-level warp from each vertical best (exact: the middle level is the
  vertical warp, skins carried over) and refine with local moves per level, lean ≤ 45°.
- **Adversarial review.** A gain could be just the extra budget; the laminate may reward lean
  optimistically. Guard: a vertical local control with the same 600 evaluations.
- **Result.** 3D / vertical: frame +110.3 / +110.2, plate +134.2 / +135.5, wrench +124.5 /
  +122.0. From a locally refined vertical optimum, 3D is worth −1.3 to +2.5 at three times the
  cost per evaluation. Not kept for now; the vertical controls are the new bests.

### c10: held-out check of local refinement — passes (process)

| layout | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| frame, c03 (before local moves) | +100.8 (0.810) | +103.9 (0.826) | +104.3 (0.805) |
| frame, c09 vertical (after) | +110.2 (0.796) | **+114.8 (0.787)** | **+113.9 (0.784)** |
| plate, c03 | +117.0 (0.644) | +100.7 (0.730) | +100.9 (0.730) |
| plate, c09 vertical | +135.5 (0.628) | **+115.1 (0.735)** | **+115.8 (0.741)** |

Local refinement doesn't fit seed 0 more than CMA-ES did: the frame keeps its whole gain on
unseen loads, strength included; the plate keeps 14–15 of its 18.5 points, though its small
strength gain doesn't carry (held-out FI_agg 0.735–0.741 against 0.730). The plate loses about
20 points to held-out loads either way, the target of c11.

### c12: larger warp amplitude (±3 mm) — not kept (`runs/ribbons/c12_amp3`)

- **Lacking.** With local moves effective, the ±1.5 mm bound might be what stops lines bunching
  further (near holes, say).
- **Change.** `--amp 3`, local refinement for 600 evaluations from c07's bests (the warm start
  carries over exactly: same grid, node displacements re-inverted through the new bound).
- **Adversarial review.** Bigger moves fail more often on a rugged score. Guard: c09's vertical
  controls, same start and budget at ±1.5 mm.
- **Result.** ±3 / ±1.5 mm: frame +109.9 / +110.2, plate +134.5 / +135.5, wrench +122.9 /
  +122.0 — no difference. With c09, 600 more local evaluations add 0.4–1.8 points: the local
  optima are nearly converged; further gains need different structure, not more refinement.
- **Where the points are now** (c09 vertical bests against the reference): ribbons lead on
  continuity (+59.6 / +67.4 / +60.0 against +8.5 / −48.6 / +10.4) and interlayer, and trail on
  strength + bulk + stiffness by 35 / 34 / 18 points, bulk (p = 2, weight 2) the largest part:
  +7.9 / +25.0 / +18.1 against +23.8 / +39.4 / +25.0.
- **Probe (no cycle):** reversing strokes changes the score through print timing (frame:
  odd layers −6.0, even −13.0, all −8.8), but the layout as laid beats every pattern tried.

### c11: search over 40 load cases — not kept (`runs/ribbons/c11_cases40`)

- **Lacking.** The plate's search fits seed 0's twenty cases (about 20 points lost on unseen ones).
- **Change.** `--cases 20`: search the plate over 20 force + 20 moment cases (CMA-ES 1,500, then
  local 750).
- **Adversarial review.** Each evaluation is slower (about 3 s per worker against 1 s), so fewer
  steps fit; a 40-case score isn't comparable to a 20-case one. Guard: held-out seeds 1–3,
  against c09's 20-case result.
- **Result** (points against each seed's slicer layout; strength / bulk / stiffness points):

  | plate | seed 0 | seed 1 | seed 2 | seed 3 |
  | --- | --- | --- | --- | --- |
  | reference | +19.5 (36/39/17) | −16.0 (23/20/14) | −23.9 (17/20/12) | −15.0 (24/20/13) |
  | 20-case search (c09) | +135.5 (23/25/9) | **+115.1** (15/14/8) | **+115.8** (15/15/8) | **+121.6** (17/17/9) |
  | 40-case search | +113.5 (15/17/6) | +104.2 (12/12/5) | +108.5 (14/13/5) | +112.4 (16/15/6) |

  Worse on every held-out seed: its CMA-ES phase found nothing better than the start in 1,500
  evaluations (local moves added 5 points), so the slower, flatter search cost more than the
  reduced fitting gained. Not kept. The table also sizes the real strength gap: on unseen loads
  the reference's strength + bulk + stiffness points fall from 92 to 49–57, against 37–43 for
  ribbons — 15–20 points of its lead are genuine, the rest was fitted to seed 0.

### c13: splitting the budget differently — not kept (`runs/ribbons/c13_budget`)

- **Lacking.** c12: local refinement converges within about 600 evaluations, while CMA-ES gets
  1,500. Is CMA-ES's second half worth more as refinement, or as a second start?
- **Change.** The same 2,250 evaluations split two other ways: SPLIT, CMA-ES 750 then local
  1,500; MULTI, two CMA-ES runs of 750 (seeds 0 and 1), then local 750 from the better.
- **Adversarial review.** A shorter CMA-ES phase may stop in a worse basin; two starts halve
  each run's depth; single runs are noisy. Guards: the current path at equal budget (c07), and
  the seed pair, which measures the noise directly.
- **Result:**

  | | frame | plate | wrench |
  | --- | --- | --- | --- |
  | CMA-ES 750, seed 0 / seed 1 | +97.3 / +92.7 | +112.9 / +118.0 | +106.7 / +111.6 |
  | SPLIT: seed 0, then local 1,500 | **+112.3** (0.805, 322 strokes) | +125.1 (0.647, 329) | +119.9 (0.936, 265) |
  | MULTI: the better seed, then local 750 | +111.0 (0.808, 322) | +127.2 (0.664, 345) | **+126.1** (0.850, 193) |
  | current: CMA-ES 1,500, then local 750 (c07) | +109.8 (0.797, 345) | **+133.7** (0.632, 340) | +121.5 (0.833, 265) |

  Two CMA-ES seeds differ by 4.6–5.1 points at 750 evaluations, and the three allocations
  differ by up to 8.6 points with no consistent sign (part means: SPLIT +119.1, MULTI +121.4,
  current +121.7). Not kept. **Process consequence:** one run per arm resolves about 5 points
  per part; smaller differences need replicate seeds before they count.

### c14: three line families rotating by layer — not kept (`runs/ribbons/c14_three`)

- **Lacking.** In-plane strength under random loads: a 0/90 cross-ply is weak at ±45°. Without
  search, three families rotating 0/60/120° by layer start 5–11% stronger (FI_agg) than each
  part's current start: frame 0.802 against 0.876, plate 0.662 against 0.744, wrench 0.887
  against 0.939.
- **Change.** `--families 3`: infill layer i lays family i mod 3, 60° apart, all bent by one
  warp; skin search; CMA-ES 1,500, then local 750.
- **Adversarial review.** Diagonal families meet the walls at slants and turn more (strokes, print
  time), and each direction gets one layer in three. The start's strength edge may not survive
  the search. Guards: the current path at equal budget; strokes and time reported.
- **Result.** Frame +96.8 (FI_agg 0.786, 363 strokes), plate +124.5 (0.632, 387), wrench +110.6
  (0.859, 248), against +109.8 (0.797, 345), +133.7 (0.632, 340) and +121.5 (0.833, 265). After
  search the strength edge is gone (frame 1.4% stronger, plate level, wrench weaker than its
  density family). The frame loses 13 points to print time, infill share (its search drifted to
  44.6%), continuity and bonds; the plate loses 9 to continuity. Not kept.
- **Where the strength still sits** (`scripts/analysis/ribbon_regions.py`, c09's vertical bests;
  p = 8 mean FI and share of Σ FI⁸):

  | | walls | solid layers | infill 0.4–1.2 mm | interior |
  | --- | --- | --- | --- | --- |
  | frame, reference | 0.740 (39%) | 0.670 (25%) | 0.659 (15%) | 0.638 (21%) |
  | frame, c09 | 0.888 (27%) | **0.911 (46%)** | 0.811 (11%) | 0.773 (15%) |
  | plate, reference | 0.581 (25%) | 0.534 (31%) | 0.523 (12%) | 0.506 (32%) |
  | plate, c09 | 0.695 (14%) | **0.731 (53%)** | 0.670 (10%) | 0.614 (23%) |
  | wrench, reference | 0.915 (71%) | 0.757 (19%) | 0.709 (8%) | 0.564 (2%) |
  | wrench, c09 | 0.987 (47%) | **0.954 (45%)** | 0.784 (6%) | 0.604 (2%) |

  The solid layers still hold about half of Σ FI⁸, 26–36% above the reference's. Layers share
  strain in this model, so a stiffer infill unloads every layer, the skins most (they are the
  stiffest); the reference's skins on our infill didn't help (screens after c04). Infill
  stiffness at equal material is the lever: where the material goes, and which way it runs.

### c15: density families normalised per cross-section — not kept (`runs/ribbons/c15_norm`)

- **Lacking.** Strength (c14's region split: the solid layers hold half of Σ FI⁸, and a stiffer
  infill unloads them). c05's density families placed material by stress but their lines
  drifted: the cumulative density runs from one edge of the bounds, so a cross-section that
  crosses more material shifts every line beyond it, and lines far from a dense spot tilt to make
  room for it.
- **Change.** `--density-norm station|segment` (`density_family(inside=, level=, norm=)`): each
  family's field is rescaled per cross-section so the part's cross-section holds as many lines as
  uniform ones ("segment": per stretch between holes). Lines gather across their family, not
  along it, and lines past the part stay straight. Screened with the stress field (c05) and with
  the gradient field step's own optimised density (`--density-from fields`, cached per part).
- **Screen** (equal material, no search; points / FI_agg / strokes):

  | start | frame (0/90) | plate (90/0) | wrench (0) |
  | --- | --- | --- | --- |
  | uniform lines | +67.0 / 0.876 / 317 | +110.9 / 0.744 / 349 | +53.8 / 1.002 / 301 |
  | stress γ 0.5 (c05) | +62.7 / 0.857 / 423 | +85.6 / 0.646 / 474 | +65.9 / 0.939 / 322 |
  | stress γ 0.5, per station | +58.1 / 0.853 / 438 | +109.8 / 0.670 / 423 | +51.4 / 1.080 / 280 |
  | stress γ 1, per station | +61.9 / 0.878 / 421 | **+113.9** / 0.625 / 413 | +49.8 / 0.948 / 280 |
  | field density | +12.3 / 0.772 / 828 | +16.2 / 0.820 / 779 | **+69.7** / 0.851 / 406 |
  | field density, per station | +40.5 / **0.754** / 622 | +97.6 / **0.580** / 468 | +54.8 / 1.022 / 301 |
  | field density, per segment | +42.8 / 0.777 / 569 | +56.7 / 0.706 / 673 | +46.2 / 1.087 / 364 |
  | field, mix 0.5, cap 0.6, per station | +74.0 / 0.761 / 428 | | |

  Normalised, the field density makes the plate's start stronger than any searched plate so far
  (FI_agg 0.580 against c09's 0.628), at 468 strokes. Segments lose (lines jump where a hole
  begins). On the wrench, whose one family runs along the handle, normalising removes the very
  placement that helped (more lines toward the loaded end): it suits alternating families.
- **Adversarial review.** The field density is fitted to seed 0's cases; its starts cost 80–300
  strokes the search may not win back (c05's density searches stalled); equal material is by the
  model's deposit accounting, and curvy lines lay more per line (the plate's normalised field
  family reaches 40% with the line count uniform lines reach 31% with). Guards: equal budget
  against the current path, strokes and terms reported, held-out seeds.
- **Result** (skin search, CMA-ES 1,500 then local 750):

  | | points | FI_agg | strokes | infill | strength + bulk + stiffness | continuity | interlayer | time |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | plate, field per station | +127.5 | **0.607** | 380 | 41.3% | **+62.0** | +59.4 | +8.8 | −1.3 |
  | plate, current (c09) | **+135.5** | 0.628 | 330 | 43.1% | +57.6 | **+67.4** | **+12.3** | +1.4 |
  | frame, field mix 0.5 per station | +99.5 | **0.775** | 363 | 38.2% | **+26.3** | +56.2 | +16.0 | +2.9 |
  | frame, current (c09) | **+110.2** | 0.796 | 345 | 40.1% | +22.9 | **+59.6** | **+22.8** | +4.6 |

  Stronger with less material on both parts, and still behind: the density layouts lose more to
  continuity, bonds and print time than they gain. CMA-ES stalled on both within 300–600
  evaluations (plate +114.5 from +108.8); local moves then added 13 (plate) and 3 (frame). Not
  kept. The strokes come from tilted lines: where the density across a family changes along it,
  its lines lean to follow (the frame's short cross-bar lines zigzag), and unevenly spaced ends
  merge worse than a regular zigzag (frame: 223 runs, 42 left unmerged per pair of templates,
  against 126 and 12 for uniform lines). That is c17's target.
- **Held-out seeds** (`runs/ribbons/c15_norm/heldout.log`; points, FI_agg):

  | layout | seed 0 | seed 1 | seed 2 | seed 3 |
  | --- | --- | --- | --- | --- |
  | plate, field per station | +127.5 (**0.607**) | +104.9 (0.741) | +99.2 (0.800) | +110.1 (0.692) |
  | plate, current (c09) | +135.5 (0.628) | **+115.1 (0.735)** | **+115.8 (0.741)** | **+121.6** (0.710) |
  | frame, field per station | +99.5 (**0.775**) | +90.8 (0.894) | +95.9 (0.824) | +96.7 (0.796) |
  | frame, current (c09) | +110.2 (0.796) | **+114.8 (0.787)** | **+113.9 (0.784)** | **+113.9 (0.777)** |

  The strength doesn't hold: on unseen cases the field-density layouts are weaker than the
  current bests (plate seed 2: FI_agg 0.800 against 0.741; frame seed 1: 0.894 against 0.787),
  while the uniform-start bests keep or improve theirs. The gradient field step fits its
  material to the twenty cases it was optimised on, and concentrating material by one sample
  of random loads leaves other samples' load paths thin. This is the risk the review named,
  and the reason held-out seeds guard every density change from here on.

### c16: longer connectors, re-routed merges — not kept (`runs/ribbons/c16_reach`)

- **Lacking.** Continuity: an infill layer of the bests still prints as 4–8 strokes besides its
  wall loops (frame 13.6 strokes per infill layer, 6 of them walls). Ends further apart than the
  4 mm connector reach never merge, and the single merge pass skips a pair whose route an
  earlier connector took.
- **Change.** `--link-reach MM` (`rltp.surfaces.layout.LINK_REACH_MM`) and `--link-passes N`: after
  the first pass, route the ends still open again around the connectors laid, until a pass adds
  none (`_merge(passes=...)`; one pass reproduces every earlier layout exactly).
- **Adversarial review.** A merge changes the print order, so travel between strokes and the
  bond timing move too; the bests are tuned to the current rule, so changing it re-scrambles
  their merges. Guard: re-lay c09's vertical bests (same warp and skins) under each rule and
  read the whole score, not the stroke count.
- **Result** (no search; points, strokes):

  | rule | frame | plate | wrench |
  | --- | --- | --- | --- |
  | 4 mm, one pass (current) | +110.2 (345) | **+135.5** (330) | **+122.0** (265) |
  | 6 mm | +110.7 (334) | +127.2 (320) | +94.8 (349)* |
  | 8 mm | **+111.7** (324) | +127.2 (320) | +94.8 (349)* |
  | 12 mm | **+111.7** (314) | +130.9 (309) | +94.5 (328)* |
  | 4 mm, 2–3 passes | +110.2 (345) | +135.5 (330) | +122.0 (265) |
  | 8 mm, 3 passes | +111.7 (324) | +127.2 (320) | +110.8 (286)* |

  \* the wrench's density family is calibrated through the layout, so a new merge rule
  recalibrates it and changes the start itself.

  Fewer strokes didn't mean more continuity points: on the plate at 6 mm, 10 fewer strokes
  came with 23% more travel (1,583 → 1,948 mm) and a print order that cost 5.8 interlayer
  points. The frame gains 1.5 at 8–12 mm. Extra passes find nothing on the tuned layouts, whose
  warps already avoid blocked routes. Not kept. The density starts of c15 are where merging
  fails most (below); the rule could be re-tested there, with search.

### c17: density smoothed along the lines — kept for the plate (`runs/ribbons/c17_smooth`)

- **Lacking.** c15's density layouts lost on continuity: lines lean wherever the density
  across their family changes along it, so short lines zigzag (the frame's cross-bar family)
  and unevenly spaced ends merge badly.
- **Change.** `--density-smooth MM`: smooth the field along each family's lines (Gaussian,
  `density_family(smooth_along=)`) before placing them, which bounds the lean and keeps the
  contrast across the lines. Also `--density-on first|second`: space one family by the density
  and keep the other straight at the same mean spacing.
- **Screens** (equal material, no search; points / FI_agg / strokes):

  | start | frame | plate | wrench |
  | --- | --- | --- | --- |
  | field per station, unsmoothed (c15) | +40.5 / 0.754 / 622 | +97.6 / 0.580 / 468 | – |
  | smoothed 1.5 / 2 mm | | +112.0 / 0.589 / 423, +105.4 / 0.590 / 477 | |
  | smoothed 3 mm | +63.3 / 0.784 / 481 | **+124.8** / 0.601 / 369 | +68.3 / 0.916 / 343 (unnormalised) |
  | smoothed 4 / 6 mm | +58.2 / 0.806 / 452 (6 mm) | +116.1 / 0.619 / 394 (4 mm) | +56.2 / 0.961 / 343 (6 mm) |
  | first family only, smoothed 3 mm | **+76.2** / 0.775 / 380 | | |
  | second family only | +61.8 / 0.821 / 451 | | |

  Smoothing trades a little strength for far fewer runs (frame 223 → 132 per pair of
  templates); stroke counts jump around between neighbouring settings (merges flip), so 3 mm
  is a reasonable middle, not a tuned optimum. On the frame, the density pays on its long 0°
  family only. The wrench gains nothing from smoothing.
- **Adversarial review.** Smoothing doesn't change what the field was fitted to, so c15's
  held-out weakness may remain; a strong start doesn't promise a strong search (c08). Guards:
  held-out seeds 1–3 against the current bests; equal budget.
- **Result** (skin search, CMA-ES 1,500 then local 750):

  | run | seed 0 | seed 1 | seed 2 | seed 3 |
  | --- | --- | --- | --- | --- |
  | plate: field per station, smoothed 3 mm | **+142.5** (0.597, 297 strokes) | **+118.6** (0.730) | **+121.1** (0.713) | **+128.9** (0.649) |
  | plate: current (c09) | +135.5 (0.628, 330) | +115.1 (0.735) | +115.8 (0.741) | +121.6 (0.710) |
  | frame: first family only, smoothed 3 mm | +100.3 (0.792, 364) | +104.9 (0.803) | +100.0 (0.830) | +102.5 (0.804) |
  | frame: current (c09) | **+110.2** (0.796, 345) | **+114.8** (0.787) | **+113.9** (0.784) | **+113.9** (0.777) |
  | wrench: field density, unnormalised | +117.1 (0.858, 238) | +119.0 (0.836) | +109.5 (0.912) | +122.8 (0.843) |
  | wrench: current (c09, stress γ 0.5) | **+122.0** (0.833, 265) | **+121.4** (0.819) | **+116.1** (0.870) | +121.9 (0.831) |

  The smoothed plate beats the current best on every seed, held-out ones included (+3.5 to
  +7.3), stronger on each and with 33 fewer strokes, so its gain isn't fitted to seed 0. Its
  CMA-ES phase found nothing better than the start in 1,500 evaluations; the local moves added
  all 17.7 points (strokes 369 → 297, interlayer +6.4 → +10.3). Kept for the plate. The frame and
  the wrench lose on every seed. The wrench's stress density (c05) holds on held-out seeds
  (FI_agg 0.819–0.870 against 0.833 on seed 0), so it stays; c13's two-start wrench (+126.1 on
  seed 0) averages +123.1 on held-out seeds against +119.8, though weaker on seed 1 (0.946).

### c18: density fitted to 100 load cases — kept for plate and wrench (`runs/ribbons/c18_cases`)

- **Lacking.** c15's held-out check: a density fitted to the twenty search cases puts material
  where those cases load the part, and other samples of random loads find the thin places.
  Smoothing (c17) rescued the plate, but the field itself is still fitted to one sample.
- **Change.** `--density-cases N`: fit the density field (the gradient field step's, or the
  stress consensus) to N random force + N moment cases of the same generator instead of the part's
  own 10 + 10, so it follows the load distribution rather than one draw from it; the search
  still scores the part's own cases. The 50 + 50 fields are cached per part (one-time cost:
  15 min for the plate, 9 for the frame).
- **Adversarial review.** A field fitted to more cases is less contrasty and may give back the
  strength that made density worth having; the search still fits seed 0 on top of it. Guards:
  held-out seeds 1–3, and the starts compared before any search.
- **Starts, no search** (plate, smoothed 3 mm, per station; FI_agg):

  | start | seed 0 | seed 1 | seed 2 | seed 3 |
  | --- | --- | --- | --- | --- |
  | uniform lines | 0.744 | 0.808 | 0.773 | 0.745 |
  | field fitted to the 20 search cases | **0.601** | 0.731 | 0.735 | 0.659 |
  | field fitted to 100 cases | 0.617 | **0.685** | **0.694** | **0.655** |

  On unseen cases the 100-case field is the strongest start, 6–15% below uniform lines.
- **Result** (skin search, CMA-ES 1,500 then local 750; points, FI_agg in brackets):

  | run | seed 0 | seed 1 | seed 2 | seed 3 | held-out mean |
  | --- | --- | --- | --- | --- | --- |
  | plate, 100-case field | +139.6 (0.617, 319 strokes) | **+122.1** (0.704) | +120.6 (0.723) | +128.2 (0.677) | **+123.6** |
  | plate, 20-case field (c17) | **+142.5** (0.597, 297) | +118.6 (0.730) | **+121.1** (0.713) | **+128.9** (0.649) | +122.9 |
  | plate, uniform (c09) | +135.5 (0.628, 330) | +115.1 (0.735) | +115.8 (0.741) | +121.6 (0.710) | +117.5 |
  | frame, 100-case field, 0° family | +112.3 (0.777, 318) | +113.2 (0.810) | +113.0 (0.784) | +115.3 (0.757) | +113.8 |
  | frame, 20-case field, 0° family (c17) | +100.3 (0.792, 364) | +104.9 (0.803) | +100.0 (0.830) | +102.5 (0.804) | +102.5 |
  | frame, uniform (c13 SPLIT) | +112.3 (0.805, 322) | **+119.4** (0.781) | **+119.3** (0.779) | **+116.0** (0.808) | **+118.2** |
  | frame, uniform (c09) | +110.2 (0.796, 345) | +114.8 (0.787) | +113.9 (0.784) | +113.9 (0.777) | +114.2 |
  | wrench, stress γ 0.5 on 100 cases | **+124.9** (0.858, 217) | **+126.0** (0.838) | +116.3 (0.913) | **+134.3** (0.810) | **+125.5** |
  | wrench, stress γ 0.5 on 20 (c09) | +122.0 (0.833, 265) | +121.4 (0.819) | +116.1 (0.870) | +121.9 (0.831) | +119.8 |
  | wrench, the same, two starts (c13 MULTI) | +126.1 (0.850, 193) | +117.0 (0.946) | +124.0 (0.871) | +128.4 (0.878) | +123.1 |

  Fitting to 100 cases removes most of the fitting: the frame's density layout gains 11 points
  on held-out seeds over the 20-case one and draws level with the uniform pipeline (c09), though
  not with c13's best uniform run. The plate is level with c17 on unseen loads (its smoothing
  already did most of the work), and the wrench's 100-case run is the best wrench on held-out
  seeds. Kept as the density setting for plate and wrench (`--density-cases 50`); the frame
  stays uniform. Strength and points disagree in places (wrench seed 2: FI_agg 0.913 against
  0.870, yet more points, from 48 fewer strokes): the gains here are as much continuity as
  strength.

### c19: replicates of the kept settings — both hold (`runs/ribbons/c19_replicate`)

- **Lacking (process).** c17 and c18 each rest on one search seed per part, and c13 measured
  about 5 points of run-to-run noise.
- **Change.** Rerun each kept setting with search seed 1: the plate's smoothed field density
  (c17's arm) and the wrench's 100-case stress density (c18), then held-out seeds 1–3.
- **Adversarial review.** A second seed could land below the old bests, which would make the
  "kept" calls noise. A replicate that lands above them would say little about an average
  gain either; two seeds per setting is the least that separates them.
- **Result** (points; FI_agg; held-out mean over seeds 1–3):

  | setting | search seed 0 | search seed 1 | held-out, seed 0 / seed 1 | old best, held-out |
  | --- | --- | --- | --- | --- |
  | plate, field per station, smoothed 3 mm | +142.5 (0.597, 297 strokes) | +142.8 (0.601, 322) | +122.9 / +121.1 | +117.5 (c09) |
  | wrench, stress γ 0.5 on 100 cases | +124.9 (0.858, 217) | **+128.3** (0.861, 217) | +125.5 / **+127.3** | +119.8 (c09) |

  Both settings repeat within 3.4 points and beat the old bests on unseen loads with either
  seed: plate +3.6 to +5.4, wrench +5.7 to +7.5. On unseen loads the plate's 20-case field gains
  through continuity (its held-out strength matches uniform lines': FI_agg 0.741 / 0.750 / 0.661
  against 0.735 / 0.741 / 0.710), while c18's 100-case field is also stronger there
  (0.704 / 0.723 / 0.677): the recommended plate setting uses the 100-case field.

### c20: final settings, audited the same way for every part (`runs/ribbons/c20_final`)

- **Lacking (process).** Each cycle was judged on its own table; the reference had held-out
  scores for seeds 1–2 only on frame and wrench. The end result needs one footing: each part's
  kept setting and the reference, on the search's cases and on unseen ones.
- **Change.** No new mechanism. Scored the reference's layouts on held-out seeds 1–3
  (`heldout_reference.log`), took each part's kept setting (frame: uniform cross-ply; plate:
  c18's 100-case field density; wrench: c18/c19's 100-case stress density), looked at their
  layers (`layers.png`: long merged serpentines, no stray fragments; the plate's lines gather
  between the holes), and ran the full test suite (178 passed).
- **Adversarial review of the whole result.** (1) The lead comes mostly from continuity and print
  time, so it depends on the score's weights: with strength weighted above them, the
  reference's strength would count for more. (2) The skin search's gain rides on the bond-timing
  model, whose τ = 30 s is a placeholder. (3) "40% infill" is by the grid model's deposit count,
  which credits curvy lines with more material per line. (4) Single runs vary by about 5 points.
  Mitigations: strength (FI_agg) reported beside every score, held-out seeds for every kept
  change, replicates for the kept density settings (c19); τ and the deposit model need
  calibration before absolute claims.
- **Result** (points; FI_agg; held-out = mean over seeds 1–3):

  | part | reference | reference, held-out | ribbons, final | ribbons, held-out |
  | --- | --- | --- | --- | --- |
  | frame | +61.0 (0.633, 940 strokes) | +46.4 (0.696) | **+112.3** (0.805, 322) | **+118.2** (0.789) |
  | plate | +19.5 (0.491, 1629) | −18.3 (0.654) | **+139.6** (0.617, 319) | **+123.6** (0.701) |
  | wrench | +73.1 (0.733, 727) | +71.2 (0.729) | **+124.9 / +128.3** (0.858 / 0.861, 217) | **+125.5 / +127.3** (0.854 / 0.860) |

  Against c00 the ribbons gained 28–44 points on the search's cases. Against the reference they
  lead by 51–120 points there and 54–142 on unseen loads, through strokes (a third to a fifth),
  print time (15–23% shorter) and, on frame and plate, bonding; the reference stays 7–15%
  stronger on unseen loads.
  Of the twenty cycles, six changes were kept (c02 skin search, c03 alternating families, c05
  density families, c07 local refinement, c17 smoothing along the lines, c18 density fitted to
  100 cases), two were process checks that held (c10, c19), and the rest are
  recorded with why they lost.
