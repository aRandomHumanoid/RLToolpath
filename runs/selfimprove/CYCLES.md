# Self-improvement cycles: working notes (summary in README.md)

Abbreviations: arm names are each cycle's --arm (A plain annealing, G guide, S stiffness, B bulk,
K gradient start, Z ratio Z index, T hotter); al2 = aligned share (FI^2-weighted); eff2/eff8 =
orientation efficiency at p = 2 / 8 (rltp/eval/adapt.py); obj = objective in points.

Setup: 5 parts (bracket, link, hook, frame, wrench), configs/printer.yaml (0.1 mm cells, 2 walls,
3 solid layers, 40% infill), FEA 0.4 mm elements (coarsen 4), 21 load cases (primary + 10 edge
force + 10 edge moment, seed 0), weights 0.5/1/1 (+Z 0.3, infill 1, cover 1 defaults), SA 3000
evals, seed 0. Objective = weighted % change vs the part's slicer start (README convention).

## Cycle 0 (baseline)
A: --start slicer (plain SA). K: --start gradient --stress-bias 0.5 --strength-first 3.
- A: objective +40..+51; FI_agg change -0.3..+0.3%; per-case within +-1.4%.
  dir changed 4-8.5% of bead cells, fill 7-9.5%; eff2 0.87->0.87, aligned2 0.40->0.40 (no move toward loads).
- ops: strength sums per op ~0.0-0.1 points; region relays accepted for continuity/Z.
- where sum FI^8 sits (start, band 1 mm): bracket walls near own regions 70.8%, walls far 20.2%,
  beads 9%; hook 81.4 / 17.4 / 1.1%. FI^2: beads 54% / 36%.
  => the p=8 objective is ~all locked walls at clamp/load-edge singularities: toolpath
  directions are nearly invisible to it.
- Perf: 21 cases, fine FEA 2.07 s/eval (bracket) -> coarse 0.4 mm + fewer systems + post opt 0.22 s.

## Cycle 1: --clamp-band 1.0 + --guide 0.5

### Cycle 0 full A results (objective, FI%, dirchg, eff2 0->1, aligned2 0->1)
bracket +40.3, -0.0%, 4.0%, 0.868->0.869, 0.400->0.397
frame   +35.1, -0.2%, 3.4%, 0.896->0.895, 0.313->0.311
hook    +44.2, -0.3%, 8.5%, 0.841->0.837, 0.280->0.272
link    +50.9, +0.3%, 4.5%, 0.862->0.856, 0.265->0.259
wrench  +57.4, -0.2%, 18.5%, 0.847->0.827, 0.287->0.271   <- turned 18% of beads, got LESS aligned
hook K: start +33.9 (gradient start aligned2 0.473), final +52.6, aligned 0.473->0.458 (SA erodes it)
- BUG found from K results: with several load cases, every start (stress, each gradient start's
  final part) re-measured case scales on its own layout -> its FI_agg was 1 by construction;
  strength not comparable across starts/arms; gradient info fi_agg meaningless. Fixed: refs
  (case scales, stiffness reference) from the slicer layout, shared by every start.
  K results of cycle 0 are therefore invalid for strength; rerun at the end.
- K field step with 21 cases: ~19-21 min per part (build_s 1118-1271 s) with 10 procs running.

### Cycle 1 results (G1: band 1 + guide 0.5), 4 of 5 parts (frame pending)
bracket +42.7, FI -1.9%, dir 9.5%, eff2 .853->.872, eff8 .385->.428, al2 .348->.377
hook    +40.0, FI -4.2%, dir 17.3%, eff2 .862->.862, eff8 .451->.486, al2 .294->.287
link    +54.2, FI -1.0%, dir 6.1%, eff2 .863->.867, eff8 .467->.475, al2 .260->.286
wrench  +61.9, FI -4.8%, dir 18.7%, eff2 .839->.833, eff8 .311->.347, al2 .304->.295
=> strength now moves (1-5% vs ~0), peaks adapt (eff8 up everywhere); bulk alignment (al2) still flat.
   Guided relays: ~10x the strength per op of cycle 0; many strength-positive guided relays still
   rejected for continuity/time cost (window borders cut lines).
## Cycle 2: + stiffness term weight 0.5 (S05) [started concurrently with cycle 1's tail]
frame   +41.9, FI -2.9%, dir 8.8%, eff2 .894->.897, eff8 .379->.395, al2 .303->.319
Cycle 1 means: obj 48.1 (c0 45.6), FI -3.0% (c0 -0.1%), dirchg 12.1% (c0 7.7%)
## Cycle 3: S05 + --align 0.5 (align_region op: re-lay connected area sharing best direction)

### Cycle 2 (S05: G1 + stiffness weight 0.5), partial
bracket +41.6, FI -2.2%, dir 7.9%, eff2 .853->.868, al2 .348->.369, compliance -0.5%
hook    +38.2, FI -4.8%, dir 14.3%, eff2 .862->.868, al2 .294->.305, compliance -1.6%
link    +58.3, FI -2.0%, dir 8.8%, eff2 .863->.873, al2 .260->.293, compliance -0.03%
=> compliance barely moves: E1/E2 = 3300/2600 = 1.27 (mild stiffness anisotropy) vs strength
   Xt/Yt = 1.8. Stiffness is a weak lever with these material constants; strength (band) is the lever.
- Field model got the stiffness term too (gradient checked by FD); field evaluate ~3 s with 21 cases.
frame   +43.4, FI -3.3%, dir 7.8%, eff2 .894->.897, al2 .303->.319
wrench  +65.6, FI -5.7%, dir 14.0%, eff2 .839->.839, al2 .304->.297, worst case -0.3%
Cycle 2 means: obj 49.4 (c1 48.1), FI -3.6% (c1 -3.0%). Keep stiffness 0.5 (cheap, slightly better).
## Cycle 4: KG = gradient start (shared refs fix) + band + guide 0.5 + D's guidance + stiff 0.5

### Cycle 3 (AL: S05 + align 0.5): REJECTED
bracket +37.3 (S05 41.6), link +45.8 (58.3), hook +40.1 (38.2), frame +31.7 (43.4), wrench +55.7 (65.6)
FI similar (-1.6..-5.2%). align_region: ~700 proposals/run (25% of budget), accepted 3-6%;
rejected ones carried +5..6 points of strength in sum but cost more in time/cover. Wastes evals.
Keep op (off by default).

### Analysis after cycle 3 (cycle 2 histories)
- guided accepted: strength +0.9..+2.7 pts per run vs unguided +0.1..+0.3 (guide = ~90% of strength gains);
  guided acceptance 8-11% vs 15-18%.
- rejected guided: strength +3..+7 pts total, but Z (inter) -1300..-1700 and cont -600..-1100.
- Z losses: SKIN layers (0-2,7-9), spiky (median -0.05 pt/edit, worst 5% = 44-70% of loss), same for
  unguided skin edits. Skin guided relays rarely change strength (median 0).
- infill layers (3-6): strength comes from here (+1.7-2.0 pts/layer, bracket); blocker = continuity
  (avg relay +0.012 pt strength vs -0.5 pt cont).
- ROOT: even with the band, ~80% of sum FI^8 is in the 2 outline walls (already along the edges where
  stress peaks). The part is wall-limited under p=8: interior direction barely moves the peak.
  => need an objective that values the bulk if the interior is to follow the loads: a p=2 "bulk" term.

### Cycle 2b (S15: stiffness weight 1.5)
FI -2.3/-1.8/-3.9/-3.8/-6.6 (mean -3.7 vs S05 -3.6); al2 final mean 0.330 vs 0.317 (S05);
compliance -0.3..-1.9% either way. Heavier stiffness = slightly more alignment, same strength.
## Cycle 6: bulk term (p=2 FI aggregate) weight 0.5 + guide uses the weighted cost (FI^8 and FI^2)
- Field model: bulk term (J2 = sum FI^2 with case_scale2) by a second adjoint; FD-checked. Gradient
  start optimises w_strength/q ln J + w_stiff ln stiff + w_bulk/2 ln J2 when the weights have them.

### Cycle 4 (KG) gradient-start step, FI_agg vs slicer start = 1 (shared refs)
gradient 0.886 / 0.985 / 0.955 / 0.964 / 0.939 vs stress start 1.012 / 0.978 / 1.424 / 1.255 / 1.044
=> with 21 random cases the stress start (primary case only) is up to 42% WORSE than slicer;
   the gradient start (all cases) is 1.5-11% better. Field step 20-44 min/part under contention.
- INCIDENT: c04 (KG) crashed after its 20-44 min field step: guide.py (edited for the bulk term) was
  imported lazily at SA start against the old part.py. Lost ~1 h. From now on every cycle runs from a
  frozen code snapshot (~/run_cycle.sh NAME ...). c04 relaunched from snapshot.

### Cycle 5 (G8: guide 0.8, else S05) vs cycle 2 (guide 0.5)
obj 45.4/56.0/37.9/52.6/56.8 (mean 49.7 vs 49.4); FI mean -3.7 vs -3.6; dirchg 12.5% vs 10.6%;
al2 final mean 0.336 vs 0.317; eff2 final 0.877 vs 0.869. => ADOPT guide 0.8.
## Cycle 7: budget test, G8 settings at 6000 evals (bracket, hook, wrench)

### Cycle 6 (B05: bulk 0.5, guide 0.5) vs cycle 2
FI mean -3.2 vs -3.6; al2 final 0.338 vs 0.317; eff2 0.874 vs 0.869; bulk FI2 -0.7..-2.2%.
=> bulk 0.5 ~ same alignment gain as guide 0.8, slightly less peak gain. 1% bulk = 0.5 pt vs
   ~0.5 pt continuity per relay: weight too low to pay for much re-orientation.
## Cycle 8: guide 0.8 + bulk 2.0 (+ stiff 0.5): what if the bulk is valued?

### Cycle 4 (KG: gradient start + band + guide 0.5 + D's guidance + stiff 0.5), 3 of 5, scored vs slicer start
bracket +54.6 (G8 +45.4), FI vs slicer -4.1% (G8 -2.2%); start obj +28.5
hook    +68.9 (G8 +37.9), FI -6.3% (-4.5%); start +40.9
link    +71.7 (G8 +56.0), FI -11.5% (-1.9%); start +15.9
=> gradient start wins every term (strength, stiffness, continuity, Z), by 9-31 points. Bulk alignment
   (al2) is not what it optimises: hook K al2 0.11->0.22 (G8 0.33), bracket 0.48 (G8 0.39).
## Cycle 9: KB = gradient start + guide 0.8 + bulk 0.5 (field model too) + stiff 0.5

### Cycle 8 (GB2: guide 0.8 + bulk 2.0) vs cycle 5 (G8)
FI mean -4.0 vs -3.7 (peaks better too); al2 final 0.362 vs 0.336; eff2 0.883 vs 0.877; dirchg 15.1 vs 12.5%;
bulk FI2 -1.5..-2.7%; print time +1.5..+4.2% (G8 +0.3..+3.5%). => bulk ~2 is the "adapt" dial; ADOPT bulk 2.
## Cycle 10: GB2 + temp0 0.3 (more exploration)
- direction maps hook: c05 (G8) skin al 44->39%, infill 21->38%; c08 (GB2) skin 47->47%, infill 21->40%.
  Skin best-direction map is a curve-following patchwork (0/45/90/135 every few mm) -> contour-
  following skin beads would suit curved parts; not attempted (new layout feature).

### Cycle 4 complete (KG vs G8, scored vs slicer start)
obj 54.6/68.9/71.7/50.8/66.3 (mean 62.5) vs 45.4/37.9/56.0/52.6/56.8 (49.7)
FI vs slicer -4.1/-6.3/-11.5/-7.7/-9.9 (mean -7.9%) vs -2.2/-4.5/-1.9/-4.1/-5.9 (-3.7%)
worst case better on link/frame/wrench; bulk alignment mixed (hook al2 0.22 vs 0.33).
## Cycle 11: KB2 = gradient start + guide 0.8 + bulk 2 (field model too) + stiff 0.5
- Cycle 12 (speed): field model wall FI only for the directions each wall group holds: evaluate
  3.0 s -> 1.6 s (bracket, 21 cases); identical values (FD tests pass). Field step ~1.8x faster.

### Cycle 7 (G8L: 6000 evals) vs cycle 5 (3000)
bracket same (+45.5 vs +45.4, stops early); hook +46.2 vs +37.9, wrench +71.1 vs +56.8 (continuity:
strokes 179 vs 196, 149 vs 167); FI -4.0 vs -4.5, -6.1 vs -5.9; al2 .353 vs .334, .301 vs .328.
=> more budget buys continuity, not adaptation. Budget is not the limit.
- Z spikes: interlayer FI is quadratic in stress ((s/qZt)^2+...), so at p=8 it acts like p=16 and a
  stacked path-end pair (q x0.7) weighs 0.7^-16 = 300x. Option --z-fi ratio (sqrt: inverse strength
  ratio, linear in load like in-plane). Same 60 guided skin relays (c08 hook final): Z pts mean
  -0.376 -> -0.115, worst -8.3 -> -2.5, rank corr 0.93. Mostly a halving of Z's scale (-ln sqrt =
  1/2 -ln) + modestly less spiky. Kept as option (default quadratic).
## Cycle 13: GB2 + --z-fi ratio (bracket, hook, wrench)

### Cycle 10 (GB2T: temp0 0.3) vs cycle 8 (GB2, temp0 0.1)
obj mean 46.2 vs 49.8; FI -4.2 vs -4.0; al2 0.376 vs 0.362; dirchg 18.1 vs 15.1%; strokes +10..16.
=> hotter = a bit more adaptation, worse continuity at equal budget. Not adopted.

### Cycles 9 (KB) / 11 (KB2) partial, all scored on GB2 weights vs slicer start (scripts/compare.py)
3-part means (bracket, hook, link): G8 +46.4 FI -2.9% al .341 | GB2 +48.2 -3.1% bulk -1.8% al .363 |
KG +65.1 -7.3% al .352 | KB +71.6 -8.2% bulk -4.5% al .369.
KB2 hook: +70.2, FI -7.1%, bulk -4.5%, aligned 0.424 (KB 0.242, KG 0.221): bulk in the field step
aligns the whole part, not only the peaks. => final config: K + guide 0.8 + band + bulk (0.5-2) + stiff 0.5.

### Cycle 13 (GB2Z: Z ratio form) vs cycle 8 (GB2)
objective not comparable (Z on another scale). FI -4.0 vs -4.1%, bulk -1.9 both, aligned 0.365 vs
0.362, strokes fewer (link 133 vs 144, frame 254 vs 266, wrench 168 vs 183). => frees a little
continuity, no adaptation change. Option only.
- Added --adapt preset (gradient start, band 1, guide 0.8, stress-bias 0.5, strength-first 3,
  stiffness 0.5, bulk 2; each unless given). Tested.
## Cycle 14: seed replicate of GB2 (seed 1) -> noise level of single-seed comparisons

### Cycle 11 complete (KB2 on bracket, hook, wrench; GB2 weights, vs slicer start)
GB2 +50.8 FI -4.4 bulk -2.2 worst +0.8 al .383 | KG +63.3 -6.8 - +1.4 .328 | KB +71.0 -7.8 -4.2 +0.7 .363 |
KB2 +69.3 -8.4 -5.1 +0.1 .442  => KB2 best on strength, bulk, worst case and alignment; -1.7 obj (strokes).
### Cycle 9 complete (5 parts, GB2 weights): KB +68.3, FI -8.8, bulk -4.3, worst -0.1, aligned 0.362
(GB2 +49.8 / -4.1 / -1.9 / +0.5 / .362; KG +62.5 / -7.9 / - / +0.1 / .332)

### Cycle 14 (GB2 seed 1 vs seed 0)
per-part objective diff +1.0/+8.0/-0.9/-0.9/-0.7 (mean 51.1 vs 49.8); FI per part within 0.9 pp
(means -4.0 vs -4.1); aligned per part +-0.02 (means .360 vs .362).
=> noise: objective ~8 pts per part, ~1-2 pts on 5-part means; FI ~0.2 pp on means; aligned ~0.01.
Real: gradient start (+12.8, -4.2 pp FI), KB2 alignment (+0.08), align-op loss (-8), temp loss (-3.6).
Marginal: guide 0.8 (+0.019 al), bulk 2 on slicer (+0.026 al). Within noise: stiffness objective effect.

### Cycle 15 (final: --adapt on all 5 parts, viewers saved), rescored under the final objective
(scripts/rescore.py, rows in final_rescore.jsonl). 5-part means vs slicer start (aligned 0.302):
plain annealing (c00 A)  +45.7, FI -0.5%, bulk +0.1%, worst +2.0%, aligned 0.293
guided, slicer (c08 GB2) +49.8, FI -4.1%, bulk -1.9%, worst +0.5%, aligned 0.362
--adapt (c15)            +69.7, FI -9.1%, bulk -4.9%, worst -1.1%, aligned 0.412
Gradient step alone: FI_agg 0.87-0.99 of the slicer start (stress start 0.98-1.42 over 21 cases).
Per part ~15-28 min field step + 15-23 min annealing with 5-13 runs on 16 cores.
History logs are gzipped (history.jsonl.gz); c15 runs keep viewer.html, part.gcode, state.npz.

## Cycle 16: one wall (configs/printer.yaml walls 2 -> 1), plain (A) and --adapt on all 5 parts
Rescored with scripts/rescore.py (walls_p8 = share of sum FI^8 in walls, band as scored).
5-part means, each against its own slicer start (2 walls -> 1 wall):
walls' share at slicer start 0.376 -> 0.307 (after --adapt 0.60 -> 0.48)
plain annealing FI -0.5 -> -0.3%   (still no adaptation)
--adapt FI -9.1 -> -11.3%, bulk -4.9 -> -6.7%, worst -1.1 -> -2.4%, aligned 0.412 -> 0.389, obj +69.7 -> +68.3
per part --adapt FI: bracket -7.8/-10.3, link -11.3/-15.6, hook -7.1/-8.1, frame -8.7/-10.5, wrench -10.4/-12.2
Absolute (1-wall layouts scored with the 2-wall slicer start's case scales):
1-wall slicer start FI +5.9%, worst +15.6%; 1-wall --adapt FI -5.9%, worst +5.1% (2-wall --adapt -9.1 / -1.1).
Print time 424 -> 375 s at the slicer start (the second ring becomes 40% infill: less material).
=> interior directions matter more (+2.2 pp FI gain, well above the 0.2-0.3 pp noise), but the 1-wall
part is weaker in absolute terms at 40% infill. Kept as the default (user request).
Note: before this cycle the README said ~80% of sum FI^8 sat in the walls; that came from fi_where.py
with case scales measured without the band. As scored (band in the scales) it's 24-52% at the start.

### Floating infill lines (diagnosis, scripts/analysis/frag.py on c15)
Infill-layer path ends with no wall or bead within 2 bead widths ahead:
slicer start 0-10%, gradient start 24-71%, --adapt final 28-50% (strokes per infill layer roughly halve
in annealing). Cause: fine_layout lays gradient-start lines only where fill_map > the line's dither
rank, and clips them at direction-area boundaries. Unpenalised: laminate FEA = perfect interlayer bond;
continuity charges strokes/travel, not anchoring.

## Cycle 17: anchored infill lines + free-end cost, --adapt (ADAPT) and --adapt --free-end-cost 0 (NOCOST)
Free end = non-wall path end with no other bead's node within kb cells (beads.is_anchored).
Code: anchor_end/anchor_ends (straight on, or a 90-degree T-joint to the next bead, <= 40 cells) in
fine_layout, _relay/refill/flip/align, stack; `anchor` op; ScoreConfig.c_free_end = 1 in continuity.
Probe first (scripts/analysis/probe_layouts.py, fixed bracket/link fields, gradient start FI_agg):
cut lines unanchored 0.899/0.840, whole runs 0.949/0.943 (+5.6%/+12.3%), cut + T-joints 0.891/0.841
(-0.9%/+0.1%, 0 free ends) => default infill_lines: cut, `--infill-lines whole` kept as an option.
A first c17 launch (whole runs) died with the WSL service; rerun with the final code.
Rescored with scripts/rescore.py under c17 ADAPT settings (free ends charged) and NOCOST (not), 5-part means:
                         c16      c17 ADAPT   c17 NOCOST
free ends, grad start    56       0           0
free ends, final         37.6     0.2         4.0
FI_agg                   -11.3%   -11.3%      -11.4%
bulk                     -6.7%    -6.6%       -6.6%
worst case               -2.4%    -3.7%       -3.4%
aligned                  0.389    0.381       0.382
obj (not charged)        +68.3    +69.3       +69.0
obj (charged)            +58.2    +69.2       +67.9
print time               386 s    387 s       388 s
per part FI c16 -> c17: bracket -10.3/-9.7, link -15.6/-15.4, hook -8.1/-8.9, frame -10.5/-10.4, wrench -12.2/-12.0
free ends final c16/ADAPT/NOCOST: bracket 34/0/3, link 28/0/4, hook 35/0/4, frame 52/1/4, wrench 39/0/5
Run time ADAPT = NOCOST (61-78 min, 10 runs on 16 cores): counting free ends is cheap. anchor op seldom
proposed (few free ends left). Figure: docs/img/anchored_ends.png (scripts/analysis/ends_fig.py).
=> kept: anchoring everywhere + c_free_end 1 (default). Strength unchanged within noise, floating lines gone.

### Budget check: hook at --evals 20000 (runs/selfimprove/c17_20k, c17 ADAPT settings + --max-passes 100)
Same seed and start as c17 hook ADAPT: the first 9 passes are identical, so it continues the 3,000 run.
Stopped itself at 5,496 evals ("converged": a greedy pass gained < eps 1e-3); never reaches 20,000.
Rescored under c17 hook ADAPT settings, 3,000 -> 5,496 evals: objective +64.9 -> +71.4 (cont +28.9 -> +34.2,
inter +22.9 -> +24.3), strokes 174 -> 166; FI_agg -8.9% -> -8.7%, bulk -6.2/-6.2, worst +1.5/+1.6,
aligned 0.428/0.426, print time 345.9 s both. FI_agg (vs annealing start) bottoms at pass 8 (-5.0%), then
drifts to -4.6% as greedy passes trade strength for continuity. Run alone: 25 min (field step 9 min).
=> a bigger budget only buys continuity/Z after ~3,000 evals; using more would need --reheat or a lower eps.

### Reheat check: hook at --evals 20000 --reheat 0.5 / 1 (runs/selfimprove/c17_reheat, + --max-passes 200)
With --reheat the "converged" stop is skipped, so both runs use all 20,000 evals (75 min each, 2 jobs at once).
Passes 0-33 are the same in both runs and continue the 5,496 run: its "converged" stop was one quiet pass;
greedy passes kept gaining for 16 more passes (to 9,855 evals) and FI_agg (vs annealing start) went from
-4.6% back to -5.4%. Reheats (2 greedy passes < 0.005) then fired every ~1,700-3,000 evals:
REHEAT05 5x (from 9,855), REHEAT1 6x. Full heat (temp 0.1) drops the total each time and the 4 cooling
passes only win it back; half heat keeps more.
Rescored under c17 hook ADAPT settings (vs slicer start):
                3,000    5,496    REHEAT05   REHEAT1
objective       +64.9    +71.4    +93.4      +85.1
  cont          +28.9    +34.2    +54.5      +50.3
  inter         +22.9    +24.3    +26.3      +24.4
  strength      +4.6     +4.5     +5.0       +5.0
FI_agg          -8.9%    -8.7%    -9.5%      -9.5%
bulk            -6.2%    -6.2%    -6.3%      -6.3%
worst case      +1.5%    +1.6%    +0.7%      +1.0%
aligned         0.428    0.426    0.444      0.437
print time      345.9 s  345.9 s  347.7 s    351.4 s
strokes         174      166      116        125
=> reheating found no strength: FI_agg is flat within 0.1 pt after the first reheat. The extra 0.6 pt
came from greedy passes that the single-pass "converged" test had cut off. The rest of the 20,000 budget
buys continuity (strokes 174 -> 116) at +0.5% print time. Option: require two quiet passes before
"converged" (as reheat_after does) instead of one.

## Setup change (Nov 15): parts 5 mm thick (25 layers), solid_layers 2
examples/make_parts.py T = 5 (bracket and link stretched from their 2 mm STLs by `thicken`), each
primary load resized so the slicer start's FI_agg is ~1: bracket 25 -> 33.1 N, link 120 -> 187.8,
hook 61.2 -> 75.9, frame 120.3 -> 171.9, wrench 24.1 -> 32.1. 2 mm files kept in examples/2mm/
(rescore.py/direction_maps build from examples/, so runs above need those files to be rescored).
21 of 25 layers now take infill (was 4 of 10).
Probe (runs/selfimprove/probe5mm): hook, --adapt, 1,000 evals. Build + gradient start ~10 min,
0.27 s/eval (2 mm: ~0.17), ~790 evals per pass (2 mm: ~315), so 3,000 evals = 3.8 passes.
Gradient start: field model rounded 0.719 vs evaluator 0.962 (2 mm hook: 0.921 vs 0.958); stress start
1.47. Rescored vs slicer start:  grad  FI -3.8%, worst +26.9%, bulk -11.0%;
after 1,000 evals FI -24.0%, worst -5.2%, bulk -14.5%, time +4.3%, strokes 759 -> 649, 3 free ends.
Wall share of sum FI^8 0.176 -> 0.822 (2 mm hook c17: 0.427 -> 0.558): after one pass the locked wall
ring holds most of the peak stress.

### c18_5mm: the five parts at 5 mm, --adapt, 8,000 evals (runs/selfimprove/c18_5mm)
Code as of the 5 mm switch, so loads as before the load-agnostic change: each part's loads.yaml
primary + 10 force + 10 moment random cases, edges drawn uniformly. 5 jobs at once, --workers 1:
88-122 min each (build + gradient start 10-25 min); peak memory 14.3 GB used.
The suite table's FI%/case% compare with each run's gradient start. Rescored vs the slicer start
(scripts/rescore.py):
            FI_agg   bulk    worst case  aligned       wall8          print time         strokes   obj
bracket     -26.4%  -10.8%   -14.6%      0.383->0.517  0.207->0.624   586->622 s (+6.1%)  513->413  +76.6
link        -27.1%  -12.2%   -11.7%      0.296->0.352  0.137->0.497   601->677 s (+12.7%) 368->494  +75.5
hook        -26.3%  -15.5%    -7.9%      0.270->0.417  0.188->0.854   679->712 s (+4.9%)  759->525  +97.3
frame       -33.4%  -10.9%   -14.7%      0.301->0.343  0.073->0.528   850->968 s (+13.8%) 696->823  +76.9
wrench      -32.1%  -14.7%   -11.4%      0.385->0.448  0.189->0.807   804->888 s (+10.5%) 700->582  +104.0
Gradient starts alone: FI +7.3/-15.1/-3.8/-6.5/-12.0%, worst case +50.0/+5.7/+26.9/+15.6/+26.0%
(the field model lets weak cases rise; the anneal then brings every case below the slicer).
For scale, c17 at 2 mm (3,000 evals, other loads): FI -9 to -15%, worst case about -4%.
Wall-like structure (scripts/analysis/wallband.py): bands by distance from the outline, over every
layer (meant to be the sparse layers only; it counts the 4 solid layers too, see c20 follow-ups).
cover = share of band area under beads (centerline x bead width, overcounts ~10%), par = share of
bead length within 22.5 deg of the local edge, fi8 = share of the layers' sum FI^8.
  band 0.4-1.2 mm (the bead rows next to the wall), cover slicer -> grad start -> final:
    bracket 0.52->0.63->0.61, link 0.50->0.70->0.66, hook 0.44->0.40->0.44, frame 0.44->0.52->0.52,
    wrench 0.46->0.45->0.49
  par there, slicer -> final: bracket 0.57->0.71, link 0.65->0.68, hook 0.28->0.45, frame 0.53->0.57,
    wrench 0.60->0.51
  wall ring's fi8, slicer -> final: 0.20->0.62, 0.14->0.49, 0.17->0.79, 0.07->0.52, 0.18->0.79
=> 5 mm parts gain ~3x what 2 mm did, every load case improves, at +5-14% print time. The peaks end
in the locked wall ring. Only bracket and link grew a denser band next to the wall, and that came
from the gradient start; the anneal kept it but never added to it.

### Why no wall-like band grows (c18_5mm layouts; scripts/analysis/probe_walls, probe_dirs, probe_bands, probe_grad.py)
1. The anneal can't move density. On the 0.1 mm grid the add/fill moves are off (ops.action_mask, kb > 1);
   refills (85% of tried moves) re-lay a window at its own infill share with evenly spaced straight
   lines (shell.line_picks, one density per window), so they even out any density gradient inside
   their 1.6-6.4 mm windows; only clear_w, extend/retract, shift, stack_w, anchor change material,
   and none adds lines beside a wall. Only the gradient start's density field can make a band.
2. The score barely asks for one. Field-model gradient (fields read off each layout; model FI_agg
   within 1-6% of the evaluator), weighted as the gradient start weights it: moving 1% of the
   optional material into the 0.4-1.2 mm band (vs spreading it evenly), in points:
                 slicer start           final layout          (+1% material everywhere: +1.5..+5)
     old loads   -0.01 -0.95 -0.65 -0.74 +1.21   +0.36 +0.01 +0.41 -0.10 +0.47   (bracket link hook frame wrench)
     new loads   +0.90 -1.16 -0.26 -0.03 -0.44   +0.11 -0.22 -0.04 +0.65 +0.20
   Stiffness always favours the band (+0.1..+0.4); strength goes with where the peaks are: at the
   start in misaligned infill 1.2-2.4 mm in (material there helps more than at the wall), after
   realignment in the walls (band helps a little on some parts). The added beads' own FI^8 terms
   are ~0 in this comparison (+-0.2), so the weakest-link sum is not what holds it back.
3. Second ring at equal material (2 walls, infill 24-28% vs 1 wall, 40%), slicer layouts:
   FI_agg bracket/link/hook/frame/wrench old loads -0.1/+4.6/+21.9/+11.3/+16.6%, new +0.1/+5.4/+0.1/
   +13.4/+3.5%. The wall and ring bands' share of sum FI^8 falls (hook 0.17 -> 0.06, 0.08 -> 0.01) but
   the thinned infill 0.8-2.4 mm in rises 2-7x: on these narrow parts most of the section is "near
   the edge" (wall + 0.4-2.4 mm bands = 62-87% of the area) and still carries load.
4. Scoring each cell at its bead's path direction instead of the worst class it holds (curved beads
   are staircases whose jogs lay 90 deg material; half the hook's wall cells hold 2 classes) moves the
   hook's walls' share of FI^8 0.19 -> 0.11 and the ring result by 0.2 points: real, not the reason.
=> can't (no density-moving move) and, on these parts, barely should. A refill that packs its lines
   where material is worth most (per-line density in line_picks) would give the anneal the freedom;
   expected payoff here is a few points at most; wider parts would test the idea better.

### c19_pack: a refill that packs its lines where material is worth most (Nov 16, 2026)
`--pack 0.25`: a quarter of guided proposals re-lay a window at its own infill share with the lines
packed by a worth map (field-model gradient, weighted like the gradient start, refreshed every 200
evals; beads.pack, guide.WorthMap). ADAPT vs ADAPT + --pack 0.25 from one shared gradient start
(--start-state), 8,000 evals, frame / wrench / plate (new 40 x 28 mm plate with five holes, the
widest part), load-agnostic loads. runs/selfimprove/c19_pack.
```
part    arm    FI%    case%  worst%  time            near-wall cover   held-out FI% (seeds 1, 2)
frame   ADAPT  -31.1  -24.4   0.8    905->955 s      0.55->0.53        -23.8, -32.7
frame   PACK   -30.0  -23.1   0.9    905->932 s      0.55->0.51        -25.8, -30.7
plate   ADAPT  -40.2  -22.7   7.5    1301->1401 s    0.59->0.61        -36.9, -21.4
plate   PACK   -39.5  -21.6   6.9    1301->1362 s    0.59->0.57        -34.7, -19.4
wrench  ADAPT  -15.3  -10.3   4.2    845->887 s      0.46->0.49        -23.7, -20.7
wrench  PACK   -15.2   -9.3   5.2    845->871 s      0.46->0.47        -23.1, -19.5
```
(near-wall cover: band 0.4-1.2 mm from the outline, gradient start -> final, wallband.py.)
Pack moves were accepted as often as the rest (guided 13-18%, random 5-10%; ~1,500 tried per run).
PACK wins the composite score on every part (+2.3..+7.8 points), all from time and continuity:
it adds 38-46% less print time for 0.1-1.1 points less FI reduction. It grew no band beside the walls (cover
there lower than ADAPT on all three), and held-out seeds agree (PACK behind in 5 of 6).
probe_grad on the finals: moving 1% of the material between bands is worth <= 0.5 points either
way after the anneal (vs up to 3 at the start): the layouts already sit where the score is flat
in density, and FI^8 ends in the locked walls (54-91%), which material beside them barely relieves.
=> The missing move was not what held back wall-like bands; the score doesn't ask for them on
these parts. Pack stays available (off by default) as a cheaper way to the same strength.

### c20_bins16: bead lines every 11.25 degrees vs the four classes, true directions (Nov 16, 2026)
Both arms `--directions path` (beads scored at their own angle, printed as least-squares segments
within 0.1 mm). G4: `--start gradient4` + 4-class edits. G16: `--start gradient16` (field rounded to
11.25-degree bins) + `--line-bins 16` (guided lay16/32/64 re-lay a window at the best of 16
directions, turn16_pos/neg turn an area by one bin). Each arm its own gradient start, 8,000 evals,
frame / plate / wrench, load-agnostic loads. runs/selfimprove/c20_bins16.
```
part    arm  FI_agg start->final  time    strokes     obj    held-out FI% (seeds 1, 2)
frame   G4   0.906 -> 0.633       890 s    940       +115.8  -26.8, -33.3
frame   G16  0.882 -> 0.649       957 s   1292        +76.6  -22.3, -33.3
plate   G4   0.782 -> 0.491      1347 s   1629        +97.2  -37.1, -28.3
plate   G16  0.724 -> 0.500      1440 s   1991        +62.6  -33.8, -21.0
wrench  G4   0.808 -> 0.733       724 s    727        +94.0  -28.5, -20.4
wrench  G16  0.828 -> 0.733       782 s   1031        +54.3  -28.5, -19.7
```
(FI_agg: slicer start = 1. obj: points vs G16's start. Held-out: vs the slicer under seeds 1, 2.)
G16's start was 2.6% / 7.4% stronger on frame / plate (2.6% weaker on wrench), but its anneal
gained less (FI -26.4/-30.9/-11.6% vs G4's -30.1/-37.2/-9.2%): final FI 2.6% / 2.0% worse, wrench
tied. Held-out seeds: G4 ahead or tied in all 6. G4 leads the composite by 35-40 points on every
part: continuity 14-24, interlayer 4-11, time 7-8, strength -0.1..+1.3. Proposals 32% slower.
Lay moves were 77% of proposals, accepted 11-23% (G4's guided refills 10-30%).
Why:
1. The in-between angles are worth little. Snap test (scripts/analysis/snap.py, `free`): same beads,
   every free bead's direction rounded to 0/45/90/135 for scoring, walls left at their true
   angles. G16 finals (52-61% of free bead length off-class): FI_agg +0.65 / +1.6 / +0.03%,
   bulk +0.9..+1.1%, worst case +3.4..+5.2%. G4 control: <= 0.05%.
2. They cost strokes. Only 0/90 lines are zigzag-linked (zigzag_links); 45/135 (diag_links) and
   in-between lines (angle_links) are linked along each line only. Frame final, free strokes: G16
   in-between 12.9 mm mean vs 0 deg 25 mm; G4 0 deg 29.5 mm, 90 deg 19.4, 45/135 7.8-8.8 mm. More
   strokes cost time and continuity, so the composite charges each in-between move for its ends
   and the anneal keeps fewer of the strength gains.
3. No wall-like effect: bead length along the edge in the 0.4-1.2 mm band G4/G16 0.55/0.52,
   0.46/0.41, 0.51/0.49; FI^8 still ends in the locked walls (0.47/0.49, 0.30/0.33, 0.84/0.84).
=> On these parts 16 bins lose: exact angles buy <= 1.6% FI and cost 7-8% print time. gradient16
and --line-bins 16 stay available, off by default; path mode with the four classes is the setup to
use. Zigzag-linking 45/135 (and in-between) lines would cut strokes in both modes.
Viewer: path-mode runs draw the fitted printed moves at their true angle ("printed path" toggle);
viewers for the six runs written with scripts/run_viewer.py (local only, see README).

### c20 follow-ups: where the score sits, what blocks edits (Nov 16, 2026)
All on the c20 finals (scripts/analysis/; G4 unless noted). README "Where the score comes from".
- Off-axis bead strength (offaxis.py, printer.yaml Tsai-Wu): tension 99.6 / 98.3 / 93.2 / 75.7 / 56%
  at 5.6 / 11.25 / 22.5 / 45 / 90 deg; compression 100.3 / 101 / 102.8 / 100.3 / 84.6%.
  => four classes are never > 22.5 deg off: <= 7% at a cell, ~2% on average.
- Region split of sum FI^8 (decomp.py; walls = sparse layers' outer 0.4 mm, solid layers apart):
                 walls      solid      0.4-1.2 mm   interior    (share of FI^8, share of material)
    frame        .39 .20    .25 .27    .15 .19      .21 .34
    plate        .25 .12    .31 .29    .12 .13      .32 .46
    wrench       .70 .21    .19 .27    .08 .18      .02 .34
  Half of FI^8 in 1.5 / 1.4 / 0.08% of printed cells, 90% in 20 / 18 / 7%. Interior FI -> 0 would
  lower FI_agg by <= 2.9 / 4.8 / 0.3% (no redistribution). G16 in-between beads: 28 / 33 / 26% of
  material, 14 / 18 / 5% of FI^8 (median FI 0.22-0.28 of peak).
  wallband.py counts the 4 solid layers in its bands (its skip test never fires), so its wall8
  numbers above (c18-c20) mix solid layers in; decomp.py separates them.
- Per-region p-mean FI, slicer -> gradient start -> final (walltrack.py):
    frame  walls .911 .770 .740  solid 1.247 1.129 .670   share of FI^8 at slicer: solid .92
    plate  walls .803 .620 .581  solid 1.255  .982 .534   solid .97
    wrench walls 1.115 .935 .915 solid 1.207  .950 .757   solid .69, walls .29
  => walls are relieved 18-28% without a wall bead moving (laminate: stiffer surroundings, less
  strain), mostly by the gradient start; the anneal adds 2-6%. Top/bottom layers held the score at
  the start; walls are the most stressed region at the end on all three. Kept walls in the score
  (dropping them would reward thinning the material that relieves them).
- Hot spots (hotspots.py, sum over layers): wrench half of FI^8 in ~27 xy cells at inside corners
  (3 around the jaw opening, 1 at the handle-ring junction); frame one slot corner holds 27%;
  plate around holes + one corner, partly infill. Per layer 3-4% each (top/bottom 5-8%): the same
  xy spots through the stack. FI vs distance from peak (singular.py): log-log slope -0.05..-0.20,
  half within ~1 mm; not a clean r^-0.46 corner singularity at this resolution.
- Blocked edits (blockers.py, history d per term, weights fitted from d_total):
                  strength-improving  accepted  blocker: cont  inter  time   cont/gain median
    frame         56%                 24%       56%            31%    12%    15x
    plate         56%                 25%       37%            42%    20%    7x
    wrench        48%                 20%       61%            23%    14%    39x
  Strength forgone in rejected / gained by accepted: 0.35 / 0.51, 0.42 / 0.62, 0.22 / 0.13
  (overlapping edits, so an upper bound). join / anchor tried 1-14 times per 8,000 edits.
  => edits toward stronger layouts are rejected for the loose ends they leave (unlinked line ends,
  lost overlap with the layers above/below), not for print time. Plan (not built): multi-layer
  re-lays (2-4 layers per edit), repair follow-ups after promising rejections, tidy-before-score;
  and a local-reinforcement probe at the hot spots.
