# Follow-up to the rotating grid hold-out: the program-emission bug, and the decisive 160-native re-solve

**Date:** 2026-08-03. **Scope:** the two follow-ups
`ROTATING_HOLDOUT_REPORT.md` Section 11 named as next. Nothing in that report is
overwritten; every new number is in a new artifact alongside the cited ones.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the two-dimensional solver uses).
IoU = intersection over union. bpp = bits per pixel. phi = melt fraction.
J = the shape-fidelity objective. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit-tested, red first, or reproduction-gated.
COMPUTED = measured from a real forward or solve run in this pass. ASSUMED = a
modelling choice or an inference not measured here.

**Conventions, carried on every number and unchanged from the hold-out report.**

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

with chi the sub-cell AREA FILL indicator (`adjoint2d/chi_area.py`).
`J_raster_chi` against the binary raster is reported alongside. **J is a sum
over cells and is NOT comparable between grids** (1036 cross part cells at grid
120 against 1920 at grid 160; 1030 keyhole cells at 120 against 1844 at 160);
read rankings and margins. `t_stop` = argmin of J over the arm's own
trajectory, HORIZON flagged when the minimum sits on the last stored step
(index 1499, 750.0 s). The melted region for IoU, growth and under-melt is
phi >= 0.5. IoU is against the BINARY part mask, which is the reading the
SOLVED class threshold of 0.95 has always been quoted on. Absorbed power is the
state-B value in W/m. Conductivity channel only, 4 bpp inside the part, dopant
map held at the nominal 1 outside the part. The drive is recalibrated at each
grid so the uniform STATIC arm absorbs 500.0 W/m in electrical state B; rotating
arms are NOT dose matched against each other.

---

## 1. The two verdicts, up front

**VERDICT 1, the program-arithmetic bug.** COMPUTED and PROVEN.
The bug is real, it is one line of apportionment arithmetic, and it is fixed and
gated. **Three of the twenty stored cyclic turntable programs are affected**
(keyhole continuous, gear8 continuous, T-shape turntable deliverable); the
sequential T and L programs are a different emitter and are structurally immune;
the cross dwell 150-move program and the cross and star fixed-interval indexing
programs divide evenly and are not affected. With the keyhole's program
re-emitted divisor aware, **its realized dwell error against its own design
falls from 20.0 percent to 0.5 percent of the design fraction**, and the arm
improves at BOTH grids:

| keyhole, continuous rotation | grid 120 | grid 160 |
|---|---|---|
| IoU, old 448-move program | 0.9684 | 0.9243 |
| **IoU, fixed 456-move program** | **0.9828** | **0.9447** |
| J, old program | 12.49 | 78.54 |
| **J, fixed program** | **7.84** | **62.39** |
| program-realization gap on J | +73.2 % | +28.0 % |
| **gap after the fix** | **+0.48 %** | **+1.04 %** |

The fix does not change the grid verdict: **0.9828 is inside the SOLVED class at
grid 120 and 0.9447 is still outside it at grid 160.** It does raise the arm's
grid-120 fidelity above the quasi-static design model it was solved against
(0.9828 against 0.9753), which the old program could not reach.

**VERDICT 2, transfer against convergence, and it SPLITS BY SHAPE.** COMPUTED.
Both shapes were re-solved natively at grid 160 under the production recipe and
scored at 160 by the hold-out harness.

| shape | IoU at 120 (solved at 120) | IoU at 160, map transferred from 120 | IoU at 160, map SOLVED at 160 | class recovered | **verdict** |
|---|---|---|---|---|---|
| **cross**, 90-degree indexed | 0.9700 | 0.8052 | **0.8751** | **NO** | **FORWARD NON-CONVERGENCE** |
| **keyhole**, continuous rotation, fixed program | 0.9828 | 0.9447 | **0.9716** | **YES** | **MAP TRANSFER** |

Stated in exactly the terms the dissertation needs:

* **On the keyhole the grid-120 class loss is MAP TRANSFER.** A map solved
  natively at grid 160 returns the arm to the SOLVED class (IoU 0.9716, and
  0.9701 for the unquantized map, both at or above 0.95). Maps are grid
  entangled, even filtered and even under rotation, and the fix is to solve at
  the deployment grid or to find a better transfer convention.
* **On the cross the grid-120 class loss is FORWARD NON-CONVERGENCE.** A map
  solved natively at grid 160 recovers only 42.4 percent of the lost IoU and
  lands at 0.8751, far outside the class. No dopant map at this budget reaches
  0.95 at grid 160 on this shape, so the failure is not the transfer; it is the
  rotating forward itself, and no absolute IoU on the cross should be quoted at
  any grid without a convergence study behind it.

**Both mechanisms are present on both shapes; what differs is which one
dominates.** COMPUTED. Re-solving natively recovers 42.4 percent of the cross's
IoU loss and 70.7 percent of the keyhole's. The transfer penalty is therefore
real on both (the native map beats the transferred one by 49.0 percent of J on
the cross and 60.1 percent on the keyhole), but only on the keyhole is the
residual small enough to clear the class line.

---

## 2. TASK 1: the program-emission arithmetic

### 2.1 What the bug was, exactly

PROVEN by a red-first test. `adjoint2d/dwell.py:cycle_program` turns a dwell
fraction vector into the ordered list of holds the machine runs. It divides the
cycle into `n_slots = cycle_time_s / dt_s` integer control steps and apportions
them across the kept positions by Hamilton's largest-remainder rule. That part
was correct. The bug was that **the allocation was computed once and then
repeated identically in every cycle**, so a rounding that is at most one control
step inside a cycle becomes a permanent bias over the exposure.

The keyhole's program is the exact failing case: a 20.0 s cycle at a 0.5 s
control step is 40 control steps, spread over 12 half-turn-distinct positions.
Forty does not divide by twelve. The per-cycle quota is 3.333 steps, largest
remainder returns 4, 4, 4, 4 and then 3 eight times, and repeating that forever
realizes dwell fractions 0.100 and 0.075 against the design 1/12 = 0.08333. The
map was solved against EQUAL twelve-angle weights, so the machine was executing
a different design from the one that was optimized.

The red test is `test_cycle_program_realizes_equal_dwell_when_the_cycle_does_not_divide`
in `fgm_solve_campaign/adjoint2d/tests/test_dwell.py`. Observed failing first,
for the right reason:

    assert np.max(np.abs(got / total - 1.0 / 12)) <= 1.0 / n_total
    AssertionError: assert 0.01833890746934226 <= (1.0 / 1495)

### 2.2 The fix

`adjoint2d/dwell.py:carry_forward_slots`, new, called by `cycle_program`.
Largest remainder is applied to a running DEBT (the exact quota accumulated so
far, minus the control steps already handed out) rather than to a single cycle
in isolation. A position rounded up this cycle carries a negative debt into the
next one and is rounded down there, so **the leftover control steps rotate**.
Every cycle still receives exactly its own slot count, every allocation is
non-negative, and the cumulative allocation tracks the exact quota to within one
control step over the whole exposure. On the keyhole the allocation returns to
exactly 1/12 every three cycles (4-4-4-4-3-3-3-3-3-3-3-3, then
3-3-3-3-4-4-4-4-3-3-3-3, then 3-3-3-3-3-3-3-3-4-4-4-4, which is 10 steps each
per 120).

Two smaller corrections travel with it, both visible in the tests:

1. The exposure's final PARTIAL cycle is apportioned by the same rule instead of
   letting the round robin run off the end. On the cross's 4-position program at
   750.0 s (37.5 cycles) the old emitter executed 0.2533 / 0.2533 / 0.2467 /
   0.2467 rather than 0.25 each; it now executes exactly 187.5 s at each
   position. **That was invisible on the cross only because its four-fold
   symmetry makes 0 and 180 degrees the same part-frame heating, so the two
   biased pairs cancel** (COMPUTED: the stored hold-out's cross dwell
   quasi-static realized-weight and equal-weight arms are bit identical at
   J 25.49, and that is why).
2. Adjacent holds at the same position are merged, so the program can never
   contain a null move.

Six new tests, all red first where they were meant to be: the twelve-position
divisor case, the cross regression, the angular sweep order, a 40-case
randomized property check that no allocation is negative and every cycle is
exactly filled, and the rewritten realized-weight test.

**Test gate.** PROVEN. `test_dwell`, `test_dwell_march`, `test_seq_dwell`,
`test_seq_dwell_march` and `test_robust_rot`: **65 passed**. Whole
`adjoint2d/tests/` directory: **446 passed, 1 failed**. The one failure is
`test_eqs_assembly.py::test_assembled_solve_is_bit_identical_to_production` and
it is PRE-EXISTING, confirmed by re-running it with `dwell.py` stashed back to
its committed state, where it fails identically. It is unrelated to this work
and is not addressed here.

**No gradient was computed and therefore none was re-gated.** This pass changes
program-emission arithmetic and runs forward scoring and one L-BFGS-B solve on
the STANDING gated gradient. `forward.py`, `adjoint.py`, `rot_kernel.py`,
`dwell_kernel.py` and `dwell_march.py` were READ and not modified. The standing
finite-difference gate of `gate_rot` applies unchanged, and it remains a
SUBGRADIENT gate whose random-direction probe bottoms at 1.22e-05 because the
pinned population is the cold powder bed. Every solved map here inherits that
caveat.

### 2.3 The bug-class inventory, all twenty cyclic programs and all eleven sequential ones

COMPUTED by `scripts/analysis/audit_turntable_programs.py`, written to
`fgm_solve_campaign/out_rot_holdout/turntable_program_audit.json`. A program is
AFFECTED when the realized dwell fraction differs from the requested one by more
than one control step over its own exposure.

| program | emitter | control steps per cycle | kept positions | divides | max realized minus requested | AFFECTED |
|---|---|---|---|---|---|---|
| `out_intake/keyhole_novel.json` continuous | cyclic | 40 | 12 | NO | **0.0167** (20.0 % of design) | **YES** |
| `out_intake/gear8_novel.json` continuous | cyclic | 40 | 12 | NO | **0.0167** (20.0 % of design) | **YES** |
| `out_dwell/T_shape_turntable_deliverable.json` | cyclic | 40 | 6 | NO | **0.0108** | **YES** |
| `out_dwell/T_shape_dwell.json` deliverable | cyclic | 40 | 6 | NO | 0.0000 | no |
| `out_dwell/cross_turntable_deliverable.json` | cyclic | 40 | 4 | yes | 0.0000 | no |
| `out_dwell/cross_dwell.json` deliverable | cyclic | 40 | 4 | yes | 0.0000 | no |
| `out_dwell/L_shape_*` deliverable | cyclic | 40 | 2 | yes | 0.0000 | no |
| `out_dwell/square_*` deliverable | cyclic | 40 | 8 | yes | 0.0000 | no |
| all eight `equal_dwell_control` programs | cyclic | 40 | 8 | yes | 0.0000 | no |
| `out_intake/gear8_novel.json` index8 | cyclic | 40 | 4 | yes | 0.0000 | no |
| eleven `out_seq/*turntable*.json` | sequential | n/a | 2 to 3 | n/a | holds land exactly on the control-step grid | no |
| cross and star 90-degree indexing | fixed interval | 4 outer steps per position | 4 | yes | n/a | no |

**Twenty cyclic programs, five of them in the structurally exposed class (the
kept positions do not divide the per-cycle slot count), three of those actually
affected.** COMPUTED.

Reading the three affected ones honestly, because they are not equally serious:

* **keyhole and gear8 continuous, the serious ones.** Both were solved against
  EQUAL twelve-angle weights and then emitted a program that realizes 0.100 and
  0.075. The design the optimizer chose and the design the machine executes are
  different objects. The keyhole's cost is measured in Section 2.4 (+73.2
  percent of J at grid 120). The gear8's is NOT measured here; its program is
  arithmetically identical (40 steps, 12 positions, equal request, same 0.0167
  error) so the same class of penalty applies, but no forward run was made for
  it in this pass and no number is claimed.
* **T-shape turntable deliverable, the mild one, and it was already scored
  honestly.** The optimizer's dwell vector has two positions at 0.0138, which is
  0.55 of a control step per cycle; largest remainder rounds them to a whole
  step, 0.025, and the repeat makes that permanent. But
  `scripts/analysis/run_dwell_solve.py` scores the deliverable AT THE REALIZED
  weights (`D_program_4bpp`), so the stored T-shape number is a true statement
  about what the machine executes. COMPUTED from the stored artifacts: the cost
  of that snapping is **+2.08 percent of J** (422.11 to 430.90) and 0.0059 of
  IoU (0.6231 to 0.6172). The keyhole differs precisely because its headline
  number was the equal-weight design model, not the executed program.
* **The same snapping exists even where the positions DO divide**, whenever the
  requested fractions are off the 1/40 grid. COMPUTED from the stored artifacts:
  the square deliverable pays **+0.70 percent of J** for it (16.808 to 16.925),
  the cross and the L-shape pay under 0.01 percent. The carry-forward rule
  removes this class too, because the exposure has 1500 control steps rather
  than 40, so a request of 0.13 is realizable to one step even though it is not
  a multiple of 1/40. Those arms were not re-run.

### 2.4 The keyhole, re-emitted and re-run at both grids

The program was re-emitted by `scripts/analysis/reemit_keyhole_program.py` from
the SAME inputs: the same twelve half-turn-distinct positions, the same equal
design weights, the same 20.0 s cycle, the same 0.5 s control step and the same
747.5 s exposure. Only the allocation of the leftover control steps changed.
Realized dwell error against the design: **0.01667 (20.0 percent of the design
fraction) to 0.00039 (0.5 percent)**, which is 0.58 of one control step over the
exposure. Moves: 448 to 456.

The arm was then re-run through the hold-out harness at both grids, unchanged in
every other respect. New artifacts; the old ones are untouched.

**Reproduction gate first, as the harness requires.** PROVEN. The equal-weight
quasi-static arm does not depend on the program at all, so it is the check that
this is the same arm: **J 8.08 and IoU 0.9753 at grid 120**, against the stored
`GEOMETRY_GENERALIZATION_REPORT.md` Section 6.1 numbers 8.0759 and 0.9753. Exact
to the printed digits, and identical to the value the previous pass reproduced.

| grid | arm | J | J raster | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|---|
| 120 | ROT_solved, **fixed program** | **7.84** | 22.52 | **0.9828** | 0.9663 | 1.36 | 0.39 | 745.0 | 245.9 | 0.13 |
| 120 | ROT_solved, old program | 12.49 | 24.79 | 0.9684 | 0.9558 | 1.46 | 1.75 | 718.5 | 251.7 | 0.14 |
| 120 | ROT_uniform, fixed program | 173.10 | 192.50 | 0.8026 | 0.7997 | 13.59 | 8.83 | 580.0 | 302.0 | 0.91 |
| 120 | QS_solved, realized dwell | 8.11 | 23.31 | 0.9762 | 0.9656 | 2.04 | 0.39 | 746.5 | 245.9 | 0.20 |
| 120 | QS_solved_equalW, design model | 8.08 | 23.32 | 0.9753 | 0.9656 | 2.14 | 0.39 | 747.5 | 245.7 | 0.20 |
| 120 | STATIC_uniform | 509.96 | 524.99 | 0.5750 | 0.5659 | 30.68 | 24.85 | 336.0 | 500.0 | 2.27 |
| 120 | STATIC_solved | 168.35 | 191.73 | 0.8163 | 0.8162 | 14.17 | 6.80 | 501.5 | 367.2 | 0.98 |
| 160 | ROT_solved, **fixed program** | **62.39** | 82.14 | **0.9447** | 0.9318 | 4.01 | 1.74 | 722.0 | 258.7 | 0.31 |
| 160 | ROT_solved, old program | 78.54 | 97.85 | 0.9243 | 0.9202 | 4.56 | 3.36 | 701.0 | 264.8 | 0.34 |
| 160 | ROT_uniform, fixed program | 384.03 | 408.85 | 0.7749 | 0.7709 | 13.45 | 12.09 | 554.5 | 314.0 | 0.95 |
| 160 | QS_solved, realized dwell | 64.24 | 87.99 | 0.9342 | 0.9273 | 3.80 | 3.04 | 716.0 | 258.7 | 0.28 |
| 160 | QS_solved_equalW, design model | 63.58 | 87.36 | 0.9331 | 0.9277 | 3.80 | 3.15 | 717.0 | 258.5 | 0.28 |
| 160 | STATIC_uniform | 759.35 | 781.97 | 0.6328 | 0.6293 | 26.14 | 20.17 | 339.0 | 500.0 | 1.94 |
| 160 | STATIC_solved | 351.91 | 376.13 | 0.7991 | 0.7945 | 14.43 | 8.57 | 440.0 | 404.0 | 1.05 |

Five things worth naming, all COMPUTED.

1. **The program-realization gap is gone.** Grid 120: **+73.2 percent to
   +0.48 percent** of J. Grid 160: **+28.0 percent to +1.04 percent**. The
   ASSUMED prediction of `ROTATING_HOLDOUT_REPORT.md` Section 5, that fixing the
   arithmetic would remove the gap entirely, holds to within half a percent.
2. **The uniform comparator improves too**, from J 191.50 to 173.10 at grid 120,
   because the dwell distribution is part of the actuator and not part of the
   map. The margin of the solved arm over its own rotating uniform comparator
   therefore moves only from +93.5 to **+95.5 percent** of J at grid 120 and
   from +80.9 to **+83.8 percent** at grid 160. Every ranking is preserved at
   both grids, as before.
3. **The executed program now beats its own quasi-static design model** at grid
   120 (J 7.84 against 8.08, IoU 0.9828 against 0.9753), which the old program
   could not do. The finite cycle time is a small help here rather than a small
   cost: the program march is better than its quasi-static limit by 3.4 percent
   of J at grid 120 and 3.0 percent at grid 160.
4. **The grid verdict is unchanged.** SOLVED at 120, not SOLVED at 160. The bug
   was never the reason the class was lost across the grid; it was a separate
   defect that happened to live in the same arm.
5. **One residual, named rather than buried.** The program ends at 747.5 s and
   the march horizon is 750.0 s, so the last 5 control steps hold the final
   position and the executed fraction there reads 0.0860 instead of 0.0833. It
   does not touch any scored number, because every keyhole arm stops at or
   before 745.0 s at grid 120 and 722.0 s at grid 160, inside the program.

**Energy gate: clean on all 14 forward runs of the fixed-program arm.** COMPUTED.
Maximum relative energy residual at any arm's own stop is 2.27 percent (static
uniform at grid 120) against the 5 percent threshold, and
`energy_gate_violations` is EMPTY. **Temperature ceiling:** the two static
keyhole comparators exceed 250 C as before (290.5 and 253.1 C at grid 120,
268.7 and 259.6 C at 160); no rotating arm does.

---

## 3. TASK 2: the decisive 160-native re-solve

### 3.1 What was held fixed and what was regenerated

`scripts/analysis/run_rot_native160.py`. The recipe, the budget and the actuator
are the grid-120 ones; the discretization-dependent objects are rebuilt at 160.

**Held fixed against the grid-120 solve.** The filtered production recipe (box
[0, 1], L-BFGS-B on the finite-difference gated filtered averaged-kernel
gradient, conductivity channel only, 4 bpp quantization for the deliverable
map); the budget class of 16 gradient evaluations per start on two starts, which
is the campaign's 40 forward-equivalent depth budget; the actuator (the cross's
90-degree indexing at 2.0 s on four candidate angles, the keyhole's twelve-
position program, both on the same 1500-step 750 s horizon); the averaged kernel
with EQUAL angle weights, which is the model every map in this campaign was
solved against.

**Regenerated at 160.** The part mask and the target chi, by the production
domain builder. The drive, recalibrated so the uniform static arm absorbs
500.0 W/m in state B: **cross 2815.4 V gives 456.3 W/m at 160, recalibrated to
2947.0 V, verified 500.00 W/m; keyhole 2428.2 V gives 380.3 W/m, recalibrated to
2784.0 V, verified 500.00 W/m.** Both match the hold-out report's recalibration
to the printed digit, which is a cross-check that the two passes are describing
the same case.

**And the design filter width, which is the one thing that would silently ruin
this experiment if it were carried across in cells.** sigma is a PHYSICAL length
of 0.75 mm (`design_filter` radius convention), which is 1.5 cells at grid 120
and **2.0 cells at grid 160**. Holding the CELL count would shrink the design
length with the grid and would let a finer solve buy fidelity with finer
features, confounding the answer. `solve_start` in
`scripts/analysis/run_rot_avg_solve.py` now takes `sigma_cells` explicitly and
defaults to the grid-120 value, so no existing call changes.

The warm start is the grid-120 static map moved to 160 by the production
resample. It is a STARTING POINT, not a transferred answer: the cold start at
saturation 1 is run alongside and the better of the two wins, exactly as at grid
120. COMPUTED: warm won on both shapes (cross 164.76 against the cold start's
376.89; keyhole 46.64 against 67.02, both on the solve objective).

### 3.2 The cross, grid 160

| arm at grid 160 | J | J raster | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|
| ROT_uniform | 442.71 | 440.36 | 0.7832 | 0.7544 | 16.25 | 8.96 | 342.5 | 500.0 | 1.28 |
| ROT, map transferred from 120 | 303.70 | 309.11 | 0.8052 | 0.7924 | 7.50 | 13.44 | 425.0 | 384.9 | 0.67 |
| **ROT, map SOLVED at 160** | **154.96** | 159.35 | **0.8751** | 0.8564 | 5.10 | 8.02 | 503.0 | 345.0 | 0.44 |
| ROT, map solved at 160, unquantized | 158.37 | 163.13 | 0.8699 | 0.8546 | 4.90 | 8.75 | 500.5 | 345.5 | 0.43 |
| QS, transferred map, equal weights | 305.50 | 311.13 | 0.8037 | 0.7922 | 7.71 | 13.44 | 426.5 | 384.9 | 0.69 |
| QS, native map, equal weights | 156.73 | 160.87 | 0.8734 | 0.8558 | 5.31 | 8.02 | 504.0 | 345.0 | 0.45 |
| STATIC_uniform | 724.72 | 746.22 | 0.6152 | 0.6034 | 6.67 | 34.38 | 258.5 | 500.0 | 0.70 |
| STATIC_solved | 631.36 | 647.39 | 0.6667 | 0.6609 | 11.88 | 25.42 | 439.5 | 372.4 | 1.01 |

**The native solve is a large, real improvement and it is not enough.** COMPUTED.
It beats the transferred map by **49.0 percent of J** and by 0.0699 of IoU, and
it repairs the limb starvation the hold-out report diagnosed (under-melt 13.44
percent to 8.02 percent, growth 7.50 to 5.10). But 0.8751 is 0.075 below the
class line, it recovers only **42.4 percent** of the 0.1648 IoU points the grid
change cost, and the same conclusion holds for the unquantized map (0.8699) and
for the quasi-static design model at 160 (0.8734), so it is not a quantization
artifact and not a finite-cycle artifact.

**VERDICT, cross: FORWARD NON-CONVERGENCE.** No map at this budget reaches 0.95
at grid 160 on this shape. This is consistent with, and now stronger than, the
hold-out report's Section 6 measurement that the cross's ROTATING uniform arm
alone (which contains no solved map at all) moves 0.097 IoU points between the
two grids while its STATIC uniform arm moves 0.064 the OTHER way.

### 3.3 The keyhole, grid 160, with the fixed program

| arm at grid 160 | J | J raster | IoU | IoU area | growth % | under % | stop s | P W/m | Eres % |
|---|---|---|---|---|---|---|---|---|---|
| ROT_uniform | 384.03 | 408.85 | 0.7749 | 0.7709 | 13.45 | 12.09 | 554.5 | 314.0 | 0.95 |
| ROT, map transferred from 120 | 62.39 | 82.14 | 0.9447 | 0.9318 | 4.01 | 1.74 | 722.0 | 258.7 | 0.31 |
| **ROT, map SOLVED at 160** | **24.93** | 47.06 | **0.9716** | 0.9549 | 1.30 | 1.57 | 750.0 HORIZON | 251.0 | 0.18 |
| ROT, map solved at 160, unquantized | 25.85 | 47.61 | 0.9701 | 0.9544 | 1.57 | 1.46 | 749.0 | 251.5 | 0.19 |
| QS, transferred map, equal weights | 63.58 | 87.36 | 0.9331 | 0.9277 | 3.80 | 3.15 | 717.0 | 258.5 | 0.28 |
| QS, native map, equal weights | 24.25 | 46.02 | 0.9743 | 0.9548 | 1.19 | 1.41 | 750.0 HORIZON | 250.7 | 0.17 |
| STATIC_uniform | 759.35 | 781.97 | 0.6328 | 0.6293 | 26.14 | 20.17 | 339.0 | 500.0 | 1.94 |
| STATIC_solved | 351.91 | 376.13 | 0.7991 | 0.7945 | 14.43 | 8.57 | 440.0 | 404.0 | 1.05 |

**VERDICT, keyhole: MAP TRANSFER.** COMPUTED. The native solve reaches
**IoU 0.9716 at grid 160**, inside the SOLVED class, recovering **70.7 percent**
of the IoU the grid change cost and beating the transferred map by 60.1 percent
of J.

**Two horizon flags, and why the verdict survives them.** The native quantized
arm and its quasi-static twin both stop on the last stored step (750.0 s), so
their J values are upper bounds and their true optima may lie later. The
unquantized native arm is NOT at the horizon (749.0 s) and reads **0.9701**,
also inside the class, so the class recovery does not depend on a
horizon-flagged number.

**Energy gate: clean on all 16 forward runs across both native re-solves**
(8 arms each), `energy_gate_violations` EMPTY on both files, maximum residual
1.94 percent. **Temperature ceiling:** two keyhole static comparators exceed
250 C (268.7 and 259.6 C); no rotating arm on either shape exceeds 235.1 C.

### 3.4 Cost

COMPUTED. Cross native re-solve 1112 s wall (cold start 457 s, warm start 556 s,
eight scored forward runs at grid 160). Keyhole native re-solve 1818 s (cold 845
s, warm 836 s). Keyhole fixed-program hold-out at both grids 188.3 s, 14 forward
runs. Total real compute for this report is under 55 minutes, three streams run
concurrently under `fgm_solve_campaign/env1.sh` single-thread pinning.

---

## 4. Proven, computed, assumed

**PROVEN**
* The divisor bug reproduces in a unit test on the exact 40-control-steps /
  12-positions case, observed failing first at 0.01834 against a tolerance of
  1/1495, and passes after the carry-forward rule.
* Under the carry-forward rule every per-cycle allocation is non-negative and
  every cycle is exactly filled, over 40 randomized weight vectors with 2 to 12
  positions.
* The cross's 4-position 40-slot cycle allocation is unchanged per cycle
  (10 steps each) and its executed dwell becomes exactly equal.
* 65 tests pass across the five modules this change touches; 446 of 447 pass
  across the whole `adjoint2d/tests/` directory, the one failure being
  pre-existing and confirmed so by re-running it against the committed
  `dwell.py`.
* Reproduction gate on the re-run keyhole arm: the equal-weight quasi-static
  model gives J 8.08 and IoU 0.9753 at grid 120 against the stored 8.0759 and
  0.9753.
* The drive recalibration at grid 160 reproduces the hold-out report's values on
  both shapes (cross 2947.0 V, keyhole 2784.0 V), each verified by a second EQS
  solve at 500.00 W/m.

**COMPUTED**
* Every number in Sections 1 through 3.
* The bug-class inventory: 20 cyclic programs, 5 structurally exposed, 3
  affected; 11 sequential programs, none affected.
* The keyhole's realization gap collapsing from +73.2 to +0.48 percent of J at
  grid 120 and from +28.0 to +1.04 percent at grid 160.
* The two native-solve verdicts and the fractions of the IoU loss they recover
  (42.4 percent cross, 70.7 percent keyhole).

**ASSUMED**
* That the leftover control steps SHOULD rotate rather than the cycle length
  being snapped to a multiple of the position count. Both remove the bias;
  rotating them keeps the commanded cycle time, which is the machine-facing
  parameter. Snapping the cycle was not tested.
* That 16 gradient evaluations per start at grid 160 is the same budget as at
  grid 120. It is the same COUNT of gradient evaluations, not the same wall
  clock and not necessarily the same distance to the grid-160 optimum. A deeper
  cross solve at 160 might do better than 0.8751, and this pass does not bound
  how much better. This is the single largest caveat on the cross verdict.
* That the sub-cell area fill is the right nominal target, that bilinear
  resampling with a clip is the right transfer convention, that an arbitrary
  stop time is realizable as a process control, and that the move between
  indexed positions is instantaneous. All unchanged from
  `ROTATING_HOLDOUT_REPORT.md` Section 9.
* That the gear8 continuous program carries the same class of penalty as the
  keyhole's. Arithmetically identical, but no forward run was made for it.

---

## 5. Honest limits

1. **Two shapes, one grid pair.** The transfer-against-convergence verdict is
   measured on the cross and the keyhole at 120 against 160. It SPLITS between
   them, which is itself the strongest reason not to generalize it to the star,
   the square, the T or the L.
2. **The cross verdict is a statement at a fixed budget.** See the ASSUMED entry
   above. "No map at this budget reaches 0.95 at grid 160" is what was measured;
   "no map can" is not.
3. **A two-grid comparison is still not a convergence study.** The right
   experiment behind any absolute IoU is a forward grid ladder on the ROTATING
   uniform arm at 96, 120, 160 and 200 with the drive recalibrated at each grid.
   `ROTATING_HOLDOUT_REPORT.md` Section 11 named it third and it is still not
   run.
4. **The three affected programs were not all re-run.** The keyhole was. The
   gear8's continuous arm and the T-shape's turntable deliverable were not, so
   no corrected number is claimed for either.
5. **The old artifacts are cited by the previous report and are untouched**, so
   two contradictory keyhole programs now exist on disk. Anything downstream
   that reads `out_intake/keyhole_novel.json` still gets the old 448-move
   program. Re-emitting into that file is a deliberate later decision, not one
   this pass took.
6. **The adjoint arms are conductivity only; the historical stored masks
   co-vary permittivity.** Unchanged, still the largest actuator gap.
7. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). Every
   IoU here is a statement about a two-dimensional model, not a printed part.

---

## 6. The single most valuable next layer

**The forward grid ladder on the rotating uniform arm**, at 96, 120, 160 and 200
with the drive recalibrated at each grid and no solved map anywhere in it.
Section 3.2 now shows the cross's class loss is the forward and not the map,
which promotes that measurement from a caveat to the blocking experiment: until
it exists, no absolute IoU on a rotating cross arm is quotable at any grid. It
is forward runs only, no solve, and on the evidence of this pass it is well
under an hour.

Second, and cheap: **re-emit and re-score the gear8 continuous arm**, the one
remaining seriously affected program, so the inventory in Section 2.3 carries a
measured number rather than an arithmetic inference.

---

## 7. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

**Changed code**
* `fgm_solve_campaign/adjoint2d/dwell.py` new `carry_forward_slots`; divisor-aware
  `cycle_program`; `TurntableProgram` gains `steps_per_position_total`,
  `cycle_slot_plan` and `allocation_rule`, and emits
  `steps_per_position_over_exposure` and `allocation_rule` in `as_json`
* `fgm_solve_campaign/adjoint2d/tests/test_dwell.py` six new or rewritten tests,
  red first
* `scripts/analysis/run_rot_avg_solve.py` `solve_start` takes `sigma_cells`,
  defaulting to the grid-120 value so no existing call changes
* `scripts/analysis/run_rot_holdout.py` new arm `keyhole_cont_fixedprog`, and
  `_dwell_program` gains a `keypath`

**New code**
* `scripts/analysis/reemit_keyhole_program.py` divisor-aware re-emission
* `scripts/analysis/audit_turntable_programs.py` the bug-class inventory
* `scripts/analysis/run_rot_native160.py` the native re-solve and its scoring

**New results, nothing overwritten**
* `fgm_solve_campaign/out_rot_holdout/keyhole_program_fixed.json` the corrected
  program with the old and new realized fractions side by side
* `fgm_solve_campaign/out_rot_holdout/keyhole_cont_fixedprog.json` and
  `_maps.npz` the arm re-run at both grids
* `fgm_solve_campaign/out_rot_holdout/turntable_program_audit.json` the inventory
* `fgm_solve_campaign/out_rot_native160/cross_native160.json` and `_maps.npz`
* `fgm_solve_campaign/out_rot_native160/keyhole_native160.json` and `_maps.npz`
* `fgm_solve_campaign/logs_rot_holdout/keyhole_cont_fixedprog.log`,
  `cross_native160.log`, `keyhole_native160.log`

**Read, not modified**
* `ROTATING_HOLDOUT_REPORT.md`, `GEOMETRY_GENERALIZATION_REPORT.md`,
  `DWELL_SCHEDULE_REPORT.md`, `CONTINUOUS_ROTATION_REPORT.md`
* `fgm_solve_campaign/out_rot_holdout/{cross_index90,cross_dwell,star_index90,keyhole_cont}.json`
* `fgm_solve_campaign/out_intake/keyhole_novel.json`, `gear8_novel.json`,
  `keyhole_maps.npz`
* `fgm_solve_campaign/out_dwell/*.json`, `fgm_solve_campaign/out_seq/*turntable*.json`
* `fgm_solve_campaign/out_lib/cross_maps.npz`, `out_rot/cross_rotavg_step90*.{json,npz}`
* `fgm_solve_campaign/adjoint2d/{forward,adjoint,rot_kernel,dwell_kernel,dwell_march,robust_rot,design_filter,chi_area,printability,topopt_objective}.py`
