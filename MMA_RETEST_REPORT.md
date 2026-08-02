# Fixing the optimizer mechanics and retesting the Heaviside projection fairly

Follow-up to `TOPOPT_REPORT.md`, which diagnosed its own headline result as possibly an
optimizer artifact: "With 3 evaluations per beta stage, L-BFGS-B spends most of a stage on
line-search trial points that are worse than the incumbent, and the curvature memory is
discarded at each restart" (Section 4.3), and named MMA as not implemented
(`FROZEN_CONVENTIONS_2D.md` Section 9 item 4). This pass implements MMA, retests the
projection at a matched budget, and settles the question.

Solve lane only. No dissertation file was edited. Nothing was committed. No file outside
`topopt*.py`, `mma.py`, their tests and the new drivers was modified.

Acronyms on first use: RFAM radio-frequency additive manufacturing; FGM functionally graded
material; EQS electro-quasi-static; IoU intersection over union; bpp bits per pixel;
L-BFGS-B limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints; MMA the
method of moving asymptotes (Svanberg 1987).

**Conventions, stated once, every number below obeys them.**
* **J** is the melt-region shape-fidelity objective summed over the whole domain, against
  the grid-independent area-fill target chi, read at **the argmin over that arm's own
  stored trajectory**. `at_horizon` is flagged and makes that arm's J an upper bound.
  J against the area-fill chi is **not** comparable to any J in `out_lib` or `out_ms`;
  every result file also stores `J_raster_chi`, the same map under the old binary target,
  and that is the cross-pass number.
* **IoU** is always against the binary part mask at the stated grid. **Every fidelity
  number carries its grid.** Solves are at grid 120; the Gate A hold-out is at grid 160
  with the drive recalibrated there.
* The deliverable arm is always the continuous map quantized to **4 bpp inside the part**
  and re-run through the real forward, never the continuous map alone.
* Actuator: **conductivity only** (`eps_covary=False`), the deployable channel, on every
  arm. **Arms are not dose matched**: absorbed power over the 30 deliverable arms of this
  pass spans **314.9 to 499.1 W per metre of depth**.

---

## 1. Verdict, up front

**THE QUESTION: does projection plus proper optimizer mechanics reach filter-only's
in-grid quality while keeping the projection's Gate B robustness and near-binary maps?**

**1. No, not at a matched budget, and not on any shape. COMPUTED.** At 40
forward-equivalents, the same 1.0 mm physical filter radius, the same grid-independent
target and the same cold start, projection plus MMA loses in grid to the filter-only
production recipe on **six of six shapes** in J and on **five of six** in IoU at grid 120,
tying on the sixth. The gaps are not marginal: square J 117.91 against 12.30, circle
240.15 against 2.50. **The projection's in-grid cost is not an optimizer artifact.** The
diagnosis in `TOPOPT_REPORT.md` Section 4.3 is measured false at this budget.

**2. The premise behind that diagnosis is itself measured false. COMPUTED.** The claim was
that L-BFGS-B spends *most* of a stage on trial points worse than the incumbent. Counting
only evaluations after the first of each stage, the share that **improved** their own
stage incumbent is **45.7 percent** for projection plus L-BFGS-B at 40 and **50.3 percent**
for projection plus MMA at 40, averaged over the six shapes. So L-BFGS-B wastes a little
over half of its follow-up evaluations, MMA a little under half, and **the mechanics fix
is worth 4.6 percentage points, not a transformation**. For contrast the filter-only arm,
which spends its whole pool in one stage with no beta jumps, improves on **74.0 percent**
of follow-up evaluations. The expensive thing is the continuation, not the line search.

**3. Optimizer at a matched budget: MMA is not a clean win. COMPUTED.** Against
projection plus L-BFGS-B at the same 40 forward-equivalents, MMA improves J on four of six
shapes (square by 23.9 percent, diamond 4.8, rectangle 2.7, triangle 2.0) and is far worse
on two (circle 240.15 against 45.13, trapezoid 141.02 against 61.35). Diagnosed in
Section 5: on those two shapes MMA reaches an excellent low-beta iterate and then destroys
it in the three evaluations the beta 8 and beta 16 stages get, because the box-only MMA
step is proportional to the asymptote distance and not to the gradient magnitude.

**4. At 80 forward-equivalents MMA wins the optimizer comparison on six of six. COMPUTED.**
Against the carried-history L-BFGS-B arm at the same 80, MMA is better on every shape:
triangle by 36.0 percent, trapezoid 29.2, circle 19.2, diamond 7.2, square 2.5, rectangle
2.4. **The mechanics fix is real; it just needs enough evaluations per stage to show.**

**5. Budget, not the optimizer, is the dominant lever. COMPUTED.** Doubling MMA's budget
from 40 to 80 cuts J by 83 percent (circle), 76 (trapezoid), 63 (square), 39 (triangle),
6 (diamond) and 2 (rectangle). That single change is larger than any optimizer difference
measured anywhere in this pass.

**6. But the extra budget is paid for in Gate B robustness and in binariness. COMPUTED.**
Gate B, sub-filter-radius blur, tolerance 10 percent change in J, map space, passes:
projection plus L-BFGS-B at 40 **5 of 6**, projection plus MMA at 40 **5 of 6**, projection
plus MMA at 80 **4 of 6**, projection plus carried L-BFGS-B at 80 **3 of 6**, filter only
at 40 **3 of 6**. Non-discreteness moves the same way: the beta 16 maps at 80
forward-equivalents are markedly less binary (MMA 80 M_nd 0.156 to 0.428) than the same
projection at 40 (L-BFGS-B 0.026 to 0.151). **Spending more budget under the projection
buys in-grid fidelity by re-introducing exactly the fine structure the projection was
adopted to remove.**

**7. Two shapes, and only two, where projection plus MMA at 80 reaches or beats filter-only
in grid. COMPUTED.** Triangle: J 49.26 against 69.93 and IoU 0.8907 against 0.8815, a clear
win at double the budget. Rectangle: J 180.31 against 182.40 and IoU 0.8580 against 0.8580,
a tie inside the noise. On the other four the filter-only arm at HALF the budget is still
better in grid.

**8. Gate A is unchanged by any of this. COMPUTED.** Exactly one shape reaches IoU >= 0.95
at grid 160, the circle, and only on the filter-only arm (0.9855), the prior projection
L-BFGS-B arm (0.9616) and the carried L-BFGS-B arm at 80 (0.9686). Neither MMA arm reaches
the SOLVED class at 160 on any shape. **The square still does not close**: its best grid-160
IoU anywhere in this pass is 0.8202 (filter only), against 0.8116 (carried L-BFGS-B 80),
0.7928 (MMA 40) and 0.7842 (MMA 80).

**9. Nothing is labelled SOLVED.** No map passes Gate A and Gate B together on any arm.

**One-sentence answer.** Proper optimizer mechanics do improve the projection arm, clearly
so at 80 forward-equivalents where MMA beats L-BFGS-B on every shape, but they do not
close the in-grid gap to the filter-only recipe at a matched budget on any of the six
shapes, so **the projection tax is physics and parameterization, not mechanics**, and the
budget that does close part of the gap simultaneously gives back the Gate B robustness and
the near-binary maps that were the projection's only measured advantage.

---

## 2. What was built, and the red that was observed

| file | what it is |
|---|---|
| `fgm_solve_campaign/adjoint2d/mma.py` | the method of moving asymptotes, box constraints only, 300 lines with the simplifications documented in the module docstring |
| `fgm_solve_campaign/adjoint2d/topopt_stage.py` | the optimizer seam: one budgeted stage, three modes, one shared evaluate callable |
| `fgm_solve_campaign/adjoint2d/tests/test_mma.py` | 22 tests, red first |
| `fgm_solve_campaign/adjoint2d/tests/test_topopt_stage.py` | 17 tests, red first |
| `fgm_solve_campaign/adjoint2d/build_mma_tables.py`, `make_mma_figures.py` | tables and the five figures |
| `fgm_solve_campaign/run_mma_arm.sh`, `run_mma_robust.sh` | the drivers |

Modified, additively and with the default path preserved: `adjoint2d/topopt_solve.py`
(optimizer parameter, shared evaluate closure, stem disambiguation) and
`adjoint2d/topopt_robust.py` (the source directory and stem are now explicit and pure).

**Red first, and the red that was observed.** `ImportError: cannot import name 'mma' from
'adjoint2d'` before `mma.py` existed; `ImportError: cannot import name 'topopt_stage'`
before `topopt_stage.py` existed; `AttributeError: module 'adjoint2d.topopt_robust' has no
attribute 'resolve_stem'`; and `KeyError: 'n_optimizer_restarts'` for the re-entry fix of
Section 2.3. **313 tests pass** across the whole `adjoint2d` suite, against a measured
**274** before this pass.

### 2.1 The regression gate that makes the comparison legitimate

The production L-BFGS-B path was refactored to run through the new stage seam. If that
changed anything, every arm comparison in this report would be against a moved baseline.
So the filter-only production recipe was **re-run from scratch on all six shapes** and
compared with the stored `out_topopt` results of the previous pass.

**PROVEN: bit-identical on all six shapes**, both the delivered J and the delivered 4 bpp
map (`np.max(np.abs(map_new - map_old)) == 0.0` exactly, on square, circle, trapezoid,
triangle, diamond and rectangle). The refactor is therefore not a confound, and arm (a) of
the matrix below is simultaneously the baseline and the regression check.

### 2.2 The MMA simplifications, stated in full

With no constraint other than the box, this is **not** full MMA and the differences are
real:

1. **The dual is zero dimensional, not one dimensional.** With m = 0 constraints there is
   no multiplier at all, the artificial variables y and z of Svanberg's formulation vanish,
   and the subproblem
   `min sum_j [ p0_j/(U_j - x_j) + q0_j/(x_j - L_j) ]` over `[alpha, beta]` has the exact
   per-variable closed form
   `x*_j = (sqrt(p0_j) L_j + sqrt(q0_j) U_j)/(sqrt(p0_j) + sqrt(q0_j))`, clipped. No
   Newton iteration, no line search, no dual gradient. Verified against a 200001-point
   brute-force scan of the same scalar model
   (`test_subproblem_solution_matches_a_brute_force_minimization_of_the_same_model`).
2. **The 1987 method, not the 2002 globally convergent variant.** There is no conservative
   inner loop, so **MMA here is not a descent method**: an iterate can be worse than its
   predecessor. Pinned by
   `test_the_iterate_can_overshoot_so_the_caller_must_keep_the_best_iterate`. The solve
   driver keeps the best iterate per stage, exactly as the L-BFGS-B driver already did.
3. **The step tracks the asymptote distance, not the gradient magnitude.** A consequence of
   simplification 1 and the most surprising property of the box-only case: for any variable
   whose gradient is above the raa0 floor, `sqrt(p0/q0)` is close to the fixed
   `sqrt(1.001/0.001) = 31.6` and the closed form lands about 94 percent of the way to the
   asymptote on the descent side, whatever the gradient's size. **MEASURED: scaling the
   gradient by 1e6 changes the step by less than one part in a thousand**
   (`test_the_step_size_tracks_the_asymptote_distance_not_the_gradient_magnitude`). In full
   MMA the constraint multipliers reintroduce magnitude sensitivity; with no constraints
   they cannot. This is the mechanism behind the circle and trapezoid failures of Section 5.
4. **The default asymptote lower clamp floors the achievable accuracy** at about
   `asy_bound_lo` times the box span, one percent at Svanberg's default, because the step
   is proportional to the asymptote distance. MEASURED and pinned by
   `test_the_default_asymptote_clamp_floors_the_accuracy_at_one_percent_of_the_span`.

**Hyperparameters, frozen BEFORE any solve was run and NOT tuned on any result**:
`asy_init 0.5, asy_decr 0.7, asy_incr 1.2, asy_bound_lo 0.01, asy_bound_hi 10.0,
albefa 0.1, raa0 1e-5` are Svanberg's published defaults unchanged; `move 0.2` is the
standard density move limit of the topology-optimization literature. Recorded in every
result file under `mma_config`.

### 2.3 One real bug in new code, found before any result was used

The carried-history L-BFGS-B arm crashed on the triangle with
`TypeError: unsupported format string passed to NoneType.__format__`. The crash was
cosmetic; **what it exposed was not**. scipy declared convergence after **12 of 32 allowed
evaluations, at beta 2**, so the beta 4, 8 and 16 stages received no evaluation at all and
that arm never reached the projection sharpness it was supposed to be compared at
(`logs_mma/triangle_lbfgsb_carry_b80.log`, first attempt, since overwritten by the fixed
run; the reason is recorded in the `topopt_stage._run_lbfgsb_carry` docstring).

The fix, test first: re-enter L-BFGS-B from the current best point until the budget is
spent, and **count the re-entries**, because each one discards the curvature memory this
arm exists to carry. **MEASURED after the fix: 1 re-entry on the square, 2 on the other
five shapes; all six reach beta 16 and spend 28 to 32 of 28 to 32 evaluations.** The whole
arm was killed and re-run from scratch. Reported here because the failure was silent in
every log line except the last.

---

## 3. The finite-difference gate, run before any optimization

**No new gradient was written in this pass.** MMA consumes the identical `dJ/dv` that
`TOPOPT_REPORT.md` gated: the chain
`v --F--> v_f --P_beta--> s --forward--> T --J--> scalar` through
`topopt.design_vjp`, unchanged. The new code path is the OPTIMIZER, and
`test_both_optimizers_consume_the_identical_evaluate_callable` pins by construction that
both optimizers receive the same closure object.

The full four-layer gate was nevertheless re-run on the square through the refactored
code, from scratch, before any arm was scored. It reproduces the earlier pass exactly.

| layer | at 1e-6 | at 1e-5 | worst probe | its analytic derivative |
|---|---|---|---|---|
| P1 target only, dJ/ds | 1/5 | **5/5** | random_direction 7.86e-06 | 0.873 |
| P2 add the filter, dJ/dv | 1/5 | 3/5 | random_direction 1.09e-04 | 0.483 |
| P3 add projection, beta 1 | 1/5 | 3/5 | smooth_random_direction 1.01e-04 | 0.314 |
| **P4 projection, beta 16** | 3/5 | **5/5** | random_cell 8.43e-06 | 0.252 |

P4 is the layer the final continuation stage of every projection arm actually uses.
Transpose exactness, the dot-product identity `<dJ/dv, d> = <dJ/ds, dS(v)[d]>` against the
real adjoint: **3.24e-16 at beta 0, 4.19e-16 at beta 1, 2.31e-16 at beta 16**. Read-state
stability at beta 16, epsilon 1e-3: the argmin index does **not** move (base 900).

**Gate verdict: PASS at the campaign's 1e-5 subgradient standard on the layer the solve
uses (P4, 5 of 5), with the chain rule PROVEN exact to 4.19e-16.** It does NOT clear 1e-6
on all 20 probes, 6 of 20 do, and the misses are localized to small-derivative directions
on the two intermediate layers, which `TOPOPT_REPORT.md` Section 3.2 measured to be a
denominator effect. Honest label: **subgradient PASS with named exceptions**. Wall 6831 s,
`out_mma/gate_topopt_square.json`.

---

## 4. The retest matrix

Every arm: six shapes, filtered at the 1.0 mm physical radius, grid-independent area-fill
chi, conductivity only, single cold start from uniform saturation 1, `enforce_generator_power`
false, budget converted to a gradient-evaluation pool by the library campaign's recorded
adjoint-to-forward ratio (1.782 on the square), stage split by `topopt.stage_split`.

| label | arm | budget | pool, square | stage split |
|---|---|---|---|---|
| a | filter only (beta = 0), L-BFGS-B | 40 | 14 | [14] |
| e | projection, L-BFGS-B, prior pass, READ not re-run | 40 | 14 | [3, 3, 3, 3, 2] |
| b | projection, MMA | 40 | 14 | [3, 3, 3, 3, 2] |
| c | projection, MMA | 80 | 28 | [6, 6, 6, 5, 5] |
| d | projection, L-BFGS-B carried across stages | 80 | 28 | [6, 6, 6, 5, 5] |

Arm (e) is the previous pass's continuation arm, read unchanged from `out_topopt`, and it
is the matched-budget L-BFGS-B counterpart to arm (b). Arm (a) was re-run and is
bit-identical to its stored copy (Section 2.1).

**What arm (d) is and is not.** scipy exposes no way to seed the curvature memory of a
fresh `minimize` call, so the only way to carry history across a beta jump is never to end
the call: beta is switched inside the objective at the stage boundaries while a single
scipy call runs. That is genuine carried history, with the honest caveat that the line
search may be evaluating a different function from the one that produced its direction,
plus the 1 to 2 forced re-entries of Section 2.3, each of which discards the memory.

### 4.1 Grid 120, the deliverable 4 bpp arm at each arm's own J-stop

**J against the area-fill target** (lower is better):

| shape | a filter 40 | e proj L-BFGS-B 40 | b proj MMA 40 | c proj MMA 80 | d proj carry 80 |
|---|---|---|---|---|---|
| square | **12.30** | 154.91 | 117.91 | 43.26 | 44.35 |
| circle | **2.50** | 45.13 | 240.15 | 41.57 | 51.47 |
| trapezoid | **15.82** | 61.35 | 141.02 | 34.18 | 48.27 |
| triangle | 69.93 | 81.92 | 80.25 | **49.26** | 77.02 |
| diamond | **187.53** | 243.96 | 232.13 | 217.52 | 234.43 |
| rectangle | 182.40 | 189.50 | 184.42 | **180.31** | 184.80 |

**IoU against the binary part mask, grid 120**:

| shape | a filter 40 | e proj L-BFGS-B 40 | b proj MMA 40 | c proj MMA 80 | d proj carry 80 |
|---|---|---|---|---|---|
| square | **1.0000** | 0.8734 | 0.8989 | 0.9574 | 0.9572 |
| circle | **0.9968** | 0.9190 | 0.7983 | 0.9375 | 0.9279 |
| trapezoid | **0.9735** | 0.9308 | 0.8465 | 0.9495 | 0.9309 |
| triangle | 0.8815 | 0.8538 | 0.8483 | **0.8907** | 0.8541 |
| diamond | **0.8501** | 0.8115 | 0.8145 | 0.8341 | 0.8127 |
| rectangle | 0.8580 | 0.8558 | **0.8585** | 0.8580 | 0.8558 |

**Growth (melted bed as a percent of part area) and under (unmelted part, same
normalization), grid 120**, at each arm's own stop:

| shape | a growth / under | b growth / under | c growth / under | d growth / under |
|---|---|---|---|---|
| square | 0.00 / 0.00 | 2.62 / 7.75 | 2.62 / 1.75 | 2.12 / 2.25 |
| circle | 0.32 / 0.00 | 13.55 / 9.35 | 3.23 / 3.23 | 2.90 / 4.52 |
| trapezoid | 1.51 / 1.18 | 8.49 / 8.16 | 3.28 / 1.93 | 3.45 / 3.70 |
| triangle | 5.50 / 7.00 | 5.50 / 10.50 | 5.25 / 6.25 | 6.25 / 9.25 |
| diamond | 2.71 / 12.68 | 3.57 / 15.64 | 4.68 / 12.68 | 2.59 / 16.63 |
| rectangle | 12.50 / 3.47 | 12.85 / 3.12 | 12.50 / 3.47 | 13.19 / 3.12 |

**Non-discreteness `M_nd = mean(4 s (1 - s))` over the part**, 0 for a binary map:

| shape | a filter 40 | e proj L-BFGS-B 40 | b proj MMA 40 | c proj MMA 80 | d proj carry 80 |
|---|---|---|---|---|---|
| square | 0.651 | **0.151** | 0.216 | 0.398 | 0.170 |
| circle | 0.690 | **0.026** | 0.114 | 0.428 | 0.179 |
| trapezoid | 0.549 | **0.056** | 0.089 | 0.369 | 0.064 |
| triangle | 0.479 | **0.093** | 0.169 | 0.302 | 0.122 |
| diamond | 0.465 | **0.087** | 0.233 | 0.156 | 0.103 |
| rectangle | 0.323 | **0.068** | 0.135 | 0.270 | 0.104 |

**One horizon flag, on every arm**: the diamond stops on the last stored step at 750.0 s in
all five arms, so every diamond J in this report is an upper bound.

**Zero energy-residual gate violations** on every scored forward run of this pass:
30 solve arms and 18 robustness runs times 9 arms each.

`figs_mma/fig_mma_scoreboard.png`, `figs_mma/fig_mma_maps.png`.

---

## 5. The mechanics, measured rather than argued

`figs_mma/fig_mma_mechanics.png`. Thin line: every evaluation. Thick line: the best WITHIN
the current beta stage, which is what the continuation actually carries forward. A running
minimum across stages would draw a curve no arm ever delivers, and it is deliberately not
plotted.

### 5.1 How much of a stage is wasted, and by whom

Percent of evaluations AFTER the first of each stage that improved their own stage
incumbent (the first is excluded because it is an improvement by definition):

| arm | square | circle | trapezoid | triangle | diamond | rectangle | mean |
|---|---|---|---|---|---|---|---|
| a filter only L-BFGS-B 40 | 76.9 | 92.3 | 78.6 | 53.3 | 78.6 | 64.3 | **74.0** |
| e projection L-BFGS-B 40 | 44.4 | 44.4 | 30.0 | 45.5 | 60.0 | 50.0 | **45.7** |
| b projection MMA 40 | 44.4 | 44.4 | 30.0 | 72.7 | 60.0 | 50.0 | **50.3** |
| c projection MMA 80 | 60.9 | 37.5 | 57.7 | 59.3 | 53.8 | 34.6 | **50.6** |
| d projection carry 80 | 34.8 | 41.7 | 50.0 | 37.0 | 42.3 | 53.8 | **43.3** |

**COMPUTED, and this is the correction to `TOPOPT_REPORT.md` Section 4.3.** L-BFGS-B does
waste slightly more than half of its follow-up evaluations under the projection, so
"most" is barely true, but MMA recovers only 4.6 percentage points of it. The far larger
effect in the table is the projection itself: the same optimizer on the same budget
recovers 74.0 percent when there is no beta continuation (arm a) against 45.7 percent when
there is (arm e). **The waste is caused by the beta jumps, not by the line search.**

### 5.2 What sharpening the projection costs, per stage

Stage-best J relative to the beta = 1 stage, arm b, projection plus MMA at 40:

| shape | beta 1 | beta 2 | beta 4 | beta 8 | beta 16 | delivered |
|---|---|---|---|---|---|---|
| square | 76.5 | 44.3 | 67.5 | 99.9 | 116.4 | 117.91 |
| circle | 130.0 | 29.0 | **14.4** | 122.1 | 246.4 | 240.15 |
| trapezoid | **50.4** | 55.3 | 73.9 | 110.1 | 148.1 | 141.02 |
| triangle | 108.6 | 69.3 | 75.6 | 87.4 | 80.1 | 80.25 |
| diamond | 538.3 | 234.4 | 261.1 | 243.1 | 241.2 | 232.13 |
| rectangle | 184.3 | 184.2 | 182.0 | 180.8 | 184.2 | 184.42 |

**COMPUTED: on the circle, MMA reaches J = 14.4 at beta 4 with IoU 0.9745, better than any
projection arm anywhere in this campaign, and then loses it entirely.** Three evaluations
at beta 8 cannot recover a design that the beta jump has just moved onto the projection's
steep part, and MMA's step is proportional to the asymptote distance rather than to the
gradient, so at high beta it keeps taking large steps where a small one is needed
(simplification 3 of Section 2.2). The trapezoid shows the same shape of failure. Those
two shapes are the entirety of MMA's matched-budget loss.

This is also why the same arm at 80 forward-equivalents recovers: six evaluations per stage
is enough for the asymptotes to contract after a jump. `figs_mma/fig_mma_asymptotes.png`
shows the mean asymptote distance falling monotonically across stage boundaries without
resetting, which is the continuation property the arm was built to have.

---

## 6. Acceptance gate A: the grid hold-out

Solve at grid 120, score at grid 160, drive recalibrated at 160 so the uniform arm absorbs
500 W per metre, target chi rebuilt from the geometry at 160. Both transfers run; the
better of the two is quoted. Recalibrated voltages at 160 are unchanged from the previous
pass: square 2834.8, circle 3783.7, trapezoid 2957.0, triangle 3191.5, diamond 2919.3,
rectangle 3588.8 volts.

| shape | arm | IoU 120 | IoU 160 best transfer | drop | uniform's own move | SOLVED at 160 |
|---|---|---|---|---|---|---|
| square | a filter 40 | 1.0000 | **0.8202** | +0.1798 | -0.0760 | no |
| square | e proj L-BFGS-B 40 | 0.8734 | 0.7833 | +0.0901 | -0.0760 | no |
| square | b proj MMA 40 | 0.8989 | 0.7928 | +0.1061 | -0.0760 | no |
| square | c proj MMA 80 | 0.9574 | 0.7842 | +0.1732 | -0.0760 | no |
| square | d proj carry 80 | 0.9572 | 0.8116 | +0.1456 | -0.0760 | no |
| circle | a filter 40 | 0.9968 | **0.9855** | +0.0113 | +0.1097 | **YES** |
| circle | e proj L-BFGS-B 40 | 0.9190 | 0.9616 | -0.0426 | +0.1097 | **YES** |
| circle | b proj MMA 40 | 0.7983 | 0.8983 | -0.1000 | +0.1097 | no |
| circle | c proj MMA 80 | 0.9375 | 0.9272 | +0.0103 | +0.1097 | no |
| circle | d proj carry 80 | 0.9279 | 0.9686 | -0.0407 | +0.1097 | **YES** |
| trapezoid | a filter 40 | 0.9735 | 0.9104 | +0.0631 | +0.0538 | no |
| trapezoid | e proj L-BFGS-B 40 | 0.9308 | 0.9040 | +0.0268 | +0.0538 | no |
| trapezoid | b proj MMA 40 | 0.8465 | 0.8934 | -0.0469 | +0.0538 | no |
| trapezoid | c proj MMA 80 | 0.9495 | 0.8954 | +0.0541 | +0.0538 | no |
| trapezoid | d proj carry 80 | 0.9309 | **0.9179** | +0.0130 | +0.0538 | no |
| triangle | a filter 40 | 0.8815 | 0.8659 | +0.0156 | +0.0027 | no |
| triangle | e proj L-BFGS-B 40 | 0.8538 | 0.8694 | -0.0156 | +0.0027 | no |
| triangle | b proj MMA 40 | 0.8483 | 0.8644 | -0.0160 | +0.0027 | no |
| triangle | c proj MMA 80 | 0.8907 | **0.8887** | +0.0021 | +0.0027 | no |
| triangle | d proj carry 80 | 0.8541 | 0.8688 | -0.0147 | +0.0027 | no |
| diamond | a filter 40 | 0.8501 | **0.8606** | -0.0104 | +0.1193 | no |
| diamond | e proj L-BFGS-B 40 | 0.8115 | 0.8435 | -0.0320 | +0.1193 | no |
| diamond | b proj MMA 40 | 0.8145 | 0.8500 | -0.0355 | +0.1193 | no |
| diamond | c proj MMA 80 | 0.8341 | 0.8545 | -0.0204 | +0.1193 | no |
| diamond | d proj carry 80 | 0.8127 | 0.8451 | -0.0324 | +0.1193 | no |
| rectangle | a filter 40 | 0.8580 | 0.8891 | -0.0311 | +0.0320 | no |
| rectangle | e proj L-BFGS-B 40 | 0.8558 | 0.8879 | -0.0321 | +0.0320 | no |
| rectangle | b proj MMA 40 | 0.8585 | 0.8895 | -0.0310 | +0.0320 | no |
| rectangle | c proj MMA 80 | 0.8580 | **0.8955** | -0.0375 | +0.0320 | no |
| rectangle | d proj carry 80 | 0.8558 | 0.8893 | -0.0335 | +0.0320 | no |

**Three readings, all COMPUTED.**

1. **The MMA arms do not add a single SOLVED shape at grid 160.** The three shapes that
   reach 0.95 there are all the circle, on arms a, e and d.
2. **The better an arm is in grid, the more it loses across grids, on the square.** Arm c
   is the best square in grid of any projection arm (IoU 0.9574) and the worst at 160
   (0.7842). Arm a is perfect in grid (1.0000) and drops 0.1798. The uniform arm itself
   moves -0.0760 on this shape, so a large part of every square drop is forward
   discretization, not the map.
3. **On four of six shapes the projection arms are grid stable or improve at 160**
   (negative drops on circle b, trapezoid b, triangle b and e, diamond all four, rectangle
   all five). That is inherited from the physical radius and the grid-independent target
   and is unchanged by the optimizer.

---

## 7. Acceptance gate B: sub-filter-radius perturbation

Blur the delivered map by a part-masked normalized-convolution Gaussian at 0.5, 1.0 and
1.5 cells, all strictly below the 1.983-cell filter radius at grid 120, and re-score.
Tolerance 10 percent change in J. The design-space form blurs `v` and re-projects.

| shape | a filter 40 | e proj L-BFGS-B 40 | b proj MMA 40 | c proj MMA 80 | d proj carry 80 |
|---|---|---|---|---|---|
| square | FAIL 24.65 % | **PASS 1.51 %** | FAIL 28.49 % | PASS 9.46 % | FAIL 96.52 % |
| circle | FAIL 264.71 % | PASS 8.98 % | **PASS 0.17 %** | PASS 8.10 % | FAIL 48.63 % |
| trapezoid | FAIL 26.49 % | FAIL 25.40 % | **PASS 4.51 %** | FAIL 11.69 % | FAIL 26.94 % |
| triangle | PASS 3.25 % | PASS 4.62 % | PASS 2.38 % | FAIL 24.38 % | **PASS 1.78 %** |
| diamond | PASS 8.99 % | PASS 1.67 % | **PASS 0.87 %** | PASS 6.58 % | PASS 4.33 % |
| rectangle | PASS 0.67 % | PASS 2.35 % | **PASS 0.12 %** | PASS 3.80 % | PASS 0.16 % |
| **passes** | **3 of 6** | **5 of 6** | **5 of 6** | **4 of 6** | **3 of 6** |

Design-space form, for reference: a 3 of 6, e 5 of 6, b 6 of 6, c 3 of 6, d 3 of 6.

**COMPUTED, and this is the finding that decides the trade.** Gate B robustness is not a
property of the projection alone; it is a property of the projection **at a small budget**.
Both 80 forward-equivalent arms are worse than their own 40-evaluation counterparts, and
the carried L-BFGS-B arm at 80 is as fragile as the filter-only arm (3 of 6, with a
96.52 percent square failure). The maps in `figs_mma/fig_mma_maps.png` show why: arm c's
square, circle and trapezoid maps carry visible interior structure that arms b, d and e do
not, and arm c's non-discreteness is two to four times theirs.

The trapezoid fails on four of five arms, as it did in the previous pass, and this pass
adds nothing to that diagnosis.

---

## 8. Per-shape answer to the question

"Does projection plus proper mechanics reach filter-only's in-grid quality while keeping
Gate B robustness and near-binary maps?" Judged at grid 120 for the in-grid column.

| shape | best projection arm in grid | reaches filter-only in grid? | keeps Gate B? | near binary? | verdict |
|---|---|---|---|---|---|
| square | c MMA 80, J 43.26, IoU 0.9574 | **NO**, filter only is J 12.30, IoU 1.0000 | marginal, 9.46 % | no, M_nd 0.398 | projection loses |
| circle | c MMA 80, J 41.57, IoU 0.9375 | **NO**, filter only is J 2.50, IoU 0.9968 | yes, 8.10 % | no, M_nd 0.428 | projection loses in grid, wins Gate B |
| trapezoid | c MMA 80, J 34.18, IoU 0.9495 | **NO**, filter only is J 15.82, IoU 0.9735 | no, 11.69 % | no, M_nd 0.369 | projection loses |
| triangle | c MMA 80, J 49.26, IoU 0.8907 | **YES**, beats filter only 69.93 / 0.8815 | no, 24.38 % | no, M_nd 0.302 | wins in grid, gives up Gate B |
| diamond | c MMA 80, J 217.52, IoU 0.8341 | **NO**, filter only is 187.53 / 0.8501 | yes, 6.58 % | partly, M_nd 0.156 | projection loses in grid, wins Gate B |
| rectangle | c MMA 80, J 180.31, IoU 0.8580 | **TIE**, filter only is 182.40 / 0.8580 | yes, 3.80 % | partly, M_nd 0.270 | tie in grid, wins Gate B |

**At a matched 40 forward-equivalents the answer is NO on six of six.** At double the
budget it is YES on one (triangle) and a tie on one (rectangle), and on both of those the
Gate B advantage or the near-binary property is given up in the process. **On no shape does
projection plus MMA achieve all three of the things the question asks for at once.**

---

## 9. Cost and wall time

**Logged after two shapes and projected, as required.** The first two arm (a) solves to
complete were the triangle at 860 s and the rectangle at 890 s, mean 875 s, which projected
to six shapes in six parallel single-threaded streams as one round of about 1450 s per
40-forward-equivalent arm and about 2900 s per 80-equivalent arm, so four arms plus 18
robustness runs projected to about 3.4 hours of wall clock. **Actual: 17:46 to 19:54, 2 h
08 min**, inside the projection because the arms overlapped. Nothing was cut.

| stage | runs | process time |
|---|---|---|
| finite-difference gate, square, one thread, concurrent with everything | 1 | 6831 s |
| arm a, filter only L-BFGS-B 40, re-run for the regression gate | 6 | 6555 s |
| arm b, projection MMA 40 | 6 | 6941 s |
| arm c, projection MMA 80 | 6 | 11730 s |
| arm d, projection carried L-BFGS-B 80, after the discarded first attempt | 6 | 10089 s |
| acceptance gates A and B, arms b, c and d | 18 | 6307 s |
| full test suite | 2 | 443 s and 104 s |

Total about **13.6 hours of process time**, about **2 hours of wall clock** on 12 cores
with every numerical library pinned to one thread (`env1.sh`). The first attempt at arm d
was discarded because of the early-convergence bug in Section 2.3, costing about 12 minutes.

Arm (e) is read from the previous pass and its wall times (278 to 453 s) were measured on a
less loaded machine, so **they are not comparable with this pass's** and no
seconds-per-evaluation comparison between arm e and the others is made anywhere in this
report.

---

## 10. Proven, computed, assumed

**PROVEN**
* The refactored production L-BFGS-B path is **bit-identical** to the pre-refactor path on
  all six shapes, in the delivered J and in the delivered 4 bpp map to the last bit.
* The MMA subproblem's closed form is the argmin of the separable model it claims,
  checked against an independent 200001-point scan on six random variables.
* The chain-rule gradient is exactly the unfiltered `dJ/ds` composed with the transpose of
  the linearized parameterization: dot-product identity to **4.19e-16** worst over beta 0,
  1 and 16, against the real adjoint, on the square.
* MMA respects the box under an adversarial gradient, drives an out-of-box optimum onto the
  bound, honours per-variable bounds and a move limit, and returns exactly `x` when the
  gradient is zero.
* 313 tests pass; the 39 new ones were red first, the reds being observed `ImportError`,
  `AttributeError` and `KeyError`.

**COMPUTED**
* Every number in Sections 1 and 3 through 9.
* Zero energy-residual gate violations on every scored forward run.
* The 4.6 percentage-point mechanics improvement, and the 74.0 against 45.7 percent gap
  between the no-continuation and the continuation arms of the same optimizer.
* The circle's beta 4 iterate at J = 14.4, IoU 0.9745, and its loss by beta 16.
* Absorbed power spread 314.9 to 499.1 W per metre over the 30 deliverable arms.

**ASSUMED, and how each one bites**
1. **That the MMA hyperparameters are right.** Frozen at Svanberg's defaults with the
   literature's 0.2 move limit, before any solve, and **not swept**. A different move limit
   or `asy_init` would change the matched-budget result, and the circle and trapezoid
   failures of Section 5.2 are exactly the kind of failure a smaller move limit might fix.
   This is the largest open knob introduced by this pass.
2. **That the beta schedule (1, 2, 4, 8, 16) and the even stage split are right.** Carried
   over unchanged so the arms are comparable. Section 5.2 shows the schedule is where the
   cost is, and a schedule that spends more of the pool at high beta was not tried.
3. **That arm (d) is a fair carried-history arm.** It is carried in the sense that a single
   scipy call spans the stages, but it takes 1 to 2 forced re-entries per shape and each one
   discards the memory. A true warm-started L-BFGS-B would need a reimplementation of the
   two-loop recursion and was not written.
4. **That 80 forward-equivalents is a meaningful "bigger budget".** It is a doubling, chosen
   because `TOPOPT_REPORT.md` Section 12 asked for exactly that. It is not a converged
   budget and every J here remains an upper bound.
5. **That 1.0 mm is the right filter radius.** Unchanged and still **not swept**.
6. **That the 10 percent Gate B tolerance and the Gaussian blur model the physical rim
   uncertainty.** Inherited. No bench measurement of the real rim blur exists.
7. **The arms are conductivity only; every historical arm co-varies permittivity.** The
   standing actuator gap.
8. **The forward is the two-dimensional `adjoint2d` engine, not heatr3d.**
9. **No experimental validation.** `ALLISON_LAW_REPLICATION.md` Section 6.1 records that
   the model over-predicts achievable tuned uniformity by roughly a factor of eight against
   hardware.

---

## 11. Honest limits

1. **Only one shape was finite-difference gated in this pass** (the square, four layers).
   The other five inherit it. The gradient is unchanged from the previous pass, which gated
   the square and the circle.
2. **Arm (e) was not re-run.** It is read from `out_topopt`. Its maps and J are the previous
   pass's, produced by code that is now proven bit-identical on arm (a) but was not
   re-verified on arm (e) itself.
3. **The diamond is at the horizon on every arm**, so all six diamond J values are upper
   bounds and the diamond column of every table should be read as such.
4. **Six shapes, not eighteen.** Nothing here says what the other twelve do.
5. **Single cold start, no multi-start, on every arm.** The rectangle's known stall is
   present and unaddressed: the best rectangle here is J 180.31 against the multi-start
   campaign's 97.19 under the old target.
6. **No arm is dose matched.**
7. **Everything is grid 120 except Section 6**, and the forward is still not IoU converged
   between 120 and 160: the uniform arm alone moves up to 0.1193 IoU points, in both
   directions. A re-solve AT 160 is still the experiment that would separate transfer from
   convergence, and it was still not run.
8. **MMA is the 1987 method with no constraints**, which as Section 2.2 documents degenerates
   into a sign-driven per-variable adaptive trust region. A volume or dose constraint would
   restore the dual and change the method's character, and that is a different experiment.

---

## 12. The single most valuable next layer

**Re-run arm (b), projection plus MMA at 40 forward-equivalents, with a beta schedule that
spends the pool unevenly, front-loaded at low beta.** Section 5.2 measures that on the
square, circle and trapezoid the best iterate of the whole run sits at beta 1, 2 or 4, and
the last two stages only destroy it. A schedule such as [5, 4, 3, 1, 1] costs nothing extra
and tests the hypothesis directly: if the delivered J then approaches the beta 4 value, the
continuation is mis-scheduled rather than the projection being expensive. This is six
solves, about 25 minutes in six streams, and it is the cheapest remaining experiment that
could change the verdict of Section 1.

**Second: sweep the MMA move limit at 0.05, 0.1 and 0.2 on the circle and the trapezoid,**
the two shapes where the matched-budget MMA arm fails. Six solves, about 25 minutes. It is
the one hyperparameter this pass introduced and did not sweep, and Section 5.2's mechanism
predicts a smaller move limit will fix exactly those two.

**Third: re-solve at grid 160.** Still unrun, still named by every report in this
workstream since `SOLVE_ROBUSTNESS_VALIDATION.md`.

---

## 13. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code, under `fgm_solve_campaign/adjoint2d/`: `mma.py`, `topopt_stage.py`,
`build_mma_tables.py`, `make_mma_figures.py`, `tests/test_mma.py`,
`tests/test_topopt_stage.py`. Modified additively: `topopt_solve.py`, `topopt_robust.py`.
Drivers under `fgm_solve_campaign/`: `run_mma_arm.sh`, `run_mma_robust.sh`.

Results, under `fgm_solve_campaign/out_mma/`:
* `gate_topopt_square.json` the four-layer finite-difference gate
* `<shape>_control_filteronly.json` and `_maps.npz`, arm a, re-run and bit-identical
* `<shape>_mma.json`, `<shape>_mma_b80.json`, `<shape>_lbfgsb_carry_b80.json` and their
  `_maps.npz`, arms b, c and d
* `<shape>_mma_robust.json`, `<shape>_mma_b80_robust.json`,
  `<shape>_lbfgsb_carry_b80_robust.json` and their `_robust_maps.npz`, Gates A and B
* `tables.json`, `tables.md`

Logs: `fgm_solve_campaign/logs_mma/*.log`.

Figures, **all five viewed before delivery**, under `fgm_solve_campaign/figs_mma/`:
* `fig_mma_mechanics.png` what one evaluation buys under each optimizer, the measured
  improvement fraction, and what sharpening beta costs
* `fig_mma_scoreboard.png` five arms, six shapes, J, IoU and non-discreteness at grid 120
* `fig_mma_maps.png` the delivered maps, cropped to the part
* `fig_mma_asymptotes.png` MMA's carried asymptote state across the stage boundaries
* `fig_mma_acceptance.png` the two acceptance gates for all five arms

Read, not modified: `TOPOPT_REPORT.md`, `FROZEN_CONVENTIONS_2D.md`,
`fgm_solve_campaign/out_topopt/*.json`, `fgm_solve_campaign/out_lib/*.json`,
`fgm_solve_campaign/out_ms/*.json`.
