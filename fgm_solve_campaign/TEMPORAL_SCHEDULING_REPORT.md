# Temporal power scheduling p(t) as a control on shape fidelity

Adjoint-gated study of a piecewise-constant radio-frequency (RF) generator power schedule
p(t) as a design variable alongside the dopant map, scored on the whole-domain
shape-fidelity objective at each arm's own optimal stop.

Everything below was produced in this worktree under
`.venv312/bin/python`. Code: `adjoint2d/schedule.py`, `adjoint2d/sched_solve.py`,
`adjoint2d/sched_warmstart.py`, `adjoint2d/sched_dose_control.py`,
`adjoint2d/sched_offperiod.py`, `adjoint2d/gate_sched.py`,
`adjoint2d/make_sched_figures.py`. Data: `out_sched/`. Figures: `figs_sched/`.

---

## 1. Verdicts, one line per shape

Fidelity is reported as **IoU, the intersection over union** of the melted region with the
nominal part, evaluated at that arm's own optimal stop.

| shape | verdict |
|---|---|
| **cross** | **SCHEDULING HELPS, AND THE GAIN IS TEMPORAL STRUCTURE, NOT DOSE.** Warm-started co-optimized 4 bits-per-pixel map plus schedule reaches **IoU 0.7504** against the library baseline's 0.6766 (**+0.0738**, J **-30.5 percent**), with mean relative density **0.941** against the baseline's 0.833. A dose-matched constant-power control on the same map at the same duty cycle reaches only 0.7091, so **+0.0413 of the +0.0738 is temporal structure and +0.0000 is dose.** The schedule contains a real 211 s generator OFF period. |
| **T_shape** | **NULL ON TEMPORAL STRUCTURE; THE SMALL GAIN IS DOSE.** Deliverable IoU 0.4680 against baseline 0.4444 (+0.0235), but the dose-matched control on the same map reaches 0.4654, so **structure buys +0.0026 and dose buys +0.0192**. The optimizer's only move is to sit on the 1.5 power ceiling. Binary on/off returns exactly all-ones. The T crossbar never melts in any arm. |
| **L_shape** | **NULL ON TEMPORAL STRUCTURE; THE GAIN IS DOSE.** Deliverable IoU 0.5554 against baseline 0.5209 (+0.0345); dose-matched control 0.5404, so **structure +0.0150, dose +0.0304**. Same ceiling-seeking behaviour. Binary on/off returns exactly all-ones and is 0.0244 IoU WORSE than the unscheduled baseline. |
| **star** | **CLEAN NULL, TO FOUR DECIMAL PLACES.** Every optimized schedule is identically p = 1 on every segment. Deliverable IoU 0.7046 against baseline 0.7032 (+0.0014), which is entirely the re-optimized map and zero the schedule. Dose-matched control differs by 0.0000. Rounding loss 0.0001 percent. |

Deliverable figure per shape: `figs_sched/fig_sched_<shape>.png`.
Census: `figs_sched/fig_sched_census.png`.
Mechanism: `figs_sched/fig_sched_offperiod_cross_WARM_CO_4bpp.png`.

---

## 2. What was inherited and what is new

The previous agent left `schedule.py`, `sched_solve.py`, `gate_sched.py` and four test
files in place, with one finite-difference gate already run on the cross at a schedule
window equal to the march horizon. All 80 of its tests passed on arrival, and its gate
result reproduced. Nothing that passed its tests was rebuilt.

New in this pass, each with a test written red first:

1. **`schedule.expand_full` and the fix to `schedule.accumulate_to_segments`.**
   A latent gradient bug. When the schedule window is shorter than the march, the forward
   clamps and keeps heating at the LAST segment's level, so those trailing steps do depend
   on p[-1]; `accumulate_to_segments` dropped them. Pinned by
   `test_accumulate_folds_the_post_window_tail_into_the_last_segment`, observed red at
   `assert 30.0 == 50.0`, then fixed and re-gated (section 3).
2. **`schedule.place_then_hold`**, a structure classifier that reads only the segments the
   march actually reached, so an unconstrained tail past the stop cannot be mistaken for a
   discovered structure.
3. **`sched_solve.iso_j_hold_gain`**, the direct densification-versus-fidelity probe.
4. **`sched_warmstart.py`**, the warm-started arm (section 6).
5. **`sched_dose_control.py`**, the dose-matched control (section 5).
6. **`sched_offperiod.py`**, the OFF-period mechanism diagnosis (section 7).
7. Mean relative density and schedule structure recorded on every arm.

Test suite: **91 passed** (`.venv312/bin/python -m pytest adjoint2d/tests -q`).

> Operational note, not a result: this machine was running other agents' jobs at load
> average 262 during this work. With default multi-threaded BLAS the test suite went from
> 34 s to unbounded. Every run reported here was made with `OMP_NUM_THREADS=1` and the
> other thread-count environment variables set to 1.

---

## 3. Finite-difference gate on dJ/dp_k, run BEFORE any optimization

Central differences, epsilon swept 1e-3 down to 1e-8, three probes: the maximum-sensitivity
segment (argmax of the analytic gradient magnitude), a fixed pseudo-random segment (seed 11),
and a random unit direction over all sixteen segments. Two layers: P1 at a FIXED stop index,
and P2 with the stop taken as the argmin of J on the arm's own trajectory. The gate point is
deliberately away from nominal: a non-uniform dopant map and a schedule with a ramp and one
forced dip to 0.35.

Pass threshold 1e-6 on the best relative error over the epsilon sweep.

| shape | march steps | schedule window | max-sensitivity segment | random segment | random direction | verdict |
|---|---|---|---|---|---|---|
| cross | 2500 | 2500 (= march, inherited) | 5.290e-09 | 3.513e-09 | 2.578e-09 | PASS |
| **cross** | **2500** | **300 (tail dominant)** | **1.304e-09** | **1.114e-08** | **1.635e-07** | **PASS** |
| cross | 2500 | 2250 (production) | 1.699e-08 | 1.565e-07 | 5.847e-08 | PASS |
| T_shape | 1500 | 700 (production) | 3.872e-10 | 1.651e-08 | 2.396e-10 | PASS |
| L_shape | 1500 | 800 (production) | 1.507e-08 | 1.326e-08 | 3.351e-08 | PASS |
| star | 1500 | 500 (production) | 3.453e-10 | 3.453e-10 | 1.619e-09 | PASS |

Raw: `out_sched/gate_sched_*.json`, logs `out_sched/gate_sched_window.log`, `out_sched/g_*.log`.

Three things this table establishes.

* **The tail term is real and is now correct.** In the window-300 row, 2200 of the 2500
  march steps run in the post-window hold. Before the fix, segment 15 carried zero gradient
  from those steps. After the fix all sixteen segments carry non-zero gradient
  (`dJdp[15] = +8.1`, opposite in sign to every other segment) and the gate passes at
  1.30e-09.
* **The envelope argument holds for the temporal actuator.** The P1 fixed-stop and P2
  optimal-stop gradients agree to a relative difference of **exactly 0.000e+00** on every
  shape. There is no dt*/dp term to add. This mirrors what `gate_shape.py` already showed
  for the dopant map.
* **The melt-front subgradient floor of about 1e-5 does NOT apply here.** That floor appears
  in the single-cell dopant-map gate (`out_sched/gate_shape_regate.log` shows a
  random-direction 1.220e-05 on the square) because moving one dopant cell perturbs a small
  number of cells sitting exactly on the phase ramp. Perturbing a power segment moves every
  cell in the domain simultaneously, so the clipped melt fraction phi = clip((T - T_pc)/dT_pc + 0.5, 0, 1)
  keeps a large smooth interior population and the non-smooth boundary contribution is a
  vanishing share. Every schedule probe bottoms between 3.5e-10 and 1.6e-07.

The gate was run before any optimization for every shape and every window used in
production. No arm was optimized on an ungated gradient.

---

## 4. Conventions, stated once

**Objective.** J(s, p, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2,
where phi is the melt fraction and chi_part the binary nominal part indicator. Under-melting a
part cell and melting a bed cell cost the same.

**Stop.** t_stop = argmin over that arm's own trajectory of J. Per-arm, always. The
shape-fidelity early stop (`shape_stop_patience`) is DISABLED on every arm including the
baselines, because a schedule with an OFF stretch makes J rise and would trip the patience
counter before a later ON stretch could act. Every march runs the full horizon. Baselines
are therefore RE-SCORED here rather than carried across from
`SHAPE_LIBRARY_SOLVE_REPORT.md`.

**Injection convention, the square-root-voltage statement.** p multiplies POWER. The
electro-quasi-static problem is linear in the potential at fixed material properties, so
scaling the drive voltage by sqrt(p) scales the electric field by sqrt(p) and the volumetric
heating Q_rf = 0.5 * power_factor * sigma * |E|^2 by exactly p. Multiplying Q_rf by p is
therefore not an approximation of a voltage change, it IS the change V -> V * sqrt(p), with
no electro-quasi-static re-solve needed. A commanded level of p = 1.5 means a drive voltage
1.225 times nominal. The `max_qrf` cap is applied AFTER the scaling, which is where it would
act on a real re-solve, and its subgradient mask is carried into the adjoint per step.

**Horizon.** 1500 outer steps (750 s at dt = 0.5 s) except the **cross, which is run to 2500
steps (1250 s)**. The library run's cross J-minimum sat on its 1500-step horizon and was
therefore a bound; measured here, the library 4-bit cross map has its minimum at step 1505,
just past the old horizon, so the extension was necessary and the optimum is now interior.
**Every cross number in this report is at the 2500-step horizon and is NOT comparable to any
cross number in `SHAPE_LIBRARY_SOLVE_REPORT.md`.**

**Schedule window (a decision made in this pass, with a measured reason).** The sixteen
segments span a WINDOW, not the whole march; past the window the generator holds the last
segment's level. Laying the segments over the whole march is what the inherited code did,
and it is a bad control parameterization here: at the uniform-map full-power stop, only 3 of
16 segments sit before the stop, so 13 carry exactly zero gradient. The window is set to
1.5 times the LATER of the uniform-map stop and the library-baseline stop, measured once per
shape before any optimization:

| shape | uniform stop | baseline stop | window (steps) | window (s) | segment length |
|---|---|---|---|---|---|
| cross | 439 | 1505 | 2250 | 1125 | 140.6 steps = 70.3 s |
| T_shape | 460 | 433 | 700 | 350 | 43.75 steps = 21.9 s |
| L_shape | 526 | 525 | 800 | 400 | 50 steps = 25 s |
| star | 284 | 327 | 500 | 250 | 31.25 steps = 15.6 s |

**Melt criterion.** melted = melt fraction >= 0.5. **Density.** mean relative density rho in
part cells; rho is NOT in the objective and is reported as a diagnostic only.

**Energy-residual gate.** On for every arm and every number in this report. Worst
relative residual across all 4 shapes and 20 arms is **1.34 percent** (cross warm arms);
**zero violations**.

**Budget.** Forward-equivalents, converted to gradient evaluations using a per-run measured
adjoint-to-forward cost ratio: 40 for the co-optimization (the deliverable), 20 for the
schedule-alone arm, 15 for each re-scheduling pass, so **90 forward-equivalents per shape per
start**, spent twice (cold and warm) plus 5 for the dose control. This is more than a literal
reading of "40 forward-equivalents per shape" and is stated rather than hidden. No arm was
silently cut. One wart: because the ratio is measured on a loaded machine it drifted, so the
cross warm run received 17 co-optimization evaluations where the cold run received 29. The
warm run won anyway, which only strengthens the conclusion in section 6.

**Wall time.** 6887 s = 1.91 h total for all optimization and control runs, single-threaded,
on a machine at load average 80 to 260. Per run: cross 1282 s cold and 858 s warm; star 575 s
and 520 s; T_shape 221 s and 229 s; L_shape 230 s and 232 s; the two cross segment-count
sensitivity runs 1178 s and 1040 s; the eight dose-control runs 523 s combined.

---

## 5. The dose confound, and the control that removes it

The continuous schedule box is [0, 1.5]. An optimizer that simply raises every segment to
the ceiling buys extra energy, not temporal structure, and would report a dose result wearing
a scheduling label. That is exactly what three of the four shapes did.

The control is one extra forward per arm: take the arm's optimized schedule, read its
time-weighted duty cycle d over the window, and re-score the SAME dopant map with a CONSTANT
schedule p == d, which delivers the same time-integrated power scale with no temporal
structure at all. A second control at constant p == 1 gives the nominal-power reference on
the same map, so the total improvement decomposes three ways: map, dose, structure.

| shape | arm | duty | scheduled IoU | constant at duty | constant at 1 | **structure** | **dose** |
|---|---|---|---|---|---|---|---|
| **cross** | **WARM_CO_4bpp** | **1.145** | **0.7504** | **0.7091** | **0.7091** | **+0.0413** | **+0.0000** |
| cross | WARM_sched_only | 1.094 | 0.7035 | 0.6755 | 0.6766 | +0.0280 | -0.0011 |
| cross | CO_4bpp (cold) | 0.883 | 0.6522 | 0.6088 | 0.6131 | +0.0434 | -0.0044 |
| cross | BIN_round (cold) | 0.812 | 0.6176 | 0.6295 | 0.6131 | -0.0118 | +0.0164 |
| T_shape | WARM_CO_4bpp | 1.313 | 0.4680 | 0.4654 | 0.4462 | +0.0026 | +0.0192 |
| L_shape | WARM_CO_4bpp | 1.344 | 0.5554 | 0.5404 | 0.5100 | +0.0150 | +0.0304 |
| star | WARM_CO_4bpp | 1.000 | 0.7046 | 0.7046 | 0.7046 | +0.0000 | +0.0000 |

Full table `out_sched/*_dose_control.json`.

**Read this way.** On the cross, dose contributes nothing and the entire scheduling gain is
temporal structure. On T_shape and L_shape it is the other way round: the schedule is a power
boost with a name, and if the generator cannot exceed nominal power there is nothing there.
On the star the schedule does not move at all.

The binary arms are the independent confirmation. The binary box is [0, 1], so the ceiling
move is unavailable by construction. On T_shape, L_shape and star the binary relaxation
returns **exactly all-ones with zero level changes**, that is, the optimizer's answer inside
[0, 1] is "never turn off". Only the cross uses OFF segments.

---

## 6. Cold start against warm start

The cold-start co-optimized arm begins from a uniform dopant map, while the baseline it is
compared against is the library's own solved map produced under a much larger budget. A cold
start losing under 40 forward-equivalents is a statement about the budget, not about
co-optimization. `sched_warmstart.py` removes that confound by starting the map at the
library deliverable.

| shape | baseline IoU | cold-start deliverable | warm-start deliverable |
|---|---|---|---|
| **cross** | 0.6766 | **0.6522 (loses, -0.0244)** | **0.7504 (wins, +0.0738)** |
| T_shape | 0.4444 | 0.4680 | 0.4680 |
| L_shape | 0.5209 | 0.5529 | 0.5554 |
| star | 0.7032 | 0.7046 | 0.7046 |

On the cross the cold-start arm is not merely worse, it reverses the verdict from
"scheduling helps" to "scheduling hurts". The three shapes whose gain is dose are insensitive
to the start, because saturating the power ceiling is a move the optimizer finds from
anywhere. **The deliverable arm reported in section 1 is the warm-started one, and the
cold-start number is kept in the tables as the honest measure of what 40 forward-equivalents
from scratch buys.**

Comparison to prior work on the cross: the best cross arm previously recorded is the
orientation study's graded map at 30 degrees, IoU 0.749 with an interior stop at 512 s at the
OLD 1500-step horizon. The warm-started scheduled cross here reaches 0.7504 at the 2500-step
horizon. **These two numbers are at different horizons and are not a like-for-like
comparison.** No orientation arm was re-run in this pass. What can be said without a caveat
is that scheduling on the fixed 0-degree orientation reaches a fidelity in the same band as
the best orientation result, by a completely different lever.

---

## 7. Place-then-hold: what the optimizer actually discovered

The triangle showcase found mean relative density only 0.684 at the shape-optimal stop,
because densification lags the melt front. Two independent checks were run.

### 7.1 Does the optimizer discover a place-then-hold structure?

`schedule.place_then_hold` splits the ACTIVE portion of the schedule at the point of maximum
time-weighted level drop and classifies. Segments entirely after the stop are excluded, so an
unconstrained tail cannot be mistaken for structure.

**Cross: yes, on every scheduled arm, at every segment count tested.** T_shape, L_shape and
star: no, every arm classifies FLAT, and inspection of the schedules confirms it (they are
1.5 held to the ceiling, with the one step down falling after the stop).

Mean relative density at each arm's own stop, cross:

| arm | IoU | mean rho at stop |
|---|---|---|
| uniform, no schedule | 0.5465 | 0.627 |
| library 4-bit baseline, no schedule | 0.6766 | 0.833 |
| schedule added to baseline map | 0.7035 | 0.907 |
| **warm co-optimized 4-bit + schedule** | **0.7504** | **0.941** |
| binary on/off, cold start | 0.6176 | 0.704 |

**On the cross the scheduled deliverable improves shape fidelity and density together**, from
0.833 to 0.941 mean relative density. That is the direct answer to the triangle showcase's
concern for this shape: the lag is not an unavoidable cost, and a schedule that stops later
at a lower average power closes most of it. On the other three shapes the deliverable's
density is FLAT or slightly LOWER than the baseline's (T_shape 0.597 against 0.608, L_shape
0.642 against 0.651, star 0.606 against 0.606), because their schedules stop EARLIER at
higher power.

### 7.2 Is there iso-fidelity densification headroom left at the stop?

`iso_j_hold_gain` walks forward from the J-optimal stop while J stays inside a 2 percent
band and reports the densest time reached. This is a clean null on all four shapes:

| shape | rho at stop | best rho inside the J band | gain | extra steps |
|---|---|---|---|---|
| cross | 0.941 | 0.942 | +0.0017 | 38 |
| T_shape | 0.597 | 0.608 | +0.0118 | 16 |
| L_shape | 0.642 | 0.661 | +0.0186 | 20 |
| star | 0.606 | 0.615 | +0.0093 | 11 |

**At most 0.019 of relative density is available by simply dwelling past the shape-optimal
stop.** The place-then-hold structure the cross found is therefore not a dwell; it is an
active reshaping of the melt front, and its density benefit comes from a longer total process
at lower mean power, not from waiting at the end.

---

## 8. The OFF period, and what it actually does

The warm-started cross deliverable commands, over a 2250-step window at 16 segments:

```
segments  0 to 10   level 1.50    t =    0.0 to  843.8 s
segment      11     level 0.32    t =  843.8 to  914.1 s
segments 12 to 14   level 0.00    t =  914.1 to 1054.7 s     GENERATOR OFF
segment      15     level 1.50    t = 1054.7 to 1125.0 s
post-window hold    level 1.50    t = 1125.0 to 1250.0 s
STOP at t = 1138.5 s (step 2277), which is 27 steps INTO the hold
```

Only 3 level changes, and the OFF stretch is 211 s including segment 11's near-off level.
This is directly realizable as generator instructions.

The pre-registered hypothesis was DIFFERENTIAL COOLING: with the generator off, melted powder
outside the part falls back below the melt threshold faster than the part interior does, so
the OFF period acts as a selective eraser. **That hypothesis is REFUTED by measurement.**

| leg | window | bed melt (percent of part cells) | part unmelted (percent) |
|---|---|---|---|
| ON, full power | 0 to 844 s | 0.0 to **30.3** | 100.0 to **9.7** |
| **OFF** | 844 to 1055 s | 30.3 to **10.6 (-19.7)** | 9.7 to **29.3 (+19.7)** |
| RE-HEAT, full power | 1055 to 1138 s | 10.6 to **16.0 (+5.4)** | 29.3 to **12.9 (-16.4)** |

Across the OFF period the ratio of bed melt removed to part melt lost is **1.00 to 1**. That
is not selective at all. The part cools 19.4 C and the melted bed ring cools 15.1 C, so if
anything the bed cools slightly SLOWER.

**The selectivity is in the RE-HEAT leg, at 3.04 to 1**: the part recovers 3.04 points of
melt for every point the bed regains. The reason is the dopant. The part is doped and
absorbs RF power volumetrically; the bed is undoped powder and can only be reheated by
conduction from the part. So the correct mechanism name is **non-selective cooling followed
by selective re-heat**, a melt, erase and selectively re-melt cycle. The OFF period is the
erase step, and it is a blunt instrument on its own; what makes the cycle profitable is that
the dopant makes the re-melt sharp.

Net effect of the whole cycle against stopping at the end of the first ON leg: bed melt
30.3 to 16.0 (14.3 points removed) at a cost of 9.7 to 12.9 part unmelted (3.2 points). That
is the +0.0413 IoU of structure the dose control isolated.

Evidence: `figs_sched/fig_sched_offperiod_cross_WARM_CO_4bpp.png` (temperature field pair at
the first and last step of the OFF period, with the melt front dashed and the nominal outline
in cyan; at OFF start the front bulges well past the cross into the bed, at OFF end it has
retreated INSIDE the cross outline), and `out_sched/cross_warm_WARM_CO_4bpp_offperiod.json`.

**The cold-start cross binary arm has OFF periods too, and they are NOT this mechanism.**
Its longest OFF run is 282 to 563 steps (141 to 282 s), and at its start bed melt is 0.0
percent and the part is 100.0 percent unmelted: nothing had melted yet, so there was nothing
to erase. That arm's OFF period is a **delay, not an erase cycle**, and no selectivity can be
read from it. Two guards in `sched_offperiod.py` produce this distinction rather than
papering over it: the labelled OFF period is the LONGEST CONTIGUOUS zero run (taking first
start to last end would have counted ON stretches inside it), and the re-heat selectivity is
suppressed entirely when nothing had melted when the generator went off. Figure:
`figs_sched/fig_sched_offperiod_cross_BIN_round.png`.

---

## 9. Binary on/off schedules and the rounding loss

The binary class was solved as a continuous relaxation inside the box [0, 1] and then rounded
onto {0, 1} at a threshold of 0.5. The rounding loss is reported as a labelled number.

| shape | J relaxed | J rounded | delta J | relative loss | level changes after rounding |
|---|---|---|---|---|---|
| cross | 391.99 | 390.95 | -1.04 | **-0.265 percent (rounding HELPED)** | 4 |
| T_shape | 609.35 | 609.35 | 0.00 | 0.000 percent | 0 |
| L_shape | 538.32 | 538.32 | 0.00 | 0.000 percent | 0 |
| star | 156.75 | 156.75 | +0.00 | +0.0001 percent | 0 |

**The rounding loss is negligible on all four shapes**, and on the cross it is slightly
negative, which the objective's non-convexity permits and which the reporting function does
not clamp away. The reason the loss is so small is that the relaxed solutions are already
nearly binary: the cross relaxation sits at 1.0 or 0.0 on 15 of 16 segments, and the other
three shapes' relaxations are exactly all-ones.

**Honest limit.** The relaxation is not a certificate. Rounding a relaxed optimum gives a
feasible binary schedule and its true objective, but says nothing about the best binary
schedule. No branch-and-bound or exhaustive check over the 2^16 binary schedules was run.
The claim is "this binary schedule scores this well", not "this is the optimal binary
schedule".

Separately, the binary class LOSES to the unscheduled baseline on L_shape (0.4965 against
0.5209) because that shape needs MORE power than nominal and the binary box forbids it; on
T_shape it is a small win (0.4591 against 0.4444) that comes entirely from re-scoring under
the disabled early stop, since its schedule is all-ones. On the cross the binary arm at duty
0.812 reaches IoU 0.6176 from a cold start, which is below both the cold-start continuous
arm (0.6522) and the warm-start continuous deliverable (0.7504), and its dose control shows
its structure contribution is NEGATIVE (-0.0118). **Binary on/off did not beat continuous
scheduling anywhere.**

---

## 10. Segment-count sensitivity, cross

Eight and thirty-two segments were run against the production sixteen, on the cross, with
everything else identical (2250-step window, 2500-step march, same budgets).

| segments | arm | J | IoU | duty | mean rho | stop (s) | structure |
|---|---|---|---|---|---|---|---|
| 8 | schedule only | 404.55 | 0.5929 | 0.886 | 0.757 | 423.5 | place then hold |
| 16 | schedule only | 424.11 | 0.5874 | 0.930 | 0.742 | 357.5 | place then hold |
| 32 | schedule only | 411.08 | 0.5955 | 0.997 | 0.741 | 253.5 | place then hold |
| 8 | co-optimized 4 bit | 394.64 | 0.6151 | 0.901 | 0.743 | 429.0 | place then hold |
| 16 | co-optimized 4 bit | 367.99 | 0.6522 | 0.883 | 0.792 | 518.5 | place then hold |
| 32 | co-optimized 4 bit | 351.95 | 0.6514 | 1.016 | 0.733 | 516.5 | place then hold |
| 8 | binary rounded | 430.77 | 0.5698 | 0.875 | 0.618 | 413.0 | place then hold |
| 16 | binary rounded | 390.95 | 0.6176 | 0.812 | 0.704 | 572.0 | place then hold |
| 32 | binary rounded | 353.67 | 0.6495 | 0.937 | 0.728 | 609.0 | ramp up |

**The place-then-hold structure is not an artifact of the segment count**: it appears at 8,
16 and 32 segments on both the schedule-only and the co-optimized arms. Fidelity spreads
0.5874 to 0.5955 across segment counts on the schedule-only arm (0.008 IoU) and 0.6151 to
0.6522 on the co-optimized arm (0.037 IoU, with 8 the weakest). Sixteen is adequate and
thirty-two buys nothing on the continuous arms. The binary arm is the one that keeps
improving with resolution (0.5698 to 0.6176 to 0.6495), which is expected: a binary control
can only approximate a level by switching, so it needs more switches to do so. All numbers
here are cold-start.

---

## 11. Per-arm tables

Read: J is the whole-domain shape objective; IoU, part unmelted and bed growth are at that
arm's own stop; duty is time-weighted over the schedule window; rho is mean relative density
in the part at the stop. `WARM_` arms start from the library 4-bit map.

### cross (horizon 2500 steps = 1250 s, window 2250 steps)

| arm | J | IoU | part unmelted % | bed growth % | stop s | duty | switches | mean rho | structure |
|---|---|---|---|---|---|---|---|---|---|
| U_uniform | 471.82 | 0.5465 | 43.2 | 3.9 | 220.0 | 1.000 | 0 | 0.627 | no schedule |
| BASE_map4bpp | 360.15 | 0.6766 | 20.8 | 17.0 | 753.0 | 1.000 | 0 | 0.833 | no schedule |
| SCHED_only | 424.11 | 0.5874 | 39.0 | 3.9 | 357.5 | 0.930 | 4 | 0.742 | place then hold |
| CO_cont | 390.44 | 0.6311 | 32.6 | 6.8 | 480.0 | 0.881 | 5 | 0.740 | place then hold |
| CO_4bpp | 367.99 | 0.6522 | 30.5 | 6.6 | 518.5 | 0.883 | 6 | 0.792 | place then hold |
| BIN_relax | 391.99 | 0.6199 | 35.1 | 4.6 | 576.5 | 0.808 | 5 | 0.696 | place then hold |
| BIN_round | 390.95 | 0.6176 | 35.1 | 5.0 | 572.0 | 0.812 | 4 | 0.704 | place then hold |
| WARM_sched_only | 344.85 | 0.7035 | 13.9 | 22.4 | 733.5 | 1.094 | 7 | 0.907 | place then hold |
| WARM_CO_cont | 249.92 | 0.7504 | 12.9 | 16.0 | 1139.0 | 1.145 | 3 | 0.941 | place then hold |
| **WARM_CO_4bpp** | **250.20** | **0.7504** | **12.9** | **16.0** | **1138.5** | **1.145** | **3** | **0.941** | **place then hold** |

### T_shape (horizon 1500 steps = 750 s, window 700 steps)

| arm | J | IoU | part unmelted % | bed growth % | stop s | duty | switches | mean rho | structure |
|---|---|---|---|---|---|---|---|---|---|
| U_uniform | 614.26 | 0.4574 | 52.4 | 4.2 | 230.5 | 1.000 | 0 | 0.623 | no schedule |
| BASE_map4bpp | 609.29 | 0.4444 | 54.3 | 2.7 | 217.0 | 1.000 | 0 | 0.608 | no schedule |
| SCHED_only | 580.95 | 0.4662 | 52.5 | 1.8 | 131.5 | 1.344 | 1 | 0.596 | flat |
| CO_cont | 575.34 | 0.4680 | 52.4 | 1.8 | 133.0 | 1.344 | 1 | 0.596 | flat |
| CO_4bpp | 575.39 | 0.4680 | 52.4 | 1.8 | 133.0 | 1.344 | 1 | 0.596 | flat |
| BIN_relax | 609.35 | 0.4591 | 52.2 | 4.2 | 232.5 | 1.000 | 0 | 0.622 | flat |
| BIN_round | 609.35 | 0.4591 | 52.2 | 4.2 | 232.5 | 1.000 | 0 | 0.622 | flat |
| WARM_sched_only | 579.66 | 0.4680 | 52.4 | 1.8 | 131.5 | 1.313 | 1 | 0.596 | flat |
| WARM_CO_cont | 575.07 | 0.4680 | 52.4 | 1.8 | 133.0 | 1.313 | 1 | 0.597 | flat |
| **WARM_CO_4bpp** | **575.01** | **0.4680** | **52.4** | **1.8** | **133.0** | **1.313** | **1** | **0.597** | **flat** |

### L_shape (horizon 1500 steps = 750 s, window 800 steps)

| arm | J | IoU | part unmelted % | bed growth % | stop s | duty | switches | mean rho | structure |
|---|---|---|---|---|---|---|---|---|---|
| U_uniform | 538.42 | 0.5142 | 44.6 | 7.8 | 263.5 | 1.000 | 0 | 0.657 | no schedule |
| BASE_map4bpp | 521.20 | 0.5209 | 44.6 | 6.4 | 263.0 | 1.000 | 0 | 0.651 | no schedule |
| SCHED_only | 502.42 | 0.5262 | 45.0 | 4.4 | 151.0 | 1.344 | 1 | 0.620 | flat |
| CO_cont | 476.77 | 0.5534 | 41.4 | 5.8 | 169.5 | 1.344 | 1 | 0.640 | flat |
| CO_4bpp | 476.60 | 0.5529 | 41.4 | 5.9 | 169.5 | 1.344 | 1 | 0.640 | flat |
| BIN_relax | 538.32 | 0.4965 | 46.7 | 7.3 | 275.5 | 1.000 | 0 | 0.655 | flat |
| BIN_round | 538.32 | 0.4965 | 46.7 | 7.3 | 275.5 | 1.000 | 0 | 0.655 | flat |
| WARM_sched_only | 486.38 | 0.5520 | 41.4 | 6.1 | 164.0 | 1.344 | 1 | 0.639 | flat |
| WARM_CO_cont | 475.44 | 0.5562 | 40.9 | 6.3 | 168.0 | 1.344 | 1 | 0.642 | flat |
| **WARM_CO_4bpp** | **475.42** | **0.5554** | **41.0** | **6.3** | **168.0** | **1.344** | **1** | **0.642** | **flat** |

### star (horizon 1500 steps = 750 s, window 500 steps)

| arm | J | IoU | part unmelted % | bed growth % | stop s | duty | switches | mean rho | structure |
|---|---|---|---|---|---|---|---|---|---|
| U_uniform | 192.58 | 0.6517 | 30.3 | 7.0 | 142.5 | 1.000 | 0 | 0.597 | no schedule |
| BASE_map4bpp | 157.39 | 0.7032 | 26.6 | 4.4 | 164.0 | 1.000 | 0 | 0.606 | no schedule |
| SCHED_only | 192.56 | 0.6517 | 30.3 | 7.0 | 142.5 | 1.000 | 3 | 0.597 | flat |
| CO_cont | 156.59 | 0.7107 | 26.6 | 3.3 | 164.0 | 1.000 | 4 | 0.606 | flat |
| CO_4bpp | 156.75 | 0.7046 | 26.9 | 3.7 | 163.5 | 1.000 | 4 | 0.606 | flat |
| BIN_relax | 156.75 | 0.7046 | 26.9 | 3.7 | 163.5 | 1.000 | 2 | 0.606 | flat |
| BIN_round | 156.75 | 0.7046 | 26.9 | 3.7 | 163.5 | 1.000 | 0 | 0.606 | flat |
| WARM_sched_only | 157.39 | 0.7032 | 26.6 | 4.4 | 164.0 | 1.000 | 1 | 0.606 | flat |
| WARM_CO_cont | 156.79 | 0.7071 | 26.9 | 3.3 | 163.0 | 1.000 | 0 | 0.605 | flat |
| **WARM_CO_4bpp** | **156.97** | **0.7046** | **26.9** | **3.7** | **163.0** | **1.000** | **1** | **0.606** | **flat** |

Note on the star's SCHED_only row: the optimizer records 3 level changes but the duty cycle
is exactly 1.000 and J moves by 0.02, so those are sub-thousandth perturbations that the
optimizer (limited-memory Broyden-Fletcher-Goldfarb-Shanno with bounds, L-BFGS-B) made and
did not keep. The star is a null.

The two design variables are solved in ALTERNATING blocks rather than as one joint vector,
in the order power, map, power, map. This is deliberate: dJ/dp_k sums thousands of cells
while dJ/ds_i is one cell, so the two gradients differ by about three orders of magnitude in
scale and L-BFGS-B is not invariant to that. Alternating removes the need for an invented
scaling constant. Both gradients still come out of ONE backward sweep, so a block that only
moves the schedule pays for the map gradient it discards, and that cost is inside the budget.

---

## 12. Proven, computed, assumed

**Proven (finite-difference gated at the production operating point).**
The temporal power-schedule gradient dJ/dp_k, at six shape-and-window configurations,
best relative error between 3.5e-10 and 1.6e-07 across max-sensitivity, random-segment and
random-direction probes, on both a fixed stop and the optimal stop. The equality of the
fixed-stop and optimal-stop gradients to 0.000e+00, which is the envelope property. The
post-window hold term in dJ/dp[-1]. The map gradient and the schedule gradient come out of
ONE backward sweep sharing the same discrete operators as the forward.

**Computed (measured in this campaign, not assumed).**
Every J, IoU, stop time, duty cycle, mean relative density and energy residual in the tables.
The uniform-map and baseline stop times used to set the schedule window. The three-way
map/dose/structure decomposition. The three-leg melt bookkeeping across the OFF period and
the 1.00-to-1 and 3.04-to-1 selectivity numbers. The binary rounding losses. The
segment-count sensitivity.

**Assumed or simplified, and how it bites.**
1. **The schedule window rule (1.5 times the later measured stop) is a convention, not a
   derivation.** It was chosen to give roughly 10 to 11 active segments. A different window
   changes the control resolution and therefore the achievable schedule.
2. **The power box [0, 1.5] is arbitrary.** It is what makes the dose confound possible.
   Three of four shapes sit on that ceiling, so their numbers are a statement about a 1.5x
   voltage-squared headroom, not about scheduling.
3. **The forward is the two-dimensional `adjoint2d` engine, not `heatr3d`.** Every
   simplification already documented for the shape-library solve applies unchanged here:
   two-dimensional cross-section, the depth treatment, the material boundary width, and the
   dopant-to-conductivity law.
4. **Density enters only as a diagnostic.** Nothing in this campaign optimized for density.
   The cross's 0.941 is a by-product of a longer, lower-power process, not a target.
5. **The relaxation-and-round binary result is feasible but not certified optimal**
   (section 9).
6. **The forward-equivalent budget is computed from a wall-clock ratio measured on a loaded
   machine**, so the number of gradient evaluations varied between runs (17 to 29 for the
   co-optimization block). This is reported per run in the JSON.
7. **No experimental validation.** These are simulation results on a fixed geometry with a
   graded dopant. Whether a real generator can hold 1.5 times nominal for 844 s, or switch
   fully off and back on with the assumed instantaneous response, was not checked.

---

## 13. The single most valuable next layer

**Re-run the cross deliverable with the power box capped at [0, 1] and the window rule held
fixed.** The cross is the one shape where the gain is structure rather than dose, and its
optimum currently sits at duty 1.145 with eleven segments pinned at the 1.5 ceiling. Capping
at nominal power separates two things that are still entangled on that shape: whether the
melt, erase and selectively re-melt cycle needs an over-drive to work, or whether it works at
nominal power and simply takes longer. That question decides whether the result is
generator-realizable on existing hardware. It is one configuration change, no new gradient,
no re-gate needed, and roughly 15 minutes of compute.

Second, and larger: **put the OFF period's mechanism to work deliberately** by adding a
second melt-erase-remelt cycle, which the current 16-segment window can barely express (the
cross uses one cycle occupying five segments). A 32-segment window at the capped box would
test whether the selectivity compounds.

---

## 14. Files

Code, all in `adjoint2d/`:
`schedule.py` (control parameterization, structure classifier, adjoint of the expansion),
`sched_solve.py` (cold-start campaign driver), `sched_warmstart.py` (warm-start arms),
`sched_dose_control.py` (dose-matched control), `sched_offperiod.py` (mechanism diagnosis),
`gate_sched.py` (finite-difference gate), `make_sched_figures.py` (figures).
Tests: `tests/test_schedule.py`, `tests/test_schedule_forward.py`,
`tests/test_schedule_gradient.py`, `tests/test_sched_solve.py`.

Data in `out_sched/`: `<shape>_sched.json` and `_maps.npz` (cold start),
`<shape>_warm.json` and `_maps.npz` (warm start), `<shape>_{sched,warm}_dose_control.json`,
`cross_{sched,warm}_<arm>_offperiod.json`, `cross_nseg{8,32}_sched.json`,
`gate_sched_*.json` and the run logs.

Figures in `figs_sched/`: `fig_sched_<shape>.png` (deliverable composite, schedules as
generator instructions on the top row and melt fields at each arm's own stop below),
`fig_sched_curves_<shape>.png` (objective and density against time),
`fig_sched_maps_<shape>.png` (dopant maps), `fig_sched_census.png` (all four shapes),
`fig_sched_offperiod_cross_WARM_CO_4bpp.png` and
`fig_sched_offperiod_cross_BIN_round.png` (temperature field pairs across the OFF period).
Every figure in this list was viewed and checked.
