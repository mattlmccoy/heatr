# Multi-start and warm-start on the melt-region shape solve, with the mandatory design filter

**Date:** 2026-08-01. **Scope:** four deterministic starts sharing one 40
forward-equivalent budget per shape, an early-kill rule, the physical-length
design filter applied to the design variable inside the solve, the full 18-shape
standardized library, a filtered single-start control on all 18 shapes, and the
two robustness probes re-run on the new square and rectangle maps. Grid 120 x
120 unless a number says 160. Nothing was committed. No dissertation file was
touched. `.claude/worktrees/` was not read or written.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the two-dimensional solver uses). IoU =
intersection over union. bpp = bits per pixel. L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. FD = finite difference.
BLAS = basic linear algebra subprograms. phi = melt fraction. rho = relative
density. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit tested or FD gated. COMPUTED = measured from a
real run in this pass. ASSUMED = a modelling choice or an inference not measured
here.

**Objective and stop convention, stated once and carried on every number.**

    J_phi(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with `chi_part` the rasterized binary part mask. Every J, IoU, growth, under-melt,
mean relative density and absorbed power below is read at that arm's OWN J-stop,
t_stop = argmin over that arm's own stored trajectory of J_phi. `HORIZON` is
flagged when the minimum sits on the last stored step, which makes that arm's J
a bound. The melted region for IoU, growth and under-melt is phi >= 0.5; J_phi
itself uses no threshold. Absorbed power is the state-B value in W/m. Mean
relative density is the part-cell mean at that same stop index.

**Grid qualifier, mandatory.** `SOLVE_ROBUSTNESS_VALIDATION.md` established that
absolute fidelity at grid 120 does not transfer to grid 160 and that the forward
itself is not converged in IoU between those grids. Every IoU in this report is
a property of the method AT GRID 120 with an exactly reproduced dopant map, not
a property of the method. Section 6 is the only place a 160 number appears and
it is labelled.

---

## 1. Verdict, up front

**1. The rectangle stall is HALVED, not eliminated, and the cold start still
stalls exactly.** COMPUTED. The deliverable rectangle arm goes from J_phi 196.94
(library single start) to **97.19**, a **50.7 percent** reduction, and IoU 0.8424
to **0.8842** at grid 120. It still loses badly to the best stored mask
(J 10.69, IoU 1.0000), so the shape stays in the NOT RESCUED class. The stall
itself is reproduced exactly and is not a fluke: the cold start's first two
gradient evaluations move J_phi from 203.07 to 203.07, **0.00 percent**, and a
filtered cold start given the FULL budget only reaches 187.57. Multi-start does
not cure the stall, it routes around it, and the route is the deterministic
perturbed start.

**2. Multi-start changes the winner on 15 of 18 shapes, and on the fixed budget
that is not the same as changing the result for the better.** COMPUTED. The
uniform cold start wins on only **3 of 18** shapes (triangle, H_shape, diamond);
the previously solved map wins on 7, the stored historical mask on 6, and the
perturbed start on 2. But splitting one 40 forward-equivalent budget four ways
costs depth, and the deliverable beats the unfiltered single-start library arm
on only **7 of 18** shapes on J_phi and 8 of 18 on IoU. The filtered SINGLE cold
start at the full budget beats the library arm on **9 of 18** and beats the
multi-start arm on **12 of 18**.

**3. The 13-of-18 census against the best-stored oracle is HELD EXACTLY, by all
three arms, on the same 13 shapes.** COMPUTED. Multi-start 13 of 18 on J_phi and
13 of 18 on IoU; the filtered single cold start 13 and 13; the library arm 13 and
13. No shape flips its beats-the-oracle boolean in either direction. The only
class change anywhere is **star6, NOT RESCUED to MATCHED**. All three arms beat
uniform on 18 of 18, and all three put 7 of 18 shapes in the IoU >= 0.95 class at
grid 120.

**4. The design filter fixes the grid-sculpture class on the square, which is
the first direct evidence that the fix named in
`SOLVE_ROBUSTNESS_VALIDATION.md` Section 10 works.** COMPUTED. A one-cell
part-masked blur of the square's solved map used to cost **+152.7 percent** of
J_phi; on the filtered multi-start map it costs **+9.9 percent**, a 15-fold
reduction in rim sensitivity. In absolute terms the filtered map starts worse
(35.56 against 25.66) and ends **39.7 percent better** once a one-cell blur is
applied (39.09 against 64.84). At grid 160 the filtered square arm scores J
503.09 and IoU 0.8063 against the unfiltered 572.79 and 0.7767, a real
improvement that still does not restore the grid-120 IoU of 0.9681.

**5. The most useful single number in this pass is the separation of the two
changes, because they moved together.** COMPUTED, from the two-shape headline
and confirmed on all 18:

| shape | unfiltered single start (library) | filtered single cold start | filtered multi-start |
|---|---|---|---|
| square | J 25.66, IoU 0.9816 | **J 18.51, IoU 1.0000** | J 35.56, IoU 0.9681 |
| rectangle | J 196.94, IoU 0.8424 | J 187.57, IoU 0.8580 | **J 97.19, IoU 0.8842** |

On the square the win is the FILTER and the multi-start is what costs; on the
rectangle the win is the MULTI-START and the filter buys almost nothing. The
mechanism is depth against breadth and it is visible in one scatter
(`figs_ms/fig_ms_starts.png` panel B): multi-start pays off exactly on the three
shapes whose cold-start probe removes less than 2 percent of J_phi in two
evaluations (L_shape 0.19 percent, T_shape 0.01 percent, rectangle 0.00 percent)
and costs on the shapes where the cold start descends well.

**6. A cheap adaptive recipe falls out of that, and it is IN-SAMPLE.** COMPUTED
but flagged: routing each shape to the filtered single cold start when its cold
probe removes at least 2 percent of J_phi, and to multi-start otherwise, gives a
summed J_phi over the library of **2572** against 2661 (filtered single only),
2679 (multi-start only) and 2676 (library single start), and the routing is
identical for any threshold between 1 and 5 percent. **The threshold was chosen
after seeing these 18 results, so this is a hypothesis fitted to its own test
set, not a validated recipe.** It is free to apply, because the cold probe it
reads is bit-identical to the first two evaluations of the multi-start run
(COMPUTED: maximum absolute difference 0 over 18 shapes and 2 evaluations).

---

## 2. The finite-difference gate, run BEFORE any optimization

The gradient the solve uses is dJ_phi/dv with the design filter in the chain,
s = F(v). That composition had never been gated end to end against the real
forward: `gate_shape.py` gated dJ_phi/ds and `tests/test_design_filter.py`
proves the transpose to 1e-12 by the dot-product identity, but not together on
the melt objective. Code: `adjoint2d/gate_ms.py`. Raw:
`out_ms/gate_ms_square.json`, `out_ms/gate_ms_rectangle.json`. Logs:
`logs_ms/gate_ms_*.log`.

Central differences, epsilon swept **1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8,
1e-8**, two shapes, two layers (S1 unfiltered, S2 filtered), read at a FIXED
index equal to the base run's J_phi argmin. Probes: the maximum-sensitivity
cell, a fixed pseudo-random in-part cell, a random unit direction, for S2 a
smooth random direction (a random direction passed through the filter, then
normalized), and a new **gradient-direction** probe.

**Why the gradient-direction probe was added, stated so it cannot look like
cherry-picking.** The relative error of a directional probe is
`floor / (2 eps |analytic|)`, so a direction nearly orthogonal to the gradient
reports its own denominator, not the gradient. COMPUTED on the square: the rough
random direction has analytic -3.53e-03 against a gradient norm of 20.63, and
its 1.58e-02 relative error corresponds to an absolute error of 5.58e-05, which
is the same absolute agreement every other probe achieves. The gradient
direction is the direction L-BFGS-B actually steps along and it has the largest
available analytic derivative. Every probe is reported below; none was dropped.

### 2.1 Results

| shape | layer | probe | analytic | best relative error | best epsilon | absolute error |
|---|---|---|---|---|---|---|
| square | S1 | max-sensitivity cell | -1.0545e+01 | 3.446e-06 | 3e-08 | 3.6e-05 |
| square | S1 | random cell | -6.3889e-01 | **1.003e-08** | 1e-05 | 6.4e-09 |
| square | S1 | random direction | -2.7213e-01 | 7.760e-05 | 1e-06 | 2.1e-05 |
| square | S1 | gradient direction | +2.9625e+01 | 6.142e-05 | 1e-07 | 1.8e-03 |
| square | S2 filtered | max-sensitivity cell | -2.2134e+00 | 1.435e-05 | 1e-06 | 3.2e-05 |
| square | S2 filtered | random cell | -6.2068e-01 | **3.703e-06** | 1e-05 | 2.3e-06 |
| square | S2 filtered | random direction | -3.5345e-03 | 1.579e-02 | 1e-06 | 5.6e-05 |
| square | S2 filtered | smooth random direction | +8.1489e-02 | 7.831e-04 | 1e-06 | 6.4e-05 |
| square | S2 filtered | **gradient direction** | +2.0626e+01 | **2.556e-06** | 1e-06 | 5.3e-05 |
| rectangle | S1 | max-sensitivity cell | -1.1117e+01 | **6.223e-07** | 1e-06 | 6.9e-06 |
| rectangle | S1 | random cell | +4.5173e-02 | **8.318e-07** | 1e-04 | 3.8e-08 |
| rectangle | S1 | random direction | +1.8315e+00 | **1.815e-06** | 1e-05 | 3.3e-06 |
| rectangle | S1 | gradient direction | +3.0292e+01 | **5.592e-06** | 3e-07 | 1.7e-04 |
| rectangle | S2 filtered | max-sensitivity cell | -2.6953e+00 | **3.094e-06** | 1e-05 | 8.3e-06 |
| rectangle | S2 filtered | random cell | -1.8290e-01 | **2.096e-06** | 1e-04 | 3.8e-07 |
| rectangle | S2 filtered | random direction | +6.9352e-01 | **2.107e-06** | 1e-05 | 1.5e-06 |
| rectangle | S2 filtered | smooth random direction | +2.4030e+00 | **2.884e-06** | 1e-05 | 6.9e-06 |
| rectangle | S2 filtered | **gradient direction** | +1.5144e+01 | **1.844e-07** | 1e-05 | 2.8e-06 |

**Verdict, stated exactly and not inflated. The gradient is VERIFIED, and the
gate does NOT reach the 1e-6 clean-smooth standard on every probe.** On the
rectangle **9 of 9 probes clear the campaign's documented 1e-5 subgradient
standard** and 3 of 9 clear 1e-6. On the square 4 of 9 clear 1e-5 and 1 of 9
clears 1e-6. The decision-relevant probe, the direction the optimizer steps
along, gates at **1.84e-07 (rectangle)** and **2.56e-06 (square)** on the
filtered layer.

### 2.2 The bisect that localizes the residual, and it is not the filter

**PROVEN in this pass, to machine precision.** `gate_ms.transpose_consistency`
checks that dJ/dv dotted with a direction d equals dJ/ds dotted with F_lin(d),
where F_lin is the filter with the nominal outside value set to zero. F is
linear in v at fixed masks, so this identity is exact if the wiring is right.
COMPUTED: maximum relative error **4.712e-14 (square)** and **1.848e-16
(rectangle)**. **The filtered analytic gradient is therefore EXACTLY the
unfiltered gradient composed with a proven-exact linear operator, and every
remaining finite-difference disagreement belongs to the forward, not to the
filter.**

### 2.3 The floor, measured rather than assumed

From the roundoff-dominated tail of each sweep, `2 eps` times the absolute
error estimates the objective's absolute evaluation floor. COMPUTED: **1.4e-11
to 2.6e-10 absolute**, that is **8e-14 to 1.5e-12 relative** to J_phi of about
170 to 180, roughly 1000 times machine epsilon. That is the same class as the
floor `DENSITY_OBJECTIVE_LIBRARY_REPORT.md` Section 2.3 measured for J_rho, and
it has the same cause, an explicit march of 1500 outer steps by 5 substeps. With
a central difference the achievable relative error is capped at
`floor / (2 eps |g|)`, and every measured number sits at that cap. **The gate is
at its information limit for a double-precision forward.**

Two secondary readings, both COMPUTED. Single-cell probes in the UNFILTERED
layer reach the smallest absolute errors (6.4e-09 on the square) because most of
the arithmetic is bit-identical between the plus and minus runs; the same
single-cell probe in the FILTERED layer is a multi-cell perturbation in the map
and lands at 2.3e-06. And at epsilon 1e-5 the smooth-direction central
difference is biased by a factor of 200 on the square (17.0 against an analytic
0.081), so the usable epsilon window is narrow, roughly 3e-7 to 1e-6, and it was
swept explicitly.

### 2.4 The read state does not move

COMPUTED. `gate_ms.stop_index_stability` recomputes the J_phi argmin under every
probe perturbation at epsilon 1e-3: **it does not move on either shape** (square
base index 882, rectangle base index 675, `moved = False` on all probes). The
envelope argument that removes the dt*/ds term is therefore not merely
theoretical here; the argmin is literally fixed over the tested range.

### 2.5 Unit tests, red first

**167 tests pass** (`.venv312/bin/python -m pytest adjoint2d/tests -q`), of which
**29 are new in this pass and every one was observed failing first**: the run of
`tests/test_multistart.py` before `multistart.py` existed gave
`ImportError: cannot import name 'multistart' from 'adjoint2d'`, then 29 passed.
The new tests cover the budget split (`probe_evals`, `continuation_evals`), the
early-kill rule including the zero-leader and missing-result cases, the two
deterministic starts, the box projection, the winner pick, and the evaluation
cache.

---

## 3. What was run, and the rules

### 3.1 The four starts, in declaration order

All four are DESIGN VARIABLES v; the injected map is s = F(v) at sigma = 1.5
cells (0.75 mm at this grid), the value `DENSITY_OBJECTIVE_LIBRARY_REPORT.md`
used and inside the 1.5 to 2 cell band the robustness verdict named.

| start | what it is | note |
|---|---|---|
| `cold` | uniform saturation 1 | the historical cold start |
| `warm` | the best stored historical 4-bpp mask for that shape, as `out_lib/<shape>.json` selected it, in the boundary convention that won there, loaded through the production loader and clipped into [0, 1] | the mask was SCORED historically in the permittivity-co-varying channel and this solve actuates conductivity only; the start is the map, not the channel |
| `prev` | the library campaign's solved CONTINUOUS map, `out_lib/<shape>_maps.npz` key `A1_cont` | |
| `pert` | the midpoint of uniform and the proportional-inverse control map at gain 0.5 | seedless and resume-reproducible |

**Why the perturbed start is a midpoint and not a sum, stated because the task
said "uniform plus the proportional-inverse map at gain 0.5".** Read literally as
a sum it is degenerate: the proportional-inverse map at gain 0.5 lies in
[0.25, 0.75] inside the part, so 1 plus that exceeds the box on every cell and
clipping returns the uniform start exactly, duplicating start (a). The midpoint
is the non-degenerate reading of the same intent. Recorded in
`multistart.perturbed_start` with that reasoning.

**Consequence that must be stated: the warm start injects a FILTERED historical
mask, not the mask.** Start `warm`'s first evaluation is therefore not the
historical arm's number, and on several shapes it is much worse (COMPUTED: on
the diamond the warm probe reaches J 768.0 against the historical arm's 355.07,
because a sigma 1.5 blur of a sharp stored mask is a different map).

### 3.2 The budget split and the early-kill rule

One evaluation of objective plus gradient costs `1 + ratio` forward-equivalents,
so the pool is `n_total = floor(40 / (1 + ratio))` evaluations, 14 to 18
depending on the shape. Spent in two phases:

* **PROBE.** Every start gets `min(2, n_total // n_starts)` evaluations, which
  was 2 on every shape. The first is the objective at the start point, so a
  probe of 2 buys one descent step.
* **KILL.** Let J_k be the best J_phi start k reached and J* the smallest. Start
  k survives when `J_k <= J* + 0.10 |J*|` AND its rank in ascending J_k is below
  2. The leader always survives. COMPUTED: **42 of 72 declared starts were
  killed at the probe**; 5 shapes kept one survivor and 13 kept two.
* **CONTINUATION.** Survivors split the remainder equally, 3 to 6 evaluations
  each.

**The continuation restart is not free and that is not hidden.**
`scipy.optimize.minimize` cannot be resumed, so a survivor's continuation is a
FRESH L-BFGS-B from its probe's last iterate and the limited-memory curvature
from the probe is discarded. With a 2-evaluation probe that memory holds at most
one secant pair, so the loss is small, but it is not zero and it was not
measured. What IS free is the repeated evaluation: the continuation's first call
lands on exactly the vector the probe last evaluated and `multistart.EvalCache`
returns it without running anything and without charging the budget. COMPUTED:
**30 cache hits across the 18 shapes**, all of them that handover.

### 3.3 Budget accounting, and a measurement problem that had to be fixed

**COMPUTED problem.** Timing one forward and one adjoint on a loaded machine is
unreliable. On the star shape, with one other job running, this pass timed 3.4 s
forward against 43.4 s adjoint, a ratio of 12.91, against the 1.11 the library
campaign timed for the same two calls on the same code path. That ratio converts
a 40 forward-equivalent budget into ONE gradient evaluation and would silently
starve the solve.

**Fix, and it also improves the comparison.** The budget conversion uses the
per-shape ratio RECORDED BY THE LIBRARY CAMPAIGN in `out_lib/<shape>.json`
(1.11 to 1.78), so the evaluation pool is deterministic and the multi-start
solve gets **exactly the same number of gradient evaluations the single-start
library solve got for the same nominal budget**. The ratio re-timed in this run
is stored alongside and labelled contention-contaminated
(`ms_solve.budget_ratio`).

COMPUTED spend: **35.1 to 39.6 forward-equivalents per shape, no overrun**, and
`probe_overruns_pool` is false on every shape. Outside that budget, and named
rather than absorbed: one cost-probe forward (which doubles as the uniform arm)
plus one adjoint, one full-horizon forward to build the perturbed start's proxy
field, and two final scoring runs. The single-start library campaign carried the
same cost-probe and final-scoring overhead.

### 3.4 A throughput finding worth keeping

COMPUTED. Unpinned, each solve process took **5.6 cores of BLAS threads** and a
single square forward still cost 10.7 s, against 9.9 s for the same call in the
library campaign. Two such jobs made a gate that takes 9 minutes single-threaded
run for more than 30 minutes without finishing a layer. Pinning every numerical
library to one thread (`fgm_solve_campaign/env1.sh`) and taking parallelism
across shapes instead cut the whole 18-shape pass to **11 minutes of wall clock
in 6 streams**. The thread parallelism buys nothing on a sparse 120 x 120 solve.

### 3.5 Reproducibility check, and it is exact

COMPUTED. Every shape re-runs the uniform arm and compares it against
`out_lib`. **Maximum absolute difference in J_phi over 18 shapes: 0.0.** The
engine, the configuration and the stop rule are the same ones the library
campaign used, so the reference arms read from `out_lib` are directly
comparable and were not re-run.

---

## 4. The census, all 18 shapes

Deliverable arm `MS_4bpp`: the winning start's filtered continuous map, quantized
to 4 bpp inside the part through the production quantizer and re-run through the
real forward. Grid 120 x 120. `win` is the winning start. `lvl` is the number of
distinct printer levels the map uses inside the part.

| shape | win | J_ms | J_ctl | J_lib | J_hist | J_unif | IoU_ms | IoU_ctl | IoU_lib | IoU_hist | grow % | under % | rho at stop | P_abs W/m | lvl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | prev | 35.56 | **18.51** | 25.66 | 12.77 | 210.19 | 0.9681 | **1.0000** | 0.9816 | 0.9975 | 1.75 | 1.50 | 0.7973 | 407.7 | 10 |
| circle | warm | 15.55 | 12.33 | 14.68 | 52.18 | 275.02 | 0.9872 | 0.9936 | 0.9904 | 0.9492 | 0.65 | 0.65 | 0.8432 | 348.9 | 13 |
| hexagon | prev | 24.57 | 24.24 | 22.15 | 69.63 | 251.27 | 0.9650 | 0.9746 | 0.9767 | 0.9211 | 1.18 | 2.36 | 0.8011 | 380.9 | 14 |
| triangle | cold | 82.29 | 82.29 | 87.99 | 171.27 | 202.82 | 0.8783 | 0.8783 | 0.8578 | 0.7768 | 4.75 | 8.00 | 0.6924 | 415.7 | 14 |
| equilateral_triangle | pert | 100.06 | 97.89 | 119.00 | 145.55 | 320.80 | 0.8451 | 0.8370 | 0.8128 | 0.8058 | 5.34 | 10.98 | 0.7737 | 347.3 | 16 |
| L_shape | prev | 530.14 | 528.35 | 521.20 | 597.86 | 538.42 | 0.5109 | 0.5135 | 0.5209 | 0.4413 | 6.67 | 45.51 | 0.6491 | 493.4 | 15 |
| H_shape | cold | 149.76 | **120.07** | 150.86 | 219.11 | 214.68 | 0.8499 | 0.8599 | 0.8423 | 0.7631 | 9.90 | 6.60 | 0.7036 | 452.5 | 8 |
| T_shape | prev | 612.03 | 612.13 | 609.29 | 676.39 | 614.26 | 0.4583 | 0.4583 | 0.4444 | 0.3983 | 4.35 | 52.17 | 0.6242 | 497.0 | 10 |
| cross | prev | 385.07 | 380.93 | 360.18 | 331.36 | 471.82 | 0.6508 | 0.6589 | 0.6755 | 0.7052 | 13.32 | 26.25 | 0.7894 | 351.3 | 14 |
| diamond | cold | 272.11 | 227.78 | 211.16 | 355.07 | 768.11 | 0.8161 | 0.8432 | 0.8520 | 0.7793 | 1.11 | 17.49 | 0.7992 | 311.2 | 16 |
| ellipse | warm | 28.84 | 21.46 | **13.98** | 74.70 | 241.72 | 0.9524 | 0.9681 | 0.9787 | 0.8854 | 1.61 | 3.23 | 0.7597 | 307.0 | 15 |
| octagon | prev | **10.58** | 11.96 | 10.66 | 16.22 | 42.96 | 0.9963 | 0.9852 | 0.9944 | 0.9888 | 0.00 | 0.37 | 0.8057 | 414.6 | 15 |
| pentagon | warm | 57.66 | 49.65 | **44.22** | 134.64 | 300.33 | 0.9253 | 0.9289 | 0.9387 | 0.8446 | 3.66 | 4.09 | 0.8277 | 323.3 | 15 |
| rectangle | pert | **97.19** | 187.57 | 196.94 | 10.69 | 203.07 | 0.8842 | 0.8580 | 0.8424 | 1.0000 | 7.99 | 4.51 | 0.6717 | 424.6 | 4 |
| rounded_rect | warm | 31.65 | 16.06 | 18.21 | 8.97 | 266.38 | 0.9795 | 0.9974 | 0.9974 | 0.9974 | 1.56 | 0.52 | 0.7894 | 414.6 | 12 |
| star | warm | **144.45** | 163.22 | 157.39 | 172.11 | 192.58 | 0.7138 | 0.6809 | 0.7032 | 0.6713 | 7.01 | 23.62 | 0.6767 | 280.3 | 8 |
| star6 | prev | **72.42** | 82.03 | 84.69 | 70.57 | 190.62 | 0.8742 | 0.8408 | 0.8408 | 0.8782 | 7.43 | 6.08 | 0.6778 | 402.8 | 13 |
| trapezoid | warm | 29.27 | 24.12 | 27.36 | 122.22 | 171.32 | 0.9693 | 0.9792 | 0.9718 | 0.8688 | 1.43 | 1.68 | 0.7670 | 414.3 | 11 |

`J_ctl` and `IoU_ctl` are the filtered SINGLE cold start at the full 40
forward-equivalent budget, the control that separates the filter from the
multi-start. The diamond deliverable's stop is **at the horizon** (750.0 s), so
its J is an upper bound; every other stop is interior.

### 4.1 The census delta table, three arms against four baselines

| baseline | library single start, unfiltered | filtered single cold start | filtered multi-start (deliverable) | adaptive routing (in-sample) |
|---|---|---|---|---|
| beats the best-stored ORACLE on J_phi | 13 of 18 | **13 of 18** | **13 of 18** | 13 of 18 |
| beats the best-stored ORACLE on IoU | 13 of 18 | **13 of 18** | **13 of 18** | 13 of 18 |
| beats UNIFORM on J_phi | 18 of 18 | 18 of 18 | 18 of 18 | 18 of 18 |
| beats the LIBRARY single-start arm on J_phi | reference | **9 of 18** | 7 of 18 | 9 of 18 |
| beats the LIBRARY single-start arm on IoU | reference | 8 of 18 | 8 of 18 | 8 of 18 |
| in the IoU >= 0.95 class AT GRID 120 | 7 of 18 | 7 of 18 | 7 of 18 | 7 of 18 |
| summed J_phi over the library | 2676 | 2661 | 2679 | **2572** |

**The oracle census does not move, and it is the same 13 shapes in every arm.**
The five losses are square, cross, rectangle, rounded_rect and star6 in the
library arm, and the identical five in both new arms. The only ladder class that
changes is star6, **NOT RESCUED to MATCHED** (J 84.69 to 72.42 against the
oracle's 70.57, IoU 0.8408 to 0.8742 against 0.8782, which lands inside the
5-percent and 2-IoU-point tie band).

**The oracle baseline is an oracle and must be quoted as one.** Selecting that
mask cost 28 to 38 forward solves per shape
(`SHAPE_LIBRARY_SOLVE_REPORT.md` Section 5 table), against the 40
forward-equivalents each solved arm was given.

### 4.2 Where each arm wins, and the mechanism

Multi-start beats the library arm on **rectangle (+50.7 percent of J_phi),
equilateral_triangle (+15.9), star6 (+14.5), star (+8.2), triangle (+6.5),
octagon (+0.7) and H_shape (+0.7)**. It loses on **ellipse (-106.3),
rounded_rect (-73.8), square (-38.6), pentagon (-30.4), diamond (-28.9),
hexagon (-10.9), cross (-6.9), trapezoid (-7.0), circle (-5.9), L_shape (-1.7)
and T_shape (-0.5)**.

COMPUTED, the discriminator is the cold start's own probe. The three shapes
whose cold probe removes less than 2 percent of J_phi in two evaluations are
**L_shape (0.19 percent), T_shape (0.01 percent) and rectangle (0.00 percent)**,
and the rectangle is where multi-start wins most. Above 2 percent the cold start
is descending and the budget is better spent going deeper with it: the filtered
single cold start beats the multi-start arm on **12 of 18** shapes.

**Winning starts:** `prev` 7, `warm` 6, `cold` 3, `pert` 2. **The uniform cold
start is the best of the four on only 3 of 18 shapes**, which is the direct
answer to whether warm-starting matters: it does, and the previously solved map
is the single most useful start.

### 4.3 The dose-match limitation, restated because it applies here too

None of these arms is power matched. COMPUTED in this pass, absorbed power at
the deliverables' own stops spans **280.3 to 497.0 W/m** across the 18-shape
multi-start census against the 500.0 W/m uniform calibration target, and 302.2 to
497.0 W/m for the adaptive selection. The objective penalizes over-melting as
well as under-melting, which removes the crudest dose gaming, but a J_phi or IoU
comparison between two arms is not a comparison at equal delivered energy. State
it every time these counts are quoted.

Mean relative density at the deliverable's own melt stop spans **0.6242 to
0.8432**, which reproduces the 0.60 to 0.82 band
`DENSITY_OBJECTIVE_LIBRARY_REPORT.md` Section 1 point 3 measured and is a
property of the READ TIME, not of the map. Nothing here is an oversinter claim:
these are melt-stop reads, not end-of-horizon reads.

---

## 5. Gates and flags

**Energy-residual gate.** COMPUTED: **zero violations on every arm of every
shape**, in the 18 multi-start solves, the 18 controls and the 4 robustness runs.
The maximum relative energy residual at any deliverable's own stop is **1.04
percent** against the standing 5 percent threshold.

**Horizon flags.** One, the **diamond** deliverable arm at 750.0 s, so its
J_phi 272.11 is an upper bound. Reported in `stop_at_horizon_arms` in that
shape's JSON.

**Gradient health.** No gradient-death assertion was wired for this objective
because J_phi does not saturate the way J_rho does (its argmin is interior, and
the stop is optimized to stationarity). Not checked, and named as not checked.

**Not monotone, and that is normal.** The per-start curves in
`figs_ms/fig_ms_starts.png` panel C rise and fall because L-BFGS-B evaluates
trial points during its line search. Each arm keeps its BEST iterate by J_phi,
which is what `MS_cont` and `MS_4bpp` are built from.

---

## 6. The two robustness probes, re-run on the new maps

This is the first direct test of whether filtering the design variable fixes the
grid-sculpture class that `SOLVE_ROBUSTNESS_VALIDATION.md` identified. Forward
runs only, nothing re-solved. Both probes reproduce the earlier protocol exactly
(`robust.smooth_in_part` for the rim, `robust.resample_map` in the production
map-injection convention for the grid), and the uniform, historical and
dose-matched reference arms at 160 are read from `out_robust/<shape>_grid.json`
rather than re-run, because that pass ran them on this engine at this grid with
this configuration.

### 6.1 Rim robustness at grid 120

| shape | map | J at r = 0 | J at r = 1 | J at r = 2 | dJ at r = 1 | dJ at r = 2 | IoU r = 0 | IoU r = 1 | IoU r = 2 |
|---|---|---|---|---|---|---|---|---|---|
| square | unfiltered single start | 25.66 | 64.84 | 88.38 | **+152.7 %** | **+244.4 %** | 0.9816 | 0.9547 | 0.9340 |
| square | filtered multi-start | 35.56 | **39.09** | 57.24 | **+9.9 %** | +61.0 % | 0.9681 | 0.9583 | 0.9395 |
| rectangle | unfiltered single start | 196.94 | 157.09 | 153.91 | -20.2 % | -21.8 % | 0.8424 | 0.8715 | 0.8770 |
| rectangle | filtered multi-start | 97.19 | 98.79 | 99.52 | **+1.6 %** | **+2.4 %** | 0.8842 | 0.8842 | 0.8842 |

**COMPUTED, and this is the clearest result in the pass. The filter removes the
single-cell sensitivity.** On the square the one-cell blur cost falls from
+152.7 percent to +9.9 percent, a 15-fold reduction, and in ABSOLUTE J the
filtered map crosses below the unfiltered one at a blur radius under one cell:
39.09 against 64.84 at r = 1 and 57.24 against 88.38 at r = 2. On the rectangle
the filtered map is essentially blur-invariant (+1.6 and +2.4 percent, IoU
unchanged to four decimals). The rectangle's unfiltered arm IMPROVES under blur,
which is what a stalled nearly uniform map does and is why it was the control in
the earlier pass; the filtered rectangle map no longer has that signature
because it is no longer stalled.

### 6.2 Grid hold-out, solve at 120 and score at 160

| shape | arm | J at 160 | IoU at 160 | IoU at 120 | P_abs W/m |
|---|---|---|---|---|---|
| square | uniform | 631.22 | 0.7711 | 0.8508 | 366.8 |
| square | best stored mask | 176.47 | 0.9261 | 0.9975 | 502.5 |
| square | unfiltered single start | 572.79 | 0.7767 | 0.9816 | 340.1 |
| square | **filtered multi-start** | **503.09** | **0.8063** | 0.9681 | 344.1 |
| square | unfiltered, dose-matched | 573.84 | 0.7775 | | 463.6 |
| square | **filtered multi-start, dose-matched** | **492.37** | **0.8157** | | 469.0 |
| rectangle | uniform | 210.20 | 0.8887 | 0.8528 | 426.9 |
| rectangle | best stored mask | 33.47 | 0.9980 | 1.0000 | 444.7 |
| rectangle | unfiltered single start | 196.87 | 0.8954 | 0.8424 | 427.3 |
| rectangle | **filtered multi-start** | **103.05** | **0.9113** | 0.8842 | 377.8 |
| rectangle | unfiltered, dose-matched | 234.99 | 0.8873 | | 500.4 |
| rectangle | **filtered multi-start, dose-matched** | **115.19** | **0.9116** | | 442.4 |

The dose-matched arms use the drive voltage the earlier pass calibrated so the
UNIFORM arm at 160 absorbs 500.0 W/m (square 2834.8 V, rectangle 3588.8 V).

**COMPUTED, and the honest reading is a partial fix.** The filtered map transfers
BETTER than the unfiltered one on both shapes and in both dose conditions:
square J 572.79 to 503.09 and IoU 0.7767 to 0.8063 at pinned voltage, 573.84 to
492.37 and 0.7775 to 0.8157 dose matched; rectangle J 196.87 to 103.05 and IoU
0.8954 to 0.9113. **But the square's absolute fidelity still collapses between
grids**, 0.9681 at 120 against 0.8063 at 160, and it does not re-enter the
IoU >= 0.95 class. The rectangle's arm IMPROVES at 160 (0.8842 to 0.9113), which
is the same direction the uniform arm moves on that shape (0.8528 to 0.8887), so
it is not evidence about the map.

**The limit named in the earlier report still binds and is not removed by this
result.** The uniform arm alone moves up to 0.13 IoU points between the grids and
in both directions, so a grid hold-out remains a joint test of map transfer AND
forward discretization convergence. This pass cannot separate them either.
ASSUMED, supported by the rim result: since the filtered map is nearly
blur-invariant and bilinear resampling is a sub-cell blur, the residual square
gap at 160 is more likely convergence than transfer. Not tested. The decisive
experiment is still a re-solve at 160.

---

## 7. Cost and wall time, with the projection that was required

**Logged after two shapes and projected, as required.** The first two multi-start
completions were rectangle at 180 s and hexagon at 191 s, mean 186 s, which
projected to 18 shapes in 6 streams as 3 rounds of about 190 s, roughly 10
minutes. **Actual: 11 minutes of wall clock**, per-shape 115 to 287 s. The
projection held and nothing was dropped.

| stage | runs | wall |
|---|---|---|
| FD gate, square and rectangle, single threaded | 2 | 659 s and 535 s, in parallel |
| FD gate, first attempt, unpinned threads | 2 | superseded after 30 minutes without completing a layer, see Section 3.4 |
| 18 multi-start solves | 18 | 3490 s of process time, 11 minutes of wall clock in 6 streams |
| 18 filtered single-start controls | 18 | 3371 s of process time, about 7 minutes of wall clock in 6 streams |
| robustness, 2 shapes | 2 | 52 s and 41 s |

Total real compute for this report is about **2.3 hours of process time** and
about **45 minutes of wall clock**.

---

## 8. Proven, computed, assumed

**PROVEN**
* The filtered gradient is exactly the unfiltered gradient composed with the
  filter transpose: the dot-product identity holds to 4.7e-14 (square) and
  1.8e-16 (rectangle) against the real adjoint.
* 167 unit tests pass; the 29 new ones were red before they were green, the red
  being an `ImportError` observed before `multistart.py` existed.
* The budget arithmetic, the kill rule, the two deterministic starts, the box
  projection, the winner pick and the evaluation cache are unit tested,
  including the zero-leader, missing-result and empty-input cases.
* The design filter itself (box preservation without a clip, no bleed from the
  nominal outside value, monotone roughness reduction, identity at radius zero)
  carries over unchanged from the previous pass.

**COMPUTED**
* Every number in Sections 1 through 7.
* The uniform arm reproduces `out_lib` exactly, maximum absolute difference 0.0
  in J_phi over 18 shapes.
* The cold probe is bit-identical between the multi-start run and the control
  run, maximum absolute difference 0 over 18 shapes and 2 evaluations.
* The J_phi evaluation floor, 1.4e-11 to 2.6e-10 absolute.
* The J_phi argmin does not move under any probe perturbation at epsilon 1e-3.
* Zero energy-residual gate violations on 40 scored forward runs plus the solve
  evaluations; maximum residual 1.04 percent at a deliverable's own stop.

**ASSUMED, and how it bites**
1. **That sigma = 1.5 cells is the right physical design length.** No bench
   measurement of the real rim blur exists, so the length is in cells (0.75 mm at
   grid 120), not in a measured process length. Section 6.1 shows the choice is
   load bearing: it is most of why the square gives up 38.6 percent of J_phi at
   radius 0 and wins 39.7 percent at radius 1.
2. **That the midpoint is the right reading of the perturbed start.** The literal
   sum is degenerate (Section 3.1) and no other perturbation family was tried.
3. **That a 2-evaluation probe is enough to rank four starts.** It is what the
   budget allows, and it is a rank on a very short descent. A start that is slow
   early and fast later is killed by this rule and would never be seen. Not
   tested.
4. **That the kill margin of 10 percent and the cap of two survivors are right.**
   Both are conventions, neither was swept.
5. **That the adaptive threshold of 2 percent generalizes.** It does not follow
   from anything measured; it was read off these 18 shapes. In-sample.
6. **That the rasterized binary part mask is the right nominal target.** Carried
   over, still untested.
7. **That an arbitrary stop time is realizable as a process control.**
8. **The adjoint arms are conductivity-only; every historical arm co-varies
   permittivity.** Unchanged, still the largest actuator gap, and it is exactly
   why the warm start's first evaluation is not the historical arm's score.
9. **The forward is the two-dimensional `adjoint2d` engine, not `heatr3d`**, and
   every simplification documented for the shape-library solve applies unchanged.
10. **No experimental validation.** These are two-dimensional model results, and
    `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the model over-predicts
    achievable tuned uniformity by roughly a factor of eight against hardware.

---

## 9. Honest limits

1. **The rectangle stall is not eliminated.** Halved, and still 9.1 times worse
   than the best stored mask on J_phi. The headline question's answer is "no,
   reduced".
2. **The primary arm changed two things at once**, the filter and the
   multi-start. The 18-shape control separates them, but only for the cold start;
   there is no unfiltered multi-start arm, so "multi-start without the filter"
   is not measured.
3. **The gate does not reach 1e-6 on most probes** (4 of 18 across both shapes
   and both layers). The limit is the measured arithmetic floor and cannot be
   swept away, but it is a weaker gate than a smooth objective would give.
4. **Two shapes were gated, not eighteen.** The other 16 solves inherit that
   gate.
5. **The budget is small.** 14 to 18 gradient evaluations on 542 to 1624 design
   variables, split four ways in the primary arm, means every reported J_phi is
   an upper bound. The multi-start arm's losses against the library arm are
   partly a budget artifact and the size of that part is not measured; the
   quadruple-budget control that `DENSITY_OBJECTIVE_LIBRARY_REPORT.md` ran for
   J_rho has no counterpart here.
6. **No arm is dose matched**, 280.3 to 497.0 W/m.
7. **Robustness was probed on two shapes**, square and rectangle, and only on the
   multi-start maps. The filtered single cold start, which is the better arm on
   12 of 18 shapes, was NOT robustness probed. That is the most obvious gap.
8. **Everything is grid 120** except Section 6.2, and the earlier report's
   finding that the forward is not grid converged in IoU still blocks a clean
   transfer verdict.
9. **gt_logo was SKIPPED, loudly.** Its geometry is rasterized from an image and
   `rfam_eqs_coupled.make_domain` raises `ModuleNotFoundError: No module named
   'cv2'` in the `.venv312` interpreter. Nothing in this pass changed that. The
   library count is 18, not 19.

---

## 10. The single most valuable next layer

**Robustness probe the filtered SINGLE cold start on square and rectangle, and
then re-solve at grid 160.** The filtered single cold start is the better arm on
12 of 18 shapes and it reached IoU 1.0000 on the square, and it has not been
tested for rim or grid robustness at all. That is four forward runs and it
decides whether the production recipe should be the filtered single start or the
adaptive router. The 160 re-solve, one solve per shape at about 500 to 1300 s,
is still the decisive experiment for separating transfer from convergence, and
Section 6.2 has now measured a filtered arm to compare against.

**Second: give the multi-start a bigger budget rather than a split one.** The
census shows breadth and depth trading against each other at 40
forward-equivalents. Running the four starts at 40 forward-equivalents EACH on
the five shapes where multi-start currently loses most (ellipse, rounded_rect,
square, pentagon, diamond) would say directly whether the losses are the budget
split or the starts themselves. That is 5 shapes times 160 forward-equivalents,
about an hour in six streams.

**Third, and cheap: sweep the filter width.** Section 6.1 shows sigma = 1.5 cells
buys a 15-fold rim robustness improvement for 38.6 percent of J_phi on the
square. Sigma 1.0 and 2.0 on three shapes would locate the knee of that trade,
and the robustness verdict asked for exactly that test.

---

## 11. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code, all under `fgm_solve_campaign/`:
* `adjoint2d/multistart.py` the budget split, the kill rule, the two
  deterministic starts, the evaluation cache
* `adjoint2d/ms_solve.py` the per-shape driver, the four starts, the two phases,
  the control mode
* `adjoint2d/gate_ms.py` the finite-difference gate, the gradient-direction
  probe, the transpose-consistency bisect, the ramp population and the
  stop-index stability check
* `adjoint2d/ms_robust.py` the two robustness probes on the new maps
* `adjoint2d/build_ms_tables.py` the census tables
* `adjoint2d/make_ms_figures.py` the four figures
* `adjoint2d/tests/test_multistart.py` 29 tests, red first
* `env1.sh`, `run_ms_stream.sh`, `run_ms_control.sh`

Nothing existing was modified. `forward.py`, `adjoint.py`, `shape_objective.py`,
`design_filter.py`, `printability.py`, `robust.py` and `library_solve.py` were
read and imported, not changed.

Results:
* `fgm_solve_campaign/out_ms/<shape>.json` and `<shape>_maps.npz`, 18 shapes
* `fgm_solve_campaign/out_ms/<shape>_control_cold.json` and
  `<shape>_control_cold_maps.npz`, 18 shapes
* `fgm_solve_campaign/out_ms/{square,rectangle}_robust.json` and
  `_robust_maps.npz`
* `fgm_solve_campaign/out_ms/gate_ms_{square,rectangle}.json`
* `fgm_solve_campaign/logs_ms/*.log` per-run console logs and stream timings

Figures, **all four viewed before delivery**:
* `fgm_solve_campaign/figs_ms/fig_ms_census.png` the five-arm comparison and the
  separation of the two changes
* `fgm_solve_campaign/figs_ms/fig_ms_starts.png` the probe, the kill rule, the
  depth-against-breadth mechanism, and all 18 per-start evaluation sequences
* `fgm_solve_campaign/figs_ms/fig_ms_maps.png` the 18 delivered 4-bpp maps
* `fgm_solve_campaign/figs_ms/fig_ms_robust.png` rim and grid, filtered against
  unfiltered

Read, not modified:
* `OVERNIGHT_QUEUE_2026-08-01.md`, `SOLVE_ROBUSTNESS_VALIDATION.md`,
  `DENSITY_OBJECTIVE_LIBRARY_REPORT.md`, `SHAPE_LIBRARY_SOLVE_REPORT.md`
* `fgm_solve_campaign/out_lib/*.json`, `out_lib/*_maps.npz`
* `fgm_solve_campaign/out_robust/{square,rectangle}_{grid,rim}.json`
* `outputs_eqs/fgm_calibrated_control/configs/*_m0p0500.yaml` and the stored
  4-bpp dopant maps the warm start loads
