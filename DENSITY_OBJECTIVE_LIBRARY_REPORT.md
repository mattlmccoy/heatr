# The density-region objective, finite-difference gated, and the 18-shape library scored under both objectives

**Date:** 2026-08-01. **Scope:** a new whole-domain relative-density objective, a
coupled temperature-and-density adjoint, a physical-length design filter applied
inside the solve, and a re-run of the full 18-shape standardized library with
every arm scored under BOTH the melt-region objective and the new density-region
objective. Grid 120 x 120 throughout. Nothing was committed. No dissertation file
was touched. `.claude/worktrees/` was not read or written.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the two-dimensional solver uses). IoU =
intersection over union. bpp = bits per pixel. L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. RF = radio frequency.
phi = melt fraction. rho = relative density. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit-tested or finite-difference gated. COMPUTED =
measured from a real run in this pass. ASSUMED = a modelling choice or an
inference not measured here.

---

## 1. Verdict, up front

**The density-region objective does NOT change the map family. It changes the
stop time, and it makes the density-at-shape-stop WORSE on 14 of 18 shapes.**
COMPUTED, all three parts.

1. **It mostly shifts the stop time.** The density read state (the flat onset of
   J_rho) is **1.42 to 3.26 times later** than the melt read state, median
   1.65 times. Across the library the density-solved deliverable reaches mean
   relative density **0.931 to 0.988** at its own density stop against
   **0.596 to 0.817** at its own melt stop. The whole density gain is bought
   with time.

2. **It does not change the map family; it collapses toward uniform.** The
   density-solved 4-bpp map sits at root-mean-square distance **0.129 (median)**
   from the uniform map and **0.252 (median)** from the melt-solved map, and it
   is closer to uniform than to the melt-solved map on **10 of 18** shapes. On
   **5 of 18** shapes (hexagon, H_shape, ellipse, star6, and to four decimals the
   T_shape) the solve returns **exactly the uniform map**, mean s in part 1.000
   and in-part roughness 0.0000. In-part roughness of the density-solved maps is
   **0.0000 to 0.0459** against **0.0105 to 0.1111** for the melt-solved maps.

3. **It does not fix the shape-versus-density tension. It loses to the melt
   objective on that tension.** Read at its OWN melt stop, the MELT-solved map
   delivers HIGHER mean relative density than the density-solved map on
   **14 of 18** shapes (for example circle 0.874 against 0.738, cross 0.832
   against 0.628, ellipse 0.839 against 0.687). The 0.68-class values of the
   triangle showcase are reproduced and are a property of the READ TIME, not of
   the map: at the melt stop every arm on every shape sits between 0.596 and
   0.822 mean relative density, and the spread between the best and worst map on
   a given shape is 0.006 to 0.205 with a median of 0.069.

4. **What it costs.** At the density read state the melted region has grown far
   past the nominal part: melt IoU falls to **0.335 to 0.726** and bed melt
   reaches **37.7 to 197.8 percent** of the part cell count. On the melt
   objective at the melt stop, the density-solved map beats the melt-solved map
   on only **2 of 18** shapes.

5. **The optimizer barely moves the objective, and that is not a budget
   artifact.** Within the 40 forward-equivalent budget the solve improves J_rho
   by **-0.2 to +14.8 percent** with a median of **0.2 percent**. Quadrupling the
   budget on three shapes (6 to 9 gradient evaluations per start becoming 26 to
   35) changes the final J_rho by **0.003 percent (square), 0.019 percent
   (triangle) and 0.033 percent (L_shape)**. The objective is genuinely weakly
   actuated by the dopant map, not under-optimized.

**The mechanism, and it is the most useful finding here.** The flat-onset stop
rule is close to self-normalizing. It reads every arm at the first step within
1 percent of that arm's own terminal J_rho, so what J_rho measures at that read
state is the ASYMPTOTIC under-densification (the part of the shape that will
never densify however long the generator runs), not the speed of densification.
Almost every map eventually densifies almost everything, so almost every arm
lands at J_rho close to 1 percent of the part cell count. The exceptions are
exactly the maps with genuinely cold regions: on the square the melt-solved map
scores 33.69 against uniform's 16.71, on the cross 312.0 against 23.5, on the
ellipse 37.2 against 7.3.

**The recommendation this produces.** The operationally meaningful density
criterion is **relative density read at the SHAPE-optimal stop**, which is
already tabulated here as `mean_rho_rel_part_at_phi_stop`, and on that criterion
the existing melt-region objective is the better design driver on 14 of 18
shapes. A separately optimized density stop is not a useful second objective; it
is a longer bake.

---

## 2. The finite-difference gate, run BEFORE any optimization

### 2.1 What was gated and how

Central differences, epsilon swept **1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8,
1e-8**, on two shapes (square and L_shape, deliberately a compact convex shape
and a shape with a limb that never melts) and two layers:

* **R1** dJ_rho/ds at a FIXED read index. The pure partial derivative.
* **R2** dJ_rho/dv with the physical-length design filter in the chain,
  s = F(v). This is the gradient the solve actually uses.

Probes per layer: the maximum-sensitivity cell (argmax of the analytic gradient
magnitude), a fixed pseudo-random in-part cell, a random unit direction over all
in-part cells, and for the filtered layer a **smooth random direction** (a random
direction passed through the filter, then normalized). Code:
`fgm_solve_campaign/adjoint2d/gate_rho.py`. Raw:
`out_rho/gate_rho_square.json`, `out_rho/gate_rho_L_shape.json`.

### 2.2 Results

| shape | layer | probe | analytic | best relative error | best epsilon | absolute error |
|---|---|---|---|---|---|---|
| square | R1 | max-sensitivity cell | -6.4655e+00 | **1.394e-06** | 3e-08 | 9.0e-06 |
| square | R1 | random cell | -7.6188e-02 | **5.800e-07** | 1e-05 | 4.4e-08 |
| square | R1 | random direction | +4.0803e-01 | **3.037e-06** | 1e-06 | 1.2e-06 |
| square | R2 filtered | max-sensitivity cell | -1.5758e+00 | **1.740e-06** | 3e-08 | 2.7e-06 |
| square | R2 filtered | random cell | -7.3856e-02 | **6.648e-06** | 1e-05 | 4.9e-07 |
| square | R2 filtered | random direction | +1.6245e-01 | 8.160e-05 | 1e-06 | 1.3e-05 |
| square | R2 filtered | smooth random direction | +8.1737e-01 | **1.848e-05** | 1e-06 | 1.5e-05 |
| L_shape | R1 | max-sensitivity cell | -1.1029e+01 | **5.218e-07** | 1e-06 | 5.8e-06 |
| L_shape | R1 | random cell | +6.8294e-04 | 1.137e-04 | 1e-04 | 7.8e-08 |
| L_shape | R1 | random direction | +1.5355e+00 | **7.889e-06** | 1e-06 | 1.2e-05 |
| L_shape | R2 filtered | max-sensitivity cell | -2.8227e+00 | **1.913e-07** | 1e-06 | 5.4e-07 |
| L_shape | R2 filtered | random cell | -8.3432e-03 | **2.555e-06** | 1e-04 | 2.1e-08 |
| L_shape | R2 filtered | random direction | +7.8321e-01 | 4.473e-05 | 1e-06 | 3.5e-05 |
| L_shape | R2 filtered | smooth random direction | +2.9751e+00 | **4.160e-06** | 3e-07 | 1.2e-05 |

**Verdict, stated exactly and not inflated. The gradient is verified, but it does
NOT reach the 1e-6 clean-smooth standard on most probes.** 3 of 14 probes reach
1e-6; **9 of 14 reach the campaign's documented subgradient standard of 1e-5**
(`SOLVE_ROBUSTNESS_VALIDATION.md` Section 7 records the melt-region gate
bottoming at 1.22e-05 for the same reason class). Best relative errors span
1.9e-07 to 1.1e-04.

**The relative error tracks 1/|analytic|, which is the signature of a fixed
absolute noise floor rather than a wrong gradient.** COMPUTED: the ABSOLUTE
finite-difference error spans only **2.1e-08 to 3.5e-05** across all 14 probes,
and every probe whose analytic derivative exceeds 1.5 gates between **1.9e-07 and
7.9e-06**. The two worst relative numbers are the two smallest analytic
derivatives: the L_shape random cell at analytic 6.8e-04 has the SMALLEST
absolute error of any probe in the table (7.8e-08), and the square filtered
random direction is damped by the filter (measured `||F d|| / ||d|| = 0.2213`,
so the perturbation the forward sees is 4.5 times smaller while the noise floor
is unchanged), which is exactly why the smooth-direction probe was added and why
it gates 4.4 times better on the square and 10.8 times better on the L_shape.

### 2.3 The floor, measured rather than assumed

From the absolute errors and their epsilons, the evaluation floor of J_rho is
**2.5e-12 to 7.0e-11 absolute**, that is **1.4e-13 to 4.1e-13 relative** to
J_rho, roughly **1000 times machine epsilon**. That is what an explicit march of
1500 outer steps by 5 substeps costs in accumulated rounding, and it is not
stochastic: repeating an identical evaluation gives a difference of exactly
0.000e+00 (`logs_rho/diag_scan.log`). With a central difference that floor caps
the achievable relative error at roughly floor divided by (2 epsilon |g|), and
every measured number sits at that cap. **The gate is at its information limit
for a double-precision forward; it cannot be improved by sweeping epsilon
further.**

### 2.4 The single-cell curvature finding, and how it was localized

The first gate run FAILED at 3.637e-02 on the square's maximum-sensitivity cell
with the melt-region gate's epsilon list (1e-3 down to 1e-7). Bisection, not
guessing:

* the random cell passed at 5.800e-07 on the same run, so the adjoint was not
  broadly wrong;
* a direct scan of J_rho along that cell over plus and minus 2e-4 in 21 steps
  (`logs_rho/diag_scan.log`) showed local secant slopes between **+2.0 and
  -12.2** around an analytic -6.47, that is a small high-curvature component,
  with the evaluation itself bit-identical on repeats;
* the SECOND-largest cell reached 4.53e-06 at epsilon 1e-7
  (`logs_rho/diag_bisect.log`), which is where the epsilon window actually opens;
* extending the sweep to 3e-8 and 1e-8 moved the maximum-sensitivity cell from
  3.637e-02 to **1.394e-06**.

The maximum-sensitivity cells are the four CORNERS of the square, where the
electric field is largest. **The cause was the epsilon window, not the
gradient**, and the fix is recorded in the code with the measurement that
motivates it.

### 2.5 The moving read state, measured instead of assumed

J_rho is monotone non-increasing (relative density only rises), so its argmin is
ALWAYS the last stored step and the envelope argument that covers the melt-region
objective does NOT transfer: the flat-onset stop is not a stationary point and a
dt_flat/ds term exists in principle. Rather than assume it small,
`gate_rho.read_index_stability` recomputes the flat-onset index under every probe
perturbation at epsilon 1e-3. **COMPUTED: the index does not move on either
shape (square base index 1338, L_shape base index 1460, `moved = False` on all
probes), so over the tested perturbation range that term is exactly zero and the
fixed-read gradient IS the rule-pinned gradient.**

### 2.6 Does the melt-front subgradient floor apply

**Partly, and it is not the binding constraint here.** The density seed reaches
the dopant map through the densification rate, which carries the melt-fraction
clip masks (`phi_out_range`, `phi_act_range`, the solid-branch mask) and the
per-substep densification cap, so subgradients are present in the chain exactly
as they are for the melt-region objective. But the binding limit measured above
is the arithmetic floor of a 7500-substep march plus, for single-cell probes at
field corners, high curvature. Evidence that the non-smooth population is not
dominant: the epsilon sweeps bottom cleanly and then degrade monotonically with
smaller epsilon, which is roundoff behaviour, not the epsilon-independent plateau
a dominant kink produces.

### 2.7 Unit tests, red first

**138 tests pass** (`.venv312/bin/python -m pytest adjoint2d/tests -q`), of which
**36 are new in this pass and every one was observed failing first**:

* `tests/test_design_filter.py`, 9 tests. Observed red as `ImportError: cannot
  import name 'design_filter'`. Includes the dot-product identity
  `<F v, w> == <v, F^T w>` to 1e-12, no bleed from the nominal value held outside
  the part, box preservation without a clip, and monotone roughness reduction.
* `tests/test_density_objective.py`, 15 tests. Observed red as `ImportError`.
  Includes the seed against a cell-by-cell central difference, the bed
  contributing exactly zero whatever its stored value, and four tests of the
  flat-onset stop rule.
* `tests/test_rho_state_access.py`, 4 tests. Observed red as 4 failures before
  `Trajectory.rho_at_end` existed.
* `tests/test_rho_solve_logic.py`, 8 tests. Observed red as 8 failures before
  `gradient_death` and `better_start` existed.

Two of my own test expectations were wrong and were corrected against the
implementation after checking the arithmetic by hand (the flat-onset band
1.0 + 0.01 x 9.0 = 1.09, and the count of normalized densities k/16 clearing
0.5 being 9 not 8). Both are noted here rather than silently fixed.

---

## 3. Definitions and conventions, stated once

### 3.1 The objective and the normalization

    J_rho(s, t) = sum over the WHOLE domain of (rho_norm(x, t) - chi_part(x))^2

    rho_norm = (rho_rel - rho_floor) / (1 - rho_floor),  rho_floor = rho_rel_init

`rho_rel` is the relative-density state integrated by `forward.substep`, which
reproduces the production `physics_dual` densification model of
`rfam_eqs_coupled`: a solid-state Arrhenius branch `k0_ss exp(-Ea_ss / R T)`
times `(1 - phi)^e_s`, plus a viscous-capillary liquid branch
`geom * surface_tension / (eta(T) * particle_radius)` times a thresholded melt
drive, the sum scaled by `(1 - rho)^e_r` and capped per substep at
`max_delta_per_step / n_substeps`. `rho_floor` is the configured powder-bed
starting relative density `densification.rho_rel_initial`, **0.55 in every
configuration of this campaign**. The forward initializes rho at the floor on the
part, drho is non-negative and rho is clipped at 1, so rho_rel lies in
[0.55, 1] and rho_norm lies in [0, 1] with **no clip required**.

### 3.2 The bed, and the asymmetry that follows

`forward.substep` integrates rho only inside the part mask
(`rho_new[pm] = clip(...)`, `forward.py:279-282`) and leaves the bed at the
stored value 0, which is bookkeeping and not physics: undensified powder sits at
the powder floor. The bed is therefore **extended at rho_floor**, giving
rho_norm = 0 = chi_part there, so **the bed contributes identically zero to the
whole-domain sum at every time**. Consequence, stated rather than hidden:
**unlike the melt-region objective, which charges the same for a melted bed cell
as for an unmelted part cell, J_rho has NO growth term.** It penalizes
under-densification of the part only. Every comparison between the two
objectives carries this.

### 3.3 The two read states

* **PHI-STOP** t = argmin over that arm's own stored trajectory of J_phi. The
  melt objective turns when melt spills into the bed, so its argmin is interior.
* **RHO-STOP** the **flat onset** of J_rho: the first stored step whose J_rho is
  within `tol = 0.01` of the terminal value, measured as a fraction of the total
  decrease `J[0] - J[-1]`.

**The saturation guard, and why it was needed.** COMPUTED on every arm of every
shape: the unguarded argmin of J_rho is the LAST stored step, 18 of 18 shapes,
every arm. Reading there hands the optimizer an objective whose per-cell terms
are zero, whose `(1 - rho)^e` rate factors are zero, and whose rho clip
subgradient has closed. The flat-onset rule pins the read state before that.
**Reported loudly on every arm**: the flat-onset index, the argmin index, an
`at_horizon` flag when the curve had not flattened inside the horizon (COMPUTED:
**zero occurrences** across the library), a `no_progress` flag, and the fraction
of the total decrease still remaining at the read state.

**The gradient-death assertion.** Per iterate, `rho_solve.gradient_death` fires
when the infinity norm of the gradient falls to 1e-6 of the first iterate's, with
an absolute floor so a run that starts dead is not certified healthy. COMPUTED:
**zero firings across the library**, and the gradient does not decay during the
solves (square 1.094 to 1.093, triangle 1.189 to 1.209 at 4x budget, L_shape
4.008 to 3.930 at 4x budget). The objective stalls with a live gradient, which is
a flatness result and not a saturation result.

**The shape-fidelity early stop is DISABLED on every arm**, because it truncates
the march on the melt objective long before the density objective flattens. Every
march runs the full 1500-step horizon. **Consequence: the J_phi numbers here are
re-scored and are NOT carried across from `SHAPE_LIBRARY_SOLVE_REPORT.md`**; a
full-horizon argmin can only find a J_phi lower than or equal to the truncated
one.

### 3.4 The design filter, applied inside the solve

Mandated by the robustness verdict. The design variable is v and the injected map
is

    s = F(v) = gaussian(v * chi) / gaussian(chi)   on the part,   s = 1 outside

a normalized convolution over the part only at **sigma = 1.5 cells**, which at
the pinned geometry is **0.75 mm** at grid 120. Two properties that matter:
there is **no clip** (a normalized convolution is a convex combination of in-part
values, so v in [0, 1] gives s in [0, 1] exactly, PROVEN by unit test), and the
transpose is `F^T g = chi * gaussian(chi * g / gaussian(chi))`, exact because a
Gaussian correlation with zero padding is self-adjoint, PROVEN to 1e-12 by the
dot-product identity. The filter acts on the DESIGN VARIABLE, so sub-resolution
rim structure is not expressible rather than merely discouraged.

**Densified region.** rho_norm >= 0.5. **Melted region.** phi >= 0.5.
**Density.** mean relative density in part cells.

---

## 4. What the filter costs, measured

Unfiltered controls were run on the square and the circle, same budget, same two
starts.

| shape | arm | J_rho | mean rho | melt IoU at the melt stop | in-part roughness |
|---|---|---|---|---|---|
| square | filtered, 4 bpp | 16.399 | 0.9615 | 0.8800 | **0.0146** |
| square | unfiltered, 4 bpp | 16.302 | 0.9607 | 0.8821 | **0.0629** |
| circle | filtered, 4 bpp | 12.309 | 0.9786 | 0.8246 | **0.0251** |
| circle | unfiltered, 4 bpp | 12.282 | 0.9814 | 0.8112 | **0.0186** |

**COMPUTED: the filter costs +0.59 percent of J_rho on the square and
+0.22 percent on the circle, and it removes 77 percent of the in-part roughness
on the square.** The square's unfiltered map carries a visible single-cell rim
stripe along the top and bottom edges (`figs_rho/fig_rho_square.png`, top row,
rightmost panel) which the filtered map does not have. **On the circle the
unfiltered map is the SMOOTHER of the two (0.0186 against 0.0251)**, because its
best start barely moved from uniform and there was nothing to sculpt; that is
reported rather than dropped. At 4 times the budget the square's unfiltered arm
reaches J_rho 16.10 with roughness 0.0453 against the filtered 16.40, so the
filter cost grows to 1.9 percent with more optimization while still buying a
3.1-fold roughness reduction.

**Reading.** For this objective the filter is nearly free. That is a weaker claim
than it sounds, because this objective barely moves at all; the filter's value
should be re-measured on an objective that the map can actually drive.

---

## 5. The census, both objectives, 18 shapes

Every number at the stated stop for that arm. Grid 120 x 120. `J_rho` at the
RHO-STOP, `J_phi` and melt IoU at the PHI-STOP, `rho@phi` is mean relative
density read at the arm's own melt stop, `rho@rho` at its own density stop.

| shape | part cells | start | J_rho solved | J_rho uniform | J_rho hist | J_rho melt-solved | rho@phi | rho@rho | melt IoU solved | melt IoU melt-solved | t_phi s | t_rho s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | 1600 | warm | 16.40 | 16.71 | 15.83 | 33.69 | 0.6913 | 0.9615 | 0.8800 | 0.9816 | 425 | 643 |
| circle | 1240 | cold | 12.31 | 12.34 | 12.23 | 19.37 | 0.7378 | 0.9786 | 0.8246 | 0.9904 | 354 | 563 |
| hexagon | 1016 | cold | 10.02 | 10.02 | 10.07 | 10.21 | 0.7951 | 0.9800 | 0.7649 | 0.9767 | 326 | 505 |
| triangle | 800 | cold | 7.99 | 7.98 | 22.81 | 7.98 | 0.6985 | 0.9713 | 0.7752 | 0.8578 | 254 | 446 |
| equilateral_triangle | 674 | cold | 6.68 | 6.73 | 6.66 | 6.73 | 0.7544 | 0.9815 | 0.5955 | 0.8128 | 240 | 460 |
| L_shape | 1079 | cold | 139.54 | 139.92 | 722.84 | 149.41 | 0.6572 | 0.9305 | 0.5137 | 0.5209 | 264 | 731 |
| H_shape | 1152 | cold | 11.38 | 11.38 | 11.38 | 11.56 | 0.6653 | 0.9686 | 0.7563 | 0.8423 | 348 | 579 |
| T_shape | 1104 | warm | 122.23 | 122.24 | 772.28 | 123.60 | 0.6232 | 0.9307 | 0.4574 | 0.4444 | 230 | 738 |
| cross | 1036 | cold | 22.94 | 23.52 | 11.53 | 312.03 | 0.6275 | 0.9757 | 0.5539 | 0.6755 | 222 | 721 |
| diamond | 1624 | cold | 133.38 | 156.52 | 112.63 | 463.40 | 0.7245 | 0.9408 | 0.6338 | 0.8520 | 465 | 734 |
| ellipse | 744 | cold | 7.34 | 7.34 | 7.33 | 37.24 | 0.6867 | 0.9840 | 0.6909 | 0.9787 | 214 | 436 |
| octagon | 1076 | cold | 10.66 | 10.69 | 10.57 | 10.74 | 0.8173 | 0.9762 | 0.9759 | 0.9944 | 324 | 466 |
| pentagon | 930 | cold | 9.29 | 9.30 | 9.27 | 9.60 | 0.7472 | 0.9761 | 0.7034 | 0.9387 | 330 | 538 |
| rectangle | 1152 | cold | 11.50 | 11.51 | 11.49 | 11.36 | 0.6568 | 0.9612 | 0.8462 | 0.8424 | 322 | 526 |
| rounded_rect | 1536 | warm | 15.49 | 15.79 | 15.16 | 72.57 | 0.7566 | 0.9677 | 0.8568 | 0.9974 | 437 | 618 |
| star | 542 | warm | 5.39 | 5.40 | 5.38 | 5.38 | 0.5957 | 0.9879 | 0.6784 | 0.7032 | 150 | 458 |
| star6 | 592 | cold | 5.87 | 5.87 | 5.81 | 5.88 | 0.6729 | 0.9758 | 0.6758 | 0.8408 | 196 | 362 |
| trapezoid | 1189 | warm | 11.86 | 11.83 | 11.83 | 11.89 | 0.7163 | 0.9681 | 0.8733 | 0.9718 | 350 | 539 |

### 5.1 Census counts, each against a named baseline

| comparison | count |
|---|---|
| density-solved beats **uniform** on J_rho | **12 of 18** |
| density-solved beats the **best stored historical mask** on J_rho | **5 of 18** |
| density-solved beats the **melt-solved 4-bpp map** on J_rho | **15 of 18** |
| density-solved beats the melt-solved map on the fraction of the part above normalized density 0.90 | **12 of 18** |
| density-solved beats the melt-solved map on **melt IoU at the melt stop** | **2 of 18** |
| melt-solved map gives HIGHER mean relative density **at its own melt stop** | **14 of 18** |

**The 12 of 18 against uniform is a weak win and must be quoted as one.** The
median J_rho improvement over uniform is 0.2 percent; the three real wins are the
diamond (156.52 to 133.38, 14.8 percent), the cross (23.52 to 22.94,
2.5 percent) and the square (16.71 to 16.40, 1.9 percent). On three shapes the
solve ends marginally WORSE than uniform (triangle +0.1 percent, trapezoid
+0.3 percent, and rectangle statistically level), which L-BFGS-B is permitted to
do when the best iterate by J_rho is quantized to 4 bpp afterwards.

**The 5 of 18 against the historical mask is the honest hard baseline.** The
stored calibrated masks were selected against a completely different criterion
and still beat the density solve on 13 shapes, usually by under 1 percent,
because at the flat-onset read state almost anything densifies.

### 5.2 The densified-region IoU degenerates, and the stricter level that does not

The densified region is a subset of the part by construction (there is no bed
densification branch), so `IoU_rho` is exactly the densified FRACTION of the
part. COMPUTED: at the flat-onset read state that fraction is **1.0000 on 24 of
54 arms** and above 0.85 on nearly all the rest, so **the half level cannot
discriminate between arms and must not be quoted as a result**. The census
therefore also reports the fraction of the part above **normalized relative
density 0.90**, which does discriminate: the melt-solved map reaches only 0.26 on
the square, 0.40 on the cross, 0.44 on the ellipse and 0.50 on the rounded_rect
against 0.68 to 0.93 for uniform and the density-solved arm.

### 5.3 An instance of the pathology the guard is for

The **pentagon** is worth naming. Its density solve carved a deep low-saturation
hole in the middle of the part (`figs_rho/fig_rho_pentagon.png`, top row,
rightmost panel, minimum saturation reaching zero), which produces an unmelted
island at the melt stop and costs melt IoU 0.7034 against uniform's 0.7130. It
bought **0.1 percent of J_rho** (9.29 against 9.30). That is structure the
objective could not pay for, discovered with a filtered design variable and a
pinned pre-saturation read state, which is a useful demonstration that the guard
narrows the pathology without eliminating it.

---

## 6. Budget and wall time, with the projection that was required

**Budget: 40 forward-equivalents per shape solve, warm and cold combined, with no
overrun.** The adjoint-to-forward cost ratio was measured per shape (1.23 to
2.01, mean about 1.8), giving **6 to 9 gradient evaluations per start** and a
**mean 37.0 forward-equivalents actually spent per shape**. No arm was silently
cut. The square and the circle additionally carry the unfiltered control, which
is a second 40-forward-equivalent solve and is labelled as such.

**Wall time, logged after two shapes and projected as required.** The first two
completions were equilateral_triangle at 260 s and circle at 484 s, mean 372 s,
which projected to 18 x 372 / 3 parallel streams = **37 minutes**. Actual:
**5063 s of process time across 18 shape jobs, 241 to 515 s each, about 24
minutes of wall clock in three streams**. The projection was honoured and nothing
was dropped.

Additional compute, all labelled as extra and outside the per-shape budget:
the two finite-difference gates at 782 s and 778 s, an earlier superseded gate
run at 540 s, the diagnostic scans at about 600 s, the three quadruple-budget
controls at 863, 1032 and 1902 s, and the figure generation. Total for the pass
is about 3.5 hours of process time.

**The quadruple-budget control, which the task did not ask for but the weak
result needed.**

| shape | gradient evaluations per start | J_rho at budget 40 | J_rho at budget 160 | change |
|---|---|---|---|---|
| square | 7 to 28 | 16.3992 | 16.3987 | **-0.003 %** |
| triangle | 9 to 35 | 7.9942 | 7.9927 | **-0.019 %** |
| L_shape | 6 to 26 | 139.5433 | 139.4977 | **-0.033 %** |

Mean relative density and melt IoU are unchanged to four decimals on all three.
**The weak result is a property of the objective, not of the budget.**

---

## 7. Gates and violations

**The standing 5 percent energy-residual gate FAILS at the density read state on
4 of 18 shapes.** COMPUTED and reported loudly:

| shape | worst residual at the DENSITY stop | residual at the MELT stop | density stop / melt stop |
|---|---|---|---|
| star | **5.66 to 6.17 %** (5 arms) | 0.55 to 0.79 % | 3.06 |
| L_shape | **6.06 to 6.10 %** (4 arms) | 0.68 to 0.85 % | 2.77 |
| T_shape | **5.74 to 5.75 %** (4 arms) | 0.33 to 0.62 % | 3.21 |
| cross | **5.26 to 5.28 %** (3 arms) | 0.5 to 0.6 % | 3.26 |

Sixteen arms in total: L_shape and T_shape (uniform, melt-solved, and both
density-solved arms), cross (uniform and both density-solved arms), star (all
five arms including the historical mask). **At
the MELT stop the gate is clean on every arm of every shape, worst residual
2.22 percent, zero violations.** The mechanism is march length: the four
offenders are exactly the four shapes whose density stop is more than 2.7 times
their melt stop, and the incremental stored-energy bookkeeping accumulates with
the number of steps. **Every density-stop number on those four shapes is a
flagged number and must not be quoted without the flag.** The 14 other shapes
pass at both stops.

**Other gates.** Zero gradient-death firings. Zero arms with the density read
state at the horizon. The three clip fractions (temperature step cap, temperature
range cap, Q_rf cap) are zero on the square's gate point, checked directly.

---

## 8. What is new in the code

All under `fgm_solve_campaign/adjoint2d/`.

* `density_objective.py` the normalization, the objective and its seed, the
  flat-onset stop rule with its guard flags, and the region metrics.
* `design_filter.py` the part-masked normalized-convolution filter, its exact
  transpose, and a roughness diagnostic.
* `forward.py` one addition, `Trajectory.rho_at_end`, the relative-density twin
  of `T_at_end`, same checkpoint convention.
* `adjoint.py` two additions, `seeds_rho` on `reverse_march` and on `gradient`.
  **The substep vector-Jacobian product was ALREADY coupled in (T, rho)**, so the
  density objective needed no new physics in the reverse sweep, only its own seed
  on the density co-state. With `seeds_rho` empty the sweep is bit-for-bit the
  temperature-seeded one and every previously reported gradient is unaffected.
* `gate_rho.py` the finite-difference gate, the extended epsilon sweep, the
  smooth-direction probe, and the read-index stability check.
* `rho_solve.py` the per-shape driver, the two starts, the gradient-death
  assertion, and the dual-objective scoring with both cross reads.
* `make_rho_figures.py` the census and per-shape figures.
* `tests/test_density_objective.py`, `tests/test_design_filter.py`,
  `tests/test_rho_state_access.py`, `tests/test_rho_solve_logic.py`, 36 tests.

---

## 9. Proven, computed, assumed

**PROVEN**
* The design filter's transpose satisfies the dot-product identity to 1e-12; the
  filter preserves the box without a clip; the nominal value held outside the
  part does not bleed in (a uniformly 0.2 part stays 0.2 to 1e-12 with 1.0
  outside); it is the identity at radius zero.
* The density objective's seed matches a cell-by-cell central difference; the bed
  contributes exactly zero whatever its stored value; the normalization maps the
  configured floor to 0 and full density to 1.
* The flat-onset rule returns the flat onset and not the argmin on a plateauing
  curve, flags the horizon when the curve is still falling, and flags no progress
  on a flat curve.
* 138 unit tests pass; the 36 new ones were red before they were green.

**COMPUTED**
* Every number in Sections 1, 2, 4, 5, 6 and 7.
* The gradient is verified against the real forward at absolute errors of
  2.1e-08 to 3.5e-05 across 14 probes on 2 shapes and 2 layers.
* The flat-onset read index does not move under any probe perturbation.
* J_rho's argmin is the horizon on every arm of every shape.
* The objective's evaluation floor, 1.4e-13 to 4.1e-13 relative.

**ASSUMED, and how it bites**
1. **That the bed should be extended at the powder floor.** It is the physically
   correct reading of a bookkeeping zero, but it means J_rho has no growth term
   at all, and that asymmetry is most of why the objective behaves as it does. An
   objective with a bed-densification penalty would be a different experiment.
2. **That a 1 percent flat-onset tolerance is the right read state.** It is a
   convention. Section 1 shows it is close to self-normalizing, which is the main
   limitation of this whole pass. A looser tolerance reads earlier and
   discriminates more; 0.20, 0.10 and 0.05 were measured on the square's uniform
   arm (read indices 997, 1087, 1161 against 1301 at 0.01) but the library was
   not re-run at another tolerance.
3. **That the design filter sigma of 1.5 cells is the right physical length.**
   It is inside the 1.5 to 2 cell band the robustness verdict named. No bench
   measurement of the real rim blur exists, so the length is in cells (0.75 mm at
   grid 120) and not in a measured process length.
4. **That the rasterized binary part mask is the right nominal target.** Carried
   over from the melt-region work, still untested.
5. **That an arbitrary stop time is realizable as a process control.** This
   matters much more here than it did for the melt objective, because the density
   read state is 1.4 to 3.3 times later and the four flagged shapes need more
   than 720 s of continuous drive.
6. **The adjoint arms are conductivity-only; every historical arm co-varies
   permittivity.** Unchanged, still the largest actuator gap.
7. **The forward is the two-dimensional `adjoint2d` engine, not `heatr3d`.**
   Every simplification documented for the shape-library solve applies unchanged.
8. **No experimental validation.** These are two-dimensional model results.

---

## 10. Honest limits

1. **The gate does not reach 1e-6 on most probes** (9 of 14 at 1e-5, 3 of 14 at
   1e-6). The limit is arithmetic, measured, and cannot be swept away, but it is
   a weaker gate than the melt-region single-cell gate at a short read state.
2. **Two shapes were gated, not eighteen.** The library solves inherit that gate.
3. **The energy-residual gate fails at the density read state on 4 of 18
   shapes.** Section 7.
4. **The densified-region IoU is degenerate at the read state** (1.0000 on 24 of
   54 arms) and the 0.90-level fraction was substituted. Do not quote a
   densified-region IoU from this pass as a fidelity result.
5. **No arm is dose matched.** Absorbed power spans 461.4 to 503.8 W/m across
   the density-solved deliverables, which is much tighter than the melt-region
   census span of 221.3 to 792.4 W/m only because the density-solved maps are
   nearly uniform.
6. **Everything is grid 120.** `SOLVE_ROBUSTNESS_VALIDATION.md` established that
   absolute fidelity at this grid does not transfer to 160. No claim here should
   be quoted without the grid qualifier.
7. **The budget buys 6 to 9 gradient evaluations per start on 542 to 1624 design
   variables.** The quadruple-budget control on three shapes shows that does not
   change the conclusion, but it was not run on the other fifteen.
8. **Cold and warm starts differ little** because the objective is flat: the warm
   start wins on 5 of 18 shapes, by 2.46 percent of J_rho on the square and by
   0.13 percent or less on the other four (T_shape 0.006, rounded_rect 0.000,
   star 0.008, trapezoid 0.126 percent).

---

## 11. The single most valuable next layer

**Solve for relative density READ AT THE SHAPE-OPTIMAL STOP, not at a separately
optimized density stop.** That is the objective the data here actually points to:

    minimize over s of  J_rho(s, t_phi*(s))  subject to  t_phi*(s) = argmin_t J_phi

or, more usefully as a scalarization, `J_phi + lambda * J_rho` both read at the
melt stop, with lambda swept. Three reasons it is the right next step, each
supported by a number in this report: the melt-solved map already beats the
density-solved map on density-at-melt-stop on 14 of 18 shapes, so there is a
real gradient to exploit in that direction; at the melt stop the density
objective is far from saturated (J_rho at the melt stop is 252 to 880 against 5
to 140 at the density stop) so the objective has leverage the flat-onset version
does not; and the read state stays inside the energy-residual gate on all 18
shapes. The adjoint needs no new physics, because the coupled (T, rho) reverse
sweep is already in place and gated; it needs both seeds active at the same read
index, which `gradient` already accepts.

**Second, and cheap: sweep the flat-onset tolerance.** One shape, four
tolerances, reusing a single forward per arm, would say directly how much of the
null in Section 1 is the rule rather than the physics.

**Third: give J_rho a bed term.** The densification model has no bed branch, so
the objective cannot penalize spilling. Adding an explicit penalty on melt in the
bed to the density objective would restore the symmetry that makes the
melt-region objective well behaved, at the cost of no longer being a pure density
objective. That is a modelling decision, not a numerical one, and it should be
made deliberately.

---

## 12. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code:
* `fgm_solve_campaign/adjoint2d/density_objective.py`
* `fgm_solve_campaign/adjoint2d/design_filter.py`
* `fgm_solve_campaign/adjoint2d/gate_rho.py`
* `fgm_solve_campaign/adjoint2d/rho_solve.py`
* `fgm_solve_campaign/adjoint2d/make_rho_figures.py`
* `fgm_solve_campaign/adjoint2d/tests/test_density_objective.py`
* `fgm_solve_campaign/adjoint2d/tests/test_design_filter.py`
* `fgm_solve_campaign/adjoint2d/tests/test_rho_state_access.py`
* `fgm_solve_campaign/adjoint2d/tests/test_rho_solve_logic.py`
* `fgm_solve_campaign/run_rho_stream.sh`, `run_rho_b160.sh`, `run_rho_figs.sh`
* `fgm_solve_campaign/diag_smoothdir.py` the filtered smooth-direction probe

Modified code, additively and with the previous behaviour bit-for-bit preserved:
* `fgm_solve_campaign/adjoint2d/forward.py` (`Trajectory.rho_at_end`)
* `fgm_solve_campaign/adjoint2d/adjoint.py` (`seeds_rho` on `reverse_march`
  and `gradient`)

Results, one JSON and one npz of maps per shape:
* `fgm_solve_campaign/out_rho/<shape>.json`, `<shape>_maps.npz`, 18 shapes
* `fgm_solve_campaign/out_rho/gate_rho_square.json`,
  `out_rho/gate_rho_L_shape.json`
* `fgm_solve_campaign/out_rho_b160/{square,triangle,L_shape}.json` the
  quadruple-budget control
* `fgm_solve_campaign/logs_rho/*.log` per-run console logs, the diagnostic scans
  and the stream timings

Figures, **all 19 viewed before delivery**:
* `fgm_solve_campaign/figs_rho/fig_rho_census.png`
* `fgm_solve_campaign/figs_rho/fig_rho_<shape>.png`, 18 shapes

Read, not modified:
* `SOLVE_ROBUSTNESS_VALIDATION.md`, `TEMPORAL_SCHEDULING_REPORT.md`,
  `SHAPE_LIBRARY_SOLVE_REPORT.md`, `OVERNIGHT_QUEUE_2026-08-01.md`
* `fgm_solve_campaign/out_lib/*.json`, `out_lib/*_maps.npz`
* `outputs_eqs/fgm_calibrated_control/configs/*_m0p0500.yaml` and the stored
  4-bpp dopant maps the historical arms load

**gt_logo was SKIPPED, loudly.** It is not in the 18-shape standardized library
for the same reason as before: its geometry is rasterized from an image and
`rfam_eqs_coupled.make_domain` raises `ModuleNotFoundError: No module named
'cv2'` in the `.venv312` interpreter. Nothing in this pass changed that. The
library count is 18, not 19.
