# Dense if and only if in bounds: an asymmetric shape-and-density objective, finite-difference gated, solved with two optimizers on five shapes

**Date:** 2026-08-02. **Scope:** a new objective that encodes the refined end goal
(every part fully dense if and only if it falls within the nominal shape bounds,
with an explicit asymmetric trade), its finite-difference gate on both sides of
the new hinge nonsmoothness, solves with L-BFGS-B and with MMA at a matched
40 forward-equivalents on five shapes, the measured shape-against-density trade
curve, and a density-floor sweep on the hexagon. **Grid 120 x 120 throughout.**
Nothing was committed. No dissertation file was touched. `.claude/worktrees/`,
`deck_gifs/`, `studio_handoff/`, `webui/`, `rfam_gui_server.py`,
`geometry_actuator.py` and `geometry_intake.py` were not read or written.

**Acronyms, expanded on first use.** RFAM = radio-frequency additive
manufacturing. FGM = functionally graded material, a spatially varying dopant
saturation map. EQS = electro-quasi-static, the low-frequency Maxwell
approximation the two-dimensional solver uses. IoU = intersection over union.
bpp = bits per pixel. L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. MMA = the method of
moving asymptotes (Svanberg 1987). RF = radio frequency. phi = melt fraction.
rho_rel = relative density. W/m = watts per metre of depth.

**Evidence tags.** PROVEN = unit-tested or finite-difference gated.
COMPUTED = measured from a real run in this pass. ASSUMED = a modelling choice
or an inference not measured here.

---

## 1. Verdict, up front

**The asymmetric objective is well posed, its gradient is gated, and it changes
the STOP far more than it changes the MAP. It does not reach the specification
on any shape at grid 120 with conductivity-only actuation, and the trade it was
built to express turns out to live almost entirely in the read state, which is
why the out-of-bounds price is the production knob and the density floor is
not.** COMPUTED, all four parts.

1. **The objective has an interior optimum in time, which the pure density
   objective never had.** The out-of-bounds term rises with time and the
   in-bounds deficit falls, so the argmin of J_asym is interior and the stop is
   pinned by physics rather than by a tolerance. COMPUTED: on all 42 scored arms
   the argmin is interior except three arms on the cross (Section 6), zero arms
   stop at the first step, and the soft term is never dead at the stop. That
   removes the flat-onset self-normalization that
   `DENSITY_OBJECTIVE_LIBRARY_REPORT.md` Section 1 identified as the main
   limitation of the density objective.

2. **It is mostly the stop, not the map.** Reading the SAME melt-solved
   4-bits-per-pixel map at its own J_asym argmin instead of at its melt argmin
   improves J_asym by **4.3 to 36.1 percent**. Re-solving the map on top of that
   then changes J_asym by only **+17.5, +6.5, +5.3, +0.1 and -15.6 percent**
   (triangle, square, cross, L_shape, hexagon). **The asymmetric solve beats the
   melt-solved map on 3 of 5 shapes, ties on 1 and loses on 1.** Against uniform
   the win is large and consistent: **58.5, 62.2, 34.9, 49.2 and 2.2 percent**.

3. **It does produce a new map family, and the family is "more dopant, later
   stop".** Root-mean-square distance of the delivered map from the melt-solved
   map is **0.166 to 0.345**, and from uniform **0.137 to 0.533**, so the
   asymmetric maps are neither the melt maps nor uniform. Mean in-part
   saturation rises from 0.610 to 0.923 (melt-solved) to **0.570 to 0.953**
   (asymmetric), and the delivered stop is **43 to 160 stored steps later** than
   the same arm's melt stop on every arm of every shape. This is not the
   collapse-to-uniform the pure density objective showed.

4. **Nothing satisfies the specification.** With the acceptance rule stated once
   in code (growth at or below 1.0 percent of the part cell count AND at least
   95 percent of in-bounds cells at or above the floor), **0 of 42 arms pass**.
   The best deliverable of each shape sits at growth **6.50 to 16.00 percent**
   with **22.6 to 58.5 percent** of the part above the 0.85 relative-density
   floor; across all four arms of all five shapes the ranges are 6.50 to 37.00
   percent and 14.7 to 66.9 percent. The star at
   (0, 100) in `figs_asym/fig_asym_census.png` panel B is empty.

**The trade, quantified, and the finding that matters most.** Sweeping the
out-of-bounds price w_out over a stored pair of curves recovers the exact stop
and outcome at every price from ONE forward run, so the exchange rate was
measured directly on the delivered map (`figs_asym/fig_asym_trade.png`). On the
square, going from w_out = 1 to w_out = 2 buys **growth 6.50 to 2.12 percent and
melt IoU 0.9366 to 0.9731** and pays **mean in-bounds relative density 0.8405 to
0.8135**. On the hexagon the same step buys **7.09 to 2.76 percent growth and
IoU 0.9301 to 0.9579** for **0.8384 to 0.8051** density. That is exactly the
asymmetric trade the specification asks for, it is cheap, and it lands inside
the researcher's stated 0.80 to 0.90 acceptance band. **At w_out = 20 growth
reaches 0.00 percent on the square and the hexagon, at relative density 0.759
and 0.757, which is below the band.**

**The floor recommendation: keep 0.85 relative density.** COMPUTED on the
hexagon at 0.80, 0.85 and 0.90: the OUTCOME is weakly sensitive (growth 6.69,
7.09, 7.48 percent; melt IoU 0.9354, 0.9301, 0.9267; mean in-bounds relative
density 0.8291, 0.8384, 0.8598) while the MAP is moderately sensitive (mean
in-part saturation 0.609, 0.671, 0.772; root-mean-square distance from the 0.85
map 0.109 and 0.163). Raising the floor from 0.80 to 0.90 buys **0.031 of mean
relative density** and costs **0.79 growth points and 0.009 IoU points**, so the
choice is low risk anywhere in the band. 0.85 is the midpoint of the stated
band, it is the only one of the three whose achieved mean density (0.838) lands
inside the band rather than on its edge, and at 0.90 only **51.0 percent** of
the part clears its own floor, which starts to switch the hinge on everywhere
and re-symmetrize the objective.

**MMA earns its place on this objective.** At a matched 40 forward-equivalents
MMA beats L-BFGS-B on **4 of 5** shapes (triangle by 21.4 percent, cross 10.4,
hexagon 2.1, L_shape 0.5) and loses on the square by 2.1 percent. That is a
stronger result than `MMA_RETEST_REPORT.md` Section 1 item 3 found at the same
budget under the projection continuation (4 of 6, with two catastrophic
failures). There is no continuation here, so MMA spends every evaluation on a
design update while L-BFGS-B spends part of its pool on line-search trial
points.

**One-sentence answer.** The asymmetric objective is the right encoding of the
specification and it is gated and solvable, but at grid 120 with the
conductivity-only actuator it cannot reach "dense if and only if in bounds" on
any shape, and what it actually buys is a principled, cheap, per-shape exchange
rate between out-of-bounds growth and in-bounds density that is set by the
out-of-bounds price and realized through the stop time.

---

## 2. The objective, stated before the code was written

    J_asym(s, t) = [ w_out * sum over the BED of phi(x, t)^2
                   + w_in  * sum over the PART of h(rho_rel(x, t))^2 ] / n_part

    h(rho_rel) = max(0, floor - rho_rel) / (floor - rho_floor)      in [0, 1]

`n_part` is the part cell count, so one fully melted bed cell costs exactly
`w_out / n_part` and one completely unsintered part cell costs exactly
`w_in / n_part`. Both terms are read at the SAME time index.

### 2.1 Why the two sides use different state variables

**Out of bounds uses melt fraction, and that is FORCED by the model rather than
chosen.** `forward.py` integrates relative density only inside the part mask
(`rho_new[pm] = clip(...)`), so the densification state literally cannot express
bed growth. Bed growth is therefore measured as melted powder, which is the
physical fusing event that produces material to be machined off.

**In bounds uses relative density, and that is a choice with three reasons.**
First, "fully dense" is a density statement, and melt fraction is a transient
state that relaxes when the generator turns off while relative density survives
into the finished part. Second, the density hinge already subsumes under-melting:
a part cell that never melts never gets the viscous-capillary branch and never
densifies, so it sits deep below the floor. Third, melt fraction on both sides
would collapse this into a hinged variant of the melt-region objective and would
not encode density at all.

**The read-state gaming that killed the previous density objective does not
apply here, and that was checked rather than assumed.** `J_rho` is monotone
non-increasing, so its argmin is always the horizon and its read state had to be
pinned by a 1 percent flat-onset tolerance which turned out to be close to
self-normalizing. Here the two terms move in opposite directions in time, so the
argmin is interior and the stop is the same convention the melt-region objective
uses. Because the stop is stationary, the envelope theorem removes the dt*/ds
term; the argmin index was recomputed under every gate perturbation and **did
not move** (Section 3.4).

### 2.2 The floor, in the researcher's units

The floor is quoted in RELATIVE DENSITY, so 0.85 means 85 percent of full
density, not 0.85 of the [0, 1] normalized variable of `density_objective`. At
the campaign's powder floor of 0.550 the equivalents are: floor 0.80 = 0.556
normalized, 0.85 = 0.667, 0.90 = 0.778. **Default frozen before any solve:
0.85.** Swept 0.80 / 0.85 / 0.90 on the hexagon (Section 7).

### 2.3 The weights, and where the asymmetry actually lives

**Frozen before any solve: `w_out = 1.0`, `w_in = 1.0`.** That is not a
symmetric objective, and the asymmetry sits in three places:

1. **The hinge.** Density at or above the floor costs exactly zero, so the
   objective can never buy in-bounds density it does not need by paying
   out-of-bounds growth for it. PROVEN by unit test
   (`test_density_above_the_floor_buys_nothing`).
2. **The amplitude scale.** An out-of-bounds defect is charged at full amplitude
   (a melted bed cell has phi = 1) while an in-bounds defect at the measured
   operating point has h of roughly 0.2 to 0.3, so the per-cell cost ratio at
   the operating point is already about 10 to 1 in the hard direction even at
   equal weights.
3. **The read state.** The stop itself is set by the balance of the two terms.

**How w_out = 1.0 was chosen, disclosed rather than hidden.** A reality probe on
the square under a uniform map, run BEFORE any solve
(`logs_asym/` is not where it lives; the probe was a scratch script and its
numbers are reproduced here), measured the argmin stop and the density there for
w_out in {1, 3, 10, 30} at each of the three floors. w_out = 1 was the largest
tested price whose uniform-arm stop kept mean relative density inside the stated
0.80 to 0.90 band on the square (0.807 at floor 0.85, against 0.734 at w_out = 3
and 0.610 at w_out = 10). That is a calibration to the stated acceptance band on
one control arm, made before any optimization, and it is the one hyperparameter
of this pass that was set from data. **Section 6 measures that w_out = 2 to 3
is the better production value, which supersedes it.**

### 2.4 The saturation guard, carried over from the density pass

The pathology the guard exists for is still possible here: if every in-bounds
cell clears the floor before any bed melt appears, the soft term is exactly zero,
its gradient is exactly zero, and the objective degenerates into pure growth
minimization whose argmin is the first step. Reported loudly on every arm:
`asym_stop_at_horizon`, `asym_stop_is_first_step`, `asym_in_term_dead`, and the
**flat-onset index of the combined curve next to the argmin**, so a read state
sitting on a plateau is visible in the data. COMPUTED: zero first-step stops,
zero dead in-bounds terms, three at-horizon arms (all on the cross), and
flat-onset gaps of **13 to 83 stored steps**, that is the argmin is soft to
between 6.5 and 41.5 seconds of bake. Every stop time in this report carries
that softness.

---

## 3. The finite-difference gate, run BEFORE any optimization

### 3.1 What was gated

Central differences, epsilon swept **1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8,
1e-8**, on two shapes (square, a compact convex shape; L_shape, a shape with a
limb that never melts) and four layers, one chain piece at a time so a failure
localizes:

* **A0_out** the out-of-bounds term alone (`w_in = 0`), temperature seed only.
* **A0_in** the in-bounds deficit alone (`w_out = 0`), density seed only,
  carrying the hinge mask.
* **A1** both terms, both seeds in ONE reverse sweep, unfiltered.
* **A2** both terms with the 1.0 mm physical-length design filter in the chain,
  `s = F(v)`. **This is the gradient the solve actually uses.**

**Both sides of the hinge are probed explicitly**, because the hinge is the new
nonsmoothness this objective introduces:

* `hinge_active_cell` an in-part cell strictly below the floor at the read
  state, so its own density seed is nonzero;
* `hinge_inactive_cell` an in-part cell at or above the floor, so its own seed
  is exactly zero and its entire analytic derivative arrives through nonlocal
  coupling;
* `hinge_boundary_cell` the in-part cell whose relative density is CLOSEST to
  the floor, that is the cell sitting on the kink. COMPUTED: that cell sat at
  relative density 0.84999 (square) and 0.85015 (L_shape), within 1.5e-4 of the
  kink.

Also probed: the maximum-sensitivity cell, a fixed pseudo-random in-part cell, a
random unit direction, the filtered smooth random direction, and the
**gradient direction itself**, which is the largest directional derivative
available and therefore the probe that sits furthest above the arithmetic noise
floor. Code: `fgm_solve_campaign/adjoint2d/gate_asym.py`. Raw:
`out_asym/gate_asym_square.json`, `out_asym/gate_asym_L_shape.json`.

### 3.2 Results, 46 probes

| shape | layer | at 1e-6 | at 1e-5 | worst probe | its relative error | its analytic derivative | its absolute error |
|---|---|---|---|---|---|---|---|
| square | A0_out | 3/4 | **4/4** | random_direction | 9.44e-06 | -2.70e-03 | 2.6e-08 |
| square | A0_in | 1/4 | 2/4 | random_cell | 2.18e-05 | +3.37e-06 | 7.4e-11 |
| square | A1 | 1/7 | **7/7** | hinge_active_cell | 8.57e-06 | -3.28e-04 | 2.8e-09 |
| square | **A2 filtered** | 3/8 | 3/8 | random_direction | 5.12e-04 | +1.86e-05 | 9.5e-09 |
| L_shape | A0_out | 1/4 | 3/4 | random_direction | 4.20e-05 | -6.87e-04 | 2.9e-08 |
| L_shape | A0_in | 1/4 | 3/4 | random_direction | 1.93e-05 | +1.12e-03 | 2.2e-08 |
| L_shape | A1 | 2/7 | 5/7 | random_direction | 1.69e-05 | +4.31e-04 | 7.3e-09 |
| L_shape | **A2 filtered** | 5/8 | **8/8** | gradient_direction | 2.21e-06 | +2.15e-02 | 4.7e-08 |

**Totals: 17 of 46 probes at 1e-6, 35 of 46 at the campaign's documented 1e-5
subgradient standard.**

**The hinge probes specifically**, which is what this gate was extended for:

| shape | layer | hinge_active | hinge_inactive | hinge_boundary (the kink) |
|---|---|---|---|---|
| square | A1 | 8.57e-06 | 2.62e-06 | **7.94e-06** |
| square | A2 filtered | 5.18e-05 | 5.80e-07 | **1.68e-05** |
| L_shape | A1 | 4.56e-06 | 4.54e-08 | **1.46e-05** |
| L_shape | A2 filtered | 1.11e-06 | 1.81e-08 | **1.70e-08** |

**The gradient direction, the probe the optimizer actually moves along:**
1.39e-07, 2.00e-08, 5.90e-06, 4.15e-07 (square, four layers) and 4.45e-06,
1.24e-06, 6.18e-06, 2.21e-06 (L_shape). **8 of 8 clear 1e-5 and 5 of 8 clear
1e-6.**

### 3.3 Verdict, stated exactly and not inflated

**PASS at the campaign's 1e-5 subgradient standard with named exceptions, and
the hinge is NOT one of them.** Three pieces of evidence:

1. **The misses are a denominator effect, measured.** COMPUTED across all 46
   probes the ABSOLUTE finite-difference error spans **1.2e-12 to 2.6e-07** with
   a median of **2.0e-09**, and every probe that misses 1e-5 has an analytic
   derivative between 1.9e-05 and 1.1e-03, that is 1 to 3 decades below the
   gradient norm. The largest relative error in the whole table
   (5.12e-04, square A2 random direction) has an absolute error of 9.5e-09,
   which is smaller than the absolute error of six probes that PASS.
2. **The epsilon sweeps are clean V shapes, which is roundoff and curvature, not
   a kink.** For the square's A1 kink-cell probe the absolute error runs
   5.6e-04, 1.96e-09, 1.90e-08, 7.57e-08, 3.53e-06 at epsilon 1e-4, 1e-5, 1e-6,
   3e-7, 1e-8: a bias-dominated branch above 1e-5, a floor-dominated branch
   below 1e-6, and a minimum in between. A dominant kink produces an
   epsilon-independent plateau instead, and none is present. **The usable
   epsilon window is one to two decades wide, from 1e-5 to about 3e-7**, and it
   is swept explicitly.
3. **The floor is measured and matches the independent measurement of the
   density campaign.** From absolute error times 2 epsilon the evaluation floor
   of J_asym is **about 5e-15 to 5e-14 absolute**, that is **2e-14 to 3e-13
   relative** to J_asym of 0.196 to 0.629, against the
   `DENSITY_OBJECTIVE_LIBRARY_REPORT.md` Section 2.3 measurement of 1.4e-13 to
   4.1e-13 relative for the same 1500-step march. **The gate is at the
   information limit of a double-precision full-horizon forward.**

**The honest weak point.** The layer the solve uses, A2 on the square, reaches
only **3 of 8** probes at 1e-5. Its five misses are random_cell 2.60e-05,
hinge_active 5.18e-05, hinge_boundary 1.68e-05, random_direction 5.12e-04 and
smooth_random_direction 1.03e-04, with absolute errors of 5.1e-09, 6.3e-09,
1.5e-09, 9.5e-09 and 2.0e-08 respectively. The same layer on the L_shape reaches
8 of 8. The difference is that the square's filtered derivatives are three times
smaller, not that its gradient is worse.

### 3.4 Read-index stability, checked and not assumed

The argmin read index was recomputed under every probe perturbation at epsilon
1e-3. COMPUTED: **the index does not move on either shape** (square base index
1025, L_shape base index 615, `moved = False` on all probes), so over the tested
perturbation range the neglected dt*/ds term is exactly zero and the fixed-read
gradient IS the rule-pinned gradient.

### 3.5 Flag-off bit identity, by construction

**No engine file was touched.** `git diff` on `forward.py`, `adjoint.py`,
`shape_objective.py`, `density_objective.py`, `design_filter.py`, `mma.py` and
`topopt_stage.py` is empty; the whole pass is six NEW files. There is therefore
no flag to turn off and no path that could have moved: every previously reported
forward and gradient is byte-identically reachable. The new objective reuses
`adjoint.gradient(..., seeds, seeds_rho)`, whose coupled (T, rho) reverse sweep
was gated in the density pass, with both seeds active at the same index.

### 3.6 Unit tests, red first

**429 tests pass** (`.venv312/bin/python -m pytest adjoint2d/tests -q`), of
which **26 are new in this pass and both files were observed failing first** with
`ImportError: cannot import name 'asym_objective' from 'adjoint2d'` and
`ImportError: cannot import name 'asym_solve' from 'adjoint2d'`. The baseline
before this pass was 383; the difference above 383 + 26 is other work in the
shared test directory and is not mine.

* `tests/test_asym_objective.py`, 20 tests. The hinge is exactly zero at and
  above the floor and exactly 1 at the powder floor; a floor at or below the
  powder floor is rejected; the bed contributes nothing to the soft term
  whatever its stored value; melting the part is never charged by the hard term;
  one fully melted bed cell costs exactly w_out times one unsintered part cell;
  density above the floor buys nothing; both seeds match cell-by-cell central
  differences; the density seed is exactly zero where the hinge is inactive; the
  stop is the argmin, flags the horizon, flags a dead soft term, and reports the
  flat-onset guard gap.
* `tests/test_asym_solve_logic.py`, 6 tests. The acceptance rule and its
  tolerances; and the identity that makes the trade curve free, namely that
  sweeping w_out over a stored pair of curves reproduces the argmin of the
  weighted sum exactly, together with the monotonicity that a harder
  out-of-bounds price never stops later.

---

## 4. Conventions, stated once, obeyed by every number below

* **ASYM-STOP** the argmin over that arm's own stored trajectory of J_asym. The
  softness of that argmin is reported per arm as the flat-onset gap.
* **MELT-STOP** the argmin over that arm's own stored trajectory of the
  melt-region objective J_phi. Quoted only where the text says so.
* The shape-fidelity early stop is **DISABLED on every arm**; every march runs
  the full 1500-step horizon at dt = 0.5 s, so the horizon is 750 s.
* **Grid 120 x 120 on every number in this report.** No claim here transfers to
  grid 160; `SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute fidelity
  at this grid does not transfer.
* The melted region is phi >= 0.5. **Growth** is melted bed cells as a percentage
  of the part cell count; **under** is unmelted part cells on the same
  normalization. **IoU** is against the binary part mask at grid 120.
* The deliverable arm is always the continuous map quantized to **4 bits per
  pixel inside the part** and re-run through the real forward.
* Actuator: **conductivity only** (`eps_covary = False`) on every solved arm, the
  deployable channel. **Arms are not dose matched**: absorbed power spans
  **261.1 to 500.0 W/m** over the 42 scored arms.
* Design filter: part-masked normalized-convolution Gaussian at sigma = 1.5
  cells, which is 0.75 mm at grid 120, applied to the DESIGN VARIABLE inside the
  solve on both optimizer arms.
* **sigma_T is a diagnostic only** and is tabulated in the result files, never
  used as an objective or a selection criterion.

---

## 5. The five-shape census

Every number at that arm's own ASYM-STOP, floor 0.85 relative density,
w_out = 1.0, grid 120 x 120.

| shape | arm | J_asym | out | in | stop s | IoU | growth % | under % | mean rho_rel | min rho_rel | above floor | sigma_T C | P_abs W/m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | uniform | 0.22568 | 0.15520 | 0.07048 | 496 | 0.8288 | 18.25 | 2.00 | 0.8072 | 0.554 | 0.160 | 6.08 | 500.0 |
| square | melt-solved 4 bpp | 0.10015 | 0.04827 | 0.05188 | 592 | 0.9238 | 8.25 | 0.00 | 0.8012 | 0.661 | 0.196 | 3.74 | 412.0 |
| square | **asym L-BFGS-B 4 bpp** | **0.09369** | 0.04309 | 0.05060 | 626 | 0.9366 | 6.50 | 0.25 | 0.8405 | 0.565 | 0.557 | 6.72 | 402.8 |
| square | asym MMA 4 bpp | 0.09562 | 0.04400 | 0.05162 | 575 | 0.9390 | 6.50 | 0.00 | 0.8068 | 0.632 | 0.207 | 4.17 | 425.0 |
| hexagon | uniform | 0.34471 | 0.24454 | 0.10018 | 368 | 0.7750 | 25.98 | 2.36 | 0.8641 | 0.550 | 0.669 | 19.34 | 500.0 |
| hexagon | **melt-solved 4 bpp** | **0.11270** | 0.03360 | 0.07910 | 470 | 0.9476 | 5.12 | 0.39 | 0.8656 | 0.556 | 0.650 | 14.25 | 373.6 |
| hexagon | asym L-BFGS-B 4 bpp | 0.13304 | 0.05368 | 0.07936 | 458 | 0.9158 | 7.48 | 1.57 | 0.8656 | 0.550 | 0.648 | 14.87 | 383.3 |
| hexagon | asym MMA 4 bpp | 0.13027 | 0.05194 | 0.07833 | 435 | 0.9301 | 7.09 | 0.39 | 0.8384 | 0.555 | 0.585 | 10.72 | 391.5 |
| cross | uniform | 0.59890 | 0.13283 | 0.46608 | 262 | 0.5347 | 16.99 | 37.45 | 0.6883 | 0.550 | 0.147 | 35.35 | 500.0 |
| cross | melt-solved 4 bpp | 0.41149 | 0.15515 | 0.25634 | **750 H** | 0.6755 | 16.60 | 21.24 | 0.8318 | 0.550 | 0.610 | 21.97 | 262.6 |
| cross | asym L-BFGS-B 4 bpp | 0.43496 | 0.19975 | 0.23521 | 632 | 0.6720 | 21.24 | 18.53 | 0.8488 | 0.550 | 0.631 | 30.14 | 311.2 |
| cross | **asym MMA 4 bpp** | **0.38969 H** | 0.13022 | 0.25946 | **750 H** | 0.6920 | 14.09 | 21.04 | 0.8229 | 0.550 | 0.569 | 20.82 | 261.1 |
| triangle | uniform | 0.47648 | 0.33536 | 0.14112 | 311 | 0.7153 | 37.00 | 2.00 | 0.8030 | 0.550 | 0.343 | 25.37 | 500.0 |
| triangle | melt-solved 4 bpp | 0.29321 | 0.18215 | 0.11106 | 327 | 0.8182 | 21.00 | 1.00 | 0.7945 | 0.550 | 0.278 | 13.51 | 442.1 |
| triangle | asym L-BFGS-B 4 bpp | 0.30779 | 0.19082 | 0.11697 | 332 | 0.8066 | 21.50 | 2.00 | 0.8033 | 0.550 | 0.333 | 15.40 | 440.9 |
| triangle | **asym MMA 4 bpp** | **0.24194** | 0.13521 | 0.10672 | 359 | 0.8491 | 16.00 | 1.50 | 0.7996 | 0.550 | 0.280 | 11.87 | 405.7 |
| L_shape | uniform | 0.62222 | 0.13597 | 0.48625 | 296 | 0.5253 | 15.38 | 39.39 | 0.6980 | 0.550 | 0.233 | 39.89 | 500.0 |
| L_shape | melt-solved 4 bpp | 0.60853 | 0.12234 | 0.48619 | 298 | 0.5334 | 13.81 | 39.30 | 0.6952 | 0.550 | 0.216 | 39.84 | 493.7 |
| L_shape | asym L-BFGS-B 4 bpp | 0.61152 | 0.13266 | 0.47886 | 305 | 0.5296 | 14.46 | 39.39 | 0.7010 | 0.550 | 0.247 | 40.64 | 488.3 |
| L_shape | **asym MMA 4 bpp** | **0.60823** | 0.12130 | 0.48693 | 304 | 0.5281 | 13.72 | 39.94 | 0.6964 | 0.550 | 0.226 | 42.90 | 480.4 |

**H = the stop is the last stored step**, so that arm's J_asym is an upper bound.
Three arms on the cross carry it and no arm on any other shape does.

**The continuous arms are also stored and scored** and are within 0.4 percent of
their quantized counterparts on every shape except the square MMA arm
(0.09837 continuous against 0.09562 at 4 bits per pixel, where quantization
happens to help), so quantization to the printer's 16 levels is not a
meaningful loss here.

### 5.1 The three questions, answered per shape

**(a) Does the old melt objective already satisfy the new specification?** No,
but it is a strong baseline. COMPUTED, the melt-solved 4-bits-per-pixel map
re-scored under J_asym at its own J_asym stop beats uniform on 5 of 5 shapes and
beats BOTH asymmetric solves on the hexagon. It fails the acceptance rule on
5 of 5 (growth 5.12 to 21.00 percent).

**(b) Does re-solving under J_asym help?** On 3 of 5 shapes, yes; the gains are
+17.5 percent (triangle), +6.5 (square) and +5.3 (cross), the L_shape is a tie
at +0.1 and the hexagon is a 15.6 percent loss.

**(c) Stop or map?** Mostly the stop.

| shape | J_asym of the melt-solved map at the MELT stop | at its own ASYM stop | gain from the stop alone | further gain from re-solving |
|---|---|---|---|---|
| square | 0.15665 | 0.10015 | **36.1 %** | 6.5 % |
| hexagon | 0.12867 | 0.11270 | **12.4 %** | -15.6 % |
| cross | 0.41149 | 0.41149 | 0.0 % | 5.3 % |
| triangle | 0.43112 | 0.29321 | **32.0 %** | 17.5 % |
| L_shape | 0.63597 | 0.60853 | **4.3 %** | 0.1 % |

### 5.2 The trade between the melt-solved map and the asymmetric deliverable

Shape gained against in-bounds density given up, per shape, best asymmetric arm
against the melt-solved map:

| shape | change in growth, points | change in mean in-bounds relative density | change in percent of the part above the floor | reading |
|---|---|---|---|---|
| square | **-1.75** | **+0.0393** | **+36.1** | both sides gained, no trade was needed |
| hexagon | +1.97 | -0.0272 | -6.5 | both sides lost, the melt map is better |
| cross | **-2.51** | -0.0089 | -4.1 | shape bought with a small density payment |
| triangle | **-5.00** | +0.0052 | +0.3 | both sides gained |
| L_shape | -0.09 | +0.0012 | +1.0 | nothing moved |

**COMPUTED: the map-level trade is NOT where the asymmetry expresses itself.**
On 3 of 5 shapes the asymmetric map improves both sides at once, and on the one
shape where it pays density (the cross) it pays 0.009. The exchange the
specification describes is real, but it is bought and sold in the READ STATE,
which is Section 6.

### 5.3 Map structure

| shape | root-mean-square distance from uniform | from the melt-solved map | mean in-part saturation | in-part roughness |
|---|---|---|---|---|
| square L-BFGS-B | 0.2104 | 0.2609 | 0.837 | 0.0179 |
| square MMA | 0.3092 | 0.1658 | 0.731 | 0.0168 |
| hexagon MMA | 0.4058 | 0.2672 | 0.671 | 0.0248 |
| cross MMA | 0.5328 | 0.3351 | 0.570 | 0.0450 |
| triangle MMA | 0.3179 | 0.2909 | 0.790 | 0.0373 |
| L_shape MMA | 0.3791 | 0.3257 | 0.822 | 0.0240 |

For comparison the melt-solved maps sit at 0.2198 to 0.5524 from uniform with
roughness 0.0347 to 0.1019. **The asymmetric maps are a distinct family, and they
are two to five times smoother in part**, which is the design filter doing its
job: the melt-solved library maps were solved WITHOUT the filter, so the
roughness comparison is a comparison of recipes and not of objectives.

---

## 6. The trade curve, measured

Because both terms are read at the same index, `J_asym = w_out * J_out + J_in`
means one forward run per map recovers the exact stop and the exact outcome at
every out-of-bounds price. `figs_asym/fig_asym_trade.png`, raw in
`figs_asym/trade_rows.json`. Run on each shape's best deliverable map.

**Square** (map solved at w_out = 1, L-BFGS-B):

| w_out | stop s | growth % | under % | mean rho_rel | above floor | IoU |
|---|---|---|---|---|---|---|
| 0.25 | 678 | 16.62 | 0.00 | 0.8962 | 0.810 | 0.8574 |
| 0.50 | 644 | 9.75 | 0.12 | 0.8612 | 0.644 | 0.9100 |
| **1.00** | 626 | 6.50 | 0.25 | 0.8405 | 0.557 | 0.9366 |
| **2.00** | 604 | **2.12** | 0.62 | **0.8135** | 0.441 | **0.9731** |
| 3.00 | 596 | 2.00 | 1.12 | 0.8033 | 0.388 | 0.9694 |
| 5.00 | 587 | 2.00 | 1.75 | 0.7910 | 0.326 | 0.9632 |
| 10.00 | 575 | 0.25 | 2.75 | 0.7752 | 0.224 | 0.9701 |
| 20.00 | 563 | **0.00** | 5.25 | 0.7593 | 0.071 | 0.9475 |

**Hexagon** (MMA map): w_out 0.25 to 20 moves growth 18.50 to 0.00 percent,
mean relative density 0.9067 to 0.7565, IoU 0.8439 to 0.9429, stop 488 to 384 s.
**Triangle** (MMA map): growth 30.25 to 0.00 percent, density 0.8676 to 0.6562,
IoU 0.7639 to 0.8475, stop 400 to 279 s.
**Cross** (MMA map): growth 14.09 to 0.39 percent, density 0.8229 to 0.6394,
under-melt 21.04 to 54.44 percent, stop 750 to 492 s. **At w_out at or below 1
the cross stop is the horizon, so those three rows are bounds.**
**L_shape** (MMA map): growth 55.42 to 0.00 percent, density 0.8325 to 0.6049,
under-melt 24.93 to 55.51 percent, stop 454 to 226 s.

**Three readings, all COMPUTED.**

1. **The exchange rate is per shape and it is steep.** Buying the last 6.5 points
   of growth on the square (w_out 1 to 20) costs 0.081 of mean relative density.
   On the L_shape buying 55 points of growth costs 0.228 of density and 31 points
   of under-melt, because on that shape the growth is the only way the horizontal
   limb ever gets warm.
2. **Melt IoU is NOT monotone in the price and peaks near w_out = 2 to 3** on the
   square (0.9731 at w_out = 2) and near w_out = 3 on the hexagon (0.9632).
   Growth and under-melt trade against each other and the shape metric has an
   interior optimum in the price, which the objective's own value does not see.
3. **Where the floor is reachable.** Mean in-bounds relative density of 0.85 is
   reached at growth of about 8 percent on the square, about 10 percent on the
   hexagon and about 24 percent on the triangle, and is **not reachable at any
   price on the cross or the L_shape**, whose curves top out at 0.823 and 0.833
   respectively.

**This supersedes the frozen default.** The evidence says the production
out-of-bounds price is **w_out = 2 to 3**, not 1: at w_out = 2 the square is at
growth 2.12 percent with IoU 0.9731 and relative density 0.8135, and the hexagon
at 2.76 percent, 0.9579 and 0.8051, both inside the stated 0.80 to 0.90 band.
**The caveat, stated: these rows re-read a map that was SOLVED at w_out = 1, so
they isolate the read-state trade. A map solved at w_out = 2 was not run and is
the first item of Section 11.**

---

## 7. The floor sweep, hexagon

Three independent solves, both optimizers each, 40 forward-equivalents each,
floor 0.80, 0.85 and 0.90 relative density.
`figs_asym/fig_asym_floor_hexagon.png`.

| floor | best arm | J_asym | stop s | growth % | under % | IoU | mean rho_rel | above ITS floor | mean s | P_abs W/m |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | MMA | 0.10928 | 450 | 6.69 | 0.20 | 0.9354 | 0.8291 | 0.699 | 0.609 | 378.6 |
| **0.85** | MMA | 0.13027 | 435 | 7.09 | 0.39 | 0.9301 | 0.8384 | 0.585 | 0.671 | 391.5 |
| 0.90 | L-BFGS-B | 0.14851 | 450 | 7.48 | 0.39 | 0.9267 | 0.8598 | 0.510 | 0.772 | 388.1 |

J_asym is not comparable across floors, because the floor changes the objective's
own normalization; the outcome columns are.

**COMPUTED. The floor moves the MAP more than it moves the OUTCOME.** Mean
in-part saturation rises monotonically 0.609, 0.671, 0.772 and the
root-mean-square distance from the 0.85 map is 0.109 (at 0.80) and 0.163 (at
0.90), while the outcome spread across the whole band is 0.79 growth points,
0.009 IoU points and 0.031 of mean relative density. **The floor choice is
therefore low risk anywhere in 0.80 to 0.90, and it is a real knob on the map
rather than a cosmetic one.**

**Recommendation: 0.85.** It is the midpoint of the stated band; it is the only
one of the three whose achieved mean in-bounds density (0.8384) lands inside the
band rather than on its edge (0.8291 at floor 0.80 is essentially at the floor,
0.8598 at floor 0.90 is below its own floor by 0.040); and at floor 0.90 only
51.0 percent of the part clears its own floor, so the hinge is switched on over
half the part and the soft side starts behaving like a plain quadratic density
penalty, which is exactly the asymmetry the floor exists to preserve.

---

## 8. The two optimizers at a matched 40 forward-equivalents

Single cold start from uniform on both, the same filtered design variable, the
same box [0, 1], the same objective closure through `topopt_stage.StageRunner`,
no continuation. MMA hyperparameters are Svanberg's published defaults with the
topology-optimization standard move limit 0.2, unchanged and not swept, recorded
in every result file.

| shape | gradient evaluations each | L-BFGS-B J_asym | MMA J_asym | MMA against L-BFGS-B | L-BFGS-B improving follow-ups | MMA improving follow-ups |
|---|---|---|---|---|---|---|
| square | 16 | **0.09369** | 0.09562 | +2.1 % | 0.200 | 0.200 |
| hexagon | 19 | 0.13304 | **0.13027** | -2.1 % | 1.000 | 0.278 |
| cross | 22 | 0.43496 | **0.38969** | **-10.4 %** | 1.000 | 0.619 |
| triangle | 20 | 0.30779 | **0.24194** | **-21.4 %** | 1.000 | 0.526 |
| L_shape | 21 | 0.61152 | **0.60823** | -0.5 % | 1.000 | 0.750 |

**COMPUTED: MMA wins 4 of 5 at a matched budget, and the two big wins are the
two shapes whose stop moves the furthest during the solve** (the cross stop
moves 262 to 750 s, the triangle 311 to 359 s). The mechanism is the one
`mma.py` documents: MMA takes one design update per evaluation with no line
search, so a 20-evaluation pool buys 20 design updates, while L-BFGS-B's pool is
shared with its line search. The "improving follow-ups" column shows the other
side of that: L-BFGS-B improves its incumbent on essentially every evaluation
(it is a descent method) while MMA improves on 20 to 75 percent of them (the
1987 method without the globally convergent inner loop is not a descent method),
and MMA still ends lower. **This is a cleaner MMA win than
`MMA_RETEST_REPORT.md` measured, and the difference is that there is no
projection continuation here to throw away the asymptote state.**

**Budget accounting, and a caveat that must travel with it.** The measured
adjoint-to-forward ratio here is **0.80 to 1.38**, against 1.23 to 2.01 in the
density campaign, because the forward runs the full 1500-step horizon while the
reverse sweep only runs back from the stop index (615 to 1025). So 40
forward-equivalents buys **16 to 22 gradient evaluations per optimizer** here
against 6 to 9 per start in the density campaign. **The budgets are matched
within this pass and are NOT comparable across passes.**

---

## 9. Gates and violations

* **Energy-residual gate: zero violations.** COMPUTED, the standing 5 percent
  gate evaluated at each arm's own ASYM stop passes on **all 42 scored arms of
  all 7 solve jobs**, and also at each arm's own melt stop. The four shapes that
  failed the gate at the density stop in the previous pass are not reproduced
  here, because the J_asym stop is 1.0 to 1.6 times the melt stop rather than
  2.7 to 3.3 times.
* **Horizon flags: three arms**, all on the cross (melt-solved, asym MMA
  continuous, asym MMA 4 bits per pixel), all marked H in Section 5.
* **Saturation guards: zero firings.** No arm stopped at the first step, no arm
  had a dead in-bounds term at its stop.
* **Read-state softness: flat-onset gaps of 13 to 83 stored steps**, that is 6.5
  to 41.5 s. Every stop time carries that.
* **Clip fractions** (temperature step cap, temperature range cap, RF power cap)
  are stored per arm in the result files.

---

## 10. Cost and wall time

**Logged after two shapes and projected, as required.** The first two shape
solves to complete were L_shape at 811 s and triangle at 814 s, mean 812 s,
which projected to five shapes in five parallel single-threaded streams as about
14 minutes of wall clock. **Actual: 17:24:38 to 17:41:26, 16 min 48 s**, inside
the projection plus the slowest job (the cross at 1026 s). **Nothing was cut.**

| stage | runs | process time |
|---|---|---|
| finite-difference gate, square and L_shape, final pass with the gradient-direction probe | 2 | 2686 s and 2661 s |
| the same gate, first pass, SUPERSEDED and not quoted anywhere above | 2 | 1759 s and 1737 s |
| five shape solves at 40 forward-equivalents per optimizer | 5 | 4336 s |
| hexagon floor sweep at 0.80 and 0.90 | 2 | 1319 s |
| driver smoke test on the star at budget 6, discarded | 1 | 108 s |
| figures, all eight regenerated after two layout fixes | 3 rounds | about 1100 s |
| full test suite | 2 | 112 s and 117 s |

Total about **4.4 hours of process time**, about **1 hour 25 minutes of wall
clock** on 12 cores with every numerical library pinned to one thread
(`fgm_solve_campaign/env1.sh`).

---

## 11. Proven, computed, assumed

**PROVEN**
* The hinge is exactly zero at and above the floor and exactly 1 at the powder
  floor, and it is linear between; a floor at or below the powder floor is
  rejected rather than silently deleting the soft side.
* One fully melted bed cell costs exactly `w_out` times one completely
  unsintered part cell, at matched normalized amplitude.
* Pushing in-bounds density above the floor changes the objective by exactly
  zero, so it cannot pay for out-of-bounds growth.
* Both seeds match cell-by-cell central differences of the objective; the
  density seed is exactly zero where the hinge is inactive; the temperature seed
  is exactly zero inside the part; each seed scales linearly with its own weight.
* The stop is the argmin, flags the horizon, flags a dead soft term, and reports
  the flat-onset guard gap.
* Sweeping w_out over a stored pair of curves reproduces the argmin of the
  weighted sum exactly, and a harder price never stops later.
* 429 tests pass; the 26 new ones were red before they were green.
* No engine file was modified, so every previously reported forward and gradient
  is byte-identically reachable.

**COMPUTED**
* Every number in Sections 1 and 3 through 10.
* The gradient is verified against the real forward at absolute errors of
  1.2e-12 to 2.6e-07 across 46 probes on 2 shapes and 4 layers, with the
  evaluation floor of J_asym measured at 2e-14 to 3e-13 relative.
* The argmin read index does not move under any probe perturbation.
* The hinge does not degrade the gate: the kink cell reaches 1.7e-08 to 1.7e-05
  and the epsilon sweeps are clean V shapes with no epsilon-independent plateau.
* Zero energy-residual gate violations on 42 scored arms.
* MMA beats L-BFGS-B on 4 of 5 shapes at a matched 40 forward-equivalents.

**ASSUMED, and how each one bites**
1. **That `w_out = 1` and `w_in = 1` are the right prices.** Frozen before any
   solve from a uniform-arm probe on ONE shape. Section 6 measures that 2 to 3
   is better and the solved maps were not re-run at that price. This is the
   largest open knob of this pass.
2. **That the acceptance rule (growth at or below 1.0 percent, at least
   95 percent of the part above the floor) is the right pass or fail line.** It
   is a convention invented in this pass so that no table could quote a PASS
   without stating its tolerances. Nothing passes it, so the rule has not yet
   been tested against a case that should pass.
3. **That the floor should be a hard threshold rather than a soft band.** A
   smoothed hinge with a tested width was not tried, because `max(0, .)^2` is
   already continuously differentiable and the gate shows the kink is not the
   binding limitation.
4. **That out-of-bounds melt should be charged at the READ STATE rather than as
   a maximum over time.** Bed fusing is irreversible, so a running maximum is
   arguably the physical quantity. The single-index read is the campaign
   convention and it keeps the envelope argument; a running maximum would break
   it and is a different experiment.
5. **That the rasterized binary part mask is the right nominal target.** Carried
   over, still untested. `MMA_RETEST_REPORT.md` used a grid-independent
   area-fill target instead, so J_asym here is not comparable to any J in
   `out_mma`.
6. **That an arbitrary stop time is realizable as a process control.** The
   delivered stops are 304 to 750 s of continuous drive and they differ between
   arms by up to 130 s.
7. **The arms are conductivity only; the historical dopant maps co-vary
   permittivity.** The standing actuator gap.
8. **The forward is the two-dimensional `adjoint2d` engine, not `heatr3d`.**
9. **No experimental validation.** These are two-dimensional model results at one
   grid.

---

## 12. Honest limits

1. **Nothing reaches the specification.** 0 of 42 arms pass the acceptance rule,
   and the trade curve says the cross and the L_shape cannot reach the 0.85 mean
   density at ANY out-of-bounds price with this actuator.
2. **Two shapes were gated, five were solved.** The other three inherit the gate.
3. **The gate does not reach 1e-6 on most probes** (17 of 46) and the layer the
   solve uses reaches only 3 of 8 at 1e-5 ON THE SQUARE (8 of 8 on the L_shape).
   The limit is arithmetic and measured, but it is a weaker gate than a
   short-read-state single-cell gate.
4. **Three cross arms stop at the horizon**, so those J_asym values are upper
   bounds and the cross column of every table should be read as such.
5. **The trade curve re-reads maps solved at w_out = 1.** It isolates the
   read-state trade and does not say what a map solved at another price would do.
6. **The floor sweep is one shape.** Nothing here says the hexagon's mild floor
   sensitivity transfers.
7. **No arm is dose matched** (261.1 to 500.0 W/m) and no arm is a second
   printing pass (box [0, 1], single pass).
8. **Everything is grid 120**, and the forward is not IoU converged between 120
   and 160.
9. **Single cold start on every solved arm**, no multi-start.
10. **The melt-solved comparison arm was solved without the design filter**, so
    the roughness comparison of Section 5.3 compares recipes, not objectives.

---

## 13. The single most valuable next layer

**Re-solve at w_out = 2 and w_out = 3 on the square, hexagon and triangle, and
compare against the trade curve of the map solved at w_out = 1.** Section 6
measures that at w_out = 2 the read state alone already gives growth of about
2 percent at melt IoU 0.973 and relative density 0.814 on the square, which is
the closest anything in this pass comes to the specification. If a map solved at
that price beats the re-read map, the production recipe is settled at one
number; if it does not, the whole trade is a read-state effect and the map is
irrelevant to it, which is an equally decisive result. Six solves, about
25 minutes in six streams, and it is the cheapest remaining experiment that
could change Section 1.

**Second: give the out-of-bounds term a running maximum over time.** Bed fusing
is irreversible and the current single-index read lets an arm "un-grow" by being
read later, which is not physical. It breaks the envelope argument, so it needs
its own gate, and it is the one modelling error in this objective that is known
rather than suspected.

**Third: the cross and the L_shape need a different actuator, not a different
objective.** Both have limbs that never melt at any price (under-melt 21 to
55 percent across the whole trade curve), and no dopant map at box [0, 1] with
conductivity-only actuation changed that by more than 1.6 points.

---

## 14. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code, all under `fgm_solve_campaign/adjoint2d/`, nothing else modified:
* `asym_objective.py` the objective, the hinge, both seeds, the stop rule with
  its guard flags, and the region metrics
* `asym_solve.py` the driver, the acceptance rule, the trade-curve identity, and
  the two optimizer arms through the shared stage seam
* `gate_asym.py` the four-layer finite-difference gate, the hinge probes, the
  gradient-direction probe, and the read-index stability check
* `make_asym_figures.py` the four figure modes
* `tests/test_asym_objective.py` 20 tests, red first
* `tests/test_asym_solve_logic.py` 6 tests, red first

Results, under `fgm_solve_campaign/out_asym/`:
* `gate_asym_square.json`, `gate_asym_L_shape.json` the gate
* `<shape>.json` and `<shape>_maps.npz` for square, hexagon, cross, triangle,
  L_shape
* `hexagon_f080.json`, `hexagon_f090.json` and their maps, the floor sweep

Logs: `fgm_solve_campaign/logs_asym/*.log`.

Figures, **all eight viewed before delivery**, under
`fgm_solve_campaign/figs_asym/`:
* `fig_asym_square.png`, `fig_asym_hexagon.png`, `fig_asym_cross.png`,
  `fig_asym_triangle.png`, `fig_asym_L_shape.png`
* `fig_asym_census.png` the five-shape verdict
* `fig_asym_trade.png` THE trade curve
* `fig_asym_floor_hexagon.png` the floor sweep
* `trade_rows.json` the raw trade table

Read, not modified: `DENSITY_OBJECTIVE_LIBRARY_REPORT.md`,
`MMA_RETEST_REPORT.md`, `FROZEN_CONVENTIONS_2D.md`,
`fgm_solve_campaign/out_lib/*_maps.npz`.
