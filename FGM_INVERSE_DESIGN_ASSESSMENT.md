# FGM Inverse-Design Assessment

**Question (Matt, 2026-07-30):** "How do we create our FGMs? Right now we invert and then we
optimize from there. But are we doing any actual solving? I feel like there needs to be some
physics solving that can figure out the ideal dopant mask based on how the simulation densified.
How confident are we in our process, results, and compensation methods? They seem fine for now,
but can they get better?"

**Scope.** Analysis and discussion only. Nothing was built, no optimizer was run, no dissertation
file was touched, nothing was pushed. One small numerical probe was run on the trusted 2-D engine
and is reported in Section 9 with exactly what was executed.

**Acronyms, expanded on first use.** FGM = functionally graded material (a spatially varying
dopant/saturation map). EQS = electro-quasi-static (the low-frequency Maxwell approximation the
solver uses). FD = finite difference. RF = radio frequency. PDE = partial differential equation.
L-BFGS-B = limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints (the optimizer used
in the A-series). CFL = Courant-Friedrichs-Lewy (the explicit time-step stability limit). bpp =
bits per pixel (the printable quantization of the saturation map). IoU = intersection over union.
CAD = computer-aided design. sigma_T = std(T_part) in deg C, always quoted with its read state
(heating-peak, or melt-onset at phi_bar = 0.90).

**Convention used throughout.** Every claim is tagged PROVEN (unit-tested or FD-gated),
COMPUTED (measured from a real solve or from a stored artifact), or ASSUMED (a modelling choice or
an inference not yet measured).

---

## 0. The verdict up front

**1. No, there is no solve anywhere in FGM creation today.** Both production paths are
proportional feedback controllers with a hand-set gain. Neither forms, factors, or transposes
anything. Neither uses a derivative of the objective with respect to the dopant. Both silently
assume the sensitivity matrix dT/ds is diagonal and positive. It is neither. (Section 1, with
file:line.)

**2. That single assumption is the measured cause of every "compensation made it worse" result in
the project, in both engines.** The proportional-inverse rule renormalizes the proxy field to its
own 2nd/98th percentiles (`fgm_generator.py:286`), so the size of the correction it applies is
independent of how non-uniform the field actually is. Given a nearly-flat baseline it manufactures
a large correction anyway. Three independent measurements of this one mechanism:
2-D square at the INTEGRAL magnitude, sigma_T 4.92 -> 9.01 C at melt-onset, **+83 % worse**;
3-D at n = 64 with the shipped binary boundary, **+48.1 % worse than doing nothing**;
3-D at the only mesh-converged boundary width ever measured, **+92 % (n = 48) to +106 % (n = 64)
worse than doing nothing.** (Section 2.)

**3. The brief's premise about the prior adjoint is out of date and should be corrected.** The
"+15.9 % worse on the n = 64 hold-out" number is real but was measured under the non-converged
binary material boundary, and it was **explicitly reversed** by the later edge-width work: at a
resolved boundary width the same map (`n32c0`) is the **best arm at -22.1 % / -28.0 % versus
uniform**, at both grids, entirely out of model. At that same converged configuration the one-shot
heuristic is the arm that loses to doing nothing. Do not carry "the adjoint lost" forward.
(Section 4.)

**3b. I measured the diagonal assumption directly and it is half wrong, in a quantified way.**
Five solves on the trusted 2-D engine, square at the canonical operating point (Section 9.1):
`d(sigma_T)/d(sat_corner) = +6.2103` and `d(sigma_T)/d(sat_interior) = +0.6646` C per unit
saturation, both at melt-onset. The interior derivative is **positive**, so the heuristic's
"cold cell, add dopant" move is an **ascent** component there; the true sensitivity ratio between
the two zones is **9.34 to 1** where the heuristic weights them about 1 to 1. Net, the heuristic's
direction is still descent but only **63 % aligned** with steepest descent, against **99 %** for a
move that simply lowers the corners. **What a solve buys is not a different direction; it is a
correctly signed and correctly weighted one.**

**4. The head-to-head that would settle Matt's question has never been run.** Every adjoint-versus-
heuristic comparison in the record pits a true gradient against an **uncalibrated** heuristic whose
gain was either fixed by hand or picked best-of-four on the same metric it is then reported on.
The correct control, a **gain-calibrated heuristic selected on a hold-out**, does not exist in any
artifact I could find. My reasoned expectation is that this control recovers most of the adjoint's
margin at roughly one tenth the cost. (Section 6.)

**5. Can compensation get meaningfully better? Yes, and the cheapest large gain is not the
adjoint.** Ranked by expected benefit per unit of compute:
   - **(P0, ~5-10 solves per shape)** Replace the fixed `magnitude` knob with a gain calibrated
     against the measured non-uniformity, and select it on a hold-out read state rather than on the
     reported one. This alone should remove the harmful cases, which are essentially all of the
     current negative results.
   - **(P0, 8 solves, ~1 h)** Run the already-designed edge-width probe on the 3-D FGM numbers.
     Until it runs, every 3-D FGM percentage in the dissertation needs the qualifier
     "at n = 48 with a one-cell material boundary."
   - **(P1, ~40-100 solves per shape)** A 2-D adjoint. It is well posed, the machinery is proven,
     and it addresses exactly the failure mode that is measured. But its incremental value over a
     properly calibrated heuristic is **unknown**, and at a truly matched budget of five solves it
     cannot win.

**6. Is invert-plus-local-search near the achievable frontier? No, but it is much closer than the
current negative results suggest, because those negatives are gain-calibration failures rather than
structural ones.** The structural ceiling is set by something else entirely: the model over-predicts
what tuning can deliver in hardware by roughly a factor of 8 (Section 3.3). Improving the map from
"good" to "optimal" moves a number the hardware cannot yet resolve.

---

## 1. What FGM creation actually is, verified against the code

I read the code rather than the descriptions. Three distinct paths exist. Only one of them
computes a derivative of the objective, and it is not the one in production.

### 1.1 Path A, the one-shot proportional inverse (the production 2-D and 3-D map)

`fgm_generator.generate_fgm` (2-D and printable path) and `heatr3d.make_fgm` (3-D path) implement
the same rule.

- `fgm_generator.py:286`: `norm = np.clip((raw_smooth - lo) / span, 0.0, 1.0)`, where `lo, hi` are
  the 2nd/98th percentiles of the proxy field over the part (default `clip_percentile=(2.0, 98.0)`,
  `fgm_generator.py:72`).
- `fgm_generator.py:296`: `sat_raw = (1.0 - norm) if invert else norm`.
- `fgm_generator.py:305`: `sat_scaled = baseline_saturation + magnitude * (sat_raw - baseline_saturation)`.
- `heatr3d.py:705-710`: the identical construction: percentile normalize, invert, scale by
  `magnitude` about `baseline`, clip to [0,1], quantize to `bpp`.

PROVEN by inspection: this is an algebraic transform of one stored field. There is no objective, no
derivative, no linear solve, no iteration. `magnitude` is a free scalar; `baseline_saturation` is a
free scalar; `dead_band` is a free scalar. Nothing in the code selects them from the physics.

**The percentile normalization is the load-bearing defect.** Because `lo` and `hi` are recomputed
from the field being corrected, `norm` always spans [0,1] no matter whether the field varies by
100 C or by 1 C. The applied correction therefore has an amplitude set entirely by `magnitude` and
`baseline_saturation`, and no dependence on the actual non-uniformity. A rule with a fixed gain
applied to a scale-free error signal overshoots by construction whenever the error is small. This
is not an inference; it is what the three measurements in Section 0 item 2 record, and the
edge-width report states the same mechanism independently
(`notes-ilt-adjoint.md:83-88`: "a proportional inverse-T rule with fixed magnitude/baseline always
over-shoots a nearly-flat field").

**On "figure out the dopant mask based on how the simulation densified": that mode already exists,
and it is the same heuristic.** `heatr3d_job.py:438-440` supports `fgm = "density"`, which runs a
baseline densification solve and then calls
`H.make_fgm(base, magnitude=mag, proxy=base.rho_final)`. The docstring at `heatr3d.py:699-702` is
explicit: "Pass proxy=rho_final for a DENSIFICATION-targeted map (regions that ended denser get less
dopant, under-densified regions get more), which flattens the final density directly." So the
project already grades on the densification field. What that changes is the **proxy**, not the
**method**: it is still the percentile-normalized proportional inverse of a stored field, with the
same fixed `magnitude`, the same diagonal-positive assumption, and the same absence of any
derivative. Changing what you invert does not turn inversion into a solve.

The `use_delta_correction` branch (`fgm_generator.py:351-433`) adds integral accumulation, a move
limit, a sensitivity filter and a volume projection. It is closer to a topology-optimization update
in form, but the quantity it descends on is still not a gradient. `fgm_generator.py:362`:

```python
_g = (norm - 0.5).astype(np.float64)   # range ~ [-0.5, +0.5]
```

and `fgm_generator.py:401`: `_delta = -_mag * _sens`. The comment at `fgm_generator.py:376` calls
this "gradient descent on sigma_T^2", which is not accurate: `norm - 0.5` is a normalized state
variable, not `d(sigma_T^2)/ds`. The true gradient is

```
dJ/ds_k = (2/N) * sum_j (T_j - Tbar) * dT_j/ds_k
```

and this code is exactly the special case obtained by substituting `dT_j/ds_k = c * delta_jk` with
`c > 0` and a fixed `c`. That substitution is the assumption, and it is the whole method.

### 1.2 Path B, the per-node two-sided adaptive-gain law (the current refinement mode)

`pernode_tuning.pernode_sigma_update`, a faithful port of Allison's `Tune_Conductivity.m`. The
update is `pernode_tuning.py:108-118`:

```python
err = float(Tt) - T
max_diff = float(np.max(np.abs(err)))
sign = np.sign(err)
flipped = (sign * sign_prev) < 0.0
K1_new = np.where(flipped, K1 * gain_halving, K1)
sigma_new = np.clip(sigma + K1_new * (err / denom), sigma_min, sigma_max)
```

PROVEN (9 unit tests, `test_pernode_tuning.py`, recorded in `PERNODE_RESULTS.md:52-62`): the law is
implemented correctly. But note what it is: `sigma_i` moves in response to `T_i` and nothing else.
It is a per-node proportional controller with a per-node scalar gain that halves on error sign flip.
The sign-flip halving is a crude one-dimensional secant damping; it is not curvature information and
it carries no cross-cell coupling. Same diagonal-positive assumption as Path A, with an adaptive
step size bolted on.

Two genuine improvements over Path A, both real:
- **Two-sided actuation.** `sat` may exceed 1, so sigma can rise above the uniform baseline
  (`rfam_eqs_coupled.py:339-343`, `sat_max` clamp instead of a [0,1] clamp). Path A can only lower
  sigma.
- **It closes the loop on the real forward.** Each iteration solves the actual physics and reads
  the actual temperature field, so the controller at least measures the plant it is acting on.

That is why it works better than Path A on the square: COMPUTED, `PERNODE_RESULTS.md:94-106`,
sigma_T at melt-onset 4.92 C baseline -> 3.14 C best (Smax = 0.0425) or 2.49 C best (Smax = 0.06),
against Path A's 9.01 C on the same case.

### 1.3 Path C, a real but coarse finite-difference gradient (exists, rarely used)

`rfam_gui_server.py:3934-4013` implements zone-wise finite-difference gradient descent against the
real 2-D engine:

- `rfam_gui_server.py:3947`: perturb `sat` by `perturbation_delta` on zone `k`,
- `rfam_gui_server.py:3991`: `gradients = [(st - sigma_T_base) / perturbation_delta for st in zone_sigma_T]`,
- `fgm_generator.py:1041`: `sat_new = sat_map - step_size * G`, with `G` made zero-mean over the
  part for volume conservation (`fgm_generator.py:1037-1038`).

This IS a derivative of the objective, and it is the only place in the production repository where
one is computed. Two honest limitations. First, cost: `n_zones + 1` full forward solves per gradient
step (`rfam_gui_server.py:3924`, `total_sims = n_gradient_steps * (n_zones_actual + 1)`), so the
resolution of the design is capped at whatever number of zones you can afford. Second, the
perturbation is one-sided and is clipped: `rfam_gui_server.py:3947` applies
`np.clip(sat_pert[zone_mask] + perturbation_delta, 0.0, 1.0)`, so for any cell already at
`sat = 1.0` the perturbation is silently zero and that cell contributes nothing to the measured
zone gradient. That biases the gradient toward zero exactly where the design is on its bound.

### 1.4 Path D, the adjoint (exists, proven, lives in the dissertation-materials workstream)

`dissertation_materials/analysis-3dfgm/ilt_adjoint.py`, `ilt_shape.py`, `ilt_transient.py`,
`ilt_eqs3d.py`, `ilt_eqs3d_thermal.py`, plus the A-series drivers. This computes the exact per-cell
`dJ/ds` at a cost of roughly two forward solves regardless of the number of design variables.
COMPUTED, `out_adjoint_fgm_a6/probe_cost.json`: at n = 48, forward 200.8 s, full gradient 404.1 s
over 34 656 part cells. The equivalent finite-difference gradient would be about 34 656 forwards.

### 1.5 Direct answer to Matt's question

**Production FGM creation is: invert (algebra, no solve), then optionally run a diagonal
proportional controller in the loop (feedback, no solve).** The forward physics is solved every
iteration; the dopant map is never solved for. The one true gradient in the production repository is
Path C's zone-wise finite difference, which is a derivative but a very coarse one. The exact
gradient exists only in Path D, which is not wired into the production 2-D engine.

---

## 2. Where it fails, and the measured mechanism

Three failure modes, each with a code cause and a measurement.

### 2.1 Failure 1: the gain does not scale with the error (the dominant one)

Cause: `fgm_generator.py:286` percentile renormalization plus a fixed `magnitude`
(`fgm_generator.py:305`). Measurements:

| configuration | baseline sigma_T | heuristic sigma_T | result | source |
|---|---|---|---|---|
| 2-D square, melt-onset phi_bar = 0.90, INTEGRAL magnitude | 4.92 C | 9.01 C | **+83 %** | `outputs_eqs/fgm_dosecheck/DOSECHECK.md` results table |
| 3-D n = 64, binary boundary, heuristic built natively | 2.878 C | 4.261 C | **+48.1 %** | `notes-ilt-adjoint.md:178` |
| 3-D n = 48, w = 1e-03 (converged) | 13.896 C | 26.689 C | **+92.1 %** | `notes-ilt-adjoint.md:54` |
| 3-D n = 64, w = 1e-03 (converged) | 13.708 C | 28.206 C | **+105.8 %** | `notes-ilt-adjoint.md:54` |

The common factor in every row is that the baseline field was already comparatively flat. The
edge-width report quantifies it directly: at w = 1e-03 the uniform Q_rf coefficient of variation is
**16.9 %** versus **204.6 %** binary, and the heuristic drives it back up to **72.8 %** with T_max
227 -> 268 C (`notes-ilt-adjoint.md:83-88`). The rule creates non-uniformity where there was little.

The current mitigation is a per-shape magnitude sweep over `{0.3, 0.5, 0.7, 0.85}`
(`outputs_eqs/geometry_dual_readstate/GEOMETRY_DUAL_READSTATE.md:3`), with the winner picked by
sigma_T. That is a four-point line search on one scalar. It works: at m = 0.85 the square is neutral
(+0.6 % melt-onset grading-only benefit, `POWER_MATCHED.md` per-shape table) rather than +83 % worse.
But it is selected in-sample on the same metric that is then reported, with no hold-out. So the
reported per-shape benefits are best-of-four on the reporting metric. That is a real, nameable
confidence caveat, and it is also the clue to the cheap fix (Section 6).

### 2.2 Failure 2: the sign of the diagonal is wrong at the operating point

COMPUTED by me from the stored uniform-sigma sweep
(`outputs_eqs/sigma_sweep/sweep_summary.json`, 17 runs, 2-D square 20 mm, voltage-driven 2428.2 V,
27.12 MHz, eps_r = 20, 60 s, 120x120; the sweep itself is prior work, the aggregation below is mine):

| sigma [S/m] | total absorbed [W/m] | interior (erode 1) [W/m] | interior (erode 4) [W/m] | Q_max/Q_mean |
|---|---|---|---|---|
| 0.020 | 358.63 | 215.11 | 146.73 | 24.83 |
| 0.030 | 447.39 | 247.26 | 168.30 | 30.24 |
| **0.035** | 476.70 | **250.47 (peak)** | **170.32 (peak)** | 33.33 |
| **0.040 = the operating point** | 500.01 | 248.50 | 168.85 | 36.55 |
| 0.050 | 536.81 | 235.98 | 160.14 | 43.02 |
| 0.100 | 703.98 | 157.29 | 106.45 | 67.27 |
| 1.000 | 5221.77 | 17.68 | 11.95 | 91.91 |

Two facts follow, and they matter:

- **Total absorbed power is monotone increasing in sigma across the whole range, while interior
  deposition peaks near sigma = 0.035 and then falls.** The peak location is the same at erosion
  depths 1, 2 and 4 cells in this data. The repository's own analytic criterion puts it at
  `sigma* = omega * eps0 * eps_r = 0.0302 S/m`
  (`outputs_eqs/sigma_sweep/SIGMA_PEAK_RECONCILIATION.md`). The uniform baseline sigma0 = 0.04 sits
  **past** the peak.
- Therefore, at the operating point, lowering a cell's sigma **raises** its coupling. A rule that
  says "this cell is hot, so give it less dopant" is acting on the wrong side of the peak. Measured
  consequence: the FGM maps in DOSECHECK push essentially 100 % of doped cells below sigma*, and the
  part then absorbs **12.5 % to 45.5 % more** power at fixed voltage, not less
  (`DOSECHECK.md` results table, dP column).

The per-node law's two-sided actuator addresses the actuation range but not the sign: it still moves
`sigma_i` from `T_i` alone.

### 2.3 Failure 3: the sensitivity is strongly non-local, and the diagonal ignores it

Structural, from the physics rather than from a measurement:
- The EQS problem is elliptic. Changing gamma at one cell changes the potential and therefore the
  field everywhere. `Q = 0.5 * sigma * |E|^2` depends on the entire sigma field, not on the local
  value.
- When power is enforced (`enforce_generator_power: true`), the P_abs renormalization adds an exact
  dense rank-one coupling to the Jacobian. The adjoint work derives and gates this term explicitly
  (`pabs_correct_r`, `ilt_adjoint.py:133`, gated to 1.78e-08 by brute-force Jacobian,
  `findings/dim_adjoint.md` ADJ-INFO-13).
- The thermal march diffuses: over the ~270 s exposure the diffusion length is 4.6-5.7 mm
  (`notes-ilt-adjoint.md:208-211`) against a part 16-20 mm across. A dopant change at one cell moves
  the temperature over a large fraction of the part.

So the true `dT/ds` is a dense matrix with a sign structure that varies with depth. Both production
paths approximate it by a positive diagonal.

---

## 3. How confident should we be in the current compensation results?

### 3.1 The 2-D / 2.5-D engine forward model: high confidence

PROVEN or COMPUTED:
- Cross-campaign baseline reproduction to better than 0.05 C (`GEOMETRY_DUAL_READSTATE.md` G1
  gate: square 4.931 vs 4.92, circle 17.428 vs 17.4, hexagon 18.083 vs 18.05, diamond 32.234 vs
  32.23).
- Independent driver reproduction gate to |d| = 0.0000 C on the iteration-4 sigma_T
  (`ALLISON_LAW_REPLICATION.md` Sec 3.1).
- Energy-residual gates below 5 % and zero temperature-step clipping on every kept solve
  (`ALLISON_LAW_REPLICATION.md` Sec 3.2; `POWER_MATCHED.md` G3).
- Time-to-state agrees with the measured untuned population using only a 60 s ramp calibration
  (208.0 s vs 207.7 +- 120.1 s at 120 C, etc., `ALLISON_LAW_REPLICATION.md` Sec 6.2). This is a
  genuine non-tautological pass.

### 3.2 The 2-D compensation results: moderate confidence in sign, weak in magnitude

COMPUTED, `POWER_MATCHED.md` per-shape table, sigma_T at melt-onset phi_bar = 0.90 at matched dose
(500 W/m):
- 11 of 12 voltage-beneficial shapes keep a grading-only benefit (circle +59.2 %, hexagon +67.6 %,
  rectangle +63.8 %, ellipse +57.5 %, octagon +55.6 %, and so on).
- Square collapses to +0.6 % at melt-onset while keeping +28.0 % at heating-peak.
- Four shapes remain harmful at matched dose: triangle -82.3 %, H_shape -60.1 %, trapezoid -55.6 %,
  T_shape -22.7 %. One shape flips from harmful to helpful (equilateral_triangle -5.9 % -> +6.5 %).
- L_shape never reaches melt and has no FGM arm at all.

What I would and would not stand behind:
- **Would stand behind:** the direction of the effect for the curved and rounded shapes, and the
  finding that spatial redistribution rather than extra dose is the dominant mechanism (the dose
  component is within +-7 points for 11 of 12 shapes).
- **Would not stand behind without qualification:** the specific percentages, because the magnitude
  was chosen best-of-four in-sample on the same metric.
- **Three untested risks, in order of severity.**
  1. **Grid convergence has never been tested in 2-D.** Every 2-D result is at a single fixed
     120x120 grid. The 3-D work PROVED that sigma_T of a sharp-cornered uniform-dopant part is not
     a mesh-converged functional, with the baseline itself moving **-44.4 % between n = 48 and
     n = 64** (`notes-ilt-adjoint.md:184-193`), and localized the cause to the EQS corner
     singularity at the one-cell material boundary. The 2-D engine has the same sharp binary
     boundary and the same corner geometry. ASSUMED, not measured: the same non-convergence is
     present in 2-D. This is the single largest open confidence hole on the 2-D side, and a 4-point
     grid ladder on one shape would close it.
  2. **The model over-predicts what tuning delivers in hardware by about 8x.** COMPUTED,
     `ALLISON_LAW_REPLICATION.md` Sec 6.1: the tuned simulated field is 1.54 C at the 135 C state
     against a measured tuned range of 12.95-24.98 C. The untuned arm lands inside the measured
     untuned population, so the forward is not broadly wrong; it is the *achievable tuned*
     uniformity that is over-predicted. Read every corrected-arm sigma_T as an upper bound.
  3. **Continuous per-cell dopant versus a 5-level printable map on a 1.82 mm track pitch**
     (`ALLISON_LAW_REPLICATION.md` Sec 10). The 2-D results are mostly quoted continuous or at
     4 bpp.

### 3.3 The 3-D (heatr3d) compensation results: low confidence, and one number needs a health warning

- Every published 3-D FGM number is single-grid n = 48 with the shipped binary (one-cell) material
  boundary (`PREWARP_HANDOFF.md:78-83`).
- At the only mesh-converged boundary width ever measured, the sign of the heuristic's benefit
  **flips**: -46.2 % (binary, n = 48) becomes +92.1 % (w = 1e-03, n = 48) and +105.8 %
  (w = 1e-03, n = 64) (`notes-ilt-adjoint.md:51-59`). The edge-width work also showed the binary
  model produces two sign flips across grids while the regularized model produces none
  (Spearman 0.929, no sign changes).
- The recommended cheap probe (uniform plus the chapter's FGM map, one sharp shape and one smooth
  shape, n = 48, w = 0 versus w = 1e-03, 8 runs, roughly 1 h) is **specified but not run**
  (`PREWARP_HANDOFF.md:78-83`). Until it runs, the honest qualifier on every 3-D FGM percentage is
  "at n = 48 with a sharp one-cell material boundary."
- Honest counterweight, stated by the edge-width report itself
  (`notes-ilt-adjoint.md:97-113`): w = 1e-03 is itself an assumed width, justified by printer drop
  pitch and applied isotropically including through thickness, which is an assumption not a
  measurement. At w = 1e-03, 38 % of the part is "skin" and 12 % of the dopant dose is removed on a
  part only 10 mm thick. We replaced one idealized boundary model with another. The **rankings** at
  a resolved width are stable; the **absolute values** are not portable.

**Net:** the 2-D compensation story is defensible with the caveats above. The 3-D compensation
percentages should not be quoted without the boundary qualifier, and the 8-run probe should be
treated as a blocker rather than a nice-to-have.

---

## 4. Why the prior adjoint "lost", and what is actually fixable

The brief states the verdict as: modest in-model gain (-8.8 % / -15.9 %) at ~300x cost, flipping to
+15.9 % worse on an n = 64 hold-out. **That is accurate for one map under one boundary model and has
since been reversed.** The corrected account, with sources:

### 4.1 The gradient was never the problem

PROVEN, and independently audited twice (`findings/dim_adjoint.md`): the adjoint is the exact
conjugate transpose of the assembled forward matrix (transpose residual 5.85e-16); the chain rule
including harmonic faces, the P_abs rank-one renormalization, the 2.5-D depth diagonal, transient
apparent-heat-capacity phase change, coupled densification and shrinkage-to-shape is complete;
independent FD agrees to 1e-08 to 1e-09; a tangent-linear cross-check matches to 1.37e-06; the
non-smooth temperature-step clip was correctly handled as a subgradient after an initial version
failed at a flat 4.3e-04. There is no gradient bug. Any explanation of the failure must be sought
elsewhere.

### 4.2 What actually went wrong, in order, with the fixability call

| # | cause | evidence | fixable? |
|---|---|---|---|
| 1 | **Regime mismatch (A1-A3).** The FD-conditioning power boost moved the optimization into Fourier number 0.019 where T is an image of Q_rf, so "flatten Q_rf" == "flatten T". Production runs at Fo ~ 1.4 where the peaked field is compensating boundary loss. | `PREWARP_HANDOFF.md:221-227`; A2 map scores 13.802 C in the 2-D engine against a 4.931 C baseline, +179.9 % (`A4_STATUS.md:153`) | **FIXED** in A5 by optimizing the actual production march. |
| 2 | **Incomplete actuator.** sigma-only, with eps_r pinned. The same map scores 2.784 C with eps_r co-varying and 17.297 C with eps_r pinned, a factor of 6.2. | `PREWARP_HANDOFF.md:231-232` | **FIXED** in A5 (dopant fraction s moves sigma and eps_r together). |
| 3 | **Over-fitting to the optimization model.** The self-score falls monotonically by ~20x while the transfer score is U-shaped in optimization depth. | `A6_STATUS.md` Sec 3.2: self-score Spearman versus production 0.643 | **FIXED** in A6 by a pre-registered hold-out stop rule; that rule reproduces A5's production argmin exactly, Spearman 1.000. |
| 4 | **The evaluation metric was not mesh converged.** The uniform baseline itself falls 11.868 -> 2.878 C over n = 32 to 64 with no plateau, so every "-X % versus uniform" divides by a moving quantity. Root cause localized to the EQS corner singularity at the one-cell material boundary: Q_rf max/mean diverges roughly linearly in n while whole-part and deep-core coefficients of variation are converged. | `notes-ilt-adjoint.md:184-211` | **FIXED** by resolving the material boundary at a fixed physical width (w >= 1.5 h); the ladder spread drops from 4.12x to 1.11x and Q_rf max/mean goes flat. Costed: sub-cell widths are strictly worse than binary; the physically principled 1e-04 needs n >= 300, roughly 12 days per arm, and is unreachable. |
| 5 | **Grid-specific tuning.** `n48c0` optimized on the production grid, so its 0.392 C is in-sample-grid; it degrades to 1.175 C at n = 64. | `notes-ilt-adjoint.md:181, 212-213` | **PARTLY STRUCTURAL.** Do not optimize on the evaluation grid. |
| 6 | **Budget, not convergence.** The A6 `n48c0` stop rule never fired: eleven of twelve validation checks set new minima and the run ended on `maxfun`. | `A6_STATUS.md` Sec 3.3; and see Section 9.2 below for the current state of the continuation | **FIXABLE with compute only.** |

### 4.3 The result the brief inverts

At the mesh-converged boundary width (`notes-ilt-adjoint.md:51-59`, four arms x two grids,
identical protocol, resampler and quantizer, dt = 0.02, zero stability-guard firings in all 22 runs):

| arm | sigma_T @ n=48 | vs uniform | sigma_T @ n=64 | vs uniform |
|---|---|---|---|---|
| uniform | 13.896 | n/a | 13.708 | n/a |
| heuristic 2 bpp, built natively | 26.689 | **+92.1 %** | 28.206 | **+105.8 %** |
| `n32c0` 2 bpp (adjoint) | **10.619** | **-23.6 %** | **9.867** | **-28.0 %** |
| `n48c0` 2 bpp (adjoint) | 11.221 | -19.3 % | 10.779 | -21.4 % |

All four adjoint maps beat uniform at both grids, **entirely out of model**: they were optimized
under the binary forward and never saw the smoothed one. The specific "+15.9 % worse at n = 64"
result the brief cites is explicitly retracted in the source: "REVERSED: meshcheck's 'n32c0 is
+15.9 % worse at n = 64.' That was binary-specific; at a converged width `n32c0` is -22.1 % and is
the best arm" (`notes-ilt-adjoint.md:74-75`).

Two corrections that follow, both of which should propagate:
- **"The adjoint does not reliably beat the heuristic" is not supported.** At the converged
  configuration it beats uniform by ~20-28 % grid-stably while the heuristic loses to uniform by
  ~92-106 %.
- **"~300x cost" is too high.** COMPUTED from `probe_cost.json` and the A6 logs: at n = 48 a full
  gradient costs 404.1 s against a 200.8 s forward, so about 2 forward-equivalents. The A6 `n48c0`
  run was 46 evaluations in 19 603 s wall, about 98 forward-equivalents. Against a heuristic that
  costs 1 forward plus a 4-point magnitude sweep (5 forwards), the ratio is roughly **20x**, not
  300x. The ratio only reaches the hundreds if the heuristic is counted as free.

### 4.4 The one thing that is structural

The **ceiling**, not the method. At a converged boundary width the best adjoint margin is about
-20 to -28 % versus uniform, permanently retiring the 13x / -92 % class of claim
(`PREWARP_HANDOFF.md:66`). And that margin is measured in a model that over-predicts achievable
tuned uniformity by roughly 8x against hardware (Section 3.2 item 2). So a better solve buys a
better number in a model whose corrected-arm predictions the hardware cannot currently confirm.

---

## 5. Is there a well-posed inverse design on the trusted 2-D engine?

Yes. Stating it precisely, because the well-posedness lives in the details.

### 5.1 The design problem

Design variable: `s in R^{N_part}`, the per-cell saturation, entering as
`sigma = sigma_d0 * s * (1 + a_T (T - T_ref)) * (1 + a_rho (rho - rho_0))`
(`rfam_eqs_coupled.py:437-443`), and, if the eps_r channel is enabled, also through
`effective_fill = fill_frac * sat_map` (`rfam_eqs_coupled.py:416`).

Constraints: `s in [0, s_max]` box; optionally a dose constraint `sum_part s = V0`; optionally a
total-variation or Gaussian-smoothing regularizer for printability; a quantizer applied only at
scoring, never inside the optimization.

Objective: this is the choice that decides whether the problem is well posed. Three candidates,
assessed:

1. **`Var_part(T)` at a FIXED exposure horizon.** Smooth, bounded, differentiable, and it is
   exactly what the A-series gated. **Recommended for layer 1.** Weakness: it is not the metric the
   dissertation reports.
2. **`Var_part(T)` at the melt-onset read state `t*(s)` where `phi_bar(t*, s) = 0.90`.** This is the
   reported metric. It is differentiable, but `t*` moves with `s`, so
   `dJ/ds = partial_s Var + (dVar/dt)|_{t*} * dt*/ds`, with
   `dt*/ds = -(partial_s phi_bar) / (partial_t phi_bar)` from the implicit function theorem, valid
   as long as `partial_t phi_bar > 0` at the crossing (it is, monotone melting). **The A-series
   deliberately omitted this term** (`A6_STATUS.md` Sec 5.3: "Fixed 284.3 s horizon in the
   optimizer; dt*/ds is not in the gradient. Production still scores at matched melt."). Adding it
   is one clean, FD-gateable layer and it would be a genuine methodological advance over the
   A-series, because it closes the gap between what was optimized and what is reported.
3. **Final relative-density uniformity, `Var_part(rho_final)`.** Attractive, since it is nearest to
   the physical goal. Layer B of the adjoint work already makes rho an evolving adjoint state with a
   gated coupled (T, rho) adjoint (`PREWARP_HANDOFF.md:369-378`, FD gate 5.3e-07 square). Risk: at
   long exposure the density field saturates and the gradient dies, which is the exact pathology
   Layer D hit for the melt objective ("at full melt the melt-fraction objective is SATURATED so
   dJ/dtheta has no useful signal and the optimizer SPURIOUSLY CARVES a lattice",
   `PREWARP_HANDOFF.md:252-254`). A density objective must be evaluated before saturation, and the
   read state must be pinned by a rule, not by convenience.

**Recommendation: objective 2, reached by layering 1 -> 2.** Objective 3 is the better long-term
target but should not be the first layer.

### 5.2 Is the coupled response differentiable enough?

Mostly yes, with four named non-smooth terms, all of which have known handling:

| term | where | status |
|---|---|---|
| temperature-step clip `max_deltaT_per_step_c = 10` | `thermal.max_deltaT_per_step_c` in the config | **SOLVED PATTERN.** The 3-D adjoint failed at a flat 4.3e-04 until the clip was handled as a subgradient (gate the num/denom sensitivities by the clip-inactive mask, leave the carried-identity term ungated); the fix produced a textbook clean V (`findings/dim_adjoint.md` ADJ-INFO-09). Port the same treatment. The clip fraction is 0.0 on all the kept 2-D solves, so it is likely inactive at the optimum, but that must be re-checked per iterate rather than assumed. |
| `max_qrf_w_per_m3` cap | `electric.max_qrf_w_per_m3 = 1e11` | measured inactive (`frac_at_qrf_cap = 0.0` across the sigma sweep). Add an assertion. |
| phase-change Heaviside | `phase_change.model: comsol_heaviside`, `smooth_shape: linear`, `dt_pc_c = 10` | already a smoothed ramp of finite width; treat `dt_pc_c` as the regularizer and verify the gate as it shrinks, exactly as the A-series did with tau_pc 2.5 -> 1.0. |
| densification per-step cap `max_delta_per_step = 0.02` | `densification` block | same subgradient treatment as the temperature clip. Only matters if the objective reaches rho. |

The remaining smoothness question is the box bound. At the optimum a large fraction of cells sits on
`s = 0` or `s = s_max` (`PERNODE_RESULTS.md:97` records 82 % of nodes at the clamp by iteration 12).
L-BFGS-B handles that correctly, but it means the useful design space is smaller than N_part and
that a plain unconstrained gradient norm is a misleading convergence signal.

### 5.3 Two engine-specific blockers that must be cleared first

1. **The 2-D direct injection hook hard-pins eps_r.** `rfam_eqs_coupled.py:342`:
   ```python
   block["eps_geometry_only"] = True
   ```
   is set unconditionally in `sat_map_npz_direct` mode. A5 measured that eps_r co-variation is worth
   a factor of 6.2 in 3-D (`PREWARP_HANDOFF.md:231-232`), and the A5 notes name this exact line as
   the reason the 2-D engine could not be used as a second production engine
   (`PREWARP_HANDOFF.md:208-209`). Fixing it is a one-flag change plus a test, but it changes the
   physics of every existing per-node result, so it must be a separate, gated change with the old
   behavior preserved by default.
2. **Any 2-D adjoint must differentiate a forward proven bit-identical to `rfam_eqs_coupled.run_sim`,
   not a look-alike.** The precedent to copy is the meshcheck work, which re-implemented heatr3d's
   march with per-term instrumentation and proved `max|diff| = 0.000e+00` against the production
   engine before drawing any conclusion (`notes-ilt-adjoint.md:153-155`). Without that gate, an
   FD-gated gradient only proves you differentiated your own re-implementation correctly. This is
   the largest single cost item in the prototype.

---

## 6. The missing control, and why it changes the recommendation

Every adjoint-versus-heuristic comparison in the record uses a heuristic whose gain was **not
calibrated to the measured non-uniformity**:

- In the 3-D converged-width comparison, the heuristic arm is the fixed-magnitude one-shot map, and
  it comes out +92 to +106 % worse than doing nothing. An arm that is worse than doing nothing is
  not a control; it is a broken baseline.
- In the 2-D campaigns the magnitude is picked best-of-four in-sample, which is closer to a control
  but is selected on the reported metric.

Since the diagnosed failure is a **gain** failure and not a **shape-of-the-map** failure, the
obvious control is a heuristic with a calibrated gain. Concretely: keep the map shape from the
proportional inverse, but set the scalar gain by a one-dimensional line search on the real forward,
selected on a hold-out read state (for example fit the gain at heating-peak and score it at
melt-onset, or fit on a coarser grid and score on the production one). That is 5 to 8 solves. It
directly removes the overshoot mechanism, because a line search on the actual objective cannot
choose a step that makes the objective worse.

**My reasoned expectation, stated as an expectation and not a measurement:** a gain-calibrated
heuristic recovers most of the ~20-28 % adjoint margin at roughly one tenth the cost, because a
one-dimensional line search on a nearly-quadratic objective captures most of the first-order
descent when the search direction is roughly right, and the proportional-inverse direction IS
roughly right for the shapes where it works (11 of 12 at matched dose). Where it is structurally
wrong (the over-critical sign inversion of Section 2.2, and the reentrant shapes) a line search
cannot save it, and there the adjoint should win outright. **This is a testable prediction and it is
the single most informative experiment available.**

---

## 7. Prototype design, IF the above is judged worth testing (designed, NOT built)

I am recommending this be **designed and held**, per the brief, and that Matt decide from the
assessment. Design below so the decision is concrete.

### 7.1 What it must answer

Not "does an adjoint work" (proven, Path D). The three open questions are:
- **Q1.** Does a **gain-calibrated** heuristic remove the harmful cases? (cheap, decisive)
- **Q2.** At what compute budget does an adjoint overtake the calibrated heuristic on a hold-out?
- **Q3.** Does including the moving melt-onset read state `dt*/ds` change the answer?

### 7.2 Cases

Two shapes, chosen because they bracket the failure mode, both already anchored with reproduced
baselines:
- **square**, 20 mm, 120x120, voltage 2428.2 V, 845 steps, config
  `outputs_eqs/fgm_dosecheck/configs/square_baseline_voltage.yaml`. Baseline melt-onset sigma_T
  4.92-4.93 C. The heuristic FAILS here (Section 2.1). Maximum headroom.
- **hexagon**, same protocol, baseline 18.05-18.08 C, heuristic already gives about -44 to -68 %.
  Tests whether a solve adds anything where the heuristic already works.

### 7.3 Layering, with an FD gate after every layer

Non-negotiable: PASS < 1e-06, central difference, epsilon swept over 1e-3 to 1e-7, expecting the
V-shaped relative error, both a random-direction probe and a single-cell probe at the maximum-|grad|
cell. Bisect any failure by re-gating the previous layer.

| layer | one change | gate |
|---|---|---|
| **L0** | Re-implement the 2-D EQS + thermal march and prove it **bit-identical** to `rfam_eqs_coupled.run_sim` on the square (target `max|diff| = 0.0` on T, Q_rf, phi and rho at every stored step). No gradient yet. | identity, not FD |
| **L1** | `dJ/ds` for `J = Var_part(T)` at a FIXED horizon, eps_r pinned (matching the current hook). | FD |
| **L2** | Temperature-step-clip subgradient (port the ADJ-INFO-09 pattern). Verify by forcing the clip active. | FD, plus a clip-active case |
| **L3** | Phase-change regularizer-width robustness: re-gate at `dt_pc_c` 10 -> 5 -> 2.5. | FD at three widths |
| **L4** | Moving read state: `J = Var_part(T(t*(s)))` with `phi_bar(t*, s) = 0.90`, `dt*/ds` by the implicit function theorem. | FD; this is the layer most likely to fail first |
| **L5** | eps_r co-varying (requires clearing the `eps_geometry_only` pin, Section 5.3). | FD |

### 7.4 The head-to-head, at matched budget, on a hold-out

Three arms, one budget `B` measured in forward-solve equivalents, swept over `B in {5, 15, 40}` so
the answer is a curve rather than a point:

- **Arm H0, current practice.** One-shot inverse, magnitude best-of-four in-sample. Cost 5.
- **Arm H1, gain-calibrated heuristic (the missing control).** One-shot inverse map shape; scalar
  gain by golden-section line search on the real forward; **selected on the hold-out read state**.
  Cost `B`.
- **Arm A1, adjoint.** L-BFGS-B on the FD-gated gradient, box [0, 1.5], start at uniform s = 1,
  **selection by the A6 rule**: validate every k-th evaluation on a hold-out, patience 3, report the
  argmin, and write the selected map to disk at the moment the running minimum is set, before any
  scoring number exists. Cost `B` (about `B/2` gradient steps).

**Pre-registration is mandatory.** Write the arms, the budgets, the hold-out definition and the
verdict criterion to a frozen file before the first optimization launch, exactly as
`A6_PREREGISTRATION.md` did. Report every pre-registered number, win or lose.

**Hold-out definition, stated up front (this is where the A-series earned its result).** Two
independent hold-outs, both cheap in 2-D:
- **read-state hold-out:** fit at heating-peak, score at melt-onset phi_bar = 0.90;
- **grid hold-out:** optimize at 120x120, score at 160x160.
The grid hold-out doubles as the missing 2-D mesh-convergence check (Section 3.2 risk 1), so it
should be run for the baseline arms regardless of whether the optimization goes ahead.

### 7.5 Cost, measured not assumed

COMPUTED during this assessment: one 845-step 120x120 2-D forward solve takes about **13-14 minutes**
wall on this machine (Section 9.1). So `B = 40` per arm per shape is about 9 h per arm, roughly
**50-55 h** for the full three-arm, two-shape, three-budget matrix. That is a weekend of compute, not
a research programme, but it is not free. If the budget is tight, **run Arm H1 alone at B = 8 on all
19 shapes first** (about 25 h) and answer Q1 before spending anything on the adjoint.

### 7.6 The honest failure mode of this prototype

If L4 (the moving read state) does not gate, the gradient is for a different objective than the one
reported and the comparison becomes apples-to-oranges. In that case fall back to L1's fixed horizon
for **all three arms**, including the heuristic, so that at least the comparison is internally
consistent, and say plainly that the optimized objective is not the reported metric.

---

## 8. Honest verdict

**Can compensation get meaningfully better?** Yes, in two distinct senses that should not be
conflated.

- **Reliability: large, cheap, high confidence.** The current method fails outright on a
  characterizable class of cases (nearly-flat baseline field; over-critical operating point;
  reentrant geometry) and the cause is a gain that does not scale with the error. Fixing the gain is
  5-8 solves per shape and should remove essentially all the harmful results. **Expected effect:
  turn 4-5 harmful shapes out of 19 into neutral-or-helpful, and remove the "+92 to +106 % worse
  than doing nothing" result in 3-D.** Confidence: high, because the mechanism is measured in two
  engines and is visible in the source.
- **Peak quality: modest, expensive, moderate confidence.** A properly regime-matched, hold-out-
  selected adjoint gets about **-20 to -28 % versus uniform** at the only mesh-converged
  configuration measured, at roughly 20x the cost of the heuristic. Whether it beats a
  *gain-calibrated* heuristic is **unknown and untested**. Confidence in the -20 to -28 % figure:
  moderate, because it is grid-stable and out-of-model, but it rests on one geometry and one assumed
  boundary width.

**Is invert-plus-local-search near the achievable frontier?** For the shapes where the direction of
the correction is right, and once the gain is calibrated, probably yes for practical purposes. For
the shapes where the direction is wrong (the over-critical sign inversion, the reentrant set) it is
not near the frontier and no amount of line searching will get it there, because the search direction
itself is wrong. That is precisely the domain where a solve earns its cost, and it is a minority of
the shape library.

**The binding constraint is not the algorithm.** The model over-predicts achievable tuned uniformity
by about 8x against the measured tuned parts. Until that gap is understood, moving a simulated
sigma_T from 2.5 C to 2.0 C is a claim about the model, not about a part. The highest-value
experiment in the whole workstream is not an inverse solve; it is the sectioned-part measurement of
the real dopant edge profile that the edge-width report asks for
(`notes-ilt-adjoint.md:112-113`), because it simultaneously fixes the boundary-width assumption and
constrains the 8x gap.

**Recommended order of work, subject to Matt's decision:**
1. **[P0]** The 8-run edge-width probe on the 3-D FGM numbers. Blocks the honest phrasing of every
   3-D percentage in the chapter. ~1 h.
2. **[P0]** Arm H1, the gain-calibrated heuristic, on the 2-D shape library. Answers Q1 and is the
   missing control for every adjoint comparison ever run here. ~25 h.
3. **[P1]** The 2-D grid hold-out on baselines only, which closes the untested 2-D mesh-convergence
   risk. ~4 solves per shape on two shapes.
4. **[P1]** The layered 2-D adjoint prototype of Section 7, only if step 2 leaves a gap worth the
   50 h.
5. **[P2]** The sectioned-part dopant edge measurement. Not compute; it is the one experiment that
   closes both the boundary-width assumption and the 8x hardware gap.

---

## 9. What I actually ran, and what it showed

### 9.1 Two-zone sensitivity probe on the trusted 2-D engine (new measurement)

**Purpose.** Test the diagonal-positive assumption directly: measure the true
`d(sigma_T)/d(sat)` for a hot zone and a cold zone on the real forward, and check whether the
direction the one-shot heuristic moves is actually a descent direction for the reported objective.

**Exactly what was executed.** Script
`scratchpad/probe_zone_sensitivity.py`, run under `./.venv312/bin/python` from the repository root.
Base configuration `outputs_eqs/fgm_dosecheck/configs/square_baseline_voltage.yaml` (square 20 mm,
120x120, voltage-driven 2428.2 V, `enforce_generator_power: false`, 845 steps at 0.5 s). Five
forward solves, all injected through the **same** two-sided direct hook
(`fgm_feedback.sat_map_npz_direct`, `sat_max = 1.5`) so no arm differs in its injection path:

| arm | saturation map |
|---|---|
| `base` | s = 1 everywhere |
| `corner_p` / `corner_m` | s = 1 +- 0.05 on cells with `|u| > 0.70` and `|v| > 0.70` (Chebyshev part coordinates), 264 cells |
| `interior_p` / `interior_m` | s = 1 +- 0.05 on cells with `max(|u|,|v|) < 0.35`, 196 cells |

Metric: `sigma_T = ui_rms * (Tmean - 23)` read at the first `phi_mean >= 0.90` crossing, the same
matched-melt read state as `PERNODE_RESULTS.md`. Central differences give
`d(sigma_T)/d(sat_zone)` for each zone; the inner product of that two-component gradient with the
heuristic's step direction (corner down, interior up) gives the directional derivative along the
heuristic's move. Negative means the heuristic descends; positive means it ascends.

**Base-arm gates, COMPUTED (`probe_zone_sens/base/summary.json`), all pass:**
`integrated_power_doped_W_per_m = 500.011` (the 500 W/m calibration reproduces),
`mean_T_part_final_c = 186.184` (reproduces the `Tt = 186.18 C` of `PERNODE_RESULTS.md:76-77`),
`frac_cells_dT_clipped_final = 0.0`, `frac_part_at_qrf_cap_final = 0.0`,
`frac_part_at_temp_cap = 0.0`. Two consequences worth carrying into the prototype design: the
injection path reproduces the published operating point exactly, and **the temperature-step clip and
the Q_rf cap are both inactive at the uniform baseline**, so the two worst non-smooth terms of
Section 5.2 are dormant here. They must still be re-checked per iterate once the optimizer starts
pushing cells to their bounds; dormant at the start is not dormant at the optimum.

**RESULT (all five solves completed; `probe_zone_sens/probe_results.json`).** Part 1600 cells,
corner zone 144 cells, interior zone 196 cells.

| arm | sigma_T @ melt-onset [C] | phi at read | P_abs [W/m] | clip fraction |
|---|---|---|---|---|
| `base` (s = 1) | **4.9196** | 0.8993 | 500.01 | 0.0 |
| `corner_p` (+0.05) | 5.2090 | 0.9006 | 504.83 | 0.0 |
| `corner_m` (-0.05) | 4.5879 | 0.8904 (see caveat) | 495.12 | 0.0 |
| `interior_p` (+0.05) | 4.9538 | 0.8977 | 499.91 | 0.0 |
| `interior_m` (-0.05) | 4.8874 | 0.9001 | 500.11 | 0.0 |

Central differences:

```
d(sigma_T)/d(sat_corner)   = +6.2103 C per unit sat
d(sigma_T)/d(sat_interior) = +0.6646 C per unit sat
```

**Four findings, all new measurements on the trusted 2-D engine:**

1. **The base arm reproduces the published number to four figures: 4.9196 C against the published
   4.92 C.** The probe is anchored.
2. **The interior derivative is POSITIVE.** Raising interior dopant makes uniformity **worse**.
   The heuristic raises interior dopant, because the interior reads as cold. So the interior
   component of the heuristic's step is an **ascent** component. This is the diagonal-sign failure
   of Section 2.2, now measured directly on the reported objective rather than inferred from the
   sigma sweep. The mechanism is visible in the absorbed power: `interior_p` and `interior_m`
   absorb 499.91 and 500.11 W/m against the baseline's 500.01, i.e. the interior sits essentially
   **at the top of the Q(sigma) peak where dQ/dsigma is approximately zero**, exactly where the
   sigma sweep put sigma0 = 0.04. The corner cells, which see a different local field, are still on
   the rising branch (504.83 / 495.12 W/m, about +-1 %).
3. **The heuristic's overall direction is nevertheless still descent, but it is badly weighted.**
   The two-zone directional derivatives:

   | move | directional derivative [C per unit step] | cosine with steepest descent |
   |---|---|---|
   | heuristic: corner down, interior up | -3.9214 | **+0.628** |
   | corner down only | **-6.2103** | **+0.994** |
   | lower both zones | -4.8613 | +0.778 |

   The true sensitivity ratio between the two zones is **9.34 to 1**. The heuristic weights them
   roughly 1 to 1 and gets the interior sign wrong. It is about **63 % aligned** with steepest
   descent on this projection, where a move that simply lowers the corners is **99 % aligned**.
   **This is the cleanest single statement of what a solve buys: not a new direction, a correctly
   weighted and correctly signed one.** It also explains why the per-node law works better than the
   one-shot map on this case (it re-measures every iteration, so it eventually finds the corners)
   and why it still leaves a plateau (it never learns the 9-to-1 weighting).
4. **The reported objective has a genuine non-differentiability at the horizon, and I hit it.**
   `corner_m` never crossed phi_bar = 0.90 inside the fixed 845-step horizon (it ends at 0.8904), so
   its read state falls back to the final step rather than a true melt-onset crossing
   (`run_pernode_square.py:69-72` implements exactly this fallback). The corner central difference is
   therefore slightly contaminated: one side is read at a crossing and the other at the horizon.
   This is a real property of the metric as defined, not a bug in the probe, and it is direct
   evidence for the Section 7.3 warning that layer L4 (the moving read state) is the layer most
   likely to fail its gate. Any adjoint of the melt-onset objective must either guarantee the
   crossing exists inside the horizon, or handle the fallback branch explicitly as a subgradient.

**Cost, measured:** the five solves took 335 s to 1294 s each depending on machine contention,
median about 460 s, worst about 22 min. The Section 7.5 budget uses the conservative end.

**What the probe does and does not settle.** It settles that on the square the heuristic's failure
is predominantly a **weighting and gain** failure rather than a wholesale direction failure, which
supports the Section 6 recommendation to test a gain-calibrated heuristic first. It does not settle
the full-resolution question: two zones is a two-dimensional projection of a 1600-dimensional
gradient, and the full gradient may contain structure that no two-zone projection reveals. That is
what the Section 7 prototype would measure.

### 9.2 Re-verification of the A6 continuation status (correction to the handoff)

The handoff and the notes both record the `n48c0` continuation as "STILL RUNNING". COMPUTED by
inspecting `out_adjoint_fgm_a6/meshcheck/optcont_n48c0_cont_progress.json` directly:

- Last write **2026-07-26 18:02**. Today is 2026-07-30. **The continuation is not running; it
  stopped four days ago**, which is why `optcont_n48c0_cont_summary.json` never appeared.
- It reached continuation evaluation **52** (global 97), further than the notes' last reading of 41.
- Hold-out score on the n = 40 grid: best **1.7243 C at continuation evaluation 49**, versus
  A6's selected map at 2.0536 C. **Thirteen consecutive validation checks, thirteen new running
  minima. The hold-out still never turned.**
- Implication, unchanged in kind but stronger in degree: A6's reported `n48c0` map is under-trained
  by **16.0 %** on its own hold-out metric. The selection rule is not implicated; the budget is.

Recommended edit to `notes-ilt-adjoint.md` and `PREWARP_HANDOFF.md`: change "STILL RUNNING" to
"STOPPED at continuation evaluation 52 on 2026-07-26; hold-out never turned; best 1.7243 C at
evaluation 49". I have not made that edit, since the brief scopes me to analysis.

### 9.3 Aggregation of the stored sigma sweep

The table in Section 2.2 was computed by me from `outputs_eqs/sigma_sweep/sweep_summary.json`
(17 stored runs, prior work). No new solves. The interior-power peak at sigma = 0.035 for erosion
depths 1, 2 and 4 cells is a straight read of that file and is consistent with the analytic
`sigma* = 0.0302 S/m` in `SIGMA_PEAK_RECONCILIATION.md`.

---

## 10. Proven / computed / assumed ledger

**PROVEN (unit-tested or FD-gated, by prior work):**
- The per-node law is a faithful implementation of Allison's `Tune_Conductivity.m` (9 unit tests);
  the two-sided continuous hook behaves as specified (3 unit tests).
- The A-series adjoint is the exact conjugate transpose of the assembled forward operator, with a
  complete chain rule including harmonic faces, P_abs renormalization, 2.5-D depth, transient phase
  change, coupled densification and shrinkage-to-shape; independently audited twice; FD 1e-08 to
  1e-09; the non-smooth temperature-step clip is correctly handled as a subgradient.
- The A6 hold-out selection rule reproduces the production argmin on an independent trajectory,
  Spearman 1.000.

**COMPUTED (measured from real solves or stored artifacts):**
- 2-D FGM benefit at matched dose across 19 shapes: 11 of 12 keep a melt-onset grading benefit,
  4 remain harmful, 1 flips helpful, 1 neutral, 1 not reached.
- 2-D square one-shot FGM at the INTEGRAL magnitude: sigma_T 4.92 -> 9.01 C at melt-onset, +83 %.
- 3-D at the converged boundary width: heuristic +92.1 % / +105.8 %; adjoint maps -19.3 % to
  -28.0 %; Spearman 0.929 across grids with no sign flips.
- Interior power peaks at sigma ~= 0.035 while total power is monotone increasing; the operating
  point sigma0 = 0.04 is past the peak; FGM maps therefore raise absorbed power by 12.5-45.5 %.
- Adjoint gradient cost about 2 forward-equivalents; a full A6 optimization about 98
  forward-equivalents; ratio to a 5-solve heuristic about 20x.
- The A6 `n48c0` continuation stopped on 2026-07-26 at evaluation 52 with the hold-out still
  improving; the reported map is 16.0 % under-trained on that metric.
- 2-D forward cost 335-1294 s per 845-step 120x120 solve on this machine (median about 460 s).
- **New this pass (5 solves):** base sigma_T 4.9196 C reproduces the published 4.92 C;
  `d(sigma_T)/d(sat_corner) = +6.2103` and `d(sigma_T)/d(sat_interior) = +0.6646` C per unit sat at
  melt-onset; sensitivity ratio 9.34 to 1; the heuristic direction is 63 % aligned with steepest
  descent against 99 % for corner-only; the interior sits at the top of the Q(sigma) peak
  (P_abs 499.91 / 500.11 / 500.01 W/m across the interior perturbations) while the corners are still
  on the rising branch (504.83 / 495.12 W/m); the temperature-step clip and the Q_rf cap are
  inactive in all five arms; and the melt-onset read state falls back to the horizon when an arm
  fails to cross phi_bar = 0.90 (`corner_m` ended at 0.8904).

**ASSUMED (not measured, flagged as such):**
- That the 2-D sigma_T shows the same mesh non-convergence the 3-D work proved. Plausible from the
  shared sharp binary boundary and sharp corners; **untested**.
- That a gain-calibrated heuristic recovers most of the adjoint's margin. This is my reasoned
  expectation, offered as the prediction to test, not as a result.
- That w = 1e-03 is a defensible physical boundary width. The source report says plainly it is an
  assumption drawn from printer pitch and applied isotropically including through thickness.
- That the shapes where the heuristic works are those where the proportional-inverse direction is
  approximately correct. Consistent with all the data I read, but not separately demonstrated.

---

## 11. Sources read (absolute paths)

Code:
- `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_generator.py`
- `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/heatr3d.py`
- `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/pernode_tuning.py`
- `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/rfam_eqs_coupled.py`
- `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/rfam_gui_server.py`
- `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/scripts/analysis/run_pernode_square.py`

Records:
- `.../geo-prewarp/outputs_eqs/pernode_square/PERNODE_RESULTS.md`
- `.../geo-prewarp/outputs_eqs/allison_law_replication/ALLISON_LAW_REPLICATION.md`
- `.../geo-prewarp/outputs_eqs/fgm_dosecheck/DOSECHECK.md`
- `.../geo-prewarp/outputs_eqs/geometry_dual_readstate/{GEOMETRY_DUAL_READSTATE.md, POWER_MATCHED.md}`
- `.../geo-prewarp/outputs_eqs/sigma_sweep/{SIGMA_PEAK_RECONCILIATION.md, sweep_summary.json}`
- `.../dissertation_materials/analysis-3dfgm/PREWARP_HANDOFF.md`
- `.../dissertation_materials/analysis-3dfgm/notes-ilt-adjoint.md`
- `.../dissertation_materials/analysis-3dfgm/findings/dim_adjoint.md`
- `.../dissertation_materials/analysis-3dfgm/out_adjoint_fgm_a6/{A6_STATUS.md, probe_cost.json}`
- `.../dissertation_materials/analysis-3dfgm/out_adjoint_fgm_a4/A4_STATUS.md`
- `.../dissertation_materials/analysis-3dfgm/out_adjoint_fgm_a6/meshcheck/optcont_n48c0_cont_progress.json`
