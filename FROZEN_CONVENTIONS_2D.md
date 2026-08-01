# Frozen two-dimensional solve conventions

Status: FROZEN as of 2026-08-01. This is the handover document the three-dimensional
port lane's Phase C and Phase D were waiting on. Every convention below is either
already in production use in `fgm_solve_campaign/adjoint2d/` or is newly frozen by the
topology-optimization pass (`TOPOPT_REPORT.md`). Each one carries its citation: a code
location, a measured number, or the report that established it.

Nothing here is a recommendation. These are the settings the two-dimensional solve
actually runs, and the port should reproduce them exactly so that a three-dimensional
result can be attributed to the extra dimension rather than to a changed convention.

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

Acronyms on first use: RFAM radio-frequency additive manufacturing; FGM functionally
graded material; EQS electro-quasi-static; IoU intersection over union; bpp bits per
pixel; L-BFGS-B limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints;
MMA the method of moving asymptotes.

---

## 1. Design parameterization

### 1.1 The chain

```
v  --F-->  v_f  --P_beta-->  s  --forward-->  T  --J-->  scalar
```

`v` is the design variable, one value per in-part cell, box `[0, 1]`.
`s` is the physical dopant saturation injected into the forward.
Code: `fgm_solve_campaign/adjoint2d/topopt.py`.

### 1.2 Filter F: type and PHYSICAL radius

* **Type**: normalized-convolution Gaussian over the part only,
  `s = gaussian(v * chi) / gaussian(chi)` inside, held at the nominal value outside.
  Code `adjoint2d/design_filter.py:50-64`; the same convention as
  `adjoint2d/robust.smooth_in_part` so the solve filter and the robustness probe cannot
  drift apart.
* **Why a normalized convolution and not a plain blur**: it is a convex combination of
  in-part values, so `v` in `[0, 1]` gives `s` in `[0, 1]` automatically and NO clip
  subgradient enters the chain rule. Tested in
  `adjoint2d/tests/test_design_filter.py::test_box_is_preserved`.
* **Radius**: **`FILTER_RADIUS_M = 1.0e-3` metres**, a Gaussian standard deviation, held
  as a LENGTH and converted per grid by `topopt.sigma_cells_for(radius_m, dx)`.
  On the pinned 60 mm domain that is **1.983 cells at grid 120** (dx = 0.5042 mm) and
  **2.650 cells at grid 160** (dx = 0.3774 mm).
* **Why 1.0 mm, in full, because this was the open question**:
  1. The printer's dopant edge scale, roughly 50 to 100 micrometres, is **NOT** the
     binding constraint and must not be used as the radius. It is five to ten times
     FINER than the cell size of any grid the solve runs on (504 micrometres at 120,
     377 micrometres at 160), so it can never restrict what the solve is able to
     express. Using it would be the same as using no filter at all.
  2. The binding constraint is solver convergence. `MULTISTART_REPORT.md` Section 6
     measured that a 1.5-cell radius (0.756 mm at grid 120) is not enough: the filtered
     square still went from IoU 0.9681 in grid to 0.8063 across the grid hold-out.
  3. The next resolvable step is two cells at grid 120, which is 1.0 mm, and 1.0 mm is
     also resolved with 2.65 cells at grid 160, so the same physical feature is
     representable on both grids with margin.
  ASSUMED, and named as assumed: the radius was frozen at 1.0 mm and **not swept**. A
  sweep at 0.75, 1.0 and 1.5 mm on three shapes is the cheapest open experiment.

### 1.3 Projection P: smoothed Heaviside, and its schedule

* **Form**: the tanh projection of Wang, Lazarov and Sigmund,
  `P(u) = [tanh(beta*eta) + tanh(beta*(u - eta))] / [tanh(beta*eta) + tanh(beta*(1 - eta))]`.
  Code `adjoint2d/topopt.py:project`.
* **Threshold**: **`ETA = 0.5`**. At this value P fixes 0, 0.5 and 1 exactly, is
  monotone, and maps `[0, 1]` onto `[0, 1]`, so again no clip enters the chain.
  Tested: `test_topopt.py::test_projection_fixes_both_endpoints_and_the_threshold`.
* **Continuation schedule**: **`BETA_SCHEDULE = (1, 2, 4, 8, 16)`**, one L-BFGS-B stage
  each, each stage restarting from the previous stage's best iterate.
  * beta = 1 first because the cold start is uniform saturation 1, which sits on the
    upper rail: at beta = 16 the projection derivative there is 1.8e-06 and the gradient
    is numerically dead. At beta = 1 it is 0.851.
  * Doubling and not a larger jump because the optimizer's curvature memory is discarded
    at each restart, so a stage must re-converge inside its own share of the budget.
  * Stop at 16 because at eta = 0.5 it already maps 0.65 above 0.95 and 0.35 below 0.05.
* **Budget split across stages**: `topopt.stage_split(pool, n_stages)`, even split with
  the remainder to the EARLIEST stages, and when the pool is smaller than the stage count
  the LATE stages are dropped entirely rather than every stage being starved. Measured
  pools on the six shapes were 14 to 16, giving splits like `[3, 3, 3, 3, 2]`.
* **Flag-off identity**: `beta <= 0` makes `design_to_map` BIT-IDENTICAL to
  `design_filter.apply_filter`, checked by
  `test_topopt.py::test_composed_map_at_beta_zero_is_bit_identical_to_the_existing_filter`.
  The port must keep an equivalent switch.
* **HONEST CAVEAT, from this pass**: the projection is frozen as the convention but it
  is NOT established as a net win. At 40 forward-equivalents the beta = 0 control at the
  same radius reaches better in-grid AND better cross-grid fidelity on four of six
  shapes, while the projection is what removes the sub-radius sensitivity on square and
  circle. See `TOPOPT_REPORT.md` Section 1. The port should carry both arms.

### 1.4 Gradient through the parameterization

`dJ/dv = F^T [ P'(F v) * dJ/ds ]`, restricted to the part, zero outside. Code
`topopt.design_vjp`. The linearization `topopt.design_jvp` is provided so the transpose
can be checked by the dot-product identity at ANY design point, which is the bisect that
separates a chain-rule error from a property of the forward.
MEASURED: exact to **4.19e-16 (square) and 2.14e-16 (circle)**, worst over beta 0, 1 and
16 (`out_topopt/gate_topopt_*.json`, key `transpose_consistency`).

---

## 2. The target indicator chi: grid independent, never a solve-grid raster

* **Construction**: the sub-cell AREA FILL, the cell average of the geometric indicator,
  evaluated by the SAME supersampling routine the production domain builder uses for its
  material fill fraction, `rfam_eqs_coupled._subpixel_fill_fraction`
  (`rfam_eqs_coupled.py:605-635`), at a higher sample count. Multi-part union by
  `np.maximum`, matching `rfam_eqs_coupled.py:1595`.
  Code: `adjoint2d/chi_area.py`.
* **Sample count**: **`CHI_N_SUB = 32`**, 1024 samples per cell, against the production
  fill's `_N_SUB = 8` (`rfam_eqs_coupled.py:1208`).
* **Evaluation is band restricted** for cost: only cells whose centre lies within half a
  cell diagonal of the polygon boundary are supersampled, the rest take their centre
  value. Proven bit-identical to the brute-force production sampler by
  `test_chi_area.py::test_band_restriction_is_bit_identical_to_the_production_sampler`
  and by a second test on a grid-aligned square. Cost falls from 16 s to 0.4 s per call
  on a 720-vertex circle at grid 120.
* **Why not the binary raster**: MEASURED on a 10 mm circle in the 60 mm domain, the
  binary raster's target area is off the closed form by +0.34 percent at grid 120 and
  -0.83 percent at grid 160, so a map solved at 120 was being scored at 160 against a
  DIFFERENT target. The area fill is within 0.01 percent at both.
  On the six campaign shapes at grid 120 the raster OVER-states the part by **+0.34
  percent (circle) to +3.23 percent (diamond)**.
* **Measured grid stability**, `out_topopt/<shape>_robust.json` key
  `chi_area_grid_consistency`: between grid 120 and grid 160 the area-fill target moves
  **-0.105 percent (square)** and **-0.052 percent (trapezoid)** while the binary raster
  moves **-1.660 percent (square)**.
* **Limit, stated**: this is an area fill, not a signed distance. It carries the correct
  cell average, which is what a cell-summed quadratic objective needs, but it carries no
  boundary normal or curvature and therefore cannot drive a level-set velocity.
* **Reporting rule**: J against the area-fill chi is NOT numerically comparable to any J
  in `out_lib` or `out_ms`. Every result file also records `J_raster_chi`, the same map
  scored under the old binary target, and IoU against the binary mask, which are the two
  comparable readings.

---

## 3. Drive convention

* **Voltage**: the per-shape calibrated `electric.voltage_v` from
  `outputs_eqs/fgm_calibrated_control/configs/<shape>_m*.yaml`, chosen by the earlier
  calibration so the UNIFORM arm absorbs 500 W per metre of depth at grid 120.
  Selection rule `adjoint2d/library_solve.shape_config`: the first name in sorted order,
  which is deterministic and physically arbitrary because within a shape every config
  differs only in the stored dopant-map path that the prototype never reads.
* **`enforce_generator_power: False`**, always. `adjoint2d/pins.py:98-99` raises
  `UnsupportedConfig` if it is true, because that branch adds a rank-one absorbed-power
  term that this adjoint layer does not carry.
* **Voltage mode `grounded`**, `pins.py:102-104`. Explicit hi/lo voltages are rejected.
* **Dose is NOT matched**, and this limit is quoted every time an arm comparison is.
  Measured spread on this pass's six deliverable arms: **316.2 to 495.7 W per metre**.
* **Grid recalibration**: when scoring at a grid other than the one the voltage was
  calibrated on, rescale once by `robust.recalibrated_voltage(v0, p_measured, 500.0)`.
  The EQS solve is exactly quadratic in the drive and the material fields do not depend
  on it, so one rescale is exact and no iteration is needed. Measured recalibrated
  voltages at grid 160: square 2834.8, circle 3783.7, trapezoid 2957.0, triangle 3191.5,
  diamond 2919.3, rectangle 3588.8 volts.

---

## 4. Boundary convention: saturation outside the part

**Outside the part the saturation is held at 1.0, the nominal value.** Citation
`adjoint2d/control.py:39-61`, whose comment records the measurement: with 0 outside
instead of 1, the uniform triangle reference moved from 23.06 C to 39.37 C in the
permittivity-co-varying channel, because the sub-pixel boundary cells carry geometry
fill between 0 and 1 and zeroing the saturation there silently changes the GEOMETRY as
well as the dopant.

The same convention is enforced in `design_filter.apply_filter` (`outside=1.0`),
`topopt.design_to_map`, `printability.quantize_in_part` and `robust.smooth_in_part`. A
design variable is defined only on the part; the filter transpose and the projection
derivative are both zero outside.

---

## 5. Finite-difference gate checklist

Every gradient must clear this list BEFORE any optimization is run with it. It is the
list this pass ran; see `adjoint2d/gate_topopt.py` for the reference implementation.

1. **L0 bit identity.** The forward reproduces a stored production run in the same
   channel to the last bit, and any new channel is bit-identical to the old one when its
   flag is off. Reference `adjoint2d/gate_l0.py`; new-channel example
   `test_topopt.py::test_composed_map_at_beta_zero_is_bit_identical_to_the_existing_filter`.
2. **Layered bisect.** Add ONE thing per layer so a failure localizes. This pass ran
   P1 target only, P2 add the filter, P3 add the projection at beta 1, P4 at beta 16.
3. **Three-probe protocol per layer, plus two.** Maximum-sensitivity cell, a fixed
   pseudo-random in-part cell, a random unit direction; plus a FILTER-SMOOTH random
   direction whenever a filter is in the chain, and the gradient direction. Code
   `gate_rho._probe_dirs` and `gate_topopt.gradient_direction`.
   Rationale for the smooth direction: a rough random direction is mostly cell-scale
   content, exactly what the filter removes, so its analytic derivative is damped while
   the objective's roundoff floor is not.
4. **Central differences, epsilon swept over eight values**,
   `EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8)` (`gate_rho.py:47-52`).
   Expect a V-shaped relative error bottoming between 1e-6 and 1e-7.
5. **Pass standard: 1e-6 preferred, 1e-5 is the campaign SUBGRADIENT standard**
   (`gate_rho.PASS_REL_ERR`, `gate_rho.SUBGRADIENT_PASS_REL_ERR`). The objective reads a
   clipped melt fraction, so cells at the clip are kinks and a smooth-function standard
   is not available. Report the count at both.
6. **Measure the evaluation floor, do not assume it.** In the roundoff-dominated tail
   the central-difference error is `floor / (2 eps)`, so `2 eps * abs_err` at the two
   smallest epsilons estimates the objective's absolute floor. MEASURED on this pass:
   **3.2e-12 to 1.0e-10** across 40 probes.
7. **Interpret a relative-error failure against the analytic magnitude.** MEASURED on
   this pass: the best ABSOLUTE error is flat, 4.1e-07 to 1.3e-04, across every layer and
   both shapes, while the analytic directional derivative spans 0.047 to 296. Every probe
   that missed the relative standard had a small derivative. Report the scatter of
   absolute error against analytic magnitude, not only the relative number.
8. **Filter and projection transpose exactness.** The dot-product identity
   `<dJ/dv, d> = <dJ/ds, dS(v)[d]>` must hold to machine precision at every beta.
   Threshold 1e-10; measured 4.19e-16 worst. This is the bisect that decides whether a
   residual belongs to the chain rule or to the forward.
9. **Read-state stability.** Report whether the objective's argmin index moves under the
   probe perturbations rather than assuming the envelope theorem covers it. Measured on
   this pass: it does NOT move at epsilon 1e-3 on either shape.
10. **Flag-off bit identity for every new channel**, so an existing gated result cannot
    be perturbed by adding a channel that is switched off.

---

## 6. Optimizer, budget and start

* **Optimizer: L-BFGS-B** (`scipy.optimize.minimize`, `jac=True`, box bounds,
  `ftol=1e-16`, `gtol=1e-16`, budget enforced by raising `StopIteration` from the
  objective). **MMA is the field standard for topology optimization and was NOT
  implemented**; L-BFGS-B is the substitution and is named as such. The practical
  consequence, visible in `figs_topopt/fig_topopt_continuation.png`, is that a
  significant share of a small budget is spent on line-search trial points that are
  worse than the incumbent, which MMA's convex separable subproblem would not do. Each
  stage keeps its own BEST iterate, so no bad trial is carried forward.
* **Budget: 40 forward-equivalents per shape**, `library_solve.BUDGET_FORWARD_EQUIVALENTS`.
  Converted to a gradient-evaluation pool by the per-shape adjoint-to-forward cost ratio
  RECORDED BY THE LIBRARY CAMPAIGN in `out_lib/<shape>.json` key `cost.ratio`, not the
  ratio timed in the current run. Reason, measured: timing a single forward and adjoint
  on a loaded machine gave a ratio of 12.91 on the star shape against the library
  campaign's 1.11 for the same calls, which would have silently starved the solve
  (`ms_solve.budget_ratio`). Measured pools on this pass: 14 to 16 gradient evaluations.
* **Start: filtered full-depth SINGLE cold start** from uniform saturation 1. Warm
  starting is used ONLY where a strong historical mask exists for that shape, and the
  warm start injects the FILTERED historical mask, so its first evaluation is not the
  historical arm's score (`ms_solve.build_starts`). Multi-start splits the same budget
  and `MULTISTART_REPORT.md` Section 4 found the single cold start better on 12 of 18
  shapes at this budget.
* **Stop rule**: `t_stop = argmin` over the arm's OWN stored trajectory of J.
  `at_horizon` is flagged whenever the minimum sits on the last stored step, which makes
  that arm's J an upper bound. Early-stop patience `library_solve.PATIENCE = 250`.
  NOTE for the port: the forward's early stop internally uses the BINARY raster chi
  (`forward.py:447`). Measured shift between the raster argmin and the area-fill argmin
  on this pass, at the gate design point: **square, index 1214 against 1224, 10 steps;
  circle, 924 against 925, 1 step**. Both are far inside the 250-step patience, so the
  truncation does not cut off the area-fill argmin. It is not exact and it is named.
* **Deliverable arm**: the continuous map quantized to **4 bits per pixel inside the
  part** (`printability.quantize_in_part`, 16 levels, nominal value held outside),
  re-run through the real forward. Never the continuous map alone.
* **Class threshold**: `library_solve.SOLVED_IOU = 0.95`, absolute, IoU against the
  binary part mask.

---

## 7. Actuator channels

* **Conductivity only is the DEPLOYABLE channel** and is what every adjoint arm in this
  and every previous pass actuates: `eps_covary=False`.
  `sigma(x,y) = sigma_v + sat(x,y) * fill_frac(x,y) * (sigma_d0 - sigma_v)`
  (`rfam_eqs_coupled.py:252`).
* **The permittivity channel is MODEL ONLY, pending the material question.** The
  gradient exists, is gated, and is large (`EPS_CHANNEL_REPORT.md` Section 1 item 5:
  the new term's norm is 0.65 to 1.54 times the conductivity gradient's, with a cosine
  of only 0.73 and 0.56 between them). But the production code contains a hook that
  holds the doped part's relative permittivity fixed while only conductivity varies
  (`rfam_eqs_coupled.py:342`, `eps_geometry_only`, consumed at
  `rfam_eqs_coupled.py:2531`). Until it is settled whether the real binder co-varies
  permittivity, **every permittivity-channel census is a model result with no deployment
  path** (`EPS_CHANNEL_REPORT.md` Section 11). The port must default the channel OFF and
  must not quote the permittivity census as deployable.
* Historical stored dopant masks were SCORED in the permittivity-co-varying channel.
  Comparing an adjoint arm against them is therefore an actuator-mismatched comparison
  and that must be stated wherever the comparison appears.

---

## 8. Acceptance gates before any SOLVED label

Adopted from the three-dimensional port lane and now implemented in
`adjoint2d/topopt_robust.py` and `adjoint2d/topopt_dosematch.py`.

* **Gate A, grid hold-out.** Solve at grid 120, score at grid 160, with the drive
  voltage recalibrated at 160 so the uniform arm absorbs 500 W per metre, and with the
  target chi REBUILT from the geometry at 160. Two transfers are run and both are
  reported: the production map transfer (bilinear then clip, `robust.resample_map`,
  reproducing `rfam_eqs_coupled.py:374-380`) and the design transfer (resample `v`,
  re-apply the filter and projection at 160 with the same physical radius). Report the
  uniform arm's OWN move between grids alongside, because it bounds how much of any drop
  is forward discretization rather than map transfer.
* **Gate B, sub-filter-radius perturbation.** Blur the delivered map by a part-masked
  normalized-convolution Gaussian at radii strictly below the filter radius and re-score.
  Tolerance: less than 10 percent change in J. Report the radius AT the filter length as
  context, not as part of the gate.
  Two extra forms, both added by this pass and both recommended for the port:
  * **design space**, blur `v` and re-project, which tests the intended claim without
    also testing how far a blur moves the level set of a nearly binary map;
  * **dose matched**, recalibrate the drive so the blurred arm absorbs the unblurred
    arm's power, because a part-masked blur moved absorbed power by up to +7.97 percent
    on this pass and that is a confound inside the gate.
* **A map does not get a SOLVED label unless Gate A and Gate B both pass.** On this pass
  no map earned that label under all forms; see `TOPOPT_REPORT.md` Section 1.
* **Standing energy-residual gate**, `adjoint2d/energy_gate.py`, 5 percent of integrated
  dose at the arm's own stop, checked on every scored forward run. Zero violations on
  this pass.

---

## 9. What is deliberately NOT frozen

Named so the port does not read silence as agreement.

1. **The filter radius value.** Frozen at 1.0 mm and not swept.
2. **Whether the projection is worth its cost.** Frozen as the convention, not
   established as a net win at 40 forward-equivalents.
3. **The beta schedule and the stage split rule.** Both are conventions, neither swept.
4. **MMA.** Not implemented; L-BFGS-B substituted.
5. **The 40 forward-equivalent budget.** Inherited, and every J reported anywhere in the
   campaign is an upper bound because of it.
6. **The permittivity channel's deployability.** One literature or bench question,
   unanswered, and it gates the value of the largest measured improvement in the
   campaign.
7. **Any experimental validation.** These are two-dimensional model results.
   `ALLISON_LAW_REPLICATION.md` Section 6.1 records that the model over-predicts
   achievable tuned uniformity by roughly a factor of eight against hardware.
