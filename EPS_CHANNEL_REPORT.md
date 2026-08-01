# The permittivity channel in the shape-fidelity solve, and the census with the actuator confound removed

**Date:** 2026-08-01. **Scope:** the dopant design variable now moves relative
permittivity as well as conductivity in the differentiable forward AND in the
adjoint; a finite-difference gate run before any optimization; a bit-identity
regression proving the new flag off changes nothing; a bit-identity check
against stored production runs in the co-varying channel; an 18-shape
re-solve; the de-confounded census; and the two robustness probes on the new
square and cross maps. Grid 120 x 120 unless a number says 160. **Nothing was
committed. No dissertation file was touched. `.claude/worktrees/` was not read
or written.**

**Acronyms, expanded on first use.** EQS = electro-quasi-static (the
low-frequency Maxwell approximation the two-dimensional solver uses). FGM =
functionally graded material (a spatially varying dopant saturation map). IoU =
intersection over union. bpp = bits per pixel. L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. FD = finite difference.
VJP = vector-Jacobian product. phi = melt fraction. rho = relative density.
W/m = watts per metre of depth. eps_r = relative permittivity. sigma =
electrical conductivity.

**Evidence tags.** PROVEN = unit tested, bit-identity checked, or FD gated.
COMPUTED = measured from a real run in this pass. ASSUMED = a modelling choice
or an inference not measured here.

**Objective and stop convention, stated once and carried on every number.**

    J_phi(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with `chi_part` the rasterized binary part mask. Every J_phi, IoU, growth,
under-melt, mean relative density and absorbed power is read at that arm's OWN
J-stop, t_stop = argmin over that arm's own stored trajectory of J_phi.
`HORIZON` is flagged when the minimum sits on the last stored step, which makes
that arm's J a bound; **COMPUTED: no arm in this pass stopped at the horizon.**
The melted region for IoU, growth and under-melt is phi >= 0.5; J_phi itself
uses no threshold. Absorbed power is the state-B value in W/m. Mean relative
density is the part-cell mean at that same stop index.

**Grid qualifier, mandatory.** `SOLVE_ROBUSTNESS_VALIDATION.md` established
that absolute fidelity at grid 120 does not transfer to grid 160 and that the
forward itself is not converged in IoU between those grids. Every IoU below is
a property of the method AT GRID 120 with an exactly reproduced dopant map, not
a property of the method. Section 7 is the only place a 160 number appears and
it is labelled.

---

## 1. Verdict, up front

**1. The actuator was most of the gap, and closing it flips the census.**
COMPUTED. With the dopant moving conductivity only, the solve beat the best
stored historical dopant mask on **13 of 18** shapes on J_phi and put **7 of 18**
in the IoU >= 0.95 class at grid 120. With the dopant moving conductivity AND
permittivity, the same objective, the same design filter, the same budget and
the same 18 shapes give **17 of 18** on J_phi, **16 of 18** on IoU, and **10 of
18** in the IoU >= 0.95 class. Summed J_phi over the library falls from **2679**
(conductivity-only multi-start) and **2661** (conductivity-only filtered cold
start) to **1401**. The four-way class ladder goes from 13 wins with 5 losses to
**10 SOLVED, 7 IMPROVED, 1 NOT RESCUED**.

**2. Yes, the solve now beats the historical mask on the compact shapes it
lost.** COMPUTED, and this was the headline question.

| shape | J_hist | J conductivity only | J permittivity, 40 fwd-equiv | J permittivity, best of two | IoU_hist | IoU new |
|---|---|---|---|---|---|---|
| square | 12.77 | 35.56 | **9.91** | **8.39** | 0.9975 | **1.0000** |
| rectangle | 10.69 | 97.19 | **9.51** | **7.00** | 1.0000 | 1.0000 |
| rounded_rect | 8.97 | 31.65 | **8.25** | **5.65** | 0.9974 | **1.0000** |
| star6 | 70.57 | 72.42 | 84.55 | 84.55 | 0.8782 | 0.8395 |

Three of the four flip. **star6 is the single remaining loss in the whole
library** and it got worse, not better: 72.42 to 84.55, IoU 0.8742 to 0.8395.

**3. Yes, the cross finally moves, and it moves further than anything else.**
COMPUTED. J_phi 385.07 (conductivity-only multi-start) to **106.50**, a **72.3
percent** reduction, and IoU 0.6508 to **0.9079**. Against the historical mask
it goes from losing by 16 percent to winning by **67.9 percent** (331.36 to
106.50). `VERIFICATION_PRINTABILITY_REPORT.md` Section 3.1 measured the actuator
gap on exactly this shape (the same historical map scores J 343.27 with
permittivity co-varying and J 1036.00, nothing melts at all, with conductivity
only) and named adding the channel as the most valuable next layer. It was.

**4. And the cross win is not bought with dose.** COMPUTED. At its own J-stop
the new cross arm absorbs **365.2 W/m** against the historical mask's **397.9
W/m**, so it scores three times better on J_phi while absorbing 8 percent LESS
power. The same holds on **square (540.8 against 559.8), rounded_rect (612.0
against 639.9), rectangle (483.5 against 483.7) and trapezoid (567.6 against
571.3)**: five shapes beat the historical mask at equal or lower absorbed
power. The other twelve wins absorb MORE than the mask they beat, up to 2.1
times on the L_shape and T_shape, and those wins ARE partly dose. Nothing here
is power matched and Section 6.3 quotes the spread.

**5. The permittivity term is not a small correction; it is most of the
gradient.** COMPUTED, from the gate's channel-split diagnostic, which recomputes
the same reverse sweep with the new term switched off:

    ||dJ/ds with permittivity minus dJ/ds without|| / ||dJ/ds without||

is **0.646 (square, no filter), 0.899 (square, filtered), 0.806 (cross, no
filter) and 1.537 (cross, filtered)**, with the cosine between the two channel
gradients only **0.73 (square)** and **0.56 (cross)**. On the filtered cross the
new term is larger than the old gradient and points half a right angle away from
it. A conductivity-only solve was not descending a slightly wrong direction; it
was descending a materially different one.

**6. The permittivity channel helps on 16 of 18 shapes at MATCHED budget, and
the two exceptions are named.** COMPUTED, comparing the filtered cold start at
40 forward-equivalents in each channel (same filter, same start, same budget,
same stop rule): J_phi falls on 16 shapes by 15.5 to 94.9 percent, and rises on
**hexagon (-75.6 percent, 24.24 to 42.57)** and **star6 (-3.1 percent, 82.03 to
84.55)**. The hexagon regression is a start-quality effect and not a channel
effect: its warm start reaches 7.47 in the same channel, so the shape ends at
**7.47 against the conductivity-only 24.24**, a 69 percent improvement, once
both starts are run.

---

## 2. What changed in the code, and the bit-identity guarantee

Three functions, all in `fgm_solve_campaign/adjoint2d/`.

**`adjoint.eqs_vjp(..., with_eps=False)`.** The complex material coefficient is
`gamma = sigma + 1j*omega*EPS0*eps_r`, so `d gamma / d sigma = 1` and
`d gamma / d eps_r = 1j*omega*EPS0`. Three consequences, and they are the whole
implementation:

* the DIRECT term of the absorbed power, `Q_raw = 0.5*pf*Re(gamma)*|E|^2`, has
  no permittivity dependence at all, because `Re(gamma) = sigma`. The entire
  permittivity sensitivity travels through the field;
* the assembled-operator term is the SAME complex quantity in both channels,
  multiplied by the respective `d gamma / d parameter`, so the permittivity
  branch reuses the existing `base`, `dgf_dk` and `dgf_dn` untouched and adds
  `2*Re(base * dgf * 1j*omega*EPS0)` into a second accumulator;
* at omega = 0 the channel is exactly dead, which is asserted rather than
  assumed.

The conductivity accumulation is left byte for byte where it was, so asking for
the permittivity channel cannot perturb the conductivity gradient by a
summation-order rounding change.

**`adjoint.gradient(..., eps_covary=False)`.** Adds
`(dJ/deps_A + dJ/deps_B) * fill_frac * (eps_d - eps_v)` to the same dJ/ds,
because one design variable drives both channels. The permittivity field is
assembled ONCE in the forward and handed to both electrical states, so both
states feed the same chain factor.

**`forward.eps_field(..., covary=...)`.** Already existed and is unchanged; the
default remains the pinned channel.

### 2.1 Red-first tests, and what each one buys

**PROVEN. 173 tests pass** (`.venv312/bin/python -m pytest adjoint2d/tests -q`),
of which **6 are new in this pass and every one was observed failing first**,
with the failures being `TypeError: eqs_vjp() got an unexpected keyword argument
'with_eps'` and `TypeError: gradient() got an unexpected keyword argument
'eps_covary'`. `adjoint2d/tests/test_eps_channel.py`:

| test | what it proves |
|---|---|
| `test_eps_field_covary_matches_the_production_effective_fill_expression` | the co-varying permittivity field equals `eps_v + (fill*s)*(eps_d - eps_v)` exactly, and the pinned field ignores s |
| `test_eqs_vjp_eps_matches_brute_force_finite_difference` | the EQS VJP in permittivity reproduces a brute-force central difference of the absorbed-power field to better than **1e-7** relative on an 11 x 13 grid |
| `test_asking_for_the_eps_channel_leaves_the_sigma_sensitivity_bit_identical` | `np.array_equal` on the conductivity sensitivity with and without `with_eps=True` |
| `test_eps_sensitivity_scales_with_omega_eps0_as_the_gamma_chain_requires` | the channel is identically zero at omega = 0, which is the chain-rule factor and not a fitted constant |
| `test_gradient_with_eps_covary_off_is_bit_identical_to_the_old_call` | **the flag-off regression**: on the real engine, at a read state with a populated phase ramp, `np.array_equal(gradient(...), gradient(..., eps_covary=False))` |
| `test_gradient_with_eps_covary_on_moves_the_gradient` | the flag-off test is not vacuous |

The bit-identity test was written first in a form that PASSED VACUOUSLY (a
200-step march leaves nothing melted, the objective seed is identically zero,
and both gradients are the zero array). That was caught by adding
`assert np.max(np.abs(a)) > 0.0` and lengthening the march, and the assertion is
kept in the test.

### 2.2 The forward reproduces a stored production run in this channel, exactly

**PROVEN.** `adjoint2d.verify_hist gate` reads a stored historical run's own
`used_config.yaml`, loads that run's own dopant map through the PRODUCTION
loader `rfam_eqs_coupled._FgmFeedback.from_config`, marches the prototype in
identity mode with the permittivity channel ON, and compares the stored
`T_phi90` field inside that run's `fields.npz`. Result file
`fgm_solve_campaign/out_eps/stored_run_gate_eps.json`:

| stored run | melt-onset outer index | max abs difference in T_phi90 | sigma_T prototype | sigma_T stored |
|---|---|---|---|---|
| `outputs_eqs/geometry_dual_readstate/runs/square/baseline` (no FGM) | 845 | **0.000e+00** | 4.930989680787 | 4.930989680787 |
| `.../runs/square/fgm_m0p85` (permittivity co-varying) | 672 | **0.000e+00** | 3.768226015489 | 3.768226015489 |
| `.../runs/triangle/fgm_m0p85` (permittivity co-varying) | 572 | **0.000e+00** | 45.357859475541 | 45.357859475541 |
| `.../runs/cross/fgm_m0p85` (permittivity co-varying) | 663 | **0.000e+00** | 31.400589027953 | 31.400589027953 |

Those three FGM runs have `fgm_feedback.enabled: true` with
`saturation_map_npz`, which is the eps-co-varying hook. The prototype forward in
that channel is bit-identical to archived production output.

### 2.3 A second identity check, free, on all 18 shapes

**COMPUTED.** With s = 1 everywhere, `fill*s = fill`, so the uniform arm must be
identical in both channels. Every one of the 18 solves re-runs the uniform arm
in the permittivity channel and compares it against `out_lib`:
**maximum absolute difference in J_phi = 0.000e+00 on all 18 shapes.**

---

## 3. The finite-difference gate, run BEFORE any optimization

Code: `adjoint2d/gate_eps.py` and `adjoint2d/gate_eps_bisect.py`. Raw:
`out_eps/gate_eps_{square,cross}.json`,
`out_eps/gate_eps_bisect_{square,cross}.json`. Logs: `logs_eps/gate_eps*.log`.

Central differences at a FIXED read index equal to the argmin of J_phi on the
base run (the envelope theorem removes the dt*/ds term because the stop is
optimized to stationarity, and whether the argmin actually moves is measured
below, not assumed). Three layers, on **square and cross**:

* **E0** conductivity only, filtered. A CONTROL at the same read state, so a
  residual that is a property of the forward at this point shows up in both
  channels instead of being blamed on the new term.
* **E1** permittivity channel, no design filter.
* **E2** permittivity channel WITH the design filter. This is the gradient the
  solve uses.

Probes: the maximum-sensitivity cell, a fixed pseudo-random in-part cell, a
random unit direction, for the filtered layers a smooth random direction, and
the gradient direction (the direction L-BFGS-B steps along, which has the
largest available analytic derivative and therefore reports the smallest
relative error the arithmetic floor permits). Epsilon swept
**1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8** and then RESWEPT densely
inside the usable window at **3e-6, 2e-6, 1.5e-6, 1e-6, 7e-7, 5e-7, 3e-7, 2e-7,
1.5e-7, 1e-7, 7e-8**. Every probe is reported; none was dropped.

### 3.1 Results, coarse sweep

| shape | layer | probe | analytic | best relative error | best epsilon |
|---|---|---|---|---|---|
| square | E0 control | max-sensitivity cell | -9.8535e+00 | 1.267e-05 | 1e-06 |
| square | E0 control | random cell | -5.4745e-01 | 3.877e-06 | 1e-05 |
| square | E0 control | random direction | +1.0149e+00 | 1.380e-04 | 1e-06 |
| square | E0 control | smooth random direction | +4.8640e+00 | 2.126e-05 | 1e-06 |
| square | E0 control | gradient direction | +4.0625e+01 | 5.018e-06 | 3e-07 |
| square | E1 eps | max-sensitivity cell | -4.7899e+00 | **7.436e-07** | 1e-06 |
| square | E1 eps | random cell | +4.9962e-01 | 5.885e-06 | 1e-06 |
| square | E1 eps | random direction | -1.2445e+00 | 2.079e-05 | 1e-06 |
| square | E1 eps | gradient direction | +3.3845e+01 | 5.261e-06 | 1e-07 |
| square | E2 eps filtered | max-sensitivity cell | +1.5579e+00 | **1.245e-06** | 1e-05 |
| square | E2 eps filtered | random cell | +4.1616e-01 | 5.630e-06 | 1e-05 |
| square | E2 eps filtered | random direction | -7.0423e-01 | 3.955e-06 | 1e-05 |
| square | E2 eps filtered | smooth random direction | -2.9567e+00 | 1.568e-05 | 1e-06 |
| square | E2 eps filtered | **gradient direction** | +2.6342e+01 | **1.044e-06** | 1e-07 |
| cross | E0 control | max-sensitivity cell | -1.5602e+00 | **1.590e-06** | 1e-05 |
| cross | E0 control | random cell | -2.5135e-01 | **4.379e-07** | 1e-04 |
| cross | E0 control | random direction | +3.5122e-01 | **7.200e-07** | 1e-05 |
| cross | E0 control | smooth random direction | +1.4755e+00 | **1.056e-06** | 1e-05 |
| cross | E0 control | gradient direction | +9.7722e+00 | **2.117e-07** | 3e-08 |
| cross | E1 eps | max-sensitivity cell | +6.2076e+00 | **7.307e-08** | 1e-06 |
| cross | E1 eps | random cell | +4.7485e-01 | **6.371e-08** | 1e-04 |
| cross | E1 eps | random direction | -1.0326e+00 | **1.891e-06** | 1e-05 |
| cross | E1 eps | gradient direction | +2.1040e+01 | **1.076e-07** | 3e-08 |
| cross | E2 eps filtered | max-sensitivity cell | +1.5531e+00 | **1.516e-06** | 1e-05 |
| cross | E2 eps filtered | random cell | +5.8695e-01 | 2.598e-05 | 3e-07 |
| cross | E2 eps filtered | random direction | -1.1360e+00 | **1.276e-06** | 1e-05 |
| cross | E2 eps filtered | smooth random direction | -4.1176e+00 | 9.323e-06 | 1e-06 |
| cross | E2 eps filtered | **gradient direction** | +1.4937e+01 | 5.475e-06 | 3e-07 |

### 3.2 Results, refined sweep inside the usable window, and the gate verdict

The coarse sweep's epsilon grid is too sparse for the objective's usable window,
which `MULTISTART_REPORT.md` Section 2.3 already located at roughly 3e-7 to
1e-6. Resweeping the window densely on the filtered permittivity layer:

| shape | probe | analytic | best relative error | best epsilon | absolute error |
|---|---|---|---|---|---|
| square | max-sensitivity cell | +1.5579e+00 | 6.922e-06 | 3e-06 | 1.08e-05 |
| square | random cell | +4.1616e-01 | **4.316e-07** | 3e-06 | 1.80e-07 |
| square | random direction | -7.0423e-01 | 6.886e-06 | 3e-07 | 4.85e-06 |
| square | smooth random direction | -2.9567e+00 | 4.108e-06 | 2e-06 | 1.21e-05 |
| square | **gradient direction** | +2.6342e+01 | **2.132e-07** | 3e-06 | 5.62e-06 |
| cross | max-sensitivity cell | +1.5531e+00 | 2.480e-06 | 3e-06 | 3.85e-06 |
| cross | random cell | +5.8695e-01 | 3.476e-06 | 7e-07 | 2.04e-06 |
| cross | random direction | -1.1360e+00 | 2.287e-06 | 3e-06 | 2.60e-06 |
| cross | smooth random direction | -4.1176e+00 | 1.682e-06 | 3e-06 | 6.93e-06 |
| cross | **gradient direction** | +1.4937e+01 | **1.835e-07** | 5e-07 | 2.74e-06 |

**GATE VERDICT, stated exactly and not inflated. The permittivity gradient is
VERIFIED at the campaign's documented 1e-5 subgradient standard, on both shapes
and on every probe, and it does NOT reach the 1e-6 clean-smooth standard on
most probes.** Counts on the filtered layer with the refined sweep:
**square 5 of 5 at 1e-5 and 2 of 5 at 1e-6; cross 5 of 5 at 1e-5 and 1 of 5 at
1e-6.** The decision-relevant probe, the direction the optimizer steps along,
gates at **2.13e-07 (square)** and **1.84e-07 (cross)**. Without the filter the
permittivity layer is better still: **cross 3 of 4 at 1e-6, best 7.31e-08**.

### 3.3 The bisect, and the residual is NOT the new term

Three independent pieces of evidence localize what is left.

**PROVEN, to machine precision.** The design filter F is linear in v at fixed
masks, so the directional derivative along d in the design variable must equal
the directional derivative along F_lin(d) in the map. That identity involves no
finite difference. In the PERMITTIVITY channel: maximum relative error
**3.004e-16 (square)** and **2.859e-16 (cross)**. The filtered permittivity
gradient is therefore EXACTLY the unfiltered permittivity gradient composed with
a proven-exact linear operator.

**COMPUTED, and this is the decisive comparison.** On the square, at the SAME
read state and with the SAME filter, the CONDUCTIVITY-ONLY control layer E0
clears the 1e-5 standard on only **2 of 5** probes (random direction 1.380e-04)
while the permittivity layer E2 clears **5 of 5** after the refined sweep. The
residual is worse in the OLD channel than in the new one at this design point.

**COMPUTED.** The floor, estimated from the roundoff-dominated tail as
`2*eps*abs_err`, is **5.95e-12 to 4.87e-10 absolute**, that is roughly 4e-14 to
3e-12 relative to a J_phi of 162 to 455, the same class
`MULTISTART_REPORT.md` Section 2.3 measured. The remaining disagreement is one
to four orders of magnitude ABOVE that floor, so it is not roundoff: it is the
objective's non-smoothness under a multi-cell perturbation. The filter turns a
one-cell probe into a Gaussian patch, and only **175 of 14400 cells (cross)**
and **536 of 14400 (square)** sit strictly inside the phase-change ramp at the
read state, so a patch perturbation is far more likely to push a cell across a
clip than a single-cell one is.

### 3.4 The read state does not move

**COMPUTED.** `gate_eps.stop_index_stability` recomputes the J_phi argmin under
every probe perturbation at epsilon 1e-3 in the permittivity channel:
**it does not move on either shape** (square base index 836, cross base index
436, `moved = False` on all probes).

---

## 4. What was solved, and the budget honestly stated

Recipe, the production one from `MULTISTART_REPORT.md`: **filtered full-depth
single-start solves**, no budget split, the physical-length design filter on the
design variable at sigma = 1.5 cells (0.75 mm at grid 120), box [0, 1],
L-BFGS-B on dJ_phi/dv, deliverable quantized to 4 bpp inside the part through
the production quantizer and re-run through the real forward.

**TWO STARTS, each at the FULL budget.**

| start | what it is |
|---|---|
| `cold` | uniform saturation 1 |
| `warm` | the best stored historical 4-bpp dopant mask as `out_lib/<shape>.json` selected it, in the boundary convention that won there, loaded through the PRODUCTION loader and clipped into the box |

**The warm start is a different object now, and that is the point.** In the
conductivity-only campaigns the historical mask was scored in the permittivity
channel and injected into a conductivity-only solve, so the start point was the
map without its channel. Here the start and the solve share an actuator for the
first time.

**BUDGET, named rather than absorbed.** Each start got the full 40
forward-equivalents, so a shape cost 80 in total. `MULTISTART_REPORT.md`
Section 4.2 measured that splitting one 40-equivalent budget across starts costs
more depth than the breadth is worth on 12 of 18 shapes, so splitting is the
wrong control. **COMPUTED spend: 75.3 to 79.1 forward-equivalents per shape, 28
to 36 gradient evaluations, 14 to 18 per start** (the per-shape
adjoint-to-forward ratio is read from `out_lib/<shape>.json` exactly as
`ms_solve` does, so the evaluation pool is deterministic and independent of
machine load). Outside the solve budget and named: one cost-probe forward (which
doubles as the uniform arm) plus one adjoint, and two final scoring runs per
start.

**Because of that, the census reports the budget-matched arm separately.**
`EPS_cold_4bpp` used exactly 40 forward-equivalents and is the like-for-like
comparison against the 40-equivalent `out_lib` and `out_ms` arms.
`EPS_best_4bpp` is the better of the two starts and cost twice that. **COMPUTED:
the class ladder is IDENTICAL for both arms (10 SOLVED, 7 IMPROVED, 1 NOT
RESCUED) and the census counts are identical (17 of 18 on J_phi, 16 of 18 on
IoU, 10 of 18 at IoU >= 0.95), so the headline does not depend on the extra
budget.** The extra budget buys summed J_phi 1636 to 1401, a 14 percent
reduction, and nothing else.

**COMPUTED. The warm start wins on 14 of 18 shapes**, the cold start on 4
(triangle, equilateral_triangle, star, star6). Once the actuator matches, the
historical mask is a good start point rather than a competitor.

**gt_logo was SKIPPED, loudly.** Its geometry is rasterized from an image and
`rfam_eqs_coupled.make_domain` raises `ModuleNotFoundError: No module named
'cv2'` in the `.venv312` interpreter. Nothing in this pass changed that. The
library count is 18, not 19.

---

## 5. The census, all 18 shapes

Deliverable arm `EPS_best_4bpp`. `win` is the winning start. `lvl` is the number
of distinct printer levels the map uses inside the part. `J_ms` is the
conductivity-only multi-start arm, `J_ctl` the conductivity-only filtered cold
start, `J_lib` the conductivity-only unfiltered single start.

| shape | win | J_eps_best | J_eps_cold | J_hist | J_ms | J_ctl | J_lib | J_unif | IoU_eps_best | IoU_hist | IoU_ms | grow % | under % | rho at stop | P_abs W/m | lvl | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | warm | **8.39** | 9.91 | 12.77 | 35.56 | 18.51 | 25.66 | 210.19 | **1.0000** | 0.9975 | 0.9681 | 0.00 | 0.00 | 0.7740 | 540.8 | 7 | SOLVED |
| circle | warm | **5.93** | 7.24 | 52.18 | 15.55 | 12.33 | 14.68 | 275.02 | **1.0000** | 0.9492 | 0.9872 | 0.00 | 0.00 | 0.7307 | 716.1 | 10 | SOLVED |
| hexagon | warm | **7.47** | 42.57 | 69.63 | 24.57 | 24.24 | 22.15 | 251.27 | 0.9941 | 0.9211 | 0.9650 | 0.00 | 0.59 | 0.6427 | 847.1 | 14 | SOLVED |
| triangle | cold | **44.30** | 44.30 | 171.27 | 82.29 | 82.29 | 87.99 | 202.82 | 0.9327 | 0.7768 | 0.8783 | 4.00 | 3.00 | 0.6715 | 609.2 | 15 | IMPROVED |
| equilateral_triangle | cold | **63.30** | 63.30 | 145.55 | 100.06 | 97.89 | 119.00 | 320.80 | 0.9000 | 0.8058 | 0.8451 | 3.86 | 6.53 | 0.6973 | 681.3 | 16 | IMPROVED |
| L_shape | warm | **337.77** | 355.84 | 597.86 | 530.14 | 528.35 | 521.20 | 538.42 | 0.6838 | 0.4413 | 0.5109 | 2.87 | 29.66 | 0.6750 | 488.2 | 16 | IMPROVED |
| H_shape | warm | **81.51** | 101.41 | 219.11 | 149.76 | 120.07 | 150.86 | 214.68 | 0.9238 | 0.7631 | 0.8499 | 4.86 | 3.12 | 0.6799 | 613.4 | 16 | IMPROVED |
| T_shape | warm | **457.74** | 476.96 | 676.39 | 612.03 | 612.13 | 609.29 | 614.26 | 0.5829 | 0.3983 | 0.4583 | 1.63 | 40.76 | 0.6443 | 480.5 | 16 | IMPROVED |
| cross | warm | **106.50** | 213.78 | 331.36 | 385.07 | 380.93 | 360.18 | 471.82 | 0.9079 | 0.7052 | 0.6508 | 6.95 | 2.90 | 0.7181 | 365.2 | 16 | IMPROVED |
| diamond | warm | **51.99** | 60.13 | 355.07 | 272.11 | 227.78 | 211.16 | 768.11 | 0.9647 | 0.7793 | 0.8161 | 1.11 | 2.46 | 0.6754 | 719.7 | 16 | SOLVED |
| ellipse | warm | **2.62** | 2.64 | 74.70 | 28.84 | 21.46 | 13.98 | 241.72 | **1.0000** | 0.8854 | 0.9524 | 0.00 | 0.00 | 0.6543 | 775.2 | 13 | SOLVED |
| octagon | warm | **4.20** | 6.43 | 16.22 | 10.58 | 11.96 | 10.66 | 42.96 | 0.9963 | 0.9888 | 0.9963 | 0.00 | 0.37 | 0.8147 | 862.5 | 14 | SOLVED |
| pentagon | warm | **8.95** | 24.05 | 134.64 | 57.66 | 49.65 | 44.22 | 300.33 | 0.9914 | 0.8446 | 0.9253 | 0.43 | 0.43 | 0.6925 | 734.5 | 13 | SOLVED |
| rectangle | warm | **7.00** | 9.51 | 10.69 | 97.19 | 187.57 | 196.94 | 203.07 | 1.0000 | 1.0000 | 0.8842 | 0.00 | 0.00 | 0.6988 | 483.5 | 9 | SOLVED |
| rounded_rect | warm | **5.65** | 8.25 | 8.97 | 31.65 | 16.06 | 18.21 | 266.38 | **1.0000** | 0.9974 | 0.9795 | 0.00 | 0.00 | 0.7314 | 612.0 | 11 | SOLVED |
| star | cold | **114.06** | 114.06 | 172.11 | 144.45 | 163.22 | 157.39 | 192.58 | 0.7862 | 0.6713 | 0.7138 | 7.01 | 15.87 | 0.6126 | 583.7 | 16 | IMPROVED |
| star6 | cold | 84.55 | 84.55 | **70.57** | 72.42 | 82.03 | 84.69 | 190.62 | 0.8395 | **0.8782** | 0.8742 | 9.46 | 8.11 | 0.6277 | 611.9 | 15 | NOT RESCUED |
| trapezoid | warm | **9.38** | 11.13 | 122.22 | 29.27 | 24.12 | 27.36 | 171.32 | 0.9966 | 0.8688 | 0.9693 | 0.25 | 0.08 | 0.7381 | 567.6 | 13 | SOLVED |

### 5.1 The census counts, six arms against two baselines

| arm | beats HIST on J_phi | beats HIST on IoU | beats UNIFORM on J_phi | IoU >= 0.95 at grid 120 | summed J_phi |
|---|---|---|---|---|---|
| permittivity, best of two full-depth starts (80 fwd-equiv) | **17 of 18** | **16 of 18** | 18 of 18 | **10 of 18** | **1401** |
| permittivity, cold full depth (40 fwd-equiv, budget matched) | **17 of 18** | **16 of 18** | 18 of 18 | **10 of 18** | 1636 |
| permittivity, warm full depth (40 fwd-equiv) | **17 of 18** | **16 of 18** | 18 of 18 | **10 of 18** | 1552 |
| conductivity only, multi-start (`out_ms`) | 13 of 18 | 13 of 18 | 18 of 18 | 7 of 18 | 2679 |
| conductivity only, filtered cold start (`out_ms` control) | 13 of 18 | 13 of 18 | 18 of 18 | 7 of 18 | 2661 |
| conductivity only, single start (`out_lib`) | 13 of 18 | 13 of 18 | 18 of 18 | 7 of 18 | 2676 |

**The two shapes that do not beat the historical mask on IoU are star6 (0.8395
against 0.8782, a real loss) and rectangle (1.0000 against 1.0000, an exact
tie which the strict comparison scores as a non-win while J_phi falls from 10.69
to 7.00).** So there is exactly ONE genuine remaining loss in the library.

**The historical baseline is an ORACLE and must be quoted as one.** Selecting
that mask cost 28 to 38 forward solves per shape
(`SHAPE_LIBRARY_SOLVE_REPORT.md` Section 5), against the 40 forward-equivalents
the budget-matched solved arm was given.

### 5.2 What the channel bought at matched budget

Filtered cold start, 40 forward-equivalents, same filter, same start, same stop
rule; only the actuator differs.

| shape | J conductivity only | J permittivity | dJ % | IoU conductivity only | IoU permittivity | dIoU |
|---|---|---|---|---|---|---|
| rectangle | 187.57 | 9.51 | **+94.9** | 0.8580 | 1.0000 | +0.1420 |
| ellipse | 21.46 | 2.64 | +87.7 | 0.9681 | 1.0000 | +0.0319 |
| diamond | 227.78 | 60.13 | +73.6 | 0.8432 | 0.9611 | +0.1179 |
| trapezoid | 24.12 | 11.13 | +53.8 | 0.9792 | 0.9950 | +0.0158 |
| pentagon | 49.65 | 24.05 | +51.6 | 0.9289 | 0.9640 | +0.0351 |
| rounded_rect | 16.06 | 8.25 | +48.6 | 0.9974 | 1.0000 | +0.0026 |
| square | 18.51 | 9.91 | +46.5 | 1.0000 | 1.0000 | +0.0000 |
| octagon | 11.96 | 6.43 | +46.2 | 0.9852 | 0.9963 | +0.0110 |
| triangle | 82.29 | 44.30 | +46.2 | 0.8783 | 0.9327 | +0.0544 |
| cross | 380.93 | 213.78 | +43.9 | 0.6589 | 0.7980 | +0.1391 |
| circle | 12.33 | 7.24 | +41.2 | 0.9936 | 1.0000 | +0.0064 |
| equilateral_triangle | 97.89 | 63.30 | +35.3 | 0.8370 | 0.9000 | +0.0630 |
| L_shape | 528.35 | 355.84 | +32.7 | 0.5135 | 0.6670 | +0.1534 |
| star | 163.22 | 114.06 | +30.1 | 0.6809 | 0.7862 | +0.1054 |
| T_shape | 612.13 | 476.96 | +22.1 | 0.4583 | 0.5653 | +0.1070 |
| H_shape | 120.07 | 101.41 | +15.5 | 0.8599 | 0.8777 | +0.0178 |
| star6 | 82.03 | 84.55 | -3.1 | 0.8408 | 0.8395 | -0.0013 |
| hexagon | 24.24 | 42.57 | -75.6 | 0.9746 | 0.9502 | -0.0244 |

---

## 6. Gates, flags and the dose-match limit

**Energy-residual gate.** COMPUTED: **zero violations on every arm of every
shape**, across the 18 solves (5 scored arms each) and the 6 robustness runs. The
maximum relative energy residual at any arm's own stop is **2.21 percent**
against the standing 5 percent threshold.

**Horizon flags.** COMPUTED: **none.** Every arm's J_phi minimum is interior, so
no J in this report is an upper bound for that reason. This is a change from the
conductivity-only pass, where the diamond deliverable stopped at the horizon.

**Melt-fraction stop.** COMPUTED. Mean relative density at the deliverable's own
melt stop spans **0.6126 to 0.8147**, inside the 0.60 to 0.82 band
`DENSITY_OBJECTIVE_LIBRARY_REPORT.md` Section 1 measured. These are melt-stop
reads, not end-of-horizon reads, so nothing here is an oversinter claim.

### 6.3 The dose-match limit, quoted every time these counts are

**No arm is power matched.** COMPUTED: absorbed power at the deliverables' own
stops spans **365.2 to 862.5 W/m** across the 18-shape census, against the
500.0 W/m uniform calibration target. The permittivity channel moves absorbed
power in both directions: 0.73 times the uniform arm on the cross, 1.73 times on
the octagon. The objective penalizes over-melting as well as under-melting,
which removes the crudest dose gaming, but a J_phi or IoU comparison between two
arms is not a comparison at equal delivered energy.

**The part of the census that survives a dose caveat, stated explicitly.**
COMPUTED: **5 of 18 shapes beat the historical mask on J_phi while absorbing
EQUAL OR LESS power than that mask** (cross 365.2 against 397.9, rectangle
483.5 against 483.7, rounded_rect 612.0 against 639.9, square 540.8 against
559.8, trapezoid 567.6 against 571.3). Those five, which include the headline
cross and three of the four previously lost compact shapes, are not dose wins.
The other twelve absorb more than the mask they beat, and on **L_shape (488.2
against 228.4) and T_shape (480.5 against 227.9) the new arm absorbs 2.1 times
the power**, so those two wins should be read as partly dose until a
power-matched repeat is run.

---

## 7. Robustness spot-check on the new square and cross maps

Forward runs only, nothing re-solved, all in the permittivity channel the maps
were solved in. Protocol identical to `ms_robust.py` so the numbers are
comparable arm for arm. Code `adjoint2d/eps_robust.py`, raw
`out_eps/{square,cross}_robust.json`.

### 7.1 Rim robustness at grid 120

| shape | map | J at r = 0 | J at r = 1 | J at r = 2 | dJ at r = 1 | dJ at r = 2 | IoU r = 0 | IoU r = 1 | IoU r = 2 |
|---|---|---|---|---|---|---|---|---|---|
| square | permittivity channel | 8.39 | 8.78 | 10.05 | **+4.7 %** | +19.8 % | 1.0000 | **1.0000** | **1.0000** |
| square | conductivity only, filtered (`out_ms`) | 35.56 | 39.09 | 57.24 | +9.9 % | +61.0 % | 0.9681 | 0.9583 | 0.9395 |
| square | conductivity only, unfiltered (`out_lib`) | 25.66 | 64.84 | 88.38 | +152.7 % | +244.4 % | 0.9816 | 0.9547 | 0.9340 |
| cross | permittivity channel | 106.50 | 113.44 | 142.40 | **+6.5 %** | +33.7 % | 0.9079 | 0.8869 | 0.8171 |

**COMPUTED. The filter holds, and it holds better than it did in the
conductivity channel.** On the square a one-cell part-masked blur costs
**+4.7 percent** of J_phi against +9.9 percent for the filtered
conductivity-only map and +152.7 percent for the unfiltered one, and the IoU
does not move at all to four decimals out to a two-cell blur. On the cross the
one-cell cost is +6.5 percent with IoU falling 0.021 points. The two-cell blur
costs more on the cross (+33.7 percent, IoU down 0.091), which is expected: two
cells is 1 mm at this grid and the cross arms are narrow.

### 7.2 Grid hold-out, solve at 120 and score at 160

| shape | arm | J at 160 | IoU at 160 | IoU at 120 | P_abs W/m |
|---|---|---|---|---|---|
| square | uniform | 631.22 | 0.7711 | 0.8508 | 366.8 |
| square | best stored historical mask | 176.47 | **0.9261** | 0.9975 | 502.5 |
| square | conductivity only, unfiltered | 572.79 | 0.7767 | 0.9816 | 340.1 |
| square | conductivity only, filtered multi-start | 503.09 | 0.8063 | 0.9681 | 344.1 |
| square | **permittivity channel, 4 bpp** | **314.90** | **0.8697** | 1.0000 | 487.8 |
| square | permittivity channel, dose matched | **296.90** | **0.8772** | | 664.9 |
| cross | uniform | 734.47 | 0.6279 | 0.5465 | 456.3 |
| cross | **permittivity channel, 4 bpp** | **545.88** | **0.7432** | 0.9079 | 549.0 |
| cross | permittivity channel, continuous | 522.14 | 0.7482 | | 572.5 |

The square dose-matched arm uses the drive voltage `out_robust/square_grid.json`
calibrated so the UNIFORM arm at 160 absorbs 500.0 W/m (2834.8 V). **The cross
has no stored 160 dose calibration and that arm was SKIPPED LOUDLY** rather than
calibrated on the fly with a different protocol.

**COMPUTED, and the honest reading is a large improvement that still does not
close the gap.** The square transfers far better than any previous solved arm
(J 572.79 to 503.09 to **314.90**; IoU 0.7767 to 0.8063 to **0.8697**), but its
grid-120 IoU of 1.0000 still collapses to 0.8697 at 160 and it still loses to
the historical mask at 160 (0.9261). The cross improves over uniform at 160
(0.7432 against 0.6279) but drops 0.16 IoU points from its grid-120 value.
**The limit named in `SOLVE_ROBUSTNESS_VALIDATION.md` still binds:** the uniform
arm alone moves up to 0.13 IoU points between the grids and in both directions,
so a grid hold-out remains a joint test of map transfer AND forward
discretization convergence, and this pass cannot separate them either. The
decisive experiment is still a re-solve at 160.

---

## 8. Cost and wall time, with the projection that was required

**Logged after two shapes and projected, as required.** The first two solve
completions were star6 at 202 s and star at 192 s, mean 197 s; the first round
of six spanned 188 to 405 s, mean 268 s, which projected to 18 shapes in 6
streams as 3 rounds of about 270 s, roughly 14 to 18 minutes. **Actual: 15
minutes 52 seconds of wall clock** (06:06:59 to 06:22:51), 4928 s of process
time, per shape 188 to 405 s. **Nothing was dropped and no shape was cut.**

| stage | runs | wall |
|---|---|---|
| unit tests, full `adjoint2d` suite | 173 tests | 64 s |
| stored production-run identity gate, 4 runs | 4 | 8 min |
| FD gate, square and cross, single threaded, in parallel | 2 | 976 s and 617 s |
| FD bisect and refined sweep, square and cross | 2 | 552 s and 290 s |
| 18 solves, 2 full-depth starts each | 36 solves | 4928 s of process time, 16 min of wall clock in 6 streams |
| robustness, square and cross | 2 | 52 s and 31 s |
| figures, including 6 per-shape re-runs | 10 figures | about 4 min |

Total real compute for this report is about **2.6 hours of process time** and
about **50 minutes of wall clock**. Every numerical library was pinned to one
thread (`fgm_solve_campaign/env1.sh`) and parallelism taken across shapes.

---

## 9. Proven, computed, assumed

**PROVEN**
* The EQS vector-Jacobian product in permittivity reproduces a brute-force
  central difference to better than 1e-7 relative on a small grid.
* Asking for the permittivity channel leaves the conductivity sensitivity
  bit-identical (`np.array_equal`), at the operator level and at the full
  `gradient` level on the real engine.
* The permittivity channel is identically zero at omega = 0, which is the
  chain-rule factor `d gamma / d eps_r = 1j*omega*EPS0`.
* The prototype forward in the permittivity channel is BIT-IDENTICAL to three
  archived production runs, maximum absolute difference 0.000e+00 in the stored
  `T_phi90` field.
* The filtered permittivity gradient is exactly the unfiltered permittivity
  gradient composed with the filter transpose: the dot-product identity holds to
  3.00e-16 (square) and 2.86e-16 (cross).
* 173 unit tests pass; the 6 new ones were red before green, the red being the
  two `TypeError`s named in Section 2.1.

**COMPUTED**
* Every number in Sections 1 and 3 through 8.
* The uniform arm is channel invariant, maximum absolute difference 0.000e+00 in
  J_phi over 18 shapes.
* The permittivity term contributes 0.646 to 1.537 of the conductivity gradient
  norm, with cosine 0.55 to 0.84 between the channels.
* The J_phi evaluation floor, 5.95e-12 to 4.87e-10 absolute.
* The J_phi argmin does not move under any probe perturbation at epsilon 1e-3.
* Zero energy-residual gate violations; maximum residual 2.21 percent.

**ASSUMED, and how it bites**
1. **That sigma = 1.5 cells is the right physical design length.** No bench
   measurement of the real rim blur exists. Carried unchanged, and Section 7.1
   shows the choice is still load bearing.
2. **That the historical maps' permittivity co-variation is the physically right
   actuator model.** It is what the production `saturation_map_npz` hook does
   (`rfam_eqs_coupled.py:2527-2532`) and what the whole historical campaign was
   scored in, but whether a real binder saturation moves eps_r and sigma in
   exactly that proportion is a materials question this pass did not test. The
   two-sided per-node hook deliberately pins eps_r for a reason
   (`rfam_eqs_coupled.py:290-292`, "Allison's part eps_r = 20 is fixed; only
   sigma varies"), and if that pinning is the physically correct one then the
   permittivity channel is not printable and this census is a statement about
   the model, not about a machine. **This is the largest open question in the
   pass and it is a physics question, not a numerics one.**
3. **That an arbitrary stop time is realizable as a process control.**
4. **That the rasterized binary part mask is the right nominal target.**
5. **No arm is power matched**, 365.2 to 862.5 W/m; Section 6.3 separates the
   five wins that survive the caveat from the twelve that do not.
6. **Two shapes were gated, not eighteen.** The other 16 solves inherit that
   gate.
7. **The forward is the two-dimensional `adjoint2d` engine, not `heatr3d`**, and
   every simplification documented for the shape-library solve applies unchanged
   (no through-thickness physics, no rotation, no experimental validation).
8. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). An IoU
   of 1.0000 is a statement about the model at grid 120 in two dimensions, not
   about a printed part.

---

## 10. Honest limits

1. **The gate does not reach 1e-6 on most probes** (3 of 10 across both shapes
   on the filtered layer). It clears the campaign's 1e-5 subgradient standard on
   10 of 10 after the refined sweep, and the conductivity-only control at the
   same read state clears it on only 2 of 5 on the square, so the residual is
   the forward and not the new term, but it is a weaker gate than a smooth
   objective would give.
2. **star6 got WORSE with the better actuator**, 72.42 to 84.55 on J_phi and
   0.8742 to 0.8395 on IoU, and it is the library's only genuine loss. No
   diagnosis was attempted. The per-shape figure shows the failure is growth
   into the two side notches.
3. **hexagon regresses at matched budget** (24.24 to 42.57 from the cold start)
   and is only rescued by the warm start. Its cold start is finding a worse
   local minimum in the new channel, which the budget-matched arm exposes.
4. **The extra budget is real.** The best-of-two arm cost 80 forward-equivalents
   per shape. The budget-matched arm gives the identical class ladder and
   identical census counts, so nothing headline rests on it, but every summed
   J_phi from the best-of-two arm carries the 80.
5. **The grid-120 fidelity still does not transfer.** The square goes 1.0000 to
   0.8697 between grids and still loses to the historical mask at 160.
6. **Robustness was probed on two shapes only**, square and cross, and only on
   the best-of-two maps.
7. **The permittivity channel raises absorbed power on 14 of 18 shapes**, up to
   1.73 times the uniform calibration. Section 6.3 quantifies which wins survive
   that.
8. **gt_logo was skipped** (no `cv2`), so the library is 18, not 19.

---

## 11. The single most valuable next layer

**Settle whether the permittivity co-variation is printable.** Everything in
this report rests on assumption 2. The production code contains BOTH hooks and a
comment asserting that the doped part's eps_r is fixed at 20 while only sigma
varies. If that comment is right for the real binder, then the census above is a
model result with no deployment path and the conductivity-only census is the
deployable one. If it is wrong, this pass is the largest single improvement the
solve has ever had. That is one literature or bench question, not a compute job,
and it gates the value of everything below.

**Second, and cheap: power-match the twelve wins that are not yet dose clean.**
Recalibrate the drive so each arm absorbs the same energy as the historical mask
it is compared against, then re-score. That is 12 forward runs and it converts
"beats the oracle on 17 of 18" into a claim that survives the dose caveat.
Section 6.3 already shows 5 of 18 survive it today.

**Third: diagnose star6 and the hexagon cold start.** Both are single-shape
failures in an otherwise uniform result, and a single-shape failure with a
proven gradient is usually a start or a local-minimum story, which a
three-start run at the same full depth would settle in about 10 minutes.

---

## 12. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

New code, all under `fgm_solve_campaign/`:
* `adjoint2d/gate_eps.py` the finite-difference gate, the conductivity-only
  control layer, the channel-split diagnostic and the stop-index stability check
* `adjoint2d/gate_eps_bisect.py` the permittivity-channel filter transpose
  identity and the refined epsilon sweep
* `adjoint2d/eps_solve.py` the per-shape driver, two full-depth starts, the
  budget accounting and the verdict
* `adjoint2d/eps_robust.py` the rim and grid probes in the permittivity channel
* `adjoint2d/build_eps_tables.py` the census tables
* `adjoint2d/make_eps_figures.py` the figures
* `adjoint2d/tests/test_eps_channel.py` 6 tests, red first
* `run_eps_stream.sh`

Modified:
* `adjoint2d/adjoint.py` only: `eqs_vjp` gained `with_eps`, `gradient` gained
  `eps_covary`. Both default to the previous behaviour and the flag-off path is
  bit-identity tested.

Unmodified but read and imported: `forward.py`, `eqs.py`, `shape_objective.py`,
`design_filter.py`, `printability.py`, `robust.py`, `library_solve.py`,
`ms_solve.py`, `multistart.py`, `verify_hist.py`, `control_eps.py`.

Results:
* `fgm_solve_campaign/out_eps/<shape>.json` and `<shape>_maps.npz`, 18 shapes
* `fgm_solve_campaign/out_eps/{square,cross}_robust.json` and
  `_robust_maps.npz`
* `fgm_solve_campaign/out_eps/gate_eps_{square,cross}.json`
* `fgm_solve_campaign/out_eps/gate_eps_bisect_{square,cross}.json`
* `fgm_solve_campaign/out_eps/stored_run_gate_eps.json`
* `fgm_solve_campaign/out_eps/_tables.md`
* `fgm_solve_campaign/logs_eps/*.log`

Figures, **all ten viewed before delivery**:
* `fgm_solve_campaign/figs_eps/fig_eps_census.png` the de-confounded census,
  J_phi, IoU, and the percent removed against the historical mask in both
  channels
* `fgm_solve_campaign/figs_eps/fig_eps_channel.png` what the channel bought at
  matched budget, the finite-difference sweeps, and the size of the new term
* `fgm_solve_campaign/figs_eps/fig_eps_maps.png` the 18 delivered 4-bpp maps
* `fgm_solve_campaign/figs_eps/fig_eps_robust.png` rim and grid on square and
  cross
* `fgm_solve_campaign/figs_eps/fig_eps_{cross,square,rectangle,rounded_rect,star6,diamond}.png`
  per-shape movers: the dopant map and the melted region against the nominal
  part, for the new arm, the historical mask and the conductivity-only arm

Read, not modified: `OVERNIGHT_QUEUE_2026-08-01.md`, `MULTISTART_REPORT.md`,
`VERIFICATION_PRINTABILITY_REPORT.md`, `SOLVE_ROBUSTNESS_VALIDATION.md`,
`SHAPE_LIBRARY_SOLVE_REPORT.md`, `DENSITY_OBJECTIVE_LIBRARY_REPORT.md`,
`fgm_solve_campaign/out_lib/*`, `fgm_solve_campaign/out_ms/*`,
`fgm_solve_campaign/out_robust/*`, `outputs_eqs/geometry_dual_readstate/runs/*`,
`outputs_eqs/fgm_calibrated_control/*`, `rfam_eqs_coupled.py`.
