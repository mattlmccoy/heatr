# Adjoint prototype on the trusted 2-D engine: verdict, gates, and what it cost

**Date:** 2026-07-31. **Scope:** step 3 of the solve workstream, building the
prototype designed in Section 7 of `FGM_INVERSE_DESIGN_ASSESSMENT.md` and running
the head-to-head it specifies against the step-2 gain-calibrated control
(`FGM_CALIBRATED_CONTROL_REPORT.md`). Nothing was committed. No dissertation file
was touched. All work is in the git worktree
`.claude/worktrees/agent-a02efc1141ba69c58`.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the 2-D solver uses). FD = finite
difference. IFT = implicit function theorem. VJP = vector-Jacobian product (the
reverse-mode building block). L-BFGS-B = limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints. CSR = compressed sparse
row. ULP = unit in the last place (the spacing between adjacent floating-point
numbers). sigma_T = the 2-D campaign uniformity metric
`ui_rms_part * (T_bar_part - 23 C)`, in deg C, always quoted with its read
state. phi_bar = mean part melt fraction. dpi = dots per inch.

**Evidence tags.** PROVEN = unit-tested, bit-identity-gated or FD-gated.
COMPUTED = measured from a real solve in this campaign or read from a stored
artifact. ASSUMED = a modelling choice or an inference not measured here.

---

## 1. Verdict

**1. The adjoint direction beats the gain-calibrated proportional map on the
hold-out read state at the largest budget on all three shapes, when both arms
use the same actuator, and it loses at the two smaller budgets on two of three
shapes.** COMPUTED, hold-out melt-onset sigma_T versus the matched-actuator
control H1 at 40 forward-solve equivalents: square 4.193 C against 6.936 C,
triangle 5.575 C against 8.511 C, cross 24.403 C against 29.970 C. At 5 and 15
equivalents the control wins on triangle (8.52 and 8.51 C against 16.50 and
13.04 C) and on cross at 15 (29.97 against 32.57 C). The crossover is between
15 and 40 forward-solve equivalents. This is the answer to assessment question
Q2 and it is budget-dependent, not a clean win.

**2. Against the STRONGEST control the answer is shape-dependent and the adjoint
does not sweep it.** The step-2 control ran in the permittivity-co-varying
injection channel, which the two-sided per-node hook the adjoint uses pins. Put
that channel on the table as arm H1eps and, at 40 equivalents: on square H1eps
reaches 3.475 C against the adjoint's 4.193 C (hold-out-fair arm) and 3.380 C
(melt-onset-fit arm); on cross H1eps reaches 14.396 C against 24.403 C; on
triangle the adjoint wins outright, 5.575 C against 19.768 C. **Two of three
shapes go to a control with a better actuator.** Naming that honestly matters
more than the score: the adjoint here is sigma-only by design, and permittivity
co-variation is worth more than the search direction on two of these shapes.

**3. The "direction-failure class" does not survive scrutiny. It is a
line-search basin artifact, not a wrong direction.** This is the most consequential
finding in this report. COMPUTED on a dense 26-point gain scan of the triangle
under step-2's own conventions (`out_adjoint/gaincurve/triangle_gaincurve_outside0_eps1.json`):
the fit metric is **bimodal** in the gain, rising from 40.45 C at m = 0.05 to a
local maximum of **48.82 C at m = 0.638**, then falling monotonically to
**23.01 C at m = 2.40**. The step-2 line search was warm-started from the stored
four-point grid {0.30, 0.50, 0.70, 0.85}, which lies entirely on the wrong side
of that local maximum (fits 45.77, 48.34, 48.36, 46.14), so it walked to the
domain floor and reported +79.8 % worse than uniform. An unseeded
golden-section over the **same** pre-registered domain [0.05, 2.50] finds
m = 2.39 and lands at **-5.1 % better than uniform**. The step-2 conclusion
"triangle is a direction failure" should be withdrawn and replaced with "the
step-2 triangle gain search was trapped by its warm start".

**4. Everything is built on a forward proven BIT-IDENTICAL to the production
engine.** PROVEN, four independent gates, `max|diff| = 0.000e+00` on every one:
all four part-history series at every one of 1500 outer steps and all four final
fields, on square, triangle and cross through the two-sided per-node hook, and on
square through the permittivity-co-varying hook. Reaching zero required
reproducing two floating-point details of the production code exactly (Section 3).

**5. Three further independent anchors, all exact.** COMPUTED: the prototype's
uniform arms reproduce the published step-2 uniform baselines to every digit
reported (square 4.931, triangle 23.043, cross 43.771 C at melt onset), and
driving the prototype with the three stored step-2 map artifacts reproduces the
published calibrated numbers to every digit reported (square 3.669, triangle
41.423, cross 13.858 C). The prototype is not a look-alike of the campaign; it
is the campaign.

**6. FD gates: three of four layers pass cleanly, the moving read state does
not.** L1 (fixed horizon), L1b (heating peak) and the clip-active layer all pass
below 1e-6 on both probe directions with the expected V-shaped error curve. L2,
the melt-onset objective with the IFT term dt*/ds, passes on the single-cell
probe on square (5.54e-07) but **fails** the random-direction probe (8.84e-06),
and fails both on triangle (1.99e-06, 1.65e-06). The cause is localized and
measured, not guessed: at the production phase-change regularizer width
dt_pc_c = 10 C, **65.7 % of part cells are pinned at melt fraction 1** at the
read state, and the IFT term multiplies any non-smoothness in phi_bar by
1/slope, about 1000 here. Widening the regularizer to 80 C drops the pinned
fraction to 3.8 % and the gate to **1.10e-08** with a textbook V. The term is
implemented correctly; the objective is genuinely non-smooth at the production
width and the gradient there is a subgradient.

---

## 2. What was built

All under `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/.claude/worktrees/agent-a02efc1141ba69c58/adjoint2d/`.

| module | contents |
|---|---|
| `prod.py` | import shim that binds the prototype to the SAME `rfam_eqs_coupled.py` the campaigns used |
| `pins.py` | configuration pins, with a loud `UnsupportedConfig` for every production branch not replicated |
| `eqs.py` | vectorized EQS assembly, forward solve, transposed solve reusing the forward factorization |
| `gradops.py` | sparse matrices for the `np.gradient(..., edge_order=1)` stencils and their transposes |
| `forward.py` | the differentiable forward march and the substep cache the reverse sweep consumes |
| `objective.py` | sigma_T, the read states, the melt-onset bracket and the seed gradients |
| `adjoint.py` | the substep VJP, the reverse outer march, and the EQS adjoint |
| `control.py` | the proportional-inverse map, golden-section search, selection rules, budget arithmetic |
| `control_eps.py` | the same control in the permittivity-co-varying channel, plus stored-map reproduction |
| `gate_l0.py`, `gate_fd.py` | the identity gate and the finite-difference gates |
| `head_to_head.py`, `bisect_control.py`, `gain_curve.py`, `build_tables.py` | experiments and reporting |

Twenty-four unit tests, all passing
(`./.venv312/bin/python -m pytest adjoint2d/tests -q`). Red-first discipline was
followed and captured for the EQS assembly (import error, then a real one-ULP
assertion failure at 6.70e-13, then green) and for the substep identity
(three failures at 9.88e-15, then green). For `control.py`, `objective.py`,
`gradops.py` and `adjoint2d/tests/test_adjoint_pieces.py` the tests were written
alongside or after the module; that is a discipline gap and it is stated here
rather than glossed. The numerical gates below are the substantive verification.

---

## 3. L0, the bit-identity gate

The precedent is the meshcheck work: prove `max|diff| = 0.000e+00` against the
production engine before drawing any conclusion. What is compared, at the full
pinned 1500-step horizon with a deliberately non-uniform, non-symmetric
saturation map: every outer step of mean part temperature, part uniformity index
root-mean-square, mean part melt fraction and mean part relative density, plus
the final T, phi, rho and Q_rf fields.

| gate | injection channel | part cells | worst max abs diff | production wall s | prototype wall s | speedup |
|---|---|---|---|---|---|---|
| square | two-sided per-node (permittivity pinned) | 1600 | **0.000e+00** | 44.6 | 5.87 | 7.6 x |
| triangle | two-sided per-node | 800 | **0.000e+00** | 42.1 | 5.05 | 8.3 x |
| cross | two-sided per-node | 1036 | **0.000e+00** | 43.0 | 5.19 | 8.3 x |
| square | `saturation_map_npz`, permittivity co-varying | 1600 | **0.000e+00** | 43.7 | 5.23 | 8.4 x |

Artifacts: `out_adjoint/l0_{square,triangle,cross}_full.json`,
`out_adjoint/l0_square_epscovary.json`. An earlier run on an idle-versus-loaded
machine measured the square gate at 206.9 s production against 9.76 s prototype,
21.2 x; the table above is from the final sweep with all runs on the same load.

**Two floating-point details had to be reproduced exactly to reach zero, and
both are worth recording as gotchas.**

1. **NumPy complex128 scalar multiply and array multiply do not agree
   bit-for-bit.** The production assembly loop multiplies numpy complex scalars
   one face at a time; a vectorized `a * b` on arrays differs by one ULP on
   about half of random pairs (measured: 2363 of 5000, `scratch_probe3.py`).
   `eqs.cmul` therefore evaluates the naive formula
   `(ar*br - ai*bi) + 1j*(ar*bi + ai*br)` with each product rounded separately,
   which matches the scalar path exactly. Complex DIVISION agrees between the
   two paths; only multiplication differs.
2. **The production conductivity array is float32, not float64, whenever the FGM
   hook is enabled.** `_FgmFeedback.effective_fill` casts to float32
   (`rfam_eqs_coupled.py:416`), so the startup expression
   `sigma = sigma_v + eff_fill*(sigma_d0 - sigma_v)` inherits float32 under the
   NumPy 2 weak-scalar rules, and the in-place state-B assignment
   `sigma[part_mask] = ...` is then rounded to float32 as well. Missing this
   left a 2.4 W per cubic metre residual in Q_rf and a 1.3e-06 C residual in the
   final temperature field. Identity mode reproduces the rounding; the
   differentiable mode keeps float64 so the design variable enters smoothly.

The third bug found this way was a wrong gas constant: the prototype used
8.314462618 against the production 8.31446261815324, which showed up as a
9.9e-15 densification residual. The constants are now imported from the
production module so they cannot drift.

**Independent anchors, both exact.** COMPUTED:

| check | prototype | published step 2 |
|---|---|---|
| square uniform, melt-onset sigma_T | 4.931 C | 4.931 C |
| triangle uniform, melt-onset sigma_T | 23.043 C | 23.043 C |
| cross uniform, melt-onset sigma_T | 43.771 C | 43.771 C |
| square stored map, melt onset / heating peak / absorbed power | 3.669 C / 6.923 C / 564.8 W per m | 3.669 / 6.923 / 564.8 |
| triangle stored map | 41.423 C / 41.360 C / 604.0 W per m | 41.423 / 41.360 / 604.0 |
| cross stored map | 13.858 C / 15.562 C / 397.9 W per m | 13.858 / 15.562 / 397.9 |

---

## 4. The adjoint, and every FD gate value

### 4.1 Structure

The pinned configuration has zero conductivity temperature and density
coefficients, so the conductivity field depends only on the design variable and
exactly TWO electrical states exist in the whole march: the startup assembly
used for outer steps 0 to 19, and the re-assembly used from step 20 to the end.
The production engine recomputes the second one on every twentieth step, which
reproduces the same numbers; the prototype computes it once. That caching is
covered by the L0 gate.

Reverse mode therefore has three pieces:

1. **Substep VJP**, coupled in (T, rho), with every clip and cap handled as a
   subgradient: the temperature-step cap, the temperature clamp, the phase-ramp
   clip at both ends, the densification per-step cap, the density box, the
   viscosity floor and the heat-capacity floor. The latent-heat term
   `L * dphi_dT` is piecewise constant in T and contributes zero subgradient.
2. **Reverse outer march** with checkpointing: temperature and density are
   stored at each outer-step boundary, each outer step's five substeps are
   recomputed from its checkpoint, and dJ/dQ_rf is accumulated separately for
   the two electrical states.
3. **EQS adjoint.** For a real functional of the complex potential, writing
   `dJ = 2 Re(p^T dV)` gives `A^T lambda = p` with the plain transpose, solved
   with the factorization of the SAME assembled matrix the forward used. The
   harmonic face conductances are then differentiated analytically,
   `dgf/dg_k = 2 g_n^2/(g_k+g_n)^2`. Complex arithmetic appears only in V, gamma
   and A; Q_rf, T, rho, phi and the objective are real throughout.

The absorbed-power renormalization is NOT in this layer: `enforce_generator_power`
is false in the pinned configuration and `pins.py` raises if it is true, so the
dense rank-one term the 3-D work derives is absent by construction, not by
omission.

### 4.2 Gate values, square, 120 x 120, 1600 design variables

Central differences, epsilon swept 1e-3 to 1e-7, two probes: a random unit
direction over the part cells and a single-cell probe at the maximum-|gradient|
cell. Pre-registered pass criterion: best relative error below 1e-6.
Artifact `out_adjoint/fd_gate_square.json`.

| layer | what it adds | random direction | single cell | verdict |
|---|---|---|---|---|
| L1 fixed horizon (outer step 163) | sigma_T at a fixed exposure horizon | **4.285e-08** (eps 1e-3) | **2.293e-09** (eps 1e-4) | PASS |
| L1b heating peak | the max over pre-melt steps, taken as a subgradient at the argmax | **4.285e-08** | **1.902e-08** (eps 1e-6) | PASS |
| L2 melt onset with IFT | the moving read state t*(s) and dt*/ds | 8.838e-06 (eps 1e-5) | 5.542e-07 (eps 1e-7) | **FAIL** on the random probe |
| L2clip, cap 0.02 C, 11.11 % of cells clipped | forces the temperature-step clip ACTIVE | **8.436e-08** (eps 1e-4) | **7.317e-08** (eps 1e-5) | PASS |
| L2clip, cap 0.01 C | fully clipped | 0.0 | 0.0 | degenerate |

The cap 0.01 C row is a consistency check, not a gate: the limiter binds
everywhere, the objective becomes independent of the design variable, and both
the analytic gradient and the finite difference are exactly zero. The cap 0.02 C
row is the substantive clip gate, with 11.11 % of cells clipped and a gradient
of magnitude 6.42e-03.

### 4.3 Gate values, triangle, 800 design variables

Artifact `out_adjoint/fd_gate_triangle.json`.

| layer | random direction | single cell | verdict |
|---|---|---|---|
| L1 fixed horizon (outer step 148) | **8.338e-07** | **1.101e-07** | PASS |
| L1b heating peak | **8.338e-07** | **1.101e-07** | PASS |
| L2 melt onset with IFT | 1.985e-06 | 1.651e-06 | **FAIL** |
| L2clip, cap 0.02 C, 5.56 % clipped | **1.244e-07** | **7.318e-09** | PASS |

### 4.4 Why L2 fails, measured

The melt-onset objective interpolates sigma_T to the exact crossing
phi_bar(t*) = 0.90 between bracketing outer steps, which is the discrete form of
`dt*/ds = -(d phi_bar/ds)/(d phi_bar/dt)`. The two theta sensitivities in
`objective.objective_melt_onset_interp` are exactly that expression, and they
are unit-tested against a direct derivative.

The failure is the objective, not the term. Widening the phase-change
regularizer and re-gating the single-cell probe
(`out_adjoint/fd_gate_l2_regularizer_sweep.txt`):

| dt_pc_c, deg C | part cells pinned at phi = 1 | phi_bar slope per outer step | best relative error | shape of the epsilon sweep |
|---|---|---|---|---|
| 10 (production) | 65.7 % | 0.001016 | 5.54e-07 | no V; only the smallest epsilon is right |
| 20 | 28.3 % | 0.001494 | 2.97e-07 | V forming |
| 40 | 15.6 % | 0.001174 | 6.91e-08 | clean V from 1e-5 |
| 80 | 3.8 % | 0.000803 | 1.10e-08 | clean V |

Mechanism: every part cell pinned at phi = 1 contributes zero to d phi_bar/ds,
and its pinning status flips as the design variable moves. The IFT term
multiplies phi_bar by 1/slope, about 1000 at the production width, so each flip
becomes a visible jump in the gradient. The objective is C0 but not C1 at the
1e-3 scale in the design variable. Per the assessment's own prescription
(treat dt_pc_c as the regularizer and verify the gate as it changes), this is a
subgradient with a measured non-smoothness scale, and the optimizer arm that
uses it is labelled accordingly.

### 4.5 Standing non-smooth-term gates

COMPUTED on every solve in the head-to-head: temperature-step clip fraction
0.0000, temperature-clamp fraction 0.0000, and **the Q_rf cap fraction is
0.0000 on all 15 selected arms** (Table E). The Q_rf cap is asserted inactive,
as the assessment requires. The clip is dormant in production, which is exactly
why the clip subgradient was gated by forcing the cap down until it bound.

---

## 5. Cost, measured

`out_adjoint/h2h/<shape>.json`, field `cost`. One adjoint sweep, measured
wall-clock on this machine with thread oversubscription disabled:

| shape | part cells | forward, s | adjoint with the heating-peak objective, s | ratio | adjoint with the melt-onset objective, s | ratio |
|---|---|---|---|---|---|---|
| square | 1600 | 3.37 | 1.47 | **0.437** | 7.13 | **2.116** |
| triangle | 800 | 2.15 | 1.42 | **0.659** | 4.64 | **2.157** |
| cross | 1036 | 4.53 | 9.10 | **2.011** | 8.27 | **1.826** |

**The ratio is objective-dependent, and that is a real property, not noise.**
The reverse sweep only spans outer steps up to the read state. The heating peak
sits at outer step 163 on the square while melt onset sits at 845, so
differentiating the early read state is nearly five times cheaper. On the cross
the heating peak sits near the end of a 1500-step march and the ratio rises to
2.0. Charging both arms one ratio would have silently over-funded the
melt-onset arm; each arm is charged its own measured ratio. One
objective-plus-gradient evaluation therefore costs between 1.44 and 3.16
forward-solve equivalents depending on shape and objective.

Against the assessment's 3-D figure of about 2 forward-equivalents per gradient,
this 2-D measurement is 0.44 to 2.16, that is, the same order and often cheaper.

---

## 6. The head-to-head

Protocol frozen in `out_adjoint/PREREGISTRATION.md` before the first
optimization launch. Tables regenerated by `adjoint2d/build_tables.py` into
`out_adjoint/h2h/tables.md`.

### 6.1 Arms

| arm | map family | actuator | fit metric | selection |
|---|---|---|---|---|
| U | none, s = 1 | n/a | n/a | n/a |
| H1 | proportional inverse, gain by golden section on [0.05, 2.50] | conductivity only, same as the adjoint | heating-peak sigma_T | argmin fit among feasible |
| H1eps | same map | conductivity AND permittivity, the step-2 production channel | heating-peak sigma_T | argmin fit among feasible |
| H1eps0 | same map, saturation zero outside the part (the step-2 boundary convention) | conductivity and permittivity | heating-peak sigma_T | argmin fit among feasible |
| A1 | per-cell, L-BFGS-B on the FD-gated gradient, box [0, 1], start s = 1 | conductivity only | heating-peak sigma_T | argmin fit among feasible |
| A2 | same optimizer, fit on the interpolated melt-onset objective with the IFT term | conductivity only | melt-onset sigma_T | argmin hold-out among feasible, LABELLED in-sample |

A1 versus H1 is the hold-out-fair, matched-actuator pairing: same fit metric,
same selection rule, same actuator, nothing selected on the reported metric.
A2 is the arm that exercises the IFT layer and is in-sample on the reported read
state by construction; it is reported separately and never presented as a
hold-out result.

### 6.2 Hold-out melt-onset sigma_T, deg C

| shape | budget | uniform | H1 | H1eps | H1eps0 | A1 | A2 |
|---|---|---|---|---|---|---|---|
| square | 5 | 4.931 | 7.878 | 3.805 | 3.805 | 3.860 | 4.931 |
| square | 15 | 4.931 | 6.943 | 3.475 | 3.475 | 4.049 | 3.380 |
| square | 40 | 4.931 | 6.936 | 3.475 | 3.475 | **4.193** | **3.380** |
| triangle | 5 | 23.043 | 8.521 | 19.722 | 21.784 | 16.499 | 23.043 |
| triangle | 15 | 23.043 | 8.509 | 19.768 | 21.875 | 13.040 | 15.568 |
| triangle | 40 | 23.043 | 8.511 | 19.768 | 21.875 | **5.575** | 11.072 |
| cross | 5 | 43.771 | NOT_REACHED | 23.263 | 14.438 | 43.771 | 43.771 |
| cross | 15 | 43.771 | 29.972 | 14.428 | 13.513 | 32.566 | 31.216 |
| cross | 40 | 43.771 | 29.970 | 14.396 | 13.526 | **24.403** | **22.821** |

Percent versus uniform at the same read state (negative is better):

| shape | budget | H1 | H1eps | H1eps0 | A1 | A2 |
|---|---|---|---|---|---|---|
| square | 5 | +59.8 % | -22.8 % | -22.8 % | -21.7 % | +0.0 % |
| square | 15 | +40.8 % | -29.5 % | -29.5 % | -17.9 % | -31.5 % |
| square | 40 | +40.7 % | -29.5 % | -29.5 % | -15.0 % | -31.5 % |
| triangle | 5 | -63.0 % | -14.4 % | -5.5 % | -28.4 % | +0.0 % |
| triangle | 15 | -63.1 % | -14.2 % | -5.1 % | -43.4 % | -32.4 % |
| triangle | 40 | -63.1 % | -14.2 % | -5.1 % | **-75.8 %** | -51.9 % |
| cross | 5 | n/a | -46.9 % | -67.0 % | +0.0 % | +0.0 % |
| cross | 15 | -31.5 % | -67.0 % | -69.1 % | -25.6 % | -28.7 % |
| cross | 40 | -31.5 % | -67.1 % | -69.1 % | -44.2 % | -47.9 % |

### 6.3 Reading the table

- **Adjoint versus the matched-actuator control (A1 versus H1).** Adjoint wins
  at 40 equivalents on all three shapes. Control wins at 5 and 15 on triangle,
  and at 15 on cross. On square the adjoint wins at every budget because the
  conductivity-only control is actively harmful there.
- **The conductivity-only control is harmful on the square, by +40.7 %.** Its
  gain search runs to the domain floor m = 0.05, where the map is a uniform
  half-dose. With permittivity co-varying the same map family finds m = 0.92 and
  is -29.5 % better. **The permittivity channel is what makes the
  proportional-inverse map work on the square.** That is a direct measurement of
  the assessment's Section 5.3 blocker and it is larger than the search-direction
  effect on that shape.
- **A2 beats A1 on the reported metric on square and cross and loses on
  triangle.** A2 is in-sample, so this is an upper bound on what the IFT term
  buys, not a hold-out result. It does answer assessment question Q3
  directionally: including the moving read state changes the answer, by -19 % on
  square (4.193 to 3.380 C) and -6 % on cross, and it is not free, because the
  melt-onset objective costs about 2.1 forward-equivalents per gradient against
  0.44 for the heating peak, so the same budget buys fewer than half as many
  iterations (12 against 27 on the square).
- **The heating-peak fit metric is not dose-normalized, and a free per-cell
  optimizer exploits that.** Measured on the coarse smoke case and visible in the
  A1 trajectories: reducing dopant everywhere lowers the pre-melt temperature
  spread simply by heating less. Only the feasibility barrier keeps A1 inside
  the melting region. The melt-onset read state does not have this defect
  because it is read at a matched thermal state. This is a metric property worth
  carrying forward to any future optimizer arm.
- **The cross arms are read very close to the horizon.** COMPUTED, Table E: the
  melt-onset outer step is 1499 of 1500 for H1 and H1eps, 1486 for A1 and 1458
  for A2, against 1069 for uniform. Those numbers are legitimate crossings, not
  fallbacks, but they sit at the edge of the pinned horizon and should be treated
  as fragile.

### 6.4 Absorbed power, so the dose confound stays visible

The arms are voltage-driven and are NOT dose matched. Absorbed power at the
melt-onset read state, watts per metre of depth:

| shape | uniform | H1 (B = 40) | H1eps (B = 40) | A1 (B = 40) | A2 (B = 40) |
|---|---|---|---|---|---|
| square | 500.0 | 356.3 | 569.6 | 443.8 | 477.9 |
| triangle | 500.0 | 353.8 | 363.0 | 373.5 | 430.2 |
| cross | 500.0 | 342.6 | 283.0 | 306.1 | 326.3 |

The spread is 283 to 570 watts per metre from a 500 watt per metre baseline, a
factor of 2.0. No arm in this report is a pure redistribution result. The
adjoint arms sit closer to the baseline dose than the controls on square and
triangle and further from it on cross.

### 6.5 Budget accounting

Nested budgets, so the smaller answers are prefixes of the largest trajectory.
Evaluations that fit in 40 forward-solve equivalents, and equivalents actually
spent:

| shape | H1 gains | A1 gradient evaluations | A2 gradient evaluations | spent H1 | spent A1 | spent A2 |
|---|---|---|---|---|---|---|
| square | 39 | 27 | 12 | 40.00 | 38.81 | 37.40 |
| triangle | 39 | 24 | 12 | 40.00 | 39.82 | 37.88 |
| cross | 39 | 13 | 14 | 40.00 | 39.14 | 39.56 |

The control gets one forward for the proxy field plus one per gain. The adjoint
arms are charged `1 + r` per evaluation with the measured, per-objective r of
Section 5.

---

## 7. The step-2 control, reproduced and then bisected

### 7.1 The triangle gain curve is bimodal

COMPUTED, 26-point scan under step-2's conventions (permittivity co-varying,
saturation zero outside the part), `out_adjoint/gaincurve/triangle_gaincurve_outside0_eps1.json`:

| gain m | heating-peak sigma_T, C | melt-onset sigma_T, C | absorbed power, W per m |
|---|---|---|---|
| 0.050 | 40.45 | 40.51 | 606.6 |
| 0.344 | 45.77 | 45.83 | 603.6 |
| 0.540 | 48.34 | 48.41 | 598.0 |
| **0.638** | **48.82 (local maximum)** | 48.88 | 592.1 |
| 0.834 | 46.14 | 46.20 | 565.9 |
| 1.030 | 33.97 | 34.02 | 502.7 |
| 1.520 | 25.04 | 23.88 | 421.1 |
| 2.108 | 23.08 | 21.80 | 387.1 |
| **2.402** | **23.01 (global minimum)** | **21.88** | 380.9 |
| 2.500 | 23.04 | 21.94 | 379.0 |

Uniform is 24.83 C at the heating peak and 23.04 C at melt onset. The step-2
stored grid {0.30, 0.50, 0.70, 0.85} straddles the local maximum, so a line
search seeded there sees the objective increasing to its right and decreasing to
its left and walks to the floor. The same scan for the square and the cross is
**unimodal** (square minimum at m = 0.932, cross minimum at m = 1.128,
`out_adjoint/gaincurve/`), which is exactly why the step-2 gains are sensible on
those shapes and pathological on the triangle.

### 7.2 What each stage of the map pipeline is worth

COMPUTED, `out_adjoint/bisect/*_bisect.json`, all arms injected through the
permittivity-co-varying hook at the step-2 selected gain, melt-onset sigma_T:

| arm | square | triangle | cross |
|---|---|---|---|
| uniform | 4.931 | 23.043 | 43.771 |
| stored step-2 map (production loader) | **3.669** | **41.423** | **13.858** |
| prototype map, continuous, saturation 1 outside | 3.629 | 35.862 | 15.911 |
| prototype map, 4 bits per pixel | 3.316 | 35.797 | 15.528 |
| prototype map, printer-dpi round trip, saturation 0 outside | 3.744 | 41.293 | 13.773 |
| prototype map, printer-dpi round trip, saturation 1 outside | 3.744 | 35.799 | 15.574 |

Two conclusions, both COMPUTED:

1. **Quantization to 4 bits per pixel is not the driver anywhere** (it is even
   slightly beneficial on the square).
2. **The saturation value OUTSIDE the part is a large, previously unnamed
   effect on shapes with a rasterized diagonal boundary.** The production maps
   are essentially zero outside the part (triangle: mean 0.0000, maximum 0.012),
   and the triangle has 40 cells with `part_mask` False but geometry fill above
   zero (mean fill 0.188). Zeroing the saturation there strips those cells'
   permittivity contribution for the entire run, worth 41.29 C against 35.80 C on
   the triangle. The square has **zero** such cells, which is why the square is
   completely insensitive to the convention and why every square number in this
   report and in step 2 agrees.

The prototype's own convention is to hold the saturation at its nominal value 1
outside the part, so that an arm changes the dopant map and nothing else. That
convention is what makes the uniform arms reproduce the published baselines
exactly. It is stated here because it differs from what the production map
generator emits.

---

## 8. Proven, computed, assumed

**PROVEN**
- The prototype forward is bit-identical to `rfam_eqs_coupled.run_sim` on the
  pinned configuration, `max|diff| = 0.000e+00`, on three shapes and both
  injection channels, over 1500 outer steps and 7500 thermal substeps.
- The vectorized EQS assembly reproduces the production potential exactly, and
  the transposed solve is the transpose (residual below 1e-8 relative).
- The diffusion VJP is the exact transpose of the bilinear diffusion operator
  (dot-product test to 1e-9 relative).
- The EQS VJP reproduces a brute-force finite difference of the absorbed-power
  field with respect to conductivity on a small grid (below 1e-7 relative).
- L1, L1b and the clip-active layer are FD-gated below 1e-6 on both probe
  directions on two shapes, with V-shaped epsilon sweeps.
- Twenty-four unit tests on the pure logic (assembly, gradient stencils, substep
  identity, read states, seeds, IFT coefficients, line search, selection rules,
  budget arithmetic).

**COMPUTED**
- Every number in Sections 3, 5, 6 and 7.
- The L2 gate at the production regularizer width is 5.54e-07 on the square
  single-cell probe and 8.84e-06 on the random probe; it improves monotonically
  to 1.10e-08 as the regularizer widens and the pinned-cell fraction falls from
  65.7 % to 3.8 %.
- The triangle fit metric is bimodal in the gain with a local maximum at
  m = 0.638 and a global minimum at m = 2.402; the square and the cross are
  unimodal.
- Adjoint cost 0.44 to 2.16 forward-solve equivalents per gradient, depending on
  shape and on which read state is differentiated.

**ASSUMED**
- That the interpolated melt-onset crossing is the right differentiable stand-in
  for the production step-indexed read state. It is
  continuous and agrees at the crossing, but it is not the same functional; the
  scoring in Section 6 always uses the production convention.
- That holding the saturation at 1 outside the part is the correct design-domain
  convention. It is defensible (the dopant map should not change the geometry)
  and it reproduces the published baselines, but the production map generator
  does something different and no experiment settles which is physical.
- That 26 points is enough to characterize a gain curve. The triangle bimodality
  is unmistakable at that resolution; a narrower second basin elsewhere could
  still be missed.
- That the two-shape FD gate coverage (square, triangle) transfers to the cross.
  The cross gradient was used in optimization without its own gate.

---

## 9. Honest limits

1. **Single grid.** Everything is 120 x 120. The 2-D mesh-convergence risk named
   in `FGM_INVERSE_DESIGN_ASSESSMENT.md` Section 3.2 is untouched. The grid
   hold-out the assessment specifies in Section 7.4 was NOT run.
2. **L2 does not gate cleanly at the production regularizer width.** The A2 arm
   therefore optimizes on a subgradient of a C0 objective. Its results are
   reported and are internally consistent, but the assessment's own pass bar is
   not met for that layer at that width.
3. **Three shapes, not nineteen.** The bimodality finding is measured on one
   shape and the absence of bimodality on two. Whether the other three step-2
   harmful shapes (H_shape, trapezoid, T_shape) are the same artifact is
   UNTESTED and is the obvious next check.
4. **The adjoint arm is conductivity-only.** The assessment's L5 layer
   (permittivity co-varying inside the gradient) was NOT built. The forward
   supports the channel and is L0-gated in it, but no gradient flows through
   permittivity, so on two of three shapes the adjoint is fighting with a weaker
   actuator than the control it is compared against.
5. **Not dose matched.** Section 6.4.
6. **In-sample labelling.** A2 selects on the metric it reports. It is labelled
   in-sample everywhere it appears and must never be quoted as a hold-out result.
7. **L-BFGS-B stalls.** On several shapes the last several evaluations repeat
   nearly identical points, which is a line-search failure on a non-smooth
   objective rather than convergence. The adjoint numbers at 40 equivalents are
   therefore lower bounds on what the direction can do, not converged optima.
8. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). Every
   corrected-arm number here is an upper bound on what a real part would show.

---

## 10. The single most valuable next layer

**Re-run the step-2 gain calibration on all nineteen shapes with an UNSEEDED
line search over the same domain.** It costs about 8 forwards per shape, roughly
25 minutes total on the prototype forward at 2 to 5 seconds per solve, and it
directly tests whether the entire step-2 harmful class (triangle, H_shape,
trapezoid, T_shape) is the same warm-start artifact the triangle turned out to
be. If it is, the assessment's central premise (that there is a residual
direction-failure class that only a solve can fix) needs revision, and the case
for the adjoint rests on peak quality rather than on rescuing failures. If it is
not, the surviving shapes are the correct target for the solve and the head-to-head
should be repeated on them.

Second priority, and cheap now that the forward is 8 times faster than
production: the **grid hold-out** (optimize at 120 x 120, score at 160 x 160) on
the baseline arms, which closes the untested 2-D mesh-convergence hole and costs
about 4 solves per shape.

---

## 11. Artifacts, absolute paths

Worktree root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/.claude/worktrees/agent-a02efc1141ba69c58`

- Code: `adjoint2d/` (13 modules), tests `adjoint2d/tests/` (24 tests)
- Pre-registration: `out_adjoint/PREREGISTRATION.md`
- L0 identity gates: `out_adjoint/l0_square_full.json`,
  `out_adjoint/l0_triangle_full.json`, `out_adjoint/l0_cross_full.json`,
  `out_adjoint/l0_square_epscovary.json`
- FD gates: `out_adjoint/fd_gate_square.json`, `out_adjoint/fd_gate_triangle.json`,
  `out_adjoint/fd_gate_l2_regularizer_sweep.txt`
- Head-to-head: `out_adjoint/h2h/{square,triangle,cross}.json`,
  `out_adjoint/h2h/{shape}_H1eps.json`, `out_adjoint/h2h/{shape}_H1eps0.json`,
  `out_adjoint/h2h/{shape}_maps.npz`, `out_adjoint/h2h/tables.md`
- Pipeline bisection: `out_adjoint/bisect/{square,triangle,cross}_bisect.json`
- Gain curves: `out_adjoint/gaincurve/`
- Run logs: `out_adjoint/{final_run,phase2,phase3,gaincurve_sq_cross}.log`
- Runners: `run_final.sh`, `run_phase2.sh`, `run_phase3.sh`, `run_gaincurves.sh`
