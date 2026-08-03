# Solving for the ideal dopant map under a geometric-fidelity objective

**Date:** 2026-07-31. **Scope:** re-target of the 2-D adjoint prototype from the
temperature-uniformity proxy to the true goal, that the melted region at the
process stop should match the nominal part shape. Nothing was committed. No
dissertation file was touched. All work is in the git worktree
`.claude/worktrees/agent-a02efc1141ba69c58`. The bit-identity gate, the adjoint
machinery and the cost accounting are carried over unchanged from
`ADJOINT_PROTOTYPE_REPORT.md` in the same worktree.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the 2-D solver uses). FD = finite
difference. IoU = intersection over union. IFT = implicit function theorem.
L-BFGS-B = limited-memory Broyden-Fletcher-Goldfarb-Shanno with box
constraints. VJP = vector-Jacobian product. sigma_T = the old uniformity metric
`ui_rms_part * (T_bar_part - 23 C)`, in deg C, reported here as a DIAGNOSTIC
only. phi = melt fraction. phi_bar = mean part melt fraction.

**Evidence tags.** PROVEN = unit-tested or bit-identity-gated. COMPUTED =
measured from a real solve in this campaign. ASSUMED = a modelling choice or an
inference not measured here.

**The objective.**

    J(s, t_stop) = sum over the WHOLE domain of w(x) * (phi(x, t_stop) - chi(x))^2

with chi the nominal part indicator (the rasterized binary part mask), w = 1
everywhere, and t_stop chosen per arm as the argmin of J along that arm's own
trajectory. J is reported raw (in units of cells) and per part cell.

---

## 1. Verdict

**1. Solving for the ideal dopant map beats the calibrated heuristic on three of
the four shapes, and the margins are large.** COMPUTED, J at each arm's own
optimal stop, best adjoint arm against the best heuristic arm:

| shape | uniform | best heuristic | best adjoint | adjoint vs best heuristic | adjoint vs uniform |
|---|---|---|---|---|---|
| square | 210.19 | **27.94** (permittivity channel) | **20.14** | **-27.9 %** | **-90.4 %** |
| triangle | 202.82 | **189.84** (permittivity channel) | **88.06** | **-53.6 %** | **-56.6 %** |
| cross | 471.82 | **353.45** (permittivity channel) | 360.07 | +1.9 % | -23.7 % |
| L_shape | 538.42 | 616.94 (both heuristics WORSE than uniform) | **396.52** | **-35.7 %** | **-26.4 %** |

**2. In shape terms the square is essentially solved and the others are not.**
COMPUTED, IoU of the melted region against the nominal part at the optimal stop:
square 0.851 uniform to **0.9975** solved (part growth 7.25 % to **0.00 %**,
under-melt 8.75 % to **0.25 %**); triangle 0.769 to **0.878**; cross 0.547 to
**0.678**; L_shape 0.514 to **0.627**. The three non-convex or high-aspect
shapes remain 12 to 32 IoU points short of the nominal geometry even with a
per-cell optimal dopant map.

**3. The ideal map is NOT in the proportional-inverse family, and that is now a
number rather than an impression.** The heuristic map is by construction a
monotone affine function of one smoothed proxy field; regressing it on that
proxy over the part cells gives a coefficient of determination of 0.85 to 0.99.
Regressing the ADJOINT map on the same proxy gives **0.027 (square), 0.010
(triangle), 0.118 (cross), 0.050 (L_shape)** for the box [0, 1] arm. Between
88 % and 99 % of the ideal map's structure is invisible to the heuristic family
no matter what gain is chosen. On the square the ideal map is a high-saturation
RIM around a depleted core (rim minus core +0.239); on the cross and the L_shape
it is the opposite sign (-0.145, -0.150). It is field-aware and shape-specific,
not a fixed rule.

**4. The new objective immediately exposes two failures the temperature proxy
hid.** COMPUTED: the conductivity-only heuristic at the window-selected gain
leaves **28.5 %** of the square unmelted and melts **nothing at all** on the
cross (J equals the part cell count exactly, optimal stop index 0, absorbed
power 105.8 W per m). Under the old melt-onset sigma_T convention neither case
could even be scored, because both fail to reach phi_bar = 0.90.

**5. The stop time is a real degree of freedom and the arms use it
differently.** COMPUTED optimal stops span 0.5 s (cross, conductivity-only
heuristic, meaning do not run the process) to 750.0 s. On the square the
uniform arm's best stop is 416.5 s while the adjoint's is 559.5 s: the graded
part can be run 34 % longer before melt escapes into the bed. phi_bar at the
optimal stop ranges 0.45 to 0.98 and is **never** 0.90, so the old fixed
phi_bar = 0.90 read state was not the shape-optimal stop for any arm on any
shape.

**6. The envelope claim is verified exactly, so the IFT machinery is not
needed.** PROVEN, `out_adjoint/gate_shape_square.json`: the gradient computed at
a FIXED stop index and the gradient of the min-over-time objective agree to a
relative difference of **0.000e+00**. Optimizing t_stop to stationarity removes
the dt*/ds term that the melt-onset objective required.

**7. The FD gate is a subgradient gate, not a clean pass, and the reason is
measured.** The single-cell probe at the maximum-sensitivity cell reaches
**3.04e-07**; a six-cell probe spanning the sensitivity range reaches 1.9e-06 to
1.9e-05 at each cell's own best epsilon; the random-direction probe bottoms at
**1.22e-05**. The objective reads melt fraction directly, so **95.0 % of domain
cells are pinned at phi = 0 or phi = 1** at the stop and carry zero sensitivity,
and their membership flips discretely as the design variable moves. Unlike the
melt-onset objective, widening the phase-change regularizer does NOT fix this
(the pinned population is the cold bed, which no realistic ramp width unpins):
dt_pc_c 10 to 40 C moves the pinned fraction only 95.0 % to 84.9 % and the gate
1.22e-05 to 9.57e-06.

---

## 2. What changed, and what was carried over

Carried over unchanged and still valid: the forward march proven BIT-IDENTICAL
to `rfam_eqs_coupled.run_sim` (`max|diff| = 0.000e+00` on four gates, three
shapes, both injection channels), the EQS assembly and its transpose, the
coupled (T, rho) substep VJP with subgradient handling of every clip, and the
per-objective cost accounting.

New in this pass:

| module | contents |
|---|---|
| `shape_objective.py` | J, its seed dJ/dT, the optimal-stop search, and the IoU / growth / under-melt / melt-window metrics |
| `gate_shape.py` | the S1 fixed-stop and S2 envelope-stop FD gates |
| `gate_shape_reg.py`, `gate_shape_cells.py` | the phase-ramp regularizer diagnostic and the multi-cell probe |
| `shape_solve.py` | the five-arm experiment |
| `map_structure.py` | the coefficient of determination of the ideal map against the heuristic's proxy, and the rim-versus-core contrast |
| `make_figures.py` | the composites and the J-against-time figure |
| `fix_heuristic.py` | recomputation of the control arms after a proxy-truncation bug (Section 7) |

Nine new unit tests written RED first and watched fail (`ImportError`, captured)
before `shape_objective.py` existed. Thirty-three tests pass in total.

**phi is differentiable in the forward and needed no new state.** It is an
algebraic function of the temperature field at the read step,
`phi = clip((T - T_pc)/dT_pc + 0.5, 0, 1)`, the same expression the
apparent-heat-capacity phase change uses inside the march. So the shape
objective seeds the EXISTING reverse march at one outer step with
`dJ/dT = 2 w (phi - chi) / dT_pc` on the cells inside the ramp and zero on the
pinned ones. No new adjoint state was required, and the coupled (T, rho) sweep
and the EQS adjoint are reused verbatim.

---

## 3. Gate values

### 3.1 The two shape layers, square, 120 x 120, 1600 design variables

Central differences, epsilon swept 1e-3 to 1e-7, random unit direction over the
part cells and a single-cell probe at the maximum-|gradient| cell. Base run:
optimal stop index 847 (424.0 s), J = 211.95.
Artifact `out_adjoint/gate_shape_square.json`.

| layer | random direction | single cell | verdict |
|---|---|---|---|
| S1 dJ/ds at a FIXED stop index | 1.220e-05 (eps 1e-5) | **3.050e-07** (eps 1e-6) | single cell PASS, random FAIL |
| S2 dJ*/ds with t_stop = argmin | 1.220e-05 | **3.050e-07** | identical to S1 |
| S1 against S2 gradient | | **0.000e+00 relative difference** | the envelope argument holds exactly |

The S1-against-S2 row is the specific verification asked for: with t_stop
separately optimized to stationarity there is no dt*/ds contribution, and the
two gradients are bit-for-bit the same because the argmin index is the same and
the seed is evaluated at that index.

### 3.2 Multi-cell probe and per-cell epsilon windows

`out_adjoint/gate_shape_cells_square.json`,
`out_adjoint/gate_shape_cellsweep_square_hi.json`. The paired-difference
estimator is used (elementwise residual difference summed afterwards) so the
about-13-digit cancellation in forming J_plus minus J_minus cannot masquerade as
a gradient error; it changed nothing, which is itself the evidence that the
residual disagreement is model-level, not roundoff.

| cell | analytic dJ/ds | best relative error | at epsilon |
|---|---|---|---|
| (40, 79), the maximum-sensitivity cell | -10.172982 | **3.04e-07** | 1e-6 |
| (57, 57) | +0.561009 | **1.94e-06** | 1e-5 |
| (42, 58) | +0.086664 | **9.85e-07** | 1e-4 |
| (74, 49) | +0.142826 | 1.88e-05 | 1e-5 |
| (65, 53) | +0.465122 | 9.30e-06 | 1e-6 |
| (71, 57) | +0.293867 | 1.29e-05 | 1e-6 |

Each cell has its own usable epsilon window and the error is strongly
non-monotone outside it (cell (57,57) is 0.78 relative at eps = 3e-5 and
1.94e-06 at eps = 1e-5). That is the signature of a piecewise-smooth objective
with dense breakpoints, which is what a melt front sweeping cells across a clip
boundary produces.

### 3.3 The regularizer does not rescue this gate

`out_adjoint/gate_shape_regularizer.json`, random-direction probe:

| dt_pc_c, deg C | domain cells pinned at phi = 0 or 1 | best relative error |
|---|---|---|
| 10 (production) | 95.01 % | 1.22e-05 |
| 20 | 89.40 % | 1.38e-04 |
| 40 | 84.86 % | 9.57e-06 |

Contrast with the melt-onset objective of the previous report, where the same
widening took the gate from 5.5e-07 to 1.1e-08. There the pinned population was
the fully melted core, which a wider ramp unpins. Here it is the cold powder
bed at about 25 C, hundreds of degrees below the ramp, which no realistic width
unpins. **The shape objective is intrinsically a subgradient problem at the melt
front.** That is stated as a limit, not worked around.

### 3.4 Standing non-smooth-term gates

COMPUTED on every solve reported here: temperature-step clip fraction 0.0000,
temperature-clamp fraction 0.0000, Q_rf cap fraction 0.0000 on all 28 scored
arms. The Q_rf cap is asserted inactive.

---

## 4. Cost

`out_shape/<shape>.json`, field `cost`, measured wall clock with thread
oversubscription disabled. One objective-plus-gradient evaluation costs
`1 + ratio` forward-solve equivalents.

| shape | part cells | forward, s | adjoint, s | ratio | evaluations inside 40 equivalents |
|---|---|---|---|---|---|
| square | 1600 | 4.18 | 6.51 | 1.559 | 15 |
| triangle | 800 | 3.12 | 4.74 | 1.521 | 15 |
| cross | 1036 | 4.05 | 6.77 | 1.670 | 14 |
| L_shape | 1079 | 3.52 | 4.97 | 1.412 | 16 |

The ratio is higher than the 0.44 of the heating-peak objective in the previous
report because the shape objective's read state is late in the march, so the
reverse sweep spans most of the trajectory. Total wall time for the four-shape,
five-arm experiment: 1564 s.

---

## 5. Results, every arm at its own optimal stop

Metrics: J and J per part cell; IoU of the melted region (phi >= 0.5) against
the nominal part; part growth (melted bed area as a percentage of part area);
part under-melt; the stop index and time; the horizon flag; phi_bar at the stop;
the melt-window metrics (part cells below 175 C, above 185 C, and the 95th
percentile of part temperature minus 185 C, floored at zero); sigma_T as a
diagnostic; and absorbed power. `A1` is box [0, 1], the same actuation range as
the heuristic; `A15` is box [0, 1.5], the two-sided per-node actuator.

| shape | arm | J | J per cell | IoU | growth % | under % | stop idx | stop s | phi_bar | under 175 % | over 185 % | p95 overshoot C | sigma_T C | P_abs W/m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | uniform | 210.19 | 0.1314 | 0.8508 | 7.25 | 8.75 | 832 | 416.5 | 0.884 | 2.75 | 43.50 | 9.4 | 4.77 | 500.0 |
| square | heuristic, conductivity only | 366.72 | 0.2292 | 0.7150 | 0.00 | 28.50 | 1499 | 750.0 | 0.711 | 13.00 | 52.88 | 13.9 | 9.12 | 299.5 |
| square | heuristic, permittivity co-varying | 27.94 | 0.0175 | 0.9765 | 0.88 | 1.50 | 738 | 369.5 | 0.976 | 0.00 | 91.50 | 14.4 | 5.11 | 570.1 |
| square | adjoint A1, 15 equivalents | 87.89 | 0.0549 | 0.9415 | 4.62 | 1.50 | 1039 | 520.0 | 0.955 | 0.00 | 81.25 | 11.3 | 4.82 | 445.0 |
| square | adjoint A15, 15 equivalents | 78.96 | 0.0493 | 0.9448 | 4.12 | 1.62 | 1004 | 502.5 | 0.952 | 0.00 | 76.50 | 9.1 | 4.20 | 450.3 |
| square | adjoint A1, 40 equivalents | 22.63 | 0.0141 | 0.9975 | 0.25 | 0.00 | 1146 | 573.5 | 0.967 | 0.00 | 72.62 | 11.4 | 4.29 | 403.2 |
| square | **adjoint A15, 40 equivalents** | **20.14** | **0.0126** | **0.9975** | **0.00** | **0.25** | 1118 | 559.5 | 0.975 | 0.00 | 82.12 | 12.4 | 4.74 | 414.7 |
| triangle | uniform | 202.82 | 0.2535 | 0.7687 | 16.75 | 10.25 | 506 | 253.5 | 0.850 | 4.00 | 55.25 | 60.9 | 21.84 | 500.0 |
| triangle | heuristic, conductivity only | 212.73 | 0.2659 | 0.7359 | 10.75 | 18.50 | 799 | 400.0 | 0.812 | 14.75 | 75.00 | 18.7 | 19.84 | 329.0 |
| triangle | heuristic, permittivity co-varying | 189.84 | 0.2373 | 0.7806 | 18.50 | 7.50 | 624 | 312.5 | 0.888 | 2.75 | 66.75 | 61.1 | 23.76 | 439.9 |
| triangle | adjoint A1, 15 equivalents | 94.02 | 0.1175 | 0.8545 | 8.25 | 7.50 | 531 | 266.0 | 0.886 | 2.75 | 60.25 | 34.3 | 12.84 | 450.5 |
| triangle | adjoint A15, 15 equivalents | 90.79 | 0.1135 | 0.8782 | 6.75 | 6.25 | 512 | 256.5 | 0.883 | 1.75 | 57.00 | 33.4 | 12.41 | 461.9 |
| triangle | **adjoint A1, 40 equivalents** | **88.06** | **0.1101** | 0.8578 | 7.25 | 8.00 | 539 | 270.0 | 0.892 | 3.25 | 63.50 | 32.3 | 12.25 | 442.3 |
| triangle | adjoint A15, 40 equivalents | 90.79 | 0.1135 | **0.8782** | 6.75 | 6.25 | 512 | 256.5 | 0.883 | 1.75 | 57.00 | 33.4 | 12.41 | 461.9 |
| cross | uniform | 471.82 | 0.4554 | 0.5465 | 3.86 | 43.24 | 439 | 220.0 | 0.563 | 39.38 | 47.88 | 21.9 | 34.97 | 500.0 |
| cross | heuristic, conductivity only | 1036.00 | 1.0000 | 0.0000 | 0.00 | 100.00 | 0 | 0.5 | 0.000 | 100.00 | 0.00 | 0.0 | 0.19 | 105.8 |
| cross | **heuristic, permittivity co-varying** | **353.45** | **0.3412** | 0.6633 | 15.83 | 23.17 | 1499 | 750.0 | 0.774 | 13.90 | 68.15 | 22.0 | 13.11 | 256.7 |
| cross | adjoint A1, 15 equivalents | 407.24 | 0.3931 | 0.6550 | 20.85 | 20.85 | 1023 | 512.0 | 0.791 | 17.37 | 74.52 | 64.8 | 32.03 | 358.0 |
| cross | adjoint A15, 15 equivalents | 372.02 | 0.3591 | 0.6764 | 18.73 | 19.69 | 1263 | 632.0 | 0.803 | 15.06 | 74.13 | 51.6 | 24.22 | 302.4 |
| cross | adjoint A1, 40 equivalents | 360.07 | 0.3476 | **0.6777** | 16.80 | 20.85 | 1498 | 749.5 | 0.790 | 15.83 | 73.36 | 46.1 | 21.96 | 263.0 |
| cross | adjoint A15, 40 equivalents | 369.28 | 0.3564 | 0.6604 | 13.71 | 24.90 | 1112 | 556.5 | 0.750 | 20.46 | 69.11 | 41.8 | 22.75 | 311.2 |
| L_shape | uniform | 538.42 | 0.4990 | 0.5142 | 7.78 | 44.58 | 526 | 263.5 | 0.550 | 39.11 | 47.54 | 36.4 | 39.21 | 500.0 |
| L_shape | heuristic, conductivity only | 616.94 | 0.5718 | 0.4379 | 6.67 | 53.29 | 910 | 455.5 | 0.465 | 47.45 | 40.59 | 25.1 | 34.51 | 314.6 |
| L_shape | heuristic, permittivity co-varying | 630.23 | 0.5841 | 0.4222 | 5.38 | 55.51 | 439 | 220.0 | 0.446 | 50.70 | 40.22 | 35.9 | 46.94 | 493.4 |
| L_shape | adjoint A1, 15 equivalents | 525.42 | 0.4869 | 0.5166 | 6.39 | 45.04 | 520 | 260.5 | 0.543 | 39.30 | 46.15 | 33.4 | 39.11 | 495.9 |
| L_shape | adjoint A15, 15 equivalents | 421.47 | 0.3906 | 0.6065 | 6.21 | 35.59 | 501 | 251.0 | 0.633 | 28.82 | 53.57 | 33.1 | 34.26 | 547.6 |
| L_shape | adjoint A1, 40 equivalents | 521.16 | 0.4830 | 0.5209 | 6.39 | 44.58 | 525 | 263.0 | 0.547 | 38.92 | 46.71 | 33.8 | 39.24 | 493.6 |
| L_shape | **adjoint A15, 40 equivalents** | **396.52** | **0.3675** | **0.6265** | 6.21 | 33.46 | 502 | 251.5 | 0.654 | 27.43 | 55.42 | 32.5 | 33.60 | 553.0 |

Two horizon flags, both stated rather than buried: the conductivity-only
heuristic on the square has its minimum at the last stored step (index 1499),
so its J is a lower bound on how bad it is; and the permittivity-co-varying
heuristic on the cross has its minimum at index 1499 as well, so its J is a
lower bound on how good it is. The cross A1 arm at index 1498 is one step short
of the same flag. **The cross comparison at 40 equivalents, adjoint 360.07
against heuristic 353.45, is inside the uncertainty those two flags create and
should not be read as a heuristic win; it should be read as a tie that the
1500-step horizon cannot resolve.**

---

## 6. What the ideal map looks like

`out_shape/<shape>_mapstructure.json`. `R2 against proxy` is the coefficient of
determination of a least-squares affine fit of the map on the heuristic's own
proxy field over the part cells: it measures how much of the map the
proportional-inverse family could express at ANY gain. `rim minus core` is the
mean saturation in the outer two cell rings minus the mean in the eroded core.

| shape | arm | mean s in part | R2 against proxy | rim mean | core mean | rim minus core |
|---|---|---|---|---|---|---|
| square | heuristic | 0.451 | **0.846** | 0.478 | 0.445 | +0.033 |
| square | adjoint A1 | 0.689 | **0.027** | 0.882 | 0.643 | **+0.239** |
| square | adjoint A15 | 0.925 | 0.182 | 1.047 | 0.896 | +0.151 |
| triangle | heuristic | 0.668 | **0.950** | 0.731 | 0.643 | +0.088 |
| triangle | adjoint A1 | 0.805 | **0.010** | 0.784 | 0.813 | -0.030 |
| triangle | adjoint A15 | 0.925 | 0.033 | 1.046 | 0.879 | +0.168 |
| cross | heuristic | 0.332 | **0.984** | 0.432 | 0.287 | +0.145 |
| cross | adjoint A1 | 0.610 | **0.118** | 0.510 | 0.655 | **-0.145** |
| cross | adjoint A15 | 0.835 | 0.036 | 0.734 | 0.881 | -0.147 |
| L_shape | heuristic | 0.491 | **0.989** | 0.498 | 0.487 | +0.011 |
| L_shape | adjoint A1 | 0.923 | **0.050** | 0.824 | 0.974 | **-0.150** |
| L_shape | adjoint A15 | 1.180 | 0.367 | 1.053 | 1.245 | -0.192 |

Read from the figures together with this table:

- **Square.** The ideal map is a high-saturation rim (0.88 to 1.05) around a
  depleted core (0.64 to 0.90). The melt front then lands on the outline to
  within a quarter of a percent of the part area. The heuristic map is the
  opposite structure, a smooth bowl that is LOWEST at the four corners, because
  the corners are the hottest cells of the proxy field. Only 2.7 % of the ideal
  map is expressible in that family.
- **Triangle.** The ideal map raises the two slanted flanks and depresses the
  apex tip and a band along the base. The heuristic instead empties the apex,
  which is why it truncates the tip and leaves 18.5 % of the part unmelted.
- **Cross and L_shape.** The sign inverts: the ideal map depresses the rim and
  raises the core of the poorly coupled limbs. The electrodes are top and
  bottom, so the horizontal arms of the cross and the foot of the L are
  perpendicular to the field and couple weakly. The solver pushes their
  saturation to the top of the box and still cannot melt them.

**The honest statement about the family:** the ideal map is not a rescaling of
any smoothed temperature or absorbed-power field. It is a boundary-aware
correction whose sign depends on whether the local limb is over-coupled (square,
triangle flanks: raise the rim to pull the front outward and shorten the run) or
under-coupled (cross arms, L foot: give up on the rim and drive the core). No
single-parameter gain on a monotone proxy can produce both signs on the same
part.

---

## 7. Two implementation errors found and fixed

Both were caught by cross-checking against a value that should not have moved,
and both are recorded because they are the kind of error that silently changes a
conclusion.

1. **Saturation outside the part.** The first pass set the design variable to
   zero outside the part mask. On shapes whose rasterized boundary has
   partially-filled cells outside the binary mask, that strips their geometry
   fill from the permittivity field for the whole run. It moved the uniform
   triangle reference from 23.043 to 39.367 C on the old metric. The convention
   is now s = 1 outside the part, so an arm changes the dopant map and nothing
   else, and the uniform arms reproduce the published baselines exactly.
2. **Proxy truncation.** The shape-fidelity early stop truncates the march long
   before phi_bar = 0.90 on slow shapes, so the first pass built the control's
   map from a truncated run on the cross and the L_shape. Recomputed with a
   full-horizon proxy (`fix_heuristic.py`); the cross permittivity control moved
   from J 353.27 to 353.45 and the L_shape conductivity control from 619.28 to
   616.94, so the effect was small here, but the control was not the published
   one until it was fixed.

---

## 8. Proven, computed, assumed

**PROVEN**
- The forward is bit-identical to `rfam_eqs_coupled.run_sim`,
  `max|diff| = 0.000e+00`, on three shapes and both injection channels, over
  1500 outer steps (carried over, `out_adjoint/l0_*.json`).
- The fixed-stop and envelope-stop shape gradients are identical, relative
  difference 0.000e+00, so no dt*/ds term is required.
- Thirty-three unit tests, nine of them new and written red first, covering the
  objective value, its symmetry between part growth and under-melt, the seed
  derivative against finite differences, the zero-subgradient of pinned cells,
  the optimal-stop argmin and its horizon flag, and the region and window
  metrics.

**COMPUTED**
- Every number in Sections 1, 3, 4, 5 and 6.
- The shape gradient agrees with paired-difference finite differences to
  3.04e-07 at the maximum-sensitivity cell and to between 9.85e-07 and 1.88e-05
  at five other cells, each at its own epsilon window.
- 95.0 % of domain cells are pinned at phi = 0 or 1 at the optimal stop, and
  widening the phase-change regularizer to 40 C reduces that only to 84.9 %.
- Adjoint cost 1.41 to 1.67 forward-solve equivalents per gradient.

**ASSUMED**
- That the rasterized binary part mask is the right nominal target chi. The
  sub-pixel geometry fill is an equally defensible target and would change every
  boundary cell's contribution. Not tested.
- That w = 1 is the right weighting. A bed-versus-part weighting is the labeled
  sensitivity check the brief asks for and was NOT run.
- That phi >= 0.5 is the right melted-region threshold for IoU. J itself does
  not use a threshold; only the IoU, growth and under-melt metrics do.
- That an optimal stop is realizable as a process control. The model assumes the
  radio-frequency drive can be cut at an arbitrary step.

---

## 9. Honest limits

1. **The FD gate is a subgradient gate.** The random-direction probe does not
   reach 1e-6 and, unlike the melt-onset objective, cannot be made to by
   widening the regularizer. Every optimization result here rests on a gradient
   that is exact between breakpoints and one-sided at them.
2. **Fifteen L-BFGS-B evaluations on 1600 design variables is a very small
   budget.** On the triangle the box [0, 1.5] arm made no progress at all
   between 15 and 40 equivalents (identical J, identical evaluation index 4),
   which is a line-search stall on a non-smooth objective, not convergence. All
   adjoint numbers are upper bounds on J, that is, lower bounds on what the
   method can do.
3. **The adjoint arm is conductivity-only; the permittivity hook is still
   pinned.** On the square and the cross the permittivity-co-varying heuristic is
   competitive or better precisely because it has an actuator the solver does
   not. A permittivity-co-varying gradient is the obvious missing layer.
4. **Not dose matched.** Absorbed power at the stop spans 105.8 to 570.1 W per
   metre against a 500.0 W per metre uniform baseline. The objective penalizes
   both under and over melting, which removes the crudest dose gaming, but the
   arms are not power-matched and the table reports the power so the confound
   stays visible.
5. **Two arms have their optimal stop at the 1500-step horizon** (square
   conductivity-only heuristic, cross permittivity heuristic) and one is one step
   short of it (cross adjoint A1). The cross ranking is unresolved at this
   horizon.
6. **Single grid, single geometry set, four shapes.** No mesh-convergence check.
7. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). An IoU of
   0.9975 is a statement about the model.
8. **The 2-D forward has no through-thickness physics.** Part growth here is
   in-plane only; melt spreading into the layer below is not represented.

---

## 10. The single most valuable next layer

**Add the permittivity channel to the gradient and re-run the same four
shapes.** The forward already supports it and is L0-gated in it
(`out_adjoint/l0_square_epscovary.json`, `max|diff| = 0.000e+00`); only the
chain rule from the design variable through the complex permittivity into gamma
is missing, which is a small extension of the existing EQS adjoint and one more
FD gate. It is the one change that would let the solve compete with the control
on its own actuator, and Section 5 shows that on two of four shapes the actuator,
not the search direction, is what decides the result.

Second: **the bed-versus-part weighting sensitivity check** (w larger on the
bed than on the part), which the brief flags as a labeled variant and which
directly tests whether the growth term is doing the work on the square.

Third: **more budget on the two re-entrant shapes.** The cross and the L_shape
are stalling, not converging, and their comparisons sit inside the horizon
uncertainty.

---

## 11. Artifacts, absolute paths

Worktree root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/.claude/worktrees/agent-a02efc1141ba69c58`

- Code: `adjoint2d/` (24 modules), tests `adjoint2d/tests/` (33 tests)
- Shape-fidelity gates: `out_adjoint/gate_shape_square.json`,
  `out_adjoint/gate_shape_regularizer.json`,
  `out_adjoint/gate_shape_cells_square.json`,
  `out_adjoint/gate_shape_cells_triangle.json`,
  `out_adjoint/gate_shape_cellsweep_square.json`,
  `out_adjoint/gate_shape_cellsweep_square_hi.json`
- Carried-over identity gates: `out_adjoint/l0_{square,triangle,cross}_full.json`,
  `out_adjoint/l0_square_epscovary.json`
- Solve results: `out_shape/{square,triangle,cross,L_shape}.json`
- Dopant maps and part masks: `out_shape/{shape}_maps.npz`
- Map structure: `out_shape/{shape}_mapstructure.json`
- Figures: `figs/fig_shape_{square,triangle,cross,L_shape}.png`,
  `figs/fig_shape_Jcurves.png`
- Run logs: `out_shape_run.log`, `out_shape_fix.log`, `out_shape_figs.log`
- Runners: `run_shape.sh`, `run_shape_fix.sh`, `run_shape_figs.sh`,
  `run_mapstruct.sh`
- Previous pass, still valid: `ADJOINT_PROTOTYPE_REPORT.md`,
  `out_adjoint/PREREGISTRATION.md`
