# Solving the whole standardized shape library for a printable dopant map

**Date:** 2026-07-31. **Scope:** the shape-fidelity adjoint solve of
`SHAPE_FIDELITY_SOLVE_REPORT.md`, run across all 18 shapes of the standardized
library instead of four, with the historical baseline strengthened from two
hand-picked masks to an exhaustive scan of every stored 4-bits-per-pixel dopant
map the campaign holds, and with the standing energy-residual gate wired in.
Nothing was committed. No dissertation file was touched. All work is in the git
worktree `.claude/worktrees/agent-a02efc1141ba69c58`.

**Acronyms, expanded on first use.** FGM = functionally graded material (a
spatially varying dopant saturation map). EQS = electro-quasi-static (the
low-frequency Maxwell approximation the 2-D solver uses). FD = finite
difference. IoU = intersection over union. bpp = bits per pixel. L-BFGS-B =
limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints.
phi = melt fraction. phi_bar = mean part melt fraction. sigma_T = the older 2-D
uniformity metric, not used here. J = the shape-fidelity objective.

**Evidence tags.** PROVEN = unit-tested or bit-identity-gated. COMPUTED =
measured from a real run in this campaign. ASSUMED = a modelling choice or an
inference not measured here.

**The objective, and the stop convention, stated once.**

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with chi_part the rasterized binary part mask. **Every J, IoU, growth,
under-melt, phi_bar and absorbed-power number in this report is read at that
arm's OWN J-stop**, t_stop = argmin over that arm's own stored trajectory of J.
No arm is read at a fixed time and none is read at phi_bar = 0.90. `horizon` is
flagged whenever the minimum sits on the last stored step (index 1499, 750.0 s),
which means the objective had not turned and that arm's J is a bound, not a
value. The melted region for IoU, growth and under-melt is phi >= 0.5; J itself
uses no threshold. Absorbed power is the state-B value in watts per metre of
depth.

---

## 1. Verdict, census first

**COMPUTED.** The printable arm is the solved per-cell dopant map quantized to
4 bits per pixel through the production `fgm_generator.py` convention and re-run
through the real forward, box [0, 1], that is, a SINGLE printing pass. The
baseline is the best by J of every distinct stored 4-bits-per-pixel dopant map
the campaign holds for that shape, in either boundary convention, scored on the
same engine at its own J-stop.

**1. The printable single-pass solved map beats the best stored historical mask
on J in 13 of 18 shapes and on IoU in 13 of 18 shapes, and it is the same 13
shapes both times.** Of the other five it LOSES on both metrics on four (square,
rectangle, cross, star6) and on the rounded_rect it loses on J while TYING on
IoU to twelve digits (both 0.9973958333333334). It beats the uniform reference on J on 18 of 18.

**2. Seven of the 18 are SOLVED in the absolute sense**, meaning the printable
map's melted region matches the nominal part to IoU >= 0.95: square (0.9816),
circle (0.9904), hexagon (0.9767), ellipse (0.9787), octagon (0.9944),
rounded_rect (0.9974) and trapezoid (0.9718). Eleven are not, and the worst
three are T_shape (0.4444), L_shape (0.5209) and cross (0.6755).

**3. The historical baseline used here is far stronger than the one the previous
two reports used, and that is why the square flipped.** Scanning all 19 stored
square masks found `cal map_m0p5477` at J = 12.77, IoU = 0.9975, which beats both
the window-reselection winner (J 28.53) and the old-grid m = 0.85 mask
(J 22.20) that `VERIFICATION_PRINTABILITY_REPORT.md` Section 3.3 called "the
genuinely best historical square arm". It is not. Against the mask the campaign
ACTUALLY standardized on, the old-grid m = 0.85 arm, the printable solved map
wins **16 of 18 on J and 16 of 18 on IoU**.

**4. One of the five losses is a solver stall, not an actuator limit, and it is
the rectangle.** COMPUTED: on the rectangle the solve moved J only
from the uniform 203.07 to 196.87 in 15 gradient evaluations, a 3.1 percent
improvement, while a stored mask reaches J = 10.69 at IoU = 1.0000. The solved
map is nearly uniform (mean saturation 0.834, and see
`figs/fig_lib_rectangle.png`, where the melt fraction in the part interior never
reaches 1). The line search stalled from the s = 1 start.

**5. Three more losses point at the permittivity actuator, which the solve still
does not have.** Every stored historical mask is injected through the
permittivity-co-varying hook, so it moves conductivity AND permittivity; every
solved arm moves conductivity only. COMPUTED, at each arm's own J-stop:

| shape | best stored mask: J / IoU / growth % / under % | solved 4 bpp: J / IoU / growth % / under % |
|---|---|---|
| square | 12.77 / 0.9975 / 0.00 / 0.25 | 25.66 / 0.9816 / 1.88 / 0.00 |
| rounded_rect | 8.97 / 0.9974 / 0.00 / 0.26 | 18.21 / 0.9974 / 0.00 / 0.26 |
| star6 | 70.57 / 0.8782 / 5.41 / 7.43 | 84.69 / 0.8408 / 6.08 / 10.81 |

On the square and the star6 the historical mask genuinely melts more of the part
with less growth. On the rounded_rect the two arms have IDENTICAL thresholded
geometry (same IoU, same growth, same under-melt to two decimals) and differ only
in J, that is, only in the partially-melted halo below phi = 0.5 that J charges
for and IoU does not: the historical arm reaches its stop at 308.5 s with a
sharper front, the solved arm at 641.5 s with a softer one. **Calling the
rounded_rect a loss is a statement about the halo, not about the printed
shape.** The permittivity channel remains the missing layer both previous reports
named as the single most valuable next step, now measured on 18 shapes instead of
four.

**5b. The fifth loss, the cross, is not resolved by this horizon and should not
be read as a loss.** COMPUTED: the cross solved 4-bits-per-pixel arm has its
J-minimum at the last stored step (index 1499, 750.0 s), so its J = 360.18 is an
upper bound that a longer run could only lower. The historical arm's stop is
interior at 430.0 s. The 8.7 percent gap is inside that ambiguity. This is the
same unresolved cross comparison both previous reports flagged.

**6. The standing 5 percent energy-residual gate is now wired in and it is clean
everywhere.** COMPUTED over **718 scored forward runs** (84 primary arms plus
634 historical-scan arms): the maximum relative energy residual at any arm's own
stop is **3.23 percent** (octagon, the extreme m = 6.0 stored mask) and the
maximum on any PRIMARY arm is **2.21 percent** (equilateral_triangle, uniform).
**Zero violations.** The three clip gates (temperature-step clip, temperature
clamp, Q_rf cap) are exactly 0.0000 on all 718.

**7. Bit depth costs almost nothing at 4 bits per pixel and is shape-dependent at
2.** COMPUTED: across the 18 shapes the 4-bits-per-pixel arm differs from the
continuous solved map on J by between -2.54 percent (hexagon, the quantized map
is BETTER) and +2.66 percent (ellipse), and on IoU by at most 0.0035 (star). At
2 bits per pixel the change ranges from -2.25 percent on J (star6, better) to
+82.9 percent (circle, J 14.56 to 26.63, IoU 0.9904 to 0.9667). Shapes whose
solved map has fine rim structure pay for bit depth; shapes whose map is nearly
flat do not. The sub-percent wins at 4 bits per pixel are breakpoint noise on a
piecewise-smooth objective, not a mechanism.

**8. The double pass was triggered on 4 shapes and it is worth it on two.**
The trigger is more than 15 percent of the part left unmelted by the printable
single-pass arm. It fired on L_shape, T_shape, cross and star. It buys +10.2 IoU
points on the L_shape and +4.3 on the T_shape; on the cross and the star it buys
nothing measurable.

---

## 2. The census figure

`figs/fig_lib_census.png`. Panel A is the intersection over union of the melted
region with the nominal part, one row per shape, an arrow from the best stored
historical mask (hollow) to the printable solved map (filled), green when the
solved map is better and red when it is worse, with the uniform reference as a
grey tick and the IoU >= 0.95 band shaded. Panel B is the relative change in J
against the same baseline, on the same row order. The axis of panel B is clamped
at 100 percent; the three bars that run off it keep their true value in the
label, because the rectangle's -1743 percent is real and comes from dividing by a
historical J of 10.69.

---

## 3. The census, shape by shape

Deliverable arm: solved map, box [0, 1], 4 bits per pixel, single printing pass,
re-run through the real forward. Baseline: best by J of all stored masks.
(H) marks an arm whose J-stop is the 1500-step horizon.

| shape | best stored mask, J | its IoU | solved 4 bpp, J | its IoU | J change | IoU change | beats on J | beats on IoU | class |
|---|---|---|---|---|---|---|---|---|---|
| square | 12.77 | 0.9975 | 25.66 | 0.9816 | -100.9 % | -0.0159 | no | no | SOLVED |
| circle | 52.18 | 0.9492 | 14.68 | 0.9904 | +71.9 % | +0.0412 | YES | YES | SOLVED |
| hexagon | 69.63 | 0.9211 | 22.15 | 0.9767 | +68.2 % | +0.0556 | YES | YES | SOLVED |
| triangle | 171.27 | 0.7768 | 87.99 | 0.8578 | +48.6 % | +0.0810 | YES | YES | IMPROVED |
| equilateral_triangle | 145.55 | 0.8058 | 119.00 | 0.8128 | +18.2 % | +0.0071 | YES | YES | IMPROVED |
| L_shape | 597.86 (H) | 0.4413 | 521.20 | 0.5209 | +12.8 % | +0.0796 | YES | YES | IMPROVED |
| H_shape | 219.11 | 0.7631 | 150.86 | 0.8423 | +31.1 % | +0.0792 | YES | YES | IMPROVED |
| T_shape | 676.39 | 0.3983 | 609.29 | 0.4444 | +9.9 % | +0.0462 | YES | YES | IMPROVED |
| cross | 331.36 | 0.7052 | 360.18 (H) | 0.6755 | -8.7 % | -0.0297 | no | no | NOT RESCUED |
| diamond | 355.07 | 0.7793 | 211.16 (H) | 0.8520 | +40.5 % | +0.0727 | YES | YES | IMPROVED |
| ellipse | 74.70 | 0.8854 | 13.98 | 0.9787 | +81.3 % | +0.0933 | YES | YES | SOLVED |
| octagon | 16.22 | 0.9888 | 10.66 | 0.9944 | +34.3 % | +0.0056 | YES | YES | SOLVED |
| pentagon | 134.64 | 0.8446 | 44.22 | 0.9387 | +67.2 % | +0.0941 | YES | YES | IMPROVED |
| rectangle | 10.69 | 1.0000 | 196.94 | 0.8424 | -1742.6 % | -0.1576 | no | no | NOT RESCUED |
| rounded_rect | 8.97 | 0.9974 | 18.21 | 0.9974 | -103.1 % | +0.0000 | no | no | SOLVED |
| star | 172.11 | 0.6713 | 157.39 | 0.7032 | +8.6 % | +0.0319 | YES | YES | IMPROVED |
| star6 | 70.57 | 0.8782 | 84.69 | 0.8408 | -20.0 % | -0.0374 | no | no | NOT RESCUED |
| trapezoid | 122.22 | 0.8688 | 27.36 | 0.9718 | +77.6 % | +0.1030 | YES | YES | SOLVED |

**How to read the class column.** SOLVED is ABSOLUTE: the printable arm reaches
IoU >= 0.95, whatever the baseline does. The other three are COMPARATIVE against
the best stored mask with a tie band of 5 percent on J and 2 IoU points; when the
two metrics disagree in direction beyond the band the verdict is MATCHED, because
a disagreement is not a win. This is why square and rounded_rect read SOLVED
while also losing to the baseline: their melted region IS the nominal part, and a
stored mask happens to reach it with a smaller partially-melted halo, which is
what J charges for and IoU does not.

**Shapes flagged as candidates for a different actuator** (printable arm below
IoU 0.80 after a 40 forward-equivalent solve, so more dopant budget is not the
answer): **T_shape (0.4444), L_shape (0.5209), cross (0.6755), star (0.7032),
equilateral_triangle (0.8128) is borderline**, plus **rectangle (0.8424)** for
the separate reason that its solve stalled. The first four all have a limb
perpendicular to the electrode axis (the electrodes are top and bottom), and in
every one of their figures that limb is the part that never melts: see the
horizontal arms in `figs/fig_lib_cross.png`, the foot in
`figs/fig_lib_L_shape.png`, the crossbar in `figs/fig_lib_T_shape.png` and the
two side points in `figs/fig_lib_star.png`. **Orientation or turntable
actuation, or temporal power scheduling, is the natural next actuator for that
class. Neither was implemented here.**

---

## 4. Against the mask the campaign actually standardized on

The census above uses an oracle: the best of roughly 18 stored masks per shape,
selected in hindsight under the very objective the solve optimizes. That is the
strongest available baseline and it is the right one for the headline. It is not
what the campaign would have printed. The old-grid m = 0.85 as-stored arm is what
`outputs_eqs/geometry_dual_readstate` actually ran on every shape.

| shape | old-grid m = 0.85 as stored, J | its IoU | solved 4 bpp, J | its IoU | J change | IoU change |
|---|---|---|---|---|---|---|
| square | 22.20 | 0.9838 | 25.66 | 0.9816 | -15.6 % | -0.0022 |
| circle | 107.97 | 0.9080 | 14.68 | 0.9904 | +86.4 % | +0.0824 |
| hexagon | 85.38 | 0.9037 | 22.15 | 0.9767 | +74.1 % | +0.0729 |
| triangle | 369.26 | 0.5865 | 87.99 | 0.8578 | +76.2 % | +0.2713 |
| equilateral_triangle | 231.57 | 0.7273 | 119.00 | 0.8128 | +48.6 % | +0.0856 |
| L_shape | 691.64 | 0.3668 | 521.20 | 0.5209 | +24.6 % | +0.1541 |
| H_shape | 316.65 | 0.6717 | 150.86 | 0.8423 | +52.4 % | +0.1705 |
| T_shape | 762.88 | 0.3195 | 609.29 | 0.4444 | +20.1 % | +0.1249 |
| cross | 414.59 | 0.6279 | 360.18 | 0.6755 | +13.1 % | +0.0476 |
| diamond | 490.77 | 0.6957 | 211.16 | 0.8520 | +57.0 % | +0.1563 |
| ellipse | 93.08 | 0.8680 | 13.98 | 0.9787 | +85.0 % | +0.1107 |
| octagon | 122.96 | 0.8724 | 10.66 | 0.9944 | +91.3 % | +0.1220 |
| pentagon | 162.36 | 0.8097 | 44.22 | 0.9387 | +72.8 % | +0.1290 |
| rectangle | 16.19 | 0.9931 | 196.94 | 0.8424 | -1116.8 % | -0.1506 |
| rounded_rect | 31.01 | 0.9702 | 18.21 | 0.9974 | +41.3 % | +0.0272 |
| star | 179.05 | 0.6757 | 157.39 | 0.7032 | +12.1 % | +0.0275 |
| star6 | 91.16 | 0.8333 | 84.69 | 0.8408 | +7.1 % | +0.0074 |
| trapezoid | 256.49 | 0.7710 | 27.36 | 0.9718 | +89.3 % | +0.2008 |

**COMPUTED: against the campaign's standard arm the printable solved map wins 16
of 18 on J and 16 of 18 on IoU**, losing only the square (-15.6 percent on J,
-0.0022 IoU, effectively a tie on shape) and the rectangle (the stall).

---

## 5. Which stored mask won, and how much the oracle selection buys

| shape | stored masks scanned | winning mask | campaign | convention | J | IoU |
|---|---|---|---|---|---|---|
| square | 19 | `cal_map_m0p5477_mag0p55` | calibration campaign | asstored | 12.77 | 0.9975 |
| circle | 18 | `oldgrid_map_m0p70_mag0p70` | old {0.30 .. 0.85} grid | outside1 | 52.18 | 0.9492 |
| hexagon | 18 | `cal_map_m0p8327_mag0p83` | calibration campaign | outside1 | 69.63 | 0.9211 |
| triangle | 14 | `cal_map_m2p7016_mag2p70` | calibration campaign | outside1 | 171.27 | 0.7768 |
| equilateral_triangle | 18 | `cal_map_m1p2164_mag1p22` | calibration campaign | asstored | 145.55 | 0.8058 |
| L_shape | 18 | `cal_map_m1p0927_mag1p09` | calibration campaign | outside1 | 597.86 | 0.4413 |
| H_shape | 17 | `cal_map_m0p1090_mag0p11` | calibration campaign | asstored | 219.11 | 0.7631 |
| T_shape | 18 | `cal_map_m1p0927_mag1p09` | calibration campaign | asstored | 676.39 | 0.3983 |
| cross | 18 | `cal_map_m1p0927_mag1p09` | calibration campaign | asstored | 331.36 | 0.7052 |
| diamond | 18 | `cal_map_m1p4854_mag1p49` | calibration campaign | asstored | 355.07 | 0.7793 |
| ellipse | 18 | `oldgrid_map_m0p85_mag0p85` | old {0.30 .. 0.85} grid | outside1 | 74.70 | 0.8854 |
| octagon | 18 | `cal_map_m0p1110_mag0p11` | calibration campaign | outside1 | 16.22 | 0.9888 |
| pentagon | 17 | `cal_map_m0p1110_mag0p11` | calibration campaign | outside1 | 134.64 | 0.8446 |
| rectangle | 17 | `oldgrid_map_m0p50_mag0p50` | old {0.30 .. 0.85} grid | asstored | 10.69 | 1.0000 |
| rounded_rect | 18 | `cal_map_m0p5569_mag0p56` | calibration campaign | asstored | 8.97 | 0.9974 |
| star | 18 | `oldgrid_map_m0p50_mag0p50` | old {0.30 .. 0.85} grid | outside1 | 172.11 | 0.6713 |
| star6 | 18 | `cal_map_m0p9644_mag0p96` | calibration campaign | asstored | 70.57 | 0.8782 |
| trapezoid | 17 | `cal_map_m0p1110_mag0p11` | calibration campaign | outside1 | 122.22 | 0.8688 |

Eight of the 18 winners come from the calibration campaign's coarse gain grid
rather than from its window-selected winner, and four come from the old
{0.30 .. 0.85} grid. **The window-reselection winner is the best stored mask
under J on none of the 18 shapes**, which is expected, because those gains were
selected on the melt-window criterion and not on shape fidelity, but it means the
comparison in `VERIFICATION_PRINTABILITY_REPORT.md` Section 3.3 was against a
weaker baseline than the campaign actually contains.

The spread inside the one-parameter family is large, so the oracle selection is
doing real work and its cost should be named: picking the winner took 34 to 38
forward solves per shape, against the 40 forward-equivalents the adjoint solve
was given.

| shape | stored arms scored | best (oracle) J | median J | worst J | old-grid m = 0.85 J | uniform J |
|---|---|---|---|---|---|---|
| square | 38 | 12.77 | 23.78 | 242.93 | 22.20 | 210.19 |
| circle | 36 | 52.18 | 110.30 | 520.05 | 107.97 | 275.02 |
| hexagon | 36 | 69.63 | 102.42 | 424.68 | 85.38 | 251.27 |
| triangle | 28 | 171.27 | 334.72 | 391.14 | 369.26 | 202.82 |
| equilateral_triangle | 36 | 145.55 | 238.17 | 400.07 | 231.57 | 320.80 |
| L_shape | 36 | 597.86 | 968.11 | 1079.00 | 691.64 | 538.42 |
| H_shape | 34 | 219.11 | 235.11 | 475.28 | 316.65 | 214.68 |
| T_shape | 36 | 676.39 | 1070.00 | 1104.00 | 762.88 | 614.26 |
| cross | 36 | 331.36 | 419.79 | 1036.00 | 414.59 | 471.82 |
| diamond | 36 | 355.07 | 490.77 | 525.67 | 490.77 | 768.11 |
| ellipse | 36 | 74.70 | 97.91 | 427.27 | 93.08 | 241.72 |
| octagon | 36 | 16.22 | 106.90 | 628.97 | 122.96 | 42.96 |
| pentagon | 34 | 134.64 | 152.75 | 434.49 | 162.36 | 300.33 |
| rectangle | 34 | 10.69 | 15.81 | 298.03 | 16.19 | 203.07 |
| rounded_rect | 36 | 8.97 | 22.72 | 940.32 | 31.01 | 266.38 |
| star | 36 | 172.11 | 179.05 | 275.63 | 179.05 | 192.58 |
| star6 | 36 | 70.57 | 124.12 | 315.62 | 91.16 | 190.62 |
| trapezoid | 34 | 122.22 | 143.39 | 335.01 | 256.49 | 171.32 |

**Two shapes where the whole stored family is worse than doing nothing.**
COMPUTED: on the L_shape the best of 18 stored masks scores J = 597.86 against
the uniform arm's 538.42, and on the H_shape 219.11 against 214.68. On those two
shapes the proportional-inverse family has no member that helps.

---

## 6. The double pass, for the shapes that needed it

Trigger, decided per shape and not by hand: the printable single-pass arm leaves
more than 15 percent of the part unmelted. Both arms at 4 bits per pixel so both
are printable; saturation above 1.0 is treated as a second printing pass at the
same 1/15 level quantum, the convention of
`VERIFICATION_PRINTABILITY_REPORT.md` Section 4.1.

| shape | single-pass under-melt % (trigger) | single pass J | its IoU | its stop s | double pass J | its IoU | its stop s | J change | IoU change |
|---|---|---|---|---|---|---|---|---|---|
| L_shape | 44.58 | 521.20 | 0.5209 | 263.0 | 402.55 | 0.6225 | 253.0 | +22.8 % | +0.1016 |
| T_shape | 54.35 | 609.29 | 0.4444 | 217.0 | 553.99 | 0.4875 | 205.5 | +9.1 % | +0.0431 |
| cross | 21.24 | 360.18 | 0.6755 | 750.0 | 368.13 | 0.6610 | 565.0 | -2.2 % | -0.0145 |
| star | 26.57 | 157.39 | 0.7032 | 164.0 | 149.79 | 0.7092 | 155.5 | +4.8 % | +0.0060 |

**The stop times say the same thing they said on four shapes.** On the L_shape
the single-pass arm's stop is 263.0 s and the double pass's is 253.0 s: the
single pass already runs LONGER and still melts less, so exposure does not
substitute for saturation. On the cross the single-pass arm's stop is at the
horizon, so its J is a bound and the double-pass comparison there is unresolved.

---

## 7. Every arm, every shape

| shape | arm | J | J per part cell | IoU | growth % | under % | stop idx | stop s | horizon | phi_bar | P_abs W/m | energy residual % of dose | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | uniform s = 1 | 210.19 | 0.1314 | 0.8508 | 7.25 | 8.75 | 832 | 416.5 | no | 0.884 | 500.0 | 0.70 | PASS |
| square | best stored historical mask | 12.77 | 0.0080 | 0.9975 | 0.00 | 0.25 | 778 | 389.5 | no | 0.982 | 559.8 | 0.16 | PASS |
| square | solved [0, 1] continuous | 25.56 | 0.0160 | 0.9804 | 2.00 | 0.00 | 1107 | 554.0 | no | 0.968 | 412.0 | 0.25 | PASS |
| square | **solved [0, 1] 4 bpp** | 25.66 | 0.0160 | 0.9816 | 1.88 | 0.00 | 1107 | 554.0 | no | 0.967 | 412.0 | 0.24 | PASS |
| square | solved [0, 1] 2 bpp | 25.52 | 0.0160 | 0.9901 | 1.00 | 0.00 | 1117 | 559.0 | no | 0.963 | 409.5 | 0.25 | PASS |
| circle | uniform s = 1 | 275.02 | 0.2218 | 0.7881 | 14.19 | 10.00 | 755 | 378.0 | no | 0.898 | 500.0 | 1.17 | PASS |
| circle | best stored historical mask | 52.18 | 0.0421 | 0.9492 | 1.61 | 3.55 | 452 | 226.5 | no | 0.959 | 689.3 | 0.22 | PASS |
| circle | solved [0, 1] continuous | 14.56 | 0.0117 | 0.9904 | 0.65 | 0.32 | 1136 | 568.5 | no | 0.983 | 365.7 | 0.14 | PASS |
| circle | **solved [0, 1] 4 bpp** | 14.68 | 0.0118 | 0.9904 | 0.65 | 0.32 | 1132 | 566.5 | no | 0.983 | 366.4 | 0.14 | PASS |
| circle | solved [0, 1] 2 bpp | 26.63 | 0.0215 | 0.9667 | 1.77 | 1.61 | 1186 | 593.5 | no | 0.977 | 354.2 | 0.17 | PASS |
| hexagon | uniform s = 1 | 251.27 | 0.2473 | 0.7649 | 18.90 | 9.06 | 652 | 326.5 | no | 0.912 | 500.0 | 1.43 | PASS |
| hexagon | best stored historical mask | 69.63 | 0.0685 | 0.9211 | 4.72 | 3.54 | 403 | 202.0 | no | 0.962 | 663.2 | 0.49 | PASS |
| hexagon | solved [0, 1] continuous | 22.73 | 0.0224 | 0.9785 | 0.59 | 1.57 | 885 | 443.0 | no | 0.973 | 376.2 | 0.17 | PASS |
| hexagon | **solved [0, 1] 4 bpp** | 22.15 | 0.0218 | 0.9767 | 1.18 | 1.18 | 896 | 448.5 | no | 0.974 | 373.6 | 0.18 | PASS |
| hexagon | solved [0, 1] 2 bpp | 23.52 | 0.0231 | 0.9728 | 1.18 | 1.57 | 911 | 456.0 | no | 0.975 | 370.7 | 0.18 | PASS |
| triangle | uniform s = 1 | 202.82 | 0.2535 | 0.7687 | 16.75 | 10.25 | 506 | 253.5 | no | 0.850 | 500.0 | 1.40 | PASS |
| triangle | best stored historical mask | 171.27 | 0.2141 | 0.7768 | 12.00 | 13.00 | 736 | 368.5 | no | 0.855 | 354.4 | 0.86 | PASS |
| triangle | solved [0, 1] continuous | 88.06 | 0.1101 | 0.8578 | 7.25 | 8.00 | 539 | 270.0 | no | 0.892 | 442.3 | 0.52 | PASS |
| triangle | **solved [0, 1] 4 bpp** | 87.99 | 0.1100 | 0.8578 | 7.25 | 8.00 | 539 | 270.0 | no | 0.892 | 442.1 | 0.52 | PASS |
| triangle | solved [0, 1] 2 bpp | 90.65 | 0.1133 | 0.8538 | 7.75 | 8.00 | 536 | 268.5 | no | 0.890 | 444.6 | 0.53 | PASS |
| equilateral_triangle | uniform s = 1 | 320.80 | 0.4760 | 0.6067 | 32.05 | 19.88 | 479 | 240.0 | no | 0.797 | 500.0 | 2.21 | PASS |
| equilateral_triangle | best stored historical mask | 145.55 | 0.2159 | 0.8058 | 13.06 | 8.90 | 348 | 174.5 | no | 0.871 | 571.3 | 1.26 | PASS |
| equilateral_triangle | solved [0, 1] continuous | 118.81 | 0.1763 | 0.8128 | 10.98 | 9.79 | 589 | 295.0 | no | 0.895 | 395.4 | 0.78 | PASS |
| equilateral_triangle | **solved [0, 1] 4 bpp** | 119.00 | 0.1766 | 0.8128 | 10.98 | 9.79 | 589 | 295.0 | no | 0.896 | 395.6 | 0.79 | PASS |
| equilateral_triangle | solved [0, 1] 2 bpp | 118.64 | 0.1760 | 0.8155 | 10.98 | 9.50 | 590 | 295.5 | no | 0.896 | 395.1 | 0.78 | PASS |
| L_shape | uniform s = 1 | 538.42 | 0.4990 | 0.5142 | 7.78 | 44.58 | 526 | 263.5 | no | 0.550 | 500.0 | 0.85 | PASS |
| L_shape | best stored historical mask | 597.86 | 0.5541 | 0.4413 | 5.00 | 53.66 | 1499 | 750.0 | YES | 0.443 | 228.4 | 0.41 | PASS |
| L_shape | solved [0, 1] continuous | 521.16 | 0.4830 | 0.5209 | 6.39 | 44.58 | 525 | 263.0 | no | 0.547 | 493.6 | 0.68 | PASS |
| L_shape | **solved [0, 1] 4 bpp** | 521.20 | 0.4830 | 0.5209 | 6.39 | 44.58 | 525 | 263.0 | no | 0.547 | 493.7 | 0.68 | PASS |
| L_shape | solved [0, 1] 2 bpp | 521.42 | 0.4832 | 0.5200 | 6.39 | 44.67 | 525 | 263.0 | no | 0.547 | 493.5 | 0.68 | PASS |
| L_shape | solved [0, 1.5] continuous | 394.59 | 0.3657 | 0.6272 | 6.39 | 33.27 | 503 | 252.0 | no | 0.656 | 552.5 | 0.64 | PASS |
| L_shape | solved [0, 1.5] 4 bpp | 402.55 | 0.3731 | 0.6225 | 6.30 | 33.83 | 505 | 253.0 | no | 0.651 | 549.3 | 0.65 | PASS |
| L_shape | solved [0, 1.5] 2 bpp | 440.63 | 0.4084 | 0.5795 | 7.78 | 37.53 | 517 | 259.0 | no | 0.623 | 533.1 | 0.73 | PASS |
| H_shape | uniform s = 1 | 214.68 | 0.1864 | 0.7563 | 9.72 | 17.01 | 694 | 347.5 | no | 0.832 | 500.0 | 0.92 | PASS |
| H_shape | best stored historical mask | 219.11 | 0.1902 | 0.7631 | 12.85 | 13.89 | 589 | 295.0 | no | 0.867 | 570.7 | 1.16 | PASS |
| H_shape | solved [0, 1] continuous | 150.78 | 0.1309 | 0.8412 | 10.42 | 7.12 | 827 | 414.0 | no | 0.904 | 450.0 | 0.94 | PASS |
| H_shape | **solved [0, 1] 4 bpp** | 150.86 | 0.1310 | 0.8423 | 10.07 | 7.29 | 825 | 413.0 | no | 0.903 | 450.3 | 0.93 | PASS |
| H_shape | solved [0, 1] 2 bpp | 156.38 | 0.1357 | 0.8331 | 10.24 | 8.16 | 826 | 413.5 | no | 0.900 | 450.5 | 0.95 | PASS |
| T_shape | uniform s = 1 | 614.26 | 0.5564 | 0.4574 | 4.17 | 52.36 | 460 | 230.5 | no | 0.462 | 500.0 | 0.62 | PASS |
| T_shape | best stored historical mask | 676.39 | 0.6127 | 0.3983 | 5.07 | 58.15 | 1497 | 749.0 | no | 0.397 | 227.9 | 0.45 | PASS |
| T_shape | solved [0, 1] continuous | 609.31 | 0.5519 | 0.4444 | 2.72 | 54.35 | 433 | 217.0 | no | 0.438 | 500.3 | 0.33 | PASS |
| T_shape | **solved [0, 1] 4 bpp** | 609.29 | 0.5519 | 0.4444 | 2.72 | 54.35 | 433 | 217.0 | no | 0.438 | 500.3 | 0.33 | PASS |
| T_shape | solved [0, 1] 2 bpp | 610.08 | 0.5526 | 0.4444 | 2.72 | 54.35 | 433 | 217.0 | no | 0.438 | 500.8 | 0.34 | PASS |
| T_shape | solved [0, 1.5] continuous | 551.60 | 0.4996 | 0.4937 | 1.27 | 50.00 | 410 | 205.5 | no | 0.493 | 557.0 | 0.28 | PASS |
| T_shape | solved [0, 1.5] 4 bpp | 553.99 | 0.5018 | 0.4875 | 1.45 | 50.54 | 410 | 205.5 | no | 0.491 | 555.3 | 0.28 | PASS |
| T_shape | solved [0, 1.5] 2 bpp | 566.58 | 0.5132 | 0.4734 | 2.17 | 51.63 | 412 | 206.5 | no | 0.477 | 544.1 | 0.28 | PASS |
| cross | uniform s = 1 | 471.82 | 0.4554 | 0.5465 | 3.86 | 43.24 | 439 | 220.0 | no | 0.563 | 500.0 | 0.59 | PASS |
| cross | best stored historical mask | 331.36 | 0.3198 | 0.7052 | 21.81 | 14.09 | 859 | 430.0 | no | 0.857 | 397.9 | 1.64 | PASS |
| cross | solved [0, 1] continuous | 360.07 | 0.3476 | 0.6777 | 16.80 | 20.85 | 1498 | 749.5 | no | 0.790 | 263.0 | 1.10 | PASS |
| cross | **solved [0, 1] 4 bpp** | 360.18 | 0.3477 | 0.6755 | 16.60 | 21.24 | 1499 | 750.0 | YES | 0.789 | 262.6 | 1.10 | PASS |
| cross | solved [0, 1] 2 bpp | 359.55 | 0.3471 | 0.6833 | 15.83 | 20.85 | 1499 | 750.0 | YES | 0.788 | 261.8 | 1.08 | PASS |
| cross | solved [0, 1.5] continuous | 367.28 | 0.3545 | 0.6542 | 13.90 | 25.48 | 1122 | 561.5 | no | 0.750 | 309.0 | 1.00 | PASS |
| cross | solved [0, 1.5] 4 bpp | 368.13 | 0.3553 | 0.6610 | 13.32 | 25.10 | 1129 | 565.0 | no | 0.750 | 307.8 | 1.01 | PASS |
| cross | solved [0, 1.5] 2 bpp | 370.39 | 0.3575 | 0.6570 | 13.71 | 25.29 | 1150 | 575.5 | no | 0.748 | 302.8 | 1.00 | PASS |
| diamond | uniform s = 1 | 768.11 | 0.4730 | 0.5903 | 24.75 | 26.35 | 946 | 473.5 | no | 0.738 | 500.0 | 2.07 | PASS |
| diamond | best stored historical mask | 355.07 | 0.2186 | 0.7793 | 9.36 | 14.78 | 1073 | 537.0 | no | 0.852 | 433.1 | 0.83 | PASS |
| diamond | solved [0, 1] continuous | 211.79 | 0.1304 | 0.8535 | 2.59 | 12.44 | 1499 | 750.0 | YES | 0.869 | 319.4 | 0.27 | PASS |
| diamond | **solved [0, 1] 4 bpp** | 211.16 | 0.1300 | 0.8520 | 2.34 | 12.81 | 1499 | 750.0 | YES | 0.867 | 318.5 | 0.25 | PASS |
| diamond | solved [0, 1] 2 bpp | 226.67 | 0.1396 | 0.8406 | 1.97 | 14.29 | 1499 | 750.0 | YES | 0.837 | 313.0 | 0.20 | PASS |
| ellipse | uniform s = 1 | 241.72 | 0.3249 | 0.6909 | 18.28 | 18.28 | 426 | 213.5 | no | 0.814 | 500.0 | 1.56 | PASS |
| ellipse | best stored historical mask | 74.70 | 0.1004 | 0.8854 | 3.23 | 8.60 | 306 | 153.5 | no | 0.909 | 608.9 | 0.37 | PASS |
| ellipse | solved [0, 1] continuous | 13.62 | 0.0183 | 0.9787 | 1.08 | 1.08 | 1314 | 657.5 | no | 0.974 | 232.9 | 0.16 | PASS |
| ellipse | **solved [0, 1] 4 bpp** | 13.98 | 0.0188 | 0.9787 | 1.08 | 1.08 | 1283 | 642.0 | no | 0.974 | 236.2 | 0.16 | PASS |
| ellipse | solved [0, 1] 2 bpp | 22.05 | 0.0296 | 0.9577 | 1.61 | 2.69 | 1433 | 717.0 | no | 0.968 | 221.3 | 0.18 | PASS |
| octagon | uniform s = 1 | 42.96 | 0.0399 | 0.9489 | 1.86 | 3.35 | 656 | 328.5 | no | 0.961 | 500.0 | 0.22 | PASS |
| octagon | best stored historical mask | 16.22 | 0.0151 | 0.9888 | 0.00 | 1.12 | 390 | 195.5 | no | 0.982 | 792.4 | 0.16 | PASS |
| octagon | solved [0, 1] continuous | 10.60 | 0.0099 | 0.9944 | 0.00 | 0.56 | 758 | 379.5 | no | 0.984 | 437.6 | 0.14 | PASS |
| octagon | **solved [0, 1] 4 bpp** | 10.66 | 0.0099 | 0.9944 | 0.00 | 0.56 | 758 | 379.5 | no | 0.984 | 437.4 | 0.14 | PASS |
| octagon | solved [0, 1] 2 bpp | 10.60 | 0.0099 | 0.9907 | 0.00 | 0.93 | 760 | 380.5 | no | 0.984 | 440.0 | 0.14 | PASS |
| pentagon | uniform s = 1 | 300.33 | 0.3229 | 0.7130 | 20.65 | 13.98 | 579 | 290.0 | no | 0.857 | 500.0 | 1.66 | PASS |
| pentagon | best stored historical mask | 134.64 | 0.1448 | 0.8446 | 5.16 | 11.18 | 411 | 206.0 | no | 0.885 | 602.1 | 0.52 | PASS |
| pentagon | solved [0, 1] continuous | 43.89 | 0.0472 | 0.9387 | 1.72 | 4.52 | 867 | 434.0 | no | 0.952 | 352.0 | 0.21 | PASS |
| pentagon | **solved [0, 1] 4 bpp** | 44.22 | 0.0475 | 0.9387 | 1.72 | 4.52 | 864 | 432.5 | no | 0.952 | 352.4 | 0.21 | PASS |
| pentagon | solved [0, 1] 2 bpp | 67.11 | 0.0722 | 0.9083 | 3.23 | 6.24 | 887 | 444.0 | no | 0.935 | 344.4 | 0.27 | PASS |
| rectangle | uniform s = 1 | 203.07 | 0.1763 | 0.8528 | 13.19 | 3.47 | 642 | 321.5 | no | 0.839 | 500.0 | 1.16 | PASS |
| rectangle | best stored historical mask | 10.69 | 0.0093 | 1.0000 | 0.00 | 0.00 | 661 | 331.0 | no | 0.972 | 483.7 | 0.19 | PASS |
| rectangle | solved [0, 1] continuous | 196.87 | 0.1709 | 0.8450 | 14.24 | 3.47 | 642 | 321.5 | no | 0.863 | 503.2 | 1.23 | PASS |
| rectangle | **solved [0, 1] 4 bpp** | 196.94 | 0.1710 | 0.8424 | 14.58 | 3.47 | 642 | 321.5 | no | 0.863 | 503.3 | 1.23 | PASS |
| rectangle | solved [0, 1] 2 bpp | 197.68 | 0.1716 | 0.8450 | 14.24 | 3.47 | 643 | 322.0 | no | 0.864 | 502.8 | 1.23 | PASS |
| rounded_rect | uniform s = 1 | 266.38 | 0.1734 | 0.8190 | 12.24 | 8.07 | 853 | 427.0 | no | 0.912 | 500.0 | 1.04 | PASS |
| rounded_rect | best stored historical mask | 8.97 | 0.0058 | 0.9974 | 0.00 | 0.26 | 616 | 308.5 | no | 0.987 | 639.9 | 0.13 | PASS |
| rounded_rect | solved [0, 1] continuous | 18.21 | 0.0119 | 0.9948 | 0.13 | 0.39 | 1282 | 641.5 | no | 0.979 | 376.6 | 0.16 | PASS |
| rounded_rect | **solved [0, 1] 4 bpp** | 18.21 | 0.0119 | 0.9974 | 0.00 | 0.26 | 1282 | 641.5 | no | 0.979 | 376.7 | 0.16 | PASS |
| rounded_rect | solved [0, 1] 2 bpp | 21.22 | 0.0138 | 0.9935 | 0.00 | 0.65 | 1225 | 613.0 | no | 0.978 | 386.4 | 0.17 | PASS |
| star | uniform s = 1 | 192.58 | 0.3553 | 0.6517 | 7.01 | 30.26 | 284 | 142.5 | no | 0.664 | 500.0 | 0.79 | PASS |
| star | best stored historical mask | 172.11 | 0.3176 | 0.6713 | 5.54 | 29.15 | 242 | 121.5 | no | 0.703 | 551.5 | 0.65 | PASS |
| star | solved [0, 1] continuous | 157.36 | 0.2903 | 0.6996 | 4.43 | 26.94 | 326 | 163.5 | no | 0.707 | 449.0 | 0.48 | PASS |
| star | **solved [0, 1] 4 bpp** | 157.39 | 0.2904 | 0.7032 | 4.43 | 26.57 | 327 | 164.0 | no | 0.708 | 448.5 | 0.49 | PASS |
| star | solved [0, 1] 2 bpp | 160.38 | 0.2959 | 0.6926 | 4.43 | 27.68 | 320 | 160.5 | no | 0.701 | 454.8 | 0.50 | PASS |
| star | solved [0, 1.5] continuous | 149.41 | 0.2757 | 0.7077 | 4.80 | 25.83 | 310 | 155.5 | no | 0.715 | 473.1 | 0.54 | PASS |
| star | solved [0, 1.5] 4 bpp | 149.79 | 0.2764 | 0.7092 | 4.06 | 26.20 | 310 | 155.5 | no | 0.712 | 472.0 | 0.53 | PASS |
| star | solved [0, 1.5] 2 bpp | 152.02 | 0.2805 | 0.7092 | 4.06 | 26.20 | 312 | 156.5 | no | 0.703 | 467.6 | 0.50 | PASS |
| star6 | uniform s = 1 | 190.62 | 0.3220 | 0.6758 | 22.97 | 16.89 | 391 | 196.0 | no | 0.850 | 500.0 | 1.82 | PASS |
| star6 | best stored historical mask | 70.57 | 0.1192 | 0.8782 | 5.41 | 7.43 | 273 | 137.0 | no | 0.894 | 585.9 | 0.73 | PASS |
| star6 | solved [0, 1] continuous | 84.80 | 0.1432 | 0.8408 | 6.08 | 10.81 | 460 | 230.5 | no | 0.874 | 410.6 | 0.64 | PASS |
| star6 | **solved [0, 1] 4 bpp** | 84.69 | 0.1431 | 0.8408 | 6.08 | 10.81 | 461 | 231.0 | no | 0.876 | 410.5 | 0.65 | PASS |
| star6 | solved [0, 1] 2 bpp | 82.88 | 0.1400 | 0.8408 | 6.08 | 10.81 | 460 | 230.5 | no | 0.873 | 409.0 | 0.61 | PASS |
| trapezoid | uniform s = 1 | 171.32 | 0.1441 | 0.8387 | 9.50 | 8.16 | 682 | 341.5 | no | 0.909 | 500.0 | 0.80 | PASS |
| trapezoid | best stored historical mask | 122.22 | 0.1028 | 0.8688 | 5.13 | 8.66 | 592 | 296.5 | no | 0.912 | 571.3 | 0.50 | PASS |
| trapezoid | solved [0, 1] continuous | 27.24 | 0.0229 | 0.9718 | 1.35 | 1.51 | 869 | 435.0 | no | 0.970 | 422.5 | 0.23 | PASS |
| trapezoid | **solved [0, 1] 4 bpp** | 27.36 | 0.0230 | 0.9718 | 1.35 | 1.51 | 867 | 434.0 | no | 0.969 | 422.9 | 0.23 | PASS |
| trapezoid | solved [0, 1] 2 bpp | 28.95 | 0.0243 | 0.9693 | 1.26 | 1.85 | 855 | 428.0 | no | 0.969 | 426.2 | 0.24 | PASS |

---

## 8. Cost

One objective-plus-gradient evaluation costs `1 + ratio` forward-solve
equivalents, with the ratio measured on that shape at run time. The budget was
**40 forward-equivalents per shape and it was NOT cut**; the resulting gradient
evaluation count is 14 to 18 depending on the measured ratio.

| shape | part cells | forward s | adjoint s | ratio | gradient evaluations inside 40 forward-equivalents | double pass run | wall s |
|---|---|---|---|---|---|---|---|
| square | 1600 | 9.87 | 17.59 | 1.782 | 14 | no | 530 |
| circle | 1240 | 14.53 | 24.54 | 1.690 | 14 | no | 861 |
| hexagon | 1016 | 12.38 | 19.48 | 1.574 | 15 | no | 753 |
| triangle | 800 | 10.19 | 14.99 | 1.472 | 16 | no | 582 |
| equilateral_triangle | 674 | 9.79 | 13.85 | 1.415 | 16 | no | 631 |
| L_shape | 1079 | 7.52 | 8.83 | 1.174 | 18 | yes | 679 |
| H_shape | 1152 | 13.75 | 20.77 | 1.510 | 15 | no | 786 |
| T_shape | 1104 | 9.72 | 12.64 | 1.301 | 17 | yes | 1019 |
| cross | 1036 | 6.99 | 10.16 | 1.454 | 16 | yes | 1720 |
| diamond | 1624 | 13.76 | 21.03 | 1.528 | 15 | no | 1349 |
| ellipse | 744 | 7.09 | 8.82 | 1.244 | 17 | no | 963 |
| octagon | 1076 | 10.08 | 14.77 | 1.465 | 16 | no | 785 |
| pentagon | 930 | 8.91 | 12.99 | 1.457 | 16 | no | 862 |
| rectangle | 1152 | 11.67 | 18.37 | 1.574 | 15 | no | 859 |
| rounded_rect | 1536 | 14.00 | 22.77 | 1.626 | 15 | no | 864 |
| star | 542 | 5.23 | 5.82 | 1.113 | 18 | yes | 661 |
| star6 | 592 | 6.23 | 8.10 | 1.299 | 17 | no | 504 |
| trapezoid | 1189 | 11.25 | 17.15 | 1.525 | 15 | no | 590 |

**A reproducibility wart, named.** The ratio is measured by wall clock, and the
shapes were run six at a time on a 12-core machine, so the ratio a shape measures
depends on machine load. The square measured 1.782 here against 1.559 in the
previous serial pass, so it got 14 gradient evaluations instead of 15 and its
solved J came out at 25.56 instead of the previously reported 22.63. The budget
is honest (40 forward-equivalents of real compute) but the eval count is not
reproducible to the integer. Total measured solve time 14 997 s of serial
equivalent, about 4.2 hours, completed in about 1.2 hours of wall clock at
six-way parallelism. Figure generation added about 20 minutes.

---

## 9. Gates

### 9.1 The standing energy-residual gate, newly wired

PROVEN and COMPUTED. New module `adjoint2d/energy_gate.py`, wired into
`adjoint2d/forward.py`. It reproduces the production definition rather than
re-deriving it: `e_in` is the cumulative integrated dose, `e_out` the cumulative
convective plus depth loss, and `e_stored` the INCREMENTAL stored energy
accumulated with beginning-of-outer-step material properties
(`rfam_eqs_coupled.py:3149-3163`); the residual is `e_in - e_out - e_stored`
(`rfam_eqs_coupled.py:3217`) and the reported quantity is
`|residual| / max(|e_in|, 1 J/m)`. The threshold is the 5 percent of
`test_energy_balance.py::test_c_full_run_residual`.

Five new tests, written RED first and the `ImportError` observed before the
module existed. The substantive one is a CAPTURED-REAL-DATA contract test, not
an invented fixture: it runs the PRODUCTION `rfam_eqs_coupled.run_sim` on the
stored square baseline configuration for 60 outer steps and requires the
prototype's `energy_in`, `energy_out` and residual series to match
`hist["energy_balance_residual_J_per_m"]` to a relative 1e-12 of the final dose.
It passes.

| quantity | value |
|---|---|
| scored forward runs with the gate evaluated at their own stop | **718** |
| gate violations (>= 5 percent) | **0** |
| worst relative residual, any arm | **3.23 %** (octagon, stored mask m = 6.0) |
| worst relative residual, any primary arm | **2.21 %** (equilateral_triangle, uniform) |
| worst temperature-step clip fraction | 0.0000 |
| worst temperature-clamp fraction | 0.0000 |
| worst Q_rf cap fraction | 0.0000 |

### 9.2 The gradient was not changed, and the gates prove the forward was not either

The only edit to `forward.py` is diagnostic accumulation. The claim that it
changed no physics is not asserted, it is gated twice.

**FD gate, square, re-run after the change** (`out_lib/gate_shape_square_postenergy.json`):

| layer | random direction | single cell | analytic dJ/ds at the probe cell |
|---|---|---|---|
| S1 dJ/ds at a FIXED stop index | 1.2201821406636123e-05 | **3.0503706577255724e-07** | -10.172982490016617 |
| S2 dJ*/ds with t_stop = argmin | 1.2201821406636123e-05 | **3.0503706577255724e-07** | -10.172982490016617 |
| S1 against S2 | | **0.000e+00 relative difference** | |

These are IDENTICAL TO EVERY PRINTED DIGIT to the pre-change values in
`out_adjoint/gate_shape_square.json`, including the base run's stop index 847 and
J = 211.95425494988422. **Verdict: single-cell probe PASS at 3.05e-07;
random-direction probe 1.22e-05, which is the standing SUBGRADIENT-gate FAIL
carried over unchanged from `SHAPE_FIDELITY_SOLVE_REPORT.md` Section 3.3.** It is
not fixed here and cannot be fixed by widening the phase-change regularizer,
because the pinned population is the cold powder bed. Every optimization number
in this report rests on a gradient that is exact between breakpoints and
one-sided at them.

**L0 bit-identity against stored production runs, re-run after the change**
(`out_lib/stored_run_gate_postenergy.json`):

| stored run | max abs diff on T_phi90, deg C | sigma_T prototype | sigma_T stored |
|---|---|---|---|
| `geometry_dual_readstate/runs/square/baseline` | **0.000e+00** | 4.930989680787 | 4.930989680787 |
| `geometry_dual_readstate/runs/square/fgm_m0p85` | **0.000e+00** | 3.768226015489 | 3.768226015489 |
| `geometry_dual_readstate/runs/triangle/fgm_m0p85` | **0.000e+00** | 45.357859475541 | 45.357859475541 |
| `geometry_dual_readstate/runs/cross/fgm_m0p85` | **0.000e+00** | 31.400589027953 | 31.400589027953 |

### 9.3 Unit tests

**56 pass** (41 carried over, 5 new for the energy gate, 10 new for the library
solve's pure logic: the double-pass trigger, the best-arm selection, the
four-way classification ladder, the deterministic configuration choice and the
deduplicated stored-mask catalogue). All 15 new tests were written red first and
the import errors were observed before the modules existed.

---

## 10. gt_logo: SKIPPED, loudly

`gt_logo` is the nineteenth shape in the campaign directories and it is NOT in
this report. COMPUTED: calling `build_case` on
`outputs_eqs/fgm_calibrated_control/configs/gt_logo_m0p6054.yaml` raises
`ModuleNotFoundError: No module named 'cv2'` from
`rfam_eqs_coupled.make_domain`, because its geometry is rasterized from an image.
The `.venv312` interpreter has no OpenCV. It was not worked around and it was not
silently dropped; the reason is recorded in
`adjoint2d/library_solve.GT_LOGO_SKIP_REASON` and asserted by a unit test.

---

## 11. Proven, computed, assumed

**PROVEN**
* The prototype forward is still bit-identical to four ACTUAL stored production
  runs after the energy-gate wiring, `max|diff| = 0.000e+00` on the archived
  T_phi90 field, sigma_T agreeing to twelve printed digits.
* The shape gradient is bit-for-bit unchanged by this pass, and the fixed-stop
  and envelope-stop gradients still agree to a relative difference of 0.000e+00.
* The prototype's energy bookkeeping reproduces the production engine's
  `energy_balance_residual_J_per_m` series to a relative 1e-12 of the final dose
  on a real 60-step production run.
* The quantizer reproduces five ACTUAL stored 1715 x 1715 `level_map` artifacts
  exactly (carried over, `test_printability.py`).
* 56 unit tests pass; the 15 new ones were written red first.

**COMPUTED**
* Every number in Sections 1 and 3 through 9.
* Zero energy-gate violations and zero clip-gate activations on 718 scored
  forward runs.

**ASSUMED**
* That the best-by-J stored mask is the right baseline. It is an oracle
  selection made in hindsight under the objective being optimized, over 34 to 38
  scored arms per shape. Section 4 gives the non-oracle comparison as well.
* That within a shape the calibration configurations are physically identical.
  Verified by `diff` on two shapes (square, star): the only differing line is the
  path of a dopant map the prototype never reads. NOT verified on the other 16.
* That the rasterized binary part mask is the right nominal target chi. Carried
  over, still untested.
* That saturation above 1.0 is a second printing pass at the same level quantum.
  Carried over, not measured on hardware.
* That an arbitrary stop time is realizable as a process control.
* That w = 1 over the whole domain is the right weighting. The bed-versus-part
  weighting sensitivity check is still not run.

---

## 12. Honest limits

1. **The FD gate is still a subgradient gate.** Random-direction probe 1.22e-05.
   Unchanged and not worked around.
2. **The adjoint arm is conductivity-only; every historical arm co-varies
   permittivity.** On 18 shapes this now clearly decides three of the five
   losses. It remains the single largest missing layer.
3. **The rectangle solve stalled** and is reported as a loss rather than
   excluded. It is a line-search failure on a non-smooth objective from the
   s = 1 start, not evidence that the rectangle cannot be graded: a stored mask
   reaches IoU 1.0000 on it.
4. **Five arms have their J-stop at the 1500-step horizon** (L_shape best
   historical mask; cross solved 4 bpp and 2 bpp; diamond solved continuous, 4
   bpp and 2 bpp), so their J is a bound. The cross ranking in particular is
   unresolved at this horizon, exactly as in the two previous reports.
5. **Not dose matched.** Absorbed power at the stop spans 221.3 W per metre
   (ellipse, solved 2 bpp) to 792.4 W per metre (octagon, best stored mask)
   across the 84 primary arms, against the 500.0 W per metre uniform baseline,
   and 66.3 to 1300.7 W per metre if the whole 718-run historical scan is
   included. The objective penalizes over-melting as well as under-melting, but
   the arms are not power-matched and every table keeps the power visible.
6. **Gradient evaluation counts vary 14 to 18 across shapes** because the budget
   is in forward-equivalents and the cost ratio is measured under machine load.
7. **Single grid (120 x 120), single geometry set, two dimensions.** No
   mesh-convergence check, no through-thickness physics, so part growth is
   in-plane only.
8. **The model over-predicts achievable tuned uniformity by roughly a factor of
   eight against hardware** (`ALLISON_LAW_REPLICATION.md` Section 6.1). An IoU of
   0.9944 is a statement about the model at grid 120 in two dimensions, not about
   a printed part.
9. **gt_logo was not solved.** 18 of 19, not 19 of 19.

---

## 13. The single most valuable next layer

Unchanged, and this pass is the strongest evidence yet for it: **add the
permittivity channel to the gradient.** The forward already supports it and is
bit-identity gated in it (`out_adjoint/l0_square_epscovary.json`). Three of the
five shapes the solve loses lose to a mask whose only advantage is that actuator,
and the previous report measured the size of the gap directly (the same cross
mask scores J 343 with permittivity and J 1036, nothing melts, without it).

Second: **fix the rectangle-class stall.** A multi-start or a scaled initial
step would tell in a few forward solves whether the conductivity-only family
contains a good rectangle map at all. Right now the report cannot distinguish
"the solver failed" from "the actuator cannot".

Third: **orientation, turntable or temporal power scheduling for the
perpendicular-limb class** (T_shape, L_shape, cross, star). Their unmelted region
is geometrically the limb that is perpendicular to the electrode axis, on every
figure, and no per-cell dopant map inside a fixed geometry and a fixed drive has
moved it. NOT implemented here, flagged as the actuator question.

---

## 14. Artifacts, absolute paths

Worktree root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/.claude/worktrees/agent-a02efc1141ba69c58`

New code:
* `adjoint2d/energy_gate.py` the standing energy-residual gate
* `adjoint2d/library_solve.py` the per-shape driver and all six arms
* `adjoint2d/make_library_figures.py` the per-shape composites and the census figure
* `adjoint2d/build_library_table.py` the tables in this report
* `adjoint2d/tests/test_energy_gate.py` 5 tests, red first, one a real-production contract test
* `adjoint2d/tests/test_library_solve.py` 10 tests, red first
* `adjoint2d/forward.py` modified: per-outer-step energy bookkeeping only
* `run_library.sh`, `run_library_figs.sh` the runners

New results:
* `out_lib/<shape>.json` for the 18 shapes: every arm, every metric, the full
  historical scan, the gates, the verdict
* `out_lib/<shape>_maps.npz` uniform, solved continuous, solved 4 bpp, solved
  2 bpp and, where triggered, the double-pass maps, plus the part mask and grid
* `out_lib/tables.md`, `out_lib/_sections.md` the generated tables
* `out_lib/gate_shape_square_postenergy.json` the re-run FD gate
* `out_lib/stored_run_gate_postenergy.json` the re-run L0 identity gate
* `logs_lib/<shape>.log` per-shape run logs

New figures, ALL viewed before delivery:
* `figs/fig_lib_census.png` the one-glance verdict
* `figs/fig_lib_<shape>.png` for all 18 shapes: dopant map on the top row,
  melted region at that arm's own optimal stop on the bottom row with the
  nominal outline in cyan and the melt front dashed white

Carried over and still valid: `SHAPE_FIDELITY_SOLVE_REPORT.md`,
`VERIFICATION_PRINTABILITY_REPORT.md`, `ADJOINT_PROTOTYPE_REPORT.md`,
`out_adjoint/`, `out_shape/`, `out_verify/`.

Main-tree sources read (read-only):
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/rfam_eqs_coupled.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_generator.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/test_energy_balance.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/HEATR_STANDARD_PARAMETERS.md`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/outputs_eqs/fgm_calibrated_control/{configs,runs}/`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/outputs_eqs/geometry_dual_readstate/runs/`
