# Gain Re-Selection on Stop-Time Melt-Window Metrics

**Date:** 2026-07-31. **Cost:** zero new solves (re-scored all cached arms from stored fields.npz).
**Why:** the earlier hold-out rule selected on melt-onset sigma_T alone. Latent-heat buffering
flattens every melt-state temperature read (settled project fact; the dual read state exists for
this reason), so "looks uniform in the shape window at the stop" is partly plateau flattery, and
end-of-horizon density is meaningless (everything oversinters; rho caps at 1.0).

## Selection rule (designed against the plateau)

Over every cached arm (old {0.3..0.85} grid runs + all v1/v2 calibration arms), from the stored
stop-time snapshot T_phi90 (T at the phi_bar = 0.90 crossing), melt window 175-185 C from config:

- FEASIBLE: melt reached, and under% (part cells < 175 C, i.e. still powder at stop)
  <= uniform's under% + 1 point.
- SELECT: minimize p95 overshoot past 185 C (cells done with latent absorption and still
  heating - the quantity the plateau cannot fake). Tie-break: under%, then max overshoot.
- Never select on %in-window (plateau-flattered).

## Winners (uniform p95 overshoot -> window-selected arm)

| shape | uni under% / p95_ov | selected m | under% | p95_ov C | sigma_T@melt | sigma_T@peak | verdict vs uniform |
|---|---|---|---|---|---|---|---|
| circle | 5.8 / 46 | 0.6402 | 4.2 | 13.1 | 7.16 | 12.01 | BETTER |
| cross | 8.1 / 113 | 1.1855 | 5.8 | 31.5 | 13.51 | 16.43 | BETTER |
| diamond | 6.9 / 97 | 2.1208 | 6.9 | 37.8 | 15.95 | 20.00 | BETTER |
| ellipse | 6.5 / 63 | 0.6713 | 5.9 | 16.0 | 8.48 | 12.76 | BETTER |
| equilateral_triangle | 3.0 / 122 | 1.2164 | 3.6 | 65.3 | 24.35 | 25.38 | BETTER |
| hexagon | 5.9 / 49 | 0.8327 | 2.4 | 16.5 | 6.73 | 11.85 | BETTER |
| octagon | 4.8 / 48 | 0.4513 | 4.5 | 16.7 | 6.96 | 11.37 | BETTER |
| pentagon | 6.5 / 73 | 0.3424 | 5.8 | 32.8 | 14.79 | 19.27 | BETTER |
| rectangle | 0.0 / 20 | 0.8500 | 0.0 | 5.3 | 2.36 | 9.69 | BETTER (old-grid arm) |
| rounded_rect | 3.4 / 18 | 0.7000 | 0.0 | 9.4 | 4.61 | 8.52 | BETTER (old-grid arm) |
| square | 2.5 / 12 | 0.9033 | 0.8 | 5.2 | 3.65 | 6.92 | BETTER |
| star6 | 0.7 / 47 | 0.9089 | 1.4 | 17.1 | 7.20 | 10.11 | BETTER |
| star | 6.6 / 67 | 0.8500 | 7.0 | 62.4 | 24.17 | 25.16 | neutral |
| triangle | 2.8 / 70 | 1.2164 | 2.0 | 74.9 | 27.63 | 27.77 | neutral (NO benefit) |
| H_shape | 0.0 / 14 | 0.5000 | 0.0 | 25.9 | 11.13 | 17.07 | HARMFUL |
| trapezoid | 2.9 / 30 | 0.2045 | 1.3 | 42.8 | 16.22 | 19.81 | HARMFUL |
| T_shape | 6.0 / 163 | 0.1110 | 6.7 | 189.2 | 69.11 | 69.09 | HARMFUL |
| L_shape | 7.1 / 165 | 0.1110 | 8.1 | 207.3 | 74.11 | 74.09 | HARMFUL |

(gt_logo not run; cv2 missing.)

## What changed vs the sigma_T-only selection

1. **The triangle m=6 "rescue" is REJECTED.** It violates the under-melt gate (6.8% vs uniform
   2.8%): its flat sigma_T came from parking the apex below melt. The best window-feasible
   triangle arm is no better than uniform. The triangle returns to the no-benefit class.
2. **The diamond m=2.1208 arm is CONFIRMED** (p95 overshoot 97 -> 37.8 C at equal under%), and
   the cross moves to a near-neighbor gain (1.1855 vs 1.0927), p95 113 -> 31.5 C.
3. **Rectangle and rounded_rect select OLD-grid arms** (0.85 / 0.70): the window rule
   independently recovers the arms the old in-sample sweep had found. The two selections agree
   wherever the old grid spanned the optimum.
4. **The benefit census converges to the familiar structure: 12 of 18 better, 2 neutral
   (star, triangle), 4 harmful (H_shape, trapezoid, T_shape, L_shape)** - consistent with the
   settled 19-shape campaign's 12-of-19, now measured on a metric that cannot be gamed by
   latent-plateau flattening or dose reduction.

## Limits (stated, not hidden)

- phi(x) at the crossing is not stored; under% uses T < 175 C as the unmelted proxy. Cells inside
  the window may be at any phi (plateau degeneracy) - the under-gate catches only clearly
  unmelted cells.
- Overshoot is capped in observation by the crossing time itself; arms are compared at their own
  crossings (equal mean melt state), which is the process-consistent comparison.
- Bed melt (part growth outside the mask) is NOT in any of these metrics and was visible in the
  GUI circle run; it needs its own metric before any of this reaches the dissertation.
- Arms are not dose-matched; the under-gate plus overshoot minimization removes the worst
  dose-reduction gaming, but a power-matched confirmation of the 4 big movers is still the
  clean close-out (~8 solves).
