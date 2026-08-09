# Premix baseline dopant 0→15 wt%: forward-physics sweep results

**Date:** 2026-08-07 · **Config:** `configs/jared_exp1_40mm_premix.yaml` (156×50, n_steps 1232,
update_interval 0) · **Solver:** 2-D `rfam_eqs_coupled.run_sim`, proven premix path (`floor_added`).
**Reproduce:** `python study_premix_sweep.py && python augment_matched_peak.py && python make_premix_figure.py`.
Records: `results/premix_sweep/level_*.json` (+ `.npz`); figure `fig_premix_sweep.png`.

## Conclusion (honest, and it contradicts the naive hypothesis)

Under a **fixed total absorbed power** (the heatr3d-faithful convention), a **uniform premix
baseline is parasitic** in this model. Because the bed is much larger than the part, even 3.75 wt%
flips **~80% of the power into the surrounding bed**, so at the drive that brings the part peak to the
250 °C ceiling the part is **under-fused** and **less uniform** than with no premix. Premix does **not**
lower the drive here (it rises ~8%). The "premix helps" intuition only survives under a *voltage-drive*
convention or a real coupling-efficiency gain — neither is modeled here.

## Matched-peak results (every level driven to a ~250 °C part peak — the fair comparison)

| wt% | premix_frac | bed absorb frac | drive→ceiling (W) | part mean φ (fusion) | part peak/mean |
|----:|----:|----:|----:|----:|----:|
| 0    | 0.00 | 0.000 | 736 | **0.999** | 1.147 |
| 3.75 | 0.15 | 0.817 | 766 | 0.510 | 1.285 |
| 7.5  | 0.30 | 0.790 | 796 | 0.584 | 1.273 |
| 11.25| 0.45 | 0.769 | 796 | 0.610 | 1.258 |
| 15   | 0.60 | 0.754 | 796 | 0.632 | 1.251 |

Three robust readings:
1. **RF absorption redistributes to the bed** (`bed absorb frac` 0 → ~0.8). Geometric, flips at low wt%.
2. **Part under-densifies at the ceiling** (`mean φ` 0.999 → 0.51–0.63): the bed steals power, so the
   part hits 250 °C at a hot spot while the bulk stays under-fused.
3. **Drive-to-ceiling rises modestly** (736 → 796 W, +8%) and **part uniformity worsens**
   (peak/mean 1.15 → 1.25–1.29). Within the premix-on range *more* premix is progressively less bad
   (φ 0.51→0.63), but never recovers the un-premixed φ≈1.0.

## Voltage-drive counterpart — the drive mode does NOT rescue it (tested 2026-08-07)

The fair counterpart (`study_premix_sweep_voltage.py`, results in `results/premix_sweep_voltage/`,
figure `fig_premix_drive_comparison.png`) bisects the applied **voltage** to the same 250 °C ceiling
instead of enforcing fixed power. Every converged run is THM-clean (0 clamp warnings; peak ~250 °C).

| wt% | voltage→ceiling (V) | part mean φ | peak/mean | bed absorb |
|----:|----:|----:|----:|----:|
| 0    | 763 | 0.996 | 1.145 | 0.000 |
| 3.75 | 167 | 0.519 | 1.288 | 0.817 |
| 15   |  95 | 0.655 | 1.259 | 0.754 |

**The part outcome (φ, uniformity, bed fraction) is essentially IDENTICAL to the power-drive sweep.**
Only the drive *knob* moves oppositely: power (W) rises +8%, **voltage (V) falls ~8×**. That voltage
drop is a units artifact (higher σ needs less voltage for the same field), **not** a part benefit — at
matched ceiling the field *shape*, hence the part state, is set by the σ *pattern* (same premix), not by
the drive mode. **Conclusion: a uniform premix baseline is parasitic under BOTH drive modes.** The bed
absorption is geometric and drive-mode-independent.

## Critical caveats — do not over-read this

1. ~~Drive-mode dependence~~ **RESOLVED (see the voltage-drive section above):** the earlier hypothesis
   that voltage drive would flip the sign and make premix helpful is **WRONG** — it flips only the drive
   knob, not the part outcome. Premix is parasitic in both modes.
2. **Uniform premix ≠ graded FGM.** This sweeps a *uniform* baseline over the whole bed. It does **not**
   test, and does not refute, *graded* dopant (the established FGM benefit of putting dopant where the
   part needs it). A uniform floor just raises absorption everywhere, including the parasitic bed.
3. **Bed/part geometry.** The jared chamber (62×20 mm) is much larger than the 40×10 mm part, which
   maximizes parasitic bed loss. A tighter bed, or a full build where bed heat preheats neighboring
   parts, changes the economics — bed heat is not necessarily "wasted" in a real build.
4. **Provenance of the σ/ε(wt%) law** is `[ASSUMED linear, no percolation]` (`premix.py`,
   `premix_frac_from_wtpct`); the real curve has a percolation toe below ~15 wt%, so low-wt% σ is
   overstated. wt% here is dopant-to-nylon in the MATRIX/bed, not in an ink.

## Not done (honest scope)
- **Per-level map re-optimization (#1 in the plan):** the cheap 2-D adjoint (`fgm_solve_campaign/adjoint2d`)
  is a separate solver without premix; wiring premix there is a follow-up. Here the printed map is held
  fixed, which isolates the premix forward effect.
- **Voltage-drive counterpart** (caveat 1) and **heatr3d 3-D cross-check** (plan Task 8 step 3).
- Loop re-solve blocks (turntable/FGM-iterate) remain premix-unaware; the study uses `update_interval:0`.
