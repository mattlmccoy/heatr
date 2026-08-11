# Constant-Magnitude Proportional FGM Sweep - Report

Date: 2026-08-11. Circle 20 mm, HEATR 2-D grid 120, 4 bpp, proxy T_phi90,
dead_band 0.02, baseline_saturation 0.5. Drive on every number in this report:
**enforced generator power, 500 W set-point at 2 % transfer efficiency, 27.12 MHz,
grounded boundary electrodes (860 V nominal), exposure t\* = 6.308 min
(optimizer-chosen at the phi = 0.90 crossing, identical to the circle_INTEGRAL_m0\*
sweeps).** Every sigma_T in this report is the **melt-onset read**:
ui_rms_part x (mean_T_part - 23 C) at the first mean_phi >= 0.90 crossing, from
each iteration's `time_series.json` (`rfam_eqs_coupled.dual_read_state_from_hist`).
No end-of-horizon reads are used anywhere.

## Verdict (first)

1. **The attractor is a FIXED POINT at every magnitude tested. Genuine period-2
   exists, but only as a transient at m = 0.9, and it damps out.** (computed)
   - m = 0.3, 0.5: **exact** fixed point. The 4 bpp quantization freezes the map
     completely from about iteration 3: RMS(s_{k+1} - s_k) = 0 identically on the
     plateau, corr(s_k, s_{k+1}) = corr(s_k, s_{k+2}) = 1.000, sigma_T constant to
     the third decimal.
   - m = 0.7: sigma_T frozen at 10.39 C while the map executes a sub-quantum
     period-4 micro-cycle (RMS step 0.006 on a 0-1 map; the 4 bpp quantum is
     0.067). Functionally a fixed point of the temperature field; classifier
     label "neither" comes only from the non-shrinking micro-steps.
   - m = 0.8: noisy quasi-fixed point (corr lag-1 = lag-2 = 0.999, RMS step ~0.01)
     with a slow upward drift, sigma_T 9.5 -> 10.3 C over iterations 5-29.
   - m = 0.9: clear period-2 alternation, sigma_T 6.3 <-> 12.2 C, lag-2 map
     correlation 0.99 vs lag-1 0.86 through roughly iteration 15; the oscillation
     then damps (RMS step 0.25 -> 0.007) and the run settles onto a fixed point at
     15.8 +/- 0.3 C. Tail-window (last 8 iterates) verdict: fixed-point.
   - Dissertation language check: "fixed-point plateau" is the right phrase for
     the settled state at every m. "Collapses onto a single fixed-point attractor
     by iter 2" is wrong for m >= 0.8 (m = 0.9 takes ~20 iterations to settle,
     via a damped period-2 transient).

2. **The quoted 14.9-16.5 C plateau band is NOT where all magnitudes land.**
   (computed; honest negative) Plateau levels depend on m and are non-monotonic:

   | m | plateau sigma_T (C, melt onset, enforced power) | in 14.9-16.5 band? |
   |---|---|---|
   | 0.3 | 14.94 (exact freeze) | yes, bottom edge |
   | 0.5 | 13.00 (exact freeze) | no, below |
   | 0.7 | 10.39 (frozen +/- 0.01) | no, below |
   | 0.8 | 9.5 drifting to 10.3 | no, below |
   | 0.9 | 15.8 +/- 0.3 (settled tail) | yes |

   Only the two ends of the sweep (m = 0.3 and m = 0.9) sit in the band. The
   historical m = 1.0 run (`circle_CONVERGENCE_prop_m10_nodecay`, dead_band 0.0,
   6 iterations, end-of-horizon read 15.0-15.4 C) also sits in the band, which is
   probably where the band came from. The corrected statement for the chapter:
   the plateau level is magnitude-dependent; the 14.9-16.5 C band describes the
   small-m and large-m ends (m <= 0.3 and m >= 0.9), not a universal ceiling.
   Mid-magnitude proportional iteration plateaus **below** the band (9.5-13 C).

3. **Iteration 1 is NOT the sigma_T minimum at any magnitude tested; iteration 2
   is (iteration 5 at m = 0.9), but the single-step claim survives at high m.**
   (computed) Baseline iter-0 = 17.43 C in all five runs (bit-identical solves).

   | m | iter-1 | minimum (iter) | iter-1 penalty vs min |
   |---|---|---|---|
   | 0.3 | 14.96 | 14.93 (2) | 0.03 C, negligible |
   | 0.5 | 13.08 | 12.93 (2) | 0.15 C |
   | 0.7 | 10.03 | 9.17 (2) | 0.86 C |
   | 0.8 | 8.19 | 6.91 (2) | 1.28 C |
   | 0.9 | 6.29 | 6.28 (5) | 0.01 C, negligible |

   Best single step in the sweep: **6.29 C at m = 0.9 (-64 % vs baseline)**,
   consistent in regime with the dissertation's 7.05-7.1 C single-step floor
   (that number came from m = 1.0 with dead_band 0; this sweep used dead_band
   0.02 to match the integral sweeps). The refined deployability statement: one
   proportional step at high magnitude (m >= 0.9) is within 0.01 C of the best
   any iterate achieves; at mid magnitudes a second step buys up to ~1.3 C.

## Provenance labels

- **Proven (code paths read, not assumed):** proportional mode is
  `use_delta_correction=false`; with `fgm_momentum=0` the prior map is ignored
  entirely (`fgm_generator.generate_fgm` lines 348-489), i.e. the map is
  recomputed from the current iterate's T_phi90 field each iteration with
  per-iteration adaptive normalization and **no iter0_ref anchoring**
  (`rfam_gui_server.py` passes `ref_lo=None, ref_hi=None` at line ~3697).
  The melt-onset sigma_T definition is `dual_read_state_from_hist`
  (`rfam_eqs_coupled.py:109`).
- **Computed:** every number above, from the 5 x 30 run matrix (150 coupled EQS +
  thermal solves), collected into `fgm_solve_campaign/figs_prop_sweep/prop_sweep_data.json`.
- **Assumed (analysis choices):** plateau window = iterations >= 10; settled-tail
  sensitivity window = last 8 iterates; Pearson correlation over the 1240
  part-mask pixels; RMS step norm. Changing the plateau start to 12 or 15 does
  not change any verdict (the m = 0.9 tail verdict is computed separately).

## Deviations from the integral sweeps (stated, not hidden)

- `regression_threshold_multiplier=100` (vs default 10) so all 30 iterations run
  for the academic sweep; no run hit any abort anyway.
- The v2 engine re-runs iter-0 at t\* after the optimizer probe (the integral
  sweeps' iter-0 fields were at the 8-min probe exposure). Melt-onset reads are
  exposure-independent, so trajectories remain comparable.
- dead_band 0.02 matches the integral m-sweeps; the historical proportional
  m = 1.0 run used dead_band 0.0.

## Data layout (for regenerating the chapter figure)

Runs (per-iteration checkpointed, all completed 30/30, no aborts):

```
outputs_eqs/runs/circle/fgm_iterate/circle_PROPORTIONAL_m03_n30/   (m=0.3)
outputs_eqs/runs/circle/fgm_iterate/circle_PROPORTIONAL_m05_n30/   (m=0.5)
outputs_eqs/runs/circle/fgm_iterate/circle_PROPORTIONAL_m007_n30/  (m=0.7)
outputs_eqs/runs/circle/fgm_iterate/circle_PROPORTIONAL_m008_n30/  (m=0.8)
outputs_eqs/runs/circle/fgm_iterate/circle_PROPORTIONAL_m009_n30/  (m=0.9)
```

Inside each: `convergence.json` (iteration log; its `sigma_T` field is the
END-OF-HORIZON std - do not quote it), and `<name>_iterK/` with
`time_series.json` (melt-onset read source), `summary.json`
(`sigma_T_melt_onset_c` key, v2 engine), `fields.npz`, `used_config.yaml`, and
`fgm_<name>_iterK_T_phi90_4bpp_mag0pXX.npz` = the saturation map **generated
from** iter K and **applied in** iter K+1 (so the applied-map sequence is
s_k = npz of iter k-1).

Analysis artifacts (all under `fgm_solve_campaign/figs_prop_sweep/`):

- `prop_sweep_analysis.py` - pure lag-stats + verdict functions (tested).
- `test_prop_sweep_analysis.py` - red-green tests incl. the exact-quantized
  fixed-point case.
- `make_fig_proportional_sweep.py` - collector + figure; rerun with
  `./.venv312/bin/python fgm_solve_campaign/figs_prop_sweep/make_fig_proportional_sweep.py`.
- `prop_sweep_data.json` - full trajectories, per-pair lag-1/lag-2 correlations,
  step norms, verdicts (the chapter figure can be drawn from this file alone).
- `fig_proportional_sweep.png` - (a) trajectories with the quoted band,
  (b) map step size, (c) plateau lag-1 vs lag-2 with verdicts.

Launcher (exact payloads used): `launch_proportional_sweep.py` (submits to the
HEATR GUI server queue, one job at a time; per-iteration checkpointing is done
by the server's fgm_iterate loop, so an interruption costs one iteration).

## Verification performed

- All five runs: 30/30 iterations, 30 time_series.json, 30 fgm npz, no abort
  notes, jobs `completed` on the server.
- iter-0 melt-onset sigma_T identical across runs to 4 decimals (17.4284 C):
  deterministic solver, shared drive confirmed.
- iter-0 sanity (m03): dT-clip fraction 0.0, energy residual 1.18 % of injected,
  T_max 244.96 C (under the 245 C damage ceiling).
- `used_config.yaml` of iter-0 diffed against `circle_INTEGRAL_m03_n30` iter-0:
  identical physics (only YAML key order, the v2 engine stamp, and n_steps
  757 = t\* instead of the probe's 960).
- Unit tests: 5 passed (`test_prop_sweep_analysis.py`).
