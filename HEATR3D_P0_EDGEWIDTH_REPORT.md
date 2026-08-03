# heatr3d P0 Edge-Width Probe + P0b Melt-Onset Fallback Check

Date: 2026-07-30. Phases P0 and P0b of `HEATR3D_REVALIDATION_BRIEF.md`, executed and
stopped before P1 as instructed. All work run under `./.venv312/bin/python` from the
geo-prewarp root.

## Verdict, first

1. **The sign of both chapter functionally-graded-material (FGM) percentages survives
   the boundary change; the magnitudes do not.** At the pre-registered width
   w = 1e-03 m the cone benefit shrinks from -37.2% to -24.4% and the sphere benefit
   from -57.7% to -31.7% versus the uniform arm at the same width. No sign flip
   anywhere in this probe. (Computed, this session.)
2. **The slab-derived pre-registered predictions mostly failed.** The uniform-arm
   sigma_T does not rise ~2.7x (cone rises 1.19x, sphere FALLS 0.70x); the sphere
   moved MORE than the sharp cone, not least; only "percentages degrade" and
   "the FGM-beats-uniform verdict survives" held. The slab edge-width intuition is
   not portable to the chapter pipeline, which conserves total absorbed power under
   smoothing (the slab pipeline removed ~12% of the dose).
3. **P0b: the silent melt-onset fallback is real and confirmed in heatr3d**
   (final-timestep read stands in for the phi_bar = 0.90 read when the crossing never
   happens). It is now loud (warning + reported field; numerical behavior unchanged,
   test-verified). The chapter's headline sphere/dumbbell/cone rows did NOT hit it;
   the chapter's own boundary-probe figure (`fig_boundary_probe`) sphere panel hit it
   in ALL EIGHT sphere arms, and 11 of 113 graphical-user-interface runs hit it.
4. **Answer to the standing question:** the published 3-D FGM percentages survive in
   sign but not in magnitude at a resolved boundary. Every quoted 3-D number must
   carry: *"at a sharp one-cell material boundary on a single grid; a
   resolved-boundary re-score preserves the sign but reduces the magnitude (curved
   class retains roughly half to two thirds, sharp-edged extruded class as little as
   a fifth), and the physical dopant edge width is unmeasured."*

Caveat on "converged width": at n = 48 the grid spacing is h = 1.25 mm, so the
pre-registered w = 1e-03 m is w/h = 0.80, inside the aliased sub-cell band by the
dissertation's own w >= 1.5h criterion. This probe was run exactly as pre-registered
and is a sensitivity probe at that width, not a criterion-compliant converged-width
measurement. The criterion-compliant evidence at w = 2e-03 m (w/h = 1.60) exists in
the prior published-pipeline probe (square/sphere; see Section 5) and agrees in
direction with this probe.

## 1. The 8-run table

sigma_T = std(T over part voxels) in degrees C. Read state: this pipeline
(densify = False, the chapter's published cone/sphere pipeline) integrates a
monotonically heating constant-power march and STOPS at the phi_bar = 0.90 crossing.
The last solved state is therefore simultaneously the heating-peak state and the
melt-onset state; the two read states coincide by construction in every run below.
**All 8 runs reached phi_bar = 0.90 within the 1500 s horizon; no run used the
final-step fallback.**

| shape | map | width w (m) | reached phi_bar=0.90 | t90 (s) | sigma_T heating-peak (C) | sigma_T melt-onset (C) | % vs uniform (same width) | T_max (C) | Qrf CoV (%) |
|---|---|---|---|---|---|---|---|---|---|
| cone (sharp) | uniform | 0 | yes | 1242.5 | 32.035 | 32.035 | n/a | 306.8 | 187.9 |
| cone (sharp) | chapter FGM | 0 | yes | 877.6 | 20.129 | 20.129 | **-37.2%** | 236.0 | 103.2 |
| cone (sharp) | uniform | 1e-03 | yes | 1110.9 | 38.240 | 38.240 | n/a | 292.9 | 54.2 |
| cone (sharp) | chapter FGM | 1e-03 | yes | 918.8 | 28.910 | 28.910 | **-24.4%** | 262.5 | 63.7 |
| sphere (smooth) | uniform | 0 | yes | 572.5 | 38.319 | 38.319 | n/a | 344.1 | 193.1 |
| sphere (smooth) | chapter FGM | 0 | yes | 427.0 | 16.225 | 16.225 | **-57.7%** | 228.8 | 123.6 |
| sphere (smooth) | uniform | 1e-03 | yes | 430.1 | 26.975 | 26.975 | n/a | 271.2 | 28.1 |
| sphere (smooth) | chapter FGM | 1e-03 | yes | 406.5 | 18.426 | 18.426 | **-31.7%** | 231.2 | 65.8 |

Health: `clamp_bound = False` in all 8 runs (no THM-01/THM-02 limiter ever bound), so
no number above was altered by a numerical limiter. (Proven from the solver's own
manifest flag.)

Reproduction anchors (computed):
- Sphere w = 0 arms reproduce the published schedule-study baseline
  (`analysis-3dfgm/schedule_summary.csv`, "constant x1.0" rows: 38.32 / 16.22 C,
  t90 572.0 / 427.0 s) to within 0.01 C and 0.5 s. The instrument is the published
  pipeline, not a re-implementation.
- Cone w = 0 baseline (32.04 C, t90 1242.5 s at n = 48) sits within 1.9% of the
  published n = 64 row (32.65 C, t90 1276.4 s,
  `analysis-3dfgm/cone_dumbbell_fgm_results_fixed.json`). Note the percentage itself
  is grid-sensitive: -37.2% at n = 48 versus the published -31.9% at n = 64, a
  5-point move from grid alone, consistent with the known non-convergence.

## 2. Full config pins

- Solver: `geo-prewarp/heatr3d.py`, the synced copy, SHA-256 verified identical in
  body to the canonical `dissertation_materials/analysis-3dfgm/heatr3d.py`
  (`SYNCED_FROM_SHA256 = 5243529f...85ea7` matches the canonical hash; proven this
  session, before the P0b instrumentation edit described below).
- Driver: `geo-prewarp/heatr3d_p0_edgewidth/run_p0_edgewidth.py` (this session).
  Raw results: `heatr3d_p0_edgewidth/p0_results.json`, per-run fields
  `p0_<label>_fields.npz`, log `p0_run.log`.
- Grid: n = 48 cells/axis, chamber L = 0.060 m cubic, h = 1.25e-03 m.
- Timestep: dt = 0.05 s (Params default, as in every published heatr3d chapter run).
- Horizon / stop: max_time_s = 1500, densify = False, stop at phi_bar = 0.90
  (exactly `run_3d_study.py` / `run_cone_dumbbell_fgm_fixed.py`, the chapter's
  cone/sphere pipeline).
- Geometry (chapter data-of-record): cone diam = 0.024 m, zspan = 0.030 m
  (2320 voxels at n = 48); sphere diam = 0.028 m (5904 voxels).
- Drive mode: electroquasistatic (EQS) solve with plate potentials 860 V / 0 V
  setting the field SHAPE, then Qrf renormalized to a fixed total absorbed power =
  power_density (1.5915e6 W/m^3, the 10 W reference) times the binary part volume
  (`compute_qrf_3d`, heatr3d.py:266-293). Because the renormalization target uses the
  binary part volume at every width, total dose is conserved across w in this
  pipeline. 27.12 MHz, sigma_doped 0.04 S/m, eps_r doped 20.
- FGM map: the chapter's exact published procedure at this grid:
  `H.make_fgm(baseline_run, magnitude=1.0, baseline=0.5, bpp=2)` (2 bits per pixel,
  4 levels), proxy = the w = 0 uniform run's T field of the same shape. The chapter's
  own stored maps are n = 64 volumes; the brief mandates n = 48, so the map was
  regenerated by the identical procedure (assumption stated: procedure identity, not
  file identity, is the published object at this grid).
- Edge width: w = 0 (shipped binary boundary, bit-for-bit original) versus
  w = 1e-03 m via heatr3d's own `edge_width_m` error-function regularization, with
  the binary-derived map nearest-neighbour extended into the bed before blending
  (function copied verbatim from the A6 `ew_common.extend_sat_nearest`, the same
  modeling decision as the prior probe and the chapter requalification).
- w/h = 0.80 at this grid: inside the aliased band per the dissertation's
  w >= 1.5h criterion. Pre-registered, therefore run; quoted as sensitivity only.

## 3. Pre-registered predictions, scored

| # | prediction | result | evidence |
|---|---|---|---|
| 1 | absolute sigma_T rises ~2.7x at w = 1e-03 | **FAIL** | uniform arms: cone 32.0 -> 38.2 C (1.19x), sphere 38.3 -> 27.0 C (0.70x, it FALLS). The 2.7x came from the slab pipeline, which removes ~12% of the dose when smoothing; heatr3d's shipped path renormalizes power to the binary part volume, so no dose is lost and the effect is field-shape only. |
| 2 | "% vs uniform" degrades sharply and may change sign | **PARTIAL PASS** | degrades: cone -37.2% -> -24.4% (loses 12.8 points, retains 66% of magnitude), sphere -57.7% -> -31.7% (loses 26.0 points, retains 55%). No sign change anywhere; the "may change sign" clause did not materialize on these shapes in this pipeline. |
| 3 | sharp shape moves most, sphere least | **FAIL** | the sphere moved MORE on every measure: uniform-arm sigma_T change 11.3 C (sphere) vs 6.2 C (cone); percentage-point loss 26.0 (sphere) vs 12.8 (cone). Mechanism (computed): the sphere's uniform-arm Qrf coefficient of variation collapses 193% -> 28% under smoothing versus the cone's 188% -> 54%, so at this width the boundary singularity was a larger share of the sphere's heating structure than expected. The prior probe's "sharp moves most" held for the extruded square at w = 2e-03 under the densify pipeline; it does not generalize to the cone here. |
| 4 | rankings survive better than percentages | **PASS, with a scope limit** | the decision-relevant ranking (graded map beats uniform, per shape, per width) survives in all 4 comparisons while percentages moved 13 to 26 points. But the cross-shape ordering does NOT survive: Spearman rho across the four arms w = 0 -> w = 1e-03 is 0.40 (cone-uniform vs sphere-uniform swap order). Rankings survive within-shape, not across shapes. |

Does the sign flip generalize off the slab? **No, not at this width, on these shapes,
in the chapter pipeline.** Both graded arms still beat uniform. The one recorded
sign flip in the published pipeline remains the square's SELF-CONSISTENTLY RECOMPUTED
map at resolved width (+82% in the chapter requalification), which is a statement
about re-deriving the map under a smoothed forward, not about re-scoring the
published map.

## 4. P0b: the melt-onset fallback

**Finding (proven, file:line):** `heatr3d.py` `run()` ends with

```
if T_phi90 is None:
    T_phi90 = T.copy()
```

at `geo-prewarp/heatr3d.py:686-687` pre-edit (identical construct in the canonical
`analysis-3dfgm/heatr3d.py`, SHA-verified same body). When phi_bar never crosses
phi_target = 0.90 within max_time_s, every downstream metric derived from `T_phi90`
(`sigma_T`, `T_max_c`, `phi_final`) silently becomes a FINAL-TIMESTEP read. The
`Result.reached` flag records the fact, and `heatr3d_job.py` writes it as
`reached_phi90`, but nothing loud marks the metrics themselves.

**Instrumentation added (loud flag, numerical behavior unchanged, red-green
tested):**
- `geo-prewarp/heatr3d.py:686-701`: a WARNING log, "MELT-ONSET FALLBACK: ... sigma_T
  / T_phi90 / T_max_c are FINAL-TIMESTEP reads, not melt-onset reads."
- `geo-prewarp/heatr3d_job.py:448-453`: new `results.json` field
  `MELT_ONSET_FALLBACK: true/false`.
- Test: `geo-prewarp/test_heatr3d_melt_fallback.py` (2 tests: warning fires when the
  crossing never happens; T_phi90/t_phi90_s/sigma_T numerics unchanged). Watched fail
  before the edit, pass after. `2 passed`.
- Scope note for the maintainer: only the geo-prewarp synced copy was edited. The
  canonical `analysis-3dfgm/heatr3d.py` is untouched (it feeds published pipelines),
  so the two copies now differ by this warning block only; propagating it and
  updating `SYNCED_FROM_SHA256` is a deliberate solver-maintainer action, flagged
  here rather than done silently.

**Which existing published runs actually hit the fallback (computed from stored
outputs):**
- **HIT, in a chapter figure:** the boundary-probe behind `fig_boundary_probe`
  (chapter requalification): ALL EIGHT sphere arms have `reached = false` in
  `analysis-3dfgm/out_adjoint_fgm_a6/meshcheck/edgewidth/probe/probe_results.json`
  (sphere uniform/graded at w = 0, 1e-03, 2e-03, both map variants). Every sphere
  sigma_T in that figure (22.97, 12.66, 21.94, 16.43, 13.86, 23.85, 17.89, 15.85 C)
  is a final-state read at the rho_bar = 0.85 densification stop; phi_bar = 0.90 was
  never crossed. The figure caption does disclose that the sphere panel stops on the
  densification criterion, applied identically in every arm, so the comparison is
  internally consistent; what was never disclosed is that the phi = 0.90 crossing
  never occurred and the `T_phi90` field name is therefore misleading for that panel.
  The square arms all reached (`reached = true`).
- **NOT hit (chapter headline rows):** `study_summary.csv` (cylinder/cone/sphere/
  dumbbell baselines + sphere fgm3d: reached = 1 in every row),
  `cone_dumbbell_fgm_results_fixed.json` (reached_base = reached_fgm = 1),
  `combination_fgm_prewarp_4bpp.json` (t_phi90_s present in all 8 arms),
  `schedule_summary.csv` (t_phi90_s recorded for every schedule including
  fast-to-soak), `densify_summary.csv`. The 2-D stacked-slice pilot table
  (`pilot_fgm_by_diameter.csv`): phi90_reached = 1 in every published row.
- **HIT, not published:** 11 of 113 graphical-user-interface runs under
  `geo-prewarp/outputs_eqs/_heatr3d/` have `reached_phi90 = false` (exploratory
  n = 32, 600 s, densify sphere/cone jobs); their `sigma_T` values (22.7 to 41.8 C)
  are final-step reads. These now get the loud `MELT_ONSET_FALLBACK` field on any
  future run.

## 5. The standing question, answered

**Do the published 3-D FGM percentages survive at a converged boundary width?**

- Sign: yes, on everything measured through the published pipelines. This probe
  (cone, sphere, melt-onset read, w = 1e-03): -37.2% -> -24.4% and -57.7% -> -31.7%.
  The prior published-pipeline probe at the criterion-compliant w = 2e-03
  (square, sphere, densification-stop read): square -60.6% -> -12.9%, sphere
  -44.9% -> -25.0% (published map) and -33.6% (recomputed map).
- Magnitude: no. Retention is class-dependent: curved/blocky class roughly one half
  to two thirds; sharp extruded class (square, and by mechanism the cylinder) as
  little as one fifth. The cone, despite its acute tip, behaves closer to the curved
  class here (retains 66%), and it was the SPHERE whose uniform baseline moved most
  at this width, so "sharp = fragile, smooth = robust" is too coarse as a law.
- The only outright inversion remains the self-consistent re-derivation of the map
  under a smoothed forward on the square (+82%): the published one-shot inverse rule
  must not be re-applied to a boundary-smoothed, nearly flat field.

**Mandatory qualifier for every quoted 3-D number:** "computed at a sharp (one-cell)
material boundary on a single grid (n = 48 or 64); a resolved-boundary re-score
preserves the sign of the effect but reduces its magnitude (curved class retains
roughly half to two thirds, sharp-edged extruded class as little as a fifth); the
boundary width is a modeling assumption and the physical dopant edge profile is
unmeasured." For any densification-stop number (the combination table and the
boundary-probe sphere panel), add: "sigma_T is read at the rho_bar = 0.85
densification stop, not at the phi_bar = 0.90 melt-onset crossing."

## 6. Proven / computed / assumed

- Proven: the fallback code path and its line numbers; the SHA-identity of the two
  heatr3d copies pre-edit; `clamp_bound = False` in all 8 probe runs; which stored
  runs carry `reached = false`.
- Computed: all 8 probe sigma_T values, percentages, ratios, Spearman rho = 0.40,
  Qrf coefficients of variation; the reproduction anchors against
  `schedule_summary.csv` and the n = 64 cone row.
- Assumed (stated): regenerating the chapter map at n = 48 by the identical published
  procedure stands in for the chapter's stored n = 64 map; w = 1e-03 itself
  (w/h = 0.80, aliased band at this grid) is a pre-registered probe width, not a
  converged one; nearest-extension of the dopant map into the graded skin (the
  A6/edgewidth modeling decision) is the right smoothing convention.

## 7. Next sensible step (not started, per the brief)

P1, the cross-engine agreement check, should pin: the densify = False melt-onset
read state (so the 2.5-D and 3-D compare like with like and the fallback cannot
enter), w reported like dt and n, and a square slab configuration whose drive mode
is matched (power-renormalized, 27.12 MHz). P0's lesson for P1: dose conservation
under smoothing differs between the slab pipeline and heatr3d's shipped path, so P1
must state which convention it uses or the comparison will inherit a hidden dose
difference.
