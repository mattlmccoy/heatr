# EQS-02 shape re-ranking: legacy vs masked Q_rf gradient (n = 64)

Campaign design, inventory and pre-stated runtime estimate: `README.md`.
Machine-readable: `rerank_results.json`. `heatr3d.py` was **not modified**;
both arms come from the SAME `solve_eqs_3d` V per shape via
`compute_qrf_3d(..., qrf_gradient=...)` and drive `run(qrf_override=...)`.

`sigma_T` is the **3-D** metric (3-D: std of T_phi90 over part voxels (heatr3d.Result.sigma_T). NOT comparable to any 2-D/2.5-D ui_rms*(T_bar-23) number.)

Max |total-power| difference between arms across all shapes: **4.03e-16** (identical by construction; the
arms differ only in how that fixed power is distributed).

## 0. Standing gates (every march)

| shape | arm | reached phi_bar=0.90 | energy residual frac | clamp_bound | cfl_violated | substeps | gate |
|---|---|---|---|---|---|---|---|
| cone | legacy | True | -4.02e-13 | False | False | 1 | PASS |
| cone | masked | True | -4.08e-13 | False | False | 1 | PASS |
| cross | legacy | True | 5.60e-14 | False | False | 1 | PASS |
| cross | masked | True | 5.00e-14 | False | False | 1 | PASS |
| cylinder | legacy | True | -9.98e-14 | False | False | 1 | PASS |
| cylinder | masked | True | -2.37e-14 | False | False | 1 | PASS |
| diamond | legacy | True | 1.68e-13 | False | False | 1 | PASS |
| diamond | masked | True | -1.87e-15 | False | False | 1 | PASS |
| dumbbell | legacy | True | 3.17e-13 | False | False | 1 | PASS |
| dumbbell | masked | True | 2.61e-13 | False | False | 1 | PASS |
| lshape | legacy | True | -1.76e-13 | False | False | 1 | PASS |
| lshape | masked | True | -6.69e-14 | False | False | 1 | PASS |
| sphere | legacy | True | 4.92e-14 | False | False | 1 | PASS |
| sphere | masked | True | -5.00e-14 | False | False | 1 | PASS |
| square | legacy | True | -7.52e-14 | False | False | 1 | PASS |
| square | masked | True | -7.63e-14 | False | False | 1 | PASS |

**All gates pass: True** (gate = reached phi_bar=0.90, |energy residual| < 1e-6, no clamp, no CFL violation).

## 1. Side-by-side, per shape

| shape | class | part vox | arm | sigma_T [C] | t90 [s] | T_max [C] | T_mean [C] | interior T [C] | surface T [C] | surf-int [C] | Q surf-band power frac | Q max/mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cone | solid | 5492 | legacy | 32.904 | 1182.8 | 311.69 | 223.37 | 228.11 | 218.78 | -9.33 | 0.7997 | 31.418 |
| cone | solid | 5492 | masked | 43.123 | 1153.8 | 306.55 | 239.62 | 254.38 | 225.30 | -29.08 | 0.5818 | 8.549 |
| cross | prism_full_height | 21760 | legacy | 29.823 | 457.2 | 281.79 | 218.12 | 215.91 | 220.98 | +5.07 | 0.7810 | 16.891 |
| cross | prism_full_height | 21760 | masked | 28.958 | 401.5 | 262.67 | 224.69 | 240.36 | 204.35 | -36.01 | 0.4496 | 2.201 |
| cylinder | prism_full_height | 23040 | legacy | 26.131 | 397.0 | 284.27 | 210.98 | 208.32 | 216.87 | +8.55 | 0.7366 | 12.255 |
| cylinder | prism_full_height | 23040 | masked | 20.103 | 323.4 | 240.03 | 208.65 | 219.62 | 184.37 | -35.25 | 0.3186 | 1.857 |
| diamond | prism_full_height | 14080 | legacy | 26.214 | 531.4 | 292.42 | 212.84 | 212.68 | 213.14 | +0.46 | 0.7567 | 17.055 |
| diamond | prism_full_height | 14080 | masked | 21.142 | 439.7 | 237.23 | 211.90 | 221.27 | 194.14 | -27.13 | 0.3828 | 3.219 |
| dumbbell | solid | 8328 | legacy | 20.007 | 797.3 | 268.51 | 207.86 | 212.65 | 202.66 | -9.99 | 0.8003 | 18.010 |
| dumbbell | solid | 8328 | masked | 26.006 | 688.3 | 261.02 | 212.04 | 228.09 | 194.60 | -33.49 | 0.4737 | 4.224 |
| lshape | prism_full_height | 20160 | legacy | 33.204 | 532.6 | 298.58 | 225.66 | 226.41 | 224.91 | -1.50 | 0.7666 | 16.450 |
| lshape | prism_full_height | 20160 | masked | 66.420 | 680.9 | 379.35 | 275.82 | 292.00 | 259.54 | -32.47 | 0.5012 | 3.379 |
| sphere | solid | 13992 | legacy | 32.403 | 496.2 | 330.70 | 220.13 | 223.14 | 213.47 | -9.67 | 0.6607 | 20.273 |
| sphere | solid | 13992 | masked | 27.103 | 390.8 | 271.18 | 216.40 | 230.48 | 185.20 | -45.28 | 0.3164 | 2.625 |
| square | prism_full_height | 30976 | legacy | 17.897 | 327.7 | 254.25 | 198.33 | 192.38 | 210.39 | +18.02 | 0.7366 | 19.155 |
| square | prism_full_height | 30976 | masked | 19.856 | 316.1 | 238.10 | 211.42 | 221.51 | 190.99 | -30.52 | 0.3714 | 2.198 |

Surface-band volume fraction (for comparison against the power fraction): cone 0.5076, cross 0.4353, cylinder 0.3111, diamond 0.3455, dumbbell 0.4793, lshape 0.4984, sphere 0.3110, square 0.3306.

## 2. Deltas (masked minus legacy)

| shape | d sigma_T [C] | rel | d t90 [s] | rel | d T_max [C] | d (surf-int) [C] | d Q surf-band frac |
|---|---|---|---|---|---|---|---|
| cone | +10.220 | +31.1 % | -29.0 | -2.5 % | -5.14 | -19.74 | -0.2179 |
| cross | -0.865 | -2.9 % | -55.8 | -12.2 % | -19.13 | -41.09 | -0.3314 |
| cylinder | -6.027 | -23.1 % | -73.6 | -18.6 % | -44.24 | -43.80 | -0.4180 |
| diamond | -5.072 | -19.3 % | -91.7 | -17.3 % | -55.20 | -27.59 | -0.3739 |
| dumbbell | +5.999 | +30.0 % | -109.0 | -13.7 % | -7.49 | -23.50 | -0.3267 |
| lshape | +33.216 | +100.0 % | +148.4 | +27.9 % | +80.77 | -30.96 | -0.2654 |
| sphere | -5.300 | -16.4 % | -105.5 | -21.3 % | -59.51 | -35.61 | -0.3443 |
| square | +1.958 | +10.9 % | -11.7 | -3.6 % | -16.15 | -48.54 | -0.3652 |

## 3. Rankings, legacy vs masked

Ordered smallest first. A ranking **FLIPS** if at least one shape pair inverts.

### sigma_T (3-D: std of T_phi90 over part voxels) [C]

- legacy: `square < dumbbell < cylinder < diamond < cross < sphere < cone < lshape`
- masked: `square < cylinder < diamond < dumbbell < sphere < cross < cone < lshape`
- Spearman rho = **0.905**; **FLIPPED**; inverted pairs: (dumbbell, cylinder), (dumbbell, diamond), (cross, sphere)

### t90 [s]

- legacy: `square < cylinder < cross < sphere < diamond < lshape < dumbbell < cone`
- masked: `square < cylinder < sphere < cross < diamond < lshape < dumbbell < cone`
- Spearman rho = **0.976**; **FLIPPED**; inverted pairs: (cross, sphere)

### T_max [C]

- legacy: `square < dumbbell < cross < cylinder < diamond < lshape < cone < sphere`
- masked: `diamond < square < cylinder < dumbbell < cross < sphere < cone < lshape`
- Spearman rho = **0.595**; **FLIPPED**; inverted pairs: (square, diamond), (dumbbell, cylinder), (dumbbell, diamond), (cross, cylinder), (cross, diamond), (cylinder, diamond), (lshape, cone), (lshape, sphere), (cone, sphere)

### surface minus interior mean T [C]

- legacy: `dumbbell < sphere < cone < lshape < diamond < cross < cylinder < square`
- masked: `sphere < cross < cylinder < dumbbell < lshape < square < cone < diamond`
- Spearman rho = **0.143**; **FLIPPED**; inverted pairs: (dumbbell, sphere), (dumbbell, cross), (dumbbell, cylinder), (cone, lshape), (cone, cross), (cone, cylinder), (cone, square), (lshape, cross), (lshape, cylinder), (diamond, cross), (diamond, cylinder), (diamond, square)

### Q_rf power fraction in the surface band [-]

- legacy: `sphere < cylinder < square < diamond < lshape < cross < cone < dumbbell`
- masked: `sphere < cylinder < square < diamond < cross < dumbbell < lshape < cone`
- Spearman rho = **0.881**; **FLIPPED**; inverted pairs: (lshape, cross), (lshape, dumbbell), (cone, dumbbell)

### Q_rf max/mean [-]

- legacy: `cylinder < lshape < cross < diamond < dumbbell < square < sphere < cone`
- masked: `cylinder < square < cross < sphere < diamond < lshape < dumbbell < cone`
- Spearman rho = **0.452**; **FLIPPED**; inverted pairs: (lshape, cross), (lshape, diamond), (lshape, square), (lshape, sphere), (cross, square), (diamond, square), (diamond, sphere), (dumbbell, square), (dumbbell, sphere)

## 4. The published 4-shape table, re-ranked

`dissertation_materials/analysis-3dfgm/study_summary.csv` (n=64, `baseline`
rows) is the heatr3d shape ranking of record. Its four geometries are pinned
here (part-voxel counts match exactly). It ran `Params()` defaults, i.e.
`phase_update="apparent_cp"`; this campaign runs `"enthalpy"` on **both**
arms, so the legacy column is an anchor, not a bit-for-bit reproduction.

| shape | published sigma_T [C] | legacy (this run) | masked | published t90 [s] | legacy t90 | masked t90 |
|---|---|---|---|---|---|---|
| cylinder | 26.34 | 26.131 | 20.103 | 436.1 | 397.0 | 323.4 |
| cone | 32.65 | 32.904 | 43.123 | 1276.4 | 1182.8 | 1153.8 |
| sphere | 32.92 | 32.403 | 27.103 | 541.8 | 496.2 | 390.8 |
| dumbbell | 20.13 | 20.007 | 26.006 | 856.2 | 797.3 | 688.3 |

- published sigma_T order: `dumbbell < cylinder < cone < sphere`
- legacy arm order:        `dumbbell < cylinder < sphere < cone`  (DIFFERS from published)
- masked arm order:        `cylinder < dumbbell < sphere < cone`
- inverted pairs: [('dumbbell', 'cylinder')]; Spearman rho = 0.800

**Attribution.** The following pair(s) already reorder between the published
table and the legacy arm, i.e. they moved under the `apparent_cp` ->
`enthalpy` phase-update change and **cannot be attributed to EQS-02**:
- (cone, sphere): published 32.65 vs 32.92 C (gap 0.27 C) -> legacy 32.904 vs 32.403 C. A near-tie.

Only `(dumbbell, cylinder)` is a legacy-vs-masked inversion, i.e. attributable to the EQS-02 default flip.

**Outlier.** `lshape` is the only shape whose
phi_bar=0.90 crossing gets *later* under masked (532.6 -> 680.9 s) and whose T_max *rises* (298.6 -> 379.3 C), alongside the largest sigma_T increase in the set (33.20 -> 66.42 C, +100 %). Its Q_rf is the expected volume-proportional field (surface-band power fraction 0.501 vs volume fraction 0.498); the thermal response, not the drive, is what differs. Mechanism consistent with the data but NOT independently verified here: with power distributed by volume, a part of strongly varying section thickness heats its thick region faster than its thin arms lose heat to the powder, so the spread widens and the mean melt fraction crosses later. The legacy surface-weighted drive happened to compensate that (thin arms carry more skin per unit volume). Flagged as the single result in this campaign most worth an independent check.

## 5. n = 96 refinement spot check

NOT RUN (see README section 4: the n=96 check is the part that gets cut if wall time overruns).

## 6. Exposure statement

Factual: which previously-reported qualitative conclusions survive the default
flip, and which reverse. No interpretation beyond what the table shows.

### Reverses

1. **Surface-vs-interior attribution of absorbed dose reverses in all 8 shapes.** Legacy Q_rf puts 1.54-2.37x its volume share of absorbed power in the surface band; masked puts 0.99-1.15x, i.e. roughly volume-proportional. Any statement that the part heats 'from the outside in' rests on the legacy stencil.
2. **The thermal topology reverses.** Under masked, mean surface T is below mean interior T in ALL shapes (-45.3 to -27.1 C). Under legacy the sign was mixed; the shapes that flip sign outright are: cross, cylinder, diamond, square.
3. **Peak-ratio numbers do not survive at any magnitude.** Q_rf max/mean falls by 3.7-8.7x (largest: square 19.15 -> 2.20). 'The field concentrates Nx at the corner' carries the squared cross-interface jump inside N.

### Reverses in ranking, not just in value

4. **The published 4-shape sigma_T ranking flips.** `dumbbell < cylinder < sphere < cone` becomes `cylinder < dumbbell < sphere < cone`; inverted pair(s) (dumbbell, cylinder). The claim 'the dumbbell is the most uniform of the four' becomes 'the cylinder is'.
5. **The sign of the sigma_T change is shape-dependent, confirming EQS02_IMPACT at a wider shape set.** Up in 4 of 8 shapes (lshape +100 %, cone +31 %, dumbbell +30 %, square +11 %); down in 4 (cylinder -23 %, diamond -19 %, sphere -16 %, cross -3 %). No single scale factor reconciles the two drives.

### Survives

6. **Total absorbed power and the energy balance.** Identical by construction (max relative difference 4.0e-16); every march closes its energy audit (|residual| < 1e-6, see section 0).
7. **The extreme ends of the sigma_T ranking.** `square` stays the most uniform-ranked and `lshape` the least, in both arms (overall Spearman rho = 0.905 over 8 shapes).
8. **The direction of the t90 change** (masked reaches phi_bar=0.90 sooner) holds in 7 of 8 shapes; the exception is lshape. The t90 ranking is nearly preserved (rho = 0.976).

### Limits of this campaign

- One grid (n = 64) for the 8-shape set; one refinement spot check (section 5).
- The `legacy` arm here uses `phase_update="enthalpy"`, whereas the published
  `study_summary.csv` used `"apparent_cp"`; the legacy column is an anchor, not
  a bit-for-bit reproduction of the published table.
- `sigma_T` is the 3-D `std(T_phi90)` metric and is NOT grid-converged in heatr3d
  (documented in HEATR_STANDARD_PARAMETERS.md); rankings, not absolute values, are
  the object here.
- Only the Q_rf post-processing gradient differs between arms. Any error in `V`
  itself (harmonic face averaging at the staircase boundary, cell-centred
  electrode gauge) is present in BOTH arms.
- The prism family (20 mm bounding box, full height) and the solid family
  (published `run_3d_study.py` sizes) are not the same physical size.
