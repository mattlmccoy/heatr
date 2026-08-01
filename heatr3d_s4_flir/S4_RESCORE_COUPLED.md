# S4 re-score with in-march EQS re-solve + sigma(T) coupling

Date: 2026-08-01. Follows `S4_GATE_REPORT.md` §4.2 / §7 items 1-2. All numbers in
`results_coupled.json`, fields in `fields_coupled.npz`, figure
`figs/coupled_topology.png`. Driver `s4_rescore_coupled.py`, merge/figure
`s4_rescore_merge.py`. The pre-registration (`README.md`) is untouched; the
measured side is loaded verbatim from the committed `results.json` /
`fields.npz` and no `.seq` was re-decoded.

## Verdict

**GATE S4: STILL NOT PASSED, and the frozen-Q_rf hypothesis is now largely
falsified as the explanation of the late-time topology miss.**

Three findings, in decreasing strength of evidence:

1. **Un-freezing Q_rf, by itself, changes exactly nothing.** With
   `eqs_update_interval_s` armed but both sigma coefficients at zero, 21 (case A)
   / 14 (case B) additional full EQS solves reproduce the frozen-drive baseline
   **to the last reported digit** on every metric (r, chamfer, M2, topology).
   This is not a null result to be explained away, it is arithmetic: with no
   sigma(T) / sigma(rho) law and no geometry evolution, gamma never moves, so
   re-solving returns the same field. Candidate mechanism 1 of §4.2 ("Q_rf is
   frozen") **cannot be an independent cause**; it can only act through a
   mechanism that changes the material.
2. **A positive sigma(T) makes the agreement monotonically WORSE, in both
   cases, on every gating metric, and moves the topology diagnostic the WRONG
   way.** Case A r(95 %) +0.687 -> +0.646 -> +0.567 as a goes 0 -> +0.002 ->
   +0.010 /K; case B +0.512 -> +0.483 -> +0.403. The positive branch is
   falsified.
3. **A NEGATIVE sigma(T) helps the pattern metrics and moves the topology
   diagnostic toward the observation, but nowhere near far enough, and the
   magnitude that would close the topology gap is numerically unsound.** At
   a = -0.002 /K case A's late-state correlation rises +0.687 -> +0.730 (into
   the pre-registered "strong" band) and case B's M1 goes from FAIL to PASS at
   both states (+0.492/+0.512 -> +0.691/+0.552) with M3 also passing
   (2.66/2.80 mm) -- but case A's M3 still fails (7.69 mm vs the 6 mm threshold)
   and **case B's discrimination margin M2 collapses to +0.099, below the 0.15
   threshold**. No coefficient makes both cases pass. The centre-hot plateau is
   not reproduced by any valid setting (case A cmr -0.017 -> +0.006 against a
   measured +0.399) and the diagonal bands are not reproduced **at all**
   (case B dmq stays at -0.02 to -0.03 against a measured +0.366, at every
   coefficient of either sign).

One-line statement: **the late-time topology miss is not explained by a frozen
Q_rf, and is not repaired by a uniform-dopant sigma(T) feedback of either sign at
any coefficient inside the law's own validity domain; the remaining §4.2
candidates (densification coupling, and above all dopant non-uniformity) are now
the live ones.**

## 1. What was added to the model

`heatr3d.Params` gained `eqs_update_interval_s`, `sigma_temp_coeff_per_K`,
`sigma_density_coeff`, `sigma_ref_temp_c`, `eqs_resolve_drift_rtol`, all
defaulting to the legacy values, with `eqs_update_interval_s = 0.0` as the master
switch. `run()` gained `t_start_s` (default 0.0) so a chained march keeps one
global re-solve schedule. `Result` reports `n_eqs_solves` /
`n_eqs_resolves_skipped`. Commit `699ed79`, 8 new tests in
`test_heatr3d_s1.py` (red before green), 42 tests green.

**The ported law and its source.** From `rfam_eqs_coupled.py`
`_FGMFeedback.sigma_at_mask` (lines 418-449) and its `update_interval` call site
(lines 3038-3046), verbatim including the clip bounds:

```
sigma_eff = clip( sigma_local * (1 + a (T - T_ref)) * (1 + b (rho_rel - rho_ref)),
                  1e-4 * sigma_doped,  25 * sigma_doped )
```

with `rho_ref = p.rho_rel` (the 2-D solver's `rho_rel_init`). Two documented
differences from the 2-D form: the factor multiplies the LOCAL baseline sigma
(so `edge_width_m`, premix and the FGM `sat` map survive; identical to the flat
2-D form in the S4 configuration, where all three are off) and only the REAL part
of gamma is coupled -- **eps_r(T) feedback is still NOT modelled**, exactly as in
the 2-D solver. Re-solves reuse the same masked-gradient stencil
(`qrf_gradient="masked"`) and the same fixed-absorbed-power renormalization, so
total absorbed power is constant across a march by construction.

## 2. Coefficient discipline (why this is not circular)

There is **no validated nonzero coefficient anywhere in this repository.** Every
current 2-D config sets both to 0.0; the only nonzero values are archived
(`configs/_archive_old/rfam_eqs_comsol_mimic.yaml` l. 83-85:
`sigma_temp_coeff_per_K: 0.002`, `sigma_density_coeff: 0.6`,
`sigma_ref_temp_c: 23.0`), never validated against anything.

The sweep was therefore fixed on physical grounds **before any score was read**,
and explicitly NOT fitted to the FLIR frames it is scored against -- fitting the
coupling to the scoring data would make the whole exercise circular:

| a [1/K] | why this value |
|---|---|
| 0.0 | control: mechanism 1 alone (Q_rf un-frozen, material unchanged) |
| +0.002 | the archived 2-D value, the only number that exists in-repo |
| +0.010 | 1 %/K, the order-of-magnitude upper end for a thermally activated (hopping) carbon-black/PA12 composite |
| -0.010 | the opposite sign: the PTC branch, where thermal expansion breaks the percolation network |

Two further points were added **after** the first pass, for a stated reason that
is not "it scored better" (both are recorded as deviations below):

| a [1/K] | why added |
|---|---|
| -0.002 | a = -0.010 left the LINEAR LAW'S OWN VALIDITY DOMAIN (see §4). -0.002 is the archived magnitude with the sign flipped, the largest in-validity negative point at that scale |
| -0.004 | one more point on the negative branch, to test whether the trend continues or the march degenerates. POST-HOC and labelled as such |

`sigma_density_coeff` is held at 0.0 and that is **not a choice**: the
pre-registered S4 march runs `densify=False`, so `rho_rel` is constant for the
whole march and the density term is provably inert. **Candidate mechanism 3
(densification coupling into the EQS) is NOT tested here.**

## 3. Scores against the pre-registered thresholds

Metrics, registration operator and thresholds are the pre-registered ones,
computed by the same `s4_flir_lib` functions. Plane P only (plane S is VOID per
§4.4 of the gate report). Thresholds: M1 `r >= 0.50` at BOTH states,
M3 `chamfer <= 6 mm`, M2 `margin >= +0.15`.

### Case A (`010320_exp1`, primary anchor)

| a [1/K] | P_abs W | r 60 % | r 95 % | chamfer 60 % | chamfer 95 % | M2 margin | cmr 95 % | dmq 95 % | gates |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 15.66 | +0.826 | +0.687 | 0.55 | **8.19** | +0.246 | -0.017 | -0.021 | clean |
| 0.0 (re-solve only) | 15.66 | +0.826 | +0.687 | 0.55 | **8.19** | +0.246 | -0.017 | -0.021 | clean |
| +0.002 | 15.67 | +0.818 | +0.646 | 0.68 | **8.57** | +0.233 | -0.025 | -0.018 | clean |
| +0.010 | 15.66 | +0.789 | +0.567 | 0.84 | **9.30** | +0.205 | -0.036 | -0.013 | clean |
| -0.002 | 15.64 | +0.826 | **+0.730** | 0.48 | **7.69** | +0.254 | -0.006 | -0.025 | clean |
| -0.004 | 15.61 | +0.811 | **+0.743** | 0.42 | **7.26** | +0.241 | +0.006 | -0.028 | clean |
| -0.010 | 15.44 | +0.191 | +0.156 | 8.81 | 13.34 | +0.148 | +0.031 | +0.001 | **VOID** (clamp bound; energy residual +1.3e-2 > 1e-2) |
| **measured** | | | | | | | **+0.399** | **+0.176** | |

Published values for the frozen baseline were +0.822 / +0.690 / 8.14 mm /
M2 +0.246; this re-run reads +0.826 / +0.687 / 8.19 mm / +0.246 at 21 checkpoints
instead of 40 (deviation R1). That agreement is the pipeline check: the re-score
harness reproduces the published gate to <0.005 in r.

M3 fails at the 95 % state for **every** coefficient. Case A does not pass.

### Case B (`010320_exp5`, independent repeat)

| a [1/K] | P_abs W | r 60 % | r 95 % | chamfer 60 % | chamfer 95 % | M2 margin | cmr 95 % | dmq 95 % | gates |
|---|---|---|---|---|---|---|---|---|---|
| frozen baseline | 14.10 | **+0.492** | +0.512 | 3.24 | 3.24 | +0.176 | -0.046 | -0.020 | clean |
| 0.0 (re-solve only) | 14.10 | **+0.492** | +0.512 | 3.24 | 3.24 | +0.176 | -0.046 | -0.020 | clean |
| +0.002 | 14.11 | **+0.437** | +0.483 | 3.46 | 3.63 | +0.210 | -0.060 | -0.018 | clean |
| +0.010 | 14.12 | **+0.327** | +0.403 | 3.77 | 4.15 | +0.256 | -0.081 | -0.015 | clean |
| -0.002 | 14.09 | +0.691 | +0.552 | 2.66 | 2.80 | **+0.099** | -0.019 | -0.029 | clean |
| -0.010 | 13.96 | +0.373 | +0.367 | 7.47 | 5.40 | -0.161 | +0.556 | +0.013 | **VOID** (clamp bound) |
| **measured** | | | | | | | **+0.295** | **+0.366** | |

Published frozen baseline: +0.463 / +0.508 / 3.24 mm / M2 +0.181; this re-run
reads +0.492 / +0.512 / 3.24 / +0.176. The 60 %-state r moves by 0.029 with the
checkpoint count because the matched-state frame selection is interpolated on a
14-point instead of a 40-point curve; it stays below 0.50 either way, so case B's
published M1 failure is unchanged.

At a = -0.002 case B's M1 and M3 both pass -- and its M2 margin drops to +0.099,
**below the +0.15 threshold**. The prediction has become more correlated with the
untuned run and more correlated with the tuned control at the same time, i.e. it
has become less specific. Under the gate's own verdict rule that is not a pass.

### The one fitted scalar

Refitted with coupling on, by the same pre-registered 60 s procedure (deviation
R2). It barely moves: 15.66 -> 15.44-15.67 W (case A), 14.10 -> 13.96-14.12 W
(case B). **The coupling effect reported above is not a disguised power effect.**

## 4. The validity boundary of the ported linear law (new numerical finding)

The law is linear in `T - T_ref`, so it is only meaningful while
`|a| * dT_max < 1`. Case A reaches a part max of 249 C from a 21.7 C ambient,
dT_max = 227 K:

| a | a * dT_max | factor 1 + a dT |
|---|---|---|
| +0.002 | +0.45 | 1.45 |
| +0.010 | +2.27 | 3.27 (no sign change; monotone, clip never reached) |
| -0.002 | -0.45 | 0.55 |
| -0.004 | -0.91 | 0.09 (near-degenerate but positive) |
| **-0.010** | **-2.27** | **-1.27 -> sigma pinned at the 1e-4 clip floor over the whole hot region** |

At a = -0.010 the conductivity of the hot interior collapses by four orders of
magnitude, the field re-concentrates into a small residual blob (visible in
`figs/coupled_topology.png`, right column), and the march **fails the standing
gates**: case A `clamp_bound=True` with `energy_residual_frac = +1.3e-2` (above
the 1e-2 gate), case B `clamp_bound=True`. Per the pre-registration ("any march
failing these is reported as a failed march, not scored") both a = -0.010 rows
are reported **VOID** and are excluded from every conclusion, exactly as the
plane-S scores were in the original report. They are shown only because the value
was in the pre-declared sweep.

The practical consequence for anyone using this feature: **`|sigma_temp_coeff_per_K|`
must satisfy `|a| * (T_max - T_ref) < 1`**, i.e. `|a| < 0.0044 /K` for a march
of this temperature span. A saturating (e.g. Arrhenius or tanh) law would be the
correct next port if a large-magnitude coefficient is ever needed.

## 5. Does the topology move in the observed direction?

Two non-pre-registered, non-gating diagnostics were added for this question
(deviation R3), both on the normalized rise field at the 95 % state:

* `centre_minus_ring` (**cmr**) = mean(r<0.20) - mean(0.33<r<0.50). Positive =
  centre-hot plateau, which is case A's measured late topology.
* `diagonal_minus_quadrant` (**dmq**) = mean(the two diagonals, |‖u|-|v‖|<0.08) -
  mean(quadrant interiors). Positive = the bright diagonal "X", which is case B's.

| | case A | case B |
|---|---|---|
| measured, 95 % state | cmr **+0.399**, dmq +0.176 | cmr +0.295, dmq **+0.366** |
| frozen prediction | cmr -0.017, dmq -0.021 | cmr -0.046, dmq -0.020 |
| best VALID coupled prediction | cmr +0.006 (a=-0.004) | cmr -0.019 (a=-0.002) |
| VOID (a=-0.010) | cmr +0.031 | cmr +0.556 |

**Direction: yes, weakly. Magnitude: no.** The negative branch moves cmr the
right way monotonically, but the best valid setting closes ~6 % of case A's gap
(-0.017 -> +0.006 against +0.399). The only run that reaches or overshoots the
measured centre-hotness (case B, a=-0.010, cmr +0.556) is the numerically unsound
one, and it destroys the correlation (r +0.367) -- it is not a centre-hot plateau,
it is a collapsed field.

**dmq never becomes positive in any valid run, in either case.** The diagonal
banding of case B is not produced by a uniform-dopant sigma(T) feedback of any
sign or magnitude tried here. That is the clearest single result of this
re-score, and it points squarely at §4.2 candidate 4 (the parts were PRINTED;
real dopant non-uniformity and a print-path signature are not in the model).

## 6. Cost (EQS-01)

196 full EQS solves, 7414 s of wall time for the 13 marches, on a machine whose
load average was 10-14 throughout (a single EQS solve at this grid measured
30.1 s idle, 35-43 s under that load).

| | frozen baseline | coupled |
|---|---|---|
| case A (615.5 s record) | 1 cached solve, 56-63 s wall | 21 solves + fit, 714-906 s wall |
| case B (429.9 s record) | 1 cached solve, 39 s wall | 14 solves + fit, 495-558 s wall |

A coupled march costs **11-15x** the frozen one. The re-solve cadence was set
equal to the checkpoint spacing (~30 s) so that each segment's mandatory pre-loop
solve IS the scheduled re-solve and no solve is wasted; consequently the
`eqs_resolve_drift_rtol` skip path was never exercised in these runs
(`n_eqs_resolves_skipped = 0` everywhere). Its correctness is covered by
`test_drift_tolerance_skips_resolves_and_reports_the_census`, not by this study.

## 7. Standing gates

Clean on every scored march: `|energy_residual_frac| <= 4.5e-15`,
`clamp_bound=False`, `cfl_violated=False`, `n_substeps_used=1`. The energy audit
holds to machine precision **through a drive that changes 14-21 times mid-march**,
which is the specific thing the audit had not previously been exercised against.
The two failing marches are the a = -0.010 pair (§4), reported VOID.

## 8. Deviations from the original S4 driver

| # | deviation | reason |
|---|---|---|
| R1 | 20/14 checkpoints instead of 40 | each chained segment now begins with a full EQS solve, so the segment length is set equal to the re-solve interval to avoid wasted solves. The frozen baseline was RE-RUN at the same checkpoint count so the comparison isolates the coupling; it reproduces the published scores to <0.03 in r (<0.005 for case A) |
| R2 | the one fitted scalar is refitted with coupling on | the coupled model is a different model; refitting by the SAME pre-registered 60 s procedure keeps "one fitted scalar" true. It moves by <1.5 % |
| R3 | two topology diagnostics added (cmr, dmq) | the gate report states its failure in topological words; r alone cannot say WHICH WAY the field moved. Non-pre-registered and NON-GATING |
| R4 | a = -0.002 and a = -0.004 added after the first pass | -0.002 because a = -0.010 left the linear law's validity domain (§4); -0.004 to test whether the negative-branch trend continues. Post-hoc, labelled, and reported in full including the metrics they fail |
| R5 | case A's positive-coefficient 64x64 registered FIELDS were not retained | the first shard was killed by the harness before its `.npz` write. The SCORES are intact in `results_coupled.json`; only the images are missing, and the figure says so in those panels rather than leaving a blank. The driver now saves after every case |
| R6 | case C was not re-run coupled | it is the discrimination control, scored through its stored measured field; re-running it coupled would add ~10 min/coefficient and change no verdict |

No threshold changes, no case substitutions, no metric removed from the verdict
rule.

## 9. What this changes in `S4_GATE_REPORT.md` §7

The priority list was: (1) re-solve the EQS, (2) add sigma(T)/sigma(rho),
(3) resolve the 1.4x scale question. Items 1 and 2 are now done and largely
answered in the negative. The revised order:

1. **Dopant non-uniformity** (§4.2 candidate 4). It is the only remaining
   mechanism that can produce case B's diagonal bands, which nothing here moved
   at all, and the parts were printed. Needs a print-path or a measured/assumed
   dopant field, not more solver physics.
2. **Densification coupling into the EQS** (§4.2 candidate 3), which this study
   could not test because the pre-registered march holds `rho_rel` fixed. The
   machinery is in place (`sigma_density_coeff`); it needs a march with
   `densify=True`, which is a change to the pre-registered thermal model and so a
   separate, explicitly re-registered study.
3. **The 1.4x scale question**, unchanged: it is the difference between M3 being
   indicative and quantitative, and no amount of solver work touches it.
4. A saturating sigma(T) law, if and only if a large-magnitude coefficient is
   ever motivated by a measurement (§4).

## 10. Reproducing this

```bash
cd geo-prewarp
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m pytest \
    test_heatr3d_s1.py -q                        # 30 tests incl. 8 coupling tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -u \
    heatr3d_s4_flir/s4_rescore_coupled.py        # ~2 h: 4 coefficients x 2 cases
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -u \
    heatr3d_s4_flir/s4_rescore_coupled.py --cases A B --coeffs -0.002 \
    --out heatr3d_s4_flir/results_coupled_neg.json \
    --fields-out heatr3d_s4_flir/fields_coupled_neg.npz
./.venv312/bin/python heatr3d_s4_flir/s4_rescore_merge.py   # merge + figure
```

Pin the BLAS thread count (the original report measured a 26 s solve becoming an
18 min one without it). `cache/qrf_square_n96.npz` is reused unchanged.

Read-only inputs, untouched by this work: `README.md` (the pre-registration),
`results.json`, `fields.npz`, `s4_run.py`, `s4_flir_lib.py`, the `JaredFiles`
archive, and `dissertation_materials/`.
