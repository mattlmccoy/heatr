# heatr3d S2 gate report: convergence

Plan: `docs/superpowers/plans/2026-08-02-heatr3d-s2-convergence.md`
Spec: `docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md` (Gate S2)
Pre-registration: `heatr3d_s2/results/s2_preregistration.json`, committed in
`3bd32d0` BEFORE any campaign run.
Date: 2026-08-03. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` throughout.

**Every number below is PRINTED from `heatr3d_s2/results/*.json` by
`heatr3d_s2/make_tables.py`.** Nothing is transcribed.

`heatr3d.py` was NOT modified by any part of this campaign. It was driven
through its public API and its documented `run(qrf_override=...)` hook, in the
isolation pattern of `heatr3d_s4_flir/`.

---

## S2 VERDICT: **FAIL**

Against the pre-registered criteria, on the completed grid ladder:

| shape | verdict | why |
|---|---|---|
| **circle** | **PASS** | all five verdict-carrying quantities monotone convergent on BOTH read states |
| **square** | **FAIL** | every verdict-carrying quantity DIVERGES |
| **lshape** | **FAIL** | every verdict-carrying quantity DIVERGES, worst on bed melt |

The verdict is not negotiated after the fact and the failure is not a problem
to be fixed: it is the answer to the question S2 was created to ask.

**The finding in one sentence.** heatr3d's melt-region geometry converges
cleanly on the one test shape WITHOUT corners and diverges on both shapes WITH
corners, because the EQS-02 stencil correction removed a discretization
artifact and thereby EXPOSED a genuine corner field singularity that the shape
metrics and std(T) both read directly.

**Consequence, stated plainly.** The deliverable S2 was supposed to produce -- a
working grid chosen from the study, with published bands every downstream
number quotes -- CAN be produced for cornered-free geometry and CANNOT be
produced for cornered geometry on this grid range. That escalates to S3: only
the COMSOL anchor can decide whether the corner singularity resolves against an
independent engine, or whether the metric itself is the wrong instrument for
cornered geometry.

---

## 1. What was run, and what was not

| shape | n | status | wall [s] |
|---|---|---|---|
| circle | 48 | completed | 75 |
| circle | 64 | completed | 189 |
| circle | 80 | completed | 384 |
| circle | 96 | NOT_RUN (not reached) | - |
| square | 48 | completed | 75 |
| square | 64 | completed | 193 |
| square | 80 | completed | 500 |
| square | 96 | NOT_RUN (not reached) | - |
| lshape | 48 | completed | 141 |
| lshape | 64 | completed | 361 |
| lshape | 80 | completed | 900 |
| lshape | 96 | NOT_RUN (not reached) | - |

Per the pre-registered partial-completion policy, any (shape, grid) not reached
is recorded NOT_RUN with its reason and is never back-filled or extrapolated.
The run order was GRID-MAJOR (every shape at 48, then 64, then 80, then 96)
precisely so that partial completion still yields a 3-grid band -- the
pre-registered minimum -- for EVERY shape rather than a complete ladder for one
shape and nothing for the others. That is what happened, and it is why a
verdict exists at all.

---

## 2. The gates

### Convergence verdict per shape per quantity

| shape | read | quantity | successive changes | finest | ceiling | status | band | PASS |
|---|---|---|---|---|---|---|---|---|
| circle | melt_onset | jaccard_dist_phi0p9_grid_to_grid | 0.042409, 0.033036 | 0.033036 | 0.05 | monotone_convergent | 0.049554 | PASS |
| circle | melt_onset | jaccard_dist_phi0p8_grid_to_grid | 0.045114, 0.033719 | 0.033719 | 0.05 | monotone_convergent | 0.050578 | PASS |
| circle | melt_onset | front_ssd_mm_grid_to_grid | 0.19839, 0.16712 | 0.16712 | 0.25 | monotone_convergent | 0.25068 | PASS |
| circle | melt_onset | in_part_melt_fraction_phi0p9 | 0.023189, 0.0034354 | 0.0034354 | 0.02 | monotone_convergent | 0.0051532 | PASS |
| circle | melt_onset | out_of_part_melt_fraction_phi0p9 | 0, 0 | 0 | 0.02 | bounded_oscillatory | 0 | PASS |
| circle | melt_onset | t90 | 0, 0.0058417 | 0.0058417 | 0.01 | diverging | - | FAIL |
| circle | melt_onset | sigma_T | 0.00068143, 0.012683 | 0.012683 | - | diverging | - | FAIL |
| circle | heating_fixed_time | jaccard_dist_phi0p9_grid_to_grid | 0.045617, 0.034868 | 0.034868 | 0.05 | monotone_convergent | 0.052302 | PASS |
| circle | heating_fixed_time | jaccard_dist_phi0p8_grid_to_grid | 0.04231, 0.032301 | 0.032301 | 0.05 | monotone_convergent | 0.048451 | PASS |
| circle | heating_fixed_time | front_ssd_mm_grid_to_grid | 0.2138, 0.17568 | 0.17568 | 0.25 | monotone_convergent | 0.26352 | PASS |
| circle | heating_fixed_time | in_part_melt_fraction_phi0p9 | 0.024621, 0.0064415 | 0.0064415 | 0.02 | monotone_convergent | 0.0096622 | PASS |
| circle | heating_fixed_time | out_of_part_melt_fraction_phi0p9 | 0, 0 | 0 | 0.02 | bounded_oscillatory | 0 | PASS |
| circle | heating_fixed_time | sigma_T | 0.0011852, 0.015339 | 0.015339 | - | diverging | - | FAIL |
| square | melt_onset | jaccard_dist_phi0p9_grid_to_grid | 0.058157, 0.11574 | 0.11574 | 0.05 | diverging | - | FAIL |
| square | melt_onset | jaccard_dist_phi0p8_grid_to_grid | 0.057029, 0.11623 | 0.11623 | 0.05 | diverging | - | FAIL |
| square | melt_onset | front_ssd_mm_grid_to_grid | 0.29885, 0.61219 | 0.61219 | 0.25 | diverging | - | FAIL |
| square | melt_onset | in_part_melt_fraction_phi0p9 | 0.03219, 0.07006 | 0.07006 | 0.02 | diverging | - | FAIL |
| square | melt_onset | out_of_part_melt_fraction_phi0p9 | 0.023836, 0.041435 | 0.041435 | 0.02 | diverging | - | FAIL |
| square | melt_onset | t90 | 0.024205, 0.017257 | 0.017257 | 0.01 | monotone_convergent | 0.025886 | FAIL |
| square | melt_onset | sigma_T | 0.0048573, 0.0096403 | 0.0096403 | - | diverging | - | FAIL |
| square | heating_fixed_time | jaccard_dist_phi0p9_grid_to_grid | 0.078538, 0.13457 | 0.13457 | 0.05 | diverging | - | FAIL |
| square | heating_fixed_time | jaccard_dist_phi0p8_grid_to_grid | 0.073641, 0.12778 | 0.12778 | 0.05 | diverging | - | FAIL |
| square | heating_fixed_time | front_ssd_mm_grid_to_grid | 0.4082, 0.71788 | 0.71788 | 0.25 | diverging | - | FAIL |
| square | heating_fixed_time | in_part_melt_fraction_phi0p9 | 0.044888, 0.083538 | 0.083538 | 0.02 | diverging | - | FAIL |
| square | heating_fixed_time | out_of_part_melt_fraction_phi0p9 | 0.030519, 0.045667 | 0.045667 | 0.02 | diverging | - | FAIL |
| square | heating_fixed_time | sigma_T | 0.02423, 0.017056 | 0.017056 | - | monotone_convergent | 0.025584 | n/a |
| lshape | melt_onset | jaccard_dist_phi0p9_grid_to_grid | 0.082277, 0.10172 | 0.10172 | 0.05 | diverging | - | FAIL |
| lshape | melt_onset | jaccard_dist_phi0p8_grid_to_grid | 0.082644, 0.10103 | 0.10103 | 0.05 | diverging | - | FAIL |
| lshape | melt_onset | front_ssd_mm_grid_to_grid | 0.64086, 0.79737 | 0.79737 | 0.25 | diverging | - | FAIL |
| lshape | melt_onset | in_part_melt_fraction_phi0p9 | 0.0089286, 0.017015 | 0.017015 | 0.02 | diverging | - | FAIL |
| lshape | melt_onset | out_of_part_melt_fraction_phi0p9 | 0.14858, 0.17773 | 0.17773 | 0.02 | diverging | - | FAIL |
| lshape | melt_onset | t90 | 0.070789, 0.024912 | 0.024912 | 0.01 | monotone_convergent | 0.037367 | FAIL |
| lshape | melt_onset | sigma_T | 0.093555, 0.10601 | 0.10601 | - | diverging | - | FAIL |
| lshape | heating_fixed_time | jaccard_dist_phi0p9_grid_to_grid | 0.043105, 0.076356 | 0.076356 | 0.05 | diverging | - | FAIL |
| lshape | heating_fixed_time | jaccard_dist_phi0p8_grid_to_grid | 0.042049, 0.075661 | 0.075661 | 0.05 | diverging | - | FAIL |
| lshape | heating_fixed_time | front_ssd_mm_grid_to_grid | 0.33639, 0.589 | 0.589 | 0.25 | diverging | - | FAIL |
| lshape | heating_fixed_time | in_part_melt_fraction_phi0p9 | 0.016088, 0.0080863 | 0.0080863 | 0.02 | monotone_convergent | 0.012129 | PASS |
| lshape | heating_fixed_time | out_of_part_melt_fraction_phi0p9 | 0.022995, 0.13671 | 0.13671 | 0.02 | diverging | - | FAIL |
| lshape | heating_fixed_time | sigma_T | 0.068146, 0.096032 | 0.096032 | - | diverging | - | FAIL |

Per shape: **circle** PASS, **square** FAIL, **lshape** FAIL

`s2_task3_verdict` = **FAIL** (scored ['circle', 'square', 'lshape'], insufficient [])

### Electrode-gauge decision

| arm | raw power density by grid | total drift | finest-pair change |
|---|---|---|---|
| cell_centred_current | 73980.738, 73824.204, 69911.760 | 0.05820 | 0.05300 |
| face_gauge | 70930.316, 71535.228, 68462.861 | 0.04488 | 0.04295 |

| n | measured cell/face ratio | predicted (n/(n-1))^2 | abs diff |
|---|---|---|---|
| 48 | 1.0430059036085566 | 1.0430058850158441 | 1.859e-08 |
| 64 | 1.0319978892714579 | 1.0319979843789366 | 9.511e-08 |
| 96 | 1.021163290404396 | 1.0211634349030472 | 1.445e-07 |

Winner by the pre-registered rule: **face_gauge** (shipped arm wins: false).

**Inertness**: renormalized Q differs between gauges by 3.523e-07 of max Q and total power by 0.0 -- the gauge cannot move any thermal output.

### Mechanism 1 -- the L-shape outlier (reentrant corner)

| n | corner power share | corner volume share | concentration | corner max/mean | whole-part max/mean | in-part CV |
|---|---|---|---|---|---|---|
| 48 | 0.05818 | 0.02857 | 2.0362 | 2.8125 | 2.8125 | 0.5879 |
| 64 | 0.03787 | 0.01587 | 2.3860 | 3.3788 | 3.3788 | 0.6247 |
| 80 | 0.04976 | 0.02217 | 2.2441 | 3.7135 | 3.7135 | 0.6154 |

`corner_peak_growing` = `true`, `concentration_growing` = `false`.

### Mechanism 2 -- the cylinder null (no attenuation across the part)

skin depth / part radius = **48.3**, loss tangent sigma/(omega eps) = **1.326**

| n | interior CV | interior max/mean | whole-part CV |
|---|---|---|---|
| 48 | 0.02007 | 1.0639 | 0.1294 |
| 64 | 0.05036 | 1.1832 | 0.1803 |
| 80 | 0.04146 | 1.1617 | 0.1214 |


---

## 2b. Task 4 -- the densify=True coupled march

**STATUS: NOT_RUN.** The runner (`heatr3d_s2/densify.py`) is implemented,
committed and was launched, but did not complete within the session. Per the
pre-registered partial-completion policy it is recorded NOT_RUN and is NOT
back-filled with an estimate or a partial arm.

What it is set up to do, so the next session does not have to re-derive it:

* **Why the term has never been exercised.** `sigma_density_coeff` enters
  `heatr3d.apply_sigma_coupling` ONLY through the factor
  `(1 + b * (rho_rel - rho_ref))`. With `densify=False` the relative density
  never leaves `rho_ref`, so that factor is exactly 1.0 for ANY b. Every prior
  study ran `densify=False`, which is why the coefficient has been provably
  inert -- the inertness is structural, not incidental.
* **The control is two-sided, deliberately.** With densify OFF a change in b
  must move NOTHING bit-for-bit (that is the prior inertness, proven rather
  than asserted); with densify ON it must move something, or the term is still
  unreachable and the assignment has not been carried out. Both halves are
  measured by the runner.
* **Arms** (pre-registered, EXPLORATORY): b in {0.0, 0.3, 0.6, -0.3} at
  n = 64 on the circle, `eqs_update_interval_s` = 60 s,
  `sigma_temp_coeff_per_K` = -0.002 (inside the 0.0044/K validity bound carried
  from S4). 0.6 is the only value with any provenance in this repo
  (`configs/_archive_old/rfam_eqs_comsol_mimic.yaml` l.83-85) and is labelled
  exploratory everywhere it appears.
* **Standing gates** (energy audit, clamp, CFL) are recorded on every arm.
* **Anti-circularity**: the S4 rule stands -- the arms are pre-registered and
  nothing is fitted to scored FLIR frames.
* Question (b) is a deliberate SPOT CHECK at two grids and is labelled as such;
  it is not a convergence claim, because the pre-registration requires three
  grids for a band.

Resume with `./.venv312/bin/python -m heatr3d_s2.densify`.

---

## 3. Reading the failure

**The pattern is corners, not shapes.** The circle has no corners and passes on
every verdict-carrying quantity, on both read states, with successive changes
that fall monotonically. The square (four convex 90-degree corners) and the
L-shape (three convex corners plus one REENTRANT corner) both fail, and they
fail the same way: the 64->80 change is roughly DOUBLE the 48->64 change. That
is divergence, not slow convergence, and the pre-registration forbids
converting it into a band.

**The mechanism is a field singularity that the correction exposed.** In the
L-shape the GLOBAL absorbed-power peak sits at the reentrant corner -- the
whole-part max/mean equals the corner max/mean at every grid tested -- and that
peak grows monotonically under refinement, 2.8125 -> 3.3788 -> 3.7135 over
n = 48/64/80. A resolved feature settles; a peak that keeps climbing is the
grid chasing a singularity it cannot resolve.

This reconciles the EQS-02 re-rank anomaly that motivated putting the L-shape in
the ladder. The corrected (masked) stencil did exactly what it was supposed to:
it stopped `compute_qrf_3d` differencing V across the material interface, which
was a pure discretization artifact. What it revealed underneath is a PHYSICAL
reentrant-corner singularity, and std(T) reads that directly -- which is why the
corrected default made the L-shape's sigma_T go 33.204 -> 66.420 (+100.0 %,
`heatr3d_eqs02_rerank/RERANK_REPORT.md`) rather than improving it. The +100 % was
never a regression in the correction; it was the correction telling the truth
about a shape whose field genuinely has a singular point.

Note the direction this cuts. D1 already established that the shipped voxel
corner metric grew like 1/h (exponent -1.016, a numerical artifact) while
dolfinx grew at -0.329 (a genuine integrable edge singularity). S2 now shows
the corrected voxel engine sits on the physical side of that split -- and that
being on the physical side is precisely what breaks convergence of the derived
melt geometry at these grids.

**Why the circle is the exception, and what that means for the cylinder null.**
The skin depth is 48.3x the part radius and the loss tangent is 1.33, so there
is no attenuation mechanism that could make the field vary across the part.
The measured interior coefficient of variation is 0.020-0.050 while the
whole-part CV is 0.121-0.180: essentially ALL the field structure lives at the
rim, and the interior is flat because it has no reason not to be.

That is the cylinder null, and it also reconciles with this lane's own Phase C
result. The inversion heuristic grades dopant by inverting `T_phi90` over the
whole part, so it is dominated by the flat interior and finds +0.3 % -- nothing
(`FGM_BENEFIT_RERUN.md` arm D). The Phase C direct solve found 10.67 % on the
same shape and the same corrected field because its objective carries an
explicit OUT-OF-BOUNDS term that sees exactly the rim: its `J_out_of_bounds`
fell 22.4 % against only 6.99 % on the in-bounds deficit
(`solve3d/results/phase_c_solves.json`). The two results do not conflict. They
were reading different parts of one field, and S2 measures which part carries
the structure.

---

## 4. Two caveats that are mine, not the engine's

**The circle's t90 `diverging` label is an artifact of quantization, not of the
solver.** t90 is quantized to `dt_s` = 0.05 s, and n = 48 and n = 64 both return
323.35 s exactly, so the first successive change is EXACTLY ZERO. Any nonzero
second change then makes the ratio test degenerate and the classifier reports
divergence. The finest change, 0.0058417, is inside the pre-registered 0.01
ceiling. t90 is co-primary rather than verdict-carrying, so per the
pre-registration this is PARTIAL and does not touch the circle's PASS. Reported
rather than reclassified: the classifier behaved as specified and the
specification has a degenerate case, which is worth knowing before the next
campaign uses it.

**An ambiguity in my own pre-registration, resolved toward the calibrated
side.** It describes the IoU quantities as measured "against the ANALYTIC
nominal shape" but anchors their ceiling on a precedent (0.0344,
`solve3d/results/phase_a_shape_gate.json`) that is the Jaccard distance between
TWO GRIDS. Those are different quantities. Both are computed; the VERDICT is
taken on the grid-to-grid form because that is what the ceiling was calibrated
against and because "convergence" means the melt region stops MOVING under
refinement, while agreement with the nominal is a physics outcome a perfectly
converged solver is free to get wrong. The nominal-referenced form is published
beside it. Nothing was widened.

---

## 5. Not covered

* **COMSOL and 2.5-D anchors are S3**, and S2's failure hands them a specific
  question rather than a vague one: does the corner singularity resolve against
  an independent engine, or is the melt-geometry metric the wrong instrument
  for cornered geometry? S2 cannot answer that from inside one engine.
* **The Phase A cross-family circle t90 offset (~1.6 s) is out of scope** and
  remains S3's, exactly as the plan states.
* **No physical data.** Nothing here is validated against hardware; the S4
  anti-circularity rule stands and nothing in this campaign is fitted to scored
  FLIR frames.
* **The metric replacement S2 was also meant to deliver is NOT delivered.** The
  spec asks S2 to replace raw std(T) with a metric that converges. This campaign
  shows the shape metrics converge for corner-free geometry and diverge for
  cornered geometry, so no candidate replacement can be selected on this
  evidence -- the choice depends on the S3 adjudication above.
* **The working grid is NOT chosen.** For the circle the bands support a working
  grid; for cornered shapes no grid in the tested range is defensible.
* **sigma_T was never verdict-carrying**, by pre-registration. It is reported
  with its band throughout and gating on it would have re-enshrined the very
  metric S2 exists to replace.

---

## 6. Sign-off items for Matt

1. **The gauge default flip.** Recommended (face gauge) but NOT implemented and
   NOT flipped -- the EQS-02 precedent. Given the measured inertness, the
   recommendation is to flip only if a downstream consumer quotes a
   pre-renormalization field quantity. `heatr3d.py` is untouched.
2. **The canonical re-sync question is NOT triggered by this campaign**, because
   `heatr3d.py` was not modified. If the gauge flag later lands, the re-sync to
   `dissertation_materials/analysis-3dfgm/heatr3d.py` is gated on your sign-off.
3. **S2 is FAIL and should be recorded as FAIL.** The corner finding is the
   deliverable; the honest next move is S3's COMSOL anchor on a cornered shape,
   not a re-run of S2 with looser criteria.
