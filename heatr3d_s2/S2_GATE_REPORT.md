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

Against the pre-registered criteria, on the completed grid ladder
(circle 48/64/80/96; square and lshape 48/64/80, their n=96 NOT_RUN):

| shape | verdict | why |
|---|---|---|
| **circle** | **FAIL** | melt-region GEOMETRY converges cleanly and roughly halves per refinement, but the in-part melt FRACTION oscillates (0.0232 -> 0.0034 -> 0.0123) and fails the non-increase criterion |
| **square** | **FAIL** | every verdict-carrying quantity DIVERGES over 48->80 |
| **lshape** | **FAIL** | every verdict-carrying quantity DIVERGES over 48->80, worst on bed melt |

The verdict is not negotiated after the fact and the failure is not a problem
to be fixed: it is the answer to the question S2 was created to ask.

**THE FOURTH GRID CHANGED THE ANSWER, and it changed it for the worse.** At
three grids (48/64/80) the circle PASSED every verdict-carrying quantity and
this report would have said so. Adding n=96 flipped it to FAIL, because
`in_part_melt_fraction` fell from 0.0232 to 0.0034 and then rose again to
0.0123. That is recorded prominently rather than buried: a 3-grid convergence
claim on this engine is not safe, the pre-registered 3-grid minimum turns out
to be the bare minimum and not a sufficient one, and any future campaign here
should treat three grids as provisional.

**The finding, in two parts.** They are different and the distinction is the
useful part:

1. **Melt-region GEOMETRY converges for the corner-free shape and diverges for
   cornered shapes.** The circle's Jaccard distance and front position fall
   monotonically and roughly halve per refinement step (0.0424 -> 0.0330 ->
   0.0169; front 0.198 -> 0.167 -> 0.0874 mm), and both yield bands. The square
   and the L-shape instead DOUBLE their change from 48->64 to 64->80 on every
   verdict-carrying quantity. The mechanism is that the EQS-02 stencil
   correction removed a discretization artifact and thereby EXPOSED a genuine
   corner field singularity that the shape metrics and std(T) both read
   directly.
2. **Melt AMOUNT does not converge monotonically even for the circle.** The
   in-part melt fraction oscillates at the ~1 % level across the ladder. So the
   failure is not purely a corner story: WHERE the front sits converges for
   smooth geometry, HOW MUCH is melted does not settle on this range for any
   shape tested.

**Consequence, stated plainly.** The deliverable S2 was supposed to produce -- a
working grid chosen from the study, with published bands every downstream
number quotes -- can be produced for the GEOMETRIC metrics on corner-free
geometry and cannot be produced for cornered geometry, nor for the melt-amount
metric on any shape tested. That escalates to S3: only the COMSOL anchor can
decide whether the corner singularity resolves against an independent engine,
or whether the metric itself is the wrong instrument for cornered geometry.

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
| circle | melt_onset | jaccard_dist_phi0p9_grid_to_grid | 0.042409, 0.033036, 0.016899 | 0.016899 | 0.05 | monotone_convergent | 0.025348 | PASS |
| circle | melt_onset | jaccard_dist_phi0p8_grid_to_grid | 0.045114, 0.033719, 0.017154 | 0.017154 | 0.05 | monotone_convergent | 0.025731 | PASS |
| circle | melt_onset | front_ssd_mm_grid_to_grid | 0.19839, 0.16712, 0.087426 | 0.087426 | 0.25 | monotone_convergent | 0.13114 | PASS |
| circle | melt_onset | in_part_melt_fraction_phi0p9 | 0.023189, 0.0034354, 0.01231 | 0.01231 | 0.02 | diverging | - | FAIL |
| circle | melt_onset | out_of_part_melt_fraction_phi0p9 | 0, 0, 0 | 0 | 0.02 | bounded_oscillatory | 0 | PASS |
| circle | melt_onset | t90 | 0, 0.0058417, 0.0077459 | 0.0077459 | 0.01 | diverging | - | FAIL |
| circle | melt_onset | sigma_T | 0.00068143, 0.012683, 0.0050416 | 0.0050416 | - | diverging | - | FAIL |
| circle | heating_fixed_time | jaccard_dist_phi0p9_grid_to_grid | 0.045617, 0.034868, 0.019575 | 0.019575 | 0.05 | monotone_convergent | 0.029363 | PASS |
| circle | heating_fixed_time | jaccard_dist_phi0p8_grid_to_grid | 0.04231, 0.032301, 0.022252 | 0.022252 | 0.05 | monotone_convergent | 0.033377 | PASS |
| circle | heating_fixed_time | front_ssd_mm_grid_to_grid | 0.2138, 0.17568, 0.09905 | 0.09905 | 0.25 | monotone_convergent | 0.14857 | PASS |
| circle | heating_fixed_time | in_part_melt_fraction_phi0p9 | 0.024621, 0.0064415, 0.014744 | 0.014744 | 0.02 | diverging | - | FAIL |
| circle | heating_fixed_time | out_of_part_melt_fraction_phi0p9 | 0, 0, 0 | 0 | 0.02 | bounded_oscillatory | 0 | PASS |
| circle | heating_fixed_time | sigma_T | 0.0011852, 0.015339, 0.0091556 | 0.0091556 | - | diverging | - | FAIL |
| square | melt_onset | jaccard_dist_phi0p9_grid_to_grid | 0.058157, 0.11574, 0.05241 | 0.05241 | 0.05 | diverging | - | FAIL |
| square | melt_onset | jaccard_dist_phi0p8_grid_to_grid | 0.057029, 0.11623, 0.054761 | 0.054761 | 0.05 | diverging | - | FAIL |
| square | melt_onset | front_ssd_mm_grid_to_grid | 0.29885, 0.61219, 0.26117 | 0.26117 | 0.25 | diverging | - | FAIL |
| square | melt_onset | in_part_melt_fraction_phi0p9 | 0.03219, 0.07006, 0.030408 | 0.030408 | 0.02 | diverging | - | FAIL |
| square | melt_onset | out_of_part_melt_fraction_phi0p9 | 0.023836, 0.041435, 0.016708 | 0.016708 | 0.02 | diverging | - | FAIL |
| square | melt_onset | t90 | 0.024205, 0.017257, 0.010526 | 0.010526 | 0.01 | monotone_convergent | 0.015789 | FAIL |
| square | melt_onset | sigma_T | 0.0048573, 0.0096403, 0.0081885 | 0.0081885 | - | diverging | - | FAIL |
| square | heating_fixed_time | jaccard_dist_phi0p9_grid_to_grid | 0.078538, 0.13457, 0.066099 | 0.066099 | 0.05 | diverging | - | FAIL |
| square | heating_fixed_time | jaccard_dist_phi0p8_grid_to_grid | 0.073641, 0.12778, 0.061003 | 0.061003 | 0.05 | diverging | - | FAIL |
| square | heating_fixed_time | front_ssd_mm_grid_to_grid | 0.4082, 0.71788, 0.32891 | 0.32891 | 0.25 | diverging | - | FAIL |
| square | heating_fixed_time | in_part_melt_fraction_phi0p9 | 0.044888, 0.083538, 0.03943 | 0.03943 | 0.02 | diverging | - | FAIL |
| square | heating_fixed_time | out_of_part_melt_fraction_phi0p9 | 0.030519, 0.045667, 0.019381 | 0.019381 | 0.02 | diverging | - | FAIL |
| square | heating_fixed_time | sigma_T | 0.02423, 0.017056, 0.013186 | 0.013186 | - | monotone_convergent | 0.019779 | n/a |
| lshape | melt_onset | jaccard_dist_phi0p9_grid_to_grid | 0.082277, 0.10172, 0.062023 | 0.062023 | 0.05 | diverging | - | FAIL |
| lshape | melt_onset | jaccard_dist_phi0p8_grid_to_grid | 0.082644, 0.10103, 0.061491 | 0.061491 | 0.05 | diverging | - | FAIL |
| lshape | melt_onset | front_ssd_mm_grid_to_grid | 0.64086, 0.79737, 0.46281 | 0.46281 | 0.25 | diverging | - | FAIL |
| lshape | melt_onset | in_part_melt_fraction_phi0p9 | 0.0089286, 0.017015, 0.0084232 | 0.0084232 | 0.02 | diverging | - | FAIL |
| lshape | melt_onset | out_of_part_melt_fraction_phi0p9 | 0.14858, 0.17773, 0.10529 | 0.10529 | 0.02 | diverging | - | FAIL |
| lshape | melt_onset | t90 | 0.070789, 0.024912, 0.040511 | 0.040511 | 0.01 | diverging | - | FAIL |
| lshape | melt_onset | sigma_T | 0.093555, 0.10601, 0.063298 | 0.063298 | - | diverging | - | FAIL |
| lshape | heating_fixed_time | jaccard_dist_phi0p9_grid_to_grid | 0.043105, 0.076356, 0.033069 | 0.033069 | 0.05 | diverging | - | FAIL |
| lshape | heating_fixed_time | jaccard_dist_phi0p8_grid_to_grid | 0.042049, 0.075661, 0.032322 | 0.032322 | 0.05 | diverging | - | FAIL |
| lshape | heating_fixed_time | front_ssd_mm_grid_to_grid | 0.33639, 0.589, 0.24553 | 0.24553 | 0.25 | diverging | - | FAIL |
| lshape | heating_fixed_time | in_part_melt_fraction_phi0p9 | 0.016088, 0.0080863, 0.0067385 | 0.0067385 | 0.02 | monotone_convergent | 0.010108 | PASS |
| lshape | heating_fixed_time | out_of_part_melt_fraction_phi0p9 | 0.022995, 0.13671, 0.035799 | 0.035799 | 0.02 | diverging | - | FAIL |
| lshape | heating_fixed_time | sigma_T | 0.068146, 0.096032, 0.049431 | 0.049431 | - | diverging | - | FAIL |

Per shape: **circle** FAIL, **square** FAIL, **lshape** FAIL

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

See the CLOSE-OUT section C4 below: this completed after the first draft.

---

## 3. Reading the failure

**The pattern is corners, for the GEOMETRIC metrics.** The circle has no
corners, and its melt-region geometry -- Jaccard distance and front position --
falls monotonically on both read states, roughly halving per refinement step,
and yields bands. (Its in-part melt FRACTION does not; see the verdict section.
The corner story below is about where the front sits, not how much melts.) The square (four convex 90-degree corners) and the
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
* **The working grid is NOT chosen.** For the circle's GEOMETRIC metrics the
  bands support a working grid; for cornered shapes, and for the melt-amount
  metric on any shape, no grid in the tested range is defensible.
* **square and lshape at n=96 are NOT_RUN**, so the key follow-up question --
  does their divergence PERSIST at 80->96 or turn over -- is unanswered. Given
  that the circle's own verdict flipped when its fourth grid landed, that
  question must be answered before the corner finding is treated as settled.
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
### Addendum -- axis-aligned staircase commensurability

| n | h [mm] | cells across | effective half-width [mm] | error [mm] | error [%] | commensurate |
|---|---|---|---|---|---|---|
| 48 | 1.2500 | 16 | 10.0000 | +0.0000 | +0.00 | true |
| 64 | 0.9375 | 22 | 10.3125 | +0.3125 | +3.12 | false |
| 80 | 0.7500 | 26 | 9.7500 | -0.2500 | -2.50 | false |
| 96 | 0.6250 | 32 | 10.0000 | +0.0000 | +0.00 | true |

| shape | pair | jaccard phi>=0.9 | geometry jump [mm] | jaccard per mm of jump |
|---|---|---|---|---|
| square | 48->64 | 0.05816 | 0.3125 | 0.1861 |
| square | 64->80 | 0.11574 | 0.5625 | 0.2058 |
| square | 80->96 | 0.05241 | 0.2500 | 0.2096 |
| lshape | 48->64 | 0.08228 | 0.3125 | 0.2633 |
| lshape | 64->80 | 0.10172 | 0.5625 | 0.1808 |
| lshape | 80->96 | 0.06202 | 0.2500 | 0.2481 |
| circle | 48->64 | 0.04241 | 0.3125 | 0.1357 |
| circle | 64->80 | 0.03304 | 0.5625 | 0.0587 |
| circle | 80->96 | 0.01690 | 0.2500 | 0.0676 |

| shape | COMMENSURATE pair | jaccard phi>=0.9 | front SSD [mm] |
|---|---|---|---|
| square | 48<->96 | 0.01238 | 0.06597 |
| lshape | 48<->96 | 0.04450 | 0.33342 |
| circle | 48<->96 | 0.01955 | 0.09798 |

### Task 4 -- the densify=True coupled march

**Control, densify OFF**: `inert` = `true`, max |dT| between b = 0.0 and b = 0.6 is **0.0** C, t90 identical: `true`. The prior inertness is proven bit-for-bit, exactly as the structural argument predicts.

**Reachable with densify ON**: `true`.

| b | t90 [s] | sigma_T [C] | surface-minus-interior [C] | rho_final mean | EQS solves | energy resid | clamp |
|---|---|---|---|---|---|---|---|
| 0.0 | 320.15 | 19.043 | -34.022 | 0.99790 | 10 | -1.05e-13 | false |
| 0.3 | 321.90 | 19.495 | -34.699 | 0.99776 | 10 | -1.06e-13 | false |
| 0.6 | 323.50 | 19.911 | -35.319 | 0.99763 | 10 | -1.03e-13 | false |
| -0.3 | 318.30 | 18.556 | -33.287 | 0.99805 | 10 | -9.83e-14 | false |

| b | d t90 [s] | d sigma_T [C] | d(surface-interior) [C] | max abs dT vs b=0 [C] |
|---|---|---|---|---|
| 0.3 | +1.750 | +0.4520 | -0.6780 | 1.4186 |
| 0.6 | +3.350 | +0.8680 | -1.2973 | 2.7107 |
| -0.3 | -1.850 | -0.4868 | +0.7343 | 1.5323 |

| spot-check grid | t90 [s] | sigma_T [C] |
|---|---|---|
| 48 | 323.55 | 19.940 |
| 64 | 323.50 | 19.911 |

TWO grids at ONE shape is a SPOT CHECK, not a convergence claim: the pre-registration requires 3 grids for a band and this deliberately does not meet it

---

# S2 CLOSE-OUT (full 4-grid ladder + densify harvested)

All twelve campaign cases and the densify march completed. This section
supersedes the 3-grid readings above where they differ; the verdict does not
change, but the DIAGNOSIS changes substantially and for the better.

## C1. The verdict stands: FAIL

Evaluated on the pre-registered ladder and criteria, unchanged:
circle FAIL, square FAIL, lshape FAIL. Criteria frozen before the campaign are
not renegotiated after it.

## C2. But the mechanism was mostly NOT a solver failure

The fourth grid turned the cornered shapes' sequences over rather than
confirming divergence. Square Jaccard changes read 0.05816, 0.11574, 0.05241
and lshape 0.08228, 0.10172, 0.06202 -- the 80->96 change is BELOW the 48->64
change in both. The spike sits at the 64->80 pair, not at the finest pair, so
what the pre-registered non-increase criterion caught is a NON-MONOTONE
sequence, not a growing one.

The cause is measured and it is geometric, not numerical. heatr3d's part mask
is evaluated at CELL CENTRES, so an AXIS-ALIGNED boundary snaps coherently to
the nearest centre and the voxelized part is a DIFFERENT SIZE at each grid:
effective half-width 10.0000, 10.3125, 9.7500, 10.0000 mm at n = 48/64/80/96
against a nominal 10 mm -- up to +3.12 %. The 64->80 pair spans the largest
geometry jump (0.5625 mm), which is exactly where the metric spikes.

The proportionality is near-exact. For the square the Jaccard distance per
millimetre of geometry jump is 0.1861, 0.2058, 0.2096 across the three pairs --
essentially constant, i.e. the disagreement is explained by how much the part
CHANGED SIZE and almost nothing else.

**The decisive test.** n = 48 and n = 96 are COMMENSURATE (both give exactly
10.0000 mm). Comparing those two directly, with the part identical on both:

| shape | successive-pair range | COMMENSURATE 48<->96 | front SSD [mm] |
|---|---|---|---|
| square | 0.0524 - 0.1157 | **0.01238** | 0.06597 |
| lshape | 0.0620 - 0.1017 | **0.04450** | 0.33342 |
| circle | 0.0169 - 0.0424 | 0.01955 | 0.09798 |

The square collapses by 4-9x and lands far inside the pre-registered ceilings
(0.05 Jaccard, 0.25 mm front). **The square's apparent non-convergence was a
geometry artifact of the grid ladder, not a property of the solver.**

The L-shape does NOT fully collapse: 0.0445 clears the Jaccard ceiling but its
front SSD, 0.33342 mm, EXCEEDS the 0.25 mm ceiling even on the commensurate
pair. So the L-shape carries a genuine residual on top of commensurability --
which is consistent with the independent corner measurement, where its global
absorbed-power peak sits at the REENTRANT corner and grows monotonically under
refinement (2.8125 -> 3.3788 -> 3.7135). The corner singularity is real; it is
just not what was driving the square.

The circle is immune to the effect by construction: a curved boundary has no
coherent snap, its staircase error averages around the perimeter, and its
ratio column is correspondingly NOT constant (0.1357, 0.0587, 0.0676).

## C3. What that means, and the concrete recommendation

1. **The grid ladder I pre-registered was, for axis-aligned shapes, the worst
   available choice**: 48/64/80/96 mixes two commensurate grids with two
   incommensurate ones. I chose it before knowing this. Recorded as my error,
   not the engine's.
2. **Use commensurate grids for axis-aligned geometry.** For a 20 mm part in a
   60 mm chamber the commensurate set is n = 24k, i.e. 24/48/72/96. On such a
   ladder the square's melt geometry is inside the pre-registered ceilings.
3. **The S3 escalation narrows.** It is no longer "cornered shapes do not
   converge". It is specifically the REENTRANT corner: the L-shape retains a
   front-position residual above ceiling even with geometry held fixed, and its
   corner power peak grows without settling. That is the question for COMSOL --
   does the reentrant singularity resolve against an independent engine, or is
   front position the wrong instrument at a reentrant corner?
4. **The circle's in-part melt FRACTION remains non-monotone** (0.02319,
   0.00344, 0.01231) and is not explained by commensurability, since the circle
   is immune to it. How much melts still does not settle for any shape tested.

## C4. Task 4 -- densification coupling, answered

**The control is clean.** With densify OFF, changing b from 0.0 to 0.6 moves the
temperature field by max |dT| = **0.0** C exactly and leaves t90 identical. The
prior inertness is now proven bit-for-bit rather than asserted, and for the
predicted structural reason: rho never leaves rho_ref, so the coupling factor
is exactly 1.0 for any b.

**With densify ON the term is reachable**, and its response is close to linear
in b: per unit b, t90 moves about +5.6 to +6.2 s, sigma_T about +1.45 to +1.62 C
and the surface-minus-interior topology about -2.2 to -2.5 C.

**Question (a): does it move late-time surface topology in the FLIR-observed
direction? NO -- it moves the wrong way, and it is far too small anyway.**
The corrected model already puts the interior hotter than the surface
(surface-minus-interior = -34.022 C at b = 0). Positive b makes that MORE
negative (-35.319 C at b = 0.6), i.e. further from the FLIR observation, and
only negative b moves toward it. The only value with any archived provenance in
this repo is +0.6. Beyond the sign, the magnitude is ~1.3 C against an S4 gap of
tens of degrees -- roughly two orders of magnitude too small. **Density coupling
does not explain the S4 late-time topology miss.**

There is a structural reason it is small, and it is the same one the gauge
decision turned on: rho_final is 0.9976-0.9981, i.e. nearly UNIFORM at
saturation, so the coupling raises conductivity almost uniformly (a factor of
about 1.27 at b = 0.6) -- and heatr3d's fixed-power renormalization divides a
uniform conductivity change straight out. Only the PATTERN survives, which is
why a 27 % conductivity change buys a 1.3 C topology shift.

**Question (b): the spot check.** At b = 0.6, n = 48 and n = 64 give
t90 323.55 vs 323.50 s (0.015 %) and sigma_T 19.94 vs 19.911 C (0.15 %). Tight
-- but this is TWO grids on ONE shape and the pre-registration requires three
for a band. It is a spot check and is labelled as one; it is not evidence that
the Task-3 bands hold with densification on.

Standing gates passed on every arm: energy residual ~1e-13, no clamp binding,
no CFL violation.
