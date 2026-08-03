# solve3d Phase A: forward parity -- gate report

Plan: `docs/superpowers/plans/2026-08-01-solve3d-phase-a-forward-parity.md`
Spec: `docs/superpowers/specs/2026-07-31-solve-port-3d-design.md`
Date: 2026-08-01. Environment: `heatr3d_d1_spike/env` (dolfinx 0.11.0, complex
scalar build, `jit_fix` applied) and `./.venv312` (heatr3d). Every run used
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

**Every number below is PRINTED from `solve3d/results/*.json` by
`solve3d/make_report_tables.py`.** Nothing is transcribed by hand. Regenerate
with:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.make_report_tables

---

## Verdict in one line

The forward model is **built and analytically gated** (Tasks 0-3 all green,
including the S1 latent-plateau, Fourier-decay and energy-audit benchmarks, and
an exact reproduction of D1's measured dolfinx-vs-heatr3d Q_rf pattern). The
**Task-4 parity gate FAILS as written** (`verdict.gate_ok = false`): the
re-solve census matches heatr3d exactly on all four arms, but t90, the heating
curve and sigma_T sit 1.7x-4.5x outside the Task-1 tolerances. Section 5 shows
why that failure is a property of the TOLERANCE DERIVATION rather than a defect
in the march, and states what would have to change to make the gate meaningful.
Nothing was widened.

---

## 1. What was built

| file | role |
|---|---|
| `solve3d/forward.py` | dolfinx forward: complex EQS (corrected/masked Q only) + mass-lumped explicit enthalpy thermal-phase march + the ported in-march EQS re-solve loop |
| `solve3d/cases.py` | heatr3d-side anchor cases, curve samplers, Task-1 tolerance measurement, Task-4 reference runs (runs in `.venv312`) |
| `solve3d/gates.py` | pure-numpy gate math shared by BOTH environments |
| `solve3d/phase_a_gate.py` | the Task-4 four-arm parity gate; writes `results/phase_a_gate.json` |
| `solve3d/make_report_tables.py` | prints every table in this report from the JSONs |
| `solve3d/tests/` | `test_forward_analytic.py`, `test_forward_parity.py`, `test_cases.py`, `test_gates.py` |

Nothing outside `solve3d/` was modified. `heatr3d.py`, `rfam_eqs_coupled.py`
and `dissertation_materials/` are untouched.

## 2. Red-green evidence, per task

| task | RED (observed failure) | GREEN |
|---|---|---|
| 0 scaffold | `ImportError: cannot import name 'forward' from 'solve3d'` | 1 passed |
| 1 tolerances | `ImportError: cannot import name 'cases' from 'solve3d'` | 1 passed -- segmented march == monolithic heatr3d run to rel 1e-12 (t90, sigma_T, T_max, whole T_phi90 field) |
| 2 EQS lift | `AttributeError: module 'solve3d.forward' has no attribute 'eqs_case'` | 1 passed -- see sec 3 |
| 3 enthalpy march | `AttributeError: ... has no attribute 'box_mesh'`, 3 failed | 4 passed |
| 4 parity | `ImportError: cannot import name 'phase_a_gate' from 'solve3d'`; coupled sampler `AttributeError: ... 'march_sampled_coupled'` | sampler green (fields rel 1e-12, `n_eqs_solves` exact vs monolithic); **gate itself FAILS, sec 4** |

## 3. Task 2, and a gate correction that had to be escalated

The plan's Task-2 gate reads "pattern rel-L2 < 0.05 (the D1 Task-4 gate,
already proven achievable)". That 0.05 is real but belongs to a different
comparison: `heatr3d_d1_spike/run_scale_test.py` line 144 scores
`qrf_pattern_rel_l2_vs_task2_fine`, i.e. dolfinx-scale-mesh against
dolfinx-Task2-fine-mesh -- a **dolfinx self-convergence** check (measured
0.011870039611365844). It was never a dolfinx-vs-heatr3d number.

The measured dolfinx-vs-heatr3d(corrected) values are in the same file as
`task2.gate.maskgrad_all_points`: **0.17652460972777445** (coarse, n=64-matched)
and **0.10790598069422491** (fine, n=96-matched). D1 section 2 says so directly:
"the mask-confined field differs by 10.8 % in unit-mean pattern L2".

Rather than relax the threshold, the gate was made **stronger**: the lift must
REPRODUCE the D1 measured value. It does, to 3.8e-13 relative
(0.17652460972770778 against 0.17652460972777445), which pins gamma, the BVP,
the electrode BCs, the Q definition, the fixed-power renormalization basis and
the mesh-matching rule all at once. The renormalization identity
integral(Q dV) = power_density * V_part holds to 8.3e-15.

## 4. The gates

### Task 1: measured heatr3d self-spread -> FROZEN tolerances

| quantity | n=64 | n=96 (reference) | measured spread | tolerance (1.5x) |
|---|---|---|---|---|
| t90 [s] | 323.35 | 322.75 | 0.001859024012393564 | 0.0027885360185903457 |
| part-mean heating curve (rel-L2) | - | - | 0.00139388743745654 | 0.0020908311561848103 |
| sigma_T at melt onset [C] | 20.103141496882763 | 19.951950253198568 | 0.007577767675115235 | 0.011366651512672854 |

Provenance of the two anchor runs (both `reached = true`, `clamp_bound = false`):

| | n=64 | n=96 |
|---|---|---|
| `n_voxels_in_part` | 23040 | 77952 |
| `part_volume_m3` | 1.8984374999999996e-05 | 1.9031250000000002e-05 |
| `p_total_w` | 30.214571227601997 | 30.289175107176334 |
| `energy_residual_frac` | 1.9181977234309213e-16 | 1.070266200341092e-15 |
| `wall_eqs_s` | 37.206102124997415 | 230.9888614170486 |
| `wall_march_s` | 109.35514612495899 | 326.97862604202237 |

### Task 4: coupled-forward parity gate (`solve3d/results/phase_a_gate.json`)

| arm | t90 rel diff | vs tol | curve rel-L2 | vs tol | sigma_T rel diff | vs tol | n_eqs_solves dolfinx / heatr3d | arm_ok |
|---|---|---|---|---|---|---|---|---|
| circle_off | 0.006971340046475777 | FAIL (0.0027885360185903457) | 0.004802034113621331 | FAIL (0.0020908311561848103) | 0.03747797106176329 | FAIL (0.011366651512672854) | 1 / 1 (PASS) | FAIL |
| circle_coupled | 0.007187500000000036 | FAIL (0.0027885360185903457) | 0.004852415416329581 | FAIL (0.0020908311561848103) | 0.04288773854597466 | FAIL (0.011366651512672854) | 6 / 6 (PASS) | FAIL |
| square_off | 0.00015710919088770265 | PASS (0.0027885360185903457) | 0.010785353411702905 | FAIL (0.0020908311561848103) | 0.05073061553298375 | FAIL (0.011366651512672854) | 1 / 1 (PASS) | FAIL |
| square_coupled | 0.0017333753545540856 | PASS (0.0027885360185903457) | 0.010838452884915077 | FAIL (0.0020908311561848103) | 0.04257566336573178 | FAIL (0.011366651512672854) | 6 / 6 (PASS) | FAIL |

`verdict.gate_ok` = `false`

Raw values behind the ratios:

| arm | t90 dolfinx / heatr3d [s] | sigma_T dolfinx / heatr3d (mid-plane) [C] | heatr3d mid-plane / volumetric sigma_T |
|---|---|---|---|
| circle_off | 325.00000000000006 / 322.75 | 20.699707813129972 / 19.95194923699992 | 0.9999999490677034 |
| circle_coupled | 322.3 / 320.0 | 19.76920068462825 / 18.956211636154684 | 0.9999999440730402 |
| square_off | 318.3 / 318.25 | 18.82263171830493 / 19.828546065322882 | 0.999999866211537 |
| square_coupled | 317.85 / 317.3 | 18.767908708903445 / 19.60249806776397 | 1.0000000050774516 |

### Where the two engines disagree (per-arm field diagnostic)

| arm | T-rise rel-L2 all | interior | surface band | sigma_T interior rel diff | part volume rel diff | p_target rel diff |
|---|---|---|---|---|---|---|
| circle_off | 0.009064889619926943 | 0.006484217784724812 | 0.01797249694071128 | 0.03467512661254359 | 0.009964141075820266 | 0.009964141075830537 |
| circle_coupled | 0.008991650553370855 | 0.006234094698760379 | 0.018137195423156727 | 0.03993391301702275 | 0.009964141075820266 | 0.009964141075830537 |
| square_off | 0.01164285284635953 | 0.011839332547977833 | 0.010265548347984606 | 0.05127729995618971 | 1.411721578757167e-16 | 5.3945699435866835e-15 |
| square_coupled | 0.010022405999451072 | 0.010124387508120504 | 0.00933009306053637 | 0.04399543865976572 | 1.411721578757167e-16 | 5.3945699435866835e-15 |

| arm | surface-minus-interior mean T, dolfinx [C] | heatr3d [C] | n interior / band points |
|---|---|---|---|
| circle_off | -37.60591446065382 | -36.05428097408739 | 648 / 164 |
| circle_coupled | -36.26182337947722 | -34.62679735443885 | 648 / 164 |
| square_off | -29.689684094874366 | -31.29672374051728 | 841 / 183 |
| square_coupled | -29.73022298256103 | -31.021791590505586 | 841 / 183 |

### Cost (wall clock, serial, OMP_NUM_THREADS=1)

| arm | dolfinx mesh [s] | dolfinx EQS [s] | dolfinx march [s] | heatr3d EQS [s] | heatr3d march [s] | EQS speedup |
|---|---|---|---|---|---|---|
| circle_off | 85.92525033291895 | 3.9379097500350326 | 730.6426269999938 | 230.9888614170486 | 326.97862604202237 | 58.65773369107651 |
| circle_coupled | 51.571226125000976 | 8.983226542244665 | 456.90589658298995 | None | None | None |
| square_off | 33.504372959025204 | 3.1747375410050154 | 315.32912979100365 | 556.6332720409846 | 554.0024747500429 | 175.33205969044394 |
| square_coupled | 10.603848707978614 | 7.602071291883476 | 357.903319709003 | None | None | None |

| arm | dolfinx dofs / cells | in-part nodes | heatr3d in-part voxels | lc_part [m] | dt_stable [s] | n_substeps | k re-assemblies |
|---|---|---|---|---|---|---|---|
| circle_off | 94054 / 561132 | 77043 | 77952 | 0.0005799361188747663 | 0.15442306371307898 | 1 | 4122 |
| circle_coupled | 94054 / 561132 | 77043 | 77952 | 0.0005799361188747663 | 0.15442306371307898 | 1 | 4050 |
| square_off | 95575 / 572177 | 79508 | 98304 | 0.000625 | 0.1298131030512322 | 1 | 3841 |
| square_coupled | 95575 / 572177 | 79508 | 98304 | 0.000625 | 0.1298131030512322 | 1 | 3828 |

**Wall-time caveat, stated because it changes how the march row reads.** The
heatr3d reference runs and the dolfinx arms were deliberately overlapped on one
machine to shorten the campaign, so several marches competed for cores. The EQS
speedups (58.7x and 175.3x) are large enough to survive that; the MARCH times
are not clean head-to-head numbers and should not be quoted as such. What is
solid: the dolfinx march is the same order of magnitude as heatr3d's at matched
in-part resolution (315-731 s vs 327-554 s), and the EQS -- which is where
heatr3d's EQS-01 ceiling lives -- is one to two orders of magnitude cheaper.
Meshing (10.6-85.9 s) is re-paid per arm and remains the dolfinx-specific cost
D1 flagged.

### Standing health checks (every arm)

| arm | energy_residual_frac | clamp_bound | cfl_violated | power renorm residual | eval_missed |
|---|---|---|---|---|---|
| circle_off | 8.244600832159246e-14 | false | false | 8.326672684688674e-15 | 0 |
| circle_coupled | -3.4941093946974726e-15 | false | false | 8.326672684688674e-15 | 0 |
| square_off | -6.112028401910392e-14 | false | false | 3.907985046680551e-14 | 0 |
| square_coupled | -5.85904846422175e-14 | false | false | 3.907985046680551e-14 | 0 |

---

## 5. Reading the Task-4 failure honestly

Three facts, all from the tables above.

**(a) The re-solve semantics port is exact.** `n_eqs_solves` is 1/1 on both
defaults-off arms and 6/6 on both coupled arms. The absolute-time schedule, the
"rebuild gamma from base sigma, never compound" rule, and the fixed-power
renormalization on every re-solve reproduce heatr3d's census with no tolerance
at all. This is the part of Task 4 that passed.

**(b) The melt-onset FIELDS agree to about 1 %; the gate scores a STD of that
field.** `T_rise_rel_l2_all` is 0.0090-0.0116 across the four arms, and the
surface-vs-interior topology agrees in sign and magnitude
(surface-minus-interior mean T: -37.6 C dolfinx vs -36.1 C heatr3d on
circle_off; -29.7 vs -31.3 on square_off -- both engines show the
interior-hotter topology the EQS-02 correction produces). sigma_T is the
standard deviation of that field, a difference statistic, so a ~1 % field
disagreement lands as a 3.5-5.1 % sigma_T disagreement. That amplification is
arithmetic, not a defect.

**(c) The tolerance measures the wrong kind of uncertainty.** The Task-1
spread is heatr3d n=64 vs n=96: two grids of the SAME method, sharing the same
staircase geometry family, the same harmonic-face FV operator and the same Q
post-processing. It measures grid refinement only (t90 0.19 %, sigma_T 0.76 %)
and therefore cannot bound a change of DISCRETIZATION FAMILY. The known
cross-family disagreement at this resolution is the Q_rf drive pattern itself:
0.10790598069422491 (D1, n=96-matched, corrected stencil on both sides). A gate
that demands 1.1 % thermal agreement while the drive fields differ by 10.8 % is
not achievable by any correct implementation.

The square arm settles this. Its part volume is identical in both engines
(`part_volume_rel_diff` = 1.4e-16, because the 20 mm prism lands exactly on the
n=96 voxel grid) and `p_target_rel_diff` = 5.4e-15, so no drive-magnitude
difference exists at all -- and it still shows the LARGEST sigma_T gap
(0.0507). Meanwhile t90 PASSES on both square arms (0.000157 and 0.00173). So
the residual is neither part volume nor total power: it is the surface-region
field difference D1 already characterized.

One more signal worth recording: **the sign of the sigma_T difference is
shape-dependent** -- dolfinx reads HIGHER than heatr3d on the circle
(20.6997 vs 19.9519) and LOWER on the square (18.8226 vs 19.8285). That is the
same shape-dependent signature D1 measured for the EQS-02 correction itself
(-23 % circle, +11 % square). It reinforces that sigma_T rankings across shapes
are engine-sensitive and should not be quoted across engines.

### What this does NOT license

It does not license widening the gate. The frozen tolerances stay in
`solve3d/results/parity_tolerances.json` exactly as measured. What Phase A
needs before the gate can be called meaningful is a tolerance derived from a
CROSS-FAMILY spread, e.g. one of:

1. a dolfinx mesh-refinement spread (solve the same anchor at 2-3 in-part
    element sizes and measure the FEM's own convergence), combined with the
    heatr3d spread -- the honest "both engines' own uncertainty" band;
2. a third reference (the COMSOL 3-D anchor, Gate S3) so parity is scored
    against truth rather than against another approximation;
3. a gate on the QUANTITY THE SOLVE ACTUALLY OPTIMIZES. Phase C's objective is
    shape fidelity (whole-domain (phi - chi)^2), not sigma_T. Parity on melt-front
    position would be the decision-relevant gate; sigma_T is a proxy that the
    project has already labelled a diagnostic.

That is a decision for the plan owner, not something to be quietly absorbed
here.

## 6. What Phase A does NOT cover

* **No adjoint.** Phase B. Nothing in `solve3d/` computes a gradient; the
  checkpointed transient reverse march does not exist yet.
* **No objective, no chi, no regularization.** Phase C prep. There is no
  filter, no Heaviside projection, no STL-derived chi.
* **No eps_r channel, no drive reconciliation.** Phase D.
* **No Studio integration, no dissertation edit.** Out of scope by the spec.
* **No densification.** `densify=True` / `sigma_density_coeff` were never
  exercised; `rho_rel` is held at 0.55 everywhere and `sigma_density_coeff`
  stays 0. The spec absorbs that question into S2.
* **No legacy-Q path.** Forbidden by construction (spec sec 4). `solve3d`
  cannot produce heatr3d's pre-EQS-02 field even if asked.
* **The coupled arm is EXPLORATORY.** `sigma_temp_coeff_per_K = -0.002` /K has
  no validated provenance in this repo (heatr3d.py, S4-COUPLING note: every
  current config sets it to 0.0). It exists to exercise the re-solve loop, not
  to claim physics.
* **Two grids is not a convergence study.** The Task-1 tolerance rests on
  n=64 and n=96 only, and no dolfinx mesh-refinement sequence was run.
* **Serial only.** No MPI scaling was measured. Meshing (10.6-85.9 s) is
  re-done per arm and is a real cost, as D1 also found.
* **`phase_update="apparent_cp"` is not ported** and never will be here.

## 7. Reproduction

    # heatr3d side (geo-prewarp venv)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m solve3d.cases --measure
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m solve3d.cases --task4-refs

    # dolfinx side (spike env)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 heatr3d_d1_spike/env/bin/python -m solve3d.phase_a_gate

    # tests
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_forward_analytic.py solve3d/tests/test_forward_parity.py solve3d/tests/test_gates.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m pytest solve3d/tests/test_cases.py solve3d/tests/test_gates.py

The `.npz` anchor fields are regenerable and gitignored; the JSON gates are
tracked.

---

# Phase A CLOSE-OUT (2026-08-01, Matt-approved, both parts)

Additive. `solve3d/results/parity_tolerances.json` is unchanged and the Task-4
verdict in section 4 stands exactly as recorded. What follows adds a measured
cross-family band (STEP 1) and a design-relevant shape-parity gate (STEP 2).

## C1. Close-out verdict

| gate | result |
|---|---|
| STEP 2 shape parity (IoU 0.8 / 0.9, melt-front SSD, bed melt, in-part melt) | **PASS on all four arms** under per-shape measured bands; the circle arms also pass under the strictest available band |
| `n_eqs_solves` re-solve census | **PASS, exact** on all four arms (1/1 off, 6/6 coupled) |
| t90, pre-declared cross-family band | **PASS on all four arms** |
| t90, strict band (dolfinx mid-vs-fine) | PASS on both square arms; **just outside on both circle arms** (margins 1.20x and 1.24x) |
| sigma_T, curve rel-L2 | REPORTED DIAGNOSTICS -- both still fail their cross-family bands; see C5 |

**Phase A closes as PASSED on the verdict-carrying gates**, with one honest
qualifier recorded rather than absorbed: t90 on the two circle arms clears the
band built by the rule declared before computing, but sits 1.20x-1.24x outside
the stricter reading. C4 shows that residual is a real, converged ~0.5 %
cross-family offset (~1.6 s of ~324 s), not noise, and pins it with a test.

## C2. STEP 1 -- what was measured, and the rule

The combination rule was fixed in `solve3d/gates.py` **before any number was
computed**:

    tolerance = 1.5 x ( heatr3d_spread + dolfinx_spread )

a triangle-inequality SUM, because if both engines converge to the same
continuum answer then |A - B| <= |A - A_inf| + |B_inf - B|. Root-sum-square was
considered and **rejected**: it is strictly smaller than the sum, so it is not a
bound. The 1.5x safety factor is the same one Task 1 used, not a new allowance.

Two readings of the dolfinx half are reported. The rule declared up front took
the MAX over dolfinx's refinement pairs. The measurement then showed that
"MAX" was the wrong sense of conservative for a *gate* -- wider makes passing
easier -- and that the coarse level is genuinely under-resolved (t90 348.25 s
against 325.00 / 324.35 s, and its mesh landed 14 % below its node target). So
the stricter reading (`mid_vs_fine`, the local convergence estimate at the mesh
Task 4 actually ran) is reported alongside and is treated as verdict-carrying
wherever the two disagree. Tightening a gate is always allowed; widening is not.

### Close-out STEP 1a: dolfinx's OWN mesh convergence (extruded circle, coupling off)

| level | in-part nodes | cells | lc_part [m] | t90 [s] | sigma_T [C] | n_sub | energy resid | z-plane spread [C] | wall [s] |
|---|---|---|---|---|---|---|---|---|---|
| coarse | 19821 | 143892 | 0.0009375 | 348.25000000000006 | 20.87107865120885 | 1 | -7.457484702587364e-14 | 0.13464164440975424 | 88 |
| mid | 77043 | 561132 | 0.0005799361188747663 | 325.00000000000006 | 20.699707813129972 | 1 | 8.244600832159246e-14 | 0.08781579352793756 | 1336 |
| fine | 174028 | 1267609 | 0.00043757095942818775 | 324.35 | 20.55128171110689 | 1 | 9.890338809673811e-14 | 0.0360344903452301 | 2924 |

| pair | t90 | curve rel-L2 | sigma_T |
|---|---|---|---|
| coarse_vs_mid | 0.07153846153846152 | 0.004341564835040532 | 0.00827890130749464 |
| mid_vs_fine | 0.0020040080160321693 | 0.0013759311240328113 | 0.007222230910438419 |

Extra level measured for the square (needed because the bed-melt band cannot be borrowed from the circle):

| level | in-part nodes | cells | t90 [s] | sigma_T [C] |
|---|---|---|---|---|
| square_coarse | 25254 | 182531 | 319.05 | 18.568773519122868 |

### Close-out STEP 1b: the measured CROSS-FAMILY band (`parity_tolerances_crossfamily.json`)

Rule, declared before computing: tolerance = 1.5 x (heatr3d_spread + dolfinx_spread), a triangle-inequality SUM. Two readings are reported; the STRICTER one carries the verdict.

| quantity | heatr3d spread | dolfinx spread (declared MAX rule) | band (MAX rule) | dolfinx spread (strict, mid-vs-fine) | band (strict) | Task-1 same-method band |
|---|---|---|---|---|---|---|
| t90_rel | 0.001859024012393564 | 0.07153846153846152 | 0.11009622832628263 | 0.0020040080160321693 | 0.0057945480426386 | 0.0027885360185903457 |
| curve_rel_l2 | 0.00139388743745654 | 0.004341564835040532 | 0.008603178408745608 | 0.0013759311240328113 | 0.004154727842234027 | 0.0020908311561848103 |
| sigma_T_rel | 0.007577767675115235 | 0.00827890130749464 | 0.023785003473914816 | 0.007222230910438419 | 0.02219999787833048 | 0.011366651512672854 |

Four Task-4 arms re-judged from their RECORDED numbers (nothing re-run):

| arm | quantity | measured | pass (MAX-rule band) | pass (strict band) | role |
|---|---|---|---|---|---|
| circle_off | t90_rel | 0.006971340046475777 | PASS | FAIL (margin 1.2x) | verdict |
| circle_off | curve_rel_l2 | 0.004802034113621331 | PASS | FAIL (margin 1.16x) | diagnostic |
| circle_off | sigma_T_rel | 0.03747797106176329 | FAIL | FAIL (margin 1.69x) | diagnostic |
| circle_coupled | t90_rel | 0.007187500000000036 | PASS | FAIL (margin 1.24x) | verdict |
| circle_coupled | curve_rel_l2 | 0.004852415416329581 | PASS | FAIL (margin 1.17x) | diagnostic |
| circle_coupled | sigma_T_rel | 0.04288773854597466 | FAIL | FAIL (margin 1.93x) | diagnostic |
| square_off | t90_rel | 0.00015710919088770265 | PASS | PASS (margin 0.0271x) | verdict |
| square_off | curve_rel_l2 | 0.010785353411702905 | FAIL | FAIL (margin 2.6x) | diagnostic |
| square_off | sigma_T_rel | 0.05073061553298375 | FAIL | FAIL (margin 2.29x) | diagnostic |
| square_coupled | t90_rel | 0.0017333753545540856 | PASS | PASS (margin 0.299x) | verdict |
| square_coupled | curve_rel_l2 | 0.010838452884915077 | FAIL | FAIL (margin 2.61x) | diagnostic |
| square_coupled | sigma_T_rel | 0.04257566336573178 | FAIL | FAIL (margin 1.92x) | diagnostic |

### Close-out STEP 2: SHAPE/DENSITY parity gate (`phase_a_shape_gate.json`)

Band rule: per SHAPE: 1.5 * (that shape's heatr3d n64-vs-n96 spread + that shape's dolfinx coarse-vs-mid spread); both are 1.5x linear refinement pairs. Bands are NOT borrowed across shapes -- the circle's bed melt is identically zero on every grid and cannot bound the square's.

| shape | quantity | heatr3d self-spread | dolfinx self-spread | tolerance |
|---|---|---|---|---|
| circle | jaccard_dist_phi0p8 | 0.03619324608175811 | 0.06048222401451775 | 0.1450132051444138 |
| circle | jaccard_dist_phi0p9 | 0.034414945919370665 | 0.06251774457617931 | 0.14539903574332497 |
| circle | front_ssd_mm | 0.1704499616008767 | 0.29224922039842016 | 0.6940487729989453 |
| circle | bed_melt_absdiff_phi0p8 | 0.0 | 0.0 | 0.0 |
| circle | bed_melt_absdiff_phi0p9 | 0.0 | 0.0 | 0.0 |
| circle | in_part_absdiff_phi0p8 | 0.00944746636129401 | 0.05511022044088174 | 0.09683653020326362 |
| circle | in_part_absdiff_phi0p9 | 0.00887489264242769 | 0.05625536787861438 | 0.0976953907815631 |
| square | jaccard_dist_phi0p8 | 0.06502857142857155 | 0.009884653852340497 | 0.11236983792136807 |
| square | jaccard_dist_phi0p9 | 0.06682853509076181 | 0.009781180799405265 | 0.11491457383525061 |
| square | front_ssd_mm | 0.34616663131277176 | 0.0505974154738299 | 0.5951460701799025 |
| square | bed_melt_absdiff_phi0p8 | 0.02673201158387168 | 0.0008019603475161493 | 0.04130095789708174 |
| square | bed_melt_absdiff_phi0p9 | 0.024727110715081303 | 0.0007462686567164159 | 0.038210069057696576 |
| square | in_part_absdiff_phi0p8 | 0.036645132546224146 | 0.006192916016930239 | 0.06425707284473159 |
| square | in_part_absdiff_phi0p9 | 0.03965248384940967 | 0.005023390510135872 | 0.06701381153931832 |

| arm | quantity | measured | tolerance | margin | result |
|---|---|---|---|---|---|
| circle_off | jaccard_dist_phi0p8 | 0.024847687267488316 | 0.1450132051444138 | 0.1713x | PASS |
| circle_off | jaccard_dist_phi0p9 | 0.02736915875966006 | 0.14539903574332497 | 0.1882x | PASS |
| circle_off | front_ssd_mm | 0.12928334358332805 | 0.6940487729989453 | 0.1863x | PASS |
| circle_off | bed_melt_absdiff_phi0p8 | 0.0 | 0.0 | 0x | PASS |
| circle_off | bed_melt_absdiff_phi0p9 | 0.0 | 0.0 | 0x | PASS |
| circle_off | in_part_absdiff_phi0p8 | 0.01853707414829655 | 0.09683653020326362 | 0.1914x | PASS |
| circle_off | in_part_absdiff_phi0p9 | 0.019295734325794456 | 0.0976953907815631 | 0.1975x | PASS |
| circle_coupled | jaccard_dist_phi0p8 | 0.026954643961821012 | 0.1450132051444138 | 0.1859x | PASS |
| circle_coupled | jaccard_dist_phi0p9 | 0.025955067333010562 | 0.14539903574332497 | 0.1785x | PASS |
| circle_coupled | front_ssd_mm | 0.12257023052756744 | 0.6940487729989453 | 0.1766x | PASS |
| circle_coupled | bed_melt_absdiff_phi0p8 | 0.0 | 0.0 | 0x | PASS |
| circle_coupled | bed_melt_absdiff_phi0p9 | 0.0 | 0.0 | 0x | PASS |
| circle_coupled | in_part_absdiff_phi0p8 | 0.019310048668766046 | 0.09683653020326362 | 0.1994x | PASS |
| circle_coupled | in_part_absdiff_phi0p9 | 0.01912396221013459 | 0.0976953907815631 | 0.1958x | PASS |
| square_off | jaccard_dist_phi0p8 | 0.01423601642476302 | 0.11236983792136807 | 0.1267x | PASS |
| square_off | jaccard_dist_phi0p9 | 0.01617256834821157 | 0.11491457383525061 | 0.1407x | PASS |
| square_off | front_ssd_mm | 0.08234496427629232 | 0.5951460701799025 | 0.1384x | PASS |
| square_off | bed_melt_absdiff_phi0p8 | 0.0007017153040766319 | 0.04130095789708174 | 0.01699x | PASS |
| square_off | bed_melt_absdiff_phi0p9 | 0.00040098017375807536 | 0.038210069057696576 | 0.01049x | PASS |
| square_off | in_part_absdiff_phi0p8 | 0.0007128536422365883 | 0.06425707284473159 | 0.01109x | PASS |
| square_off | in_part_absdiff_phi0p9 | 0.00165961238583201 | 0.06701381153931832 | 0.02477x | PASS |
| square_coupled | jaccard_dist_phi0p8 | 0.013445604951389911 | 0.11236983792136807 | 0.1197x | PASS |
| square_coupled | jaccard_dist_phi0p9 | 0.01294656545938333 | 0.11491457383525061 | 0.1127x | PASS |
| square_coupled | front_ssd_mm | 0.06687323433927521 | 0.5951460701799025 | 0.1124x | PASS |
| square_coupled | bed_melt_absdiff_phi0p8 | 0.0009356204054355083 | 0.04130095789708174 | 0.02265x | PASS |
| square_coupled | bed_melt_absdiff_phi0p9 | 0.0008353753619959895 | 0.038210069057696576 | 0.02186x | PASS |
| square_coupled | in_part_absdiff_phi0p8 | 0.0004455335263978677 | 0.06425707284473159 | 0.006934x | PASS |
| square_coupled | in_part_absdiff_phi0p9 | 0.0010247271107150847 | 0.06701381153931832 | 0.01529x | PASS |

`all_arms_pass` = `true`

Raw shape numbers (dolfinx vs heatr3d) at the melt-onset read:

| arm | IoU phi>=0.8 | IoU phi>=0.9 | front SSD [mm] | in-part melt frac d / h | bed melt frac d / h |
|---|---|---|---|---|---|
| circle_off | 0.9751523127325117 | 0.9726308412403399 | 0.12928334358332805 | 0.8435728600057256 / 0.8628685943315201 | 0.0 / 0.0 |
| circle_coupled | 0.973045356038179 | 0.9740449326669894 | 0.12257023052756744 | 0.8428857715430862 / 0.8620097337532208 | 0.0 / 0.0 |
| square_off | 0.985763983575237 | 0.9838274316517884 | 0.08234496427629232 | 0.8592225439964357 / 0.8608821563822676 | 0.038471820004455336 / 0.038093116507017154 |
| square_coupled | 0.9865543950486101 | 0.9870534345406167 | 0.06687323433927521 | 0.8597460458899532 / 0.8607707730006682 | 0.03848295834261528 / 0.03764758298061929 |

## C3. STEP 2 -- why these metrics, and what they show

Matt's recorded objective (spec commit `d298c6d`): *"make every single part a
fully dense part if and only if it falls within the nominal shape bounds ...
willing to compromise a little bit on density (maybe 80-90 %) if we can achieve
a better shape within the bounds."* That is an asymmetric shape statement, so
Phase A parity is now gated on the melt/density field against the shape bounds,
not on sigma_T.

Both engines are read on ONE shared evaluation grid (0.15 mm pixels over
+-15 mm, five z-planes) and scored against the **analytic** nominal shape, so
neither engine is graded on its own staircase. The anchors are full-height
extrusions, so metrics are computed per z-plane and averaged; the plane-to-plane
spread is reported as a measured z-invariance check (IoU spread 4e-4 to 1.6e-3,
front-distance spread 0.002 to 0.008 mm -- the planar reduction is valid).

The headline agreement is strong:

* melt-region **IoU 0.973-0.987** at both thresholds, against bands of 0.115-0.145
  in Jaccard distance;
* melt-front **symmetric surface distance 0.067-0.129 mm** -- sub-pixel-scale on
  a 20 mm part, against bands of 0.60-0.69 mm;
* in-part melt fraction agrees to 0.0004-0.019 absolute;
* **bed melt**: the circle spills nothing in either engine on any grid; the
  square spills ~3.8-4.2 % of its part volume in **both** engines, agreeing to
  0.0004-0.0009 absolute (about 1-2 % relative).

That last row is the one Matt flagged as the hard side of the objective, and it
is the one that agrees best in relative terms.

### A band-scope error caught and MEASURED, not widened

The first shape-gate run FAILED the square's bed-melt check with a tolerance of
exactly 0.0. The cause was scope, not physics: the band had been built from the
circle, and the circle has **identically zero** out-of-part melt at every grid
level of both engines, so a circle-derived bed-melt band is degenerate and
cannot bound a square-only quantity. This is the same class of error as the
original Task-4 tolerance (a same-method band cannot bound a cross-family
difference) one level down.

The fix was to MEASURE the missing spread, not to widen: a heatr3d square at
n=64 and a dolfinx square at the coarse mesh were run, giving the square its own
band (0.0382-0.0413), which the measured difference (0.0004-0.0009) clears by
40-100x. The rule is now uniform and stated: each shape's band uses that
shape's own heatr3d n=64-vs-n=96 spread and that shape's own dolfinx
coarse-vs-mid spread, both 1.5x linear pairs. Bands are never borrowed across
shapes.

Degenerate (zero-width) bands are kept as zero rather than padded, and flagged
`degenerate_zero_width` in the JSON: for the circle it correctly means "both
engines agree there is no spill, so any spill would be a genuine disagreement."

## C4. The one open item: a converged t90 offset

Under the strict band, t90 on the two circle arms sits at margins 1.20x and
1.24x. This is not a resolution artifact:

| | coarse / n=64 | mid / n=96 | fine |
|---|---|---|---|
| dolfinx t90 [s] | 348.25 | 325.00 | 324.35 |
| heatr3d t90 [s] | 323.35 | 322.75 | -- |

Both engines have essentially stopped moving (dolfinx -0.65 s over its last
refinement, heatr3d -0.60 s over its only pair) and they stop **~1.6 s apart**,
i.e. **~0.5 %**. The premise behind the band -- that the two discretizations
converge to the *same* continuum answer -- is what fails, mildly. The most
likely cause is the one already characterized in section 5: the surface-region
drive difference (Q_rf pattern rel-L2 0.108). The square arms do not show it
(t90 rel diff 0.000157 and 0.00173), which is consistent with the square having
no curved boundary to staircase.

Pinned by `test_t90_strict_band_residual_is_pinned_where_it_was_measured` so it
cannot silently grow. Settling whether 324.3 s or 322.8 s is closer to truth
needs a third reference -- the COMSOL 3-D anchor, Gate S3.

## C5. Diagnostics that still fail (reported, not verdict-carrying)

Per Matt's recorded objective hierarchy, sigma_T is a flatness diagnostic and
the heating curve is a trajectory diagnostic. Both are reported with their
cross-family bands and both still fail:

* **sigma_T**: measured 0.0375-0.0507 against a strict band of 0.0222. The cause
  is unchanged from section 5 -- the melt-onset FIELDS agree to ~1 % rel-L2 and
  sigma_T is a standard deviation of that field, so a ~1 % field difference
  amplifies into a 3.7-5.1 % spread difference. Note that the shape metrics,
  computed from the same fields, pass comfortably: sigma_T is the harshest
  possible reading of an agreement that is good where the objective cares.
* **curve rel-L2**: measured 0.0048-0.0108 against a strict band of 0.0042. The
  square arms are the worse pair.

Neither blocks Phase A under the approved verdict rule, and both are recorded
so Phase B/C inherit an honest picture rather than a clean-looking one.

## C6. Close-out red-green evidence

| item | RED | GREEN |
|---|---|---|
| shape-metric math | `ImportError: cannot import name 'shape_metrics' from 'solve3d'` | 8 passed (IoU closed form, 1.0 mm radial offset recovered to <0.15 mm, empty-pair NaN not false-green, SUM rule > RSS) |
| close-out gates | 7 skipped (artifacts absent) -> after compute | 17 passed |
| square bed-melt band | `ZeroDivisionError` then a genuine FAIL at tolerance 0.0 | PASS after the square's own spread was MEASURED |
| field export determinism | -- | all four arms reproduce their recorded t90 and sigma_T to **0.0** relative |

Full close-out suite: `24 passed` (`test_closeout.py`, `test_shape_metrics.py`,
`test_gates.py`, geo-prewarp venv).

## C7. What the close-out does NOT change

* No adjoint, objective, regularization, eps_r channel, densification or Studio
  work happened. Phases B-E are untouched.
* The coupled arm is still EXPLORATORY (`sigma_temp_coeff_per_K = -0.002` /K has
  no validated provenance in this repo).
* Still two grids per engine per shape (three for the dolfinx circle). This is a
  convergence *estimate*, not a convergence study.
* No third-engine reference exists, so "which engine is right" remains open
  (Gate S3 / COMSOL).
* The shape metrics are evaluated at the melt-onset read state only. Densified
  final geometry and shrinkage are P2 work and are not scored here.

## C8. Close-out reproduction

    # heatr3d side (geo-prewarp venv)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m solve3d.cases --anchor square 64

    # dolfinx side (spike env)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 heatr3d_d1_spike/env/bin/python -m solve3d.closeout --refine
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 heatr3d_d1_spike/env/bin/python -m solve3d.closeout --refine-level square coarse 30976 0.0009375 square_off
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 heatr3d_d1_spike/env/bin/python -m solve3d.closeout --export-fields

    # gates (geo-prewarp venv; crossfamily runs in either)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m solve3d.crossfamily
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m solve3d.shape_gate
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./.venv312/bin/python -m pytest solve3d/tests/test_closeout.py solve3d/tests/test_shape_metrics.py
