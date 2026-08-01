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
