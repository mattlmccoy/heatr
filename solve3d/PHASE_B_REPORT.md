# solve3d Phase B: transient adjoint -- gate report

Plan: `docs/superpowers/plans/2026-08-01-solve3d-phase-b-transient-adjoint.md`
Spec: `docs/superpowers/specs/2026-07-31-solve-port-3d-design.md`
Conventions ported from: `FROZEN_CONVENTIONS_2D.md` (repo root, commit b04e356)
Semantic template (READ ONLY): `fgm_solve_campaign/adjoint2d/`
Date: 2026-08-01. Environment: `heatr3d_d1_spike/env` (dolfinx 0.11.0, complex
scalar build, `jit_fix` first). `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` on
every run.

**Every number below is PRINTED from `solve3d/results/*.json` by
`solve3d/make_phase_b_tables.py`.** Nothing is transcribed. Regenerate with:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.make_phase_b_tables

---

## Verdict

**Phase B passes.** A gradient through the FULL Phase A coupled forward --
steady complex EQS, in-march EQS re-solves with sigma(T) coupling, the enthalpy
thermal-phase march with its melt clips, and the fixed-power renormalization --
is FD-gated at the frozen 2-D thresholds on four layers, both mutants are
rejected, the envelope stop-time rule is exact, and the checkpointed gradient
costs **1.774 forward-equivalents** against a `<= ~2` target while holding
**6.67x less march state**.

The strongest single number: at layers B2, B3 and B4 **all four probes clear the
PREFERRED 1e-6 standard**, not merely the 1e-5 subgradient standard the frozen
convention would have allowed -- and that is with the melt-fraction clips live
(437 of 3091 nodes strictly inside the melt window at the read state).

One honest qualifier, carried from Task 1: on the B1 steady layer at 105191
design dofs, one probe of four reads 1.42e-06 against the preferred 1e-6. It
clears the frozen 1e-5 standard, and section 4 shows with measurements that it
is a finite-difference NOISE FLOOR, not a gradient error.

---

## Red/green evidence, per task

| task | RED (observed) | GREEN |
|---|---|---|
| 0 protocol | 6 failed -- protocol artifact missing (hard assert, deliberately not a skip) | 6 passed |
| (gate machinery) | `ImportError: cannot import name 'gate_fd'` | 8 passed, incl. that it rejects a 5%-wrong gradient and that its measured floor tracks both double precision and an injected quantizer |
| 1 steady EQS | `ImportError: ... 'adjoint'` (4 errors); then 1 failed on an over-strict `all_pass_preferred` assertion I had invented | 5 passed |
| 2 transient | `AttributeError: ... 'TransientCase'` | 5 passed |
| 3 design + 4 envelope | 9 failed (`AttributeError`, all runners missing) | 7 passed |
| 5 checkpointing | same red | 2 passed |

Full Phase B suite: **33 passed, 0 failed**, as the sum of the six per-file
runs each verified above: `test_phase_b_protocol.py` 6, `test_gate_fd.py` 8,
`test_adjoint_steady.py` 5, `test_adjoint_transient.py` 5,
`test_adjoint_design.py` 7, `test_checkpointing.py` 2.

COST NOTE for anyone re-running it in one invocation: the layer drivers are not
cached across tests, so the three envelope tests each rebuild the 300 s case and
re-run its 64-forward sweep, and the two design tests each re-run theirs. That
makes a single combined invocation roughly an hour, against about 24 minutes if
the files are run separately. The per-file counts above are what was measured;
caching the drivers the way `run_fd_gate(reuse=True)` already does for B1 is the
obvious cleanup and is NOT done.

---

## What was built

| file | role |
|---|---|
| `solve3d/phase_b_protocol.py` | Task 0: pre-registers the protocol and MEASURES the FD case |
| `solve3d/gate_fd.py` | the ported FD/subgradient machinery; holds NO thresholds of its own (all read from the protocol JSON) |
| `solve3d/adjoint.py` | B1 steady EQS adjoint (lifted from `adjoint_core.py`) + B2/B3/B4 transient reverse march, design map, envelope read, checkpointing |
| `solve3d/transient_gate.py` | the layer drivers that emit the gate JSONs |
| `solve3d/make_phase_b_tables.py` | prints every table in this report from the JSONs |

`solve3d/forward.py` gained exactly ONE hook (`record`), inert when `None` and
pinned bit-identical (`max|dT| == 0.0`). Nothing else outside `solve3d/` was
touched; `heatr3d.py`, `rfam_eqs_coupled.py`, `dissertation_materials/` and
`fgm_solve_campaign/adjoint2d/` are unmodified.

---

### Pre-registered protocol (`phase_b_protocol.json`)

Thresholds source: FROZEN_CONVENTIONS_2D.md (repo root, commit b04e356), section 5

| item | value | citation |
|---|---|---|
| epsilon sweep | [0.001, 0.0001, 1e-05, 1e-06, 3e-07, 1e-07, 3e-08, 1e-08] | FROZEN_CONVENTIONS_2D.md section 5 item 4 (adjoint2d/gate_rho.py:47-52) |
| pass_rel_err | 1e-06 | FROZEN_CONVENTIONS_2D.md section 5 item 5 (gate_rho.PASS_REL_ERR) |
| subgradient_pass_rel_err | 1e-05 | FROZEN_CONVENTIONS_2D.md section 5 item 5 (gate_rho.SUBGRADIENT_PASS_REL_ERR) |
| transpose_rel_err | 1e-10 | FROZEN_CONVENTIONS_2D.md section 5 item 8 (2-D measured 4.19e-16 worst) |

| checklist item | status |
|---|---|
| 1. L0 bit identity: the forward reproduces a stored run in the same channel to th... | ported |
| 2. Layered bisect: add ONE thing per layer so a failure localizes... | ported |
| 3. Three-probe protocol per layer, plus two: max-sensitivity cell, fixed pseudo-r... | ported |
| 4. Central differences, epsilon swept over eight values, expect a V-shaped relati... | ported |
| 5. Pass standard 1e-6 preferred, 1e-5 is the campaign SUBGRADIENT standard; repor... | ported |
| 6. Measure the evaluation floor, do not assume it: in the roundoff-dominated tail... | ported |
| 7. Interpret a relative-error failure against the analytic magnitude; report the ... | ported |
| 8. Filter and projection transpose exactness by the dot-product identity, thresho... | ported |
| 9. Read-state stability: report whether the objective's argmin index moves under ... | ported |
| 10. Flag-off bit identity for every new channel... | ported |

FD case, MEASURED (not asserted):

| quantity | value |
|---|---|
| `wall_forward_s` | 1.9343576660612598 |
| `n_dofs_total` | 3091 |
| `n_cells_total` | 16549 |
| `n_eqs_solves` | 4 |
| `resolve_times_s` | [25.0, 50.0, 75.0] |
| `n_march_steps` | 200 |
| `part_mean_phi` | 0.6694775218214674 |
| `n_nodes_in_melt_window` | 437 |
| `n_nodes_total` | 3091 |
| `n_substeps_used` | 1 |
| `dt_stable_s` | 2.1770063475094665 |
| `energy_residual_frac` | 5.1763061469314535e-17 |
| `clamp_bound` | False |

### Assembly consistency and standing identities

| quantity | value |
|---|---|
| B1 assembly consistency (LU vs LU) | 3.1540875622780237e-12 |
| B1 same, vs Phase A's GMRES (recorded, NOT gated) | 2.0812190969942587e-07 |
| B1 Qbar / power_density - 1 | 2.220446049250313e-16 |
| B1 Q clip active | False |
| B1 LU residual norm | 1.6012024099836067e-19 |

### Every FD gate, all probes (the count at BOTH standards, per checklist item 5)

| layer | probe | best rel err | best abs err | analytic directional | best eps | measured floor | pass 1e-6 / 1e-5 |
|---|---|---|---|---|---|---|---|
| B1 steady | max_sensitivity_cell | 1.104527565131573e-07 | 0.0019181353745807428 | 17366.11593167665 | 0.0001 | 8.422896066857408e-09 | yes / yes |
| B1 steady | random_cell | 1.4242701146841632e-06 | 0.0002557940224505728 | -179.59656655949425 | 0.0001 | 9.620999247809715e-09 | no / yes |
| B1 steady | random_direction | 9.013859686433251e-07 | 0.0003315689598935023 | 367.8434892796771 | 0.001 | 7.996217529688965e-09 | yes / yes |
| B1 steady | gradient_direction | 2.2308422808276671e-10 | 0.00013168249279260635 | 590281.5000608232 | 0.001 | 3.407626406348319e-08 | yes / yes |
| B2 transient | max_sensitivity_cell | 1.2326511054200018e-10 | 1.351688559783512e-17 | -1.0965702734862275e-07 | 0.001 | 8.307896676353507e-21 | yes / yes |
| B2 transient | random_cell | 3.786061055641351e-09 | 7.768279779973322e-17 | 2.051810487418934e-08 | 1e-05 | 9.4894258178699e-21 | yes / yes |
| B2 transient | random_direction | 3.4346550563482964e-07 | 1.4199872176997772e-15 | 4.134293529928734e-09 | 0.0001 | 9.333185779959916e-21 | yes / yes |
| B2 transient | gradient_direction | 1.7938131580356196e-10 | 2.52467487291564e-16 | 1.4074346938565092e-06 | 0.0001 | 1.2145585658128507e-20 | yes / yes |
| B3 design | max_sensitivity_cell | 2.6825999526405e-09 | 1.4808452023720318e-17 | -5.5201864926390776e-09 | 0.001 | 7.920799856915721e-21 | yes / yes |
| B3 design | random_cell | 5.775569349560013e-10 | 6.607309869968915e-19 | 1.1440101347709148e-09 | 0.0001 | 1.339710628177043e-20 | yes / yes |
| B3 design | random_direction | 7.202924643929975e-08 | 7.564790195659192e-18 | 1.0502386974205217e-10 | 0.001 | 7.386402593484227e-21 | yes / yes |
| B3 design | gradient_direction | 3.737743527274459e-10 | 3.190763847884089e-17 | 8.536604570647936e-08 | 0.0001 | 1.714694874742689e-21 | yes / yes |
| B4 envelope | max_sensitivity_cell | 6.958327663348178e-09 | 8.577142808733608e-17 | 1.2326442823197688e-08 | 0.001 | 2.019827436705042e-20 | yes / yes |
| B4 envelope | random_cell | 3.700495096095019e-09 | 9.400120806107468e-19 | -2.540233282845593e-10 | 1e-05 | 4.301824853267298e-20 | yes / yes |
| B4 envelope | random_direction | 3.7019141822293514e-08 | 5.782346018739469e-17 | 1.5619881321120329e-09 | 0.0001 | 1.923796510335292e-20 | yes / yes |
| B4 envelope | gradient_direction | 1.5697784852732066e-08 | 2.6577651100414293e-15 | 1.6930828998964575e-07 | 1e-05 | 6.955981098681722e-21 | yes / yes |

| layer | probes passing 1e-6 | probes passing 1e-5 |
|---|---|---|
| B1 steady | 3 / 4 | 4 / 4 |
| B2 transient | 4 / 4 | 4 / 4 |
| B3 design | 4 / 4 | 4 / 4 |
| B4 envelope | 4 / 4 | 4 / 4 |

### Mutation tests -- both mutants MUST fail the gate

| layer | mutant | best rel err | passes 1e-5? | worst per-dof deviation vs the true gradient |
|---|---|---|---|---|
| B1 steady | `renorm_frozen` | 0.005344497035039635 | no -- REJECTED | 14036.702402320509 |
| B1 steady | `adjoint_dropped` | 0.237423062173416 | no -- REJECTED | 129968.97938819272 |
| B2 transient | `renorm_frozen` | 0.11470945675270387 | no -- REJECTED | 3547.4094874460266 |
| B2 transient | `adjoint_dropped` | 0.22606297506552453 | no -- REJECTED | 5256.172479606137 |

### B4 envelope stop time

| quantity | value |
|---|---|
| argmin step | 314 of 600 |
| at_horizon | False |
| argmin moves under probes (eps 0.001) | False |
| J at start / argmin / horizon | 1.6145721431616122e-05 / 6.732798390911482e-07 / 7.674796070028112e-06 |
| envelope vs fixed-index gradient, max abs diff | 0.0 |
| envelope vs fixed-index gradient, rel diff | 0.0 |

Per-probe argmin under perturbation:

| probe | argmin |
|---|---|
| max_sensitivity_cell+ | 314 |
| max_sensitivity_cell- | 314 |
| random_cell+ | 314 |
| random_cell- | 314 |
| random_direction+ | 314 |
| random_direction- | 314 |
| gradient_direction+ | 314 |
| gradient_direction- | 314 |

### Cost, in forward-equivalents

Accounting rule: wall time of ONE gradient evaluation (reverse march + all adjoint solves, EXCLUDING the forward that produced the trajectory) divided by the wall time of one forward on the same case and machine

Target: <= ~2 forward-equivalents CHECKPOINTED | 2-D reference: 1.4-1.7 store-everything under 2-D memory conditions; 3-D must checkpoint

| scheme | wall gradient [s] | forward-equivalents | stored state [B] | recomputed steps | gradient vs store-everything |
|---|---|---|---|---|---|
| store-everything | 1.8928094169823453 | 1.3610637961032714 | 4945600 | 0 | - |
| interval 10 | 2.466506000026129 | 1.77359220077167 | 741840 | 180 | max rel diff 0.0 |
| interval 20 | 2.3839003330795094 | 1.7141928858563547 | 741840 | 190 | max rel diff 0.0 |
| interval 50 | 2.494874457945116 | 1.7939911277203582 | 1335312 | 196 | max rel diff 0.0 |

Chosen: interval 10, 1.77359220077167 forward-equivalents, 741840 B (6.67x less state than store-everything).

Scheme note: INTERVAL checkpointing (uniform anchors + forward recompute of one segment at a time). Full binomial Griewank was NOT needed: the interval scheme already meets the <= ~2 forward-equivalent target, so the extra machinery would buy nothing measurable here. That justification is a measurement, not a preference.

Wall forward on the FD case: 1.3906838330440223 s over 200 march steps.


---

## 4. The one preferred-standard miss, and why it is a floor and not an error

On B1 the `random_cell` probe reads 1.4243e-06 against the preferred 1e-6. It
clears the frozen 1e-5 standard, and three independent measurements say it is
the finite-difference evaluation floor:

1. **The absolute error is flat while the derivative is not.** Across the four
   B1 probes the best absolute error spans 1.3168e-04 to 1.9181e-03, a factor of
   15, while the analytic directional derivative spans 1.7960e+02 to
   5.9028e+05, a factor of 3286. A gradient that is *wrong* produces an error
   proportional to the derivative; a *floor* produces a constant absolute error,
   and the relative number is then just 1/|analytic|.
2. **The probe that misses is exactly the smallest-derivative probe.** This is
   asserted by a test, not observed by eye
   (`test_the_preferred_standard_miss_is_a_measured_floor_artifact`).
3. **The floor-free probe is clean by four orders.** The gradient direction,
   whose derivative is the largest available (5.9028e+05), reads 2.2308e-10.

This is the identical signature D1 recorded for the same solve
(`heatr3d_d1_spike/results.json` `task5.gate.fd_noise_floor_note`), and its
physical cause is also the same: gamma spans `sigma_doped/sigma_virgin` = 4e6,
so the LU backward error sets the floor on J. Two iterative-refinement sweeps
are already applied.

**No threshold was widened anywhere in Phase B.** What changed at this step was
that I removed an assertion *I* had invented (`all_pass_preferred` on B1),
which is not the frozen convention: FROZEN_CONVENTIONS_2D.md section 5 item 5
sets 1e-5 as the pass standard, calls 1e-6 preferred, and asks for the count at
both. The report gives the count at both.

## 5. Deviations from the plan, each named

1. **A second case for layer B4.** The plan assumed the envelope gate could run
   on the pre-registered FD case. It cannot: MEASURED, that case's argmin sits
   at step 200 of 200, i.e. AT the horizon, where `dJ/dt` is not zero, the
   envelope argument does not apply, and the gate would have been vacuous.
   Extending the horizon 100 s -> 300 s puts the argmin at step 314 of 600,
   interior, because the part finishes melting and then the bed starts melting
   and drives J back up. Only the horizon changed; every threshold is the
   pre-registered one. The 2-D lane's `at_horizon` flag is implemented and
   reported for exactly this reason.
2. **Mutation tests use the gradient-direction probe, not all four.** One probe
   is enough to disqualify a gradient, and the gradient direction has the
   highest signal-to-floor ratio -- which is what D1 used
   (`task5.mutations` reports a directional relative error). The full four-probe
   sweep is spent on the real gradient, where it buys information.
3. **The B1 gate artifact is cached (`reuse=True`).** That sweep is 8 epsilons x
   4 probes x 2 solves on a 24784-dof complex LU and takes 39 minutes measured.
   The artifact comes from a real run; `reuse=False` forces a fresh one.
4. **`assembly_consistency_gate` is the LU-vs-LU comparison.** Comparing the
   adjoint's LU path against Phase A's production GMRES mixes two questions.
   MEASURED: LU-vs-LU agrees to 3.15e-12 (B1) and 5.07e-13 (B2), while the same
   operator read against GMRES differs by 1.83e-07 -- which is Phase A's
   `ksp_rtol = 1e-10` showing through `|E|^2`, not an operator difference. Both
   are recorded; the gate is on the former.

## 6. What Phase B does NOT cover

* **No optimizer loop.** No L-BFGS-B, no budget accounting, no multi-start. The
  gradient exists and is gated; nothing has been optimized with it.
* **No regularization chain.** No filter, no Heaviside projection, no filter
  transpose in 3-D. That is Phase C. The transpose-check discipline is already
  in the protocol and was applied to the two linear operators Phase B does
  introduce (the P1 cell-average feeding the sigma(T) coupling, and the design
  map), both at the frozen 1e-10 threshold.
  Note for Phase C, from the 2-D lane's `MMA_RETEST_REPORT.md`: the chain I am
  NOT building has a settled shape -- **filter-only is the in-grid production
  recipe and the projection is the robustness arm**. Phase C should carry both
  arms rather than treat projection as the default.
* **No permittivity channel.** Conductivity only, per FROZEN_CONVENTIONS_2D
  section 7, which records that the eps channel is MODEL ONLY with no
  deployment path pending the binder question. Phase D.
* **No chi from STL, no shape objective as Phase C will define it.** The Phase B
  objective is a melt-state functional on the analytic part indicator; it exists
  to have something differentiable to gate, not to be the design objective.
* **No density (rho) co-state.** The Phase A forward holds `rho_rel` fixed, so
  the adjoint matches the forward exactly. The 2-D lane's coupled (T, rho)
  reverse march is the template for when `densify` ports into solve3d; until
  then this is a matched simplification, not a missing term.
* **One shape, one mesh, one design point per layer.** The FD case is a coarse
  circle with `dt_s = 0.5 s` and 2x power density, chosen so central differences
  are affordable. It gates the DISCRETE gradient of the DISCRETE forward, which
  is what an adjoint must reproduce; it is not a physics claim and no Phase A
  number is restated from it.
* **Cost measured on one case.** 1.774 forward-equivalents is the small FD case.
  The ratio at Phase A mesh sizes is unmeasured, and FROZEN_CONVENTIONS_2D
  section 6 records that machine load can move such a ratio by an order of
  magnitude, which is why the budget rule uses a recorded campaign ratio rather
  than a freshly timed one.
* **Phase A's open item is untouched by design.** The converged ~1.6 s circle
  t90 cross-family offset is a forward-model question for Gate S3. The adjoint
  differentiates the dolfinx forward as it is.

## 7. Reproduction

    # protocol (measures the FD case)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_b_protocol

    # layer drivers
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.transient_gate

    # full suite (B1's 39-minute sweep is cached; delete
    # results/phase_b_steady_gate.json to force it)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_phase_b_protocol.py \
      solve3d/tests/test_gate_fd.py solve3d/tests/test_adjoint_steady.py \
      solve3d/tests/test_adjoint_transient.py solve3d/tests/test_adjoint_design.py \
      solve3d/tests/test_checkpointing.py
