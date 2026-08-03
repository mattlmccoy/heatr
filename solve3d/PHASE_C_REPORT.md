# solve3d Phase C: first 3-D solve -- the cylinder-null demonstration

Plan: `docs/superpowers/plans/2026-08-02-solve3d-phase-c-first-solve.md`
Spec: `docs/superpowers/specs/2026-07-31-solve-port-3d-design.md`
Conventions: `FROZEN_CONVENTIONS_2D.md` (their b04e356)
Pre-registration: `solve3d/results/phase_c_preregistration.json`, committed in
`08e3b97` BEFORE any objective or solve code existed.
Date: 2026-08-02. Spike env (dolfinx 0.11.0 complex, `jit_fix` first, scipy
1.17.1 -- see the environment deviation below). `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1` on every run.

**Every number below is PRINTED from `solve3d/results/*.json` by
`solve3d/make_phase_c_tables.py`.** Nothing is transcribed.

---

## The answer

**The direct solve finds improvement the inversion heuristic cannot, and the
map earns the SOLVED label.**

On the corrected-field extruded circle -- the cylinder null, where the
inversion rule's own re-run yields **+0.3 % sigma_T (nothing)** -- the solve
reduces the primary asymmetric objective by **10.67 %** against the uniform
baseline, and the win **survives both acceptance gates**:

* mesh hold-out (solve on the Phase A coarse mesh, score on the mid mesh):
  PASS on every banded metric, and the solved map still beats uniform at the
  finer mesh (7.982756e-08 against 8.273248e-08, a 3.51 % margin). The map also
  moves LESS across meshes than the uniform arm does (22.4 % against 28.1 %),
  so it is not more mesh-sensitive than the forward itself.
* smoothing robustness: J changes 0.217 % under a 0.5 mm sub-filter-radius
  blur, against a 10 % tolerance -- a 46x margin.

`solved_label = true`. For context, `TOPOPT_REPORT.md` records that on the 2-D
lane's own pass **no** map earned that label under all forms.

Two qualifications stated up front, because the headline depends on them:

1. **The pre-registered arms as written produced nothing, and the result comes
   from a RECORDED-DEVIATION arm.** Both pre-registered filter-only arms
   stalled after 2 of 12 gradient evaluations, delivering v ~= 1. That was
   diagnosed, not patched: it is the FROZEN section 1.3 upper-rail hazard, not
   the physics (section 3 below). The remedy -- rescaling J by the constant
   1/|g0| -- is a pure reparameterization that cannot move the minimizer, and
   it was run as an ADDITIONAL arm without displacing the pre-registered ones,
   which remain in the results exactly as they ran.
2. **The in-grid margin (10.67 %) is about three times the cross-mesh margin
   (3.51 %).** The win transfers; its size does not. Both numbers are reported
   everywhere rather than the flattering one alone.

---

### The arms (pre-registered priority order)

| priority | arm | status | J_asymmetric (PRIMARY) | J_symmetric | IMPROVEMENT vs uniform (primary; + is better) |
|---|---|---|---|---|---|
| 1 | uniform_baseline | scored | 1.1511359005643762e-07 | 9.855033669884453e-08 | reference |
| 3 | inversion_map | **DROPPED** | - | - | transfer moved total in-part dopant beyond the pre-registered 2 % drop condition |
| 2 | solve_filter_only_symmetric | converged (2/12 evals) | 1.1511359005322288e-07 | 9.855033669739705e-08 | +0.00 % |
| 2/dev | solve_filter_only_asymmetric | converged (2/12 evals) | 1.1511359004906023e-07 | 9.855033669563119e-08 | +0.00 % |
| 2/dev | solve_filter_only_asymmetric_scaled | budget_exhausted (12/12 evals) | 1.0283280687027586e-07 | 9.491814369786978e-08 | +10.67 % |
| 4 | solve_projection_beta_continuation | **NOT_RUN** | - | - | session compute exhausted; pre-registered unrun_arms_policy applies |
| 5 | solve_filter_only_w3 | **NOT_RUN** | - | - | session compute exhausted; pre-registered unrun_arms_policy applies |

### Cold-start stationarity diagnostic (why the pre-registered arms stalled)

| objective | \|g\| | \|Pg\| | \|Pg\|/\|g\| | best probe t | dJ at best | verdict |
|---|---|---|---|---|---|---|
| symmetric | 1.688696607389328e-09 | 1.203025391916237e-09 | 0.7123987734990932 | 2.0 | -0.019841884492188093 | rail_stall_with_available_descent |
| asymmetric | 3.75646091110831e-09 | 2.7185137000082866e-09 | 0.7236901339687344 | 2.0 | -0.03848191657494872 | rail_stall_with_available_descent |

Line probe along the projected steepest-descent direction:

| objective | t | J | rel change vs v=1 |
|---|---|---|---|
| symmetric | 0.5 | 9.79798903392282e-08 | -0.005788375552310226 |
| symmetric | 2.0 | 9.659491230140297e-08 | -0.019841884492188093 |
| symmetric | 10.0 | 9.764495413247917e-08 | -0.009187006322822325 |
| symmetric | 50.0 | 3.225806561620884e-07 | 2.273257778386261 |
| asymmetric | 0.5 | 1.1381750035277977e-07 | -0.011259224067569877 |
| asymmetric | 2.0 | 1.1068379848725263e-07 | -0.03848191657494872 |
| asymmetric | 10.0 | 1.1363840020067518e-07 | -0.01281507991410167 |
| asymmetric | 50.0 | 3.6087790310283616e-07 | 2.134972186393233 |

### The solve trajectory (scaled-start arm, asymmetric primary)

| eval | J | t_stop [s] | at_horizon | \|grad\| | map mean | wall [s] |
|---|---|---|---|---|---|---|
| 1 | 1.151135900564477e-07 | 358.6 | false | 3.75646091110831e-09 | 1.0 | 466 |
| 2 | 1.1327919891855508e-07 | 358.6 | false | 3.2726310348293058e-09 | 0.9985695102145788 | 913 |
| 3 | 1.0765863160774848e-07 | 358.3 | false | 1.062111815870677e-09 | 0.9891900557393776 | 1348 |
| 4 | 1.0712506126267239e-07 | 358.3 | false | 7.708802705035177e-10 | 0.9873939140494286 | 1785 |
| 5 | 1.063837983086363e-07 | 358.35 | false | 8.749980397885989e-10 | 0.9831466816356917 | 2220 |
| 6 | 1.0549954854924422e-07 | 358.40000000000003 | false | 1.0913470153017507e-09 | 0.9759400643746609 | 2654 |
| 7 | 1.0492329962076859e-07 | 358.5 | false | 1.6905954819252321e-09 | 0.9654548459810979 | 3088 |
| 8 | 1.0422351925500967e-07 | 358.5 | false | 6.938332177326176e-10 | 0.9629299382878533 | 3522 |
| 9 | 1.0390987834185277e-07 | 358.5 | false | 6.047979017053157e-10 | 0.9608250822176043 | 3960 |
| 10 | 1.0359388812415053e-07 | 358.5 | false | 7.176668959712909e-10 | 0.9577449784785652 | 4394 |
| 11 | 1.0297993916671718e-07 | 358.5 | false | 9.18450502860215e-10 | 0.9495486045531635 | 4828 |
| 12 | 1.0283280687027586e-07 | 358.5 | false | 2.186095091513688e-09 | 0.933441006102861 | 5267 |

status `budget_exhausted`, 12 of 12 gradient evaluations, 39.936 forward-equivalents spent, wall 5267 s.
Delivered map: mean 0.933441006102861, min 0.6833835718234699, max 0.9999837652694947.

### The decisive comparison (solve mesh, same read rule)

| quantity | uniform | solved | change |
|---|---|---|---|
| J asymmetric (PRIMARY) | 1.1511359005643762e-07 | 1.0283280687027586e-07 | -10.67 % |
| J symmetric (control, 2-D comparable) | 9.855033669884453e-08 | 9.491814369786978e-08 | -3.69 % |
| J out-of-bounds (bed growth) | 2.7419553229641498e-08 | 2.1265418490449242e-08 | -22.44 % |
| J in-bounds deficit | 8.769403682679613e-08 | 8.156738837982663e-08 | -6.99 % |
| out-of-part melt / part volume | 0.00238498203886267 | 0.0022676046529119453 | -4.92 % |
| in-bounds fraction below the 0.85 floor | 0.08423965346900332 | 0.08206148388886499 | -2.59 % |
| part mean melt fraction | 0.9191950888049109 | 0.9197733582201735 | +0.06 % |
| sigma_T [C] (DIAGNOSTIC, never optimized) | 21.41672479024056 | 21.230714791387136 | -0.87 % |
| in-part melt fraction at phi>=0.9 | 0.9222874320068708 | 0.9225450901803608 | +0.03 % |
| bed melt fraction at phi>=0.9 | 0.0 | 0.0 | n/a |

### Acceptance gates

**Mesh hold-out** (phase_a_coarse -> phase_a_mid), map transfer moved total in-part dopant by 0.0006465387111868232:

| metric | measured move | band | uniform's OWN move | pass |
|---|---|---|---|---|
| J_rel | 0.22371504427265723 | 0.42194553862546325 | 0.2812970257503088 | PASS |
| in_part_absdiff_phi0p8 | 0.0020469510449472317 | 0.08266533066132262 | 0.0026481534497566628 | PASS |
| in_part_absdiff_phi0p9 | 0.001846550243343792 | 0.08438305181792158 | 0.0022187231606068947 | PASS |
| bed_melt_absdiff_phi0p8 | 0.0 | 0.0 | 0.0 | PASS |
| bed_melt_absdiff_phi0p9 | 0.0 | 0.0 | 0.0 | PASS |

| J at each mesh | solve mesh | score mesh |
|---|---|---|
| solved | 1.0283280687027586e-07 | 7.982756092861049e-08 |
| uniform | 1.1511359005643762e-07 | 8.27324795501214e-08 |

**Smoothing robustness**: blur at 0.0005 m (filter radius 0.001 m), J 1.0283280687027586e-07 -> 1.0305643673414576e-07, relative change 0.0021746937643353577 against a tolerance of 0.1 -> PASS.

**SOLVED label**: `true` (beats uniform true, in-grid margin 0.1066840429539272; hold-out true; smoothing true).


---

## 3. Why the pre-registered arms stalled, and why that is not the null

Both pre-registered filter-only arms stopped after 2 gradient evaluations on
L-BFGS-B's factr criterion with v ~= 1 (map min 0.99999999999). There were
exactly two explanations with opposite meanings, so they were separated by
measurement rather than argument:

* **(a) uniform is a constrained stationary point.** Then the projected
  gradient is ~0, and "uniform is locally optimal" would BE the cylinder-null
  answer under that objective -- the physically meaningful null the
  pre-registration explicitly anticipates.
* **(b) the projected gradient is nonzero and the optimizer stalled anyway.**
  Then it is the FROZEN section 1.3 upper-rail hazard: the cold start sits ON
  the upper box bound, L-BFGS-B's first trial step is O(1/|g|), and with the
  measured |g| ~ 1.7e-09 that step is ~6e8, which leaves the box instantly, so
  the line search collapses.

The measurement says **(b)**, unambiguously: the projected gradient is 71-72 %
of the full gradient, and a line probe along it reduces J by 1.98 %
(symmetric) and 3.85 % (asymmetric) with a clear interior optimum at t = 2
(J turns back up by t = 10 and blows up past +200 % by t = 50).

**Reporting those two arms as the cylinder null would have been wrong.** They
measure an optimizer's first step, not the physics.

## 4. The inversion arm was DROPPED, and its provenance was proven first

Order matters here. The arm's provenance was established BEFORE the drop, so
the drop is about the transfer and not about doubt over which map it is:

* `heatr3d.make_fgm` regenerated from the stored masked-baseline `T_phi90`
  reproduces the recorded map to **mean saturation 0.40740740740740733 against
  FGM_BENEFIT_RERUN.md's 0.4074, relative 1.8e-05** (tolerance 1e-3), with the
  2-bpp levels {0, 1/3, 2/3, 1}. This IS the published cylinder-null map.
* the pre-registered trilinear-then-clip transfer onto the conforming FEM part
  then moved total in-part dopant by **7.15 %** (mean saturation 0.4074 ->
  0.3814), against the 2 % drop condition. Arm DROPPED, and **no substitute
  transfer was tried** -- the pre-registration forbids replacing it with an
  approximation presented as the published map.

The cause is a reusable finding, not bad luck: the voxel map's support is the
heatr3d STAIRCASE part and is zero outside it, so interpolation at conforming
cells near the rim pulls in those zeros and thins the map at the boundary.
That the *solved* map's own coarse-to-mid transfer moved only **0.065 %**
confirms the mechanism is the staircase support and not the transfer operator.

Consequence for the headline: the comparison against the inversion heuristic is
made against its **published re-run number** (+0.3 % sigma_T on this shape,
FGM_BENEFIT_RERUN.md arm D), not against a re-scored map in this campaign. That
is a weaker comparison than the plan intended and it is stated as such.

## 5. Deviations, each named

1. **Scaled first step** (section 3). A recorded deviation, run as an
   additional arm, pure reparameterization.
2. **scipy added to the spike env** (`phase_c_env_deviation.json`). Needed for
   the neighbour search and L-BFGS-B. `pip install scipy==1.14.1` silently
   downgraded numpy 2.4.6 -> 2.2.6, the line this project flags for a
   buffer-elision bug; caught, reverted, scipy 1.17.1 installed instead. Phase
   B's B2 gate was re-run: single-cell probes bit-identical, the two direction
   probes moved 5.8e-10 and 3.4e-06 relative on quantities already at 1e-7 and
   1e-10, every verdict unchanged.
3. **Filter is an explicit matrix, not a Helmholtz PDE filter.** Chosen because
   the Helmholtz operator is self-adjoint, making its transpose gate trivially
   true; the explicit matrix is measurably asymmetric (0.1322) so the gate
   tests something. It is also the literal 2-D convention.
4. **Volume-weighted kernel.** The 2-D grid is uniform and needs no volume
   factor; a tetrahedral mesh does, or the "physical radius" claim would be
   false on a graded mesh.
5. **Barycentric tet quadrature for chi**, since the shared contract's
   rectangular n_sub grid has no tetrahedral analogue. Measured inert: chi
   reproduces the conforming-mesh indicator to max difference 0.0 with 0
   partial cells of 16549.

## 6. What Phase C does NOT cover

* **Two pre-registered arms were NOT RUN** -- the projection/beta-continuation
  robustness arm and the 3x weight sensitivity arm -- because session compute
  ran out. The pre-registered `unrun_arms_policy` applies: recorded as
  NOT_RUN, never back-filled with an estimate. The consequence is real: with
  the sensitivity arm unrun, **it is not known whether the ranking depends on
  the 10x weight ratio**, and the pre-registration says that if it flips, that
  is the finding. Treat the weight choice as unvalidated.
* **One shape.** The extruded circle only; the library campaign is Phase E. A
  single shape cannot establish that the method generalizes, and the cylinder
  was chosen precisely because it is the HARDEST case for a heuristic, not a
  representative one.
* **The symmetric control arm never moved**, so the cross-lane comparable
  number comes from the scaled arm's symmetric score (-3.69 %), not from a
  solve that optimized it.
* **No permittivity channel, no drive reconciliation** (Phase D). The drive is
  the Phase A fixed-power renormalization convention throughout.
* **No densification in the forward**; rho is held fixed, so the adjoint
  matches the forward and there is no rho co-state.
* **Coupling OFF.** Chosen for like-for-like comparison with the heatr3d FGM
  re-run, which marched through `qrf_override`. The coupled path is Phase
  B-gated but unexercised here.
* **Physics trust is unchanged.** This certifies the DESIGN METHOD on the
  dolfinx forward. Phase A's converged ~1.6 s circle t90 cross-family offset
  remains an S3 question, and no Studio badge is earned by this report.
* **Budget.** 40 forward-equivalents per arm is the inherited 2-D figure, so
  every J here is an upper bound. The solve was still descending at the budget
  limit (eval 11 -> 12 improved), so the 10.67 % is not a converged optimum.

## 7. Reproduction

    # pre-registration, objective, chi, design chain (fast)
    ./.venv312/bin/python -m solve3d.phase_c_prereg
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_phase_c_prereg.py \
      solve3d/tests/test_objective.py solve3d/tests/test_fill_contract.py \
      solve3d/tests/test_design_chain.py

    # the campaign (hours)
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_c_run --chain-gate
    #   then run_baselines(), run_stationarity_diagnostic(),
    #   run_solve_arm(..., scale_first_step=True), run_acceptance(...)

    ./.venv312/bin/python -m solve3d.make_phase_c_tables
