# D1 decision report: adopt dolfinx (FEniCSx) as the high-fidelity EQS cross-check and P2 mechanics engine

Decision point D1 of `docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md`.
Plan: `docs/superpowers/plans/2026-07-31-d1-dolfinx-spike.md`. Date: 2026-07-31.

**Every number in this report is quoted from `heatr3d_d1_spike/results.json`,
with its JSON key path given.** Two quantities are explicitly NOT from that file
and are labelled where they appear: the plan's `~29.8 GB` heatr3d direct-LU
estimate at n = 200, and the effort log (derived from git commit timestamps).
`heatr3d.py` was not modified by any part of this spike
(`results.json["eqs02_impact"]["heatr3d_edited"] = false`).

---

## 1. Environment reality

The complex-scalar build resolved on the **first** `micromamba create` attempt,
so the documented real/imaginary split fallback was never needed:

| Key | Value |
|---|---|
| `task0.dolfinx_version` | `0.11.0` |
| `task0.petsc_scalar_type` | `<class 'numpy.complex128'>` |
| `task0.complex_build` | `true` |
| `task1.scalar_path` | `complex` |
| `task0.poisson_umax` vs `task0.poisson_ref` | 0.056064658380808416 vs 0.0562 (`task0.poisson_rel_err` = 0.0024082138646189335) |
| `task0.gate_ok` | `true` |

Every downstream EQS task ran the single complex bilinear form, not a mixed
two-field system. That is the cheaper of the two branches the plan priced.

### The deployability cost: a JIT shim for space-containing paths

`task0.jit_space_fix.needed = true`. FFCx compiles every UFL form through a C
JIT that inherits conda's `sysconfig` flags; those flags contain the environment
prefix, and `distutils` splits them on whitespace. This repo lives under
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/...`, so clang received
`Dropbox/Matthew` as a filename and **nothing assembled at all**. The fix
(`jit_fix.py`) is a 0-byte symlink at `task0.jit_space_fix.link` =
`/Users/mattmccoy/.cache/heatr3d_d1_spike/env` plus a rewrite of the 11 config
variables listed in `task0.jit_space_fix.patched_vars`
(`BLDSHARED`, `CFLAGS`, `CONFIGURE_CFLAGS`, `CONFIGURE_CPPFLAGS`,
`CONFIGURE_LDFLAGS`, `CPPFLAGS`, `LDFLAGS`, `LDSHARED`, `PY_CFLAGS`,
`PY_CPPFLAGS`, `PY_LDFLAGS`).

Score this honestly: the shim is three lines of user code and installs nothing
outside the spike directory, but it is a **hard blocker that a successor will
hit** on any machine whose project path has a space — which includes every
Dropbox-synced lab machine in this group. Adoption must ship `jit_fix.py` (or an
equivalent documented step) as part of the environment, not as folklore.
heatr3d has no comparable install-time failure mode: it is numpy/scipy.

### Formulation gate (Task 1)

Uniform virgin bed, parallel plates, 24 cells/axis, `task1.n_dofs` = 15625:

| Key | Value | Tolerance |
|---|---|---|
| `task1.v_err_max` | 2.580691216280684e-11 | `task1.v_err_tol` 0.00086 |
| `task1.transverse_std` | 5.027568745526734e-12 | `task1.transverse_std_tol` 8.6e-07 |
| `task1.vi_max_abs` | 1.1102230246251565e-16 | — |
| `task1.gate_ok` | `true` | |

The Q_rf convention was separately pinned analytically against
`heatr3d.compute_qrf_3d`: `task1_qrf_convention_check.gate_ok = true`, worst
relative deviation across the four checks 5.4127968932884905e-14 (`.checks.q_uniform_min.rel`),
against `tol` 1e-09. So the FEM engine reproduces heatr3d's Q definition **and**
its fixed-power renormalization basis exactly; every later comparison is
like-for-like.

---

## 2. Fidelity (Task 2) — and the EQS-02 discovery

Extruded circle d = 20 mm, full height, compared on heatr3d's own in-part
mid-plane voxel centres (`task2.levels.*.comparison.n_points_compared` = 360
coarse, 812 fine; `eval_missed_q` = 0 in both, so nothing was silently dropped).
Resolution matching is by in-part unknown count:

| | coarse (matched to n=64) | fine (matched to n=96) |
|---|---|---|
| `levels.*.n_nodes_in_part` | 19821 | 77043 |
| `levels.*.target_nodes_in_part` | 23040 | 77952 |
| `levels.*.node_count_ratio_vs_voxels` | 0.8602864583333333 | 0.9883389778325123 |
| `levels.*.n_dofs_total` | 24784 | 94054 |
| `levels.*.part_volume_rel_err_vs_analytic` | 0.0 | 0.0 |

### The gate as written FAILS, and that is the finding

`task2.gate`:

| Variant | coarse | fine | `gate_ok` |
|---|---|---|---|
| `as_written_all_points` | 0.9031803180394822 | 0.9037810456580881 | `false` |
| `interior_only` | 0.04775482891950163 | 0.02553801373366744 | `true` |
| `surface_band_only` | 0.8090219426770544 | 0.8011004738155877 | `false` |
| `maskgrad_all_points` | 0.17652460972777445 | 0.10790598069422491 | `false` |
| `maskgrad_interior` | 0.047754828919521514 | 0.025538013733723816 | `true` |

All five variants satisfy `improves_or_holds = true`. The 90 % disagreement is
**entirely in the surface band**, and it is not a mesh-resolution effect: it is
`compute_qrf_3d` differencing `V` across the part boundary before masking
(heatr3d.py:393-399), where `grad V` jumps by
`sigma_doped/sigma_virgin` = 4e6 and `Q ~ |E|^2` squares the jump. Re-post-
processing **heatr3d's own V** with a part-confined stencil
(`comparison.diagnostic_maskgrad`) collapses the disagreement from 0.9038 to
0.10790598069422491 overall and to 0.025538013733723816 in the interior.

The FEM answer is the one the analytic limit endorses. An infinite cylinder in a
uniform transverse field has a **uniform** interior field
(`comparison.analytic_anchor`), so the coefficient of variation should tend to
zero. `comparison.uniformity_cv` at the fine level:

| | value |
|---|---|
| `fem` | 0.011873150219774569 |
| `heatr3d` (shipped Q) | 2.120846029633705 |
| `heatr3d_maskgrad` | 0.10929602243246767 |

and the FEM CV *halves* under refinement (0.024789058677560608 → 0.011873150219774569)
while the shipped-Q CV does not (2.11557105235421 → 2.120846029633705).
Same story in the energy split: `comparison.power_fraction_in_surface_band` at
the fine level is `fem` 0.202407483723139 against an `area_fraction` of
0.2019704433497537 — volume-proportional, as an almost-uniform interior field
requires — while `heatr3d` puts 0.6178420619955252 there.

The absolute scale agrees to 7e-4 (`qrf_abs_mean_ratio_fem_over_heatr3d` =
1.0006694869867458), which it must, since both engines renormalize to the same
fixed total power. The residual `|E|` offset is also explained: the raw
`emag_gauge_residual` is -0.18484422797591116, but with the cross-interface
stencil removed it drops to -0.010997591457920453
(`diagnostic_maskgrad.emag_maskgrad_gauge_residual`), i.e. to the
cell-centred-electrode gauge difference the plan predicted and nothing else.

**Verdict on fidelity:** in the part interior the two engines agree to 2.6 % and
converge toward each other. At the part surface they disagree by ~80–90 %, and
three independent lines of evidence (the analytic uniform-interior limit, the
surface-band energy fraction, and heatr3d's own V re-post-processed) all say the
FEM is right and the voxel post-processing is wrong. That is a fidelity result in
dolfinx's favour, obtained *because* an independent engine existed.

### Cost, at matched in-part resolution

| Key | coarse | fine |
|---|---|---|
| `levels.*.wall_solve_s` (dolfinx GMRES+gamg) | 0.45022175007034093 | 1.4775482079712674 |
| `levels.*.heatr3d_wall_eqs_s` | 102.98091095802374 | 636.7955465000123 |
| `levels.*.eqs_solve_speedup_vs_heatr3d` | **228.73375384892978** | **430.9812316542673** |
| `levels.*.peak_rss_gb` | 0.772186279296875 | 1.48321533203125 |
| `levels.*.heatr3d_peak_rss_gb` | 0.763946533203125 | 1.8428955078125 |

Meshing is the real dolfinx cost, not solving: `wall_mesh_s` 2.7209705419372767
and 24.913597416016273 against solve times under 1.5 s. A complex direct LU was
run at the coarse level as an independent cross-check of the iterative solve:
`direct_lu_cross_check.max_dV_over_860` = 2.240432948827558e-08 at
`wall_lu_s` = 18.93924820900429, i.e. `speedup_lu_over_iterative` =
42.06648880478407 in favour of GMRES+gamg.

---

## 3. Corner control (Task 3) — the voxel-killer test

Extruded square 20 mm, full height. Metric: `corner_max_over_mean` = max in-part
Q_rf within `task3.corner_band_m` = 0.001 m of a vertical corner edge, and
`bulk_p99_over_mean` away from it.

### dolfinx (`task3.dolfinx`)

| label | `h_corner_m` | `n_dofs_total` | `corner_max_over_mean` | `bulk_p99_over_mean` | `wall_solve_s` | `peak_rss_gb` |
|---|---|---|---|---|---|---|
| `uniform_n64` | 0.0004960681304483504 | 31183 | 21.655906531098807 | 2.134983771628248 | 0.5272805830463767 | 0.394866943359375 |
| `uniform_n96` | 0.0003481674469130965 | 95575 | 23.198907412310916 | 1.9277071010665976 | 1.5617420000489801 | 0.9089508056640625 |
| `uniform_n128` | 0.0002593311377965431 | 215192 | 27.775774328542642 | 1.9316929436477586 | 4.177374207996763 | 1.5114593505859375 |
| `corner_refined` | 7.009051124403837e-05 | 1573289 | 40.710357946911536 | 1.8350349802778887 | 37.58296887506731 | 5.0117645263671875 |

### heatr3d (`task3.heatr3d`)

| n | `h_m` | `raw.corner_max_over_mean` | `raw.bulk_p99_over_mean` | `maskgrad.corner_max_over_mean` | `maskgrad.bulk_p99_over_mean` | `wall_eqs_s` |
|---|---|---|---|---|---|---|
| 64 | 0.0009375 | 19.154676154007085 | 5.9320398276965145 | 2.198335698709164 | 1.6941611965562469 | 206.15408391691744 |
| 96 | 0.000625 | 29.11753369314547 | 6.470449710457635 | 2.709556057075584 | 1.7425452953667566 | 676.6130994580453 |
| 128 | 0.00046875 | 38.70663194436349 | 6.988607835897863 | 3.1690280371273563 | 1.8212343270137583 | 1684.4803166660713 |

### Growth-law fits (`task3.fits`)

| series | exponent | R² |
|---|---|---|
| `dolfinx_uniform` | -0.3774267830858815 | 0.9095990852659674 |
| `dolfinx_all_incl_corner_refined` | -0.3289000895385861 | **0.9852656158082325** |
| `heatr3d_raw` | -1.016064787247657 | 0.9998597375583892 |
| `heatr3d_maskgrad` | -0.5268434876897898 | 0.9997692555156256 |

`task3.gate.gate_ok = true` on the plan's criterion (`dolfinx_r2` =
0.9852656158082325 > 0.98). Read the fits carefully, because the plan's stated
expectation was only half right:

* The plan expected the voxel engine to jump erratically. **It did not.**
  `heatr3d_raw` fits a power law better than dolfinx does (R² 0.99986). What
  disqualifies it is the *exponent*: -1.016064787247657 means
  `corner_max ~ 1/h`, exactly the rate at which a finite-difference gradient
  across a discontinuity diverges. That is a discretization artifact with a very
  clean scaling law, not a physical edge singularity. dolfinx grows at
  -0.3289000895385861 — a genuine, slowly divergent integrable edge singularity.
* Between n = 64 and n = 128 the shipped heatr3d corner ratio doubles
  (19.15 → 38.71) while dolfinx moves 21.66 → 27.78 over a *finer* corner
  element size range.
* Away from the corner, `task3.gate` records the bulk-metric stability spread
  across refinements: `heatr3d_raw_bulk_p99_over_mean_spread` =
  1.0565680082013484, `dolfinx_bulk_p99_over_mean_spread` =
  0.29994879135035935, `heatr3d_maskgrad_bulk_p99_over_mean_spread` =
  0.12707313045751145. dolfinx is 3.5x more stable than shipped heatr3d in the
  bulk — but **less** stable than heatr3d's own solve read with a corrected
  stencil. State that plainly: the bulk-stability win belongs mostly to fixing
  the post-processing, not to the mesh.

The mesh win is specifically at the corner, and it comes with a price:
`corner_refined` cost `wall_mesh_s` = 336.0033413749188 (9x its solve time) and
`peak_rss_gb` = 5.0117645263671875 for `n_dofs_total` = 1573289.

**This spike did NOT establish the true corner value.** Both engines diverge
under refinement, as a real edge singularity must. What is established is which
divergence is physical (-0.33) and which is numerical (-1.02), and that dolfinx
lets the corner be resolved locally at controlled cost. Settling the absolute
number needs the COMSOL 3-D anchor, i.e. Gate S3.

---

## 4. Scale (Task 4) — the solve heatr3d cannot do

Extruded circle at `task4.lc_part_m` = 0.0003 m in-part element size, i.e. the
n = 200 voxel pitch (`task4.target.voxel_h_m` = 0.0003):

| Key | Value |
|---|---|
| `task4.n_dofs_total` | 623900 |
| `task4.n_cells_total` | 3812849 |
| `task4.n_nodes_in_part` | 522837 |
| `task4.target.heatr3d_uniform_unknowns_at_n200` | 8000000 |
| `task4.dof_reduction_vs_n200_voxels` | **12.822567719185766** |
| `task4.wall_solve_s` | 29.59482370794285 |
| `task4.wall_mesh_s` | 198.41988833399955 |
| `task4.gate.peak_rss_gb` | **2.4009552001953125** (gate `rss_lt_gb` 34.0) |
| `task4.gate.qrf_pattern_rel_l2_all` | 0.011870039611365844 (gate `pattern_rel_l2_lt` 0.05) |
| `task4.gate.gate_ok` | `true` |

The 12.8x unknown reduction is the structural point: the powder bed can be
graded coarse (`task4.lc_bed_m` = 0.0012) while the part keeps 0.3 mm elements.
A voxel grid must pay for resolution everywhere.

Comparison against heatr3d's documented ceilings:

* Measured in this spike, heatr3d EQS-only at n = 128 needs
  `task3_ref_n128.runs.n128.peak_rss_gb` = 3.7201385498046875 GB and
  `wall_eqs_s` = 1684.4803166660713 s for
  `n_unknowns` = 2097152. dolfinx at *higher in-part resolution* used **less**
  memory (2.40 GB) and 57x less solve time (29.59 s).
* At n = 200 heatr3d has 8e6 unknowns and, per the plan's documented estimate
  (**not measured here, and not in results.json**), a direct LU would need
  ~29.8 GB — which is why the S1 gate report records that n = 200 cannot
  currently be solved at all. dolfinx reached the equivalent in-part resolution
  at 2.4 GB, a ~12x margin under the 34 GB machine limit.

Sanity that the big solve is real and not silently broken:
`task4.comparison.uniformity_cv.scale` = 0.007923204652072354 (still heading to
the analytic uniform-interior limit, below the Task-2 fine value of
0.011873150219774569), and `power_fraction_in_surface_band.scale` =
0.20199028319302817 against `area_fraction` 0.2019704433497537.

---

## 5. Adjoint-readiness (Task 5) — the north-star criterion

The north star (spec §1, restated 2026-07-30) is inverse design: solve for the
dopant field that produces the intended geometry. That makes gradient
availability a first-class D1 criterion, not a nice-to-have.

**Tooling reality:** `task5.adjoint_tooling` records that dolfinx-adjoint /
pyadjoint has no release wired to dolfinx 0.11, so the adjoint was
**hand-assembled** (`adjoint_core.py`). For one linear solve this is textbook
work, and it took a small fraction of the spike (§7).

Setup: Task-2 coarse mesh (`task5.mesh.n_dofs_total` = 24784,
`n_cells_total` = 143892), sigma a DG0 Function on the part with
`task5.n_design_dofs` = **105191** design variables, objective
`J = int_part (Q_rf - mean_part Q_rf)^2 dV` with Q_rf built exactly as
`eqs_common.qrf_dg0` builds it, **including** the fixed-power renormalization,
differentiated through rather than frozen.

Assembly consistency first — forward and adjoint must share one operator
(`task5.consistency_vs_eqs_common`): `max_dV_over_860` = 4.255324774719558e-13,
`max_dQ_over_Qmax` = 3.15408756228332e-12, `scale_rel_diff` =
3.965716643961059e-13, `ok` = `true`.

### The FD gate (`task5.per_dof`, 5 seeded-random sigma dofs, central differences)

| `dof_local` | `sigma` [S/m] | adjoint `dJ/dsigma` | best FD rel err | best FD abs err | `grad_over_grad_max` | `pass_1pct` |
|---|---|---|---|---|---|---|
| 67680 | 0.04278480707464558 | 31.676030404903827 | 7.407950091118488e-05 | 0.002346370705271994 | 0.0018240135289621779 | `true` |
| 2456 | 0.04053004804244084 | 822.2239543063645 | 8.932203947710685e-07 | 0.0007344278610617039 | 0.04734645084377143 | `true` |
| 2867 | 0.02735343491220618 | -3747.0551676934256 | 1.9661830378840393e-07 | 0.0007367397761299799 | 0.2157681765131265 | `true` |
| 8408 | 0.05727418414886424 | 554.6882211499258 | 1.7594996543006809e-06 | 0.000975972016135529 | 0.03194083370928943 | `true` |
| 93450 | 0.02092181195079609 | -7152.595429940419 | 4.883335493254683e-08 | 0.0003492852483759634 | 0.41187076362272423 | `true` |

`task5.gate.worst_best_rel_err` = **7.407950091118488e-05**, against
`tol_rel` = 0.01. `task5.gate.all_dofs_pass` = `true`.

Note the `best_abs_err` column: the FD error bottoms out at a nearly constant
**absolute** value (0.0003492852483759634 to 0.002346370705271994) regardless of the gradient's magnitude,
which is the signature of an FD noise floor (LU backward error at a 4e6
conductivity contrast), not of a gradient error. The dof with the worst relative
error is simply the one with the smallest gradient (`grad_over_grad_max` =
0.0018240135289621779). Recorded as `task5.gate.fd_noise_floor_note`.

Because that floor makes the per-dof numbers a weak upper bound, a
floor-free check was added — the directional derivative of the full gradient
along one seeded random direction over all 105191 dofs
(`task5.directional`): adjoint 2396.8878904068424 vs central FD
2396.888024716473, `best_rel_err` = **5.603500420298189e-08**.

### The gate is not vacuous (`task5.mutations`)

Two deliberately wrong gradients were run through the same gate:

| mutant | directional rel err | rejected by the 1 % gate |
|---|---|---|
| `renorm_frozen` (fixed-power scale treated as constant) | 0.017159885771654493 | `true` |
| `adjoint_dropped` (explicit partial only, no adjoint solve) | 0.46174542475102587 | `true` |

`renorm_frozen` errs by up to 1.4983795363711947 (150 %) on an individual dof.
So the renormalization derivative — the term the plan warned about — is a real,
measurable part of the gradient and it is being computed.

The renormalization also produces an exact identity worth recording: it pins
`task5.forward.qbar` = 1591549.4309189538 to
`power_density_w_per_m3` = 1591549.4309189534, i.e.
`qbar_over_power_density_minus_1` = 2.220446049250313e-16, and the measured FD
drift of that mean is `max_qbar_fd_drift_rel` = 2.9258361585343184e-16. The part
mean of Q_rf is a constant of the problem, independent of the dopant field.
`task5.forward.clip_active` = `false`, so no subgradient question arises here.

### The number that matters for inverse design

`task5.timing`: `wall_forward_s` = 21.110703374957666,
`wall_gradient_s` = 0.04366762505378574, so
`gradient_over_forward` = **0.0020685064006718966**. A gradient with respect to
105191 design variables costs **0.2 % of one forward solve**, because A is
complex symmetric, so `A^H = conj(A)` and the adjoint reuses the forward LU
factorization. Finite differences over the same design space would cost
2 x 105191 forward solves; at 21.1 s each that is about 3.5 machine-months.
`task5.gate.gate_ok` = `true`.

---

## 6. EQS-02 impact summary — the thermal topology inverts

This was not a planned task. It exists because Task 2 exposed the artifact, and
its consequences had to be measured before any published heatr3d number could be
relied on. `heatr3d.py` was not edited; the corrected drive was injected through
the existing `run(qrf_override=...)` hook. Both drives carry **identical** total
absorbed power (`eqs02_impact.shapes.cylinder.power_identity.rel_diff` = 0.0;
square 3.498331035896751e-16), so every difference below is redistribution, not
energy creation.

Heating field, n = 64 (`eqs02_impact.shapes.*.qrf`):

| | circle shipped | circle corrected | square shipped | square corrected |
|---|---|---|---|---|
| `max_over_mean` | 12.25493303782372 | 1.857307742257783 | 19.154676717105826 | 2.198335863744568 |
| `cv` | 2.115571006819856 | 0.18032886484116506 | 2.108182070527277 | 0.2901614160876311 |
| `power_fraction_in_surface_band` | 0.7365865624496305 | 0.3185966843456619 | 0.736607946311936 | 0.3714467981738855 |

The in-part **interior pattern is unchanged**
(`pattern_rel_l2_corrected_vs_shipped.interior` = 1.7079263682273255e-16 circle,
2.035933932153387e-16 square); only the surface band and the global scale move.

Thermal answers, `phase_update = "enthalpy"`, `phi_target` = 0.9
(`eqs02_impact.shapes.*.thermal` and `.delta_corrected_minus_shipped`):

| | circle | square |
|---|---|---|
| `sigma_T_c` shipped → corrected | 26.130549178929297 → 20.103142032804296 | 17.89741487156853 → 19.85577084524189 |
| `delta.sigma_T_rel` | **-0.23066515383401423** | **+0.10942116432604826** |
| `t_phi90_s` shipped → corrected | 397.0 → 323.35 | 327.70000000000005 → 316.05 |
| `delta.t_phi90_rel` | -0.18551637279596972 | -0.035550808666463325 |
| `surface_minus_interior_mean_c` shipped | +8.547095976515465 | +18.015160437969882 |
| `surface_minus_interior_mean_c` corrected | **-35.252093432559434** | **-30.523435032896373** |

Two facts drive the D1 decision:

1. **The sign of the sigma_T change is shape-dependent** (-23 % circle,
   +11 % square). The artifact is not a bias that can be scaled out; it
   interacts with geometry, so any ranking of shapes, orientations, or FGM
   designs by sigma_T is affected in a way that cannot be assumed to survive.
   The *magnitude* is also grid-dependent (the circle's -23 % becomes -3.7 % at
   n = 96); see the n-dependence subsection below.
2. **The thermal topology inverts.** Under the shipped drive the part surface is
   hotter than the interior; under the corrected drive the interior is hotter.
   Where the hot region *is* flips.

Full breakdown and exposure ranking: `EQS02_IMPACT.md`.

### n-dependence (`eqs02_impact.shapes.cylinder_n96`)

The circle case was repeated at n = 96 so the n = 64 deltas are not left as a
lower bound of unknown tightness. `power_identity.rel_diff` =
3.5187954107988544e-16, `z_invariance.ez_max_over_emag_mean` =
1.8059906972438017e-06, `energy_residual_frac` = -8.627945557411796e-14 (shipped) and
-8.805692389645806e-14 (corrected), `clamp_bound` = `false` in both, so the n = 96 pair is as
clean as the n = 64 pair. Caveat: the surface band is "within 1.5 voxels", so
its volume fraction is *not* the same at the two refinements; band-relative
quantities are comparable in direction, not term-by-term.

| circle, `delta_corrected_minus_shipped` | n = 64 | n = 96 |
|---|---|---|
| `sigma_T_rel` | -0.23066515383401423 | **-0.03677621588928825** |
| `t_phi90_rel` | -0.18551637279596972 | -0.12746688294133557 |
| `T_max_c` | -44.243879152394356 | -17.38527868992145 |
| `surface_minus_interior_c` | -43.7991894090749 | -37.07158733230409 |

Two conclusions, and they point in opposite directions:

1. **The n = 64 sigma_T delta was NOT a lower bound — it was an over-estimate.**
   The circle's sigma_T impact shrinks from -23.1 % to -3.7 % under refinement.
   Anyone quoting "-23 %" as the size of the EQS-02 sigma_T error is quoting a
   grid-specific number. (The square was not repeated at n = 96; its +10.9 %
   remains single-grid.)
2. **The topology inversion is robust and, if anything, sharper.**
   `surface_minus_interior_mean_c` goes +1.1448128748674833 (shipped) to
   -35.92677445743661 (corrected) at n = 96, against +8.55 to -35.25 at n = 64.
   The corrected interior is ~36 C hotter than the surface at both refinements.

The most useful number here is not a delta at all. It is which drive gives a
**grid-stable** sigma_T:

| circle `thermal.*.sigma_T_c` | n = 64 | n = 96 | change |
|---|---|---|---|
| shipped | 26.130549178929297 | 20.713722586926192 | **-20.7 %** |
| corrected | 20.103142032804296 | 19.951950253198568 | **-0.75 %** |

Spec §1 lists "the reported sigma_T = std(T) is not grid-converged" as one of
the three reasons heatr3d is untrusted, and Gate S2 exists to replace the
metric because of it. These two rows say a large part of that non-convergence
was the cross-interface Q artifact, not the metric: with the artifact removed,
sigma_T moves 0.75 % between n = 64 and n = 96 instead of 20.7 %. That does not
retire the S2 metric question — the corner singularity of §3 is still real and
two grid levels are not a convergence study — but it strongly suggests S2 should
be run on corrected Q before any metric is declared unfixable.

---

## 7. Effort log

**Not from results.json.** Derived from git commit timestamps on
2026-07-31 plus the timings recorded per task. These are *agent session*
wall-clock intervals, a large share of which is unattended machine time; a human
implementing the same work would take substantially longer.

| Task | Interval | Wall | Notes |
|---|---|---|---|
| Plan | `bae3b03` 12:46 | — | plan committed |
| 0 environment | 12:46 → `885c86d` 12:52 | 6 min | includes finding and fixing the FFCx JIT space bug |
| 1 EQS formulation + plate gate | 12:52 → `79acf7d` 12:53 | 1 min | |
| 2 + 3 fidelity and corner study | 12:53 → `bd4f0e0`/`a7fec00` 14:41 | 108 min | dominated by heatr3d reference solves |
| EQS-02 impact (unplanned) | 14:41 → `9ff48fe` 15:13 | 32 min | |
| 4 scale test | 15:13 → `667ce57` 15:14 | ~1 min commit; 228 s of measured mesh+solve | |
| 5 adjoint demo | 15:14 → `a7e0df4` 15:47 | 33 min | 21 min of it is the 60-solve FD sweep |
| EQS-02 at n = 96 (optional add) | 15:47 → | 26 min | 396 s EQS + 310 s + 281 s marches |
| 6 this report | concurrent with the n = 96 run | | |

Total D1 spike: **~3 h wall-clock**, against the plan's one-focused-week time
box. Of that, measured machine time is ~1.81 h: heatr3d reference EQS solves
3307 s, EQS-02 n=64 solves and marches 1241 s, all dolfinx meshing+solving
704 s, and the Task-5 FD sweep 1270 s (60 forward solves x 21.11 s). The time box
was never approached.

The adjoint specifically — the item the plan flagged as possibly needing "deeper
than ~a day of effort" — took **one working session**, and the physics-specific
part (the fixed-power renormalization derivative) was the only non-mechanical
piece. It is not a research problem; it is bookkeeping over one linear solve.

---

## 8. Recommendation: **(a)** — adopt dolfinx as the high-fidelity cross-check engine AND the P2 mechanics engine

heatr3d remains the fast in-tool planner in the Studio. dolfinx becomes the
engine that says what is true.

Reasoning, tied to the numbers:

1. **It found a real defect in the shipped tool within two hours.** The Task-2
   surface-band disagreement (`task2.gate.as_written_all_points.fine` =
   0.9037810456580881) is not FEM-vs-FV noise; it is a cross-interface stencil
   in `compute_qrf_3d`, and its thermal consequence is a topology inversion
   (§6). An independent engine is exactly the instrument that catches this
   class of bug, and heatr3d has been trusted-but-unanchored precisely because
   no such instrument existed.
2. **It is faster than heatr3d at the same in-part resolution, by 229x and
   431x** (`task2.levels.*.eqs_solve_speedup_vs_heatr3d`), at comparable or
   lower memory. The "voxel cost is acceptable" premise of option (c) is
   false for EQS.
3. **It solves the case heatr3d cannot.** n = 200-equivalent in-part resolution
   at `task4.gate.peak_rss_gb` = 2.4009552001953125 GB and 29.59 s, versus
   heatr3d's 8e6 unknowns and the plan's ~29.8 GB direct-LU estimate. This
   directly unblocks Gate S2, which the S1 report recorded as blocked.
4. **The corner divergence becomes diagnosable.** dolfinx grows at
   -0.3289000895385861 with R² 0.9852656158082325 under controllable local
   refinement; the shipped voxel metric grows at -1.016064787247657, the rate of
   a numerical artifact. S2's replacement uniformity metric can now be chosen
   against a reference that converges for physical reasons.
5. **The north star is reachable.** A machine-precision-gated gradient over
   105191 design variables at 0.2 % of a forward solve
   (`task5.timing.gradient_over_forward` = 0.0020685064006718966,
   `task5.gate.gate_ok` = `true`) is the difference between inverse dopant
   design being a research project and being a loop.

Why not (b) mechanics-only: options (b) and (c) both assume heatr3d's EQS is
good enough to keep as the sole field engine. §2, §3 and §6 show it is not, at
the part surface — which is where melt onset, skin heating, and every
surface-vs-interior claim live.

Why the recommendation is still *cross-check*, not *replacement*: nothing here
argues for retiring heatr3d. It stays the in-tool planner (its interior Q
pattern is identical to the FEM's once the post-processing is corrected —
`eqs02_impact.shapes.cylinder.qrf.pattern_rel_l2_corrected_vs_shipped.interior`
= 1.7079263682273255e-16 and `task2.gate.maskgrad_interior.fine` = 0.025538013733723816), and
a rewrite would restart the trust ladder, which the spec explicitly rejects.

---

## 9. Implications

### S1 completion (large-N EQS)

The S1b deferral said "the scalable large-N EQS investment waits for D1." D1's
answer: **do not make that investment in heatr3d.** Task 4 shows the large-N EQS
requirement is already met by dolfinx at 12.8x fewer unknowns and 2.4 GB. What
heatr3d still needs from S1 is (i) the EQS-02 post-processing fix — a
part-confined gradient in `compute_qrf_3d`, which `metrics.masked_grad_3d`
already implements and `test_metrics.py` already covers — and (ii) a regression
test pinning the corrected surface-band energy fraction. Neither is a scalability
project. EQS-02 should be raised as an S1 defect in its own right, since its
thermal deltas (§6) reach 23 % in sigma_T.

### S2 design (convergence)

* Run the refinement studies in **both** engines. dolfinx supplies the
  converging reference the metric selection needs; heatr3d supplies the
  in-tool number that must track it.
* The corner metric decision now has data: raw voxel `corner_max` is unusable
  (exponent -1.02); the candidates should be scored against the dolfinx
  sequence in `task3.dolfinx`, and `bulk_p99_over_mean` is the most stable
  quantity measured on either engine (`task3.gate` spreads: 0.127 maskgrad,
  0.300 dolfinx, 1.057 raw).
* S2 must be re-baselined on the corrected Q. Any pre-EQS-02 sigma_T is a
  different quantity — and per §6, a much less grid-stable one (shipped sigma_T
  moves -20.7 % from n = 64 to n = 96; corrected moves -0.75 %). Fix EQS-02
  *before* choosing the replacement metric, or S2 will be tuning a metric to
  absorb an artifact.

### P2 mechanics

Unchanged in direction, but note honestly: **mechanics was not spiked.** The
recommendation to use dolfinx for P2 rests on (i) the environment, complex/real
solver, and meshing path now being proven here, (ii) conforming meshes being a
precondition for any stress/distortion calculation on a curved boundary, and
(iii) the adjoint machinery of §5 transferring directly to shape sensitivities.
The first mechanics task should be its own small gate (a manufactured
elastic solution plus a thermal-expansion patch test) before any P2 claim.
CalculiX remains the fallback per the spec.

### Studio

No badge changes are earned by this spike. It is engine selection, not
validation. The one Studio-visible consequence is that heatr3d-derived
hot-spot ratios and surface-vs-interior statements should be treated as
provisional until EQS-02 is fixed.

---

## 10. What this spike did NOT settle

* **Absolute corner truth.** Both engines diverge under refinement (they should).
  Which value to report needs the COMSOL 3-D anchor — Gate S3.
* **The thermal-phase march in FEM.** The plan asked for an enthalpy-based
  thermal-phase march in dolfinx; **only the EQS half was spiked.** All Task 2-4
  numbers are EQS fields. The thermal comparisons in §6 are heatr3d-vs-heatr3d
  (shipped vs corrected drive), not FEM-vs-heatr3d. A dolfinx thermal march with
  latent heat is unspiked work and its effort is unmeasured.
* **Mechanics.** Not spiked at all (§9).
* **COMSOL comparison.** Deliberately out of scope here (the repo exports are
  2-D mid-plane extractions); it belongs to S3.
* **Meshing at scale as a workflow.** `task3.dolfinx.corner_refined.wall_mesh_s`
  = 336.0033413749188 and `task4.wall_mesh_s` = 198.41988833399955 mean meshing,
  not solving, is the dolfinx bottleneck. STL-driven meshing of real Studio
  geometry (as opposed to the analytic cylinder/box used here) is unmeasured.
* **Parallelism.** Everything ran serial on one machine. MPI scaling is
  unmeasured.
* **The EQS-02 n-dependence beyond two grids and one shape.** §6 adds n = 96 for
  the circle only. The square's +10.9 % sigma_T delta is still single-grid, and
  two levels do not establish a convergence rate for either.

---

## 11. Sign-off

Recommendation: **(a) adopt dolfinx as the high-fidelity cross-check engine and
the P2 mechanics engine; heatr3d remains the fast in-tool planner.**

Immediate consequences if approved:
1. EQS-02 is filed as an S1 defect and fixed in `heatr3d.compute_qrf_3d`.
2. The large-N EQS investment in heatr3d is cancelled; S2 unblocks via dolfinx.
3. S2 is designed as a two-engine convergence study and re-baselined on
   corrected Q.
4. A separate small mechanics gate is scheduled before any P2 commitment.
5. `jit_fix.py` (or equivalent) ships with the environment documentation.

Matt — approve / modify / reject: ______________________  date: ____________


---
SIGNED OFF: Matt McCoy, 2026-07-31 (via session): D1 recommendation (a) approved.
