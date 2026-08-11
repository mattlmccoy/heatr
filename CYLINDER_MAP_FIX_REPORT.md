# Phase C cylinder dopant map: symmetry fix

## Verdict

**STAGE A SHIPPED.** The published map is now the Phase C solved map projected
onto the problem's symmetry subspace and re-filtered once. No new solve was
run. Stage B (a symmetric-subspace re-solve at budget 40) was not needed and
was not started.

| quantity | raw solved map (old) | symmetrized map (new) |
|---|---|---|
| J on the solve mesh, vs uniform | **-10.67 %** | **-6.94 %** |
| J on the hold-out mesh, vs uniform | **-3.51 %** | **-7.64 %** |
| mesh hold-out gate | PASS | PASS |
| smoothing robustness gate (0.5 mm) | PASS, 0.22 % | PASS, **0.041 %** |
| SOLVED label | yes | yes |
| symmetric-variance fraction, full part | 0.4257 | **0.9956** |
| symmetric-variance fraction, mid-height slab | 0.4989 | **0.9966** |
| smooth field-frame modes, full part | R2 = 0.277 | **R2 = 0.733** |
| smooth field-frame modes, mid-height slab | R2 = 0.463 | **R2 = 0.952** |

Absolute objective values (asymmetric, 10x, at each arm's own envelope stop):

|  | uniform | raw solved | symmetrized |
|---|---|---|---|
| solve mesh `phase_a_coarse` | 1.1511359e-07 | 1.0283281e-07 | 1.0712930e-07 |
| hold-out mesh `phase_a_mid` | 8.2732480e-08 | 7.9827561e-08 | 7.6415689e-08 |

The headline trade is explicit: the symmetrized map gives up 3.7 points of
in-grid margin and gains 4.1 points of hold-out margin. The in-grid number was
partly bought by fitting the solve mesh's own discretization error, which is
exactly what a hold-out is for. The symmetrized map is the better map by the
only measure that is not contaminated by the mesh it was designed on.

## The symmetry group, and how it was fixed mid-task

The group is the intersection of part, chamber, electrode-field, objective and
**convection-boundary** symmetry. Read directly from the source, not assumed:

* **Axis is z.** `solve3d/forward.py in_part_predicate("circle")` is
  `sqrt(x^2 + y^2) <= 10 mm` with no z condition, and the mesher builds
  `occ.addCylinder(0, 0, -L/2, 0, 0, L, half)`
  (`heatr3d_d1_spike/mesh_gmsh.py`). The cylinder spans the full 60 mm chamber.
* **Electrodes at y = +/- L/2** (`forward.py` l.274-278). Deposited power goes
  as `|E|^2`, which is even under y -> -y even though V is odd. The electrodes
  fix the field direction along y, so **there is no rotation about the
  cylinder axis** and the map may legitimately keep smooth azimuthal structure.
* **Convection is one-sided.** `_top_facet_measure` restricts `ds` to the open
  top face **y = +L/2 only** (`forward.py` l.111, l.563-577). This breaks
  y -> -y in the thermal field and therefore in the objective.

**Group G = {identity, x-mirror, z-mirror, xz-mirror}, a Klein four-group. The
y-mirror is excluded.** A map's odd-in-y content on this case is physical, not
residue: it is the response to the one-sided convection. An earlier version of
this task projected with the y-mirror included; that run was **discarded and
re-run**, because deleting the odd-in-y content deletes real design content.
The discarded run is archived outside the repo and none of its numbers appear
here.

Independent agreement on the residue measurement: the 3-D lane's
`solve3d/symmetry_gate.py` reports 0.4258 (corrected group) and 0.2772 (full
three-mirror group) for the original map at
`solve3d/results/symmetry_retro_3d.json -> results.cylinder_phase_c`. This
task's independent implementation (cell-centroid KDTree matching on `s_map`,
volume-weighted) reports **0.42568** and **0.27765**. The two agree to about
2e-4, so the residue statement does not depend on either implementation.

Matt's original "only 25 percent" was measured on the mid-height slab with the
full three-mirror group, and reproduces exactly (0.2518). **The corrected
number for the same slab is 0.4989, and 0.4257 over the full part.** The defect
conclusion survives: roughly half of the solved map's spatial variance was
mesh-frame fitting from a budget-limited 12-evaluation stop. The magnitude of
the defect is smaller than first stated, and the corrected number is the one to
quote.

## What the new map actually looks like

Volume-weighted over the part: mean 0.9334, min 0.8478, max 0.9919, standard
deviation 0.0248. The design is therefore a roughly 7 percent uniform
reduction in saturation carrying a modulation of about plus or minus 2.5
percent, not a strongly graded field.

Decomposing the delivered map onto smooth modes that are invariant under G
(azimuthal factors cos 0t, cos 2t, cos 4t from the electrode field, plus sin 1t
and sin 3t which are odd in y and are the one-sided-convection response; times
r-polynomials up to cubic; times even powers of z):

| component | RMS amplitude | share of the map's std |
|---|---|---|
| radial and axial profile, m = 0 | 0.0163 | 66 % |
| azimuthal modulation, m = 2 and m = 4 | 0.0080 | 32 % |
| odd-in-y convection response | 0.0111 | 45 % |
| all invariant smooth modes together | R2 = 0.733 | |

Read plainly: **the map is dominated by a radial and axial profile; the
azimuthal modulation is real but about half the amplitude of the radial
profile, and the one-sided-convection tilt toward the cooled top face is of
comparable size to the azimuthal modulation.** The remaining 27 percent of
variance that the smooth basis does not capture is short-wavelength content at
the 1.0 mm filter scale, which the filter permits; on the mid-height slab,
where only 20 basis modes are needed, the smooth basis captures 95.2 percent.

The physics reads correctly at a glance in panel (b): saturation is highest
near +y, the face that loses heat to convection, and lowest in a core below
centre. That is the sign a heat-loss boundary should produce.

## What this implies for the 10.67 percent quote

**The defensible number changes. Say so in the dissertation.**

* `10.67 percent in-grid / 3.51 percent hold-out` was measured on a map that is
  only 42.6 percent consistent with the problem's own symmetry. It should no
  longer be quoted as the result.
* The replacement is **6.94 percent in-grid / 7.64 percent hold-out**, on a map
  that is 99.6 percent symmetry-consistent and passes the same two acceptance
  gates.
* The qualitative claim is unchanged and is now better supported: a direct 3-D
  design solve beats a uniform dopant on the cylinder, and the win transfers
  across meshes. The new pair is also easier to defend, because the in-grid and
  hold-out numbers now agree with each other instead of the hold-out being a
  third of the in-grid.
* The budget caveat still stands. The source solve stopped at its
  pre-registered 12 gradient evaluations while still descending, so neither
  number is a converged optimum.

## Mean-shift control: the win is shaping, not level

The delivered map has volume-weighted mean 0.9334, so the margin could in
principle have come from simply putting less dopant everywhere rather than from
putting it in the right places. That is now measured, not argued.

**Control arm:** a spatially uniform map at s = 0.933439748305557 on every part
cell, scored through the exact same `phase_c_run.score_arm` read rule (envelope
argmin of the asymmetric objective), one forward on each mesh. On the hold-out
mesh the constant is used directly, and that is legitimate as a measurement,
not an assumption: `transfer_map_across_meshes` is a row-normalized
convolution, and re-evaluating a constant through it on 2000 randomly chosen
destination cells reproduces the constant to a maximum absolute deviation of
**4.44e-16**, that is to machine precision.

| | solve mesh | hold-out mesh |
|---|---|---|
| J, uniform s = 1 | 1.1511359e-07 | 8.2732480e-08 |
| J, uniform at the level s = 0.9334 | 1.1502099e-07 | 8.2633161e-08 |
| J, symmetrized map | 1.0712930e-07 | 7.6415689e-08 |
| **total margin** | **-6.936 %** | **-7.635 %** |
| of which LEVEL | -0.080 % | -0.120 % |
| of which SHAPE | **-6.856 %** | **-7.515 %** |
| shape share of the total | **98.8 %** | **98.4 %** |

**Verdict, plainly: the level accounts for essentially none of the win.**
Lowering the dopant uniformly from 1.0 to 0.9334 buys 0.08 percent on the solve
mesh and 0.12 percent on the hold-out mesh. Between 98 and 99 percent of the
margin on both meshes comes from the spatial structure of the map. The
confound named in the first version of this report is closed, and it closed in
the favourable direction.

This matters for how the dissertation words the win. The claim
"a direct three-dimensional design solve beats a uniform dopant" is safe, and
the stronger reading is now also supported: the benefit is attributable to
grading the dopant, not to using less of it. The wording should not hedge with
a mean-level caveat, because the control rules it out on both meshes.

Artifacts: `scripts/analysis/score_mean_shift_control.py`,
`scripts/analysis/mean_shift_control.json` (checkpointed per stage, including
the transfer-of-a-constant check and the full `score_arm` records for both
control arms).

## Named limits

1. **Not a solve in the symmetric subspace.** The delivered map is a projection
   of a full-space solve. A Stage B re-solve parameterized on the quadrant
   orbit at budget 40 would very likely do better than 6.94 percent in-grid,
   and its in-grid number would be trustworthy rather than inflated. Stage A
   met the acceptance rule, so per the staged mandate Stage B was not run.
2. ~~The mean-shift confound is not yet separated.~~ **CLOSED**, see the
   mean-shift control section above. The level accounts for 1.2 percent of the
   margin on the solve mesh and 1.6 percent on the hold-out mesh; the rest is
   shaping.
3. **Nearest-centroid matching is not exact.** The tetrahedral mesh is not
   itself mirror-symmetric, so the group action is approximate; the worst match
   distance is 7.7e-04 m, comparable to the 1.0 mm filter radius. The residual
   asymmetry after projection and filtering is 0.44 percent of variance, and
   total in-part dopant moved by 1.3e-06 relative, so the projection neither
   invented nor destroyed dopant.
4. **Physics scope is unchanged.** Coupling off, no densification, Phase A
   fixed-power drive, one shape. This certifies the design method on the
   dolfinx forward and certifies nothing against experiment.

## Artifacts

| path | what |
|---|---|
| `solve3d/results/phase_c_map_symmetrized_filter_only_asymmetric_scaled.npz` | the delivered map (`s_map`), plus `s_map_source` (the pre-projection map) |
| `solve3d/results/phase_c_solves.json` -> `arms.symmetrized_filter_only_asymmetric_scaled` | the scored arm, with the group and the deviation recorded in the record itself |
| `solve3d/results/phase_c_gate.json` -> same arm | mesh hold-out and smoothing robustness |
| `scripts/analysis/cylinder_map_symmetry.py` / `.json` | the group derivation, the projection, and the decomposition numbers in this report |
| `scripts/analysis/score_symmetrized_cylinder_map.py` | the scoring driver, checkpointed per stage |
| `scripts/analysis/score_mean_shift_control.py` / `mean_shift_control.json` | the level-versus-shape control, both meshes |
| `fgm_solve_campaign/figs_3d/export_phase_c_fields_sym.py` | the one forward that produced the figure's read-state fields |
| `fgm_solve_campaign/figs_3d/phase_c_cylinder_fields_sym_gate.txt` | reproduction gate, eight scalars, relative error 0.000e+00 |
| `fgm_solve_campaign/figs_3d/fig_cylinder_3d_solve.png` | the regenerated dissertation figure |

No `solve3d` solver module was modified.
