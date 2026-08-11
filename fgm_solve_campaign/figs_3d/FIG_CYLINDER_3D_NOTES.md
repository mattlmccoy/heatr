# fig_cylinder_3d_solve.png -- provenance, caption, and limits

Figure: `fgm_solve_campaign/figs_3d/fig_cylinder_3d_solve.png`
(6.5 in x 7.1 in at 300 DPI, 1950 x 2130 px, white background,
viridis for the design field, no title inside the image).

Renderer: `fgm_solve_campaign/figs_3d/render_fig_cylinder_3d_solve.py`
(`./.venv312/bin/python fgm_solve_campaign/figs_3d/render_fig_cylinder_3d_solve.py`).
No solve3d solver code was modified.

Source of record: `solve3d/PHASE_C_REPORT.md`. Every percentage drawn in the
image is computed at render time from the stored JSON, not transcribed.

---

## THE DELIVERED ARM CHANGED (2026-08-10): symmetry projection

The figure now shows the arm
`symmetrized_filter_only_asymmetric_scaled`, not
`solve_filter_only_asymmetric_scaled`. Set
`PHASE_C_ARM=solve_filter_only_asymmetric_scaled` to render the old arm.

**Why.** The Phase C case has a symmetry group and the 12-evaluation solved map
did not respect it. The group is the intersection of part, chamber,
electrode-field, objective and convection-boundary symmetry, read from the
source: the cylinder axis is z and it spans the full chamber
(`in_part_predicate("circle")` has no z condition;
`occ.addCylinder(0,0,-L/2, 0,0,L, half)` in `heatr3d_d1_spike/mesh_gmsh.py`);
the electrodes are at y = +/- L/2 (`forward.py` l.274-278) and the deposited
power goes as `|E|^2`, which is even in y, but they fix the field direction so
no rotation about the axis survives; and convection acts on the **top face
y = +L/2 only** (`forward.py` l.111, l.563-577), which breaks the y-mirror.
The group is therefore the Klein four-group
**{identity, x-mirror, z-mirror, xz-mirror}, with the y-mirror excluded**.
Odd-in-y content in a map on this case is the physical response to the
one-sided convection and must be kept.

The original solved map retained only **42.57 percent** of its
volume-weighted variance under this projection (49.89 percent on the mid-height
slab), so about half its structure was mesh-frame fitting from the
budget-limited stop. Independently confirmed by the 3-D lane's
`solve3d/symmetry_gate.py`, which reports 0.4258 for the same map
(`solve3d/results/symmetry_retro_3d.json -> results.cylinder_phase_c`).

**What was done.** `scripts/analysis/cylinder_map_symmetry.py` averages the map
over the four group elements using nearest-centroid matching, applies the
pre-registered 1.0 mm design filter once to clean the matching artefacts, and
clips to [0, 1]. `scripts/analysis/score_symmetrized_cylinder_map.py` then
scores it through the existing `phase_c_run.score_arm` and
`phase_c_run.run_acceptance`, so the numbers come from the same code path as
every other arm. No solve was re-run and no `solve3d` solver module was
modified. The arm is recorded as an ADDITIONAL recorded-deviation arm and does
not displace the pre-registered one.

**Result.** In-grid margin falls from 10.67 to **6.94 percent**; the hold-out
margin rises from 3.51 to **7.64 percent**; both acceptance gates still pass
and the smoothing sensitivity improves from 0.22 to 0.041 percent. The
delivered map is 99.56 percent symmetry-consistent. Full accounting in
`CYLINDER_MAP_FIX_REPORT.md` at the repository root.

**Panel (f) honesty.** The trajectory drawn is still the SOURCE solve's twelve
evaluations, because the delivered map is the symmetric part of that solve's
best iterate and not a new solve. The delivered map's own J is marked as a
separate point above the trajectory endpoint, so the reader sees that the
projection costs in-grid objective.

---

## What each panel reads

| panel | content | artifact(s) read |
|---|---|---|
| (a) | uniform arm design field, s = 1 on every part cell | constructed as ones on the part-cell set of `solve3d/results/phase_c_map_symmetrized_filter_only_asymmetric_scaled.npz` (`centroids`), matching `phase_c_run.run_baselines`, which scores `np.ones(npart)` |
| (b) | delivered dopant map, mid-height cross-section | `solve3d/results/phase_c_map_symmetrized_filter_only_asymmetric_scaled.npz`, keys `s_map` and `centroids` |
| (c) | objective density binned by azimuth, both arms | `fgm_solve_campaign/figs_3d/phase_c_cylinder_fields_sym.npz`, keys `uniform_phi`, `solved_phi`, `chi_nodal`, `vol_nodal`, `node_xyz` |
| (d) | objective density binned by height, both arms | same npz as (c) |
| (e) | J total, in-bounds deficit, bed-melt term, both arms; hold-out annotation | `solve3d/results/phase_c_baselines.json` (`arms.uniform_baseline`), `solve3d/results/phase_c_solves.json` (`arms.symmetrized_filter_only_asymmetric_scaled`), `solve3d/results/phase_c_gate.json` (`arms.symmetrized_filter_only_asymmetric_scaled.mesh_holdout`) |
| (f) | J against gradient evaluation, 12 of 12, for the SOURCE solve, plus the delivered map's own J as a separate marked point | `solve3d/results/phase_c_solves.json`, `arms.solve_filter_only_asymmetric_scaled.trajectory` and `arms.symmetrized_filter_only_asymmetric_scaled.J_asymmetric` |
| footer line 1 | mesh size, coupling, read time | `phase_c_baselines.json` `mesh.n_cells`, `case.coupling`; `phase_c_solves.json` `t_stop_s` |
| footer line 2 | inversion heuristic result | `heatr3d_eqs02_rerank/FGM_BENEFIT_RERUN.md`, cylinder row, arm D against arm B: +0.3 % sigma_T |

## The one piece of new compute, and its gate

Phase C's scorer recorded scalars only; it never wrote the nodal read state, so
panels (c) and (d) had no stored field to read. One forward evaluation per arm
was re-run on the stored maps by
`fgm_solve_campaign/figs_3d/export_phase_c_fields.py`
(spike env, `heatr3d_d1_spike/env/bin/python`, `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`), reproducing `phase_c_run.score_arm`'s read rule
exactly. No solve was re-run and no optimizer was invoked.

For the symmetrized arm the same was done by
`fgm_solve_campaign/figs_3d/export_phase_c_fields_sym.py`, which costs ONE
forward rather than two because the uniform arm's fields are copied from the
existing npz (same mesh, same case, same map of ones). Its reproduction gate,
`phase_c_cylinder_fields_sym_gate.txt`, checks eight scalars against
`phase_c_solves.json` and reports relative error **0.000e+00, that is
bit-identical**, on all eight.

Reproduction gate for the original arm, recorded in
`fgm_solve_campaign/figs_3d/phase_c_cylinder_fields_gate.txt`: sixteen scalars
(J_symmetric, J_asymmetric, both argmins, J_out_of_bounds,
J_in_bounds_deficit, out_of_part_melt_fraction_of_part,
in_bounds_below_floor_fraction, for each of the two arms) reproduce the
recorded Phase C JSON at **relative error 0.000e+00, that is bit-identical**.
The script raises rather than writing a figure if any check exceeds rtol 1e-9.

Field file: `fgm_solve_campaign/figs_3d/phase_c_cylinder_fields.npz`
(also holds `uniform_T_eval` / `solved_T_eval` on the shared Phase A
evaluation grid, unused by the final figure). It is 7.8 MB and is NOT
committed; it is regenerated in about seven minutes by the export command at
the bottom of this file, and the gate text file that certifies it is
committed.

## Rendering conventions and the choices behind them

* **Mid-height cross-section, panels (a) and (b).** The design field is DG0,
  cellwise constant on tetrahedra, and no centroid lies exactly on z = 0, so
  the panel takes the cells with |z| <= 0.5 mm and colours each pixel by its
  nearest centroid. Nearest-centroid is the field itself, not a smoothing.
  Pixels outside r = 9.8 mm are left blank so the outline is not aliased.
* **Panels (c) and (d) are the objective, not a proxy.** They bin the exact
  per-node integrand
  `vol_i [ 10 max(phi_i - chi_i, 0)^2 + max(0.85 chi_i - phi_i, 0)^2 ]`
  by azimuth (36 bins) and by height (24 bins). The bars therefore sum to the
  J values shown in (e).
* **Why a surface quantity.** Measured on the re-exported fields: every
  interior node is fully melted (phi = 1) at the read state and the flat ends
  of the part sit on the domain boundary with chi = 1, so **100.0 % of J sits
  on the lateral surface nodes**. The objective is a melt-boundary quantity on
  this geometry. That is a new observation from this render, not a Phase C
  report number.
* **Bin counts that improve (computed at render time):** 23 of 24 height bins
  and 33 of 36 azimuth bins fall under the delivered symmetrized map. The raw
  arm scored 24 of 24 and 35 of 36; the symmetrized map trades a little
  in-grid bin coverage for the hold-out margin, consistent with the J numbers.
* Colormaps: viridis (sequential) and two flat hues distinguished by lightness
  as well as hue, so the figure survives greyscale and common colour-vision
  deficiencies. No dark theme, no figure title inside the image.

---

## Proposed LaTeX caption

> Direct three-dimensional design solve on the Phase C cylinder, against a
> uniform dopant baseline. (a) The uniform arm places saturation $s = 1$ on
> every part cell. (b) The delivered map at mid height. The cylinder axis is
> $z$ and the electrodes lie at $y = \pm L/2$, so the problem is invariant
> under the mirrors $x \to -x$ and $z \to -z$ but not under $y \to -y$, which
> the one-sided convection on the top face breaks. The solved map is projected
> onto that symmetry group and re-filtered once, which removes the
> discretization-frame content that the budget-limited solve had fitted; the
> map retains 42.6 percent of its variance under the projection. Saturation
> runs from 0.85 to 0.99 with a volume-weighted mean of 0.933, and is highest
> near the convectively cooled face. (c, d) The shape-fidelity objective binned
> by azimuth and by height over the part surface, where all of the objective
> sits at the read state; the shaded band is the part the design removes, and
> it is negative in 23 of 24 height bins and 33 of 36 azimuth bins. (e) On the
> solve mesh the objective falls 6.94 percent, split into a 5.70 percent
> reduction in the in-bounds density deficit and a 10.89 percent reduction in
> the out-of-bounds bed-melt term. Both arms are read at the asymmetric
> objective's own stopping step, so the margin is an effect of the map and not
> of the stopping time. (f) The underlying solve is stopped by its
> pre-registered budget of twelve gradient evaluations while the objective is
> still falling, so this is not a converged optimum; the delivered map is the
> symmetric part of that solve's best iterate and is marked separately, above
> the trajectory, since the projection gives up in-grid objective. The map was
> designed on the coarse mesh and re-scored on a finer hold-out mesh, where it
> beats the uniform arm by 7.64 percent, so the win transfers across meshes.
> For comparison, the published inversion heuristic produces no benefit on this
> shape, returning $+0.3$ percent in $\sigma_T$ under its own engine and
> metric.

Wording constraints honoured: plain academic voice, no em dashes, the word
"surrogate" does not appear, the budget-limited caveat, the symmetry-projection
step and the two-mesh statement are all stated plainly in the caption.

## What could not be verified here

1. **The +0.3 percent inversion number is cross-engine.** It comes from
   `FGM_BENEFIT_RERUN.md` (heatr3d, n = 64 voxels, sigma_T), not from the
   dolfinx asymmetric objective used everywhere else in this figure. It is
   therefore reported as a separate sentence with its engine and metric named,
   and is deliberately NOT drawn on the same axis as J. Phase C dropped the
   inversion arm because the trilinear transfer onto the conforming mesh moved
   total in-part dopant by 7.15 percent, past the pre-registered 2 percent
   drop condition, so there is no in-campaign J for that arm to plot.
2. **The 0.5 mm mid-height slab in (a) and (b) is a rendering choice.** It has
   no counterpart in the report and is not part of any gate.
3. **Physics trust is unchanged by this figure.** Coupling is off, there is no
   densification in the forward, and the drive is the Phase A fixed-power
   renormalization. Phase C certifies the design method on the dolfinx
   forward; it does not certify the physics against experiment, and no Studio
   badge follows from it.
4. **One shape.** The cylinder only. Generalization is Phase E work.
5. **The mean-shift confound is CLOSED, in the favourable direction.** The
   delivered map has volume-weighted mean 0.9334, so the margin could have come
   from using less dopant rather than from placing it well. A control arm at a
   spatially uniform $s = 0.9334$ was scored through the same read rule on both
   meshes: it beats $s = 1$ by only 0.080 percent on the solve mesh and 0.120
   percent on the hold-out mesh, so **98.8 and 98.4 percent of the margin is
   spatial structure, not level**. The caption's claim is therefore about
   grading the dopant, and needs no mean-level hedge. Numbers and method in
   `CYLINDER_MAP_FIX_REPORT.md`, mean-shift control section.
6. **This is a projection, not a symmetric-subspace solve.** A re-solve
   parameterized on the quadrant orbit at the full budget would probably beat
   6.94 percent in grid, and its in-grid number would not be inflated by
   mesh-frame fitting. Stage A met the acceptance rule, so it was shipped.

## Reproduce

    # 1. build and diagnose the symmetrized map (seconds, no solver)
    ./.venv312/bin/python scripts/analysis/cylinder_map_symmetry.py

    # 2. score it and run both acceptance gates (~70 min, 4 forwards,
    #    two of them on the finer hold-out mesh; checkpointed per stage)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
      heatr3d_d1_spike/env/bin/python \
      scripts/analysis/score_symmetrized_cylinder_map.py

    # 3. export the read-state fields for panels (c) and (d) (~5 min, 1 forward)
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
      heatr3d_d1_spike/env/bin/python \
      fgm_solve_campaign/figs_3d/export_phase_c_fields_sym.py

    # 4. render
    ./.venv312/bin/python \
      fgm_solve_campaign/figs_3d/render_fig_cylinder_3d_solve.py

For the superseded raw arm, run `export_phase_c_fields.py` (~7 min, 2 forwards)
and render with `PHASE_C_ARM=solve_filter_only_asymmetric_scaled`.
