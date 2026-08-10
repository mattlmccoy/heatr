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

## What each panel reads

| panel | content | artifact(s) read |
|---|---|---|
| (a) | uniform arm design field, s = 1 on every part cell | constructed as ones on the part-cell set of `solve3d/results/phase_c_map_solve_filter_only_asymmetric_scaled.npz` (`centroids`), matching `phase_c_run.run_baselines`, which scores `np.ones(npart)` |
| (b) | solved dopant map, mid-height cross-section | `solve3d/results/phase_c_map_solve_filter_only_asymmetric_scaled.npz`, keys `s_map` and `centroids` |
| (c) | objective density binned by azimuth, both arms | `fgm_solve_campaign/figs_3d/phase_c_cylinder_fields.npz`, keys `uniform_phi`, `solved_phi`, `chi_nodal`, `vol_nodal`, `node_xyz` |
| (d) | objective density binned by height, both arms | same npz as (c) |
| (e) | J total, in-bounds deficit, bed-melt term, both arms; hold-out annotation | `solve3d/results/phase_c_baselines.json` (`arms.uniform_baseline`), `solve3d/results/phase_c_solves.json` (`arms.solve_filter_only_asymmetric_scaled`), `solve3d/results/phase_c_gate.json` (`arms.solve_filter_only_asymmetric_scaled.mesh_holdout`) |
| (f) | J against gradient evaluation, 12 of 12 | `solve3d/results/phase_c_solves.json`, `arms.solve_filter_only_asymmetric_scaled.trajectory` |
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

Reproduction gate, recorded in
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
* **Bin counts that improve (new numbers, computed at render time):** 24 of 24
  height bins and 35 of 36 azimuth bins fall under the solved map.
* Colormaps: viridis (sequential) and two flat hues distinguished by lightness
  as well as hue, so the figure survives greyscale and common colour-vision
  deficiencies. No dark theme, no figure title inside the image.

---

## Proposed LaTeX caption

> Direct three-dimensional design solve on the Phase C cylinder, against a
> uniform dopant baseline. (a) The uniform arm places saturation
> $s = 1$ on every part cell. (b) The solved map at mid height varies both
> radially and azimuthally, with saturation between 0.68 and 1.00 and a
> volume-weighted mean of 0.933. (c, d) The shape-fidelity objective binned by
> azimuth and by height over the part surface, where all of the objective sits
> at the read state; the shaded band is the part the solve removes, and it is
> negative in 24 of 24 height bins and 35 of 36 azimuth bins. (e) On the solve
> mesh the objective falls 10.67 percent, split into a 6.99 percent reduction
> in the in-bounds density deficit and a 22.44 percent reduction in the
> out-of-bounds bed-melt term. Both arms are read at the asymmetric
> objective's own stopping step, so the margin is an effect of the map and not
> of the stopping time. (f) The solve is stopped by its pre-registered budget
> of twelve gradient evaluations while the objective is still falling, so
> 10.67 percent is an upper bound on the achievable objective and not a
> converged optimum. The map was designed on the coarse mesh and re-scored on
> a finer hold-out mesh, where it still beats the uniform arm by 3.51 percent;
> the win transfers across meshes, its magnitude does not. For comparison, the
> published inversion heuristic produces no benefit on this shape, returning
> $+0.3$ percent in $\sigma_T$ under its own engine and metric.

Wording constraints honoured: plain academic voice, no em dashes, the word
"surrogate" does not appear, the budget-limited caveat and the two-mesh
statement are both in the caption.

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

## Reproduce

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
      heatr3d_d1_spike/env/bin/python \
      fgm_solve_campaign/figs_3d/export_phase_c_fields.py     # ~7 min, 2 forwards
    ./.venv312/bin/python \
      fgm_solve_campaign/figs_3d/render_fig_cylinder_3d_solve.py
