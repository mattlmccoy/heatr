# Phase E cube, light-style dissertation figures: provenance, captions, limits

Two figures, both white background, no title inside the image, no deck chrome,
300 DPI, 6.9 in wide.

| file | size | renderer |
|---|---|---|
| `fgm_solve_campaign/figs_3d/fig_cube_three_arms.png` | 2070 x 1620 px | `fgm_solve_campaign/figs_3d/render_fig_cube_three_arms.py` |
| `fgm_solve_campaign/figs_3d/fig_cube_solved_map.png` | 2070 x 1455 px | `fgm_solve_campaign/figs_3d/render_fig_cube_solved_map.py` |

Shared plumbing and the reproduction gate live in
`fgm_solve_campaign/figs_3d/cube_light_common.py`. Both renderers call the gate
before they draw anything and refuse to write a figure if it fails.

Reproduce:

    PYTHONPATH="$PWD:$PWD/fgm_solve_campaign/figs_3d" \
      ./.venv312/bin/python fgm_solve_campaign/figs_3d/render_fig_cube_three_arms.py
    PYTHONPATH="$PWD:$PWD/fgm_solve_campaign/figs_3d" \
      ./.venv312/bin/python fgm_solve_campaign/figs_3d/render_fig_cube_solved_map.py

No solve3d source file was modified. No solve, no forward evaluation and no
optimizer call happens at render time. Unlike the Phase C cylinder figure, no
field export was needed: Phase E already stored the nodal read state for all
three arms, so the export/forward-rerun pattern of
`export_phase_c_fields.py` does not apply here.

Source of record: `solve3d/phase_e/PHASE_E_OPENER_REPORT.md`, sections 7a to
7e. The pyramid arm is deliberately absent from both figures.

---

## Per-panel provenance

### fig_cube_three_arms.png

| panel | content | artifact(s) read |
|---|---|---|
| (a) (b) (c) | melt body, the isosurface $\varphi = 0.9$, amber where inside the nominal solid and dark red where outside, inside the analytic nominal cube wireframe | `solve3d/phase_e/results/vol_cube.npz`, keys `T__uniform_baseline`, `T__heuristic_grading_law`, `T__solve_filter_only`, `inside`, `axis_m`, `nominal_base_side_m`; isosurface by `solve3d.phase_e.render_deck_melt_body.melt_body` (imported, unmodified) |
| per-arm text: melt fraction, stop time | `part_mean_phi`, `t_stop_s` | `solve3d/phase_e/results/phase_e_cube.json`, `arms.<arm>` |
| per-arm text: objective $J$ percent | computed at render time as `J_asymmetric / uniform.J_asymmetric - 1` | same JSON |
| (d) (e) (f) | vertical mid-plane cut of the same three volumes; the black dashed curve on (e) and (f) is the uniform arm's own $\varphi = 0.9$ contour on the same plane; the black square is the analytic nominal outline, not a contour of the voxel mask | same `vol_cube.npz` |

### fig_cube_solved_map.png

| panel | content | artifact(s) read |
|---|---|---|
| (a) to (d) | four z slabs of the delivered map, drawn as volume-weighted bin averages on a 26 x 26 grid, each titled with its own volume-weighted mean saturation, computed at render time | `solve3d/phase_e/results/map_cube_solve_filter_only.npz`, keys `s_map`, `centroids`, `volumes` |
| map statistics block (cell count, mean, range) | recomputed from the same arrays, gated against `arms.solve_filter_only.map_stats` | same npz, `phase_e_cube.json` |
| (e) | mean saturation in 40 build-height bins against the uniform arm's $s = 1$ | same npz |
| (f) | quarter cutaway of the same map, lifted to the render grid by nearest design cell | `map_cube_solve_filter_only.npz` plus `vol_cube.npz` keys `inside`, `design_nearest_index`, `axis_m`; geometry by `render_deck_cutaway3d.surface` / `.face_colors` and `render_deck_loop3d.design_volume` (imported, unmodified) |
| (g) | five-row metrics table; every ratio and percentage is computed at render time from the JSON, none is transcribed | `phase_e_cube.json`, `arms.uniform_baseline` and `arms.solve_filter_only` |
| footer | mesh size, design-cell count | `phase_e_cube.json`, `arms._mesh.n_cells` |

Values that appear in the table, read from the JSON at render time and matching
`PHASE_E_OPENER_REPORT.md` section 7a and 7b: mean melt fraction 0.64 to 0.75,
melt-region IoU at $\varphi \geq 0.9$ 0.600 to 0.738, melt-front distance 1.94
to 1.20 mm, fraction below the melt floor 0.403 to 0.279, objective
7.842e-07 to 4.228e-07, that is -46 percent. Slab means 0.60, 0.75, 0.76, 0.60.

---

## Reproduction gate

`cube_light_common.gate` recomputes every drawn scalar that also exists in the
recorded JSON and writes
`fgm_solve_campaign/figs_3d/fig_cube_3d_gate.txt`. Result, 14 of 14 PASS:

* Bit-identical, relative error exactly 0.000e+00: volume-weighted mean of the
  delivered map against `map_stats.mean`, map min, map max, design-cell count,
  the three per-arm sampler `missed_in_part` counts, and both stored symmetry
  fractions.
* Floating-point identical at 1.9e-13 and 9.3e-13: the two headline objective
  deltas, -46.0861458043 percent and +49.9975737291 percent, against the
  full-precision values behind the report's "-46.09 %" and "+50.00 % worse".
* Tolerance checks, stated rather than hidden: the volume-weighted mean melt
  fraction recomputed on the 96^3 render grid against the FEM scorer moves
  1.63e-03 (uniform), 4.25e-03 (heuristic) and 1.64e-02 (solve), against a 3
  percent tolerance. These are two different quadratures of the same field, so
  they cannot be bit-identical, and the figure's melt-fraction numbers are the
  FEM ones from the JSON, not the grid ones.

The gate was confirmed to be capable of failing: an early run with a mistyped
reference percentage returned `reproduction gate FAILED: dJ_solve_percent rel
7.147e-05` and no figure was written.

---

## Proposed captions

### fig_cube_three_arms.png

> Three correction arms on the Phase E cube, each evaluated on the same
> transient three-dimensional forward model and read at its own envelope stop.
> The solid body in (a) to (c) is the melted material, the isosurface
> $\varphi = 0.9$, which is the part that would form; amber is melt inside the
> nominal solid and dark red is melt that has escaped it, and the thin black
> wireframe is the nominal cube. (a) With no correction, a uniform dopant
> loading reaches a volume-weighted mean melt fraction of 0.64 at a stop time
> of 514 s. (b) The hand-built grading law from the earlier heuristic study
> reaches 0.58 at 439 s and raises the shape-fidelity objective by 50 percent,
> a loss rather than a gain, which the objective split attributes to a factor
> 6.1 increase in the out-of-bounds term. (c) The direct three-dimensional
> adjoint solve reaches 0.75 at 624 s and lowers the objective by 46 percent.
> (d) to (f) cut the same three bodies on the vertical mid-plane through the
> part centre, with the no-correction melt front repeated as a dashed curve, so
> amber outside that curve is the material the correction added. The solve was
> stopped by its pre-registered budget of twelve gradient evaluations while the
> objective was still falling, so the 46 percent figure is a lower bound on
> what this arm can reach and not a converged optimum. Simulation only; no
> experimental measurement enters this figure.

### fig_cube_solved_map.png

> The dopant map produced by the direct three-dimensional solve on the Phase E
> cube, 24042 design cells with saturation between 0.005 and 1.000 and a
> volume-weighted mean of 0.676. (a) to (d) Four slabs through build height,
> each drawn as a volume-weighted bin average and labelled with its own mean
> saturation. The solve holds dopant back at both ends of the build axis, 0.60
> in the outer slabs against 0.75 and 0.76 in the two central slabs, and inside
> the central slabs it empties the two faces whose normals lie along $x$. (e)
> The same structure as a profile against build height, against the uniform
> arm's $s = 1$. The profile is symmetric about the mid-plane, which is the
> right answer for a symmetric part and is not enforced anywhere in the
> objective or the design chain. (f) The map as a quarter cutaway. (g) Shape
> fidelity for the uniform arm and the solved map, each read at its own
> envelope stop: the mean melt fraction rises from 0.64 to 0.75, the
> melt-region intersection over union at $\varphi \geq 0.9$ from 0.600 to
> 0.738, the melt-front distance falls from 1.94 mm to 1.20 mm, the fraction of
> the part below the melt floor falls from 0.403 to 0.279, and the objective
> falls from 7.842e-07 to 4.228e-07, a reduction of 46 percent. The solve was
> stopped by its twelve-evaluation budget while still descending, so this is a
> lower bound rather than a converged optimum. The map is spatially consistent
> under the symmetry group its own problem admits, scoring 0.912 for the
> mirror group in $x$ and $z$ against a threshold of 0.8. Simulation only.

Wording constraints honoured in both captions: plain academic voice, no em
dashes, the word "surrogate" does not appear, the budget-limited caveat is
stated, and the scope is tagged as simulation only.

---

## The symmetry-consistency gate, stated precisely

`solve3d/results/symmetry_retro_3d.json` records the cube as **PASS**.

* Under the group the cube's problem actually admits, the mirror group
  $\{x, z\}$, the symmetric component carries **0.9121** of the
  volume-weighted variance of the map.
* Under the full $\{x, y, z\}$ mirror group it carries **0.9063**, which is the
  0.906 quoted in the request.

Both are above the 0.8 threshold, so the verdict does not depend on the choice.
The correct group excludes the $y$ mirror because convection is applied on the
top face only, at $y = +L/2$, so the thermal problem and the objective are not
$y$-mirror symmetric even though the electrostatic field is. For the square in
the same retro-check that distinction is load bearing (0.860 corrected against
0.699 uncorrected); for the cube it is not, because the cube scores above
threshold either way. The pyramid fails the same gate at 0.354 and appears
nowhere in these figures or notes.

---

## What is NOT claimed, and what could not be verified here

1. **The cube map does not carry the pre-registered SOLVED label.**
   `phase_e_gate_cube.json` gives `solved_label = false`. Smoothing robustness
   passes at 2.111 percent against a 10 percent tolerance, and all five shape
   metrics pass their measured mesh hold-out bands, but the scalar objective
   moves 22.61 percent coarse-to-fine against a 9.64 percent band and fails.
   At the finer score mesh the solved map still beats uniform by 37.89 percent.
   Neither figure states or implies that the map is validated. If a chapter
   sentence needs one line on this, the honest one is that the improvement
   transfers across meshes but its magnitude does not.
2. **Absolute fidelity numbers on a cornered shape are ungated.** The cube has
   twelve edges and eight corners and the campaign's own S2 finding is that
   corner behaviour is unsettled. What is meaningful is the arm-to-arm
   comparison at matched mesh and matched read state.
3. **Two different out-of-part quantities exist and only one is drawn.** On the
   96^3 render grid the melt body of the heuristic arm has 3505 triangle faces
   outside the nominal solid, while the uniform and solve arms have none. The
   JSON `out_of_part_melt_fraction_of_part` is the finite-element integral of
   continuous $\varphi$ over out-of-part cells and is non-zero for all three
   arms: 0.00370 uniform, 0.01181 heuristic, 0.00296 solve. Absence of red in a
   panel is therefore not evidence of zero bed melt, and no caption says it is.
4. **The red spill is visible in the cut row, not in the three-dimensional
   row.** In panel (b) the escaped melt sits under the part and is occluded by
   the body at this camera. Panel (e) carries the annotation instead. This is a
   rendering limitation, not a data difference between the rows.
5. **Display-only smoothing.** The melt isosurface is drawn from a 0.8 cell
   Gaussian blur of $\varphi$, inherited unchanged from
   `render_deck_melt_body.SMOOTH_CELLS`, because the field is piecewise linear
   and the raw isosurface rings on element boundaries. No number anywhere in
   either figure is computed from the smoothed field.
6. **Physics scope.** Coupling is off, there is no densification in the forward
   march, the drive is the fixed-total-power renormalization differentiated
   through, and the dopant enters as a saturation field on part cells only.
   These figures certify a design result on the dolfinx forward model. They
   certify nothing against experiment.
7. **One shape.** The cube alone. Nothing here says how the method behaves on
   reentrant or thin-neck geometry.
