# Pyramid FGM demo (quick visual, 2026-08-03; strong-law update same day)

**Honest scoping (applies to everything here): heuristic graded map (not solved),
heatr3d n=48, simulation-only.** The simulation is REAL heatr3d physics (EQS +
enthalpy thermal march + densification; no fabricated fields), but the dopant map
is HAND-CONSTRUCTED and physically motivated, not the output of any solve or
optimization, and heatr3d itself is the exploratory, less-validated 3-D engine
(see HEATR_STANDARD_PARAMETERS.md). Nothing here is validated physics or a
performance claim.

## Geometry
- `shape_library_3d/stl/pyramid.stl` (equal-volume class, 4188.79 mm^3, apex +z),
  dimensions read from the STL, occupancy evaluated analytically at voxel centers
  (trimesh's voxelizer under-fills this minimal 6-face mesh: base read 13 cells
  instead of the true ~18.6 at h = 1.25 mm, so the analytic primitive is used).
- Grid n = 48, h = 1.25 mm, part = 1956 voxels (3820 mm^3, 91% of nominal at this
  resolution), centered in the chamber, tip pointing +z.

## Grading laws (heuristic, vary in x, y, AND z)
Shared form: `sat = clip(peak * (d_floor + d_gain * d_hat) * (1 - z_taper * z_hat),
s_min, 1.00)` with `d_hat` = normalized Euclidean distance-to-surface inside the
part and `z_hat` = normalized height above the base.

- **Mild (v1, first demo):** `sat = clip(0.95*(0.35 + 0.65*d_hat)*(1 - 0.45*z_hat), 0.20, 1.00)`
- **Strong (headline, current figures):** `sat = clip(0.95*(0.15 + 0.85*d_hat)*(1 - 0.75*z_hat), 0.10, 1.00)`

Motivation for both: RF field concentration at exterior corners, sharp edges, and
the apex overheats those regions, so dopant is graded DOWN near the surface (which
contains all corners/edges) and DOWN toward the apex, higher in the core and base.
Implemented in `grading.py`; red-green tested in `test_grading_law.py` (bounds,
zero outside part, variation along every axis, core>skin, base>apex, and
strong-has-more-contrast-than-mild).

## The saturation lesson (why the exposure changed from 1200 s to 900 s)
The first comparison ran both arms to 1200 s, past the uniform arm's
phi_bar = 0.90 crossing (1195.5 s). By then nearly the whole bulk had densified to
~1.0 in BOTH arms, so the final-density side-by-side looked identical even though
the underlying fields genuinely differed (fig5's dT and dQ panels prove the
mid-run differences from the run artifacts alone). Lesson: a saturating readout
hides a real difference; pick the read state before the metric ceilings out. The
update reads both arms at 900 s, chosen from the uniform arm's own trajectory
(pre-crossing, mid-densification). Same exposure for both arms, always.

## Run settings (arms identical except the map)
- `heatr3d.run(grid, part, Params(phase_update="enthalpy"), sat=..., max_time_s=900,
  densify=True)`; masked Qrf gradient (solver default); OMP/OPENBLAS threads = 1.
- Absorbed power is renormalized by the solver, so both arms receive the same
  total power (5472.2 J at 900 s); grading only redistributes it.
- Neither 900 s arm crosses phi_bar = 0.90 (by design, pre-saturation read), so
  the stored T field is the END-OF-EXPOSURE read, labeled as such on the figures.

## Results
900 s read (current figures, strong law vs uniform):
| quantity | uniform 900 s | strong graded 900 s |
|---|---|---|
| wall time (s) | 179.2 | 179.1 |
| melt height above base, phi>=0.5 (mm) | 12.5 | 13.75 |
| melted voxels (phi>=0.5) | 1616 | 1636 |
| mean rho_final over part | 0.874 | 0.885 |
| std(rho_final) over part | 0.1737 | 0.1693 |
| std(T) at end of exposure (C) | 28.953 | 30.343 |
| T_max (C) | 245.1 | 262.3 |
| energy standing gate (residual_frac) | +0.0000 | +0.0000 |
Mid-slice differences (fig5): max |delta Qrf| = 12.1% of the uniform peak, max
|delta T| = 20.7 C, max |delta rho_final| = 0.28 in the band where the graded
melt front advanced one layer higher.

1200 s read (first demo, mild law vs uniform, kept in out_graded / out_uniform):
| quantity | uniform | mild graded |
|---|---|---|
| t at phi_bar=0.90 (s) | 1195.5 | 1116.8 |
| sintered fraction | 0.901 | 0.900 |
| std(rho_final) | 0.1404 | 0.1326 |
| std(T) at phi=0.90 latch (C) | 29.234 | 29.436 |

Honest reading: the strong grading VISIBLY moves the melt front one build layer
higher (+1.25 mm) and concentrates heating in the core/base (hotter core, +17 C
T_max), at the cost of slightly higher std(T). This demo shows the CAPABILITY of
a 3-axis graded map running through real physics, not a tuned benefit claim.

## Files
- `grading.py`, `test_grading_law.py` - both laws + tests
- `run_demo.py` - voxelize + solve one arm (`graded` | `strong` | `uniform`), optional exposure arg
- `render_figs.py` - deck-style renders (style: `deck_figures_3d/style3d.py`)
- `render_fig5_diff.py` - difference panels (graded minus uniform)
- `out_strong_900/`, `out_uniform_900/` - current-figure artifacts (fields.npz incl. phi_hist, results.json)
- `out_graded/`, `out_uniform/` - first-demo 1200 s artifacts (mild law)
- `fig1_graded_all_axes.png` - quarter-cut 3-D dopant volume (strong law) + CAD wireframe
- `fig2_densification.png` - XZ mid-slice density + end-of-exposure temperature, melt front overlaid
- `fig3_graded_vs_uniform.png` - final density at 900 s side by side, melt heights annotated
- `fig4_layer_stack.png` - six per-layer dopant rasters (strong law), base to apex
- `fig5_what_changed.png` - delta Qrf / delta T / delta rho panels, graded minus uniform
