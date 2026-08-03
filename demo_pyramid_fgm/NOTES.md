# Pyramid FGM demo (quick visual, 2026-08-03)

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

## Grading law (heuristic, varies in x, y, AND z)
```
d_hat = depth / max depth          (Euclidean distance-to-surface inside the part)
z_hat = (z - z_base) / (z_apex - z_base)
sat   = clip(0.95 * (0.35 + 0.65 * d_hat) * (1 - 0.45 * z_hat), 0.20, 1.00)
```
Motivation: measured RF field concentration at exterior corners, sharp edges, and
the apex overheats those regions, so dopant is graded DOWN near the surface (which
contains all corners/edges) and DOWN toward the apex, higher in the core and base.
Implemented in `grading.py`; red-green tested in `test_grading_law.py` (bounds,
zero outside part, variation along every axis, core>skin, base>apex).

## Run settings (both arms identical except the map)
- `heatr3d.run(grid, part, Params(phase_update="enthalpy"), sat=..., max_time_s=1200,
  densify=True)`; masked Qrf gradient (solver default); OMP/OPENBLAS threads = 1.
- Graded arm: sat from the law above. Uniform arm: sat=None (full dopant everywhere).

## Results (out_graded / out_uniform, results.json)
| quantity | uniform | graded |
|---|---|---|
| wall time (s) | 329.2 | 329.2 |
| t at phi_bar=0.90 (s) | 1195.5 | 1116.8 |
| sigma_T = std(T) at phi=0.90 latch (C) | 29.234 | 29.436 |
| T_max (C) | 261.9 | 266.6 |
| sintered fraction (phi>=0.5) | 0.901 | 0.900 |
| std(rho_final) over part | 0.1404 | 0.1326 |
| energy standing gate (residual_frac) | +0.0000 | +0.0000 |

The graded arm melts sooner and ends with a slightly more uniform density field;
its std(T) at the latch is essentially unchanged. This demo shows the CAPABILITY
(a 3-axis graded map running through real physics), not a tuned benefit claim.

## Files
- `grading.py`, `test_grading_law.py` - the law + its tests
- `run_demo.py` - voxelize + solve one arm (`graded` | `uniform`)
- `render_figs.py` - deck-style renders (style: `deck_figures_3d/style3d.py`)
- `out_graded/`, `out_uniform/` - `fields.npz` (part, sat, T_phi90, phi_final,
  rho_final, Qrf, h, L) + `results.json`; `log_*.txt` - run logs
- `fig1_graded_all_axes.png` - quarter-cut 3-D dopant volume + CAD wireframe
- `fig2_densification.png` - XZ mid-slice density + temperature, melt front overlaid
- `fig3_graded_vs_uniform.png` - final density side by side, shared scale
- `fig4_layer_stack.png` - six per-layer dopant rasters, base to apex
