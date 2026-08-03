# EQS-02 impact: what the cross-interface Q_rf artifact does to heatr3d's thermal numbers

All numbers below are read from `results.json["eqs02_impact"]`, produced by
`run_eqs02_impact.py` (geo-prewarp venv). `heatr3d.py` was **not modified**: the
corrected heating field is injected through the existing validation hook
`run(qrf_override=...)`.

## The artifact

`heatr3d.compute_qrf_3d` (heatr3d.py:393-399) forms

```python
Ex, Ey, Ez = np.gradient(V, grid.h, edge_order=1)   # over the WHOLE domain
...
Q[~doped] = 0.0                                     # masked only AFTERWARDS
```

The gradient is taken before the part mask is applied, so the outermost in-part
voxel is centrally differenced against an **outside** voxel — across the
material interface, where `grad V` jumps by the conductivity contrast
(`sigma_doped/sigma_virgin = 4e6`). `|E|` in that skin is inflated and, because
`Q ~ |E|^2`, the inflation is squared. The subsequent `P_abs` renormalization
rescales the entire field down to hit the same fixed total power, so **no energy
is invented — energy is moved from the interior to the surface skin.**

## The correction

`Q` is recomputed from the **same** `V` and the **same** `gamma` with
`metrics.masked_grad_3d`, a stencil that never crosses the part boundary
(second-order central where both neighbours are in-part; one-sided — exact for a
linear field — where only one is). Every other step of the `Q` definition is
byte-identical to `compute_qrf_3d`, including the renormalization to
`p_target = power_density_w_per_m3 * doped_volume`. Recorded power identity:
`rel_diff` = 0.0 (circle) and 3.5e-16 (square), so shipped and corrected drives
carry **identical total absorbed power** and differ only in its distribution.

**2-D vs 3-D choice.** Both shapes are full-height extrusions, so `V` is exactly
z-invariant and a per-z-slice application of `masked_grad_2d` would suffice. The
volumetric `masked_grad_3d` was used instead because (a) the thermal march needs
a full 3-D drive, and (b) it lets the z-invariance be *measured* rather than
assumed: `max|Ez| / mean|E|` in the part = 6.5e-07 (circle), 1.3e-06 (square).
Equivalence of the two routes for a z-invariant field is pinned by
`test_masked_grad_3d_matches_2d_per_slice_for_a_z_invariant_field`
(`test_metrics.py`); the full metrics suite is 20/20 green.

**Setup.** n = 64, L = 60 mm, extruded circle d = 20 mm and extruded square
20 mm (both full height). "Surface" = within 1.5 voxels of the part boundary
(the Task-2 rule); interior = the rest. Thermal march: `phase_update="enthalpy"`
(via `dataclasses.replace`, `Params` is frozen), `phi_target = 0.90`.
Energy-residual fraction < 1e-13 and `clamp_bound = False` in all four marches.

## Corrected vs shipped: the heating field

| Q_rf (in-part) | circle shipped | circle corrected | square shipped | square corrected |
|---|---|---|---|---|
| max / mean | 12.25 | 1.86 | 19.15 | 2.20 |
| p99 / mean | 12.25 | 1.86 | 7.49 | 1.89 |
| CV (std/mean) | 2.116 | 0.180 | 2.108 | 0.290 |
| power fraction in surface band | 0.737 | 0.319 | 0.737 | 0.371 |
| (surface band volume fraction) | 0.311 | 0.311 | 0.331 | 0.331 |
| interior mean [W/m^3] | 6.086e5 | 1.574e6 | 6.262e5 | 1.494e6 |

The **interior pattern is unchanged** by the correction: unit-mean pattern
relative L2 (corrected vs shipped, interior points only) = 1.7e-16 (circle) and
2.0e-16 (square). The correction changes only (i) the surface-band values
(pattern L2 0.73 / 0.81) and (ii) the global scale that the renormalization then
assigns to the interior — the interior absolute mean rises by **2.59x** (circle)
and **2.39x** (square).

## Corrected vs shipped: the thermal answers

| Metric (3-D basis) | circle shipped | circle corrected | delta | square shipped | square corrected | delta |
|---|---|---|---|---|---|---|
| sigma_T [C] | 26.131 | 20.103 | **-6.03 (-23.1 %)** | 17.897 | 19.856 | **+1.96 (+10.9 %)** |
| t90 [s] | 397.00 | 323.35 | **-73.65 (-18.6 %)** | 327.70 | 316.05 | **-11.65 (-3.6 %)** |
| T_max [C] | 284.27 | 240.03 | -44.24 | 254.25 | 238.10 | -16.15 |
| T_mean over part [C] | 210.98 | 208.65 | -2.33 | 198.33 | 211.42 | +13.09 |
| interior mean T [C] | 208.32 | 219.62 | +11.30 | 192.38 | 221.51 | +29.14 |
| surface mean T [C] | 216.87 | 184.37 | -32.50 | 210.39 | 190.99 | -19.40 |
| surface - interior [C] | +8.55 | **-35.25** | -43.80 | +18.02 | **-30.52** | -48.54 |
| interior std T [C] | 17.33 | 12.74 | -4.59 | 10.13 | 13.02 | +2.89 |
| surface std T [C] | 38.46 | 9.13 | -29.33 | 23.32 | 15.02 | -8.30 |

`sigma_T` here is the **3-D** metric (std of `T_phi90` over part voxels,
`heatr3d.Result.sigma_T`). It is not comparable to the 2.5-D study's `ui_rms`
numbers; only shipped-vs-corrected within this table is like-for-like.

Two facts worth stating plainly:

1. **The sign of the sigma_T change is shape-dependent** — down 23 % for the
   circle, up 11 % for the square. The artifact is not a uniform bias that can
   be scaled out; it interacts with geometry.
2. **The thermal topology inverts.** Under the shipped drive the part surface is
   hotter than its interior (+8.6 C circle, +18.0 C square); under the corrected
   drive the interior is hotter than the surface (-35.3 C, -30.5 C). The
   published shape of the temperature field — where the hot region is — flips.

## Which classes of published heatr3d numbers are most exposed

Ordered by how directly the artifact enters them:

1. **Hot-spot / peak-ratio numbers** (max/mean, p99/mean, corner-concentration
   ratios, "Q_rf is Nx mean at the corners"). Most exposed. The shipped values
   are 6.6x (circle max/mean) and 8.7x (square max/mean) the corrected values,
   and the Task-3 refinement study shows the shipped corner max keeps growing
   with n (19.2 -> 29.1 -> 38.7 at n = 64/96/128) while the mask-confined
   recomputation grows slowly and smoothly (2.20 -> 2.71 -> 3.17). Any statement
   of the form "the field concentrates Nx at the surface/corner" carries the
   squared cross-interface jump inside N.

2. **Surface-vs-interior attribution of absorbed dose.** Directly exposed: 73.7 %
   of absorbed power sits in a 31-33 % volume band under the shipped Q, versus
   32-37 % (roughly volume-proportional) corrected. Statements about skin
   heating, surface-driven melt onset, or the part heating "from the outside in"
   rest on this split, and it reverses.

3. **sigma_T and uniformity metrics.** Exposed but not by a fixed factor:
   -23 % (circle) and +11 % (square). Anything that ranks or optimizes shapes,
   orientations, or FGM/prewarp designs by sigma_T is affected in a
   geometry-dependent way, so relative rankings cannot be assumed to survive.

4. **Timing / throughput numbers (t90, exposure).** Least exposed of the four,
   but not clean: -18.6 % (circle), -3.6 % (square). Total absorbed power is
   identical in both drives, so the shift is purely redistribution — the
   corrected drive puts power where it raises mean melt fraction faster.

5. **Not exposed:** total absorbed power and the energy balance (identical by
   construction, residual < 1e-13), and the in-part *interior* Q pattern
   (relative L2 1.7e-16 / 2.0e-16 — unchanged shape, changed scale).

## Scope and limits

- Single grid (n = 64) and two full-height extruded shapes. The n-dependence of
  the thermal deltas was not measured here; the Q-field n-dependence is in
  `results.json["task3"]`.
- The correction fixes only the *post-processing* gradient. It uses heatr3d's
  own `V` from the unchanged finite-volume EQS solve, so any error in `V` itself
  (harmonic face averaging at the staircase boundary, cell-centred electrode
  gauge) is still present in both columns.
- The corrected column is not claimed to be the true answer; it is the same
  discretization with the cross-interface stencil removed. The independent
  conforming-FEM answer is Task 2: over all in-part mid-plane points at
  the fine (n=96-matched) level, the conforming FEM differs from the
  mask-confined recomputation by 10.8 % in unit-mean pattern L2, but from the
  shipped Q by 90.4 % (`results.json["task2"]["gate"]`). In the interior the two
  heatr3d post-processings are identical (2.55 % vs FEM either way); the whole
  disagreement lives in the surface band.
