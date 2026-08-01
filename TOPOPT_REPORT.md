# Topology-optimization parameterization of the dopant solve: does fidelity transfer across grids?

Batch item 6 of the 2026-08-01 overnight queue, with the specification upgraded per the
three-dimensional port lane's converged diagnosis. Solve lane only; the heatr3d tool lane
was not touched, no dissertation file was edited, nothing was committed.

Acronyms on first use: RFAM radio-frequency additive manufacturing; FGM functionally
graded material; EQS electro-quasi-static; IoU intersection over union; bpp bits per
pixel; L-BFGS-B limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints;
MMA the method of moving asymptotes.

Reading conventions used throughout, stated once:
* **J** is the melt-region shape-fidelity objective summed over the WHOLE domain at the
  arm's own J-stop. In this pass its target is the grid-independent AREA-FILL chi, so
  **this pass's J is not numerically comparable to any J in `out_lib` or `out_ms`**.
  Every result file also stores `J_raster_chi`, the same map scored under the old binary
  target, which IS comparable, and that is the number used in every cross-pass table.
* **IoU** is always against the BINARY part mask at the stated grid, which is what every
  earlier report quotes. `IoU_area` is the area-weighted reading against the area-fill
  target and is not comparable to earlier reports.
* Every fidelity number carries its grid. Solves are at grid 120; hold-outs at grid 160.
* The stop rule is `argmin` over the arm's own trajectory; `at_horizon` is flagged and
  makes that arm's J an upper bound.
* Actuator: conductivity only, the deployable channel, on every arm.
* Arms are NOT dose matched. Absorbed power spread on the six deliverable arms:
  316.2 to 495.7 W per metre of depth.

---

## 1. Verdict, up front

**1. Yes, fidelity now transfers across grids, and this is the first pass in the campaign
where it does. But the projection is NOT what buys it: the physical filter radius and the
grid-independent target are.** COMPUTED. Solving at grid 120 and scoring at grid 160 with
the drive recalibrated there, the topology-optimization continuation arm **improves or
holds on four of six shapes** (circle 0.9190 to 0.9616, triangle 0.8538 to 0.8694,
diamond 0.8115 to 0.8435, rectangle 0.8558 to 0.8879) and loses on two (square 0.8734 to
0.7833, trapezoid 0.9308 to 0.9040). For contrast, `SOLVE_ROBUSTNESS_VALIDATION.md`
Section 3.2 found that none of the three shapes in the SOLVED class at 120 stayed there
at 160, and `MULTISTART_REPORT.md` Section 6.2 got the filtered square only from 0.7767
to 0.8063.

**2. The square, the shape the queue named, is still not fixed, and most of its remaining
drop is the forward and not the map.** COMPUTED. The best square arm this pass produced
reaches **IoU 0.8202 at grid 160** (the beta = 0 control, design transfer), against
0.8063 for the filtered multi-start and 0.7767 for the unfiltered single start. That is
the best square yet at 160 and it is still not the IoU >= 0.95 class. The **uniform arm
itself falls from 0.8528 at 120 to 0.7768 at 160 on this shape**, so 0.076 IoU points of
any square drop is forward discretization that no map can fix. The queue asked for an
honest report if the full fix did not close most of the gap: on the square it did not.

**3. Exactly one shape reaches the SOLVED class at grid 160, and it does so on both
arms.** COMPUTED: the **circle**, IoU **0.9616** (continuation, design transfer) and
**0.9855** (beta = 0 control, map transfer), both above the 0.95 threshold. No shape in
any earlier pass reached that class at 160.

**4. No map earns the SOLVED label under the port lane's rule, because Gate B is not
clean on the shapes that pass Gate A.** COMPUTED. The circle continuation arm passes
Gate A and both pre-registered forms of Gate B (map space 8.98 percent, design space
5.40 percent, both inside the 10 percent tolerance) but fails the dose-matched form this
pass added afterwards at **11.41 percent**. The beta = 0 control reaches a higher IoU at
160 on the circle but fails Gate B catastrophically there, **264.7 percent**. Under a
strict reading of "both gates before any SOLVED label", **zero of six**.

**5. Gate B is the clearest win in the pass, and it is the projection that delivers it.**
COMPUTED, at a fixed one-cell (0.50 mm) blur of the delivered map, absolute change in J:

| shape | unfiltered single start | filtered multi-start, 0.76 mm | 1.00 mm radius, NO projection | 1.00 mm radius WITH projection |
|---|---|---|---|---|
| square | +152.7 % | +9.9 % | +10.09 % | **-1.44 %** |
| circle | +891.9 % | not run | +45.04 % | **-8.98 %** |
| trapezoid | +171.2 % | not run | +4.59 % | +7.67 % |
| triangle | +37.3 % | not run | +0.01 % | **-2.62 %** |
| diamond | +115.2 % | not run | +5.18 % | **-1.14 %** |
| rectangle | -20.2 % | +1.6 % | +0.23 % | **-1.43 %** |

Across the full sub-radius sweep the projection arm passes Gate B on **five of six**
shapes in map space and **five of six** in design space, against **three of six** for the
same radius without the projection.

**6. And the projection costs a great deal of in-grid fidelity at this budget.** COMPUTED.
At the same 40 forward-equivalents, the same 1.00 mm radius and the same target, the
beta = 0 control reaches **IoU 1.0000 on the square and 0.9968 on the circle at grid
120**, against 0.8734 and 0.9190 for the continuation. Scored under the OLD binary target
so it is directly comparable, the control beats the single-start library campaign's
deliverable on five of six shapes (square 18.88 against 25.66, circle 13.07 against 14.68,
trapezoid 26.08 against 27.36, triangle 81.79 against 87.99, rectangle 188.67 against
196.94) and essentially ties on the diamond (214.84 against 211.16).

**7. The grid-independent target does what it was built to do, and it removes one of the
two confounds the earlier hold-outs could not separate.** COMPUTED: between grid 120 and
grid 160 the area-fill target area moves **-0.105 percent** on the square while the
binary raster target moves **-1.660 percent**. The map is now scored against the same
physical target at both grids.

**8. The composed gradient is finite-difference gated and the chain rule is exact.**
COMPUTED. The transpose identity holds to **4.19e-16 (square)** and **2.14e-16 (circle)**,
worst over beta 0, 1 and 16. On the layer the solve's final stage actually uses,
beta = 16, the square passes **5 of 5** probes at the campaign's 1e-5 subgradient standard
and 3 of 5 at 1e-6; the circle passes 4 of 5 at both, its single miss being a probe whose
own analytic derivative is 0.047. The gate does NOT clear 1e-5 on every probe of every
layer, and Section 3 shows why that is a denominator effect and not a wrong gradient.

**One-sentence answer to the queue's headline question**: the full fix makes fidelity
transfer on four of six shapes and puts one shape in the SOLVED class at grid 160 for the
first time, but the transfer comes from the physical filter radius plus the
grid-independent target rather than from the smoothed-Heaviside projection, and the
square still does not close.

---

## 2. What was built

All new code under `fgm_solve_campaign/adjoint2d/`. Nothing existing was modified;
`forward.py`, `adjoint.py`, `shape_objective.py`, `design_filter.py`, `robust.py`,
`printability.py`, `library_solve.py` and `ms_solve.py` were read and imported.

| file | what it is |
|---|---|
| `topopt.py` | the parameterization: physical-radius filter, tanh projection, chain-rule VJP and JVP, the frozen beta schedule, the stage split |
| `chi_area.py` | the grid-independent target: band-restricted sub-cell area fill, bit-identical to the production sampler |
| `topopt_objective.py` | the melt objective with chi as an ARGUMENT, bit-identical to `shape_objective` when handed the raster back |
| `gate_topopt.py` | the four-layer finite-difference gate, transpose bisect, read-state stability |
| `topopt_solve.py` | the per-shape continuation solve and the beta = 0 control |
| `topopt_robust.py` | Gate A grid hold-out, Gate B in map space and design space |
| `topopt_dosematch.py` | the dose-matched repeat of Gate B |
| `build_topopt_tables.py`, `make_topopt_figures.py` | tables and the five figures |
| `tests/test_topopt.py`, `tests/test_chi_area.py`, `tests/test_topopt_objective.py` | 38 tests, red first |

Shell drivers: `run_topopt_gate.sh`, `run_topopt_stream.sh`, `run_topopt_robust.sh`.

### 2.1 Red first, and the red that was observed

The red state was an `ImportError` on `adjoint2d.topopt`, `adjoint2d.chi_area` and
`adjoint2d.topopt_objective`, observed before each module existed. **224 tests pass**
across the whole `adjoint2d` suite, against a measured **186** before this pass.

Two tests are worth naming because they are the ones that would catch a real error:

* `test_design_vjp_is_the_exact_transpose_of_the_linearized_map` checks
  `<dJ/dv, d> = <dJ/ds, dS(v)[d]>` on a non-convex mask with an interior hole. It holds to
  better than 1e-12 relative.
* `test_band_restriction_is_bit_identical_to_the_production_sampler` checks the fast
  area-fill path against the brute-force production routine by `np.array_equal`, on a
  circle and on a grid-aligned square, so the speed-up cannot have become a second
  slightly different convention.

### 2.2 One real bug, found and fixed, with a regression test

The first run of all six solves wrote every result to `out_topopt/multistart.json`. The
cause: `topopt_solve.main` held the output stem in a local named `tag`, and a later
reference loop `for tag, path in (("library", ...), ("multistart", ...))` rebound it.
The fix is a pure function `topopt_solve.output_tag(shape, control)` recomputed at write
time, so the stem cannot be shadowed, plus two tests. The six solves were re-run from
scratch. This is recorded because the failure mode is silent: five of six results were
overwritten and every log line still looked correct until the last one.

---

## 3. The finite-difference gate, run BEFORE any optimization

Two shapes, square and circle. Four layers, each adding exactly one thing.
Five probes per layer: the maximum-sensitivity cell, a fixed pseudo-random in-part cell,
a random unit direction, a filter-smooth random direction and the gradient direction.
Epsilon swept over eight values, 1e-3 down to 1e-8. Read at a FIXED index, the argmin of
J on the base run.

The gate design point is NOT `gate_rho.default_v`. That point sits around 0.80, which at
beta = 16 is on the projection's upper rail where dP/du is 1.8e-06 and the gradient is
numerically dead: gating there would be gating nothing. `gate_topopt.gate_v0` centres the
design on the threshold eta = 0.5. MEASURED at that point, beta = 16, square: dP/du spans
**0.094 to 6.54** with a median of 0.77 and **zero** cells below 1e-3 of the peak.

### 3.1 Results

| shape | layer | at 1e-6 | at 1e-5 | worst probe | its analytic derivative |
|---|---|---|---|---|---|
| square | P1 target only, dJ/ds | 1/5 | **5/5** | random_direction 7.86e-06 | 0.873 |
| square | P2 add the filter, dJ/dv | 1/5 | 3/5 | random_direction 1.09e-04 | 0.483 |
| square | P3 add projection, beta 1 | 1/5 | 3/5 | smooth_random_direction 1.01e-04 | 0.314 |
| square | **P4 projection, beta 16** | 3/5 | **5/5** | random_cell 8.43e-06 | 0.252 |
| circle | P1 target only, dJ/ds | 1/5 | **5/5** | random_direction 9.33e-06 | 0.260 |
| circle | P2 add the filter, dJ/dv | 3/5 | 4/5 | random_cell 3.11e-05 | 0.106 |
| circle | P3 add projection, beta 1 | 2/5 | **5/5** | smooth_random_direction 4.09e-06 | 5.797 |
| circle | **P4 projection, beta 16** | 4/5 | 4/5 | random_cell 1.16e-05 | 0.047 |

P4 is the layer the solve's final and sharpest stage actually uses. It is also the
CLEANEST layer on both shapes, because the projection amplifies the analytic derivative
(square gradient norm 296 at beta 16 against 33 at beta 0).

### 3.2 The bisect, and the residual is not the chain rule

**Transpose exactness, the dot-product identity `<dJ/dv, d> = <dJ/ds, dS(v)[d]>` against
the REAL adjoint**, at three betas and four probe directions each:

| shape | beta 0 | beta 1 | beta 16 |
|---|---|---|---|
| square | 3.24e-16 | 4.19e-16 | 2.31e-16 |
| circle | 1.31e-16 | 2.14e-16 | 1.63e-16 |

PROVEN to machine precision. The filtered and projected analytic gradient is EXACTLY the
unfiltered `dJ/ds` composed with a proven-exact linear operator, so any remaining
finite-difference disagreement belongs to the forward.

**The denominator diagnosis, measured rather than argued.** Across all 40 probes on both
shapes and all four layers, the best ABSOLUTE finite-difference error lies between
**4.09e-07 and 1.25e-04**, essentially independent of layer, while the analytic
directional derivative spans **0.047 to 296**. Every probe that missed the relative
standard is a probe whose own derivative is small. Panel at the bottom of
`figs_topopt/fig_topopt_gate.png` shows the scatter with the 1e-6 and 1e-5 relative lines
as diagonals; the points lie in a flat horizontal band, not along either diagonal.

Mechanism for the two worst probes: the filter damps a rough random direction, so the
directional derivative shrinks while the objective's roundoff floor does not. The
measured evaluation floor of J on this pass is **3.17e-12 to 1.04e-10 absolute**.

### 3.3 The read state does not move

`stop_index_stability` at beta 16, epsilon 1e-3, over three probe directions: the argmin
index does **not** move on either shape (square base 900, circle base 792). The envelope
argument that removes the `dt*/ds` term is therefore exactly valid over the tested range
rather than assumed.

### 3.4 Gate verdict

**PASS at the campaign's 1e-5 subgradient standard on the layer the solve uses (P4:
square 5/5, circle 4/5 with a single 1.16e-05 miss on a probe of magnitude 0.047), with
the chain rule PROVEN exact to 4.19e-16.** The composed gradient is FD-gated.
It does NOT pass 1e-5 on all 40 probes, 34 of 40 do, and the six misses are localized to
small-derivative directions on the two intermediate layers. Honest label: **subgradient
PASS with named exceptions**, not a clean 1e-6 gate.

---

## 4. The parameterization, and why each piece is what it is

See `figs_topopt/fig_topopt_parameterization.png`, all four panels.

### 4.1 The filter radius is a length, 1.0 mm, and the printer scale is not binding

The queue asked for the reasoning to be stated, so it is stated in full.

The printer's dopant edge scale of roughly 50 to 100 micrometres is **finer than the cell
size of any grid the solve runs on**: 504 micrometres at grid 120, 377 micrometres at
grid 160. A constraint finer than the mesh cannot restrict what the solve is able to
express, so it is NOT the binding scale and must not be used as the radius.

The binding scale is solver convergence. `MULTISTART_REPORT.md` Section 6 measured that a
1.5-cell radius (0.756 mm at 120) leaves the square going from IoU 0.9681 in grid to
0.8063 across the hold-out. The cell-count evidence says at least two cells at grid 120,
which is 1.0 mm, and 1.0 mm is also 2.65 cells at grid 160, so the same physical feature
is representable on both grids with margin. **Frozen at 1.0 mm. NOT swept.**

### 4.2 The target: measured, not asserted

| shape | area-fill target, mm2 | binary raster target, mm2 | raster over-statement |
|---|---|---|---|
| square | 400.420 | 406.751 | +1.58 % |
| circle | 314.155 | 315.232 | +0.34 % |
| trapezoid | 300.158 | 302.267 | +0.70 % |
| triangle | 200.210 | 203.375 | +1.58 % |
| diamond | 399.952 | 412.852 | +3.23 % |
| rectangle | 288.303 | 292.861 | +1.58 % |

The nominal square is 20 mm by 20 mm, 400.000 mm2; the area fill gives 400.420, a
+0.105 percent residual from the polygon edge falling between cell centres at this grid.

Across grids, `out_topopt/<shape>_robust.json` key `chi_area_grid_consistency`:
the area-fill target moves **-0.105 percent (square)** and **-0.052 percent (trapezoid)**
from grid 120 to grid 160; the binary raster moves **-1.660 percent (square)** and
**+0.016 percent (trapezoid)**.

**The delta against the raster target on one shape, as the queue asked.** On the square
the two targets differ on **156 cells by more than 0.01**, the largest single-cell
disagreement is **0.288**, and the per-cell absolute differences sum to **24.90 cells**,
which is the +1.58 percent (6.331 mm2) area over-statement of the raster.
Scored under the two targets, the same delivered square map gives
J = 154.91 (area fill) against J_raster_chi = 165.72, a **7.0 percent** difference, and at
the gate design point the J-stop index moves by 10 steps, 1214 against 1224.

### 4.3 The continuation, and what it does to the design

`figs_topopt/fig_topopt_continuation.png`. Non-discreteness `M_nd = mean(4 s (1 - s))`
over the part, 0 for a binary map:

| shape | M_nd, continuation deliverable | M_nd, beta = 0 control |
|---|---|---|
| square | 0.151 | 0.651 |
| circle | 0.026 | 0.690 |
| trapezoid | 0.056 | 0.549 |
| triangle | 0.093 | 0.479 |
| diamond | 0.087 | 0.465 |
| rectangle | 0.068 | 0.323 |

The projection does what it is supposed to do: the delivered maps are near binary. The
figure also shows the cost mechanism plainly, and it is an optimizer artifact as much as
a physics one. With 3 evaluations per beta stage, L-BFGS-B spends most of a stage on
line-search trial points that are worse than the incumbent, and the curvature memory is
discarded at each restart. Each stage keeps its own best iterate so nothing bad is
carried forward, but the effective depth per stage is one or two useful steps.

---

## 5. Results at grid 120

| shape | pool | uniform J | uniform IoU | TO_cont J | TO_4bpp J | TO_4bpp IoU | IoU_area | M_nd | P_abs W/m | stop s | horizon |
|---|---|---|---|---|---|---|---|---|---|---|---|
| square | 14 | 197.23 | 0.8528 | 154.96 | 154.91 | 0.8734 | 0.8583 | 0.151 | 472.6 | 473.5 | no |
| circle | 14 | 256.67 | 0.7853 | 44.50 | 45.13 | 0.9190 | 0.9370 | 0.026 | 371.2 | 557.5 | no |
| trapezoid | 15 | 158.82 | 0.8387 | 60.62 | 61.35 | 0.9308 | 0.9171 | 0.056 | 421.2 | 437.5 | no |
| triangle | 16 | 190.54 | 0.7666 | 81.71 | 81.92 | 0.8538 | 0.8394 | 0.093 | 405.0 | 297.0 | no |
| diamond | 15 | 743.00 | 0.5781 | 243.40 | 243.96 | 0.8115 | 0.8124 | 0.087 | 316.2 | 750.0 | **YES** |
| rectangle | 15 | 197.25 | 0.8528 | 189.34 | 189.50 | 0.8558 | 0.7385 | 0.068 | 495.7 | 325.5 | no |

**One horizon flag**: the diamond continuation arm stops at 750.0 s, the last stored step,
so its J of 243.96 is an upper bound. Reported in `stop_at_horizon_arms`.

Quantization to 4 bpp costs almost nothing here (J moves by at most 0.63 on any shape),
which is expected for a near-binary map.

### 5.1 The control that separates the projection from everything else

`MULTISTART_REPORT.md` Section 9 limit 2 could not separate its filter from its
multi-start. This pass runs the separating control: beta = 0, the SAME 1.0 mm radius, the
SAME area-fill target, the SAME 40 forward-equivalents, the same cold start.

| shape | continuation J | control J (beta 0) | continuation IoU | control IoU | continuation M_nd | control M_nd |
|---|---|---|---|---|---|---|
| square | 154.91 | **12.30** | 0.8734 | **1.0000** | 0.151 | 0.651 |
| circle | 45.13 | **2.50** | 0.9190 | **0.9968** | 0.026 | 0.690 |
| trapezoid | 61.35 | **15.82** | 0.9308 | **0.9735** | 0.056 | 0.549 |
| triangle | 81.92 | **69.93** | 0.8538 | **0.8815** | 0.093 | 0.479 |
| diamond | 243.96 | **187.53** | 0.8115 | **0.8501** | 0.087 | 0.465 |
| rectangle | 189.50 | **182.40** | 0.8558 | **0.8580** | 0.068 | 0.323 |

The control wins in grid on all six. That is the honest attribution: at this budget the
projection is a robustness purchase paid for with in-grid fidelity, not a free win.

### 5.2 Against the earlier passes, on the comparable metric

J here is `J_raster_chi`, the same map scored under the old binary target, so these
columns are like for like.

| shape | uniform IoU | out_lib A1_4bpp IoU | out_ms MS_4bpp IoU | control IoU | continuation IoU | control J_raster | continuation J_raster | out_lib J | out_ms J |
|---|---|---|---|---|---|---|---|---|---|
| square | 0.8528 | 0.9816 | 0.9681 | **1.0000** | 0.8734 | **18.88** | 165.72 | 25.66 | 35.56 |
| circle | 0.7853 | 0.9904 | 0.9872 | **0.9968** | 0.9190 | **13.07** | 69.05 | 14.68 | 15.55 |
| trapezoid | 0.8387 | 0.9718 | 0.9693 | **0.9735** | 0.9308 | **26.08** | 72.16 | 27.36 | 29.27 |
| triangle | 0.7666 | 0.8578 | 0.8783 | **0.8815** | 0.8538 | **81.79** | 94.45 | 87.99 | 82.29 |
| diamond | 0.5781 | 0.8520 | 0.8161 | 0.8501 | 0.8115 | 214.84 | 282.01 | **211.16** | 272.11 |
| rectangle | 0.8528 | 0.8424 | 0.8842 | 0.8580 | 0.8558 | 188.67 | 194.86 | 196.94 | **97.19** |

**COMPUTED: the beta = 0 control at the 1.0 mm physical radius is the best arm the
campaign has produced at grid 120 on four of six shapes** (square, circle, trapezoid,
triangle) on both IoU and the comparable J, losing the diamond to the library single
start by 1.7 percent and the rectangle to the multi-start by a factor of 1.9. The
rectangle loss is the known rectangle stall; the multi-start's warm start is what beats
it there and this pass ran a single cold start only.

---

## 6. Acceptance gate A: the grid hold-out

Solve at 120, score at 160, drive recalibrated at 160 so the uniform arm absorbs 500 W
per metre, target chi rebuilt from the geometry at 160. Both transfer routes reported.

| shape | arm | IoU 120 | IoU 160 map transfer | IoU 160 design transfer | uniform IoU 160 | drop, best route | SOLVED at 160 |
|---|---|---|---|---|---|---|---|
| square | continuation | 0.8734 | 0.7833 | 0.7771 | 0.7768 | +0.0901 | no |
| square | control | 1.0000 | 0.7999 | **0.8202** | 0.7768 | +0.1798 | no |
| circle | continuation | 0.9190 | 0.9484 | **0.9616** | 0.8950 | **-0.0426** | **YES** |
| circle | control | 0.9968 | **0.9855** | 0.9695 | 0.8950 | +0.0113 | **YES** |
| trapezoid | continuation | 0.9308 | **0.9040** | 0.8983 | 0.8925 | +0.0268 | no |
| trapezoid | control | 0.9735 | **0.9104** | 0.9098 | 0.8925 | +0.0631 | no |
| triangle | continuation | 0.8538 | 0.8679 | **0.8694** | 0.7693 | **-0.0156** | no |
| triangle | control | 0.8815 | 0.8653 | **0.8659** | 0.7693 | +0.0156 | no |
| diamond | continuation | 0.8115 | **0.8435** | 0.8430 | 0.6974 | **-0.0320** | no |
| diamond | control | 0.8501 | 0.8605 | **0.8606** | 0.6974 | **-0.0105** | no |
| rectangle | continuation | 0.8558 | 0.8877 | **0.8879** | 0.8848 | **-0.0321** | no |
| rectangle | control | 0.8580 | 0.8891 | 0.8891 | 0.8848 | **-0.0311** | no |

Recalibrated voltages at grid 160: square 2834.8, circle 3783.7, trapezoid 2957.0,
triangle 3191.5, diamond 2919.3, rectangle 3588.8 volts.

**Three readings, all COMPUTED.**

1. **Absolute fidelity now transfers on four of six shapes for the continuation arm and
   three of six for the control.** Negative drops mean the arm got BETTER at 160. That is
   new: `SOLVE_ROBUSTNESS_VALIDATION.md` Section 3.2 had every solved shape falling out
   of its class.
2. **The design transfer buys essentially nothing over the production map transfer.** The
   two routes agree to within **1.5 percent of J on every shape and both arms**, largest
   gap trapezoid continuation 180.43 against 186.58. That is worth knowing for the port:
   the printer pipeline's bilinear resample is not the bottleneck, so a design-space
   transfer is not required.
3. **The uniform arm's own move still bounds the interpretation, but it no longer hides
   the target.** The uniform arm moves from -0.0760 (square) to +0.1193 (diamond) between
   grids, so the forward is still not IoU converged. What has changed is that the TARGET
   no longer moves: chi is rebuilt from the geometry at each grid and it shifts by
   0.105 percent where the raster shifted by 1.660 percent. One of the two confounds is
   removed; the forward-convergence one is not, and it is still the reason the square's
   verdict cannot be clean.

---

## 7. Acceptance gate B: sub-filter-radius perturbation

Blur the delivered map (or, in the second form, the design variable) by a part-masked
normalized-convolution Gaussian at 0.5, 1.0 and 1.5 cells, all strictly below the
1.983-cell filter radius, and re-score. Tolerance 10 percent change in J. The 2.0-cell
radius sits AT the filter length and is reported as context, not gated.

### 7.1 The continuation arm, three forms of the gate

| shape | 0.25 mm | 0.50 mm | 0.76 mm | 1.01 mm (context) | max sub-radius | map space | design space | dose matched |
|---|---|---|---|---|---|---|---|---|
| square | +0.32 % | -1.44 % | +1.51 % | +2.13 % | 1.51 % | PASS | PASS 6.49 % | PASS 2.93 % |
| circle | -4.23 % | -8.98 % | -3.98 % | +66.38 % | 8.98 % | PASS | PASS 5.40 % | **FAIL 11.41 %** |
| trapezoid | +0.22 % | +7.67 % | +25.40 % | +46.86 % | 25.40 % | **FAIL** | **FAIL 52.95 %** | **FAIL 14.79 %** |
| triangle | -1.55 % | -2.62 % | -4.62 % | -3.70 % | 4.62 % | PASS | PASS 0.79 % | PASS 5.73 % |
| diamond | -0.61 % | -1.14 % | +1.67 % | +6.65 % | 1.67 % | PASS | PASS 1.31 % | PASS 3.36 % |
| rectangle | -0.40 % | -1.43 % | -2.35 % | -2.82 % | 2.35 % | PASS | PASS 0.79 % | PASS 2.74 % |

### 7.2 The same gate on the beta = 0 control, which is the attribution

| shape | control, map space | control, design space | continuation, map space |
|---|---|---|---|
| square | **FAIL 24.65 %** | **FAIL 23.18 %** | PASS 1.51 % |
| circle | **FAIL 264.71 %** | **FAIL 227.12 %** | PASS 8.98 % |
| trapezoid | **FAIL 26.49 %** | **FAIL 26.59 %** | FAIL 25.40 % |
| triangle | PASS 3.25 % | PASS 2.30 % | PASS 4.62 % |
| diamond | PASS 8.99 % | PASS 8.31 % | PASS 1.67 % |
| rectangle | PASS 0.67 % | PASS 0.67 % | PASS 2.35 % |

**COMPUTED: the projection is what removes the sub-radius sensitivity.** Three of six for
the control against five of six for the continuation, and on the two shapes where the
control is most spectacular in grid it is also most fragile: the circle's control reaches
IoU 0.9968 at 120 and then loses **264.7 percent of J** to a blur the filter radius was
supposed to have made irrelevant. That is the classic signature of a solution that lives
in the grey-scale texture, and `M_nd = 0.690` on that map says the same thing.

### 7.3 Why the second and third forms of Gate B exist, and what they found

Gate B as specified blurs the DELIVERED map at the PINNED voltage. Two confounds were
found in it and both are measured rather than argued.

* **It also tests crispness.** After a beta = 16 projection the map is nearly binary, so
  part of what a blur does is move the 0.5 level set rather than remove sub-radius
  structure. The design-space form blurs `v` and re-projects. It changed the verdict on
  no shape but moved the numbers substantially (trapezoid 25.40 to 52.95 percent, square
  1.51 to 6.49 percent).
* **It also changes the dose.** A part-masked blur of a non-uniform map changes absorbed
  power: MEASURED at 1.5 cells, +0.35 percent (rectangle) to +7.97 percent (trapezoid),
  even though the mean saturation moves by at most 0.33 percent, so it is a field effect
  and not a mean shift. The dose-matched form removes it exactly with one rescale. It
  cut the trapezoid failure from 25.40 to **14.79 percent**, so roughly 40 percent of the
  trapezoid failure was dose, and it pushed the circle from 8.98 to **11.41 percent**,
  turning a pass into a fail.

**Recommendation to the port lane, stated as a recommendation and not a result: run Gate
B dose matched.** It is one extra forward run per blurred arm and it removes a confound
that changed a verdict on this pass.

### 7.4 The trapezoid failure, diagnosed as far as it was

The trapezoid fails all three forms. Two hypotheses were tested and both were rejected:

* **Near-threshold population.** The fraction of in-part cells with the filtered design
  within 0.05 of eta is **5.05 percent** on the trapezoid, LOWER than the square's 14.50
  percent, which passes. Rejected.
* **Dose.** Removing it exactly leaves 14.79 percent, still a failure. Rejected as the
  sole cause.

What is left, and it is stated as UNTESTED: the trapezoid's base J is the second smallest
of the six at 61.35, and under blur its melt grows into the bed monotonically
(3.28 to 6.73 percent of the part area) while under-melt barely moves. That pattern is a
solution sitting close to a growth cliff, where a relative tolerance on a small J is a
harsh test. No experiment isolating that was run.

---

## 8. Cost and wall time

**Logged after two shapes and projected, as required.** The first two solves to complete
were triangle at 278 s and rectangle at 291 s, mean 285 s, which projected to six shapes
in six parallel streams as one round of about 300 s, plus the same again for the
beta = 0 control and about 350 s for each robustness run: roughly 20 minutes of wall
clock for the whole solve and gate campaign. **Actual: 11:00 to 11:24, 24 minutes**, plus
the two finite-difference gates which ran concurrently from 10:40 and took 3021 s
(square) and 2623 s (circle). The projection held within 20 percent and nothing was cut.

| stage | runs | process time |
|---|---|---|
| finite-difference gate, square and circle, one thread each, in parallel | 2 | 3021 s and 2623 s |
| six continuation solves | 6 | 2140 s |
| six beta = 0 controls | 6 | 3801 s |
| six robustness runs on the continuation arm, both gates | 6 | 1920 s |
| six robustness runs on the beta = 0 control, both gates | 6 | 932 s |
| six dose-matched repeats | 6 | 172 s |
| figures, six extra forward runs for the melt panels | 1 | about 120 s |
| full test suite | 1 | 538 s |

Total about **3.9 hours of process time**, about **55 minutes of wall clock** on 12 cores
with every numerical library pinned to one thread (`env1.sh`). One first attempt at the
six solves was discarded because of the output-stem bug in Section 2.2, costing 8 minutes.

Per-shape solve wall: 278 to 453 s for the continuation, 459 to 845 s for the control
(the control spends its whole pool in one stage, so L-BFGS-B keeps stepping rather than
restarting, and its steps land in the slower part of the trajectory).

---

## 9. Gates and flags

* **Energy-residual gate**: `adjoint2d.energy_gate`, 5 percent of integrated dose at each
  arm's own stop. **Zero violations** on every scored forward run in this pass:
  6 solves times 3 arms, 6 controls times 3 arms, 12 robustness runs times 9 arms and 18
  dose-matched arms.
* **Horizon flags**: one, the diamond continuation arm at 750.0 s. Its J is an upper
  bound.
* **Clip diagnostics** (`frac_dT_clipped_max`, `frac_temp_cap_max`, `frac_qrf_cap`) are
  recorded per arm in every result file.
* **Gradient health**: not separately asserted; the objective's argmin is interior and
  the read state was measured not to move, which is the relevant check here.

---

## 10. Proven, computed, assumed

**PROVEN**
* The chain-rule gradient is exactly the unfiltered `dJ/ds` composed with the transpose
  of the linearized parameterization: dot-product identity to 4.19e-16 worst, against the
  real adjoint, at beta 0, 1 and 16, on two shapes.
* The area-fill target is bit-identical to the production supersampler
  (`np.array_equal`) on a circle and on a grid-aligned square, so the band-restricted
  fast path is the same convention and not a second one.
* The composed map at beta 0 is bit-identical to the existing `design_filter.apply_filter`
  path, so the new channel cannot perturb the previously gated filtered arm.
* The chi-parameterized objective handed the binary raster reproduces
  `shape_objective.shape_J_and_seed` bitwise, in both the value and the adjoint seed.
* The projection fixes 0, 0.5 and 1 exactly, is strictly monotone, and maps [0, 1] onto
  [0, 1], so no clip subgradient enters the chain anywhere.
* 224 tests pass; the 38 new ones were red first, the red being an observed `ImportError`.

**COMPUTED**
* Every number in Sections 1, 3 through 9.
* The finite-difference evaluation floor of J, 3.17e-12 to 1.04e-10 absolute.
* The J argmin does not move under any probe perturbation at epsilon 1e-3.
* The area-fill target moves 0.105 percent between grids where the binary raster moves
  1.660 percent (square).
* Zero energy-residual gate violations.

**ASSUMED, and how each one bites**
1. **That 1.0 mm is the right filter radius.** Derived from a cell-count argument and a
   single prior measurement, NOT swept. Section 5.1 shows the radius choice is load
   bearing: at the same radius, adding the projection costs the square 0.13 IoU points in
   grid and buys a 16-fold Gate B improvement.
2. **That beta = (1, 2, 4, 8, 16) with an even budget split is the right schedule.** A
   convention. With a 14 to 16 evaluation pool each stage gets 3, which Section 4.3 shows
   is one or two useful L-BFGS-B steps. The continuation may simply be under-resourced
   rather than wrong, and that is NOT distinguished by anything measured here.
3. **That L-BFGS-B is an adequate substitute for MMA.** MMA is the field standard and was
   not implemented. The line-search waste in `fig_topopt_continuation.png` is the visible
   cost.
4. **That the 10 percent Gate B tolerance is the right threshold.** Inherited from the
   port lane's specification, not derived. On a shape with a small base J it is a harsh
   test; see the trapezoid.
5. **That a Gaussian blur of the delivered map models the physical rim uncertainty.** No
   bench measurement of the real rim blur exists. Carried over from
   `SOLVE_ROBUSTNESS_VALIDATION.md`, which names both directions of the error.
6. **That the forward's raster-based early stop does not truncate the area-fill argmin.**
   Supported by measurement (10 steps and 1 step against a 250-step patience), not proven.
7. **The arms are conductivity only; every historical arm co-varies permittivity.** The
   standing actuator gap.
8. **The forward is the two-dimensional `adjoint2d` engine, not heatr3d**, with every
   simplification documented for the shape-library solve.
9. **No experimental validation.** `ALLISON_LAW_REPLICATION.md` Section 6.1 records that
   the model over-predicts achievable tuned uniformity by roughly a factor of eight
   against hardware.

---

## 11. Honest limits

1. **The square is not fixed.** The headline shape of the queue item goes from 0.8734 at
   120 to 0.7833 at 160 on the continuation arm and 1.0000 to 0.8202 on the control. The
   0.8202 is the best square at 160 the campaign has produced, and it is still 0.13 IoU
   points short of the SOLVED class.
2. **No map passes both gates in all forms, so nothing is labelled SOLVED.** The closest
   is the circle continuation arm, which passes Gate A and two of three forms of Gate B.
3. **The projection's net value is not established.** It wins Gate B and loses in-grid
   fidelity, on the same six shapes, at the same budget. Which matters more depends on
   whether the rim blur model is right, and Section 10 assumption 5 says it is not
   measured.
4. **Two shapes gated, not six.** The other four solves inherit the square and circle
   gates.
5. **Six shapes, not eighteen.** The robustness-study six, as specified. Nothing here
   says what the other twelve do.
6. **Single cold start.** No warm start, no multi-start. The rectangle's known stall is
   therefore present and unaddressed: 189.50 against the multi-start's 97.19.
7. **The budget is 40 forward-equivalents, 14 to 16 gradient evaluations on 800 to 1624
   design variables.** Every J is an upper bound and the continuation is the arm most
   likely to be budget limited, because it spends the pool five ways.
8. **No arm is dose matched** except inside the dose-matched Gate B repeat. Spread 316.2
   to 495.7 W per metre.
9. **Everything is grid 120 except Section 6.** The forward is still not IoU converged
   between 120 and 160: the uniform arm alone moves up to 0.1193 IoU points, and in both
   directions. A re-solve AT 160 is still the only experiment that fully separates
   transfer from convergence, and it was not run.
10. **The area-fill target is an area fill, not a signed distance.** It cannot support a
    level-set velocity if that is where the port goes.

---

## 12. The single most valuable next layer

**Re-solve at grid 160 with the frozen conventions, on the square and the circle.** It is
two solves of about 900 s each and it is the last experiment that can separate map
transfer from forward discretization convergence. Every report in this workstream since
`SOLVE_ROBUSTNESS_VALIDATION.md` has named it and none has run it. With the target now
grid independent, the experiment is finally clean: if the square solved AT 160 reaches
IoU 0.95 there, then the 0.8202 hold-out number is a transfer failure and the
parameterization needs more work; if it does not, the ceiling is the forward and no
amount of design work will move it.

**Second: sweep the filter radius at 0.75, 1.0 and 1.5 mm on three shapes.** It is nine
solves, about 45 minutes in six streams, and it is the largest unswept convention in
`FROZEN_CONVENTIONS_2D.md`. Both prior reports asked for it.

**Third: give the continuation a doubled budget and re-run the control at the same
doubled budget.** Section 5.1's gap between the projection arm and the beta = 0 control
may be a budget artifact of splitting the pool five ways, and nothing measured here
separates that from a real cost of the projection. Six shapes times two arms at 80
forward-equivalents is about 40 minutes in six streams and it decides whether the
projection stays in the recipe.

---

## 13. Artifacts, absolute paths

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

Documents:
* `TOPOPT_REPORT.md` this file
* `FROZEN_CONVENTIONS_2D.md` the frozen two-dimensional conventions for the port lane

New code, all under `fgm_solve_campaign/adjoint2d/`: `topopt.py`, `chi_area.py`,
`topopt_objective.py`, `gate_topopt.py`, `topopt_solve.py`, `topopt_robust.py`,
`topopt_dosematch.py`, `build_topopt_tables.py`, `make_topopt_figures.py`,
`tests/test_topopt.py`, `tests/test_chi_area.py`, `tests/test_topopt_objective.py`.
Shell drivers under `fgm_solve_campaign/`: `run_topopt_gate.sh`, `run_topopt_stream.sh`,
`run_topopt_robust.sh`.

Results, all under `fgm_solve_campaign/out_topopt/`:
* `gate_topopt_{square,circle}.json` the four-layer finite-difference gate
* `<shape>.json` and `<shape>_maps.npz`, six shapes, the continuation solve
* `<shape>_control_filteronly.json` and `_maps.npz`, six shapes, the beta = 0 control
* `<shape>_robust.json` and `_robust_maps.npz`, six shapes, Gates A and B on the
  continuation
* `<shape>_control_filteronly_robust.json` and `_robust_maps.npz`, six shapes, the same
  two gates on the control
* `<shape>_dosematch.json`, six shapes, the dose-matched Gate B repeat

Logs: `fgm_solve_campaign/logs_topopt/*.log`.

Figures, **all five viewed before delivery**, under `fgm_solve_campaign/figs_topopt/`:
* `fig_topopt_parameterization.png` the projection ladder, the physical radius on two
  grids, the area-fill target against the raster it replaces, and the target's
  grid convergence
* `fig_topopt_gate.png` the gate, eight layers of V curves, plus the diagnosis panel that
  shows the absolute error is flat and every relative failure is a small denominator
* `fig_topopt_continuation.png` the beta stages, the objective, the non-discreteness, and
  the beta = 0 control on the same axes
* `fig_topopt_maps.png` the delivered maps, what they melt, and where the residual error
  sits
* `fig_topopt_acceptance.png` Gate A across grids for both arms, Gate B across four
  generations of arm, and the three forms of Gate B

Read, not modified: `OVERNIGHT_QUEUE_2026-08-01.md`, `SOLVE_ROBUSTNESS_VALIDATION.md`,
`MULTISTART_REPORT.md`, `EPS_CHANNEL_REPORT.md`, `fgm_solve_campaign/out_lib/*.json`,
`fgm_solve_campaign/out_ms/*.json`, `fgm_solve_campaign/out_robust/*.json`,
`rfam_eqs_coupled.py`.
