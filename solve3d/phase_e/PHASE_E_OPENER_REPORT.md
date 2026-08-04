# Phase E opener: pyramid and cube, two shapes solved in 3-D

Pre-registration: `solve3d/phase_e/results/phase_e_preregistration.json`,
committed in `56ee531` BEFORE any Phase E solve code ran.
Conventions: `FROZEN_CONVENTIONS_2D.md` (their b04e356), plus the Phase C
recorded deviation (1/|g0| objective rescale) which is now standing.
Environment: spike env (dolfinx 0.11.0 complex, `jit_fix` imported first,
scipy 1.17.1). `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` on every run.
Date: 2026-08-03.

Every number below is read out of `solve3d/phase_e/results/*.json`. Nothing is
transcribed by hand.

**Both shapes are complete: three arms each, both acceptance gates on each
solve arm, figures for both.** The pyramid is section 1 to 6, the cube is
section 7, and section 8 closes the two-shape opener.

## The two-shape verdict in one table

| | pyramid | cube |
|---|---|---|
| solve vs uniform (primary J) | **-57.98 %** | **-46.09 %** |
| hand-built grading law vs uniform | **+4.33 % worse** | **+50.00 % worse** |
| smoothing robustness | PASS (1.48 % vs 10 %) | PASS (2.11 % vs 10 %) |
| mesh hold-out, shape metrics | PASS, all five | PASS, all five |
| mesh hold-out, scalar J | **FAIL** (15.08 % vs 7.70 % band) | **FAIL** (22.61 % vs 9.64 % band) |
| solve still beats uniform at score mesh | yes, by 54.00 % | yes, by 37.89 % |
| `solved_label` | **false** | **false** |

Neither map earns the pre-registered label, for the same reason on both
shapes, with the same mechanism behind it (sections 3b, 7c and 8).

---

## The answer for the pyramid

**The 3-D adjoint solve produces a large, mesh-robust improvement in shape
fidelity, but the map does NOT earn the pre-registered SOLVED label.**

`solved_label = false`, because the pre-registration requires *all three* of
beats-uniform, mesh hold-out, and smoothing robustness, and the hold-out fails
on one check.

What actually happened, in one table
(`results/phase_e_pyramid.json`, asymmetric objective, each arm read at its
own envelope stop):

| arm | J_asym | vs uniform | t_stop (s) | mean phi | below floor | IoU(phi 0.9) | front SSD (mm) | sigma_T (C) |
|---|---|---|---|---|---|---|---|---|
| uniform_baseline | 1.452004e-06 | -- | 418.20 | 0.3933 | 0.6633 | 0.1848 | 3.064 | 29.49 |
| heuristic_grading_law | 1.514836e-06 | **+4.33 % (worse)** | 327.15 | 0.3805 | 0.6514 | 0.1948 | 2.957 | 38.17 |
| solve_filter_only | 6.101507e-07 | **-57.98 %** | 638.75 | 0.6752 | 0.3550 | 0.3802 | 1.633 | 25.16 |

The solve nearly doubles the part's mean melt fraction (0.393 -> 0.675), cuts
the in-bounds below-floor fraction by 46 % (0.663 -> 0.355), doubles the
melt-region IoU (0.185 -> 0.380) and halves the melt-front distance (3.06 ->
1.63 mm), while holding out-of-part melt essentially flat (0.00390 ->
0.00353, i.e. it does not buy in-bounds melt by spilling into the bed).

Per the S2 honesty constraint carried into the registration, the ABSOLUTE
fidelity numbers on this shape are ungated -- the pyramid has an apex, a
physical field singularity that a conforming mesh does not remove. What is
meaningful is the solved-vs-uniform comparison at matched mesh and matched
read state, and the hold-out gate result itself. Both are reported below.

---

## 1. The heuristic grading law is worse than doing nothing

The hand-constructed depth/height law from `demo_pyramid_fgm` (commit
`6c2aab9`, strong variant), transferred by evaluating the closed form directly
at the FEM cell centroids, **loses to uniform on the primary objective by
4.33 %**. It does move the diagnostic shape metrics slightly the right way
(IoU 0.1848 -> 0.1948, front SSD 3.064 -> 2.957 mm), but it stops the part
much earlier (327 s against 418 s) and its temperature field is *less* uniform
than uniform dopant (sigma_T 29.5 -> 38.2 C, diagnostic only).

This is a clean instance of the standing lane finding: a physically motivated
grading law is feedback, not a solve. It reproduces the right qualitative
intent -- more dopant deep and low -- without finding an operating point that
actually improves the objective.

## 2. What the solve did: it emptied the base, not the apex

`results/map_pyramid_solve_filter_only.npz`, 24 442 design cells, saturation
in [0.0201, 1.0000], volume-weighted mean 0.7759, 46.5 % of cells below 0.9.

The structure is almost purely a function of build height. Mean saturation by
z-slab (from the deck figure's own read):

| z slab (mm) | -11.6 to -5.8 | -5.8 to 0.0 | 0.0 to +5.8 | +5.8 to +11.6 |
|---|---|---|---|---|
| mean saturation | 0.63 | 0.96 | 1.00 | 0.99 |

The profile is monotone: ~0.51 at the base plane rising to 1.00 by z ~ -3 mm
and flat above. In-plane there is a secondary pattern in the bottom slab (the
base face is cooler at its mid-edges than at its corners).

So the solve's move is to **hold dopant back in the wide base** and let the
apex run at saturation. The direction is the *opposite* of the heuristic law,
which grades dopant UP with depth (more dopant low). That sign disagreement is
the most interesting single fact in the pyramid arm, and it is consistent with
what the objective is actually rewarding: the base is the part's thermal sink
and its own volume, so under fixed total power the way to raise mean melt
fraction is to stop over-driving the base and let the stop time extend
(t_stop 418 -> 639 s).

## 3. Acceptance gates

`results/phase_e_gate_pyramid.json`. Bands are MEASURED in this campaign per
shape (registration `acceptance.bands_are_per_shape`), never inherited from
the circle/square anchors.

### 3a. Smoothing robustness -- PASS

0.5 mm sub-filter-radius blur of the delivered map (filter radius 1.0 mm),
FROZEN section 8 Gate B:

J 6.101507e-07 -> 6.191518e-07, a **1.475 % change against a 10 % tolerance**
(6.8x margin). The map is not a knife-edge feature of the design space.

### 3b. Mesh hold-out -- FAIL on J, PASS on every shape metric

Solve mesh lc = 0.9375 mm, score mesh lc = 0.625 mm. Map transferred by
re-evaluating the same normalized-convolution kernel that produced it; total
in-part dopant moved **0.041 %** (the Phase C inversion arm was dropped at
7.15 % on this same check, so the transfer here is clean by two orders of
magnitude).

| metric | solved arm moved | band (1.5x uniform's own move) | uniform's own move | verdict |
|---|---|---|---|---|
| J_asym (relative) | 0.15084 | 0.07702 | 0.05134 | **FAIL** |
| IoU(phi 0.9) | 0.00934 | 0.01418 | 0.00946 | PASS |
| IoU(phi 0.8) | 0.00846 | 0.01460 | 0.00973 | PASS |
| in-part phi 0.9 | 0.00934 | 0.01418 | 0.00946 | PASS |
| out-of-part phi 0.9 | 0.0 | 0.0 | 0.0 | PASS |
| front SSD (mm) | 0.0898 | 0.2441 | 0.1627 | PASS |

At the score mesh the solved map **still beats uniform**: 7.021863e-07 against
1.526556e-06, a **54.00 % margin** (against 57.98 % in grid). The win is not a
coarse-mesh artifact. Every verdict-carrying shape metric is inside its
measured band, and on the shape metrics the solved arm actually moves LESS
across meshes than the uniform arm does.

The single failing check is the scalar J, which moves 15.08 % coarse-to-fine
against a 7.70 % band. **Mechanism**: J is read at each arm's own envelope
stop, and the solved arm's stop sits at step 12 775 of 13 000, i.e. 98.3 % of
the way along the 650 s horizon. (It is a genuine interior argmin --
`at_horizon = false` -- not a clipped read.) A late stop sits on a flatter,
longer-lever part of the trajectory, so a small mesh-induced change in march
rate moves the stop a lot: the solved arm's t_stop moved 638.75 -> 603.90 s
(-5.5 %) while uniform's moved 418.20 -> 411.15 s (-1.7 %). The J band is
derived from the uniform arm's move, so it is calibrated on an arm that is
three times less stop-sensitive than the one being tested. The band is
therefore tight for this arm by construction, and that is a property of the
band rule, not evidence that the map is wrong.

I am NOT relaxing the gate on that reasoning. The pre-registered rule says
all-of, one check failed, so `solved_label = false` and the pyramid map is
reported as an improvement that is robust on shape but not yet on the scalar
objective. Two honest reads of the same failure, both stated:

* conservative: the delivered map's *objective value* is not mesh-converged to
  the pre-registered band, so it should not carry the SOLVED label.
* mechanistic: the failure is concentrated in the stop-time read, the shape
  metrics that actually carry the verdict all pass, and the improvement itself
  survives refinement at 54 %.

The apex-singularity escalation from S2 stands: settling whether the
near-apex field is mesh-converged at all belongs to S3's COMSOL anchor, not
to Phase E.

## 4. Budget limit -- the solve is still descending

`status = budget_exhausted`, 12 of 12 pre-registered gradient evaluations
used, wall 7374 s. The trajectory descends to the last evaluation (eval 10 is
a rejected line-search probe at 8.407e-07, the only non-decreasing step), with
the big drop between evals 4 and 5 (1.4147e-06 -> 1.0204e-06) when the stop
time first extended (425.65 -> 522.10 s). The last two accepted steps still
gain 4.4 % and 4.0 %. **The -57.98 % is a
lower bound on what this arm can do, not a converged optimum.** Every quoted
improvement carries that caveat.

The Phase C 1/|g0| rescale was applied (`scale_first_step = true`, scale
6.2652e+07) and this arm did not stall on the upper rail -- 12 productive
evaluations against Phase C's 2 before the rescale.

## 5. What is proven, simplified, and assumed

**Proven (gated):** the transient 3-D adjoint gradient driving this solve is
the Phase B FD-gated gradient, unchanged; the design chain's filter transpose
is gated; the forward's energy residual on the delivered arm is 1.22e-13 with
no clamp and no CFL violation; the map's total dopant is conserved to 0.041 %
across the hold-out transfer; smoothing robustness passes at 6.8x margin.

**Simplified vs heatr3d:** conforming tetrahedral mesh instead of the voxel
staircase (this is the point of solve3d, and S2 showed the staircase is the
larger error on cornered shapes); dopant enters as a saturation field on the
part cells only; fixed total power with the renormalization differentiated
through; no densification feedback in the solve arm's march.

**Assumed / ungated:** all ABSOLUTE fidelity numbers on this shape (apex
singularity, per the S2 constraint); that 12 gradient evaluations is anywhere
near the achievable optimum (it is not -- see section 4); that the 0.625 mm
score mesh is itself converged near the apex (S3's question).

## 6. Deck figure

`results/fig_deck_pyramid_solved.png` (Space Mono dark, `style3d`), rendered
by `render_deck_pyramid.py` for Matt's conference deck. Left: four z-slabs of
the solved dopant map plus the mean-saturation-vs-z profile, so the through-z
structure reads at a glance. Right: vertical x-z sections through the part
centre for uniform vs SOLVED, with the nominal outline and the phi = 0.9 melt
front, plus the numbers block. Honesty line on the figure: "solved by 3-D
adjoint (filter-only arm), budget-limited, still descending; simulation-only".
No title block (the deck adds its own).

`results/fig_deck_pyramid_cutaway3d.png`, rendered by
`render_deck_cutaway3d.py` in the `fig1_dense_inside_bounds` quarter-cutaway
style: the nominal pyramid solid with the x>0, y>0 wedge removed, uniform vs
the filter-only solve arm at the same camera, inside the cyan nominal
wireframe, with the phi = 0.9 front contoured on the two cut faces.

**What colours that figure, for the caption: MELT FRACTION phi -- a real full
3-D field, not the dopant map.** The data contract was checked before the
figure was designed: `field_<shape>_<arm>.npz` stores `T_read`, the complete
nodal temperature at that arm's own envelope read state, and
`forward.phase_fraction` is pointwise in T, so T_read determines a genuine
melt-fraction volume. `sample_volume.py` evaluates it on a 96^3 grid (zero
missed points in-part). Cross-check: in-part mean phi off that grid is 0.3763
and 0.6671, against 0.3933 and 0.6752 from the FEM volume-weighted scorer --
the small gap is grid vs FEM weighting, not a different quantity.

Caption constraint: the arm may be called "solve (filter-only arm)" but NOT
described as validated or SOLVED, because the pre-registered label was not
earned (section 3b).

---

## 7. The cube

Mesh: 105 853 cells, 18 028 nodes, 24 042 design cells, part volume matching
the library solid to 2.2e-16 relative. Same lc, same kernel, same objective,
same 12-evaluation budget as the pyramid.

### 7a. Three arms

`results/phase_e_cube.json`, asymmetric objective, each arm at its own
envelope stop (all three are interior argmins, `at_horizon = false`):

| arm | J_asym | vs uniform | t_stop (s) | mean phi | below floor | out-of-part | sigma_T (C) |
|---|---|---|---|---|---|---|---|
| uniform_baseline | 7.841949e-07 | -- | 514.30 | 0.6365 | 0.4028 | 0.00370 | 23.37 |
| heuristic_grading_law | 1.176273e-06 | **+50.00 % worse** | 439.00 | 0.5803 | 0.4523 | 0.01181 | 38.62 |
| solve_filter_only | 4.227897e-07 | **-46.09 %** | 623.55 | 0.7530 | 0.2786 | 0.00296 | 27.67 |

On shape-relative evaluation planes (`rescore.py`, the same fix the pyramid
needed):

| arm | IoU (phi 0.9) | front SSD (mm) |
|---|---|---|
| uniform | 0.59991 | 1.940 |
| heuristic | 0.53062 | 2.724 |
| solve | 0.73789 | 1.203 |

The cube starts from a far better baseline than the pyramid (uniform IoU 0.600
against 0.185): it is a much easier shape, with no apex and a melt region that
already fills most of the part. The solve still finds 46 % of the objective.

### 7b. The hand-built law fails much harder here, and the reason is legible

On the cube the `demo_pyramid_fgm` law is **50.00 % worse than uniform**, an
order of magnitude worse than its 4.33 % loss on the pyramid. Splitting the
objective says exactly why:

| J component | uniform | heuristic | ratio |
|---|---|---|---|
| out-of-bounds (melt outside the part) | 3.827010e-08 | 2.336303e-07 | **6.10x** |
| in-bounds deficit (unmelted inside) | 7.459248e-07 | 9.426430e-07 | 1.26x |

**The failure is dominated by the out-of-bounds term, which the law multiplies
by 6.1x.** The volumetric check agrees independently: melt outside the part
goes from 0.370 % to 1.181 % of part volume, 3.20x. The two ratios differ
because the objective weights out-of-bounds melt asymmetrically (Matt's
recorded "dense if and only if in-bounds" refinement); both say the same
thing, and the figure shows it as a field.

This is the asymmetric objective doing its job. A law that grades dopant up
with depth pushes heat toward the part boundary, and on a cube every boundary
is a flat face with bed powder directly against it. The pyramid's sloped faces
are more forgiving. The solve moves the opposite way on both shapes and cuts
out-of-bounds melt to 0.54x uniform.

### 7c. Acceptance gates: the same pattern as the pyramid

`results/phase_e_gate_cube.json`. Bands measured on this shape, not inherited.

**Smoothing robustness: PASS.** 0.5 mm sub-filter blur, J 4.227897e-07 to
4.317132e-07, a **2.111 % change against a 10 % tolerance** (4.7x margin).

**Mesh hold-out (lc 0.9375 to 0.625 mm): FAIL on J, PASS on all five shape
metrics.** Map transferred by the same normalized-convolution kernel; total
in-part dopant moved **0.309 %** (against the 7.15 % that dropped the Phase C
inversion arm).

| metric | solved moved | band (1.5x uniform's own) | uniform's own | verdict |
|---|---|---|---|---|
| J_asym (relative) | 0.22610 | 0.09644 | 0.06430 | **FAIL** |
| IoU(phi 0.9) | 0.01312 | 0.02060 | 0.01374 | PASS |
| IoU(phi 0.8) | 0.01252 | 0.01906 | 0.01271 | PASS |
| in-part phi 0.9 | 0.01312 | 0.02060 | 0.01374 | PASS |
| out-of-part phi 0.9 | 0.0 | 0.0 | 0.0 | PASS |
| front SSD (mm) | 0.06738 | 0.11639 | 0.07760 | PASS |

At the score mesh the solved map still beats uniform: 5.183813e-07 against
8.346144e-07, a **37.89 % margin** (against 46.09 % in grid).

Same mechanism as the pyramid, now measured twice: the solved arm's stop time
is far more mesh-sensitive than the uniform arm the band is calibrated on.
Cube t_stop moved 623.55 to 592.50 s (**-4.98 %**) against uniform's 514.30 to
507.15 s (**-1.39 %**), a **3.6x** sensitivity ratio; on the pyramid it was
3.2x. `solved_label = false`.

### 7d. What the cube solve did: it emptied the two x-normal faces and both z ends

`results/map_cube_solve_filter_only.npz`, 24 042 design cells, saturation in
[0.0050, 1.0000], volume-weighted mean 0.6760, 68.1 % of cells below 0.9.

Mean saturation by z slab: 0.60, 0.75, 0.76, 0.60 -- symmetric about the
mid-plane, unlike the pyramid's monotone rise. That symmetry is the right
answer for a symmetric part and is a free sanity check on the solve: nothing
in the objective or the chain enforces it.

In plane, the mid slabs show two dark bands on the **x-normal faces**. That is
an independent echo of a result this lane already has: `fig1_dense_inside_bounds`
measured that the square's out-of-part melt is not at the corners but is a 1
to 2 voxel skin outside the two x-normal faces. The solve, with no knowledge
of that finding, pulls dopant out of exactly those faces.

### 7e. Budget

`status = budget_exhausted`, 12 of 12 evaluations, wall 11 524 s, no resume.
The trajectory is **strictly monotone decreasing** on all 12 evaluations
(7.842e-07 to 4.228e-07), with no rejected line-search trial at all, and the
last step still gains 1.2 %. The 1/|g0| rescale was applied (scale 1.7398e+08).
As on the pyramid, **-46.09 % is a lower bound**.

### 7f. Cube figures

* `results/fig_deck_cube_three_arms.png` -- the three arms as 3-D quarter
  cutaways at one camera coloured by melt fraction phi, over a difference row
  against the uniform baseline. The heuristic's difference panel is blue
  (melts less) through the core with orange spilling below the part; the
  solve's is orange around the faces. Same data contract as the pyramid
  cutaway: phi from the full `T_read` volume, pointwise in T.
* `results/fig_deck_cube_solved_map.png` -- the delivered dopant map: four z
  slabs (volume-weighted bin averages, not scatter), the mean-saturation
  profile through build height, the same map as a 3-D cutaway, and the
  numbers.

---

## 8. Closing the two-shape opener

### What this campaign established

1. **The 3-D transient adjoint solve beats uniform dopant on both shapes, by a
   large margin, and the win survives mesh refinement.** -57.98 % and -46.09 %
   in grid; +54.00 % and +37.89 % margins still present at the finer score
   mesh. Every verdict-carrying shape metric passes its measured band on both
   shapes.
2. **The hand-built grading law loses on both shapes**, by 4.33 % and 50.00 %.
   It is feedback, not a solve, and on the cube the loss is traceable to a
   6.10x increase in the out-of-bounds objective component. Two shapes is not
   a survey, but the law lost on the shape it was built for.
3. **Neither map earns the pre-registered SOLVED label**, because the same
   single check fails on both: the scalar J moves too far across meshes. The
   shape metrics do not. The mechanism is measured, not asserted: the solve
   extends the stop time, later stops sit on a flatter part of the trajectory,
   and the band is calibrated on the uniform arm, which is 3.2x to 3.6x less
   stop-sensitive. **This is a property of the band rule as pre-registered.**
   I am not relaxing it after the fact; I am recording that a future
   registration should either band J against the tested arm's own stop
   sensitivity or make the shape metrics the sole verdict carriers, and that
   change must be pre-registered before the next solve, not chosen now.
4. **The solve's structure is physically legible on both shapes and is not
   what the heuristic assumed.** Pyramid: empty the wide base, run the apex at
   saturation. Cube: pull dopant symmetrically out of both z ends and out of
   the two x-normal faces, the same faces where this lane independently
   measured the square's melt spill.
5. **Determinism and gradient provenance are pinned.** A recomputed forward at
   the delivered pyramid map reproduces the stored objective exactly, relative
   difference 0.0 (`results/grad_pyramid_solve_filter_only.json`).

### What stays open

* **The corner and apex adjudication belongs to S3's COMSOL anchor.** Both
  Phase E shapes are cornered and the pyramid has an apex. No ABSOLUTE
  fidelity number on either shape is gated by anything, and none is quoted as
  if it were. Whether the near-apex field is mesh-converged at all is not a
  question Phase E can settle.
* **The projection arm was never run.** `solve_projection_beta_continuation`
  was pre-registered as optional on one shape; the budget went to finishing
  both filter-only arms and their gates. It is untried, not tried and rejected.
* **Both solves are budget-limited and still descending.** Every improvement
  quoted here is a lower bound on what 12 evaluations of this arm can reach,
  let alone what the method can reach.
* **Two shapes out of the fourteen-shape library.** The pyramid and cube were
  chosen for the opener; nothing here says how the method behaves on reentrant
  or thin-neck geometry, which is where S2 found the engine itself unsettled.
* **The J hold-out band rule needs a pre-registered fix** before the next
  campaign, per point 3 above.

---

## Reproduce

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SPIKE=heatr3d_d1_spike/env/bin/python

for S in pyramid cube; do
  $SPIKE -m solve3d.phase_e.run        --shape $S
  $SPIKE -m solve3d.phase_e.rescore    --shape $S
  $SPIKE -m solve3d.phase_e.acceptance --shape $S
  $SPIKE -m solve3d.phase_e.sample_volume --shape $S \
      --arms uniform_baseline,heuristic_grading_law,solve_filter_only
done
$SPIKE -m solve3d.phase_e.gradient_probe --shape pyramid

.venv312/bin/python -m solve3d.phase_e.render_deck_pyramid
.venv312/bin/python -m solve3d.phase_e.render_deck_cutaway3d
.venv312/bin/python -m solve3d.phase_e.render_deck_loop3d
.venv312/bin/python -m solve3d.phase_e.render_deck_cube
```

## Figure index

| file | what |
|---|---|
| `fig_deck_pyramid_solved.png` | pyramid solved map: z slabs, profile, uniform vs solved sections |
| `fig_deck_pyramid_cutaway3d.png` | pyramid uniform vs solve, 3-D quarter cutaway, melt fraction |
| `fig_deck_solve_loop_3d.png` | the five-stage 3-D adjoint loop, real recomputed gradient at stage 4 |
| `fig_deck_cube_three_arms.png` | cube three arms plus difference row, melt fraction |
| `fig_deck_cube_solved_map.png` | cube solved map: z slabs, profile, cutaway, numbers |
