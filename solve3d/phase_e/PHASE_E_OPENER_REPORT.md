# Phase E opener: the SOLVED pyramid (cube arm to follow)

Pre-registration: `solve3d/phase_e/results/phase_e_preregistration.json`,
committed in `56ee531` BEFORE any Phase E solve code ran.
Conventions: `FROZEN_CONVENTIONS_2D.md` (their b04e356), plus the Phase C
recorded deviation (1/|g0| objective rescale) which is now standing.
Environment: spike env (dolfinx 0.11.0 complex, `jit_fix` imported first,
scipy 1.17.1). `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` on every run.
Date: 2026-08-03.

Every number below is read out of `solve3d/phase_e/results/*.json`. Nothing is
transcribed by hand.

**The cube arm is still solving (auto-resumed by the scheduler). Its gates,
figures and report section follow when it lands. This document covers the
pyramid only.**

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

## Reproduce

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.run --shape pyramid
heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.rescore --shape pyramid
heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.acceptance --shape pyramid
.venv312/bin/python -m solve3d.phase_e.render_deck_pyramid
```
