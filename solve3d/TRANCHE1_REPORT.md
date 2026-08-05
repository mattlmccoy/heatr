# Shrinkage v2 Tranche 1: L0, arbitrary-STL meshing, CG/AMG spike

Plan: `docs/superpowers/plans/2026-08-04-shrinkage-v2-tranche1.md` (9494a89)
Spec: `docs/superpowers/specs/2026-08-04-shrinkage-prewarp-v2-design.md` (APPROVED)
Coefficients: `SHRINKAGE_COEFFICIENTS_MEMO.md` (c2a8563)
Date: 2026-08-04. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` on every run.
Environment: spike env, dolfinx 0.11.0 complex, gmsh 4.15.2, PETSc via
petsc4py; `.venv312` for anything that needs trimesh.

Every number below is printed from `solve3d/results/*.json` or from a test
run quoted in the commit it landed in. Red-first TDD throughout: each task's
tests were written and run FAILING before any implementation existed.

---

## Gates table

| task | gate | measured | standard | verdict |
|---|---|---|---|---|
| 1 | L0 affine map on analytic vertices | exact, `atol=0.0` | exact | PASS |
| 1 | L0 round trip (scale up, shrink by s) | `rtol=1e-15` | 1e-15 | PASS |
| 1 | double-counting guard, 4 refusals | all raise + name the key | must refuse | PASS |
| 1 | `s_*=0` bit-identity | `tobytes()` equal | bit-identical | PASS |
| 1 | default path mesh unchanged | 106822/18172 and 105853/18028 cells/nodes | identical to committed JSON | PASS |
| 1 | full suite regression | 135 passed, 1 pre-existing fail | no new failures | PASS |
| 2 | pyramid STL tet volume vs STL enclosed volume | **2.220e-16** | 5.0e-3 (stated) | PASS |
| 2 | chi fill contract + independent ray-cast cross-check | >0.995 cells inside | contract | PASS |
| 2 | OCC vs STL equivalence | `rel_diff = 0.0` | band 0.10731 | PASS |
| 2 | Tier-3 refusals before gmsh is touched | both raise | must refuse | PASS |
| 3 | FD re-gate, direct control | worst 1.4243e-06 | 1e-5 subgradient | PASS |
| 3 | FD re-gate, iterative rtol 1e-10 | worst **6.7833e-06** | 1e-5 subgradient | PASS |
| 3 | FD re-gate, iterative rtol 1e-14 | worst 7.3168e-06 | 1e-5 subgradient | PASS |
| 3 | iterative field agreement vs direct | V rel L2 2.3e-15 | machine precision | PASS |

Commits: `4cf5d13`, `4703532` (Task 1), `7bdd679` (Task 2), `091863d`,
`c1a7cb2`, `ab7b16a` (Task 3).

---

## 1. Level 0 pre-compensation

Coefficients live in ONE shared repo-root file, `shrinkage_precomp.json`
(schema 1.0), which the Studio lane landed first (`79c1835`); my write was a
no-op and I verified their tracked content byte-for-byte against my tests
rather than assuming it matched. `s_xy = 0.030` (band 0.020-0.040),
`s_z_mat = 0.020` (band 0.010-0.030). Nothing is hardcoded.

**The double-counting guard is Matt's material-vs-consolidation distinction
made executable, and it refuses rather than ignores.** Four separate
refusals: the coefficient schema is exactly `{s_xy, s_z_mat}`; a mapping
offering a densification/consolidation/collapse/rho/density/total key raises
`DoubleCountingError` naming the offender; the shared file must assert
`material_only: true` or it is refused; an unknown `schema_version` raises
rather than being read optimistically.

### Default flipped ON, and two corrections it forced

Authority: the approved spec section 2, verbatim, *"Off by default until the
memo lands; then default ON with the coefficients displayed."* The memo has
landed. I initially shipped OFF and escalated; the coordinator ruled that
flipping executes approved intent rather than making a new default decision,
and I verified that spec line myself before acting.

The flip exposed two silent-wrong-number bugs:

* `part_volume_rel_err_vs_library` compared the pre-compensated solid against
  the NOMINAL library volume, reporting a ~8 % "error" that was the
  correction working as designed. Now compared against the L0-scaled volume.
* The gmsh refinement box was origin-centred and did not follow the scaled
  part, so L0 would have quietly coarsened the mesh at the part boundary.

Both are the class of bug that produces confident wrong numbers rather than a
crash, which is what the tests exist for.

### Reproduction consequence (carry this forward)

Every existing solve3d artifact predates L0 and was built with no
pre-compensation. **Re-running any pre-L0 campaign reproduces its recorded
numbers only if the coefficients are pinned to zero explicitly:**

```python
build_case(shape, precomp_coeffs=precomp.ShrinkageL0(0.0, 0.0))
```

The flip governs new runs; it does not retroactively describe old ones. A
test pins this against the committed `phase_e_cube.json` counts.

### Provenance

Every run records coefficients, both coefficient bands, both derived scale
bands, source, `material_only`, and the applicability caveat: *SLS literature
values; RFAM material shrinkage unmeasured, pending P1 measurement.* That
caveat travels with the number rather than sitting only in the memo.

---

## 2. Arbitrary-STL tet meshing

**Route and versions.** Meshing is gmsh 4.15.2, the same kernel the OCC path
uses, so the equivalence gate compares like with like. Mesh object is dolfinx
0.11.0. Validation is plain numpy, **not trimesh**: trimesh is absent from the
dolfinx spike env (it lives only in `.venv312`), and adding a solver-env
dependency to answer questions that are a dozen lines of array code is the
wrong trade.

**`gmsh.merge` refuses these library binary STLs outright** ("Error loading"),
so the surface is handed to gmsh as a discrete entity built from the triangles
this module already parsed and validated. That also guarantees the meshed
surface is exactly the one the refusal checks ran on.

**The equivalence gate is not self-graded.** The band is read from
`dolfinx_refinement.json`: 1.5 x the engine's own mesh-refinement spread
(0.10731). The STL route counts as equivalent only if it differs from the OCC
route by no more than remeshing does. Measured `rel_diff = 0.0`.

**Three bugs the tests caught, all mine.** `classifySurfaces` at angle pi
spline-fitted one patch across planar facets and shrank the pyramid 2 %
(fixed to 40 degrees; volume is now exact at 2.220e-16). The refinement box
was origin-centred while the library pyramid sits at z in [0, 23.2] mm. The
ray-cast inside test reported axis points as outside, because the +z ray
passes exactly through the apex shared by four facets; degeneracy is now
retried at an offset larger than the tolerance.

### Named blocker: chamber embedding, and what the fix needs

**`with_chamber=True` does not work and is not claimed to.** Wrapping the part
in a bed as a geo-kernel volume-with-a-hole fails in tetgen with
`PLC Error: a segment and a facet intersect` when the inner surface comes from
a discrete entity. Default is `False`. The part-only mesh is exact and is what
Task 2's gates cover.

**A full solve needs the bed, so the Studio's `direct_solve` unblock is NOT
delivered by this tranche.** The Studio lane has not been notified.

What the fix needs, as the next-increment recommendation, in preference order:

1. **PLC-compatible bed embedding.** Re-mesh the STL surface into a clean
   conforming triangulation before it becomes a hole boundary, so the bed and
   part share an identical facet set. In gmsh terms: `classifySurfaces` ->
   `createGeometry` -> remesh the surface at the target `lc` -> then build the
   bed volume against that remeshed boundary rather than against the raw STL
   facets. The PLC error is a boundary-conformity complaint, and this removes
   the mismatch at its source.
2. **OCC route instead of the geo kernel.** Convert the validated surface to
   an OCC solid (`healShapes` then `fragment` against the chamber box), which
   is exactly what `phase_e/geometry.py` already does successfully for OCC
   primitives. This reuses a known-good code path; the risk is OCC's STL
   import fidelity on faceted solids.
3. **Immersed/ersatz boundary.** Keep the bed mesh independent and represent
   the part by a volume-fraction chi rather than a conforming boundary. This
   abandons the conforming-boundary advantage that motivates solve3d over the
   voxel engine, so it is a fallback, not a preference.

Whichever route is taken, the acceptance gate should be the one already
written: part volume against the STL's own enclosed volume, plus the fill
contract, plus the same OCC-vs-STL equivalence band.

### Fixture gap (library owner, not this lane)

The shape library ships two Tier-3 rejection fixtures, `open_cylinder`
(non-watertight) and `flat_plane` (zero volume). **It has no self-intersecting
fixture.** That refusal is exercised against two interpenetrating tetrahedra
constructed in the test module and labelled as constructed. A one-line gap the
library owner can close; I did not build it into the library.

Refusal names are a cross-component contract: `shape_library_3d` records the
exception it expects ingestion to raise in each meta JSON's `rejection_error`,
and a test asserts the names match so library and ingestion cannot drift.

---

## 3. CG/AMG iterative EQS spike

### The rtol-floor finding (this changed the experiment)

The plan required reporting the direct path's own rtol floor BEFORE choosing
an acceptance tolerance. Doing so was load-bearing.

**Measured: the incumbent solves to 9.59e-20 (coarse, 2520 dofs) and 8.26e-20
(mid, 9257 dofs) relative residual** — PETSc `preonly` + LU plus `N_REFINE=2`
iterative-refinement sweeps.

That is ~1e-19, **not** ~1e-10. `forward.py`'s existing `KSP_ITER` preset uses
`ksp_rtol 1e-10`, nine to ten orders looser, and `adjoint.py` records that the
refinement sweeps "buy roughly two decades of FD headroom". So a 1e-10
iterative solve was expected *a priori* to damage the Phase B FD gate. The
pre-registration therefore made the **FD re-gate the binding criterion** rather
than field agreement, swept rtol to 1e-14, and stated that keeping the direct
path was a valid outcome.

Also pre-registered before any run: CG requires a Hermitian positive definite
operator, but the EQS operator is complex **symmetric** — not the same thing.
GMRES+GAMG was named as the path, with the reason recorded in advance rather
than discovered afterwards.

### A false green, caught

The first spike pass reported the FD gate passing at every rtol and
recommended adoption. **It measured nothing.** `run_fd_gate(reuse=True)`
returns the cached `phase_b_steady_gate.json` artifact without solving. The
tell was that all four arms, including the direct control, had `wall_s = 0.0`
and rel_errs identical to twelve significant figures. The spike now passes
`reuse=False` and raises if any arm returns in under a second, so an arm that
does not solve can never again be scored as a pass.

This is recorded because the corrected result agrees with the false one in
verdict but not in evidence, and the difference matters.

### Results

FD re-gate, 24784 complex dofs, 8 epsilons x 4 probes, thresholds read from
the frozen protocol (1e-5 subgradient pass, 1e-6 preferred):

| arm | wall | worst best_rel_err | preferred | verdict |
|---|---|---|---|---|
| direct control | 1206 s | 1.4243e-06 | 3/4 | PASS |
| iterative rtol 1e-10 | 54 s | **6.7833e-06** | 3/4 | PASS |
| iterative rtol 1e-14 | 65 s | 7.3168e-06 | 3/4 | PASS |

The arms **differ**, which is what proves the solver swap reached the loop.

Speedup, same assembled operator and machine state, best of 3, load-checked:

| level | dofs | cells | direct | iterative (1e-10) | speedup |
|---|---|---|---|---|---|
| coarse | 2520 | 13969 | 0.1696 s | 0.0428 s | **3.96x** |
| mid | 9257 | 53480 | 3.6241 s | 0.1803 s | **20.10x** |

Field agreement is at machine precision (V rel L2 2.3e-15) and the iterative
`res_norm` reaches ~1e-19, **the same floor as direct**.

**The mechanism matters and is not what the framing suggested.** The existing
`N_REFINE=2` refinement sweeps clean up after the loose inner solve, so rtol
barely moves final accuracy — 1e-14 is marginally *worse* than 1e-10, i.e.
noise. Speedup comes from the inner solve; accuracy comes from the refinement.

**Cost disclosed.** The iterative path degrades worst-probe FD accuracy about
4.8x, 1.42e-06 to 6.78e-06. It still clears the 1e-5 campaign standard with
the same 3/4 preferred count, but it sits closer to the limit, and that is a
real price rather than a rounding detail.

**Honest scaling caveat.** D1's precedent was 229-431x. Measured here is
4-20x at 2.5k-9.3k dofs. The trend rises steeply with size, but nothing in
this tranche measured production-scale meshes and **the D1 figure is not
reproduced**.

### Recommendation

Adopt GMRES+GAMG at rtol 1e-10 behind the flag. **Default stays DIRECT and the
off-path is bit-identical**; the flip goes to Matt with this evidence, per the
EQS-02 precedent. Before flipping, I would want the speedup measured at a
production-scale mesh, since 20x at 9k dofs is the interesting end of a curve
this tranche only sampled twice.

---

## What is NOT covered

* **L1 prewarp fixed-point loop** — Studio lane, not started here.
* **L2 differentiable shrinkage in the objective** — the research core. Needs
  the rho co-state and the densify march ported into solve3d's forward and
  adjoint, plus one new VJP layer for the kinematics map. Task 2's meshing was
  a prerequisite and is only partly delivered (see the chamber blocker).
* **L3a/L3b sintering mechanics** — not started.
* **Metal/MPS march kernel** and **CUDA backend** — compute items 2 and 3,
  untouched.
* **Chamber/bed embedding for STL parts** — the named blocker above.
* **Production-scale CG/AMG benchmark** — the speedup curve is sampled at two
  small sizes only.
* **Any dissertation edit** — out of scope for this lane by the spec.

## Next-tranche recommendation

1. **PLC-compatible bed embedding** (route 1 above). It is the single item
   blocking both L2 and the Studio's `direct_solve` rung, and the acceptance
   gates for it already exist.
2. **Production-scale CG/AMG benchmark**, then take the default flip to Matt
   with a curve rather than two points.
3. **L2 groundwork**: port the densify march into solve3d's forward and
   adjoint. Largest engineering item at that level, and it is on the critical
   path for the research core.
4. Ask the library owner to add a self-intersecting Tier-3 fixture.

## Reproduce

```
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SPIKE=heatr3d_d1_spike/env/bin/python

$SPIKE -m pytest solve3d/tests/test_precomp.py -q        # 23 passed
$SPIKE -m pytest solve3d/tests/test_stl_mesh.py -q       # 13 passed
$SPIKE -m pytest solve3d/tests/test_cgamg.py -q          # flag + agreement
$SPIKE -m solve3d.cgamg_spike                            # ~40 min: real FD arms
```
