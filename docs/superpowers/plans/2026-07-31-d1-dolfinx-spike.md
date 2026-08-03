# D1 dolfinx Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, with numbers, whether FEniCSx/dolfinx becomes (a) the high-fidelity cross-check + P2 mechanics engine, (b) mechanics engine only, or (c) not adopted - per decision point D1 of docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md. Time-boxed: if the cumulative spike effort exceeds one focused week of work, stop and write the report with whatever is measured.

**Architecture:** A new isolated directory `heatr3d_d1_spike/` at the geo-prewarp repo root holds everything (its own environment, scripts, results JSON, report). Nothing in heatr3d.py or the Studio changes. Every task ends with a numeric gate recorded in `heatr3d_d1_spike/results.json`; the final report reads only from that file (no transcribed numbers).

**Tech Stack:** FEniCSx (dolfinx) from conda-forge via micromamba (macOS arm64), gmsh for meshing, PETSc. Complex-valued EQS requires a complex-scalar PETSc build: install `fenics-dolfinx` with the complex variant if available on this platform; if only the real build resolves, use the documented real/imaginary split formulation (two coupled real fields) and record that choice prominently - it changes effort scoring, not validity.

**Scoring criteria (from the spec + the north star):** field agreement vs anchors; corner behavior under refinement (the voxel staircase is the thing FEM should kill); wall-clock + memory at effective resolutions matching heatr3d n=96 and the infeasible n=200; adjoint-readiness (the inverse-design destination); implementation effort, honestly logged.

**Anchor numbers available today (do not re-derive):** 2.5-D circle d=20 anchor: V_cal 3399.6 V, heating-peak sigma_T 19.96 C, melt-onset 17.43 C, t90 ~380 s at 500 W/m. heatr3d n=96 full physics: t90 802.6 s, sigma_T 19.278 (different drive normalization - the 3-D power-density basis; compare like with like, never across bases). heatr3d EQS ceilings: n=96 full / n=128 EQS-only; n=200 direct LU needs ~29.8 GB. COMSOL volumetric exports: outputs_eqs/comsol_exports/ plus the Tuned_Sigma.mph geometry documented in project memory.

---

### Task 0: Environment (gate: solves a Poisson problem)

**Files:**
- Create: `heatr3d_d1_spike/setup_env.sh`
- Create: `heatr3d_d1_spike/check_env.py`
- Create: `heatr3d_d1_spike/results.json` (initialized `{}`)

- [ ] **Step 1: Write setup_env.sh**

```bash
#!/bin/bash
# D1 spike environment: micromamba + fenics-dolfinx (conda-forge), macOS arm64.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v micromamba >/dev/null 2>&1; then
  mkdir -p mm && cd mm
  curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj bin/micromamba
  cd ..
fi
MM="${PWD}/mm/bin/micromamba"; command -v micromamba >/dev/null 2>&1 && MM=micromamba
# try the complex-PETSc variant first; fall back to default (real) build
"$MM" create -y -p ./env -c conda-forge python=3.11 fenics-dolfinx "petsc=*=*complex*" gmsh python-gmsh pyvista || \
"$MM" create -y -p ./env -c conda-forge python=3.11 fenics-dolfinx gmsh python-gmsh pyvista
./env/bin/python check_env.py
```

- [ ] **Step 2: Write check_env.py (the gate)**

```python
"""Gate: dolfinx imports, reports scalar type, and solves -laplace(u)=1 on a
unit cube with u=0 walls; the max must match the known reference 0.0562+-5%
(center value of the unit-cube Poisson problem)."""
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

msh = mesh.create_unit_cube(MPI.COMM_WORLD, 24, 24, 24)
V = fem.functionspace(msh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(fem.Constant(msh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx
facets = mesh.exterior_facet_indices(msh.topology)
dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
bc = fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, V)
uh = LinearProblem(a, L, bcs=[bc]).solve()
umax = float(np.abs(uh.x.array).max())
scalar = str(dolfinx.default_scalar_type)
ok = abs(umax - 0.0562) / 0.0562 < 0.05
out = {"task0": {"dolfinx_version": dolfinx.__version__,
                 "scalar_type": scalar, "poisson_umax": umax, "gate_ok": ok}}
p = Path(__file__).parent / "results.json"
d = json.loads(p.read_text()) if p.exists() else {}
d.update(out)
p.write_text(json.dumps(d, indent=1))
print(out)
assert ok, "Poisson gate failed"
```

- [ ] **Step 3: Run setup, record whether the complex build resolved**

Run: `bash heatr3d_d1_spike/setup_env.sh`
Expected: env created, gate printed with `gate_ok: true`. Record
`scalar_type` (complex128 = complex build; float64 = real build, split
formulation required downstream). If BOTH create attempts fail on this
platform, try the docker route (`docker run dolfinx/dolfinx:stable`) and
record the pivot; if that also fails, STOP the spike and report - that is
itself a D1 finding (deployability on lab machines is a scoring criterion).

- [ ] **Step 4: Commit**

```bash
git add heatr3d_d1_spike/
git commit -m "feat(d1): dolfinx spike environment + Poisson gate"
```

### Task 1: EQS parallel plate in dolfinx (gate: matches analytic + heatr3d)

**Files:**
- Create: `heatr3d_d1_spike/eqs_common.py` (materials + formulation shared by all EQS tasks)
- Create: `heatr3d_d1_spike/run_plate.py`

- [ ] **Step 1: Write eqs_common.py**

```python
"""Shared D1 EQS pieces. Physics matches heatr3d.Params defaults:
freq 27.12 MHz, v_lo 860 V at y=-L/2, v_hi 0 at y=+L/2, Neumann side walls,
gamma = sigma + i*omega*eps0*eps_r with sigma_doped=0.04, eps_doped=20,
sigma_virgin=1e-8, eps_virgin=2. Domain: 60 mm cube.

Complex handling: with a complex dolfinx build, assemble gamma directly.
With a real build, solve the split system for (Vr, Vi):
  div(a grad Vr) - div(b grad Vi) = 0
  div(b grad Vr) + div(a grad Vi) = 0,  a=sigma, b=omega*eps0*eps_r
using a mixed function space; document which path ran in results.json.
Q_rf = 0.5? NO: match heatr3d.compute_qrf_3d conventions exactly - read that
function first and replicate its Q definition and its power renormalization
(fixed total absorbed power over the part at power_density_w_per_m3 basis)
so fields are comparable. Record the convention used.
"""
FREQ_HZ = 27.12e6
V_LO, V_HI = 860.0, 0.0
SIGMA_DOPED, EPS_DOPED = 0.04, 20.0
SIGMA_VIRGIN, EPS_VIRGIN = 1e-8, 2.0
L_DOMAIN = 0.060
EPS0 = 8.8541878128e-12
```

(plus the mesh/space/solve helpers the implementer writes; keep them in this
one file so every task shares one formulation.)

- [ ] **Step 2: Write run_plate.py**

Uniform virgin bed, plate BCs, tetrahedral mesh ~24 cells across. Gates:
max |V - linear(y)| < 1e-6 * 860 at the vertices; transverse std < 1e-9 * 860.
Append `{"task1": {"v_err_max": ..., "transverse_std": ..., "gate_ok": ...,
"wall_s": ..., "n_dofs": ...}}` to results.json.

- [ ] **Step 3: Run and check**

Run: `heatr3d_d1_spike/env/bin/python heatr3d_d1_spike/run_plate.py`
Expected: gate_ok true. This is the same characterization heatr3d passed at
1.25e-11 V; a FEM failure here is a formulation bug, not a discretization
property - fix before proceeding.

- [ ] **Step 4: Commit**

```bash
git add heatr3d_d1_spike/
git commit -m "feat(d1): dolfinx EQS formulation + parallel-plate gate"
```

### Task 2: Extrusion anchor - extruded circle vs heatr3d (the fidelity core)

**Files:**
- Create: `heatr3d_d1_spike/run_extruded_circle.py`
- Create: `heatr3d_d1_spike/run_heatr3d_reference.py`

- [ ] **Step 1: heatr3d reference fields**

`run_heatr3d_reference.py` (runs with the geo-prewarp venv, NOT the spike
env): extruded circle d=20 mm, full height, n=64 and n=96, EQS only
(build_gamma + solve_eqs_3d + compute_qrf_3d), saving V, |E|, Q_rf on the
mid-height plane plus the in-part Q_rf histogram to
`heatr3d_d1_spike/ref_heatr3d_circle_n{64,96}.npz`, and timing/memory to
results.json (`task2_ref`).

- [ ] **Step 2: dolfinx same case**

gmsh cylinder (d=20 mm, full-height extrusion) inside the 60 mm box; two
mesh resolutions with in-part element sizes chosen to match n=64 and n=96
voxel counts-in-part within ~20% (record actual dof counts). Solve EQS,
sample the SAME mid-height plane on the voxel grid points (interpolation),
save `d1_circle_{coarse,fine}.npz`.

- [ ] **Step 3: Compare (the gate)**

In-part mid-plane comparison, both resolutions: relative L2 error of Q_rf
pattern (normalized to unit mean over the part) dolfinx-vs-heatr3d, and the
same for |E|. Gate: agreement improves or holds with refinement and the
fine-mesh normalized-Q_rf L2 difference < 10% (pattern agreement; absolute
scale is fixed by the shared power normalization). Append `task2` metrics +
wall/memory to results.json. If the gate fails, investigate boundary-layer
resolution near the part surface before concluding; document either way.

- [ ] **Step 4: Commit**

### Task 3: Corner behavior - extruded square refinement study (the voxel-killer test)

**Files:**
- Create: `heatr3d_d1_spike/run_corner_study.py`

- [ ] **Step 1: The study**

Extruded square 20 mm in both engines, three refinements each (heatr3d
n=64/96/128 EQS-only; dolfinx meshes of comparable in-part resolution plus
one with local corner refinement). Metric: max in-part Q_rf within 1 mm of
a vertical corner edge, and the p99/mean Q_rf ratio, vs refinement.
Expected physics: the continuum solution has a genuine (integrable) edge
singularity, so BOTH engines grow the max under refinement; the question is
CONTROL: dolfinx with local refinement should show a smooth, predictable
growth law (power-law fit exponent stable across refinements) and unchanged
p99/mean away from the edge, while the voxel staircase produces erratic,
orientation-dependent jumps. Fit and record both growth sequences.

- [ ] **Step 2: Gate**

Not pass/fail physics (the singularity is real); the gate is DIAGNOSTIC
QUALITY: dolfinx growth-law fit R^2 > 0.98 across its refinements, and the
comparison table lands in results.json (`task3`). This section's numbers
feed the S2 metric decision directly.

- [ ] **Step 3: Commit**

### Task 4: Scale test - the n=200-equivalent solve heatr3d cannot do

**Files:**
- Create: `heatr3d_d1_spike/run_scale_test.py`

- [ ] **Step 1: The solve**

Extruded circle with dolfinx at in-part resolution equivalent to voxel
n=200 (~8e6-voxel domain; the FEM mesh needs far fewer dofs for equal
in-part resolution because the powder bed can be graded coarse - record
the dof count that achieves 0.3 mm in-part element size). Iterative KSP
(GMRES/BiCGSTAB + AMG via PETSc gamg or hypre if present). Gates: solve
completes on this machine (< 34 GB RSS, record actual), wall time recorded,
and mid-plane fields consistent with Task 2's fine solution within 5%
(pattern L2). Append `task4` to results.json. If it cannot complete, that
is a scoring datum, not a plan failure - record how it fails.

- [ ] **Step 2: Commit**

### Task 5: Adjoint-readiness demo (north-star criterion)

**Files:**
- Create: `heatr3d_d1_spike/run_adjoint_demo.py`

- [ ] **Step 1: The demo**

On the Task 2 coarse mesh: define sigma as a spatially varying Function on
the part, objective J = integral over the part of (Q_rf - mean Q_rf)^2
(uniformity of heating - a toy stand-in for the inverse-grading objective).
Compute dJ/dsigma with UFL/dolfinx-adjoint machinery (or hand-assembled
adjoint solve if dolfinx-adjoint is unavailable for this version - record
which). Gate: the adjoint gradient matches central finite differences on 5
randomly chosen sigma dofs to < 1% relative (the FD-gate discipline).
Append `task5` (per-dof FD vs adjoint values) to results.json.

- [ ] **Step 2: Commit**

### Task 6: D1 decision report

**Files:**
- Create: `heatr3d_d1_spike/D1_DECISION.md`

- [ ] **Step 1: Write the report from results.json only**

Sections: environment reality (install path that worked, complex vs split);
fidelity (Task 2 numbers); corner control (Task 3 table); scale (Task 4 vs
heatr3d's documented ceilings and 29.8 GB direct-LU estimate);
adjoint-readiness (Task 5 FD gate); effort log (honest hours per task);
recommendation (a) / (b) / (c) with reasoning tied to the numbers; and
what adopting the recommendation implies for S1 completion (large-N EQS),
S2 design, and the P2 mechanics engine. End with a sign-off line for Matt.

- [ ] **Step 2: Commit and present**

```bash
git add heatr3d_d1_spike/
git commit -m "docs(d1): D1 decision report"
```

---

## Self-review notes

- Spec coverage: reproduces the S3 extrusion-anchor case (Task 2), corner
  behavior under refinement (Task 3), cost (Task 4), effort (Task 6 log),
  adjoint criterion from the north-star amendment (Task 5). COMSOL
  comparison is deliberately thin here (the exports are 2-D mid-plane
  extractions); the full COMSOL 3-D anchor remains in Gate S3 where it
  belongs - the spike compares engines to each other and to the analytic
  plate.
- Gates are numeric and recorded in one JSON; the report cannot transcribe
  numbers from prose.
- Escape hatches: env failure is a finding (Task 0), scale failure is a
  datum (Task 4), real-build split formulation is documented not fatal.
- Time-box is binding: one focused week of cumulative effort, then report
  with what exists.
