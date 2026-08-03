"""Gate: dolfinx imports, reports scalar type, and solves -laplace(u)=1 on a
unit cube with u=0 walls; the max must match the known reference 0.0562+-5%
(center value of the unit-cube Poisson problem).

API-adaptation note: the exact dolfinx entry points move between versions
(fem.functionspace vs fem.FunctionSpace; LinearProblem in dolfinx.fem.petsc;
mesh.exterior_facet_indices requires the (dim-1, dim) connectivity to exist;
dolfinx 0.11 made `petsc_options_prefix` a REQUIRED keyword-only argument of
LinearProblem and lets .solve() return a sequence). The GATE is the number,
not the call sequence, so the calls below probe for whatever the resolved
build provides.
"""
import json
import sys
from pathlib import Path

import jit_fix                       # must precede dolfinx (see jit_fix docstring)
JIT_FIX = jit_fix.apply()

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

REF_UMAX = 0.0562          # unit-cube -lap(u)=1, u=0 walls; center value
REL_TOL = 0.05


def _functionspace(msh, element):
    """fem.functionspace (>=0.7) or fem.FunctionSpace (<=0.6)."""
    fn = getattr(fem, "functionspace", None) or getattr(fem, "FunctionSpace")
    return fn(msh, element)


def _exterior_facets(msh):
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)     # required before the scan
    return mesh.exterior_facet_indices(msh.topology)


def main() -> int:
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 24, 24, 24)
    V = _functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(fem.Constant(msh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, _exterior_facets(msh))
    bc = fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, V)
    opts = {"ksp_type": "preonly", "pc_type": "lu"}
    try:
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=opts,
                                petsc_options_prefix="d1_check_")
    except TypeError:                       # dolfinx <= 0.10: no prefix argument
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=opts)
    solved = problem.solve()
    uh = solved[0] if isinstance(solved, (tuple, list)) else solved
    umax = float(np.abs(uh.x.array).max())
    scalar = str(dolfinx.default_scalar_type)
    ok = abs(umax - REF_UMAX) / REF_UMAX < REL_TOL
    out = {"task0": {"dolfinx_version": dolfinx.__version__,
                     "petsc_scalar_type": scalar,
                     "complex_build": "complex" in scalar,
                     "poisson_umax": umax,
                     "poisson_ref": REF_UMAX,
                     "poisson_rel_err": abs(umax - REF_UMAX) / REF_UMAX,
                     "gate_ok": bool(ok),
                     "jit_space_fix": JIT_FIX}}
    p = Path(__file__).parent / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(out)
    p.write_text(json.dumps(d, indent=1))
    print(out)
    assert ok, "Poisson gate failed"
    return 0


if __name__ == "__main__":
    sys.exit(main())
