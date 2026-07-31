"""Shared D1 EQS pieces (dolfinx). One formulation for every EQS task.

=============================================================================
CONVENTIONS REPLICATED FROM heatr3d.py  (read verbatim from the repo-root file
before writing this module; quoted line numbers are from the 2026-07-31 state)
=============================================================================

1. gamma  --  heatr3d.build_gamma (heatr3d.py:442-504)
       omega = 2*pi*p.freq_hz
       gamma = sigma + 1j * omega * EPS0 * eps_r
   with Params defaults (heatr3d.py:47-59):
       freq_hz 27.12e6, sigma_doped 0.04, eps_doped 20.0,
       sigma_virgin 1e-8, eps_virgin 2.0, EPS0 = 8.8541878128e-12.
   The binary path (edge_width_m = 0, premix_frac = 0) is what this spike
   replicates: sigma = sigma_virgin + part*(sigma_doped - sigma_virgin),
   eps likewise. NOT replicated here: the erf edge regularization
   (_material_fraction) and the premix variants -- both are voxel-grid
   remedies for the staircase boundary, which is precisely the artifact a
   conforming FEM mesh is being spiked to remove.

2. EQS boundary value problem  --  heatr3d.solve_eqs_3d (heatr3d.py:286-376)
       div(gamma grad V) = 0                       (source-free, complex V)
       V = p.v_lo = 860 V   on the y_min electrode plane
       V = p.v_hi =   0 V   on the y_max electrode plane
       zero-flux (natural/Neumann) on the four x and z walls.
   heatr3d discretizes this as a cell-centered finite volume with a HARMONIC
   mean of gamma on each face (_harmonic, heatr3d.py:250) and h^2 scaling,
   plus a 1e-18 identity shift for regularization. DEVIATION (unavoidable and
   intended): the FEM form uses the Galerkin weak form
       integral gamma grad(V) . grad(w) dV = 0
   with element-wise gamma (DG0). Face harmonic averaging has no FEM analogue;
   the two agree in the continuum limit, and for the uniform-material plate
   case of Task 1 they are identical.
   Second DEVIATION: heatr3d's grid is CELL-CENTERED, so its electrode planes
   sit at y = -L/2 + h/2 and y = +L/2 - h/2, i.e. the imposed potential drop
   spans L - h, not L. The FEM mesh puts the Dirichlet planes on the true
   domain faces y = -+L/2. Analytic comparisons here use the FEM plate
   positions; any future heatr3d-vs-dolfinx field comparison must account for
   this h-dependent gauge difference (it is an O(h/L) effect: 1.6% at n = 64).

3. Q_rf  --  heatr3d.compute_qrf_3d (heatr3d.py:379-406)
       E    = -grad V                       (np.gradient, edge_order=1)
       e2   = Re(E . conj(E))               (a real quantity)
       Q    = 0.5 * Re(gamma * e2) = 0.5 * sigma * |E|^2
              -- only Re(gamma) heats; the displacement (eps) part does not.
       Q    = clip(nan_to_num(Q), 0, None)  (subgradient-safe non-negativity)
       Q[~doped] = 0                        (premix=False, the default:
                                             jet-only absorption)
   With V = Vr + i Vi,  |E|^2 = |grad Vr|^2 + |grad Vi|^2.

4. POWER RENORMALIZATION BASIS  --  compute_qrf_3d (heatr3d.py:400-405)
       p_target = p.power_density_w_per_m3 * (doped_voxel_count * dV)
                = power_density_w_per_m3 * V_doped        [W]
       Q *= p_target / integral(Q dV)      (integral over the WHOLE domain)
   with power_density_w_per_m3 = 10.0 / (pi * 0.010^2 * 0.020) W/m^3
   = 10 W into the 2-D reference part (20 mm circle x 20 mm depth). So the
   absolute Q scale is FIXED BY THE DOPED VOLUME, not by the 860 V drive:
   the applied voltage sets only the PATTERN. This is the "3-D power-density
   basis" the plan warns never to compare against the 2.5-D drive basis.
   For premix=False the renormalization integral has support only inside the
   part, so it is equivalent to renormalizing over the part.

Complex handling
----------------
With a complex-scalar dolfinx/PETSc build, gamma is assembled directly as one
complex bilinear form. With a real build, the identical physics is solved as
the split system for (Vr, Vi) on a mixed space, a = sigma, b = omega*eps0*eps_r:
    div(a grad Vr) - div(b grad Vi) = 0
    div(b grad Vr) + div(a grad Vi) = 0
Which path ran is recorded in results.json under "scalar_path".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jit_fix                       # must precede dolfinx (see jit_fix docstring)
JIT_FIX = jit_fix.apply()

import numpy as np
import ufl
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# --- heatr3d.Params defaults (verbatim values) ------------------------------
FREQ_HZ = 27.12e6
V_LO, V_HI = 860.0, 0.0
SIGMA_DOPED, EPS_DOPED = 0.04, 20.0
SIGMA_VIRGIN, EPS_VIRGIN = 1e-8, 2.0
L_DOMAIN = 0.060
EPS0 = 8.8541878128e-12
# heatr3d.py:59 -- 10 W into a 20 mm circle x 20 mm depth
POWER_DENSITY_W_PER_M3 = 10.0 / (np.pi * 0.010 ** 2 * 0.020)

OMEGA = 2.0 * np.pi * FREQ_HZ

IS_COMPLEX = np.issubdtype(dolfinx.default_scalar_type, np.complexfloating)
SCALAR_PATH = "complex" if IS_COMPLEX else "real_split"


def gamma_of(sigma: float | np.ndarray, eps_r: float | np.ndarray):
    """heatr3d.build_gamma: gamma = sigma + j*omega*eps0*eps_r."""
    return sigma + 1j * OMEGA * EPS0 * eps_r


def b_of(eps_r: float | np.ndarray):
    """Imaginary part of gamma: b = omega * eps0 * eps_r."""
    return OMEGA * EPS0 * eps_r


# --------------------------------------------------------------------------- #
# API shims (dolfinx moved these between 0.6 and 0.9)
# --------------------------------------------------------------------------- #
def functionspace(msh, element):
    fn = getattr(fem, "functionspace", None) or getattr(fem, "FunctionSpace")
    return fn(msh, element)


_PREFIX_COUNTER = [0]


def _solve_linear(a, L, bcs, petsc_options=None):
    """dolfinx 0.11 made petsc_options_prefix a required keyword; older
    versions do not accept it. Probe, don't assume."""
    opts = petsc_options or {"ksp_type": "preonly", "pc_type": "lu"}
    _PREFIX_COUNTER[0] += 1
    pfx = f"d1_{_PREFIX_COUNTER[0]}_"
    try:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts,
                                petsc_options_prefix=pfx)
    except TypeError:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts)
    out = problem.solve()
    return out[0] if isinstance(out, (tuple, list)) else out


# --------------------------------------------------------------------------- #
# Mesh + materials
# --------------------------------------------------------------------------- #
def box_mesh(n: int, L: float = L_DOMAIN, comm=MPI.COMM_WORLD):
    """Tetrahedral mesh of the cubic chamber [-L/2, L/2]^3, n cells per axis."""
    return mesh.create_box(
        comm,
        [np.array([-L / 2, -L / 2, -L / 2]), np.array([L / 2, L / 2, L / 2])],
        [n, n, n], cell_type=mesh.CellType.tetrahedron)


@dataclass
class Materials:
    """DG0 sigma / eps_r fields plus the doped-cell indicator."""
    sigma: fem.Function
    eps_r: fem.Function
    doped: fem.Function          # 1.0 in the part, 0.0 outside (DG0)
    dg0: object


def materials(msh, in_part: Callable[[np.ndarray], np.ndarray] | None = None) -> Materials:
    """Build DG0 material fields. `in_part(midpoints[3,ncell]) -> bool array`;
    None = uniform virgin bed (the Task 1 plate case)."""
    Q = functionspace(msh, ("DG", 0))
    sig, eps, dop = fem.Function(Q), fem.Function(Q), fem.Function(Q)
    ncell = msh.topology.index_map(msh.topology.dim).size_local + \
        msh.topology.index_map(msh.topology.dim).num_ghosts
    cells = np.arange(ncell, dtype=np.int32)
    if in_part is None:
        m = np.zeros(ncell, dtype=bool)
    else:
        mp = dolfinx.mesh.compute_midpoints(msh, msh.topology.dim, cells)
        m = np.asarray(in_part(mp.T), dtype=bool)
    st = dolfinx.default_scalar_type
    # heatr3d binary blend: value = virgin + part*(doped - virgin)
    sig.x.array[:] = np.where(m, SIGMA_DOPED, SIGMA_VIRGIN).astype(st)
    eps.x.array[:] = np.where(m, EPS_DOPED, EPS_VIRGIN).astype(st)
    dop.x.array[:] = m.astype(st)
    return Materials(sig, eps, dop, Q)


# --------------------------------------------------------------------------- #
# EQS solve
# --------------------------------------------------------------------------- #
def _electrode_dofs(V, msh, L: float, sub=None):
    tol = 1e-9 * L
    lo = mesh.locate_entities_boundary(msh, msh.topology.dim - 1,
                                       lambda x: np.isclose(x[1], -L / 2, atol=tol))
    hi = mesh.locate_entities_boundary(msh, msh.topology.dim - 1,
                                       lambda x: np.isclose(x[1], +L / 2, atol=tol))
    fdim = msh.topology.dim - 1
    if sub is None:
        return (fem.locate_dofs_topological(V, fdim, lo),
                fem.locate_dofs_topological(V, fdim, hi))
    return (fem.locate_dofs_topological((sub[0], sub[1]), fdim, lo),
            fem.locate_dofs_topological((sub[0], sub[1]), fdim, hi))


def solve_eqs(msh, mats: Materials, degree: int = 1, L: float = L_DOMAIN,
              petsc_options=None):
    """Solve div(gamma grad V)=0 with the heatr3d electrode BCs.

    Returns (Vr, Vi) as real-valued CG functions on a scalar space, whichever
    scalar path the build supports, so downstream code is path-agnostic."""
    st = dolfinx.default_scalar_type
    if IS_COMPLEX:
        W = functionspace(msh, ("Lagrange", degree))
        u, w = ufl.TrialFunction(W), ufl.TestFunction(W)
        gam = mats.sigma + 1j * OMEGA * EPS0 * mats.eps_r
        a = ufl.inner(gam * ufl.grad(u), ufl.grad(w)) * ufl.dx
        Lf = ufl.inner(fem.Constant(msh, st(0.0)), w) * ufl.dx
        d_lo, d_hi = _electrode_dofs(W, msh, L)
        bcs = [fem.dirichletbc(st(V_LO), d_lo, W), fem.dirichletbc(st(V_HI), d_hi, W)]
        Vh = _solve_linear(a, Lf, bcs, petsc_options)
        Wr = functionspace(msh, ("Lagrange", degree))
        Vr, Vi = fem.Function(Wr), fem.Function(Wr)
        Vr.x.array[:] = np.real(Vh.x.array)
        Vi.x.array[:] = np.imag(Vh.x.array)
        return Vr, Vi

    # ---- real build: split (Vr, Vi) mixed system --------------------------
    import basix.ufl
    el = basix.ufl.element("Lagrange", msh.basix_cell(), degree)
    ME = functionspace(msh, basix.ufl.mixed_element([el, el]))
    ur, ui = ufl.TrialFunctions(ME)
    wr, wi = ufl.TestFunctions(ME)
    a_c, b_c = mats.sigma, OMEGA * EPS0 * mats.eps_r
    a = (ufl.inner(a_c * ufl.grad(ur), ufl.grad(wr))
         - ufl.inner(b_c * ufl.grad(ui), ufl.grad(wr))
         + ufl.inner(b_c * ufl.grad(ur), ufl.grad(wi))
         + ufl.inner(a_c * ufl.grad(ui), ufl.grad(wi))) * ufl.dx
    Lf = ufl.inner(fem.Constant(msh, st(0.0)), wr) * ufl.dx \
        + ufl.inner(fem.Constant(msh, st(0.0)), wi) * ufl.dx
    S0, m0 = ME.sub(0).collapse()
    S1, m1 = ME.sub(1).collapse()
    lo0, hi0 = _electrode_dofs(None, msh, L, sub=(ME.sub(0), S0))
    lo1, hi1 = _electrode_dofs(None, msh, L, sub=(ME.sub(1), S1))
    f_lo, f_hi = fem.Function(S0), fem.Function(S0)
    f_lo.x.array[:] = st(V_LO)
    f_hi.x.array[:] = st(V_HI)
    f_zero = fem.Function(S1)
    f_zero.x.array[:] = st(0.0)
    bcs = [fem.dirichletbc(f_lo, lo0, ME.sub(0)),
           fem.dirichletbc(f_hi, hi0, ME.sub(0)),
           fem.dirichletbc(f_zero, lo1, ME.sub(1)),
           fem.dirichletbc(f_zero, hi1, ME.sub(1))]
    Vh = _solve_linear(a, Lf, bcs, petsc_options)
    Vr = Vh.sub(0).collapse()
    Vi = Vh.sub(1).collapse()
    return Vr, Vi


# --------------------------------------------------------------------------- #
# Q_rf (heatr3d.compute_qrf_3d conventions)
# --------------------------------------------------------------------------- #
def qrf_expression(Vr, Vi, mats: Materials):
    """UFL for Q = 0.5 * sigma * |E|^2, E = -grad V (heatr3d convention 3).

    |E|^2 = |grad Vr|^2 + |grad Vi|^2; the sign of E is irrelevant here."""
    return 0.5 * mats.sigma * (ufl.inner(ufl.grad(Vr), ufl.grad(Vr))
                               + ufl.inner(ufl.grad(Vi), ufl.grad(Vi)))


def qrf_dg0(msh, Vr, Vi, mats: Materials, premix: bool = False):
    """Project Q onto DG0, apply the heatr3d masking + power renormalization.

    Returns (Q_dg0_function, scale_factor, p_target_W)."""
    Q = mats.dg0
    q = fem.Function(Q)
    expr = qrf_expression(Vr, Vi, mats)
    # DG0 projection == cell average; do it with the exact cell-wise formula
    p_ = ufl.TrialFunction(Q)
    w_ = ufl.TestFunction(Q)
    a = ufl.inner(p_, w_) * ufl.dx
    Lf = ufl.inner(expr, w_) * ufl.dx
    q = _solve_linear(a, Lf, [], {"ksp_type": "preonly", "pc_type": "jacobi"})
    arr = np.real(q.x.array)
    arr = np.clip(np.nan_to_num(arr), 0.0, None)          # heatr3d clip
    if not premix:
        arr = np.where(np.real(mats.doped.x.array) > 0.5, arr, 0.0)
    q.x.array[:] = arr.astype(dolfinx.default_scalar_type)
    # power renormalization on the heatr3d basis
    v_doped = fem.assemble_scalar(fem.form(mats.doped * ufl.dx))
    v_doped = float(np.real(msh.comm.allreduce(v_doped, op=MPI.SUM)))
    p_target = POWER_DENSITY_W_PER_M3 * v_doped
    p_now = fem.assemble_scalar(fem.form(q * ufl.dx))
    p_now = float(np.real(msh.comm.allreduce(p_now, op=MPI.SUM)))
    scale = (p_target / p_now) if p_now > 1e-18 else 1.0
    q.x.array[:] = q.x.array * dolfinx.default_scalar_type(scale)
    return q, scale, p_target
