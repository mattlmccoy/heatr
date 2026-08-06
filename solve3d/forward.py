"""solve3d Phase A forward model: complex EQS + enthalpy thermal-phase march.

RUNS IN THE SPIKE ENV ONLY (dolfinx 0.11 complex build):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 heatr3d_d1_spike/env/bin/python ...

`jit_fix` MUST be imported and applied before dolfinx (Dropbox-path FFCx shim);
that is done at the top of this module, exactly as heatr3d_d1_spike/eqs_common.py
does it.

=============================================================================
PORTED SEMANTICS (source anchors read at heatr3d.py commit 699ed79)
=============================================================================
* gamma = sigma + 1j*omega*EPS0*eps_r          -- heatr3d.build_gamma (binary path)
* div(gamma grad V) = 0, V=v_lo on y_min, v_hi on y_max, Neumann elsewhere
                                               -- heatr3d.solve_eqs_3d
* Q = 0.5*Re(gamma |E|^2) = 0.5*sigma|E|^2, clipped >= 0, zeroed outside the
  part, then renormalized so integral(Q dV) = power_density_w_per_m3 * V_part
                                               -- heatr3d.compute_qrf_3d
  ONLY the corrected (part-confined) drive exists here. heatr3d's
  qrf_gradient="legacy" cross-interface np.gradient has no analogue and is
  forbidden by construction (spec sec 4). On a conforming FEM mesh grad(V) is
  cell-wise constant per element and elements never straddle the interface, so
  the DG0 Q is a part-confined quantity by construction -- this is precisely
  what heatr3d's `qrf_gradient="masked"` mode approximates on voxels.
* H(T) piecewise linear, latent plateau ramped over the melt window; exact
  inversion                                    -- heatr3d.enthalpy_from_T /
                                                  heatr3d.T_from_enthalpy
* sigma(T, rho) coupling + clip bounds         -- heatr3d.apply_sigma_coupling
* in-march EQS re-solve schedule on ABSOLUTE time, fixed-power renormalization
  on every re-solve, drift-tolerance skip      -- heatr3d.run (S4-COUPLING)

DELIBERATE DEVIATIONS (named, not hidden):
1. Electrode gauge. heatr3d's cell-centred Dirichlet rows span L-h; the FEM
   planes sit on the true faces and span L. This is an O(h/L) scale factor on
   E (1.6 % at n=64) that CANCELS out of Q because both engines renormalize to
   the same fixed absorbed power (heatr3d_d1_spike/metrics.unit_mean note).
2. Face harmonic averaging of gamma / k has no FEM analogue; the Galerkin form
   uses element-wise (DG0) coefficients. The two agree in the continuum limit.
3. Part geometry is CONFORMING (exact cylinder / prism), not a voxel staircase.
   Part volumes therefore differ from heatr3d's by the staircase error, which
   changes the absolute power target (p_target = power_density * V_part). This
   is recorded in every parity artifact, never silently absorbed.
4. The thermal march is mass-LUMPED explicit Euler on nodal volumetric
   enthalpy -- the FEM analogue of heatr3d's explicit cell-centred FV update.
   Lumping is what makes the pointwise H->T inversion (heatr3d's exact
   piecewise-linear inverse) well defined node-by-node.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_SPIKE = Path(__file__).resolve().parents[1] / "heatr3d_d1_spike"
if str(_SPIKE) not in sys.path:
    sys.path.insert(0, str(_SPIKE))

import jit_fix                       # noqa: E402  MUST precede dolfinx
JIT_FIX = jit_fix.apply()

import numpy as np                   # noqa: E402
import ufl                           # noqa: E402
import dolfinx                       # noqa: E402
from dolfinx import fem, mesh        # noqa: E402
from dolfinx.fem.petsc import LinearProblem   # noqa: E402
from mpi4py import MPI               # noqa: E402

DOLFINX_VERSION = dolfinx.__version__
IS_COMPLEX = bool(np.issubdtype(dolfinx.default_scalar_type, np.complexfloating))
SCALAR_PATH = "complex" if IS_COMPLEX else "real_split"

EPS0 = 8.8541878128e-12              # heatr3d.EPS0, verbatim
R_GAS = 8.314                        # heatr3d.R_GAS, verbatim (densification only)


@dataclass(frozen=True)
class ForwardParams:
    """The heatr3d.Params fields Phase A depends on, with heatr3d's defaults.

    Frozen (coding-style rule: immutable config). Every value below is copied
    verbatim from heatr3d.Params; `phase_update` is pinned to "enthalpy"
    because Phase A only ports the energy-conserving scheme (the legacy
    apparent_cp path is explicitly out of scope, spec sec 5 / plan Task 3).
    """
    # --- drive / EQS ---
    freq_hz: float = 27.12e6
    v_lo: float = 860.0
    v_hi: float = 0.0
    sigma_doped: float = 0.04
    eps_doped: float = 20.0
    sigma_virgin: float = 1e-8
    eps_virgin: float = 2.0
    power_density_w_per_m3: float = 10.0 / (np.pi * 0.010 ** 2 * 0.020)
    # --- powder thermal ---
    k_powder: float = 0.197
    rho_powder: float = 490.0
    cp_powder: float = 1072.0
    # --- melt (doped solid -> liquid) ---
    k_solid: float = 0.10
    k_liquid: float = 0.26
    rho_solid: float = 460.0
    rho_liquid: float = 1010.0
    cp_solid: float = 2500.0
    cp_liquid: float = 3279.0
    latent_j_per_kg: float = 96700.0
    t_pc_c: float = 180.0
    dt_pc_c: float = 10.0
    rho_rel: float = 0.55
    ambient_c: float = 23.0
    preheat_c: float = 23.0
    conv_h: float = 5.0              # top face (y = +L/2) only
    # --- densification (heatr3d.Params dens_* fields, verbatim defaults;
    #     physics_dual solid-state Arrhenius creep + liquid viscous-capillary
    #     flow. Inert unless march_enthalpy(densify=True) is requested -- the
    #     Phase A melt-onset march never reads them. Ported for Stage A of the
    #     thermal-ceiling spec: the ceiling peak is an end-state quantity that
    #     only exists once the forward marches densify to rho_target.) ---
    dens_k0_ss: float = 0.005
    dens_ea_ss: float = 48000.0
    dens_phi_solid_exp: float = 0.8
    dens_phi_threshold: float = 0.01
    dens_phi_liq_exp: float = 1.0
    dens_geom_factor: float = 0.05
    dens_surface_tension: float = 0.03
    dens_particle_radius_m: float = 3.5e-5
    dens_eta_ref_pa_s: float = 8.0e3
    dens_eta_ref_temp_k: float = 458.15
    dens_eta_activation: float = 6.0e4
    dens_rho_exp: float = 1.0
    dens_max_drho_rate: float = 0.04   # per second
    # --- numerics ---
    dt_s: float = 0.05
    max_dt_step_c: float = 10.0
    temp_min_c: float = -50.0
    temp_max_c: float = 600.0
    phase_update: str = "enthalpy"   # pinned; see class docstring
    enforce_cfl: bool = True
    # --- S4-COUPLING (heatr3d 699ed79); all inert at these defaults ---
    eqs_update_interval_s: float = 0.0
    sigma_temp_coeff_per_K: float = 0.0
    sigma_density_coeff: float = 0.0
    sigma_ref_temp_c: float = 23.0
    eqs_resolve_drift_rtol: float = 0.0

    def __post_init__(self) -> None:
        if self.phase_update != "enthalpy":
            raise ValueError(
                "solve3d Phase A only implements phase_update='enthalpy'; "
                f"got {self.phase_update!r}")

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq_hz


# --------------------------------------------------------------------------- #
# dolfinx API shims + linear solve (lifted from heatr3d_d1_spike/eqs_common.py)
# --------------------------------------------------------------------------- #
def functionspace(msh, element):
    fn = getattr(fem, "functionspace", None) or getattr(fem, "FunctionSpace")
    return fn(msh, element)


_PREFIX_COUNTER = [0]

KSP_ITER = {"ksp_type": "gmres", "pc_type": "gamg",
            "ksp_rtol": "1e-10", "ksp_max_it": "500"}
KSP_LU = {"ksp_type": "preonly", "pc_type": "lu"}
KSP_JACOBI = {"ksp_type": "preonly", "pc_type": "jacobi"}


def _solve_linear(a, L, bcs, petsc_options=None):
    """dolfinx 0.11 made petsc_options_prefix required; older versions reject
    it. Probe, don't assume (eqs_common._solve_linear, verbatim behaviour)."""
    opts = petsc_options or KSP_LU
    _PREFIX_COUNTER[0] += 1
    pfx = f"solve3d_{_PREFIX_COUNTER[0]}_"
    try:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts,
                                petsc_options_prefix=pfx)
    except TypeError:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts)
    out = problem.solve()
    return out[0] if isinstance(out, (tuple, list)) else out


# --------------------------------------------------------------------------- #
# Geometry predicates for the Phase A anchor cases
# --------------------------------------------------------------------------- #
PART_DIAM_M = 0.020
L_DOMAIN = 0.060


def in_part_predicate(shape: str, diam: float = PART_DIAM_M):
    """Midpoint predicate matching heatr3d.make_geometry for the FULL-HEIGHT
    extrusions (`cylinder`; `square` with zspan = L). `mp` is (3, ncell)."""
    half = diam / 2.0
    if shape == "circle":
        return lambda mp: np.sqrt(mp[0] ** 2 + mp[1] ** 2) <= half
    if shape == "square":
        return lambda mp: (np.abs(mp[0]) <= half) & (np.abs(mp[1]) <= half)
    raise ValueError(f"unknown anchor shape {shape!r}")


# --------------------------------------------------------------------------- #
# Materials (heatr3d.build_gamma binary path)
# --------------------------------------------------------------------------- #
@dataclass
class Materials:
    """DG0 sigma / eps_r fields plus the doped-cell indicator and space."""
    sigma: "fem.Function"
    eps_r: "fem.Function"
    doped: "fem.Function"
    dg0: object
    mask: np.ndarray                 # bool, per local cell


def n_cells_local(msh) -> int:
    tdim = msh.topology.dim
    return (msh.topology.index_map(tdim).size_local
            + msh.topology.index_map(tdim).num_ghosts)


def build_materials(msh, in_part, p: ForwardParams) -> Materials:
    """heatr3d.build_gamma binary blend on DG0:
        sigma = sigma_virgin + part*(sigma_doped - sigma_virgin)
        eps_r = eps_virgin   + part*(eps_doped   - eps_virgin)
    The erf edge regularization and the premix variants are NOT ported: both
    are voxel-staircase remedies with no meaning on a conforming mesh."""
    Q = functionspace(msh, ("DG", 0))
    sig, eps, dop = fem.Function(Q), fem.Function(Q), fem.Function(Q)
    ncell = n_cells_local(msh)
    cells = np.arange(ncell, dtype=np.int32)
    if in_part is None:
        m = np.zeros(ncell, dtype=bool)
    else:
        mp = dolfinx.mesh.compute_midpoints(msh, msh.topology.dim, cells)
        m = np.asarray(in_part(mp.T), dtype=bool)
    st = dolfinx.default_scalar_type
    sig.x.array[:] = np.where(m, p.sigma_doped, p.sigma_virgin).astype(st)
    eps.x.array[:] = np.where(m, p.eps_doped, p.eps_virgin).astype(st)
    dop.x.array[:] = m.astype(st)
    return Materials(sig, eps, dop, Q, m)


def apply_sigma_coupling(mats: Materials, T_cells: np.ndarray,
                         rho_cells: np.ndarray, p: ForwardParams) -> np.ndarray:
    """Port of heatr3d.apply_sigma_coupling (clip bounds are the 2-D
    convention verbatim: 1e-4 and 25 x sigma_doped, inside the part only).

    Returns the coupled per-cell sigma array; the imaginary (displacement)
    part of gamma is untouched, exactly as heatr3d leaves it."""
    s0 = np.real(mats.sigma.x.array).astype(float)
    f = ((1.0 + p.sigma_temp_coeff_per_K * (np.asarray(T_cells, float) - p.sigma_ref_temp_c))
         * (1.0 + p.sigma_density_coeff * (np.asarray(rho_cells, float) - p.rho_rel)))
    s = np.nan_to_num(s0 * f, nan=p.sigma_doped,
                      posinf=SIGMA_COUPLING_CLIP_HI * p.sigma_doped,
                      neginf=SIGMA_COUPLING_CLIP_LO * p.sigma_doped)
    s = np.clip(s, SIGMA_COUPLING_CLIP_LO * p.sigma_doped,
                SIGMA_COUPLING_CLIP_HI * p.sigma_doped)
    return np.where(mats.mask, s, s0)


SIGMA_COUPLING_CLIP_LO = 1e-4        # heatr3d.SIGMA_COUPLING_CLIP_LO
SIGMA_COUPLING_CLIP_HI = 25.0        # heatr3d.SIGMA_COUPLING_CLIP_HI


# --------------------------------------------------------------------------- #
# EQS solve (heatr3d.solve_eqs_3d BVP, Galerkin weak form)
# --------------------------------------------------------------------------- #
def _electrode_dofs(W, msh, L: float):
    tol = 1e-9 * L
    fdim = msh.topology.dim - 1
    lo = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], -L / 2, atol=tol))
    hi = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], +L / 2, atol=tol))
    return (fem.locate_dofs_topological(W, fdim, lo),
            fem.locate_dofs_topological(W, fdim, hi))


def solve_eqs(msh, mats: Materials, p: ForwardParams, degree: int = 1,
              L: float = L_DOMAIN, petsc_options=None):
    """Solve div(gamma grad V) = 0 with heatr3d's electrode BCs.

    Returns (Vr, Vi) as real CG functions so downstream code is scalar-path
    agnostic. Phase A requires the complex build (asserted at import)."""
    if not IS_COMPLEX:
        raise RuntimeError(
            "solve3d Phase A requires the complex-scalar dolfinx build "
            "(heatr3d_d1_spike/env). The real split path is not ported.")
    st = dolfinx.default_scalar_type
    W = functionspace(msh, ("Lagrange", degree))
    u, w = ufl.TrialFunction(W), ufl.TestFunction(W)
    gam = mats.sigma + 1j * p.omega * EPS0 * mats.eps_r
    a = ufl.inner(gam * ufl.grad(u), ufl.grad(w)) * ufl.dx
    Lf = ufl.inner(fem.Constant(msh, st(0.0)), w) * ufl.dx
    d_lo, d_hi = _electrode_dofs(W, msh, L)
    bcs = [fem.dirichletbc(st(p.v_lo), d_lo, W),
           fem.dirichletbc(st(p.v_hi), d_hi, W)]
    Vh = _solve_linear(a, Lf, bcs, petsc_options or KSP_ITER)
    Vr, Vi = fem.Function(W), fem.Function(W)
    Vr.x.array[:] = np.real(Vh.x.array)
    Vi.x.array[:] = np.imag(Vh.x.array)
    return Vr, Vi


def dg0_project(expr, Q, msh):
    """L2 projection onto DG0 == exact cell average. For P1 V any function of
    grad(V) is already cell-wise constant, so this is exact, not smoothing."""
    p_, w_ = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    return _solve_linear(ufl.inner(p_, w_) * ufl.dx,
                         ufl.inner(expr, w_) * ufl.dx, [], KSP_JACOBI)


def qrf_dg0(msh, Vr, Vi, mats: Materials, p: ForwardParams) -> dict:
    """Q = 0.5*sigma*|E|^2 on DG0, part-confined, then renormalized so
    integral(Q dV) = power_density_w_per_m3 * V_part (heatr3d.compute_qrf_3d,
    premix=False basis).

    Only Re(gamma) heats; the displacement part does no work. |E|^2 =
    |grad Vr|^2 + |grad Vi|^2 (the sign of E is irrelevant).

    NOTE ON THE 'MASKED' CONVENTION: on a conforming mesh each element lies
    wholly inside or wholly outside the part and grad(V) is element-local, so
    no difference is ever taken across the material interface. That is the
    corrected drive by construction -- heatr3d's `qrf_gradient="masked"` is the
    voxel approximation of it. Legacy Q has no analogue here (spec sec 4)."""
    expr = 0.5 * mats.sigma * (ufl.inner(ufl.grad(Vr), ufl.grad(Vr))
                               + ufl.inner(ufl.grad(Vi), ufl.grad(Vi)))
    q = dg0_project(expr, mats.dg0, msh)
    arr = np.clip(np.nan_to_num(np.real(q.x.array)), 0.0, None)   # heatr3d clip
    arr = np.where(mats.mask, arr, 0.0)                           # premix=False
    q.x.array[:] = arr.astype(dolfinx.default_scalar_type)
    v_part = float(np.real(msh.comm.allreduce(
        fem.assemble_scalar(fem.form(mats.doped * ufl.dx)), op=MPI.SUM)))
    p_target = p.power_density_w_per_m3 * v_part
    p_raw = float(np.real(msh.comm.allreduce(
        fem.assemble_scalar(fem.form(q * ufl.dx)), op=MPI.SUM)))
    scale = (p_target / p_raw) if p_raw > 1e-18 else 1.0
    q.x.array[:] = q.x.array * dolfinx.default_scalar_type(scale)
    p_now = float(np.real(msh.comm.allreduce(
        fem.assemble_scalar(fem.form(q * ufl.dx)), op=MPI.SUM)))
    return {"q": q, "scale": scale, "p_target_w": p_target,
            "p_raw_w": p_raw, "p_now_w": p_now, "v_part_m3": v_part}


# --------------------------------------------------------------------------- #
# Point evaluation on heatr3d's voxel-centre grid
# --------------------------------------------------------------------------- #
def eval_at(fn, msh, pts: np.ndarray):
    """Evaluate `fn` at pts (N,3). Returns (values[N], n_missed); misses are
    NaN and COUNTED, never silently dropped (femutils.eval_points port)."""
    from dolfinx import geometry
    pts = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = np.zeros(pts.shape[0], dtype=np.int32)
    missed = np.zeros(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) == 0:
            missed[i] = True
        else:
            cells[i] = links[0]
    vals = np.real(fn.eval(pts, cells)).reshape(-1)
    vals[missed] = np.nan
    return vals, int(missed.sum())


def midplane_part_points(ref):
    """heatr3d mid-plane in-part voxel centres from an anchor npz, as (N,3)
    points plus the 2-D selection mask used to unpack the reference field."""
    x, y = np.asarray(ref["x"]), np.asarray(ref["y"])
    n = int(ref["n"])
    k = n // 2
    z_mid = float(np.asarray(ref["z"])[k])
    sel = np.asarray(ref["part"])[:, :, k].astype(bool)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([X[sel], Y[sel],
                           np.full(int(sel.sum()), z_mid)])
    return pts, sel


# --------------------------------------------------------------------------- #
# One-call EQS case (mesh + solve + drive)
# --------------------------------------------------------------------------- #
def eqs_case(shape: str, target_nodes_in_part: int, lc0: float,
             p: ForwardParams | None = None, petsc_options=None) -> dict:
    """Mesh the anchor at heatr3d-matched in-part resolution, solve the EQS and
    build the corrected drive. Meshing uses the D1-proven
    heatr3d_d1_spike/mesh_gmsh.match_lc (in-part NODE count matched to
    heatr3d's in-part VOXEL count, i.e. matched UNKNOWNS)."""
    import time

    import mesh_gmsh as mg

    p = p or ForwardParams()
    kind = "cylinder" if shape == "circle" else "square"
    t0 = time.perf_counter()
    msh, info, hist = mg.match_lc(kind, int(target_nodes_in_part), float(lc0))
    t_mesh = time.perf_counter() - t0
    mats = build_materials(msh, in_part_predicate(shape), p)
    t0 = time.perf_counter()
    Vr, Vi = solve_eqs(msh, mats, p, petsc_options=petsc_options)
    t_solve = time.perf_counter() - t0
    drive = qrf_dg0(msh, Vr, Vi, mats, p)
    out = {"msh": msh, "info": info, "match_history": hist, "mats": mats,
           "Vr": Vr, "Vi": Vi, "wall_mesh_s": t_mesh, "wall_solve_s": t_solve,
           "n_dofs_total": int(info.n_nodes_total),
           "n_cells_total": int(info.n_cells_total),
           "n_nodes_in_part": int(info.n_nodes_in_part),
           "lc_part_m": float(info.lc_part)}
    out.update(drive)
    return out


# --------------------------------------------------------------------------- #
# Enthalpy / phase (verbatim ports of heatr3d.py)
# --------------------------------------------------------------------------- #
def phase_fraction(T, p: ForwardParams):
    """heatr3d.phase_fraction: melt fraction phi and its derivative dphi/dT."""
    Te = np.array(T, dtype=float, copy=True)
    arg = (Te - p.t_pc_c) / p.dt_pc_c
    phi = np.clip(arg + 0.5, 0.0, 1.0)
    dphi = np.where(np.abs(arg) <= 0.5, 1.0 / p.dt_pc_c, 0.0)
    return phi, dphi


def densify_rate(T, phi, rho_rel, p: ForwardParams):
    """Pure-numpy port of heatr3d.densify_rate (source anchor heatr3d.py:707).

    physics_dual densification rate d(rho_rel)/dt (>= 0): solid-state Arrhenius
    creep + liquid viscous-capillary flow, both gated by available porosity
    (1 - rho_rel)^dens_rho_exp. Verbatim arithmetic, verbatim R_GAS = 8.314; a
    unit test pins it bit-for-bit against heatr3d.densify_rate. This is the ONE
    piece of densification physics Stage A ports; the march below evolves rho
    with it and marches to rho_target so the ceiling peak (an end-state
    quantity) exists in the solve3d forward.
    """
    Tk = np.maximum(np.array(T, dtype=float, copy=True) + 273.15, 1.0)
    rho_term = np.power(np.clip(1.0 - rho_rel, 0.0, 1.0), p.dens_rho_exp)
    # solid-state
    kss = p.dens_k0_ss * np.exp(-p.dens_ea_ss / (R_GAS * Tk))
    ss_drive = np.power(np.clip(1.0 - phi, 0.0, 1.0), p.dens_phi_solid_exp)
    # liquid viscous-capillary
    eta = p.dens_eta_ref_pa_s * np.exp(
        p.dens_eta_activation / R_GAS * (1.0 / Tk - 1.0 / p.dens_eta_ref_temp_k))
    kliq = p.dens_geom_factor * p.dens_surface_tension / (
        np.maximum(eta, 1e-12) * p.dens_particle_radius_m)
    phi_act = np.clip((phi - p.dens_phi_threshold)
                      / max(1.0 - p.dens_phi_threshold, 1e-9), 0.0, 1.0)
    liq_drive = np.power(phi_act, p.dens_phi_liq_exp)
    return (kss * ss_drive + kliq * liq_drive) * rho_term


def enthalpy_from_T(T, rho_cp, rho_L, p: ForwardParams):
    """heatr3d.enthalpy_from_T: volumetric enthalpy [J/m^3], piecewise linear
    (sensible slope rho_cp everywhere plus the latent plateau rho_L ramped
    linearly across the melt window)."""
    T = np.asarray(T, dtype=np.float64)
    lo = p.t_pc_c - p.dt_pc_c / 2.0
    frac = np.clip((T - lo) / p.dt_pc_c, 0.0, 1.0)
    return rho_cp * T + rho_L * frac


def T_from_enthalpy(H, rho_cp, rho_L, p: ForwardParams):
    """heatr3d.T_from_enthalpy: exact inverse of enthalpy_from_T."""
    H = np.asarray(H, dtype=np.float64)
    lo = p.t_pc_c - p.dt_pc_c / 2.0
    H_lo = rho_cp * lo
    H_hi = rho_cp * (lo + p.dt_pc_c) + rho_L
    T_below = H / rho_cp
    T_window = (H + rho_L * lo / p.dt_pc_c) / (rho_cp + rho_L / p.dt_pc_c)
    T_above = (H - rho_L) / rho_cp
    return np.where(H <= H_lo, T_below, np.where(H >= H_hi, T_above, T_window))


# --------------------------------------------------------------------------- #
# Transient enthalpy thermal-phase march
# --------------------------------------------------------------------------- #
def box_mesh(n: int, L: float = L_DOMAIN, comm=None):
    """Tetrahedral mesh of the cubic chamber [-L/2, L/2]^3 (eqs_common.box_mesh)."""
    comm = comm if comm is not None else MPI.COMM_WORLD
    return mesh.create_box(
        comm, [np.array([-L / 2, -L / 2, -L / 2]), np.array([L / 2, L / 2, L / 2])],
        [n, n, n], cell_type=mesh.CellType.tetrahedron)


def _top_facet_measure(msh, L: float):
    """ds restricted to the OPEN TOP FACE y = +L/2 (heatr3d's `top` slice).

    heatr3d applies q_conv = conv_h*(T - preheat)/h in the last y layer, i.e. a
    SURFACE flux h*(T - T_inf) spread over that layer's thickness. The FEM
    analogue is the Robin surface integral -- the same physics without the /h
    voxel-thickness bookkeeping."""
    fdim = msh.topology.dim - 1
    tol = 1e-9 * L
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], +L / 2, atol=tol))
    facets = np.sort(np.asarray(facets, dtype=np.int32))
    mt = mesh.meshtags(msh, fdim, facets,
                       np.ones(facets.size, dtype=np.int32))
    return ufl.Measure("ds", domain=msh, subdomain_data=mt)(1)


def _assemble_real(form) -> np.ndarray:
    """assemble_vector -> a real numpy array (the complex build carries a zero
    imaginary part on every thermal quantity; only V, gamma and A are complex)."""
    b = fem.assemble_vector(form)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    return np.real(b.array).copy()


def _stability_dt(msh, W, k_max_cells: np.ndarray, rho_cp_nodal: np.ndarray,
                  vol_nodal: np.ndarray, Q0) -> float:
    """Explicit lumped-mass stability limit dt <= min_i 2 V_i (rho cp)_i / K_ii.

    Assembled ONCE at the worst-case (largest) per-cell conductivity, so the
    bound holds for every property state the march visits. The FEM analogue of
    heatr3d.dt_stable_thermal; using rho*cp (not the latent-augmented slope) is
    the conservative choice, since the latent plateau only ever makes the
    temperature response stiffer-in-H and softer-in-T."""
    kf = fem.Function(Q0)
    kf.x.array[:] = k_max_cells.astype(dolfinx.default_scalar_type)
    u, v = ufl.TrialFunction(W), ufl.TestFunction(W)
    A = dolfinx.fem.petsc.assemble_matrix(
        fem.form(ufl.inner(kf * ufl.grad(u), ufl.grad(v)) * ufl.dx))
    A.assemble()
    d = np.real(A.getDiagonal().array).copy()
    A.destroy()
    good = d > 0.0
    return float(np.min(2.0 * vol_nodal[good] * rho_cp_nodal[good] / d[good]))


CFL_SAFETY = 0.9                     # heatr3d.CFL_SAFETY, verbatim


def march_enthalpy(msh, p: ForwardParams, in_part=None,
                   q_uniform: float | None = None,
                   q_dg0=None, mats: "Materials | None" = None,
                   max_time_s: float = 1500.0, phi_target: float = 0.90,
                   T0_fn=None, L: float = L_DOMAIN,
                   sample_dt_s: float | None = None,
                   resolve_hook=None, record: dict | None = None,
                   record_stride: int = 1, record_scalar_fn=None,
                   densify: bool = False,
                   stop_mean_rho: float | None = None) -> dict:
    """Mass-lumped explicit enthalpy march on the FEM mesh.

    DISCRETIZATION (the FEM analogue of heatr3d's explicit cell-centred FV
    update, one line at a time):

        V_i dH_i/dt = -(K T)_i + F_i - C_i(T)

      V_i = integral(phi_i dx)              lumped nodal volume      [m^3]
      H_i                                   volumetric enthalpy      [J/m^3]
      (K T)_i = integral(k grad T . grad phi_i dx)                   [W]
      F_i     = integral(Q phi_i dx)        RF source                [W]
      C_i     = integral_top conv_h (T - T_preheat) phi_i ds         [W]

    then T_i = T_from_enthalpy(H_i, ...) -- heatr3d's exact piecewise-linear
    inversion, applied pointwise, which is what MASS LUMPING makes well defined
    node by node. Row sums of K vanish (K applied to a constant is zero), so the
    scheme is exactly conservative: whatever leaves one node arrives at another.

    NODAL PART FRACTION (deviation from heatr3d, named): heatr3d classifies each
    voxel as wholly in or out of the part. On a conforming mesh the interface
    passes through nodes, so the nodal part fraction is taken as the exact
    volume ratio m_i = integral(doped phi_i dx) / integral(phi_i dx). This makes
    sum_i V_i m_i == the exact part volume, so the total latent budget is exact;
    a 0/1 threshold would not conserve it.

    CLAMPS: heatr3d's per-step dT cap and temp_min/temp_max clamp are ported
    verbatim, including the latching diagnostic -- a bound clamp means a
    numerical limiter altered the physics and the result is suspect.

    `record` (Phase B hook; default None == the Phase A path, bit-identical):
    when a dict is supplied it is filled with the state the reverse march needs
    and NOTHING else changes. Two lists:
      record["T_steps"][j]  -- the nodal T at the TOP of substep j, i.e. the
                              input state of that step. Every other per-step
                              quantity is a pure function of it and of that
                              step's drive, so this is the complete checkpoint.
      record["q_events"]    -- (substep_index, q_cells) for the pre-loop drive
                              and for every drive change a re-solve produced.
    The only cost when `record is None` is one `is not None` test per substep,
    and the flag-off bit identity is pinned by
    test_adjoint_transient.py::test_recording_flag_off_is_bit_identical.
    """
    if record is not None:
        record.setdefault("T_steps", [])
        record.setdefault("q_events", [])
        # BOUNDED RECORDING. `record_stride` stores only every Nth substep
        # state (an ANCHOR); `record_scalar_fn` records a per-step scalar tuple
        # instead. Storing every substep state cost 51.15 GB on the Tamper
        # (18000 sample steps x 36 CFL substeps x 9866 nodes x 8 B) and the run
        # was silently SIGKILLed by the OS three times. The reverse sweep never
        # needed them: adjoint._checkpoint_reader replays from anchors every
        # `checkpoint_interval` steps, so the forward was building 51 GB to
        # extract 2 GB of anchors. DEFAULTS (stride 1, no scalar fn) are the
        # original store-everything path, bit-identical.
        record.setdefault("T_step_index", [])
        record.setdefault("scalars", [])
        record["record_stride"] = int(record_stride)
    if int(record_stride) < 1:
        raise ValueError(f"record_stride must be >= 1, got {record_stride}")
    if not (q_uniform is None) ^ (q_dg0 is None):
        raise ValueError("march_enthalpy: pass exactly one of q_uniform / q_dg0")
    tdim = msh.topology.dim
    W = functionspace(msh, ("Lagrange", 1))
    Q0 = mats.dg0 if mats is not None else functionspace(msh, ("DG", 0))
    v = ufl.TestFunction(W)
    st = dolfinx.default_scalar_type

    # ---- geometry-derived, time-independent ------------------------------- #
    one = fem.Constant(msh, st(1.0))
    vol = _assemble_real(fem.form(ufl.inner(one, v) * ufl.dx))  # V_i
    if mats is not None:
        doped_cells = np.real(mats.doped.x.array).astype(float)
        cell_mask = mats.mask
    else:
        ncell = n_cells_local(msh)
        cells = np.arange(ncell, dtype=np.int32)
        if in_part is None:
            cell_mask = np.zeros(ncell, dtype=bool)
        else:
            mp = dolfinx.mesh.compute_midpoints(msh, tdim, cells)
            cell_mask = np.asarray(in_part(mp.T), dtype=bool)
        doped_cells = cell_mask.astype(float)
    dop = fem.Function(Q0)
    dop.x.array[:] = doped_cells.astype(st)
    m_nodal = np.where(vol > 0.0,
                       _assemble_real(fem.form(ufl.inner(dop, v) * ufl.dx)) / np.where(vol > 0, vol, 1.0),
                       0.0)
    m_nodal = np.clip(m_nodal, 0.0, 1.0)
    part_vol_m3 = float(np.sum(vol * m_nodal))

    # ---- drive ------------------------------------------------------------ #
    qf = fem.Function(Q0)
    if q_dg0 is not None:
        qf.x.array[:] = q_dg0.x.array
    else:
        qf.x.array[:] = np.where(cell_mask, float(q_uniform), 0.0).astype(st)
    F = _assemble_real(fem.form(ufl.inner(qf, v) * ufl.dx))    # W per node

    # ---- state ------------------------------------------------------------ #
    T_fn = fem.Function(W)
    if T0_fn is None:
        T = np.full(vol.size, p.preheat_c, dtype=np.float64)
    else:
        T_fn.interpolate(lambda x: T0_fn(x).astype(st))
        T = np.real(T_fn.x.array).astype(np.float64).copy()
    T0_snapshot = T.copy()
    rho_rel = np.full(vol.size, p.rho_rel)

    # ---- property model (heatr3d.run, per step) --------------------------- #
    rho_s_eff = p.rho_powder + rho_rel * (p.rho_solid - p.rho_powder)
    k_s_eff = p.k_powder + p.rho_rel * (p.k_solid - p.k_powder)

    def nodal_props(phi):
        rho_part = (1.0 - phi) * rho_s_eff + phi * p.rho_liquid
        cp_part = (1.0 - phi) * p.cp_solid + phi * p.cp_liquid
        rho = (1.0 - m_nodal) * p.rho_powder + m_nodal * rho_part
        cp = (1.0 - m_nodal) * p.cp_powder + m_nodal * cp_part
        return rho, cp, m_nodal * rho_s_eff * p.latent_j_per_kg

    cell_dofs = p1_cell_dofs(W)

    def cell_k(phi_nodal):
        """k on DG0 cells: powder outside, blended solid/liquid inside, using
        the cell-averaged melt fraction (the FEM analogue of heatr3d's per-voxel
        k with harmonic face averaging)."""
        phi_c = cell_average_p1(cell_dofs, phi_nodal)
        k_part = (1.0 - phi_c) * k_s_eff + phi_c * p.k_liquid
        return np.where(doped_cells > 0.5, k_part, p.k_powder)

    # ---- CFL (heatr3d.cfl_substeps analogue) ------------------------------ #
    k_max = np.where(doped_cells > 0.5, max(k_s_eff, p.k_liquid), p.k_powder)
    rho0, cp0, _ = nodal_props(phase_fraction(T, p)[0])
    dt_stable = _stability_dt(msh, W, k_max, rho0 * cp0, vol, Q0)
    n_sub = 1 if (not p.enforce_cfl or p.dt_s <= CFL_SAFETY * dt_stable) \
        else int(np.ceil(p.dt_s / (CFL_SAFETY * dt_stable)))
    dt_sub = p.dt_s / n_sub
    cfl_violated = bool(dt_sub > CFL_SAFETY * dt_stable)

    # ---- compiled forms (coefficients updated in place) ------------------- #
    k_fn = fem.Function(Q0)
    diff_form = fem.form(ufl.inner(k_fn * ufl.grad(T_fn), ufl.grad(v)) * ufl.dx)
    conv_form = None
    if p.conv_h != 0.0:
        ds_top = _top_facet_measure(msh, L)
        conv_form = fem.form(ufl.inner(
            fem.Constant(msh, st(p.conv_h)) * (T_fn - fem.Constant(msh, st(p.preheat_c))),
            v) * ds_top)

    # ---- march ------------------------------------------------------------ #
    nsteps = int(max_time_s / p.dt_s)
    e_in = e_loss = e_stored = 0.0
    clamp_bound = False
    reached = False
    t90 = float("nan")
    T_phi90 = None
    curve_t: list[float] = [0.0]
    curve_T: list[float] = [_wmean(T, vol * m_nodal) if part_vol_m3 > 0 else float("nan")]
    curve_phi: list[float] = [_wmean(phase_fraction(T, p)[0], vol * m_nodal)
                              if part_vol_m3 > 0 else 0.0]
    next_sample = sample_dt_s if sample_dt_s else float("inf")
    k_last = None
    n_k_assemblies = 0

    # ---- densification state (Stage A; inert unless densify) -------------- #
    # Guarded so the OFF path executes exactly the pre-densify statements and
    # stays BIT-IDENTICAL to the Phase A forward (record-hook precedent). The
    # ceiling peak is an end-state quantity, so the TRUE peak is the running
    # maximum of the in-part temperature over the WHOLE trajectory, not the
    # end-state snapshot -- studio3d ab08872 is the record of what a snapshot
    # peak understated.
    if densify:
        drho_cap = p.dens_max_drho_rate * dt_sub
        w_part_live = vol * m_nodal
        part_peak_mask = m_nodal > 0.5
        if not part_peak_mask.any():
            part_peak_mask = m_nodal > 0.0
        true_peak = float(T[part_peak_mask].max()) if part_peak_mask.any() \
            else float("nan")
        true_peak_step = 0
        # per-node running peak over the trajectory: the melt-completeness check
        # (min in-part peak >= melt onset) needs the PEAK each node ever reached,
        # not the end-state (a node can melt then cool as the front moves).
        T_peak_nodal = T.copy()
        reached_rho = False
        part_mean_rho0 = _wmean(rho_rel, w_part_live) if part_vol_m3 > 0 else float("nan")
        rho_traj_t: list[float] = [0.0]
        rho_traj_mean: list[float] = [part_mean_rho0]

    for it_sub in range(nsteps * n_sub):
        it, isub = divmod(it_sub, n_sub)
        t_now = it * p.dt_s + isub * dt_sub
        if record is not None and it_sub == 0:
            record["q_events"].append(
                (0, np.real(qf.x.array).astype(float).copy()))
        if resolve_hook is not None and isub == 0:
            new_q = resolve_hook(t_now,
                                 cell_average_p1(cell_dofs, T),
                                 cell_average_p1(cell_dofs, rho_rel))
            if new_q is not None:
                qf.x.array[:] = np.asarray(new_q).astype(st)
                F = _assemble_real(fem.form(ufl.inner(qf, v) * ufl.dx))
                if record is not None:
                    record["q_events"].append(
                        (it_sub, np.real(qf.x.array).astype(float).copy()))
        if record is not None:
            if it_sub % int(record_stride) == 0:
                record["T_steps"].append(T.copy())
                record["T_step_index"].append(it_sub)
            if record_scalar_fn is not None:
                record["scalars"].append(tuple(record_scalar_fn(T)))
        phi, _ = phase_fraction(T, p)
        if densify:
            # Density-dependent solid properties track the evolving rho_rel
            # (heatr3d.run per-step rho_s_eff / k_s_eff). The closures below
            # read these names; reassigning here updates them. On the first
            # substep rho_rel is still the initial constant, so these equal the
            # OFF-path constants -- which is why a zero-rate densify march
            # reproduces the constant-rho march bit-for-bit (pinned test).
            rho_s_eff = p.rho_powder + rho_rel * (p.rho_solid - p.rho_powder)
            k_s_eff = p.k_powder + cell_average_p1(cell_dofs, rho_rel) * (
                p.k_solid - p.k_powder)
        rho, cp, rho_L = nodal_props(phi)
        k_cells = cell_k(phi)
        if k_last is None or not np.array_equal(k_cells, k_last):
            k_fn.x.array[:] = k_cells.astype(st)
            k_last = k_cells.copy()
            n_k_assemblies += 1
        T_fn.x.array[:] = T.astype(st)
        num = -_assemble_real(diff_form) + F                   # W per node
        c_loss = np.zeros_like(num)
        if conv_form is not None:
            c_loss = _assemble_real(conv_form)
            num = num - c_loss

        e_in += float(F.sum()) * dt_sub
        e_loss += float(c_loss.sum()) * dt_sub

        rho_cp = rho * cp
        H = enthalpy_from_T(T, rho_cp, rho_L, p)
        H = H + dt_sub * np.nan_to_num(num) / np.where(vol > 0, vol, 1.0)
        T_new = T_from_enthalpy(H, rho_cp, rho_L, p)
        dT_raw = T_new - T
        dT = np.clip(dT_raw, -p.max_dt_step_c, p.max_dt_step_c)
        if int(np.count_nonzero(np.abs(dT_raw) > p.max_dt_step_c)):
            clamp_bound = True
        T_cand = T + dT
        if int(np.count_nonzero((T_cand > p.temp_max_c) | (T_cand < p.temp_min_c))):
            clamp_bound = True
        T_prev = T
        T = np.clip(T_cand, p.temp_min_c, p.temp_max_c)

        phi_now = phase_fraction(T, p)[0]
        # heatr3d audit v2: bank with THIS step's property maps, and bank the
        # ACTUAL applied change (post-clamp), so clamp-destroyed energy shows up
        e_stored += float(np.sum(vol * rho_cp * (T - T_prev)))
        e_stored += float(np.sum(vol * m_nodal * rho_s_eff * p.latent_j_per_kg
                                 * (phi_now - phi)))

        if densify:
            # heatr3d.run densify update (heatr3d.py:1211): evolve rho_rel with
            # the Arrhenius creep + viscous-capillary rate, capped per step, on
            # part-carrying nodes only. Track the running TRUE peak of the
            # in-part temperature for the ceiling constraint.
            drho = np.clip(dt_sub * densify_rate(T, phi_now, rho_rel, p),
                           0.0, drho_cap)
            rho_rel = np.where(m_nodal > 0.0,
                               np.clip(rho_rel + drho, 0.0, 1.0), rho_rel)
            step_peak = float(T[part_peak_mask].max()) if part_peak_mask.any() \
                else float("nan")
            if not (step_peak <= true_peak):     # NaN-safe first assignment
                true_peak = step_peak
                true_peak_step = it_sub
            np.maximum(T_peak_nodal, T, out=T_peak_nodal)

        t_end_step = t_now + dt_sub
        mean_phi = _wmean(phi_now, vol * m_nodal) if part_vol_m3 > 0 else 0.0
        if not reached and mean_phi >= phi_target:
            reached = True
            t90 = t_end_step
            T_phi90 = T.copy()
            if not densify:
                curve_t.append(t90)
                curve_T.append(_wmean(T, vol * m_nodal))
                curve_phi.append(mean_phi)
                break
        if densify:
            # In densify mode the march does NOT stop at melt onset; it stops at
            # the target MEAN relative density (the realistic process stop,
            # heatr3d.py:1231), so the ceiling peak is read at the densified
            # end-state, not at melt-onset.
            part_mean_rho = (_wmean(rho_rel, w_part_live)
                             if part_vol_m3 > 0 else float("nan"))
            if t_end_step >= next_sample - 1e-12:
                rho_traj_t.append(t_end_step)
                rho_traj_mean.append(part_mean_rho)
            if (stop_mean_rho is not None and part_vol_m3 > 0
                    and part_mean_rho >= stop_mean_rho):
                reached_rho = True
                curve_t.append(t_end_step)
                curve_T.append(_wmean(T, vol * m_nodal))
                curve_phi.append(mean_phi)
                break
        if t_end_step >= next_sample - 1e-12:
            curve_t.append(t_end_step)
            curve_T.append(_wmean(T, vol * m_nodal) if part_vol_m3 > 0 else float("nan"))
            curve_phi.append(mean_phi)
            next_sample += sample_dt_s

    if T_phi90 is None:
        T_phi90 = T.copy()
    w_part = vol * m_nodal
    phi_final = phase_fraction(T_phi90, p)[0]
    if record is not None:
        # total substeps a state existed for: what T_step_index is indexed
        # against, and what `n_steps` must mean downstream now that T_steps
        # holds anchors rather than one entry per step
        record["n_recorded_of"] = int(nsteps * n_sub)
    out = {
        "T": T, "T_phi90": T_phi90, "T0": T0_snapshot,
        "vol_nodal": vol, "m_nodal": m_nodal, "part_volume_m3": part_vol_m3,
        "reached": bool(reached), "t90_s": t90,
        "part_mean_T_c": _wmean(T, w_part) if part_vol_m3 > 0 else float("nan"),
        "part_std_T_c": _wstd(T, w_part) if part_vol_m3 > 0 else float("nan"),
        "sigma_T_c": _wstd(T_phi90, w_part) if part_vol_m3 > 0 else float("nan"),
        "part_mean_phi": _wmean(phi_final, w_part) if part_vol_m3 > 0 else 0.0,
        "domain_mean_T_c": _wmean(T, vol),
        "T0_domain_mean_c": _wmean(T0_snapshot, vol),
        "T_max_c": float(T.max()), "T_min_c": float(T.min()),
        "T0_max_c": float(T0_snapshot.max()), "T0_min_c": float(T0_snapshot.min()),
        "energy_in_j": e_in, "energy_stored_j": e_stored, "energy_loss_j": e_loss,
        "energy_residual_frac": (e_in - e_stored - e_loss) / max(e_in, 1e-30),
        "clamp_bound": bool(clamp_bound),
        "n_substeps_used": int(n_sub), "cfl_violated": cfl_violated,
        "dt_stable_s": float(dt_stable), "n_k_assemblies": int(n_k_assemblies),
        "curve_t_s": curve_t, "curve_part_mean_T_c": curve_T,
        "curve_part_mean_phi": curve_phi,
        "n_steps_taken": int(it_sub + 1) if nsteps * n_sub > 0 else 0,
    }
    if densify:
        out.update({
            "densify": True,
            "rho_final": rho_rel,
            "part_mean_rho": (_wmean(rho_rel, w_part_live)
                              if part_vol_m3 > 0 else float("nan")),
            "reached_rho": bool(reached_rho),
            "stop_mean_rho": stop_mean_rho,
            # the TRUE trajectory peak of the in-part temperature -- the ceiling
            # quantity. This is the physical max over (t, x); the KS aggregate
            # (solve3d.ceiling) is only the smooth gradient proxy and must
            # never be reported in this slot.
            "true_peak_T_c": true_peak,
            "true_peak_step_index": int(true_peak_step),
            "T_peak_nodal": T_peak_nodal,
            "part_peak_mask": part_peak_mask,
            "T_end_max_c": (float(T[part_peak_mask].max())
                            if part_peak_mask.any() else float("nan")),
            "rho_traj_t_s": rho_traj_t,
            "rho_traj_part_mean_rho": rho_traj_mean,
            "exposure_s": float(it_sub + 1) * dt_sub,
        })
    return out


def p1_cell_dofs(W) -> np.ndarray:
    """(ncells, 4) P1 dof indices per tetrahedron."""
    dm = W.dofmap
    lst = dm.list
    return np.asarray(lst).reshape(-1, dm.dof_layout.num_dofs)


def cell_average_p1(cell_dofs: np.ndarray, nodal: np.ndarray) -> np.ndarray:
    """Cell average of a P1 field = mean of its vertex values (EXACT on a
    simplex for a linear function).

    Deliberately not an L2 projection: dg0_project builds and solves a linear
    system, which inside a 6000-step march costs more than the whole physics.
    """
    return nodal[cell_dofs].mean(axis=1)


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    tot = float(w.sum())
    return float(np.dot(v, w) / tot) if tot > 0 else float("nan")


def _wstd(v: np.ndarray, w: np.ndarray) -> float:
    tot = float(w.sum())
    if tot <= 0:
        return float("nan")
    mu = float(np.dot(v, w) / tot)
    return float(np.sqrt(float(np.dot((v - mu) ** 2, w)) / tot))


def run_forward(shape: str, target_nodes_in_part: int, lc0: float,
                p: ForwardParams | None = None, max_time_s: float = 1500.0,
                phi_target: float = 0.90, sample_dt_s: float = 10.0,
                L: float = L_DOMAIN, petsc_options=None) -> dict:
    """The Phase A coupled forward: EQS -> enthalpy march, with the in-march
    EQS re-solve schedule ported from heatr3d.run (S4-COUPLING, 699ed79).

    Re-solve semantics, port-for-port:
      * schedule on ABSOLUTE time; the next re-solve is the first multiple of
        eqs_update_interval_s STRICTLY greater than the march start, so the
        pre-loop solve is never double-counted;
      * the check runs at the TOP of a full step, before any thermal work, so
        the drive used by step `it` belongs to time it*dt_s;
      * every re-solve rebuilds gamma from the BASE sigma through
        apply_sigma_coupling (never compounding), re-solves, and reapplies the
        fixed-power renormalization;
      * eqs_resolve_drift_rtol > 0 turns a low-drift re-solve into a pointwise
        Q *= sigma/sigma_last rescale plus a re-renormalization (heatr3d's D1
        adaptive skip), counted separately.
    n_eqs_solves counts FULL solves including the pre-loop one, exactly as
    heatr3d.Result.n_eqs_solves does.
    """
    p = p or ForwardParams()
    case = eqs_case(shape, target_nodes_in_part, lc0, p=p,
                    petsc_options=petsc_options)
    msh, mats = case["msh"], case["mats"]
    sigma_base = np.real(mats.sigma.x.array).astype(float).copy()
    coupling_on = float(p.eqs_update_interval_s) > 0.0

    census = {"n_eqs_solves": 1, "n_eqs_resolves_skipped": 0,
              "wall_eqs_s": case["wall_solve_s"], "resolve_times_s": []}
    sigma_last = sigma_base.copy()
    if coupling_on:
        # heatr3d applies the coupling BEFORE the pre-loop solve, using the
        # initial T / rho state.
        T0 = np.full(sigma_base.size, p.preheat_c)
        rho0 = np.full(sigma_base.size, p.rho_rel)
        mats.sigma.x.array[:] = apply_sigma_coupling(
            mats, T0, rho0, p).astype(dolfinx.default_scalar_type)
        sigma_last = np.real(mats.sigma.x.array).astype(float).copy()
        import time as _t
        t0 = _t.perf_counter()
        Vr, Vi = solve_eqs(msh, mats, p, petsc_options=petsc_options)
        census["wall_eqs_s"] = _t.perf_counter() - t0
        drive = qrf_dg0(msh, Vr, Vi, mats, p)
        case.update(drive)
        case["Vr"], case["Vi"] = Vr, Vi

    q_fn = case["q"]
    p_target = case["p_target_w"]
    state = {"next": (float(p.eqs_update_interval_s) if coupling_on
                      else float("inf")),
             "q": np.real(q_fn.x.array).astype(float).copy()}

    vol_cells = None

    def hook(t_now: float, T_cells: np.ndarray, rho_cells: np.ndarray):
        if not coupling_on or t_now < state["next"] - 1e-12:
            return None
        while t_now >= state["next"] - 1e-12:
            state["next"] += float(p.eqs_update_interval_s)
        nonlocal sigma_last, vol_cells
        mats.sigma.x.array[:] = sigma_base.astype(dolfinx.default_scalar_type)
        sigma_new = apply_sigma_coupling(mats, T_cells, rho_cells, p)
        skip = False
        if p.eqs_resolve_drift_rtol > 0.0:
            drift = float(np.max(np.abs(sigma_new - sigma_last))) / max(
                p.sigma_doped, 1e-30)
            skip = drift < p.eqs_resolve_drift_rtol
        if skip:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(sigma_last > 0.0, sigma_new / sigma_last, 1.0)
            q_new = np.clip(np.nan_to_num(state["q"] * ratio), 0.0, None)
            q_new = np.where(mats.mask, q_new, 0.0)
            if vol_cells is None:
                import femutils as fu
                vol_cells = np.real(fu.cell_volumes(msh, mats.dg0)).astype(float)
            p_now = float(np.dot(q_new, vol_cells))
            if p_now > 1e-18:
                q_new = q_new * (p_target / p_now)
            state["q"] = q_new
            census["n_eqs_resolves_skipped"] += 1
        else:
            mats.sigma.x.array[:] = sigma_new.astype(dolfinx.default_scalar_type)
            import time as _t
            t0 = _t.perf_counter()
            Vr_, Vi_ = solve_eqs(msh, mats, p, petsc_options=petsc_options)
            census["wall_eqs_s"] += _t.perf_counter() - t0
            d = qrf_dg0(msh, Vr_, Vi_, mats, p)
            state["q"] = np.real(d["q"].x.array).astype(float).copy()
            sigma_last = sigma_new.copy()
            census["n_eqs_solves"] += 1
        census["resolve_times_s"].append(float(t_now))
        return state["q"]

    import time as _t
    t0 = _t.perf_counter()
    march = march_enthalpy(msh, p, mats=mats, q_dg0=q_fn,
                           max_time_s=max_time_s, phi_target=phi_target,
                           L=L, sample_dt_s=sample_dt_s,
                           resolve_hook=hook if coupling_on else None)
    march["wall_march_s"] = _t.perf_counter() - t0
    out = {k: v for k, v in case.items() if k not in ("msh", "mats", "q")}
    out.update(march)
    out.update(census)
    out["msh"] = msh
    out["mats"] = mats
    out["q"] = q_fn
    out["shape"] = shape
    out["coupling_on"] = bool(coupling_on)
    return out
