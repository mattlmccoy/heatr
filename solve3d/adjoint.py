"""solve3d Phase B: adjoint of the Phase A forward.

RUNS IN THE SPIKE ENV ONLY (dolfinx 0.11 complex build).

LAYER B1 (this file, first section): the STEADY EQS adjoint, lifted from
heatr3d_d1_spike/adjoint_core.py (D1 Task 5, already gated: worst per-dof FD
error 7.4e-5 over 105191 design dofs, mutation-tested). Two things are lifted
verbatim because they are the load-bearing parts:

  * A is complex SYMMETRIC (real basis functions, complex coefficient), so
    A^H = conj(A) and the adjoint solve REUSES the forward LU factorization:
    solve A x = conj(rho), then lambda = conj(x). This is why a gradient over
    every design dof costs a fraction of one forward.
  * The fixed-power renormalization is DIFFERENTIATED THROUGH, never frozen.
    D1 measured that freezing it is wrong by up to 150 % on an individual dof.

THE FORWARD MAP (identical to solve3d/forward.py; consistency is CHECKED, not
assumed -- see SteadyCase.consistency_vs_phase_a):

    gamma_c = sigma_c + i*omega*eps0*eps_r_c        (eps_r fixed in Phase B)
    A(sigma) V = b                                   div(gamma grad V) = 0
    e2_c    = |grad Vr|^2 + |grad Vi|^2              (P1 -> cell-constant)
    Qraw_c  = 0.5 * sigma_c * e2_c                   (zero outside the part)
    scale   = p_target / sum_c Qraw_c |K_c|
    Q_c     = scale * Qraw_c

`vjp_q` is the reusable interface: given dJ/dQ_c it returns dJ/dsigma_c. The
transient layer (Task 2) produces that dJ/dQ_c from the reverse march, so the
EQS adjoint is written once and used by every layer.

CONVENTIONS (protocol JSON / FROZEN_CONVENTIONS_2D.md section 7): conductivity
is the ONLY actuator; the permittivity channel is model-only and is not
implemented here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solve3d import forward as fwd
from solve3d import gate_fd, gates

import dolfinx
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fp
from petsc4py import PETSc

RESULTS = Path(__file__).resolve().parent / "results"

N_REFINE = 2        # iterative-refinement sweeps (adjoint_core.N_REFINE)
MUTANTS = ("renorm_frozen", "adjoint_dropped")


@dataclass
class SteadyState:
    """Everything the gradient needs from one EQS forward."""
    sigma_part: np.ndarray
    V: np.ndarray                # complex nodal
    e2: np.ndarray               # per-cell
    qraw: np.ndarray             # per-cell, masked to the part
    q: np.ndarray                # per-cell, renormalized
    scale: float
    p_now: float
    qbar: float
    clip_active: bool
    res_norm: float


class SteadyEqs:
    """One mesh + one part geometry; sigma on the part cells is the design.

    Owns the forms (built ONCE, coefficients updated in place) and the LU, so
    the forward and the adjoint cannot drift apart.
    """

    def __init__(self, msh, mats: fwd.Materials, p: fwd.ForwardParams,
                 L: float = fwd.L_DOMAIN, degree: int = 1):
        if not fwd.IS_COMPLEX:
            raise RuntimeError("Phase B assumes the complex scalar build")
        self.msh, self.mats, self.p, self.L = msh, mats, p, L
        # Task 3 spike flag. DEFAULT "direct" reproduces every existing gate
        # bit-identically; "iterative" swaps in GMRES+GAMG (see set_solver).
        self.solver_kind = "direct"
        self.solver_rtol = None
        self.W = fwd.functionspace(msh, ("Lagrange", degree))
        self.Q0 = mats.dg0
        import femutils as fu
        self.vol = np.real(fu.cell_volumes(msh, self.Q0)).astype(float)
        self.part = np.flatnonzero(mats.mask)
        self.v_doped = float(self.vol[self.part].sum())
        self.p_target = p.power_density_w_per_m3 * self.v_doped

        st = dolfinx.default_scalar_type
        u, w = ufl.TrialFunction(self.W), ufl.TestFunction(self.W)
        gam = mats.sigma + 1j * p.omega * fwd.EPS0 * mats.eps_r
        self.a_form = fem.form(ufl.inner(gam * ufl.grad(u), ufl.grad(w)) * ufl.dx)
        self.L_form = fem.form(ufl.inner(fem.Constant(msh, st(0.0)), w) * ufl.dx)
        d_lo, d_hi = fwd._electrode_dofs(self.W, msh, L)
        self.bcs = [fem.dirichletbc(st(p.v_lo), d_lo, self.W),
                    fem.dirichletbc(st(p.v_hi), d_hi, self.W)]
        self.bc_dofs = np.unique(np.concatenate([d_lo, d_hi]))

        self.Vfun = fem.Function(self.W)
        self.lam = fem.Function(self.W)
        self.what = fem.Function(self.Q0)
        m = ufl.TestFunction(self.Q0)
        self.e2_form = fem.form(
            ufl.inner(ufl.inner(ufl.grad(self.Vfun), ufl.grad(self.Vfun)), m) * ufl.dx)
        self.rho_form = fem.form(
            ufl.inner(2.0 * self.what * ufl.grad(self.Vfun), ufl.grad(w)) * ufl.dx)
        self.dRds_form = fem.form(
            ufl.inner(ufl.inner(ufl.grad(self.Vfun), ufl.grad(self.lam)), m) * ufl.dx)
        self._ksp = None
        self._A = None
        self.n_forward = 0
        self.n_adjoint = 0

    # ------------------------------------------------------------------ #
    def set_sigma(self, sigma_part: np.ndarray) -> None:
        arr = np.real(self.mats.sigma.x.array).astype(float).copy()
        arr[self.part] = np.asarray(sigma_part, dtype=float)
        self.mats.sigma.x.array[:] = arr.astype(dolfinx.default_scalar_type)

    def set_sigma_all(self, sigma_all: np.ndarray) -> None:
        self.mats.sigma.x.array[:] = np.asarray(
            sigma_all, dtype=float).astype(dolfinx.default_scalar_type)

    def set_solver(self, kind: str = "direct", rtol: float | None = None):
        """Select the EQS linear solver. Task 3 spike (cgamg_protocol.json).

        "direct"    PETSc preonly + LU, the incumbent. Bit-identical to today.
        "iterative" GMRES + GAMG at `rtol`. GMRES, not CG: the EQS operator is
                    complex SYMMETRIC (A^T = A), which is NOT Hermitian
                    positive definite, so CG has no convergence guarantee here.
                    That was named in the pre-registration before any run.

        Changing the solver invalidates any existing factorization, so the
        cached KSP is dropped rather than reused with the wrong type.
        """
        if kind not in ("direct", "iterative"):
            raise ValueError(f"unknown solver kind {kind!r}")
        if kind == "iterative" and not rtol:
            raise ValueError("iterative solver requires an explicit rtol")
        self.solver_kind = kind
        self.solver_rtol = None if kind == "direct" else float(rtol)
        if self._ksp is not None:
            self._ksp.destroy()
            self._A.destroy()
            self._ksp = self._A = None
        return self

    def _factorize(self):
        if self._ksp is not None:
            self._ksp.destroy()
            self._A.destroy()
        A = fp.assemble_matrix(self.a_form, bcs=self.bcs)
        A.assemble()
        ksp = PETSc.KSP().create(self.msh.comm)
        ksp.setOperators(A)
        if self.solver_kind == "iterative":
            ksp.setType("gmres")
            ksp.getPC().setType("gamg")
            ksp.setTolerances(rtol=self.solver_rtol, max_it=2000)
        else:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
        self._A, self._ksp = A, ksp
        return ksp

    def _rhs(self):
        b = fp.assemble_vector(self.L_form)
        fp.apply_lifting(b, [self.a_form], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(b, self.bcs)
        return b

    # ------------------------------------------------------------------ #
    def solve_state(self) -> SteadyState:
        """EQS solve + Q_rf from the CURRENT sigma coefficient."""
        ksp = self._factorize()
        b = self._rhs()
        x = self._A.createVecRight()
        ksp.solve(b, x)
        # Iterative refinement: gamma spans sigma_doped/sigma_virgin = 4e6, so
        # the LU backward error sets the FD noise floor on J (D1 measured the
        # per-dof FD error to be a CONSTANT absolute quantity -- the signature
        # of a floor, not of a wrong gradient). Two sweeps = two triangular
        # solves, and they buy roughly two decades of FD headroom.
        r = self._A.createVecRight()
        dx = self._A.createVecRight()
        res = float("nan")
        for _ in range(N_REFINE):
            self._A.mult(x, r)
            r.aypx(-1.0, b)
            ksp.solve(r, dx)
            x.axpy(1.0, dx)
        if N_REFINE:
            self._A.mult(x, r)
            r.aypx(-1.0, b)
            res = float(r.norm() / b.norm())
        r.destroy(); dx.destroy()
        self.Vfun.x.array[:] = x.array
        self.Vfun.x.scatter_forward()
        b.destroy(); x.destroy()
        self.n_forward += 1

        e2 = np.real(fem.assemble_vector(self.e2_form).array) / self.vol
        sig_all = np.real(self.mats.sigma.x.array).astype(float)
        qraw = np.zeros_like(e2)
        qraw[self.part] = 0.5 * sig_all[self.part] * e2[self.part]
        raw_neg = bool(np.any(qraw[self.part] < 0.0))
        p_now = float((qraw[self.part] * self.vol[self.part]).sum())
        scale = self.p_target / p_now if p_now > 1e-18 else 1.0
        q = scale * qraw
        qbar = float((q[self.part] * self.vol[self.part]).sum() / self.v_doped)
        return SteadyState(
            sigma_part=np.real(self.mats.sigma.x.array).astype(float)[self.part].copy(),
            V=np.array(self.Vfun.x.array), e2=e2, qraw=qraw, q=q, scale=scale,
            p_now=p_now, qbar=qbar, clip_active=raw_neg, res_norm=res)

    def forward(self, sigma_part: np.ndarray) -> SteadyState:
        self.set_sigma(sigma_part)
        return self.solve_state()

    # ------------------------------------------------------------------ #
    def vjp_q(self, st: SteadyState, gQ: np.ndarray,
              mutate: str | None = None) -> np.ndarray:
        """dJ/dsigma on the part cells, given dJ/dQ_c on all cells.

        THE reusable interface: the transient layer hands its accumulated
        dJ/dQ here and gets back a conductivity gradient, so there is exactly
        one EQS adjoint in the codebase.

        `mutate` deliberately breaks the gradient so the gate can be shown to be
        non-vacuous (pre-registered in the protocol):
          renorm_frozen    -- treat the fixed-power scale as a constant
          adjoint_dropped  -- keep only the explicit dQ/dsigma partial
        """
        if mutate not in (None,) + MUTANTS:
            raise ValueError(mutate)
        p_idx = self.part
        gQ = np.asarray(gQ, dtype=float)

        # --- through the renormalization  Q = (p_target / sum Qraw|K|) Qraw ---
        S = float((gQ[p_idx] * st.qraw[p_idx]).sum())
        g_raw = np.zeros_like(gQ)
        g_raw[p_idx] = st.scale * gQ[p_idx]
        if mutate != "renorm_frozen":
            g_raw[p_idx] -= (S * st.scale / st.p_now) * self.vol[p_idx]

        # --- explicit sigma dependence of Qraw = 0.5 sigma e2 -----------------
        d_expl = g_raw[p_idx] * 0.5 * st.e2[p_idx]
        if mutate == "adjoint_dropped":
            return d_expl

        # --- through the field: A^H lambda = rho, reusing the forward LU ------
        sig_all = np.real(self.mats.sigma.x.array).astype(float)
        w = np.zeros_like(gQ)
        w[p_idx] = g_raw[p_idx] * 0.5 * sig_all[p_idx]
        self.what.x.array[:] = (w / self.vol).astype(dolfinx.default_scalar_type)

        rho = fem.assemble_vector(self.rho_form).array.copy()
        rhs = np.conj(rho)
        rhs[self.bc_dofs] = 0.0                 # dR_bc/dsigma = 0
        bvec = self._A.createVecRight()
        bvec.array[:] = rhs
        xvec = self._A.createVecRight()
        self._ksp.solve(bvec, xvec)             # A x = conj(rho); A^H = conj(A)
        self.lam.x.array[:] = np.conj(xvec.array)
        self.lam.x.array[self.bc_dofs] = 0.0
        self.lam.x.scatter_forward()
        bvec.destroy(); xvec.destroy()
        self.n_adjoint += 1

        dR = np.real(fem.assemble_vector(self.dRds_form).array)
        return d_expl - dR[p_idx]


# --------------------------------------------------------------------------- #
# Layer B1 gate case
# --------------------------------------------------------------------------- #
def smooth_sigma(mp: np.ndarray, sigma0: float, amp: float = 0.4,
                 r_part: float = 0.010, L: float = fwd.L_DOMAIN) -> np.ndarray:
    """A smooth, non-degenerate design point (adjoint_core.smooth_sigma).

    Stays inside [0.4, 1.6] * sigma0 > 0 so the Q clip is never active and no
    subgradient question contaminates layer B1."""
    x, y, z = mp[0], mp[1], mp[2]
    return sigma0 * (1.0
                     + amp * np.sin(np.pi * x / r_part) * np.cos(np.pi * y / r_part)
                     + 0.2 * np.sin(np.pi * z / (L / 2.0)))


class SteadyCase:
    """Layer B1: J is a quadratic of Q_rf only, no thermal march.

        J = sum_c (Q_c - Qbar)^2 |K_c|      over part cells

    Qbar is the volume-weighted part mean, and dJ/dQ_c = 2 (Q_c - Qbar)|K_c|
    EXACTLY, because the correction term carries sum_c (Q_c - Qbar)|K_c| = 0.
    """

    def __init__(self, msh, mats, p, eqs: SteadyEqs, s0: np.ndarray, info):
        self.msh, self.mats, self.p, self.eqs, self.s0, self.info = \
            msh, mats, p, eqs, s0, info
        self.power_density = p.power_density_w_per_m3

    @classmethod
    def build(cls, shape: str = "circle", target_nodes_in_part: int = 23040,
              lc0: float = 0.0009375, p: fwd.ForwardParams | None = None):
        import mesh_gmsh as mg
        p = p or fwd.ForwardParams()
        kind = "cylinder" if shape == "circle" else "square"
        msh, info, _ = mg.match_lc(kind, int(target_nodes_in_part), float(lc0))
        mats = fwd.build_materials(msh, fwd.in_part_predicate(shape), p)
        eqs = SteadyEqs(msh, mats, p)
        mp = dolfinx.mesh.compute_midpoints(
            msh, msh.topology.dim,
            np.arange(fwd.n_cells_local(msh), dtype=np.int32))
        s0 = smooth_sigma(mp.T[:, eqs.part], p.sigma_doped)
        return cls(msh, mats, p, eqs, s0, info)

    # ------------------------------------------------------------------ #
    def forward(self, s: np.ndarray) -> SteadyState:
        return self.eqs.forward(s)

    def J_of_state(self, st: SteadyState) -> float:
        p_idx = self.eqs.part
        vol = self.eqs.vol
        return float((((st.q[p_idx] - st.qbar) ** 2) * vol[p_idx]).sum())

    def J(self, s: np.ndarray) -> float:
        return self.J_of_state(self.forward(s))

    def seed(self, st: SteadyState) -> np.ndarray:
        g = np.zeros_like(st.q)
        p_idx = self.eqs.part
        g[p_idx] = 2.0 * (st.q[p_idx] - st.qbar) * self.eqs.vol[p_idx]
        return g

    def gradient(self, s: np.ndarray, mutate: str | None = None) -> np.ndarray:
        st = self.forward(s)
        return self.eqs.vjp_q(st, self.seed(st), mutate=mutate)

    # ------------------------------------------------------------------ #
    def consistency_vs_phase_a(self) -> dict:
        """Forward and adjoint must share ONE discrete operator.

        The adjoint carries its own LU-factorized solve (so the factorization
        can be reused for A^H); this checks that solve against the Phase A
        production path (GMRES + gamg, forward.solve_eqs / forward.qrf_dg0) on
        the same mesh and the same sigma."""
        self.eqs.set_sigma(self.s0)
        st = self.eqs.solve_state()
        out = {"lu_residual_norm": st.res_norm,
               "adjoint_solver":
                   f"direct LU + {N_REFINE} iterative-refinement sweeps"}
        for label, opts in (("vs_phase_a_iterative", fwd.KSP_ITER),
                            ("vs_phase_a_lu", fwd.KSP_LU)):
            self.eqs.set_sigma(self.s0)
            Vr, Vi = fwd.solve_eqs(self.msh, self.mats, self.p,
                                   petsc_options=opts)
            drive = fwd.qrf_dg0(self.msh, Vr, Vi, self.mats, self.p)
            q_pa = np.real(drive["q"].x.array).astype(float)
            v_ref = np.real(Vr.x.array) + 1j * np.real(Vi.x.array)
            qmax = float(np.max(np.abs(q_pa))) or 1.0
            out[label] = {
                "max_dV_over_v_lo":
                    float(np.max(np.abs(st.V - v_ref)) / abs(self.p.v_lo)),
                "max_dQ_over_Qmax":
                    float(np.max(np.abs(st.q - q_pa)) / qmax),
                "scale_rel_diff": float(abs(st.scale / drive["scale"] - 1.0)),
                "solver": opts["ksp_type"] + "+" + opts["pc_type"]}
        # The GATE is the LU-vs-LU comparison (same operator, same solver
        # class). The GMRES figure is Phase A's ksp_rtol = 1e-10 showing
        # through |E|^2, a property of the production solver, and is recorded
        # rather than gated. MEASURED on the small case: 5.07e-13 (LU) against
        # 1.83e-07 (GMRES) for the same operator.
        out["assembly_consistency_gate"] = out["vs_phase_a_lu"]["max_dQ_over_Qmax"]
        return out

    # ------------------------------------------------------------------ #
    def run_fd_gate(self, seed: int = 7, reuse: bool = True) -> dict:
        """The B1 gate. `reuse` returns the stored artifact if one exists.

        The sweep is 8 epsilons x 4 probes x 2 solves on a 24784-dof complex LU
        (~39 minutes measured), so re-running it to re-read a number that is
        already recorded would be waste, not rigour. The artifact is produced by
        a real run; `reuse=False` forces a fresh one."""
        cached = RESULTS / "phase_b_steady_gate.json"
        if reuse and cached.exists():
            d = json.loads(cached.read_text())
            if "gate" in d:
                return d
        cons = self.consistency_vs_phase_a()
        t0 = time.perf_counter()
        st = self.forward(self.s0)
        t_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        g = self.eqs.vjp_q(st, self.seed(st))
        t_grad = time.perf_counter() - t0
        mask = np.ones(g.shape, dtype=bool)      # every part cell is a design dof
        gate = gate_fd.run_probes(self.J, self.s0, g, mask, seed=seed)
        doc = {
            "what": "Phase B layer B1: steady EQS adjoint dJ/dsigma, "
                    "J = sum_c (Q_c - Qbar)^2 |K_c| over part cells",
            "mesh": {"n_dofs_total": int(self.info.n_nodes_total),
                     "n_cells_total": int(self.info.n_cells_total),
                     "n_nodes_in_part": int(self.info.n_nodes_in_part),
                     "n_design_dofs": int(g.size),
                     "lc_part_m": float(self.info.lc_part)},
            "design_point": "adjoint_core.smooth_sigma, amp 0.4 (clip inactive "
                            "by construction, so B1 is a SMOOTH layer)",
            "forward": {"J": self.J_of_state(st), "scale": st.scale,
                        "qbar": st.qbar,
                        "qbar_over_power_density_minus_1":
                            float(st.qbar / self.power_density - 1.0),
                        "clip_active": st.clip_active,
                        "lu_residual_norm": st.res_norm},
            "consistency": cons,
            "cost": {"wall_forward_s": t_fwd, "wall_gradient_s": t_grad,
                     "gradient_over_forward": t_grad / t_fwd},
            "gate": gate,
            "thresholds": {"pass_rel_err": gate_fd.PASS_REL_ERR,
                           "subgradient_pass_rel_err": gate_fd.SUBGRADIENT_PASS_REL_ERR},
        }
        gates.write_json("phase_b_steady_gate.json", doc)
        return doc

    def run_mutation_tests(self, seed: int = 7, reuse: bool = True) -> dict:
        """Both pre-registered mutants, on the GRADIENT DIRECTION probe.

        One probe is enough to disqualify a gradient, and the gradient direction
        is the highest-signal one -- which is exactly what D1 used for its
        mutation tests (task5.mutations reports a directional rel err). The full
        four-probe sweep is reserved for the real gradient, where the cost buys
        information rather than confirming a known failure."""
        cached = RESULTS / "phase_b_steady_gate.json"
        if reuse and cached.exists():
            d0 = json.loads(cached.read_text())
            if "mutations" in d0:
                return d0["mutations"]
        st = self.forward(self.s0)
        g_true = self.eqs.vjp_q(st, self.seed(st))
        d = g_true / float(np.linalg.norm(g_true))
        x_scale = float(np.mean(np.abs(self.s0)))
        out: dict = {"what": "pre-registered mutants; both MUST fail the gate",
                     "probe": "gradient_direction"}
        for name in MUTANTS:
            g = self.eqs.vjp_q(st, self.seed(st), mutate=name)
            sw = gate_fd.sweep(self.J, self.s0, d, float(np.dot(g, d)),
                               x_scale=x_scale)
            sw.update(gate_fd.verdict(sw["best_rel_err"]))
            out[name] = {
                "probes": {"gradient_direction": sw},
                "worst_best_rel_err": sw["best_rel_err"],
                "all_pass_subgradient": sw["pass_subgradient"],
                "all_pass_preferred": sw["pass_preferred"],
                "max_rel_dev_vs_true_gradient": float(np.max(
                    np.abs(g - g_true) / np.maximum(np.abs(g_true), 1e-300))),
                "mutant": name}
        p = RESULTS / "phase_b_steady_gate.json"
        doc = json.loads(p.read_text()) if p.exists() else {}
        doc["mutations"] = out
        gates.write_json(p.name, doc)
        return out


# =========================================================================== #
# LAYER B2: the TRANSIENT adjoint (reverse march), store-everything
#
# Semantic template: fgm_solve_campaign/adjoint2d/adjoint.py (READ ONLY) --
# substep_vjp / reverse_march / eqs_vjp. The 3-D differences, each named:
#   * the enthalpy scheme inverts H(T) EXACTLY instead of dividing by an
#     apparent heat capacity, so the step VJP carries dT_new/dH AND
#     dT_new/d(rho cp) (the property map itself depends on T through phi);
#   * rho_rel is HELD FIXED in the Phase A forward, so there is no density
#     co-state here. The 2-D coupled (T, rho) sweep is the template for when
#     densify ports into solve3d; until then the adjoint matches the forward.
#   * the drive changes DURING the march (in-march EQS re-solves), and the new
#     sigma depends on T at the re-solve instant, so the reverse sweep injects
#     a dJ/dT contribution at every re-solve boundary. That coupling has no
#     2-D analogue.
# =========================================================================== #
from dataclasses import field as _field       # noqa: E402


@dataclass
class _Event:
    """One EQS solve and the state it produced."""
    step: int                        # substep index at whose TOP it happened
    sigma_eff: np.ndarray            # full-domain sigma actually used
    state: SteadyState
    T_cells: np.ndarray | None       # cell-averaged T fed to the coupling
    rho_cells: np.ndarray | None
    q: np.ndarray                    # per-cell drive it produced


@dataclass
class Trajectory:
    T_steps: list = _field(default_factory=list)
    events: list = _field(default_factory=list)
    step_event: np.ndarray | None = None
    T_final: np.ndarray | None = None
    n_steps: int = 0
    out: dict | None = None


class TransientCase:
    """Layer B2/B3: the FULL Phase A coupled forward, differentiated.

        design -> sigma_base -> {EQS re-solves} -> Q(t) -> march -> T
        J = sum_i vol_i * (phi(T_i) - chi_i)^2      at a FIXED read step

    chi_i is the nodal part fraction m_i, the EXACT volume fraction on a
    conforming mesh (protocol: no sub-cell area fill is needed because no cell
    is partial).
    """

    def __init__(self, msh, mats, p, eqs: SteadyEqs, info, sample_dt_s: float,
                 max_time_s: float, L: float = fwd.L_DOMAIN):
        # THE CHAMBER FRAME. Default is the frozen 60 mm, so every existing
        # caller is bit-identical; adaptive-chamber callers must pass the same
        # L their MESH was built with. This was hardcoded to fwd.L_DOMAIN and
        # on an 85 mm mesh it looked for the convective top facet -- and, via
        # SteadyEqs, the electrode dofs -- at +-30 mm where there is nothing.
        # The Dirichlet rows were never set, the RHS was identically zero, and
        # the EQS solve died on r.norm() / b.norm(). It failed loudly, which is
        # the only reason it was a crash and not a silently wrong field.
        self.L = float(L)
        self.msh, self.mats, self.p, self.eqs, self.info = msh, mats, p, eqs, info
        self.max_time_s, self.sample_dt_s = max_time_s, sample_dt_s
        self.W = eqs.W
        self.Q0 = eqs.Q0
        self.cell_dofs = fwd.p1_cell_dofs(self.W)
        self.ncells = fwd.n_cells_local(msh)
        st = dolfinx.default_scalar_type
        v = ufl.TestFunction(self.W)
        m = ufl.TestFunction(self.Q0)
        self.k_fn = fem.Function(self.Q0)
        self.T_fn = fem.Function(self.W)
        self.G_fn = fem.Function(self.W)
        self.q_fn = fem.Function(self.Q0)
        self._one = fem.Constant(msh, st(1.0))
        self.vol_nodal = fwd._assemble_real(
            fem.form(ufl.inner(self._one, v) * ufl.dx))
        self.m_nodal = np.clip(
            fwd._assemble_real(fem.form(ufl.inner(mats.doped, v) * ufl.dx))
            / np.where(self.vol_nodal > 0, self.vol_nodal, 1.0), 0.0, 1.0)
        self.doped_cells = np.real(mats.doped.x.array).astype(float)
        self.ds_top = fwd._top_facet_measure(msh, self.L)
        self.diff_form = fem.form(
            ufl.inner(self.k_fn * ufl.grad(self.T_fn), ufl.grad(v)) * ufl.dx)
        self.diffG_form = fem.form(
            ufl.inner(self.k_fn * ufl.grad(self.G_fn), ufl.grad(v)) * ufl.dx)
        self.conv_form = fem.form(ufl.inner(
            fem.Constant(msh, st(p.conv_h))
            * (self.T_fn - fem.Constant(msh, st(p.preheat_c))), v) * self.ds_top)
        self.convG_form = fem.form(ufl.inner(
            fem.Constant(msh, st(p.conv_h)) * self.G_fn, v) * self.ds_top)
        self.gk_form = fem.form(ufl.inner(
            ufl.inner(ufl.grad(self.T_fn), ufl.grad(self.G_fn)), m) * ufl.dx)
        self.gq_form = fem.form(ufl.inner(self.G_fn, m) * ufl.dx)
        self.F_form = fem.form(ufl.inner(self.q_fn, v) * ufl.dx)
        self.rho_s_eff = p.rho_powder + p.rho_rel * (p.rho_solid - p.rho_powder)
        self.k_s_eff = p.k_powder + p.rho_rel * (p.k_solid - p.k_powder)
        self.rho_L = self.m_nodal * self.rho_s_eff * p.latent_j_per_kg
        self.n_forward = 0
        self.n_reverse = 0

    @classmethod
    def build(cls, shape: str, target_nodes_in_part: int, lc0: float,
              p: fwd.ForwardParams, max_time_s: float,
              sample_dt_s: float | None = None):
        import mesh_gmsh as mg
        kind = "cylinder" if shape == "circle" else "square"
        msh, info, _ = mg.match_lc(kind, int(target_nodes_in_part), float(lc0))
        mats = fwd.build_materials(msh, fwd.in_part_predicate(shape), p)
        eqs = SteadyEqs(msh, mats, p)
        return cls(msh, mats, p, eqs, info,
                   sample_dt_s or p.eqs_update_interval_s, max_time_s)

    # ---- helpers ------------------------------------------------------ #
    def cell_avg(self, nodal: np.ndarray) -> np.ndarray:
        return fwd.cell_average_p1(self.cell_dofs, nodal)

    def cell_avg_T(self, gc: np.ndarray) -> np.ndarray:
        """Transpose of cell_avg (checked by the dot-product identity)."""
        gn = np.zeros(self.vol_nodal.size, dtype=float)
        np.add.at(gn, self.cell_dofs,
                  (np.asarray(gc, float) / self.cell_dofs.shape[1])[:, None])
        return gn

    def sigma_all_from_part(self, sigma_part: np.ndarray) -> np.ndarray:
        a = np.full(self.ncells, self.p.sigma_virgin, dtype=float)
        a[self.eqs.part] = np.asarray(sigma_part, dtype=float)
        return a

    def coupling_factor(self, T_cells, rho_cells) -> np.ndarray:
        p = self.p
        return ((1.0 + p.sigma_temp_coeff_per_K
                 * (np.asarray(T_cells, float) - p.sigma_ref_temp_c))
                * (1.0 + p.sigma_density_coeff
                   * (np.asarray(rho_cells, float) - p.rho_rel)))

    def apply_coupling(self, base_all, T_cells, rho_cells):
        """forward.apply_sigma_coupling, plus the clip-active mask the VJP needs."""
        p = self.p
        f = self.coupling_factor(T_cells, rho_cells)
        s = base_all * f
        lo = fwd.SIGMA_COUPLING_CLIP_LO * p.sigma_doped
        hi = fwd.SIGMA_COUPLING_CLIP_HI * p.sigma_doped
        s_cl = np.clip(np.nan_to_num(s, nan=p.sigma_doped, posinf=hi, neginf=lo),
                       lo, hi)
        inside = self.mats.mask
        active = inside & (s > lo) & (s < hi)      # clip-inactive subgradient
        return np.where(inside, s_cl, base_all), f, active

    # ---- forward with recording --------------------------------------- #
    def forward(self, sigma_base_part: np.ndarray) -> Trajectory:
        p, eqs = self.p, self.eqs
        base_all = self.sigma_all_from_part(sigma_base_part)
        rho_cells = np.full(self.ncells, p.rho_rel)
        events: list[_Event] = []

        sig0, _f0, _a0 = self.apply_coupling(
            base_all, np.full(self.ncells, p.preheat_c), rho_cells)
        eqs.set_sigma_all(sig0)
        st0 = eqs.solve_state()
        events.append(_Event(0, sig0, st0, None, rho_cells.copy(), st0.q.copy()))

        self.q_fn.x.array[:] = st0.q.astype(dolfinx.default_scalar_type)
        state = {"next": float(p.eqs_update_interval_s)
                 if p.eqs_update_interval_s > 0 else float("inf")}
        rec: dict = {}

        def hook(t_now, T_cells, rho_c):
            if t_now < state["next"] - 1e-12:
                return None
            while t_now >= state["next"] - 1e-12:
                state["next"] += float(p.eqs_update_interval_s)
            sig, _f, _a = self.apply_coupling(base_all, T_cells, rho_c)
            eqs.set_sigma_all(sig)
            stk = eqs.solve_state()
            events.append(_Event(-1, sig, stk, np.asarray(T_cells, float).copy(),
                                 np.asarray(rho_c, float).copy(), stk.q.copy()))
            return stk.q

        out = fwd.march_enthalpy(
            self.msh, p, mats=self.mats, q_dg0=self.q_fn,
            max_time_s=self.max_time_s, phi_target=2.0, L=self.L,
            sample_dt_s=self.sample_dt_s, resolve_hook=hook, record=rec)
        self.n_forward += 1

        n_steps = len(rec["T_steps"])
        ev_steps = [e[0] for e in rec["q_events"]]
        for k, e in enumerate(events):
            e.step = int(ev_steps[k])
        step_event = np.zeros(n_steps, dtype=int)
        for k, s_i in enumerate(ev_steps):
            step_event[int(s_i):] = k
        return Trajectory(T_steps=rec["T_steps"], events=events,
                          step_event=step_event, T_final=out["T"],
                          n_steps=n_steps, out=out)

    # ---- objective ----------------------------------------------------- #
    # Phase C selector. The DEFAULT is "symmetric", which is bit-identically
    # the Phase B functional sum_i vol_i (phi_i - chi_i)^2, so every Phase B
    # gate still measures what it measured.
    objective_name: str = "symmetric"
    objective_w_ratio: float | None = None

    def set_objective(self, name: str, w_ratio: float | None = None) -> None:
        from solve3d import objective as _obj
        if name not in _obj.OBJECTIVES:
            raise ValueError(f"unknown objective {name!r}")
        self.objective_name = name
        self.objective_w_ratio = w_ratio

    def J_of_T(self, T: np.ndarray) -> float:
        from solve3d import objective as _obj
        phi = fwd.phase_fraction(T, self.p)[0]
        fn = _obj.OBJECTIVES[self.objective_name][0]
        if self.objective_name == "asymmetric":
            return fn(phi, self.m_nodal, self.vol_nodal,
                      w_ratio=self.objective_w_ratio)
        return fn(phi, self.m_nodal, self.vol_nodal)

    def J(self, sigma_base_part: np.ndarray) -> float:
        return self.J_of_T(self.forward(sigma_base_part).T_final)

    def seed_T(self, T: np.ndarray) -> np.ndarray:
        """dJ/dT at the read state.

        Two subgradients compose here and both take the forward's own side:
        the melt-fraction clip (dphi/dT is zero outside the melt window) and,
        for the asymmetric objective, its two hinges."""
        from solve3d import objective as _obj
        p = self.p
        u = (T - p.t_pc_c) / p.dt_pc_c + 0.5
        phi = np.clip(u, 0.0, 1.0)
        dphi = np.where((u > 0.0) & (u < 1.0), 1.0 / p.dt_pc_c, 0.0)
        dfn = _obj.OBJECTIVES[self.objective_name][1]
        if self.objective_name == "asymmetric":
            dJ_dphi = dfn(phi, self.m_nodal, self.vol_nodal,
                          w_ratio=self.objective_w_ratio)
        else:
            dJ_dphi = dfn(phi, self.m_nodal, self.vol_nodal)
        return dJ_dphi * dphi

    # ---- one march step, forward cache + VJP --------------------------- #
    def _step_cache(self, T_in: np.ndarray, F: np.ndarray) -> dict:
        p = self.p
        u = (T_in - p.t_pc_c) / p.dt_pc_c + 0.5
        phi = np.clip(u, 0.0, 1.0)
        m_phi = (u > 0.0) & (u < 1.0)
        rho_part = (1.0 - phi) * self.rho_s_eff + phi * p.rho_liquid
        cp_part = (1.0 - phi) * p.cp_solid + phi * p.cp_liquid
        rho = (1.0 - self.m_nodal) * p.rho_powder + self.m_nodal * rho_part
        cp = (1.0 - self.m_nodal) * p.cp_powder + self.m_nodal * cp_part
        rho_cp = rho * cp
        phi_c = self.cell_avg(phi)
        k_part = (1.0 - phi_c) * self.k_s_eff + phi_c * p.k_liquid
        k_cells = np.where(self.doped_cells > 0.5, k_part, p.k_powder)
        self.k_fn.x.array[:] = k_cells.astype(dolfinx.default_scalar_type)
        self.T_fn.x.array[:] = T_in.astype(dolfinx.default_scalar_type)
        KT = fwd._assemble_real(self.diff_form)
        C = (fwd._assemble_real(self.conv_form) if p.conv_h != 0.0
             else np.zeros_like(KT))
        num = -KT + F - C
        lo = p.t_pc_c - p.dt_pc_c / 2.0
        frac_u = (T_in - lo) / p.dt_pc_c
        m_frac = (frac_u > 0.0) & (frac_u < 1.0)
        H = fwd.enthalpy_from_T(T_in, rho_cp, self.rho_L, p)
        H2 = H + p.dt_s * np.nan_to_num(num) / np.where(
            self.vol_nodal > 0, self.vol_nodal, 1.0)
        T_new = fwd.T_from_enthalpy(H2, rho_cp, self.rho_L, p)
        dT_raw = T_new - T_in
        dT = np.clip(dT_raw, -p.max_dt_step_c, p.max_dt_step_c)
        m_cap = (np.abs(dT_raw) <= p.max_dt_step_c)
        T_cand = T_in + dT
        T_out = np.clip(T_cand, p.temp_min_c, p.temp_max_c)
        m_tmp = (T_cand >= p.temp_min_c) & (T_cand <= p.temp_max_c)
        return {"T_in": T_in, "phi": phi, "m_phi": m_phi, "rho": rho, "cp": cp,
                "rho_cp": rho_cp, "phi_c": phi_c, "k_cells": k_cells,
                "num": num, "H2": H2, "m_frac": m_frac, "m_cap": m_cap,
                "m_tmp": m_tmp, "T_out": T_out}

    def _dT_from_enthalpy(self, H, rho_cp):
        """(dT/dH, dT/d(rho_cp)) for the exact piecewise-linear inversion.

        The branch boundaries themselves depend on rho_cp, but T_from_enthalpy
        is CONTINUOUS across them, so the subgradient is the forward's own
        branch -- the same rule the 2-D lane applies to its clips."""
        p = self.p
        lo = p.t_pc_c - p.dt_pc_c / 2.0
        rL = self.rho_L
        H_lo = rho_cp * lo
        H_hi = rho_cp * (lo + p.dt_pc_c) + rL
        below, above = H <= H_lo, H >= H_hi
        D = rho_cp + rL / p.dt_pc_c
        dT_dH = np.where(below | above, 1.0 / rho_cp, 1.0 / D)
        dT_drc = np.where(
            below, -H / rho_cp ** 2,
            np.where(above, -(H - rL) / rho_cp ** 2,
                     -(H + rL * lo / p.dt_pc_c) / D ** 2))
        return dT_dH, dT_drc

    def step_vjp(self, c: dict, gT_out: np.ndarray):
        """Return (dJ/dT_in, dJ/dF) for one march step."""
        p = self.p
        g = np.asarray(gT_out, float) * c["m_tmp"]        # T_out clip
        gT_in = g.copy()                                   # carried identity
        g_dT = g * c["m_cap"]                              # dT cap
        g_Tnew = g_dT
        gT_in -= g_dT                                      # dT_raw = T_new - T_in

        dT_dH, dT_drc = self._dT_from_enthalpy(c["H2"], c["rho_cp"])
        g_H2 = g_Tnew * dT_dH
        g_rho_cp = g_Tnew * dT_drc

        g_num = g_H2 * p.dt_s / np.where(self.vol_nodal > 0, self.vol_nodal, 1.0)
        g_H = g_H2
        gT_in += g_H * (c["rho_cp"] + self.rho_L * c["m_frac"] / p.dt_pc_c)
        g_rho_cp = g_rho_cp + g_H * c["T_in"]

        g_F = g_num.copy()
        self.G_fn.x.array[:] = g_num.astype(dolfinx.default_scalar_type)
        self.k_fn.x.array[:] = c["k_cells"].astype(dolfinx.default_scalar_type)
        self.T_fn.x.array[:] = c["T_in"].astype(dolfinx.default_scalar_type)
        gT_in -= fwd._assemble_real(self.diffG_form)        # K symmetric
        if p.conv_h != 0.0:
            gT_in -= fwd._assemble_real(self.convG_form)    # M_top symmetric
        g_k_cells = -np.real(fem.assemble_vector(self.gk_form).array)

        g_phi_c = g_k_cells * (self.doped_cells > 0.5) * (p.k_liquid - self.k_s_eff)
        g_phi = self.cell_avg_T(g_phi_c)

        g_rho = g_rho_cp * c["cp"]
        g_cp = g_rho_cp * c["rho"]
        g_phi += (g_rho * self.m_nodal * (p.rho_liquid - self.rho_s_eff)
                  + g_cp * self.m_nodal * (p.cp_liquid - p.cp_solid))

        gT_in += g_phi * c["m_phi"] / p.dt_pc_c            # phi clip subgradient
        return gT_in, g_F

    def gF_to_gq(self, g_F: np.ndarray) -> np.ndarray:
        """dJ/dQ_c from dJ/dF_i, since F_i = sum_c Q_c * int_c phi_i."""
        self.G_fn.x.array[:] = np.asarray(g_F, float).astype(
            dolfinx.default_scalar_type)
        return np.real(fem.assemble_vector(self.gq_form).array).astype(float)

    # ---- reverse march ------------------------------------------------- #
    def gradient(self, sigma_base_part: np.ndarray, tr: Trajectory | None = None,
                 mutate: str | None = None,
                 checkpoint_interval: int | None = None,
                 read_step: int | None = None):
        """dJ/d(sigma_base) on the part cells; store-everything by default.

        `read_step` = the number of march steps completed at the read state
        (None = the horizon). The envelope layer (B4) passes the trajectory
        argmin here; nothing else about the sweep changes, which is exactly the
        statement that no dt*/ds term exists."""
        p, eqs = self.p, self.eqs
        tr = tr or self.forward(sigma_base_part)
        base_all = self.sigma_all_from_part(sigma_base_part)
        n_read = tr.n_steps if read_step is None else int(read_step)
        gT = self.seed_T(self.state_at(tr, n_read))
        g_base = np.zeros(self.ncells, dtype=float)
        n_ev = len(tr.events)
        gQ = [np.zeros(self.ncells) for _ in range(n_ev)]
        recompute_steps = 0

        T_of_step = self._checkpoint_reader(tr, checkpoint_interval)
        for j in range(n_read - 1, -1, -1):
            k = int(tr.step_event[j])
            T_in, extra = T_of_step(j)
            recompute_steps += extra
            c = self._step_cache(T_in, self._F_of_event(tr.events[k]))
            gT, g_F = self.step_vjp(c, gT)
            gQ[k] += self.gF_to_gq(g_F)
            if j == tr.events[k].step:
                ev = tr.events[k]
                dsig_eff = self._eqs_vjp_at(ev, gQ[k], mutate=mutate)
                T_c = (ev.T_cells if ev.T_cells is not None
                       else np.full(self.ncells, p.preheat_c))
                _s, f, active = self.apply_coupling(base_all, T_c, ev.rho_cells)
                d_full = np.zeros(self.ncells)
                d_full[eqs.part] = dsig_eff
                g_base += np.where(self.mats.mask, d_full * f * active, d_full)
                # d sigma_eff / d T -- the re-solve coupling, injected as a
                # dJ/dT contribution at exactly this instant
                if ev.T_cells is not None and p.sigma_temp_coeff_per_K != 0.0:
                    dens = (1.0 + p.sigma_density_coeff
                            * (ev.rho_cells - p.rho_rel))
                    g_cells = (d_full * base_all * p.sigma_temp_coeff_per_K
                               * dens * active)
                    gT += self.cell_avg_T(g_cells)
        self.n_reverse += 1
        info = {"n_events": n_ev, "n_steps": tr.n_steps, "read_step": n_read,
                "recomputed_steps": recompute_steps,
                "checkpoint_interval": checkpoint_interval,
                "stored_state_bytes": self.stored_state_bytes(tr, checkpoint_interval)}
        return g_base[eqs.part], info

    def _F_of_event(self, ev: _Event) -> np.ndarray:
        self.q_fn.x.array[:] = ev.q.astype(dolfinx.default_scalar_type)
        return fwd._assemble_real(self.F_form)

    def _eqs_vjp_at(self, ev: _Event, gQ: np.ndarray,
                    mutate: str | None = None) -> np.ndarray:
        """EQS adjoint for one event: restore its sigma and V, re-factorize,
        then reuse the SAME operator for A^H."""
        self.eqs.set_sigma_all(ev.sigma_eff)
        self.eqs._factorize()
        self.eqs.Vfun.x.array[:] = ev.state.V
        self.eqs.Vfun.x.scatter_forward()
        return self.eqs.vjp_q(ev.state, gQ, mutate=mutate)

    def _checkpoint_reader(self, tr: Trajectory, interval: int | None):
        """Store-everything (interval None) or interval checkpointing."""
        if interval is None:
            return lambda j: (tr.T_steps[j], 0)
        anchors = {i: tr.T_steps[i] for i in range(0, tr.n_steps, int(interval))}
        cache: dict[int, np.ndarray] = {}

        def read(j: int):
            if j in cache:
                return cache.pop(j), 0
            a = max(i for i in anchors if i <= j)
            T = anchors[a].copy()
            n = 0
            cache.clear()
            for i in range(a, j + 1):
                cache[i] = T.copy()
                if i < j:
                    k = int(tr.step_event[i])
                    c = self._step_cache(T, self._F_of_event(tr.events[k]))
                    T = c["T_out"]
                    n += 1
            return cache.pop(j), n
        return read


    # ---- state / trajectory readers ------------------------------------ #
    def state_at(self, tr: Trajectory, k: int) -> np.ndarray:
        """The march state after `k` steps (k = n_steps is the horizon)."""
        k = int(k)
        if k >= tr.n_steps:
            return tr.T_final
        return tr.T_steps[k]

    def J_trajectory(self, tr: Trajectory) -> np.ndarray:
        """J at every stored read state, index j = after j steps."""
        return np.array([self.J_of_T(self.state_at(tr, j))
                         for j in range(tr.n_steps + 1)], dtype=float)

    def envelope_read(self, tr: Trajectory) -> dict:
        """t_stop = argmin over the arm's OWN stored trajectory (the frozen 2-D
        stop rule, FROZEN_CONVENTIONS_2D.md section 6). `at_horizon` is flagged
        whenever the minimum sits on the last stored step, which makes that
        arm's J an UPPER BOUND rather than a converged read."""
        Js = self.J_trajectory(tr)
        k = int(np.argmin(Js))
        return {"argmin_step": k, "J": float(Js[k]), "n_steps": tr.n_steps,
                "at_horizon": bool(k >= tr.n_steps), "J_trajectory": Js}

    def J_envelope(self, sigma_base_part: np.ndarray) -> float:
        return float(np.min(self.J_trajectory(self.forward(sigma_base_part))))

    def stored_state_bytes(self, tr: Trajectory, interval: int | None) -> int:
        """Bytes of march state the reverse sweep must hold."""
        per = int(tr.T_steps[0].nbytes) if tr.T_steps else 0
        if interval is None:
            return per * tr.n_steps
        n_anchor = len(range(0, tr.n_steps, int(interval)))
        return per * (n_anchor + int(interval))     # anchors + one live segment

    # ---- design map (FROZEN_CONVENTIONS_2D sections 4 and 7) ------------- #
    def design_to_sigma(self, v: np.ndarray) -> np.ndarray:
        """sigma = sigma_v + sat * fill * (sigma_d0 - sigma_v), conductivity ONLY.

        On a conforming mesh `fill` is exactly 1 on part cells and 0 elsewhere
        (no partial cells -- asserted in the test, not assumed), so the 2-D
        sub-pixel fill factor collapses to the part indicator. The section-4
        rule "outside the part the saturation is held at 1.0" is therefore
        inert here: with fill = 0 the outside conductivity is sigma_virgin
        whatever the saturation is. It is honoured rather than dropped, and the
        reason it cannot bite is recorded."""
        p = self.p
        return p.sigma_virgin + np.asarray(v, float) * (p.sigma_doped - p.sigma_virgin)

    def design_jvp(self, dv: np.ndarray) -> np.ndarray:
        p = self.p
        return np.asarray(dv, float) * (p.sigma_doped - p.sigma_virgin)

    def design_vjp(self, g_sigma: np.ndarray) -> np.ndarray:
        p = self.p
        return np.asarray(g_sigma, float) * (p.sigma_doped - p.sigma_virgin)

    def J_of_design(self, v: np.ndarray) -> float:
        return self.J(self.design_to_sigma(v))

    def J_envelope_of_design(self, v: np.ndarray) -> float:
        return self.J_envelope(self.design_to_sigma(v))

    def gradient_design(self, v: np.ndarray, **kw):
        g_sigma, info = self.gradient(self.design_to_sigma(v), **kw)
        return self.design_vjp(g_sigma), info
