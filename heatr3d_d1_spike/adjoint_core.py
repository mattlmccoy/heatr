"""D1 Task 5: hand-assembled adjoint of the EQS solve w.r.t. a DG0 sigma field.

WHY BY HAND
-----------
dolfinx-adjoint / pyadjoint has no release wired to dolfinx 0.11 (the spike env
resolved dolfinx 0.11.0, see results.json["task0"]["dolfinx_version"]), so the
adjoint here is assembled by hand. For ONE linear solve that is standard work:
one transposed solve plus the partial derivatives. The point of Task 5 is to
measure how much work it is and whether the gradient survives an FD gate.

THE FORWARD MAP  s -> J
-----------------------
s in R^{Ncell_part}   sigma on the part cells (DG0; one dof per tetrahedron).
gamma_c = sigma_c + i*omega*eps0*eps_r_c   (eps_r fixed; d gamma/d sigma = 1)

A(s) V = b(s)          div(gamma grad V) = 0, V = 860 / 0 on the electrodes
                       (eqs_common.solve_eqs, identical forms)
e2_c    = |grad Vr|^2 + |grad Vi|^2          (P1 -> exactly cell-constant)
Qraw_c  = 0.5 * sigma_c * e2_c               (heatr3d.compute_qrf_3d, masked)
scale   = p_target / sum_c Qraw_c |K_c|      (the fixed-power renormalization)
Q_c     = scale * Qraw_c
Qbar    = sum_c Q_c |K_c| / sum_c |K_c|      (volume-weighted part mean)
J       = sum_c (Q_c - Qbar)^2 |K_c|         (heating non-uniformity)

sums over part cells; p_target = power_density_w_per_m3 * V_doped is a CONSTANT
(geometry only), so the renormalization makes `scale` a nonlocal function of
EVERY sigma dof. It is differentiated through, not frozen.

IDENTITY WORTH RECORDING: because the renormalization pins total power,
Qbar == p_target / V_doped == power_density_w_per_m3 EXACTLY, independent of s.
So dQbar/ds = 0 analytically. That is an output of the renormalization, not an
assumption: run_adjoint_demo.py measures Qbar's FD sensitivity and records it.

THE ADJOINT
-----------
Residual (BC rows folded in, so db/ds is not a separate term):
    R_i(V, s) = (A_full(s) V)_i        i free      (V_bc already = g)
    R_i(V, s) = V_i - g_i              i on an electrode
    => dR_i/ds_c = (K_c V)_i for free i, 0 for BC i, where K_c is the (REAL)
       element stiffness of cell c: A_full = sum_c gamma_c K_c.

J is real; V is complex. Working in the real unknowns u = (Vr, Vi) and
translating back (see the derivation in the report), the adjoint is the
CONJUGATE transpose:
    A^H lambda = rho,   rho_j = dJ/dVr_j + i dJ/dVi_j
    dJ/ds_c = dJ/ds_c|_explicit - Re( lambda^H K_c V )

and A (real basis functions, complex coefficient) is complex SYMMETRIC, so
A^H = conj(A) and the adjoint solve reuses the FORWARD LU factorization:
    A x = conj(rho)  ->  lambda = conj(x).
BC rows/cols are eliminated symmetrically and the adjoint rhs is zeroed there,
so lambda = 0 on the electrodes, which is exactly the "dR_bc/ds = 0" statement.

Both partials are assembled with dolfinx forms built ONCE and re-evaluated with
updated coefficients, so the forward and the adjoint share the same discrete
operators by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import eqs_common as ec           # applies jit_fix before dolfinx
import femutils as fu

import dolfinx
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fp
from mpi4py import MPI
from petsc4py import PETSc

KSP_LU = {"ksp_type": "preonly", "pc_type": "lu"}
N_REFINE = 2          # iterative-refinement sweeps after the LU back-substitution


def smooth_sigma(mp: np.ndarray, sigma0: float = ec.SIGMA_DOPED,
                 amp: float = 0.4, r_part: float = 0.010,
                 L: float = ec.L_DOMAIN) -> np.ndarray:
    """A smooth, non-degenerate baseline grading on the part cells.

    mp is (3, N) cell midpoints. Stays inside [0.4, 1.6] * sigma0 > 0, so the
    Q clip is never active and no subgradient question arises.
    """
    x, y, z = mp[0], mp[1], mp[2]
    return sigma0 * (1.0
                     + amp * np.sin(np.pi * x / r_part) * np.cos(np.pi * y / r_part)
                     + 0.2 * np.sin(np.pi * z / (L / 2.0)))


@dataclass
class Forward:
    """Everything the gradient needs from one forward solve."""
    s: np.ndarray
    V: np.ndarray                 # complex nodal values
    e2: np.ndarray                # per-cell |grad V|^2 (all cells)
    qraw: np.ndarray              # per-cell, masked to the part
    scale: float
    p_now: float
    q: np.ndarray                 # per-cell renormalized Q
    qbar: float
    J: float
    qraw_min_in_part: float


class AdjointCase:
    """One mesh + one part geometry; sigma on the part is the design vector."""

    def __init__(self, msh, in_part: Callable[[np.ndarray], np.ndarray],
                 L: float = ec.L_DOMAIN, degree: int = 1):
        if not ec.IS_COMPLEX:
            raise RuntimeError("Task 5 assumes the complex scalar build; "
                               "the real-split path needs its own adjoint.")
        self.msh = msh
        self.mats = ec.materials(msh, in_part=in_part)
        self.Q0 = self.mats.dg0
        self.W = ec.functionspace(msh, ("Lagrange", degree))

        self.vol = fu.cell_volumes(msh, self.Q0)          # |K_c| per DG0 dof
        self.part = np.where(np.real(self.mats.doped.x.array) > 0.5)[0]
        self.v_doped = float(self.vol[self.part].sum())
        self.p_target = ec.POWER_DENSITY_W_PER_M3 * self.v_doped

        # --- forms, built once, coefficients updated in place ---------------
        st = dolfinx.default_scalar_type
        u, w = ufl.TrialFunction(self.W), ufl.TestFunction(self.W)
        gam = self.mats.sigma + 1j * ec.OMEGA * ec.EPS0 * self.mats.eps_r
        self.a_form = fem.form(ufl.inner(gam * ufl.grad(u), ufl.grad(w)) * ufl.dx)
        self.L_form = fem.form(
            ufl.inner(fem.Constant(msh, st(0.0)), w) * ufl.dx)

        d_lo, d_hi = ec._electrode_dofs(self.W, msh, L)
        self.bcs = [fem.dirichletbc(st(ec.V_LO), d_lo, self.W),
                    fem.dirichletbc(st(ec.V_HI), d_hi, self.W)]
        self.bc_dofs = np.unique(np.concatenate([d_lo, d_hi]))

        self.Vfun = fem.Function(self.W)          # forward solution (complex)
        self.lam = fem.Function(self.W)           # adjoint solution (complex)
        self.what = fem.Function(self.Q0)         # dJ/de2_c / |K_c|

        m = ufl.TestFunction(self.Q0)
        # per-cell integral of |grad V|^2 -> e2_c after dividing by |K_c|
        self.e2_form = fem.form(
            ufl.inner(ufl.inner(ufl.grad(self.Vfun), ufl.grad(self.Vfun)), m) * ufl.dx)
        # rho_j = dJ/dVr_j + i dJ/dVi_j = 2 sum_c w_c grad(V)_c . grad(phi_j)_c
        self.rho_form = fem.form(
            ufl.inner(2.0 * self.what * ufl.grad(self.Vfun), ufl.grad(w)) * ufl.dx)
        # per-cell lambda^H K_c V = int_c grad(V) . conj(grad(lambda))
        self.dRds_form = fem.form(
            ufl.inner(ufl.inner(ufl.grad(self.Vfun), ufl.grad(self.lam)), m) * ufl.dx)

        self._ksp = None
        self._A = None

    # ------------------------------------------------------------------ #
    def _set_sigma(self, s: np.ndarray) -> None:
        arr = np.real(self.mats.sigma.x.array).copy()
        arr[self.part] = np.asarray(s, dtype=float)
        self.mats.sigma.x.array[:] = arr.astype(dolfinx.default_scalar_type)

    def _factorize(self):
        if self._ksp is not None:
            self._ksp.destroy()
            self._A.destroy()
        A = fp.assemble_matrix(self.a_form, bcs=self.bcs)
        A.assemble()
        ksp = PETSc.KSP().create(self.msh.comm)
        ksp.setOperators(A)
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
    def forward(self, s: np.ndarray) -> Forward:
        self._set_sigma(s)
        ksp = self._factorize()
        b = self._rhs()
        x = self._A.createVecRight()
        ksp.solve(b, x)
        # Iterative refinement. gamma spans sigma_doped/sigma_virgin = 4e6, so
        # the LU backward error, while tiny, sets the FD noise floor on J (the
        # per-dof FD error was measured to be a CONSTANT absolute quantity,
        # independent of |dJ/ds| -- the signature of a noise floor, not of a
        # wrong gradient). Two refinement sweeps cost two triangular solves.
        r = self._A.createVecRight()
        dx = self._A.createVecRight()
        for _ in range(N_REFINE):
            self._A.mult(x, r)
            r.aypx(-1.0, b)                 # r <- b - A x
            ksp.solve(r, dx)
            x.axpy(1.0, dx)
        self.res_norm = float(r.norm() / b.norm()) if N_REFINE else float("nan")
        r.destroy()
        dx.destroy()
        self.Vfun.x.array[:] = x.array_r if not ec.IS_COMPLEX else x.array
        self.Vfun.x.scatter_forward()
        b.destroy()
        x.destroy()

        e2 = np.real(fem.assemble_vector(self.e2_form).array) / self.vol
        sig_all = np.real(self.mats.sigma.x.array)
        qraw = np.zeros_like(e2)
        qraw[self.part] = 0.5 * sig_all[self.part] * e2[self.part]
        p_now = float((qraw[self.part] * self.vol[self.part]).sum())
        scale = self.p_target / p_now
        q = scale * qraw
        qbar = float((q[self.part] * self.vol[self.part]).sum() / self.v_doped)
        J = float((((q[self.part] - qbar) ** 2) * self.vol[self.part]).sum())
        return Forward(s=np.asarray(s, dtype=float).copy(),
                       V=np.array(self.Vfun.x.array), e2=e2, qraw=qraw,
                       scale=scale, p_now=p_now, q=q, qbar=qbar, J=J,
                       qraw_min_in_part=float(qraw[self.part].min()))

    # ------------------------------------------------------------------ #
    def gradient(self, fwd: Forward, mutate: str | None = None) -> np.ndarray:
        """dJ/ds on the part cells (same ordering as self.part).

        Assumes `fwd` is the most recent forward (the LU and Vfun are reused).

        `mutate` deliberately breaks the gradient so the FD gate can be shown to
        be non-vacuous: "renorm_frozen" treats the fixed-power scale as a
        constant, "adjoint_dropped" keeps only the explicit dQ/dsigma partial.
        """
        if mutate not in (None, "renorm_frozen", "adjoint_dropped"):
            raise ValueError(mutate)
        p = self.part
        vol = self.vol
        # dJ/dQ_c  (Qbar is s-independent: see the module docstring identity)
        G = np.zeros_like(fwd.q)
        G[p] = 2.0 * (fwd.q[p] - fwd.qbar) * vol[p]
        # through the renormalization: Q = (p_target / sum Qraw|K|) * Qraw
        S = float((G[p] * fwd.qraw[p]).sum())
        g = np.zeros_like(fwd.q)
        g[p] = fwd.scale * G[p]
        if mutate != "renorm_frozen":
            g[p] -= (S * fwd.scale / fwd.p_now) * vol[p]

        sig_all = np.real(self.mats.sigma.x.array)
        # explicit sigma dependence of Qraw = 0.5 sigma e2
        d_expl = g[p] * 0.5 * fwd.e2[p]
        if mutate == "adjoint_dropped":
            return d_expl
        # dJ/de2_c, divided by |K_c| for the DG0 coefficient in rho_form
        w = np.zeros_like(fwd.q)
        w[p] = g[p] * 0.5 * sig_all[p]
        self.what.x.array[:] = (w / vol).astype(dolfinx.default_scalar_type)

        rho = fem.assemble_vector(self.rho_form).array.copy()
        rhs = np.conj(rho)
        rhs[self.bc_dofs] = 0.0                    # dR_bc/ds = 0
        bvec = self._A.createVecRight()
        bvec.array[:] = rhs
        xvec = self._A.createVecRight()
        self._ksp.solve(bvec, xvec)                # A x = conj(rho); A^H = conj(A)
        self.lam.x.array[:] = np.conj(xvec.array)
        self.lam.x.array[self.bc_dofs] = 0.0
        self.lam.x.scatter_forward()
        bvec.destroy()
        xvec.destroy()

        dR = np.real(fem.assemble_vector(self.dRds_form).array)   # lambda^H K_c V
        return d_expl - dR[p]


def fd_central(case: AdjointCase, s: np.ndarray, dof_local: int,
               rel_eps: float) -> tuple[float, float, float]:
    """Central difference of J w.r.t. one part-cell sigma dof."""
    d = rel_eps * abs(float(s[dof_local]))
    sp = s.copy(); sp[dof_local] += d
    sm = s.copy(); sm[dof_local] -= d
    jp = case.forward(sp).J
    jm = case.forward(sm).J
    return (jp - jm) / (2.0 * d), jp, jm
