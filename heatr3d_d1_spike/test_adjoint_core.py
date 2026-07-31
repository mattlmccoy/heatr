"""Fast FD gate for adjoint_core on a TINY mesh (seconds, not minutes).

This is the development loop for Task 5; run_adjoint_demo.py runs the same gate
on the Task-2 coarse mesh and writes results.json. Kept separate so the gate can
be re-run cheaply after any change to eqs_common / adjoint_core.

Also includes two MUTATION checks -- deliberately wrong gradients that the gate
must REJECT -- so a passing gate cannot be a vacuous one:
  * renorm frozen  (scale treated as constant: the term the plan warns about)
  * adjoint term dropped (explicit partial only)

Run: ./env/bin/python test_adjoint_core.py
"""
from __future__ import annotations

import numpy as np

import eqs_common as ec
import adjoint_core as ac

N_BOX = 10
R_PART = 0.010
SEED = 20260731
N_DOFS_TESTED = 3
EPS_SWEEP = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
TOL = 0.01          # the plan's per-dof gate: < 1 % relative
TOL_DIR = 1e-7      # the directional check has no small-|g| noise floor


def in_cylinder(mp: np.ndarray) -> np.ndarray:
    return np.sqrt(mp[0] ** 2 + mp[1] ** 2) <= R_PART


def build_case():
    msh = ec.box_mesh(N_BOX)
    case = ac.AdjointCase(msh, in_cylinder)
    mp = ac.fu.cell_midpoints(msh).T[:, case.part]
    s0 = ac.smooth_sigma(mp)
    return case, s0


def _consistency(case, s0):
    """The forward here must be the SAME operator eqs_common solves with."""
    fwd = case.forward(s0)
    Vr, Vi = ec.solve_eqs(case.msh, case.mats, petsc_options=ac.KSP_LU)
    dv = max(np.abs(np.real(Vr.x.array) - np.real(fwd.V)).max(),
             np.abs(np.real(Vi.x.array) - np.imag(fwd.V)).max()) / ec.V_LO
    q_ec, scale_ec, p_ec = ec.qrf_dg0(case.msh, Vr, Vi, case.mats, premix=False)
    dq = np.abs(np.real(q_ec.x.array) - fwd.q).max() / fwd.q.max()
    return fwd, float(dv), float(dq), float(scale_ec / fwd.scale - 1.0)


def main() -> int:
    case, s0 = build_case()
    print(f"cells={case.vol.size} part_cells={case.part.size} "
          f"V_doped={case.v_doped:.6e} m^3")
    fwd, dv, dq, dscale = _consistency(case, s0)
    print(f"consistency vs eqs_common: dV/860={dv:.3e} dQ/Qmax={dq:.3e} "
          f"dscale={dscale:.3e}")
    assert dv < 1e-10 and dq < 1e-10, "forward is not eqs_common's operator"
    print(f"J={fwd.J:.10e} scale={fwd.scale:.6e} qbar={fwd.qbar:.10e} "
          f"power_density={ec.POWER_DENSITY_W_PER_M3:.10e} "
          f"qraw_min={fwd.qraw_min_in_part:.3e}")
    assert fwd.qraw_min_in_part > 0.0, "Q clip would be active (non-smooth)"

    grad = case.gradient(fwd)
    rng = np.random.default_rng(SEED)
    dofs = rng.choice(case.part.size, size=N_DOFS_TESTED, replace=False)

    ok = True
    gmax = np.abs(grad).max()
    for k in dofs:
        best, best_abs = np.inf, np.inf
        row = []
        for eps in EPS_SWEEP:
            gfd, _, _ = ac.fd_central(case, s0, int(k), eps)
            rel = abs(gfd - grad[k]) / max(abs(gfd), 1e-300)
            row.append((eps, gfd, rel))
            best = min(best, rel)
            best_abs = min(best_abs, abs(gfd - grad[k]))
        print(f"dof {int(k):6d} adj={grad[k]:+.10e}  " +
              "  ".join(f"[{e:.0e}] {r:.2e}" for e, _, r in row) +
              f"  best_rel={best:.2e} best_abs={best_abs:.2e} "
              f"|g|/|g|max={abs(grad[k])/gmax:.2e}")
        ok &= best < TOL
    case.forward(s0)          # the FD sweep clobbered the stored forward state

    # --- directional derivative: exercises ALL part dofs at once, so the FD
    # signal is O(N) larger than a single-dof bump and the noise floor that
    # limits small-|g| dofs above is irrelevant. This is the sharpest test of
    # the NONLOCAL renormalization term.
    d = rng.standard_normal(case.part.size)
    d /= np.linalg.norm(d)
    gd = float(grad @ d)
    print(f"directional: adj={gd:+.12e}")
    dir_best = np.inf
    for eps in EPS_SWEEP:
        h = eps * float(np.abs(s0).mean())
        jp = case.forward(s0 + h * d).J
        jm = case.forward(s0 - h * d).J
        fd = (jp - jm) / (2 * h)
        rel = abs(fd - gd) / abs(fd)
        dir_best = min(dir_best, rel)
        print(f"  [{eps:.0e}] fd={fd:+.12e} rel={rel:.3e}")
    case.forward(s0)

    # --- MUTATION checks: gradients that are deliberately wrong must FAIL ---
    fwd = case.forward(s0)
    for name in ("renorm_frozen", "adjoint_dropped"):
        gm = case.gradient(fwd, mutate=name)
        rel = np.abs(gm[dofs] - grad[dofs]) / np.abs(grad[dofs])
        rel_dir = abs(float(gm @ d) - gd) / abs(gd)
        print(f"mutation {name:16s} rel-vs-true at the FD dofs: "
              f"{np.array2string(rel, precision=3)}  directional: {rel_dir:.3e}")
        assert rel.max() > 0.01, f"mutation {name} is NOT detected by the gate"

    assert ok, "per-dof FD gate FAILED"
    assert dir_best < TOL_DIR, f"directional FD gate FAILED ({dir_best:.2e})"
    print("PASS: hand adjoint matches central FD; both mutants are rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
