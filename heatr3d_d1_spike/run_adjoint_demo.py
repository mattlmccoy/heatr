"""D1 Task 5: adjoint-readiness demo on the Task-2 coarse mesh.

RUNS IN THE SPIKE ENV:
    heatr3d_d1_spike/env/bin/python heatr3d_d1_spike/run_adjoint_demo.py

Same mesh as Task 2 "coarse" (extruded circle d = 20 mm, in-part element size
matched to heatr3d n = 64 by in-part node count -- mesh_gmsh.match_lc with the
same target and the same gmsh seed, so it is the same mesh).

Design variable: sigma as a spatially varying DG0 Function on the part
(one dof per tetrahedron in the part).
Objective:  J = int_part (Q_rf - mean_part(Q_rf))^2 dV, with Q_rf built exactly
as eqs_common.qrf_dg0 builds it -- INCLUDING the fixed-power renormalization,
which is differentiated through (see adjoint_core's docstring).

GATE (from the plan): the hand adjoint matches CENTRAL finite differences on 5
seeded-random sigma dofs to < 1 % relative each.

Two things are also recorded because they are what make the gate non-vacuous:
  * a directional-derivative FD (the full gradient contracted with one random
    direction) -- no small-|g| noise floor, so it pins the gradient ~5 orders
    tighter than the per-dof gate;
  * two MUTANT gradients (renorm frozen / adjoint term dropped) which the gate
    must reject. If the gate could not tell them apart it would prove nothing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import eqs_common as ec           # applies jit_fix before dolfinx
import femutils as fu
import mesh_gmsh as mg
import adjoint_core as ac

HERE = Path(__file__).resolve().parent
R_PART = 0.010
SEED = 20260731
N_DOFS = 5
EPS_SWEEP = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
TOL_PER_DOF = 0.01           # the plan's gate
TOL_DIR = 1e-7


def in_cylinder(mp: np.ndarray) -> np.ndarray:
    return np.sqrt(mp[0] ** 2 + mp[1] ** 2) <= R_PART


def main() -> int:
    p = HERE / "results.json"
    res = json.loads(p.read_text())
    ref64 = res["task2_ref"]["runs"]["n64"]
    target = int(ref64["n_voxels_in_part"])
    h = float(ref64["h_m"])

    t0 = time.perf_counter()
    msh, info, hist = mg.match_lc("cylinder", target, lc0=h)
    t_mesh = time.perf_counter() - t0
    print(f"mesh: lc_part={info.lc_part:.6e} nodes_in_part={info.n_nodes_in_part} "
          f"dofs={info.n_nodes_total} cells={info.n_cells_total} "
          f"({t_mesh:.1f} s)", flush=True)

    case = ac.AdjointCase(msh, in_cylinder)
    mp = fu.cell_midpoints(msh).T[:, case.part]
    s0 = ac.smooth_sigma(mp)
    print(f"design vector: {case.part.size} part cells, "
          f"sigma in [{s0.min():.5f}, {s0.max():.5f}] S/m", flush=True)

    # --- assembly consistency: this forward must BE eqs_common's operator ---
    t0 = time.perf_counter()
    fwd = case.forward(s0)
    t_fwd = time.perf_counter() - t0
    print(f"forward: {t_fwd:.2f} s  J={fwd.J:.10e}", flush=True)
    Vr, Vi = ec.solve_eqs(msh, case.mats, petsc_options=ac.KSP_LU)
    dv = float(max(np.abs(np.real(Vr.x.array) - np.real(fwd.V)).max(),
                   np.abs(np.real(Vi.x.array) - np.imag(fwd.V)).max()) / ec.V_LO)
    q_ec, scale_ec, p_ec = ec.qrf_dg0(msh, Vr, Vi, case.mats, premix=False)
    dq = float(np.abs(np.real(q_ec.x.array) - fwd.q).max() / fwd.q.max())
    print(f"consistency vs eqs_common: dV/860={dv:.3e} dQ/Qmax={dq:.3e}", flush=True)

    fwd = case.forward(s0)          # restore state after the eqs_common solves
    t0 = time.perf_counter()
    grad = case.gradient(fwd)
    t_grad = time.perf_counter() - t0
    print(f"adjoint gradient ({grad.size} dofs): {t_grad:.2f} s "
          f"= {t_grad / t_fwd:.2f} x one forward", flush=True)

    rng = np.random.default_rng(SEED)
    dofs = rng.choice(case.part.size, size=N_DOFS, replace=False)

    per_dof = []
    for k in dofs:
        k = int(k)
        rows, best, best_abs = [], np.inf, np.inf
        for eps in EPS_SWEEP:
            gfd, jp, jm = ac.fd_central(case, s0, k, eps)
            rel = abs(gfd - grad[k]) / max(abs(gfd), 1e-300)
            rows.append({"rel_eps": eps, "h_abs": eps * abs(float(s0[k])),
                         "fd_central": float(gfd),
                         "rel_err": float(rel),
                         "abs_err": float(abs(gfd - grad[k]))})
            best = min(best, rel)
            best_abs = min(best_abs, abs(gfd - grad[k]))
        per_dof.append({
            "dof_local": k, "cell_index": int(case.part[k]),
            "sigma": float(s0[k]),
            "cell_midpoint_m": [float(v) for v in mp[:, k]],
            "cell_volume_m3": float(case.vol[case.part[k]]),
            "adjoint": float(grad[k]),
            "fd_sweep": rows,
            "best_rel_err": float(best), "best_abs_err": float(best_abs),
            "grad_over_grad_max": float(abs(grad[k]) / np.abs(grad).max()),
            "pass_1pct": bool(best < TOL_PER_DOF)})
        print(f"dof {k:6d} adj={grad[k]:+.10e} best_rel={best:.3e} "
              f"best_abs={best_abs:.3e} "
              f"{'PASS' if best < TOL_PER_DOF else 'FAIL'}", flush=True)
    case.forward(s0)

    # --- directional derivative + the Qbar identity ------------------------
    d = rng.standard_normal(case.part.size)
    d /= np.linalg.norm(d)
    gd = float(grad @ d)
    dir_rows, dir_best, qbar_sens = [], np.inf, []
    for eps in EPS_SWEEP:
        hh = eps * float(np.abs(s0).mean())
        fp_, fm_ = case.forward(s0 + hh * d), case.forward(s0 - hh * d)
        fd = (fp_.J - fm_.J) / (2 * hh)
        rel = abs(fd - gd) / abs(fd)
        dir_rows.append({"rel_eps": eps, "h_abs": hh, "fd_central": float(fd),
                         "rel_err": float(rel)})
        dir_best = min(dir_best, rel)
        qbar_sens.append(abs(fp_.qbar - fm_.qbar) / fwd.qbar)
        print(f"directional [{eps:.0e}] fd={fd:+.12e} rel={rel:.3e}", flush=True)
    fwd = case.forward(s0)

    mutations = {}
    for name in ("renorm_frozen", "adjoint_dropped"):
        gm = case.gradient(fwd, mutate=name)
        mutations[name] = {
            "rel_vs_true_at_fd_dofs":
                [float(v) for v in np.abs(gm[dofs] - grad[dofs]) / np.abs(grad[dofs])],
            "directional_rel_err": float(abs(float(gm @ d) - gd) / abs(gd)),
            "rejected_by_1pct_gate":
                bool((np.abs(gm[dofs] - grad[dofs]) / np.abs(grad[dofs])).max()
                     > TOL_PER_DOF)}
        print(f"mutation {name}: {mutations[name]}", flush=True)

    worst = max(r["best_rel_err"] for r in per_dof)
    out = {
        "engine": "dolfinx (P1 CG, conforming tets), complex scalars",
        "scalar_path": ec.SCALAR_PATH,
        "adjoint_tooling": "hand-assembled (dolfinx-adjoint/pyadjoint has no "
                           "release wired to dolfinx 0.11; adjoint_core.py)",
        "adjoint_method": "A^H lambda = rho with A complex-symmetric, so the "
                          "FORWARD LU factorization is reused via "
                          "A x = conj(rho), lambda = conj(x); BC rows folded "
                          "into the residual so db/ds needs no separate term",
        "design_variable": "sigma as a DG0 Function on the part cells "
                           "(one dof per tetrahedron)",
        "n_design_dofs": int(case.part.size),
        "sigma_baseline": "smooth grading sigma0*(1 + 0.4 sin(pi x/R) cos(pi y/R)"
                          " + 0.2 sin(2 pi z/L)); adjoint_core.smooth_sigma",
        "objective": "J = int_part (Q_rf - mean_part Q_rf)^2 dV, Q_rf per "
                     "eqs_common.qrf_dg0 INCLUDING the fixed-power renorm",
        "linear_solver": "PETSc LU (preonly) + 2 iterative-refinement sweeps",
        "seed": SEED, "eps_sweep": list(EPS_SWEEP),
        "mesh": {"matched_to": "n64 (Task 2 coarse)",
                 "lc_part_m": info.lc_part, "lc_bed_m": info.lc_bed,
                 "n_nodes_in_part": info.n_nodes_in_part,
                 "target_nodes_in_part": target,
                 "n_dofs_total": info.n_nodes_total,
                 "n_cells_total": info.n_cells_total,
                 "match_history": hist},
        "consistency_vs_eqs_common": {
            "what": "the same mesh/materials solved through eqs_common.solve_eqs"
                    " + qrf_dg0; forward and adjoint must share ONE operator",
            "max_dV_over_860": dv, "max_dQ_over_Qmax": dq,
            "scale_rel_diff": float(scale_ec / fwd.scale - 1.0),
            "ok": bool(dv < 1e-10 and dq < 1e-10)},
        "forward": {"J": fwd.J, "scale": fwd.scale, "p_now_w": fwd.p_now,
                    "p_target_w": case.p_target, "qbar": fwd.qbar,
                    "power_density_w_per_m3": ec.POWER_DENSITY_W_PER_M3,
                    "qbar_over_power_density_minus_1":
                        float(fwd.qbar / ec.POWER_DENSITY_W_PER_M3 - 1.0),
                    "qraw_min_in_part": fwd.qraw_min_in_part,
                    "clip_active": bool(fwd.qraw_min_in_part <= 0.0),
                    "max_qbar_fd_drift_rel": float(max(qbar_sens))},
        "timing": {"wall_mesh_s": t_mesh, "wall_forward_s": t_fwd,
                   "wall_gradient_s": t_grad,
                   "gradient_over_forward": float(t_grad / t_fwd),
                   "n_forward_solves_for_fd":
                       2 * (N_DOFS * len(EPS_SWEEP) + len(EPS_SWEEP)),
                   "peak_rss_gb": fu.peak_rss_gb()},
        "per_dof": per_dof,
        "directional": {"what": "grad . d for one seeded unit random direction "
                                "over ALL part dofs; no small-|g| noise floor, "
                                "and the nonlocal renorm term dominates it",
                        "adjoint": gd, "fd_sweep": dir_rows,
                        "best_rel_err": float(dir_best),
                        "pass": bool(dir_best < TOL_DIR)},
        "mutations": mutations,
        "gate": {"criterion": "adjoint vs central FD on 5 seeded sigma dofs, "
                              "< 1 % relative each (plan Task 5)",
                 "n_dofs": N_DOFS, "tol_rel": TOL_PER_DOF,
                 "worst_best_rel_err": float(worst),
                 "all_dofs_pass": bool(all(r["pass_1pct"] for r in per_dof)),
                 "directional_best_rel_err": float(dir_best),
                 "mutants_rejected":
                     bool(all(m["rejected_by_1pct_gate"] for m in mutations.values())),
                 "fd_noise_floor_note":
                     "the per-dof FD error bottoms at a CONSTANT ABSOLUTE value "
                     "(~1e-2 in dJ/dsigma) independent of |dJ/dsigma|, i.e. a "
                     "noise floor from the LU backward error, not a gradient "
                     "error; the directional check is floor-free",
                 "gate_ok": bool(all(r["pass_1pct"] for r in per_dof)
                                 and dir_best < TOL_DIR
                                 and all(m["rejected_by_1pct_gate"]
                                         for m in mutations.values())
                                 and dv < 1e-10 and dq < 1e-10)}}
    res["task5"] = out
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(out["gate"], indent=1))
    return 0 if out["gate"]["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
