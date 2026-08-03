"""Phase E opener campaign driver.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.run --pilot pyramid
    ... --shape pyramid

Everything is READ from solve3d/phase_e/results/phase_e_preregistration.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import (adjoint, design_chain as dc, forward as fwd, gate_fd,
                     gates, objective as obj)
from solve3d.phase_e import geometry as geo

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHECKPOINT_INTERVAL = 25
LC_PART_M = 0.9375e-3            # matches the Phase A coarse element size
LC_FINE_M = 0.625e-3             # the hold-out mesh, the Phase A mid size
MAX_TIME_S = 650.0   # pilot: envelope argmin at t=418 s, comfortably interior


def prereg() -> dict:
    return json.loads((RESULTS / "phase_e_preregistration.json").read_text())


def build_case(shape: str, lc_part: float = LC_PART_M,
               max_time_s: float = MAX_TIME_S,
               p: fwd.ForwardParams | None = None) -> adjoint.TransientCase:
    """A Phase B TransientCase on a Phase E conforming mesh.

    Constructed directly rather than through TransientCase.build, because that
    helper only knows the Phase A anchor primitives. Nothing in adjoint.py is
    modified."""
    p = p or fwd.ForwardParams()
    msh, info = geo.build_mesh(shape, lc_part=lc_part)
    mats = fwd.build_materials(msh, geo.in_part_predicate(shape), p)
    eqs = adjoint.SteadyEqs(msh, mats, p)
    return adjoint.TransientCase(msh, mats, p, eqs, info, 50.0, max_time_s)


# --------------------------------------------------------------------------- #
# The heuristic arm: the closed-form law evaluated at FEM centroids
# --------------------------------------------------------------------------- #
def heuristic_map(tc: adjoint.TransientCase, shape: str,
                  strong: bool = True) -> dict:
    """demo_pyramid_fgm's grading law (commit 6c2aab9), evaluated ANALYTICALLY.

    The law is a function of depth-to-surface and normalized height, so it can
    be evaluated exactly at the FEM centroids -- which avoids the voxel-to-mesh
    resampling loss that dropped Phase C's inversion arm. Depth is computed
    from the analytic solid rather than from a distance transform of a raster,
    for the same reason."""
    import dolfinx
    mp = np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]
    x, y, z = mp[:, 0], mp[:, 1], mp[:, 2]
    if shape == "pyramid":
        b2, h = geo.PYR_B_M / 2.0, geo.PYR_H_M
        z_base, z_apex = -h / 2.0, +h / 2.0
        s_of_z = b2 * (z_apex - z) / h
        # distance to the four slanted faces and to the base, analytically
        slope = b2 / h                       # half-width lost per unit height
        denom = float(np.sqrt(1.0 + slope ** 2))
        d_side = (s_of_z - np.maximum(np.abs(x), np.abs(y))) / denom
        d_base = z - z_base
        depth = np.minimum(d_side, d_base)
        z_hat = (z - z_base) / (z_apex - z_base)
    else:                                     # cube
        a2 = geo.CUBE_A_M / 2.0
        depth = a2 - np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
        z_hat = (z + a2) / (2 * a2)
    d_hat = depth / float(depth.max())
    if strong:
        peak, d_floor, d_gain, z_taper, s_min = 0.95, 0.15, 0.85, 0.75, 0.10
    else:
        peak, d_floor, d_gain, z_taper, s_min = 0.95, 0.35, 0.65, 0.45, 0.20
    sat = np.clip(peak * (d_floor + d_gain * d_hat) * (1.0 - z_taper * z_hat),
                  s_min, 1.0)
    # transfer fidelity: the same law on a fine reference grid
    n = 200
    L = 0.060
    c = (np.arange(n) + 0.5) * (L / n) - L / 2
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    inside = geo.in_part_predicate(shape)(
        np.vstack([X.ravel(), Y.ravel(), Z.ravel()]))
    ref = _law_on_points(shape, np.vstack([X.ravel(), Y.ravel(), Z.ravel()])[:, inside],
                         strong)
    dv = (L / n) ** 3
    dop_ref = float(ref.sum() * dv)
    dop_fem = float(np.dot(sat, tc.eqs.vol[tc.eqs.part]))
    rel = abs(dop_fem - dop_ref) / dop_ref
    return {"map": sat, "strong": strong,
            "mean_sat_fem": float(np.average(sat, weights=tc.eqs.vol[tc.eqs.part])),
            "total_dopant_fem_m3": dop_fem,
            "total_dopant_reference_m3": dop_ref,
            "total_dopant_rel_move": rel,
            "drop_condition_rel": 0.02, "dropped": bool(rel > 0.02),
            "method": "closed-form law evaluated at FEM centroids"}


def _law_on_points(shape: str, mp: np.ndarray, strong: bool) -> np.ndarray:
    x, y, z = mp[0], mp[1], mp[2]
    if shape == "pyramid":
        b2, h = geo.PYR_B_M / 2.0, geo.PYR_H_M
        z_base, z_apex = -h / 2.0, +h / 2.0
        s_of_z = b2 * (z_apex - z) / h
        slope = b2 / h
        denom = float(np.sqrt(1.0 + slope ** 2))
        depth = np.minimum((s_of_z - np.maximum(np.abs(x), np.abs(y))) / denom,
                           z - z_base)
        z_hat = (z - z_base) / (z_apex - z_base)
    else:
        a2 = geo.CUBE_A_M / 2.0
        depth = a2 - np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
        z_hat = (z + a2) / (2 * a2)
    d_hat = depth / float(depth.max())
    if strong:
        peak, d_floor, d_gain, z_taper, s_min = 0.95, 0.15, 0.85, 0.75, 0.10
    else:
        peak, d_floor, d_gain, z_taper, s_min = 0.95, 0.35, 0.65, 0.45, 0.20
    return np.clip(peak * (d_floor + d_gain * d_hat) * (1.0 - z_taper * z_hat),
                   s_min, 1.0)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_arm(tc: adjoint.TransientCase, s_map: np.ndarray, shape: str,
              name: str, extra: dict | None = None) -> dict:
    from solve3d import shape_metrics as sm
    t0 = time.perf_counter()
    tr = tc.forward(tc.design_to_sigma(s_map))
    wall = time.perf_counter() - t0
    tc.set_objective("symmetric")
    Js = tc.J_trajectory(tr)
    ks = int(np.argmin(Js))
    tc.set_objective("asymmetric")
    Ja = tc.J_trajectory(tr)
    ka = int(np.argmin(Ja))
    T_read = tc.state_at(tr, ka)
    phi = fwd.phase_fraction(T_read, tc.p)[0]
    vol, chi = tc.vol_nodal, tc.m_nodal
    split = obj.split_asymmetric(phi, chi, vol)
    split3 = obj.split_asymmetric(phi, chi, vol, w_ratio=obj.W_SENSITIVITY)

    pts, shp, h = gates.eval_grid_points()
    W = fwd.functionspace(tc.msh, ("Lagrange", 1))
    Tf = fwd.fem.Function(W)
    Tf.x.array[:] = T_read.astype(fwd.dolfinx.default_scalar_type)
    Te, missed = fwd.eval_at(Tf, tc.msh, pts)
    Te = Te.reshape(shp)
    # the pyramid's nominal cross-section varies with z, so each eval plane
    # gets its OWN analytic nominal -- a single mid-plane mask would score the
    # apex against the base
    per = []
    for i, zc in enumerate(gates.EVAL_Z_M):
        nom = geo.nominal_mask_2d(shape, z=zc)
        p2 = gates.phase_fraction_phi(Te[i])
        row = {"nominal_cells": int(nom.sum())}
        if nom.any():
            for t in (0.8, 0.9):
                m = p2 >= t
                k = f"phi{t:g}".replace(".", "p")
                row[f"iou_{k}"] = sm.iou(m, nom)
                row[f"in_part_{k}"] = sm.in_part_melt_fraction(m, nom)
                row[f"out_of_part_{k}"] = sm.out_of_part_fraction(m, nom)
            row["front_ssd_mm"] = sm.symmetric_surface_distance_mm(
                p2 >= 0.9, nom, h)
        per.append(row)
    # derive keys from the first NON-EMPTY plane: the shared grid's fixed
    # z-planes mostly miss these compact shapes, and taking keys from an empty
    # plane silently deletes every verdict-carrying metric (it did, on the
    # first pass). solve3d/phase_e/rescore.py places planes shape-relative.
    live = [r for r in per if len(r) > 1]
    keys = [k for k in (live[0] if live else {}) if k != "nominal_cells"]
    agg = {k: float(np.nanmean([r[k] for r in live if k in r])) for k in keys}
    part_w = vol * chi
    rec = {"arm": name, "shape": shape, "wall_forward_s": wall,
           "n_steps": tr.n_steps, "n_eqs_solves": len(tr.events),
           "J_symmetric": float(Js[ks]), "argmin_symmetric": ks,
           "J_asymmetric": float(Ja[ka]), "argmin_asymmetric": ka,
           "t_stop_s": float(ka * tc.p.dt_s),
           "at_horizon_asymmetric": bool(ka >= tr.n_steps),
           "J_asym_w3_at_same_read": split3["J_asym"],
           "J_out_of_bounds": split["J_out_of_bounds"],
           "J_in_bounds_deficit": split["J_in_bounds_deficit"],
           "out_of_part_melt_fraction_of_part":
               split["out_of_bounds_melt_fraction_of_part"],
           "in_bounds_below_floor_fraction": split["in_bounds_below_floor_fraction"],
           "part_mean_phi": float(np.dot(phi, part_w) / part_w.sum()),
           "sigma_T_diagnostic_c": float(np.sqrt(
               np.dot((T_read - np.dot(T_read, part_w) / part_w.sum()) ** 2, part_w)
               / part_w.sum())),
           "eval_missed": int(missed),
           "map_stats": {"mean": float(np.average(s_map, weights=tc.eqs.vol[tc.eqs.part])),
                         "min": float(s_map.min()), "max": float(s_map.max())},
           "gates": {"energy_residual_frac": float(tr.out["energy_residual_frac"]),
                     "clamp_bound": bool(tr.out["clamp_bound"]),
                     "cfl_violated": bool(tr.out["cfl_violated"])}}
    rec.update(agg)
    if extra:
        rec.update(extra)
    np.savez_compressed(RESULTS / f"field_{shape}_{name}.npz",
                        T_eval=Te, s_map=s_map, T_read=T_read)
    return rec


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
class _BudgetExhausted(Exception):
    pass


def run_solve_arm(tc, chain, shape: str, name: str, objective_name: str,
                  beta: float = 0.0, w_ratio: float | None = None,
                  budget_evals: int = 12) -> dict:
    """L-BFGS-B on the Phase B gradient. The 1/|g0| rescale is the STANDING
    CONVENTION (registration conventions.scale_first_step), not a deviation."""
    from scipy.optimize import minimize
    tc.set_objective(objective_name, w_ratio)
    n = chain.n_design
    v0 = np.ones(n)
    st = {"n": 0, "best_J": np.inf, "best_v": v0.copy(), "hist": [],
          "t0": time.perf_counter(), "scale": 1.0}

    def fg(v):
        if st["n"] >= budget_evals:
            raise _BudgetExhausted()
        s = chain.design_to_map(v, beta)
        tr = tc.forward(tc.design_to_sigma(s))
        Jt = tc.J_trajectory(tr)
        k = int(np.argmin(Jt))
        J = float(Jt[k])
        g_s, _ = tc.gradient_design(s, tr=tr, read_step=k,
                                    checkpoint_interval=CHECKPOINT_INTERVAL)
        g_v = chain.design_vjp(v, g_s, beta=beta)
        st["n"] += 1
        if st["n"] == 1:
            gn = float(np.linalg.norm(g_v))
            st["scale"] = (1.0 / gn) if gn > 0 else 1.0
        if J < st["best_J"]:
            st["best_J"], st["best_v"] = J, np.asarray(v, float).copy()
        st["hist"].append({"eval": st["n"], "J": J, "argmin_step": k,
                           "t_stop_s": float(k * tc.p.dt_s),
                           "at_horizon": bool(k >= tr.n_steps),
                           "grad_norm": float(np.linalg.norm(g_v)),
                           "wall_s": time.perf_counter() - st["t0"]})
        print(f"  [{name}] eval {st['n']}/{budget_evals} J={J:.6e} "
              f"t_stop={k * tc.p.dt_s:.1f}s |g|={np.linalg.norm(g_v):.3e}",
              flush=True)
        return J * st["scale"], g_v * st["scale"]

    status = "budget_exhausted"
    try:
        r = minimize(fg, v0, jac=True, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * n,
                     options={"ftol": 1e-16, "gtol": 1e-16, "maxiter": 10000})
        status = f"converged:{r.message}"
    except _BudgetExhausted:
        pass
    v_best = st["best_v"]
    s_best = chain.design_to_map(v_best, beta)
    rec = score_arm(tc, s_best, shape, name, extra={
        "status": status, "objective_optimized": objective_name,
        "beta": beta, "w_ratio": w_ratio,
        "budget_gradient_evaluations": budget_evals,
        "gradient_evaluations_used": st["n"],
        "J_first_eval": st["hist"][0]["J"] if st["hist"] else None,
        "trajectory": st["hist"], "scale_first_step": True,
        "objective_scale_applied": st["scale"],
        "wall_total_s": time.perf_counter() - st["t0"]})
    np.savez_compressed(RESULTS / f"map_{shape}_{name}.npz", v_raw=v_best,
                        s_map=s_best, centroids=chain.centroids,
                        volumes=chain.volumes)
    return rec


def _merge(shape: str, key: str, rec) -> None:
    p = RESULTS / f"phase_e_{shape}.json"
    doc = json.loads(p.read_text()) if p.exists() else {"shape": shape, "arms": {}}
    doc.setdefault("arms", {})[key] = rec
    gates.write_json(p.name.replace("phase_e_", "phase_e_"), doc) if False else \
        p.write_text(json.dumps(doc, indent=1, default=float))


def run_shape(shape: str, budget_evals: int = 12) -> dict:
    print(f"[phase-e] {shape}: building case", flush=True)
    tc = build_case(shape)
    chain = dc.DesignChain(
        np.asarray(__import__("dolfinx").mesh.compute_midpoints(
            tc.msh, tc.msh.topology.dim,
            np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part],
        tc.eqs.vol[tc.eqs.part], 1.0e-3, [0.0])
    _merge(shape, "_mesh", {"n_cells": int(tc.ncells),
                            "n_nodes": int(tc.vol_nodal.size),
                            "n_design_cells": int(chain.n_design),
                            "lc_part_m": LC_PART_M,
                            "part_volume_rel_err_vs_library":
                                float(tc.info.part_volume_rel_err_vs_library),
                            "kernel": chain.kernel_report()})
    n = chain.n_design
    print(f"[phase-e] {shape}: uniform baseline", flush=True)
    _merge(shape, "uniform_baseline", score_arm(tc, np.ones(n), shape,
                                                "uniform_baseline"))
    print(f"[phase-e] {shape}: heuristic arm", flush=True)
    hm = heuristic_map(tc, shape)
    smap = hm.pop("map")
    if hm["dropped"]:
        _merge(shape, "heuristic_grading_law",
               {"arm": "heuristic_grading_law", "status": "DROPPED",
                "reason": "transfer moved total in-part dopant beyond the "
                          "pre-registered 2 % condition", "transfer": hm})
    else:
        _merge(shape, "heuristic_grading_law",
               score_arm(tc, smap, shape, "heuristic_grading_law",
                         extra={"transfer": hm}))
    print(f"[phase-e] {shape}: filter-only solve", flush=True)
    _merge(shape, "solve_filter_only",
           run_solve_arm(tc, chain, shape, "solve_filter_only", "asymmetric",
                         budget_evals=budget_evals))
    return json.loads((RESULTS / f"phase_e_{shape}.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=None)
    ap.add_argument("--budget", type=int, default=12)
    a = ap.parse_args()
    if a.shape:
        run_shape(a.shape, budget_evals=a.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
