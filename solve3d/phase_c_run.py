"""Phase C drivers: the composed-chain gradient gate (Task 2) and the solves.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_c_run --chain-gate

Every case parameter is READ from solve3d/results/phase_c_preregistration.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import (adjoint, design_chain as dc, forward as fwd, gate_fd,
                     gates, objective as obj)

RESULTS = Path(__file__).resolve().parent / "results"


def prereg() -> dict:
    return json.loads((RESULTS / "phase_c_preregistration.json").read_text())


# --------------------------------------------------------------------------- #
def design_point_raw(chain: dc.DesignChain, seed: int = 5) -> np.ndarray:
    """A smooth, non-degenerate raw design strictly inside the box.

    Strictly interior so no BOX clip enters the chain: the frozen
    parameterization is built so the only kinks are the physical ones (the melt
    clips and the objective hinges), and the gate design point keeps that true.
    """
    c = chain.centroids
    v = (0.70
         + 0.15 * np.sin(np.pi * c[:, 0] / 0.010) * np.cos(np.pi * c[:, 1] / 0.010)
         + 0.05 * np.sin(np.pi * c[:, 2] / 0.030))
    return np.clip(v, 0.05, 0.95)


def run_chain_gate(tc: adjoint.TransientCase, chain: dc.DesignChain,
                   betas=(0.0, 1.0), seed: int = 7,
                   objective_name: str = "asymmetric") -> dict:
    """Task 2's gate: the COMPOSED dJ/d(raw design) through filter (+ projection)
    and the full coupled forward, re-gated with the Phase B protocol BEFORE any
    solve runs."""
    tc.set_objective(objective_name)
    v0 = design_point_raw(chain)
    out = {"what": "Phase C layer C1: composed dJ/dv through the design chain "
                   "and the full Phase A/B coupled forward",
           "objective": objective_name,
           "kernel": chain.kernel_report(),
           "design_point": {"min": float(v0.min()), "max": float(v0.max()),
                            "strictly_interior": bool(v0.min() > 0 and v0.max() < 1)},
           "gates": {}, "transposes": {}}
    rng = np.random.default_rng(4)
    for beta in betas:
        b = float(beta)
        tr_res = gate_fd.transpose_residual(
            lambda d, _b=b: chain.design_jvp(v0, d, beta=_b),
            lambda g, _b=b: chain.design_vjp(v0, g, beta=_b),
            rng.standard_normal(chain.n_design), rng.standard_normal(chain.n_design))
        out["transposes"][f"beta_{b:g}"] = tr_res

        def J_raw(v, _b=b):
            return tc.J_of_design(chain.design_to_map(v, _b))

        t0 = time.perf_counter()
        s0 = chain.design_to_map(v0, b)
        tr = tc.forward(tc.design_to_sigma(s0))
        t_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        g_s, info = tc.gradient_design(s0, tr=tr)
        g_v = chain.design_vjp(v0, g_s, beta=b)
        t_grad = time.perf_counter() - t0
        g = gate_fd.run_probes(J_raw, v0, g_v, np.ones(g_v.shape, bool), seed=seed,
                               x_scale_direction=float(np.mean(np.abs(v0))))
        g["cost"] = {"wall_forward_s": t_fwd, "wall_gradient_s": t_grad,
                     "forward_equivalents": t_grad / t_fwd}
        g["J"] = float(J_raw(v0))
        out["gates"][f"beta_{b:g}"] = g
    gates.write_json("phase_c_chain_gate.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-gate", action="store_true")
    args = ap.parse_args()
    if args.chain_gate:
        from solve3d import transient_gate as TG
        tc = TG.build_case()
        ch = dc.DesignChain.build(tc)
        d = run_chain_gate(tc, ch)
        print(json.dumps({k: {p: v["probes"][p]["best_rel_err"]
                              for p in v["probes"]}
                          for k, v in d["gates"].items()}, indent=1))
        print(json.dumps({k: v["rel_err"] for k, v in d["transposes"].items()},
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =========================================================================== #
# Task 3: the solve-scale case, arm scoring, and the arms themselves
# =========================================================================== #
SOLVE_MESH = {"target_nodes_in_part": 23040, "lc0": 0.0009375,
              "label": "phase_a_coarse"}
SCORE_MESH = {"target_nodes_in_part": 77952, "lc0": 0.000625,
              "label": "phase_a_mid"}
MAX_TIME_S = 500.0


def build_solve_case(mesh: dict = SOLVE_MESH) -> adjoint.TransientCase:
    """The Phase C case: Phase A anchor circle, coupling OFF, envelope read."""
    p = fwd.ForwardParams()                       # anchor params, coupling off
    return adjoint.TransientCase.build(
        "circle", int(mesh["target_nodes_in_part"]), float(mesh["lc0"]),
        p, max_time_s=MAX_TIME_S, sample_dt_s=50.0)


def transfer_inversion_map(tc: adjoint.TransientCase) -> dict:
    """Trilinear-then-clip transfer of the regenerated heatr3d voxel map onto
    the FEM part cells -- the 3-D analogue of the 2-D production transfer
    (robust.resample_map, bilinear then clip).

    Reports the transfer fidelity so the loss is visible; the pre-registered
    DROP CONDITION is a >2 % move in total in-part dopant."""
    from scipy.interpolate import RegularGridInterpolator
    import dolfinx
    z = np.load(RESULTS / "phase_c_inversion_map.npz")
    sat, part = np.asarray(z["sat"], float), np.asarray(z["part"], bool)
    interp = RegularGridInterpolator(
        (np.asarray(z["x"]), np.asarray(z["y"]), np.asarray(z["zc"])), sat,
        method="linear", bounds_error=False, fill_value=None)
    mp = np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim, np.arange(tc.ncells, dtype=np.int32)))
    s = np.clip(interp(mp[tc.eqs.part]), 0.0, 1.0)
    vox_dv = float(z["h"]) ** 3
    dop_vox = float(sat[part].sum() * vox_dv)
    dop_fem = float(np.sum(s * tc.eqs.vol[tc.eqs.part]))
    rel = abs(dop_fem - dop_vox) / dop_vox
    return {"map": s,
            "mean_sat_voxel": float(sat[part].mean()),
            "mean_sat_fem": float(np.average(s, weights=tc.eqs.vol[tc.eqs.part])),
            "total_dopant_voxel_m3": dop_vox, "total_dopant_fem_m3": dop_fem,
            "total_dopant_rel_move": rel,
            "drop_condition_rel": 0.02, "dropped": bool(rel > 0.02),
            "method": "trilinear_then_clip"}


def score_arm(tc: adjoint.TransientCase, s_map: np.ndarray, name: str,
              extra: dict | None = None) -> dict:
    """Score ONE delivered saturation map: both objective weightings, the
    shape metrics on the shared read grid, and sigma_T as a diagnostic."""
    from solve3d import shape_metrics as sm
    t0 = time.perf_counter()
    tr = tc.forward(tc.design_to_sigma(s_map))
    wall = time.perf_counter() - t0
    tc.set_objective("symmetric")
    Js = tc.J_trajectory(tr)
    k = int(np.argmin(Js))
    tc.set_objective("asymmetric")
    Ja = tc.J_trajectory(tr)
    ka = int(np.argmin(Ja))
    T_read = tc.state_at(tr, ka)                 # PRIMARY read = asymmetric argmin
    phi = fwd.phase_fraction(T_read, tc.p)[0]
    vol, chi = tc.vol_nodal, tc.m_nodal
    split = obj.split_asymmetric(phi, chi, vol)
    split3 = obj.split_asymmetric(phi, chi, vol, w_ratio=obj.W_SENSITIVITY)

    # shape metrics on the shared Phase A read grid
    pts, shp, h = gates.eval_grid_points()
    W = fwd.functionspace(tc.msh, ("Lagrange", 1))
    Tf = fwd.fem.Function(W)
    Tf.x.array[:] = T_read.astype(fwd.dolfinx.default_scalar_type)
    Te, missed = fwd.eval_at(Tf, tc.msh, pts)
    Te = Te.reshape(shp)
    part2d = gates.nominal_part_mask("circle")
    per = []
    for i in range(Te.shape[0]):
        p2 = gates.phase_fraction_phi(Te[i])
        per.append({"iou_none": 0.0,
                    "in08": sm.in_part_melt_fraction(p2 >= 0.8, part2d),
                    "in09": sm.in_part_melt_fraction(p2 >= 0.9, part2d),
                    "bed08": sm.out_of_part_fraction(p2 >= 0.8, part2d),
                    "bed09": sm.out_of_part_fraction(p2 >= 0.9, part2d)})
    agg = {k2: float(np.mean([r[k2] for r in per])) for k2 in per[0]}
    part_w = vol * chi
    rec = {
        "arm": name, "wall_forward_s": wall,
        "n_steps": tr.n_steps, "n_eqs_solves": len(tr.events),
        "J_symmetric": float(Js[k]), "argmin_symmetric": k,
        "at_horizon_symmetric": bool(k >= tr.n_steps),
        "J_asymmetric": float(Ja[ka]), "argmin_asymmetric": ka,
        "t_stop_s": float(ka * tc.p.dt_s),
        "at_horizon_asymmetric": bool(ka >= tr.n_steps),
        "J_asym_w3_at_same_read": split3["J_asym"],
        "J_out_of_bounds": split["J_out_of_bounds"],
        "J_in_bounds_deficit": split["J_in_bounds_deficit"],
        "out_of_part_melt_fraction_of_part": split["out_of_bounds_melt_fraction_of_part"],
        "in_bounds_below_floor_fraction": split["in_bounds_below_floor_fraction"],
        "part_mean_phi": float(np.dot(phi, part_w) / part_w.sum()),
        "sigma_T_diagnostic_c": float(np.sqrt(
            np.dot((T_read - np.dot(T_read, part_w) / part_w.sum()) ** 2, part_w)
            / part_w.sum())),
        "in_part_melt_frac_phi08": agg["in08"], "in_part_melt_frac_phi09": agg["in09"],
        "bed_melt_frac_phi08": agg["bed08"], "bed_melt_frac_phi09": agg["bed09"],
        "eval_missed": int(missed),
        "map_stats": {"mean": float(np.average(s_map, weights=tc.eqs.vol[tc.eqs.part])),
                      "min": float(s_map.min()), "max": float(s_map.max())},
    }
    if extra:
        rec.update(extra)
    return rec


def run_baselines() -> dict:
    tc = build_solve_case()
    npart = tc.eqs.part.size
    out = {"what": "Phase C Task 3 baselines on the solve mesh",
           "mesh": {**SOLVE_MESH, "n_cells": int(tc.ncells),
                    "n_design_cells": int(npart)},
           "case": {"max_time_s": MAX_TIME_S, "dt_s": tc.p.dt_s,
                    "coupling": "off"},
           "arms": {}}
    out["arms"]["uniform_baseline"] = score_arm(tc, np.ones(npart), "uniform_baseline")
    inv = transfer_inversion_map(tc)
    smap = inv.pop("map")
    if inv["dropped"]:
        out["arms"]["inversion_map"] = {"arm": "inversion_map", "status": "DROPPED",
                                        "reason": "transfer moved total in-part "
                                                  "dopant beyond the pre-registered "
                                                  "2 % drop condition",
                                        "transfer": inv}
    else:
        out["arms"]["inversion_map"] = score_arm(tc, smap, "inversion_map",
                                                 extra={"transfer": inv})
    gates.write_json("phase_c_baselines.json", out)
    return out


# =========================================================================== #
# Task 3: the solve arms
# =========================================================================== #
CHECKPOINT_INTERVAL = 25


class _BudgetExhausted(Exception):
    pass


def run_solve_arm(name: str, objective_name: str, beta: float,
                  w_ratio: float | None = None,
                  budget_evals: int | None = None,
                  tc: adjoint.TransientCase | None = None,
                  chain: dc.DesignChain | None = None) -> dict:
    """One solve arm: L-BFGS-B on the Phase B gradient through the design chain.

    Frozen conventions (FROZEN_CONVENTIONS_2D section 6): L-BFGS-B with
    jac=True, box [0,1], ftol/gtol 1e-16, budget enforced by raising from the
    objective, single full-depth COLD START from uniform saturation 1.0, and
    each arm keeps its own BEST iterate so a bad line-search trial is never
    carried forward. MMA was not implemented in 2-D either; L-BFGS-B is the
    named substitution.
    """
    from scipy.optimize import minimize
    pr = prereg()
    budget = int(budget_evals or pr["budget"]["gradient_evaluations_per_arm"])
    tc = tc or build_solve_case()
    chain = chain or dc.DesignChain.build(tc)
    tc.set_objective(objective_name, w_ratio)
    n = chain.n_design
    v0 = np.ones(n)                      # cold start: uniform saturation 1.0
    state = {"n": 0, "best_J": np.inf, "best_v": v0.copy(), "hist": [],
             "t0": time.perf_counter()}

    def fg(v):
        if state["n"] >= budget:
            raise _BudgetExhausted()
        s = chain.design_to_map(v, beta)
        tr = tc.forward(tc.design_to_sigma(s))
        Jt = tc.J_trajectory(tr)
        k = int(np.argmin(Jt))
        J = float(Jt[k])
        g_s, _info = tc.gradient_design(s, tr=tr, read_step=k,
                                        checkpoint_interval=CHECKPOINT_INTERVAL)
        g_v = chain.design_vjp(v, g_s, beta=beta)
        state["n"] += 1
        if J < state["best_J"]:
            state["best_J"], state["best_v"] = J, np.asarray(v, float).copy()
        state["hist"].append({
            "eval": state["n"], "J": J, "argmin_step": k,
            "t_stop_s": float(k * tc.p.dt_s),
            "at_horizon": bool(k >= tr.n_steps),
            "grad_norm": float(np.linalg.norm(g_v)),
            "map_mean": float(np.average(s, weights=tc.eqs.vol[tc.eqs.part])),
            "wall_s": time.perf_counter() - state["t0"]})
        print(f"  [{name}] eval {state['n']}/{budget} J={J:.6e} "
              f"t_stop={k * tc.p.dt_s:.1f}s |g|={np.linalg.norm(g_v):.3e}",
              flush=True)
        return J, g_v

    status = "budget_exhausted"
    try:
        res = minimize(fg, v0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * n,
                       options={"ftol": 1e-16, "gtol": 1e-16, "maxiter": 10000})
        status = f"converged:{res.message}"
    except _BudgetExhausted:
        pass
    v_best = state["best_v"]
    s_best = chain.design_to_map(v_best, beta)
    rec = score_arm(tc, s_best, name, extra={
        "status": status, "objective_optimized": objective_name,
        "w_ratio": w_ratio, "beta": beta,
        "budget_gradient_evaluations": budget,
        "gradient_evaluations_used": state["n"],
        "forward_equivalents_spent":
            state["n"] * float(prereg()["budget"]["measured_at_solve_scale"]
                               ["per_gradient_eval_forward_equivalents"]),
        "J_first_eval": state["hist"][0]["J"] if state["hist"] else None,
        "J_best_during_solve": state["best_J"],
        "wall_total_s": time.perf_counter() - state["t0"],
        "trajectory": state["hist"],
        "start": "single full-depth cold start, uniform saturation 1.0",
    })
    np.savez_compressed(RESULTS / f"phase_c_map_{name}.npz",
                        v_raw=v_best, s_map=s_best,
                        centroids=chain.centroids, volumes=chain.volumes)
    p = RESULTS / "phase_c_solves.json"
    doc = json.loads(p.read_text()) if p.exists() else {
        "what": "Phase C Task 3 solve arms", "arms": {}}
    doc["arms"][name] = rec
    gates.write_json(p.name, doc)
    return rec
