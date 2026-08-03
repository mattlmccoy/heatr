"""Finite-difference gate for the COMPOSED topology-optimization gradient.

The solve optimizes dJ/dv through the full chain

    v  --F-->  v_f  --P_beta-->  s  --forward-->  T  --J-->  scalar

against the GRID-INDEPENDENT area-fill target chi. None of that composition has
been gated before: `gate_shape` gated dJ/ds against the raster chi, `gate_ms`
added the filter, and the projection and the new target are both new. This runs
BEFORE any optimization.

Layers, each one adding exactly one thing so a failure can be bisected:

  P1  dJ/ds, no filter, no projection, area-fill chi. Isolates the new TARGET.
  P2  dJ/dv, filter only (beta = 0). Reproduces the previously gated
      `gate_ms` layer S2 except for the target, so P2 against P1 isolates the
      FILTER.
  P3  dJ/dv, filter and projection at beta = 1, the first continuation stage.
  P4  dJ/dv, filter and projection at beta = 16, the last continuation stage
      and the sharpest projection the solve ever uses. This is the worst case:
      the projection derivative spans six orders of magnitude across the part,
      so it is where a chain-rule error would show first.

Probes per layer: the maximum-sensitivity cell, a fixed pseudo-random in-part
cell, a random unit direction, a filter-smooth random direction, and the
gradient direction. Epsilon is swept over eight values from 1e-3 to 1e-8.

THE DESIGN POINT. `gate_rho.default_v` puts the design around 0.80, which at
beta = 16 sits on the upper rail of the projection where dP/du is 2e-06 and the
gradient is numerically dead. Gating there would be gating nothing. The gate
therefore uses `gate_v0`, a design point centred on the projection THRESHOLD
eta = 0.5, which is where the projection is most active and where the chain
rule has the most to get wrong.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_topopt <shape> <out.json>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, chi_area, gradops
from . import forward as fwd
from . import library_solve as lib
from . import topopt
from . import topopt_objective as tobj
from .gate_rho import EPSILONS, PASS_REL_ERR, SUBGRADIENT_PASS_REL_ERR, _probe_dirs
from .pins import build_case, load_cfg

LAYERS = (("P1_target_only_dJds", None),
          ("P2_filter_only_dJdv", 0.0),
          ("P3_projection_beta1_dJdv", 1.0),
          ("P4_projection_beta16_dJdv", 16.0))


def gate_v0(case, seed: int = 4242) -> np.ndarray:
    """A design point centred on the projection threshold, with noise."""
    rng = np.random.default_rng(seed)
    ny, nx = case.part_mask.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    smooth = 0.50 + 0.18 * np.cos(1.7 * xx + 0.4) * np.sin(2.1 * yy + 0.9)
    v = np.ones((ny, nx))
    v[case.part_mask] = (smooth[case.part_mask]
                         + 0.02 * rng.standard_normal(int(case.part_mask.sum())))
    return np.clip(v, 0.0, 1.0)


def run_forward(case, s):
    """Exactly the forward the topology-optimization solve runs."""
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=lib.PATIENCE)


def _to_map(v, case, beta, dx):
    pm = case.part_mask
    if beta is None:
        return np.where(pm, v, 1.0)
    return topopt.design_to_map(v, pm, dx=dx, radius_m=topopt.FILTER_RADIUS_M, beta=beta)


def _to_vjp(g_s, v, case, beta, dx):
    pm = case.part_mask
    if beta is None:
        return np.where(pm, g_s, 0.0)
    return topopt.design_vjp(g_s, v, pm, dx=dx, radius_m=topopt.FILTER_RADIUS_M, beta=beta)


def gradient_direction(pm, g) -> np.ndarray:
    d = np.zeros(pm.shape)
    n = float(np.linalg.norm(g[pm]))
    if n > 0.0:
        d[pm] = g[pm] / n
    return d


def projection_activity(case, v0, beta, dx) -> dict:
    """How active the projection is at the gate point, per part cell."""
    if beta is None or float(beta) <= 0.0:
        return {"active": False}
    pm = case.part_mask
    vf = topopt.filtered(v0, pm, dx, topopt.FILTER_RADIUS_M)
    dp = topopt.project_deriv(vf, beta)[pm]
    return {"active": True, "dPdu_min": float(dp.min()), "dPdu_max": float(dp.max()),
            "dPdu_median": float(np.median(dp)),
            "frac_cells_dPdu_below_1e-3_of_peak":
                float(np.mean(dp < 1e-3 * dp.max()))}


def gate_layer(case, name, ops, v0, chi, read_index, beta, sigma_probe) -> dict:
    pm = case.part_mask
    dx = case.dx

    def J_of(v):
        tr = run_forward(case, _to_map(v, case, beta, dx))
        i = min(int(read_index), tr.n_outer - 1)
        return tobj.J_and_seed(tr.T_at_end(i), case, chi)[0]

    s0 = _to_map(v0, case, beta, dx)
    tr0 = run_forward(case, s0)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = tobj.J_and_seed(tr0.T_at_end(i0), case, chi)
    g_s = adjoint.gradient(case, s0, tr0, {i0: seed}, grad_ops=ops)
    g = _to_vjp(g_s, v0, case, beta, dx)

    _phi, inside = tobj.phi_field(tr0.T_at_end(i0), case)
    out = {"layer": name, "beta": beta, "J0": float(J0), "read_index": int(i0),
           "n_outer": int(tr0.n_outer),
           "grad_norm": float(np.linalg.norm(g[pm])),
           "grad_max_abs": float(np.max(np.abs(g[pm]))),
           "n_in_ramp": int(inside.sum()),
           "n_in_ramp_in_part": int((inside & pm).sum()),
           "projection_activity": projection_activity(case, v0, beta, dx),
           "probes": {}}
    del tr0

    probes = list(_probe_dirs(pm, g, sigma_cells=sigma_probe))
    probes.append(("gradient_direction", gradient_direction(pm, g), None))
    for pname, d, cell in probes:
        ana = float(np.sum(g * d))
        rows = []
        for eps in EPSILONS:
            fd = (J_of(v0 + eps * d) - J_of(v0 - eps * d)) / (2.0 * eps)
            rows.append({"eps": eps, "fd": float(fd),
                         "rel_err": abs(fd - ana) / max(abs(ana), 1e-30),
                         "abs_err": abs(fd - ana)})
        best = min(rows, key=lambda r: r["rel_err"])
        tail = sorted(rows, key=lambda r: r["eps"])[:2]
        floor_est = float(np.median([2.0 * r["eps"] * r["abs_err"] for r in tail]))
        out["probes"][pname] = {
            "cell": None if cell is None else [int(c) for c in cell],
            "analytic": ana, "sweep": rows, "best_rel_err": best["rel_err"],
            "best_abs_err": best["abs_err"], "best_eps": best["eps"],
            "J_eval_floor_estimate": floor_est,
            "PASS": bool(best["rel_err"] < PASS_REL_ERR),
            "PASS_subgradient": bool(best["rel_err"] < SUBGRADIENT_PASS_REL_ERR)}
    out["PASS"] = all(p["PASS"] for p in out["probes"].values())
    out["PASS_subgradient"] = all(p["PASS_subgradient"] for p in out["probes"].values())
    out["n_probes"] = len(out["probes"])
    out["n_probes_pass_1e-6"] = sum(p["PASS"] for p in out["probes"].values())
    out["n_probes_pass_1e-5"] = sum(p["PASS_subgradient"] for p in out["probes"].values())
    return out


def transpose_consistency(case, ops, v0, chi, read_index, beta) -> dict:
    """Is dJ/dv exactly the transpose of the linearized parameterization?

    The bisect that separates a chain-rule wiring error from a property of the
    forward. The composed map is nonlinear in v through the projection, but its
    LINEARIZATION is exact and available (`topopt.design_jvp`), so the identity
    <dJ/dv, d> = <dJ/ds, dS(v)[d]> must hold to machine precision at any v.
    """
    pm = case.part_mask
    dx = case.dx
    s0 = _to_map(v0, case, beta, dx)
    tr = run_forward(case, s0)
    i = min(int(read_index), tr.n_outer - 1)
    _J, seed = tobj.J_and_seed(tr.T_at_end(i), case, chi)
    g_s = adjoint.gradient(case, s0, tr, {i: seed}, grad_ops=ops)
    g_v = _to_vjp(g_s, v0, case, beta, dx)
    del tr
    out = {"beta": beta, "read_index": int(i), "probes": {}}
    sig = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, dx)
    for pname, d, _c in _probe_dirs(pm, g_v, sigma_cells=sig):
        lhs = float(np.sum(g_v * d))
        rhs = float(np.sum(g_s * topopt.design_jvp(
            d, v0, pm, dx=dx, radius_m=topopt.FILTER_RADIUS_M, beta=beta)))
        out["probes"][pname] = {"dJdv_dot_d": lhs, "dJds_dot_dS_d": rhs,
                                "rel_err": abs(lhs - rhs) / max(abs(lhs), 1e-30)}
    out["max_rel_err"] = max(p["rel_err"] for p in out["probes"].values())
    out["PASS_machine_precision"] = bool(out["max_rel_err"] < 1e-10)
    return out


def stop_index_stability(case, v0, chi, beta, eps: float = 1e-3) -> dict:
    pm = case.part_mask
    dx = case.dx
    base = tobj.optimal_stop(run_forward(case, _to_map(v0, case, beta, dx)), case, chi)
    out = {"eps": float(eps), "base_index": int(base.index), "moved": False, "probes": {}}
    for pname, d, _c in _probe_dirs(pm, np.where(pm, 1.0, 0.0)):
        ip = tobj.optimal_stop(run_forward(
            case, _to_map(v0 + eps * d, case, beta, dx)), case, chi).index
        im = tobj.optimal_stop(run_forward(
            case, _to_map(v0 - eps * d, case, beta, dx)), case, chi).index
        moved = bool(ip != base.index or im != base.index)
        out["probes"][pname] = {"plus": int(ip), "minus": int(im), "moved": moved}
        out["moved"] = out["moved"] or moved
    return out


def main(shape: str, out_path: str) -> dict:
    t0 = time.perf_counter()
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    ops = gradops.gradient_matrices(case.x, case.y)
    chi, chi_info = chi_area.chi_from_cfg(cfg, case.x, case.y)
    v0 = gate_v0(case)
    sig = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)

    base = run_forward(case, np.where(case.part_mask, v0, 1.0))
    st = tobj.optimal_stop(base, case, chi)
    st_raster = tobj.optimal_stop(base, case, case.part_mask.astype(float))
    del base

    res = {"shape": shape, "config": str(cfg_path),
           "objective": "J = sum over the WHOLE domain of (phi - chi_area)^2",
           "filter_radius_m": topopt.FILTER_RADIUS_M,
           "sigma_cells_at_this_grid": sig, "dx_m": case.dx,
           "eta": topopt.ETA, "beta_schedule": list(topopt.BETA_SCHEDULE),
           "chi": chi_info,
           "chi_vs_raster": chi_area.raster_vs_area_delta(
               case.part_mask, chi, case.dx, case.dy),
           "base": {"read_index": int(st.index), "read_time_s": st.time_s,
                    "J": st.J, "at_horizon": st.at_horizon,
                    "read_index_under_raster_chi": int(st_raster.index),
                    "read_index_shift_vs_raster": int(st.index - st_raster.index)},
           "layers": []}
    print(f"[{shape}] dx {case.dx*1e3:.4f} mm, filter radius {topopt.FILTER_RADIUS_M*1e3:.2f} "
          f"mm = {sig:.3f} cells; chi area {chi_info['area_m2']*1e6:.3f} mm2 against "
          f"raster {res['chi_vs_raster']['area_raster_m2']*1e6:.3f} mm2 "
          f"({res['chi_vs_raster']['area_rel_delta']*100:+.2f} percent); read index "
          f"{st.index} (raster chi would read {st_raster.index})", flush=True)

    for name, beta in LAYERS:
        r = gate_layer(case, name, ops, v0, chi, st.index, beta, sig)
        res["layers"].append(r)
        print(f"{r['layer']:28s} J0={r['J0']:.6f} |g|={r['grad_norm']:.4e} "
              f"maxcell={r['probes']['max_sensitivity_cell']['best_rel_err']:.3e} "
              f"randcell={r['probes']['random_cell']['best_rel_err']:.3e} "
              f"randdir={r['probes']['random_direction']['best_rel_err']:.3e} "
              f"smoothdir={r['probes']['smooth_random_direction']['best_rel_err']:.3e} "
              f"graddir={r['probes']['gradient_direction']['best_rel_err']:.3e} "
              f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
              f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    res["transpose_consistency"] = {}
    for beta in (0.0, 1.0, 16.0):
        tc = transpose_consistency(case, ops, v0, chi, st.index, beta)
        res["transpose_consistency"][f"beta_{beta:g}"] = tc
        print(f"transpose consistency beta={beta:g}: max relative error "
              f"{tc['max_rel_err']:.3e}, machine-precision PASS = "
              f"{tc['PASS_machine_precision']}", flush=True)

    res["stop_index_stability"] = stop_index_stability(case, v0, chi, 16.0)
    print(f"stop index stability at beta 16: base "
          f"{res['stop_index_stability']['base_index']}, moved = "
          f"{res['stop_index_stability']['moved']}", flush=True)

    res["ALL_GATES_PASS"] = all(l["PASS"] for l in res["layers"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = all(l["PASS_subgradient"] for l in res["layers"])
    res["ALL_TRANSPOSES_EXACT"] = all(
        t["PASS_machine_precision"] for t in res["transpose_consistency"].values())
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"ALL_GATES_PASS at 1e-6 = {res['ALL_GATES_PASS']}, at the 1e-5 subgradient "
          f"standard = {res['ALL_GATES_PASS_SUBGRADIENT']}, transposes exact = "
          f"{res['ALL_TRANSPOSES_EXACT']}, wall {res['wall_s']:.1f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
