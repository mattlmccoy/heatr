"""Finite-difference gate for the MELT-region gradient with the design filter.

The multi-start solve optimizes dJ_phi/dv with the physical-length design
filter in the chain, s = F(v). That composition has never been gated end to end
against the real forward: `gate_shape.py` gated dJ_phi/ds, and
`tests/test_design_filter.py` proves the transpose to 1e-12 by the dot-product
identity, but the two had not been run together on the melt objective. This
runs before any multi-start optimization.

Layers:

  S1  dJ_phi/ds at a FIXED read index, no filter. The reference layer, so a
      failure in S2 can be bisected to the filter rather than guessed at.
  S2  dJ_phi/dv with the filter in the chain. THE gradient the solve uses.

Probes per layer: the maximum-sensitivity cell, a fixed pseudo-random in-part
cell, a random unit direction over in-part cells, and for S2 a SMOOTH random
direction (a random direction passed through the filter, then normalized),
because a rough random direction is mostly cell-scale content and that is
exactly what the filter removes, so the finite-difference signal is damped while
the objective's roundoff floor is not.

Read state. The gate is taken at a FIXED read index, the argmin of J_phi on the
base run. That is the right layer to gate because the stop is optimized to
stationarity and the envelope theorem removes the dt*/ds term
(`shape_objective.py` module docstring). Whether the argmin index actually moves
under the probe perturbations is measured rather than assumed and reported.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_ms <shape> <out.json> [sigma_cells]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, design_filter as df, forward as fwd, gradops
from . import library_solve as lib
from . import shape_objective as so
from .gate_rho import EPSILONS, PASS_REL_ERR, SUBGRADIENT_PASS_REL_ERR, _probe_dirs, default_v
from .pins import build_case, load_cfg


def run_forward(case, s):
    """Exactly the forward the multi-start solve runs, early stop included."""
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=lib.PATIENCE)


def gradient_direction(pm, g) -> np.ndarray:
    """The unit descent direction the optimizer actually steps along.

    Added because it is the DECISION-RELEVANT probe and because the other
    direction probes are dominated by their own small analytic derivative: a
    rough random direction is nearly orthogonal to the gradient (MEASURED on the
    square, analytic -3.5e-03 against a gradient norm of 20.6), so its relative
    error is a statement about the denominator, not about the gradient. Along
    the gradient the analytic derivative is the full gradient norm, which is the
    largest available, so this probe reports the smallest relative error the
    measured arithmetic floor permits. It is reported alongside every other
    probe, never instead of them.
    """
    d = np.zeros(pm.shape)
    n = float(np.linalg.norm(g[pm]))
    if n > 0.0:
        d[pm] = g[pm] / n
    return d


def ramp_population(case, T) -> dict:
    """How many cells sit strictly inside the phase-change ramp at the read state.

    The melt fraction is clipped, phi = clip((T - t_pc) / dt_pc + 0.5, 0, 1), and
    the adjoint seed carries the mask of cells strictly inside that ramp. Cells
    at the clip are kinks in J, so the size of that population bounds how
    subgradient-like the objective is at this point.
    """
    from . import shape_objective as _so
    _phi, inside = _so.phi_field(T, case)
    pm = case.part_mask
    return {"n_in_ramp": int(inside.sum()),
            "n_in_ramp_in_part": int((inside & pm).sum()),
            "n_in_ramp_in_bed": int((inside & ~pm).sum()),
            "n_part_cells": int(pm.sum())}


def transpose_consistency(case, ops, v0, sigma_cells: float, read_index: int) -> dict:
    """Is dJ/dv exactly F^T applied to dJ/ds, to machine precision?

    This is the bisect that separates a filter-wiring error from a property of
    the forward. F is linear in v at fixed masks, so the directional derivative
    along d in the design variable must equal the directional derivative along
    F_lin(d) in the map, where F_lin is the filter with the nominal outside
    value set to zero. If this identity holds to machine precision, then the
    filtered analytic gradient is EXACTLY the unfiltered one composed with a
    proven-exact linear operator, and any remaining finite-difference
    disagreement belongs to the forward and not to the filter.
    """
    pm = case.part_mask
    s0 = df.apply_filter(v0, pm, sigma_cells)
    tr = run_forward(case, s0)
    i = min(int(read_index), tr.n_outer - 1)
    _J, seed = so.shape_J_and_seed(tr.T_at_end(i), case)
    g_s = adjoint.gradient(case, s0, tr, {i: seed}, grad_ops=ops)
    g_v = df.filter_vjp(g_s, pm, sigma_cells)
    out = {"read_index": int(i), "probes": {}}
    for pname, d, _c in _probe_dirs(pm, g_v, sigma_cells=sigma_cells):
        lhs = float(np.sum(g_v * d))
        rhs = float(np.sum(g_s * df.apply_filter(d, pm, sigma_cells, outside=0.0)))
        out["probes"][pname] = {"dJdv_dot_d": lhs, "dJds_dot_Fd": rhs,
                                "rel_err": abs(lhs - rhs) / max(abs(lhs), 1e-30)}
    out["max_rel_err"] = max(p["rel_err"] for p in out["probes"].values())
    out["PASS_machine_precision"] = bool(out["max_rel_err"] < 1e-10)
    del tr
    return out


def gate_layer(case, name, ops, v0, read_index: int, sigma_cells: float) -> dict:
    pm = case.part_mask
    filt = float(sigma_cells) > 0.0

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells) if filt else np.where(pm, v, 1.0)

    def J_of(v):
        tr = run_forward(case, to_map(v))
        i = min(int(read_index), tr.n_outer - 1)
        return so.shape_J_and_seed(tr.T_at_end(i), case)[0]

    tr0 = run_forward(case, to_map(v0))
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    g_s = adjoint.gradient(case, to_map(v0), tr0, {i0: seed}, grad_ops=ops)
    g = df.filter_vjp(g_s, pm, sigma_cells) if filt else np.where(pm, g_s, 0.0)

    out = {"layer": name, "sigma_cells": float(sigma_cells), "J0": float(J0),
           "read_index": int(i0), "n_outer": int(tr0.n_outer),
           "grad_norm": float(np.linalg.norm(g[pm])),
           "grad_max_abs": float(np.max(np.abs(g[pm]))),
           "ramp_population": ramp_population(case, tr0.T_at_end(i0)), "probes": {}}
    probes = list(_probe_dirs(pm, g, sigma_cells=sigma_cells))
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
        # In the roundoff-dominated tail the central-difference error is
        # floor / (2 eps), so 2 eps times the absolute error estimates the
        # objective's absolute evaluation floor. Taken at the two smallest
        # epsilons, where roundoff and not curvature dominates.
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


def stop_index_stability(case, v0, sigma_cells: float, eps: float = 1e-3) -> dict:
    """Does the J_phi argmin move under the probe perturbations?"""
    pm = case.part_mask
    to_map = ((lambda v: df.apply_filter(v, pm, sigma_cells)) if sigma_cells > 0
              else (lambda v: np.where(pm, v, 1.0)))
    base = so.optimal_stop(run_forward(case, to_map(v0)), case)
    out = {"eps": float(eps), "base_index": int(base.index), "moved": False, "probes": {}}
    for pname, d, _c in _probe_dirs(pm, np.where(pm, 1.0, 0.0)):
        ip = so.optimal_stop(run_forward(case, to_map(v0 + eps * d)), case).index
        im = so.optimal_stop(run_forward(case, to_map(v0 - eps * d)), case).index
        out["probes"][pname] = {"plus": int(ip), "minus": int(im),
                                "moved": bool(ip != base.index or im != base.index)}
        out["moved"] = out["moved"] or out["probes"][pname]["moved"]
    return out


def main(shape: str, out_path: str, sigma_cells: float = df.DEFAULT_SIGMA_CELLS) -> dict:
    t0 = time.perf_counter()
    cfg_path = lib.shape_config(shape)
    case = build_case(load_cfg(cfg_path))
    ops = gradops.gradient_matrices(case.x, case.y)
    v0 = default_v(case)

    base = run_forward(case, np.where(case.part_mask, v0, 1.0))
    st = so.optimal_stop(base, case)
    res = {"shape": shape, "config": str(cfg_path), "sigma_cells": float(sigma_cells),
           "objective": "J_phi = sum over the WHOLE domain of (phi - chi_part)^2",
           "base": {"read_index": int(st.index), "read_time_s": st.time_s,
                    "J_phi": st.J, "at_horizon": st.at_horizon,
                    "n_outer": int(base.n_outer)},
           "layers": []}

    for name, sig in (("S1_unfiltered_fixed_read", 0.0),
                      ("S2_filtered_fixed_read", float(sigma_cells))):
        r = gate_layer(case, name, ops, v0, st.index, sig)
        res["layers"].append(r)
        extra = (f"smoothdir={r['probes']['smooth_random_direction']['best_rel_err']:.3e} "
                 if "smooth_random_direction" in r["probes"] else "")
        print(f"{r['layer']:26s} J0={r['J0']:.6f} "
              f"maxcell={r['probes']['max_sensitivity_cell']['best_rel_err']:.3e} "
              f"randcell={r['probes']['random_cell']['best_rel_err']:.3e} "
              f"randdir={r['probes']['random_direction']['best_rel_err']:.3e} "
              + extra
              + f"graddir={r['probes']['gradient_direction']['best_rel_err']:.3e} "
              + f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
                f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    res["transpose_consistency"] = transpose_consistency(
        case, ops, v0, float(sigma_cells), st.index)
    print(f"filter transpose consistency: max relative error "
          f"{res['transpose_consistency']['max_rel_err']:.3e}, machine-precision "
          f"PASS = {res['transpose_consistency']['PASS_machine_precision']}", flush=True)

    res["stop_index_stability"] = stop_index_stability(case, v0, float(sigma_cells))
    print(f"stop index stability: base {res['stop_index_stability']['base_index']}, "
          f"moved = {res['stop_index_stability']['moved']}", flush=True)
    res["ALL_GATES_PASS"] = all(l["PASS"] for l in res["layers"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = all(l["PASS_subgradient"] for l in res["layers"])
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"ALL_GATES_PASS at 1e-6 = {res['ALL_GATES_PASS']}, at the 1e-5 "
          f"subgradient standard = {res['ALL_GATES_PASS_SUBGRADIENT']}, "
          f"wall {res['wall_s']:.1f} s")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else df.DEFAULT_SIGMA_CELLS)
