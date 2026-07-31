"""Head-to-head: adjoint direction versus the gain-calibrated proportional map.

Protocol is frozen in `out_adjoint/PREREGISTRATION.md`. Run:

    ./.venv312/bin/python -m adjoint2d.head_to_head <config.yaml> <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import adjoint, control, forward as fwd, gradops, objective as obj
from .pins import build_case, load_cfg

BUDGETS = (5.0, 15.0, 40.0)
SAT_BOX = (0.0, 1.0)
# The infeasible branch must be worse than EVERY feasible point, otherwise it
# is not a barrier. The offset dominates any sigma_T the engine can produce
# (the worst uniform baseline in the 19-shape library is about 60 C).
MELT_PENALTY_OFFSET = 1.0e4
MELT_PENALTY = 1.0e4


def _evaluate(case, s, ops=None, want_grad_obj=None):
    tr = fwd.forward(case, s, keep_checkpoints=want_grad_obj is not None,
                     stop_after_phi=0.90, stop_margin_steps=2)
    m = obj.scored_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["feasible"] = m["status"] == "OK"
    m["fit"] = m["heating_peak_sigma_T_c"]
    m["holdout"] = m["melt_onset_sigma_T_c"]
    return tr, m


def measure_cost_ratio(case, s, ops) -> dict:
    """Measure the adjoint cost separately for EACH fit objective.

    The reverse sweep only spans outer steps up to the read state, so the
    heating-peak objective (an early read state) is much cheaper to
    differentiate than the melt-onset objective (a late one). Charging both
    arms the same ratio would silently over-fund the melt-onset arm.
    """
    t0 = time.perf_counter()
    tr = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=0.90, stop_margin_steps=2)
    t1 = time.perf_counter()
    forward_s = t1 - t0

    out = {"forward_s": forward_s}
    for tag, fn in (("peak", obj.objective_heating_peak),
                    ("melt", obj.objective_melt_onset_interp)):
        try:
            _J, seeds = fn(tr, case)
        except obj.MeltNotReached:
            out[f"adjoint_{tag}_s"] = None
            out[f"ratio_{tag}"] = None
            continue
        t2 = time.perf_counter()
        adjoint.gradient(case, s, tr, seeds, grad_ops=ops)
        t3 = time.perf_counter()
        out[f"adjoint_{tag}_s"] = t3 - t2
        out[f"ratio_{tag}"] = (t3 - t2) / max(forward_s, 1e-9)
    return out


# ---------------------------------------------------------------------------

def uniform_s(case) -> np.ndarray:
    """s = 1 everywhere. Outside the part the saturation is not a design
    variable and stays at its nominal value, so this arm is exactly the
    production no-dopant-map baseline in BOTH injection channels."""
    return np.ones(case.part_mask.shape)


def run_uniform(case, ops):
    s = uniform_s(case)
    tr, m = _evaluate(case, s)
    m["arm"] = "U_uniform"
    return s, tr, m


def run_control(case, tr_uniform, n_gain_evals: int):
    """Arm H1. Proxy is the melt-onset temperature field of the uniform run."""
    rs = obj.read_states(tr_uniform)
    if rs.melt_onset_index is None:
        raise obj.MeltNotReached("uniform arm never melts; the proxy field is undefined")
    proxy = tr_uniform.T_at_end(rs.melt_onset_index)
    rows: list[dict] = []

    def fit_of(m: float) -> float:
        sat = control.proportional_inverse_map(proxy, case.part_mask, magnitude=m)
        _tr, met = _evaluate(case, sat)
        met["magnitude"] = float(m)
        met["eval_index"] = len(rows) + 1
        met["sat"] = sat
        rows.append(met)
        if met["feasible"]:
            return met["fit"]
        return control.infeasible_rank(met, t_pc_c=case.pins.t_pc_c)

    control.golden_section(fit_of, 0.05, 2.50, n_evals=n_gain_evals)
    return proxy, rows


def run_adjoint(case, ops, n_evals: int, fit: str, box=SAT_BOX):
    """Arm A1 (fit='peak') or A2 (fit='melt')."""
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
    s0 = uniform_s(case)
    rows: list[dict] = []
    store: dict[str, np.ndarray] = {}

    def unpack(v):
        s = np.ones(pm.shape)
        s.ravel()[idx] = v
        return s

    def fun(v):
        if len(rows) >= n_evals:
            raise StopIteration
        s = unpack(v)
        tr = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=0.90, stop_margin_steps=2)
        met = obj.scored_metrics(tr, case)
        met["P_abs_W_per_m"] = tr.P_abs_B
        met["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
        met["frac_qrf_cap"] = tr.frac_qrf_cap
        met["feasible"] = met["status"] == "OK"
        met["fit"] = met["heating_peak_sigma_T_c"]
        met["holdout"] = met["melt_onset_sigma_T_c"]
        met["eval_index"] = len(rows) + 1
        if not met["feasible"]:
            # Feasibility barrier. The objective is UNDEFINED when the part
            # never reaches the melt-onset read state, so instead of a silent
            # final-step fallback the arm is pushed back toward melting by a
            # differentiable shortfall penalty. Infeasible iterates can never
            # be selected (see control.select_on_fit).
            n_best = int(np.argmax(tr.mean_phi_part))
            phi_max, gp = obj.phi_bar_and_seed(tr.T_at_end(n_best), case)
            J = MELT_PENALTY_OFFSET + MELT_PENALTY * (obj.PHI_TARGET - phi_max)
            seeds = {n_best: -MELT_PENALTY * gp}
            met["infeasible_penalty"] = True
        elif fit == "melt":
            J, seeds = obj.objective_melt_onset_interp(tr, case)
        else:
            J, seeds = obj.objective_heating_peak(tr, case)
        met["fit_objective"] = float(J)
        g = adjoint.gradient(case, s, tr, seeds, grad_ops=ops)
        rows.append(met)
        store[str(met["eval_index"])] = s.copy()
        return float(J), g.ravel()[idx].astype(float)

    v0 = s0.ravel()[idx]
    try:
        minimize(fun, v0, jac=True, method="L-BFGS-B",
                 bounds=[box] * len(idx),
                 options={"maxiter": 10_000, "maxfun": n_evals, "ftol": 1e-14, "gtol": 1e-14})
    except StopIteration:
        pass
    return rows, store


# ---------------------------------------------------------------------------

def main(cfg_path: str, shape: str, outdir: str) -> dict:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    ops = gradops.gradient_matrices(case.x, case.y)

    t0 = time.perf_counter()
    s_u, tr_u, m_u = run_uniform(case, ops)
    cost = measure_cost_ratio(case, s_u, ops)
    r_peak = cost["ratio_peak"]
    r_melt = cost["ratio_melt"] if cost.get("ratio_melt") is not None else r_peak

    n1_max = control.max_gradient_evals(max(BUDGETS), r_peak)
    n2_max = control.max_gradient_evals(max(BUDGETS), r_melt)
    n_gain_max = int(max(BUDGETS)) - 1

    proxy, ctrl_rows = run_control(case, tr_u, n_gain_max)
    a1_rows, a1_store = run_adjoint(case, ops, n1_max, fit="peak")
    a2_rows, a2_store = run_adjoint(case, ops, n2_max, fit="melt")

    def strip(rows):
        return [{k: v for k, v in r.items() if k != "sat"} for r in rows]

    result = {
        "shape": shape,
        "config": str(cfg_path),
        "n_part_cells": case.n_part,
        "cost": cost,
        "n_gradient_evals_at_B40_peak": n1_max,
        "n_gradient_evals_at_B40_melt": n2_max,
        "n_gain_evals_at_B40": n_gain_max,
        "uniform": m_u,
        "control_rows": strip(ctrl_rows),
        "adjoint_fit_peak_rows": a1_rows,
        "adjoint_fit_melt_rows": a2_rows,
        "budgets": {},
    }

    for B in BUDGETS:
        n_gain = max(0, int(B) - 1)
        n1 = control.max_gradient_evals(B, r_peak)
        n2 = control.max_gradient_evals(B, r_melt)
        sel_c = control.select_on_fit(ctrl_rows[:n_gain])
        sel_cm = control.select_on_holdout(ctrl_rows[:n_gain])
        sel_1 = control.select_on_fit(a1_rows[:n1])
        sel_2 = control.select_on_holdout(a2_rows[:n2])
        result["budgets"][str(int(B))] = {
            "n_gain_evals": n_gain,
            "n_gradient_evals_A1": n1,
            "n_gradient_evals_A2": n2,
            "spent_forward_equivalents_control": control.forward_equivalents(1 + n_gain, 0, r_peak),
            "spent_forward_equivalents_A1": control.forward_equivalents(n1, n1, r_peak),
            "spent_forward_equivalents_A2": control.forward_equivalents(n2, n2, r_melt),
            "H1_peak": None if sel_c is None else {k: v for k, v in sel_c.items() if k != "sat"},
            "H1_melt": None if sel_cm is None else {k: v for k, v in sel_cm.items() if k != "sat"},
            "A1": sel_1,
            "A2": sel_2,
        }

    t_all = time.perf_counter() - t0
    result["wall_s"] = t_all
    (out / f"{shape}.json").write_text(json.dumps(result, indent=2, default=float))

    best_c = control.select_on_fit(ctrl_rows)
    maps = {"uniform": s_u, "proxy_T_melt": proxy}
    if best_c is not None:
        maps["control_best"] = best_c["sat"]
    for tag, store, rows in (("a1", a1_store, a1_rows), ("a2", a2_store, a2_rows)):
        sel = control.select_on_fit(rows)
        if sel is not None:
            maps[f"adjoint_{tag}_best"] = store[str(sel["eval_index"])]
    np.savez_compressed(out / f"{shape}_maps.npz", **maps)
    return result


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(r["budgets"], indent=2, default=float))
