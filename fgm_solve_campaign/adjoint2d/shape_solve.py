"""Solve for the ideal dopant map under the shape-fidelity objective.

Arms, all scored under the SAME J with their OWN optimal stop time:

  U        uniform s = 1
  H_sig    the window-selected heuristic gain, conductivity-only injection
           (the same actuator the adjoint uses)
  H_eps    the same gain, permittivity-co-varying injection (the channel the
           published gains were selected in)
  A1       L-BFGS-B on the finite-difference-gated shape gradient, box [0, 1]
           (matched to the heuristic's actuation range)
  A15      the same, box [0, 1.5] (the two-sided per-node actuator)

Run:  ./.venv312/bin/python -m adjoint2d.shape_solve <config.yaml> <shape> <gain> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import adjoint, control, forward as fwd, gradops, objective as obj
from . import shape_objective as so
from .pins import build_case, load_cfg

BUDGETS = (15.0, 40.0)
PATIENCE = 250


def _forward(case, s, checkpoints=False, eps_covary=False):
    return fwd.forward(case, s, keep_checkpoints=checkpoints, stop_after_phi=None,
                       shape_stop_patience=PATIENCE, eps_covary=eps_covary)


def _score(case, tr) -> dict:
    m = so.full_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    return m


def run_adjoint(case, ops, n_evals: int, box) -> tuple[list[dict], dict]:
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
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
        tr = _forward(case, s, checkpoints=True)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        g = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops)
        met = _score(case, tr)
        met["eval_index"] = len(rows) + 1
        rows.append(met)
        store[str(met["eval_index"])] = s.copy()
        return float(J), g.ravel()[idx].astype(float)

    v0 = np.ones(len(idx))
    try:
        minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=[box] * len(idx),
                 options={"maxiter": 10_000, "maxfun": n_evals,
                          "ftol": 1e-16, "gtol": 1e-16})
    except StopIteration:
        pass
    return rows, store


def best_by_J(rows: list[dict], k: int | None = None) -> dict | None:
    sub = rows if k is None else rows[:k]
    if not sub:
        return None
    return min(sub, key=lambda r: r["J"])


def heuristic_map(case, gain: float, outside: float = 1.0) -> tuple[np.ndarray, dict]:
    """Proportional-inverse map built from the uniform arm's melt-onset field.

    That is the proxy the published gains were selected against
    (`FGM_WINDOW_RESELECTION.md` reads the stored T_phi90 snapshot). When the
    uniform arm never reaches the phi_bar = 0.90 crossing (L_shape), the
    temperature field at the uniform arm's own optimal shape stop is used
    instead, and the fallback is recorded.
    """
    s_u = np.ones(case.part_mask.shape)
    # FULL horizon: the shape-fidelity early stop truncates the march long
    # before the phi_bar = 0.90 crossing on slow shapes, and using a truncated
    # run here would silently build the control's map from a different proxy
    # than the published gains were selected against.
    tr = fwd.forward(case, s_u, stop_after_phi=None)
    rs = obj.read_states(tr)
    info = {"proxy": "T_phi90"}
    if rs.melt_onset_index is None:
        st = so.optimal_stop(tr, case)
        proxy = tr.T_at_end(st.index)
        info = {"proxy": "T_at_uniform_optimal_stop", "fallback": True,
                "reason": "uniform arm never reaches phi_bar = 0.90"}
    else:
        proxy = tr.T_at_end(rs.melt_onset_index)
    return control.proportional_inverse_map(proxy, case.part_mask, magnitude=gain,
                                            outside=outside), info


def main(cfg_path: str, shape: str, gain: float, outdir: str) -> dict:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    ops = gradops.gradient_matrices(case.x, case.y)

    t0 = time.perf_counter()
    s_u = np.ones(case.part_mask.shape)
    t_a = time.perf_counter()
    tr_u = _forward(case, s_u, checkpoints=True)
    t_b = time.perf_counter()
    st_u = so.optimal_stop(tr_u, case)
    _J, seed = so.shape_J_and_seed(tr_u.T_at_end(st_u.index), case)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed}, grad_ops=ops)
    t_d = time.perf_counter()
    cost = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c,
            "ratio": (t_d - t_c) / max(t_b - t_a, 1e-9)}
    n_max = control.max_gradient_evals(max(BUDGETS), cost["ratio"])

    m_u = _score(case, tr_u)
    m_u["arm"] = "U_uniform"

    sat_h, proxy_info = heuristic_map(case, gain)
    m_hs = _score(case, _forward(case, sat_h))
    m_hs.update({"arm": "H_sig", "magnitude": gain})
    m_he = _score(case, _forward(case, sat_h, eps_covary=True))
    m_he.update({"arm": "H_eps", "magnitude": gain})

    a1_rows, a1_store = run_adjoint(case, ops, n_max, (0.0, 1.0))
    a15_rows, a15_store = run_adjoint(case, ops, n_max, (0.0, 1.5))

    res = {"shape": shape, "config": str(cfg_path), "gain": gain,
           "n_part_cells": case.n_part, "cost": cost,
           "n_gradient_evals_at_B40": n_max, "proxy_info": proxy_info,
           "uniform": m_u, "H_sig": m_hs, "H_eps": m_he,
           "A1_rows": a1_rows, "A15_rows": a15_rows, "budgets": {}}
    for B in BUDGETS:
        k = control.max_gradient_evals(B, cost["ratio"])
        res["budgets"][str(int(B))] = {
            "n_gradient_evals": k,
            "spent_forward_equivalents": control.forward_equivalents(k, k, cost["ratio"]),
            "A1": best_by_J(a1_rows, k), "A15": best_by_J(a15_rows, k),
        }
    res["wall_s"] = time.perf_counter() - t0

    maps = {"uniform": s_u, "heuristic": sat_h}
    for tag, rows, store in (("A1", a1_rows, a1_store), ("A15", a15_rows, a15_store)):
        b = best_by_J(rows)
        if b is not None:
            maps[tag] = store[str(b["eval_index"])]
    np.savez_compressed(out / f"{shape}_maps.npz", **maps,
                        part_mask=case.part_mask, x=case.x, y=case.y)
    (out / f"{shape}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4])
    u = r["uniform"]
    print(f"{'arm':10s} {'J':>10s} {'IoU':>7s} {'grow%':>7s} {'under%':>7s} "
          f"{'t_stop':>7s} {'phibar':>7s} {'P_abs':>8s}")
    def line(tag, m):
        if m is None:
            print(f"{tag:10s} {'n/a':>10s}")
            return
        print(f"{tag:10s} {m['J']:10.2f} {m['IoU']:7.4f} {m['bed_melt_pct_of_part']:7.2f} "
              f"{m['part_under_melt_pct']:7.2f} {m['t_stop_index']:7d} "
              f"{m['phi_bar_part_at_stop']:7.4f} {m['P_abs_W_per_m']:8.1f}")
    line("uniform", u)
    line("H_sig", r["H_sig"])
    line("H_eps", r["H_eps"])
    for B in ("15", "40"):
        line(f"A1@{B}", r["budgets"][B]["A1"])
        line(f"A15@{B}", r["budgets"][B]["A15"])
