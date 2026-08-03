"""Bisect for the permittivity-channel gate: where does the residual live?

`gate_eps.py` measures that the FILTERED permittivity layer does not reach the
1e-6 clean-smooth standard on every probe. Two cheap checks localize that
residual instead of leaving it as a guess.

1. TRANSPOSE CONSISTENCY, permittivity channel. The design filter F is linear
   in v at fixed masks, so the directional derivative along d in the design
   variable must equal the directional derivative along F_lin(d) in the map,
   where F_lin is the filter with the nominal outside value set to zero. This
   identity involves NO finite difference. If it holds to machine precision the
   filtered analytic gradient is EXACTLY the unfiltered one composed with a
   proven-exact linear operator, and every remaining finite-difference
   disagreement belongs to the forward and not to the filter or the new term.

2. A REFINED EPSILON SWEEP inside the usable window. The coarse sweep shows the
   central difference is wild above about 1e-6 and roundoff dominated below
   about 1e-7, which is the same narrow window `MULTISTART_REPORT.md`
   Section 2.3 measured for the conductivity channel. This resweeps the window
   densely so the reported best relative error is not an artifact of a coarse
   grid of epsilons.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_eps_bisect <shape> <out.json> [sigma_cells]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, design_filter as df, gradops
from . import shape_objective as so
from .gate_eps import run_forward
from .gate_ms import gradient_direction
from .gate_rho import _probe_dirs, default_v
from .library_solve import shape_config
from .pins import build_case, load_cfg

FINE_EPSILONS = (3e-6, 2e-6, 1.5e-6, 1e-6, 7e-7, 5e-7, 3e-7, 2e-7, 1.5e-7, 1e-7, 7e-8)


def transpose_consistency_eps(case, ops, v0, sigma_cells: float,
                              read_index: int) -> dict:
    pm = case.part_mask
    s0 = df.apply_filter(v0, pm, sigma_cells)
    tr = run_forward(case, s0, True)
    i = min(int(read_index), tr.n_outer - 1)
    _J, seed = so.shape_J_and_seed(tr.T_at_end(i), case)
    g_s = adjoint.gradient(case, s0, tr, {i: seed}, grad_ops=ops, eps_covary=True)
    del tr
    g_v = df.filter_vjp(g_s, pm, sigma_cells)
    out = {"read_index": int(i), "probes": {}}
    for pname, d, _c in _probe_dirs(pm, g_v, sigma_cells=sigma_cells):
        lhs = float(np.sum(g_v * d))
        rhs = float(np.sum(g_s * df.apply_filter(d, pm, sigma_cells, outside=0.0)))
        out["probes"][pname] = {"dJdv_dot_d": lhs, "dJds_dot_Fd": rhs,
                                "rel_err": abs(lhs - rhs) / max(abs(lhs), 1e-30)}
    out["max_rel_err"] = max(p["rel_err"] for p in out["probes"].values())
    out["PASS_machine_precision"] = bool(out["max_rel_err"] < 1e-10)
    return out


def refined_sweep(case, ops, v0, sigma_cells: float, read_index: int) -> dict:
    pm = case.part_mask

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells)

    def J_of(v):
        tr = run_forward(case, to_map(v), True)
        i = min(int(read_index), tr.n_outer - 1)
        j = so.shape_J_and_seed(tr.T_at_end(i), case)[0]
        del tr
        return j

    s0 = to_map(v0)
    tr0 = run_forward(case, s0, True)
    i0 = min(int(read_index), tr0.n_outer - 1)
    _J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    g_s = adjoint.gradient(case, s0, tr0, {i0: seed}, grad_ops=ops, eps_covary=True)
    del tr0
    g = df.filter_vjp(g_s, pm, sigma_cells)

    probes = list(_probe_dirs(pm, g, sigma_cells=sigma_cells))
    probes.append(("gradient_direction", gradient_direction(pm, g), None))
    out: dict = {"read_index": int(i0), "epsilons": list(FINE_EPSILONS), "probes": {}}
    for pname, d, cell in probes:
        ana = float(np.sum(g * d))
        rows = []
        for eps in FINE_EPSILONS:
            fd = (J_of(v0 + eps * d) - J_of(v0 - eps * d)) / (2.0 * eps)
            rows.append({"eps": eps, "fd": float(fd),
                         "rel_err": abs(fd - ana) / max(abs(ana), 1e-30),
                         "abs_err": abs(fd - ana)})
        best = min(rows, key=lambda r: r["rel_err"])
        out["probes"][pname] = {
            "cell": None if cell is None else [int(c) for c in cell],
            "analytic": ana, "sweep": rows,
            "best_rel_err": best["rel_err"], "best_eps": best["eps"],
            "best_abs_err": best["abs_err"],
            "PASS_1e-6": bool(best["rel_err"] < 1e-6),
            "PASS_1e-5": bool(best["rel_err"] < 1e-5)}
        print(f"  {pname:26s} ana {ana:+.5e}  best_rel {best['rel_err']:.3e} "
              f"at eps {best['eps']:g}", flush=True)
    out["n_probes"] = len(out["probes"])
    out["n_pass_1e-6"] = sum(p["PASS_1e-6"] for p in out["probes"].values())
    out["n_pass_1e-5"] = sum(p["PASS_1e-5"] for p in out["probes"].values())
    return out


def main(shape: str, out_path: str, sigma_cells: float = df.DEFAULT_SIGMA_CELLS) -> dict:
    t0 = time.perf_counter()
    case = build_case(load_cfg(shape_config(shape)))
    ops = gradops.gradient_matrices(case.x, case.y)
    v0 = default_v(case)
    base = run_forward(case, np.where(case.part_mask, v0, 1.0), True)
    idx = int(so.optimal_stop(base, case).index)
    del base

    res = {"shape": shape, "sigma_cells": float(sigma_cells), "read_index": idx}
    res["transpose_consistency_eps_channel"] = transpose_consistency_eps(
        case, ops, v0, float(sigma_cells), idx)
    print(f"[{shape}] filter transpose consistency in the permittivity channel: "
          f"max relative error "
          f"{res['transpose_consistency_eps_channel']['max_rel_err']:.3e}, "
          f"machine-precision PASS = "
          f"{res['transpose_consistency_eps_channel']['PASS_machine_precision']}",
          flush=True)

    print(f"[{shape}] refined epsilon sweep, filtered permittivity layer:", flush=True)
    res["refined_sweep_E2"] = refined_sweep(case, ops, v0, float(sigma_cells), idx)
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    r = res["refined_sweep_E2"]
    print(f"[{shape}] refined: {r['n_pass_1e-6']}/{r['n_probes']} at 1e-6, "
          f"{r['n_pass_1e-5']}/{r['n_probes']} at 1e-5, wall {res['wall_s']:.0f} s")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else df.DEFAULT_SIGMA_CELLS)
