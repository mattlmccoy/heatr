"""Finite-difference gates for the DENSITY-region gradient dJ_rho/ds.

Run before any optimization. Central differences, epsilon swept 1e-3 down to
1e-8, with probes: the maximum-sensitivity cell (argmax of the analytic
gradient magnitude), a fixed pseudo-random in-part cell, a random unit
direction over all in-part cells, and for the filtered layer a smooth random
direction (a random direction passed through the filter, then normalized).

Layers:

  R1  dJ_rho/ds at a FIXED read index. The pure partial derivative.
  R2  dJ_rho/dv with the physical-length design filter in the chain,
      s = F(v). Gates the filter transpose end to end against the real forward,
      on top of R1. The transpose itself is separately proven to machine
      precision by the dot-product identity in `tests/test_design_filter.py`.

  read-index stability. The flat-onset stop is NOT a stationary point of J_rho
      (J_rho is monotone non-increasing, so its argmin is always the horizon
      and the envelope argument that covers the melt-region objective does NOT
      transfer). Instead of assuming the neglected dt_flat/ds term is small,
      the read index is recomputed under each probe perturbation and reported:
      if it does not move, the term is exactly zero over the tested range.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_rho <shape> <out.json> [sigma_cells]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, density_objective as dobj, design_filter as df
from . import forward as fwd, gradops, library_solve as lib
from .pins import build_case, load_cfg

# The sweep runs two decades further down than the melt-region gate does.
# MEASURED reason (`logs_rho/diag_bisect.log`): along a SINGLE-CELL direction
# at a corner of the part, J_rho carries a small high-curvature component, so
# central differences at 1e-3 to 1e-5 are biased by tens of percent and only
# epsilon at or below 1e-7 sits inside one smooth piece. The roundoff floor of
# J_rho was measured at about 2e-12 absolute (from the random-cell probe), so
# epsilon below 1e-8 is noise dominated. The usable window is narrow and it is
# swept explicitly rather than assumed.
EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8)
PASS_REL_ERR = 1e-6
# The campaign's documented subgradient standard: the melt-region gate of
# `SOLVE_ROBUSTNESS_VALIDATION.md` Section 7 bottoms at 1.22e-05 on its
# random-direction probe because the pinned population sits on the phase ramp.
SUBGRADIENT_PASS_REL_ERR = 1e-5


def default_v(case, seed: int = 4242) -> np.ndarray:
    """A deliberately non-nominal design point: smooth structure plus noise."""
    rng = np.random.default_rng(seed)
    ny, nx = case.part_mask.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    smooth = 0.80 + 0.18 * np.cos(1.7 * xx + 0.4) * np.sin(2.1 * yy + 0.9)
    v = np.ones((ny, nx))
    v[case.part_mask] = smooth[case.part_mask] + 0.02 * rng.standard_normal(int(case.part_mask.sum()))
    return v


def run_forward(case, s):
    """Full horizon, rho checkpoints kept.

    The shape-fidelity early stop is DISABLED on every density run: it
    truncates the march on the melt objective long before the density objective
    flattens, and truncating there would silently redefine the read state.
    """
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=None)


def obj_fixed(tr, case, index: int):
    idx = min(int(index), tr.n_outer - 1)
    J, g = dobj.rho_J_and_seed(tr.rho_at_end(idx), case)
    return J, {idx: g}


def obj_rule(tr, case, tol: float = dobj.FLAT_TOL):
    st = dobj.rho_stop(tr, case, tol=tol)
    J, g = dobj.rho_J_and_seed(tr.rho_at_end(st.index), case)
    return J, {st.index: g}


def _probe_dirs(pm, g, seed=7, sigma_cells: float = 0.0):
    rng = np.random.default_rng(seed)
    d_rand = np.zeros(pm.shape)
    u = rng.standard_normal(int(pm.sum()))
    d_rand[pm] = u / np.linalg.norm(u)
    gp = np.where(pm, np.abs(g), -np.inf)
    i_max = np.unravel_index(int(np.argmax(gp)), pm.shape)
    d_max = np.zeros(pm.shape)
    d_max[i_max] = 1.0
    idx = np.argwhere(pm)
    i_rnd = tuple(idx[rng.integers(len(idx))])
    d_rnd = np.zeros(pm.shape)
    d_rnd[i_rnd] = 1.0
    out = [("max_sensitivity_cell", d_max, i_max),
           ("random_cell", d_rnd, i_rnd),
           ("random_direction", d_rand, None)]
    if float(sigma_cells) > 0.0:
        # A rough random unit vector is mostly cell-scale content, which is
        # exactly what the filter removes: MEASURED damping ||F d|| / ||d|| =
        # 0.2213 at sigma = 1.5 cells on the square. The finite-difference
        # signal is damped by that factor while the objective's roundoff floor
        # is not, so the rough direction is the worst probe available for the
        # filtered layer. The smooth direction is the design perturbation the
        # filter actually passes.
        fdr = df.apply_filter(d_rand, pm, sigma_cells, outside=0.0)
        d_sm = np.zeros(pm.shape)
        d_sm[pm] = fdr[pm] / np.linalg.norm(fdr[pm])
        out.append(("smooth_random_direction", d_sm, None))
    return tuple(out)


def gate(case, name, ops, v0, objective, sigma_cells: float, is_gate: bool = True) -> dict:
    """One layer. `sigma_cells` = 0 means the design variable IS the map."""
    pm = case.part_mask
    filt = float(sigma_cells) > 0.0

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells) if filt else v

    def J_of(v):
        return objective(run_forward(case, to_map(v)), case)[0]

    tr0 = run_forward(case, to_map(v0))
    J0, seeds_rho = objective(tr0, case)
    g_s = adjoint.gradient(case, to_map(v0), tr0, {}, grad_ops=ops, seeds_rho=seeds_rho)
    g = df.filter_vjp(g_s, pm, sigma_cells) if filt else g_s

    out = {"layer": name, "sigma_cells": float(sigma_cells), "J0": float(J0),
           "read_index": int(max(seeds_rho)), "n_outer": int(tr0.n_outer),
           "grad_norm": float(np.linalg.norm(g[pm])),
           "grad_max_abs": float(np.max(np.abs(g[pm]))),
           "is_pass_fail_gate": bool(is_gate), "probes": {}}
    for pname, d, cell in _probe_dirs(pm, g, sigma_cells=sigma_cells):
        ana = float(np.sum(g * d))
        rows = []
        for eps in EPSILONS:
            fd = (J_of(v0 + eps * d) - J_of(v0 - eps * d)) / (2.0 * eps)
            rows.append({"eps": eps, "fd": float(fd),
                         "rel_err": abs(fd - ana) / max(abs(ana), 1e-30)})
        best = min(rows, key=lambda r: r["rel_err"])
        out["probes"][pname] = {
            "cell": None if cell is None else [int(c) for c in cell],
            "analytic": ana, "sweep": rows, "best_rel_err": best["rel_err"],
            "best_eps": best["eps"], "PASS": bool(best["rel_err"] < PASS_REL_ERR),
            "PASS_subgradient": bool(best["rel_err"] < SUBGRADIENT_PASS_REL_ERR)}
    out["PASS"] = all(p["PASS"] for p in out["probes"].values())
    out["PASS_subgradient"] = all(p["PASS_subgradient"] for p in out["probes"].values())
    out["n_probes"] = len(out["probes"])
    out["n_probes_pass_1e-6"] = sum(p["PASS"] for p in out["probes"].values())
    out["n_probes_pass_1e-5"] = sum(p["PASS_subgradient"] for p in out["probes"].values())
    return out


def read_index_stability(case, v0, sigma_cells, tol, eps=1e-3) -> dict:
    """Does the flat-onset read index move under the probe perturbations?

    If it does not, the neglected dt_flat/ds term is exactly zero over the
    tested range and the fixed-read gradient IS the rule-pinned gradient. If it
    does, that is the size of the term the solve neglects, and it is reported.
    """
    pm = case.part_mask
    to_map = (lambda v: df.apply_filter(v, pm, sigma_cells)) if sigma_cells > 0 else (lambda v: v)
    base = dobj.rho_stop(run_forward(case, to_map(v0)), case, tol=tol)
    g = np.zeros(pm.shape)
    out = {"eps": float(eps), "base_index": base.index, "moved": False, "probes": {}}
    for pname, d, _cell in _probe_dirs(pm, g if g.any() else np.where(pm, 1.0, 0.0)):
        ip = dobj.rho_stop(run_forward(case, to_map(v0 + eps * d)), case, tol=tol).index
        im = dobj.rho_stop(run_forward(case, to_map(v0 - eps * d)), case, tol=tol).index
        out["probes"][pname] = {"plus": ip, "minus": im,
                                "moved": bool(ip != base.index or im != base.index)}
        out["moved"] = out["moved"] or out["probes"][pname]["moved"]
    return out


def main(shape: str, out_path: str, sigma_cells: float = df.DEFAULT_SIGMA_CELLS,
         tol: float = dobj.FLAT_TOL) -> dict:
    t0 = time.perf_counter()
    cfg_path = lib.shape_config(shape)
    case = build_case(load_cfg(cfg_path))
    ops = gradops.gradient_matrices(case.x, case.y)
    v0 = default_v(case)

    base = run_forward(case, v0)
    st = dobj.rho_stop(base, case, tol=tol)
    res = {"shape": shape, "config": str(cfg_path), "flat_tol": float(tol),
           "sigma_cells": float(sigma_cells),
           "base": {"read_index": st.index, "read_time_s": st.time_s,
                    "J_rho": st.J, "argmin_index": st.argmin_index,
                    "argmin_J": st.argmin_J, "at_horizon": st.at_horizon,
                    "no_progress": st.no_progress,
                    "frac_remaining": st.frac_remaining,
                    "n_outer": base.n_outer,
                    "mean_rho_rel_part": float(np.mean(
                        base.rho_at_end(st.index)[case.part_mask]))},
           "layers": []}

    layers = (
        ("R1_fixed_read", lambda tr, c, i=st.index: obj_fixed(tr, c, i), 0.0, True),
        ("R2_filtered_fixed_read", lambda tr, c, i=st.index: obj_fixed(tr, c, i),
         float(sigma_cells), True),
    )
    for name, fn, sig, is_gate in layers:
        r = gate(case, name, ops, v0, fn, sig, is_gate)
        res["layers"].append(r)
        print(f"{r['layer']:24s} J0={r['J0']:.6f} "
              f"maxcell={r['probes']['max_sensitivity_cell']['best_rel_err']:.3e} "
              f"randcell={r['probes']['random_cell']['best_rel_err']:.3e} "
              f"randdir={r['probes']['random_direction']['best_rel_err']:.3e} "
              + (f"smoothdir={r['probes']['smooth_random_direction']['best_rel_err']:.3e} "
                 if 'smooth_random_direction' in r['probes'] else "")
              + f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
                f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    res["read_index_stability"] = read_index_stability(case, v0, 0.0, tol)
    print(f"read index stability: base {res['read_index_stability']['base_index']}, "
          f"moved = {res['read_index_stability']['moved']}", flush=True)
    res["ALL_GATES_PASS"] = all(l["PASS"] for l in res["layers"] if l["is_pass_fail_gate"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = all(
        l["PASS_subgradient"] for l in res["layers"] if l["is_pass_fail_gate"])
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"ALL_GATES_PASS at 1e-6 = {res['ALL_GATES_PASS']}, "
          f"at the 1e-5 subgradient standard = {res['ALL_GATES_PASS_SUBGRADIENT']}, "
          f"wall {res['wall_s']:.1f} s")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else df.DEFAULT_SIGMA_CELLS)
