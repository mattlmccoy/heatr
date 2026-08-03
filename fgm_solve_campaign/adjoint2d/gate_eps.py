"""Finite-difference gate for the PERMITTIVITY-CO-VARYING dopant channel.

Run BEFORE any optimization in this channel.

What is new against `gate_ms.py`. The design variable `s` now moves BOTH
material properties through the same effective fill `fill_frac * s`:

    sigma = sigma_v + (fill*s) * (sigma_d0 - sigma_v)
    eps_r = eps_v   + (fill*s) * (eps_d   - eps_v)

which is the production hook `fgm_feedback.saturation_map_npz`
(`rfam_eqs_coupled.py:2527-2532`), the channel every stored historical dopant
map was scored in. The conductivity-only channel the solve used until now is
the other production hook, `sat_map_npz_direct`, which pins permittivity to
geometry fill (`rfam_eqs_coupled.py:342`).

Layers, both with the permittivity channel ON in the forward AND in the
gradient:

  E1  dJ_phi/ds at a FIXED read index, no design filter. The reference layer,
      so a failure in E2 bisects to the filter rather than being guessed at.
  E2  dJ_phi/dv with the physical-length design filter in the chain, s = F(v).
      THE gradient the solve uses.

Probes per layer: the maximum-sensitivity cell, a fixed pseudo-random in-part
cell, a random unit direction over in-part cells, for E2 a smooth random
direction, and the gradient direction (the direction the optimizer steps
along, which carries the largest available analytic derivative and therefore
the smallest relative error the measured arithmetic floor permits). Every
probe is reported; none is dropped.

Two extra diagnostics that exist only because this channel is new:

  * CHANNEL SPLIT. The same reverse sweep is re-run with the permittivity term
    switched off, so the report can state what fraction of the gradient the new
    term actually contributes. A channel that changes the gradient by 1e-9 is
    not worth a census.
  * A CONDUCTIVITY-ONLY CONTROL LAYER (E0) at the same design point, so a
    failure that is really a property of the forward at this point is visible
    as a failure in both layers rather than being blamed on the new term.

Read state. The gate is taken at a FIXED read index, the argmin of J_phi on the
base run, because the stop is optimized to stationarity and the envelope
theorem removes the dt*/ds term. Whether the argmin actually moves under the
probe perturbations is MEASURED and reported, not assumed.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_eps <shape> <out.json> [sigma_cells]
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
from .gate_ms import gradient_direction, ramp_population
from .pins import build_case, load_cfg


def run_forward(case, s, eps_covary: bool):
    """Exactly the forward the eps-channel solve runs, early stop included."""
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=lib.PATIENCE, eps_covary=eps_covary)


def gate_layer(case, name, ops, v0, read_index: int, sigma_cells: float,
               eps_covary: bool) -> dict:
    pm = case.part_mask
    filt = float(sigma_cells) > 0.0

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells) if filt else np.where(pm, v, 1.0)

    def J_of(v):
        tr = run_forward(case, to_map(v), eps_covary)
        i = min(int(read_index), tr.n_outer - 1)
        return so.shape_J_and_seed(tr.T_at_end(i), case)[0]

    s0 = to_map(v0)
    tr0 = run_forward(case, s0, eps_covary)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    g_s = adjoint.gradient(case, s0, tr0, {i0: seed}, grad_ops=ops,
                           eps_covary=eps_covary)
    g = df.filter_vjp(g_s, pm, sigma_cells) if filt else np.where(pm, g_s, 0.0)

    out = {"layer": name, "eps_covary": bool(eps_covary),
           "sigma_cells": float(sigma_cells), "J0": float(J0),
           "read_index": int(i0), "n_outer": int(tr0.n_outer),
           "grad_norm": float(np.linalg.norm(g[pm])),
           "grad_max_abs": float(np.max(np.abs(g[pm]))),
           "ramp_population": ramp_population(case, tr0.T_at_end(i0)), "probes": {}}

    # channel split: the SAME trajectory, the SAME seed, the permittivity term
    # switched off in the reverse sweep only.
    if eps_covary:
        g_sig_only = adjoint.gradient(case, s0, tr0, {i0: seed}, grad_ops=ops,
                                      eps_covary=False)
        g_so = df.filter_vjp(g_sig_only, pm, sigma_cells) if filt else np.where(pm, g_sig_only, 0.0)
        num = float(np.linalg.norm((g - g_so)[pm]))
        den = float(np.linalg.norm(g_so[pm]))
        out["channel_split"] = {
            "grad_norm_sigma_only": den,
            "grad_norm_with_eps": float(np.linalg.norm(g[pm])),
            "eps_term_norm": num,
            "eps_term_relative_norm": num / max(den, 1e-30),
            "cosine_between_channels": float(
                np.sum(g[pm] * g_so[pm]) / max(np.linalg.norm(g[pm]) * den, 1e-30)),
        }
    del tr0

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
        # objective's absolute evaluation floor.
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


def stop_index_stability(case, v0, sigma_cells: float, eps_covary: bool,
                         eps: float = 1e-3) -> dict:
    """Does the J_phi argmin move under the probe perturbations?"""
    pm = case.part_mask
    to_map = ((lambda v: df.apply_filter(v, pm, sigma_cells)) if sigma_cells > 0
              else (lambda v: np.where(pm, v, 1.0)))
    base = so.optimal_stop(run_forward(case, to_map(v0), eps_covary), case)
    out = {"eps": float(eps), "base_index": int(base.index), "moved": False, "probes": {}}
    for pname, d, _c in _probe_dirs(pm, np.where(pm, 1.0, 0.0)):
        ip = so.optimal_stop(run_forward(case, to_map(v0 + eps * d), eps_covary), case).index
        im = so.optimal_stop(run_forward(case, to_map(v0 - eps * d), eps_covary), case).index
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

    base = run_forward(case, np.where(case.part_mask, v0, 1.0), True)
    st = so.optimal_stop(base, case)
    res = {"shape": shape, "config": str(cfg_path), "sigma_cells": float(sigma_cells),
           "channel": "permittivity co-varying: sigma AND eps_r both blend with "
                      "fill_frac*s, the production fgm_feedback.saturation_map_npz hook",
           "objective": "J_phi = sum over the WHOLE domain of (phi - chi_part)^2",
           "base": {"read_index": int(st.index), "read_time_s": st.time_s,
                    "J_phi": st.J, "at_horizon": st.at_horizon,
                    "n_outer": int(base.n_outer)},
           "layers": []}
    del base

    layers = (("E0_sigma_only_filtered_control", float(sigma_cells), False),
              ("E1_eps_unfiltered_fixed_read", 0.0, True),
              ("E2_eps_filtered_fixed_read", float(sigma_cells), True))
    for name, sig, cov in layers:
        r = gate_layer(case, name, ops, v0, st.index, sig, cov)
        res["layers"].append(r)
        extra = (f"smoothdir={r['probes']['smooth_random_direction']['best_rel_err']:.3e} "
                 if "smooth_random_direction" in r["probes"] else "")
        split = (f"eps_frac={r['channel_split']['eps_term_relative_norm']:.4f} "
                 if "channel_split" in r else "")
        print(f"{r['layer']:34s} J0={r['J0']:.6f} "
              f"maxcell={r['probes']['max_sensitivity_cell']['best_rel_err']:.3e} "
              f"randcell={r['probes']['random_cell']['best_rel_err']:.3e} "
              f"randdir={r['probes']['random_direction']['best_rel_err']:.3e} "
              + extra
              + f"graddir={r['probes']['gradient_direction']['best_rel_err']:.3e} "
              + split
              + f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
                f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    res["stop_index_stability"] = stop_index_stability(
        case, v0, float(sigma_cells), True)
    print(f"stop index stability (eps channel): base "
          f"{res['stop_index_stability']['base_index']}, "
          f"moved = {res['stop_index_stability']['moved']}", flush=True)

    eps_layers = [l for l in res["layers"] if l["eps_covary"]]
    res["ALL_EPS_GATES_PASS"] = all(l["PASS"] for l in eps_layers)
    res["ALL_EPS_GATES_PASS_SUBGRADIENT"] = all(l["PASS_subgradient"] for l in eps_layers)
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"ALL_EPS_GATES_PASS at 1e-6 = {res['ALL_EPS_GATES_PASS']}, at the 1e-5 "
          f"subgradient standard = {res['ALL_EPS_GATES_PASS_SUBGRADIENT']}, "
          f"wall {res['wall_s']:.1f} s")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else df.DEFAULT_SIGMA_CELLS)
