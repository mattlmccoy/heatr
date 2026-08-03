"""Finite-difference gate for the ROTATIONALLY-AVERAGED kernel gradient.

Nothing in the continuous-rotation campaign may be optimized before this
passes. Three layers, so a failure can be bisected instead of guessed at:

  R0  a single averaging angle at zero degrees, unfiltered. The kernel is then
      the ordinary forward and this layer re-gates the INHERITED gradient in
      the new code path. If R0 fails, the wiring is broken, not the rotation.
  R1  the full angle set, unfiltered. Adds the two rotation transposes and the
      per-angle EQS adjoint sum.
  R2  the full angle set WITH the physical-length design filter. THE gradient
      the solve uses.

Probes per layer are the campaign's standing set (maximum-sensitivity cell, a
fixed pseudo-random in-part cell, a rough random unit direction, a filtered
smooth direction on the filtered layer) plus the gradient direction, which is
the decision-relevant one and the one with the largest analytic derivative.

Read state. Fixed read index, taken as the argmin of J_phi on the base run, so
the envelope theorem removes any dt*/ds term. Whether that argmin moves under
the probes is measured and reported, never assumed.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_rot <shape> <out.json> [step_deg]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import design_filter as df, gradops
from . import library_solve as lib
from . import shape_objective as so
from .gate_rho import (EPSILONS, PASS_REL_ERR, SUBGRADIENT_PASS_REL_ERR,
                       _probe_dirs, default_v)
from .gate_ms import gradient_direction, ramp_population
from .pins import load_cfg
from .rot_kernel import AveragedKernel, quasistatic_numbers
from .rot_frame import averaging_angles

N_STEPS_GATE = 400          # enough to melt and turn J_phi; keeps the gate cheap


def gate_layer(kern: AveragedKernel, name: str, ops, v0: np.ndarray,
               read_index: int, sigma_cells: float) -> dict:
    case = kern.case0
    pm = case.part_mask
    filt = float(sigma_cells) > 0.0

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells) if filt else np.where(pm, v, 1.0)

    def J_of(v):
        tr = kern.forward(to_map(v), keep_checkpoints=False, n_steps=N_STEPS_GATE)
        i = min(int(read_index), tr.n_outer - 1)
        return so.shape_J_and_seed(tr.T_at_end(i), case)[0]

    tr0 = kern.forward(to_map(v0), keep_checkpoints=True, n_steps=N_STEPS_GATE)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    g_s = kern.gradient(to_map(v0), tr0, {i0: seed}, grad_ops=ops)
    g = df.filter_vjp(g_s, pm, sigma_cells) if filt else np.where(pm, g_s, 0.0)

    out = {"layer": name, "n_angles": kern.n_angles,
           "sigma_cells": float(sigma_cells), "J0": float(J0),
           "read_index": int(i0), "n_outer": int(tr0.n_outer),
           "grad_norm": float(np.linalg.norm(g[pm])),
           "grad_max_abs": float(np.max(np.abs(g[pm]))),
           "ramp_population": ramp_population(case, tr0.T_at_end(i0)),
           "probes": {}}
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


def stop_index_stability(kern, v0, sigma_cells: float, eps: float = 1e-3) -> dict:
    pm = kern.case0.part_mask
    to_map = ((lambda v: df.apply_filter(v, pm, sigma_cells)) if sigma_cells > 0
              else (lambda v: np.where(pm, v, 1.0)))

    def stop(v):
        return so.optimal_stop(
            kern.forward(to_map(v), n_steps=N_STEPS_GATE), kern.case0).index

    base = stop(v0)
    out = {"eps": float(eps), "base_index": int(base), "moved": False, "probes": {}}
    for pname, d, _c in _probe_dirs(pm, np.where(pm, 1.0, 0.0)):
        ip, im = stop(v0 + eps * d), stop(v0 - eps * d)
        out["probes"][pname] = {"plus": int(ip), "minus": int(im),
                                "moved": bool(ip != base or im != base)}
        out["moved"] = out["moved"] or out["probes"][pname]["moved"]
    return out


def main(shape: str, out_path: str, step_deg: float = 15.0,
         sigma_cells: float = df.DEFAULT_SIGMA_CELLS) -> dict:
    t0 = time.perf_counter()
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    angles = averaging_angles(float(step_deg))

    kern1 = AveragedKernel.build(cfg, angles=np.array([0.0]))
    kernM = AveragedKernel.build(cfg, angles=angles)
    case = kernM.case0
    ops = gradops.gradient_matrices(case.x, case.y)
    v0 = default_v(case)

    base = kernM.forward(np.where(case.part_mask, v0, 1.0), n_steps=N_STEPS_GATE)
    st = so.optimal_stop(base, case)
    res = {"shape": shape, "config": str(cfg_path),
           "objective": "J_phi = sum over the WHOLE domain of (phi - chi_part)^2",
           "kernel": "rotationally averaged, part frame",
           "angles_deg": [float(a) for a in angles], "n_angles": int(len(angles)),
           "step_deg": float(step_deg), "n_steps_gate": N_STEPS_GATE,
           "sigma_cells": float(sigma_cells),
           "base": {"read_index": int(st.index), "read_time_s": st.time_s,
                    "J_phi": st.J, "at_horizon": st.at_horizon,
                    "n_outer": int(base.n_outer)},
           "layers": []}

    for name, k, sig in (("R0_single_angle_zero_unfiltered", kern1, 0.0),
                         ("R1_full_angle_set_unfiltered", kernM, 0.0),
                         ("R2_full_angle_set_filtered", kernM, float(sigma_cells))):
        r = gate_layer(k, name, ops, v0, st.index, sig)
        res["layers"].append(r)
        extra = (f"smoothdir={r['probes']['smooth_random_direction']['best_rel_err']:.3e} "
                 if "smooth_random_direction" in r["probes"] else "")
        print(f"{r['layer']:34s} M={r['n_angles']:3d} J0={r['J0']:.6f} "
              f"maxcell={r['probes']['max_sensitivity_cell']['best_rel_err']:.3e} "
              f"randcell={r['probes']['random_cell']['best_rel_err']:.3e} "
              f"randdir={r['probes']['random_direction']['best_rel_err']:.3e} "
              + extra
              + f"graddir={r['probes']['gradient_direction']['best_rel_err']:.3e} "
              + f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
                f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    res["stop_index_stability"] = stop_index_stability(kernM, v0, float(sigma_cells))
    print(f"stop index stability: base {res['stop_index_stability']['base_index']}, "
          f"moved = {res['stop_index_stability']['moved']}", flush=True)

    res["quasistatic"] = quasistatic_numbers(case, rotation_period_s=float("nan"))
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
         float(sys.argv[3]) if len(sys.argv) > 3 else 15.0,
         float(sys.argv[4]) if len(sys.argv) > 4 else df.DEFAULT_SIGMA_CELLS)
