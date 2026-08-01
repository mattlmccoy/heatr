"""Finite-difference gate for the ASYMMETRIC DWELL gradients.

Nothing in the dwell campaign may be optimized before this passes. Four layers,
so a failure localizes instead of being guessed at:

  D0  the MAP gradient at UNIFORM weights, unfiltered. The weighted kernel is
      then the inherited `AveragedKernel`, so this layer re-gates the already
      gated gradient inside the new code path. If D0 fails the wiring is
      broken, not the dwell.
  D1  the DWELL gradient dJ/dw at NON-UNIFORM weights. The new object. Probe
      directions are simplex TANGENTS (they sum to zero), because a
      single-coordinate move off the simplex is not a legal design move and
      the forward refuses it.
  D2  the same through the SOFTMAX, dJ/dz. This is the gradient L-BFGS-B
      actually receives, and its probes are unconstrained coordinates.
  D3  the MAP gradient at non-uniform weights WITH the 1.0 mm physical-length
      design filter. The other gradient the solve uses.

Read state. Fixed read index, taken as the argmin of J_phi on the base run, so
the envelope theorem removes any dt*/d(design) term. Whether that argmin moves
under the probes is measured and reported, never assumed.

Pass standard: 1e-6 preferred, 1e-5 is the campaign's documented subgradient
standard (`FROZEN_CONVENTIONS_2D.md` Section 5 item 5).

Run:
  ./.venv312/bin/python -m adjoint2d.gate_dwell <shape> <out.json>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import design_filter as df, dwell, gradops
from . import library_solve as lib
from . import shape_objective as so
from . import topopt
from .dwell_kernel import DwellKernel
from .gate_ms import gradient_direction
from .gate_rho import (EPSILONS, PASS_REL_ERR, SUBGRADIENT_PASS_REL_ERR,
                       _probe_dirs, default_v)
from .pins import load_cfg

# The gate horizon must be long enough that the objective has TURNED at the
# gate design point, otherwise nothing is on the melt ramp, the analytic
# gradient is identically zero and every relative error is a meaningless 0/0.
# MEASURED at the design point of `gate_rho.default_v` with uniform dwells:
# the cross turns by step 400, the T_shape not until step 1140 and the L_shape
# 1060, so the horizon is per shape and is recorded in the result file.
N_STEPS_GATE = 400
CANDIDATE_ANGLES = np.arange(8, dtype=float) * 45.0
W_SEED = 909


def default_weights(n: int, seed: int = W_SEED) -> np.ndarray:
    """A deliberately NON-uniform, non-degenerate dwell vector."""
    rng = np.random.default_rng(seed)
    return dwell.softmax_weights(0.8 * rng.standard_normal(n))


def _simplex_probe_dirs(gw: np.ndarray, seed: int = 7):
    """Probe directions that stay on the simplex: every one sums to zero."""
    k = gw.size
    rng = np.random.default_rng(seed)

    def tangent_unit(i):
        d = np.full(k, -1.0 / (k - 1))
        d[i] = 1.0
        return d / np.linalg.norm(d)

    i_max = int(np.argmax(np.abs(gw)))
    i_rnd = int(rng.integers(k))
    u = rng.standard_normal(k)
    u = u - u.mean()
    return (("max_sensitivity_dwell", tangent_unit(i_max), i_max),
            ("random_dwell", tangent_unit(i_rnd), i_rnd),
            ("random_direction", u / np.linalg.norm(u), None))


def _sweep(J_of, base_point, d, ana) -> dict:
    rows = []
    for eps in EPSILONS:
        fd = (J_of(base_point + eps * d) - J_of(base_point - eps * d)) / (2.0 * eps)
        rows.append({"eps": eps, "fd": float(fd),
                     "rel_err": abs(fd - ana) / max(abs(ana), 1e-30),
                     "abs_err": abs(fd - ana)})
    best = min(rows, key=lambda r: r["rel_err"])
    tail = sorted(rows, key=lambda r: r["eps"])[:2]
    return {"analytic": float(ana), "sweep": rows,
            "best_rel_err": best["rel_err"], "best_abs_err": best["abs_err"],
            "best_eps": best["eps"],
            "J_eval_floor_estimate": float(
                np.median([2.0 * r["eps"] * r["abs_err"] for r in tail])),
            "PASS": bool(best["rel_err"] < PASS_REL_ERR),
            "PASS_subgradient": bool(best["rel_err"] < SUBGRADIENT_PASS_REL_ERR)}


def _finish(out: dict) -> dict:
    out["n_probes"] = len(out["probes"])
    out["n_probes_pass_1e-6"] = sum(p["PASS"] for p in out["probes"].values())
    out["n_probes_pass_1e-5"] = sum(p["PASS_subgradient"] for p in out["probes"].values())
    out["PASS"] = out["n_probes_pass_1e-6"] == out["n_probes"]
    out["PASS_subgradient"] = out["n_probes_pass_1e-5"] == out["n_probes"]
    return out


def _log(r: dict) -> None:
    bits = " ".join(f"{k}={v['best_rel_err']:.3e}" for k, v in r["probes"].items())
    print(f"{r['layer']:38s} J0={r['J0']:.6f} {bits} "
          f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
          f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)


# ---------------------------------------------------------------------------
# the map layers, D0 and D3
# ---------------------------------------------------------------------------

def gate_map(kern: DwellKernel, name: str, ops, v0, w0, read_index: int,
             sigma_cells: float) -> dict:
    case = kern.case0
    pm = case.part_mask
    filt = float(sigma_cells) > 0.0
    kern.set_weights(w0)

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells) if filt else np.where(pm, v, 1.0)

    def J_of(v):
        tr = kern.forward(to_map(v), keep_checkpoints=False, n_steps=N_STEPS_GATE)
        return so.shape_J_and_seed(tr.T_at_end(min(read_index, tr.n_outer - 1)), case)[0]

    tr0 = kern.forward(to_map(v0), keep_checkpoints=True, n_steps=N_STEPS_GATE)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    g_s, _gw = kern.both_gradients(to_map(v0), tr0, {i0: seed}, grad_ops=ops)
    g = df.filter_vjp(g_s, pm, sigma_cells) if filt else np.where(pm, g_s, 0.0)

    out = {"layer": name, "variable": "dopant map v", "J0": float(J0),
           "read_index": int(i0), "sigma_cells": float(sigma_cells),
           "weights": [float(x) for x in kern._w()],
           "grad_norm": float(np.linalg.norm(g[pm])), "probes": {}}
    probes = list(_probe_dirs(pm, g, sigma_cells=sigma_cells))
    probes.append(("gradient_direction", gradient_direction(pm, g), None))
    for pname, d, cell in probes:
        out["probes"][pname] = _sweep(J_of, v0, d, float(np.sum(g * d)))
        out["probes"][pname]["cell"] = None if cell is None else [int(c) for c in cell]
    return _finish(out)


# ---------------------------------------------------------------------------
# the dwell layers, D1 and D2
# ---------------------------------------------------------------------------

def gate_dwell_weights(kern: DwellKernel, name: str, s0, w0, read_index: int) -> dict:
    case = kern.case0

    def J_of(w):
        kern.set_weights(w)
        tr = kern.forward(s0, keep_checkpoints=False, n_steps=N_STEPS_GATE)
        return so.shape_J_and_seed(tr.T_at_end(min(read_index, tr.n_outer - 1)), case)[0]

    kern.set_weights(w0)
    tr0 = kern.forward(s0, keep_checkpoints=True, n_steps=N_STEPS_GATE)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    gw = kern.weight_gradient(tr0, {i0: seed})

    out = {"layer": name, "variable": "dwell fractions w", "J0": float(J0),
           "read_index": int(i0), "weights": [float(x) for x in w0],
           "dJ_dw": [float(x) for x in gw],
           "grad_norm": float(np.linalg.norm(gw)), "probes": {}}
    probes = list(_simplex_probe_dirs(gw))
    gt = gw - gw.mean()
    probes = probes + [("gradient_direction", gt / max(np.linalg.norm(gt), 1e-30), None)]
    for pname, d, idx in probes:
        out["probes"][pname] = _sweep(J_of, w0, d, float(np.dot(gw, d)))
        out["probes"][pname]["position_index"] = idx
    return _finish(out)


def gate_dwell_logits(kern: DwellKernel, name: str, s0, z0, read_index: int) -> dict:
    case = kern.case0

    def J_of(z):
        kern.set_weights(dwell.softmax_weights(z))
        tr = kern.forward(s0, keep_checkpoints=False, n_steps=N_STEPS_GATE)
        return so.shape_J_and_seed(tr.T_at_end(min(read_index, tr.n_outer - 1)), case)[0]

    w0 = dwell.softmax_weights(z0)
    kern.set_weights(w0)
    tr0 = kern.forward(s0, keep_checkpoints=True, n_steps=N_STEPS_GATE)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    gz = dwell.softmax_vjp(w0, kern.weight_gradient(tr0, {i0: seed}))

    k = z0.size
    rng = np.random.default_rng(21)
    e_max = np.zeros(k)
    e_max[int(np.argmax(np.abs(gz)))] = 1.0
    e_rnd = np.zeros(k)
    e_rnd[int(rng.integers(k))] = 1.0
    u = rng.standard_normal(k)
    probes = (("max_sensitivity_logit", e_max), ("random_logit", e_rnd),
              ("random_direction", u / np.linalg.norm(u)),
              ("gradient_direction", gz / max(np.linalg.norm(gz), 1e-30)))
    out = {"layer": name, "variable": "softmax logits z", "J0": float(J0),
           "read_index": int(i0), "weights": [float(x) for x in w0],
           "dJ_dz": [float(x) for x in gz],
           "grad_norm": float(np.linalg.norm(gz)), "probes": {}}
    for pname, d in probes:
        out["probes"][pname] = _sweep(J_of, z0, d, float(np.dot(gz, d)))
    return _finish(out)


# ---------------------------------------------------------------------------

def stop_index_stability(kern, s0, w0, eps: float = 1e-3) -> dict:
    case = kern.case0

    def stop(w):
        kern.set_weights(w)
        return so.optimal_stop(
            kern.forward(s0, n_steps=N_STEPS_GATE), case).index

    base = stop(w0)
    out = {"eps": float(eps), "base_index": int(base), "moved": False, "probes": {}}
    for pname, d, _i in _simplex_probe_dirs(np.ones(w0.size)):
        wp, wm = w0 + eps * d, w0 - eps * d
        if wp.min() < 0 or wm.min() < 0:
            out["probes"][pname] = {"skipped": "probe leaves the simplex"}
            continue
        ip, im = stop(wp), stop(wm)
        out["probes"][pname] = {"plus": int(ip), "minus": int(im),
                                "moved": bool(ip != base or im != base)}
        out["moved"] = out["moved"] or out["probes"][pname]["moved"]
    return out


def main(shape: str, out_path: str, n_steps: int = N_STEPS_GATE) -> dict:
    t0 = time.perf_counter()
    global N_STEPS_GATE
    N_STEPS_GATE = int(n_steps)
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    kern = DwellKernel.build(cfg, angles=CANDIDATE_ANGLES)
    case = kern.case0
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)

    v0 = default_v(case)
    k = int(len(CANDIDATE_ANGLES))
    w_uniform = np.full(k, 1.0 / k)
    w0 = default_weights(k)
    z0 = dwell.softmax_logits_for(w0)
    s0 = df.apply_filter(v0, pm, sigma_cells)

    kern.set_weights(w_uniform)
    base = kern.forward(np.where(pm, v0, 1.0), n_steps=N_STEPS_GATE)
    st = so.optimal_stop(base, case)

    res = {"shape": shape, "config": str(cfg_path),
           "objective": "J_phi = sum over the WHOLE domain of (phi - chi_part)^2",
           "kernel": "dwell-weighted angle average, part frame",
           "angles_deg": [float(a) for a in CANDIDATE_ANGLES], "n_angles": k,
           "n_steps_gate": N_STEPS_GATE,
           "filter_radius_m": topopt.FILTER_RADIUS_M,
           "sigma_cells": float(sigma_cells),
           "weights_uniform": [float(x) for x in w_uniform],
           "weights_gate_point": [float(x) for x in w0],
           "base": {"read_index": int(st.index), "read_time_s": st.time_s,
                    "J_phi": st.J, "at_horizon": st.at_horizon},
           "layers": []}

    r = gate_map(kern, "D0_map_uniform_weights_unfiltered", ops, v0, w_uniform,
                 st.index, 0.0)
    res["layers"].append(r)
    _log(r)
    r = gate_dwell_weights(kern, "D1_dwell_weights_nonuniform", s0, w0, st.index)
    res["layers"].append(r)
    _log(r)
    r = gate_dwell_logits(kern, "D2_dwell_softmax_logits", s0, z0, st.index)
    res["layers"].append(r)
    _log(r)
    r = gate_map(kern, "D3_map_nonuniform_weights_filtered", ops, v0, w0,
                 st.index, float(sigma_cells))
    res["layers"].append(r)
    _log(r)

    res["stop_index_stability"] = stop_index_stability(kern, s0, w0)
    print(f"stop index stability: base {res['stop_index_stability']['base_index']}, "
          f"moved = {res['stop_index_stability']['moved']}", flush=True)

    degenerate = [l["layer"] for l in res["layers"] if l["grad_norm"] == 0.0]
    res["degenerate_layers"] = degenerate
    if degenerate:
        print(f"[{shape}] DEGENERATE GATE: zero analytic gradient on "
              f"{degenerate}; the horizon is too short for the objective to "
              f"have turned at this design point.", flush=True)
    res["ALL_GATES_PASS"] = (not degenerate) and all(l["PASS"] for l in res["layers"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = (not degenerate) and all(
        l["PASS_subgradient"] for l in res["layers"])
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] ALL_GATES_PASS at 1e-6 = {res['ALL_GATES_PASS']}, at the 1e-5 "
          f"subgradient standard = {res['ALL_GATES_PASS_SUBGRADIENT']}, "
          f"wall {res['wall_s']:.1f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else N_STEPS_GATE)
