"""Finite-difference gates for every gradient layer.

Central differences, epsilon swept over 1e-3 to 1e-7, two probe directions:
a random unit direction over the part cells and a single-cell probe at the
maximum-|gradient| cell. A gradient that is not FD-gated is presumed wrong.

Layers
------
L1  sigma_T at a FIXED outer step (the `Var_part(T)` objective of the
    assessment, written in campaign units)
L1b sigma_T at the heating-peak read state (a max over pre-melt steps, taken as
    a subgradient at the argmax)
L2  sigma_T at the interpolated melt-onset crossing, including the implicit
    function theorem term dt*/ds
L2clip  the same as L2 but with the per-substep temperature-step cap lowered
    until it BINDS, so the non-smooth clip subgradient is exercised rather than
    assumed dormant
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import adjoint, forward as fwd, gradops, objective as obj
from .pins import build_case, load_cfg

EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


@dataclass
class LayerSpec:
    name: str
    objective: Callable
    forward_kwargs: dict


def _make_forward(case, spec: LayerSpec):
    def run(s, checkpoints: bool):
        return fwd.forward(case, s, float32_sat=False,
                           keep_checkpoints=checkpoints, **spec.forward_kwargs)
    return run


def gate_layer(case, spec: LayerSpec, s0: np.ndarray, seed: int = 7) -> dict:
    run = _make_forward(case, spec)
    tr = run(s0, True)
    J0, seeds = spec.objective(tr, case)
    ops = gradops.gradient_matrices(case.x, case.y)
    g = adjoint.gradient(case, s0, tr, seeds, grad_ops=ops)

    pm = case.part_mask
    rng = np.random.default_rng(seed)
    d_rand = np.zeros_like(s0)
    v = rng.standard_normal(int(pm.sum()))
    v /= np.linalg.norm(v)
    d_rand[pm] = v

    gp = np.where(pm, np.abs(g), -np.inf)
    i_max = np.unravel_index(int(np.argmax(gp)), s0.shape)
    d_cell = np.zeros_like(s0)
    d_cell[i_max] = 1.0

    out = {
        "layer": spec.name,
        "J0": float(J0),
        "grad_norm": float(np.linalg.norm(g[pm])),
        "grad_max_abs": float(np.max(np.abs(g[pm]))),
        "max_grad_cell": [int(i_max[0]), int(i_max[1])],
        "n_outer_base": int(tr.n_outer),
        "probes": {},
    }
    for pname, d in (("random_direction", d_rand), ("single_cell", d_cell)):
        ana = float(np.sum(g * d))
        rows = []
        for eps in EPSILONS:
            jp = spec.objective(run(s0 + eps * d, False), case)[0]
            jm = spec.objective(run(s0 - eps * d, False), case)[0]
            fd = (jp - jm) / (2.0 * eps)
            rel = abs(fd - ana) / max(abs(ana), 1e-30)
            rows.append({"eps": eps, "fd": float(fd), "rel_err": float(rel)})
        best = min(rows, key=lambda r: r["rel_err"])
        out["probes"][pname] = {
            "analytic": ana,
            "sweep": rows,
            "best_rel_err": best["rel_err"],
            "best_eps": best["eps"],
            "PASS": bool(best["rel_err"] < 1e-6),
        }
    out["PASS"] = all(v["PASS"] for v in out["probes"].values())
    return out


def default_s(case) -> np.ndarray:
    """A non-uniform starting map so no symmetry can hide a sign error."""
    rng = np.random.default_rng(4242)
    ny, nx = case.part_mask.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    smooth = 0.90 + 0.18 * np.cos(1.7 * xx + 0.4) * np.sin(2.1 * yy + 0.9)
    # Outside the part the saturation is held at its nominal value 1, matching
    # the design-domain convention used by the optimization arms.
    s = np.ones((ny, nx))
    s[case.part_mask] = smooth[case.part_mask] + 0.02 * rng.standard_normal(int(case.part_mask.sum()))
    return s


def build_specs(case, s0) -> list[LayerSpec]:
    probe = fwd.forward(case, s0, stop_after_phi=0.90, stop_margin_steps=2)
    rs = obj.read_states(probe)
    n_fix = int(max(10, min(rs.heating_peak_index, probe.n_outer - 2)))
    specs = [
        LayerSpec("L1_fixed_horizon",
                  lambda tr, c, n=n_fix: obj.objective_fixed_horizon(tr, c, n),
                  {"stop_after_phi": None, "n_steps": n_fix + 1}),
        LayerSpec("L1b_heating_peak", obj.objective_heating_peak,
                  {"stop_after_phi": 0.90, "stop_margin_steps": 2}),
        LayerSpec("L2_melt_onset_ift", obj.objective_melt_onset_interp,
                  {"stop_after_phi": 0.90, "stop_margin_steps": 2}),
    ]
    return specs, {"fixed_step": n_fix, "heating_peak_index": rs.heating_peak_index,
                   "melt_onset_index": rs.melt_onset_index, "n_outer": probe.n_outer}


def main(cfg_path: str, out_path: str, clip_test: bool = True) -> dict:
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    s0 = default_s(case)
    specs, info = build_specs(case, s0)
    results = {"config": str(cfg_path), "info": info, "layers": []}
    for spec in specs:
        r = gate_layer(case, spec, s0)
        results["layers"].append(r)
        print(f"{r['layer']:24s} J0={r['J0']:.6f}  "
              f"rand={r['probes']['random_direction']['best_rel_err']:.3e}  "
              f"cell={r['probes']['single_cell']['best_rel_err']:.3e}  "
              f"{'PASS' if r['PASS'] else 'FAIL'}", flush=True)

    if clip_test:
        # Force the per-substep temperature-step cap to BIND, then re-gate.
        cfg2 = load_cfg(Path(cfg_path).resolve())
        base_cap = float(cfg2["thermal"]["max_deltaT_per_step_c"])
        case2 = build_case(cfg2)
        tr_probe = fwd.forward(case2, s0, stop_after_phi=0.90, stop_margin_steps=2)
        rs = obj.read_states(tr_probe)
        n_fix = int(max(10, min(rs.heating_peak_index, tr_probe.n_outer - 2)))
        for cap in (0.02, 0.01):
            cfg2["thermal"]["max_deltaT_per_step_c"] = cap
            c2 = build_case(cfg2)
            spec = LayerSpec(f"L2clip_cap{cap}_fixed_horizon",
                             lambda tr, c, n=n_fix: obj.objective_fixed_horizon(tr, c, n),
                             {"stop_after_phi": None, "n_steps": n_fix + 1})
            tr = fwd.forward(c2, s0, keep_checkpoints=False, stop_after_phi=None, n_steps=n_fix + 1)
            frac = tr.frac_dT_clipped_max
            r = gate_layer(c2, spec, s0)
            r["clip_fraction_max"] = frac
            r["cap_c"] = cap
            results["layers"].append(r)
            print(f"{r['layer']:24s} clipfrac={frac:.4f} "
                  f"rand={r['probes']['random_direction']['best_rel_err']:.3e}  "
                  f"cell={r['probes']['single_cell']['best_rel_err']:.3e}  "
                  f"{'PASS' if r['PASS'] else 'FAIL'}", flush=True)
        cfg2["thermal"]["max_deltaT_per_step_c"] = base_cap

    results["ALL_PASS"] = all(layer["PASS"] for layer in results["layers"])
    Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).resolve().write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
