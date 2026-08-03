"""Finite-difference gates for the shape-fidelity gradient.

Two layers, both central differences with epsilon swept 1e-3 to 1e-7, a random
unit direction over the part cells and a single-cell probe at the
maximum-|gradient| cell.

  S1  dJ/ds at a FIXED stop index. The pure partial derivative.
  S2  dJ*/ds where J* = min over t of J(t). If the envelope argument holds,
      the S2 gradient is the S1 gradient evaluated at the argmin, with NO
      dt*/ds term. S2 gating to the same tolerance as S1 is the numerical
      verification of that claim, which is why both are run.

Run:  ./.venv312/bin/python -m adjoint2d.gate_shape <config.yaml> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import adjoint, forward as fwd, gradops, shape_objective as so
from .pins import build_case, load_cfg

EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
PATIENCE = 250


def default_s(case) -> np.ndarray:
    rng = np.random.default_rng(4242)
    ny, nx = case.part_mask.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    smooth = 0.90 + 0.18 * np.cos(1.7 * xx + 0.4) * np.sin(2.1 * yy + 0.9)
    s = np.ones((ny, nx))
    s[case.part_mask] = smooth[case.part_mask] + 0.02 * rng.standard_normal(int(case.part_mask.sum()))
    return s


def run_forward(case, s, checkpoints=False):
    return fwd.forward(case, s, keep_checkpoints=checkpoints, stop_after_phi=None,
                       shape_stop_patience=PATIENCE)


def objective_fixed(tr, case, index: int):
    idx = min(index, tr.n_outer - 1)
    J, g = so.shape_J_and_seed(tr.T_at_end(idx), case)
    return J, {idx: g}


def objective_envelope(tr, case):
    st = so.optimal_stop(tr, case)
    J, g = so.shape_J_and_seed(tr.T_at_end(st.index), case)
    return J, {st.index: g}


def gate(case, name, objective, s0, ops, seed=7) -> dict:
    tr = run_forward(case, s0, True)
    J0, seeds = objective(tr, case)
    g = adjoint.gradient(case, s0, tr, seeds, grad_ops=ops)
    pm = case.part_mask
    rng = np.random.default_rng(seed)
    d_rand = np.zeros_like(s0)
    v = rng.standard_normal(int(pm.sum()))
    d_rand[pm] = v / np.linalg.norm(v)
    gp = np.where(pm, np.abs(g), -np.inf)
    i_max = np.unravel_index(int(np.argmax(gp)), s0.shape)
    d_cell = np.zeros_like(s0)
    d_cell[i_max] = 1.0

    out = {"layer": name, "J0": float(J0), "grad_max_abs": float(np.max(np.abs(g[pm]))),
           "grad_norm": float(np.linalg.norm(g[pm])), "probes": {}}
    for pname, d in (("random_direction", d_rand), ("single_cell", d_cell)):
        ana = float(np.sum(g * d))
        rows = []
        for eps in EPSILONS:
            jp = objective(run_forward(case, s0 + eps * d), case)[0]
            jm = objective(run_forward(case, s0 - eps * d), case)[0]
            fd = (jp - jm) / (2.0 * eps)
            rows.append({"eps": eps, "fd": float(fd),
                         "rel_err": abs(fd - ana) / max(abs(ana), 1e-30)})
        best = min(rows, key=lambda r: r["rel_err"])
        out["probes"][pname] = {"analytic": ana, "sweep": rows,
                                "best_rel_err": best["rel_err"], "best_eps": best["eps"],
                                "PASS": bool(best["rel_err"] < 1e-6)}
    out["PASS"] = all(v["PASS"] for v in out["probes"].values())
    return out


def main(cfg_path: str, out_path: str) -> dict:
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    ops = gradops.gradient_matrices(case.x, case.y)
    s0 = default_s(case)
    base = run_forward(case, s0, False)
    st = so.optimal_stop(base, case)
    res = {"config": str(cfg_path),
           "base": {"t_stop_index": st.index, "t_stop_s": st.time_s, "J": st.J,
                    "at_horizon": st.at_horizon, "n_outer": base.n_outer},
           "layers": []}
    for name, obj_fn in (("S1_fixed_stop", lambda tr, c, i=st.index: objective_fixed(tr, c, i)),
                         ("S2_envelope_stop", objective_envelope)):
        r = gate(case, name, obj_fn, s0, ops)
        res["layers"].append(r)
        print(f"{r['layer']:20s} J0={r['J0']:.6f} "
              f"rand={r['probes']['random_direction']['best_rel_err']:.3e} "
              f"cell={r['probes']['single_cell']['best_rel_err']:.3e} "
              f"{'PASS' if r['PASS'] else 'FAIL'}", flush=True)
    # Do the two gradients agree? If they do, the envelope argument holds and
    # there is no dt*/ds term to add.
    g1 = res["layers"][0]["probes"]["single_cell"]["analytic"]
    g2 = res["layers"][1]["probes"]["single_cell"]["analytic"]
    res["fixed_vs_envelope_rel_diff"] = abs(g1 - g2) / max(abs(g1), 1e-30)
    res["ALL_PASS"] = all(layer["PASS"] for layer in res["layers"])
    Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).resolve().write_text(json.dumps(res, indent=2))
    print("fixed vs envelope single-cell gradient relative difference: "
          f"{res['fixed_vs_envelope_rel_diff']:.3e}")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
