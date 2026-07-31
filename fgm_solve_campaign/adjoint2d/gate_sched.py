"""Finite-difference gate for the temporal power-schedule gradient dJ/dp_k.

Central differences, epsilon swept 1e-3 to 1e-8, on the PRODUCTION horizon and
the production shape, at a non-trivial schedule (not all ones), for three
probes:

  max_sensitivity_segment   argmax |dJ/dp_k|
  random_segment            a fixed pseudo-random segment (seed 11)
  random_direction          a random unit direction over all segments

Two layers, mirroring `gate_shape.py`:

  P1  dJ/dp at a FIXED stop index. The pure partial derivative.
  P2  dJ*/dp with t_stop = argmin over the arm's own trajectory of J. If the
      envelope argument holds for the temporal actuator as it does for the
      dopant map, P2 equals P1 evaluated at the argmin with no dt*/dp term.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_sched <shape> <out.json> [n_seg] [horizon]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import adjoint, forward as fwd, gradops, schedule as sch
from . import shape_objective as so
from .library_solve import shape_config
from .pins import build_case, load_cfg

EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
PASS_TOL = 1e-6


def probe_schedule(n_seg: int) -> np.ndarray:
    """A non-trivial schedule to gate at: a slow ramp with one dip.

    Gating at p identically 1 would hide any error that is proportional to
    (p - 1), so the gate point is deliberately away from nominal.
    """
    rng = np.random.default_rng(2024)
    p = 0.85 + 0.25 * np.cos(np.linspace(0.0, 2.4, n_seg))
    p += 0.03 * rng.standard_normal(n_seg)
    p[max(n_seg // 3, 0)] = 0.35
    return np.clip(p, 0.0, 1.5)


def probe_map(case) -> np.ndarray:
    """A non-uniform dopant map, so the gate is not taken at s = 1 either."""
    ny, nx = case.part_mask.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    smooth = 0.90 + 0.18 * np.cos(1.7 * xx + 0.4) * np.sin(2.1 * yy + 0.9)
    s = np.ones((ny, nx))
    s[case.part_mask] = smooth[case.part_mask]
    return s


def run_forward(case, s, p, n_seg, horizon, checkpoints=False):
    return fwd.forward(case, s, keep_checkpoints=checkpoints, stop_after_phi=None,
                       shape_stop_patience=None, n_steps=horizon,
                       p_seg=p, n_seg=n_seg, p_horizon=horizon)


def gate(case, ops, s, p0, n_seg, horizon, layer: str, fixed_index: int | None) -> dict:
    tr = run_forward(case, s, p0, n_seg, horizon, checkpoints=True)
    if fixed_index is None:
        st = so.optimal_stop(tr, case)
        index = st.index
    else:
        index = min(fixed_index, tr.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr.T_at_end(index), case)
    _gs, gp = adjoint.gradient(case, s, tr, {index: seed}, grad_ops=ops,
                               with_schedule=True)

    def J_of(p):
        t = run_forward(case, s, p, n_seg, horizon)
        if fixed_index is None:
            return so.optimal_stop(t, case).J
        return so.shape_J_and_seed(t.T_at_end(min(fixed_index, t.n_outer - 1)), case)[0]

    rng = np.random.default_rng(11)
    k_max = int(np.argmax(np.abs(gp)))
    k_rand = int(rng.integers(0, n_seg))
    d_rand = rng.standard_normal(n_seg)
    d_rand /= np.linalg.norm(d_rand)

    dirs = {}
    e_max = np.zeros(n_seg); e_max[k_max] = 1.0
    e_rnd = np.zeros(n_seg); e_rnd[k_rand] = 1.0
    dirs["max_sensitivity_segment"] = (e_max, {"segment": k_max})
    dirs["random_segment"] = (e_rnd, {"segment": k_rand})
    dirs["random_direction"] = (d_rand, {})

    out = {"layer": layer, "J0": float(J0), "stop_index": int(index),
           "n_outer": int(tr.n_outer), "dJdp": [float(v) for v in gp],
           "probes": {}}
    for name, (d, meta) in dirs.items():
        ana = float(np.dot(gp, d))
        rows = []
        for eps in EPSILONS:
            fd = (J_of(p0 + eps * d) - J_of(p0 - eps * d)) / (2.0 * eps)
            rows.append({"eps": eps, "fd": float(fd),
                         "rel_err": abs(fd - ana) / max(abs(ana), 1e-30)})
        best = min(rows, key=lambda r: r["rel_err"])
        out["probes"][name] = dict(meta, analytic=ana, sweep=rows,
                                   best_rel_err=best["rel_err"], best_eps=best["eps"],
                                   PASS=bool(best["rel_err"] < PASS_TOL))
    out["PASS"] = all(v["PASS"] for v in out["probes"].values())
    return out


def main(shape: str, out_path: str, n_seg: int = 16, horizon: int | None = None) -> dict:
    case = build_case(load_cfg(shape_config(shape)))
    horizon = int(case.pins.n_steps if horizon is None else horizon)
    ops = gradops.gradient_matrices(case.x, case.y)
    s = probe_map(case)
    p0 = probe_schedule(n_seg)

    base = run_forward(case, s, p0, n_seg, horizon)
    st = so.optimal_stop(base, case)
    res = {"shape": shape, "n_seg": n_seg, "horizon": horizon,
           "p_probe": [float(v) for v in p0],
           "duty_cycle": sch.duty_cycle(p0, horizon, n_seg),
           "base": {"t_stop_index": st.index, "t_stop_s": st.time_s, "J": st.J,
                    "at_horizon": bool(st.at_horizon), "n_outer": base.n_outer},
           "layers": []}
    for layer, fixed in (("P1_fixed_stop", st.index), ("P2_envelope_stop", None)):
        r = gate(case, ops, s, p0, n_seg, horizon, layer, fixed)
        res["layers"].append(r)
        pr = r["probes"]
        print(f"{layer:20s} J0={r['J0']:.6f} "
              f"maxseg={pr['max_sensitivity_segment']['best_rel_err']:.3e} "
              f"randseg={pr['random_segment']['best_rel_err']:.3e} "
              f"randdir={pr['random_direction']['best_rel_err']:.3e} "
              f"{'PASS' if r['PASS'] else 'FAIL'}", flush=True)
    a1 = res["layers"][0]["probes"]["max_sensitivity_segment"]["analytic"]
    a2 = res["layers"][1]["probes"]["max_sensitivity_segment"]["analytic"]
    res["fixed_vs_envelope_rel_diff"] = abs(a1 - a2) / max(abs(a1), 1e-30)
    res["ALL_PASS"] = all(x["PASS"] for x in res["layers"])
    Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).resolve().write_text(json.dumps(res, indent=2))
    print(f"fixed vs envelope max-segment gradient relative difference: "
          f"{res['fixed_vs_envelope_rel_diff']:.3e}")
    return res


if __name__ == "__main__":
    _shape = sys.argv[1]
    _out = sys.argv[2]
    _nseg = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    _hor = int(sys.argv[4]) if len(sys.argv) > 4 else None
    main(_shape, _out, _nseg, _hor)
