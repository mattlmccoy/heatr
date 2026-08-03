"""Resumable finite-difference gate at one out-of-bounds price.

Identical layers, probes, epsilon sweep and pass rule to
`adjoint2d.gate_asym.main`, which owns all of the logic; the only difference is
that the result file is written after EVERY layer and an existing file is read
back so an interrupted gate resumes at the layer it lost instead of repeating
the ones it already paid for. This exists because a full four-layer gate is
about 45 minutes of single-threaded wall time and this pass lost one to a
process restart.

Run:
  ./.venv312/bin/python run_wout_gate.py <shape> <out.json> <floor> <w_out>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from adjoint2d import asym_objective as ao, design_filter as df
from adjoint2d import gate_asym as ga, gradops, library_solve as lib
from adjoint2d.pins import build_case, load_cfg


def main(shape: str, out_path: str, floor: float = 0.85, w_out: float = 2.0,
         w_in: float = 1.0, sigma_cells: float = df.DEFAULT_SIGMA_CELLS) -> dict:
    t0 = time.perf_counter()
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = lib.shape_config(shape)
    case = build_case(load_cfg(cfg_path))
    ops = gradops.gradient_matrices(case.x, case.y)
    v0 = ga.default_v(case)

    res: dict = ({} if not p.exists() else json.loads(p.read_text()))
    if "base" not in res:
        base = ga.run_forward(case, v0)
        st = ao.asym_stop(base, case, floor=floor, w_out=w_out, w_in=w_in)
        m0 = ao.region_metrics(base.T_at_end(st.index), base.rho_at_end(st.index),
                               case, floor=floor)
        res = {"shape": shape, "config": str(cfg_path),
               "floor_rho_rel": float(floor), "w_out": float(w_out),
               "w_in": float(w_in), "sigma_cells": float(sigma_cells),
               "base": {"read_index": st.index, "read_time_s": st.time_s,
                        "J_asym": st.J, "J_out": st.J_out, "J_in": st.J_in,
                        "at_horizon": st.at_horizon,
                        "stop_is_first_step": st.stop_is_first_step,
                        "in_term_dead": st.in_term_dead,
                        "flat_onset_index": st.flat_onset_index,
                        "flat_onset_gap_steps": st.flat_onset_gap_steps,
                        "n_outer": base.n_outer, **m0},
               "layers": [], "wall_s": 0.0}
        del base
        p.write_text(json.dumps(res, indent=2, default=float))
    b = res["base"]
    print(f"[{shape} w={w_out}] base read index {b['read_index']} at "
          f"{b['read_time_s']:.0f} s, J_asym {b['J_asym']:.5f} = out "
          f"{b['J_out']:.5f} + in {b['J_in']:.5f}, hinge active "
          f"{b['hinge_active_frac']:.3f}, mean rho_rel "
          f"{b['mean_rho_rel_part']:.4f}", flush=True)

    layers = (
        ("A0_out_only", w_out, 0.0, 0.0, ga.ISOLATION_PROBES),
        ("A0_in_only", 0.0, w_in, 0.0, ga.ISOLATION_PROBES),
        ("A1_combined", w_out, w_in, 0.0, ga.FULL_PROBES),
        ("A2_combined_filtered", w_out, w_in, float(sigma_cells), ga.FULL_PROBES),
    )
    done = {l["layer"] for l in res["layers"]}
    for name, wo, wi, sig, want in layers:
        if name in done:
            print(f"[{shape} w={w_out}] {name:22s} already stored, skipped",
                  flush=True)
            continue
        r = ga.gate(case, name, ops, v0, b["read_index"], floor, wo, wi, sig, want)
        res["layers"].append(r)
        res["wall_s"] = float(res.get("wall_s", 0.0)) + (time.perf_counter() - t0)
        t0 = time.perf_counter()
        p.write_text(json.dumps(res, indent=2, default=float))
        bits = " ".join(f"{k}={v['best_rel_err']:.2e}" for k, v in r["probes"].items())
        print(f"[{shape} w={w_out}] {name:22s} J0={r['J0']:.6f} {bits} "
              f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
              f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    if "read_index_stability" not in res:
        res["read_index_stability"] = ga.read_index_stability(
            case, v0, floor, w_out, w_in, 0.0)
    res["ALL_GATES_PASS"] = all(l["PASS"] for l in res["layers"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = all(l["PASS_subgradient"]
                                            for l in res["layers"])
    res["wall_s"] = float(res.get("wall_s", 0.0)) + (time.perf_counter() - t0)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape} w={w_out}] read index stability moved = "
          f"{res['read_index_stability']['moved']}; ALL_GATES_PASS at 1e-6 = "
          f"{res['ALL_GATES_PASS']}, at the 1e-5 subgradient standard = "
          f"{res['ALL_GATES_PASS_SUBGRADIENT']}, wall {res['wall_s']:.1f} s",
          flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.85,
         float(sys.argv[4]) if len(sys.argv) > 4 else 2.0)
