"""Phase C drivers: the composed-chain gradient gate (Task 2) and the solves.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_c_run --chain-gate

Every case parameter is READ from solve3d/results/phase_c_preregistration.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import (adjoint, design_chain as dc, forward as fwd, gate_fd,
                     gates, objective as obj)

RESULTS = Path(__file__).resolve().parent / "results"


def prereg() -> dict:
    return json.loads((RESULTS / "phase_c_preregistration.json").read_text())


# --------------------------------------------------------------------------- #
def design_point_raw(chain: dc.DesignChain, seed: int = 5) -> np.ndarray:
    """A smooth, non-degenerate raw design strictly inside the box.

    Strictly interior so no BOX clip enters the chain: the frozen
    parameterization is built so the only kinks are the physical ones (the melt
    clips and the objective hinges), and the gate design point keeps that true.
    """
    c = chain.centroids
    v = (0.70
         + 0.15 * np.sin(np.pi * c[:, 0] / 0.010) * np.cos(np.pi * c[:, 1] / 0.010)
         + 0.05 * np.sin(np.pi * c[:, 2] / 0.030))
    return np.clip(v, 0.05, 0.95)


def run_chain_gate(tc: adjoint.TransientCase, chain: dc.DesignChain,
                   betas=(0.0, 1.0), seed: int = 7,
                   objective_name: str = "asymmetric") -> dict:
    """Task 2's gate: the COMPOSED dJ/d(raw design) through filter (+ projection)
    and the full coupled forward, re-gated with the Phase B protocol BEFORE any
    solve runs."""
    tc.set_objective(objective_name)
    v0 = design_point_raw(chain)
    out = {"what": "Phase C layer C1: composed dJ/dv through the design chain "
                   "and the full Phase A/B coupled forward",
           "objective": objective_name,
           "kernel": chain.kernel_report(),
           "design_point": {"min": float(v0.min()), "max": float(v0.max()),
                            "strictly_interior": bool(v0.min() > 0 and v0.max() < 1)},
           "gates": {}, "transposes": {}}
    rng = np.random.default_rng(4)
    for beta in betas:
        b = float(beta)
        tr_res = gate_fd.transpose_residual(
            lambda d, _b=b: chain.design_jvp(v0, d, beta=_b),
            lambda g, _b=b: chain.design_vjp(v0, g, beta=_b),
            rng.standard_normal(chain.n_design), rng.standard_normal(chain.n_design))
        out["transposes"][f"beta_{b:g}"] = tr_res

        def J_raw(v, _b=b):
            return tc.J_of_design(chain.design_to_map(v, _b))

        t0 = time.perf_counter()
        s0 = chain.design_to_map(v0, b)
        tr = tc.forward(tc.design_to_sigma(s0))
        t_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        g_s, info = tc.gradient_design(s0, tr=tr)
        g_v = chain.design_vjp(v0, g_s, beta=b)
        t_grad = time.perf_counter() - t0
        g = gate_fd.run_probes(J_raw, v0, g_v, np.ones(g_v.shape, bool), seed=seed,
                               x_scale_direction=float(np.mean(np.abs(v0))))
        g["cost"] = {"wall_forward_s": t_fwd, "wall_gradient_s": t_grad,
                     "forward_equivalents": t_grad / t_fwd}
        g["J"] = float(J_raw(v0))
        out["gates"][f"beta_{b:g}"] = g
    gates.write_json("phase_c_chain_gate.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-gate", action="store_true")
    args = ap.parse_args()
    if args.chain_gate:
        from solve3d import transient_gate as TG
        tc = TG.build_case()
        ch = dc.DesignChain.build(tc)
        d = run_chain_gate(tc, ch)
        print(json.dumps({k: {p: v["probes"][p]["best_rel_err"]
                              for p in v["probes"]}
                          for k, v in d["gates"].items()}, indent=1))
        print(json.dumps({k: v["rel_err"] for k, v in d["transposes"].items()},
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
