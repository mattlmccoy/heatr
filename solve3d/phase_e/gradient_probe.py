"""Recompute ONE gradient dJ/ds at a delivered map, for display and for a
reproduction check.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.gradient_probe --shape pyramid

The solve stores the map it converged to but not the gradient field at it, so a
figure that wants to show "one adjoint sweep gives dJ/ds at every design cell"
has to recompute it. That is one forward plus one adjoint, the same cost as one
optimizer evaluation.

BUILT-IN REPRODUCTION CHECK. This re-runs the identical forward the scorer ran,
so the objective it recomputes must equal the stored `J_asymmetric` of that arm
to round-off. The script ASSERTS that (relative tolerance 1e-9). If the assert
fires, the delivered map, the mesh or the objective settings have drifted and
the gradient field is not the one that belongs to the reported result.

The gradient machinery itself is not re-verified here; it is the Phase B
FD-gated transient adjoint, unchanged (solve3d/results/phase_b_transient_gate.json).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
J_REPRO_RTOL = 1.0e-9


def probe(shape: str, arm: str = "solve_filter_only") -> dict:
    from solve3d import design_chain as dc
    from solve3d.phase_e import run as R

    doc = json.loads((RESULTS / f"phase_e_{shape}.json").read_text())
    rec = doc["arms"][arm]
    z = np.load(RESULTS / f"map_{shape}_{arm}.npz")
    v = np.asarray(z["v_raw"], float)
    beta = float(rec.get("beta", 0.0) or 0.0)

    tc = R.build_case(shape)
    import dolfinx
    cen = np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]
    chain = dc.DesignChain(cen, tc.eqs.vol[tc.eqs.part], 1.0e-3, [0.0])
    tc.set_objective(rec["objective_optimized"], rec.get("w_ratio"))

    t0 = time.perf_counter()
    s = chain.design_to_map(v, beta)
    tr = tc.forward(tc.design_to_sigma(s))
    Jt = tc.J_trajectory(tr)
    k = int(np.argmin(Jt))
    J = float(Jt[k])
    g_s, _ = tc.gradient_design(s, tr=tr, read_step=k,
                                checkpoint_interval=R.CHECKPOINT_INTERVAL)
    wall = time.perf_counter() - t0

    J_stored = float(rec["J_asymmetric"])
    rel = abs(J - J_stored) / abs(J_stored)
    assert rel <= J_REPRO_RTOL, (
        f"recomputed J {J:.12e} does not reproduce the stored "
        f"{J_stored:.12e} (rel {rel:.3e} > {J_REPRO_RTOL:.0e}): the gradient "
        f"field would not belong to the reported result")

    out = {"shape": shape, "arm": arm, "J": J, "J_stored": J_stored,
           "J_reproduction_rel": rel, "argmin_step": k,
           "t_stop_s": float(k * tc.p.dt_s), "wall_s": wall,
           "n_design": int(g_s.size), "grad_norm": float(np.linalg.norm(g_s)),
           "grad_min": float(g_s.min()), "grad_max": float(g_s.max()),
           "grad_frac_negative": float((g_s < 0).mean()),
           "note": "dJ/ds on the design cells, one adjoint sweep, "
                   "Phase B FD-gated gradient"}
    np.savez_compressed(RESULTS / f"grad_{shape}_{arm}.npz",
                        g_s=g_s, s_map=s, centroids=cen,
                        volumes=tc.eqs.vol[tc.eqs.part])
    (RESULTS / f"grad_{shape}_{arm}.json").write_text(
        json.dumps(out, indent=1, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--arm", default="solve_filter_only")
    a = ap.parse_args()
    print(json.dumps(probe(a.shape, a.arm), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
