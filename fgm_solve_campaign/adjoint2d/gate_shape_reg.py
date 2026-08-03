"""Is the shape-fidelity gate limited by the phase-ramp clip?

The objective reads melt fraction directly, so every cell pinned at phi = 0
(cold bed) or phi = 1 (fully melted core) has zero sensitivity, and its pinning
status flips discretely as the design variable moves. Widening the
phase-change regularizer dt_pc_c shrinks the pinned fraction. If the gate
improves with it, the clip is the cause and the gradient is a subgradient at the
production width rather than wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import adjoint, forward as fwd, gradops, shape_objective as so
from .gate_shape import default_s, run_forward
from .pins import build_case, load_cfg


def main(cfg_path: str, out_path: str, widths=(10.0, 20.0, 40.0)) -> dict:
    rows = []
    for dtpc in widths:
        cfg = load_cfg(Path(cfg_path).resolve())
        cfg["thermal"]["phase_change"]["dt_pc_c"] = float(dtpc)
        case = build_case(cfg)
        ops = gradops.gradient_matrices(case.x, case.y)
        s0 = default_s(case)
        tr = run_forward(case, s0, True)
        st = so.optimal_stop(tr, case)
        T = tr.T_at_end(st.index)
        phi, inside = so.phi_field(T, case)
        pinned = float(1.0 - np.mean(inside))
        J, seed = so.shape_J_and_seed(T, case)
        g = adjoint.gradient(case, s0, tr, {st.index: seed}, grad_ops=ops)
        pm = case.part_mask
        rng = np.random.default_rng(7)
        d = np.zeros_like(s0)
        v = rng.standard_normal(int(pm.sum()))
        d[pm] = v / np.linalg.norm(v)
        ana = float(np.sum(g * d))
        errs = []
        for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            jp = so.shape_J_and_seed(run_forward(case, s0 + eps * d).T_at_end(
                so.optimal_stop(run_forward(case, s0 + eps * d), case).index), case)[0]
            jm = so.shape_J_and_seed(run_forward(case, s0 - eps * d).T_at_end(
                so.optimal_stop(run_forward(case, s0 - eps * d), case).index), case)[0]
            fd = (jp - jm) / (2 * eps)
            errs.append(abs(fd - ana) / max(abs(ana), 1e-30))
        row = {"dt_pc_c": dtpc, "pinned_fraction_of_domain": pinned,
               "t_stop_index": st.index, "J": st.J, "analytic": ana,
               "rel_err_sweep": errs, "best_rel_err": float(min(errs))}
        rows.append(row)
        print(f"dt_pc={dtpc:5.1f} pinned={pinned:.4f} J={st.J:9.3f} "
              f"best_rel={min(errs):.3e} sweep={['%.2e' % e for e in errs]}", flush=True)
    Path(out_path).resolve().write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
