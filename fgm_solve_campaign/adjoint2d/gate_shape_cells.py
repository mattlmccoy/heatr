"""Multi-cell single-probe gate for the shape gradient.

The random-direction probe is limited by phase-ramp breakpoints (many cells
cross the melt front inside one epsilon step). A single-cell probe moves far
fewer cells across the front. Gating several INDEPENDENT single cells at the
epsilon that the sweep identified as the V bottom shows whether the
3e-07-level agreement is typical or a fluke of one cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import adjoint, gradops, shape_objective as so
from .gate_shape import default_s, run_forward
from .pins import build_case, load_cfg


def main(cfg_path: str, out_path: str, n_cells: int = 6, eps: float = 1e-6) -> dict:
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    ops = gradops.gradient_matrices(case.x, case.y)
    s0 = default_s(case)
    tr = run_forward(case, s0, True)
    st = so.optimal_stop(tr, case)
    _J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
    g = adjoint.gradient(case, s0, tr, {st.index: seed}, grad_ops=ops)

    pm = case.part_mask
    cells = np.argwhere(pm)
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(cells))
    # sample across the whole sensitivity range, not only the biggest cell
    mags = np.abs(g[pm])
    ranks = np.argsort(-mags)
    picks = [ranks[0], ranks[len(ranks) // 8], ranks[len(ranks) // 4],
             ranks[len(ranks) // 2], ranks[3 * len(ranks) // 4], order[0]][:n_cells]

    chi = so.chi_part(case)

    def _resid(s_pert):
        t = run_forward(case, s_pert)
        st_ = so.optimal_stop(t, case)
        return so.phi_field(t.T_at_end(st_.index), case)[0] - chi

    rows = []
    for k in picks:
        i, j = cells[k]
        d = np.zeros_like(s0)
        d[i, j] = 1.0
        ana = float(g[i, j])
        rp = _resid(s0 + eps * d)
        rm = _resid(s0 - eps * d)
        # PAIRED difference. J is a sum of about 14400 order-one terms while a
        # single-cell perturbation moves only the melt-front annulus, so
        # forming (J_plus - J_minus) directly loses about 13 digits to
        # cancellation. Differencing ELEMENTWISE first and summing afterwards
        # removes the cancellation exactly; it changes the estimator, not the
        # model.
        fd = float(np.sum(rp * rp - rm * rm)) / (2 * eps)
        rel = abs(fd - ana) / max(abs(ana), 1e-30)
        rows.append({"cell": [int(i), int(j)], "analytic": ana, "fd": float(fd),
                     "rel_err": float(rel)})
        print(f"cell ({i:3d},{j:3d}) ana={ana:12.6f} fd={fd:12.6f} rel={rel:.3e}", flush=True)
    res = {"eps": eps, "t_stop_index": st.index, "rows": rows,
           "median_rel_err": float(np.median([r["rel_err"] for r in rows])),
           "max_rel_err": float(max(r["rel_err"] for r in rows))}
    res["PASS"] = bool(res["max_rel_err"] < 1e-6)
    Path(out_path).resolve().write_text(json.dumps(res, indent=2))
    print(f"median {res['median_rel_err']:.3e}  max {res['max_rel_err']:.3e}  "
          f"{'PASS' if res['PASS'] else 'FAIL'}")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
