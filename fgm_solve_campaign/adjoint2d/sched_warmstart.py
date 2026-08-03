"""Warm-started co-optimization: the library map is the STARTING POINT.

WHY THIS EXISTS. In `sched_solve` the co-optimized arm starts from a uniform
dopant map and gets 40 forward-equivalents, while the baseline it is compared
against is the library's own solved map, produced under a much larger budget in
`SHAPE_LIBRARY_SOLVE_REPORT.md`. A cold start under a small budget losing to a
warm map is a statement about the budget, not about co-optimization. This arm
removes that confound by starting the map at the library deliverable.

ARMS
  WARM_sched_only   library 4 bit map held FIXED, schedule optimized. The direct
                    question: what does adding a schedule buy on top of the
                    shipped map?
  WARM_CO_cont      library map as the start, map and schedule co-optimized.
  WARM_CO_4bpp      that map re-quantized to 4 bits per pixel with the schedule
                    re-optimized. The honest deliverable.

Same conventions as `sched_solve`: whole-domain shape objective, per-arm J-stop,
early stop disabled, schedule window from `sched_solve.WINDOW`.

Run:
  ./.venv312/bin/python -m adjoint2d.sched_warmstart <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, control, gradops
from . import printability as pq
from . import schedule as sch
from . import shape_objective as so
from .library_solve import shape_config
from .pins import build_case, load_cfg
from .sched_solve import (BLOCK_ORDER, BUDGET_CO, BUDGET_RESCHED,
                          BUDGET_SCHED_ONLY, HORIZON, LIB_MAPS, P_BOX_CONT,
                          WINDOW, iso_j_hold_gain, joint_solve, run_forward,
                          schedule_only_solve, score)


def main(shape: str, outdir: str, n_seg: int = 16) -> dict:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    case = build_case(load_cfg(shape_config(shape)))
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    horizon, win = HORIZON[shape], WINDOW[shape]
    ones = np.ones(n_seg)
    s_base = np.asarray(np.load(LIB_MAPS / f"{shape}_maps.npz")["A1_4bpp"], dtype=float)

    t_a = time.perf_counter()
    tr = run_forward(case, s_base, ones, n_seg, horizon, checkpoints=True, window=win)
    t_b = time.perf_counter()
    st = so.optimal_stop(tr, case)
    _J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_base, tr, {st.index: seed}, grad_ops=ops, with_schedule=True)
    ratio = (time.perf_counter() - t_c) / max(t_b - t_a, 1e-9)
    n_sched = control.max_gradient_evals(BUDGET_SCHED_ONLY, ratio)
    n_co = control.max_gradient_evals(BUDGET_CO, ratio)
    n_re = control.max_gradient_evals(BUDGET_RESCHED, ratio)

    res: dict = {"shape": shape, "n_seg": n_seg, "horizon_steps": horizon,
                 "schedule_window_steps": win, "dt_s": case.pins.dt,
                 "start_map": str(LIB_MAPS / f"{shape}_maps.npz") + " key A1_4bpp",
                 "budgets": {"sched_only_evals": n_sched, "co_evals": n_co,
                             "resched_evals": n_re, "ratio": ratio,
                             "block_order": list(BLOCK_ORDER)},
                 "arms": {}}
    maps = {"part_mask": pm}

    def record(tag, s, p):
        m = score(case, s, p, n_seg, horizon, window=win)
        m["arm"] = tag
        m["iso_J_hold"] = iso_j_hold_gain(m["J_curve"], m["rho_curve"], m["t_stop_index"])
        res["arms"][tag] = m
        maps[f"map_{tag}"] = s
        print(f"[{shape}] {tag:16s} J {m['J']:9.2f} IoU {m['IoU']:.4f} "
              f"stop {m['t_stop_s']:7.1f}s duty {m['duty_cycle']:.3f} "
              f"rho {m['mean_rho_part_at_stop']:.3f} "
              f"{m['structure']['structure']}", flush=True)
        return m

    m_base = record("WARM_base_noshed", s_base, None)

    _b, _s, p_w, rows_w = schedule_only_solve(
        case, ops, n_seg, horizon, s_base, ones, n_sched, P_BOX_CONT, window=win)
    res["rows_WARM_sched_only"] = rows_w
    record("WARM_sched_only", s_base, p_w)

    _b, s_w, p_c, rows_c = joint_solve(
        case, ops, n_seg, horizon, s_base, ones, n_co, p_box=P_BOX_CONT, window=win)
    res["rows_WARM_CO"] = rows_c
    record("WARM_CO_cont", s_w, p_c)

    s_w4 = pq.quantize_in_part(s_w, pm, bpp=4, sat_max=1.0)
    _b, _s, p_c4, rows_r = schedule_only_solve(
        case, ops, n_seg, horizon, s_w4, p_c, n_re, P_BOX_CONT, window=win)
    res["rows_WARM_resched"] = rows_r
    m_del = record("WARM_CO_4bpp", s_w4, p_c4)

    res["verdict"] = {
        "baseline_arm": "WARM_base_noshed",
        "baseline_J": m_base["J"], "baseline_IoU": m_base["IoU"],
        "deliverable_arm": "WARM_CO_4bpp",
        "deliverable_J": m_del["J"], "deliverable_IoU": m_del["IoU"],
        "dIoU_vs_baseline": m_del["IoU"] - m_base["IoU"],
        "dJ_rel_vs_baseline": (m_base["J"] - m_del["J"]) / max(abs(m_base["J"]), 1e-30),
        "sched_only_dIoU": res["arms"]["WARM_sched_only"]["IoU"] - m_base["IoU"],
        "structure_by_arm": {k: v["structure"]["structure"] for k, v in res["arms"].items()},
        "mean_rho_by_arm": {k: v["mean_rho_part_at_stop"] for k, v in res["arms"].items()},
        "duty_by_arm": {k: v["duty_cycle"] for k, v in res["arms"].items()},
    }
    res["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(out / f"{shape}_warm_maps.npz", **maps,
                        p_WARM_sched_only=p_w, p_WARM_CO_cont=p_c, p_WARM_CO_4bpp=p_c4)
    (out / f"{shape}_warm.json").write_text(json.dumps(res, indent=2, default=float))
    v = res["verdict"]
    print(f"[{shape}] WARM VERDICT dIoU {v['dIoU_vs_baseline']:+.4f} "
          f"dJ {v['dJ_rel_vs_baseline']*100:+.1f}% wall {res['wall_s']:.0f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 16)
