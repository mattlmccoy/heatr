"""Solve at the price from a WARM start, to separate two explanations.

The cold pass (`run_wout_solve.py`) starts every at-price solve from uniform
saturation, which is what the previous campaign did. If the map solved at the
price loses to the melt-region-solved map re-read at that price, there are two
completely different reasons and they have opposite consequences:

  (1) the OBJECTIVE does not want a different map, so the price is a read-state
      effect and the shipped recipe stands; or
  (2) the objective does want a different map but 40 forward-equivalents from
      uniform cannot reach it, so the finding is about the BUDGET and the fix is
      to start the at-price solve from the melt-solved map rather than to keep
      the melt-solved map.

Starting the SAME optimizer, at the SAME budget, from the melt-solved map tells
the two apart: if the warm solve cannot improve on its own start, that is (1).

The budget is taken from the cold run's own measured cost model, so the two
starts get exactly the same number of gradient evaluations on the same shape.

Run:
  ./.venv312/bin/python run_wout_warm.py <shape> <w_out>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from adjoint2d import asym_objective as ao, asym_solve as asy
from adjoint2d import design_filter as df, gradops, printability as pq
from adjoint2d.pins import build_case, load_cfg
from adjoint2d import library_solve as lib

HERE = Path(__file__).resolve().parent
OUT_W = HERE / "out_wout"


def main(shape: str, w_out: float, floor: float = 0.85, w_in: float = 1.0,
         sigma_cells: float = asy.SIGMA_CELLS) -> dict:
    t0 = time.perf_counter()
    tag = f"_w{str(w_out).replace('.', 'p')}"
    cold = json.loads((OUT_W / f"{shape}{tag}.json").read_text())
    n_each = int(cold["cost"]["n_gradient_evals_per_optimizer"])

    case = build_case(load_cfg(lib.shape_config(shape)))
    pm = np.asarray(case.part_mask, dtype=bool)
    ops = gradops.gradient_matrices(case.x, case.y)
    v_init = asy.previous_phi_map(shape)
    if v_init is None:
        raise ValueError(f"no melt-solved map stored for {shape!r}")

    rows, info, v_best, best = asy.solve_asym(
        case, ops, n_each, (0.0, 1.0), v_init, sigma_cells,
        floor, w_out, w_in, optimizer="mma")
    s_cont = df.apply_filter(v_best, pm, sigma_cells)
    s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)

    res = {"shape": shape, "w_out": float(w_out), "w_in": float(w_in),
           "floor_rho_rel": float(floor), "sigma_cells": float(sigma_cells),
           "start": "melt_solved_4bpp_map_of_the_library_campaign",
           "n_gradient_evals": n_each,
           "budget_matched_to": str(OUT_W / f"{shape}{tag}.json"),
           "solve": {"info": info, "rows": rows},
           "arms": {}}
    for name, s in (("WARM_mma_cont", s_cont), ("WARM_mma_4bpp", s_q)):
        m = asy.score_asym(case, s, floor, w_out, w_in)
        m["arm"] = name
        res["arms"][name] = m
        print(f"[{shape} w={w_out}] {name} J_asym {m['J_asym']:.5f} "
              f"(out {m['J_asym_out']:.5f} + in {m['J_asym_in']:.5f}) stop "
              f"{m['asym_stop_s']:.0f} s IoU {m['IoU']:.4f} growth "
              f"{m['growth_pct']:.2f}% under {m['under_pct']:.2f}% rho "
              f"{m['mean_rho_rel_part']:.4f} above-floor "
              f"{m['frac_part_at_or_above_floor']:.3f}", flush=True)
    res["start_J_asym"] = rows[0]["J_asym"] if rows else None
    res["best_J_asym_in_solve"] = best["J_asym"] if best else None
    res["improved_on_its_start"] = bool(
        rows and best["J_asym"] < rows[0]["J_asym"])
    res["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(OUT_W / f"{shape}{tag}_warm_maps.npz", x=case.x, y=case.y,
                        part_mask=pm.astype(np.uint8),
                        WARM_mma_cont=s_cont, WARM_mma_4bpp=s_q, v_best=v_best)
    (OUT_W / f"{shape}{tag}_warm.json").write_text(
        json.dumps(res, indent=2, default=float))
    print(f"[{shape} w={w_out}] warm start {res['start_J_asym']:.5f} -> best "
          f"{res['best_J_asym_in_solve']:.5f} in {len(rows)} evaluations, "
          f"improved = {res['improved_on_its_start']}, wall {res['wall_s']:.0f} s",
          flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
