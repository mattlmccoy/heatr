"""Dose-matched repeat of Gate B, to separate a shape effect from a dose effect.

MEASURED reason this exists. Gate B blurs the delivered map and re-scores at the
PINNED drive voltage, and a part-masked blur of a non-uniform map changes the
absorbed power: at a blur of 1.5 cells the six shapes move P_abs by +0.35 to
+7.97 percent. So a Gate B failure could be either of two different things, a
real sensitivity of the shape to sub-radius structure or a dose change the arm
never asked for. Recalibrating the drive so each blurred arm absorbs exactly the
unblurred arm's power separates them. The rescale is exact
(`robust.recalibrated_voltage`, the electro-quasi-static solve is quadratic in
the drive), so this costs ONE extra forward run per blurred arm.

Run:
  ./.venv312/bin/python -m adjoint2d.topopt_dosematch <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import chi_area, library_solve as lib
from . import printability as pq
from . import robust as rb
from . import topopt_solve as tos
from .pins import build_case, load_cfg
from .topopt_robust import GATE_B_TOL, OUT_TOPOPT, SUB_RADII_CELLS


def main(shape: str, outdir: str) -> dict:
    out = Path(outdir).resolve()
    t0 = time.perf_counter()
    src = json.loads((OUT_TOPOPT / f"{shape}.json").read_text())
    with np.load(OUT_TOPOPT / f"{shape}_maps.npz") as d:
        s_cont = np.asarray(d["TO_cont"], dtype=float)
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    chi, _ = chi_area.chi_from_cfg(cfg, case.x, case.y)
    base = src["arms"]["TO_4bpp"]
    p_base = float(base["P_abs_W_per_m"])
    v0 = float(cfg["electric"]["voltage_v"])

    res = {"shape": shape, "P_abs_target_W_per_m": p_base, "voltage_v_pinned": v0,
           "baseline_120_TO_4bpp": {k: base[k] for k in
                                    ("J", "IoU", "IoU_area", "P_abs_W_per_m")},
           "arms": {}}
    prev = json.loads((OUT_TOPOPT / f"{shape}_robust.json").read_text())
    for r in SUB_RADII_CELLS:
        key = f"r{r:.1f}"
        p_pinned = float(prev["gate_B_sub_radius"][key]["P_abs_W_per_m"])
        v_re = rb.recalibrated_voltage(v0, p_pinned, p_base)
        cfg_re = json.loads(json.dumps(cfg))
        cfg_re["electric"]["voltage_v"] = v_re
        case_re = build_case(cfg_re)
        sm = rb.smooth_in_part(s_cont, pm, float(r))
        sq = pq.quantize_in_part(sm, pm, bpp=4, sat_max=1.0)
        m = tos.score(case_re, sq, chi)
        m["arm"] = f"TO_4bpp_blur{r:.1f}_dosematched"
        m["gaussian_sigma_cells"] = float(r)
        m["gaussian_sigma_mm"] = float(r) * case.dx * 1e3
        m["voltage_v"] = v_re
        m["dJ_rel"] = (m["J"] - base["J"]) / max(abs(base["J"]), 1e-30)
        m["dJ_rel_at_pinned_voltage"] = prev["gate_B_sub_radius"][key]["dJ_rel"]
        m["dP_rel_removed"] = p_pinned / p_base - 1.0
        res["arms"][key] = m
        print(f"[{shape}] blur {r:.1f} dose matched: V {v0:.1f} -> {v_re:.1f}, "
              f"P {p_pinned:.1f} -> {m['P_abs_W_per_m']:.1f} against target "
              f"{p_base:.1f}; dJ {100*m['dJ_rel']:+.2f} percent against "
              f"{100*m['dJ_rel_at_pinned_voltage']:+.2f} percent at the pinned "
              f"voltage", flush=True)
    res["verdict"] = {
        "tolerance": GATE_B_TOL,
        "max_abs_dJ_rel": max(abs(m["dJ_rel"]) for m in res["arms"].values()),
        "PASS": bool(all(abs(m["dJ_rel"]) < GATE_B_TOL for m in res["arms"].values()))}
    res["wall_s"] = time.perf_counter() - t0
    (out / f"{shape}_dosematch.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] GATE B dose matched: max |dJ| "
          f"{100*res['verdict']['max_abs_dJ_rel']:.2f} percent, PASS = "
          f"{res['verdict']['PASS']}, wall {res['wall_s']:.0f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(OUT_TOPOPT))
