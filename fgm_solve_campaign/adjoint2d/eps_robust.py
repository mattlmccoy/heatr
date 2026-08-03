"""The two robustness probes on the PERMITTIVITY-CHANNEL solved maps.

FORWARD RUNS ONLY, nothing is re-solved, and every run is in the
permittivity-co-varying channel the map was solved in. The protocol is the one
`ms_robust.py` used, so the numbers are comparable arm for arm:

  rim   the solved CONTINUOUS map is blurred by a part-masked
        normalized-convolution Gaussian of 1 and 2 cells and re-quantized at
        4 bits per pixel, at grid 120. Radius 0 is the stored deliverable and is
        not re-run.

  grid  the solved CONTINUOUS map is resampled 120 -> 160 in the production
        map-injection convention (bilinear then clipped) and re-quantized. The
        dose-matched repeat reuses the drive voltage `out_robust/<shape>_grid.json`
        calibrated so the UNIFORM arm at 160 absorbs 500 watts per metre of
        depth. GRID QUALIFIER: `SOLVE_ROBUSTNESS_VALIDATION.md` measured that
        the forward itself is not converged in intersection over union between
        these two grids, so a 160 number is a joint test of map transfer AND
        forward discretization and cannot separate them.

Run:
  ./.venv312/bin/python -m adjoint2d.eps_robust <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eps_solve as es
from . import library_solve as lib
from . import printability as pq
from . import robust as rb
from .pins import build_case, load_cfg

OUT_EPS = Path(__file__).resolve().parents[1] / "out_eps"
OUT_ROBUST = Path(__file__).resolve().parents[1] / "out_robust"
GRID_HOLDOUT = 160
RADII = (1.0, 2.0)


def _log(shape, name, m):
    print(f"[{shape}] {name:24s} J {m['J']:9.2f}  IoU {m['IoU']:.4f}  "
          f"grow {m['bed_melt_pct_of_part']:6.2f}  under {m['part_under_melt_pct']:6.2f}  "
          f"stop {m['t_stop_s']:6.1f} s{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
          f"rho {m['mean_rho_rel_part_at_stop']:.4f}  P {m['P_abs_W_per_m']:6.1f}  "
          f"Eres {m['energy_gate']['rel_residual_at_index']*100:.2f}%  "
          f"{m['wall_s']:.0f} s", flush=True)


def main(shape: str, outdir: str) -> dict:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    src = json.loads((OUT_EPS / f"{shape}.json").read_text())
    with np.load(OUT_EPS / f"{shape}_maps.npz") as d:
        s_cont_120 = np.asarray(d["EPS_best_cont"], dtype=float)

    cfg_path = lib.shape_config(shape)
    res: dict = {"shape": shape, "config": str(cfg_path),
                 "channel": src["channel"], "source": str(OUT_EPS / f"{shape}.json"),
                 "winner_start": src["winner_start"],
                 "sigma_cells": src["sigma_cells"],
                 "stop_convention": src["stop_convention"],
                 "baseline_120": {k: src["arms"][k] for k in
                                  ("EPS_best_4bpp", "EPS_best_cont")},
                 "grid": {}, "rim": {}}

    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    for r in RADII:
        sm = rb.smooth_in_part(s_cont_120, pm, float(r))
        sq = pq.quantize_in_part(sm, pm, bpp=4, sat_max=1.0)
        m = es.score(case, sq)
        m["arm"] = f"EPS_best_4bpp_g{r:.0f}"
        m["gaussian_sigma_cells"] = float(r)
        m["rms_change_vs_r0"] = float(np.sqrt(np.mean((sm[pm] - s_cont_120[pm]) ** 2)))
        m.update({f"census_{k}": v for k, v in pq.level_census(sq, pm, bpp=4).items()})
        res["rim"][f"r{r:.1f}"] = m
        _log(shape, f"rim r={r:.0f}", m)

    b = res["baseline_120"]["EPS_best_4bpp"]
    res["rim_verdict"] = {}
    for r in RADII:
        k = f"r{r:.1f}"
        res["rim_verdict"][f"dJ_rel_{k}"] = (
            (res["rim"][k]["J"] - b["J"]) / max(abs(b["J"]), 1e-30))
        res["rim_verdict"][f"dIoU_{k}"] = res["rim"][k]["IoU"] - b["IoU"]

    cfg160 = load_cfg(cfg_path)
    cfg160["geometry"]["grid_nx"] = GRID_HOLDOUT
    cfg160["geometry"]["grid_ny"] = GRID_HOLDOUT
    case160 = build_case(cfg160)
    pm160 = case160.part_mask
    s_c = np.where(pm160, rb.resample_map(s_cont_120, GRID_HOLDOUT, GRID_HOLDOUT), 1.0)
    m = es.score(case160, s_c)
    m["arm"] = "EPS_cont_160"
    res["grid"]["EPS_cont_160"] = m
    _log(shape, "EPS_cont_160", m)

    s_q = pq.quantize_in_part(s_c, pm160, bpp=4, sat_max=1.0)
    m = es.score(case160, s_q)
    m["arm"] = "EPS_4bpp_160"
    m.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm160, bpp=4).items()})
    res["grid"]["EPS_4bpp_160"] = m
    _log(shape, "EPS_4bpp_160", m)

    # the uniform arm at 160 in THIS channel; s = 1 makes the two channels the
    # same map, so it is also the conductivity-channel uniform arm.
    m = es.score(case160, np.ones(pm160.shape))
    m["arm"] = "U_uniform_160"
    res["grid"]["U_uniform_160"] = m
    _log(shape, "U_uniform_160", m)

    prev_grid_path = OUT_ROBUST / f"{shape}_grid.json"
    if prev_grid_path.exists():
        prev = json.loads(prev_grid_path.read_text())
        v_re = float(prev["recal"]["voltage_v_recalibrated"])
        cfg_re = json.loads(json.dumps(cfg160))
        cfg_re["electric"]["voltage_v"] = v_re
        case_re = build_case(cfg_re)
        m = es.score(case_re, s_q)
        m["arm"] = "EPS_4bpp_160_recal"
        m["voltage_v"] = v_re
        res["grid"]["EPS_4bpp_160_recal"] = m
        _log(shape, "EPS_4bpp_160_recal", m)
        res["reference_160"] = {k: prev["arms"][k] for k in
                                ("U_uniform_160", "HIST_best_160", "A1_cont_160",
                                 "A1_4bpp_160") if k in prev["arms"]}
        res["reference_160"]["_source"] = str(prev_grid_path)
        res["reference_160"]["_note"] = (
            "the stored 160 reference arms were run in the conductivity-only "
            "channel EXCEPT HIST_best_160, which was run in the permittivity "
            "channel it was scored in; U_uniform_160 is channel invariant")
        res["reference_recal_voltage_v"] = v_re
    else:
        print(f"[{shape}] no out_robust grid reference, dose-matched repeat "
              f"SKIPPED LOUDLY", flush=True)

    ms_robust_path = Path(__file__).resolve().parents[1] / "out_ms" / f"{shape}_robust.json"
    if ms_robust_path.exists():
        res["reference_ms_robust"] = json.loads(ms_robust_path.read_text())
        res["reference_ms_robust"]["_source"] = str(ms_robust_path)

    res["energy_gate_violations"] = [
        a for grp in ("grid", "rim") for a, m in res[grp].items()
        if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(out / f"{shape}_robust_maps.npz",
                        EPS_cont_120=s_cont_120, EPS_cont_160=s_c, EPS_4bpp_160=s_q,
                        part_mask_160=pm160.astype(np.uint8))
    (out / f"{shape}_robust.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] robust done, wall {res['wall_s']:.0f} s, energy gate violations "
          f"{res['energy_gate_violations'] or 'none'}", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(OUT_EPS))
