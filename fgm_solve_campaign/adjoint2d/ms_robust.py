"""Re-run the two robustness probes on the FILTERED MULTI-START maps.

`SOLVE_ROBUSTNESS_VALIDATION.md` measured that the unfiltered single-start
solved maps are grid sculptures: a one-cell part-masked blur costs +37 to +892
percent of J_phi on five of six shapes, and none of the three shapes in the
absolute SOLVED class at grid 120 stays there at grid 160. Its Section 10 named
the fix, which the multi-start solve now applies: filter the DESIGN VARIABLE so
sub-resolution structure is not expressible. This is the first test of whether
that fix actually works.

FORWARD RUNS ONLY. Nothing is re-solved. Both probes reproduce the earlier
protocol exactly so the numbers are comparable arm for arm:

  grid   the solved CONTINUOUS map is resampled 120 -> 160 in the production
         map-injection convention (`robust.resample_map`, bilinear then clipped)
         and re-quantized at 4 bits per pixel inside the part. The uniform and
         historical arms at 160 are NOT re-run: they are read from
         `out_robust/<shape>_grid.json`, which ran them on this same engine at
         this same grid with this same configuration. The dose-matched repeat
         reuses the drive voltage recorded there, which was calibrated so the
         UNIFORM arm at 160 absorbs 500 W per metre of depth.

  rim    the solved CONTINUOUS map is blurred by a part-masked
         normalized-convolution Gaussian of 1 and 2 cells and re-quantized, at
         grid 120. Radius 0 is the stored MS_4bpp arm and is not re-run.

Run:
  ./.venv312/bin/python -m adjoint2d.ms_robust <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import library_solve as lib
from . import ms_solve as msv
from . import printability as pq
from . import robust as rb
from .pins import build_case, load_cfg

OUT_MS = Path(__file__).resolve().parents[1] / "out_ms"
OUT_ROBUST = Path(__file__).resolve().parents[1] / "out_robust"
GRID_HOLDOUT = 160
RADII = (1.0, 2.0)


def _log(shape, name, m):
    print(f"[{shape}] {name:22s} J {m['J']:9.2f}  IoU {m['IoU']:.4f}  "
          f"grow {m['bed_melt_pct_of_part']:6.2f}  under {m['part_under_melt_pct']:6.2f}  "
          f"stop {m['t_stop_s']:6.1f} s{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
          f"rho {m['mean_rho_rel_part_at_stop']:.4f}  P {m['P_abs_W_per_m']:6.1f}  "
          f"Eres {m['energy_gate']['rel_residual_at_index']*100:.2f}%  "
          f"{m['wall_s']:.0f} s", flush=True)


def main(shape: str, outdir: str) -> dict:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ms_json = json.loads((OUT_MS / f"{shape}.json").read_text())
    with np.load(OUT_MS / f"{shape}_maps.npz") as d:
        s_cont_120 = np.asarray(d["MS_cont"], dtype=float)

    cfg_path = lib.shape_config(shape)
    res: dict = {"shape": shape, "config": str(cfg_path),
                 "source": str(OUT_MS / f"{shape}.json"),
                 "winner_start": ms_json["winner_start"],
                 "sigma_cells": ms_json["sigma_cells"],
                 "stop_convention": ms_json["stop_convention"],
                 "baseline_120": {"MS_4bpp": ms_json["arms"]["MS_4bpp"],
                                  "MS_cont": ms_json["arms"]["MS_cont"]},
                 "grid": {}, "rim": {}}

    # ----- rim, grid 120 ---------------------------------------------------
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    for r in RADII:
        sm = rb.smooth_in_part(s_cont_120, pm, float(r))
        sq = pq.quantize_in_part(sm, pm, bpp=4, sat_max=1.0)
        m = msv.score(case, sq)
        m["arm"] = f"MS_4bpp_g{r:.0f}"
        m["gaussian_sigma_cells"] = float(r)
        m["rms_change_vs_r0"] = float(np.sqrt(np.mean((sm[pm] - s_cont_120[pm]) ** 2)))
        m["max_abs_change_vs_r0"] = float(np.max(np.abs(sm[pm] - s_cont_120[pm])))
        m.update({f"census_{k}": v for k, v in pq.level_census(sq, pm, bpp=4).items()})
        res["rim"][f"r{r:.1f}"] = m
        _log(shape, f"MS_4bpp rim r={r:.0f}", m)

    b = res["baseline_120"]["MS_4bpp"]
    res["rim_verdict"] = {
        f"dJ_rel_r{r:.1f}": (res["rim"][f"r{r:.1f}"]["J"] - b["J"]) / max(abs(b["J"]), 1e-30)
        for r in RADII}
    res["rim_verdict"].update({
        f"dIoU_r{r:.1f}": res["rim"][f"r{r:.1f}"]["IoU"] - b["IoU"] for r in RADII})

    # ----- grid hold-out ---------------------------------------------------
    cfg160 = load_cfg(cfg_path)
    cfg160["geometry"]["grid_nx"] = GRID_HOLDOUT
    cfg160["geometry"]["grid_ny"] = GRID_HOLDOUT
    case160 = build_case(cfg160)
    pm160 = case160.part_mask
    s_c = np.where(pm160, rb.resample_map(s_cont_120, GRID_HOLDOUT, GRID_HOLDOUT), 1.0)
    m = msv.score(case160, s_c)
    m["arm"] = "MS_cont_160"
    res["grid"]["MS_cont_160"] = m
    _log(shape, "MS_cont_160", m)

    s_q = pq.quantize_in_part(s_c, pm160, bpp=4, sat_max=1.0)
    m = msv.score(case160, s_q)
    m["arm"] = "MS_4bpp_160"
    m.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm160, bpp=4).items()})
    res["grid"]["MS_4bpp_160"] = m
    _log(shape, "MS_4bpp_160", m)

    # dose-matched repeat at the voltage the earlier pass calibrated at 160
    prev_grid_path = OUT_ROBUST / f"{shape}_grid.json"
    if prev_grid_path.exists():
        prev = json.loads(prev_grid_path.read_text())
        v_re = float(prev["recal"]["voltage_v_recalibrated"])
        cfg_re = json.loads(json.dumps(cfg160))
        cfg_re["electric"]["voltage_v"] = v_re
        case_re = build_case(cfg_re)
        m = msv.score(case_re, s_q)
        m["arm"] = "MS_4bpp_160_recal"
        m["voltage_v"] = v_re
        res["grid"]["MS_4bpp_160_recal"] = m
        _log(shape, "MS_4bpp_160_recal", m)
        res["reference_160"] = {k: prev["arms"][k] for k in
                                ("U_uniform_160", "HIST_best_160", "A1_cont_160",
                                 "A1_4bpp_160") if k in prev["arms"]}
        res["reference_160"]["A1_4bpp_160_recal"] = prev["recal"]["arms"].get(
            "A1_4bpp_160_recal")
        res["reference_160"]["_source"] = str(prev_grid_path)
        res["reference_recal_voltage_v"] = v_re
    else:
        print(f"[{shape}] no out_robust grid reference, dose-matched repeat SKIPPED "
              f"LOUDLY", flush=True)

    prev_rim_path = OUT_ROBUST / f"{shape}_rim.json"
    if prev_rim_path.exists():
        prev_rim = json.loads(prev_rim_path.read_text())
        res["reference_rim"] = {k: prev_rim["arms"][k] for k in prev_rim["arms"]}
        res["reference_rim"]["_source"] = str(prev_rim_path)

    res["energy_gate_violations"] = [
        a for grp in ("grid", "rim") for a, m in res[grp].items()
        if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(out / f"{shape}_robust_maps.npz",
                        MS_cont_120=s_cont_120, MS_cont_160=s_c, MS_4bpp_160=s_q,
                        part_mask_160=pm160.astype(np.uint8))
    (out / f"{shape}_robust.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] robust done, wall {res['wall_s']:.0f} s, energy gate violations "
          f"{res['energy_gate_violations'] or 'none'}", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(OUT_MS))
