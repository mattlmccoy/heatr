"""Robustness validation of the solved dopant maps. FORWARD RUNS ONLY.

Nothing here re-solves. Every arm takes a map that was already solved on the
120 x 120 grid, perturbs it outside the solve, and re-runs the real forward,
re-optimizing only the STOP.

Stop convention, identical to `SHAPE_LIBRARY_SOLVE_REPORT.md`:
t_stop = argmin over that arm's OWN stored trajectory of

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

`at_horizon` is flagged when the minimum sits on the last stored step. The
melted region for intersection over union, growth and under-melt is phi >= 0.5.

Mode `grid`: optimize at 120, score at 160.
  U_uniform_160     saturation 1 everywhere, conductivity-only channel
  HIST_best_160     the SAME stored dopant map npz and the SAME boundary
                    convention that won the 120 census for this shape, loaded
                    through the PRODUCTION loader onto the 160 grid (the loader
                    resamples its 1715 x 1715 printer level map straight to
                    160 x 160), permittivity-co-varying channel as stored
  A1_cont_160       the solved continuous map, resampled 120 -> 160 in the
                    production convention (`robust.resample_map`)
  A1_4bpp_160       that resampled map re-quantized to 4 bits per pixel inside
                    the part through the production quantizer
                    (`printability.quantize_in_part`). THE deliverable arm.

Mode `rim`: grid 120 throughout.
  A1_4bpp_g<r>      the solved CONTINUOUS map blurred by a part-masked Gaussian
                    of r cells (`robust.smooth_in_part`), clipped to [0, 1],
                    then re-quantized to 4 bits per pixel. r = 1 and r = 2.
  The r = 0 baseline is the stored `A1_4bpp` arm of `out_lib/<shape>.json`; it is
  not re-run.

Run:
  ./.venv312/bin/python -m adjoint2d.robust_run grid <shape> <outdir>
  ./.venv312/bin/python -m adjoint2d.robust_run rim  <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import energy_gate as eg, forward as fwd
from . import printability as pq
from . import robust as rb
from . import shape_objective as so
from .library_solve import PATIENCE, shape_config
from .pins import build_case, load_cfg
from .verify_hist import load_stored_map

OUT_LIB = Path(__file__).resolve().parent.parent / "out_lib"
GRID_HOLDOUT = 160


def score(case, s: np.ndarray, eps_covary: bool = False) -> dict:
    """One real forward run, scored at its own J-stop, with every standing gate."""
    t0 = time.perf_counter()
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE,
                     eps_covary=eps_covary)
    m = so.full_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["eps_covary"] = bool(eps_covary)
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    m["sat_min_in_part"] = float(np.min(s[case.part_mask]))
    m["energy_gate"] = eg.gate_from_trajectory(tr, m["t_stop_index"])
    m["wall_s"] = time.perf_counter() - t0
    return m


def _stored(shape: str) -> tuple[dict, dict]:
    res = json.loads((OUT_LIB / f"{shape}.json").read_text())
    maps = np.load(OUT_LIB / f"{shape}_maps.npz")
    return res, maps


def _log(shape: str, name: str, m: dict) -> None:
    print(f"[{shape}] {name:16s} J {m['J']:9.2f}  J/cell {m['J_per_part_cell']:.5f}  "
          f"IoU {m['IoU']:.4f}  grow {m['bed_melt_pct_of_part']:6.2f}  "
          f"under {m['part_under_melt_pct']:6.2f}  stop {m['t_stop_index']:4d} "
          f"({m['t_stop_s']:6.1f} s){' HORIZON' if m['t_stop_at_horizon'] else ''}  "
          f"P {m['P_abs_W_per_m']:6.1f}  Eres {m['energy_gate']['rel_residual_at_index']*100:.2f}%  "
          f"{m['wall_s']:.1f} s", flush=True)


# ---------------------------------------------------------------------------
# Task A: grid hold-out
# ---------------------------------------------------------------------------

def run_grid(shape: str, outdir: str, n_grid: int = GRID_HOLDOUT) -> dict:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    stored, maps = _stored(shape)
    cfg_path = shape_config(shape)
    cfg = load_cfg(cfg_path)
    cfg["geometry"]["grid_nx"] = int(n_grid)
    cfg["geometry"]["grid_ny"] = int(n_grid)
    case = build_case(cfg)
    pm = case.part_mask

    hist_name = stored["best_hist_arm"]
    hist_meta = stored["hist_scan"][hist_name]

    res = {
        "task": "A_grid_holdout", "shape": shape, "config": str(cfg_path),
        "n_grid": int(n_grid), "n_grid_solved_at": 120,
        "n_part_cells": case.n_part,
        "n_part_cells_at_120": stored["arms"]["U_uniform"]["part_cells"],
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "n_substeps": case.pins.n_substeps,
        "stop_convention": "t_stop = argmin over the arm's own trajectory of J",
        "resample_convention": "scipy.ndimage.zoom order=1 then clip, "
                               "rfam_eqs_coupled.py:374-380",
        "best_hist_arm_at_120": hist_name,
        "hist_map_npz": hist_meta["map_npz"],
        "hist_convention": hist_meta["convention"],
        "arms": {}, "arms_120": {},
    }
    for a in ("U_uniform", "HIST_best", "A1_cont", "A1_4bpp"):
        res["arms_120"][a] = stored["arms"][a]

    store: dict[str, np.ndarray] = {"part_mask": pm, "x": case.x, "y": case.y}

    # uniform baseline at the hold-out grid
    m = score(case, np.ones(pm.shape))
    m["arm"] = "U_uniform_160"
    res["arms"]["U_uniform_160"] = m
    _log(shape, "U_uniform_160", m)

    # the same stored historical mask, loaded onto the 160 grid by the
    # PRODUCTION loader, in the convention that won at 120
    s_h = load_stored_map(case, Path(hist_meta["map_npz"]), cfg)
    if hist_meta["convention"] == "outside1":
        s_h = np.where(pm, s_h, 1.0)
    m = score(case, s_h, eps_covary=True)
    m["arm"] = "HIST_best_160"
    m["source_arm_at_120"] = hist_name
    res["arms"]["HIST_best_160"] = m
    store["HIST_best_160"] = s_h
    _log(shape, "HIST_best_160", m)

    # the solved map, transferred
    s_c = rb.resample_map(np.asarray(maps["A1_cont"]), n_grid, n_grid)
    s_c = np.where(pm, s_c, 1.0)
    m = score(case, s_c)
    m["arm"] = "A1_cont_160"
    res["arms"]["A1_cont_160"] = m
    store["A1_cont_160"] = s_c
    _log(shape, "A1_cont_160", m)

    s_q = pq.quantize_in_part(s_c, pm, bpp=4, sat_max=1.0)
    m = score(case, s_q)
    m["arm"] = "A1_4bpp_160"
    m.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm, bpp=4).items()})
    res["arms"]["A1_4bpp_160"] = m
    store["A1_4bpp_160"] = s_q
    _log(shape, "A1_4bpp_160", m)

    d, h, u = (res["arms"]["A1_4bpp_160"], res["arms"]["HIST_best_160"],
               res["arms"]["U_uniform_160"])
    d0 = res["arms_120"]["A1_4bpp"]
    h0 = res["arms_120"]["HIST_best"]
    res["verdict"] = {
        "deliverable_arm": "A1_4bpp_160",
        "beats_hist_on_J_160": bool(d["J"] < h["J"]),
        "beats_hist_on_IoU_160": bool(d["IoU"] > h["IoU"]),
        "beats_hist_on_J_120": bool(d0["J"] < h0["J"]),
        "beats_hist_on_IoU_120": bool(d0["IoU"] > h0["IoU"]),
        "dJ_rel_160": (h["J"] - d["J"]) / max(abs(h["J"]), 1e-30),
        "dJ_rel_120": (h0["J"] - d0["J"]) / max(abs(h0["J"]), 1e-30),
        "dIoU_160": d["IoU"] - h["IoU"],
        "dIoU_120": d0["IoU"] - h0["IoU"],
        "beats_uniform_on_J_160": bool(d["J"] < u["J"]),
        "ranking_preserved_J": bool((d["J"] < h["J"]) == (d0["J"] < h0["J"])),
        "ranking_preserved_IoU": bool((d["IoU"] > h["IoU"]) == (d0["IoU"] > h0["IoU"])),
    }
    # --- the dose-matched repeat -------------------------------------------
    # The pinned drive voltage was calibrated so the UNIFORM arm absorbs
    # 500 W per metre AT GRID 120. It does not at 160. Re-applying the
    # campaign's own calibration convention at the hold-out grid separates
    # "the map does not transfer" from "the voltage calibration does not
    # transfer". Absorbed power is exactly quadratic in the drive, so one
    # rescale suffices (`robust.recalibrated_voltage`).
    v0 = float(cfg["electric"]["voltage_v"])
    p_u = float(res["arms"]["U_uniform_160"]["P_abs_W_per_m"])
    v_re = rb.recalibrated_voltage(v0, p_u, 500.0)
    cfg_re = json.loads(json.dumps(cfg))
    cfg_re["electric"]["voltage_v"] = v_re
    case_re = build_case(cfg_re)
    res["recal"] = {"voltage_v_pinned": v0, "voltage_v_recalibrated": v_re,
                    "P_abs_uniform_at_pinned_v": p_u, "P_abs_target": 500.0,
                    "arms": {}}
    for name, sat, ec in (("U_uniform_160_recal", np.ones(pm.shape), False),
                          ("HIST_best_160_recal", s_h, True),
                          ("A1_4bpp_160_recal", s_q, False)):
        m = score(case_re, sat, eps_covary=ec)
        m["arm"] = name
        res["recal"]["arms"][name] = m
        _log(shape, name, m)
    rd = res["recal"]["arms"]["A1_4bpp_160_recal"]
    rh = res["recal"]["arms"]["HIST_best_160_recal"]
    res["recal"]["dJ_rel"] = (rh["J"] - rd["J"]) / max(abs(rh["J"]), 1e-30)
    res["recal"]["dIoU"] = rd["IoU"] - rh["IoU"]
    res["recal"]["ranking_preserved_J"] = bool((rd["J"] < rh["J"]) == (d0["J"] < h0["J"]))
    res["recal"]["ranking_preserved_IoU"] = bool(
        (rd["IoU"] > rh["IoU"]) == (d0["IoU"] > h0["IoU"]))
    res["recal"]["uniform_P_abs_check_W_per_m"] = float(
        res["recal"]["arms"]["U_uniform_160_recal"]["P_abs_W_per_m"])

    res["energy_gate_violations"] = [a for a, m in res["arms"].items()
                                     if not m["energy_gate"]["PASS"]]
    res["energy_gate_violations"] += [a for a, m in res["recal"]["arms"].items()
                                      if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t_start
    np.savez_compressed(out / f"{shape}_grid_maps.npz", **store)
    (out / f"{shape}_grid.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] GRID ranking preserved J={res['verdict']['ranking_preserved_J']} "
          f"IoU={res['verdict']['ranking_preserved_IoU']}  "
          f"dJ 120 {res['verdict']['dJ_rel_120']*100:+.1f}% -> 160 "
          f"{res['verdict']['dJ_rel_160']*100:+.1f}% (dose-matched "
          f"{res['recal']['dJ_rel']*100:+.1f}%)  wall {res['wall_s']:.1f} s", flush=True)
    return res


# ---------------------------------------------------------------------------
# Task B: rim robustness
# ---------------------------------------------------------------------------

RADII = (1.0, 2.0)


def run_rim(shape: str, outdir: str, radii=RADII) -> dict:
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    stored, maps = _stored(shape)
    cfg_path = shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    s0 = np.asarray(maps["A1_cont"], dtype=float)
    assert s0.shape == pm.shape

    res = {
        "task": "B_rim_robustness", "shape": shape, "config": str(cfg_path),
        "n_grid": int(pm.shape[0]), "n_part_cells": case.n_part,
        "stop_convention": "t_stop = argmin over the arm's own trajectory of J",
        "smoothing_convention": "part-masked normalized-convolution Gaussian on the "
                                "CONTINUOUS solved map, clipped to [0, 1], then "
                                "re-quantized 4 bits per pixel in part "
                                "(printability.quantize_in_part)",
        "baseline_arm": "A1_4bpp of out_lib (radius 0, not re-run)",
        "arms": {"r0.0": stored["arms"]["A1_4bpp"]},
        "hist_best": stored["arms"]["HIST_best"],
        "uniform": stored["arms"]["U_uniform"],
        "map_stats": {},
    }
    store: dict[str, np.ndarray] = {"part_mask": pm, "x": case.x, "y": case.y,
                                    "A1_4bpp_r0": np.asarray(maps["A1_4bpp"])}
    for r in radii:
        sm = rb.smooth_in_part(s0, pm, float(r))
        sq = pq.quantize_in_part(sm, pm, bpp=4, sat_max=1.0)
        key = f"r{float(r):.1f}"
        res["map_stats"][key] = {
            "sat_mean_in_part": float(np.mean(sm[pm])),
            "sat_std_in_part": float(np.std(sm[pm])),
            "sat_std_in_part_r0": float(np.std(s0[pm])),
            "max_abs_change_vs_r0": float(np.max(np.abs(sm[pm] - s0[pm]))),
            "rms_change_vs_r0": float(np.sqrt(np.mean((sm[pm] - s0[pm]) ** 2))),
        }
        m = score(case, sq)
        m["arm"] = f"A1_4bpp_g{r:.0f}"
        m["gaussian_sigma_cells"] = float(r)
        m.update({f"census_{k}": v for k, v in pq.level_census(sq, pm, bpp=4).items()})
        res["arms"][key] = m
        store[f"A1_cont_smooth_{key}"] = sm
        store[f"A1_4bpp_{key}"] = sq
        _log(shape, f"A1_4bpp_g{r:.0f}", m)

    b = res["arms"]["r0.0"]
    res["verdict"] = {
        f"dJ_rel_{k}": (res["arms"][k]["J"] - b["J"]) / max(abs(b["J"]), 1e-30)
        for k in res["arms"] if k != "r0.0"}
    res["verdict"].update({
        f"dIoU_{k}": res["arms"][k]["IoU"] - b["IoU"]
        for k in res["arms"] if k != "r0.0"})
    res["verdict"]["still_beats_hist_on_J_at_r2"] = bool(
        res["arms"]["r2.0"]["J"] < res["hist_best"]["J"])
    res["verdict"]["still_beats_uniform_on_J_at_r2"] = bool(
        res["arms"]["r2.0"]["J"] < res["uniform"]["J"])
    res["energy_gate_violations"] = [
        k for k, m in res["arms"].items() if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t_start
    np.savez_compressed(out / f"{shape}_rim_maps.npz", **store)
    (out / f"{shape}_rim.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] RIM dJ r1 {res['verdict']['dJ_rel_r1.0']*100:+.2f}% "
          f"r2 {res['verdict']['dJ_rel_r2.0']*100:+.2f}%  "
          f"dIoU r1 {res['verdict']['dIoU_r1.0']:+.4f} r2 {res['verdict']['dIoU_r2.0']:+.4f}  "
          f"wall {res['wall_s']:.1f} s", flush=True)
    return res


if __name__ == "__main__":
    _mode, _shape, _outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    if _mode == "grid":
        run_grid(_shape, _outdir)
    elif _mode == "rim":
        run_rim(_shape, _outdir)
    else:
        raise SystemExit(f"unknown mode {_mode!r}")
