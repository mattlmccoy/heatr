"""Task 1: comparability audit against the ACTUAL stored historical artifacts.

Two independent things happen here.

1. `stored_run_gate` reproduces a STORED historical run. It reads that run's own
   `used_config.yaml`, loads that run's own dopant map through the PRODUCTION
   loader `rfam_eqs_coupled._FgmFeedback.from_config`, marches the prototype
   forward in identity mode, and compares the resulting phi_bar = 0.90 snapshot
   field to the `T_phi90` array stored inside that run's `fields.npz`. This is a
   bit-identity check against a real archived result, not against a fresh
   production run started by this script.

   Read/stop convention for every number it prints: the field is T at the END of
   the first outer step whose mean part melt fraction reaches 0.90, cast to
   float32 exactly as `rfam_eqs_coupled.py:3181` stores it; sigma_T is
   `ui_rms_part * (T_bar_part - 23 C)` on that field.

2. `score_historical_arms` takes the ACTUAL stored 4-bits-per-pixel maps of the
   window-reselection winners and scores them under the shape-fidelity objective
   J on the prototype engine, each at its OWN optimal stop. Read/stop convention:
   t_stop = argmin over the stored trajectory of J, per arm.

Run:
  ./.venv312/bin/python -m adjoint2d.verify_hist gate  <out.json>
  ./.venv312/bin/python -m adjoint2d.verify_hist arms  <out.json>
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import forward as fwd, objective as obj
from . import shape_objective as so
from .pins import build_case, load_cfg
from .prod import rfam

MAIN = Path("/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research"
            "/research/binderjet/code/geo-prewarp")
GDR = MAIN / "outputs_eqs/geometry_dual_readstate/runs"
FCC = MAIN / "outputs_eqs/fgm_calibrated_control/runs"
CFGD = MAIN / "outputs_eqs/fgm_calibrated_control/configs"

# The window-reselection winners of FGM_WINDOW_RESELECTION.md, with the stored
# map each winning gain actually produced.
WINNERS = {
    "square": ("square_m0p8775", 0.9033, FCC / "square/map_m0p9033/fgm_baseline_T_phi90_4bpp_mag0p90.npz"),
    "triangle": ("triangle_m0p1090", 1.2164, FCC / "triangle/map_m1p2164/fgm_baseline_T_phi90_4bpp_mag1p22.npz"),
    "cross": ("cross_m1p0927", 1.1855, FCC / "cross/map_m1p1855/fgm_baseline_T_phi90_4bpp_mag1p19.npz"),
    "L_shape": ("L_shape_m0p1110", 0.1110, FCC / "L_shape/map_m0p1110/fgm_baseline_T_phi90_4bpp_mag0p11.npz"),
}
# The old {0.30 .. 0.85} grid arm that the geometry_dual_readstate campaign ran.
OLDGRID = {s: GDR / f"{s}/map_m0p85/fgm_baseline_T_phi90_4bpp_mag0p85.npz"
           for s in WINNERS}


def load_stored_map(case, npz_path: Path, cfg: dict) -> np.ndarray:
    """The stored map exactly as the production engine loads it."""
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["fgm_feedback"] = {"enabled": True, "saturation_map_npz": str(npz_path),
                            "magnitude": 1.0, "baseline_saturation": 0.5,
                            "iterate": False}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fb = rfam._FgmFeedback.from_config(cfg2, case.x, case.y, case.part_mask)
    return np.asarray(fb.sat_map, dtype=np.float64)


def sigma_T_of(T: np.ndarray, case) -> dict:
    tp = np.asarray(T, dtype=np.float64)[case.part_mask]
    tbar = float(np.mean(tp))
    ui = float(np.sqrt(np.mean((tp - tbar) ** 2)) / (tbar - case.pins.ambient_c))
    return {"T_bar_part_c": tbar, "ui_rms_part": ui,
            "sigma_T_c": ui * (tbar - case.pins.ambient_c)}


# ---------------------------------------------------------------------------
# 1. bit-identity against a stored historical run
# ---------------------------------------------------------------------------

def stored_run_gate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    cfg = load_cfg(run_dir / "used_config.yaml")
    case = build_case(cfg)
    fields = np.load(run_dir / "fields.npz")
    T_stored = np.asarray(fields["T_phi90"])

    fgm = cfg.get("fgm_feedback", {}) or {}
    if bool(fgm.get("enabled", False)):
        s = load_stored_map(case, Path(str(fgm["saturation_map_npz"])), cfg)
        eps_covary = True
    else:
        s = np.ones(case.part_mask.shape, dtype=np.float64)
        eps_covary = False

    # `_FgmFeedback.effective_fill` casts to float32 ONLY when the hook is
    # enabled (`rfam_eqs_coupled.py:416`); with FGM disabled it returns
    # fill_frac untouched and the production conductivity array stays float64.
    # Reproducing that branch is part of the identity claim.
    t0 = time.perf_counter()
    tr = fwd.forward(case, s, float32_sat=eps_covary, stop_after_phi=None,
                     eps_covary=eps_covary, keep_checkpoints=False)
    wall = time.perf_counter() - t0

    rs = obj.read_states(tr)
    if rs.melt_onset_index is None:
        raise obj.MeltNotReached(str(run_dir))
    T_proto = tr.T_at_end(rs.melt_onset_index).astype(np.float32)

    d = np.abs(T_proto.astype(np.float64) - T_stored.astype(np.float64))
    got = sigma_T_of(T_proto, case)
    ref = sigma_T_of(T_stored, case)
    return {
        "run_dir": str(run_dir),
        "fgm_enabled": bool(fgm.get("enabled", False)),
        "map_npz": str(fgm.get("saturation_map_npz", "")) if fgm.get("enabled", False) else None,
        "melt_onset_outer_index": int(rs.melt_onset_index),
        "melt_onset_time_s": float(tr.time_s[rs.melt_onset_index]),
        "T_phi90_max_abs_diff_c": float(d.max()),
        "T_phi90_max_abs_diff_part_c": float(d[case.part_mask].max()),
        "prototype": got,
        "stored": ref,
        "sigma_T_abs_diff_c": abs(got["sigma_T_c"] - ref["sigma_T_c"]),
        "PASS_bit_identical": bool(d.max() == 0.0),
        "P_abs_state_B_W_per_m": tr.P_abs_B,
        "wall_s": wall,
    }


# ---------------------------------------------------------------------------
# 2. the historical arms under the shape objective
# ---------------------------------------------------------------------------

PATIENCE = 250


def _score(case, s, eps_covary: bool) -> dict:
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE,
                     eps_covary=eps_covary)
    m = so.full_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    return m


def score_historical_arms(shape: str) -> dict:
    cfg_name, gain, win_npz = WINNERS[shape]
    cfg = load_cfg(CFGD / f"{cfg_name}.yaml")
    case = build_case(cfg)
    pm = case.part_mask

    s_win = load_stored_map(case, win_npz, cfg)
    s_old = load_stored_map(case, OLDGRID[shape], cfg)
    win_out1 = np.where(pm, s_win, 1.0)

    arms = {
        # exactly the historical injection: stored 4 bits per pixel map, zero
        # outside the part, permittivity co-varying
        "hist_win_asstored_eps": _score(case, s_win, True),
        # same map, prototype boundary convention (nominal fill outside)
        "hist_win_outside1_eps": _score(case, win_out1, True),
        # same map, actuator matched to the adjoint arm (conductivity only)
        "hist_win_outside1_sig": _score(case, win_out1, False),
        # the old {0.30..0.85} grid arm of geometry_dual_readstate
        "hist_m0p85_asstored_eps": _score(case, s_old, True),
    }
    for k, v in arms.items():
        v["arm"] = k
    return {
        "shape": shape, "config": str(CFGD / f"{cfg_name}.yaml"),
        "window_selected_gain": gain, "winner_map_npz": str(win_npz),
        "oldgrid_map_npz": str(OLDGRID[shape]),
        "n_part_cells": case.n_part,
        "winner_map_stats": {
            "mean_in_part": float(np.mean(s_win[pm])),
            "min_in_part": float(np.min(s_win[pm])),
            "max_in_part": float(np.max(s_win[pm])),
            "mean_outside_part": float(np.mean(s_win[~pm])),
            "max_outside_part": float(np.max(s_win[~pm])),
            "n_outside_cells_with_geometry_fill": int(np.sum((~pm) & (case.fill_frac > 0.0))),
        },
        "arms": arms,
    }


def main(mode: str, out_path: str) -> dict:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if mode == "gate":
        res = {"gates": [stored_run_gate(GDR / p) for p in
                         ("square/baseline", "square/fgm_m0p85",
                          "triangle/fgm_m0p85", "cross/fgm_m0p85")]}
        for g in res["gates"]:
            print(f"{Path(g['run_dir']).parent.name:10s}/{Path(g['run_dir']).name:12s} "
                  f"idx {g['melt_onset_outer_index']:4d}  "
                  f"maxdiff {g['T_phi90_max_abs_diff_c']:.3e}  "
                  f"sigma_T proto {g['prototype']['sigma_T_c']:.12f} "
                  f"stored {g['stored']['sigma_T_c']:.12f}  "
                  f"PASS={g['PASS_bit_identical']}")
    elif mode == "arms":
        res = {s: score_historical_arms(s) for s in WINNERS}
        for s, r in res.items():
            print(f"=== {s}  gain {r['window_selected_gain']}")
            for k, m in r["arms"].items():
                print(f"  {k:26s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
                      f"grow {m['bed_melt_pct_of_part']:6.2f}  "
                      f"under {m['part_under_melt_pct']:6.2f}  "
                      f"stop {m['t_stop_index']:4d} ({m['t_stop_s']:6.1f} s)  "
                      f"P_abs {m['P_abs_W_per_m']:6.1f}")
    else:  # pragma: no cover
        raise ValueError(mode)
    out.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
