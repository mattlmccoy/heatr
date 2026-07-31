"""L0 gate: prove the prototype forward is BIT-IDENTICAL to the production march.

The precedent is the meshcheck work, which required `max|diff| = 0.000e+00`
against the production engine before drawing any conclusion. Without this gate,
a finite-difference-verified gradient only proves the prototype differentiated
its own re-implementation correctly.

What is compared, on the pinned configuration and a NON-UNIFORM saturation map:
  * every outer step of the four part-history series the campaign metrics are
    built from (mean part temperature, uniformity index root-mean-square,
    mean part melt fraction, mean part relative density);
  * the final full fields T, phi, rho and the electrical field Q_rf.

Run:  ./.venv312/bin/python -m adjoint2d.gate_l0 <config.yaml> <out.json>
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import forward as fwd
from .pins import build_case, load_cfg
from .prod import rfam


def make_probe_sat(case, seed: int = 20260731) -> np.ndarray:
    """A deliberately non-uniform, non-symmetric saturation map on the part."""
    rng = np.random.default_rng(seed)
    ny, nx = case.part_mask.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    smooth = 0.85 + 0.30 * np.cos(2.4 * xx + 0.7) * np.sin(1.9 * yy - 0.3) + 0.10 * xx
    sat = np.ones((ny, nx), dtype=np.float64)
    sat[case.part_mask] = smooth[case.part_mask] + 0.05 * rng.standard_normal(int(case.part_mask.sum()))
    return np.clip(sat, 0.0, 1.5)


def write_direct_npz(path: Path, sat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sat_map=sat.astype(np.float32))


def write_levelmap_npz(path: Path, sat: np.ndarray, bpp: int = 4) -> np.ndarray:
    """Write a map in the format the production `saturation_map_npz` hook reads.

    Returns the saturation the production loader will reconstruct, so the
    prototype can be driven with exactly the same numbers.
    """
    max_val = float((1 << bpp) - 1)
    lm = np.clip(np.round(np.asarray(sat) * max_val), 0, max_val).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, level_map=lm, bpp=np.array(bpp, dtype=np.int32))
    return np.clip(lm.astype(np.float32) / np.float32(max_val), 0.0, 1.0).astype(np.float64)


def run_gate(cfg_path: Path, out_json: Path, n_steps: int | None = None,
             mode: str = "direct") -> dict:
    cfg_path = Path(cfg_path).resolve()
    out_json = Path(out_json).resolve()
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    if n_steps is not None:
        cfg["thermal"]["n_steps"] = int(n_steps)
        case = build_case(cfg)

    sat = make_probe_sat(case)
    cfg_run = json.loads(json.dumps(cfg))
    if mode == "direct":
        npz = out_json.parent / f"{out_json.stem}_sat.npz"
        write_direct_npz(npz, sat)
        cfg_run["fgm_feedback"] = {
            "enabled": True,
            "sat_map_npz_direct": str(npz),
            "sat_max": 1.5,
            "iterate": False,
        }
        # The production engine loads the map as float32 and clips it to sat_max.
        s_used = np.clip(sat.astype(np.float32), 0.0, np.float32(1.5)).astype(np.float64)
        eps_covary = False
    elif mode == "npz_covary":
        npz = out_json.parent / f"{out_json.stem}_levelmap.npz"
        s_used = write_levelmap_npz(npz, np.clip(sat, 0.0, 1.0), bpp=4)
        cfg_run["fgm_feedback"] = {
            "enabled": True,
            "saturation_map_npz": str(npz),
            "magnitude": 1.0,
            "baseline_saturation": 0.5,
            "iterate": False,
        }
        eps_covary = True
    else:  # pragma: no cover
        raise ValueError(mode)

    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        state, summary, hist = rfam.run_sim(cfg_run)[:3]
    t_prod = time.perf_counter() - t0

    t1 = time.perf_counter()
    tr = fwd.forward(case, s_used, float32_sat=True, stop_after_phi=None,
                     eps_covary=eps_covary,
                     n_steps=int(cfg_run["thermal"]["n_steps"]))
    t_proto = time.perf_counter() - t1

    def dmax(a, b):
        return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))

    res = {
        "config": str(cfg_path),
        "mode": mode,
        "n_steps": int(cfg_run["thermal"]["n_steps"]),
        "n_substeps": case.pins.n_substeps,
        "n_part_cells": case.n_part,
        "sat_range_in_part": [float(s_used[case.part_mask].min()), float(s_used[case.part_mask].max())],
        "series_max_abs_diff": {
            "mean_T_part_c": dmax(tr.mean_T_part_c, hist["mean_T_part_c"]),
            "ui_rms_part": dmax(tr.ui_rms_part, hist["ui_rms_part"]),
            "mean_phi_part": dmax(tr.mean_phi_part, hist["mean_phi_part"]),
            "mean_rho_rel_part": dmax(tr.mean_rho_rel_part, hist["mean_rho_rel_part"]),
        },
        "field_max_abs_diff": {
            "T_final": dmax(tr.T_final, state.T),
            "phi_final": dmax(tr.phi_final, state.phi),
            "rho_final": dmax(tr.rho_final, state.rho_rel),
            "Qrf_final": dmax(tr.state_b.Qrf, state.Qrf),
        },
        "gates": {
            "frac_cells_dT_clipped_max": tr.frac_dT_clipped_max,
            "frac_cells_temp_cap_max": tr.frac_temp_cap_max,
            "frac_part_at_qrf_cap": tr.frac_qrf_cap,
        },
        "P_abs_state_A_W_per_m": tr.P_abs_A,
        "P_abs_state_B_W_per_m": tr.P_abs_B,
        "wall_s_production": t_prod,
        "wall_s_prototype": t_proto,
        "speedup": t_prod / max(t_proto, 1e-9),
    }
    worst = max(list(res["series_max_abs_diff"].values()) + list(res["field_max_abs_diff"].values()))
    res["worst_max_abs_diff"] = worst
    res["PASS"] = bool(worst == 0.0)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    cfgp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    ns = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    md = sys.argv[4] if len(sys.argv) > 4 else "direct"
    r = run_gate(cfgp, outp, ns, mode=md)
    print(json.dumps(r, indent=2))
