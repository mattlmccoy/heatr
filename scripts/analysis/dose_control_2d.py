#!/usr/bin/env python3
"""Driver for the dose-controlled three-arm 2-D grading study.

Three arms per shape at the SAME voltage, SAME grid, SAME exposure
horizon, on equal-cross-sectional-area cross-sections (A* = 314.159 mm^2):

  A  uniform baseline at the reference saturation (sat 1.0, sigma 0.04 S/m)
  B  graded, one-shot proportional inverse-field FGM (framework M1)
  C  flat control, uniform saturation at arm B's OWN part-mask mean saturation

Arm C is implemented by feeding arm B's own map through the identical
`fgm_feedback` code path with `magnitude: 0.0` and
`baseline_saturation: <mean sat of arm B>`, which makes the applied saturation
exactly uniform at that mean while leaving the eps_r blending path identical to
arm B's.

Resumable: every stage writes a marker artifact and is skipped if already done.
Per-shape results are written to disk after EVERY shape.

All artifacts are grid-scoped (`runs_g<GRID>` etc.). The 240 grid is KNOWN
BROKEN: the coupled thermal step blows up at melt onset (max dT_raw 1557 C
against a 10 C clip, energy residual 2.3e5 times the integrated dose, phi_bar
advancing 0.019 -> 0.953 in one outer step). Not shape-specific, not cured by
refining dt. The production grid is 160, verified clean on square, circle,
diamond, star6 and equilateral_triangle.

Usage:
    python3 scripts/analysis/dose_control_2d.py --shapes square
    python3 scripts/analysis/dose_control_2d.py --shapes nine
    python3 scripts/analysis/dose_control_2d.py --shapes all --grid 160
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import dose_control_lib as L  # noqa: E402

OUT = ROOT / "outputs_eqs" / "dose_control_2d"
BASE_CFG = ROOT / "outputs_eqs" / "fgm_dosecheck" / "configs" / "circle_baseline_voltage.yaml"


def dirs_for_grid(grid: int) -> dict[str, Path]:
    """Grid-scoped artifact directories.

    Every artifact path carries the grid in its name so a run at one resolution
    can never be resumed, or averaged, as a run at another. The 240-grid
    artifacts are retained as the evidence for why 240 was abandoned.
    """
    return {
        "runs": OUT / f"runs_g{grid}",
        "results": OUT / f"results_g{grid}",
        "configs": OUT / f"configs_g{grid}",
        "logs": OUT / f"logs_g{grid}",
    }


def flat_control_fgm(fgm_b: dict, mean_sat: float) -> dict:
    """Arm C: arm B's OWN map through the identical loader, flattened.

    magnitude 0.0 makes the loader collapse the applied saturation to
    ``baseline_saturation`` at every cell, exactly uniform at arm B's own
    part-mask mean, while leaving the eps_r blending path identical to arm B's.
    """
    fgm_c = dict(fgm_b)
    fgm_c["magnitude"] = 0.0
    fgm_c["baseline_saturation"] = float(mean_sat)
    return fgm_c


GRID = 160
N_STEPS = 2400          # 1200 s horizon at dt 0.5 s
# The probe must run PAST the first update_interval (20) tick. The solver builds
# the initial sigma with sub-pixel fill_frac anti-aliasing but the periodic
# EQS refresh at step 20 hard-stamps sigma[part_mask] = sigma_d0 * sat, which
# raises absorbed power by roughly 11 % at the 240 grid and then stays there for
# the rest of the run. Calibrating the voltage on a 2-step probe would therefore
# target a power level the run only holds for its first 10 s. This is
# pre-existing solver behaviour, present in every published run at every grid;
# it is calibrated around here, not modified.
PROBE_STEPS = 25
V_REF = 860.0
TARGET_P_W_PER_M = 500.0

# shape -> (base_width_m, base_height_m, role)
SHAPES: dict[str, tuple[float, float, str]] = {
    "square":               (0.020, 0.020, "control"),
    "circle":               (0.020, 0.020, "test"),
    "hexagon":              (0.020, 0.020, "test"),
    "pentagon":             (0.020, 0.020, "test"),
    "ellipse":              (0.020, 0.012, "test"),
    "octagon":              (0.020, 0.020, "test"),
    "star6":                (0.020, 0.020, "test"),
    "diamond":              (0.020, 0.020, "test"),
    "triangle":             (0.020, 0.020, "test"),
    "equilateral_triangle": (0.020, 0.020, "test"),
}
# Ordered so the square control runs first (gate G3).
ORDER = list(SHAPES.keys())


# ---------------------------------------------------------------------------
# config construction
# ---------------------------------------------------------------------------

def _base() -> dict:
    return yaml.safe_load(BASE_CFG.read_text())


def build_config(
    shape: str,
    w_m: float,
    h_m: float,
    n_steps: int,
    voltage_v: Optional[float],
    enforce: bool,
    fgm: Optional[dict] = None,
) -> dict:
    cfg = _base()
    g = cfg["geometry"]
    g["grid_nx"] = GRID
    g["grid_ny"] = GRID
    g["part"]["shape"] = shape
    g["part"]["width"] = float(w_m)
    g["part"]["height"] = float(h_m)
    g["part"]["rotation_deg"] = 0.0
    g["part"]["center_x"] = 0.0
    g["part"]["center_y"] = 0.0

    e = cfg["electric"]
    e["enforce_generator_power"] = bool(enforce)
    if voltage_v is not None:
        e["voltage_v"] = float(voltage_v)
    else:
        e["voltage_v"] = V_REF

    cfg["thermal"]["n_steps"] = int(n_steps)

    if fgm is None:
        cfg["fgm_feedback"] = {"enabled": False}
    else:
        cfg["fgm_feedback"] = fgm
    return cfg


def write_cfg(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    return path


def launch(cfg_path: Path, out_dir: Path, log_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "rfam_eqs_coupled.py"),
        "--config", str(cfg_path),
        "--output-dir", str(out_dir),
    ]
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                              stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"solver failed rc={proc.returncode} for {out_dir}; see {log_path}"
        )
    print(f"    [{dt/60:.1f} min] {out_dir}", flush=True)


def load_run(d: Path) -> tuple[dict, dict]:
    summ = json.loads((d / "summary.json").read_text())
    ts = json.loads((d / "time_series.json").read_text())
    return summ, ts


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_probe(shape: str, design: dict) -> float:
    """2-step enforced-power probe -> the voltage that makes arm A absorb 500 W/m."""
    D_ = dirs_for_grid(GRID)
    d = D_["runs"] / shape / "probe"
    marker = d / "summary.json"
    if not marker.exists():
        cfg = build_config(shape, design["new_w_m"], design["new_h_m"],
                           PROBE_STEPS, None, enforce=True)
        write_cfg(cfg, D_["configs"] / f"{shape}_probe.yaml")
        launch(D_["configs"] / f"{shape}_probe.yaml", d,
               D_["logs"] / f"{shape}_probe.log")
    summ = json.loads(marker.read_text())
    scale = float(summ["qrf_scale_applied_final"])
    return V_REF * math.sqrt(scale)


def stage_arm(shape: str, arm: str, design: dict, voltage: float,
              fgm: Optional[dict], n_steps: int = N_STEPS) -> Path:
    D_ = dirs_for_grid(GRID)
    d = D_["runs"] / shape / arm
    if not (d / "summary.json").exists():
        cfg = build_config(shape, design["new_w_m"], design["new_h_m"],
                           n_steps, voltage, enforce=False, fgm=fgm)
        write_cfg(cfg, D_["configs"] / f"{shape}_{arm}.yaml")
        launch(D_["configs"] / f"{shape}_{arm}.yaml", d,
               D_["logs"] / f"{shape}_{arm}.log")
    return d


def stage_map(shape: str, arm_a_dir: Path) -> Path:
    """Framework M1: one-shot proportional inverse-field map from arm A."""
    from fgm_generator import generate_fgm

    map_dir = dirs_for_grid(GRID)["runs"] / shape / "map_M1"
    existing = sorted(map_dir.glob("fgm_*.npz"))
    if existing:
        return existing[0]

    fields = np.load(arm_a_dir / "fields.npz", allow_pickle=True)
    if "T_phi90_reached" in fields:
        reached = bool(np.asarray(fields["T_phi90_reached"]).item())
        if not reached:
            raise RuntimeError(
                f"{shape}: arm A never reached phi_bar=0.90, so its T_phi90 field "
                f"is a silent fallback to final T. Refusing to build the M1 map."
            )
    map_dir.mkdir(parents=True, exist_ok=True)
    res = generate_fgm(
        run_output_dir=arm_a_dir,
        bpp=4,
        proxy_field="T_phi90",
        invert=True,
        magnitude=1.0,
        baseline_saturation=0.5,
        dead_band=0.05,
        emit_formats=("npz", "json", "png"),
        output_dir=map_dir,
    )
    return Path(res["npz_path"])


def applied_mean_sat(cfg: dict, shape: str, design: dict) -> dict[str, float]:
    """Part-mask statistics of the saturation map AS THE SOLVER APPLIES IT.

    Uses the solver's own loader so no value is re-derived by hand.
    """
    from rfam_eqs_coupled import _FgmFeedback

    x, y = L.make_grid(GRID, GRID)
    mask = L.rasterized_mask(shape, design["new_w_m"], design["new_h_m"], x, y)
    fb = _FgmFeedback.from_config(cfg, x, y, mask)
    v = fb.sat_map[mask]
    return {
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "std": float(v.std()),
        "n_cells": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# per-shape orchestration
# ---------------------------------------------------------------------------

def run_shape(shape: str, design_240: dict) -> dict:
    res_path = dirs_for_grid(GRID)["results"] / f"{shape}.json"
    if res_path.exists():
        print(f"[{shape}] resume: results already on disk")
        return json.loads(res_path.read_text())

    design = design_240[shape]
    role = SHAPES[shape][2]
    print(f"[{shape}] role={role}  w={design['new_w_mm']:.3f} mm  "
          f"A={design['new_A_mm2']:.2f} mm^2 ({design['err_pct']:+.3f} %)")

    voltage = stage_probe(shape, design)
    print(f"[{shape}] probe voltage = {voltage:.1f} V")

    # ---- arm A -----------------------------------------------------------
    a_dir = stage_arm(shape, "armA_uniform", design, voltage, fgm=None)
    a_summ, a_ts = load_run(a_dir)

    # ---- M1 map from arm A ----------------------------------------------
    map_npz = stage_map(shape, a_dir)

    # ---- arm B -----------------------------------------------------------
    fgm_b = {
        "enabled": True,
        "saturation_map_npz": str(map_npz),
        "magnitude": 1.0,
        "baseline_saturation": 0.5,
        "invert": True,
        "iterate": False,
        "proxy_field": "T_phi90",
        "bpp": 4,
    }
    cfg_b_preview = build_config(shape, design["new_w_m"], design["new_h_m"],
                                 N_STEPS, voltage, enforce=False, fgm=fgm_b)
    sat_b = applied_mean_sat(cfg_b_preview, shape, design)
    print(f"[{shape}] arm B applied sat: mean={sat_b['mean']:.4f} "
          f"range=[{sat_b['min']:.3f}, {sat_b['max']:.3f}]")
    b_dir = stage_arm(shape, "armB_graded", design, voltage, fgm=fgm_b)
    b_summ, b_ts = load_run(b_dir)

    # ---- arm C: flat at arm B's own mean sat ------------------------------
    fgm_c = flat_control_fgm(fgm_b, sat_b["mean"])
    cfg_c_preview = build_config(shape, design["new_w_m"], design["new_h_m"],
                                 N_STEPS, voltage, enforce=False, fgm=fgm_c)
    sat_c = applied_mean_sat(cfg_c_preview, shape, design)
    if sat_c["std"] > 1e-6 or abs(sat_c["mean"] - sat_b["mean"]) > 1e-6:
        raise RuntimeError(
            f"{shape}: arm C map is not uniform at arm B's mean "
            f"(std={sat_c['std']:.2e}, mean={sat_c['mean']:.6f} vs "
            f"{sat_b['mean']:.6f})"
        )
    c_dir = stage_arm(shape, "armC_flat", design, voltage, fgm=fgm_c)
    c_summ, c_ts = load_run(c_dir)

    # ---- analysis --------------------------------------------------------
    arms = {}
    for name, d, summ, ts, sat in (
        ("A", a_dir, a_summ, a_ts, {"mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0}),
        ("B", b_dir, b_summ, b_ts, sat_b),
        ("C", c_dir, c_summ, c_ts, sat_c),
    ):
        at90 = L.read_at_phi90(ts)
        arms[name] = {
            "run_dir": str(d.relative_to(ROOT)),
            "at_phi90": at90,
            # G2a, standing requirement. Both windows reported: the read-point
            # window certifies the sigma_T measurement, the full-run window shows
            # what happens in the post-melt overshoot tail.
            "stability_gate": L.stability_gate(
                ts, upto_index=at90["index"] if at90["reached"] else None),
            "stability_gate_full_run": L.stability_gate(ts),
            "peak_T_over_run_c": L.peak_T_over_run(ts),
            "P_abs_W_per_m": float(summ["integrated_power_doped_W_per_m"]),
            "sigma_mean_S_per_m": 0.04 * sat["mean"],
            "sat": sat,
            # G2 gate quantities, reported on every solve
            "frac_part_at_qrf_cap_final": float(summ["frac_part_at_qrf_cap_final"]),
            "frac_cells_dT_clipped_final": float(summ["frac_cells_dT_clipped_final"]),
            "frac_cells_dT_clipped_mean": float(summ["frac_cells_dT_clipped_mean"]),
            "qrf_part_max_w_per_m3": float(summ["qrf_part_max_w_per_m3"]),
            "max_qrf_cap_w_per_m3": 1.0e11,
            "energy_doped_total_J_per_m": float(summ["energy_doped_total_J_per_m"]),
            "energy_balance_residual_J_per_m": float(
                summ["energy_balance_residual_final_J_per_m"]),
            "energy_residual_pct_of_dose": (
                100.0 * float(summ["energy_balance_residual_final_J_per_m"])
                / max(float(summ["energy_doped_total_J_per_m"]), 1e-12)
            ),
            "frac_part_ge_melt_ref_final": float(summ["frac_part_ge_melt_ref"]),
            "mean_rho_rel_part_final": float(summ["mean_rho_rel_part_final"]),
        }

    sA = arms["A"]["at_phi90"]["sigma_T_c"]
    sB = arms["B"]["at_phi90"]["sigma_T_c"]
    sC = arms["C"]["at_phi90"]["sigma_T_c"]
    total = L.benefit_pct(sA, sB)
    grading = L.benefit_pct(sC, sB)
    pA = arms["A"]["P_abs_W_per_m"]
    for n in ("B", "C"):
        arms[n]["P_dev_pct_vs_A"] = 100.0 * (arms[n]["P_abs_W_per_m"] - pA) / pA
        arms[n]["P_flag_gt20pct"] = abs(arms[n]["P_dev_pct_vs_A"]) > 20.0
    arms["A"]["P_dev_pct_vs_A"] = 0.0
    arms["A"]["P_flag_gt20pct"] = False

    if grading is None:
        verdict = "NOT REACHED (excluded from ratios)"
    elif grading <= -25.0:
        verdict = "SURVIVES (strong)"
    elif grading <= -10.0:
        verdict = "SURVIVES"
    elif grading < 0.0:
        verdict = "MARGINAL (does not survive)"
    else:
        verdict = "GRADING HARMFUL at matched dose"

    out = {
        "shape": shape,
        "role": role,
        "design": design,
        "grid_cells_per_axis": GRID,
        "voltage_v": voltage,
        "n_steps": N_STEPS,
        "grid": GRID,
        "map_npz": str(map_npz.relative_to(ROOT)),
        "arms": arms,
        "sigma_T_A_c": sA,
        "sigma_T_B_c": sB,
        "sigma_T_C_c": sC,
        "total_benefit_pct": total,
        "grading_only_benefit_pct": grading,
        "verdict": verdict,
    }
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(out, indent=2))
    print(f"[{shape}] sigma_T A/B/C = "
          f"{sA if sA is None else round(sA,2)} / "
          f"{sB if sB is None else round(sB,2)} / "
          f"{sC if sC is None else round(sC,2)} C   "
          f"total={total if total is None else round(total,1)} %  "
          f"grading-only={grading if grading is None else round(grading,1)} %  "
          f"-> {verdict}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="square",
                    help="comma list, or 'all', or 'nine' for the test shapes")
    ap.add_argument("--grid", type=int, default=GRID,
                    help="simulation grid (cells per axis). Artifacts are "
                         "grid-scoped; 240 is known broken at melt onset.")
    args = ap.parse_args()
    globals()["GRID"] = int(args.grid)

    design = json.loads((OUT / "design_table.json").read_text())[str(GRID)]

    if args.shapes == "all":
        shapes = ORDER
    elif args.shapes == "nine":
        shapes = [s for s in ORDER if SHAPES[s][2] == "test"]
    else:
        shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]

    for s in shapes:
        if s not in SHAPES:
            raise SystemExit(f"unknown shape {s!r}")
        run_shape(s, design)


if __name__ == "__main__":
    main()
