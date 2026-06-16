#!/usr/bin/env python3
"""Ceiling-respecting L-shape orientation sigma_T sweep.

Corrects the committed run lshape_orient_20260304_6min, which drove the part to
maxT=352 C (>>250 C ceiling) with a 26% energy-balance residual.

Each angle is a plain static run_sim at a FIXED ceiling-respecting exposure
(n_steps set so the angle-0 maxT just reaches ~250 C). We then report, on the
FINAL T field over the part mask:
    sigma_T, maxT (confirm <250 C), phi-bar reached, residual %, dT-clip frac.

phi-bar=0.90 is NOT reachable under the ceiling for the L-shape, so we measure
on the final field (T_phi90 would falsely fall back to the same field anyway).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
BASE_CFG = ROOT / "configs" / "_tmp_L_orient_singlebase.yaml"
OUT_ROOT = ROOT / "outputs_eqs" / "runs" / "L_shape" / "orientation_sigmaT_ceiling" / "experimental"
N_STEPS = 550  # ceiling-respecting exposure (angle-0 maxT ~= 250 C)


def run_angle(angle_deg: float) -> dict:
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["geometry"]["part"]["rotation_deg"] = float(angle_deg)
    cfg["thermal"]["n_steps"] = N_STEPS
    tag = f"ang{angle_deg:05.1f}".replace(".", "p")
    run_dir = OUT_ROOT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_in.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)

    cmd = [sys.executable, str(ROOT / "rfam_eqs_coupled.py"),
           "--config", str(cfg_path), "--output-dir", str(run_dir)]
    print(f"\n=== angle {angle_deg} deg -> {run_dir} ===", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:]); print(res.stderr[-2000:])
        raise RuntimeError(f"run failed at angle {angle_deg}")

    summ = json.load(open(run_dir / "summary.json"))
    fields = np.load(run_dir / "fields.npz")
    part = fields["part_mask"].astype(bool)
    Tfin = fields["T"][part]
    Edop = summ.get("energy_doped_total_J_per_m", 0.0) or 1.0
    resid = summ.get("energy_balance_residual_final_J_per_m", float("nan"))
    rec = {
        "angle_deg": float(angle_deg),
        "sigma_T_final_C": float(np.std(Tfin)),
        "mean_T_final_C": float(np.mean(Tfin)),
        "max_T_part_final_c": float(summ.get("max_T_part_final_c", np.max(Tfin))),
        "phi_bar_final": float(summ.get("mean_phi_part_final", float("nan"))),
        "frac_melt_ref": float(summ.get("frac_part_ge_melt_ref", float("nan"))),
        "residual_pct": float(100.0 * resid / Edop),
        "dT_clip_frac_final": float(summ.get("frac_cells_dT_clipped_final", float("nan"))),
        "mean_rho_rel_final": float(summ.get("mean_rho_rel_part_final", float("nan"))),
        "run_dir": str(run_dir),
    }
    print(f"  sigma_T={rec['sigma_T_final_C']:.2f}  maxT={rec['max_T_part_final_c']:.1f}  "
          f"phi_bar={rec['phi_bar_final']:.3f}  resid={rec['residual_pct']:.2f}%  "
          f"dTclip={rec['dT_clip_frac_final']:.3g}", flush=True)
    return rec


def main():
    angles = [float(a) for a in sys.argv[1:]] or [0.0, 45.0, 90.0, 135.0, 140.0]
    records = [run_angle(a) for a in angles]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUT_ROOT / "sigmaT_vs_angle_ceiling.json"
    existing = {}
    if out_json.exists():
        for r in json.load(open(out_json)):
            existing[r["angle_deg"]] = r
    for r in records:
        existing[r["angle_deg"]] = r
    merged = [existing[k] for k in sorted(existing)]
    json.dump({"n_steps": N_STEPS, "dt_s": 0.5, "records": merged}, open(out_json, "w"), indent=2)
    print(f"\nWrote {out_json}")
    s0 = next((r["sigma_T_final_C"] for r in merged if r["angle_deg"] == 0.0), None)
    print("\nangle | sigma_T C | maxT C | phi-bar | resid % | %red_vs_0")
    for r in merged:
        red = "" if s0 in (None, 0) else f"{(s0 - r['sigma_T_final_C']) / s0 * 100:+6.1f}%"
        print(f"{r['angle_deg']:6.1f} | {r['sigma_T_final_C']:8.2f} | {r['max_T_part_final_c']:6.1f} | "
              f"{r['phi_bar_final']:6.3f} | {r['residual_pct']:6.2f} | {red}")


if __name__ == "__main__":
    main()
