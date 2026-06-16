#!/usr/bin/env python3
"""Drive a HEATR 2.5D orientation sweep of the L-shape and record the REAL
baseline sigma_T(angle) measured on the melt-state-normalized T field (T_phi90,
i.e. the T field at the first part-mean phi-bar = 0.90 crossing).

Each angle is a plain single run_sim (no optimizer block). We then load
fields.npz and compute sigma_T = std(T_phi90[part_mask]) in degrees C.
We also read time_series to confirm phi-bar actually crossed 0.90 (otherwise
T_phi90 falls back to final T and the point is flagged).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
BASE_CFG = ROOT / "configs" / "_tmp_L_orient_singlebase.yaml"
OUT_ROOT = ROOT / "outputs_eqs" / "runs" / "L_shape" / "orientation_sigmaT" / "baseline"


def run_angle(angle_deg: float) -> dict:
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["geometry"]["part"]["rotation_deg"] = float(angle_deg)
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
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        raise RuntimeError(f"run failed at angle {angle_deg}")

    # phi-bar crossing check
    ts = json.load(open(run_dir / "time_series.json"))
    phi_series = ts.get("mean_phi_part") or ts.get("phi_mean") or []
    phi_max = float(max(phi_series)) if phi_series else float("nan")
    crossed = phi_max >= 0.90

    fields = np.load(run_dir / "fields.npz")
    part = fields["part_mask"].astype(bool)
    t_phi90 = fields["T_phi90"]
    t_final = fields["T"]
    tp = t_phi90[part]
    sigma_T = float(np.std(tp))
    rec = {
        "angle_deg": float(angle_deg),
        "sigma_T_phi90_C": sigma_T,
        "sigma_T_final_C": float(np.std(t_final[part])),
        "mean_T_phi90_C": float(np.mean(tp)),
        "max_T_phi90_C": float(np.max(tp)),
        "phi_bar_max": phi_max,
        "phi90_crossed": bool(crossed),
        "run_dir": str(run_dir),
    }
    flag = "" if crossed else "  [!! phi-bar<0.90: T_phi90==final T]"
    print(f"  sigma_T(phi90)={sigma_T:.2f} C  meanT={rec['mean_T_phi90_C']:.1f}  "
          f"phi_max={phi_max:.3f}{flag}", flush=True)
    return rec


def main():
    angles = [float(a) for a in sys.argv[1:]]
    if not angles:
        angles = list(range(0, 181, 15))
    records = [run_angle(a) for a in angles]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUT_ROOT / "sigmaT_vs_angle.json"
    # merge with any existing records
    existing = {}
    if out_json.exists():
        for r in json.load(open(out_json)):
            existing[r["angle_deg"]] = r
    for r in records:
        existing[r["angle_deg"]] = r
    merged = [existing[k] for k in sorted(existing)]
    json.dump(merged, open(out_json, "w"), indent=2)
    print(f"\nWrote {out_json}")
    s0 = next((r["sigma_T_phi90_C"] for r in merged if r["angle_deg"] == 0.0), None)
    print("\nangle  sigma_T(phi90)  %red_vs_0  phi_crossed")
    for r in merged:
        red = "" if s0 in (None, 0) else f"{(s0 - r['sigma_T_phi90_C']) / s0 * 100:+6.1f}%"
        print(f"{r['angle_deg']:6.1f}  {r['sigma_T_phi90_C']:8.2f}      {red:>8}   {r['phi90_crossed']}")


if __name__ == "__main__":
    main()
