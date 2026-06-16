#!/usr/bin/env python3
"""Multi-part study at FIXED DRIVE (enforce_generator_power=false).

Removes the 500 W dose-splitting confound of the main study: voltage is held
fixed, so each part's absorbed Qrf is set purely by the EQS field it sees. The
per-part absorbed power vs the isolated single-part value is then a direct
measure of mutual EM coupling (shielding <1, enhancement >1).

Outputs: outputs_eqs/runs/circle/multipart_interaction/<family>/fixeddrive_<run>.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE_CFG = ROOT / "configs" / "shape_circle_6min.yaml"
CFG_DIR = ROOT / "configs" / "_multipart_study"
RUN_ROOT = ROOT / "outputs_eqs" / "runs" / "circle" / "multipart_interaction"

DIAM = 0.012
N_STEPS = 120          # 60 s; field/Qrf is instantaneous, short run is enough
N_CIRCLE_PTS = 400


def base_cfg() -> dict:
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg["thermal"]["n_steps"] = N_STEPS
    cfg["electric"]["enforce_generator_power"] = False  # fixed-voltage drive
    return cfg


def circle_part(cx: float, cy: float) -> dict:
    return {"shape": "circle", "width": DIAM, "height": DIAM,
            "center_x": float(cx), "center_y": float(cy),
            "n_circle_pts": N_CIRCLE_PTS, "rotation_deg": 0.0}


def run(name: str, family: str, parts: list[dict]) -> Path:
    cfg = base_cfg()
    cfg["geometry"].pop("part", None)
    cfg["geometry"]["parts"] = parts
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_DIR / f"fixeddrive_{name}.yaml"
    yaml.safe_dump(cfg, cfg_path.open("w"), sort_keys=False)
    out_dir = RUN_ROOT / family / f"fixeddrive_{name}"
    cmd = [sys.executable, str(ROOT / "rfam_eqs_coupled.py"),
           "--config", str(cfg_path), "--output-dir", str(out_dir)]
    print(f"[run] fixeddrive {family}/{name} ({len(parts)} parts)", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        raise RuntimeError(name)
    return out_dir


def main() -> None:
    spacings = [0.014, 0.018, 0.024, 0.032]
    run("single_baseline", "baseline", [circle_part(0.0, 0.0)])
    for s in spacings:
        h = s / 2.0
        mm = int(round(s * 1000))
        run(f"pair_inline_s{mm}mm", "experimental", [circle_part(0.0, -h), circle_part(0.0, +h)])
        run(f"pair_side_s{mm}mm", "experimental", [circle_part(-h, 0.0), circle_part(+h, 0.0)])
    s3 = 0.018
    run("triple_inline_s18mm", "experimental",
        [circle_part(0.0, -s3), circle_part(0.0, 0.0), circle_part(0.0, +s3)])
    run("triple_side_s18mm", "experimental",
        [circle_part(-s3, 0.0), circle_part(0.0, 0.0), circle_part(+s3, 0.0)])
    print("fixeddrive done")


if __name__ == "__main__":
    main()
