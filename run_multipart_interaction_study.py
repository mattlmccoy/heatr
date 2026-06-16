#!/usr/bin/env python3
"""Multi-part EM/thermal interaction study driver for HEATR (2.5D EQS).

Places N identical doped circles at controlled center-to-center spacing and
arrangement (inline = stacked along the applied-field axis y; side = along x,
perpendicular to the field). Each run reuses the shipped circle 6-min physics;
only geometry.parts and exposure are varied. Per-part metrics come from the
solver's own summary.part_stats; an isolated single-part run is the baseline.

Field axis note: electrodes mode=boundary top/bottom => E-field is along +y.
  inline  -> parts separated along y  (shadowing along field lines)
  side    -> parts separated along x  (lateral, perpendicular to field)

Usage: python run_multipart_interaction_study.py
Outputs land in outputs_eqs/runs/circle/multipart_interaction/<family>/<run>.
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE_CFG = ROOT / "configs" / "shape_circle_6min.yaml"
CFG_DIR = ROOT / "configs" / "_multipart_study"
RUN_ROOT = ROOT / "outputs_eqs" / "runs" / "circle" / "multipart_interaction"

DIAM = 0.012          # 12 mm circle
N_STEPS = 240         # 120 s exposure (dt=0.5)
N_CIRCLE_PTS = 400


def base_cfg() -> dict:
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg["thermal"]["n_steps"] = N_STEPS
    return cfg


def circle_part(cx: float, cy: float) -> dict:
    return {
        "shape": "circle",
        "width": DIAM,
        "height": DIAM,
        "center_x": float(cx),
        "center_y": float(cy),
        "n_circle_pts": N_CIRCLE_PTS,
        "rotation_deg": 0.0,
    }


def write_cfg(name: str, parts: list[dict]) -> Path:
    cfg = base_cfg()
    geom = cfg["geometry"]
    geom.pop("part", None)
    geom["parts"] = parts
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, path.open("w"), sort_keys=False)
    return path


def run(name: str, family: str, parts: list[dict]) -> Path:
    cfg_path = write_cfg(name, parts)
    out_dir = RUN_ROOT / family / name
    cmd = [
        sys.executable, str(ROOT / "rfam_eqs_coupled.py"),
        "--config", str(cfg_path),
        "--output-dir", str(out_dir),
    ]
    print(f"[run] {family}/{name}  ({len(parts)} part(s))", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        raise RuntimeError(f"run failed: {name}")
    return out_dir


def main() -> None:
    # center-to-center spacings (m). gap = spacing - DIAM.
    spacings = [0.014, 0.018, 0.024, 0.032]  # gaps 2,6,12,20 mm

    jobs: list[tuple[str, str, list[dict]]] = []

    # baseline: isolated single part
    jobs.append(("single_baseline", "baseline", [circle_part(0.0, 0.0)]))

    # 2-part inline (along field, y-axis) and side-by-side (x-axis)
    for s in spacings:
        h = s / 2.0
        mm = int(round(s * 1000))
        jobs.append((f"pair_inline_s{mm}mm", "experimental",
                     [circle_part(0.0, -h), circle_part(0.0, +h)]))
        jobs.append((f"pair_side_s{mm}mm", "experimental",
                     [circle_part(-h, 0.0), circle_part(+h, 0.0)]))

    # 3-part inline rows at a representative spacing (18 mm), both axes,
    # to capture center-vs-edge differential (asymmetric).
    s3 = 0.018
    jobs.append(("triple_side_s18mm", "experimental",
                 [circle_part(-s3, 0.0), circle_part(0.0, 0.0), circle_part(+s3, 0.0)]))
    jobs.append(("triple_inline_s18mm", "experimental",
                 [circle_part(0.0, -s3), circle_part(0.0, 0.0), circle_part(0.0, +s3)]))

    # 4-part square cluster at 18 mm pitch
    jobs.append(("quad_cluster_s18mm", "experimental",
                 [circle_part(-s3 / 2, -s3 / 2), circle_part(+s3 / 2, -s3 / 2),
                  circle_part(-s3 / 2, +s3 / 2), circle_part(+s3 / 2, +s3 / 2)]))

    out_dirs = []
    for name, family, parts in jobs:
        out_dirs.append(run(name, family, parts))
    print("\nAll runs complete:")
    for d in out_dirs:
        print("  ", d)


if __name__ == "__main__":
    main()
