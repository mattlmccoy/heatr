#!/usr/bin/env python3
"""Phase 2: matched single-mask FGM comparison + edge-void test + wavelength check."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_assist_voids_pilot as P  # noqa: E402
import orchestrate_pilot as O  # noqa: E402

OUT = O.OUT
MELT_REF_C = 180.0
N = 900


def gen_fgm_map(baseline_dir: Path, magnitude: float) -> Path:
    """Single-mask FGM from baseline T_phi90 proxy (hot -> less dopant)."""
    cmd = [
        sys.executable, str(ROOT / "fgm_generator.py"), str(baseline_dir),
        "--bpp", "4", "--proxy", "T_phi90",
        "--magnitude", str(magnitude), "--baseline", "0.5",
        "--output-dir", str(baseline_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    npzs = sorted(baseline_dir.glob("fgm_*T_phi90*.npz"))
    if not npzs:
        npzs = sorted(baseline_dir.glob("fgm_*.npz"))
    return npzs[-1]


def main():
    base = O.base_square_cfg(N)
    nominal = P.nominal_part_mask(base)
    results = json.load(open(OUT / "pilot_results.json"))
    baseline_dir = OUT / "baseline" / f"square_baseline_n{N}"

    # --- FGM matched correction ---
    for mag in (0.5, 0.8):
        npz = gen_fgm_map(baseline_dir, mag)
        cfg = copy.deepcopy(base)
        cfg["fgm_feedback"] = {
            "enabled": True,
            "saturation_map_npz": str(npz),
            "magnitude": 1.0,
            "baseline_saturation": 0.5,
            "bpp": 4,
        }
        label = f"fgm_singlemask_mag{mag:.1f}".replace(".", "p")
        fdir = OUT / "experimental" / f"square_{label}_n{N}"
        P.run_one(cfg, fdir, label)
        rec, *_ = P.analyze(fdir, nominal, MELT_REF_C, use_field="T")
        results[label] = rec

    # --- Edge voids: ring right at the boundary (8.5 mm) to attack bleed ---
    se = 0.0090
    edge_voids = [(se, se), (-se, se), (se, -se), (-se, -se),
                  (se, 0), (-se, 0), (0, se), (0, -se)]
    for d_mm in (1.0, 2.0):
        d = d_mm * 1e-3
        voids = [{"cx": cx, "cy": cy, "d": d} for cx, cy in edge_voids]
        cfg = P.make_void_cfg(base, voids)
        vm = O.void_cell_mask(cfg)
        label = f"void_d{d_mm:.0f}mm_edge8"
        vdir = OUT / "experimental" / f"square_{label}_n{N}"
        P.run_one(cfg, vdir, label)
        rec, *_ = P.analyze(vdir, nominal, MELT_REF_C, void_mask=vm, use_field="T")
        rec["n_voids"] = len(voids)
        rec["void_cells"] = int(vm.sum())
        results[label] = rec

    (OUT / "pilot_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k]["iou"] for k in results}, indent=2))


if __name__ == "__main__":
    main()
