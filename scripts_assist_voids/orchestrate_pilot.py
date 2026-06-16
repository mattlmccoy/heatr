#!/usr/bin/env python3
"""Orchestrate the assist-voids pilot: baseline + void variants + FGM match.

All variants share one exposure (matched dose). The as-built melt boundary is
read from the T_phi90 snapshot (T field at first phi_bar=0.90 crossing), so the
comparison is at the cure target, not at full melt.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_assist_voids_pilot as P  # noqa: E402  (same dir)

BASE_CFG = ROOT / "configs" / "shape_circle_6min.yaml"
OUT = ROOT / "outputs_eqs" / "runs" / "square" / "assist_voids"
MELT_REF_C = 180.0

# Square part, 20 mm, centred. Corners at (+-10, +-10) mm overshoot at phi90.
PART_HALF = 0.010  # m

def base_square_cfg(n_steps=720):
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["geometry"]["part"]["shape"] = "square"
    cfg["geometry"]["part"]["width"] = 0.020
    cfg["geometry"]["part"]["height"] = 0.020
    cfg["thermal"]["n_steps"] = n_steps
    return cfg


def void_pattern(d, layout):
    """Return list of void dicts. Bleed (melt overshoot into exterior powder)
    originates along the whole part edge, strongest at corners. Voids are
    placed in a ring just inside the 10 mm edge to suppress the boundary
    heat source -> pull the external melt boundary inward toward nominal.

    ``s`` = offset of the void ring from centre (m). Edge is at 10 mm.
    """
    s = 0.0075  # 7.5 mm -> ring sits ~2.5 mm inside the edge
    sc = 0.0080  # corners slightly further out (hottest)
    if layout == "ring8":
        cs = [(sc, sc), (-sc, sc), (sc, -sc), (-sc, -sc),   # 4 corners
              (s, 0), (-s, 0), (0, s), (0, -s)]              # 4 edge midpoints
    elif layout == "corners4":
        cs = [(sc, sc), (-sc, sc), (sc, -sc), (-sc, -sc)]
    elif layout == "ring12":
        cs = [(sc, sc), (-sc, sc), (sc, -sc), (-sc, -sc)]
        for a in (s, 0, -s):                                  # 3 per edge x4 edges
            cs += [(a, s), (a, -s), (s, a), (-s, a)] if a != 0 else [(0, s), (0, -s), (s, 0), (-s, 0)]
        cs = list(dict.fromkeys(cs))  # dedup
    else:
        raise ValueError(layout)
    return [{"cx": cx, "cy": cy, "d": d} for cx, cy in cs]


def void_cell_mask(cfg):
    """Reconstruct the carved void-cell mask for analysis (powder cells)."""
    import rfam_eqs_coupled as R
    out = P._ORIG_MAKE_DOMAIN(cfg)
    x, y, _, p_mask = out[0], out[1], out[2], out[3]
    XX, YY = np.meshgrid(x, y, indexing="xy")
    vm = np.zeros_like(p_mask, dtype=bool)
    for v in cfg["geometry"]["internal_voids"]["voids"]:
        disk = (XX - v["cx"]) ** 2 + (YY - v["cy"]) ** 2 <= (0.5 * v["d"]) ** 2
        vm |= (disk & p_mask)
    return vm


def main():
    # n_steps default 900: just past the baseline phi_bar=0.90 crossing (844),
    # T_max ~210 C (< 250 ceiling). Matched dose across all variants.
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    base = base_square_cfg(n_steps)
    nominal = P.nominal_part_mask(base)
    UF = "T"  # matched-dose comparison on the final melt region

    results = {}

    # --- 1. Baseline (no voids) ---
    bdir = OUT / "baseline" / f"square_baseline_n{n_steps}"
    P.run_one(base, bdir, "baseline")
    rec, *_ = P.analyze(bdir, nominal, MELT_REF_C, use_field=UF)
    results["baseline"] = rec

    # --- 2. Void variants: d in {1,2,3} mm, ring8 layout ---
    for d_mm in (1.0, 2.0, 3.0):
        d = d_mm * 1e-3
        voids = void_pattern(d, "ring8")
        cfg = P.make_void_cfg(base, voids)
        vm = void_cell_mask(cfg)
        label = f"void_d{d_mm:.0f}mm_ring8"
        vdir = OUT / "experimental" / f"square_{label}_n{n_steps}"
        P.run_one(cfg, vdir, label)
        rec, *_ = P.analyze(vdir, nominal, MELT_REF_C, void_mask=vm, use_field=UF)
        rec["n_voids"] = len(voids)
        rec["void_cells"] = int(vm.sum())
        results[label] = rec

    # --- 3. Scaling check: vary void COUNT at fixed d=2mm ---
    for layout in ("corners4", "ring12"):
        d = 2.0e-3
        voids = void_pattern(d, layout)
        cfg = P.make_void_cfg(base, voids)
        vm = void_cell_mask(cfg)
        label = f"void_d2mm_{layout}"
        vdir = OUT / "experimental" / f"square_{label}_n{n_steps}"
        P.run_one(cfg, vdir, label)
        rec, *_ = P.analyze(vdir, nominal, MELT_REF_C, void_mask=vm, use_field=UF)
        rec["n_voids"] = len(voids)
        rec["void_cells"] = int(vm.sum())
        results[label] = rec

    (OUT / "pilot_results.json").write_text(json.dumps(results, indent=2))
    print("\n===== PILOT RESULTS =====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
