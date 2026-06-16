#!/usr/bin/env python3
"""Probe which reentrant/thick shapes over-melt their INTERIOR.

The convex-square regime heats corners/edges fastest (interior cooler via
electrostatic shielding). The SRAF "promising regime" needs the opposite:
the interior is the HOTTEST / over-densified zone, so a void there both
reshapes the field and heals.

This runs short void-free baselines for candidate shapes and reports, for
the part interior vs the part edge band, the mean/max T and rho. A shape
qualifies if T_interior > T_edge (interior over-heat) AND interior reaches
melt before edges.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rfam_eqs_coupled as R  # noqa: E402

BASE_CFG = ROOT / "configs" / "shape_cross_6min.yaml"


def base_cfg(shape, w, h, n_steps):
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["geometry"]["part"]["shape"] = shape
    cfg["geometry"]["part"]["width"] = w
    cfg["geometry"]["part"]["height"] = h
    cfg["geometry"]["part"]["center_x"] = 0.0
    cfg["geometry"]["part"]["center_y"] = 0.0
    cfg["thermal"]["n_steps"] = n_steps
    return cfg


def interior_edge_split(part_mask, n_erode=4):
    """Return (interior_mask, edge_mask). interior = eroded core."""
    interior = ndimage.binary_erosion(part_mask, iterations=n_erode)
    edge = part_mask & ~interior
    return interior, edge


def probe(shape, w, h, n_steps):
    cfg = base_cfg(shape, w, h, n_steps)
    state, summary, hist, tt, opt = R.run_sim(cfg)
    # fields from state (SimState dataclass; attribute access)
    T = np.asarray(state.T, dtype=float)           # final T (C)
    rho = np.asarray(state.rho_rel, dtype=float)
    pm = np.asarray(state.part_mask, dtype=bool)
    interior, edge = interior_edge_split(pm, n_erode=4)
    def stats(m):
        return dict(meanT=float(T[m].mean()), maxT=float(T[m].max()),
                    meanrho=float(rho[m].mean()), maxrho=float(rho[m].max()),
                    n=int(m.sum()))
    rec = {
        "shape": shape, "w_mm": w*1e3, "h_mm": h*1e3, "n_steps": n_steps,
        "interior": stats(interior), "edge": stats(edge),
        "ui_rms_part": summary["ui_rms_part_final"],
        "mean_T_part_c": summary["mean_T_part_final_c"],
        "max_T_part_c": summary["max_T_part_final_c"],
        "energy_resid": summary["energy_balance_residual_final_J_per_m"],
        "dT_clip": summary.get("frac_cells_dT_clipped_final", 0.0),
    }
    rec["interior_minus_edge_meanT_c"] = rec["interior"]["meanT"] - rec["edge"]["meanT"]
    rec["interior_overheats"] = rec["interior_minus_edge_meanT_c"] > 0
    return rec


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    candidates = [
        ("cross",  0.030, 0.030),   # reentrant
        ("cross",  0.024, 0.024),
        ("square", 0.030, 0.030),   # thick solid
        ("square", 0.020, 0.020),   # reference (prior pilot)
        ("L_shape",0.026, 0.026),   # reentrant shadow
        ("plus",   0.030, 0.030),   # alias check (may not be supported)
    ]
    out = []
    for shape, w, h in candidates:
        try:
            rec = probe(shape, w, h, n_steps)
            out.append(rec)
            print(f"\n{shape} {w*1e3:.0f}mm: int-edge dMeanT={rec['interior_minus_edge_meanT_c']:+.1f}C "
                  f"intMaxT={rec['interior']['maxT']:.0f} edgeMaxT={rec['edge']['maxT']:.0f} "
                  f"intRho={rec['interior']['meanrho']:.3f} edgeRho={rec['edge']['meanrho']:.3f} "
                  f"overheats={rec['interior_overheats']}")
        except Exception as e:
            print(f"\n{shape} {w*1e3:.0f}mm: FAILED {type(e).__name__}: {e}")
    pdir = ROOT / "outputs_eqs" / "runs" / "reentrant" / "assist_voids"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "probe_interior_overheat.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {pdir/'probe_interior_overheat.json'}")


if __name__ == "__main__":
    main()
