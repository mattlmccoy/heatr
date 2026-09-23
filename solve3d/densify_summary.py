"""Summarize a densify march into the green->dense Z factor that staging consumes.

    ./.venv312/bin/python -m solve3d.densify_summary solve3d/results/densify_cube/fields.npz

Reads the march's fields.npz (rho_final, part, h) and runs heatr3d.shrinkage_analysis
-- the SINGLE source of truth for the anisotropic sintering shrink law -- then
writes densify_summary.json next to it. The MetPrint staging tools read that JSON
(stage_job --densify-summary); they never import heatr3d or re-derive the law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path

import numpy as np

import heatr3d as H

SCHEMA = 1
SUMMARY_NAME = "densify_summary.json"


def summarize_densify(fields_npz, xy_frac: float = 0.04) -> dict:
    """Green->dense Z factor + shrink/warp metrics from one densify march."""
    fields_npz = Path(fields_npz)
    d = np.load(fields_npz, allow_pickle=False)
    for k in ("rho_final", "part", "h"):
        if k not in d.files:
            raise ValueError(f"{fields_npz}: missing '{k}' (not a densify march output?)")
    part = np.asarray(d["part"], bool)
    if not part.any():
        raise ValueError(f"{fields_npz}: empty part mask")
    h = float(d["h"])
    p = H.Params()
    res = types.SimpleNamespace(rho_final=np.asarray(d["rho_final"], float), part=part)
    sh = H.shrinkage_analysis(res, p, h, xy_frac=xy_frac)
    return {"schema": SCHEMA, "kind": "densify_summary",
            "layer_multiplier": float(sh["layer_multiplier"]),
            "rho_final_mean": float(sh["rho_final_mean"]),
            "rho_final_std": float(sh["rho_final_std"]),
            "warp_std_pct": float(sh["warp_std_pct"]),
            "warp_range_pct": float(sh["warp_range_pct"]),
            "z_shrink_pct": float(sh["z_shrink_pct"]),
            "xy_shrink_pct": float(sh["xy_shrink_pct"]),
            "rho_green": float(p.rho_rel), "xy_frac": float(xy_frac),
            "grid_n": int(part.shape[0]), "h_mm": h * 1e3,
            "law": "heatr3d.shrinkage_factors (via shrinkage_analysis)",
            "source": str(fields_npz),
            "source_sha256": hashlib.sha256(fields_npz.read_bytes()).hexdigest()}


def write_summary(fields_npz, out_json=None, xy_frac: float = 0.04) -> Path:
    """Write the summary JSON (default: densify_summary.json beside fields.npz)."""
    fields_npz = Path(fields_npz)
    out = Path(out_json) if out_json else fields_npz.with_name(SUMMARY_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summarize_densify(fields_npz, xy_frac), indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fields_npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--xy-frac", type=float, default=0.04)
    a = ap.parse_args(argv)
    out = write_summary(a.fields_npz, a.out, a.xy_frac)
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
