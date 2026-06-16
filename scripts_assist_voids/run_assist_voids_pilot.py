#!/usr/bin/env python3
"""Sub-Resolution Internal Assist Voids (RFAM SRAF) pilot — HEATR 2.5D.

Realization of the void: a hole punched in ``doped_mask`` + ``fill_frac``
(removes the local RF source, "no binder/dopant") while ``part_mask`` is KEPT
(loose powder present -> thermal mass + sinterable). With d < L_diff the void
powder heats by conduction from neighbours and densifies -> sinters shut.

This driver monkey-patches ``make_domain`` (solver source untouched) to carve
voids defined by a ``geometry.internal_voids`` config block, then runs the
coupled transient and writes standard HEATR outputs per variant.

Void config block:
  geometry:
    internal_voids:
      enabled: true
      voids:
        - {cx: 0.006, cy: 0.006, d: 0.002}   # centre (m), diameter (m)
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

import rfam_eqs_coupled as R  # noqa: E402

_ORIG_MAKE_DOMAIN = R.make_domain


def _carve_voids(cfg, x, y, p_mask, d_mask, fill_frac, part_id_mask):
    """Punch void disks into doped_mask/fill_frac; keep part_mask intact."""
    vcfg = cfg.get("geometry", {}).get("internal_voids", {})
    if not (isinstance(vcfg, dict) and vcfg.get("enabled", False)):
        return d_mask, fill_frac, 0, 0
    XX, YY = np.meshgrid(x, y, indexing="xy")
    void_mask = np.zeros_like(p_mask, dtype=bool)
    for v in vcfg.get("voids", []):
        cx, cy, d = float(v["cx"]), float(v["cy"]), float(v["d"])
        disk = (XX - cx) ** 2 + (YY - cy) ** 2 <= (0.5 * d) ** 2
        void_mask |= (disk & p_mask)  # only carve inside the part
    # Remove RF source + thermal-mass-as-DOPED, keep part_mask (powder present).
    d_mask = d_mask & ~void_mask
    fill_frac = fill_frac.copy()
    fill_frac[void_mask] = 0.0
    n_void = int(void_mask.sum())
    n_part = int(p_mask.sum())
    return d_mask, fill_frac, n_void, n_part


def _patched_make_domain(cfg):
    out = _ORIG_MAKE_DOMAIN(cfg)
    x, y, part_poly_ref, p_mask, d_mask, hi, lo, fill_frac, part_id_mask, part_polys = out
    d_mask, fill_frac, n_void, n_part = _carve_voids(
        cfg, x, y, p_mask, d_mask, fill_frac, part_id_mask
    )
    if n_void:
        print(f"  [assist_voids] carved {n_void} void cells "
              f"({100*n_void/max(n_part,1):.1f}% of part) from doped_mask/fill_frac")
    return (x, y, part_poly_ref, p_mask, d_mask, hi, lo, fill_frac,
            part_id_mask, part_polys)


R.make_domain = _patched_make_domain


def nominal_part_mask(cfg):
    """The void-free part_mask = nominal CAD target."""
    c = copy.deepcopy(cfg)
    c.setdefault("geometry", {}).pop("internal_voids", None)
    out = _ORIG_MAKE_DOMAIN(c)
    return out[3]  # p_mask


def melt_iou(T_phi90, melt_ref_c, nominal_mask):
    """IoU of as-built melt region (T_phi90 >= melt_ref) vs nominal CAD mask."""
    melt = T_phi90 >= melt_ref_c
    inter = np.logical_and(melt, nominal_mask).sum()
    union = np.logical_or(melt, nominal_mask).sum()
    bleed = np.logical_and(melt, ~nominal_mask).sum()   # melt outside CAD
    miss = np.logical_and(~melt, nominal_mask).sum()     # CAD not melted
    return {
        "iou": float(inter / max(union, 1)),
        "melt_cells": int(melt.sum()),
        "nominal_cells": int(nominal_mask.sum()),
        "bleed_cells": int(bleed),
        "miss_cells": int(miss),
        "bleed_frac_of_nominal": float(bleed / max(nominal_mask.sum(), 1)),
    }


def run_one(cfg, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== RUN {label} -> {out_dir} ===")
    state, summary, hist, tt_steps, opt_data = R.run_sim(cfg)
    R.save_outputs(cfg, state, summary, hist, out_dir,
                   tt_rotation_steps=tt_steps, opt_data=opt_data)
    return out_dir


def analyze(out_dir: Path, nominal_mask, melt_ref_c, void_mask=None, use_field="T"):
    """use_field: 'T' = final melt region (matched-dose comparison) or
    'T_phi90' = melt region at the phi_bar=0.90 crossing snapshot."""
    f = np.load(out_dir / "fields.npz")
    summ = json.load(open(out_dir / "summary.json"))
    T_eval = np.asarray(f[use_field], dtype=float)
    part_mask = np.asarray(f["part_mask"], dtype=bool)
    rho = np.asarray(f["rho_rel"], dtype=float)
    iou = melt_iou(T_eval, melt_ref_c, nominal_mask)
    rec = {
        "iou": iou,
        "phi90_reached": bool(summ.get("T_phi90_reached", True)),
        "mean_T_part_c": summ["mean_T_part_final_c"],
        "max_T_part_c": summ["max_T_part_final_c"],
        "mean_rho_part": summ["mean_rho_rel_part_final"],
        "min_rho_part": float(np.min(rho[part_mask])),
        "ui_rms_part": summ["ui_rms_part_final"],
        "energy_residual_J_per_m": summ["energy_balance_residual_final_J_per_m"],
        "frac_dT_clipped": summ.get("frac_cells_dT_clipped_final",
                                    summ.get("frac_part_at_temp_cap", 0.0)),
        "mean_phi_part": summ["mean_phi_part_final"],
    }
    if void_mask is not None and void_mask.any():
        rec["void_mean_rho"] = float(np.mean(rho[void_mask]))
        rec["void_min_rho"] = float(np.min(rho[void_mask]))
        rec["void_mean_T_c"] = float(np.mean(T_eval[void_mask]))
        solid_mask = part_mask & ~void_mask
        rec["solid_mean_rho"] = float(np.mean(rho[solid_mask]))
    return rec, T_eval, rho, part_mask


def make_void_cfg(base, voids):
    c = copy.deepcopy(base)
    c["geometry"]["internal_voids"] = {"enabled": True, "voids": voids}
    return c
