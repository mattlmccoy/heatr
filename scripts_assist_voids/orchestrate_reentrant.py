#!/usr/bin/env python3
"""SRAF assist-voids pilot — REENTRANT / over-hot-interior regime.

Promising regime (prior-pilot recommendation): shapes whose INTERIOR
over-heats / over-densifies, so the void-closure zone (hot interior) and
the shape/uniformity-error zone COINCIDE. Probe (probe_interior_overheat.py)
confirms cross & L_shape over-heat the interior (+9..+12 C interior-minus-edge,
ui_rms 0.34-0.43) while the convex square does the opposite.

For each shape we run, matched-dose at one exposure:
  - baseline (void-free)
  - SRAF void variants: d in {1,2} mm placed in the over-hot interior core
  - a matched FGM correction (single-mask dopant knockdown of the hot core)
And report: interior sigma_T, melt IoU vs nominal CAD, void closure density,
SRAF-vs-FGM, plus a frequency cross-check for wavelength-independence.

Reuses the monkey-patched make_domain void model from run_assist_voids_pilot.
Numerically gated: dT-clip, energy residual, phi90.
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

import run_assist_voids_pilot as P  # noqa: E402  (monkey-patches make_domain)
import rfam_eqs_coupled as R        # noqa: E402

BASE_CFG = ROOT / "configs" / "shape_cross_6min.yaml"
OUT = ROOT / "outputs_eqs" / "runs" / "reentrant" / "assist_voids"
MELT_REF_C = 180.0


def base_cfg(shape, w, h, n_steps):
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["geometry"]["part"]["shape"] = shape
    cfg["geometry"]["part"]["width"] = w
    cfg["geometry"]["part"]["height"] = h
    cfg["geometry"]["part"]["center_x"] = 0.0
    cfg["geometry"]["part"]["center_y"] = 0.0
    cfg["thermal"]["n_steps"] = n_steps
    return cfg


def find_phi90_steps(cfg, cap_steps=1100):
    """Single long run; return step index where mean_phi_part crosses 0.90,
    and the max-T at that step. Uses time series hist."""
    c = copy.deepcopy(cfg)
    c["thermal"]["n_steps"] = cap_steps
    state, summary, hist, tt, opt = R.run_sim(c)
    # hist keys vary; find phi-mean and Tmax series
    phi_key = next((k for k in hist if "phi" in k.lower() and "mean" in k.lower()), None)
    if phi_key is None:
        phi_key = next((k for k in hist if "phi" in k.lower()), None)
    tmax_key = next((k for k in hist if ("tmax" in k.lower() or ("t_max" in k.lower()))), None)
    phis = np.asarray(hist[phi_key], dtype=float)
    cross = np.argmax(phis >= 0.90) if (phis >= 0.90).any() else len(phis) - 1
    info = {"phi_key": phi_key, "n_hist": len(phis), "phi90_step": int(cross),
            "phi_final": float(phis[-1]), "phi90_reached": bool((phis >= 0.90).any())}
    if tmax_key:
        tm = np.asarray(hist[tmax_key], dtype=float)
        info["Tmax_at_phi90"] = float(tm[min(cross, len(tm)-1)])
        info["Tmax_final"] = float(tm[-1])
    return info, hist


def interior_core_mask(part_mask, n_erode):
    return ndimage.binary_erosion(part_mask, iterations=n_erode)


def sigma_T_interior(out_dir, n_erode=3, use_field="T"):
    f = np.load(out_dir / "fields.npz")
    T = np.asarray(f[use_field], dtype=float)
    pm = np.asarray(f["part_mask"], dtype=bool)
    core = interior_core_mask(pm, n_erode)
    return float(np.std(T[core])), float(np.std(T[pm])), int(core.sum())


def hotcore_center(out_dir, use_field="T", hot_pct=90.0):
    """Centroid (cx,cy in metres) of the hottest part cells from a baseline run.
    This is the over-hot interior zone where SRAF voids should sit."""
    f = np.load(out_dir / "fields.npz")
    x = np.asarray(f["x"], dtype=float); y = np.asarray(f["y"], dtype=float)
    T = np.asarray(f[use_field], dtype=float)
    pm = np.asarray(f["part_mask"], dtype=bool)
    thr = np.percentile(T[pm], hot_pct)
    hot = pm & (T >= thr)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    return float(XX[hot].mean()), float(YY[hot].mean())


def core_void_layout(shape, d, center, r=None):
    """Voids placed as a small cross-cluster around `center` (cx,cy metres),
    the over-hot interior centroid. cross: wider cluster; L_shape: tighter."""
    cx0, cy0 = center
    if r is None:
        r = 0.004 if shape == "cross" else 0.003
    cs = [(cx0, cy0), (cx0+r, cy0), (cx0-r, cy0), (cx0, cy0+r), (cx0, cy0-r)]
    return [{"cx": cx, "cy": cy, "d": d} for cx, cy in cs]


def void_cell_mask(cfg):
    out = P._ORIG_MAKE_DOMAIN(cfg)
    x, y, _, p_mask = out[0], out[1], out[2], out[3]
    XX, YY = np.meshgrid(x, y, indexing="xy")
    vm = np.zeros_like(p_mask, dtype=bool)
    for v in cfg["geometry"]["internal_voids"]["voids"]:
        disk = (XX - v["cx"]) ** 2 + (YY - v["cy"]) ** 2 <= (0.5 * v["d"]) ** 2
        vm |= (disk & p_mask)
    return vm


def make_fgm_core_cfg(base, shape, w, knockdown=0.5, n_erode=3):
    """Matched FGM correction: knock down doped sigma in the over-hot interior
    core via a per-cell saturation/fill multiplier. We implement the same
    'remove RF source' lever as the void but GRADED (continuous fill_frac
    reduction in the core) instead of binary holes, so it is the FGM analogue.
    Uses a config flag consumed by a second monkey-patch below."""
    c = copy.deepcopy(base)
    c["geometry"]["fgm_core_knockdown"] = {
        "enabled": True, "knockdown": knockdown, "n_erode": n_erode}
    return c


# ---- second monkey-patch: graded core knockdown (FGM analogue) ----
_PREV_MAKE_DOMAIN = R.make_domain  # the void-patched one


def _patched_with_fgm(cfg):
    out = _PREV_MAKE_DOMAIN(cfg)  # applies voids if present
    fcfg = cfg.get("geometry", {}).get("fgm_core_knockdown", {})
    if isinstance(fcfg, dict) and fcfg.get("enabled", False):
        x, y, ppoly, p_mask, d_mask, hi, lo, fill_frac, pid, polys = out
        core = ndimage.binary_erosion(p_mask, iterations=int(fcfg.get("n_erode", 3)))
        kd = float(fcfg.get("knockdown", 0.5))
        fill_frac = fill_frac.copy()
        fill_frac[core] *= kd  # graded reduction of RF source in hot core
        print(f"  [fgm_core] knocked down fill_frac x{kd} on {int(core.sum())} core cells")
        out = (x, y, ppoly, p_mask, d_mask, hi, lo, fill_frac, pid, polys)
    return out


R.make_domain = _patched_with_fgm


def analyze_full(out_dir, nominal, void_mask=None, n_erode=3, use_field="T"):
    rec, T, rho, pm = P.analyze(out_dir, nominal, MELT_REF_C,
                                void_mask=void_mask, use_field=use_field)
    core = interior_core_mask(pm, n_erode)
    rec["sigma_T_interior_c"] = float(np.std(T[core]))
    rec["sigma_T_part_c"] = float(np.std(T[pm]))
    rec["mean_T_interior_c"] = float(np.mean(T[core]))
    rec["interior_core_cells"] = int(core.sum())
    return rec


def run_shape(shape, w, h, n_steps, freq_check=False):
    base = base_cfg(shape, w, h, n_steps)
    nominal = P.nominal_part_mask(base)
    UF = "T"
    res = {}

    bdir = OUT / "baseline" / f"{shape}_baseline_n{n_steps}"
    P.run_one(base, bdir, f"{shape}-baseline")
    res["baseline"] = analyze_full(bdir, nominal, use_field=UF)

    # over-hot interior centroid from the baseline -> void placement target
    center = hotcore_center(bdir, use_field=UF)
    res["_hotcore_center_m"] = list(center)
    print(f"  [hotcore] {shape} over-hot centroid = ({center[0]*1e3:+.1f},{center[1]*1e3:+.1f}) mm")

    for d_mm in (1.0, 2.0):
        d = d_mm * 1e-3
        voids = core_void_layout(shape, d, center)
        cfg = P.make_void_cfg(base, voids)
        vm = void_cell_mask(cfg)
        label = f"void_d{d_mm:.0f}mm_core"
        vdir = OUT / "experimental" / f"{shape}_{label}_n{n_steps}"
        P.run_one(cfg, vdir, f"{shape}-{label}")
        rec = analyze_full(vdir, nominal, void_mask=vm, use_field=UF)
        rec["n_voids"] = len(voids); rec["void_cells"] = int(vm.sum())
        res[label] = rec

    for kd in (0.5,):
        cfg = make_fgm_core_cfg(base, shape, w, knockdown=kd)
        label = f"fgm_core_kd{kd}"
        fdir = OUT / "experimental" / f"{shape}_{label}_n{n_steps}"
        P.run_one(cfg, fdir, f"{shape}-{label}")
        res[label] = analyze_full(fdir, nominal, use_field=UF)

    if freq_check:
        # wavelength-independence: rerun d=2mm void at 13.56 & 40.68 MHz
        d = 2.0e-3
        voids = core_void_layout(shape, d, center)
        for fmhz, ftag in ((13.56e6, "13p56"), (40.68e6, "40p68")):
            cfg = P.make_void_cfg(base, voids)
            cfg["electric"]["frequency_hz"] = fmhz
            vm = void_cell_mask(cfg)
            label = f"void_d2mm_core_{ftag}MHz"
            vdir = OUT / "experimental" / f"{shape}_{label}_n{n_steps}"
            P.run_one(cfg, vdir, f"{shape}-{label}")
            rec = analyze_full(vdir, nominal, void_mask=vm, use_field=UF)
            res[label] = rec
    return res


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 720
    shapes = [("cross", 0.024, 0.024), ("L_shape", 0.026, 0.026)]
    OUT.mkdir(parents=True, exist_ok=True)

    all_res = {}
    # exposure locate on the primary shape (cross)
    base0 = base_cfg("cross", 0.024, 0.024, n_steps)
    info, _ = find_phi90_steps(base0, cap_steps=max(n_steps, 1100))
    all_res["_exposure_probe_cross"] = info
    print("\n[exposure] cross:", json.dumps(info, indent=2))

    for shape, w, h in shapes:
        print(f"\n########## SHAPE {shape} {w*1e3:.0f}mm n_steps={n_steps} ##########")
        all_res[shape] = run_shape(shape, w, h, n_steps,
                                   freq_check=(shape == "cross"))

    (OUT / "reentrant_results.json").write_text(json.dumps(all_res, indent=2))
    print("\n===== REENTRANT SRAF RESULTS =====")
    print(json.dumps(all_res, indent=2))


if __name__ == "__main__":
    main()
