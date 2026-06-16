"""LAYER D step 2 -- the MAKE-OR-BREAK transfer re-test.

Take the Layer-D prewarp-v2 geometries (optimized against the FD-gated Layer-C as-built-shape
objective on the calibrated transient+densification forward; saved in layerD_prewarp_v2.npz by
run_layerD_prewarp_v2.py) and run BOTH the NOMINAL and the PREWARP-v2 geometry through the
SAME validated REAL 2.5D engine (rfam_eqs_coupled.run_sim) the original non-transfer baseline was
measured on. Report as-built melt IoU AND boundary-RMS vs the nominal CAD, BEFORE vs AFTER prewarp.

This is IDENTICAL ENGINE / cfg / IoU definition to run_prewarp_realheatr_crosscheck.py (the
negative baseline prewarp_realheatr_crosscheck.npz: square 0.91->0.30, L 0.77->0.62, cross
0.33->0.14), so the prewarp-v2 result compares directly. The ONLY change is the prewarp boundary
source (Layer-C as-built objective, not the steady level-set melt-match).

Run under .venv312:  ./.venv312/bin/python run_layerD_transfer.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

HERE = Path(__file__).resolve().parent
ANALYSIS = (HERE / "../../../dissertation_materials/analysis-3dfgm").resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANALYSIS))

import rfam_eqs_coupled as R          # noqa: E402

GEOMS = ("square", "lshape", "cross")
GRID = 120
SRC_GRID = 56
N_STEPS = 720


def base_cfg() -> dict:
    """IDENTICAL to run_prewarp_realheatr_crosscheck.base_cfg (the negative-baseline engine)."""
    return {
        "geometry": {"chamber_x": 0.06, "chamber_y": 0.06, "grid_nx": GRID, "grid_ny": GRID,
                     "part": {"shape": "square", "width": 0.02, "height": 0.02,
                              "center_x": 0.0, "center_y": 0.0, "rotation_deg": 0.0}},
        "electrodes": {"mode": "boundary", "hi_boundary": "top", "lo_boundary": "bottom",
                       "boundary_cells": 1},
        "electric": {"frequency_hz": 27120000.0, "voltage_v": 860.0, "voltage_mode": "grounded",
                     "solver_steps": 3000, "solver_tol": 1.0e-09, "update_interval": 20,
                     "power_factor": 1.0, "max_qrf_w_per_m3": 1.0e11,
                     "zero_qrf_outside_doped": True, "enforce_generator_power": True,
                     "generator_power_w": 800.0, "generator_transfer_efficiency": 0.02,
                     "effective_depth_m": 0.02},
        "thermal": {"ambient_c": 23.0, "convection_h_w_per_m2k": 5.0,
                    "convective_boundaries": ["top"], "dt_s": 0.5, "n_steps": N_STEPS,
                    "max_deltaT_per_step_c": 10.0, "min_temp_c": -50.0, "max_temp_c": 600.0,
                    "phase_change": {"model": "comsol_heaviside", "t_pc_c": 180.0,
                                     "dt_pc_c": 10.0, "latent_heat_j_per_kg": 96700.0,
                                     "smooth_shape": "linear"},
                    "depth_correction": {"enabled": False}},
        "densification": {"rho_rel_initial": 0.55, "model": "physics_dual", "rho_exponent": 1.0,
                          "max_delta_per_step": 0.02, "k0_ss_per_s": 0.005,
                          "activation_energy_ss_j_per_mol": 48000.0, "phi_solid_exponent": 0.8,
                          "k0_liq_per_s": 0.08, "activation_energy_liq_j_per_mol": 0.0,
                          "phi_threshold": 0.01, "phi_liq_exponent": 1.0,
                          "liquid_rate_mode": "viscous_capillary", "surface_tension_n_per_m": 0.03,
                          "particle_radius_m": 3.5e-05, "eta_ref_pa_s": 8.0e3,
                          "eta_ref_temp_k": 458.15, "eta_activation_j_per_mol": 6.0e4,
                          "geom_factor": 0.05, "k0_per_s": 0.08,
                          "activation_energy_j_per_mol": 42000.0},
        "materials": {"virgin": {"sigma_s_per_m": 0.0, "eps_r": 2.0},
                      "powder": {"rho_solid_kg_per_m3": 490.0, "k_solid_w_per_mk": 0.197,
                                 "cp_solid_j_per_kgk": 1072.0},
                      "doped": {"sigma_s_per_m": 0.04, "sigma_profile": "uniform", "eps_r": 20.0,
                                "sigma_temp_coeff_per_K": 0.0, "sigma_density_coeff": 0.0,
                                "sigma_ref_temp_c": 23.0, "rho_solid_kg_per_m3": 460.0,
                                "rho_liquid_kg_per_m3": 1010.0, "k_solid_w_per_mk": 0.1,
                                "k_liquid_w_per_mk": 0.26, "cp_solid_j_per_kgk": 2500.0,
                                "cp_liquid_j_per_kgk": 3279.0}},
        "reporting": {"potential_contours": 18, "temp_cbar_min_c": 20.0, "temp_cbar_max_c": 400.0},
    }


def _resample(mask56: np.ndarray, out_n: int) -> np.ndarray:
    z = out_n / mask56.shape[0]
    m = ndi.zoom(mask56.astype(float), z, order=0) > 0.5
    if m.shape != (out_n, out_n):
        out = np.zeros((out_n, out_n), bool)
        a = min(out_n, m.shape[0]); b = min(out_n, m.shape[1])
        out[:a, :b] = m[:a, :b]; m = out
    return m


_INJECT = {"mask": None}
_ORIG_MAKE_DOMAIN = R.make_domain


def _patched_make_domain(cfg):
    geom = cfg["geometry"]
    nx = int(geom["grid_nx"]); ny = int(geom["grid_ny"])
    wx = float(geom["chamber_x"]); wy = float(geom["chamber_y"])
    x = np.linspace(-0.5 * wx, 0.5 * wx, nx)
    y = np.linspace(-0.5 * wy, 0.5 * wy, ny)
    p_mask = np.asarray(_INJECT["mask"], bool)
    assert p_mask.shape == (ny, nx), f"mask {p_mask.shape} != ({ny},{nx})"
    d_mask = p_mask.copy()
    fill_frac = p_mask.astype(float)
    part_id_mask = p_mask.astype(np.int32)
    elec = cfg["electrodes"]
    hi, lo = R.build_boundary_electrodes(
        x=x, y=y, hi_boundary=str(elec.get("hi_boundary", "top")),
        lo_boundary=str(elec.get("lo_boundary", "bottom")),
        boundary_cells=int(elec.get("boundary_cells", 1)))
    ys, xs = np.where(p_mask)
    poly = np.array([[x[xs.min()], y[ys.min()]], [x[xs.max()], y[ys.min()]],
                     [x[xs.max()], y[ys.max()]], [x[xs.min()], y[ys.max()]]])
    return x, y, poly, p_mask, d_mask, hi, lo, fill_frac, part_id_mask, [poly]


def run_real(mask120: np.ndarray) -> dict:
    _INJECT["mask"] = mask120
    R.make_domain = _patched_make_domain
    try:
        cfg = base_cfg()
        ret = R.run_sim(cfg)
        state, summary = ret[0], ret[1]
    finally:
        R.make_domain = _ORIG_MAKE_DOMAIN
    phi = np.asarray(state.phi, float)
    rho = np.asarray(state.rho_rel, float)
    pm = np.asarray(state.part_mask, bool)
    melt = (phi >= 0.5) & pm
    return dict(melt=melt, phi=phi, rho=rho, part_mask=pm, Tmax=float(state.T.max()),
                phi_mean=float(phi[pm].mean()) if pm.any() else 0.0,
                rho_mean=float(rho[pm].mean()) if pm.any() else 0.0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0.5; b = b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def boundary_rms(melt: np.ndarray, nominal: np.ndarray) -> float:
    """Symmetric boundary RMS distance (vox) between the melt-region contour and the
    nominal-CAD contour. For each boundary pixel of one mask, EDT distance to the other's
    region edge; symmetrize."""
    a = melt > 0.5; b = nominal > 0.5
    if not a.any() or not b.any():
        return float("nan")
    # distance from outside-of-b to b, and outside-of-a to a
    dist_to_b = ndi.distance_transform_edt(~b)
    dist_to_a = ndi.distance_transform_edt(~a)
    # boundary pixels = region XOR eroded region
    ea = a & ~ndi.binary_erosion(a)
    eb = b & ~ndi.binary_erosion(b)
    da = dist_to_b[ea]            # how far melt-boundary sits from nominal region
    db = dist_to_a[eb]            # how far nominal-boundary sits from melt region
    vals = np.concatenate([da, db])
    return float(np.sqrt((vals ** 2).mean())) if vals.size else float("nan")


def main() -> None:
    pre = np.load(ANALYSIS / "layerD_prewarp_v2.npz", allow_pickle=True)
    rows = []; out = {}
    for geo in GEOMS:
        nominal56 = np.asarray(pre[f"{geo}_nominal56"]) > 0.5
        prewarp56 = np.asarray(pre[f"{geo}_prewarp56"]) > 0.5
        nominal120 = _resample(nominal56, GRID)
        prewarp120 = _resample(prewarp56, GRID)
        print(f"\n=== LAYER D transfer: {geo} (nominal {int(nominal120.sum())} vox, "
              f"prewarp-v2 {int(prewarp120.sum())} vox) ===", flush=True)
        print("  [nominal geometry] real 2.5D HEATR ...", flush=True)
        rn = run_real(nominal120)
        print(f"    Tmax={rn['Tmax']:.1f}C  phi_mean={rn['phi_mean']:.3f}  "
              f"melt_vox={int(rn['melt'].sum())}", flush=True)
        print("  [prewarp-v2 geometry] real 2.5D HEATR ...", flush=True)
        rp = run_real(prewarp120)
        print(f"    Tmax={rp['Tmax']:.1f}C  phi_mean={rp['phi_mean']:.3f}  "
              f"melt_vox={int(rp['melt'].sum())}", flush=True)

        iou_nom = iou(rn["melt"], nominal120)
        iou_pre = iou(rp["melt"], nominal120)
        brms_nom = boundary_rms(rn["melt"], nominal120)
        brms_pre = boundary_rms(rp["melt"], nominal120)
        transfers = bool(iou_pre > iou_nom)
        print(f"  IoU(real melt, nominal CAD): nominal {iou_nom:.4f} -> prewarp-v2 {iou_pre:.4f} "
              f"(delta {iou_pre - iou_nom:+.4f})", flush=True)
        print(f"  boundary-RMS (vox):          nominal {brms_nom:.3f} -> prewarp-v2 {brms_pre:.3f}",
              flush=True)
        print(f"  TRANSFERS (prewarp beats nominal IoU): {transfers}", flush=True)
        rows.append(dict(geometry=geo, iou_nominalgeom=round(iou_nom, 4),
                         iou_prewarpgeom=round(iou_pre, 4),
                         delta_iou=round(iou_pre - iou_nom, 4),
                         bndrms_nominalgeom=round(brms_nom, 3),
                         bndrms_prewarpgeom=round(brms_pre, 3),
                         melt_vox_nominal=int(rn["melt"].sum()),
                         melt_vox_prewarp=int(rp["melt"].sum()),
                         nominal_vox=int(nominal120.sum()), prewarp_vox=int(prewarp120.sum()),
                         Tmax_nominal=round(rn["Tmax"], 1), Tmax_prewarp=round(rp["Tmax"], 1),
                         phimean_nominal=round(rn["phi_mean"], 3),
                         phimean_prewarp=round(rp["phi_mean"], 3),
                         rhomean_nominal=round(rn["rho_mean"], 3),
                         rhomean_prewarp=round(rp["rho_mean"], 3),
                         transfers=transfers))
        out[f"{geo}_nominal_mask"] = nominal120.astype(np.uint8)
        out[f"{geo}_prewarp_mask"] = prewarp120.astype(np.uint8)
        out[f"{geo}_melt_nominalgeom"] = rn["melt"].astype(np.uint8)
        out[f"{geo}_melt_prewarpgeom"] = rp["melt"].astype(np.uint8)
        out[f"{geo}_rho_nominalgeom"] = rn["rho"].astype(np.float32)
        out[f"{geo}_rho_prewarpgeom"] = rp["rho"].astype(np.float32)
        # checkpoint after each shape
        np.savez_compressed(ANALYSIS / "layerD_transfer.npz", **out)
        (ANALYSIS / "layerD_transfer.json").write_text(json.dumps(rows, indent=2))
        print(f"  checkpoint written ({len(rows)} shapes)", flush=True)

    import csv
    with open(ANALYSIS / "layerD_transfer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n{'geom':<8} {'iou_nom':>8} {'iou_pre':>8} {'d_iou':>8} "
          f"{'brms_nom':>9} {'brms_pre':>9} {'xfer':>6}")
    for r in rows:
        print(f"{r['geometry']:<8} {r['iou_nominalgeom']:>8} {r['iou_prewarpgeom']:>8} "
              f"{r['delta_iou']:>8} {r['bndrms_nominalgeom']:>9} {r['bndrms_prewarpgeom']:>9} "
              f"{str(r['transfers']):>6}")


if __name__ == "__main__":
    main()
