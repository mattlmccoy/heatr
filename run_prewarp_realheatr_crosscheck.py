"""TASK 2: validate the adjoint-prewarped geometry against the REAL 2.5D HEATR engine.

For square / lshape / cross: take the adjoint-prewarped BOUNDARY produced by the FD-gated
level-set ILT melt-shape-match prewarp (ilt_levelset.levelset_ilt -- the prewarp whose
objective is exactly melt->nominal, the right lever to test against a melt-IoU criterion;
its dJ/dphi is FD-gated in ilt_levelset.run_gate), and run BOTH the NOMINAL geometry and the
PREWARPED geometry through the REAL 2.5D coupled engine rfam_eqs_coupled.run_sim (uniform
dopant, full EQS->Qrf->transient melt->densification). Report melt-vs-nominal IoU for each.

(The transient-adjoint prewarp -- ilt_transient -- optimizes a DIFFERENT objective, in-part
sigma_T uniformity in degrees C; that is deliverable 2's table, not a melt-shape-match, so it
is not the right boundary to score on a melt IoU. We use the melt-match level-set prewarp.)

THE RESULT THAT MATTERS: does the prewarped geometry's REAL-HEATR melt land on the nominal
CAD better than the nominal geometry's? If the simplified-port prewarp does NOT transfer to
the real engine, this script says so plainly (it just reports the two IoUs).

IMPLEMENTATION NOTE (honest): shapes.py (the engine's analytic mask builder) needs matplotlib,
which is NOT in the .venv312 we must run under. So we INJECT the part mask via a monkeypatch of
make_domain -- building BOTH the nominal and the prewarped mask from the SAME source the prewarp
optimization used (ilt_levelset._nominal2d for the nominal target, the saved prewarped theta for
the prewarped geometry), resampled onto the engine's 120x120 grid. This makes the comparison
apples-to-apples: identical engine, identical dopant, identical electrodes; ONLY the boundary
differs. Electrodes are the config's boundary plates (top/bottom), built directly.

Run under .venv312 (system numpy has a buffer-elision bug):
    ./.venv312/bin/python run_prewarp_realheatr_crosscheck.py
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
import ilt_levelset as LS             # noqa: E402

GEOMS = ("square", "lshape", "cross")
GRID = 120
SRC_GRID = 56          # the prewarp engine grid
N_STEPS = 720          # real-engine thermal steps (config default)


# ── base config (the working 2.5D square optimizer config, as a dict; no yaml dep) ──
def base_cfg() -> dict:
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
                     # 800 W * 0.02 efficiency: the nominal square reaches phi_mean~0.78 (phi90
                     # crossing ~step 500), so the melt is INCOMPLETE at the boundary -- the
                     # regime where boundary distortion is visible and prewarp can act. (At 500 W
                     # the part never reaches 180 C melt; at >=1200 W it fully melts to IoU 1.0
                     # and there is nothing for prewarp to correct.) Same power for all shapes.
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
    """Nearest-neighbour resample a 56x56 binary mask to out_n x out_n, centred, preserving
    relative size (zoom by out_n/56)."""
    z = out_n / mask56.shape[0]
    m = ndi.zoom(mask56.astype(float), z, order=0) > 0.5
    if m.shape != (out_n, out_n):
        out = np.zeros((out_n, out_n), bool)
        a = min(out_n, m.shape[0]); b = min(out_n, m.shape[1])
        out[:a, :b] = m[:a, :b]
        m = out
    return m


# ── monkeypatch state: the mask to inject for the next run_sim call ─────────────────
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
    # representative poly: bounding box of the mask (used only for reporting downstream)
    ys, xs = np.where(p_mask)
    poly = np.array([[x[xs.min()], y[ys.min()]], [x[xs.max()], y[ys.min()]],
                     [x[xs.max()], y[ys.max()]], [x[xs.min()], y[ys.max()]]])
    return x, y, poly, p_mask, d_mask, hi, lo, fill_frac, part_id_mask, [poly]


def run_real(mask120: np.ndarray) -> dict:
    """Run the real 2.5D engine on an injected 120x120 mask; return melt + densification."""
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
    # melt region = part voxels that reached the melt fraction phi>=0.5 (HEATR's melt criterion)
    melt = (phi >= 0.5) & pm
    return dict(melt=melt, phi=phi, rho=rho, part_mask=pm,
                Tmax=float(state.T.max()),
                phi_mean=float(phi[pm].mean()) if pm.any() else 0.0,
                rho_mean=float(rho[pm].mean()) if pm.any() else 0.0,
                summary=summary)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0.5; b = b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main() -> None:
    rows = []
    out = {}
    for geo in GEOMS:
        # Generate the FD-gated level-set melt-match prewarp at SRC_GRID (the boundary that
        # makes the SIMPLIFIED-engine melt land on the nominal CAD). theta_final = prewarped
        # material field; nominal = the crisp CAD target.
        print(f"\n--- level-set prewarp (simplified engine) for {geo} ---", flush=True)
        cap = LS.levelset_ilt(geometry=geo, nx=SRC_GRID, ny=SRC_GRID, iters=400, lr=1.0,
                              melt_frac=0.95, verbose=False, capture=True)
        nominal56 = np.asarray(cap["nominal"]) > 0.5
        prewarp56 = np.asarray(cap["theta_final"]) > 0.5
        print(f"  simplified-engine IoU(melt,nominal) naive->prewarp: "
              f"{float(cap['iou0']):.4f} -> {float(cap['iou_final']):.4f}", flush=True)
        nominal120 = _resample(nominal56, GRID)
        prewarp120 = _resample(prewarp56, GRID)

        print(f"\n=== REAL-HEATR cross-check: {geo} "
              f"(nominal {int(nominal120.sum())} vox, prewarp {int(prewarp120.sum())} vox) ===",
              flush=True)
        print("  [nominal geometry] running real 2.5D HEATR ...", flush=True)
        rn = run_real(nominal120)
        print(f"    Tmax={rn['Tmax']:.1f}C  phi_mean={rn['phi_mean']:.3f}  "
              f"rho_mean={rn['rho_mean']:.3f}  melt_vox={int(rn['melt'].sum())}", flush=True)
        print("  [prewarped geometry] running real 2.5D HEATR ...", flush=True)
        rp = run_real(prewarp120)
        print(f"    Tmax={rp['Tmax']:.1f}C  phi_mean={rp['phi_mean']:.3f}  "
              f"rho_mean={rp['rho_mean']:.3f}  melt_vox={int(rp['melt'].sum())}", flush=True)

        # IoU of each real-HEATR melt region against the crisp NOMINAL CAD mask
        iou_nominal = iou(rn["melt"], nominal120)
        iou_prewarp = iou(rp["melt"], nominal120)
        print(f"  IoU(real-HEATR melt, nominal CAD):  nominal-geom {iou_nominal:.4f}  ->  "
              f"prewarped-geom {iou_prewarp:.4f}  (delta {iou_prewarp - iou_nominal:+.4f})",
              flush=True)
        rows.append(dict(geometry=geo, iou_nominalgeom=round(iou_nominal, 4),
                         iou_prewarpgeom=round(iou_prewarp, 4),
                         delta_iou=round(iou_prewarp - iou_nominal, 4),
                         melt_vox_nominal=int(rn["melt"].sum()),
                         melt_vox_prewarp=int(rp["melt"].sum()),
                         nominal_vox=int(nominal120.sum()), prewarp_vox=int(prewarp120.sum()),
                         Tmax_nominal=round(rn["Tmax"], 1), Tmax_prewarp=round(rp["Tmax"], 1),
                         phimean_nominal=round(rn["phi_mean"], 3),
                         phimean_prewarp=round(rp["phi_mean"], 3),
                         transfers=bool(iou_prewarp > iou_nominal)))
        out[f"{geo}_nominal_mask"] = nominal120.astype(np.uint8)
        out[f"{geo}_prewarp_mask"] = prewarp120.astype(np.uint8)
        out[f"{geo}_melt_nominalgeom"] = rn["melt"].astype(np.uint8)
        out[f"{geo}_melt_prewarpgeom"] = rp["melt"].astype(np.uint8)
        out[f"{geo}_rho_nominalgeom"] = rn["rho"].astype(np.float32)
        out[f"{geo}_rho_prewarpgeom"] = rp["rho"].astype(np.float32)

    import csv
    csv_path = ANALYSIS / "prewarp_realheatr_crosscheck.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    np.savez_compressed(ANALYSIS / "prewarp_realheatr_crosscheck.npz", **out)
    (ANALYSIS / "prewarp_realheatr_crosscheck.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path}")
    print(f"\n{'geom':<8} {'iou_nom':>8} {'iou_pre':>8} {'delta':>8} {'transfers':>10}")
    for r in rows:
        print(f"{r['geometry']:<8} {r['iou_nominalgeom']:>8} {r['iou_prewarpgeom']:>8} "
              f"{r['delta_iou']:>8} {str(r['transfers']):>10}")


if __name__ == "__main__":
    main()
