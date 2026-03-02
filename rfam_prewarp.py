#!/usr/bin/env python3
"""rfam_prewarp.py — ILT-inspired geometry prewarp for RFAM RF sintering.

Given a desired TARGET shape, iteratively finds an INPUT geometry such that
after RF exposure the densified part boundary matches the target.

Algorithm  (EPE-based boundary correction; RFAM analog of optical proximity
correction in ILT / semiconductor lithography):
  1.  Initialise prewarp polygon  P = target polygon T  (N equally-spaced pts)
  2.  For each iteration:
      a.  Inject P into the sim config as shape='polygon'; run full forward
          model via rfam_eqs_coupled.run_sim(cfg).
      b.  Sample rho_rel ~1.5 grid-cells inward from each boundary vertex of P
          (bilinear interpolation of the final-state rho_rel field).
      c.  Compute local linear shrinkage
              s[i] = 1 - sqrt(rho_0 / rho_local[i])    (2-D mass conservation)
      d.  Predicted post-sinter boundary:
              B_pred[i] = P[i] - s[i] * L_eff * n_out[i]
      e.  Edge Placement Error (normal component):
              EPE[i] = (T[i] - B_pred[i]) · n_out[i]
      f.  Gradient-descent update with Laplacian smoothing:
              P_new = P + lr * EPE * n_out  +  smooth * Laplacian(P)
      g.  Resample P_new to N pts; check convergence (RMS EPE < tol).
  3.  Run one final VERIFICATION forward simulation on the converged P.
      Full save_outputs() + rf_summary_v5.png are written to output_dir/verification/.
  4.  Save: prewarp_vertices.csv, prewarp_config.yaml, prewarp_summary.json,
            prewarp_report.png,  verification/ (full dump).

Usage
-----
    python3 rfam_prewarp.py \\
        --config configs/rfam_prewarp_square.yaml \\
        --output-dir outputs_eqs/prewarp_square

    python3 rfam_prewarp.py \\
        --config configs/rfam_prewarp_circle.yaml \\
        --output-dir outputs_eqs/prewarp_circle

TODO (future 3-D extension)
----------------------------
Replace the polygon-vertex representation with a binary voxel part_mask.
Use scipy.ndimage.binary_erosion / binary_dilation with an EPE-weighted
structuring kernel to update the mask each iteration — no polygon extraction
needed, scales naturally to 3-D geometries.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

# ── Import forward model and shape utilities (same directory) ─────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import rfam_eqs_coupled as _fwd          # noqa: E402  (must come after sys.path tweak)
from shapes import make_shape, resample_polygon  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _polygon_area(poly: np.ndarray) -> float:
    """Signed area via the shoelace formula.  Positive = CCW winding."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def compute_outward_normals(poly: np.ndarray) -> np.ndarray:
    """
    Outward unit normals at each vertex of a closed polygon ring (open array).

    Uses centred finite differences along the polygon perimeter.
    Orientation is enforced: normals point away from the polygon interior.
    """
    # Centred difference gives the tangent direction at each vertex
    dp = np.roll(poly, -1, axis=0) - np.roll(poly, 1, axis=0)
    # Rotate tangent 90° CW to get the rightward normal
    nrm = np.column_stack([dp[:, 1], -dp[:, 0]])
    # For a CCW polygon the rightward normal is outward; flip for CW
    if _polygon_area(poly) < 0:
        nrm = -nrm
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    mag = np.maximum(mag, 1e-12)
    return nrm / mag


def bilinear_sample(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """
    Vectorised bilinear interpolation of a 2-D field at arbitrary query pts.

    Parameters
    ----------
    field : (ny, nx) float array
    x     : (nx,)  grid x-coordinates [m]
    y     : (ny,)  grid y-coordinates [m]
    pts   : (M, 2) query points  [[x0,y0], [x1,y1], …]

    Returns
    -------
    (M,) float array, clamped to the grid edges.
    """
    dx_g = float(x[1] - x[0])
    dy_g = float(y[1] - y[0])
    nx, ny = len(x), len(y)

    xi = np.clip((pts[:, 0] - x[0]) / dx_g, 0.0, nx - 1 - 1e-9)
    yi = np.clip((pts[:, 1] - y[0]) / dy_g, 0.0, ny - 1 - 1e-9)

    ix = xi.astype(int)
    iy = yi.astype(int)
    fx = xi - ix
    fy = yi - iy

    v00 = field[iy,     ix    ]
    v10 = field[iy,     ix + 1]
    v01 = field[iy + 1, ix    ]
    v11 = field[iy + 1, ix + 1]

    return (v00 * (1.0 - fx) * (1.0 - fy)
          + v10 *        fx  * (1.0 - fy)
          + v01 * (1.0 - fx) *        fy
          + v11 *        fx  *        fy)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {path}")
    return cfg


def _inject_polygon(cfg: dict, poly: np.ndarray) -> dict:
    """
    Return a deep-copy of *cfg* with geometry.part overridden to use *poly*.

    The shape type is set to 'polygon' and polygon_points holds the current
    prewarp vertices.  All other simulation parameters are left unchanged.
    """
    c = copy.deepcopy(cfg)
    part = c["geometry"]["part"]
    part["shape"] = "polygon"
    part["polygon_points"] = poly.tolist()
    # n_circle_pts is ignored for polygon shapes; keep it for reference
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Main prewarp loop
# ─────────────────────────────────────────────────────────────────────────────

def run_prewarp(cfg: dict, output_dir: Path) -> None:
    """Execute the full ILT-inspired prewarp optimisation and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pw  = cfg["prewarp"]
    tgt = cfg["target"]

    N          = int(tgt.get("n_resample_pts", 512))
    max_iter   = int(pw["max_iter"])
    lr         = float(pw["lr"])
    smooth     = float(pw["smooth"])
    L_eff      = float(pw["L_eff_m"])
    rho_0      = float(pw.get("rho_0", 0.55))
    tol        = float(pw["epe_tol_mm"]) * 1e-3
    inset_frac = float(pw.get("inset_cells", 1.5))

    t_w   = float(tgt["width"])
    t_h   = float(tgt.get("height", t_w))

    print(f"\n{'='*64}")
    print(f"  RFAM Geometry Prewarp — ILT-inspired EPE correction")
    print(f"  Target : {tgt['shape']}  {t_w*1e3:.1f} × {t_h*1e3:.1f} mm")
    print(f"  N={N} pts   max_iter={max_iter}   lr={lr}   L_eff={L_eff*1e3:.1f} mm")
    print(f"  tol={pw['epe_tol_mm']:.2f} mm   inset={inset_frac} cells")
    print(f"{'='*64}\n")

    # ── Build target polygon (N equally-spaced perimeter points) ─────────────
    T_raw  = make_shape(tgt["shape"], t_w, t_h)
    T_poly = resample_polygon(T_raw, N)
    # Normalise to CCW winding
    if _polygon_area(T_poly) < 0:
        T_poly = T_poly[::-1].copy()

    # Prewarp starts as a copy of the target
    P = T_poly.copy()

    history: list[dict]         = []
    B_pred_last: np.ndarray     = np.empty((0, 2))
    state_last                  = None

    # Best-iterate tracking: keep the polygon with the lowest RMSE seen so far.
    # Used for the final verification run even if later iterations diverge.
    best_P:      np.ndarray     = P.copy()
    best_rmse:   float          = float("inf")
    best_B_pred: np.ndarray     = np.empty((0, 2))
    best_state                  = None
    n_no_improve: int           = 0          # consecutive non-improving iterations

    for it in range(max_iter):
        t_start = time.perf_counter()

        # ── 1. Inject polygon and run forward model ───────────────────────────
        sim_cfg = _inject_polygon(cfg, P)
        print(f"  [prewarp] iter {it:2d}  running forward sim …")
        state, summary, hist_ts, _, _ = _fwd.run_sim(sim_cfg)
        t_sim = time.perf_counter() - t_start
        print(f"  [prewarp]           done in {t_sim:.1f} s  "
              f"(rho_mean={summary.get('mean_rho_rel_part_final', 0):.3f})")

        # ── 2. Sample rho_rel slightly inward from each boundary vertex ───────
        dx_g  = float(state.x[1] - state.x[0])
        n_out = compute_outward_normals(P)          # (N, 2) outward unit normals
        P_inset = P - inset_frac * dx_g * n_out    # sample points (inside part)
        rho_loc = bilinear_sample(state.rho_rel, state.x, state.y, P_inset)

        # ── 3. Local shrinkage → predicted post-sinter boundary ───────────────
        # s[i] = fractional LINEAR inward displacement (2-D mass conservation)
        # Use radial distance from centroid as the effective length scale:
        #   d[i] = s[i] * r[i]   where r[i] = |P[i] - centroid|
        # This is physically motivated: each boundary point contracts toward the
        # centroid proportional to its distance from it, weighted by local density.
        centroid = P.mean(axis=0)
        r_loc    = np.linalg.norm(P - centroid, axis=1)          # (N,)
        eff_Leff = np.clip(r_loc, 0.001, 0.030)                  # clamp to [1mm,30mm]
        # Blend with fixed L_eff for stability: use whichever is smaller
        eff_Leff = np.minimum(eff_Leff, L_eff)
        s        = 1.0 - np.sqrt(rho_0 / np.maximum(rho_loc, rho_0 + 1e-9))
        B_pred   = P - s[:, None] * eff_Leff[:, None] * n_out   # (N, 2)

        # ── 4. EPE (normal component) + RMSE ──────────────────────────────────
        # Point-by-point correspondence: T_poly[i] ↔ B_pred[i] ↔ P[i]
        epe_vec = T_poly - B_pred                                 # (N, 2) vector EPE
        epe_n   = np.einsum("ij,ij->i", epe_vec, n_out)          # (N,) signed normal EPE
        rmse    = float(np.sqrt(np.mean(epe_n**2)))

        rho_mean = float(np.mean(rho_loc))
        rho_std  = float(np.std(rho_loc))
        history.append({
            "iter":        it,
            "rmse_mm":     round(rmse * 1e3, 4),
            "rho_mean":    round(rho_mean, 4),
            "rho_std":     round(rho_std, 4),
            "sim_time_s":  round(t_sim, 1),
            "best":        False,
        })

        print(f"  [prewarp] iter {it:2d}  RMSE = {rmse*1e3:.3f} mm  "
              f"rho = {rho_mean:.3f}±{rho_std:.3f}  "
              f"epe_max = {np.max(np.abs(epe_n))*1e3:.2f} mm")

        B_pred_last = B_pred.copy()
        state_last  = state

        # ── Track best iterate ────────────────────────────────────────────────
        if rmse < best_rmse:
            best_rmse   = rmse
            best_P      = P.copy()
            best_B_pred = B_pred.copy()
            best_state  = state
            history[-1]["best"] = True
            n_no_improve = 0
            print(f"  [prewarp]           ★ new best  (RMSE={best_rmse*1e3:.3f} mm)")
        else:
            n_no_improve += 1

        # ── 5. Convergence / early-stop checks ────────────────────────────────
        if rmse < tol:
            print(f"\n  [prewarp] ✓ Converged at iteration {it}  "
                  f"(RMSE={rmse*1e3:.3f} mm < tol={tol*1e3:.2f} mm)")
            break
        if it == max_iter - 1:
            print(f"\n  [prewarp] ⚠  Max iterations reached  "
                  f"(RMSE={rmse*1e3:.3f} mm after {max_iter} iters)")
            break
        # Early stop: if 3 consecutive non-improving iterations, revert to best
        if n_no_improve >= 3:
            print(f"\n  [prewarp] ↩  Early stop — {n_no_improve} non-improving iters; "
                  f"reverting to best (RMSE={best_rmse*1e3:.3f} mm)")
            P = best_P.copy()
            break

        # ── 6. EPE correction + Laplacian smoothing ───────────────────────────
        # Gradient descent with simple step-halving backtracking to prevent
        # self-intersection of the prewarp polygon.
        # If current RMSE is worse than best, use a reduced step size.
        base_step = lr if rmse <= best_rmse * 1.1 else lr * 0.4
        step      = base_step
        accepted  = False
        for _bs in range(4):
            lap   = np.roll(P, -1, axis=0) - 2 * P + np.roll(P, 1, axis=0)
            P_cand = P + step * epe_n[:, None] * n_out + smooth * lap
            P_cand = resample_polygon(P_cand, N)
            if _polygon_area(P_cand) > 0:
                P = P_cand
                accepted = True
                break
            step *= 0.5
        if not accepted:
            print(f"  [prewarp]           backtracking failed — applying minimal smoothing")
            lap = np.roll(P, -1, axis=0) - 2 * P + np.roll(P, 1, axis=0)
            P = resample_polygon(P + 0.05 * smooth * lap, N)

    # Use best polygon (not necessarily the last one) for final verification
    print(f"\n  [prewarp] Best polygon: iter {[h['iter'] for h in history if h.get('best')][-1] if any(h.get('best') for h in history) else '?'}  "
          f"RMSE={best_rmse*1e3:.3f} mm")
    P           = best_P
    B_pred_last = best_B_pred
    state_last  = best_state

    # ── Final verification run ────────────────────────────────────────────────
    # Run a clean full simulation on the converged prewarp polygon and save all
    # standard outputs to output_dir/verification/.
    print(f"\n  [prewarp] Running final VERIFICATION simulation on converged prewarp …")
    verif_dir    = output_dir / "verification"
    sim_cfg_v    = _inject_polygon(cfg, P)
    t_v0         = time.perf_counter()
    state_v, summary_v, hist_v, tt_v, opt_v = _fwd.run_sim(sim_cfg_v)
    print(f"  [prewarp] Verification done in {time.perf_counter()-t_v0:.1f} s  "
          f"(rho_mean={summary_v.get('mean_rho_rel_part_final', 0):.3f}  "
          f"phi_mean={summary_v.get('mean_phi_part_final', 0):.3f})")

    _fwd.save_outputs(sim_cfg_v, state_v, summary_v, hist_v, verif_dir,
                      tt_rotation_steps=None)
    _fwd.generate_optimizer_report(sim_cfg_v, opt_v, hist_v, verif_dir)
    try:
        import make_rf_summary_v5 as _v5mod
        _v5mod.make_rf_summary_v5(str(verif_dir))
        print(f"  [rf_summary_v5] Generated in verification/")
    except Exception as _e:
        print(f"  [rf_summary_v5] Skipped: {_e}")

    # Verification EPE: apply the same shrinkage model to the verification output.
    # Use the same radial eff_Leff formula as the optimisation loop (not fixed L_eff).
    dx_g_v     = float(state_v.x[1] - state_v.x[0])
    n_out_v    = compute_outward_normals(P)
    P_ins_v    = P - inset_frac * dx_g_v * n_out_v
    rho_v      = bilinear_sample(state_v.rho_rel, state_v.x, state_v.y, P_ins_v)
    centroid_v = P.mean(axis=0)
    r_loc_v    = np.linalg.norm(P - centroid_v, axis=1)
    eff_Leff_v = np.minimum(np.clip(r_loc_v, 0.001, 0.030), L_eff)
    s_v        = 1.0 - np.sqrt(rho_0 / np.maximum(rho_v, rho_0 + 1e-9))
    B_pred_v   = P - s_v[:, None] * eff_Leff_v[:, None] * n_out_v
    epe_n_v    = np.einsum("ij,ij->i", T_poly - B_pred_v, n_out_v)
    rmse_v     = float(np.sqrt(np.mean(epe_n_v**2)))
    print(f"  [prewarp] Verification RMSE = {rmse_v*1e3:.3f} mm")

    # ── Save all prewarp outputs ──────────────────────────────────────────────
    save_prewarp_outputs(
        output_dir   = output_dir,
        T_poly       = T_poly,
        P_prewarp    = P,
        B_pred_last  = B_pred_last,
        B_pred_verif = B_pred_v,
        epe_n_verif  = epe_n_v,
        history      = history,
        rmse_verif   = rmse_v,
        state_last   = state_last,
        state_verif  = state_v,
        cfg          = cfg,
        sim_cfg      = sim_cfg_v,
        L_eff        = L_eff,
        rho_0        = rho_0,
    )

    print(f"\n  [prewarp] All done.")
    print(f"  Outputs      → {output_dir.resolve()}")
    print(f"  Verification → {verif_dir.resolve()}")
    print(f"  RMSE progression: "
          + "  →  ".join(f"{h['rmse_mm']:.3f}" for h in history)
          + f"  →  {rmse_v*1e3:.3f} mm (verif)")


# ─────────────────────────────────────────────────────────────────────────────
# Output saving
# ─────────────────────────────────────────────────────────────────────────────

def save_prewarp_outputs(
    output_dir:   Path,
    T_poly:       np.ndarray,
    P_prewarp:    np.ndarray,
    B_pred_last:  np.ndarray,
    B_pred_verif: np.ndarray,
    epe_n_verif:  np.ndarray,
    history:      list[dict],
    rmse_verif:   float,
    state_last,
    state_verif,
    cfg:          dict,
    sim_cfg:      dict,
    L_eff:        float,
    rho_0:        float,
) -> None:
    """Write prewarp_vertices.csv, sintered_prediction.csv, prewarp_config.yaml,
    prewarp_summary.json, and prewarp_report.png to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tgt = cfg["target"]
    pw  = cfg["prewarp"]

    # ── CSV: prewarp polygon vertices [mm] ────────────────────────────────────
    csv_path = output_dir / "prewarp_vertices.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_mm", "y_mm"])
        for pt in P_prewarp:
            w.writerow([f"{pt[0]*1e3:.6f}", f"{pt[1]*1e3:.6f}"])
    print(f"  [prewarp] Saved {csv_path.name}  ({len(P_prewarp)} vertices)")

    # ── CSV: predicted post-sinter boundary [mm] ─────────────────────────────
    # B_pred_verif is the predicted shape after sintering the prewarp polygon.
    # Saved as a standalone CSV so it can be loaded, compared, or used in the
    # calibration sweep figure without re-running any simulations.
    pred_path = output_dir / "sintered_prediction.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_mm", "y_mm"])
        for pt in B_pred_verif:
            w.writerow([f"{pt[0]*1e3:.6f}", f"{pt[1]*1e3:.6f}"])
    print(f"  [prewarp] Saved {pred_path.name}  ({len(B_pred_verif)} vertices)")

    # ── YAML: simulation config with injected polygon ─────────────────────────
    yaml_path = output_dir / "prewarp_config.yaml"
    yaml_path.write_text(yaml.dump(sim_cfg, default_flow_style=False))
    print(f"  [prewarp] Saved {yaml_path.name}")

    # ── JSON: summary ─────────────────────────────────────────────────────────
    T_bbox = (T_poly.max(axis=0) - T_poly.min(axis=0)) * 1e3
    P_bbox = (P_prewarp.max(axis=0) - P_prewarp.min(axis=0)) * 1e3
    n_iters_run = len(history)
    converged   = (n_iters_run < pw["max_iter"] and n_iters_run > 0
                   and history[-1]["rmse_mm"] < pw["epe_tol_mm"])
    summary = {
        "target_shape":         tgt["shape"],
        "target_width_mm":      tgt["width"]             * 1e3,
        "target_height_mm":     tgt.get("height", tgt["width"]) * 1e3,
        "n_resample_pts":       len(T_poly),
        "L_eff_mm":             L_eff * 1e3,
        "rho_0":                rho_0,
        "max_iter":             pw["max_iter"],
        "lr":                   pw["lr"],
        "smooth":               pw["smooth"],
        "epe_tol_mm":           pw["epe_tol_mm"],
        "iterations_run":       n_iters_run,
        "converged":            converged,
        "rmse_iter0_mm":        history[0]["rmse_mm"]  if history else None,
        "rmse_final_mm":        history[-1]["rmse_mm"] if history else None,
        "rmse_verification_mm": round(rmse_verif * 1e3, 4),
        "target_bbox_x_mm":     round(float(T_bbox[0]), 4),
        "target_bbox_y_mm":     round(float(T_bbox[1]), 4),
        "prewarp_bbox_x_mm":    round(float(P_bbox[0]), 4),
        "prewarp_bbox_y_mm":    round(float(P_bbox[1]), 4),
        "iteration_history":    history,
        # Per-vertex signed EPE [mm] from the verification run.
        # Positive = target is outward of predicted (under-correction);
        # Negative = target is inward of predicted (over-correction).
        # Useful for calibration sweep polar plots and assist-feature detection.
        "epe_final_per_vertex_mm": np.round(epe_n_verif * 1e3, 4).tolist(),
    }
    json_path = output_dir / "prewarp_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  [prewarp] Saved {json_path.name}")

    # ── PNG: 4-panel prewarp report ───────────────────────────────────────────
    _save_prewarp_figure(
        output_dir   = output_dir,
        T_poly       = T_poly,
        P_prewarp    = P_prewarp,
        B_pred_last  = B_pred_last,
        B_pred_verif = B_pred_verif,
        history      = history,
        rmse_verif   = rmse_verif,
        state_last   = state_last,
        state_verif  = state_verif,
        tgt          = tgt,
        pw           = pw,
    )


def _save_prewarp_figure(
    output_dir:   Path,
    T_poly:       np.ndarray,
    P_prewarp:    np.ndarray,
    B_pred_last:  np.ndarray,
    B_pred_verif: np.ndarray,
    history:      list[dict],
    rmse_verif:   float,
    state_last,
    state_verif,
    tgt:          dict,
    pw:           dict,
) -> None:
    """
    4-panel prewarp report:
      A (top-left)  — Target vs. prewarp polygon overlay
      B (top-right) — Predicted boundary vs. target (from last iteration)
      C (bot-left)  — rho_rel field from VERIFICATION run + both boundaries
      D (bot-right) — RMSE convergence curve
    """
    _sig = 1.2   # Gaussian smoothing sigma for display (standard, locked)

    fig = plt.figure(figsize=(13, 10), facecolor="white", dpi=180)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ── Convert polygons to mm ────────────────────────────────────────────────
    T_mm   = T_poly      * 1e3
    P_mm   = P_prewarp   * 1e3
    Bl_mm  = B_pred_last * 1e3   if B_pred_last.size  else T_mm
    Bv_mm  = B_pred_verif * 1e3

    def _closed(arr):
        return np.vstack([arr, arr[0]])

    # ── Panel A: Target vs. prewarp polygon ──────────────────────────────────
    axA.fill(*_closed(T_mm).T, color="#dce8ff", alpha=0.5, zorder=1)
    axA.plot(*_closed(T_mm).T,  color="#2266cc", lw=1.5, label="Target",  zorder=3)
    axA.fill(*_closed(P_mm).T, color="#ffe8cc", alpha=0.5, zorder=2)
    axA.plot(*_closed(P_mm).T,  color="#cc6600", lw=1.5, ls="--", label="Prewarp", zorder=4)
    axA.set_aspect("equal"); axA.set_xlabel("x [mm]"); axA.set_ylabel("y [mm]")
    axA.set_title("A — Target vs. Prewarp input", fontsize=9, fontweight="bold")
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(True, lw=0.4, alpha=0.5)

    # ── Panel B: Predicted final boundary vs. target (last iteration) ─────────
    axB.fill(*_closed(T_mm).T, color="#dce8ff", alpha=0.5, zorder=1)
    axB.plot(*_closed(T_mm).T,  color="#2266cc", lw=1.5, label="Target",    zorder=3)
    axB.plot(*_closed(Bl_mm).T, color="#22aa44", lw=1.5, ls=":",
             label="Predicted final\n(last iter)",  zorder=4)
    if len(history) > 0:
        rmse_last = history[-1]["rmse_mm"]
        axB.set_title(f"B — Predicted boundary vs. target\n"
                      f"(last iter RMSE = {rmse_last:.3f} mm)", fontsize=9, fontweight="bold")
    else:
        axB.set_title("B — Predicted boundary vs. target", fontsize=9, fontweight="bold")
    axB.set_aspect("equal"); axB.set_xlabel("x [mm]"); axB.set_ylabel("y [mm]")
    axB.legend(fontsize=8, loc="upper right")
    axB.grid(True, lw=0.4, alpha=0.5)

    # ── Panel C: rho_rel from verification run + boundaries ───────────────────
    if state_verif is not None:
        x_mm_g = state_verif.x * 1e3
        y_mm_g = state_verif.y * 1e3
        rho_disp = gaussian_filter(state_verif.rho_rel.astype(float), sigma=_sig)
        rho_disp = np.clip(rho_disp, 0.0, 1.0)   # prevent gaussian overshoot from mismatching colorbar
        pm_disp  = gaussian_filter(state_verif.part_mask.astype(float), sigma=1.0)

        im = axC.imshow(rho_disp, origin="lower",
                        extent=[x_mm_g[0], x_mm_g[-1], y_mm_g[0], y_mm_g[-1]],
                        cmap="plasma", vmin=0.0, vmax=1.0,
                        interpolation="bilinear", aspect="auto")
        plt.colorbar(im, ax=axC, label="ρ_rel", fraction=0.046, pad=0.04)

        # Part boundary (prewarp shape, as printed)
        axC.contour(x_mm_g, y_mm_g, pm_disp, levels=[0.5],
                    colors=["#ff9900"], linewidths=[1.2], linestyles=["-"])
        # Target boundary overlay
        axC.plot(*_closed(T_mm).T, color="#00ccff", lw=1.2, ls="--",
                 label="Target boundary", zorder=5)
        # Predicted final boundary
        axC.plot(*_closed(Bv_mm).T, color="#00ff88", lw=1.0, ls=":",
                 label="Predicted post-sinter", zorder=5)

        axC.set_xlim(x_mm_g[0], x_mm_g[-1])
        axC.set_ylim(y_mm_g[0], y_mm_g[-1])
        axC.set_xlabel("x [mm]"); axC.set_ylabel("y [mm]")
        axC.set_title(f"C — Verification: ρ_rel field\n"
                      f"Verif. RMSE = {rmse_verif*1e3:.3f} mm", fontsize=9, fontweight="bold")
        axC.legend(fontsize=7, loc="upper right",
                   handles=[
                       plt.Line2D([0],[0], color="#ff9900", lw=1.2, label="Prewarp boundary"),
                       plt.Line2D([0],[0], color="#00ccff", lw=1.2, ls="--", label="Target"),
                       plt.Line2D([0],[0], color="#00ff88", lw=1.0, ls=":", label="Predicted post-sinter"),
                   ])
    else:
        axC.text(0.5, 0.5, "No field data", ha="center", va="center", transform=axC.transAxes)
        axC.set_title("C — Verification rho_rel", fontsize=9, fontweight="bold")

    # ── Panel D: RMSE convergence curve ──────────────────────────────────────
    if history:
        iters   = [h["iter"]    for h in history]
        rmses   = [h["rmse_mm"] for h in history]
        axD.plot(iters, rmses, "o-", color="#cc3333", lw=1.8, ms=6, label="EPE RMSE")
        # Verification point
        axD.plot(len(history), rmse_verif * 1e3, "D",
                 color="#0044cc", ms=8, zorder=5, label=f"Verification ({rmse_verif*1e3:.3f} mm)")
        axD.axhline(pw["epe_tol_mm"], color="gray", lw=1.0, ls="--", label=f"Tol {pw['epe_tol_mm']} mm")
        axD.set_xlabel("Iteration"); axD.set_ylabel("RMSE EPE [mm]")
        axD.set_title("D — Convergence history", fontsize=9, fontweight="bold")
        axD.legend(fontsize=8)
        axD.grid(True, lw=0.4, alpha=0.5)
        if len(iters) > 1:
            axD.set_xlim(-0.5, len(history) + 0.5)
    else:
        axD.text(0.5, 0.5, "No iterations", ha="center", va="center",
                 transform=axD.transAxes)
        axD.set_title("D — Convergence history", fontsize=9, fontweight="bold")

    # ── Suptitle ──────────────────────────────────────────────────────────────
    n_it   = len(history)
    tgt_s  = tgt["shape"]
    t_w_mm = tgt["width"] * 1e3
    t_h_mm = tgt.get("height", tgt["width"]) * 1e3
    fig.suptitle(
        f"RFAM Geometry Prewarp — Target: {tgt_s}  {t_w_mm:.1f}×{t_h_mm:.1f} mm  "
        f"|  {n_it} iter{'s' if n_it != 1 else ''}  "
        f"|  Verif. RMSE = {rmse_verif*1e3:.3f} mm",
        fontsize=10, fontweight="bold", y=1.01,
    )

    png_path = output_dir / "prewarp_report.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [prewarp] Saved {png_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone evaluation (--evaluate mode)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_polygon(cfg: dict, polygon_csv: Path, output_dir: Path) -> None:
    """Standalone forward check: load an arbitrary prewarp polygon CSV, run one
    forward simulation, compute the predicted sintered output, and save a 3-panel
    evaluation figure + outputs.

    This is the **independent feedback loop**: any polygon CSV (from a calibration
    sweep, manual edit, or other source) can be evaluated without re-running the
    full optimisation.

    Usage::

        python3 rfam_prewarp.py \\
            --config configs/rfam_prewarp_circle.yaml \\
            --evaluate outputs_eqs/prewarp_calibrate_circle/L_eff_7mm/prewarp_vertices.csv \\
            --output-dir outputs_eqs/prewarp_eval_L7mm
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pw  = cfg["prewarp"]
    tgt = cfg["target"]
    N          = int(tgt.get("n_resample_pts", 512))
    L_eff      = float(pw["L_eff_m"])
    rho_0      = float(pw.get("rho_0", 0.55))
    inset_frac = float(pw.get("inset_cells", 1.5))

    # Build target polygon
    t_w  = float(tgt["width"])
    t_h  = float(tgt.get("height", t_w))
    T_poly = resample_polygon(make_shape(tgt["shape"], t_w, t_h), N)
    if _polygon_area(T_poly) < 0:
        T_poly = T_poly[::-1].copy()

    # Load prewarp polygon from CSV (x_mm, y_mm → metres)
    poly_data = np.loadtxt(polygon_csv, delimiter=",", skiprows=1)  # (M, 2) mm
    P = poly_data * 1e-3
    P = resample_polygon(P, N)
    if _polygon_area(P) < 0:
        P = P[::-1].copy()

    print(f"\n{'='*64}")
    print(f"  RFAM Prewarp — Standalone Evaluation")
    print(f"  Polygon : {polygon_csv.name}")
    print(f"  Target  : {tgt['shape']}  {t_w*1e3:.1f} × {t_h*1e3:.1f} mm")
    print(f"  L_eff={L_eff*1e3:.1f} mm   rho_0={rho_0}   N={N} pts")
    print(f"{'='*64}\n")

    # ── Forward simulation ────────────────────────────────────────────────────
    sim_cfg = _inject_polygon(cfg, P)
    print(f"  [evaluate] running forward sim …")
    t0 = time.perf_counter()
    state, summary, hist_ts, _, _ = _fwd.run_sim(sim_cfg)
    print(f"  [evaluate] done in {time.perf_counter()-t0:.1f} s  "
          f"(rho_mean={summary.get('mean_rho_rel_part_final', 0):.3f}  "
          f"phi_mean={summary.get('mean_phi_part_final', 0):.3f})")

    # Save full forward outputs (electric_fields, thermal_fields, rf_summary_v5 …)
    _fwd.save_outputs(sim_cfg, state, summary, hist_ts, output_dir,
                      tt_rotation_steps=None)
    try:
        import make_rf_summary_v5 as _v5mod
        _v5mod.make_rf_summary_v5(str(output_dir))
    except Exception as _e:
        print(f"  [rf_summary_v5] Skipped: {_e}")

    # ── Predicted sintered boundary (same formula as optimisation loop) ───────
    dx_g      = float(state.x[1] - state.x[0])
    n_out     = compute_outward_normals(P)
    P_inset   = P - inset_frac * dx_g * n_out
    rho_loc   = bilinear_sample(state.rho_rel, state.x, state.y, P_inset)
    centroid  = P.mean(axis=0)
    r_loc     = np.linalg.norm(P - centroid, axis=1)
    eff_Leff  = np.minimum(np.clip(r_loc, 0.001, 0.030), L_eff)
    s         = 1.0 - np.sqrt(rho_0 / np.maximum(rho_loc, rho_0 + 1e-9))
    B_pred    = P - s[:, None] * eff_Leff[:, None] * n_out
    epe_n     = np.einsum("ij,ij->i", T_poly - B_pred, n_out)
    rmse      = float(np.sqrt(np.mean(epe_n**2)))
    epe_max   = float(np.max(np.abs(epe_n))) * 1e3
    print(f"  [evaluate] Predicted RMSE = {rmse*1e3:.3f} mm  max EPE = {epe_max:.3f} mm")

    # ── Save sintered_prediction.csv ──────────────────────────────────────────
    pred_path = output_dir / "sintered_prediction.csv"
    with open(pred_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["x_mm", "y_mm"])
        for pt in B_pred:
            w_csv.writerow([f"{pt[0]*1e3:.6f}", f"{pt[1]*1e3:.6f}"])
    print(f"  [evaluate] Saved {pred_path.name}")

    # ── Density uniformity metrics at part boundary ───────────────────────────
    # Boundary cells = part_mask cells NOT in eroded part_mask (1-cell shell)
    from scipy.ndimage import binary_erosion as _bin_erode
    _eroded = _bin_erode(state.part_mask)
    _bnd_mask = state.part_mask & ~_eroded
    _rho_b = state.rho_rel[_bnd_mask] if _bnd_mask.any() else np.array([0.0])
    _rho_b_std   = float(np.std(_rho_b))
    _rho_b_min   = float(np.min(_rho_b))
    _rho_b_max   = float(np.max(_rho_b))
    _rho_b_ratio = float(_rho_b_min / max(_rho_b_max, 1e-9))

    # ── Save eval_summary.json ────────────────────────────────────────────────
    eval_summary = {
        "polygon_csv":              str(polygon_csv.resolve()),
        "target_shape":             tgt["shape"],
        "L_eff_mm":                 L_eff * 1e3,
        "rho_0":                    rho_0,
        "rmse_mm":                  round(rmse * 1e3, 4),
        "epe_max_mm":               round(epe_max, 4),
        "rho_mean":                 round(float(np.mean(rho_loc)), 4),
        "rho_std":                  round(float(np.std(rho_loc)), 4),
        "rho_boundary_std":         round(_rho_b_std, 4),
        "rho_boundary_min_max_ratio": round(_rho_b_ratio, 4),
        "rho_boundary_min":         round(_rho_b_min, 4),
        "rho_boundary_max":         round(_rho_b_max, 4),
        "ui_rms_part_final":        round(float(hist_ts["ui_rms_part"][-1]) if hist_ts.get("ui_rms_part") else 0.0, 4),
        "epe_per_vertex_mm":        np.round(epe_n * 1e3, 4).tolist(),
    }
    (output_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
    print(f"  [evaluate] Density uniformity:  rho_boundary_std={_rho_b_std:.4f}  "
          f"min/max_ratio={_rho_b_ratio:.3f}  UI_rms={eval_summary['ui_rms_part_final']:.4f}")

    # ── 3-panel evaluation figure ─────────────────────────────────────────────
    _save_evaluate_figure(
        output_dir = output_dir,
        T_poly     = T_poly,
        P_prewarp  = P,
        B_pred     = B_pred,
        state      = state,
        epe_n      = epe_n,
        rmse       = rmse,
        tgt        = tgt,
    )

    print(f"\n  [evaluate] All done.  Outputs → {output_dir.resolve()}")


def _save_evaluate_figure(
    output_dir: Path,
    T_poly:     np.ndarray,
    P_prewarp:  np.ndarray,
    B_pred:     np.ndarray,
    state,
    epe_n:      np.ndarray,
    rmse:       float,
    tgt:        dict,
) -> None:
    """3-panel evaluation figure:
      A — Prewarp input vs. target  ("what goes in")
      B — ρ_rel field from the forward simulation  ("what the physics sees")
      C — Predicted sintered output vs. target  ("what comes out")
    """
    _sig = 1.2  # Gaussian smoothing sigma (locked standard)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white", dpi=180)
    axA, axB, axC = axes

    T_mm = T_poly * 1e3
    P_mm = P_prewarp * 1e3
    B_mm = B_pred * 1e3

    def _closed(arr):
        return np.vstack([arr, arr[0]])

    # ── Panel A: prewarp input vs. target ────────────────────────────────────
    axA.fill(*_closed(T_mm).T, color="#dce8ff", alpha=0.5, zorder=1)
    axA.plot(*_closed(T_mm).T,  color="#2266cc", lw=1.5, label="Target",       zorder=3)
    axA.fill(*_closed(P_mm).T, color="#ffe8cc", alpha=0.4, zorder=2)
    axA.plot(*_closed(P_mm).T,  color="#cc6600", lw=1.5, ls="--",
             label="Prewarp input", zorder=4)
    axA.set_aspect("equal"); axA.set_xlabel("x [mm]"); axA.set_ylabel("y [mm]")
    axA.set_title("A — Input: Prewarp vs. Target", fontsize=9, fontweight="bold")
    axA.legend(fontsize=8); axA.grid(True, lw=0.4, alpha=0.5)

    # ── Panel B: ρ_rel field from forward sim ────────────────────────────────
    x_mm_g = state.x * 1e3
    y_mm_g = state.y * 1e3
    rho_disp = gaussian_filter(state.rho_rel.astype(float), sigma=_sig)
    rho_disp = np.clip(rho_disp, 0.0, 1.0)   # prevent gaussian overshoot from mismatching colorbar
    pm_disp  = gaussian_filter(state.part_mask.astype(float), sigma=1.0)
    im = axB.imshow(rho_disp, origin="lower",
                    extent=[x_mm_g[0], x_mm_g[-1], y_mm_g[0], y_mm_g[-1]],
                    cmap="plasma", vmin=0.0, vmax=1.0,
                    interpolation="bilinear", aspect="auto")
    plt.colorbar(im, ax=axB, label="ρ_rel", fraction=0.046, pad=0.04)
    axB.contour(x_mm_g, y_mm_g, pm_disp, levels=[0.5],
                colors=["#ff9900"], linewidths=[1.2])
    axB.plot(*_closed(T_mm).T, color="#00ccff", lw=1.0, ls="--", zorder=5,
             label="Target")
    axB.set_xlim(x_mm_g[0], x_mm_g[-1]); axB.set_ylim(y_mm_g[0], y_mm_g[-1])
    axB.set_xlabel("x [mm]"); axB.set_ylabel("y [mm]")
    rho_mean = float(state.rho_rel[state.part_mask].mean()) if state.part_mask.any() else 0.0
    axB.set_title(f"B — Forward sim: ρ_rel field  (ρ̄={rho_mean:.3f})",
                  fontsize=9, fontweight="bold")
    axB.legend(fontsize=7, loc="upper right")

    # ── Panel C: predicted sintered output vs. target ────────────────────────
    epe_max_mm = float(np.max(np.abs(epe_n))) * 1e3
    axC.fill(*_closed(T_mm).T, color="#dce8ff", alpha=0.5, zorder=1)
    axC.plot(*_closed(T_mm).T,  color="#2266cc", lw=1.5, label="Target",    zorder=3)
    axC.plot(*_closed(B_mm).T,  color="#22aa44", lw=1.5, ls=":",
             label=f"Predicted sintered\n(RMSE={rmse*1e3:.3f} mm)", zorder=4)
    axC.set_aspect("equal"); axC.set_xlabel("x [mm]"); axC.set_ylabel("y [mm]")
    axC.set_title(f"C — Output: Predicted sintered vs. Target\n"
                  f"max EPE = {epe_max_mm:.3f} mm",
                  fontsize=9, fontweight="bold")
    axC.legend(fontsize=8); axC.grid(True, lw=0.4, alpha=0.5)

    # ── Super-title ───────────────────────────────────────────────────────────
    t_w_mm = tgt["width"] * 1e3
    t_h_mm = tgt.get("height", tgt["width"]) * 1e3
    fig.suptitle(
        f"RFAM Prewarp Evaluation  |  {tgt['shape']}  {t_w_mm:.1f}×{t_h_mm:.1f} mm  "
        f"|  RMSE = {rmse*1e3:.3f} mm  |  max EPE = {epe_max_mm:.3f} mm",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    png_path = output_dir / "eval_report.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [evaluate] Saved {png_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Antenna / Spike Assist Features  (Phase 2 prewarp)
# ─────────────────────────────────────────────────────────────────────────────

def add_spike_features(
    poly: np.ndarray,
    spike_regions: list[dict],
    h_mm: float = 2.0,
    w_base_mm: float = 4.0,
    n_pts: int = 15,
) -> np.ndarray:
    """Insert smooth Gaussian-shaped protrusions (antenna spikes) at specified
    angular regions of a polygon.

    The spike profile is a Gaussian bump in the outward-normal direction:
        delta_r(t) = h * exp(-t² / (2 σ²))
    where t is the arc-distance from the spike centre and σ = w_base / (2√(2 ln 2)).

    Parameters
    ----------
    poly : (N, 2) polygon in metres (CCW, not necessarily closed)
    spike_regions : list of dicts with keys:
        - ``angle_center_deg`` : centre of spike in polar angle from centroid [°]
        - ``angle_width_deg``  : full angular width of the spike footprint [°]
    h_mm : spike peak height [mm]
    w_base_mm : full-width at base of the Gaussian bump [mm]
    n_pts : number of extra polygon vertices inserted inside each spike region

    Returns
    -------
    new_poly : (M, 2) resampled polygon with spikes, same vertex count N as input
    """
    h = h_mm * 1e-3          # metres
    w = w_base_mm * 1e-3      # metres
    sigma_g = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # Gaussian σ

    centroid = poly.mean(axis=0)
    # Compute polar angle of each vertex [radians]
    angles_v = np.arctan2(poly[:, 1] - centroid[1], poly[:, 0] - centroid[0])

    # Build list of new polygon segments; spikes inserted at matching regions
    segments: list[np.ndarray] = []
    i = 0
    N = len(poly)
    while i < N:
        pt = poly[i]
        ang_v = float(angles_v[i])  # current vertex angle [rad]
        inserted = False
        for reg in spike_regions:
            ctr_deg = float(reg["angle_center_deg"])
            wid_deg = float(reg["angle_width_deg"])
            ctr_rad = math.radians(ctr_deg)
            wid_rad = math.radians(wid_deg)
            # Angular distance from spike centre (wrap ±π)
            d_ang = ang_v - ctr_rad
            d_ang = (d_ang + math.pi) % (2 * math.pi) - math.pi
            if abs(d_ang) <= wid_rad / 2.0:
                # This vertex is inside a spike region — build the spike
                n_out_v = np.array([math.cos(ang_v), math.sin(ang_v)])
                # Spike arc: n_pts points spanning the angular width
                ang_lo = ctr_rad - wid_rad / 2.0
                ang_hi = ctr_rad + wid_rad / 2.0
                t_vals = np.linspace(ang_lo, ang_hi, n_pts)
                # Radius of each vertex from centroid (interpolated around the original polygon)
                r_base = np.linalg.norm(pt - centroid)
                spike_pts = []
                for t_ang in t_vals:
                    # Arc-distance from spike centre
                    t_arc = r_base * abs(t_ang - ctr_rad)
                    bump  = h * np.exp(-t_arc**2 / (2.0 * sigma_g**2))
                    n_dir = np.array([math.cos(t_ang), math.sin(t_ang)])
                    spike_pts.append(centroid + r_base * n_dir + bump * n_dir)
                segments.append(np.array(spike_pts))
                # Skip all original vertices inside this spike region
                while i < N:
                    ang_i = float(angles_v[i])
                    d_i   = (ang_i - ctr_rad + math.pi) % (2 * math.pi) - math.pi
                    if abs(d_i) <= wid_rad / 2.0:
                        i += 1
                    else:
                        break
                inserted = True
                break
        if not inserted:
            segments.append(poly[i:i+1].copy())
            i += 1

    new_pts = np.vstack(segments) if segments else poly.copy()
    # Resample back to original vertex count for uniform spacing
    return resample_polygon(new_pts, N)


def detect_spike_regions(
    poly: np.ndarray,
    epe_per_vertex_mm: list[float] | np.ndarray,
    epe_threshold_mm: float = 0.15,
    min_width_deg: float = 20.0,
    merge_gap_deg: float = 10.0,
) -> list[dict]:
    """Auto-detect angular regions with residual EPE above threshold.

    Reads ``epe_per_vertex_mm`` from ``prewarp_summary.json`` and identifies
    connected arc segments where |EPE| > epe_threshold_mm.  These are the
    faces that received insufficient RF energy and are candidates for spike
    assist features.

    Parameters
    ----------
    poly : (N, 2) polygon in metres
    epe_per_vertex_mm : signed EPE at each vertex [mm]; positive = face inside target
    epe_threshold_mm : minimum |EPE| to flag a region [mm]
    min_width_deg : minimum angular width [°] for a region to be returned
    merge_gap_deg : gap [°] below which adjacent regions are merged

    Returns
    -------
    list of {``angle_center_deg``, ``angle_width_deg``} — suitable for
    passing directly to :func:`add_spike_features`.
    """
    epe = np.asarray(epe_per_vertex_mm, dtype=float)
    N = len(poly)
    centroid = poly.mean(axis=0)
    angles_v = np.degrees(np.arctan2(
        poly[:, 1] - centroid[1], poly[:, 0] - centroid[0]
    ))  # –180…+180

    # Flag vertices above threshold
    flagged = np.abs(epe) >= epe_threshold_mm

    # Find contiguous flagged runs (circular)
    regions: list[dict] = []
    in_run = False
    run_start_ang = 0.0
    run_angles: list[float] = []

    def _close_run(run_angs: list[float]) -> None:
        a_lo = min(run_angs); a_hi = max(run_angs)
        w = a_hi - a_lo
        if w >= min_width_deg:
            regions.append({
                "angle_center_deg": (a_lo + a_hi) / 2.0,
                "angle_width_deg":  w + 5.0,   # small padding
            })

    for idx in range(N + 1):   # +1 to close circular run
        i = idx % N
        if flagged[i]:
            if not in_run:
                in_run = True
                run_angles = []
            run_angles.append(float(angles_v[i]))
        else:
            if in_run:
                _close_run(run_angles)
                in_run = False
                run_angles = []

    # Merge adjacent regions within merge_gap_deg
    merged: list[dict] = []
    for reg in sorted(regions, key=lambda r: r["angle_center_deg"]):
        if merged and (reg["angle_center_deg"] - (merged[-1]["angle_center_deg"] + merged[-1]["angle_width_deg"] / 2.0)) < merge_gap_deg:
            # Merge
            lo = merged[-1]["angle_center_deg"] - merged[-1]["angle_width_deg"] / 2.0
            hi = reg["angle_center_deg"] + reg["angle_width_deg"] / 2.0
            merged[-1] = {"angle_center_deg": (lo + hi) / 2.0, "angle_width_deg": hi - lo}
        else:
            merged.append(reg)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Spike co-optimisation: density equalization across faces
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_density_by_angle(
    state,
    poly:            np.ndarray,
    angle_center_deg: float,
    angle_width_deg: float,
    inset_cells:     float = 3.0,
) -> float:
    """Return mean rho_rel sampled *inset_cells* inward from the polygon boundary
    within an angular arc centred at *angle_center_deg*.

    Uses the polygon-vertex inset approach (same as the EPE calculation) rather
    than the outermost grid-cell boundary ring, which stays at initial density
    for face midpoints due to the thin thermal gradient at the part surface.

    Parameters
    ----------
    state            : SimState  (has .rho_rel, .x, .y)
    poly             : (N, 2) polygon in metres
    angle_center_deg : centre angle of the arc [degrees, −180..+180]
    angle_width_deg  : full angular width of the arc [degrees]
    inset_cells      : number of cells to inset inward from boundary [grid cells]

    Returns
    -------
    Mean rho_rel at the inset-boundary arc.  Returns NaN if no vertices match.
    """
    dx_g   = float(state.x[1] - state.x[0])
    n_out  = compute_outward_normals(poly)
    P_in   = poly - inset_cells * dx_g * n_out
    rho_loc = bilinear_sample(state.rho_rel, state.x, state.y, P_in)

    centroid = poly.mean(axis=0)
    angs = np.degrees(np.arctan2(poly[:, 1] - centroid[1],
                                  poly[:, 0] - centroid[0]))

    half_w = angle_width_deg / 2.0
    d_ang  = angs - angle_center_deg
    d_ang  = (d_ang + 180.0) % 360.0 - 180.0
    sel    = np.abs(d_ang) <= half_w
    if not sel.any():
        return float("nan")
    return float(np.mean(rho_loc[sel]))


def _boundary_std_from_state(state) -> tuple[float, float, float]:
    """Compute rho_boundary_std, min/max ratio from a SimState.

    Uses the 1-cell-thick outer shell of part_mask (binary erosion boundary).
    Returns (std, min_max_ratio, rho_mean_boundary).
    """
    from scipy.ndimage import binary_erosion as _be
    eroded = _be(state.part_mask)
    bnd    = state.part_mask & ~eroded
    rho_b  = state.rho_rel[bnd] if bnd.any() else np.array([0.0])
    std    = float(np.std(rho_b))
    ratio  = float(rho_b.min() / max(rho_b.max(), 1e-9))
    mean   = float(rho_b.mean())
    return std, ratio, mean


def run_spike_coopt(
    prewarp_poly:    np.ndarray,
    cfg:             dict,
    output_dir:      Path,
    spike_angles:    list[float],
    angle_width_deg: float,
    w_base_mm:       float = 2.0,
    n_pts_per_spike: int   = 20,
    h_values:        list[float] | None = None,
    h_lo:            float = 0.0,
    h_hi:            float = 6.0,
    n_coarse:        int   = 6,
    refine_bracket:  float = 1.0,
    n_refine:        int   = 4,
) -> dict:
    """Minimise *rho_boundary_std* by tuning spike height h_mm.

    **Strategy: coarse-to-fine search**

    1. *Coarse pass* — evaluate at *n_coarse* evenly-spaced h values in
       [*h_lo*, *h_hi*].  Identifies a bracket around the minimum.
    2. *Refine pass* — evaluate *n_refine* points in a ±*refine_bracket*/2 mm
       window centred on the coarse minimum.
    3. Return the h with lowest *rho_boundary_std*.

    If the user supplies *h_values* explicitly, only those heights are evaluated
    (no coarse-to-fine; useful when you already have sweep data).

    **Why rho_boundary_std?**

    The outermost boundary ring has face midpoints stuck near the initial density
    (0.55) for both ⊥ and ∥ faces (due to the coarse 0.5 mm grid and convective
    cooling at the top face), making face-level density comparisons ambiguous.
    *rho_boundary_std* measures the GLOBAL spread of the 1-cell-thick outer shell
    density.  A lower std means more uniform surface densification — exactly the
    metric that determines geometric accuracy after sintering.

    Parameters
    ----------
    prewarp_poly     : (N,2) polygon in metres (from calibration CSV)
    cfg              : full prewarp YAML config dict
    output_dir       : directory for per-h outputs + final summary
    spike_angles     : angular centres of spike faces [degrees]
    angle_width_deg  : Gaussian spike angular width [degrees] (keep 5–15°)
    w_base_mm        : Gaussian spike base width [mm]
    n_pts_per_spike  : polygon vertices per spike
    h_values         : explicit list of h values to sweep [mm] (overrides coarse/refine)
    h_lo / h_hi      : coarse search range [mm]
    n_coarse         : number of coarse samples in [h_lo, h_hi]
    refine_bracket   : ±half-width [mm] for refinement around coarse minimum
    n_refine         : number of refinement samples

    Returns
    -------
    dict with keys: h_optimal_mm, rho_bnd_std_optimal, rho_bnd_ratio_optimal,
                    improvement_pct, converged, all_iters, poly_optimal
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    spike_regions = [
        {"angle_center_deg": a, "angle_width_deg": angle_width_deg}
        for a in spike_angles
    ]

    print(f"\n{'='*64}")
    print(f"  Spike Co-Optimisation — Boundary Std Minimisation")
    print(f"  Spike faces  : {spike_angles}°  width={angle_width_deg}°  w_base={w_base_mm} mm")
    print(f"  Search range : h_mm ∈ [{h_lo:.1f}, {h_hi:.1f}] mm")
    print(f"{'='*64}\n")

    def _make_spiked_poly(h_mm: float) -> np.ndarray:
        return add_spike_features(
            prewarp_poly, spike_regions=spike_regions,
            h_mm=h_mm, w_base_mm=w_base_mm, n_pts=n_pts_per_spike,
        )

    def _eval_h(h_mm: float, tag: str) -> dict:
        """Run one forward sim; return metrics dict."""
        iter_dir = output_dir / tag
        iter_dir.mkdir(parents=True, exist_ok=True)
        poly_s = _make_spiked_poly(h_mm)

        # Save polygon CSV
        import csv as _csv
        with open(iter_dir / "spiked_polygon.csv", "w", newline="") as f:
            w_csv = _csv.writer(f)
            w_csv.writerow(["x_mm", "y_mm"])
            for pt in poly_s: w_csv.writerow([f"{pt[0]*1e3:.6f}", f"{pt[1]*1e3:.6f}"])

        sim_cfg = _inject_polygon(cfg, poly_s)
        state, summary, hist_ts, _, _ = _fwd.run_sim(sim_cfg)
        _fwd.save_outputs(sim_cfg, state, summary, hist_ts, iter_dir,
                          tt_rotation_steps=None)

        bnd_std, bnd_ratio, bnd_mean = _boundary_std_from_state(state)
        rho_mean = float(summary.get("mean_rho_rel_part_final", 0))
        print(f"    h={h_mm:.3f} mm  bnd_std={bnd_std:.4f}  "
              f"min/max={bnd_ratio:.4f}  rho_mean={rho_mean:.4f}")
        return {
            "h_mm": h_mm, "tag": tag,
            "rho_bnd_std":   bnd_std,
            "rho_bnd_ratio": bnd_ratio,
            "rho_bnd_mean":  bnd_mean,
            "rho_mean":      rho_mean,
            "poly_spiked":   poly_s,
        }

    all_iters: list[dict] = []

    # ── Determine h values to evaluate ───────────────────────────────────────
    if h_values is not None:
        h_eval = sorted(set(h_values))
        print(f"  Using explicit h_values: {h_eval}")
    else:
        # Coarse pass
        h_coarse = list(np.linspace(h_lo, h_hi, n_coarse))
        h_eval   = h_coarse
        print(f"  Coarse pass: {n_coarse} samples ∈ [{h_lo}, {h_hi}]")

    # Evaluate all h values
    for h in h_eval:
        tag = f"h{h:.2f}mm".replace(".", "p")
        r = _eval_h(h, tag)
        all_iters.append(r)

    # Find coarse minimum
    stds = [r["rho_bnd_std"] for r in all_iters]
    best_idx = int(np.argmin(stds))
    h_best   = all_iters[best_idx]["h_mm"]
    std_best = stds[best_idx]

    # ── Refinement pass (only if explicit h_values not given) ────────────────
    if h_values is None and n_refine > 0 and best_idx > 0:
        h_lo_r = max(h_lo, h_best - refine_bracket / 2.0)
        h_hi_r = min(h_hi, h_best + refine_bracket / 2.0)
        h_refine = [h for h in np.linspace(h_lo_r, h_hi_r, n_refine + 2)[1:-1]
                    if not any(abs(h - r["h_mm"]) < 0.01 for r in all_iters)]
        if h_refine:
            print(f"\n  Refinement pass: {len(h_refine)} samples around h≈{h_best:.2f}mm")
            for h in h_refine:
                tag = f"refine_h{h:.3f}mm".replace(".", "p")
                r   = _eval_h(h, tag)
                all_iters.append(r)
            # Re-find best
            stds     = [r["rho_bnd_std"] for r in all_iters]
            best_idx = int(np.argmin(stds))
            h_best   = all_iters[best_idx]["h_mm"]
            std_best = stds[best_idx]

    std_baseline = all_iters[0]["rho_bnd_std"]  # h=h_lo (usually 0)
    improvement  = 100.0 * (std_baseline - std_best) / max(std_baseline, 1e-9)
    poly_opt     = all_iters[best_idx]["poly_spiked"]

    print(f"\n  [coopt] h_optimal = {h_best:.3f} mm  "
          f"bnd_std = {std_best:.4f}  (baseline {std_baseline:.4f}  "
          f"improvement {improvement:.1f}%)")

    # ── Save summary ──────────────────────────────────────────────────────────
    json_iters = [{k: v for k, v in r.items() if k != "poly_spiked"} for r in all_iters]
    result = {
        "h_optimal_mm":          round(h_best, 4),
        "rho_bnd_std_optimal":   round(std_best, 4),
        "rho_bnd_ratio_optimal": round(all_iters[best_idx]["rho_bnd_ratio"], 4),
        "rho_bnd_std_baseline":  round(std_baseline, 4),
        "improvement_pct":       round(improvement, 2),
        "spike_angles":          spike_angles,
        "angle_width_deg":       angle_width_deg,
        "w_base_mm":             w_base_mm,
        "all_iters":             json_iters,
        "poly_optimal":          poly_opt,   # serialised separately
    }

    (output_dir / "coopt_summary.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "poly_optimal"}, indent=2)
    )
    import csv as _csv
    with open(output_dir / "spiked_polygon_optimal.csv", "w", newline="") as f:
        w_csv = _csv.writer(f)
        w_csv.writerow(["x_mm", "y_mm"])
        for pt in poly_opt: w_csv.writerow([f"{pt[0]*1e3:.6f}", f"{pt[1]*1e3:.6f}"])

    _save_coopt_figure(output_dir, all_iters, prewarp_poly, poly_opt, spike_angles)

    print(f"  [coopt] Outputs → {output_dir.resolve()}")
    return result

    # Write JSON summary (excluding poly array)
    json_summary = {k: v for k, v in result.items() if k != "poly_optimal"}
    (output_dir / "coopt_summary.json").write_text(json.dumps(json_summary, indent=2))

    # Write optimal polygon CSV
    poly_csv_path = output_dir / "spiked_polygon_optimal.csv"
    with open(poly_csv_path, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["x_mm", "y_mm"])
        for pt in poly_opt:
            w.writerow([f"{pt[0]*1e3:.6f}", f"{pt[1]*1e3:.6f}"])

    # ── Summary figure ────────────────────────────────────────────────────────
    _save_coopt_figure(output_dir, all_iters, prewarp_poly, poly_opt,
                       spike_angles, ref_angles)

    print(f"\n  [coopt] Done.  h_optimal={h_opt:.3f} mm  gap={gap_final:.4f}  "
          f"converged={converged}")
    print(f"  [coopt] Outputs → {output_dir.resolve()}")

    return result


def _save_coopt_figure(
    output_dir:   Path,
    all_iters:    list[dict],
    poly_base:    np.ndarray,
    poly_opt:     np.ndarray,
    spike_angles: list[float],
) -> None:
    """3-panel co-optimisation summary figure.

    Panel A — rho_boundary_std vs. h_spike  (primary optimisation metric)
    Panel B — rho_bnd_ratio (min/max) and rho_mean vs. h_spike
    Panel C — Polygon comparison: baseline vs. optimal spiked input
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white", dpi=180)
    axA, axB, axC = axes

    # Sort by h_mm for clean line plot
    iters_sorted = sorted(all_iters, key=lambda d: d["h_mm"])
    h_vals  = [d["h_mm"]         for d in iters_sorted]
    stds    = [d["rho_bnd_std"]  for d in iters_sorted]
    ratios  = [d["rho_bnd_ratio"] for d in iters_sorted]
    means   = [d["rho_mean"]     for d in iters_sorted]

    _best_idx = int(np.argmin(stds))
    _h_best   = h_vals[_best_idx]
    _std_best = stds[_best_idx]

    # Panel A — boundary std vs. h
    axA.plot(h_vals, stds, "o-", color="tab:red", lw=2, ms=7, label="ρ_bnd_std")
    axA.axvline(_h_best, color="tab:red", lw=1.0, ls="--", alpha=0.6,
                label=f"h_opt={_h_best:.2f} mm")
    axA.axhline(_std_best, color="tab:red", lw=0.8, ls=":", alpha=0.4)
    axA.set_xlabel("Spike height h [mm]")
    axA.set_ylabel("Boundary ρ_rel std  (lower = more uniform)")
    axA.set_title("A — Boundary Density Std vs. h_spike", fontsize=9, fontweight="bold")
    axA.legend(fontsize=8); axA.grid(True, lw=0.4, alpha=0.5)

    # Panel B — min/max ratio and mean density vs. h
    ax2 = axB.twinx()
    axB.plot(h_vals, ratios, "s-", color="tab:blue",   lw=2, ms=7, label="min/max ratio ↑")
    ax2.plot(h_vals, means,  "^-", color="tab:green",  lw=1.5, ms=6, label="rho_mean →")
    axB.set_xlabel("Spike height h [mm]")
    axB.set_ylabel("min/max ratio  (higher = more uniform)", color="tab:blue")
    ax2.set_ylabel("Mean ρ_rel (part)", color="tab:green")
    axB.tick_params(axis="y", colors="tab:blue")
    ax2.tick_params(axis="y", colors="tab:green")
    _l1, _lb1 = axB.get_legend_handles_labels()
    _l2, _lb2 = ax2.get_legend_handles_labels()
    axB.legend(_l1 + _l2, _lb1 + _lb2, fontsize=8)
    axB.set_title("B — Min/Max Ratio & Mean Density", fontsize=9, fontweight="bold")
    axB.grid(True, lw=0.4, alpha=0.5)

    # Panel C — polygon comparison
    def _closed(arr):
        return np.vstack([arr, arr[0]])
    pb_mm = poly_base * 1e3
    po_mm = poly_opt  * 1e3
    axC.plot(*_closed(pb_mm).T, "-",  color="tab:blue",   lw=1.5, label="Baseline (no spike)")
    axC.plot(*_closed(po_mm).T, "--", color="tab:orange",  lw=1.5,
             label=f"Optimal h={_h_best:.2f} mm")
    axC.set_aspect("equal")
    axC.set_xlabel("x [mm]"); axC.set_ylabel("y [mm]")
    axC.set_title("C — Baseline vs. Optimal Spiked Input", fontsize=9, fontweight="bold")
    axC.legend(fontsize=8); axC.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    out_path = output_dir / "coopt_report.png"
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [coopt] Figure saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="ILT-inspired geometry prewarp for RFAM RF sintering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     type=Path, required=True,
                   help="Path to rfam_prewarp_*.yaml config file")
    p.add_argument("--output-dir", type=Path,
                   default=Path("./outputs_eqs/prewarp_run"),
                   help="Directory for all prewarp outputs")
    p.add_argument("--evaluate",   type=Path, default=None, metavar="POLYGON_CSV",
                   help="Standalone evaluation mode: load polygon from CSV and run "
                        "one forward check (skip optimisation loop entirely)")

    # ── Spike co-optimisation mode ────────────────────────────────────────────
    p.add_argument("--coopt",      type=Path, default=None, metavar="POLYGON_CSV",
                   help="Spike co-optimisation mode: load polygon CSV and iteratively "
                        "adjust spike height to equalise face densities")
    p.add_argument("--coopt-spike-angles", type=float, nargs="+", default=[0.0, 180.0],
                   help="Angular centres of spike faces [degrees] for co-optimisation")
    p.add_argument("--coopt-angle-width",  type=float, default=10.0,
                   help="Gaussian spike angular width [degrees] (keep narrow: 5–15°)")
    p.add_argument("--coopt-w-base",       type=float, default=2.0,
                   help="Spike Gaussian base width [mm]")
    p.add_argument("--coopt-h-hi",         type=float, default=6.0,
                   help="Upper bound for spike height search [mm]")
    p.add_argument("--coopt-n-coarse",     type=int,   default=6,
                   help="Number of coarse h samples in [0, h_hi]")
    p.add_argument("--coopt-n-refine",     type=int,   default=4,
                   help="Number of refinement samples around coarse minimum")
    p.add_argument("--coopt-h-values",     type=float, nargs="+", default=None,
                   help="Explicit h_mm values to sweep (overrides coarse/refine search)")

    args = p.parse_args()
    cfg  = _load_cfg(args.config)

    if args.coopt is not None:
        # Load polygon CSV
        poly_data = np.loadtxt(args.coopt, delimiter=",", skiprows=1)  # (M,2) mm
        poly = poly_data * 1e-3
        from shapes import resample_polygon as _rsp
        N = int(cfg["target"].get("n_resample_pts", 512))
        poly = _rsp(poly, N)
        run_spike_coopt(
            prewarp_poly    = poly,
            cfg             = cfg,
            output_dir      = args.output_dir,
            spike_angles    = args.coopt_spike_angles,
            angle_width_deg = args.coopt_angle_width,
            w_base_mm       = args.coopt_w_base,
            h_hi            = args.coopt_h_hi,
            n_coarse        = args.coopt_n_coarse,
            n_refine        = args.coopt_n_refine,
            h_values        = args.coopt_h_values,
        )
    elif args.evaluate is not None:
        evaluate_polygon(cfg, args.evaluate, args.output_dir)
    else:
        run_prewarp(cfg, args.output_dir)


if __name__ == "__main__":
    main()
