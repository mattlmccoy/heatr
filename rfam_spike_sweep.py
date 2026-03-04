"""rfam_spike_sweep.py — Antenna/Spike Assist Feature Parameter Sweep

Sweeps spike height (h_mm) for a fixed spike geometry and placement, evaluating
how antenna spikes on under-heated faces affect final part density uniformity.

Concept
-------
The EPE prewarp corrects geometry but cannot equalize density across faces with
very different RF coupling (e.g., left/right faces parallel to E-field receive
~1.6× mean Q_rf vs. top/bottom perpendicular faces at ~11× mean Q_rf). Adding
small convex spike protrusions at under-heated faces concentrates E-field at the
tip (RF antenna effect), increasing local Q_rf and densification.

This script:
1. Starts from an existing prewarp polygon CSV (output of rfam_prewarp.py)
2. Generates spiked variants at h_mm = 0 (baseline) + specified values
3. Runs a forward sim for each variant via rfam_prewarp.evaluate_polygon()
4. Plots density uniformity metrics vs. h_spike in a 4-panel figure

Usage
-----
python3 rfam_spike_sweep.py \\
    --prewarp-csv  outputs_eqs/prewarp_square_v2/L_eff_m_7mm/prewarp_vertices.csv \\
    --config       outputs_eqs/prewarp_square_v2/L_eff_m_7mm/prewarp_config.yaml \\
    --output-dir   outputs_eqs/spike_sweep_square \\
    --spike-angles 0 180 \\
    --angle-width  40 \\
    --h-values     0.0 1.0 2.0 3.0 4.0 \\
    --w-base       5.0 \\
    --r-tip        0.5

Output files
------------
spike_sweep.png          — 4-panel comparison figure
spike_sweep.csv          — one row per h_spike with all metrics
<h_mm>mm/               — per-variant subdirectory with eval outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml

# ── Import project modules ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import rfam_prewarp as pw


# ─────────────────────────────────────────────────────────────────────────────
# Core sweep logic
# ─────────────────────────────────────────────────────────────────────────────

def run_spike_sweep(
    prewarp_csv: Path,
    cfg_path: Path,
    output_dir: Path,
    spike_angles: list[float],
    angle_width_deg: float,
    h_values_mm: list[float],
    w_base_mm: float = 5.0,
    n_pts_per_spike: int = 15,
) -> list[dict]:
    """Run the spike height sweep and return list of result dicts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(cfg_path.read_text())
    antennae_enabled = pw.is_antennae_enabled(cfg)
    if not antennae_enabled:
        print("  [antennae] disabled by config; forcing baseline-only sweep (h=0)")
        h_values_mm = [0.0]

    # Load base prewarp polygon (mm → metres)
    poly_data = np.loadtxt(prewarp_csv, delimiter=",", skiprows=1)  # (N, 2) mm
    P_base = poly_data * 1e-3   # metres

    # Build spike regions list
    spike_regions = [
        {"angle_center_deg": ang, "angle_width_deg": angle_width_deg}
        for ang in spike_angles
    ]

    results: list[dict] = []
    total = len(h_values_mm)
    print(f"\n{'='*64}")
    print(f"  RFAM Spike Sweep — {total} variants")
    print(f"  Spike angles: {spike_angles}°  width: {angle_width_deg}°")
    print(f"  h_values [mm]: {h_values_mm}   w_base={w_base_mm} mm")
    print(f"  Output: {output_dir}")
    print(f"{'='*64}\n")

    for i, h_mm in enumerate(h_values_mm):
        h_tag = f"h{h_mm:.1f}mm".replace(".", "p")
        val_dir = output_dir / h_tag
        val_dir.mkdir(parents=True, exist_ok=True)

        # Build spiked polygon (h=0 → baseline, no spikes added)
        if h_mm > 0.0:
            P_spiked = pw.add_antennae_features(
                P_base,
                spike_regions,
                h_mm=h_mm,
                w_base_mm=w_base_mm,
                n_pts=n_pts_per_spike,
            )
        else:
            P_spiked = P_base.copy()

        # Save spiked polygon as CSV for evaluate_polygon()
        spiked_csv = val_dir / "spiked_polygon.csv"
        with open(spiked_csv, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(["x_mm", "y_mm"])
            for pt in P_spiked:
                w_csv.writerow([f"{pt[0]*1e3:.6f}", f"{pt[1]*1e3:.6f}"])

        print(f"  [{i+1}/{total}] h={h_mm:.1f} mm → {h_tag}/")
        t0 = time.perf_counter()
        pw.evaluate_polygon(cfg, spiked_csv, val_dir)
        elapsed = time.perf_counter() - t0
        print(f"    done in {elapsed:.1f} s")

        # Load eval summary
        summary_path = val_dir / "eval_summary.json"
        eval_sum = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        results.append({
            "h_mm":                  h_mm,
            "h_tag":                 h_tag,
            "val_dir":               val_dir,
            "poly_spiked":           P_spiked,
            "rmse_mm":               float(eval_sum.get("rmse_mm", 0.0)),
            "epe_max_mm":            float(eval_sum.get("epe_max_mm", 0.0)),
            "rho_boundary_std":      float(eval_sum.get("rho_boundary_std", 0.0)),
            "rho_boundary_min_max":  float(eval_sum.get("rho_boundary_min_max_ratio", 0.0)),
            "ui_rms_final":          float(eval_sum.get("ui_rms_part_final", 0.0)),
            "rho_mean":              float(eval_sum.get("rho_mean", 0.0)),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def _plot_spike_sweep(
    results: list[dict],
    output_dir: Path,
    cfg: dict,
    spike_angles: list[float],
    angle_width_deg: float,
) -> None:
    """Generate the 4-panel spike sweep comparison figure."""
    n = len(results)
    cmap = cm.get_cmap("plasma", max(n, 2))
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    tgt = cfg.get("target", {})
    t_w = float(tgt.get("width", 0.020))
    t_h = float(tgt.get("height", t_w))
    T_poly = pw.resample_polygon(pw.make_shape(tgt.get("shape", "square"), t_w, t_h), 512)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Panel A: spiked polygon overlays ─────────────────────────────────────
    _T_mm = T_poly * 1e3
    axA.fill(np.append(_T_mm[:, 0], _T_mm[0, 0]),
             np.append(_T_mm[:, 1], _T_mm[0, 1]),
             alpha=0.08, color="tab:blue", zorder=1)
    axA.plot(np.append(_T_mm[:, 0], _T_mm[0, 0]),
             np.append(_T_mm[:, 1], _T_mm[0, 1]),
             "-", lw=2.0, color="tab:blue", label="Target", zorder=2)
    # Mark spike regions
    for ang_c in spike_angles:
        for r_sign in [-0.5, 0.5]:
            ang_r = math.radians(ang_c + r_sign * angle_width_deg / 2.0)
            axA.axline((0, 0), slope=math.tan(ang_r), ls=":", lw=0.6,
                       color="gray", alpha=0.5)
    for res, col in zip(results, colors):
        P_mm = res["poly_spiked"] * 1e3
        lw = 1.8 if res["h_mm"] == 0.0 else 1.0
        ls = "--" if res["h_mm"] == 0.0 else "-"
        label = f"h={res['h_mm']:.1f} mm"
        axA.plot(np.append(P_mm[:, 0], P_mm[0, 0]),
                 np.append(P_mm[:, 1], P_mm[0, 1]),
                 ls, lw=lw, color=col, label=label, zorder=3)
    axA.set_aspect("equal"); axA.set_title("A — Spiked polygons vs. target")
    axA.set_xlabel("x [mm]"); axA.set_ylabel("y [mm]")
    axA.legend(fontsize=7, loc="upper right")
    axA.grid(alpha=0.2)

    # ── Panel B: rho_rel heatmaps from each sim (small subimages) ────────────
    # Load rho_rel fields and show as mini-heatmaps tiled into Panel B
    n_cols = min(n, 3)
    n_rows_sub = math.ceil(n / n_cols)
    axB.set_title("B — Final ρ_rel fields (h=0 → h_max)")
    axB.axis("off")
    # Create sub-axes grid inside Panel B
    _b_pos = axB.get_position()
    _sub_w = _b_pos.width  / n_cols
    _sub_h = _b_pos.height / max(n_rows_sub, 1)
    sub_axes = []
    for ri in range(n_rows_sub):
        for ci in range(n_cols):
            idx = ri * n_cols + ci
            if idx >= n:
                break
            _ax = fig.add_axes([
                _b_pos.x0 + ci * _sub_w,
                _b_pos.y0 + (n_rows_sub - 1 - ri) * _sub_h,
                _sub_w * 0.92,
                _sub_h * 0.88,
            ])
            sub_axes.append((_ax, idx))

    for _ax, idx in sub_axes:
        res = results[idx]
        # Try to load rho_rel from the saved fields file
        fields_path = res["val_dir"] / "fields.npz"
        if fields_path.exists():
            _npz = np.load(fields_path)
            rho_arr = _npz.get("rho_rel", None)
            if rho_arr is not None:
                from scipy.ndimage import gaussian_filter as _gf
                rho_d = np.clip(_gf(rho_arr.astype(float), sigma=1.2), 0.0, 1.0)
                _ax.imshow(rho_d, origin="lower", cmap="plasma",
                           vmin=0.0, vmax=1.0, interpolation="bilinear",
                           aspect="auto")
            else:
                _ax.text(0.5, 0.5, "no rho_rel", ha="center", va="center",
                         transform=_ax.transAxes, fontsize=7)
        else:
            _ax.text(0.5, 0.5, "no fields.npz", ha="center", va="center",
                     transform=_ax.transAxes, fontsize=7)
        _ax.set_xticks([]); _ax.set_yticks([])
        _ax.set_title(f"h={res['h_mm']:.1f} mm", fontsize=7, pad=2)

    # ── Panel C: RMSE vs. h_spike ─────────────────────────────────────────────
    h_arr  = np.array([r["h_mm"]              for r in results])
    rmse   = np.array([r["rmse_mm"]           for r in results])
    axC.plot(h_arr, rmse, "o-", lw=1.8, color="tab:blue", label="Pred. RMSE [mm]")
    axC.set_xlabel("Spike height h [mm]")
    axC.set_ylabel("Predicted RMSE [mm]")
    axC.set_title("C — Geometric accuracy vs. spike height")
    axC.grid(alpha=0.25)
    axC.legend(fontsize=8)

    # ── Panel D: Density uniformity vs. h_spike ───────────────────────────────
    rho_b_std  = np.array([r["rho_boundary_std"]     for r in results])
    rho_b_rat  = np.array([r["rho_boundary_min_max"]  for r in results])
    ui_rms     = np.array([r["ui_rms_final"]           for r in results])

    axD_l = axD
    axD_r = axD.twinx()
    l1, = axD_l.plot(h_arr, rho_b_std, "s-", lw=1.8, color="tab:red",
                     label="ρ_bnd std ↓ better")
    l2, = axD_l.plot(h_arr, ui_rms,    "^-", lw=1.2, color="tab:orange",
                     label="UI_rms ↓ better")
    l3, = axD_r.plot(h_arr, rho_b_rat, "D--", lw=1.2, color="tab:green",
                     label="ρ_bnd min/max ↑ better")
    axD_l.set_xlabel("Spike height h [mm]")
    axD_l.set_ylabel("ρ_boundary_std / UI_rms", color="tab:red")
    axD_r.set_ylabel("ρ_boundary min/max ratio", color="tab:green")
    axD_l.tick_params(axis="y", colors="tab:red")
    axD_r.tick_params(axis="y", colors="tab:green")
    axD_l.set_title("D — Density uniformity vs. spike height")
    axD_l.legend(handles=[l1, l2, l3], fontsize=8, loc="best")
    axD_l.grid(alpha=0.25)

    # Annotate best h_spike (lowest rho_b_std)
    best_i = int(np.argmin(rho_b_std))
    axD_l.axvline(h_arr[best_i], color="tab:red", lw=0.8, ls="--", alpha=0.6)
    axD_l.text(h_arr[best_i] + 0.05, axD_l.get_ylim()[1] * 0.95,
               f"best h={h_arr[best_i]:.1f} mm",
               color="tab:red", fontsize=7, va="top")

    fig.suptitle(
        f"Spike Assist Feature Sweep — spike angles={spike_angles}°  "
        f"width={angle_width_deg}°  w_base={results[0]['poly_spiked'].shape[0]} pts",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    out_path = output_dir / "spike_sweep.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [spike_sweep] Saved {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV summary
# ─────────────────────────────────────────────────────────────────────────────

def _save_sweep_csv(results: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "spike_sweep.csv"
    fields = ["h_mm", "rmse_mm", "epe_max_mm",
              "rho_boundary_std", "rho_boundary_min_max", "ui_rms_final", "rho_mean"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: round(r[k], 5) for k in fields})
    print(f"  [spike_sweep] Saved {csv_path.name}")

    # Print summary table
    print(f"\n  {'h_mm':>6} | {'RMSE mm':>8} | {'ρ_bnd_std':>10} | {'min/max':>8} | {'UI_rms':>7}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}")
    for r in results:
        print(f"  {r['h_mm']:6.1f} | {r['rmse_mm']:8.4f} | {r['rho_boundary_std']:10.4f} | "
              f"{r['rho_boundary_min_max']:8.4f} | {r['ui_rms_final']:7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="RFAM Spike Assist Feature Sweep — sweep spike height for density uniformity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prewarp-csv",   type=Path, required=True,
                   help="Path to prewarp_vertices.csv (base polygon without spikes)")
    p.add_argument("--config",        type=Path, required=True,
                   help="Path to prewarp_config.yaml (same config used to generate the polygon)")
    p.add_argument("--output-dir",    type=Path,
                   default=Path("./outputs_eqs/spike_sweep"),
                   help="Output directory")
    p.add_argument("--spike-angles",  type=float, nargs="+", required=True,
                   metavar="DEG",
                   help="Polar angles [°] at which to place spikes (e.g. 0 180 for left/right faces)")
    p.add_argument("--angle-width",   type=float, default=40.0,
                   help="Full angular width of each spike footprint [°]")
    p.add_argument("--h-values",      type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 3.0, 4.0],
                   help="Spike height values to sweep [mm]. Include 0.0 for baseline.")
    p.add_argument("--w-base",        type=float, default=5.0,
                   help="Gaussian bump base width [mm]")
    p.add_argument("--r-tip",         type=float, default=0.5,
                   help="(Reserved) Tip radius of curvature [mm] — future use")
    p.add_argument("--n-pts",         type=int,   default=15,
                   help="Number of vertices inserted per spike region")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    results = run_spike_sweep(
        prewarp_csv     = args.prewarp_csv,
        cfg_path        = args.config,
        output_dir      = args.output_dir,
        spike_angles    = args.spike_angles,
        angle_width_deg = args.angle_width,
        h_values_mm     = args.h_values,
        w_base_mm       = args.w_base,
        n_pts_per_spike = args.n_pts,
    )

    _plot_spike_sweep(
        results         = results,
        output_dir      = args.output_dir,
        cfg             = cfg,
        spike_angles    = args.spike_angles,
        angle_width_deg = args.angle_width,
    )
    _save_sweep_csv(results, args.output_dir)

    # Report best h_spike
    best = min(results, key=lambda r: r["rho_boundary_std"])
    print(f"\n  ✓ Best h_spike = {best['h_mm']:.1f} mm  "
          f"(ρ_boundary_std={best['rho_boundary_std']:.4f}  "
          f"RMSE={best['rmse_mm']:.3f} mm)")
    print(f"  → spike_sweep.png + spike_sweep.csv in {args.output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
