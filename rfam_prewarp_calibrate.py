#!/usr/bin/env python3
"""rfam_prewarp_calibrate.py — Parameter sweep for RFAM prewarp calibration.

Sweeps a scalar YAML config parameter (e.g. prewarp.L_eff_m) over a list of
values, runs rfam_prewarp.run_prewarp() for each, and generates a multi-panel
comparison figure so the user can select the best calibration.

Usage
-----
    python3 rfam_prewarp_calibrate.py \\
        --config configs/rfam_prewarp_circle.yaml \\
        --output-dir outputs_eqs/prewarp_calibrate_circle \\
        --sweep-param prewarp.L_eff_m \\
        --values 0.003 0.005 0.007 0.010 0.013 0.016

The --sweep-param accepts any dot-separated YAML key path, so other scalar
params (e.g. prewarp.lr, prewarp.smooth) can also be swept.

Output layout
--------------
    <output-dir>/
      L_eff_m_3mm/          ← one sub-dir per sweep value
        prewarp_vertices.csv
        sintered_prediction.csv
        prewarp_summary.json
        prewarp_report.png
        verification/
      L_eff_m_5mm/
        ...
      calibration_sweep.png  ← 4-panel comparison figure (all L_eff values)
      calibration_sweep.csv  ← one row per value (summary stats)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import yaml

# ── Import prewarp module ─────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import rfam_prewarp as pw   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_nested(d: dict, dotted_key: str, value) -> None:
    """Set a value in a nested dict using a dot-separated key path."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _fmt_tag(param_name: str, value: float) -> str:
    """Create a short directory-safe tag for a parameter value."""
    short = param_name.split(".")[-1]          # e.g. "L_eff_m" → "L_eff_m"
    if "m" in short and not short.endswith("m3"):  # likely in metres
        return f"{short}_{value * 1e3:.0f}mm"
    return f"{short}_{value}"


def _closed(arr: np.ndarray) -> np.ndarray:
    """Close a polygon by appending the first point."""
    return np.vstack([arr, arr[0]])


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep runner
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    base_cfg_path: Path,
    output_dir:    Path,
    param_path:    str,
    values:        list[float],
) -> list[dict]:
    """Run prewarp for each parameter value and collect results.

    Returns a list of result dicts, one per sweep value, suitable for plotting
    and CSV export.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    t_sweep_start = time.perf_counter()
    print(f"\n{'='*64}")
    print(f"  RFAM Prewarp Calibration Sweep")
    print(f"  Config : {base_cfg_path}")
    print(f"  Param  : {param_path}")
    print(f"  Values : {values}")
    print(f"{'='*64}\n")

    results: list[dict] = []

    for i, val in enumerate(values):
        tag     = _fmt_tag(param_path, val)
        val_dir = output_dir / tag
        print(f"\n  ── Sweep {i+1}/{len(values)}: {param_path} = {val}  [{tag}] ──")

        # Load fresh config and inject sweep value
        cfg = yaml.safe_load(base_cfg_path.read_text())
        _set_nested(cfg, param_path, float(val))

        # Run full prewarp (optimisation + verification)
        t0 = time.perf_counter()
        pw.run_prewarp(cfg, val_dir)
        elapsed = time.perf_counter() - t0
        print(f"  └─ Sweep {i+1}/{len(values)} done in {elapsed:.1f} s")

        # ── Load results ──────────────────────────────────────────────────────
        summary = json.loads((val_dir / "prewarp_summary.json").read_text())

        poly_csv = np.loadtxt(val_dir / "prewarp_vertices.csv",
                              delimiter=",", skiprows=1)          # (N, 2) mm
        pred_csv = np.loadtxt(val_dir / "sintered_prediction.csv",
                              delimiter=",", skiprows=1)          # (N, 2) mm

        results.append({
            "label":         tag,
            "param_val":     val,
            "poly_mm":       poly_csv,             # (N, 2) prewarp polygon [mm]
            "pred_mm":       pred_csv,             # (N, 2) predicted sintered [mm]
            "rmse_mm":       summary.get("rmse_final_mm"),
            "verif_rmse_mm": summary.get("rmse_verification_mm"),
            "bbox_x_mm":     summary.get("prewarp_bbox_x_mm"),
            "bbox_y_mm":     summary.get("prewarp_bbox_y_mm"),
            "converged":     summary.get("converged", False),
            "iterations":    summary.get("iterations_run", 0),
            "epe_final":     summary.get("epe_final_per_vertex_mm", []),
        })

    elapsed_total = time.perf_counter() - t_sweep_start
    print(f"\n  {'='*60}")
    print(f"  Sweep complete  ({len(results)} runs in {elapsed_total:.1f} s)")
    print(f"  {'='*60}")

    # ── Generate comparison figure ────────────────────────────────────────────
    tgt_cfg = yaml.safe_load(base_cfg_path.read_text())["target"]
    _plot_sweep(results, output_dir, tgt_cfg, param_path)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    _save_sweep_csv(results, output_dir, param_path)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def _plot_sweep(
    results:    list[dict],
    output_dir: Path,
    tgt_cfg:    dict,
    param_path: str,
) -> None:
    """Generate the 4-panel calibration sweep comparison figure.

    Panel A — 3-body overlay: target + prewarp polygons + predicted sintered
               outputs, colour-coded by parameter value.
    Panel B — Verification RMSE vs. parameter value.
    Panel C — Prewarp aspect ratio (height/width) vs. parameter value.
    Panel D — Polar EPE: signed EPE vs. angle around part (°) for each value.
    """
    from shapes import make_shape, resample_polygon as _resamp

    n_vals = len(results)
    if n_vals == 0:
        return

    # Colour map: blue (small) → red (large)
    colours = [cm.coolwarm(i / max(n_vals - 1, 1)) for i in range(n_vals)]

    # Build target polygon (closed, mm)
    t_w = float(tgt_cfg["width"])
    t_h = float(tgt_cfg.get("height", t_w))
    T_raw    = make_shape(tgt_cfg["shape"], t_w, t_h)
    T_poly_m = _resamp(T_raw, 512)
    T_mm     = T_poly_m * 1e3

    param_label = param_path.split(".")[-1]
    param_units = "mm" if "m" in param_label and not param_label.endswith("m3") else ""
    x_vals_disp = [r["param_val"] * 1e3 if param_units == "mm" else r["param_val"]
                   for r in results]

    fig = plt.figure(figsize=(16, 12), facecolor="white", dpi=180)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1], polar=True)

    # ── Panel A: 3-body comparison ────────────────────────────────────────────
    # Target (single blue reference)
    axA.fill(*_closed(T_mm).T, color="#dce8ff", alpha=0.35, zorder=1)
    axA.plot(*_closed(T_mm).T, color="#2266cc", lw=2.0, zorder=5, label="Target")

    for r, c in zip(results, colours):
        lw = 1.2
        # Prewarp polygon (dashed) — farther from blue = more correction
        axA.plot(*_closed(r["poly_mm"]).T, color=c, lw=lw, ls="--",
                 alpha=0.85, zorder=3)
        # Predicted sintered output (dotted) — should overlap target
        axA.plot(*_closed(r["pred_mm"]).T, color=c, lw=lw, ls=":",
                 alpha=0.85, zorder=4)

    # Legend: one entry for dashed and one for dotted to explain the coding
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#2266cc", lw=2.0, label="Target"),
        Line2D([0], [0], color="gray",    lw=1.2, ls="--", label="Prewarp polygon"),
        Line2D([0], [0], color="gray",    lw=1.2, ls=":",  label="Predicted sintered"),
    ]
    # Add individual L_eff colour chips
    for r, c in zip(results, colours):
        lbl = f"{r['param_val']*1e3:.0f} mm" if param_units == "mm" else str(r["param_val"])
        handles.append(Line2D([0], [0], color=c, lw=2.0, label=lbl))

    axA.legend(handles=handles, fontsize=6.5, loc="upper right",
               ncol=2, framealpha=0.85)
    axA.set_aspect("equal")
    axA.set_xlabel("x [mm]"); axA.set_ylabel("y [mm]")
    axA.set_title(f"A — 3-body: Target / Prewarp (dashed) / Sintered (dotted)\n"
                  f"Colour = {param_label} value  (blue=small → red=large)",
                  fontsize=8.5, fontweight="bold")
    axA.grid(True, lw=0.4, alpha=0.5)

    # ── Panel B: Verification RMSE vs. parameter value ────────────────────────
    rmse_vals = [r["verif_rmse_mm"] for r in results]
    axB.plot(x_vals_disp, rmse_vals, "o-", color="#cc3333", lw=1.8, ms=7,
             label="Verification RMSE")
    # Mark the minimum
    best_idx = int(np.argmin(rmse_vals))
    axB.plot(x_vals_disp[best_idx], rmse_vals[best_idx], "*",
             color="#ff6600", ms=14, zorder=5,
             label=f"Best: {x_vals_disp[best_idx]:.1f} {param_units} "
                   f"(RMSE={rmse_vals[best_idx]:.3f} mm)")
    # Tolerance line (from config of first result)
    tol_val = yaml.safe_load(
        Path(results[0]["label"]).parent.__class__.__name__  # placeholder
        if False else "0"
    ) if False else None
    axB.set_xlabel(f"{param_label} [{param_units}]")
    axB.set_ylabel("Verification RMSE [mm]")
    axB.set_title("B — Verification RMSE vs. parameter value",
                  fontsize=9, fontweight="bold")
    axB.legend(fontsize=8); axB.grid(True, lw=0.4, alpha=0.5)

    # ── Panel C: Prewarp aspect ratio (height / width) ────────────────────────
    ratios = [r["bbox_y_mm"] / max(r["bbox_x_mm"], 1e-6) for r in results]
    axC.plot(x_vals_disp, ratios, "s-", color="#0066cc", lw=1.8, ms=7,
             label="height / width")
    axC.axhline(1.0, color="gray", lw=1.0, ls="--", label="Isotropic (=1)")
    axC.set_xlabel(f"{param_label} [{param_units}]")
    axC.set_ylabel("Prewarp height / width")
    axC.set_title("C — Prewarp aspect ratio  (1.0 = circular prewarp)",
                  fontsize=9, fontweight="bold")
    axC.legend(fontsize=8); axC.grid(True, lw=0.4, alpha=0.5)

    # ── Panel D: Polar EPE — signed EPE vs. angle ────────────────────────────
    # Use centroid-relative angle of each prewarp polygon vertex.
    has_epe = any(len(r["epe_final"]) > 0 for r in results)
    if has_epe:
        for r, c in zip(results, colours):
            epe = np.array(r["epe_final"])   # (N,) signed mm
            if len(epe) == 0:
                continue
            poly = r["poly_mm"]              # (N, 2) mm
            centroid = poly.mean(axis=0)
            angles   = np.arctan2(poly[:, 1] - centroid[1],
                                  poly[:, 0] - centroid[0])   # (N,) radians
            # Sort by angle for a clean polar line
            order = np.argsort(angles)
            ang_s = np.append(angles[order], angles[order[0]] + 2 * np.pi)
            epe_s = np.append(epe[order], epe[order[0]])
            lbl   = (f"{r['param_val']*1e3:.0f} mm"
                     if param_units == "mm" else str(r["param_val"]))
            axD.plot(ang_s, epe_s, lw=1.2, color=c, alpha=0.85, label=lbl)

        axD.axhline(0, color="gray", lw=0.8, ls="--")
        axD.set_title("D — Polar EPE: signed residual [mm] vs. angle\n"
                      "(positive = under-correction; 0 = perfect)",
                      fontsize=8.5, fontweight="bold", pad=15)
        axD.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.0))
        axD.grid(True, lw=0.4, alpha=0.5)
    else:
        axD.set_visible(False)
        fig.text(0.75, 0.25, "No per-vertex EPE data\n(re-run with updated rfam_prewarp.py)",
                 ha="center", va="center", fontsize=9, color="gray")

    # ── Super-title ───────────────────────────────────────────────────────────
    t_w_mm = tgt_cfg["width"] * 1e3
    t_h_mm = tgt_cfg.get("height", tgt_cfg["width"]) * 1e3
    fig.suptitle(
        f"RFAM Prewarp Calibration Sweep — {tgt_cfg['shape']}  "
        f"{t_w_mm:.1f}×{t_h_mm:.1f} mm  |  {len(results)} {param_label} values",
        fontsize=11, fontweight="bold", y=1.01,
    )

    png_path = output_dir / "calibration_sweep.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [sweep] Saved comparison figure: {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def _save_sweep_csv(
    results:    list[dict],
    output_dir: Path,
    param_path: str,
) -> None:
    param_label = param_path.split(".")[-1]
    csv_path = output_dir / "calibration_sweep.csv"
    fields = [
        param_label, "verif_rmse_mm", "rmse_final_mm",
        "bbox_x_mm", "bbox_y_mm", "aspect_ratio",
        "converged", "iterations",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {
                param_label:      r["param_val"],
                "verif_rmse_mm":  r["verif_rmse_mm"],
                "rmse_final_mm":  r["rmse_mm"],
                "bbox_x_mm":      r["bbox_x_mm"],
                "bbox_y_mm":      r["bbox_y_mm"],
                "aspect_ratio":   round(r["bbox_y_mm"] / max(r["bbox_x_mm"], 1e-6), 4),
                "converged":      r["converged"],
                "iterations":     r["iterations"],
            }
            w.writerow(row)
    print(f"  [sweep] Saved summary CSV: {csv_path}")

    # Print a quick table to stdout
    print(f"\n  {'─'*70}")
    print(f"  {'':>20s}  {'verif RMSE':>12s}  {'bbox x':>8s}  {'bbox y':>8s}  "
          f"{'aspect':>7s}  {'conv':>5s}")
    print(f"  {'─'*70}")
    for r in results:
        ar = r["bbox_y_mm"] / max(r["bbox_x_mm"], 1e-6)
        print(f"  {r['label']:>20s}  "
              f"{r['verif_rmse_mm']:>12.4f} mm  "
              f"{r['bbox_x_mm']:>6.3f} mm  "
              f"{r['bbox_y_mm']:>6.3f} mm  "
              f"{ar:>7.4f}  "
              f"{'✓' if r['converged'] else '✗':>5s}")
    print(f"  {'─'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Parameter sweep for RFAM prewarp calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       type=Path, required=True,
                   help="Base rfam_prewarp_*.yaml config file")
    p.add_argument("--output-dir",   type=Path,
                   default=Path("./outputs_eqs/prewarp_calibrate"),
                   help="Root directory; one sub-dir created per sweep value")
    p.add_argument("--sweep-param",  type=str, required=True,
                   help="Dot-separated YAML key to sweep (e.g. prewarp.L_eff_m)")
    p.add_argument("--values",       type=float, nargs="+", required=True,
                   help="List of scalar values to test (in native YAML units)")
    args = p.parse_args()

    run_sweep(
        base_cfg_path = args.config,
        output_dir    = args.output_dir,
        param_path    = args.sweep_param,
        values        = args.values,
    )


if __name__ == "__main__":
    main()
