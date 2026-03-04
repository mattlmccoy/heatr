#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_eqs"


def _load_json(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    if not isinstance(d, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return d


def _baseline_summary(geometry: str) -> dict[str, Any]:
    if geometry == "H":
        path = OUT / "shape_H" / "summary.json"
    elif geometry == "L":
        path = OUT / "shapes" / "shape_L_shape" / "summary.json"
    elif geometry == "T":
        path = OUT / "shapes" / "shape_T_shape" / "summary.json"
    else:
        path = OUT / "shapes" / f"shape_{geometry}" / "summary.json"
    return _load_json(path)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    p = argparse.ArgumentParser(description="Generate improved visuals for static geometry screen.")
    p.add_argument("--report-dir", required=True)
    p.add_argument("--ignore", default="gt_logo")
    args = p.parse_args()

    report_dir = Path(args.report_dir).resolve()
    rows_csv = report_dir / "static_geometry_screen_rows.csv"
    ignore = {x.strip() for x in args.ignore.split(",") if x.strip()}
    exposures = [360, 420, 480]

    by_geom: dict[str, dict[int, dict[str, Any]]] = {}
    for row in csv.DictReader(rows_csv.open()):
        geom = str(row["geometry"])
        if geom in ignore:
            continue
        t = int(round(_as_float(row["exposure_s"])))
        by_geom.setdefault(geom, {})[t] = {
            "mean_phi": _as_float(row["mean_phi"]),
            "mean_rho": _as_float(row["mean_rho"]),
            "temp_rmse_c": _as_float(row["baseline_temp_rmse_c"], float("nan")),
            "phi_iou_0p5": _as_float(row["baseline_phi_iou_0p5"], float("nan")),
            "rho_rmse": _as_float(row["baseline_rho_rmse"], float("nan")),
            "run_dir": Path(str(row["run_dir"])),
        }

    geoms = sorted(by_geom.keys())
    if not geoms:
        raise ValueError("No geometries found after filtering.")

    # Heatmap deltas (experimental - baseline@360s), shown in percentage points.
    phi_delta = np.zeros((len(geoms), len(exposures)), dtype=float)
    rho_delta = np.zeros((len(geoms), len(exposures)), dtype=float)
    for i, g in enumerate(geoms):
        b = _baseline_summary(g)
        b_phi = _as_float(b.get("mean_phi_part_final"))
        b_rho = _as_float(b.get("mean_rho_rel_part_final"))
        for j, t in enumerate(exposures):
            r = by_geom[g].get(t)
            if r is None:
                phi_delta[i, j] = np.nan
                rho_delta[i, j] = np.nan
            else:
                phi_delta[i, j] = (r["mean_phi"] - b_phi) * 100.0
                rho_delta[i, j] = (r["mean_rho"] - b_rho) * 100.0

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=180)
    vphi = np.nanmax(np.abs(phi_delta)) if np.isfinite(phi_delta).any() else 1.0
    vrho = np.nanmax(np.abs(rho_delta)) if np.isfinite(rho_delta).any() else 1.0
    vphi = max(vphi, 1e-3)
    vrho = max(vrho, 1e-3)
    im0 = ax[0].imshow(phi_delta, cmap="coolwarm", aspect="auto", vmin=-vphi, vmax=vphi)
    im1 = ax[1].imshow(rho_delta, cmap="coolwarm", aspect="auto", vmin=-vrho, vmax=vrho)
    for k, (im, title, arr) in enumerate(
        [
            (im0, "Delta mean melt fraction (percentage points)", phi_delta),
            (im1, "Delta mean relative density (percentage points)", rho_delta),
        ]
    ):
        ax[k].set_title(title)
        ax[k].set_xticks(range(len(exposures)))
        ax[k].set_xticklabels([f"{t}s" for t in exposures])
        ax[k].set_yticks(range(len(geoms)))
        ax[k].set_yticklabels(geoms)
        cbar = plt.colorbar(im, ax=ax[k], fraction=0.046, pad=0.04)
        cbar.set_label("pp")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    ax[k].text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=6, color="black")
    fig.suptitle("HEATR Static Screen: Baseline vs Experimental Delta (GT excluded)")
    fig.tight_layout()
    fig.savefig(report_dir / "figure_geometry_delta_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    # Improved sweep trends with legend and only real simulated exposures.
    fig, ax = plt.subplots(1, 2, figsize=(13.8, 5.0), dpi=180)
    for g in geoms:
        points = by_geom[g]
        x = [360.0, 420.0, 480.0]
        y_phi = [points.get(360, {}).get("mean_phi", np.nan), points.get(420, {}).get("mean_phi", np.nan), points.get(480, {}).get("mean_phi", np.nan)]
        y_rho = [points.get(360, {}).get("mean_rho", np.nan), points.get(420, {}).get("mean_rho", np.nan), points.get(480, {}).get("mean_rho", np.nan)]
        ax[0].plot(x, y_phi, marker="o", lw=1.6, label=g)
        ax[1].plot(x, y_rho, marker="o", lw=1.6, label=g)

    ax[0].set_title("Mean Melt Fraction vs Exposure")
    ax[0].set_xlabel("Exposure (s)")
    ax[0].set_ylabel("mean_phi")
    ax[0].set_xlim(355, 485)
    ax[0].set_ylim(0.0, 1.05)
    ax[0].set_xticks([360, 420, 480])

    ax[1].set_title("Mean Relative Density vs Exposure")
    ax[1].set_xlabel("Exposure (s)")
    ax[1].set_ylabel("mean_rho")
    ax[1].set_xlim(355, 485)
    ax[1].set_xticks([360, 420, 480])

    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=8, title="Geometry")
    fig.suptitle("HEATR Static Screen: Exposure Sweep Trends (GT excluded)")
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    fig.savefig(report_dir / "figure_exposure_sweep_trends.png", bbox_inches="tight")
    plt.close(fig)

    # Nominal bars (360s) with GT excluded.
    temp = [by_geom[g][360]["temp_rmse_c"] for g in geoms]
    phi_iou = [by_geom[g][360]["phi_iou_0p5"] for g in geoms]
    rho_rmse = [by_geom[g][360]["rho_rmse"] for g in geoms]
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), dpi=180, sharex=True)
    ax[0].bar(geoms, temp, color="#457b9d")
    ax[0].axhline(15.0, ls="--", c="k", lw=1)
    ax[0].set_ylabel("temp_rmse_c")
    ax[0].set_ylim(0.0, 16.0)

    ax[1].bar(geoms, phi_iou, color="#1d3557")
    ax[1].axhline(0.70, ls="--", c="k", lw=1)
    ax[1].set_ylabel("phi_iou_0p5")
    ax[1].set_ylim(0.0, 1.05)

    ax[2].bar(geoms, rho_rmse, color="#6d597a")
    ax[2].axhline(0.10, ls="--", c="k", lw=1)
    ax[2].set_ylabel("rho_rmse")
    ax[2].set_ylim(0.0, 0.12)
    ax[2].set_xticks(range(len(geoms)))
    ax[2].set_xticklabels(geoms, rotation=45, ha="right")
    fig.suptitle("HEATR Static Screen: Nominal (360s) Baseline Metrics (GT excluded)")
    fig.tight_layout()
    fig.savefig(report_dir / "figure_nominal_metric_bars.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] wrote {report_dir / 'figure_geometry_delta_heatmaps.png'}")
    print(f"[viz] wrote {report_dir / 'figure_exposure_sweep_trends.png'}")
    print(f"[viz] wrote {report_dir / 'figure_nominal_metric_bars.png'}")


if __name__ == "__main__":
    main()
