#!/usr/bin/env python3
"""
Extract 2D (XY plane) Q_rf map from COMSOL vol_fields export.

The COMSOL vol_fields export contains nodes from the part domain
(x=[0,10mm], y=[0,10mm], z=[10,20mm] in the quadrant).  The electrode
axis is Z (22mm spacing from z=-2mm to z=+20mm), so the relevant 2D
cross-section for a Python XY thermal simulation is the XY plane
perpendicular to the electrode axis.

This script:
  1. Reads the export (streaming, handles large files), extracting
     (x, y, z, Qrf, sigma) at t=0 (first time point, Q_rf is time-invariant).
  2. Depth-averages Qrf over z for each (x,y) location, producing Q_rf_2D(x,y).
  3. Mirrors from the quadrant (x>=0, y>=0) to the full domain (x and y symmetric).
  4. Interpolates onto a regular 2D grid covering the full chamber XY extent.
  5. Saves as .npy + .json metadata for use by rfam_eqs_coupled.py
     with the qrf_file_npy option.

Usage:
    python extract_comsol_qrf_2d.py \\
        --input "/path/to/vol_fields_single_time-export.txt" \\
        --out-dir outputs_eqs/comsol_exports \\
        --chamber-x-m 0.042 --chamber-y-m 0.032 \\
        --grid-nx 169 --grid-ny 129
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _parse_header(path: Path) -> tuple[list[str], dict[str, np.ndarray], list[float]]:
    """Return (unique_fields, field_col_index, unique_times)."""
    header_line: str | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("%"):
                break
            if line.startswith("% x"):
                header_line = line.strip()

    if header_line is None:
        raise RuntimeError("Could not find '% x ...' header line in file")

    pairs = re.findall(r"([^@]+?)\s*@\s*t=([0-9.Ee+\-]+)", header_line)
    if not pairs:
        raise RuntimeError("Could not parse field@time pairs from header")

    fields_in_order: list[str] = []
    time_vals: list[float] = []
    for i, (expr, t) in enumerate(pairs):
        expr_clean = expr.strip()
        if i == 0:
            expr_clean = re.sub(r"^%?\s*x\s+y\s+z\s+", "", expr_clean)
        name = re.sub(r"\s*\([^)]*\)\s*$", "", expr_clean).strip()
        fields_in_order.append(name)
        time_vals.append(float(t))

    unique_times: list[float] = []
    for t in time_vals:
        if t not in unique_times:
            unique_times.append(t)

    unique_fields: list[str] = []
    for f in fields_in_order:
        if f not in unique_fields:
            unique_fields.append(f)

    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    n_times = len(unique_times)

    field_col_index: dict[str, np.ndarray] = {
        f: np.full(n_times, -1, dtype=int) for f in unique_fields
    }
    for k, (f, t) in enumerate(zip(fields_in_order, time_vals)):
        field_col_index[f][time_to_idx[t]] = 3 + k

    return unique_fields, field_col_index, unique_times


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract 2D Q_rf map from COMSOL vol_fields export.")
    ap.add_argument("--input", type=Path, required=True, help="Path to COMSOL text export.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_eqs/comsol_exports"),
        help="Output directory for the .npy and .json files.",
    )
    # Full chamber XY extents after mirroring the quadrant.
    ap.add_argument(
        "--chamber-x-m",
        type=float,
        default=0.042,
        help="Full chamber x-width after mirroring [m]. Default: 0.042 (2×21mm).",
    )
    ap.add_argument(
        "--chamber-y-m",
        type=float,
        default=0.032,
        help="Full chamber y-width after mirroring [m]. Default: 0.032 (2×16mm).",
    )
    ap.add_argument(
        "--grid-nx",
        type=int,
        default=169,
        help="Number of grid points in x. Default: 169 (~0.25mm for 42mm).",
    )
    ap.add_argument(
        "--grid-ny",
        type=int,
        default=129,
        help="Number of grid points in y. Default: 129 (~0.25mm for 32mm).",
    )
    ap.add_argument(
        "--out-name",
        type=str,
        default="qrf_2d_map",
        help="Base output filename (without extension). Default: qrf_2d_map.",
    )
    ap.add_argument(
        "--z-target-m",
        type=float,
        default=None,
        help=(
            "If given, filter nodes to those within --z-tol-m of this z coordinate [m] "
            "before projecting to 2D. Use the part midplane, e.g. 0.015 for z=15mm. "
            "Default: None (use all z levels)."
        ),
    )
    ap.add_argument(
        "--z-tol-m",
        type=float,
        default=0.001,
        help="Half-width tolerance in z for the z-filter [m]. Default: 0.001 (±1mm).",
    )
    ap.add_argument(
        "--part-x-half-m",
        type=float,
        default=None,
        help=(
            "Override the part x half-width [m] for zeroing outside the part. "
            "If not set, detected from data. Set to 0.010 for 20mm COMSOL part."
        ),
    )
    ap.add_argument(
        "--part-y-half-m",
        type=float,
        default=None,
        help=(
            "Override the part y half-width [m]. Set to 0.010 for 20mm COMSOL part."
        ),
    )
    args = ap.parse_args()

    print(f"Parsing header of {args.input} ...")
    fields, field_col_index, times = _parse_header(args.input)
    n_times = len(times)
    n_fields = len(fields)
    min_cols = 3 + n_fields * n_times

    print(f"  Fields: {fields}")
    print(f"  Times (min): {times}")
    print(f"  Columns per row: {min_cols}")

    if "exp_Qrf" not in field_col_index:
        raise RuntimeError(
            "'exp_Qrf' not found among exported fields. Available: "
            + ", ".join(fields)
        )
    qrf_col_t0 = int(field_col_index["exp_Qrf"][0])

    sigma_col_t0 = -1
    if "exp_sigma" in field_col_index:
        sigma_col_t0 = int(field_col_index["exp_sigma"][0])

    print(f"  Q_rf column at t=0: {qrf_col_t0}")
    if sigma_col_t0 >= 0:
        print(f"  sigma column at t=0: {sigma_col_t0}")

    # --- Stream the data file ---
    print("Reading data (streaming) ...")
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    qrfs: list[float] = []
    sigmas: list[float] = []

    n_rows = 0
    with args.input.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] == "%":
                continue
            vals = np.fromstring(line, sep=" ")
            if vals.size < min_cols:
                continue
            xs.append(float(vals[0]))
            ys.append(float(vals[1]))
            zs.append(float(vals[2]))
            qrfs.append(float(vals[qrf_col_t0]))
            if sigma_col_t0 >= 0:
                sigmas.append(float(vals[sigma_col_t0]))
            n_rows += 1
            if n_rows % 50_000 == 0:
                print(f"  ... {n_rows} rows read")

    print(f"Total rows read: {n_rows}")
    if n_rows == 0:
        raise RuntimeError("No data rows found in file.")

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    zs_arr = np.asarray(zs, dtype=float)
    qrfs_arr = np.maximum(np.asarray(qrfs, dtype=float), 0.0)

    print(f"  x range: [{xs_arr.min():.5f}, {xs_arr.max():.5f}] m")
    print(f"  y range: [{ys_arr.min():.5f}, {ys_arr.max():.5f}] m")
    print(f"  z range: [{zs_arr.min():.5f}, {zs_arr.max():.5f}] m")
    print(f"  Q_rf range: [{qrfs_arr.min():.3e}, {qrfs_arr.max():.3e}] W/m³")
    nz_mask = qrfs_arr > 0
    if nz_mask.any():
        print(f"  Q_rf mean (non-zero nodes): {qrfs_arr[nz_mask].mean():.3e} W/m³")

    # Optional z-level filter: select only nodes near a target z (e.g. part midplane).
    # This avoids 3-to-2D projection artifacts from mixing z-face and interior nodes.
    if args.z_target_m is not None:
        z_mask = np.abs(zs_arr - args.z_target_m) <= args.z_tol_m
        n_before = len(xs_arr)
        xs_arr   = xs_arr[z_mask]
        ys_arr   = ys_arr[z_mask]
        zs_arr   = zs_arr[z_mask]
        qrfs_arr = qrfs_arr[z_mask]
        print(f"  z-filter: z={args.z_target_m*1e3:.1f}mm ± {args.z_tol_m*1e3:.1f}mm  "
              f"({z_mask.sum()} / {n_before} nodes kept)")
        if xs_arr.size == 0:
            raise RuntimeError(
                f"No nodes within z={args.z_target_m} ± {args.z_tol_m} m. "
                "Try increasing --z-tol-m."
            )

    # Determine part half-extents: use override if provided, else detect from
    # Q_rf-positive nodes (which are inside the part).
    if args.part_x_half_m is not None:
        part_x_half = float(args.part_x_half_m)
        part_y_half = float(args.part_y_half_m) if args.part_y_half_m is not None else part_x_half
        print(f"  Part half-extents (override): x=±{part_x_half*1e3:.2f} mm, y=±{part_y_half*1e3:.2f} mm")
    else:
        # Use nodes with non-zero Q_rf to find part extent.
        part_nodes = qrfs_arr > 0
        if part_nodes.any():
            part_x_half = float(xs_arr[part_nodes].max())
            part_y_half = float(ys_arr[part_nodes].max())
        else:
            part_x_half = float(xs_arr.max())
            part_y_half = float(ys_arr.max())
        print(f"  Detected part half-extents: x=±{part_x_half*1e3:.2f} mm, y=±{part_y_half*1e3:.2f} mm")

    # --- Mirror from quadrant to full domain ---
    # Quadrant covers x>=0, y>=0. Mirror to obtain all four quadrants.
    xs_full = np.concatenate([xs_arr, -xs_arr,  xs_arr, -xs_arr])
    ys_full = np.concatenate([ys_arr,  ys_arr, -ys_arr, -ys_arr])
    qrfs_full = np.concatenate([qrfs_arr, qrfs_arr, qrfs_arr, qrfs_arr])

    # --- Build output grid ---
    x_half = 0.5 * args.chamber_x_m
    y_half = 0.5 * args.chamber_y_m
    x_grid = np.linspace(-x_half, x_half, args.grid_nx)
    y_grid = np.linspace(-y_half, y_half, args.grid_ny)
    XX, YY = np.meshgrid(x_grid, y_grid)

    # --- Interpolate scattered points onto regular grid ---
    print(f"Interpolating {len(xs_full)} scattered points onto "
          f"{args.grid_ny}×{args.grid_nx} grid ...")
    pts_src = np.column_stack([xs_full, ys_full])
    pts_dst = np.column_stack([XX.ravel(), YY.ravel()])

    qrf_2d = griddata(pts_src, qrfs_full, pts_dst, method="linear", fill_value=0.0)
    qrf_2d = qrf_2d.reshape(args.grid_ny, args.grid_nx)
    qrf_2d = np.maximum(qrf_2d, 0.0)

    # Zero out any points outside the part boundary (with a small tolerance).
    # Powder always has Q_rf = 0 since sigma_powder = 0.
    outside = (np.abs(XX) > part_x_half * 1.02) | (np.abs(YY) > part_y_half * 1.02)
    qrf_2d[outside] = 0.0

    part_region = ~outside
    if part_region.any():
        print(f"  Q_rf in part region — mean: {qrf_2d[part_region].mean():.3e}  "
              f"max: {qrf_2d.max():.3e}  W/m³")

    # --- Save outputs ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / f"{args.out_name}.npy"
    meta_path = args.out_dir / f"{args.out_name}_meta.json"
    png_path = args.out_dir / f"{args.out_name}.png"

    np.save(str(npy_path), qrf_2d)
    print(f"Saved Q_rf array: {npy_path}  shape={qrf_2d.shape}")

    meta: dict = {
        "source_file": str(args.input),
        "n_source_rows": int(n_rows),
        "x_min": float(-x_half),
        "x_max": float(x_half),
        "y_min": float(-y_half),
        "y_max": float(y_half),
        "grid_nx": int(args.grid_nx),
        "grid_ny": int(args.grid_ny),
        "part_x_half_m": float(part_x_half),
        "part_y_half_m": float(part_y_half),
        "qrf_mean_in_part_w_per_m3": float(qrf_2d[part_region].mean()) if part_region.any() else 0.0,
        "qrf_max_w_per_m3": float(qrf_2d.max()),
        "description": (
            "2D Q_rf map (W/m³) depth-projected from COMSOL 3D part export. "
            "Grid shape: (grid_ny, grid_nx). X is axis 1 (columns), Y is axis 0 (rows). "
            "Use with rfam_eqs_coupled.py electric.qrf_file_npy option."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved metadata: {meta_path}")

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vmax = float(np.percentile(qrf_2d[qrf_2d > 0], 99)) if (qrf_2d > 0).any() else 1.0
    im0 = axes[0].contourf(
        x_grid * 1e3, y_grid * 1e3, qrf_2d,
        levels=30, cmap="hot", vmin=0, vmax=vmax,
    )
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("Q_rf 2D map from COMSOL (W/m³)")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], label="W/m³")
    rect0 = mpatches.Rectangle(
        (-part_x_half * 1e3, -part_y_half * 1e3),
        2 * part_x_half * 1e3, 2 * part_y_half * 1e3,
        fill=False, edgecolor="white", linewidth=2, linestyle="--",
    )
    axes[0].add_patch(rect0)

    qrf_safe = np.where(qrf_2d > 1.0, qrf_2d, np.nan)
    im1 = axes[1].contourf(
        x_grid * 1e3, y_grid * 1e3, np.log10(qrf_safe),
        levels=30, cmap="hot",
    )
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    axes[1].set_title("log₁₀(Q_rf) 2D map")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], label="log₁₀(W/m³)")
    rect1 = mpatches.Rectangle(
        (-part_x_half * 1e3, -part_y_half * 1e3),
        2 * part_x_half * 1e3, 2 * part_y_half * 1e3,
        fill=False, edgecolor="white", linewidth=2, linestyle="--",
    )
    axes[1].add_patch(rect1)

    fig.suptitle(
        f"COMSOL Q_rf 2D projection — part ±{part_x_half*1e3:.1f}×{part_y_half*1e3:.1f} mm "
        f"in {args.chamber_x_m*1e3:.0f}×{args.chamber_y_m*1e3:.0f} mm chamber",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)
    print(f"Saved visualisation: {png_path}")

    print("\nDone.")
    print(f"  To use in simulation: set electric.qrf_file_npy: {npy_path}")


if __name__ == "__main__":
    main()
