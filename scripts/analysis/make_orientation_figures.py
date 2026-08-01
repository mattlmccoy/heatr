#!/usr/bin/env python3
"""Per-shape orientation-optimization figures.

Layout per shape: 2 rows (uniform, graded) x 2 columns (0 degrees, best angle
by J for that arm). Each panel shows the melt-fraction field at that arm's own
J-stop with the ROTATED nominal part outline in cyan and the melt front
(phi = 0.5) dashed white, the conventions of the shape-library figures
(fgm_solve_campaign/adjoint2d/make_library_figures.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs_eqs/orientation_optimization"
DPI = 170
EXTENT = (-30.0, 30.0, -30.0, 30.0)  # mm


def field_path(shape: str, angle: float, arm: str) -> Path:
    tag = f"ang{angle:07.2f}_{arm}".replace(".", "p")
    return ROOT / shape / "fields" / f"{tag}.npz"


def panel(ax, shape: str, row: dict) -> None:
    d = np.load(field_path(shape, row["angle_deg"], row["arm"]))
    phi = d["phi_stop"]
    pm = d["part_mask"].astype(float)
    ny, nx = phi.shape
    xs = np.linspace(EXTENT[0], EXTENT[1], nx)
    ys = np.linspace(EXTENT[2], EXTENT[3], ny)
    ax.imshow(phi, origin="lower", extent=EXTENT, vmin=0.0, vmax=1.0,
              cmap="inferno", interpolation="bilinear")
    ax.contour(xs, ys, pm, levels=[0.5], colors="#00e5ff", linewidths=1.4)
    ax.contour(xs, ys, phi, levels=[0.5], colors="#ffffff",
               linewidths=1.0, linestyles="--")
    hz = " (H)" if row["t_stop_at_horizon"] else ""
    ax.set_title(
        f"{row['arm']}  {row['angle_deg']:g} deg\n"
        f"J {row['J']:.1f}  IoU {row['IoU']:.3f}  stop {row['t_stop_s']:.0f} s{hz}\n"
        f"under {row['part_under_melt_pct']:.1f}%  bed {row['bed_melt_pct_of_part']:.1f}%",
        fontsize=9)
    ax.set_xlim(-18.0, 18.0)
    ax.set_ylim(-18.0, 18.0)
    ax.set_xticks([])
    ax.set_yticks([])


def make_figure(shape: str) -> Path:
    res = json.load(open(ROOT / shape / "results.json"))
    rows = res["rows"]

    def get(angle, arm):
        return next(r for r in rows if r["angle_deg"] == angle and r["arm"] == arm)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 9.4), layout="constrained")
    for i, arm in enumerate(("uniform", "graded")):
        arm_rows = [r for r in rows if r["arm"] == arm]
        best = min(arm_rows, key=lambda r: r["J"])
        panel(axes[i, 0], shape, get(0.0, arm))
        panel(axes[i, 1], shape, best)
    fig.suptitle(
        f"{shape}: orientation vs J at each arm's own J-stop\n"
        "Cyan is the rotated nominal part, dashed white is the melt front.",
        fontsize=11)
    out = ROOT / shape / f"fig_orient_{shape}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    for shape in (sys.argv[1:] or ["T_shape", "L_shape", "cross", "star"]):
        make_figure(shape)
