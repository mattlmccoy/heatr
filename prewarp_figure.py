#!/usr/bin/env python3
"""Geometry Pre-Warp GUI worker (FIGURE stage).

Reads <out_dir>/prewarp_capture.npz (written by prewarp_worker.py under .venv312)
and renders <out_dir>/prewarp.png — the offline-campaign panel layout:
    nominal CAD | printed melt (naive) | pre-warped input (theta) | corrected melt | IoU bar

Run under a python that HAS matplotlib (e.g. /usr/bin/python3); .venv312 does not.
This stage does only plotting from saved arrays, so the numpy buffer-elision bug is
not a correctness concern here (no physics compute).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _contour_overlay(ax, mask2d, color, lw=1.6, ls="-"):
    """Draw the boundary of a binary mask as a contour line."""
    m = np.asarray(mask2d, float)
    if m.max() <= 0:
        return
    ax.contour(m.T, levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    d = np.load(out_dir / "prewarp_capture.npz", allow_pickle=True)
    geometry = str(d["geometry"])
    nominal = d["nominal"]
    theta_final = d["theta_final"]
    m_naive = d["m_naive"]
    m_final = d["m_final"]
    iou0 = float(d["iou0"])
    iouf = float(d["iou_final"])

    grey = ListedColormap(["#0e1726", "#7f8da3"])
    red_cm = ListedColormap(["#0e1726", "#e0533a"])
    grn_cm = ListedColormap(["#0e1726", "#3ab06a"])

    fig = plt.figure(figsize=(15.5, 3.7), facecolor="#0a0f1a")
    gs = fig.add_gridspec(1, 5, wspace=0.18, left=0.02, right=0.985,
                          top=0.86, bottom=0.06)

    def style(ax, title):
        ax.set_title(title, color="#cdd9ec", fontsize=11, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a3550")

    # 1. nominal CAD (the fixed crisp target)
    ax = fig.add_subplot(gs[0, 0]); style(ax, "Nominal CAD (target)")
    ax.imshow(nominal.T, origin="lower", cmap=grey, vmin=0, vmax=1)

    # 2. printed melt — naive (no pre-warp)
    ax = fig.add_subplot(gs[0, 1]); style(ax, f"Printed melt — naive\nIoU = {iou0:.3f}")
    ax.imshow(m_naive.T, origin="lower", cmap=red_cm, vmin=0, vmax=1)
    _contour_overlay(ax, nominal, "#7f8da3", lw=1.4, ls="--")

    # 3. pre-warped input geometry (theta) — the corrected boundary
    ax = fig.add_subplot(gs[0, 2]); style(ax, "Pre-warped input (geometry)")
    ax.imshow(theta_final.T, origin="lower", cmap=grn_cm, vmin=0, vmax=1)
    _contour_overlay(ax, nominal, "#5fa8ff", lw=1.4, ls="--")

    # 4. corrected melt (forward of the pre-warped geometry)
    ax = fig.add_subplot(gs[0, 3]); style(ax, f"Corrected melt — pre-warp\nIoU = {iouf:.3f}")
    ax.imshow(m_final.T, origin="lower", cmap=red_cm, vmin=0, vmax=1)
    _contour_overlay(ax, nominal, "#7f8da3", lw=1.4, ls="--")

    # 5. IoU before/after bars
    ax = fig.add_subplot(gs[0, 4]); style(ax, "IoU(melt, nominal)")
    ax.set_title("IoU(melt, nominal)", color="#cdd9ec", fontsize=11, pad=6)
    ax.set_facecolor("#0e1726")
    bars = ax.bar(["naive", "pre-warp"], [iou0, iouf],
                  color=["#e0533a", "#3ab06a"], width=0.6)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#9fb0cc", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#2a3550")
    for b, v in zip(bars, [iou0, iouf]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", color="#cdd9ec", fontsize=10)

    fig.suptitle(f"Geometry Pre-Warp — {geometry}   "
                 f"(uniform dopant, boundary moved, center fixed)",
                 color="#e8eef8", fontsize=13, y=0.985)

    png = out_dir / "prewarp.png"
    fig.savefig(png, dpi=130, facecolor="#0a0f1a")
    print(f"[prewarp-figure] wrote {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
