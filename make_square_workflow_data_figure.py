#!/usr/bin/env python3
"""Build an information-dense square workflow + real metrics figure.

No external APIs required. Uses metrics extracted from local outputs_eqs NPZ files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).parent
METRICS_JSON = ROOT / "paperbanana_figures" / "data" / "figure1_square_metrics_real.json"
OUT_MAIN = ROOT / "rfam_paper_overleaf" / "figures" / "fig_square_workflow_data_dense.png"
OUT_ALT = ROOT / "paperbanana_figures" / "generated" / "fig_square_workflow_data_dense.png"


def _load_metrics() -> dict:
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _box(ax, x, y, w, h, title, body, fc="#f4f6f8", ec="#444"):
    r = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(r)
    ax.text(x + 0.01, y + h - 0.035, title, fontsize=9.5, fontweight="bold", va="top", ha="left")
    ax.text(x + 0.01, y + h - 0.065, body, fontsize=8.1, va="top", ha="left", linespacing=1.25)
    return r


def _arrow(ax, x1, y1, x2, y2, color="#333"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.4, color=color, shrinkA=2, shrinkB=2),
    )


def main() -> None:
    data = _load_metrics()
    cases = data["cases"]
    id_to_case = {c["id"]: c for c in cases}

    order = [
        "baseline_no_rotation",
        "single_90deg_at_3min",
        "four_90deg_rotations",
        "prewarp_only",
        "prewarp_plus_four_90deg",
    ]
    label = {
        "baseline_no_rotation": "Baseline",
        "single_90deg_at_3min": "1x90deg @3min",
        "four_90deg_rotations": "4x90deg",
        "prewarp_only": "Prewarp",
        "prewarp_plus_four_90deg": "Prewarp+4x90deg",
    }

    bnd = [id_to_case[k]["bnd_std"] for k in order]
    ratio = [id_to_case[k]["r_bnd_min_max"] for k in order]
    baseline = bnd[0]
    reduction = [(baseline - v) / baseline * 100.0 for v in bnd]

    fig = plt.figure(figsize=(16.2, 9.6), dpi=220, facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.15, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.04, right=0.985, top=0.93, bottom=0.08,
        wspace=0.14, hspace=0.20,
    )

    # Left: workflow diagram
    ax_wf = fig.add_subplot(gs[:, 0])
    ax_wf.set_xlim(0, 1)
    ax_wf.set_ylim(0, 1)
    ax_wf.axis("off")

    _box(
        ax_wf, 0.03, 0.80, 0.42, 0.15,
        "1) Input Target Geometry",
        "Square target T (20 mm x 20 mm)\nDiscretize boundary with N=512 vertices.",
        fc="#eef6ff", ec="#2f5d99",
    )
    _box(
        ax_wf, 0.03, 0.59, 0.42, 0.16,
        "2) Coupled Forward Physics",
        "EQS solve -> |E| field -> Q_rf\nTransient thermal + densification\nOutputs rho_rel(x,z,t), T(x,z,t).",
        fc="#f3f8ec", ec="#4e7f2b",
    )
    _box(
        ax_wf, 0.03, 0.37, 0.42, 0.17,
        "3) Boundary Metrics",
        "Boundary mask B = part - erode(part)\n"
        "bnd_std = std(rho_rel on B)\n"
        "r_bnd = min(rho_rel_B)/max(rho_rel_B)",
        fc="#fff8eb", ec="#9b6b16",
    )
    _box(
        ax_wf, 0.03, 0.14, 0.42, 0.18,
        "4) Inverse Geometry Update (Prewarp)",
        "EPE[i]=(T[i]-B_hat[i])·n_hat[i]\n"
        "P_{k+1}[i]=P_k[i]+alpha*EPE[i]*n_hat[i]\n"
        "Repeat until RMSE/EPE tolerance.",
        fc="#f9efff", ec="#6d3f8f",
    )
    _arrow(ax_wf, 0.24, 0.80, 0.24, 0.75)
    _arrow(ax_wf, 0.24, 0.59, 0.24, 0.54)
    _arrow(ax_wf, 0.24, 0.37, 0.24, 0.32)

    # Strategy branch box
    _box(
        ax_wf, 0.50, 0.37, 0.46, 0.58,
        "Strategy Evaluation Branches (Real Runs)",
        "A) Baseline: no rotation, no prewarp\n"
        "B) 1x90deg at 180 s\n"
        "C) 4x90deg at 72,144,216,288 s\n"
        "D) Prewarp only\n"
        "E) Prewarp + 4x90deg\n\n"
        "Turntable schedule:\n"
        "t_k = k/(N_r+1) * 360 s, N_r=4, DeltaTheta=90deg",
        fc="#f5f5f5", ec="#555555",
    )
    _arrow(ax_wf, 0.45, 0.455, 0.50, 0.66)
    _arrow(ax_wf, 0.24, 0.225, 0.50, 0.48)
    ax_wf.text(
        0.50, 0.30,
        "Key interpretation:\n"
        "Spatial correction (prewarp) + temporal averaging (turntable)\n"
        "gives largest measured uniformity improvement.",
        fontsize=8.5, ha="left", va="top", color="#222",
    )

    # Right top: bar chart for bnd_std
    ax_b = fig.add_subplot(gs[0, 1])
    colors = ["#8e8e8e", "#5f83c4", "#2f6bb5", "#f39c46", "#2ca58d"]
    x = range(len(order))
    bars = ax_b.bar(x, bnd, color=colors, edgecolor="white", linewidth=1.0, width=0.62)
    ax_b.set_xticks(list(x), [label[k] for k in order], rotation=0, fontsize=8.5)
    ax_b.set_ylabel("Boundary density std (bnd_std)", fontsize=9)
    ax_b.set_title("Measured Uniformity Metric 1 (Lower is Better)", fontsize=10, fontweight="bold")
    ax_b.grid(axis="y", lw=0.4, alpha=0.45)
    ax_b.set_ylim(0.0, max(bnd) * 1.24)
    for i, bar in enumerate(bars):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0012,
            f"{bnd[i]:.4f}\n({reduction[i]:.1f}%)",
            ha="center", va="bottom", fontsize=7.8, fontweight="bold",
        )
    ax_b.text(0.01, 0.98, "Percent = reduction vs baseline", transform=ax_b.transAxes, fontsize=7.8, va="top")

    # Right bottom: min/max ratio + compact table text
    ax_r = fig.add_subplot(gs[1, 1])
    bars2 = ax_r.bar(x, ratio, color=colors, edgecolor="white", linewidth=1.0, width=0.62)
    ax_r.set_xticks(list(x), [label[k] for k in order], rotation=0, fontsize=8.5)
    ax_r.set_ylabel("Boundary min/max ratio (r_bnd)", fontsize=9)
    ax_r.set_title("Measured Uniformity Metric 2 (Higher is Better)", fontsize=10, fontweight="bold")
    ax_r.grid(axis="y", lw=0.4, alpha=0.45)
    ax_r.set_ylim(0.64, 0.90)
    for i, bar in enumerate(bars2):
        ax_r.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{ratio[i]:.3f}",
            ha="center", va="bottom", fontsize=8.2, fontweight="bold",
        )

    table_lines = [
        "Data source: boundary stats computed from fields.npz",
        "Runs:",
        "baseline  -> outputs_eqs/turntable_test/no_rotation",
        "1x90deg   -> outputs_eqs/turntable_test/rot90_fixed",
        "4x90deg   -> outputs_eqs/turntable_test/rot90x4_fixed",
        "prewarp   -> outputs_eqs/spike_coopt_v2_square/h0p00mm",
        "combined  -> outputs_eqs/prewarp_turntable_4x_eval",
    ]
    ax_r.text(0.01, 0.02, "\n".join(table_lines), transform=ax_r.transAxes, fontsize=7.3, family="monospace", va="bottom")

    fig.suptitle(
        "RFAM Square Workflow With Real Measured Performance: "
        "Process + Equations + Strategy Metrics in One Figure",
        fontsize=12.5,
        fontweight="bold",
    )

    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_ALT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MAIN, bbox_inches="tight")
    fig.savefig(OUT_ALT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_MAIN}")
    print(f"Saved: {OUT_ALT}")


if __name__ == "__main__":
    main()
