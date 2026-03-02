#!/usr/bin/env python3
"""make_uniformity_comparison.py — Density uniformity comparison figure.

Compares all uniformity-improvement strategies tested:
  • No prewarp (perfect square)
  • Prewarp only
  • Prewarp + spike features (varying h_mm, 5° width)
  • Prewarp + turntable 4× 90°
  • Perfect square + turntable 4× 90°

Outputs: outputs_eqs/uniformity_comparison.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_erosion, gaussian_filter

BASE = Path(__file__).parent

def bnd_stats(npz_path):
    f = np.load(str(npz_path))
    rho = f["rho_rel"]
    pm  = f["part_mask"].astype(bool)
    T   = f["T"]
    eroded = binary_erosion(pm)
    bnd    = pm & ~eroded
    rho_b  = rho[bnd]
    return dict(
        rho_mean    = float(rho[pm].mean()),
        T_mean      = float(T[pm].mean()),
        bnd_std     = float(rho_b.std()),
        bnd_min     = float(rho_b.min()),
        bnd_max     = float(rho_b.max()),
        min_max     = float(rho_b.min() / max(rho_b.max(), 1e-9)),
    )

# ── collect data ─────────────────────────────────────────────────────────────
od = BASE / "outputs_eqs"

cases_left = [                         # (label, npz_path)
    ("Square (no prewarp, no rotation)",      od / "turntable_test/no_rotation/fields.npz"),
    ("Square + 1× 90° (t=3min)",              od / "turntable_test/rot90_fixed/fields.npz"),
    ("Square + 4× 90°",                       od / "turntable_test/rot90x4_fixed/fields.npz"),
]
cases_right = [
    ("Prewarp only",                          od / "spike_coopt_v2_square/h0p00mm/fields.npz"),
    ("Prewarp + h=1.6mm spikes",              od / "spike_coopt_v2_square/h1p60mm/fields.npz"),
    ("Prewarp + h=3.2mm spikes",              od / "spike_coopt_v2_square/h3p20mm/fields.npz"),
    ("Prewarp + h=4.8mm spikes",              od / "spike_coopt_v2_square/h4p80mm/fields.npz"),
    ("Prewarp + h=6.4mm spikes",              od / "spike_coopt_v2_square/h6p40mm/fields.npz"),
    ("Prewarp + h=8.0mm spikes",              od / "spike_coopt_v2_square/h8p00mm/fields.npz"),
    ("Prewarp + 4× 90° turntable",            od / "prewarp_turntable_4x_eval/fields.npz"),
]

stats_l = [(lb, bnd_stats(p)) for lb, p in cases_left if p.exists()]
stats_r = [(lb, bnd_stats(p)) for lb, p in cases_right if p.exists()]

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor="white", dpi=180)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                         left=0.07, right=0.97, top=0.92, bottom=0.10)

axA = fig.add_subplot(gs[0, 0])  # bnd_std — turntable only
axB = fig.add_subplot(gs[0, 1])  # bnd_std — spike + turntable combo
axC = fig.add_subplot(gs[0, 2])  # min/max ratio
axD = fig.add_subplot(gs[1, 0])  # rho_rel field: prewarp only
axE = fig.add_subplot(gs[1, 1])  # rho_rel field: prewarp + turntable
axF = fig.add_subplot(gs[1, 2])  # rho_rel field: spike h=8mm

# ── Panel A: turntable sweep ─────────────────────────────────────────────────
colors_l = ["#888888", "#3366cc", "#ee4422"]
labels_l = [lb for lb, _ in stats_l]
stds_l   = [s["bnd_std"] for _, s in stats_l]
xs_l     = range(len(labels_l))
bars = axA.bar(xs_l, stds_l, color=colors_l, width=0.55, edgecolor="white", linewidth=1.2)
axA.set_xticks(xs_l)
axA.set_xticklabels([lb.replace(" + ", "\n+ ").replace("rotation)", "rot)") for lb in labels_l],
                    fontsize=7.5, rotation=0, ha="center")
axA.set_ylabel("ρ_boundary std  (lower = uniform)", fontsize=9)
axA.set_title("A — Turntable Rotation Effect\n(perfect square, no prewarp)", fontsize=9, fontweight="bold")
axA.set_ylim(0, 0.085)
for bar, std in zip(bars, stds_l):
    axA.text(bar.get_x() + bar.get_width()/2, std + 0.001, f"{std:.4f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold")
axA.axhline(stds_l[0], color="#888888", lw=0.8, ls="--", alpha=0.5)
axA.grid(axis="y", lw=0.4, alpha=0.5)

# ── Panel B: spike + turntable comparison (prewarp polygon) ──────────────────
labels_r = [lb for lb, _ in stats_r]
stds_r   = [s["bnd_std"] for _, s in stats_r]
colors_r = ["#555555"] + [plt.cm.YlOrRd(0.3 + 0.12*i) for i in range(5)] + ["#2266cc"]
xs_r     = range(len(labels_r))
bars2 = axB.bar(xs_r, stds_r, color=colors_r, width=0.6, edgecolor="white", linewidth=1.2)
axB.set_xticks(xs_r)
short_labels = ["Prewarp\nonly", "h=1.6mm", "h=3.2mm", "h=4.8mm", "h=6.4mm", "h=8.0mm", "Prewarp\n+turntable\n4×90°"]
axB.set_xticklabels(short_labels, fontsize=7.5, rotation=0, ha="center")
axB.set_ylabel("ρ_boundary std  (lower = uniform)", fontsize=9)
axB.set_title("B — Spike Assist vs. Turntable\n(prewarp polygon baseline)", fontsize=9, fontweight="bold")
axB.set_ylim(0, 0.050)
for bar, std in zip(bars2, stds_r):
    axB.text(bar.get_x() + bar.get_width()/2, std + 0.0006, f"{std:.4f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold")
axB.axhline(stds_r[0], color="#555555", lw=0.8, ls="--", alpha=0.5)
axB.grid(axis="y", lw=0.4, alpha=0.5)

# ── Panel C: min/max ratio comparison (all cases) ────────────────────────────
all_labels = [lb for lb, _ in stats_l] + ["─"] + [lb for lb, _ in stats_r]
all_ratios = [s["min_max"] for _, s in stats_l] + [None] + [s["min_max"] for _, s in stats_r]
all_colors = colors_l + ["white"] + colors_r
ax_xs = range(len(all_labels))
for xi, (label, ratio, color) in enumerate(zip(all_labels, all_ratios, all_colors)):
    if ratio is not None:
        axC.bar(xi, ratio, color=color, width=0.7, edgecolor="white", linewidth=1.0)
        axC.text(xi, ratio + 0.003, f"{ratio:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
axC.set_xticks(range(len(all_labels)))
xlabels_c = [lb.split("(")[0].strip().replace("Prewarp + ", "PW+").replace("Square + ", "Sq+")
             for lb in all_labels]
axC.set_xticklabels(xlabels_c, fontsize=6, rotation=45, ha="right")
axC.set_ylabel("ρ_bnd min/max ratio  (higher = uniform)", fontsize=9)
axC.set_title("C — Min/Max Boundary Density Ratio", fontsize=9, fontweight="bold")
axC.set_ylim(0.60, 0.92)
axC.grid(axis="y", lw=0.4, alpha=0.5)

# ── Panels D, E, F: rho_rel field heatmaps ───────────────────────────────────
def _plot_rho(ax, npz_path, title):
    if not Path(str(npz_path)).exists():
        ax.text(0.5, 0.5, "not found", transform=ax.transAxes, ha="center")
        ax.set_title(title, fontsize=9, fontweight="bold")
        return
    f    = np.load(str(npz_path))
    rho  = f["rho_rel"]
    pm   = f["part_mask"].astype(bool)
    x_g  = f["x"] * 1e3
    y_g  = f["y"] * 1e3
    rho_d = np.clip(gaussian_filter(rho.astype(float), sigma=1.2), 0.0, 1.0)
    pm_d  = gaussian_filter(pm.astype(float), sigma=1.0)
    # Crop to part + small margin
    part_cols = np.where(pm.any(axis=0))[0]
    part_rows = np.where(pm.any(axis=1))[0]
    pad = 6
    r0, r1 = max(0, part_rows[0]-pad), min(rho.shape[0], part_rows[-1]+pad+1)
    c0, c1 = max(0, part_cols[0]-pad), min(rho.shape[1], part_cols[-1]+pad+1)
    im = ax.imshow(rho_d[r0:r1, c0:c1],
                   origin="lower",
                   extent=[x_g[c0], x_g[c1-1], y_g[r0], y_g[r1-1]],
                   cmap="plasma", vmin=0.0, vmax=0.85,
                   interpolation="bilinear", aspect="equal")
    ax.contour(x_g[c0:c1], y_g[r0:r1], pm_d[r0:r1, c0:c1],
               levels=[0.5], colors="white", linewidths=1.2)
    plt.colorbar(im, ax=ax, label="ρ_rel", fraction=0.046, pad=0.04)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")

_plot_rho(axD, od / "spike_coopt_v2_square/h0p00mm/fields.npz",
          "D — Prewarp only\n(bnd_std=0.0403)")
_plot_rho(axE, od / "prewarp_turntable_4x_eval/fields.npz",
          "E — Prewarp + 4× 90° turntable\n(bnd_std=0.0253  ←  best!)")
_plot_rho(axF, od / "spike_coopt_v2_square/h8p00mm/fields.npz",
          "F — Prewarp + spike h=8mm\n(bnd_std=0.0347)")

fig.suptitle("RFAM Density Uniformity: Turntable vs. Spike Assist Features",
             fontsize=12, fontweight="bold", y=0.97)

out = od / "uniformity_comparison.png"
fig.savefig(str(out), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
