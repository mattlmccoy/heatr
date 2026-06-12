#!/usr/bin/env python3
"""make_H_comparison.py — H shape density uniformity comparison figure.

Compares all strategies on the letter H target:
  • Baseline H (no prewarp, no turntable)
  • H + 4× 90° turntable only (no prewarp)
  • H prewarp only (no turntable)
  • H prewarp + 4× 90° turntable (best)

Outputs: outputs_eqs/H_uniformity_comparison.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_erosion, gaussian_filter

BASE = Path(__file__).parent
od   = BASE / "outputs_eqs"

# ── data paths ────────────────────────────────────────────────────────────────
cases = [
    ("H baseline\n(no prewarp, no rotation)",
     od / "shape_H/fields.npz"),
    ("H + 4× 90° turntable\n(no prewarp)",
     od / "shape_H_turntable_4x/fields.npz"),
    ("H prewarp only\n(no turntable)",
     od / "prewarp_H/verification/fields.npz"),
    ("H prewarp +\n4× 90° turntable",
     od / "prewarp_H_turntable_4x_eval/fields.npz"),
]

def bnd_stats(npz_path):
    f = np.load(str(npz_path))
    rho = f["rho_rel"]
    pm  = f["part_mask"].astype(bool)
    T   = f["T"]
    eroded = binary_erosion(pm)
    bnd    = pm & ~eroded
    rho_b  = rho[bnd]
    return dict(
        rho_mean   = float(rho[pm].mean()),
        T_mean     = float(T[pm].mean()),
        bnd_std    = float(rho_b.std()),
        bnd_min    = float(rho_b.min()),
        bnd_max    = float(rho_b.max()),
        min_max    = float(rho_b.min() / max(rho_b.max(), 1e-9)),
    )

stats = [(lb, bnd_stats(p)) for lb, p in cases if p.exists()]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor="white", dpi=180)
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.46, wspace=0.35,
                         left=0.06, right=0.97, top=0.91, bottom=0.08)

axA  = fig.add_subplot(gs[0, 0:2])   # bar chart: bnd_std all 4
axB  = fig.add_subplot(gs[0, 2:4])   # min/max ratio all 4
axD  = fig.add_subplot(gs[1, 0])     # rho_rel heatmap: baseline
axE  = fig.add_subplot(gs[1, 1])     # rho_rel heatmap: turntable only
axF  = fig.add_subplot(gs[1, 2])     # rho_rel heatmap: prewarp only
axG  = fig.add_subplot(gs[1, 3])     # rho_rel heatmap: prewarp + turntable

# ── colours ───────────────────────────────────────────────────────────────────
bar_colors = ["#888888", "#3a86ff", "#ff6b35", "#2ec4b6"]

# ── Panel A: bnd_std bar chart ────────────────────────────────────────────────
labels   = [lb for lb, _ in stats]
stds     = [s["bnd_std"] for _, s in stats]
xs       = range(len(labels))
bars = axA.bar(xs, stds, color=bar_colors, width=0.55, edgecolor="white", linewidth=1.2)
axA.set_xticks(xs)
axA.set_xticklabels(labels, fontsize=8, ha="center")
axA.set_ylabel("ρ_boundary std  (lower = uniform)", fontsize=9)
axA.set_title("A — Boundary Density Std\n(H shape, all strategies)", fontsize=9, fontweight="bold")
axA.set_ylim(0, max(stds) * 1.22)
for bar, std in zip(bars, stds):
    axA.text(bar.get_x() + bar.get_width() / 2, std + 0.0015,
             f"{std:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
axA.axhline(stds[0], color="#888888", lw=0.8, ls="--", alpha=0.45)
axA.grid(axis="y", lw=0.4, alpha=0.5)

# ── Panel B: min/max ratio bar chart ──────────────────────────────────────────
ratios = [s["min_max"] for _, s in stats]
bars2 = axB.bar(xs, ratios, color=bar_colors, width=0.55, edgecolor="white", linewidth=1.2)
axB.set_xticks(xs)
axB.set_xticklabels(labels, fontsize=8, ha="center")
axB.set_ylabel("ρ_bnd min/max ratio  (higher = uniform)", fontsize=9)
axB.set_title("B — Min/Max Boundary Density Ratio\n(H shape, all strategies)", fontsize=9, fontweight="bold")
axB.set_ylim(0.60, 1.01)
for bar, ratio in zip(bars2, ratios):
    axB.text(bar.get_x() + bar.get_width() / 2, ratio + 0.005,
             f"{ratio:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
axB.axhline(ratios[0], color="#888888", lw=0.8, ls="--", alpha=0.45)
axB.grid(axis="y", lw=0.4, alpha=0.5)

# ── Panels D–G: rho_rel heatmaps ─────────────────────────────────────────────
def _plot_rho(ax, npz_path, title, stat):
    if not Path(str(npz_path)).exists():
        ax.text(0.5, 0.5, "not found", transform=ax.transAxes, ha="center")
        ax.set_title(title, fontsize=8, fontweight="bold")
        return
    f     = np.load(str(npz_path))
    rho   = f["rho_rel"]
    pm    = f["part_mask"].astype(bool)
    x_g   = f["x"] * 1e3
    y_g   = f["y"] * 1e3
    rho_d = np.clip(gaussian_filter(rho.astype(float), sigma=1.2), 0.0, 1.0)
    pm_d  = gaussian_filter(pm.astype(float), sigma=1.0)
    part_cols = np.where(pm.any(axis=0))[0]
    part_rows = np.where(pm.any(axis=1))[0]
    pad = 8
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
    subtitle = (f"std={stat['bnd_std']:.4f}  "
                f"min/max={stat['min_max']:.3f}")
    ax.set_title(f"{title}\n{subtitle}", fontsize=8, fontweight="bold")

panel_labels = ["C", "D", "E", "F"]
axes_hm = [axD, axE, axF, axG]

for ax, (lb, stat), (label, path), pl in zip(axes_hm, stats, cases, panel_labels):
    short_title = lb.replace("\n", " ")
    _plot_rho(ax, path, f"{pl} — {short_title}", stat)

# ── overall title ─────────────────────────────────────────────────────────────
# Build annotation with improvement %
baseline_std = stats[0][1]["bnd_std"]
best_std     = stats[-1][1]["bnd_std"]
improv_pct   = 100 * (baseline_std - best_std) / baseline_std

fig.suptitle(
    f"RFAM Density Uniformity: Letter H  |  "
    f"Prewarp + Turntable achieves {improv_pct:.0f}% reduction in boundary std",
    fontsize=11, fontweight="bold", y=0.97)

# ── add summary table as text ─────────────────────────────────────────────────
table_lines = ["Method                              bnd_std  min/max  improvement"]
for lb, s in stats:
    lb_short = lb.replace("\n", " ").replace("  ", " ")
    improv   = 100 * (baseline_std - s["bnd_std"]) / baseline_std
    table_lines.append(
        f"  {lb_short[:38]:38s}  {s['bnd_std']:.4f}   {s['min_max']:.3f}   {improv:+.1f}%")
fig.text(0.06, 0.005, "\n".join(table_lines),
         fontsize=6.5, family="monospace", va="bottom",
         color="#444444")

out = od / "H_uniformity_comparison.png"
fig.savefig(str(out), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# Print results to console
print()
print("=== H shape uniformity results ===")
for lb, s in stats:
    improv = 100 * (baseline_std - s["bnd_std"]) / baseline_std
    print(f"  {lb.replace(chr(10),' '):45s}  bnd_std={s['bnd_std']:.4f}  min/max={s['min_max']:.3f}  {improv:+.1f}%")
