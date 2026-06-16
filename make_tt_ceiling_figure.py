#!/usr/bin/env python3
"""Figure for the CEILING-RESPECTING turntable composite-mask study (note 31).

Reports sigma_T (deg C) AND mean melt fraction at the ceiling-respecting evaluation
point (max-density snapshot if the 250 C ceiling is crossed, else T_phi90), for the
circle (convex) and L-shape (reentrant).  A second panel shows the melt fraction
reached at the ceiling -- the key limit for the reentrant L-shape.

Run with system matplotlib: python make_tt_ceiling_figure.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SHAPES = ["circle", "L_shape"]
STRATS = ["baseline", "turntable_noFGM", "single_static",
          "avgfield_composite", "composite_of_masks"]
LABELS = ["baseline\n(no tt)", "turntable\n(no FGM)", "single\nstatic",
          "avgfield\ncomposite", "composite\nof masks"]
COLORS = ["#888888", "#2c7fb8", "#d95f0e", "#31a354", "#756bb1"]

data = {s: json.load(open(HERE / f"turntable_composite_ceiling_{s}.json"))["strategies"]
        for s in SHAPES}

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
nS = len(STRATS)
width = 0.38
xc = np.arange(nS)

# ── Row 0: sigma_T, one panel per shape ──────────────────────────────────────
for col, shape in enumerate(SHAPES):
    ax = axes[0, col]
    sig = [data[shape][k]["sigma_T_c"] for k in STRATS]
    mx = [data[shape][k]["max_T_c"] for k in STRATS]
    bars = ax.bar(xc, sig, 0.62, color=COLORS, edgecolor="black", linewidth=0.6)
    for b, v, m in zip(bars, sig, mx):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, 1.0, f"maxT\n{m:.0f}",
                ha="center", va="bottom", fontsize=6.5, color="darkred")
    tt = data[shape]["turntable_noFGM"]["sigma_T_c"]
    ax.axhline(tt, color=COLORS[1], ls=":", lw=1.0, alpha=0.6,
               label="turntable-alone")
    ax.set_xticks(xc); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel(r"$\sigma_T$ at ceiling ($^\circ$C)", fontsize=10)
    ax.set_title(f"{shape}: $\\sigma_T$ at the 250 $^\\circ$C ceiling", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

# ── Row 1: mean melt fraction reached at the ceiling ─────────────────────────
for col, shape in enumerate(SHAPES):
    ax = axes[1, col]
    phi = [data[shape][k]["mean_phi"] for k in STRATS]
    bars = ax.bar(xc, phi, 0.62, color=COLORS, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, phi):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0.90, color="green", ls="--", lw=1.0, alpha=0.6, label="full-melt (0.90)")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(xc); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel(r"mean melt fraction $\bar\varphi$ at ceiling", fontsize=10)
    ax.set_title(f"{shape}: melt reached under the ceiling", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Ceiling-respecting (max part T < 250 $^\\circ$C) turntable composite-mask study "
             "— FULL coupled transient (note 31, corrected operating point)",
             fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = HERE / "fig_tt_composite_ceiling.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
