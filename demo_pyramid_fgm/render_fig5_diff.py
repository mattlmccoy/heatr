"""Fig 5: what the grading actually changed - difference panels (graded minus
uniform) on the XZ mid-slice, rendered from the existing run artifacts only.

Run with the deck_figures_3d venv python from the repo root:
    deck_figures_3d/.venv/bin/python demo_pyramid_fgm/render_fig5_diff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deck_figures_3d"))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import style3d as S  # noqa: E402
from demo_pyramid_fgm.grading import HONESTY_TEXT  # noqa: E402

HERE = ROOT / "demo_pyramid_fgm"
g = np.load(HERE / "out_graded" / "fields.npz")
u = np.load(HERE / "out_uniform" / "fields.npz")

part = (g["Qrf"] > 0) | (u["Qrf"] > 0)
mid = g["T_phi90"].shape[1] // 2

dT = (g["T_phi90"] - u["T_phi90"])[:, mid, :].T
dQ = ((g["Qrf"] - u["Qrf"]) / max(u["Qrf"].max(), 1.0))[:, mid, :].T
mask = part[:, mid, :].T

dT_m = np.where(mask, dT, np.nan)
dQ_m = np.where(mask, dQ * 100.0, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4), dpi=S.DPI)
fig.patch.set_facecolor(S.BG)

vT = float(np.nanmax(np.abs(dT_m)))
vQ = float(np.nanmax(np.abs(dQ_m)))
specs = [
    (axes[0], dT_m, vT, "delta T at phi=0.90 [C]",
     f"graded minus uniform, max |dT| = {vT:.1f} C"),
    (axes[1], dQ_m, vQ, "delta Q_rf [% of uniform peak]",
     f"heating redistributed, max {vQ:.1f}% of peak"),
]
for ax, fld, v, cbl, sub in specs:
    ax.set_facecolor(S.BG)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#22252c")
    im = ax.imshow(fld, origin="lower", cmap=cmap, vmin=-v, vmax=v,
                   interpolation="bilinear")
    ax.set_title(sub, fontsize=11, color=S.FG, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=8, colors=S.DIM, length=2)
    cb.outline.set_edgecolor(S.DIM)
    cb.outline.set_linewidth(0.4)
    cb.set_label(cbl, fontsize=9, color=S.DIM)

S.title_block(fig, "WHAT THE GRADING CHANGED",
              "XZ mid-slice differences, graded minus uniform, apex up; "
              "same grid, exposure, and total absorbed power in both arms")
fig.text(0.975, 0.02, HONESTY_TEXT, fontsize=8.5, color=S.DIM,
         ha="right")

out = HERE / "fig5_what_changed.png"
fig.savefig(out, facecolor=S.BG, bbox_inches="tight")
print("wrote", out, "| max|dT|", round(vT, 2), "C | max|dQ|", round(vQ, 2), "%")
