"""Render fig_solve_loop_schematic.png: the filter-only adjoint solve loop.

One STATIC 16:9 conceptual schematic for the deck, built from STORED artifacts
only (no new forward solves):

  * square nominal mask, mid-march temperature, melt-minus-nominal residual,
    filtered solved map and the J-per-evaluation sparkline all come from
    deck_gifs/cache/c2_square.npz (the stored solve-arm capture documented in
    DECK_GIFS_NOTES.md: filtered adjoint solve at the campaign's 1.0 mm
    physical radius, conductivity-only channel, 15 evaluations).
  * the 4 bpp exit thumbnail is the production filter-only deliverable map
    TO_4bpp from fgm_solve_campaign/out_topopt/square_control_filteronly_maps.npz.
  * the adjoint stage is a labeled backward arrow, NOT a field: no stored
    dJ/ds field exists on disk (gate files store scalar probes only).

No Heaviside projection stage anywhere: it is retired from the production
recipe (MMA_RETEST_REPORT.md verdict; CHANGELOG_ENGINE.md v2.0.0).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from style import BG, PANEL, FG, DIM, ACCENT, WARM, GOOD  # noqa: E402

import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyArrowPatch        # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C2 = np.load(REPO / "deck_gifs/cache/c2_square.npz")
TOPOPT = np.load(REPO / "fgm_solve_campaign/out_topopt/"
                        "square_control_filteronly_maps.npz")
OUT = REPO / "deck_gifs/fig_solve_loop_schematic.png"

pm = C2["part_mask"]
r0, r1, c0, c1 = style.crop_indices(pm, pad=14)


def crop(a: np.ndarray) -> np.ndarray:
    return a[r0:r1, c0:c1]


PM = crop(pm)
T_solve = C2["T_solve"]                      # snapshots along the best forward
T_MID = crop(T_solve[len(T_solve) // 2])
T_STOP = crop(T_solve[-1])                   # snapshot at the optimal stop
t_pc, dt_pc = float(C2["t_pc_c"]), float(C2["dt_pc_c"])
PHI_STOP = np.clip((T_STOP - t_pc) / dt_pc + 0.5, 0.0, 1.0)
RES = PHI_STOP - PM                          # melt fraction minus nominal
SAT = crop(C2["sat_solved"])                 # filtered solved map
J_EVALS = C2["j_evals"]
RASTER = crop(TOPOPT["TO_4bpp"])             # production 4 bpp deliverable

fig = plt.figure(figsize=(16, 9), dpi=200)
fig.patch.set_facecolor(BG)

# ---------------------------------------------------------------- layout ----
TH = 0.295                     # thumbnail height (fig fraction)
TW = TH * 9.0 / 16.0           # thumbnail width, square on a 16:9 canvas
TOP_Y, BOT_Y = 0.545, 0.105
X1, X2, X3 = 0.060, 0.310, 0.560
PANELS = {
    1: (X1, TOP_Y), 2: (X2, TOP_Y), 3: (X3, TOP_Y),
    4: (X3, BOT_Y), 5: (X2, BOT_Y),
}


def panel(n: int, num: str, title: str, note: str):
    x, y = PANELS[n]
    ax = fig.add_axes([x, y, TW, TH])
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(DIM); s.set_linewidth(0.5)
    fig.text(x, y + TH + 0.030, num, color=WARM, fontsize=13, ha="left")
    fig.text(x + 0.024, y + TH + 0.030, title, color=FG, fontsize=11.5,
             ha="left")
    fig.text(x, y - 0.033, note, color=DIM, fontsize=8.0, ha="left")
    return ax


def outline(ax, color=ACCENT, lw=1.1):
    ax.contour(PM, levels=[0.5], colors=[color], linewidths=[lw])


# 1 nominal geometry
ax1 = panel(1, "1", "NOMINAL GEOMETRY",
            "chi: the part we want, and nothing else")
ax1.imshow(PM, cmap="gray", vmin=0, vmax=6, interpolation="nearest")
outline(ax1, lw=1.6)

# 2 forward march
ax2 = panel(2, "2", "FORWARD MARCH",
            "RF heating + conduction march T, mid-march shown")
ax2.imshow(T_MID, cmap=style.CMAP_T, interpolation="bilinear")
ax2.contour(T_MID, levels=[t_pc], colors=["white"], linewidths=[0.9],
            linestyles="dashed")
outline(ax2)

# 3 mismatch vs nominal
ax3 = panel(3, "3", "MISMATCH VS NOMINAL",
            "J: melt fraction minus chi, squared, over the whole bed")
ax3.imshow(RES, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear")
outline(ax3, color="#666666", lw=0.8)
ax3.text(0.04, 0.115, "red: melted beyond the part", color="#c94f4f",
         fontsize=7.5, transform=ax3.transAxes)
ax3.text(0.04, 0.045, "blue: unmelted inside it", color="#4f7ac9",
         fontsize=7.5, transform=ax3.transAxes)

# 4 adjoint backward sweep: labeled arrow stage, no field (none is stored)
ax4 = panel(4, "4", "ADJOINT BACKWARD SWEEP",
            "one backward pass in time gives dJ/ds at every cell")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
arr = FancyArrowPatch((0.88, 0.55), (0.12, 0.55), transform=ax4.transAxes,
                      arrowstyle="-|>", mutation_scale=22, color=WARM, lw=2.2)
ax4.add_artist(arr)
for xt, lab in ((0.88, "t*"), (0.50, ""), (0.12, "0")):
    ax4.plot([xt, xt], [0.51, 0.59], color=DIM, lw=0.9,
             transform=ax4.transAxes)
    if lab:
        ax4.text(xt, 0.64, lab, color=FG, fontsize=10, ha="center",
                 transform=ax4.transAxes)
ax4.text(0.5, 0.40, "residual seeds the sweep at t*", color=DIM,
         fontsize=8.0, ha="center", transform=ax4.transAxes)
ax4.text(0.5, 0.31, "cost: about one forward run", color=DIM,
         fontsize=8.0, ha="center", transform=ax4.transAxes)

# 5 filtered map update
ax5 = panel(5, "5", "FILTERED MAP UPDATE",
            "the dopant map moves downhill and stays smooth")
ax5.imshow(np.where(PM > 0, SAT, np.nan), cmap=style.CMAP_SAT, vmin=0, vmax=1,
           interpolation="bilinear")
outline(ax5)

# ---------------------------------------------------------------- arrows ----
def arrow(p, q, rad=0.0, color=DIM, lw=1.4):
    fa = FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle="-|>",
                         mutation_scale=15, color=color, lw=lw, shrinkA=0,
                         shrinkB=0, connectionstyle=f"arc3,rad={rad}")
    fig.add_artist(fa)


MIDT = TOP_Y + TH / 2.0
MIDB = BOT_Y + TH / 2.0
arrow((X1 + TW + 0.006, MIDT), (X2 - 0.006, MIDT))
arrow((X2 + TW + 0.006, MIDT), (X3 - 0.006, MIDT))
arrow((X3 + TW / 2.0, TOP_Y - 0.040), (X3 + TW / 2.0, BOT_Y + TH + 0.048))
arrow((X3 - 0.006, MIDB), (X2 + TW + 0.006, MIDB))
# loop back 5 -> 2, around the left edge of the bottom-row panel
arrow((X2 - 0.006, MIDB), (X2 - 0.060, (MIDB + MIDT) / 2.0), rad=-0.30,
      color=ACCENT)
arrow((X2 - 0.060, (MIDB + MIDT) / 2.0), (X2 - 0.006, MIDT), rad=-0.30,
      color=ACCENT)


def arrow_label(x, y, s, color=DIM, ha="center", fs=8.2):
    fig.text(x, y, s, color=color, fontsize=fs, ha=ha, va="center",
             linespacing=1.4)


arrow_label((X1 + TW + X2) / 2.0, MIDT + 0.038,
            "seed the dopant\nmap inside chi")
arrow_label((X2 + TW + X3) / 2.0, MIDT + 0.038,
            "read at its own\noptimal stop t*")
arrow_label(X3 + TW / 2.0 + 0.012, (TOP_Y + BOT_Y + TH) / 2.0,
            "backpropagate\nthe residual", ha="left")
arrow_label((X2 + TW + X3) / 2.0, MIDB + 0.040,
            "gradient through the\n1.0 mm physical filter", color=FG)
arrow_label(X2 - 0.068, (MIDT + MIDB) / 2.0, "next\nevaluation",
            color=ACCENT, ha="right")

# ---------------------------------------------------------------- exits -----
EX = X3 + TW + 0.045
arrow((X3 + TW + 0.006, MIDB), (EX + 0.030, MIDB), color=GOOD)
fig.text(EX + 0.038, MIDB + 0.135, "TWO EXITS, WHEN J STOPS FALLING",
         color=GOOD, fontsize=10.5, ha="left")
axr = fig.add_axes([EX + 0.130, MIDB - 0.075, TW * 0.52, TH * 0.52])
axr.set_facecolor(PANEL)
axr.set_xticks([]); axr.set_yticks([])
for s in axr.spines.values():
    s.set_edgecolor(GOOD); s.set_linewidth(0.7)
axr.imshow(np.where(PM > 0, RASTER, np.nan), cmap=style.CMAP_SAT, vmin=0,
           vmax=1, interpolation="nearest")
fig.text(EX + 0.038, MIDB + 0.095, "4 bpp production raster",
         color=FG, fontsize=9.5, ha="left")
fig.text(EX + 0.038, MIDB + 0.068,
         "quantized map,\nre-run through\nthe real forward",
         color=DIM, fontsize=8.0, ha="left", va="top", linespacing=1.4)
fig.text(EX + 0.038, MIDB - 0.075, "optimal stop t*", color=FG,
         fontsize=9.5, ha="left", va="top")
fig.text(EX + 0.038, MIDB - 0.103,
         "the run's own argmin\nof J: stop the heat there",
         color=DIM, fontsize=8.0, ha="left", va="top", linespacing=1.4)

# ---------------------------------------------------------------- sparkline -
sx, sy, sw, sh = X1, BOT_Y + 0.012, TW, TH - 0.085
axs = fig.add_axes([sx, sy, sw, sh])
axs.set_facecolor(PANEL)
best = np.minimum.accumulate(J_EVALS)
axs.plot(np.arange(1, len(J_EVALS) + 1), best, color=ACCENT, lw=1.7)
axs.plot(np.arange(1, len(J_EVALS) + 1), J_EVALS, ls="none", marker="o",
         ms=3.2, color=DIM)
axs.set_yticks([])
axs.set_xticks([1, len(J_EVALS)])
axs.tick_params(labelsize=8, colors=DIM, length=2)
for s in axs.spines.values():
    s.set_edgecolor(DIM); s.set_linewidth(0.5)
axs.set_xlabel("evaluation", fontsize=8, color=DIM, labelpad=1)
fig.text(sx, sy + sh + 0.030, "J FALLS AS THE LOOP TURNS", color=FG,
         fontsize=10.5, ha="left")
fig.text(sx, sy - 0.052,
         "mismatch J per gradient evaluation,\nbest so far, stored solve run",
         color=DIM, fontsize=8.0, ha="left", va="top", linespacing=1.4)

# ---------------------------------------------------------------- titles ----
fig.text(0.060, 0.945, "THE SHAPE-FIDELITY SOLVE LOOP", color=FG, fontsize=19)
fig.text(0.060, 0.902,
         "filter-only adjoint recipe: the physics chooses the dopant map",
         color=DIM, fontsize=11)

fig.savefig(OUT, facecolor=BG)
print(f"wrote {OUT}")
