"""Render fig_solve_census_wide.png: the 18-shape library census, deck style.

Same data and verdict as fgm_solve_campaign/figs/fig_lib_census.png (all
numbers read fresh from out_lib/<shape>.json, none altered), re-laid-out as a
denser 16:9 slide: IoU dumbbell left, J-change bars right, bigger labels,
near-black deck background. The verdict counts are recomputed from the JSONs
and ASSERTED to equal the stored verdict (J: 13 of 18, IoU: 13 of 18,
IoU >= 0.95: 7 of 18).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
OUT = REPO / "deck_gifs/fig_solve_census_wide.png"

RED = "#e57373"
BLUE = "#64b5f6"


def rows_from_lib() -> list[dict]:
    rows = []
    for f in sorted(OUT_LIB.glob("*.json")):
        r = json.loads(f.read_text())
        if not isinstance(r, dict) or "arms" not in r or "verdict" not in r:
            continue
        d, h, u = r["arms"]["A1_4bpp"], r["arms"]["HIST_best"], r["arms"]["U_uniform"]
        rows.append({
            "shape": r["shape"], "iou_solved": d["IoU"], "iou_hist": h["IoU"],
            "iou_uniform": u["IoU"], "J_solved": d["J"], "J_hist": h["J"],
            "dJ": (h["J"] - d["J"]) / max(abs(h["J"]), 1e-30),
            "dIoU": d["IoU"] - h["IoU"],
            "horizon": bool(h["t_stop_at_horizon"] or d["t_stop_at_horizon"]),
        })
    return rows


def main() -> None:
    rows = sorted(rows_from_lib(), key=lambda r: r["iou_solved"])
    n = len(rows)
    n_J = sum(1 for r in rows if r["dJ"] > 0)
    n_I = sum(1 for r in rows if r["dIoU"] > 0)
    n_solved = sum(1 for r in rows if r["iou_solved"] >= 0.95)
    assert (n, n_J, n_I, n_solved) == (18, 13, 13, 7), \
        f"verdict changed: {(n, n_J, n_I, n_solved)}"

    fig = plt.figure(figsize=(19.2, 10.8), dpi=style.DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0],
                          left=0.135, right=0.975, top=0.845, bottom=0.12,
                          wspace=0.05)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    ypos = np.arange(n)

    fig.text(0.5, 0.955,
             "shape-library census: does one solved printable dopant map beat "
             "the best map ever stored?",
             ha="center", fontsize=19, color=style.FG)
    fig.text(0.5, 0.905,
             f"VERDICT   J: {n_J} of {n}      IoU: {n_I} of {n}      "
             f"melted region matches the nominal part (IoU >= 0.95): "
             f"{n_solved} of {n}",
             ha="center", fontsize=15, color=style.GOOD)

    # --- panel A: IoU dumbbell ---------------------------------------------
    axL.set_facecolor(style.PANEL)
    axL.axvspan(0.95, 1.02, color=style.GOOD, alpha=0.10, zorder=0)
    axL.axvline(0.95, color=style.GOOD, lw=1.2, ls="--", alpha=0.7, zorder=1)
    for i, r in enumerate(rows):
        a, b = r["iou_hist"], r["iou_solved"]
        col = style.GOOD if b > a else (RED if b < a else style.DIM)
        axL.annotate("", xy=(b, i), xytext=(a, i),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4,
                                     shrinkA=0, shrinkB=0, alpha=0.95), zorder=3)
        axL.plot([r["iou_uniform"]], [i], marker="|", ms=15, color=style.DIM,
                 mew=2.0, zorder=2)
    axL.scatter([r["iou_hist"] for r in rows], ypos, s=70,
                facecolor=style.PANEL, edgecolor=style.FG, lw=1.4, zorder=4,
                label="best stored historical mask")
    axL.scatter([r["iou_solved"] for r in rows], ypos, s=85, color=BLUE,
                zorder=5, label="solved map, 4 bits per pixel, single pass")
    axL.plot([], [], marker="|", ls="none", ms=13, mew=2.0, color=style.DIM,
             label="uniform s = 1")
    axL.set_yticks(ypos)
    axL.set_yticklabels([r["shape"].replace("_", " ") for r in rows],
                        fontsize=13)
    axL.set_xlabel("IoU of the melted region with the nominal part\n"
                   "(each arm at its own optimal stop)", fontsize=12,
                   color=style.DIM)
    axL.set_xlim(0.34, 1.025)
    axL.set_ylim(-0.7, n - 0.3)
    axL.grid(axis="x", alpha=0.15, color=style.DIM)
    axL.tick_params(labelsize=11)
    leg = axL.legend(loc="upper left", fontsize=11, framealpha=0.9,
                     facecolor=style.PANEL, edgecolor=style.DIM,
                     labelcolor=style.FG)
    axL.set_title("A.  shape fidelity (IoU)", fontsize=14, loc="left",
                  color=style.FG, pad=10)

    # --- panel B: J change bars, clamped at 100 % ---------------------------
    axR.set_facecolor(style.PANEL)
    LIM = 100.0
    for i, r in enumerate(rows):
        v = 100.0 * r["dJ"]
        vd = float(np.clip(v, -LIM * 0.97, LIM * 0.97))
        axR.barh(i, vd, color=style.GOOD if v > 0 else RED, alpha=0.9,
                 height=0.62)
        clipped = abs(v) > LIM * 0.97
        txt = f"{v:+.0f}%" + (" off scale" if clipped else "")
        if clipped:
            axR.plot([vd], [i], marker=">" if v > 0 else "<", ms=8,
                     color=style.FG, clip_on=False)
            axR.text(vd - 3.0 * np.sign(v), i, txt, va="center",
                     ha="right" if v > 0 else "left", fontsize=10.5,
                     color=style.BG, fontweight="bold")
        else:
            axR.text(vd + (2.5 if v >= 0 else -2.5), i, txt, va="center",
                     ha="left" if v >= 0 else "right", fontsize=10.5,
                     color=style.FG)
        if r["horizon"]:
            axR.text(0.985, i, "H", transform=axR.get_yaxis_transform(),
                     va="center", ha="right", fontsize=10, color=style.WARM)
    axR.axvline(0, color=style.DIM, lw=1.0)
    axR.set_xlabel("reduction in the objective J against the best stored mask, "
                   "percent\n(positive = the solved map is better; axis "
                   "clamped at 100 %)", fontsize=12, color=style.DIM)
    axR.grid(axis="x", alpha=0.15, color=style.DIM)
    axR.tick_params(labelleft=False, labelsize=11)
    axR.set_xlim(-LIM, LIM)
    axR.set_title("B.  objective J vs the best stored mask", fontsize=14,
                  loc="left", color=style.FG, pad=10)

    fig.text(0.5, 0.018,
             "grid 120; each arm scored at its own optimal stop; melt-region "
             "framing; H = an arm stopped at the horizon; "
             "source: fgm_solve_campaign/out_lib",
             ha="center", fontsize=10.5, color=style.DIM)

    fig.savefig(OUT, dpi=style.DPI)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
