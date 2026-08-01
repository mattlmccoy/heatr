"""Figures for the robustness validation. One figure per task.

  fig_robust_grid.png  Task A, optimize at 120 and score at 160
  fig_robust_rim.png   Task B, Gaussian rim smoothing at grid 120

Run:
  ./.venv312/bin/python -m adjoint2d.make_robust_figures <outdir> <figdir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHAPES = ("square", "circle", "trapezoid", "triangle", "diamond", "rectangle")
DPI = 180


def _load(outdir: Path, shape: str, task: str) -> dict:
    return json.loads((outdir / f"{shape}_{task}.json").read_text())


def fig_grid(outdir: Path, figpath: Path) -> None:
    rows = [(s, _load(outdir, s, "grid")) for s in SHAPES
            if (outdir / f"{s}_grid.json").exists()]
    n = len(rows)
    fig, axs = plt.subplots(2, 2, figsize=(14.0, 9.4))
    y = np.arange(n)
    h = 0.26
    C120, C160, CREC = "#1f77b4", "#d62728", "#ff9f40"
    offs = (h, 0.0, -h)
    labs = ("grid 120 (solve grid, calibrated)",
            "grid 160, pinned voltage",
            "grid 160, dose-matched voltage")
    cols = (C120, C160, CREC)

    def solved(r, k):
        return (r["arms_120"]["A1_4bpp"], r["arms"]["A1_4bpp_160"],
                r["recal"]["arms"]["A1_4bpp_160_recal"])[k]

    def hist(r, k):
        return (r["arms_120"]["HIST_best"], r["arms"]["HIST_best_160"],
                r["recal"]["arms"]["HIST_best_160_recal"])[k]

    # Panel A: intersection over union
    a = axs[0, 0]
    for k in range(3):
        a.barh(y + offs[k], [solved(r, k)["IoU"] for _s, r in rows], height=h,
               color=cols[k], label=labs[k])
        a.plot([hist(r, k)["IoU"] for _s, r in rows], y + offs[k], "k|", ms=11, mew=1.8,
               label="best stored mask" if k == 0 else None)
    a.axvline(0.95, color="green", ls="--", lw=1, alpha=0.8)
    a.set_yticks(y, [s for s, _r in rows]); a.invert_yaxis()
    a.set_xlim(0.4, 1.02)
    a.set_xlabel("intersection over union of the melted region with the nominal part")
    a.set_title("A. Absolute fidelity does NOT transfer, and dose matching does not\n"
                "recover it (bars: solved 4 bits-per-pixel arm; ticks: best stored mask;\n"
                "dashed: the IoU = 0.95 solved band)", fontsize=10)
    a.grid(axis="x", alpha=0.3)

    # Panel B: relative J margin against the best stored mask
    b = axs[0, 1]
    m = [[100.0 * r["verdict"]["dJ_rel_120"] for _s, r in rows],
         [100.0 * r["verdict"]["dJ_rel_160"] for _s, r in rows],
         [100.0 * r["recal"]["dJ_rel"] for _s, r in rows]]
    for k in range(3):
        b.barh(y + offs[k], np.clip(m[k], -300, None), height=h, color=cols[k],
               label=labs[k])
    for k in range(3):
        for yy, v in zip(y, m[k]):
            if v < -300:
                b.text(-298, yy + offs[k], f"{v:.0f}", fontsize=7, va="center",
                       ha="left", color="white")
    b.axvline(0.0, color="k", lw=1)
    b.set_yticks(y, [s for s, _r in rows]); b.invert_yaxis()
    b.set_xlim(-320, None)
    b.set_xlabel("J margin against the best stored mask, percent "
                 "(positive = the solved map wins)")
    b.set_title("B. Ranking transfer. A sign change between the blue bar and the\n"
                "orange bar is a win that did not survive the grid change", fontsize=10)
    b.grid(axis="x", alpha=0.3)
    for yy, (v0, v2) in zip(y, zip(m[0], m[2])):
        if (v0 > 0) != (v2 > 0):
            b.text(2, yy, "sign flip", fontsize=8, color="darkred", va="center")

    # Panel C: absorbed power, the named dose confound
    c = axs[1, 0]
    for k in range(3):
        c.barh(y + offs[k], [solved(r, k)["P_abs_W_per_m"] for _s, r in rows],
               height=h, color=cols[k], label=labs[k])
    c.axvline(500.0, color="k", ls="--", lw=1)
    c.set_yticks(y, [s for s, _r in rows]); c.invert_yaxis()
    c.set_xlabel("absorbed power of the solved arm at its stop, watts per metre of depth")
    c.set_title("C. The confound. The drive voltage was calibrated so the UNIFORM\n"
                "arm absorbs 500 W/m at grid 120; at 160 it absorbs 359 to 443",
                fontsize=10)
    c.grid(axis="x", alpha=0.3)

    # Panel D: under-melt, the starvation signature
    d = axs[1, 1]
    for k in range(3):
        d.barh(y + offs[k], [solved(r, k)["part_under_melt_pct"] for _s, r in rows],
               height=h, color=cols[k], label=labs[k])
    d.set_yticks(y, [s for s, _r in rows]); d.invert_yaxis()
    d.set_xlabel("part cells left below phi = 0.5 at the stop, percent")
    d.set_title("D. Starvation signature. The 160 arms leave more of the part\n"
                "unmelted than the 120 solve did, and on the square and the diamond\n"
                "matching the dose does not bring the under-melt back down", fontsize=10)
    d.grid(axis="x", alpha=0.3)

    fig.suptitle("Task A. Grid hold-out: solve the dopant map at 120 x 120, score it at "
                 "160 x 160. Forward runs only; only the stop is re-optimized.",
                 fontsize=12.5)
    hh, ll = a.get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=4, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(figpath, dpi=DPI)
    plt.close(fig)


def fig_rim(outdir: Path, figpath: Path) -> None:
    rows = [(s, _load(outdir, s, "rim")) for s in SHAPES
            if (outdir / f"{s}_rim.json").exists()]
    radii = [0.0, 1.0, 2.0]
    keys = ["r0.0", "r1.0", "r2.0"]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))
    cmap = plt.get_cmap("tab10")

    a, b, c = ax
    for i, (s, r) in enumerate(rows):
        col = cmap(i)
        J = [r["arms"][k]["J"] for k in keys]
        I = [r["arms"][k]["IoU"] for k in keys]
        a.plot(radii, [v / J[0] for v in J], "o-", color=col, label=s)
        b.plot(radii, I, "o-", color=col, label=s)
        c.plot(radii, [v / r["hist_best"]["J"] for v in J], "o-", color=col, label=s)
        c.plot(radii, [v / r["uniform"]["J"] for v in J], "s--", color=col, alpha=0.45,
               ms=4, lw=1)

    a.set_xticks(radii)
    a.set_xlabel("Gaussian smoothing radius, cells")
    a.set_ylabel("J relative to the unsmoothed solved map")
    a.set_yscale("log")
    a.axhline(1.0, color="k", lw=1)
    a.set_title("A. The rim structure is load-bearing:\n"
                "J relative to radius 0 (log scale)", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    b.set_xticks(radii)
    b.set_xlabel("Gaussian smoothing radius, cells")
    b.set_ylabel("intersection over union with the nominal part")
    b.axhline(0.95, color="green", ls="--", lw=1, alpha=0.7)
    b.set_title("B. Intersection over union against smoothing radius\n"
                "(dashed: the IoU = 0.95 solved band)", fontsize=10)
    b.grid(alpha=0.3)

    c.set_xticks(radii)
    c.set_xlabel("Gaussian smoothing radius, cells")
    c.set_ylabel("J of the smoothed arm, relative to the reference")
    c.set_yscale("log")
    c.axhline(1.0, color="k", lw=1.2)
    c.set_title("C. Where the smoothing costs the win. Circles and solid lines:\n"
                "against the best stored mask. Squares and dashed lines: against\n"
                "uniform. Above the black line the reference is better.", fontsize=10)
    c.grid(alpha=0.3)

    fig.suptitle("Task B. Rim robustness: blur the solved continuous map, re-quantize "
                 "4 bits per pixel, re-run, re-optimize the stop (grid 120)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figpath, dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    _out = Path(sys.argv[1]).resolve()
    _fig = Path(sys.argv[2]).resolve()
    _fig.mkdir(parents=True, exist_ok=True)
    if (_out / "square_grid.json").exists():
        fig_grid(_out, _fig / "fig_robust_grid.png")
        print("wrote", _fig / "fig_robust_grid.png")
    fig_rim(_out, _fig / "fig_robust_rim.png")
    print("wrote", _fig / "fig_robust_rim.png")
