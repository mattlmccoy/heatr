#!/usr/bin/env python3
"""One figure for the rotating grid hold-out: where the SOLVED class is lost.

Only the arms whose CLASS changes between the grids get a panel, which is the
campaign's rule for when a figure is owed. Each column is one arm. The top row
puts the melted region at grid 120 and at grid 160 on the same axes against the
nominal outline, so the mechanism (which part of the part stops melting) is
visible at a glance. The bottom row is the intersection-over-union ladder at the
two grids with the 0.95 SOLVED line drawn, so the class change is readable
without going to the table.

Run:
  ./.venv312/bin/python scripts/analysis/make_rot_holdout_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "fgm_solve_campaign/out_rot_holdout"
FIGS = REPO / "fgm_solve_campaign/figs_rot_holdout"

TITLES = {
    "cross_index90": "cross, 90-degree indexing\nsolved four-angle map",
    "cross_dwell": "cross, co-solved dwell\nstored 20 s-cycle program",
    "keyhole_cont": "keyhole, continuous rotation\nstored 448-move program",
    "star_index90": "star, 90-degree indexing\nsolved four-angle map",
}
C120, C160 = "#1f77b4", "#d62728"
LADDER = [("ROT_solved", "rotating, solved map"),
          ("ROT_uniform", "rotating, uniform map"),
          ("STATIC_solved", "static, solved map"),
          ("STATIC_uniform", "static, uniform map")]


def _melt_outline(ax, phi, x, y, color, label, lw):
    ax.contour(x * 1e3, y * 1e3, phi, levels=[0.5], colors=[color],
               linewidths=lw)
    ax.plot([], [], color=color, lw=lw, label=label)


def _bar_label(ax, val, yy, color):
    """Outside the bar normally; inside it when the 0.95 line is in the way."""
    if val > 0.88:
        ax.text(val - 0.012, yy, f"{val:.3f}", va="center", ha="right",
                fontsize=6.5, color="white", fontweight="bold")
    else:
        ax.text(val + 0.012, yy, f"{val:.3f}", va="center", fontsize=6.5,
                color=color)


def panel(axes, name: str, show_ylabels: bool = True) -> dict:
    res = json.loads((OUT / f"{name}.json").read_text())
    z = np.load(OUT / f"{name}_maps.npz")
    v = res["verdict"]
    ax_map, ax_bar = axes

    chi = z["g160_chi"]
    x160, y160 = z["g160_x"], z["g160_y"]
    ax_map.contourf(x160 * 1e3, y160 * 1e3, chi, levels=[0.5, 2.0],
                    colors=["0.86"])
    ax_map.contour(x160 * 1e3, y160 * 1e3, chi, levels=[0.5], colors=["0.35"],
                   linewidths=1.0, linestyles=":")
    _melt_outline(ax_map, z["g120_phi_ROT_solved"], z["g120_x"], z["g120_y"],
                  C120, f"melt at grid 120, IoU {v['IoU_at_120']:.4f}", 2.0)
    _melt_outline(ax_map, z["g160_phi_ROT_solved"], x160, y160,
                  C160, f"melt at grid 160, IoU {v['IoU_at_160']:.4f}", 1.6)
    lim = np.max(np.abs(np.array([x160[chi.sum(0) > 0].min(),
                                  x160[chi.sum(0) > 0].max(),
                                  y160[chi.sum(1) > 0].min(),
                                  y160[chi.sum(1) > 0].max()]))) * 1e3 * 1.35
    ax_map.set_xlim(-lim, lim)
    ax_map.set_ylim(-lim, lim)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_title(TITLES[name], fontsize=9)
    ax_map.legend(loc="lower center", fontsize=6.5, frameon=False,
                  bbox_to_anchor=(0.5, -0.16))

    a0 = res["grids"]["120"]["arms"]
    a1 = res["grids"]["160"]["arms"]
    ypos = np.arange(len(LADDER))[::-1]
    h = 0.36
    ax_bar.barh(ypos + h / 2, [a0[k]["IoU"] for k, _ in LADDER], height=h,
                color=C120, label="grid 120")
    ax_bar.barh(ypos - h / 2, [a1[k]["IoU"] for k, _ in LADDER], height=h,
                color=C160, label="grid 160")
    for i, (k, _lab) in enumerate(LADDER):
        yy = ypos[i]
        _bar_label(ax_bar, a0[k]["IoU"], yy + h / 2, C120)
        _bar_label(ax_bar, a1[k]["IoU"], yy - h / 2, C160)
    ax_bar.axvline(0.95, color="k", lw=1.2, ls="--")
    ax_bar.text(0.95, len(LADDER) - 0.55, "SOLVED\nclass 0.95", fontsize=6.5,
                ha="center", va="bottom")
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels([lab for _, lab in LADDER] if show_ylabels
                           else [""] * len(LADDER), fontsize=8)
    ax_bar.set_xlim(0.0, 1.14)
    ax_bar.set_ylim(-0.75, len(LADDER) - 0.05)
    ax_bar.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.set_xlabel("intersection over union at the arm's own stop",
                      fontsize=7.5)
    ax_bar.tick_params(axis="x", labelsize=7)
    for sp in ("top", "right"):
        ax_bar.spines[sp].set_visible(False)
    return v


def main() -> None:
    names = [n for n in ("cross_index90", "cross_dwell", "keyhole_cont",
                         "star_index90")
             if json.loads((OUT / f"{n}.json").read_text())["verdict"]
             ["class_changes"]]
    if not names:
        print("no arm changed class; no figure is owed")
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(names), figsize=(3.9 * len(names), 7.0),
                             gridspec_kw={"height_ratios": [1.35, 1.0],
                                          "hspace": 0.30, "wspace": 0.10})
    axes = np.atleast_2d(axes)
    for j, n in enumerate(names):
        panel((axes[0, j], axes[1, j]), n, show_ylabels=(j == 0))
    axes[1, 0].legend(fontsize=7.5, frameon=False, ncol=2,
                      loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.suptitle("Rotating and dwell arms, grid hold-out: solved at grid 120, "
                 "scored at grid 160\n"
                 "every arm below leaves the SOLVED class at the hold-out grid; "
                 "the ranking against uniform and against static survives",
                 fontsize=10.5, y=0.985)
    p = FIGS / "fig_rot_holdout_class_change.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    sys.exit(main())
