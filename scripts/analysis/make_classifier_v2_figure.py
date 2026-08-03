#!/usr/bin/env python3
"""The updated actuator-classifier figure: version 1 against version 2.

Writes `fgm_solve_campaign/figs_intake/fig_intake_classifier.png`.

NOTE ON THE PATH. `make_intake_figures.py classifier` writes the VERSION-1
figure to this same path. Run this driver after it, or alone; the version-1
figure is reproducible at any time from that driver.

WHAT THE FIGURE HAS TO SAY AT A GLANCE, in this order:
  (top row) the decision the classifier actually makes is a REDUCTION FACTOR,
    and it is measured against a different injected dopant map in each of the
    three columns. On the free proportional-inverse stand-in the four measured
    rotation WINS sit above the threshold and the three FAILURES below it; on
    the uniform map (version 1) and on the expensive solved-map variant they
    interleave, so no threshold can work there. That single row is the result.
  (bottom row) every geometry the pipeline has measured, static residual against
    best-mode residual under the free stand-in, coloured by the version-2 class,
    with the geometries whose class CHANGED from version 1 marked.

Run:  ./.venv312/bin/python scripts/analysis/make_classifier_v2_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import geometry_actuator as ga        # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
FIGS = REPO / "fgm_solve_campaign/figs_intake"
DPI = 180

CLASS_COLOR = {"MAP_SUFFICES": "#7b3294", "MODE_SUFFICES": "#2c7bb6",
               "MAP_PLUS_MODE": "#fdae61", "PHYSICAL_LIMIT": "#d7191c"}
WIN_C, FAIL_C = "#1a9641", "#d7191c"

COLUMNS = [
    ("uniform", "v1",
     "version 1: uniform dopant map\n(actuator against NO actuator)",
     ga.MIN_REDUCTION),
    ("prop_inverse", "v2_prop_inverse",
     "version 2 FREE: proportional-inverse stand-in\n(zero extra gradient solves)",
     ga.MIN_REDUCTION_V2),
    ("solved", "v2_solved",
     "version 2 ONE FORWARD: static SOLVED map\n(one filtered gradient solve first)",
     ga.MIN_REDUCTION_V2),
]


def _reduction(res: dict) -> float:
    others = {k: v for k, v in res.items() if k != "static"}
    best = min(others.values()) if others else res["static"]
    return res["static"] / max(min(best, res["static"]), 1e-30)


def main() -> None:
    S = json.loads((OUT / "classifier_v2_summary.json").read_text())
    per = S["per_geometry"]
    conf = S["confusion"]

    fig = plt.figure(figsize=(15.0, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.42,
                          wspace=0.22)

    # ---------------- top row: the decision, three injected maps -------------
    measured = [n for n, r in per.items() if r["ground_truth"]]
    for col, (basis, key, title, thr) in enumerate(COLUMNS):
        ax = fig.add_subplot(gs[0, col])
        pts = []
        for n in measured:
            r = per[n]
            if basis not in r["residual"]:
                continue
            pts.append((n, _reduction(r["residual"][basis]),
                        r["ground_truth"]["outcome"] == "rotation_wins"))
        pts.sort(key=lambda t: t[1])
        for i, (n, red, win) in enumerate(pts):
            c = WIN_C if win else FAIL_C
            ax.scatter([red], [i], s=110, color=c, zorder=3,
                       marker="o" if win else "X")
            ax.annotate(n, (red, i), textcoords="offset points", xytext=(11, 0),
                        va="center", fontsize=8.5, color=c)
        ax.axvline(thr, color="k", ls="--", lw=1.3)
        ax.annotate(f"threshold {thr:.3f}", (thr, len(pts) - 0.25),
                    textcoords="offset points", xytext=(-4, 0), fontsize=7.5,
                    rotation=90, va="top", ha="right", color="0.25")
        c = conf[key]
        sep = S["reduction_separation"][basis]
        ok = sep["separates"]
        ax.set_title(f"{title}\n{c['correct']} of {c['n']} correct   |   "
                     + ("SEPARATES, margin "
                        f"{sep.get('margin_pct', 0.0):.1f} pct"
                        if ok else "DOES NOT SEPARATE at any threshold"),
                     fontsize=9,
                     color=("#1a9641" if ok else "#d7191c"))
        ax.set_yticks([])
        ax.set_ylim(-0.7, len(pts) - 0.05)
        ax.set_xlim(0.95, 2.45)
        ax.set_xlabel("reduction factor  A(static) / A(best mode)", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    fig.text(0.008, 0.965,
             "Circle, green = rotation BEAT the solved static arm (measured).   "
             "Cross, red = it did not.   The classifier recommends a turntable "
             "to the RIGHT of the dashed threshold.",
             fontsize=9)

    # ---------------- bottom row: every geometry, free stand-in --------------
    ax = fig.add_subplot(gs[1, :])
    names = sorted(per, key=lambda n: per[n]["decisions"]["v2_prop_inverse"]
                   ["residual_anisotropy"])
    a_static = [per[n]["residual"]["prop_inverse"]["static"] for n in names]
    a_best = [per[n]["decisions"]["v2_prop_inverse"]["residual_anisotropy"]
              for n in names]
    cls = [per[n]["decisions"]["v2_prop_inverse"]["actuator_class"]
           for n in names]
    v1cls = [per[n]["decisions"]["v1"]["actuator_class"] for n in names]
    xi = np.arange(len(names))
    ax.bar(xi - 0.21, a_static, width=0.42, color="0.75")
    ax.bar(xi + 0.21, a_best, width=0.42, color=[CLASS_COLOR[c] for c in cls])
    ax.axhline(ga.A2_MODE_SUFFICES, color="k", ls="--", lw=1.0)
    ax.axhline(ga.A2_PHYSICAL_LIMIT, color="k", ls=":", lw=1.0)
    ax.text(len(names) - 0.4, ga.A2_MODE_SUFFICES + 0.015,
            f"A2_MODE_SUFFICES {ga.A2_MODE_SUFFICES:.2f}", ha="right", fontsize=8)
    ax.text(len(names) - 0.4, ga.A2_PHYSICAL_LIMIT + 0.015,
            f"A2_PHYSICAL_LIMIT {ga.A2_PHYSICAL_LIMIT:.2f}", ha="right",
            fontsize=8)
    for i, n in enumerate(names):
        gt = per[n]["ground_truth"]
        if gt:
            w = gt["outcome"] == "rotation_wins"
            ax.annotate("rotation WINS" if w else "rotation FAILS",
                        (i + 0.21, a_best[i]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5,
                        color=WIN_C if w else FAIL_C, rotation=90)
        if cls[i] != v1cls[i]:
            ax.annotate(f"was {v1cls[i]}", (i - 0.21, a_static[i]),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=6.5, rotation=90, color="0.35")
    from matplotlib.patches import Patch
    handles = [Patch(color="0.75",
                     label="static kernel WITH the free stand-in map injected")]
    handles += [Patch(color=CLASS_COLOR[c], label=f"best mode, class {c}")
                for c in ("MAP_SUFFICES", "MODE_SUFFICES", "MAP_PLUS_MODE",
                          "PHYSICAL_LIMIT")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_xticks(xi)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    for t, n in zip(ax.get_xticklabels(), names):
        if per[n]["ground_truth"]:
            t.set_fontweight("bold")
    ax.set_ylabel("residual azimuthal anisotropy (occupancy weighted)")
    ax.set_ylim(0, max(a_static) * 1.65)
    ax.set_title("Version-2 spectrum on the eighteen library shapes plus the two "
                 "novel geometries, free stand-in, grid 120. Bold names carry a "
                 "MEASURED rotation-versus-solved-map outcome; grey labels mark "
                 "a class that CHANGED from version 1.", fontsize=9.5, pad=12)

    fig.suptitle("Actuator classifier version 2: predict against the SOLVED arm, "
                 "not the uniform arm. The free stand-in is 7 of 7; the "
                 "expensive one-forward variant is 4 of 7.", fontsize=12.5,
                 y=0.995)
    fig.savefig(FIGS / "fig_intake_classifier.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / "fig_intake_classifier.png")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    main()
