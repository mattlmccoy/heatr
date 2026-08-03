#!/usr/bin/env python3
"""The ROTATING FORWARD GRID LADDER figure, and the merged ladder table.

ONE figure, three panels, designed so the verdict is readable at a glance:

  A  cross: intersection over union against grid number, the rotating
     uniform-dopant arm, the static uniform arm and the transferred solved map,
     with the 0.95 SOLVED class line and the dissertation's grid 120 marked.
     The cross's rotating curve does NOT settle; it scatters.
  B  keyhole: the same three arms on the same axes. The rotating uniform curve
     DOES settle.
  C  the size of each successive ladder step, on a logarithmic axis, for the two
     rotating uniform arms. A converging forward walks this line downwards. The
     keyhole does. The cross does not.

Merges the main ladder file with the alignment-probe file for each shape, so a
shape run in two invocations reads as one ladder.

Run:
  ./.venv312/bin/python scripts/analysis/make_rot_ladder_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "fgm_solve_campaign/out_rot_ladder"
FIGS = REPO / "fgm_solve_campaign/figs_rot_ladder"

SOURCES = {
    "cross": ["cross_ladder.json", "cross_align_probe_ladder.json"],
    "keyhole": ["keyhole_ladder.json", "keyhole_align_probe_ladder.json"],
}
ARMS = ("ROT_uniform", "STATIC_uniform", "ROT_transfer")
STYLE = {
    "ROT_uniform": ("#1f77b4", "o", "rotating, uniform dopant"),
    "STATIC_uniform": ("#7f7f7f", "s", "static, uniform dopant"),
    "ROT_transfer": ("#d62728", "^", "rotating, map solved at 120"),
}
SOLVED_CLASS_IOU = 0.95


def merged(shape: str) -> dict:
    grids: dict[int, dict] = {}
    files = []
    for fn in SOURCES[shape]:
        p = OUT / fn
        if not p.exists():
            continue
        files.append(str(p))
        d = json.loads(p.read_text())
        for g, rec in d["grids"].items():
            grids[int(g)] = rec
    if not grids:
        raise SystemExit(f"no ladder files for {shape}")
    return {"shape": shape, "sources": files, "grids": grids}


def series(m: dict, arm: str):
    ns = sorted(n for n in m["grids"] if arm in m["grids"][n]["arms"])
    return (np.array(ns, dtype=float),
            np.array([m["grids"][n]["arms"][arm]["IoU"] for n in ns]),
            np.array([m["grids"][n]["arms"][arm]["J_per_part_cell"]
                      for n in ns]))


def panel(ax, m: dict, title: str) -> None:
    for arm in ARMS:
        try:
            n, iou, _ = series(m, arm)
        except (KeyError, ValueError):
            continue
        if n.size == 0:
            continue
        c, mk, lab = STYLE[arm]
        ax.plot(n, iou, marker=mk, color=c, lw=1.8, ms=6, label=lab)
    ax.axhline(SOLVED_CLASS_IOU, color="#2ca02c", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], SOLVED_CLASS_IOU + 0.006, "SOLVED class 0.95",
            color="#2ca02c", fontsize=8, ha="right", va="bottom")
    ax.axvline(120, color="k", ls=":", lw=1.0, alpha=0.5)
    ax.text(120, 0.995, " grid 120, the grid\n every map was solved on",
            fontsize=7.5, color="k", alpha=0.8, ha="left", va="top")
    # the spread of the rotating uniform arm over the grids AT AND ABOVE 120,
    # which is the quantity the verdict is read from
    n, iou, _ = series(m, "ROT_uniform")
    sel = iou[n >= 120]
    ax.axhspan(float(sel.min()), float(sel.max()), color="#1f77b4", alpha=0.09,
               zorder=0)
    ax.text(360, float(sel.max()) + 0.004,
            f"spread over n >= 120: {float(sel.max() - sel.min()):.3f}",
            fontsize=7.5, color="#1f77b4", ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([96, 120, 160, 180, 200, 240, 360])
    ax.set_xticklabels(["96", "120", "160", "180", "200", "240", "360"],
                       fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("grid number n (cells across the 60 mm chamber)", fontsize=9)
    ax.set_ylabel("intersection over union at the arm's own optimal stop",
                  fontsize=9)
    ax.set_ylim(0.50, 1.0)
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    mc, mk = merged("cross"), merged("keyhole")

    fig = plt.figure(figsize=(13.0, 5.0), dpi=180)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.85], wspace=0.28,
                          left=0.055, right=0.985, top=0.86, bottom=0.14)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    panel(axa, mc, "A. cross, 90-degree indexing: the rotating forward SCATTERS")
    panel(axb, mk, "B. keyhole, continuous rotation: the rotating forward SETTLES")

    for m, col, lab in ((mc, "#1f77b4", "cross, rotating uniform"),
                        (mk, "#ff7f0e", "keyhole, rotating uniform")):
        n, iou, _ = series(m, "ROT_uniform")
        d = np.abs(np.diff(iou))
        mid = n[1:]
        axc.plot(mid, np.maximum(d, 3e-4), marker="o", color=col, lw=1.8,
                 ms=6, label=lab)
    axc.set_ylim(3e-4, 3e-1)
    axc.set_yscale("log")
    axc.set_xscale("log")
    axc.set_xticks([120, 160, 180, 200, 240, 360])
    axc.set_xticklabels(["120", "160", "180", "200", "240", "360"], fontsize=8)
    axc.minorticks_off()
    axc.set_xlabel("grid number n, step measured from the previous grid",
                   fontsize=9)
    axc.set_ylabel("size of the ladder step in intersection over union",
                   fontsize=9)
    axc.grid(alpha=0.25, which="both")
    axc.set_title("C. does the ladder step shrink?", fontsize=10)
    axc.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig.suptitle(
        "Rotating forward grid ladder, uniform dopant, drive recalibrated at "
        "every grid so the static uniform arm absorbs 500 W/m. Forward runs "
        "only, nothing solved here.", fontsize=10.5, y=0.965)
    p = FIGS / "fig_rot_grid_ladder.png"
    fig.savefig(p)
    print(f"wrote {p}", flush=True)

    # the merged table, for the report
    tab = {}
    for m in (mc, mk):
        rows = []
        for n in sorted(m["grids"]):
            g = m["grids"][n]
            r = {"n_grid": n, "n_part_cells": g["n_part_cells"],
                 "n_substeps": g["n_substeps"],
                 "voltage_v": g["calibration"]["voltage_v"],
                 "p_verified_w_per_m": g["calibration"]["p_verified_w_per_m"],
                 "raster_minus_area_pct":
                     100.0 * g["raster_vs_area"]["area_rel_delta"],
                 "wall_s": g["wall_s"]}
            for arm in ARMS + ("QS_uniform",):
                a = g["arms"].get(arm)
                if a is None:
                    continue
                r[arm] = {k: a[k] for k in
                          ("J", "J_per_part_cell", "IoU", "IoU_area",
                           "bed_melt_pct_of_part", "part_under_melt_pct",
                           "t_stop_s", "t_stop_at_horizon", "P_abs_W_per_m")}
                r[arm]["energy_res_pct"] = (
                    100.0 * a["energy_gate"]["rel_residual_at_index"])
                r[arm]["energy_gate_PASS"] = a["energy_gate"]["PASS"]
                r[arm]["max_T_at_stop_c"] = a["max_T_at_stop_c"]
            rows.append(r)
        tab[m["shape"]] = {"sources": m["sources"], "rows": rows}
    (OUT / "ladder_merged_table.json").write_text(
        json.dumps(tab, indent=2, default=float))
    print(f"wrote {OUT / 'ladder_merged_table.json'}", flush=True)

    for m in (mc, mk):
        n, iou, _ = series(m, "ROT_uniform")
        print(f"{m['shape']:8s} ROT_uniform n=" +
              " ".join(f"{int(x)}" for x in n) + "  IoU=" +
              " ".join(f"{v:.4f}" for v in iou) + "  steps=" +
              " ".join(f"{v:+.4f}" for v in np.diff(iou)), flush=True)


if __name__ == "__main__":
    main()
