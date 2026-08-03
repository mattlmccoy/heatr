"""Figures for the shape-library shape-fidelity solve.

Two families.

  fig_lib_<shape>.png   one composite per shape: the dopant map of every arm on
      the top row and the melt-fraction field at that arm's OWN optimal stop on
      the bottom row, with the nominal part outline in cyan and the melt front
      dashed white. The question the figure answers at a glance is where the
      melted region departs from the nominal shape, and whether the printer's
      level grid moves it.

  fig_lib_census.png    the whole-library verdict in one glance: for every shape
      the intersection over union of the best historical stored mask against the
      printable 4-bits-per-pixel solved map, as a slope chart, with the relative
      change in the objective J beside it.

Read/stop convention for every number drawn: t_stop = argmin over that arm's own
trajectory of J; the melted region is melt fraction >= 0.5.

Run:
  ./.venv312/bin/python -m adjoint2d.make_library_figures shape <shape> <out_lib> <figs>
  ./.venv312/bin/python -m adjoint2d.make_library_figures census <out_lib> <figs>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import forward as fwd, shape_objective as so
from .library_solve import SHAPES, shape_config
from .make_figures import crop_box
from .pins import build_case, load_cfg
from .verify_hist import load_stored_map

DPI = 180
PATIENCE = 250

PANEL_ORDER = [
    ("U_uniform", "uniform s = 1\nno grading", False),
    ("HIST_best", "best stored historical mask\n4 bits per pixel, permittivity co-varying", True),
    ("A1_cont", "solved map, box [0, 1]\ncontinuous, NOT printable", False),
    ("A1_4bpp", "solved map, box [0, 1], 4 bits per pixel\nSINGLE PASS, printable", False),
    ("A1_2bpp", "solved map, box [0, 1], 2 bits per pixel\nsingle pass", False),
    ("A15_4bpp", "solved map, box [0, 1.5], 4 bits per pixel\nDOUBLE PASS", False),
]


def _map_panel(ax, s, pm, extent, vmax, title):
    im = ax.imshow(np.where(pm, s, np.nan), origin="lower", extent=extent,
                   vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
    ny, nx = pm.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#ffffff", linewidths=1.0)
    ax.set_title(title, fontsize=7.0, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _phi_panel(ax, phi, pm, extent, title):
    im = ax.imshow(phi, origin="lower", extent=extent, vmin=0.0, vmax=1.0,
                   cmap="inferno", interpolation="bilinear")
    ny, nx = pm.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.4)
    ax.contour(xs, ys, phi, levels=[0.5], colors="#ffffff", linewidths=1.0, linestyles="--")
    ax.set_title(title, fontsize=7.0, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _run(case, s, eps_covary=False):
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE,
                     eps_covary=eps_covary)
    st = so.optimal_stop(tr, case)
    phi, _ = so.phi_field(tr.T_at_end(st.index), case)
    return phi, st


def shape_figure(shape: str, outdir: Path, figdir: Path) -> Path:
    res = json.loads((outdir / f"{shape}.json").read_text())
    cfg = load_cfg(shape_config(shape))
    case = build_case(cfg)
    pm = case.part_mask
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    maps = np.load(outdir / f"{shape}_maps.npz")

    hist_m = res["arms"]["HIST_best"]
    s_hist = load_stored_map(case, Path(hist_m["map_npz"]), cfg)
    if hist_m["convention"] == "outside1":
        s_hist = np.where(pm, s_hist, 1.0)

    cols = []
    for key, label, covary in PANEL_ORDER:
        if key == "HIST_best":
            cols.append((key, label, s_hist, True))
        elif key in maps.files:
            cols.append((key, label, np.asarray(maps[key], dtype=float), covary))

    fig, axes = plt.subplots(2, len(cols), figsize=(2.85 * len(cols), 7.3))
    im_map = im_phi = None
    for j, (key, label, s, covary) in enumerate(cols):
        m = res["arms"][key]
        extra = ""
        if key == "HIST_best":
            extra = f"\n{res['best_hist_arm'].replace('hist_', '')}"
        im_map = _map_panel(axes[0, j], s, pm, extent, 1.5,
                            f"{label}{extra}\nmean s in part {float(np.mean(s[pm])):.3f}")
        phi, st = _run(case, s, eps_covary=covary)
        flag = " (AT HORIZON)" if m["t_stop_at_horizon"] else ""
        im_phi = _phi_panel(
            axes[1, j], phi, pm, extent,
            f"melted region at own optimal stop\n{st.time_s:.0f} s{flag}\n"
            f"J {m['J']:.1f}     IoU {m['IoU']:.4f}\n"
            f"growth {m['bed_melt_pct_of_part']:.2f} %     "
            f"under {m['part_under_melt_pct']:.2f} %\n"
            f"energy residual {m['energy_gate']['rel_residual_at_index']*100:.2f} % of dose")
        for i in range(2):
            axes[i, j].set_xlim(box[0], box[1]); axes[i, j].set_ylim(box[2], box[3])
    fig.colorbar(im_map, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01,
                 label="binder saturation s")
    fig.colorbar(im_phi, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01,
                 label="melt fraction")
    v = res["verdict"]
    fig.suptitle(
        f"{shape}  [{v['class']}]   printable single-pass 4-bit solved map against the best of "
        f"{res['n_stored_masks_scanned']} stored historical masks.   "
        f"dJ {v['dJ_rel']*100:+.1f} %,  dIoU {v['dIoU']:+.4f}.   "
        "Cyan is the nominal part, dashed white is the melt front.", fontsize=9.5)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"fig_lib_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# the census figure
# ---------------------------------------------------------------------------

def census_rows(outdir: Path) -> list[dict]:
    rows = []
    for sh in SHAPES:
        f = outdir / f"{sh}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        d, h, u = r["arms"]["A1_4bpp"], r["arms"]["HIST_best"], r["arms"]["U_uniform"]
        rows.append({
            "shape": sh, "iou_solved": d["IoU"], "iou_hist": h["IoU"], "iou_uniform": u["IoU"],
            "J_solved": d["J"], "J_hist": h["J"], "J_uniform": u["J"],
            "dJ": (h["J"] - d["J"]) / max(abs(h["J"]), 1e-30),
            "dIoU": d["IoU"] - h["IoU"], "class": r["verdict"]["class"],
            "hist_at_horizon": h["t_stop_at_horizon"],
            "solved_at_horizon": d["t_stop_at_horizon"],
        })
    return rows


def census_figure(outdir: Path, figdir: Path) -> Path:
    rows = sorted(census_rows(outdir), key=lambda r: r["iou_solved"])
    n = len(rows)
    n_J = sum(1 for r in rows if r["dJ"] > 0)
    n_I = sum(1 for r in rows if r["dIoU"] > 0)
    n_solved = sum(1 for r in rows if r["iou_solved"] >= 0.95)

    fig = plt.figure(figsize=(13.5, 0.52 * n + 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1], sharey=axL)
    ypos = np.arange(n)

    axL.axvspan(0.95, 1.02, color="#d8f3dc", zorder=0)
    axL.axvline(0.95, color="#2d6a4f", lw=1.2, ls="--", zorder=1)
    for i, r in enumerate(rows):
        a, b = r["iou_hist"], r["iou_solved"]
        col = "#1b7f3b" if b > a else ("#b3261e" if b < a else "#666666")
        axL.annotate("", xy=(b, i), xytext=(a, i),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                     shrinkA=0, shrinkB=0, alpha=0.9), zorder=3)
        axL.plot([r["iou_uniform"]], [i], marker="|", ms=11, color="#999999",
                 mew=1.6, zorder=2)
    axL.scatter([r["iou_hist"] for r in rows], ypos, s=42, facecolor="white",
                edgecolor="#333333", zorder=4, label="best stored historical mask")
    axL.scatter([r["iou_solved"] for r in rows], ypos, s=52, color="#0b4f9e",
                zorder=5, label="solved map, 4 bits per pixel, single pass")
    axL.plot([], [], marker="|", ls="none", ms=11, color="#999999",
             label="uniform s = 1")
    axL.set_yticks(ypos)
    axL.set_yticklabels([f"{r['shape']}" for r in rows], fontsize=9)
    axL.set_xlabel("intersection over union of the melted region with the nominal part\n"
                   "(each arm at its own optimal stop, melted = melt fraction >= 0.5)",
                   fontsize=9)
    axL.set_xlim(0.0, 1.03)
    axL.set_ylim(-0.8, n - 0.2)
    axL.grid(axis="x", alpha=0.25)
    axL.legend(loc="upper left", fontsize=8.2, framealpha=0.98)
    axL.set_title("A.  shape fidelity, intersection over union", fontsize=10.5, loc="left")

    # The relative change explodes when the historical J is near zero (rectangle:
    # J_hist = 10.69). The axis is clamped so the readable range is not destroyed
    # by one outlier; every clamped bar keeps its TRUE value in its label and is
    # marked with a break arrow, so nothing is hidden.
    LIM = 100.0
    for i, r in enumerate(rows):
        v = 100.0 * r["dJ"]
        vd = float(np.clip(v, -LIM * 0.97, LIM * 0.97))
        axR.barh(i, vd, color="#1b7f3b" if v > 0 else "#b3261e", alpha=0.85, height=0.62)
        clipped = abs(v) > LIM * 0.97
        if clipped:
            axR.plot([vd], [i], marker=">" if v > 0 else "<", ms=7,
                     color="#000000", clip_on=False)
        txt = f"{v:+.0f}%" + (" off scale" if clipped else "")
        # Always keep the label INSIDE the axes: outside the bar when there is
        # room, inside the bar (white) when the bar runs to the clamp.
        if clipped:
            axR.text(vd - 2.0 * np.sign(v), i, txt, va="center",
                     ha="right" if v > 0 else "left", fontsize=7.6, color="white")
        else:
            axR.text(vd + (2.0 if v >= 0 else -2.0), i, txt, va="center",
                     ha="left" if v >= 0 else "right", fontsize=7.6)
        if r["hist_at_horizon"] or r["solved_at_horizon"]:
            axR.text(0.985, i, "H", transform=axR.get_yaxis_transform(),
                     va="center", ha="right", fontsize=7.5, color="#8a6d00")
    axR.axvline(0, color="#333333", lw=1.0)
    axR.set_xlabel("reduction in the objective J against the best stored historical mask, percent\n"
                   "(positive = the solved printable map is better; axis clamped at 100 %)",
                   fontsize=9)
    axR.grid(axis="x", alpha=0.25)
    axR.tick_params(labelleft=False)
    axR.set_xlim(-LIM, LIM)
    axR.set_title("B.  objective J against the best stored mask    "
                  "(H = a compared arm stopped at the horizon)",
                  fontsize=10.5, loc="left")

    fig.suptitle(
        "Shape-library census: does a solved, printable, single-pass dopant map beat the best "
        f"dopant map the campaign has ever stored?\nVERDICT   J: {n_J} of {n}    "
        f"IoU: {n_I} of {n}    melted region matches the nominal part (IoU >= 0.95): "
        f"{n_solved} of {n}",
        fontsize=12.5, y=0.995)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / "fig_lib_census.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "shape":
        print(shape_figure(sys.argv[2], Path(sys.argv[3]).resolve(),
                           Path(sys.argv[4]).resolve()), flush=True)
    elif mode == "census":
        print(census_figure(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()),
              flush=True)
    else:  # pragma: no cover
        raise SystemExit(f"unknown mode {mode!r}")
