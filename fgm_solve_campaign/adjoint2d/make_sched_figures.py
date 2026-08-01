"""Figures for the temporal power-scheduling campaign.

Four families, one composite per shape plus one census.

  fig_sched_<shape>.png        the deliverable figure. Top row: each arm's
      optimized schedule drawn as GENERATOR INSTRUCTIONS, a step plot of power
      level against time in seconds, with the segment edges visible, the
      schedule window edge marked, and that arm's own J-optimal stop as a
      vertical line. Bottom row: the melt-fraction field at that arm's own stop,
      nominal part outline in cyan, melt front dashed white.

  fig_sched_curves_<shape>.png the objective J and the mean relative density
      rho against time for every arm, with each arm's stop marked. This is where
      the place-then-hold question is read: rho is not in J, so a hold shows up
      as rho climbing while J stays flat.

  fig_sched_maps_<shape>.png   dopant maps, baseline against deliverable.

  fig_sched_census.png         every shape on one axis: intersection over union
      of the library baseline against the scheduled deliverable, and the mean
      relative density at the stop beside it.

Stop convention for every number drawn: t_stop = argmin over that arm's own
trajectory of J. The melted region is melt fraction >= 0.5.

Run:
  ./.venv312/bin/python -m adjoint2d.make_sched_figures shape <shape> <outdir> <figs>
  ./.venv312/bin/python -m adjoint2d.make_sched_figures census <outdir> <figs>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import forward as fwd, schedule as sch, shape_objective as so
from .library_solve import shape_config
from .make_figures import crop_box
from .make_library_figures import _map_panel, _phi_panel
from .pins import build_case, load_cfg
from .sched_solve import HORIZON, SHAPES, WINDOW

DPI = 180

SCHED_PANELS = [
    ("sched", "BASE_map4bpp", "BASELINE\nlibrary 4 bit map, no schedule"),
    ("sched", "CO_4bpp", "cold start co-optimized\n4 bit map + schedule, from s = 1"),
    ("warm", "WARM_sched_only", "schedule added to the baseline map\nmap FIXED, continuous p"),
    ("warm", "WARM_CO_4bpp", "DELIVERABLE\nwarm start co-optimized, 4 bit map + schedule"),
    ("sched", "BIN_round", "binary on/off\ncold start 4 bit map, p in {0, 1}"),
]
ARM_COLORS = {
    "U_uniform": "#8a8a8a", "BASE_map4bpp": "#1f77b4", "SCHED_only": "#2ca02c",
    "CO_cont": "#9467bd", "CO_4bpp": "#d62728", "BIN_relax": "#e377c2",
    "BIN_round": "#ff7f0e", "WARM_base_noshed": "#1f77b4",
    "WARM_sched_only": "#17becf", "WARM_CO_cont": "#8c564b",
    "WARM_CO_4bpp": "#d62728",
}


# ---------------------------------------------------------------------------
# schedule as generator instructions
# ---------------------------------------------------------------------------

def _step_curve(p_seg, window, n_seg, dt_s, t_end_s):
    """Step-plot vertices for a piecewise-constant schedule, including the
    post-window hold at the last level."""
    xs, ys = [], []
    for _k, (lo, hi) in enumerate(sch.segment_bounds(window, n_seg)):
        xs += [lo * dt_s, hi * dt_s]
    lv = [float(v) for v in p_seg]
    for v in lv:
        ys += [v, v]
    if t_end_s > window * dt_s:
        xs += [window * dt_s, t_end_s]
        ys += [lv[-1], lv[-1]]
    return np.asarray(xs), np.asarray(ys)


def _sched_panel(ax, m, res, title):
    dt = res["dt_s"]
    window = res["schedule_window_steps"]
    n_seg = res["n_seg"]
    t_end = res["horizon_steps"] * dt
    p = m["p_seg"]
    if p is None:
        p = [1.0] * n_seg
    xs, ys = _step_curve(p, window, n_seg, dt, t_end)
    ax.fill_between(xs, 0.0, ys, color="#d62728", alpha=0.18, lw=0)
    ax.plot(xs, ys, color="#d62728", lw=1.5)
    for lo, _hi in sch.segment_bounds(window, n_seg):
        ax.axvline(lo * dt, color="#dddddd", lw=0.4, zorder=0)
    ax.axvline(window * dt, color="#444444", lw=1.0, ls=":")
    ax.axvline(m["t_stop_s"], color="#000000", lw=1.6)
    late = m["t_stop_s"] > 0.62 * t_end
    ax.annotate(f"stop {m['t_stop_s']:.0f} s", xy=(m["t_stop_s"], 1.52),
                xytext=(-3 if late else 3, 0), textcoords="offset points",
                fontsize=6.4, va="top", ha="right" if late else "left")
    ax.set_xlim(0.0, t_end)
    ax.set_ylim(0.0, 1.62)
    ax.set_xlabel("time, s", fontsize=7, labelpad=1.5)
    ax.set_ylabel("power level p", fontsize=7, labelpad=1.5)
    ax.tick_params(labelsize=6.2, pad=1.5)
    st = m["structure"]["structure"].lower().replace("_", " ")
    ax.set_title(f"{title}\nduty {m['duty_cycle']:.2f}, {m['n_switches']} level "
                 f"changes, {st}", fontsize=7.0, linespacing=1.3, pad=4)
    return ax


# ---------------------------------------------------------------------------
# per-shape composites
# ---------------------------------------------------------------------------

def _melt_at_stop(case, s, p, n_seg, horizon, window, stop_index):
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=None,
                     n_steps=horizon, p_seg=p, n_seg=n_seg, p_horizon=window)
    phi, _ = so.phi_field(tr.T_at_end(stop_index), case)
    return phi, tr


def _load(outdir: Path, shape: str):
    docs, maps = {}, {}
    for stem in ("sched", "warm"):
        f = outdir / f"{shape}_{stem}.json"
        if f.exists():
            docs[stem] = json.loads(f.read_text())
            maps[stem] = np.load(outdir / f"{shape}_{stem}_maps.npz")
    return docs, maps


def shape_figure(shape: str, outdir: Path, figdir: Path) -> list[Path]:
    docs, maps = _load(outdir, shape)
    res = docs["sched"]
    case = build_case(load_cfg(shape_config(shape)))
    pm = case.part_mask
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    n_seg, horizon = res["n_seg"], res["horizon_steps"]
    window = res["schedule_window_steps"]
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    cols = [(st, k, lab) for st, k, lab in SCHED_PANELS
            if st in docs and k in docs[st]["arms"]]
    nc = len(cols)
    fig = plt.figure(figsize=(2.72 * nc + 0.9, 6.35))
    gs = fig.add_gridspec(2, nc + 1, height_ratios=[1.0, 1.28],
                          width_ratios=[1.0] * nc + [0.055],
                          left=0.052, right=0.965, top=0.825, bottom=0.062,
                          hspace=0.26, wspace=0.34)
    im_phi = None
    for j, (st, key, label) in enumerate(cols):
        d = docs[st]
        m = d["arms"][key]
        _sched_panel(fig.add_subplot(gs[0, j]), m, d, label)
        ax = fig.add_subplot(gs[1, j])
        p = None if m["p_seg"] is None else np.asarray(m["p_seg"], dtype=float)
        phi, _tr = _melt_at_stop(case, np.asarray(maps[st][f"map_{key}"], dtype=float),
                                 p, n_seg, horizon, window, m["t_stop_index"])
        im_phi = _phi_panel(
            ax, phi, pm, extent,
            f"IoU {m['IoU']:.3f}   mean rho {m['mean_rho_part_at_stop']:.3f}\n"
            f"part unmelted {m['part_under_melt_pct']:.1f}%, "
            f"bed growth {m['bed_melt_pct_of_part']:.1f}%")
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    cax = fig.add_subplot(gs[1, nc])
    fig.colorbar(im_phi, cax=cax, label="melt fraction")
    cax.tick_params(labelsize=6.5)

    w = docs["warm"]["verdict"] if "warm" in docs else res["verdict"]
    fig.suptitle(
        f"{shape}: temporal power scheduling on a fixed geometry. Cyan is the nominal "
        f"part, dashed white the melt front at that arm's own stop.\n"
        f"Warm start baseline IoU {w['baseline_IoU']:.3f} -> deliverable "
        f"{w['deliverable_IoU']:.3f} ({w['dIoU_vs_baseline']:+.3f}), "
        f"J {w['dJ_rel_vs_baseline']*100:+.1f} percent.  "
        f"Horizon {horizon} steps = {res['horizon_s']:.0f} s; schedule window "
        f"{window} steps = {res['schedule_window_s']:.0f} s over {n_seg} segments, "
        f"level held after the window (dotted line).\n"
        f"Stop convention: t_stop = argmin of J on each arm's own trajectory, "
        f"shape-fidelity early stop disabled on every arm.",
        fontsize=8.0, y=0.985, linespacing=1.55)
    q = figdir / f"fig_sched_{shape}.png"
    fig.savefig(q, dpi=DPI); plt.close(fig); written.append(q)

    # --- J and rho curves ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    t = np.arange(horizon) * res["dt_s"]
    seen = set()
    for st in ("sched", "warm"):
        if st not in docs:
            continue
        for key, m in docs[st]["arms"].items():
            if key in seen or key in ("CO_cont", "BIN_relax", "WARM_CO_cont"):
                continue
            seen.add(key)
            c = ARM_COLORS.get(key, "#333333")
            ls = "--" if st == "warm" else "-"
            jc = np.asarray(m["J_curve"]); rc = np.asarray(m["rho_curve"])
            axes[0].plot(t[:jc.size], jc, color=c, lw=1.2, ls=ls, label=key)
            axes[0].plot(m["t_stop_s"], m["J"], "o", color=c, ms=4.5)
            axes[1].plot(t[:rc.size], rc, color=c, lw=1.2, ls=ls, label=key)
            axes[1].plot(m["t_stop_s"], m["mean_rho_part_at_stop"], "o", color=c, ms=4.5)
    axes[0].set_ylabel("J, whole-domain sum of (melt fraction minus part indicator) squared",
                       fontsize=7.0)
    axes[1].set_ylabel("mean relative density in the part", fontsize=8)
    for a in axes:
        a.set_xlabel("time, s", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, lw=0.4)
        a.axvline(window * res["dt_s"], color="#444444", lw=0.9, ls=":")
    axes[1].axhline(1.0, color="#888888", lw=0.7, ls="--")
    axes[0].legend(fontsize=6.2, ncol=2)
    fig.suptitle(f"{shape}: objective and densification against time. Filled circles mark "
                 f"each arm's own J-optimal stop; dashed lines are warm-start arms.\n"
                 f"Density is NOT in the objective, so a place-then-hold appears as "
                 f"density climbing while J stays flat. Dotted vertical line is the "
                 f"schedule window edge.", fontsize=8.2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    q = figdir / f"fig_sched_curves_{shape}.png"
    fig.savefig(q, dpi=DPI); plt.close(fig); written.append(q)

    # --- dopant maps --------------------------------------------------------
    keys = [("sched", "BASE_map4bpp"), ("sched", "CO_4bpp"), ("warm", "WARM_CO_4bpp")]
    keys = [(s, k) for s, k in keys if s in docs and f"map_{k}" in maps[s].files]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.15 * len(keys) + 0.7, 3.7))
    axes = np.atleast_1d(axes)
    im = None
    for j, (st, k) in enumerate(keys):
        m = docs[st]["arms"][k]
        im = _map_panel(axes[j], np.asarray(maps[st][f"map_{k}"], dtype=float), pm,
                        extent, 1.0,
                        f"{k}\nmean s {m['sat_mean_in_part']:.3f}, "
                        f"max s {m['sat_max_in_part']:.3f}")
        axes[j].set_xlim(box[0], box[1]); axes[j].set_ylim(box[2], box[3])
    fig.colorbar(im, ax=axes.tolist(), fraction=0.032, pad=0.015,
                 label="dopant fraction s")
    fig.suptitle(f"{shape}: dopant maps. The geometry is FIXED and the dopant is graded, "
                 f"so this is the functionally graded material lever, not geometry "
                 f"pre-warp.", fontsize=8.4)
    q = figdir / f"fig_sched_maps_{shape}.png"
    fig.savefig(q, dpi=DPI, bbox_inches="tight"); plt.close(fig); written.append(q)
    return written


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def census_figure(outdir: Path, figdir: Path) -> Path:
    rows = []
    for shape in SHAPES:
        f = outdir / f"{shape}_warm.json"
        g = outdir / f"{shape}_sched.json"
        if f.exists() and g.exists():
            r = json.loads(f.read_text())
            r["cold"] = json.loads(g.read_text())["verdict"]
            rows.append(r)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    names = [r["shape"] for r in rows]
    yy = np.arange(len(rows))
    for i, r in enumerate(rows):
        v = r["verdict"]
        axes[0].plot([v["baseline_IoU"], v["deliverable_IoU"]], [i, i],
                     color="#bbbbbb", lw=1.2, zorder=0)
        axes[0].plot(v["baseline_IoU"], i, "o", color="#1f77b4", ms=6)
        axes[0].plot(v["deliverable_IoU"], i, "o", color="#d62728", ms=6)
        axes[0].text(max(v["baseline_IoU"], v["deliverable_IoU"]) + 0.006, i,
                     f" {v['dIoU_vs_baseline']:+.3f}", fontsize=7, va="center")
        axes[1].barh(i - 0.18, v["mean_rho_by_arm"]["WARM_base_noshed"], height=0.34,
                     color="#1f77b4")
        axes[1].barh(i + 0.18, v["mean_rho_by_arm"]["WARM_CO_4bpp"], height=0.34,
                     color="#d62728")
        hold = r["arms"]["WARM_CO_4bpp"]["iso_J_hold"]
        axes[2].barh(i, hold["d_rho"], height=0.5, color="#2ca02c")
        axes[2].text(hold["d_rho"], i, f"  +{hold['extra_steps']} steps",
                     fontsize=7, va="center")
    for a in axes:
        a.set_yticks(yy); a.set_yticklabels(names, fontsize=8)
        a.grid(axis="x", alpha=0.25, lw=0.4)
        a.tick_params(labelsize=7)
    axes[0].margins(x=0.16)
    axes[0].set_xlabel("intersection over union at each arm's own stop", fontsize=8)
    axes[0].set_title("blue: library 4 bit baseline, no schedule\n"
                      "red: warm-start co-optimized 4 bit map + schedule",
                      fontsize=8, linespacing=1.4)
    axes[1].set_xlabel("mean relative density in the part at the stop", fontsize=8)
    axes[1].set_title("densification at the shape-optimal stop", fontsize=8)
    axes[2].set_xlabel("extra mean relative density from holding at iso J", fontsize=8)
    axes[2].set_title("place-then-hold headroom, deliverable arm\n"
                      "J allowed to rise 2 percent", fontsize=8, linespacing=1.4)
    fig.suptitle("Temporal power scheduling census. Every number at that arm's own "
                 "J-optimal stop, energy-residual gate on.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    q = figdir / "fig_sched_census.png"
    fig.savefig(q, dpi=DPI); plt.close(fig)
    return q


if __name__ == "__main__":
    if sys.argv[1] == "shape":
        for _p in shape_figure(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])):
            print(_p)
    else:
        print(census_figure(Path(sys.argv[2]), Path(sys.argv[3])))
