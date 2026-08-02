#!/usr/bin/env python3
"""Figures for the SEQUENTIAL dwell pass.

The composite must show the MECHANISM, not only the score: that the limb melted
in phase one is still melted while phase two heats the other limb. That is why
the middle row is a time series of melt fields with the two limb outlines drawn
on it, and not another pair of end-state thumbnails.

Run:
  ./.venv312/bin/python scripts/analysis/make_seq_figures.py [shape ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from matplotlib.patches import Rectangle                 # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "fgm_solve_campaign/out_seq"
FIGS = REPO / "fgm_solve_campaign/figs_seq"
SHAPES = ("L_shape", "T_shape")
DPI = 180
POS_COLORS = plt.cm.twilight(np.linspace(0.08, 0.92, 8))

ARM_ORDER = ("S_static_best_uniform", "S_cycled_equal_uniform",
             "S_prior_best_on_record", "S_seq_uniform_two_refined_interior",
             "S_seq_cosolved_4bpp_interior_switch")
ARM_LABEL = {
    "S_static_best_uniform": "static best angle,\nuniform map",
    "S_cycled_equal_uniform": "CYCLED equal dwell,\nuniform map",
    "S_prior_best_on_record": "PREVIOUS BEST on record\n(cycled dwell, solved map)",
    "S_seq_uniform_two_refined_interior": "SEQUENTIAL,\nUNIFORM map",
    "S_seq_cosolved_4bpp_interior_switch": "SEQUENTIAL,\nco-solved map 4 bpp",
}


def _crop(pm, pad=10):
    r = np.flatnonzero(pm.any(axis=1))
    c = np.flatnonzero(pm.any(axis=0))
    return (max(int(r[0]) - pad, 0), min(int(r[-1]) + pad + 1, pm.shape[0]),
            max(int(c[0]) - pad, 0), min(int(c[-1]) + pad + 1, pm.shape[1]))


def _melt_panel(ax, phi, pm, x, y, title, limbs=None, fs=8):
    r0, r1, c0, c1 = _crop(pm)
    ph = np.asarray(phi)[r0:r1, c0:c1]
    m = np.asarray(pm)[r0:r1, c0:c1]
    ext = [x[c0] * 1e3, x[c1 - 1] * 1e3, y[r0] * 1e3, y[r1 - 1] * 1e3]
    im = ax.imshow(ph, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1,
                   interpolation="bilinear")
    ax.contour(np.asarray(m, dtype=float), levels=[0.5], colors="cyan",
               linewidths=1.1, extent=ext, origin="lower")
    ax.contour(ph, levels=[0.5], colors="white", linewidths=0.9, linestyles="--",
               extent=ext, origin="lower")
    if limbs is not None:
        for lm, col in zip(limbs, ("#39ff14", "#ff45c8")):
            ax.contour(np.asarray(lm[r0:r1, c0:c1], dtype=float), levels=[0.5],
                       colors=col, linewidths=0.8, extent=ext, origin="lower")
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _timeline(ax, segments_deg, durations_s, horizon, stop_s, switch_marks=True):
    t = 0.0
    for j, (a, d) in enumerate(zip(segments_deg, durations_s)):
        d = min(float(d), horizon - t)
        if j == len(segments_deg) - 1:
            d = horizon - t
        if d <= 0:
            break
        ax.add_patch(Rectangle((t, 0.0), d, 1.0,
                               facecolor=POS_COLORS[(j * 3) % 8],
                               edgecolor="white", lw=0.6))
        ax.text(t + 0.5 * d, 0.5, f"{a:.0f} deg", ha="center", va="center",
                fontsize=8, color="w", weight="bold")
        t += d
        if switch_marks and j < len(segments_deg) - 1:
            ax.axvline(t, color="k", lw=1.0, ls=":")
    if stop_s is not None:
        ax.axvline(stop_s, color="k", lw=1.6)
        ax.text(stop_s, 0.06, f" stop {stop_s:.0f} s", fontsize=8, va="bottom",
                ha="left", color="w", weight="bold")
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time from exposure start (s)", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7, pad=1)


def figure_for(shape: str) -> Path | None:
    js, npz = OUT / f"{shape}_arms.json", OUT / f"{shape}_seq_maps.npz"
    if not (js.exists() and npz.exists()):
        print(f"  {shape}: no data, skipped")
        return None
    r = json.loads(js.read_text())
    d = np.load(npz)
    arms = r["arms"]
    pm = np.asarray(d["part_mask"], dtype=bool)
    x, y = d["x"], d["y"]
    limbs = (np.asarray(d["wide_limb"], dtype=bool),
             np.asarray(d["narrow_limb"], dtype=bool))
    snaps = r["snapshots"]
    present = [a for a in ARM_ORDER if a in arms]
    win = r.get("deliverable_arm", present[-1])
    a_win = arms[win]

    n_snap = len(snaps["seq"]["steps"])
    fig = plt.figure(figsize=(2.9 * max(n_snap, len(present)) + 1.0, 10.2))
    gs = fig.add_gridspec(4, max(n_snap, len(present)),
                          height_ratios=[0.42, 1.0, 1.0, 1.0],
                          hspace=0.62, wspace=0.08)

    # (a) the schedule as a timeline
    ax = fig.add_subplot(gs[0, :])
    _timeline(ax, a_win["segments_deg"], a_win["durations_s"],
              r["n_steps"] * r["dt_s"], a_win["t_stop_s"])
    sw = snaps["seq"]["switch_s"]
    for t_s in snaps["seq"]["times_s"]:
        ax.plot([t_s], [1.02], marker="v", color="k", ms=5, clip_on=False)
    ax.set_title(
        f"(a) {shape}: the SEQUENTIAL turntable program. Hold "
        f"{a_win['segments_deg'][0]:.0f} degrees for {sw:.0f} s, then move to "
        f"{a_win['segments_deg'][1]:.0f} degrees and hold. Triangles mark the "
        f"melt snapshots below.", fontsize=9, pad=14)

    # (b) the mechanism: melt through phase 1 and phase 2
    for k, (i_step, t_s) in enumerate(zip(snaps["seq"]["steps"],
                                          snaps["seq"]["times_s"])):
        ax = fig.add_subplot(gs[1, k])
        phase = 1 if t_s <= sw + 1e-9 else 2
        _melt_panel(ax, d[f"snap_seq_{i_step}"], pm, x, y,
                    f"t = {t_s:.0f} s, phase {phase}\n"
                    f"({a_win['segments_deg'][phase - 1]:.0f} degrees)", limbs)
    fig.text(0.012, 0.63, "(b) SEQUENTIAL", fontsize=10, rotation=90,
             va="center", weight="bold")

    # (c) the same times on the static baseline, so the comparison is in time
    for k, (i_step, t_s) in enumerate(zip(snaps["static"]["steps"],
                                          snaps["static"]["times_s"])):
        ax = fig.add_subplot(gs[2, k])
        past = t_s > snaps["static"]["stop_s"] + 1e-9
        _melt_panel(ax, d[f"snap_static_{i_step}"], pm, x, y,
                    f"t = {t_s:.0f} s ({snaps['static']['segments_deg'][0]:.0f} "
                    f"degrees, held)\n"
                    + ("past its own stop" if past else "before its own stop"),
                    limbs)
    fig.text(0.012, 0.40, "(c) STATIC best angle,\nsame clock", fontsize=10,
             rotation=90, va="center", weight="bold")

    # (d) melt against nominal at each arm's own stop
    for k, name in enumerate(present):
        ax = fig.add_subplot(gs[3, k])
        a = arms[name]
        im = _melt_panel(
            ax, d[f"phi_{name}"], pm, x, y,
            f"{ARM_LABEL[name]}\nJ {a['J']:.1f}   IoU {a['IoU']:.4f}\n"
            f"grow {a['bed_melt_pct_of_part']:.1f}%   under "
            f"{a['part_under_melt_pct']:.1f}%\nstop {a['t_stop_s']:.0f} s   peak "
            f"{a['max_T_upto_stop_c']:.0f} C"
            + ("  OVER CEILING" if a["over_ceiling_250c"] else ""), limbs, fs=8)
    fig.text(0.012, 0.15, "(d) each arm at its own stop", fontsize=10,
             rotation=90, va="center", weight="bold")
    cax = fig.add_axes([0.93, 0.06, 0.011, 0.15])
    fig.colorbar(im, cax=cax).set_label("melt fraction phi", fontsize=8)
    cax.tick_params(labelsize=7)

    fig.suptitle(
        f"{shape}, grid 120: sequential hold scheduling. Cyan is the nominal "
        f"part, dashed white the melted region (phi >= 0.5); green and magenta "
        f"outline the wide and narrow limbs.", fontsize=10, y=0.995)
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / f"fig_seq_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def figure_curves(shapes) -> Path | None:
    """The objective against time, which is where the two-phase story is legible."""
    got = [s for s in shapes if (OUT / f"{s}_arms.json").exists()]
    if not got:
        return None
    fig, axes = plt.subplots(1, len(got), figsize=(6.6 * len(got), 4.4),
                             squeeze=False)
    for ax, shape in zip(axes[0], got):
        r = json.loads((OUT / f"{shape}_arms.json").read_text())
        d = np.load(OUT / f"{shape}_seq_maps.npz")
        arms = r["arms"]
        dt = r["dt_s"]
        for name, col in zip(ARM_ORDER, ("0.55", "0.25", "tab:orange",
                                         "tab:blue", "tab:red")):
            if name not in arms or f"J_curve_{name}" not in d:
                continue
            c = np.asarray(d[f"J_curve_{name}"], dtype=float)
            t = (np.arange(c.size) + 1) * dt
            ax.plot(t, c, color=col, lw=1.5,
                    label=ARM_LABEL[name].replace("\n", " "))
            i = int(arms[name]["t_stop_index"])
            ax.plot([t[i]], [c[i]], marker="o", color=col, ms=5)
        a_win = arms.get(r.get("deliverable_arm"))
        if a_win:
            sw = float(np.cumsum(a_win["durations_s"])[0])
            ax.axvline(sw, color="k", ls=":", lw=1.2)
            ax.text(sw, ax.get_ylim()[1], " switch", fontsize=8, va="top")
        ax.set_xlabel("time (s)", fontsize=9)
        ax.set_ylabel("J_phi (whole domain)", fontsize=9)
        ax.set_title(f"{shape}: objective against time, each arm's own stop "
                     f"marked", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / "fig_seq_curves.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def figure_screen(shape: str) -> Path | None:
    """What the switch time buys, and what it costs in bed growth."""
    rows = []
    for name in (f"{shape}_screen.json", f"{shape}_screen2.json"):
        p = OUT / name
        if p.exists():
            j = json.loads(p.read_text())
            rows += j.get("rows", []) + j.get("two_segment", [])
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    pairs = sorted({(r["a1_deg"], r["a2_deg"]) for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(pairs), 2)))
    for (a1, a2), col in zip(pairs, cmap):
        sel = sorted([r for r in rows if r["a1_deg"] == a1 and r["a2_deg"] == a2],
                     key=lambda r: r["switch_s"])
        t = [r["switch_s"] for r in sel]
        lab = f"{a1:.0f} -> {a2:.0f} deg"
        axes[0].plot(t, [r["J"] for r in sel], "o-", color=col, label=lab, ms=4)
        axes[1].plot(t, [r["IoU"] for r in sel], "o-", color=col, ms=4)
        axes[2].plot(t, [r["bed_melt_pct_of_part"] for r in sel], "o-",
                     color=col, ms=4)
        axes[2].plot(t, [r["part_under_melt_pct"] for r in sel], "s--",
                     color=col, ms=4, alpha=0.6)
    for ax, lab in zip(axes, ("J_phi at the arm's own stop", "IoU at the stop",
                              "growth (circles) and under-melt (squares), % of part")):
        ax.set_xlabel("switch time (s)", fontsize=9)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{shape}: uniform dopant map, two-segment sequential "
                      f"schedule", fontsize=10)
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / f"fig_seq_screen_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def main(shapes) -> None:
    for s in shapes:
        figure_for(s)
        figure_screen(s)
    figure_curves(shapes)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SHAPES))
