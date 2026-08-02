#!/usr/bin/env python3
"""Figures for the asymmetric dwell-schedule pass.

One composite per shape, which must answer the two questions the pass exists to
answer at a glance: WHERE does the turntable park and for how long, and does
the melted region look more like the nominal part than the equal-dwell control
does. Plus one census across the shapes.

Run:
  ./.venv312/bin/python scripts/analysis/make_dwell_figures.py [shape ...]
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
OUT = REPO / "fgm_solve_campaign/out_dwell"
FIGS = REPO / "fgm_solve_campaign/figs_dwell"
SHAPES = ("cross", "square", "T_shape", "L_shape")
DPI = 180
POS_COLORS = plt.cm.twilight(np.linspace(0.05, 0.95, 8))


def best_arms(r: dict, d) -> dict:
    """Pick the deliverable and the equal-dwell control by their own J.

    Both classes are chosen the same way, from the 4-bits-per-pixel arms that
    have a stored map, so the comparison cannot be won by quoting a deeper
    solve on one side only.
    """
    arms = r["arms"]

    def pick(pred):
        c = [(v["J"], k) for k, v in arms.items()
             if "4bpp" in k and f"sat_{k}" in d and pred(k)]
        return min(c)[1] if c else None

    deliv = pick(lambda k: "equal" not in k)
    ctrl = pick(lambda k: "equal" in k)
    def tr_of(k):
        if k == "D_program_4bpp":
            return "D_timeresolved"
        if k == "D_map_equal_4bpp":
            return "D_timeresolved_equal"
        c = k.replace("_4bpp_", "_timeresolved_")
        return c if c in arms else k

    tr_d, tr_c = tr_of(deliv), tr_of(ctrl)
    return {"deliverable": deliv, "control": ctrl,
            "deliverable_tr": tr_d if tr_d in arms else deliv,
            "control_tr": tr_c if tr_c in arms else ctrl}


def fold_to_distinct(w: np.ndarray) -> np.ndarray:
    """Fold the 8 candidate positions onto the 4 DISTINCT heating patterns.

    MEASURED (`DWELL_SCHEDULE_REPORT.md` Section on the redundancy): the
    part-frame heating at theta and at theta + 180 degrees is the same field to
    4e-13 relative, because the parallel-plate drive is invariant under a half
    turn of the whole system. Reporting the raw eight-vector would therefore
    show an asymmetry that carries no physics.
    """
    w = np.asarray(w, dtype=float).ravel()
    return w[:4] + w[4:] if w.size == 8 else w


def _crop(pm, pad=14):
    r = np.flatnonzero(pm.any(axis=1))
    c = np.flatnonzero(pm.any(axis=0))
    return (max(int(r[0]) - pad, 0), min(int(r[-1]) + pad + 1, pm.shape[0]),
            max(int(c[0]) - pad, 0), min(int(c[-1]) + pad + 1, pm.shape[1]))


def _melt_panel(ax, phi, pm, x, y, title):
    r0, r1, c0, c1 = _crop(pm)
    phi = np.asarray(phi)[r0:r1, c0:c1]
    pm = np.asarray(pm)[r0:r1, c0:c1]
    ext = [x[c0] * 1e3, x[c1 - 1] * 1e3, y[r0] * 1e3, y[r1 - 1] * 1e3]
    ax.imshow(phi, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1,
              interpolation="bilinear")
    ax.contour(np.asarray(pm, dtype=float), levels=[0.5], colors="cyan",
               linewidths=1.1, extent=ext, origin="lower")
    ax.contour(phi, levels=[0.5], colors="white", linewidths=0.9,
               linestyles="--", extent=ext, origin="lower")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _timeline(ax, prog, t_max, stop_s):
    """The turntable program as a timeline bar: one coloured band per hold."""
    angles = list(prog["positions_deg"])
    shown = set()
    for m in prog["moves"]:
        if m["move_at_s"] > t_max:
            break
        j = int(np.argmin(np.abs(np.asarray(angles) - m["position_deg"])))
        ax.add_patch(Rectangle((m["move_at_s"], 0.0),
                               min(m["dwell_s"], t_max - m["move_at_s"]), 1.0,
                               facecolor=POS_COLORS[::2][j % 4], edgecolor="white",
                               lw=0.4))
        if j not in shown and m["dwell_s"] >= 0.06 * t_max:
            shown.add(j)
            ax.text(m["move_at_s"] + 0.5 * m["dwell_s"], 0.5,
                    f"{m['position_deg']:.0f}", ha="center", va="center",
                    fontsize=7, color="w", weight="bold")
    if stop_s is not None and stop_s <= t_max:
        ax.axvline(stop_s, color="k", lw=1.4)
        ax.text(stop_s, 1.06, f" stop {stop_s:.0f} s", fontsize=7, va="bottom")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=7)


def figure_for(shape: str) -> Path | None:
    js = OUT / f"{shape}_dwell.json"
    npz = OUT / f"{shape}_dwell_maps.npz"
    if not (js.exists() and npz.exists()):
        print(f"  {shape}: no data, skipped")
        return None
    r = json.loads(js.read_text())
    d = np.load(npz)
    pm = np.asarray(d["part_mask"], dtype=bool)
    x, y = d["x"], d["y"]
    arms = r["arms"]
    ang = np.asarray(r["candidate_angles_deg"], dtype=float)
    sel = best_arms(r, d)
    prog_full = json.loads(
        (OUT / f"{shape}_turntable_deliverable.json").read_text())
    prog = dict(prog_full["reduced_program"])
    prog["total_exposure_s"] = prog_full["total_exposure_s"]
    w_del = np.asarray(arms[sel["deliverable"]]["dwell_weights"], dtype=float)
    ang4 = ang[:4]
    w4 = fold_to_distinct(w_del)
    k = len(ang4)

    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.22)

    # (a) dwell fractions
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(k), w4, color=POS_COLORS[::2][:k], edgecolor="k", lw=0.5)
    ax.axhline(1.0 / k, color="crimson", ls="--", lw=1.2,
               label=f"equal dwell = {1.0 / k:.3f}")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"{a:.0f}\n(+{a + 180:.0f})" for a in ang4], fontsize=7)
    ax.set_xlabel("distinct turntable position (degrees)", fontsize=8)
    ax.set_ylabel("dwell fraction", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    asym = float(np.max(w4)) * k
    ax.set_title(f"(a) executed dwell, 4 distinct positions\nmax/equal "
                 f"{asym:.2f}x", fontsize=8)
    ax.text(0.02, 0.90, "theta and theta+180 heat identically (4e-13),\nso the "
            "eight candidates carry four patterns", transform=ax.transAxes,
            fontsize=6, va="top")

    # (b) timeline
    ax = fig.add_subplot(gs[0, 1])
    t_show = min(6.0 * float(prog["cycle_time_s"]), float(prog["total_exposure_s"]))
    _timeline(ax, prog, t_show, None)
    ax.set_title(f"(b) turntable program (reduced), first {t_show:.0f} s of "
                 f"{prog['total_exposure_s']:.0f} s\ncycle "
                 f"{prog['cycle_time_s']:.0f} s, {prog['n_moves']} moves; "
                 f"positions " + ", ".join(f"{a:.0f}" for a in prog["positions_deg"])
                 + " degrees", fontsize=8)

    # (c) objective against time
    ax = fig.add_subplot(gs[0, 2])
    for key, lab, col, ls in (
            (f"J_curve_{sel['control']}", "equal dwell, solved map", "0.35", "-"),
            (f"J_curve_{sel['deliverable']}", "dwell schedule (quasi-static)",
             "tab:blue", "--"),
            (f"J_curve_{sel['deliverable_tr']}", "dwell schedule (time resolved)",
             "tab:red", "-")):
        if key not in d:
            continue
        jc = np.asarray(d[key], dtype=float)
        t = (np.arange(jc.size) + 1) * r["dt_s"]
        ax.plot(t, jc, color=col, lw=1.3, ls=ls, label=lab)
        i = int(np.argmin(jc))
        ax.plot(t[i], jc[i], "o", color=col, ms=4)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel(r"$J_\phi$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.set_title("(c) objective and each arm's own stop", fontsize=8)

    # (d) dopant map
    ax = fig.add_subplot(gs[1, 0])
    sat = np.asarray(d[f"sat_{sel['deliverable']}"], dtype=float)
    ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    _r0, _r1, _c0, _c1 = _crop(pm)
    ext = [x[_c0] * 1e3, x[_c1 - 1] * 1e3, y[_r0] * 1e3, y[_r1 - 1] * 1e3]
    im = ax.imshow(np.where(pm, sat, np.nan)[_r0:_r1, _c0:_c1], origin="lower",
                   extent=ext,
                   cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.contour(pm.astype(float)[_r0:_r1, _c0:_c1], levels=[0.5], colors="k",
               linewidths=0.8, extent=ext, origin="lower")
    ax.set_title(f"(d) co-solved dopant map, 4 bits per pixel\n"
                 f"arm {sel['deliverable']}", fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    # (e) equal-dwell control melt, (f) dwell deliverable melt
    a_eq = arms[sel["control_tr"]]
    a_dw = arms[sel["deliverable_tr"]]
    ax = fig.add_subplot(gs[1, 1])
    _melt_panel(ax, np.asarray(d[f"phi_{sel['control_tr']}"]), pm, x, y,
                f"(e) EQUAL dwell control, melt at its stop\n"
                f"J {a_eq['J']:.1f}  IoU {a_eq['IoU']:.4f}  "
                f"stop {a_eq['t_stop_s']:.0f} s")
    ax = fig.add_subplot(gs[1, 2])
    _melt_panel(ax, np.asarray(d[f"phi_{sel['deliverable_tr']}"]), pm, x, y,
                f"(f) ASYMMETRIC dwell, melt at its stop\n"
                f"J {a_dw['J']:.1f}  IoU {a_dw['IoU']:.4f}  "
                f"stop {a_dw['t_stop_s']:.0f} s")

    fig.suptitle(
        f"{shape}: asymmetric dwell scheduling of the turntable, grid 120, "
        f"conductivity channel, constant radio-frequency power\n"
        f"cyan outline = nominal part, dashed white = melt front "
        f"(melt fraction 0.5); every metric read at that arm's own "
        f"argmin-of-$J_\\phi$ stop", fontsize=9.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / f"fig_dwell_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def census() -> Path | None:
    have = [s for s in SHAPES if (OUT / f"{s}_dwell.json").exists()]
    if not have:
        return None
    fig, axes = plt.subplots(2, len(have), figsize=(3.3 * len(have), 6.0))
    axes = np.atleast_2d(axes)
    for c, shape in enumerate(have):
        r = json.loads((OUT / f"{shape}_dwell.json").read_text())
        d = np.load(OUT / f"{shape}_dwell_maps.npz")
        arms = r["arms"]
        sel = best_arms(r, d)
        ang = np.asarray(r["candidate_angles_deg"], dtype=float)[:4]
        w = fold_to_distinct(np.asarray(arms[sel["deliverable"]]["dwell_weights"], float))
        k = len(ang)
        ax = axes[0, c]
        ax.bar(np.arange(k), w, color=POS_COLORS[::2][:k], edgecolor="k", lw=0.5)
        ax.axhline(1.0 / k, color="crimson", ls="--", lw=1.0)
        ax.set_xticks(np.arange(k))
        ax.set_xticklabels([f"{a:.0f}" for a in ang], fontsize=6)
        ax.set_ylim(0, 1.08)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{shape}\nmax/equal {float(w.max()) * k:.2f}x", fontsize=8)
        if c == 0:
            ax.set_ylabel("dwell fraction", fontsize=8)

        ax = axes[1, c]
        names = ["D_uniform_equal", sel["control"], sel["deliverable"],
                 sel["deliverable_tr"]]
        labs = ["uniform map\nequal dwell", "solved map\nequal dwell",
                "dwell schedule\n(quasi-static)", "dwell schedule\n(time resolved)"]
        vals = [arms[n]["IoU"] for n in names if n in arms]
        labs = [l for n, l in zip(names, labs) if n in arms]
        cols = ["0.7", "0.45", "tab:blue", "tab:red"][:len(vals)]
        ax.bar(np.arange(len(vals)), vals, color=cols, edgecolor="k", lw=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labs, fontsize=6)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.95, color="green", ls=":", lw=1.0)
        ax.tick_params(labelsize=6)
        if c == 0:
            ax.set_ylabel("intersection over union at own stop", fontsize=8)
    fig.suptitle("Asymmetric dwell scheduling: where the turntable parks (top) and "
                 "what it buys (bottom)\ngrid 120, conductivity channel, constant "
                 "radio-frequency power; green dotted line is the 0.95 SOLVED class",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / "fig_dwell_census.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def cycle_figure() -> Path | None:
    have = [s for s in SHAPES if (OUT / f"{s}_dwell.json").exists()]
    if not have:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for shape in have:
        r = json.loads((OUT / f"{shape}_dwell.json").read_text())
        rows = r.get("cycle_sweep", [])
        if not rows:
            continue
        c = [x["cycle_time_s"] for x in rows]
        e = [x["quasi_static_error_pct"] for x in rows]
        ax.plot(c, e, "o-", lw=1.3, ms=4, label=shape)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("cycle time (s), one pass through the kept positions", fontsize=9)
    ax.set_ylabel(r"time-resolved $J_\phi$ against the quasi-static value (%)",
                  fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.set_title("The quasi-static approximation error, measured\n"
                 "(part frame, no interpolation anywhere)", fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / "fig_dwell_cycle_time.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


if __name__ == "__main__":
    todo = sys.argv[1:] or list(SHAPES)
    for s in todo:
        figure_for(s)
    census()
    cycle_figure()
