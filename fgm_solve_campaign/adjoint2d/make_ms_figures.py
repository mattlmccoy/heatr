"""Figures for the multi-start pass. Four, each designed to say one thing.

  fig_ms_census      the four-arm comparison per shape and where each arm wins
  fig_ms_starts      which start won, and the depth-against-breadth mechanism
  fig_ms_maps        the delivered 4-bits-per-pixel maps, all 18
  fig_ms_robust      the rim and grid probes, filtered against unfiltered
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_MS = ROOT / "out_ms"
OUT_LIB = ROOT / "out_lib"
OUT_ROBUST = ROOT / "out_robust"
FIGS = ROOT / "figs_ms"
DPI = 180
SHAPES = ("square", "circle", "hexagon", "triangle", "equilateral_triangle",
          "L_shape", "H_shape", "T_shape", "cross", "diamond", "ellipse",
          "octagon", "pentagon", "rectangle", "rounded_rect", "star", "star6",
          "trapezoid")
START_COLOR = {"cold": "#4C72B0", "warm": "#DD8452", "prev": "#55A868",
               "pert": "#C44E52"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def collect() -> list[dict]:
    out = []
    for sh in SHAPES:
        j = _load(OUT_MS / f"{sh}.json")
        c = _load(OUT_MS / f"{sh}_control_cold.json")
        r = {"shape": sh, "ms": j, "ctl": c}
        out.append(r)
    return out


# ---------------------------------------------------------------------------

def fig_census(data) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 13.0))
    labels = [d["shape"] for d in data]
    x = np.arange(len(labels))
    w = 0.19

    def get(d, key):
        a = d["ms"]["arms"]["MS_4bpp"]
        ref = d["ms"]["library_reference"]
        ctl = None if d["ctl"] is None else d["ctl"]["arms"]["MS_4bpp"]
        return {"unif": d["ms"]["arms"]["U_uniform"][key],
                "hist": ref["HIST_best"][key], "lib": ref["A1_4bpp"][key],
                "ctl": None if ctl is None else ctl[key], "ms": a[key]}

    ax = axes[0]
    series = [("unif", "uniform s = 1", "#BBBBBB"),
              ("hist", "best stored mask (oracle)", "#8172B3"),
              ("lib", "library single start, unfiltered", "#937860"),
              ("ctl", "filtered single cold start", "#4C72B0"),
              ("ms", "filtered multi-start (deliverable)", "#C44E52")]
    for i, (k, lab, col) in enumerate(series):
        v = [get(d, "J")[k] for d in data]
        v = [np.nan if u is None else u for u in v]
        ax.bar(x + (i - 2) * w, v, w, label=lab, color=col)
    ax.set_yscale("log")
    ax.set_ylabel("J_phi at each arm's own J-stop (log)")
    ax.set_title("A. Melt-region objective, lower is better. Grid 120 x 120, "
                 "4 bits per pixel deliverable arms, each read at its own J-stop.")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(ncol=5, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.06),
              frameon=False)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for i, (k, lab, col) in enumerate(series):
        v = [get(d, "IoU")[k] for d in data]
        v = [np.nan if u is None else u for u in v]
        ax.bar(x + (i - 2) * w, v, w, label=lab, color=col)
    ax.axhline(0.95, color="k", ls="--", lw=1.0)
    ax.text(11.6, 0.962, "IoU 0.95, the SOLVED class at grid 120", fontsize=8.5)
    ax.set_ylabel("intersection over union at the J-stop")
    ax.set_ylim(0.3, 1.03)
    ax.set_title("B. Melted region against the nominal part. The 0.95 line is a "
                 "grid-120 statement only (SOLVE_ROBUSTNESS_VALIDATION).")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    d_lib, d_ctl = [], []
    for d in data:
        g = get(d, "J")
        d_lib.append(100.0 * (g["lib"] - g["ms"]) / g["lib"])
        d_ctl.append(np.nan if g["ctl"] is None
                     else 100.0 * (g["lib"] - g["ctl"]) / g["lib"])
    ax.bar(x - w, d_ctl, 2 * w, label="filtered single cold start", color="#4C72B0")
    ax.bar(x + w, d_lib, 2 * w, label="filtered multi-start", color="#C44E52")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_ylabel("percent of J_phi won against the\nunfiltered single-start library arm")
    ax.set_title("C. What each change buys, separately. Positive is better than the "
                 "library arm.\nWhere the blue bar is high and the red is low, the "
                 "budget split across starts cost more than the extra starts bought.",
                 fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p = FIGS / "fig_ms_census.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def fig_starts(data) -> Path:
    fig = plt.figure(figsize=(16.0, 12.4))
    gs = fig.add_gridspec(4, 6, height_ratios=[1.5, 1, 1, 1], hspace=0.95,
                          wspace=0.55, left=0.06, right=0.985, top=0.945, bottom=0.05)

    ax = fig.add_subplot(gs[0, :3])
    labels = [d["shape"] for d in data]
    x = np.arange(len(labels))
    for d, xi in zip(data, x):
        pb = d["ms"]["probe"]["best_J"]
        for k, v in pb.items():
            if v is None:
                continue
            surv = k in d["ms"]["probe"]["survivors"]
            win = k == d["ms"]["winner_start"]
            if surv:
                ax.scatter(xi, v, s=52, color=START_COLOR[k], marker="o",
                           edgecolor="k" if win else "none",
                           linewidth=1.4, zorder=3)
            else:
                ax.scatter(xi, v, s=26, color=START_COLOR[k], marker="x",
                           linewidth=1.2, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=7.5)
    ax.set_ylabel("best J_phi after the\n2-evaluation probe", fontsize=9)
    ax.set_title("A. The probe and the kill rule. Circles survive, crosses are killed,\n"
                 "the black ring is the eventual winner.", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for k, c in START_COLOR.items():
        ax.scatter([], [], color=c, label=k)
    ax.legend(ncol=4, fontsize=8.5, loc="upper left")

    ax = fig.add_subplot(gs[0, 3:])
    cold_delta, ms_gain = [], []
    for d in data:
        rows = d["ms"]["starts"]["cold"]["rows"]
        c0 = rows[0]["J"]
        c1 = min(r["J"] for r in rows[:2])
        cold_delta.append(100.0 * (c0 - c1) / c0)
        lib = d["ms"]["library_reference"]["A1_4bpp"]["J"]
        ms_gain.append(100.0 * (lib - d["ms"]["arms"]["MS_4bpp"]["J"]) / lib)
    ax.scatter(cold_delta, ms_gain,
               c=[START_COLOR[d["ms"]["winner_start"]] for d in data], s=58, zorder=3)
    for d, xx, yy in zip(data, cold_delta, ms_gain):
        ax.annotate(d["shape"], (xx, yy), fontsize=6.8, xytext=(5, 4),
                    textcoords="offset points")
    ax.axhline(0, color="k", lw=1.0)
    ax.axvline(2.0, color="grey", ls="--", lw=1.2)
    ax.set_xlabel("percent of J_phi the COLD start's own probe removed\n"
                  "in its first 2 evaluations", fontsize=9)
    ax.set_ylabel("percent won by multi-start against\nthe library single-start arm",
                  fontsize=9)
    ax.set_title("B. The mechanism. Multi-start pays off only where the cold start\n"
                 "cannot move, left of the dashed line at 2 percent.", fontsize=10)
    ax.grid(alpha=0.3)

    for i, d in enumerate(data):
        ax = fig.add_subplot(gs[1 + i // 6, i % 6])
        for k, s in d["ms"]["starts"].items():
            r = s["rows"]
            if not r:
                continue
            ax.plot([q["eval_index"] for q in r], [q["J"] for q in r],
                    marker="o", ms=2.6, lw=1.2, color=START_COLOR[k],
                    alpha=1.0 if not s["killed"] else 0.45,
                    ls="-" if not s["killed"] else ":")
        lib = d["ms"]["library_reference"]["A1_4bpp"]["J"]
        hist = d["ms"]["library_reference"]["HIST_best"]["J"]
        ax.axhline(lib, color="#937860", lw=1.0, ls="--")
        ax.axhline(hist, color="#8172B3", lw=1.0, ls=":")
        if d["ctl"] is not None:
            ax.axhline(d["ctl"]["arms"]["MS_4bpp"]["J"], color="#4C72B0", lw=1.0, ls="-.")
        ax.set_title(d["shape"], fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_yscale("log")
        if i % 6 == 0:
            ax.set_ylabel("J_phi", fontsize=8)
        if i >= 12:
            ax.set_xlabel("evaluation", fontsize=8)
    fig.text(0.5, 0.700,
             "C. Per-start evaluation sequences. Dotted and faded = killed at the probe. "
             "The curves are NOT monotone because L-BFGS-B evaluates trial points during "
             "its line search;\neach arm keeps its best iterate. Horizontal references: "
             "dashed brown = unfiltered single-start library arm, dotted purple = best "
             "stored mask, dash-dot blue = filtered single cold start at the full budget.",
             ha="center", va="top", fontsize=9.5)
    p = FIGS / "fig_ms_starts.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def fig_maps(data) -> Path:
    fig, axes = plt.subplots(3, 6, figsize=(16.0, 8.6))
    for ax, d in zip(axes.ravel(), data):
        with np.load(OUT_MS / f"{d['shape']}_maps.npz") as z:
            s = np.asarray(z["MS_4bpp"], dtype=float)
            pm = np.asarray(z["part_mask"], dtype=bool)
        m = np.where(pm, s, np.nan)
        im = ax.imshow(m, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
                       interpolation="bilinear")
        a = d["ms"]["arms"]["MS_4bpp"]
        ax.set_title(f"{d['shape']}  ({d['ms']['winner_start']})\n"
                     f"J {a['J']:.1f}  IoU {a['IoU']:.3f}  {a['census_n_levels_used']} levels",
                     fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(data):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.015, label="binder saturation, 4 bits per pixel")
    fig.suptitle("The delivered filtered multi-start dopant maps, quantized to the "
                 "printer's 16 levels inside the part and held at the nominal 1 outside. "
                 "Winning start in brackets. Grid 120 x 120.", fontsize=11)
    p = FIGS / "fig_ms_maps.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def fig_robust() -> Path:
    shapes = ("square", "rectangle")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    w = 0.35
    for i, sh in enumerate(shapes):
        new = _load(OUT_MS / f"{sh}_robust.json")
        old = _load(OUT_ROBUST / f"{sh}_rim.json")
        if new is None or old is None:
            continue
        d_new = [0.0] + [100.0 * new["rim_verdict"][f"dJ_rel_r{r:.1f}"] for r in (1.0, 2.0)]
        b = old["arms"]["r0.0"]["J"]
        d_old = [0.0] + [100.0 * (old["arms"][f"r{r:.1f}"]["J"] - b) / b for r in (1.0, 2.0)]
        ax.plot([0, 1, 2], d_old, "--o", color=["#937860", "#C0A080"][i],
                label=f"{sh}, unfiltered single start")
        ax.plot([0, 1, 2], d_new, "-o", color=["#C44E52", "#E48A8C"][i],
                label=f"{sh}, filtered multi-start")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("part-masked Gaussian blur applied to the solved map, cells")
    ax.set_ylabel("percent change in J_phi against that arm's own radius 0")
    ax.set_title("A. Rim robustness.\nFlat is robust.", fontsize=10)
    ax.axhline(0, color="k", lw=0.9); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xticks([0, 1, 2])

    ax = axes[1]
    x = np.arange(2); w = 0.2
    for j, (lab, key, col) in enumerate((
            ("unfiltered single start, 120", "old120", "#937860"),
            ("unfiltered single start, 160", "old160", "#C0A080"),
            ("filtered multi-start, 120", "new120", "#C44E52"),
            ("filtered multi-start, 160", "new160", "#E48A8C"))):
        vals = []
        for sh in shapes:
            new = _load(OUT_MS / f"{sh}_robust.json")
            old = _load(OUT_ROBUST / f"{sh}_grid.json")
            v = {"old120": old["arms_120"]["A1_4bpp"]["IoU"],
                 "old160": old["arms"]["A1_4bpp_160"]["IoU"],
                 "new120": new["baseline_120"]["MS_4bpp"]["IoU"],
                 "new160": new["grid"]["MS_4bpp_160"]["IoU"]}[key]
            vals.append(v)
        ax.bar(x + (j - 1.5) * w, vals, w, label=lab, color=col)
    ax.axhline(0.95, color="k", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(shapes)
    ax.set_ylabel("intersection over union at the J-stop")
    ax.set_ylim(0.6, 1.03)
    ax.set_title("B. Grid hold-out.\nSolve at 120, score at 160.", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    for i, sh in enumerate(shapes):
        new = _load(OUT_MS / f"{sh}_robust.json")
        old = _load(OUT_ROBUST / f"{sh}_rim.json")
        j_old = [old["arms"]["r0.0"]["J"]] + [old["arms"][f"r{r:.1f}"]["J"] for r in (1.0, 2.0)]
        j_new = [new["baseline_120"]["MS_4bpp"]["J"]] + [new["rim"][f"r{r:.1f}"]["J"]
                                                         for r in (1.0, 2.0)]
        ax.plot([0, 1, 2], j_old, "--o", color=["#937860", "#C0A080"][i],
                label=f"{sh}, unfiltered")
        ax.plot([0, 1, 2], j_new, "-o", color=["#C44E52", "#E48A8C"][i],
                label=f"{sh}, filtered multi-start")
    ax.set_xlabel("blur radius, cells")
    ax.set_ylabel("J_phi, absolute")
    ax.set_yscale("log")
    ax.set_title("C. The same in absolute J_phi.\nThe crossing is where the blunter "
                 "map wins.", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xticks([0, 1, 2])

    fig.tight_layout()
    p = FIGS / "fig_ms_robust.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    data = collect()
    for f in (fig_census(data), fig_starts(data), fig_maps(data), fig_robust()):
        print("wrote", f)


if __name__ == "__main__":
    main()
