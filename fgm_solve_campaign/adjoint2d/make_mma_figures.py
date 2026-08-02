"""Figures for `MMA_RETEST_REPORT.md`. Every one is viewed before delivery.

Run: ./.venv312/bin/python -m adjoint2d.make_mma_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .build_mma_tables import ARMS, OUT_MMA, OUT_TOPOPT, SHAPES, load

FIGS = Path(__file__).resolve().parents[1] / "figs_mma"
DPI = 180
COL = {"a_filteronly_lbfgsb_40": "#444444",
       "e_projection_lbfgsb_40": "#c8461e",
       "b_projection_mma_40": "#1f6fb4",
       "c_projection_mma_80": "#4fa3d9",
       "d_projection_lbfgsbcarry_80": "#e39b2b"}
SHORT = {"a_filteronly_lbfgsb_40": "a  filter only\nL-BFGS-B 40",
         "e_projection_lbfgsb_40": "e  projection\nL-BFGS-B 40",
         "b_projection_mma_40": "b  projection\nMMA 40",
         "c_projection_mma_80": "c  projection\nMMA 80",
         "d_projection_lbfgsbcarry_80": "d  projection\nL-BFGS-B carry 80"}


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p, flush=True)


# ---------------------------------------------------------------------------
# 1. the mechanics: what an evaluation buys under each optimizer
# ---------------------------------------------------------------------------

def stage_best_curve(rows):
    """Running best WITHIN each beta stage, which is what is carried forward.

    A running minimum across the whole trajectory would be wrong here: the
    continuation carries only the CURRENT stage's best iterate into the next
    stage, and the deliverable is built at the final beta. A cross-stage
    running minimum would draw a curve no arm ever delivers.
    """
    out, cur, prev_beta = [], np.inf, None
    for r in rows:
        if r["beta"] != prev_beta:
            cur = np.inf
            prev_beta = r["beta"]
        cur = min(cur, r["J"])
        out.append(cur)
    return out


def useful_fraction(rows) -> float:
    """Fraction of evaluations that improved their own stage's incumbent."""
    if not rows:
        return float("nan")
    n, cur, prev_beta = 0, np.inf, None
    for r in rows:
        if r["beta"] != prev_beta:
            cur = np.inf
            prev_beta = r["beta"]
        if r["J"] < cur:
            n += 1
            cur = r["J"]
    return n / len(rows)


def fig_mechanics():
    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 4)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    arms = ("e_projection_lbfgsb_40", "b_projection_mma_40",
            "d_projection_lbfgsbcarry_80", "c_projection_mma_80")
    for ax, shape in zip(axes, SHAPES):
        for arm in arms:
            j = load(arm, shape)
            if j is None or not j.get("rows"):
                continue
            rows = j["rows"]
            k = [r["eval_index"] for r in rows]
            ls = "--" if "80" in arm else "-"
            ax.plot(k, [r["J"] for r in rows], ls, color=COL[arm], lw=0.8, alpha=0.40)
            ax.plot(k, stage_best_curve(rows), ls, color=COL[arm], lw=2.0,
                    label=SHORT[arm].replace("\n", " "))
            ax.plot(k[-1] + 1, j["arms"]["TO_4bpp"]["J"], "*", color=COL[arm],
                    ms=13, mec="k", mew=0.5)
            b = [r["beta"] for r in rows]
            for i in range(1, len(b)):
                if b[i] != b[i - 1]:
                    ax.axvline(k[i] - 0.5, color="0.75", lw=0.5)
        a = load("a_filteronly_lbfgsb_40", shape)
        if a is not None:
            ax.axhline(a["arms"]["TO_4bpp"]["J"], color=COL["a_filteronly_lbfgsb_40"],
                       lw=1.6, ls=":",
                       label="a  filter only L-BFGS-B 40, DELIVERED")
        ax.set_yscale("log")
        ax.set_title(shape, fontsize=11)
        ax.set_xlabel("gradient evaluation")
        ax.set_ylabel("J, area-fill target, grid 120")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="upper right")

    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(len(SHAPES))
    w = 0.2
    for i, arm in enumerate(arms):
        v = []
        for s in SHAPES:
            j = load(arm, s)
            v.append(np.nan if j is None else 100.0 * useful_fraction(j.get("rows", [])))
        ax.barh(x + (i - 1.5) * w, v, w, color=COL[arm])
    ax.set_yticks(x)
    ax.set_yticklabels(SHAPES, fontsize=8)
    ax.set_xlabel("percent of evaluations that improved\ntheir own stage incumbent",
                  fontsize=8)
    ax.set_title("the mechanics claim, measured", fontsize=10)
    ax.grid(alpha=0.25, axis="x")

    ax2 = fig.add_subplot(gs[1, 3])
    for i, arm in enumerate(arms):
        for s_i, s in enumerate(SHAPES):
            j = load(arm, s)
            if j is None:
                continue
            st = [x for x in j["stages"] if x.get("best_J") is not None]
            if not st:
                continue
            b0 = st[0]["best_J"]
            ax2.plot([x["beta"] for x in st], [x["best_J"] / b0 for x in st],
                     "-o" if "40" in SHORT[arm] else "--o", color=COL[arm],
                     lw=0.9, ms=3, alpha=0.75,
                     label=SHORT[arm].replace("\n", " ") if s_i == 0 else None)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set_xlabel("beta stage")
    ax2.set_ylabel("stage best J, relative to the beta = 1 stage")
    ax2.set_title("what sharpening the projection costs\n(all six shapes, each arm)",
                  fontsize=9)
    ax2.legend(fontsize=6.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("What one gradient evaluation buys. Thin: every evaluation. Thick: "
                 "best WITHIN the current beta stage, which is what the continuation "
                 "carries forward. Star: the delivered 4-bits-per-pixel arm.\n"
                 "Grey verticals: beta stage boundaries. Dotted: the filter-only "
                 "production recipe's delivered J.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_mma_mechanics.png")


# ---------------------------------------------------------------------------
# 2. the scoreboard at grid 120
# ---------------------------------------------------------------------------

def fig_scoreboard():
    arms = list(ARMS)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    x = np.arange(len(SHAPES))
    w = 0.16
    for ax, key, lab, logy in (
            (axes[0], "J_raster_chi",
             "J under the OLD binary target (comparable across passes)", True),
            (axes[1], "IoU_120", "IoU against the binary part mask, grid 120", False),
            (axes[2], "M_nd", "non-discreteness M_nd, 0 is binary", False)):
        for i, arm in enumerate(arms):
            vals = []
            for s in SHAPES:
                j = load(arm, s)
                vals.append(np.nan if j is None else j["arms"]["TO_4bpp"][
                    {"J_raster_chi": "J_raster_chi", "IoU_120": "IoU",
                     "M_nd": "non_discreteness"}[key]])
            ax.bar(x + (i - 2) * w, vals, w, color=COL[arm],
                   label=SHORT[arm].replace("\n", "  "))
        if key == "IoU_120":
            ax.axhline(0.95, color="k", ls="--", lw=1, label="SOLVED threshold 0.95")
            ax.set_ylim(0.5, 1.02)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(SHAPES)
        ax.set_title(lab, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 3.0)   # headroom for the legend
    axes[0].legend(fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("Five arms at grid 120, deliverable 4-bits-per-pixel map, each read "
                 "at its own J-stop", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig_mma_scoreboard.png")


# ---------------------------------------------------------------------------
# 3. the delivered maps
# ---------------------------------------------------------------------------

def fig_maps():
    arms = ["a_filteronly_lbfgsb_40", "e_projection_lbfgsb_40",
            "b_projection_mma_40", "c_projection_mma_80",
            "d_projection_lbfgsbcarry_80"]
    fig, axes = plt.subplots(len(arms), len(SHAPES),
                             figsize=(2.15 * len(SHAPES), 2.35 * len(arms)))
    for i, arm in enumerate(arms):
        d, stem, _ = ARMS[arm]
        for k, shape in enumerate(SHAPES):
            ax = axes[i, k]
            p = d / f"{stem.format(s=shape)}_maps.npz"
            if not p.exists():
                ax.axis("off")
                continue
            with np.load(p) as z:
                m = np.asarray(z["TO_4bpp"], dtype=float)
                pm = np.asarray(z["part_mask"], dtype=bool)
            show = np.where(pm, m, np.nan)
            ys, xs = np.where(pm)                 # crop to the part, with a margin
            pad = 3
            show = show[max(ys.min() - pad, 0):ys.max() + pad + 1,
                        max(xs.min() - pad, 0):xs.max() + pad + 1]
            ax.imshow(show, origin="lower", cmap="viridis", vmin=0, vmax=1,
                      interpolation="bilinear")
            j = load(arm, shape)
            mm = j["arms"]["TO_4bpp"]
            ax.set_title(f"IoU {mm['IoU']:.3f}  M_nd {mm['non_discreteness']:.2f}",
                         fontsize=7.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(SHORT[arm], fontsize=8)
            if i == 0:
                ax.annotate(shape, xy=(0.5, 1.0), xycoords="axes fraction",
                            xytext=(0, 26), textcoords="offset points",
                            ha="center", va="bottom", fontsize=11)
    fig.suptitle("Delivered dopant saturation maps, 4 bits per pixel inside the part, "
                 "grid 120. Dark is low saturation.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig_mma_maps.png")


# ---------------------------------------------------------------------------
# 4. the MMA internal state, which is the mechanism claim
# ---------------------------------------------------------------------------

def fig_asymptotes():
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for ax, shape in zip(axes.ravel(), SHAPES):
        ax2 = ax.twinx()
        for arm, ls, lab in (("b_projection_mma_40", "-", "b  MMA 40"),
                             ("c_projection_mma_80", "--", "c  MMA 80")):
            j = load(arm, shape)
            if j is None:
                continue
            st = [s for s in j.get("stages", []) if s.get("mma_state")]
            if not st:
                continue
            k = [s["mma_state"]["iteration"] for s in st]
            lo = [s["mma_state"]["asy_dist_lo_mean"] for s in st]
            con = [s["mma_state"]["gamma_frac_contracted"] for s in st]
            ax.plot(k, lo, ls + "o", color="#1f6fb4",
                    label=f"{lab}, mean asymptote distance (left)")
            ax2.plot(k, con, ls + "s", color="#c8461e", ms=4,
                     label=f"{lab}, fraction of cells contracting (right)")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("fraction contracting", color="#c8461e", fontsize=8)
        ax.set_title(shape, fontsize=11)
        ax.set_xlabel("MMA iteration, counted across every beta stage")
        ax.set_ylabel("mean asymptote distance", color="#1f6fb4", fontsize=8)
        ax.grid(alpha=0.25)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 0].get_shared_x_axes if False else ([], [])
    axes[0, 0].legend(h1, l1, fontsize=6.5, loc="lower left")
    fig.suptitle("MMA's carried state, read at the end of each beta stage. The "
                 "iteration index does not reset at a stage boundary: that is the "
                 "continuation property L-BFGS-B does not have.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_mma_asymptotes.png")


# ---------------------------------------------------------------------------
# 5. the two acceptance gates
# ---------------------------------------------------------------------------

def _robust(stem):
    p = OUT_MMA / f"{stem}_robust.json"
    if p.exists():
        return json.loads(p.read_text())
    p = OUT_TOPOPT / f"{stem}_robust.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_acceptance():
    arms = [a for a in ARMS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = np.arange(len(SHAPES))
    w = 0.16
    for i, arm in enumerate(arms):
        d, stem, _ = ARMS[arm]
        iou120, iou160, gateb = [], [], []
        for s in SHAPES:
            r = _robust(stem.format(s=s))
            if r is None:
                iou120.append(np.nan)
                iou160.append(np.nan)
                gateb.append(np.nan)
                continue
            v = r["gate_A_verdict"]
            iou120.append(v["IoU_at_120"])
            iou160.append(max(v["IoU_at_160_maptransfer_recal"],
                              v["IoU_at_160_designtransfer_recal"]))
            gateb.append(100.0 * r["gate_B_verdict"]["max_abs_dJ_rel"])
        axes[0].bar(x + (i - 2) * w, iou120, w, color=COL[arm],
                    label=SHORT[arm].replace("\n", "  "))
        axes[1].bar(x + (i - 2) * w, iou160, w, color=COL[arm])
        axes[2].bar(x + (i - 2) * w, gateb, w, color=COL[arm])
    for ax, t in ((axes[0], "IoU at grid 120, in grid"),
                  (axes[1], "Gate A: IoU at grid 160, best transfer, drive recalibrated"),
                  (axes[2], "Gate B: max |dJ| over sub-radius blurs, percent")):
        ax.set_xticks(x)
        ax.set_xticklabels(SHAPES, rotation=20)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    for ax in axes[:2]:
        ax.axhline(0.95, color="k", ls="--", lw=1)
        ax.set_ylim(0.5, 1.02)
    axes[2].axhline(10.0, color="k", ls="--", lw=1)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("The two acceptance gates. Dashed: the SOLVED threshold 0.95 and the "
                 "10 percent Gate B tolerance.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_mma_acceptance.png")


def main():
    fig_mechanics()
    fig_scoreboard()
    fig_maps()
    fig_asymptotes()
    fig_acceptance()


if __name__ == "__main__":
    main()
