#!/usr/bin/env python3
"""Analyze the per-node square tuning campaign: convergence plot + map
correlation against Jared Allison's converged tuned-sigma ground truth.

Reuses analysis_fgm_vs_jared_square.load_jared_grid (the exact machinery that
produced the r=-0.35 anti-correlation for HEATR's one-sided FGM map).

Outputs (in outputs_eqs/pernode_square/):
  convergence.png, map_vs_jared.png, analysis_stats.json

Run: ./.venv312/bin/python analyze_pernode_square.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from analysis_fgm_vs_jared_square import load_jared_grid, norm01, structure_stats, N

REPO = Path(__file__).resolve().parents[2]  # repo root (moved to scripts/analysis/)
OUT = REPO / "outputs_eqs/pernode_square"
HEATR_HALF_MM = 10.0
BASELINE_SIGMA_T = 4.92   # untouched-baseline matched-melt sigma_T (DOSECHECK)


def _load_converged_grid(variant: str) -> np.ndarray:
    """Extract the part sub-region of the converged sigma map, resample to (N,N)."""
    d = np.load(OUT / variant / "converged_sigma_map.npz")
    sig = d["sigma_map"].astype(float)          # (120,120), 0 outside part
    x = d["x"].astype(float) * 1e3              # m -> mm
    y = d["y"].astype(float) * 1e3
    ix = np.where(np.abs(x) <= HEATR_HALF_MM + 1e-9)[0]
    iy = np.where(np.abs(y) <= HEATR_HALF_MM + 1e-9)[0]
    sub = sig[np.ix_(iy, ix)]
    g = zoom(sub, (N / sub.shape[0], N / sub.shape[1]), order=1)
    return g[:N, :N]


def main() -> None:
    variants = [v.name for v in sorted(OUT.iterdir())
                if (v / "convergence.json").exists()]
    convs = {v: json.loads((OUT / v / "convergence.json").read_text()) for v in variants}

    # ── Convergence figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [("sigma_T_matched_c", "$\\sigma_T$ at matched melt (°C)"),
               ("maxDiff_c", "maxDiff = max|Tt−T| (°C)"),
               ("P_abs_W_per_m", "absorbed power (W/m)"),
               ("frac_nodes_at_clamp", "fraction of nodes at clamp")]
    for ax, (key, lab) in zip(axes.ravel(), metrics):
        for v in variants:
            rows = convs[v]["iterations"]
            it = [r["iter"] for r in rows]
            ax.plot(it, [r[key] for r in rows], "o-", label=v)
        ax.set_xlabel("iteration"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
        if key == "sigma_T_matched_c":
            ax.axhline(BASELINE_SIGMA_T, ls="--", color="k",
                       label=f"baseline {BASELINE_SIGMA_T}")
        if key == "maxDiff_c":
            ax.axhline(5.0, ls="--", color="r", label="conv 5 °C")
        ax.legend(fontsize=8)
    fig.suptitle("Per-node adaptive-gain two-sided tuning — square (voltage-driven)")
    fig.savefig(OUT / "convergence.png", dpi=170)
    plt.close(fig)

    # ── Map correlation vs Jared ──────────────────────────────────────────────
    jared = load_jared_grid()
    jared_n = norm01(jared)
    stats = {"baseline_sigma_T": BASELINE_SIGMA_T, "variants": {}}
    ncol = len(variants) + 1
    fig2, axs = plt.subplots(1, ncol, figsize=(5 * ncol, 4.6), constrained_layout=True)
    im = axs[0].imshow(jared_n, origin="lower", extent=[-1, 1, -1, 1],
                       cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
    axs[0].set_title("Jared converged σ (norm)")
    for j, v in enumerate(variants, start=1):
        g = _load_converged_grid(v)
        gn = norm01(g)
        r = float(np.corrcoef(jared_n.ravel(), gn.ravel())[0, 1])
        axs[j].imshow(gn, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis",
                      vmin=0, vmax=1, interpolation="bilinear")
        rows = convs[v]["iterations"]
        best = min(r_["sigma_T_matched_c"] for r_ in rows)
        axs[j].set_title(f"{v}  (Pearson r={r:+.3f})\nbest σ_T={best:.2f} °C")
        st = structure_stats(gn, v)
        stats["variants"][v] = {
            "pearson_r_vs_jared": round(r, 4),
            "best_sigma_T_matched_c": round(best, 3),
            "final_sigma_T_matched_c": round(rows[-1]["sigma_T_matched_c"], 3),
            "n_iters": len(rows),
            "converged": rows[-1]["converged"],
            "sigma_max": convs[v]["sigma_max"],
            "raw_sigma_range": [round(float(g.min()), 5), round(float(g.max()), 5)],
            "shell_edge_minus_center": st["edge_minus_center"],
            "corner_mean": st["corner_mean(both>0.85)"],
            "center_mean": st["center_mean(|r|<0.25)"],
        }
        print(f"{v}: r_vs_jared={r:+.3f} best_sigT={best:.2f} "
              f"edge-center={st['edge_minus_center']:+.3f}")
    fig2.colorbar(im, ax=axs, shrink=0.8, label="normalized level")
    fig2.savefig(OUT / "map_vs_jared.png", dpi=170)
    plt.close(fig2)

    (OUT / "analysis_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {OUT/'convergence.png'}\nwrote {OUT/'map_vs_jared.png'}\n"
          f"wrote {OUT/'analysis_stats.json'}")


if __name__ == "__main__":
    main()
