"""Figure: premix vs graded vs uniform, at the 250 C ceiling (run_sim).

Two bars per arm: part densification phi (higher=better) and part peak/mean
(lower=more uniform=better). The story: grading (dopant where the part needs it)
beats a uniform bed floor (premix), which is parasitic.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/premix_vs_graded")
FIG = OUT / "fig_premix_vs_graded.png"
ORDER = ["uniform", "premix_15wt", "graded"]
LABELS = {"uniform": "uniform\n(s=1, printed)", "premix_15wt": "premix\n(15 wt% bed floor)",
          "graded": "graded\n(adjoint optimal)"}
COLORS = {"uniform": "#7f8c8d", "premix_15wt": "#c0392b", "graded": "#1e8449"}


def main():
    d = json.loads((OUT / "premix_vs_graded.json").read_text())
    arms = [a for a in ORDER if a in d]
    phi = [d[a]["part_mean_phi"] for a in arms]
    p2m = [d[a]["part_peak_to_mean"] for a in arms]
    drive = [d[a]["drive_to_ceiling_W"] for a in arms]
    cols = [COLORS[a] for a in arms]
    x = np.arange(len(arms))

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=180)
    fig.suptitle("Premix vs graded at the 250 °C ceiling: put the dopant where the part needs it",
                 fontsize=12, weight="bold")

    axs[0].bar(x, phi, color=cols)
    for i, v in enumerate(phi):
        axs[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axs[0].set_ylabel("part mean φ (densification)")
    axs[0].set_title("fusion — higher is better", fontsize=10)
    axs[0].set_ylim(0, 1.08)
    axs[0].set_xticks(x); axs[0].set_xticklabels([LABELS[a] for a in arms], fontsize=8.5)

    axs[1].bar(x, p2m, color=cols)
    for i, v in enumerate(p2m):
        axs[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    axs[1].axhline(1.0, color="#888", ls=":", lw=1)
    axs[1].set_ylabel("part peak / mean")
    axs[1].set_title("uniformity — lower (→1.0) is better", fontsize=10)
    axs[1].set_xticks(x); axs[1].set_xticklabels([LABELS[a] for a in arms], fontsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG, bbox_inches="tight")
    print(f"wrote {FIG}  (drives W: {dict(zip(arms, [round(v) for v in drive]))})")


if __name__ == "__main__":
    main()
