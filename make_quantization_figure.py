"""Does the graded uniformity win survive printable quantization? (Yes.)

Part peak/mean at the 250 C ceiling for uniform, premix, and the graded map at
continuous / 4-bpp / 2-bpp. Grading (all variants) stays near 1.0; 4-bpp is
identical to continuous, 2-bpp erodes only slightly -- all still beat uniform and
premix. Fusion phi stays 1.000 for every graded variant.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/premix_vs_graded")
FIG = OUT / "fig_graded_quantization.png"


def main():
    hh = json.loads((OUT / "premix_vs_graded.json").read_text())
    q = json.loads((OUT / "premix_graded_quantization.json").read_text())
    bars = [
        ("uniform\n(s=1)", hh["uniform"]["part_peak_to_mean"], hh["uniform"]["part_mean_phi"], "#7f8c8d"),
        ("premix\n(15 wt%)", hh["premix_15wt"]["part_peak_to_mean"], hh["premix_15wt"]["part_mean_phi"], "#c0392b"),
        ("graded\ncontinuous", hh["graded"]["part_peak_to_mean"], hh["graded"]["part_mean_phi"], "#145a32"),
        ("graded\n4-bpp", q["graded_4bpp"]["part_peak_to_mean"], q["graded_4bpp"]["part_mean_phi"], "#1e8449"),
        ("graded\n2-bpp", q["graded_2bpp"]["part_peak_to_mean"], q["graded_2bpp"]["part_mean_phi"], "#52be80"),
    ]
    labels = [b[0] for b in bars]
    p2m = [b[1] for b in bars]
    phi = [b[2] for b in bars]
    cols = [b[3] for b in bars]
    x = np.arange(len(bars))

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=180, constrained_layout=True)
    fig.suptitle("The graded uniformity win survives printable quantization",
                 fontsize=12.5, weight="bold")
    ax.bar(x, p2m, color=cols)
    for i, (v, f) in enumerate(zip(p2m, phi)):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9.5, weight="bold")
        ax.text(i, 1.008, f"φ={f:.3f}", ha="center", fontsize=8.5,
                color="white", weight="bold")
    ax.axhline(1.0, color="#888", ls=":", lw=1)
    ax.set_ylabel("part peak / mean at 250 °C ceiling\n(lower = more uniform)")
    ax.set_title("4-bpp = continuous; 2-bpp erodes only slightly; both beat uniform & premix",
                 fontsize=9.5)
    ax.set_ylim(1.0, 1.30)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(FIG)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
