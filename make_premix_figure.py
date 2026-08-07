"""Composite figure for the premix 0->15 wt% sweep (matched-peak, the fair test).

Every level is driven to the SAME 250 C part-peak (ceiling). The panels show, at
that matched peak, that a UNIFORM premix baseline is parasitic in this 2-D model
under fixed-total-absorbed-power: the large bed steals the power (top), so the part
under-densifies (middle) and gets less uniform (bottom). Honest caveats live in
PREMIX_SWEEP_RESULTS.md (voltage-drive flips the sign; uniform premix != graded FGM).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path("results/premix_sweep")
FIG = OUTDIR / "fig_premix_sweep.png"


def main() -> None:
    recs = [json.loads(p.read_text()) for p in
            sorted(OUTDIR.glob("level_*.json"), key=lambda p: json.loads(p.read_text())["wtpct"])]
    if not recs:
        raise SystemExit("no level records -- run study_premix_sweep.py + augment_matched_peak.py")
    wt = [r["wtpct"] for r in recs]
    bed = [r["bed_absorption_fraction"] for r in recs]
    phi = [r.get("matched_part_mean_phi") for r in recs]
    p2m = [r.get("matched_part_peak_to_mean") for r in recs]
    drive = [r["drive_to_ceiling_W"] for r in recs]

    fig, axs = plt.subplots(3, 1, figsize=(7.4, 9.0), sharex=True, dpi=180)
    fig.suptitle("Uniform premix baseline (0–15 wt%), all driven to the 250 °C ceiling:\n"
                 "the bed parasitically absorbs, so the part under-fuses",
                 fontsize=11.5, weight="bold")

    axs[0].plot(wt, bed, "o-", color="#c0392b", lw=2)
    axs[0].set_ylabel("bed absorption\nfraction")
    axs[0].set_title("premix flips ~80% of the fixed power into the (parasitic) bed", fontsize=9)
    axs[0].set_ylim(0, 1)

    axs[1].plot(wt, phi, "o-", color="#1e8449", lw=2)
    axs[1].set_ylabel("part mean φ\n(densification)")
    axs[1].set_title("so at the SAME ceiling the part under-densifies (1.00 → 0.5–0.63)", fontsize=9)
    axs[1].set_ylim(0, 1.05)

    axs[2].plot(wt, p2m, "o-", color="#2471a3", lw=2, label="part peak/mean")
    axs[2].axhline(1.0, color="#888", ls=":", lw=1)
    ax2b = axs[2].twinx()
    ax2b.plot(wt, drive, "s--", color="#7d3c98", lw=1.5, label="drive→ceiling (W)")
    ax2b.set_ylabel("drive→250°C (W)", color="#7d3c98")
    axs[2].set_ylabel("part peak / mean")
    axs[2].set_title("part gets LESS uniform (peak/mean ↑); drive rises modestly (+8%)", fontsize=9)
    axs[2].set_xlabel("premix baseline dopant (wt% dopant-to-nylon)")

    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
