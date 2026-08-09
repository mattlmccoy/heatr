"""The payoff figure: premix under POWER drive vs VOLTAGE drive -- the sign flip.

Power drive (fixed total absorbed power): premix is parasitic -- part under-fuses at
the ceiling, drive rises. Voltage drive (fixed applied voltage): premix raises absorbed
power in the part, so the voltage to reach the ceiling DROPS and the part fuses. The
honest answer to 'does premix help?' is: it depends entirely on the drive mode.

Reads results/premix_sweep/ (power) + results/premix_sweep_voltage/ (voltage).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POWER = Path("results/premix_sweep")
VOLT = Path("results/premix_sweep_voltage")
FIG = Path("results/premix_sweep_voltage/fig_premix_drive_comparison.png")


def _load(d):
    return [json.loads(p.read_text()) for p in
            sorted(d.glob("level_*.json"), key=lambda p: json.loads(p.read_text())["wtpct"])]


def main():
    P = _load(POWER)
    V = _load(VOLT)
    if not P or not V:
        raise SystemExit("need both results/premix_sweep and results/premix_sweep_voltage")
    wt = [r["wtpct"] for r in P]
    phi_p = [r.get("matched_part_mean_phi") for r in P]
    phi_v = [r.get("matched_part_mean_phi") for r in V]
    # normalized drive to wt%=0 (power in W rising; voltage in V falling)
    d_p = [r["drive_to_ceiling_W"] for r in P]
    d_v = [r["voltage_to_ceiling_V"] for r in V]
    d_p_n = [x / d_p[0] for x in d_p]
    d_v_n = [x / d_v[0] for x in d_v]

    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=180)
    fig.suptitle("Drive mode moves the KNOB, not the part: uniform premix is parasitic either way",
                 fontsize=12, weight="bold")

    axs[0].plot(wt, phi_p, "o-", color="#c0392b", lw=2, label="power drive (fixed total power)")
    axs[0].plot(wt, phi_v, "s--", color="#1e8449", lw=2, label="voltage drive (fixed field)")
    axs[0].set_ylabel("part mean φ at the 250 °C ceiling")
    axs[0].set_xlabel("premix baseline dopant (wt%)")
    axs[0].set_title("part fusion at matched ceiling — SAME (bad) in both modes", fontsize=9.5)
    axs[0].set_ylim(0, 1.05)
    axs[0].legend(fontsize=8.5, loc="upper right")
    axs[0].grid(alpha=0.25)

    axs[1].plot(wt, d_p_n, "o-", color="#c0392b", lw=2, label="power drive (W) rises +8%")
    axs[1].plot(wt, d_v_n, "s--", color="#1e8449", lw=2, label="voltage drive (V) falls ~8×")
    axs[1].axhline(1.0, color="#888", ls=":", lw=1)
    axs[1].set_ylabel("drive to ceiling, ÷ its wt%=0 value")
    axs[1].set_xlabel("premix baseline dopant (wt%)")
    axs[1].set_title("the knob diverges — but it's units, not a rescue", fontsize=9.5)
    axs[1].set_yscale("log")
    axs[1].legend(fontsize=8.5)
    axs[1].grid(alpha=0.25, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
