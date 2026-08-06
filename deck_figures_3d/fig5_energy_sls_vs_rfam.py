"""Figure 5: Where the energy goes - SLS vs RFAM, per unit volume.

Energy to sinter one cm3 of PA12, on common (volumetric) footing. The point
is structural, not a precise ratio: SLS's LASER DOSE is a tiny sliver of what
the machine spends; the dominant SLS cost is holding a 170-180 C chamber for a
long serial build, energy that never forms the part. RFAM deposits its energy
directly in the part via RF absorption over tens of minutes with no sustained
hot chamber, so on TOTAL SYSTEM energy RFAM should win - the open unknown is
RF coupling efficiency, shown as an uncertainty band.

ALL VALUES ARE ESTIMATES for comparison; the figure labels them as such and
draws the RFAM total as a hatched uncertainty band. Numbers must track the
dissertation - swap any of the constants below if a source-of-record differs.

Data provenance (each constant is derived, not measured here; rendering only):
  SLS_LASER_JCM3  - Matt's areal laser dose 2.8-3.0 J/cm2 / 0.110 mm layer
                    (Formlabs Fuse 1+ 30W, ~247 um spot). = 255-273 J/cm3.
  SLS_SYS_JCM3    - published SLS specific energy ~30-100 kWh/kg at ~1 g/cm3
                    dense PA12 -> ~1.08e5-3.6e5 J/cm3 (chamber-dominated).
  RFAM_PART_JCM3  - this repo's 3-D sim: volumetric input = power_density x
                    exposure_time (part volume cancels), cross-checked to the
                    joule against heatr3d's own S1 energy audit; ~2400 J/cm3
                    at the efficient (nominal) drive.
  RFAM_SYS_JCM3   - RFAM_PART / RF coupling efficiency (5-30%, UNMEASURED -
                    the P-gate). = ~8.0e3-4.8e4 J/cm3. Even at pessimistic
                    5 percent this stays below the SLS system band.
  FLOOR_JCM3      - theoretical melt energy for PA12: ~130 J/cm3 from a 170 C
                    preheat, ~420 J/cm3 from room temp (cp~2000 J/kgK,
                    latent~70 kJ/kg, rho~1000 kg/m3 dense).

Honest scoping: delivered/absorbed energy, not wall-plug for RFAM (RF coupling
is the uncertain factor and is drawn as such). SLS system energy is general SLS
literature, not a Fuse-1+-specific measurement. The comparison is volumetric
(SLS areal / layer thickness) because RFAM is a bulk process; the two deposit
energy very differently (SLS surface-serial into a preheated bed, RFAM bulk-
parallel), which the figure states rather than hides.
"""
from __future__ import annotations

from matplotlib.patches import Patch
import matplotlib.pyplot as plt

import style3d as st

# --- estimates (J/cm3); see docstring for provenance. Edit here to retrack. ---
SLS_LASER = 262.0
SLS_SYS = (1.08e5, 3.6e5)
RFAM_PART = 2400.0
RFAM_COUPLING = (0.05, 0.30)                       # RF coupling band (unmeasured)
RFAM_SYS = (RFAM_PART / RFAM_COUPLING[1], RFAM_PART / RFAM_COUPLING[0])
FLOOR = (130.0, 420.0)


def _rangebar(ax, x, lo, hi, color, hatch=None):
    ax.bar(x, hi, width=0.62, color=color, alpha=0.30, zorder=2,
           hatch=hatch, edgecolor=color, linewidth=0)
    ax.plot([x, x], [lo, hi], color=color, lw=2.2, zorder=4)
    for yy in (lo, hi):
        ax.plot([x - 0.16, x + 0.16], [yy, yy], color=color, lw=2.2, zorder=4)


def main() -> None:
    fig = plt.figure(figsize=(12.6, 7.8), dpi=st.DPI)
    st.title_block(
        fig, "WHERE THE ENERGY GOES",
        "energy to sinter one cm3 of PA12  -  SLS burns most of it heating a "
        "chamber; RFAM puts it in the part")

    leg = [Patch(facecolor=st.GOOD, label="into part formation"),
           Patch(facecolor=st.WARM, alpha=0.5,
                 label="SLS overhead (chamber, whole build)"),
           Patch(facecolor=st.MELT, alpha=0.5, hatch="////",
                 label="RFAM total est. (RF coupling 5-30%, unmeasured)")]
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.098, 0.845),
               ncol=3, facecolor=st.PANEL, edgecolor=st.DIM, labelcolor=st.FG,
               fontsize=8.6, framealpha=0.92, handlelength=1.4, columnspacing=1.6)

    ax = fig.add_axes([0.10, 0.16, 0.86, 0.60])
    ax.set_facecolor(st.BG)
    ax.set_yscale("log")
    ax.axhspan(*FLOOR, color=st.ACCENT, alpha=0.10, zorder=0)
    ax.text(-0.46, 470, "theoretical melt\nfloor PA12\n130-420 J/cm3",
            color=st.ACCENT, fontsize=8, va="bottom", ha="left")

    xs = [0, 1, 2.4, 3.4]
    ax.bar(xs[0], SLS_LASER, width=0.62, color=st.GOOD, zorder=3)
    ax.bar(xs[2], RFAM_PART, width=0.62, color=st.GOOD, zorder=3)
    _rangebar(ax, xs[1], *SLS_SYS, st.WARM)
    _rangebar(ax, xs[3], *RFAM_SYS, st.MELT, hatch="////")

    ax.set_xticks(xs)
    ax.set_xticklabels(["into part\n(laser dose)",
                        "TOTAL system\n(chamber-dominated)",
                        "into part\n(absorbed)", "TOTAL system\n(est.)"],
                       color=st.FG, fontsize=9)
    ax.set_ylabel("energy per cm3 of part   [J/cm3, log]", color=st.FG, fontsize=10)
    ax.set_ylim(80, 6e5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=st.DIM)

    ax.text(0.5, 4.3e5, "SLS  (Fuse 1+ 30W)", color=st.DIM, ha="center",
            fontsize=12, fontweight="bold")
    ax.text(2.9, 4.3e5, "RFAM  (this work)", color=st.DIM, ha="center",
            fontsize=12, fontweight="bold")
    ax.axvline(1.7, color=st.DIM, lw=0.6, alpha=0.4)
    ax.text(0, SLS_LASER * 1.2, f"~{SLS_LASER:.0f}", color=st.GOOD, ha="center",
            va="bottom", fontsize=9.5, fontweight="bold")
    ax.text(2.4, RFAM_PART * 1.16, f"~{RFAM_PART:,.0f}", color=st.GOOD,
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.annotate("~99.7% of SLS energy is\nchamber heat, not the part",
                xy=(1.31, 1.9e5), xytext=(-0.02, 2.6e4), color=st.WARM,
                fontsize=9.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=st.WARM, lw=1.0))
    ax.annotate("energy goes into the part;\nstill below SLS even at\n"
                "pessimistic RF coupling",
                xy=(3.08, 2.0e4), xytext=(2.02, 6.2e2), color=st.MELT,
                fontsize=9.5, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=st.MELT, lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))

    fig.text(0.10, 0.052,
             "ESTIMATES for comparison.  SLS laser dose 2.8-3.0 J/cm2 areal / "
             "0.11 mm layer (Fuse 1+ 30W, 247 um spot).",
             color=st.DIM, fontsize=8)
    fig.text(0.10, 0.030,
             "SLS system: published 30-100 kWh/kg.  RFAM absorbed: this 3-D "
             "sim.  RFAM system: absorbed / RF coupling (unmeasured, the P-gate).",
             color=st.DIM, fontsize=8)

    out = st.OUT / "fig5_energy_sls_vs_rfam.png"
    fig.savefig(out, dpi=st.DPI, facecolor=st.BG)
    print("wrote", out,
          f"| SLS sys/laser ~{(SLS_SYS[0]*SLS_SYS[1])**0.5/SLS_LASER:.0f}x"
          f"  RFAM sys worst {RFAM_SYS[1]:.0f} < SLS sys best {SLS_SYS[0]:.0f}")


if __name__ == "__main__":
    main()
