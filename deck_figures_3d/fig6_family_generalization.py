"""Figure 6: One method, three shapes - shippable on two independent engines.

The cross-engine is_sendable result across the shape family. For each of three
geometrically distinct parts (full-height square column, compact cube, sharp
pyramid), the ceiling-coupled joint solve picks a PER-PART drive and shapes the
dopant so the fully-dense part clears the 250 C degradation ceiling - and a
SECOND, independent engine (heatr3d voxel FDM) confirms it does. Both engines'
end-state peaks sit under the ceiling on all three, with 3-6 C margin.

Honest scope, drawn into the figure, not hidden:
  - cross-engine SIMULATION agreement (dolfinx FEM vs heatr3d FDM), NOT a
    printed-and-measured part; the physical DSC/TGA + density coupon (P-gate)
    is what makes the ceiling a measured quantity.
  - the pyramid is verified at n=80 (its sharp apex aliases the DG0->voxel
    transfer grid-dependently; n=64 and n=96 exceed the 2 percent fidelity gate)
    and its solve did not fully converge.
  - the engine offset (heatr3d minus dolfinx) is +8.8 to +11.8 C: POSITIVE and
    covered by the 15 C headroom, but NOT a tight constant - no per-shape
    re-measurement claim is made.

Data provenance (rendering only; peaks are pinned results from the committed
verify artifacts):
  square   solve3d/results/verify_stage_b4_square_heatr3d.json  (n64, 0.34x)
  cube     solve3d/results/verify_stage_b4_cube_heatr3d.json    (n64, 0.57x)
  pyramid  solve3d/results/verify_stage_b4_pyramid_heatr3d.json (n80, 0.585x)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon, FancyArrowPatch

import style3d as st

CEILING = 250.0

# pinned results (shape, drive_frac, grid, dolfinx_peak, heatr3d_peak)
SHAPES = [
    ("SQUARE",  "0.34x", "n64", 235.03, 246.3),
    ("CUBE",    "0.57x", "n64", 235.08, 246.9),
    ("PYRAMID", "0.585x", "n80", 235.43, 244.2),
]


def _poly(iax, pts, lw=1.7, alpha=1.0):
    iax.add_patch(Polygon(pts, closed=True, fill=False, ec=st.ACCENT,
                          lw=lw, alpha=alpha, joinstyle="round"))


def draw_icon(iax, kind):
    """3-D shape glyph in an equal-box inset (0..1 coords, undistorted)."""
    iax.set_xlim(0, 1)
    iax.set_ylim(0, 1)
    iax.axis("off")
    if kind == "SQUARE":              # tall square PRISM (full-height column)
        _poly(iax, [(0.28, 0.05), (0.60, 0.05), (0.60, 0.72), (0.28, 0.72)])          # front
        _poly(iax, [(0.28, 0.72), (0.44, 0.90), (0.76, 0.90), (0.60, 0.72)], lw=1.3)  # top
        _poly(iax, [(0.60, 0.05), (0.76, 0.23), (0.76, 0.90), (0.60, 0.72)], lw=1.3)  # side
    elif kind == "CUBE":              # equal-sided iso cube
        _poly(iax, [(0.16, 0.12), (0.56, 0.12), (0.56, 0.52), (0.16, 0.52)])          # front
        _poly(iax, [(0.16, 0.52), (0.40, 0.76), (0.80, 0.76), (0.56, 0.52)], lw=1.3)  # top
        _poly(iax, [(0.56, 0.12), (0.80, 0.36), (0.80, 0.76), (0.56, 0.52)], lw=1.3)  # side
    elif kind == "PYRAMID":           # square-base pyramid in perspective
        b = [(0.14, 0.24), (0.50, 0.10), (0.86, 0.24), (0.50, 0.40)]                  # base diamond
        apex = (0.52, 0.90)
        _poly(iax, b, lw=1.2, alpha=0.75)
        _poly(iax, [b[0], apex, b[2]])          # front two faces
        iax.plot([b[3][0], apex[0]], [b[3][1], apex[1]], color=st.ACCENT, lw=1.2, alpha=0.75)


def main() -> None:
    fig = plt.figure(figsize=(13.0, 7.6), dpi=st.DPI)
    st.title_block(
        fig, "ONE METHOD, THREE SHAPES",
        "shaped, ceiling-respecting, fully-dense parts - both independent "
        "engines clear the 250 C degradation ceiling")
    ax = fig.add_axes([0.09, 0.20, 0.80, 0.60])
    ax.set_facecolor(st.BG)

    # over-ceiling zone (warning) + the ceiling line
    ax.axhspan(CEILING, 256, color=st.WARM, alpha=0.10, zorder=0)
    ax.axhline(CEILING, color=st.WARM, lw=2.2, ls="--", zorder=3)
    ax.text(2.62, CEILING + 0.5, "250 C degradation ceiling",
            color=st.WARM, fontsize=9.5, ha="right", va="bottom")

    xs = [0, 1, 2]
    for x, (name, drive, grid, dpk, hpk) in zip(xs, SHAPES):
        margin = CEILING - hpk
        off = hpk - dpk
        # dumbbell: dolfinx -> heatr3d
        ax.plot([x, x], [dpk, hpk], color=st.DIM, lw=1.4, zorder=2)
        ax.plot(x, dpk, "o", ms=11, color=st.ACCENT, zorder=4)          # solve engine
        ax.plot(x, hpk, "o", ms=13, color=st.MELT, zorder=5,
                markeredgecolor=st.FG, markeredgewidth=0.6)             # verify arbiter
        # margin bracket up to the ceiling (green = the safety room)
        ax.add_patch(FancyArrowPatch((x + 0.16, hpk), (x + 0.16, CEILING),
                     arrowstyle="<->", mutation_scale=8, color=st.GOOD, lw=1.3, zorder=4))
        ax.text(x + 0.22, (hpk + CEILING) / 2, f"{margin:.1f} C\nunder",
                color=st.GOOD, fontsize=8.5, va="center", ha="left")
        # value labels
        ax.text(x - 0.13, hpk, f"{hpk:.1f}", color=st.MELT, fontsize=9.5,
                ha="right", va="center", fontweight="bold")
        ax.text(x - 0.13, dpk, f"{dpk:.1f}", color=st.ACCENT, fontsize=9.5,
                ha="right", va="center")
        ax.text(x, dpk - 1.6, f"offset +{off:.1f}", color=st.DIM, fontsize=8, ha="center", va="top")
        # icon in a near-square inset (undistorted) in the empty over-ceiling band
        fx = (x - (-0.5)) / 3.2                      # data-x -> axes fraction
        iax = ax.inset_axes([fx - 0.028, 0.855, 0.056, 0.125])   # ~square in px
        draw_icon(iax, name)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\ndrive {d}  ·  verify {g}" for n, d, g, _, _ in SHAPES],
                       color=st.FG, fontsize=10)
    ax.set_ylabel("densified end-state peak temperature  [C]", color=st.FG, fontsize=10)
    ax.set_ylim(228, 255.5)
    ax.set_xlim(-0.5, 2.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=st.DIM)

    leg = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=st.ACCENT,
                   markersize=10, label="solve3d dolfinx FEM (picks drive + shapes dopant)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=st.MELT,
                   markeredgecolor=st.FG, markersize=11,
                   label="heatr3d voxel FDM (independent is_sendable verify)"),
        Patch(facecolor=st.GOOD, alpha=0.7, label="margin under the ceiling"),
    ]
    # legend inside the axes' empty lower-left, clear of the subtitle
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.005, 0.02),
              facecolor=st.PANEL, edgecolor=st.DIM, labelcolor=st.FG, fontsize=8.2,
              framealpha=0.92, handletextpad=0.6)

    fig.text(0.09, 0.085,
             "ALL THREE shippable cross-engine: both engines under 250 C, 3-6 C margin.  "
             "Drive is PER-PART (0.34 / 0.57 / 0.585x).",
             color=st.FG, fontsize=9.3)
    fig.text(0.09, 0.052,
             "Honest scope: cross-engine SIMULATION agreement, not a printed/measured part (physical DSC/TGA + density coupon pending).",
             color=st.DIM, fontsize=8)
    fig.text(0.09, 0.030,
             "Pyramid verified at n=80 (sharp-apex transfer aliasing) from a not-fully-converged solve. Offset +8.8 to +11.8 C: positive, headroom covers it, NOT a fixed constant.",
             color=st.DIM, fontsize=8)

    out = st.OUT / "fig6_family_generalization.png"
    fig.savefig(out, dpi=st.DPI, facecolor=st.BG)
    print("wrote", out, "| margins C:",
          [round(CEILING - h, 1) for *_, h in SHAPES])


if __name__ == "__main__":
    main()
