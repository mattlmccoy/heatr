"""Figure 2: The correction that mattered (EQS-02).

Before/after volumetric-heating topology on the 20 mm circle, rendered as 3-D
height fields of the unit-mean Q_rf pattern at the D1 evaluation points.
Legacy stencil (gradient taken across the material interface): a spiked
surface-hot rim, max 12.3x the in-part mean. Masked stencil (EQS-02
corrected): near-uniform, max 1.9x; the interior absorbed power rises 2.59x
and the thermal topology flips to interior-hot. Both fields carry identical total
absorbed power (D1: power identity rel diff 0.0), so the correction MOVES
energy from the rim to the interior; it does not add any.

Data: heatr3d_d1_spike/d1_circle_coarse.npz (pts, q_heatr3d legacy,
q_heatr3d_maskgrad corrected; 812 mid-plane points, n=64-matched, the same
pair EQS02_IMPACT.md reports). Saved artifact only; no physics is run.
Scope: the points are one z plane of the extruded circle (the field is
z-invariant, max|Ez|/mean|E| = 6.5e-7); the 3-D relief is a linear
display of those 812 points, height =
local Q over in-part mean, displayed on a refined triangulation of those
points. Color is CLIPPED at 3x the mean so the corrected panel's topology is
visible; the heights are unclipped and true.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.tri as mtri

import style3d as st

MM = 1000.0
R_PART = 10.0
COLOR_CLIP = 3.0
NG = 240


def load_pair():
    d = np.load(st.REPO / "heatr3d_d1_spike" / "d1_circle_coarse.npz")
    pts = np.asarray(d["pts"]) * MM
    q_leg = np.asarray(d["q_heatr3d"])
    q_cor = np.asarray(d["q_heatr3d_maskgrad"])
    return pts[:, 0], pts[:, 1], q_leg / q_leg.mean(), q_cor / q_cor.mean()


def refined(x, y, q):
    """Refine the scattered-point triangulation for a smooth relief."""
    tri = mtri.Triangulation(x, y)
    ref = mtri.UniformTriRefiner(tri)
    tri_f, q_f = ref.refine_field(q, subdiv=2)
    return tri_f, np.clip(q_f, 0.0, None)


def panel(ax, tri_f, q_f, zmax, label, peak, peak_color):
    norm = colors.Normalize(vmin=0.0, vmax=COLOR_CLIP)
    ax.plot_trisurf(tri_f, q_f, cmap=st.CMAP_Q, norm=norm,
                    linewidth=0, antialiased=True, shade=False)
    t = np.linspace(0, 2 * np.pi, 181)
    ax.plot(R_PART * np.cos(t), R_PART * np.sin(t), np.zeros_like(t),
            color=st.ACCENT, lw=1.3, alpha=0.95)
    st.dark_3d_axes(ax)
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((1, 1, 0.9))
    ax.view_init(elev=26, azim=-52)
    ax.set_title(label, fontsize=13, color=st.FG, pad=-8)
    ax.text2D(0.5, 0.10, peak, transform=ax.transAxes, ha="center", va="top",
              fontsize=12, color=peak_color, fontweight="bold")


def main() -> None:
    x, y, q_leg, q_cor = load_pair()
    zmax = float(q_leg.max()) * 1.02

    fig = plt.figure(figsize=(12.6, 7.2), dpi=st.DPI)
    ax1 = fig.add_axes([0.00, 0.06, 0.50, 0.76], projection="3d")
    ax2 = fig.add_axes([0.46, 0.06, 0.50, 0.76], projection="3d")

    panel(ax1, *refined(x, y, q_leg), zmax,
          "BEFORE: gradient crosses the interface",
          f"rim spikes to {q_leg.max():.1f}x the mean\n"
          "the skin steals 74% of the power", st.WARM)
    panel(ax2, *refined(x, y, q_cor), zmax,
          "AFTER: masked stencil (EQS-02)",
          f"near-uniform, max {q_cor.max():.1f}x the mean\n"
          "interior absorbed power up 2.59x", st.GOOD)

    st.title_block(fig, "THE CORRECTION THAT MATTERED",
                   "RF heating density Q_rf on the 20 mm circle, height = "
                   "local Q over in-part mean, same total power in both")
    sm = cm.ScalarMappable(norm=colors.Normalize(0, COLOR_CLIP),
                           cmap=st.CMAP_Q)
    cax = fig.add_axes([0.935, 0.16, 0.012, 0.55])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8, colors=st.DIM, length=2)
    cb.outline.set_edgecolor(st.DIM)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Q_rf / mean (color clipped at 3, heights true)",
                 fontsize=8.5, color=st.DIM)
    fig.text(0.975, 0.012,
             "saved D1 field pair, mid plane of the extruded circle, "
             "n=64-matched  |  identical absorbed power: energy is moved, "
             "not added",
             fontsize=8.5, color=st.DIM, ha="right")

    out = st.OUT / "fig2_eqs02_correction.png"
    fig.savefig(out, dpi=st.DPI)
    print("wrote", out,
          "| legacy max/mean %.2f | corrected max/mean %.2f"
          % (q_leg.max(), q_cor.max()))


if __name__ == "__main__":
    main()
