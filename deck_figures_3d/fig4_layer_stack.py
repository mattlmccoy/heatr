"""Figure 4: One 3-D simulation, read layer by layer.

Exploded z-stack of the melt-fraction field phi on the five exported z planes
of the square arm (dolfinx engine, shared 0.15 mm evaluation grid), each
plane with its phi = 0.9 melt front and the nominal 20 mm outline. The
visual bridge from the 3-D simulation to layerwise printing: every layer
the printer will rasterize is a slice of one simulated field. Forward
field only; the 3-D dopant solve is in progress (Phase C).

Data: solve3d/results/eval_dolfinx_square_off.npz (T: 5 z planes at
z = -20, -10, 0, +10, +20 mm, 200x200, 0.15 mm pixels; committed Phase A
close-out export). phi via the shared phase_fraction_phi. The vertical
spacing in the render is exploded for legibility; the real planes are 10 mm
apart in a 40 mm part. Rendering only; no physics is run.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import style3d as st

MM = 1000.0
Z_MM = [-20, -10, 0, 10, 20]
CROP = 11.5          # display crop [mm]
GAP = 13.0           # exploded vertical gap between layers [render units]


def load_planes():
    d = np.load(st.REPO / "solve3d" / "results" / "eval_dolfinx_square_off.npz")
    T = np.asarray(d["T"], dtype=float)
    return st.phase_fraction_phi(T)


def eval_axes():
    h = 2.0 * 0.015 / 200
    c = ((np.arange(200) + 0.5) * h - 0.015) * MM
    return c


def main() -> None:
    phi = load_planes()
    c = eval_axes()
    keep = np.abs(c) <= CROP
    cc = c[keep]
    X, Y = np.meshgrid(cc, cc, indexing="ij")
    cmap = plt.get_cmap(st.CMAP_PHI)
    norm = colors.Normalize(0.0, 1.0)

    fig = plt.figure(figsize=(12.6, 7.6), dpi=st.DPI)
    ax = fig.add_axes([0.04, -0.03, 0.74, 0.95], projection="3d")
    ax.computed_zorder = False        # manual painter order, bottom to top

    for k in range(5):
        P = phi[k][np.ix_(keep, keep)]
        zoff = k * GAP
        fc = cmap(norm(P))
        fc[..., 3] = 0.96
        zo = 10 * k
        ax.plot_surface(X, Y, np.full_like(X, zoff), facecolors=fc,
                        rstride=2, cstride=2, linewidth=0, shade=False,
                        antialiased=False, zorder=zo)
        # nominal outline
        ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10],
                [zoff + 0.3] * 5, color=st.ACCENT, lw=1.2, alpha=0.95,
                zorder=zo + 2)
        # melt front on the layer
        cs = plt.figure().add_subplot().contour(cc, cc, P.T, levels=[0.9])
        plt.close(plt.gcf())
        for path in cs.get_paths():
            v = path.vertices
            ax.plot(v[:, 0], v[:, 1], np.full(len(v), zoff + 0.5),
                    color=st.FG, lw=1.1, ls=(0, (3, 2)), alpha=0.95,
                    zorder=zo + 3)
        zlab = f"{Z_MM[k]:+d}" if Z_MM[k] else "0"
        ax.text(-CROP - 2.0, -CROP - 2.0, zoff, f"z = {zlab} mm",
                fontsize=9.5, color=st.DIM, ha="right", va="center",
                zorder=zo + 4)

    st.dark_3d_axes(ax)
    ax.set_xlim(-CROP, CROP)
    ax.set_ylim(-CROP, CROP)
    ax.set_zlim(-2, 4 * GAP + 2)
    ax.set_box_aspect((1, 1, 1.35))
    ax.view_init(elev=33, azim=-60)

    st.title_block(fig, "ONE 3-D SIMULATION, READ LAYER BY LAYER",
                   "melt fraction phi on the five exported z planes of the "
                   "40 mm square arm, FEM engine, shared 0.15 mm grid")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.86, 0.20, 0.013, 0.50])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8, colors=st.DIM, length=2)
    cb.outline.set_edgecolor(st.DIM)
    cb.outline.set_linewidth(0.4)
    cb.set_label("melt fraction phi", fontsize=9, color=st.DIM)
    fig.text(0.86, 0.76, "dash  phi=0.9 front", fontsize=9.5, color=st.FG)
    fig.text(0.86, 0.73, "cyan  nominal 20 mm", fontsize=9.5, color=st.ACCENT)
    fig.text(0.975, 0.048,
             "z spacing exploded for display (real planes 10 mm apart)",
             fontsize=8.5, color=st.DIM, ha="right")
    fig.text(0.975, 0.022,
             "forward field, the 3-D dopant solve is in progress  |  "
             "every printed layer is a slice of one simulated field",
             fontsize=8.5, color=st.DIM, ha="right")

    out = st.OUT / "fig4_layer_stack.png"
    fig.savefig(out, dpi=st.DPI)
    print("wrote", out, "| per-plane melt frac:",
          [round(float((phi[k] >= 0.9).mean()), 4) for k in range(5)])


if __name__ == "__main__":
    main()
