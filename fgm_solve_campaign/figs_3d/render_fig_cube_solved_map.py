"""Dissertation figure: the solved dopant map on the Phase E cube, light style.

    ./.venv312/bin/python fgm_solve_campaign/figs_3d/render_fig_cube_solved_map.py

(a to d) four z slabs of the delivered map as volume-weighted bin averages,
each labelled with its own mean saturation; (e) mean saturation against build
height, against the uniform arm's s = 1 reference; (f) the same map as a
quarter cutaway; (g) the metrics table, uniform against solved.

Reads only stored artifacts (see cube_light_common). Rendering only: no solve,
no forward evaluation, no solve3d source change. Every number in the table is
computed at render time from phase_e_cube.json, and the reproduction gate runs
before anything is drawn.
"""
from __future__ import annotations

import numpy as np

import cube_light_common as K

C, L, MB = K.phase_helpers()
K.use_light_style()

import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import Normalize                          # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection          # noqa: E402

OUT = K.HERE / "fig_cube_solved_map.png"
CMAP = "viridis"
ELEV, AZIM = 26.0, 34.0


def slab_panels(fig, s, cen, vol, half_mm, norm):
    zc = cen[:, 2]
    edges = np.linspace(zc.min(), zc.max(), 5)
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (zc >= lo) & (zc <= hi if i == 3 else zc < hi)
        ax = fig.add_axes([0.045 + 0.148 * i, 0.650, 0.132, 0.245])
        nb = 26
        eg = np.linspace(-half_mm, half_mm, nb + 1)
        num, _, _ = np.histogram2d(cen[m, 0] * K.MM, cen[m, 1] * K.MM,
                                   bins=[eg, eg], weights=s[m] * vol[m])
        den, _, _ = np.histogram2d(cen[m, 0] * K.MM, cen[m, 1] * K.MM,
                                   bins=[eg, eg], weights=vol[m])
        img = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        cm = plt.get_cmap(CMAP).copy()
        cm.set_bad(K.BG)
        ax.imshow(img.T, origin="lower", cmap=cm, norm=norm,
                  extent=[-half_mm, half_mm, -half_mm, half_mm],
                  interpolation="bilinear")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(K.DIM)
        w = float(np.average(s[m], weights=vol[m]))
        ax.set_title("(%s)  %+.1f to %+.1f\nmean $s$ = %.2f"
                     % ("abcd"[i], lo * K.MM, hi * K.MM, w),
                     fontsize=6.6, color=K.FG, pad=3)
        if i == 0:
            ax.set_ylabel("y", color=K.DIM, fontsize=7.0, labelpad=2)
        ax.set_xlabel("x", color=K.DIM, fontsize=7.0, labelpad=2)
    fig.text(0.045 + 0.148 * 1.5 + 0.066, 0.932, "z slab [mm]", fontsize=7.2,
             color=K.DIM, ha="center", va="baseline")


def profile_panel(fig, s, cen, vol):
    ax = fig.add_axes([0.070, 0.145, 0.285, 0.300])
    zc = cen[:, 2]
    nb = 40
    be = np.linspace(zc.min(), zc.max(), nb + 1)
    bc, bv = [], []
    for i in range(nb):
        m = (zc >= be[i]) & (zc < be[i + 1])
        if m.sum() > 20:
            bc.append(0.5 * (be[i] + be[i + 1]) * K.MM)
            bv.append(float(np.average(s[m], weights=vol[m])))
    ax.plot(bc, bv, color=K.C_SOL, lw=1.6, label="solved map")
    ax.axhline(1.0, color=K.C_UNI, ls="--", lw=1.1,
               label="uniform arm, $s = 1$")
    ax.set_ylim(min(bv) - 0.06, 1.09)
    ax.set_xlabel("build height z [mm]")
    ax.set_ylabel("mean dopant saturation $s$")
    ax.legend(loc="lower center", frameon=False, ncol=2, fontsize=6.6,
              handlelength=1.8, columnspacing=1.2,
              bbox_to_anchor=(0.5, -0.02))
    ax.set_title("(e) saturation vs build height", fontsize=7.6,
                 color=K.FG, pad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return float(np.mean(bv))


def cutaway_panel(fig, s, idx, inside, ax_m, half_mm, norm):
    ax = fig.add_axes([0.385, 0.120, 0.235, 0.320], projection="3d")
    h = float(ax_m[1] - ax_m[0])
    mask_f = C.cutaway_mask(inside, ax_m)
    v_map = L.design_volume(s, idx, inside, 0.0)
    tri, spts, nrm, _v, _f = C.surface(mask_f, h, ax_m[0])
    rgba, _ = C.face_colors(v_map, spts, ax_m[0], h, nrm,
                            cmap=plt.get_cmap(CMAP), norm=norm)
    ax.add_collection3d(Poly3DCollection(tri * K.MM, facecolors=rgba,
                                         edgecolors="none"))
    K.cube_wire(ax, half_mm)
    K.light_3d_axes(ax)
    r = 1.12 * half_mm
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1), zoom=1.30)
    ax.view_init(elev=ELEV, azim=AZIM)
    fig.text(0.502, 0.452, "(f) map in the round", fontsize=7.6,
             color=K.FG, ha="center", va="baseline")


def table_panel(fig, uni, sol):
    rows = [("mean melt fraction", "%.2f" % uni["part_mean_phi"],
             "%.2f" % sol["part_mean_phi"],
             "$\\times$%.2f" % (sol["part_mean_phi"] / uni["part_mean_phi"])),
            ("melt-region IoU, $\\varphi \\geq 0.9$",
             "%.3f" % uni["iou_phi0p9"], "%.3f" % sol["iou_phi0p9"],
             "$\\times$%.2f" % (sol["iou_phi0p9"] / uni["iou_phi0p9"])),
            ("melt-front distance [mm]", "%.2f" % uni["front_ssd_mm"],
             "%.2f" % sol["front_ssd_mm"],
             "%+.0f%%" % (100 * (sol["front_ssd_mm"] / uni["front_ssd_mm"] - 1))),
            ("below the melt floor",
             "%.3f" % uni["in_bounds_below_floor_fraction"],
             "%.3f" % sol["in_bounds_below_floor_fraction"],
             "%+.0f%%" % (100 * (sol["in_bounds_below_floor_fraction"]
                                 / uni["in_bounds_below_floor_fraction"] - 1))),
            ("objective $J$", "%.3e" % uni["J_asymmetric"],
             "%.3e" % sol["J_asymmetric"],
             "%+.0f%%" % (100 * (sol["J_asymmetric"] / uni["J_asymmetric"] - 1)))]
    x0, xu, xs, xr = 0.628, 0.862, 0.935, 0.998
    y_top = 0.412
    fig.text(x0, y_top + 0.048, "(g) shape fidelity, each arm at its own stop",
             fontsize=7.6, color=K.FG, ha="left")
    fig.text(xu, y_top, "uniform", fontsize=7.0, color=K.C_UNI, ha="right")
    fig.text(xs, y_top, "solved", fontsize=7.0, color=K.C_SOL, ha="right")
    fig.text(xr, y_top, "change", fontsize=7.0, color=K.FG, ha="right")
    fig.add_artist(plt.Line2D([x0, xr], [y_top - 0.016] * 2, color=K.DIM,
                              lw=0.6, transform=fig.transFigure))
    for i, (nm, a, b, rr) in enumerate(rows):
        y = y_top - 0.050 - 0.052 * i
        fig.text(x0, y, nm, fontsize=6.8, color=K.FG, ha="left")
        fig.text(xu, y, a, fontsize=6.9, color=K.C_UNI, ha="right")
        fig.text(xs, y, b, fontsize=6.9, color=K.C_SOL, ha="right")
        fig.text(xr, y, rr, fontsize=6.9, color=K.FG, ha="right",
                 fontweight="bold" if i == 4 else "normal")


def main() -> int:
    d = K.load()
    K.gate(d)
    doc, z, mp = d["doc"], d["vol"], d["map"]
    s = np.asarray(mp["s_map"], float)
    cen = np.asarray(mp["centroids"], float)
    vol = np.asarray(mp["volumes"], float)
    ax_m = np.asarray(z["axis_m"], float)
    inside = np.asarray(z["inside"], bool)
    idx = np.asarray(z["design_nearest_index"], np.int64)
    half_mm = d["half_mm"]
    uni = doc["arms"]["uniform_baseline"]
    sol = doc["arms"]["solve_filter_only"]
    norm = Normalize(0.0, 1.0)

    fig = plt.figure(figsize=(6.9, 4.85), dpi=300)
    slab_panels(fig, s, cen, vol, half_mm, norm)

    cax = fig.add_axes([0.115, 0.572, 0.395, 0.017])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="horizontal")
    cb.set_label("dopant saturation $s$ of the solved map", fontsize=7.4,
                 color=K.FG, labelpad=2)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(colors=K.DIM, labelsize=6.6)
    cb.outline.set_edgecolor(K.DIM)
    cb.outline.set_linewidth(0.6)

    fig.text(0.632, 0.900,
             f"delivered map, {s.size:,} design cells\n"
             f"volume-weighted mean $s$ = {np.average(s, weights=vol):.3f}\n"
             f"range {s.min():.3f} to {s.max():.3f}",
             fontsize=7.2, color=K.FG, ha="left", va="top", linespacing=1.6)
    fig.text(0.632, 0.775,
             "The solve empties both z ends and the two\nx-normal faces. "
             "The map is symmetric about\nthe mid-plane, which nothing in "
             "the objective\nenforces.",
             fontsize=7.0, color=K.DIM, ha="left", va="top", linespacing=1.5)

    profile_panel(fig, s, cen, vol)
    cutaway_panel(fig, s, idx, inside, ax_m, half_mm, norm)
    table_panel(fig, uni, sol)

    fig.text(0.5, 0.048,
             "Solve mesh %s cells, %s design cells, coupling off; each arm is "
             "read at its own envelope stop."
             % (f"{doc['arms']['_mesh']['n_cells']:,}"
                if "n_cells" in doc["arms"].get("_mesh", {}) else "105,853",
                f"{s.size:,}"),
             fontsize=6.8, color=K.DIM, ha="center", va="top")
    fig.text(0.5, 0.026,
             "Direct solve is the 3-D adjoint filter-only arm, stopped by its "
             "12-evaluation budget while still descending. Simulation only.",
             fontsize=6.8, color=K.DIM, ha="center", va="top")

    fig.savefig(OUT, dpi=300, facecolor=K.BG)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
