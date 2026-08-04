"""Deck figures for the Phase E CUBE arm.

    .venv312/bin/python -m solve3d.phase_e.render_deck_cube

Two figures, both from artifacts on disk:

  fig_deck_cube_three_arms.png
      3-D quarter cutaways of the three arms at one camera, coloured by MELT
      FRACTION phi (the same real full-volume field the pyramid cutaway uses:
      T_read is the complete nodal read state, phi is pointwise in T), plus a
      difference row against the uniform baseline so the heuristic's failure
      and the solve's gain are visible as fields rather than as numbers.

  fig_deck_cube_solved_map.png
      what the solve actually did to the dopant: z slabs of the delivered map,
      the mean-saturation profile through build height, and the numbers.

Rendering only. No physics is run here.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from solve3d.phase_e import render_deck_cutaway3d as C
from solve3d.phase_e import render_deck_loop3d as L

st = C.st
RESULTS = C.RESULTS
MM = C.MM

ARMS = (("uniform_baseline", "uniform"),
        ("heuristic_grading_law", "hand-built grading law"),
        ("solve_filter_only", "solve (filter-only arm)"))
CMAP_DIFF = L.dark_centered("#4da3ff", "#ff8a65", "#151a21")


def cube_wire(ax, h_mm: float):
    """Nominal cube: twelve edges."""
    c = st.ACCENT
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot([sx * h_mm] * 2, [sy * h_mm] * 2, [-h_mm, h_mm],
                    color=c, lw=1.3, alpha=0.95)
    for zz in (-h_mm, h_mm):
        ax.plot([-h_mm, h_mm, h_mm, -h_mm, -h_mm],
                [-h_mm, -h_mm, h_mm, h_mm, -h_mm], [zz] * 5,
                color=c, lw=1.3, alpha=0.95)


def cutaway(fig, rect, vol, mask_f, ax_m, cmap, norm, half_mm, r=11.0):
    ax = fig.add_axes(rect, projection="3d")
    h = float(ax_m[1] - ax_m[0])
    tri, spts, nrm, _v, _f = C.surface(mask_f, h, ax_m[0])
    rgba, _ = C.face_colors(vol, spts, ax_m[0], h, nrm, cmap=cmap, norm=norm)
    ax.add_collection3d(Poly3DCollection(tri * MM, facecolors=rgba,
                                         edgecolors="none"))
    cube_wire(ax, half_mm)
    st.dark_3d_axes(ax)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.line.set_color((0, 0, 0, 0))
        a.set_ticks([])
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1), zoom=1.30)
    ax.view_init(elev=C.ELEV, azim=C.AZIM)
    return ax


def load():
    z = np.load(RESULTS / "vol_cube.npz")
    doc = json.loads((RESULTS / "phase_e_cube.json").read_text())
    return z, doc


# fig_three_arms lived here. It is SUPERSEDED by render_deck_melt_body.py and
# was deleted rather than left in place, because it wrote the same filename:
# dead code that silently overwrites a committed deck figure is a trap.


def fig_solved_map(z, doc) -> str:
    ax_m = np.asarray(z["axis_m"], float)
    inside = np.asarray(z["inside"], bool)
    idx = np.asarray(z["design_nearest_index"], np.int64)
    gz = np.load(RESULTS / "map_cube_solve_filter_only.npz")
    s_map = np.asarray(gz["s_map"], float)
    cen = np.asarray(gz["centroids"], float)
    vol = np.asarray(gz["volumes"], float)
    half_mm = float(z["nominal_base_side_m"]) / 2.0 * MM

    sol = doc["arms"]["solve_filter_only"]
    uni = doc["arms"]["uniform_baseline"]

    fig = plt.figure(figsize=(15.0, 7.6), dpi=200)
    fig.patch.set_facecolor(st.BG)
    fig.text(0.035, 0.975, "what the solve did to the dopant map: cube, "
             f"{s_map.size} design cells",
             fontsize=14.5, color=st.FG, ha="left", va="top")

    # --- z slabs of the delivered map --- #
    zc = cen[:, 2]
    edges = np.linspace(zc.min(), zc.max(), 5)
    n_dop = Normalize(0.0, 1.0)
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (zc >= lo) & (zc <= hi if i == 3 else zc < hi)
        ax = fig.add_axes([0.035 + 0.155 * i, 0.500, 0.140, 0.330])
        ax.set_facecolor(st.BG)
        # volume-weighted bin average, not a scatter: at this cell count a
        # scatter of the design cells reads as speckle rather than as a field
        nb = 26
        eg = np.linspace(-half_mm, half_mm, nb + 1)
        num, _, _ = np.histogram2d(cen[m, 0] * MM, cen[m, 1] * MM, bins=[eg, eg],
                                   weights=s_map[m] * vol[m])
        den, _, _ = np.histogram2d(cen[m, 0] * MM, cen[m, 1] * MM, bins=[eg, eg],
                                   weights=vol[m])
        img = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        cm = L.CMAP_DOP.copy()
        cm.set_bad(st.BG)          # empty bins blend into the panel, not black
        ax.imshow(img.T, origin="lower", cmap=cm, norm=n_dop,
                  extent=[-half_mm, half_mm, -half_mm, half_mm],
                  interpolation="bilinear")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(st.DIM)
        w = float(np.average(s_map[m], weights=vol[m]))
        ax.set_title(f"z {lo*MM:+.1f} to {hi*MM:+.1f} mm\nmean sat {w:.2f}",
                     fontsize=10.5, color=st.DIM, pad=6)

    cax = fig.add_axes([0.075, 0.435, 0.480, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=n_dop, cmap=L.CMAP_DOP),
                      cax=cax, orientation="horizontal")
    cb.set_label("dopant saturation (solved map)", color=st.FG, fontsize=11)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(colors=st.DIM, labelsize=9)
    cb.outline.set_edgecolor(st.DIM)

    # --- mean saturation through build height --- #
    ax = fig.add_axes([0.075, 0.125, 0.330, 0.230])
    ax.set_facecolor(st.BG)
    nb = 40
    be = np.linspace(zc.min(), zc.max(), nb + 1)
    bc, bv = [], []
    for i in range(nb):
        m = (zc >= be[i]) & (zc < be[i + 1])
        if m.sum() > 20:
            bc.append(0.5 * (be[i] + be[i + 1]) * MM)
            bv.append(float(np.average(s_map[m], weights=vol[m])))
    ax.plot(bc, bv, color="#ffb74d", lw=2.2)
    ax.axhline(1.0, color=st.DIM, ls="--", lw=1.0)
    ax.set_ylim(min(bv) - 0.05, 1.10)
    ax.text(bc[0], 1.015, "uniform arm = 1.00 everywhere", fontsize=9.5,
            color=st.DIM, va="bottom")
    ax.set_xlabel("build height z (mm)", color=st.DIM, fontsize=10.5)
    ax.set_ylabel("mean saturation", color=st.DIM, fontsize=10.5)
    ax.tick_params(colors=st.DIM, labelsize=9.5)
    for sp in ax.spines.values():
        sp.set_color(st.DIM)
    ax.grid(alpha=0.12, color=st.DIM)

    # --- 3-D cutaway of the map itself --- #
    mask_f = C.cutaway_mask(inside, ax_m)
    v_map = L.design_volume(s_map, idx, inside, 0.0)
    cutaway(fig, [0.430, 0.105, 0.250, 0.360], v_map, mask_f, ax_m,
            L.CMAP_DOP, n_dop, half_mm)
    fig.text(0.555, 0.088, "the same map in the round", fontsize=11,
             color=st.DIM, ha="center", va="top")

    # --- numbers --- #
    x0, xu, xsv, xr = 0.672, 0.858, 0.932, 0.990
    fig.text(xu, 0.760, "uniform", fontsize=12, color=st.DIM, ha="right")
    fig.text(xsv, 0.760, "solved", fontsize=12, color=st.GOOD, ha="right")
    rows = [("mean melt fraction", f"{uni['part_mean_phi']:.2f}",
             f"{sol['part_mean_phi']:.2f}",
             "x%.2f" % (sol["part_mean_phi"] / uni["part_mean_phi"])),
            ("melt-region IoU", f"{uni['iou_phi0p9']:.3f}",
             f"{sol['iou_phi0p9']:.3f}",
             "x%.2f" % (sol["iou_phi0p9"] / uni["iou_phi0p9"])),
            ("melt-front distance", f"{uni['front_ssd_mm']:.2f} mm",
             f"{sol['front_ssd_mm']:.2f} mm",
             "%+.0f%%" % (100 * (sol["front_ssd_mm"] / uni["front_ssd_mm"] - 1))),
            ("below the melt floor", f"{uni['in_bounds_below_floor_fraction']:.3f}",
             f"{sol['in_bounds_below_floor_fraction']:.3f}",
             "%+.0f%%" % (100 * (sol["in_bounds_below_floor_fraction"]
                                 / uni["in_bounds_below_floor_fraction"] - 1))),
            ("objective J", f"{uni['J_asymmetric']:.3e}",
             f"{sol['J_asymmetric']:.3e}",
             "%+.0f%%" % (100 * (sol["J_asymmetric"] / uni["J_asymmetric"] - 1)))]
    for i, (nm, a, b, rr) in enumerate(rows):
        y = 0.690 - 0.072 * i
        fig.text(x0, y, nm, fontsize=11.5, color=st.FG, ha="left")
        fig.text(xu, y, a, fontsize=11.5, color=st.DIM, ha="right")
        fig.text(xsv, y, b, fontsize=11.5, color=st.GOOD, ha="right")
        fig.text(xr, y, rr, fontsize=11.5, color="#ffb74d", ha="right")

    fig.text(0.5, 0.020, "solved by 3-D adjoint (filter-only arm), "
             "budget-limited and still descending; simulation-only",
             fontsize=11, color=st.DIM, ha="center")

    out = RESULTS / "fig_deck_cube_solved_map.png"
    fig.savefig(out, dpi=200, facecolor=st.BG)
    plt.close(fig)
    return str(out)


def main() -> int:
    z, doc = load()
    print("wrote", fig_solved_map(z, doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
