"""Render fig_hexagon_ungraded_vs_graded.png: static 16:9 two-sample compare.

LEFT: hexagon with UNIFORM (ungraded) dopant, at that arm's own optimal stop.
RIGHT: hexagon with the SOLVED graded 4 bpp map, at its own optimal stop.
Large panel: end-state temperature with the melt front (white dashed) and the
nominal outline (cyan). Small inset: the dopant map. Captions quote the
STORED library numbers (out_lib/hexagon.json U_uniform and A1_4bpp); the
compute stage c2h_hexagon.py gated the re-run fields against them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "deck_gifs/cache/c2h_hexagon.npz"
OUT = REPO / "deck_gifs/fig_hexagon_ungraded_vs_graded.png"


def main() -> None:
    d = np.load(CACHE)
    pm = np.asarray(d["part_mask"], dtype=bool)
    x, y = d["x"], d["y"]
    t_pc = float(d["t_pc_c"])

    r0, r1c, c0, c1 = style.crop_indices(pm, pad=10)
    ext = [x[c0] * 1e3, x[c1 - 1] * 1e3, y[r0] * 1e3, y[r1c - 1] * 1e3]
    pmc = pm[r0:r1c, c0:c1].astype(float)

    def cr(a):
        return np.asarray(a)[r0:r1c, c0:c1]

    T_uni = cr(d["T_uni_final"])
    T_sol = cr(d["T_a1"][-1])
    s_uni = np.ones_like(pmc)
    s_sol = cr(d["sat_a1"])

    sides = [
        ("UNGRADED", "uniform dopant, s = 1 everywhere", style.WARM,
         T_uni, s_uni, float(d["uni_J_stored"]), float(d["uni_IoU_stored"]),
         float(d["uni_stop_s_stored"])),
        ("GRADED", "solved single-pass 4 bpp dopant map", style.GOOD,
         T_sol, s_sol, float(d["a1_J_stored"]), float(d["a1_IoU_stored"]),
         float(d["a1_stop_s_stored"])),
    ]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
    gs = fig.add_gridspec(1, 2, left=0.045, right=0.955, top=0.775,
                          bottom=0.135, wspace=0.18)

    fig.suptitle("same part, same power: the graded map turns a hot-cored "
                 "blob into the hexagon", fontsize=16, color=style.FG, y=0.955)
    fig.text(0.5, 0.885,
             "hexagon, end-state temperature at each arm's own optimal stop; "
             "white dashed = melt front, cyan = nominal part",
             ha="center", fontsize=10, color=style.DIM)

    for k, (tag, sub, col, T, s_map, J, iou, stop_s) in enumerate(sides):
        ax = fig.add_subplot(gs[0, k])
        xc = 0.27 + 0.455 * k
        fig.text(xc, 0.835, tag, ha="center", fontsize=14, color=col)
        im = ax.imshow(T, origin="lower", extent=ext, cmap=style.CMAP_T,
                       vmin=25.0, vmax=220.0, interpolation="bilinear")
        ax.contour(T, levels=[t_pc], colors="white", linewidths=1.2,
                   linestyles="--", extent=ext, origin="lower")
        ax.contour(pmc, levels=[0.5], colors=style.ACCENT, linewidths=1.3,
                   extent=ext, origin="lower")
        style.field_axes(ax)
        style.slim_colorbar(fig, im, ax, "temperature (deg C)")

        # dopant-map inset, lower-left corner of the panel (square in inches)
        bb = ax.get_position()
        wi = 0.058
        hi = wi * 12.8 / 7.2
        axi = fig.add_axes([bb.x0 + 0.005, bb.y0 + 0.008, wi, hi])
        axi.imshow(s_map, origin="lower", cmap=style.CMAP_SAT, vmin=0, vmax=1,
                   interpolation="nearest")
        axi.contour(pmc, levels=[0.5], colors=style.ACCENT, linewidths=0.4)
        axi.set_xticks([])
        axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_color(style.FG)
            sp.set_linewidth(0.8)
        axi.text(0.5, 1.06, "dopant map", transform=axi.transAxes,
                 ha="center", va="bottom", fontsize=7, color=style.FG,
                 bbox=dict(facecolor=style.BG, edgecolor="none", pad=1.2))

        fig.text(xc, 0.085,
                 f"{sub}\nat its stop {stop_s:.0f} s:  J_phi {J:.1f}   "
                 f"IoU {iou:.4f}", ha="center", fontsize=10.5, color=style.FG)

    fig.text(0.5, 0.022,
             "grid 120; melt-region framing; stored numbers: "
             "fgm_solve_campaign/out_lib/hexagon.json arms U_uniform and "
             "A1_4bpp; fields re-run and gated against the stored stops",
             ha="center", fontsize=8, color=style.DIM)

    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
