"""Render gif_solve_vs_inversion_hexagon.gif: the head-to-head, on the hexagon.

Left, INVERT: the stored best historical proportional-inverse mask, one static
guess, then its melt front runs to its own stop.
Right, SOLVE: phase A shows a fresh filtered adjoint solve's iterates purely as
a visualization of the process; phase B runs the STORED solved 4 bpp
deliverable map (out_lib/hexagon_maps.npz A1_4bpp) to its own stop.
End-state captions quote the STORED library numbers for both sides; the
compute stage gated the re-runs against them. The hexagon is a genuine win,
so the footer makes the head-to-head claim, grid-120 qualified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "deck_gifs/cache/c2h_hexagon.npz"
OUT = REPO / "deck_gifs/gif_solve_vs_inversion_hexagon.gif"
FPS = 12
F_MAP = 48       # phase A: map evolution frames
F_MELT = 84      # phase B: melt evolution frames
F_HOLD = 26      # final hold


def main() -> None:
    d = np.load(CACHE)
    pm = np.asarray(d["part_mask"], dtype=bool)
    x, y = d["x"], d["y"]
    dt = float(d["dt_s"])
    t_pc = float(d["t_pc_c"])

    r0, r1c, c0, c1 = style.crop_indices(pm, pad=10)
    ext = [x[c0] * 1e3, x[c1 - 1] * 1e3, y[r0] * 1e3, y[r1c - 1] * 1e3]
    pmc = pm[r0:r1c, c0:c1].astype(float)

    def cr(a):
        return np.asarray(a)[r0:r1c, c0:c1]

    maps_ev = d["maps_evals"]
    j_ev = d["j_evals"]
    n_ev = len(j_ev)
    T_h, T_s = d["T_hist"], d["T_a1"]
    steps_h, steps_s = d["steps_hist"], d["steps_a1"]

    # stored library end-state numbers (the captions quote these)
    hist_J = float(d["hist_J_stored"])
    hist_iou = float(d["hist_IoU_stored"])
    solve_J = float(d["a1_J_stored"])
    solve_iou = float(d["a1_IoU_stored"])
    stop_h = float(d["hist_stop_s_stored"])
    stop_s = float(d["a1_stop_s_stored"])
    dj_pct = 100.0 * float(d["dJ_rel_stored"])

    n_frames = F_MAP + F_MELT + F_HOLD
    fig = plt.figure(figsize=(11.6, 6.6), dpi=style.DPI)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.42],
                          left=0.04, right=0.985, top=0.80, bottom=0.135,
                          wspace=0.24)
    ax_L = fig.add_subplot(gs[0, 0])
    ax_R = fig.add_subplot(gs[0, 1])

    fig.suptitle("two ways to a dopant map  (hexagon)", fontsize=14,
                 color=style.FG, y=0.965)
    fig.text(0.235, 0.885, "INVERT", ha="center", fontsize=12, color=style.WARM)
    fig.text(0.235, 0.845, "best stored proportional-inverse mask, fixed",
             ha="center", fontsize=8.5, color=style.DIM)
    fig.text(0.615, 0.885, "SOLVE", ha="center", fontsize=12, color=style.GOOD)
    fig.text(0.615, 0.845, "adjoint optimization through the physics",
             ha="center", fontsize=8.5, color=style.DIM)

    im_L = ax_L.imshow(cr(d["sat_hist"]), origin="lower", extent=ext,
                       cmap=style.CMAP_SAT, vmin=0, vmax=1,
                       interpolation="nearest")
    im_R = ax_R.imshow(cr(maps_ev[0]), origin="lower", extent=ext,
                       cmap=style.CMAP_SAT, vmin=0, vmax=1,
                       interpolation="nearest")
    cbs = []
    for ax, im in ((ax_L, im_L), (ax_R, im_R)):
        style.field_axes(ax)
        ax.contour(pmc, levels=[0.5], colors=style.ACCENT, linewidths=1.0,
                   extent=ext, origin="lower")
        cbs.append(style.slim_colorbar(fig, im, ax, "dopant saturation"))

    # objective panel in the right sidebar (process visualization)
    ax_J = fig.add_axes([0.845, 0.50, 0.135, 0.26])
    ax_J.set_facecolor(style.PANEL)
    ax_J.tick_params(labelsize=6, colors=style.DIM, length=2)
    for s in ax_J.spines.values():
        s.set_color(style.DIM)
        s.set_linewidth(0.5)
    j_best = np.minimum.accumulate(j_ev)
    y_cap = float(j_ev[0]) * 1.15
    ax_J.set_xlim(1, n_ev)
    ax_J.set_ylim(0, y_cap)
    ax_J.set_xlabel("evaluation", fontsize=6.5, color=style.DIM, labelpad=1)
    ax_J.set_ylabel("best J_phi", fontsize=6.5, color=style.DIM, labelpad=1)
    j_raw, = ax_J.plot([], [], ".", color=style.DIM, ms=2.5)
    j_line, = ax_J.plot([], [], color=style.GOOD, lw=1.4)
    j_dot, = ax_J.plot([], [], "o", color=style.GOOD, ms=3)
    ax_J.set_title("shape objective", fontsize=8.5, color=style.FG, pad=5)
    fig.text(0.845, 0.42,
             "the iterate sweep visualizes\nthe solve process; the melt\n"
             "run and the end numbers\nare the STORED library\n"
             "deliverable (solved\nsingle-pass 4 bpp)",
             fontsize=7, color=style.DIM, va="top")

    cap_L = fig.text(0.235, 0.075, "", ha="center", fontsize=8.5, color=style.FG)
    cap_R = fig.text(0.615, 0.075, "", ha="center", fontsize=8.5, color=style.FG)
    verdict = fig.text(0.5, 0.033, "", ha="center", fontsize=9,
                       color=style.GOOD)
    footer = fig.text(0.5, 0.008,
                      "grid 120; melt-region framing; each arm at its own "
                      "optimal stop; stored numbers: "
                      "fgm_solve_campaign/out_lib/hexagon.json",
                      ha="center", fontsize=6.5, color=style.DIM)
    del footer
    front = {"L": None, "R": None}
    melt_mode = [False]

    def show_melt(ax, key, T, im):
        im.set_data(cr(T))
        im.set_cmap(style.CMAP_T)
        im.set_clim(25.0, 220.0)
        if not melt_mode[0]:
            melt_mode[0] = True
            for cb in cbs:
                cb.set_label("temperature (deg C)", fontsize=8, color=style.DIM)
        if front[key] is not None:
            front[key].remove()
        front[key] = ax.contour(cr(T), levels=[t_pc], colors="white",
                                linewidths=0.9, linestyles="--", extent=ext,
                                origin="lower")

    def update(f):
        if f < F_MAP:
            k = min(int(f / F_MAP * n_ev), n_ev - 1)
            im_R.set_data(cr(maps_ev[int(np.argmin(j_ev[:k + 1]))]))
            e = np.arange(1, k + 2)
            j_raw.set_data(e, np.minimum(j_ev[:k + 1], y_cap * 0.98))
            j_line.set_data(e, j_best[:k + 1])
            j_dot.set_data([k + 1], [j_best[k]])
            cap_L.set_text("one guess from one proxy field")
            cap_R.set_text(f"solve process, evaluation {k + 1} of {n_ev}")
        else:
            g = min(f - F_MAP, F_MELT - 1)
            fr = g / (F_MELT - 1)
            ih = min(int(round(fr * (len(T_h) - 1))), len(T_h) - 1)
            i_s = min(int(round(fr * (len(T_s) - 1))), len(T_s) - 1)
            if g == 0:
                im_R.set_data(cr(d["sat_a1"]))
                cap_R.set_text("the stored solved 4 bpp map takes over")
            show_melt(ax_L, "L", T_h[ih], im_L)
            show_melt(ax_R, "R", T_s[i_s], im_R)
            th = (int(steps_h[ih]) + 1) * dt
            ts = (int(steps_s[i_s]) + 1) * dt
            if f >= F_MAP + F_MELT - 1:
                cap_L.set_text(f"at its stop {stop_h:.0f} s:  J_phi "
                               f"{hist_J:.1f}  IoU {hist_iou:.4f}")
                cap_R.set_text(f"at its stop {stop_s:.0f} s:  J_phi "
                               f"{solve_J:.1f}  IoU {solve_iou:.4f}")
                verdict.set_text(f"the solve wins the head-to-head: "
                                 f"J_phi {hist_J:.1f} -> {solve_J:.1f} "
                                 f"({dj_pct:+.1f} percent), "
                                 f"IoU {hist_iou:.4f} -> {solve_iou:.4f}")
            else:
                cap_L.set_text(f"melting, t = {th:4.0f} s")
                cap_R.set_text(f"melting, t = {ts:4.0f} s")
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(OUT, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {OUT}  frames {n_frames}  duration {n_frames / FPS:.1f} s  "
          f"size {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
