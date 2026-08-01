"""Render gif_dwell_cross.gif: the cross under its solved asymmetric dwell
program. Left temperature, right relative density, turntable dial and program
timeline below. The loop ends at the campaign's recommended stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "deck_gifs/cache/c1_dwell_cross.npz"
OUT = REPO / "deck_gifs/gif_dwell_cross.gif"
FPS = 12
FRAME_STRIDE = 2          # use every 2nd stored snapshot (4 s process time)
HOLD_FRAMES = 20          # freeze on the final crisp cross

POS_COLORS = plt.cm.twilight(np.linspace(0.05, 0.95, 8))[::2]


def main() -> None:
    d = np.load(CACHE)
    deliv = json.loads((REPO / "fgm_solve_campaign/out_dwell/"
                        "cross_turntable_deliverable.json").read_text())
    meta = json.loads((REPO / "fgm_solve_campaign/out_dwell/"
                       "cross_dwell.json").read_text())
    arm = meta["arms"]["D_refined_4bpp_discovered_lib"]

    pm = np.asarray(d["part_mask"], dtype=bool)
    x, y = d["x"], d["y"]
    dt = float(d["dt_s"])
    snaps_T = d["snaps_T"]
    snaps_rho = d["snaps_rho"]
    snap_steps = d["snap_steps"]
    pos_index = d["pos_index"]
    angles = d["angles_deg"]
    stop_s = float(deliv["recommended_stop_s"])
    t_pc = float(d["t_pc_c"])

    r0, r1c, c0, c1 = style.crop_indices(pm, pad=10)
    ext = [x[c0] * 1e3, x[c1 - 1] * 1e3, y[r0] * 1e3, y[r1c - 1] * 1e3]
    pmc = pm[r0:r1c, c0:c1].astype(float)

    sel = list(range(0, len(snap_steps), FRAME_STRIDE))
    if sel[-1] != len(snap_steps) - 1:
        sel.append(len(snap_steps) - 1)
    n_frames = len(sel) + HOLD_FRAMES

    cmap_rho = plt.get_cmap(style.CMAP_RHO).copy()
    cmap_rho.set_bad(style.BG)

    def rho_masked(a):
        return np.where(pm, a, np.nan)[r0:r1c, c0:c1]

    fig = plt.figure(figsize=(11.2, 6.6), dpi=style.DPI)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.22],
                          width_ratios=[1.0, 1.0, 0.42],
                          left=0.045, right=0.965, top=0.83, bottom=0.09,
                          hspace=0.30, wspace=0.16)
    ax_T = fig.add_subplot(gs[0, 0])
    ax_R = fig.add_subplot(gs[0, 1])
    ax_D = fig.add_subplot(gs[0, 2])
    ax_L = fig.add_subplot(gs[1, :])

    fig.suptitle("the turntable solve: cross under its asymmetric dwell program",
                 fontsize=13, color=style.FG, y=0.965)
    fig.text(0.045, 0.905,
             "indexed turntable, constant radio frequency power; dwell "
             "positions 0 / 90 / 180 / 270 deg, 5 s holds, 20 s cycle",
             fontsize=8.5, color=style.DIM)

    im_T = ax_T.imshow(snaps_T[0][r0:r1c, c0:c1], origin="lower", extent=ext,
                       cmap=style.CMAP_T, vmin=25.0, vmax=220.0,
                       interpolation="bilinear")
    style.field_axes(ax_T, "temperature")
    style.slim_colorbar(fig, im_T, ax_T, "deg C")
    im_R = ax_R.imshow(rho_masked(snaps_rho[0]), origin="lower", extent=ext,
                       cmap=cmap_rho, vmin=0.55, vmax=0.85,
                       interpolation="bilinear")
    style.field_axes(ax_R, "relative density")
    style.slim_colorbar(fig, im_R, ax_R, "rho / rho_solid")
    for ax in (ax_T, ax_R):
        ax.contour(pmc, levels=[0.5], colors=style.ACCENT, linewidths=1.0,
                   extent=ext, origin="lower")

    # turntable dial ---------------------------------------------------------
    ax_D.set_xlim(-1.6, 1.6)
    ax_D.set_ylim(-2.05, 1.55)
    ax_D.set_aspect("equal")
    style.field_axes(ax_D)
    ax_D.add_patch(Circle((0, 0), 1.0, fill=False, ec=style.DIM, lw=1.0))
    for k, a in enumerate([0.0, 90.0, 180.0, 270.0]):
        th = np.deg2rad(a)
        ax_D.plot([0.86 * np.cos(th), 1.0 * np.cos(th)],
                  [0.86 * np.sin(th), 1.0 * np.sin(th)],
                  color=POS_COLORS[k], lw=2.2)
        ax_D.text(1.27 * np.cos(th), 1.27 * np.sin(th), f"{a:.0f}",
                  ha="center", va="center", fontsize=8, color=style.DIM)
    needle, = ax_D.plot([0, 0.78], [0, 0], color=style.FG, lw=2.0,
                        solid_capstyle="round")
    hub = Circle((0, 0), 0.07, fc=style.FG, ec="none")
    ax_D.add_patch(hub)
    txt_pos = ax_D.text(0, -1.82, "position 0 deg", ha="center", fontsize=8.5,
                        color=style.FG)
    ax_D.set_title("turntable", fontsize=10, color=style.FG, pad=6)

    # program timeline -------------------------------------------------------
    for m in deliv["moves"]:
        if m["move_at_s"] >= stop_s:
            break
        j = int(np.argmin(np.abs(np.asarray([0., 90., 180., 270.])
                                 - (m["position_deg"] % 360.0))))
        ax_L.add_patch(Rectangle((m["move_at_s"], 0.0),
                                 min(m["dwell_s"], stop_s - m["move_at_s"]), 1.0,
                                 facecolor=POS_COLORS[j], edgecolor=style.BG,
                                 lw=0.3))
    ax_L.axvline(stop_s, color=style.WARM, lw=1.4)
    ax_L.text(stop_s - 4, 1.22, f"stop {stop_s:.1f} s", ha="right", fontsize=8,
              color=style.WARM)
    cursor = ax_L.axvline(0.0, color=style.FG, lw=1.6)
    ax_L.set_xlim(0, stop_s * 1.02)
    ax_L.set_ylim(0, 1)
    ax_L.set_yticks([])
    ax_L.tick_params(labelsize=8, colors=style.DIM, length=3)
    for s in ("top", "right", "left"):
        ax_L.spines[s].set_visible(False)
    ax_L.spines["bottom"].set_color(style.DIM)
    ax_L.set_xlabel("process time (s)", fontsize=8.5, color=style.DIM)

    txt_time = fig.text(0.955, 0.905, "t = 0 s", ha="right", fontsize=10,
                        color=style.FG)
    fig.text(0.045, 0.022,
             f"solved dwell map + schedule, stored campaign result: J_phi "
             f"{arm['J']:.1f}  IoU {arm['IoU']:.3f} at its stop  |  cyan "
             f"outline = nominal part, white dashes = melt front",
             fontsize=8, color=style.DIM)

    front = [ax_T.contour(snaps_T[0][r0:r1c, c0:c1], levels=[t_pc],
                          colors="white", linewidths=0.9, linestyles="--",
                          extent=ext, origin="lower")]

    def update(f):
        i = sel[min(f, len(sel) - 1)]
        step = int(snap_steps[i])
        t_now = (step + 1) * dt
        im_T.set_data(snaps_T[i][r0:r1c, c0:c1])
        im_R.set_data(rho_masked(snaps_rho[i]))
        front[0].remove()
        front[0] = ax_T.contour(snaps_T[i][r0:r1c, c0:c1], levels=[t_pc],
                                colors="white", linewidths=0.9,
                                linestyles="--", extent=ext, origin="lower")
        a = float(angles[int(pos_index[min(step, len(pos_index) - 1)])]) % 360.0
        th = np.deg2rad(a)
        needle.set_data([0, 0.78 * np.cos(th)], [0, 0.78 * np.sin(th)])
        txt_pos.set_text(f"position {a:.0f} deg")
        cursor.set_xdata([t_now, t_now])
        txt_time.set_text(f"t = {t_now:5.0f} s")
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(OUT, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {OUT}  frames {n_frames}  duration {n_frames / FPS:.1f} s  "
          f"size {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
