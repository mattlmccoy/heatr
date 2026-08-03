"""Render gif_sequential_L.gif: the L_shape melting limb by limb under its
DELIVERABLE sequential program. Same layout language as gif_dwell_cross:
temperature left, relative density right, turntable dial and program timeline
below. Phase 1 melts the foot, the quarter-turn is flashed on the dial, phase 2
grows the upright while the melted foot holds. Ends at the recommended stop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "deck_gifs/cache/c4_seq_L.npz"
OUT = REPO / "deck_gifs/gif_sequential_L.gif"
FPS = 12
FRAME_STRIDE = 2          # every 2nd snapshot = 4 s process time per frame
HOLD_FRAMES = 22
FLASH_S = 30.0            # dial highlight duration after the quarter-turn

POS_COLORS = plt.cm.twilight(np.linspace(0.05, 0.95, 8))[::2]
COL_90 = POS_COLORS[1]
COL_0 = POS_COLORS[0]


def main() -> None:
    d = np.load(CACHE)
    pm = np.asarray(d["part_mask"], dtype=bool)
    x, y = d["x"], d["y"]
    dt = float(d["dt_s"])
    snaps_T = d["snaps_T"]
    snaps_rho = d["snaps_rho"]
    snap_steps = d["snap_steps"]
    switch_s = float(d["switch_s"])
    stop_s = (int(d["stop_index"]) + 1) * dt
    t_pc = float(d["t_pc_c"])
    J_stop = float(d["J_at_stop"])
    iou_stop = float(d["IoU_at_stop"])

    r0, r1c, c0, c1 = style.crop_indices(pm, pad=10)
    ext = [x[c0] * 1e3, x[c1 - 1] * 1e3, y[r0] * 1e3, y[r1c - 1] * 1e3]
    pmc = pm[r0:r1c, c0:c1].astype(float)

    cmap_rho = plt.get_cmap(style.CMAP_RHO).copy()
    cmap_rho.set_bad(style.BG)

    def rho_masked(a):
        return np.where(pm, a, np.nan)[r0:r1c, c0:c1]

    sel = list(range(0, len(snap_steps), FRAME_STRIDE))
    if sel[-1] != len(snap_steps) - 1:
        sel.append(len(snap_steps) - 1)
    n_frames = len(sel) + HOLD_FRAMES

    fig = plt.figure(figsize=(11.2, 6.6), dpi=style.DPI)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.22],
                          width_ratios=[1.0, 1.0, 0.42],
                          left=0.045, right=0.965, top=0.83, bottom=0.155,
                          hspace=0.30, wspace=0.16)
    ax_T = fig.add_subplot(gs[0, 0])
    ax_R = fig.add_subplot(gs[0, 1])
    ax_D = fig.add_subplot(gs[0, 2])
    ax_L = fig.add_subplot(gs[1, :])

    fig.suptitle("the sequential solve: L melts limb by limb", fontsize=13,
                 color=style.FG, y=0.965)
    fig.text(0.045, 0.905,
             "indexed turntable, constant radio frequency power; hold 90 deg "
             f"for {switch_s:.0f} s, quarter-turn to 0 deg, stop {stop_s:.1f} s",
             fontsize=8.5, color=style.DIM)

    im_T = ax_T.imshow(snaps_T[0][r0:r1c, c0:c1], origin="lower", extent=ext,
                       cmap=style.CMAP_T, vmin=25.0, vmax=225.0,
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
    flash_ring = Circle((0, 0), 1.14, fill=False, ec=style.WARM, lw=2.2,
                        visible=False)
    ax_D.add_patch(flash_ring)
    for a, col in ((0.0, COL_0), (90.0, COL_90)):
        th = np.deg2rad(a)
        ax_D.plot([0.86 * np.cos(th), 1.0 * np.cos(th)],
                  [0.86 * np.sin(th), 1.0 * np.sin(th)], color=col, lw=2.6)
        ax_D.text(1.30 * np.cos(th), 1.30 * np.sin(th), f"{a:.0f}",
                  ha="center", va="center", fontsize=8, color=style.DIM)
    needle, = ax_D.plot([0, 0], [0, 0.78], color=style.FG, lw=2.0,
                        solid_capstyle="round")
    ax_D.add_patch(Circle((0, 0), 0.07, fc=style.FG, ec="none"))
    txt_pos = ax_D.text(0, -1.55, "hold 90 deg", ha="center", fontsize=8.5,
                        color=style.FG)
    txt_flash = ax_D.text(0, -1.90, "", ha="center", fontsize=8.5,
                          color=style.WARM)
    ax_D.set_title("turntable", fontsize=10, color=style.FG, pad=6)

    # program timeline: exactly two holds ------------------------------------
    ax_L.add_patch(Rectangle((0.0, 0.0), switch_s, 1.0, facecolor=COL_90,
                             edgecolor=style.BG, lw=0.3))
    ax_L.add_patch(Rectangle((switch_s, 0.0), stop_s - switch_s, 1.0,
                             facecolor=COL_0, edgecolor=style.BG, lw=0.3))
    ax_L.text(switch_s * 0.5, 0.5, "hold 90 deg", ha="center", va="center",
              fontsize=8, color="w")
    ax_L.text(switch_s + (stop_s - switch_s) * 0.5, 0.5, "hold 0 deg",
              ha="center", va="center", fontsize=8, color="w")
    ax_L.axvline(switch_s, color=style.WARM, lw=1.4)
    ax_L.text(switch_s, 1.22, f"quarter turn {switch_s:.0f} s", ha="center",
              fontsize=8, color=style.WARM)
    ax_L.axvline(stop_s, color=style.WARM, lw=1.4)
    ax_L.text(stop_s - 4, 1.22, f"stop {stop_s:.1f} s", ha="right", fontsize=8,
              color=style.WARM)
    cursor = ax_L.axvline(0.0, color=style.FG, lw=1.6)
    ax_L.set_xlim(0, stop_s * 1.02)
    ax_L.set_ylim(0, 1)
    ax_L.set_yticks([])
    ax_L.tick_params(labelsize=8, colors=style.DIM, length=3)
    for sp in ("top", "right", "left"):
        ax_L.spines[sp].set_visible(False)
    ax_L.spines["bottom"].set_color(style.DIM)
    ax_L.set_xlabel("process time (s)", fontsize=8.5, color=style.DIM,
                    labelpad=2)

    txt_time = fig.text(0.955, 0.905, "t = 0 s", ha="right", fontsize=10,
                        color=style.FG)
    txt_phase = fig.text(0.045, 0.058, "", fontsize=9, color=style.FG)
    fig.text(0.045, 0.022,
             f"stored campaign result at the stop: J_phi {J_stop:.1f}  IoU "
             f"{iou_stop:.4f}, best on record for the L, not solved  |  "
             f"cyan = nominal outline, white dashes = melt front",
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
        in_p2 = t_now >= switch_s
        a = 0.0 if in_p2 else 90.0
        th = np.deg2rad(a)
        needle.set_data([0, 0.78 * np.cos(th)], [0, 0.78 * np.sin(th)])
        txt_pos.set_text(f"hold {a:.0f} deg")
        flashing = in_p2 and (t_now - switch_s) <= FLASH_S
        flash_ring.set_visible(flashing)
        txt_flash.set_text("quarter turn" if flashing else "")
        txt_phase.set_text(
            "phase 2: the upright grows while the melted foot holds"
            if in_p2 else
            "phase 1: the foot melts, the upright stays cold")
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
