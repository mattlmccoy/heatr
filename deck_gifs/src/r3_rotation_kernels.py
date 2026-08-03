"""Render gif_rotation_kernels.gif: the part-frame heating kernel of the cross
as the rotation mode changes. Static electrode-axis bands, then the annular
blur of continuous rotation built up angle by angle, then the four-fold
pattern restored by 90 degree indexing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "deck_gifs/cache/c3_kernels_cross.npz"
OUT = REPO / "deck_gifs/gif_rotation_kernels.gif"
FPS = 12

F_STATIC = 28
F_SWEEP_PER_ANGLE = 3
F_HOLD_CONT = 20
F_PER_INDEX = 8
F_HOLD_IDX = 28


def main() -> None:
    d = np.load(CACHE)
    Qk = np.asarray(d["Qk"], dtype=float)
    angles = np.asarray(d["angles_deg"], dtype=float)
    pm = np.asarray(d["part_mask"], dtype=bool)
    x, y = d["x"], d["y"]
    iou = {"static": float(d["iou_static"]), "cont": float(d["iou_cont"]),
           "idx": float(d["iou_index90"])}

    r0, r1c, c0, c1 = style.crop_indices(pm, pad=10)
    ext = [x[c0] * 1e3, x[c1 - 1] * 1e3, y[r0] * 1e3, y[r1c - 1] * 1e3]
    pmc = pm[r0:r1c, c0:c1].astype(float)

    def cr(a):
        return np.asarray(a)[r0:r1c, c0:c1]

    n_ang = len(angles)
    idx90 = [int(np.argmin(np.abs(angles - a))) for a in (0., 90., 180., 270.)]
    vmax = 1.2e7   # edge singularities saturate; gamma keeps the bulk readable
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)

    # (field, mode, sub-label) per frame
    frames = []
    for _ in range(F_STATIC):
        frames.append((Qk[0], "static", "electrode-axis bands"))
    for m in range(2, n_ang + 1):
        f = Qk[:m].mean(axis=0)
        for _ in range(F_SWEEP_PER_ANGLE):
            frames.append((f, "cont", f"averaging 0 to {angles[m - 1]:.0f} deg"))
    cont = Qk.mean(axis=0)
    for _ in range(F_HOLD_CONT):
        frames.append((cont, "cont", "annular blur, corners starved"))
    for m in range(1, 5):
        f = Qk[idx90[:m]].mean(axis=0)
        for _ in range(F_PER_INDEX):
            lab = " + ".join(f"{angles[j]:.0f}" for j in idx90[:m])
            frames.append((f, "idx", f"positions {lab} deg"))
    idxf = Qk[idx90].mean(axis=0)
    for _ in range(F_HOLD_IDX):
        frames.append((idxf, "idx", "four-fold symmetry restored"))

    fig = plt.figure(figsize=(10.4, 6.4), dpi=style.DPI)
    ax = fig.add_axes([0.045, 0.10, 0.55, 0.76])
    im = ax.imshow(cr(Qk[0]), origin="lower", extent=ext, cmap=style.CMAP_Q,
                   norm=norm, interpolation="bilinear")
    ax.contour(pmc, levels=[0.5], colors=style.ACCENT, linewidths=1.0,
               extent=ext, origin="lower")
    style.field_axes(ax)
    style.slim_colorbar(fig, im, ax,
                        "radio frequency heating (W per m3, gamma 0.4 scale)")

    fig.suptitle("what rotation does to the heating kernel  (cross, part frame)",
                 fontsize=13, color=style.FG, y=0.955)

    modes = [
        ("static", "static", "bands along the electrode axis"),
        ("cont", "continuous rotation", "annular blur"),
        ("idx", "90 degree indexing", "four-fold pattern restored"),
    ]
    mode_txt = {}
    for k, (key, name, desc) in enumerate(modes):
        yy = 0.70 - 0.17 * k
        mode_txt[key] = (
            fig.text(0.66, yy, name, fontsize=11, color=style.DIM),
            fig.text(0.66, yy - 0.045, desc, fontsize=8, color=style.DIM),
            fig.text(0.66, yy - 0.085,
                     f"IoU {iou[key]:.3f}", fontsize=9, color=style.DIM),
        )
    sub = fig.text(0.045, 0.035, "", fontsize=9, color=style.FG)
    fig.text(0.66, 0.13, "IoU = intersection over union of the\nmelted region "
             "with the nominal part,\nuniform dopant map, each mode at\nits own "
             "stop (stored campaign values)", fontsize=7.5, color=style.DIM,
             va="top")

    def set_active(key):
        for k2, txts in mode_txt.items():
            on = k2 == key
            txts[0].set_color(style.FG if on else style.DIM)
            txts[0].set_fontweight("bold" if on else "normal")
            txts[2].set_color(style.GOOD if on else style.DIM)

    def update(f):
        field, key, lab = frames[f]
        im.set_data(cr(field))
        set_active(key)
        sub.set_text(lab)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(OUT, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {OUT}  frames {len(frames)}  duration {len(frames) / FPS:.1f} s"
          f"  size {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
