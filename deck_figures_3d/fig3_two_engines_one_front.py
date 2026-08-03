"""Figure 3: Two engines, one melt front.

Overlay of the heatr3d (finite-volume voxel) and dolfinx (FEM) melt fronts at
melt onset on the SAME shared evaluation grid (0.15 mm pixels), mid z plane,
with zoom insets where the sub-pixel separation is visible. Annotated with
the measured front agreement from the Phase A close-out shape gate:
symmetric surface distance 0.067 to 0.129 mm across the four arms.

Data (all committed Phase A close-out artifacts):
  dolfinx melt-onset T on the eval grid: solve3d/results/eval_dolfinx_circle_off.npz,
    eval_dolfinx_square_off.npz (T, 5 z planes, 200x200).
  heatr3d melt-onset T: solve3d/results/anchor_heatr3d_circle_n96.npz,
    anchor_heatr3d_square_n96.npz (T_phi90 voxel field), trilinearly sampled
    onto the same grid exactly as solve3d/shape_gate.py does.
  Front SSD numbers: solve3d/results/phase_a_shape_gate.json
    (arms.*.metrics.front_ssd_mm_phi0p9).
Rendering + the shape gate's own resampling only; no physics is run.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import style3d as st

MM = 1000.0
EVAL_HALF = 0.015
EVAL_N = 200
Z_PLANES = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
MID = 2
H_COLOR = st.MELT      # heatr3d front
D_COLOR = st.ACCENT    # dolfinx front


def eval_axes():
    h = 2.0 * EVAL_HALF / EVAL_N
    c = (np.arange(EVAL_N) + 0.5) * h - EVAL_HALF
    return c, h


def heatr3d_mid(name: str) -> np.ndarray:
    z = np.load(st.REPO / "solve3d" / "results" / name)
    interp = RegularGridInterpolator(
        (np.asarray(z["x"]), np.asarray(z["y"]), np.asarray(z["z"])),
        np.asarray(z["T_phi90"], dtype=float), method="linear",
        bounds_error=True)
    c, _ = eval_axes()
    X, Y = np.meshgrid(c, c, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(),
                           np.full(X.size, Z_PLANES[MID])])
    return interp(pts).reshape(EVAL_N, EVAL_N)


def dolfinx_mid(tag: str) -> np.ndarray:
    return np.asarray(np.load(st.REPO / "solve3d" / "results"
                              / f"eval_dolfinx_{tag}.npz")["T"][MID],
                      dtype=float)


def ssd_numbers() -> dict:
    doc = json.loads((st.REPO / "solve3d" / "results"
                      / "phase_a_shape_gate.json").read_text())
    return {arm: doc["arms"][arm]["metrics"]["front_ssd_mm_phi0p9"]
            for arm in doc["arms"]}


def draw_front(ax, phi, color, lw, ls="-"):
    c, _ = eval_axes()
    ax.contour(c * MM, c * MM, phi.T, levels=[0.9], colors=[color],
               linewidths=[lw], linestyles=[ls])


def nominal(ax, shape, lw=0.9):
    if shape == "circle":
        t = np.linspace(0, 2 * np.pi, 361)
        ax.plot(10 * np.cos(t), 10 * np.sin(t), color=st.DIM, lw=lw,
                ls=(0, (2, 2)), alpha=0.8)
    else:
        ax.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10],
                color=st.DIM, lw=lw, ls=(0, (2, 2)), alpha=0.8)


def panel(ax, shape, tag, anchor, zoom, ssd):
    phi_d = st.phase_fraction_phi(dolfinx_mid(tag))
    phi_h = st.phase_fraction_phi(heatr3d_mid(anchor))
    for a in (ax,):
        nominal(a, shape)
        draw_front(a, phi_h, H_COLOR, 2.2)
        draw_front(a, phi_d, D_COLOR, 1.4)
    ax.set_aspect("equal")
    ax.set_xlim(-12.5, 12.5)
    ax.set_ylim(-12.5, 12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{shape}, 20 mm", fontsize=13, color=st.FG, pad=8)
    ax.text(0.03, 0.03, f"front distance {ssd:.3f} mm",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=12,
            color=st.GOOD, fontweight="bold")

    # zoom inset
    (x0, x1, y0, y1) = zoom
    axi = ax.inset_axes([0.615, 0.615, 0.38, 0.38])
    axi.set_facecolor(st.PANEL)
    nominal(axi, shape, lw=0.8)
    draw_front(axi, phi_h, H_COLOR, 2.6)
    draw_front(axi, phi_d, D_COLOR, 1.6)
    axi.set_xlim(x0, x1)
    axi.set_ylim(y0, y1)
    axi.set_aspect("equal")
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_color(st.DIM)
        s.set_linewidth(0.7)
    ax.indicate_inset_zoom(axi, edgecolor=st.DIM, alpha=0.7)
    # scale bar: 0.5 mm, anchored top-left in axes coords
    x_a = x0 + 0.08 * (x1 - x0)
    y_a = y0 + 0.90 * (y1 - y0)
    axi.plot([x_a, x_a + 0.5], [y_a, y_a], color=st.FG, lw=1.8,
             solid_capstyle="butt", zorder=5)
    axi.text(x_a + 0.25, y_a - 0.045 * (y1 - y0), "0.5 mm", ha="center",
             va="top", fontsize=8, color=st.FG, zorder=5)
    axi.text(0.96, 0.05, "grid pixel 0.15 mm", transform=axi.transAxes,
             ha="right", va="bottom", fontsize=7.5, color=st.DIM)


def main() -> None:
    ssd = ssd_numbers()
    fig = plt.figure(figsize=(12.6, 7.2), dpi=st.DPI)
    ax1 = fig.add_axes([0.045, 0.10, 0.42, 0.72])
    ax2 = fig.add_axes([0.535, 0.10, 0.42, 0.72])

    panel(ax1, "circle", "circle_off", "anchor_heatr3d_circle_n96.npz",
          (5.6, 9.1, -8.2, -4.7), ssd["circle_off"])
    panel(ax2, "square", "square_off", "anchor_heatr3d_square_n96.npz",
          (5.6, 10.9, -10.9, -5.6), ssd["square_off"])

    st.title_block(fig, "TWO ENGINES, ONE MELT FRONT",
                   "phi = 0.9 melt front at melt onset, both engines read on "
                   "one shared 0.15 mm grid, mid z plane")
    fig.text(0.965, 0.955, "voxel engine (heatr3d)", fontsize=10.5,
             color=H_COLOR, ha="right")
    fig.text(0.965, 0.925, "FEM engine (dolfinx)", fontsize=10.5,
             color=D_COLOR, ha="right")
    fig.text(0.965, 0.895, "nominal bounds", fontsize=10.5, color=st.DIM,
             ha="right")
    lo = min(ssd.values())
    hi = max(ssd.values())
    fig.text(0.5, 0.038,
             f"front agreement {lo:.2f} to {hi:.2f} mm across all four "
             "Phase A arms: sub-pixel on a 20 mm part",
             fontsize=11.5, color=st.FG, ha="center")
    fig.text(0.975, 0.006,
             "Phase A close-out shape gate, all arms PASS "
             "(phase_a_shape_gate.json)",
             fontsize=8.5, color=st.DIM, ha="right")

    out = st.OUT / "fig3_two_engines_one_front.png"
    fig.savefig(out, dpi=st.DPI)
    print("wrote", out, "| ssd:", {k: round(v, 4) for k, v in ssd.items()})


if __name__ == "__main__":
    main()
