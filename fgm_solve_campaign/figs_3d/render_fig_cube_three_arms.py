"""Dissertation figure: Phase E cube, three arms, light style.

    ./.venv312/bin/python fgm_solve_campaign/figs_3d/render_fig_cube_three_arms.py

Top row: the melted body (phi >= 0.9) that would form under each arm, drawn as
an opaque isosurface inside the nominal cube wireframe, amber where the melt is
inside the nominal solid and dark red where it has escaped. Bottom row: the
same three bodies cut on the vertical mid-plane, with the no-correction melt
boundary repeated as a dashed reference on the two corrected panels.

Reads only stored artifacts (see cube_light_common). Rendering only: no solve,
no forward evaluation, no solve3d source change. The reproduction gate in
cube_light_common.gate runs before anything is drawn.
"""
from __future__ import annotations

import numpy as np

import cube_light_common as K

C, L, MB = K.phase_helpers()
K.use_light_style()

import matplotlib.pyplot as plt                                  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection          # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

OUT = K.HERE / "fig_cube_three_arms.png"
ELEV, AZIM = 26.0, 34.0
FRONT = 0.9


def shaded(nrm: np.ndarray, ins: np.ndarray) -> np.ndarray:
    """Lambert shading on white: keep the body light enough to read as a solid."""
    light = np.array([0.55, 0.5, 0.68])
    light /= np.linalg.norm(light)
    lam = np.clip(-(nrm @ light), 0.0, 1.0)
    base = np.where(ins[:, None],
                    np.array(plt.matplotlib.colors.to_rgb(K.MELT_IN))[None, :],
                    np.array(plt.matplotlib.colors.to_rgb(K.MELT_OUT))[None, :])
    rgb = np.clip(base * (0.62 + 0.38 * lam)[:, None], 0.0, 1.0)
    return np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)


def body_panel(fig, rect, phi, inside, ax_m, half_mm, r):
    ax = fig.add_axes(rect, projection="3d")
    tri, nrm, ins = MB.melt_body(phi, inside, ax_m)
    ax.add_collection3d(Poly3DCollection(tri * K.MM, facecolors=shaded(nrm, ins),
                                         edgecolors="none"))
    K.cube_wire(ax, half_mm)
    K.light_3d_axes(ax)
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1), zoom=1.33)
    ax.view_init(elev=ELEV, azim=AZIM)
    return int((~ins).sum())


def section_panel(fig, rect, phi, phi_ref, inside, ax_m, half_mm, r,
                  show_ref: bool):
    ax = fig.add_axes(rect)
    j = int(np.argmin(np.abs(ax_m)))
    X, Z = np.meshgrid(ax_m * K.MM, ax_m * K.MM, indexing="ij")
    p, ins = phi[:, j, :], inside[:, j, :].astype(float)
    ax.contourf(X, Z, np.where(ins > 0.5, p, np.nan), levels=[FRONT, 10.0],
                colors=[K.MELT_IN])
    ax.contourf(X, Z, np.where(ins <= 0.5, p, np.nan), levels=[FRONT, 10.0],
                colors=[K.MELT_OUT])
    ax.plot([-half_mm, half_mm, half_mm, -half_mm, -half_mm],
            [-half_mm, -half_mm, half_mm, half_mm, -half_mm],
            color=K.NOMINAL, lw=0.9)
    if show_ref:
        ax.contour(X, Z, phi_ref[:, j, :], levels=[FRONT], colors=["#1a1a1a"],
                   linewidths=0.9, linestyles=[(0, (3.5, 2.5))])
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def main() -> int:
    d = K.load()
    K.gate(d)
    doc, z = d["doc"], d["vol"]
    ax_m = np.asarray(z["axis_m"], float)
    inside = np.asarray(z["inside"], bool)
    half_mm = d["half_mm"]
    r = 1.16 * half_mm
    phi = {a: K.phi_of_T(z[f"T__{a}"]) for a in K.ARMS}
    ref = phi["uniform_baseline"]
    uni = doc["arms"]["uniform_baseline"]

    fig = plt.figure(figsize=(6.9, 5.40), dpi=300)
    W, H = 0.310, 0.370
    xs = (0.017, 0.345, 0.673)
    yt, yb = 0.545, 0.190
    n_out_cells = {}

    for i, arm in enumerate(K.ARMS):
        rec = doc["arms"][arm]
        n_out_cells[arm] = body_panel(fig, [xs[i], yt, W, H], phi[arm], inside,
                                      ax_m, half_mm, r)
        dJ = 100.0 * (rec["J_asymmetric"] / uni["J_asymmetric"] - 1.0)
        col = K.C_SOL if arm == "solve_filter_only" else (
            K.FG if i == 0 else K.C_HEU)
        fig.text(xs[i] + W / 2, yt + H + 0.040, f"({'abc'[i]}) {K.ARM_LABEL[arm]}",
                 fontsize=8.6, color=col, ha="center", va="baseline")
        tag = ("uniform dopant, reference" if i == 0 else
               ("objective $J$ %+.0f%%" % dJ))
        fig.text(xs[i] + W / 2, yt + H + 0.010, tag, fontsize=8.0,
                 color=col if i else K.DIM, ha="center", va="baseline",
                 fontweight="bold" if i else "normal")
        fig.text(xs[i] + W / 2, yt - 0.014,
                 "melt fraction %.0f%%   stop %.0f s"
                 % (100 * rec["part_mean_phi"], rec["t_stop_s"]),
                 fontsize=7.6, color=K.DIM, ha="center", va="top")
        section_panel(fig, [xs[i], yb, W, 0.285], phi[arm], ref, inside,
                      ax_m, half_mm, r, show_ref=(i > 0))
        fig.text(xs[i] + W / 2, yb + 0.283, f"({'def'[i]})", fontsize=8.0,
                 color=K.DIM, ha="center", va="baseline")
        if arm == "heuristic_grading_law":
            fig.text(xs[i] + W / 2, yb - 0.002,
                     "melt escapes below the part", fontsize=6.8,
                     color=K.MELT_OUT, ha="center", va="top")


    handles = [Patch(facecolor=K.MELT_IN, edgecolor="none",
                     label=r"melted, inside the nominal solid"),
               Patch(facecolor=K.MELT_OUT, edgecolor="none",
                     label="melted, outside it"),
               Line2D([], [], color=K.NOMINAL, lw=0.9, label="nominal cube"),
               Line2D([], [], color="#1a1a1a", lw=0.9, ls=(0, (3.5, 2.5)),
                      label="melt front with no correction")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.092), handlelength=1.6,
               columnspacing=2.2, fontsize=7.2)

    fig.text(0.5, 0.062,
             "Bottom row: the same three melted bodies cut on the vertical "
             "mid-plane through the part centre.",
             fontsize=6.8, color=K.DIM, ha="center", va="top")
    fig.text(0.5, 0.040,
             "Melted body is $\\varphi \\geq 0.9$ on a %d$^3$ render grid; melt "
             "fraction is the volume-weighted mean of $\\varphi$ over the part."
             % ax_m.size,
             fontsize=6.8, color=K.DIM, ha="center", va="top")
    fig.text(0.5, 0.018,
             "Direct solve is the 3-D adjoint filter-only arm, stopped by its "
             "12-evaluation budget while still descending. Simulation only.",
             fontsize=6.8, color=K.DIM, ha="center", va="top")

    fig.savefig(OUT, dpi=300, facecolor=K.BG)
    print("wrote", OUT)
    print("out-of-bounds isosurface faces per arm:", n_out_cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
