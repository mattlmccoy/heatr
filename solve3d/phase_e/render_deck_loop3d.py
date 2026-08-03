"""Deck figure: the 3-D analog of the 2-D solve-loop schematic.

    .venv312/bin/python -m solve3d.phase_e.render_deck_loop3d

Visual language follows deck_gifs/fig_solve_loop_schematic.png: five numbered
stages, the bottom row running right to left as the return leg, a J trajectory
and two exits. Rendered dark (deck_figures_3d/style3d) to match the other two
Phase E deck figures.

Every stage thumbnail is the SAME 3-D quarter cutaway of the pyramid at the
SAME camera, so the loop reads as one object being worked on in the round,
which is the point: the solve is a single 3-D problem, not a stack of 2-D
layer problems.

WHERE EACH THUMBNAIL COMES FROM (all on disk, nothing invented):
  1 seed          s = 1 on every design cell, the arm's own first iterate
  2 forward       phi from field_pyramid_uniform_baseline.npz T_read; the
                  uniform arm IS the loop's first forward (its J equals the
                  solve's J_first_eval to round-off, asserted below)
  3 mismatch      phi - chi from the same field, over the whole volume
  4 gradient      dJ/ds recomputed at the delivered map by gradient_probe.py
                  (one forward plus one adjoint); NOT a stand-in graphic
  5 update        s_map from map_pyramid_solve_filter_only.npz

Rendering only; gradient_probe.py does the one physics run this figure needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from solve3d.phase_e import render_deck_cutaway3d as C

st = C.st
RESULTS = C.RESULTS
MM = C.MM

CMAP_DOP = LinearSegmentedColormap.from_list(
    "viridis_lifted", plt.get_cmap("viridis")(np.linspace(0.16, 1.0, 256)))
CMAP_PHI = C.CMAP
def dark_centered(neg: str, pos: str, bg: str):
    """Diverging map whose CENTRE is the panel background.

    On a dark figure a white-centred diverging map makes "matched" and "no
    leverage" the brightest thing on the panel, which is backwards. Here zero
    fades into the background and both signs light up, so brightness reads as
    the quantity that drives the update.
    """
    return LinearSegmentedColormap.from_list(
        "divdark", [(0.0, neg), (0.5, bg), (1.0, pos)])


CMAP_MIS = dark_centered("#4da3ff", "#ff8a65", "#151a21")
CMAP_GRD = dark_centered("#ff9d3d", "#4dd0e1", "#151a21")

NUM = "#ff8a65"          # stage numbers, the reference figure's accent
ARROW = "#9aa0a6"


def design_volume(vals: np.ndarray, idx: np.ndarray, inside: np.ndarray,
                  fill: float) -> np.ndarray:
    """Lift a per-design-cell field onto the render grid (nearest cell)."""
    out = np.full(idx.shape, fill, dtype=float)
    out[inside] = vals[idx[inside]]
    return out


def stage(fig, rect, vol, mask_f, ax_m, cmap, norm, b_m, hn_m):
    ax = fig.add_axes(rect, projection="3d")
    h = float(ax_m[1] - ax_m[0])
    tri, spts, nrm, _v, _f = C.surface(mask_f, h, ax_m[0])
    rgba, _ = C.face_colors(vol, spts, ax_m[0], h, nrm, cmap=cmap, norm=norm)
    ax.add_collection3d(Poly3DCollection(tri * MM, facecolors=rgba,
                                         edgecolors="none"))
    C.pyramid_wire(ax, b_m / 2.0 * MM, hn_m * MM)
    st.dark_3d_axes(ax)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.line.set_color((0, 0, 0, 0))
        a.set_ticks([])
    r = 12.2
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1), zoom=1.30)
    ax.view_init(elev=C.ELEV, azim=C.AZIM)
    return ax


def label(fig, x, y, num, title):
    fig.text(x, y, num, fontsize=17, color=NUM, ha="left", va="baseline")
    fig.text(x + 0.021, y, title, fontsize=13.5, color=st.FG, ha="left",
             va="baseline")


def caption(fig, xc, y, lines, color=None):
    for i, t in enumerate(lines):
        fig.text(xc, y - 0.026 * i, t, fontsize=10.5,
                 color=color or st.DIM, ha="center", va="top")


def arrow(fig, x0, y0, x1, y1, text=(), rad=0.0, color=None):
    fig.patches.append(matplotlib.patches.FancyArrowPatch(
        (x0, y0), (x1, y1), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=17, lw=1.6,
        color=color or ARROW,
        connectionstyle=f"arc3,rad={rad}"))
    for i, t in enumerate(text):
        fig.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.030 - 0.024 * i, t,
                 fontsize=10, color=color or ARROW, ha="center", va="bottom")


def j_panel(fig, rect, traj):
    ax = fig.add_axes(rect)
    ax.set_facecolor(st.BG)
    e = [r["eval"] for r in traj]
    J = [r["J"] * 1e6 for r in traj]
    best = np.minimum.accumulate(J)
    ax.plot(e, best, color=st.ACCENT, lw=2.0, zorder=2, label="best so far")
    ax.scatter(e, J, s=26, color=st.FG, zorder=3, label="J at each evaluation")
    for sp in ax.spines.values():
        sp.set_color(st.DIM)
    ax.tick_params(colors=st.DIM, labelsize=9.5)
    ax.set_xlabel("gradient evaluation", color=st.DIM, fontsize=10.5)
    ax.set_ylabel("J  (1e-6)", color=st.DIM, fontsize=10.5)
    ax.grid(alpha=0.12, color=st.DIM)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    lg = ax.legend(fontsize=9, labelcolor=st.DIM, loc="upper right",
                   facecolor=st.BG, framealpha=0.92)
    lg.get_frame().set_edgecolor("none")
    return ax


def main() -> int:
    z = np.load(RESULTS / "vol_pyramid.npz")
    ax_m = np.asarray(z["axis_m"], float)
    inside = np.asarray(z["inside"], bool)
    idx = np.asarray(z["design_nearest_index"], np.int64)
    mask_f = C.cutaway_mask(inside, ax_m)
    b_m = float(z["nominal_base_side_m"])
    hn_m = float(z["nominal_height_m"])

    doc = json.loads((RESULTS / "phase_e_pyramid.json").read_text())
    sol = doc["arms"]["solve_filter_only"]
    uni = doc["arms"]["uniform_baseline"]
    mesh = doc["arms"]["_mesh"]
    traj = sol["trajectory"]

    # the uniform arm IS this loop's first forward: check, do not assume
    rel = abs(uni["J_asymmetric"] - sol["J_first_eval"]) / uni["J_asymmetric"]
    assert rel < 1e-12, f"uniform arm is not the solve's first eval (rel {rel:.2e})"

    gj = json.loads((RESULTS / "grad_pyramid_solve_filter_only.json").read_text())
    gz = np.load(RESULTS / "grad_pyramid_solve_filter_only.npz")
    g_s = np.asarray(gz["g_s"], float)
    s_map = np.asarray(gz["s_map"], float)

    phi_u = np.where(inside, C.phase_fraction_phi(z["T__uniform_baseline"]), 0.0)
    chi = inside.astype(float)
    v_seed = design_volume(np.ones_like(s_map), idx, inside, 0.0)
    v_mis = phi_u - chi
    v_grad = design_volume(g_s, idx, inside, 0.0)
    v_upd = design_volume(s_map, idx, inside, 0.0)

    # colour cap: a 98th-percentile cap leaves most of the visible surface
    # unlit, which reads as a broken panel rather than as a field. Saturating
    # at the 75th percentile lights the structure up; the cap is stated on the
    # figure so nobody reads the colours as absolute magnitudes.
    GCAP_PCT = 75.0
    gcap = float(np.percentile(np.abs(g_s), GCAP_PCT))
    n_dop = Normalize(0.0, 1.0)
    n_phi = Normalize(0.0, 1.0)
    n_mis = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    n_grd = TwoSlopeNorm(vmin=-gcap, vcenter=0.0, vmax=gcap)

    fig = plt.figure(figsize=(16.0, 9.0), dpi=200)
    fig.patch.set_facecolor(st.BG)

    fig.text(0.035, 0.985, "one 3-D solve on the full mesh "
             f"({mesh['n_nodes']} nodes, {mesh['n_design_cells']} design "
             "cells), not a stack of per-layer solves",
             fontsize=15, color=st.FG, ha="left", va="top")
    fig.text(0.035, 0.948, "follow the arrows: the bottom row is the return "
             "leg and runs right to left",
             fontsize=11, color=st.DIM, ha="left", va="top")

    W, H = 0.185, 0.270
    yt, yb = 0.615, 0.155
    ylt, ylb = 0.895, 0.455          # label baselines, top and bottom rows
    yct, ycb = 0.605, 0.145          # caption tops
    xs = (0.045, 0.320, 0.595)

    # ---- top row, left to right ---- #
    stage(fig, [xs[0], yt, W, H], v_seed, mask_f, ax_m, CMAP_DOP, n_dop, b_m, hn_m)
    label(fig, xs[0], ylt, "1", "SEED THE 3-D MAP")
    caption(fig, xs[0] + W / 2, yct,
            ["uniform dopant saturation on every",
             "design cell inside chi, the pyramid"])

    stage(fig, [xs[1], yt, W, H], phi_u, mask_f, ax_m, CMAP_PHI, n_phi, b_m, hn_m)
    label(fig, xs[1], ylt, "2", "FORWARD MARCH")
    caption(fig, xs[1] + W / 2, yct,
            ["coupled RF heating plus conduction,",
             f"melt fraction at the stop t* = {uni['t_stop_s']:.0f} s"])

    stage(fig, [xs[2], yt, W, H], v_mis, mask_f, ax_m, CMAP_MIS, n_mis, b_m, hn_m)
    label(fig, xs[2], ylt, "3", "MISMATCH OVER THE VOLUME")
    caption(fig, xs[2] + W / 2, yct,
            ["melt fraction minus chi through the",
             "whole part, every cell, not a slice"])
    fig.text(xs[2] + W / 2, yct - 0.048, "blue: unmelted inside the part",
             fontsize=10, color="#4da3ff", ha="center", va="top")

    arrow(fig, xs[0] + W + 0.012, yt + H / 2, xs[1] - 0.012, yt + H / 2,
          ("seed the dopant", "map inside chi"))
    arrow(fig, xs[1] + W + 0.012, yt + H / 2, xs[2] - 0.012, yt + H / 2,
          ("read at its own", "optimal stop t*"))

    # ---- bottom row, right to left ---- #
    stage(fig, [xs[2], yb, W, H], v_grad, mask_f, ax_m, CMAP_GRD, n_grd, b_m, hn_m)
    label(fig, xs[2], ylb, "4", "ADJOINT BACKWARD SWEEP")
    caption(fig, xs[2] + W / 2, ycb,
            [f"ONE backward pass gives dJ/ds at all {g_s.size} design",
             "cells at once: X, Y and Z together, never per layer"],
            color=st.GOOD)
    fig.text(xs[2] + W / 2, ycb - 0.050,
             "bright where the map has leverage; cost about one forward run",
             fontsize=10, color=st.DIM, ha="center", va="top")
    fig.text(xs[2] + W / 2, ycb - 0.070,
             f"colour saturates at the {GCAP_PCT:.0f}th percentile of |dJ/ds|",
             fontsize=9.5, color=st.DIM, ha="center", va="top")

    stage(fig, [xs[1], yb, W, H], v_upd, mask_f, ax_m, CMAP_DOP, n_dop, b_m, hn_m)
    label(fig, xs[1], ylb, "5", "FILTER AND UPDATE")
    caption(fig, xs[1] + W / 2, ycb,
            ["gradient through the 1.0 mm physical filter,",
             "the whole 3-D map moves downhill together"])

    xv = xs[2] + W / 2
    arrow(fig, xv, 0.545, xv, 0.478)
    fig.text(xv - 0.014, 0.512, "backpropagate the residual",
             fontsize=10, color=ARROW, ha="right", va="center")
    arrow(fig, xs[2] - 0.012, yb + H / 2, xs[1] + W + 0.012, yb + H / 2,
          ("filtered", "map update"))
    arrow(fig, xs[1] - 0.014, yb + H / 2, xs[1] - 0.030, yt + H / 2 - 0.020,
          (), rad=-0.35, color=st.ACCENT)
    fig.text(xs[1] - 0.052, yt + H / 2 - 0.055, "next\nevaluation",
             fontsize=10.5, color=st.ACCENT, ha="center", va="top")

    # ---- J trajectory ---- #
    j_panel(fig, [xs[0] + 0.032, yb + 0.038, W - 0.022, H - 0.055], traj)
    fig.text(xs[0], ylb, "J FALLS AS THE LOOP TURNS",
             fontsize=13.5, color=st.FG, ha="left", va="baseline")
    caption(fig, xs[0] + W / 2, ycb,
            ["one point per gradient evaluation;",
             "points above the line are rejected trials"])

    # ---- exits ---- #
    xe = 0.805
    fig.text(xe, 0.700, "TWO EXITS", fontsize=13, color=st.GOOD, ha="left")
    fig.text(xe, 0.655, "budget-limited", fontsize=12, color=st.FG, ha="left")
    fig.text(xe, 0.622, "stopped at the pre-registered\nbudget of "
             f"{sol['budget_gradient_evaluations']} gradient\nevaluations",
             fontsize=10.5, color=st.DIM, ha="left", va="top")
    fig.text(xe, 0.535, "still descending", fontsize=12, color=st.FG, ha="left")
    fig.text(xe, 0.502, "J was still falling at the\nlast accepted step, so the\n"
             "reported gain is a lower bound",
             fontsize=10.5, color=st.DIM, ha="left", va="top")
    fig.text(xe, 0.395, f"J  {traj[0]['J']:.3e}", fontsize=11.5, color=st.DIM,
             ha="left")
    fig.text(xe, 0.360, f"to {sol['J_asymmetric']:.3e}", fontsize=11.5,
             color=st.GOOD, ha="left")
    fig.text(xe, 0.320, "%.0f%% below the first evaluation" %
             (100 * (traj[0]["J"] - sol["J_asymmetric"]) / traj[0]["J"]),
             fontsize=10.5, color=st.DIM, ha="left")

    fig.text(0.5, 0.036, "filter-only arm; steady adjoint proven (D1), "
             "transient adjoint FD-gated (Phase B); simulation-only",
             fontsize=11, color=st.DIM, ha="center")
    repro = ("exactly, to the last bit" if gj["J_reproduction_rel"] == 0.0
             else "to %.1e relative" % gj["J_reproduction_rel"])
    fig.text(0.5, 0.010, "stage 4 is a real recomputed gradient at the "
             f"delivered map (one forward plus one adjoint, {gj['wall_s']:.0f} s); "
             f"its forward reproduces the stored J {repro}",
             fontsize=9.5, color=st.DIM, ha="center")

    out = RESULTS / "fig_deck_solve_loop_3d.png"
    fig.savefig(out, dpi=200, facecolor=st.BG)
    print("wrote", out, "| grad cap %.3e" % gcap,
          "| grad frac negative %.3f" % gj["grad_frac_negative"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
