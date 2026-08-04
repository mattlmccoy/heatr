"""Deck figure: THE PART THAT WOULD FORM, with and without correction.

    .venv312/bin/python -m solve3d.phase_e.render_deck_melt_body --shape cube

Replaces the volume-coloured three-arm figure. That version coloured every
voxel, so the unmelted material formed an opaque purple box that hid the
result; the difference row was a fog that needed a legend to decode. Matt's
reframe: show the part with and without corrections.

So this figure draws OBJECTS, not colormaps.

  top row     the phi >= 0.9 melt body as an opaque isosurface: the solid that
              would actually form. Amber inside the nominal bounds, warning red
              where it has spilled outside them. Everything unmelted is gone;
              only the cyan nominal wireframe marks where the part should be.
              Three arms, one camera: the viewer compares three physical bodies.

  bottom row  the same three bodies cut on the vertical mid-plane, with the
              uniform arm's melt boundary repeated as a white dashed line on
              the corrected panels. Amber beyond that dashed line is exactly
              what the correction added.

Rendering only. No physics is run here.

HONESTY NOTE ON THE TWO OUT-OF-PART NUMBERS. The red surface is melt at
phi >= 0.9 outside the nominal solid, counted on this render grid. The
`out_of_part_melt_fraction_of_part` in the JSON is a different quantity: the
FEM integral of the continuous phi over out-of-part cells. Arms can show no
red here and still carry a non-zero volumetric figure, and both are printed
with their own labels rather than being conflated.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates
from skimage import measure

from solve3d.phase_e import render_deck_cutaway3d as C
from solve3d.phase_e import render_deck_cube as CU

st = C.st
RESULTS = C.RESULTS
MM = C.MM

ARMS = (("uniform_baseline", "no correction"),
        ("heuristic_grading_law", "hand-built grading law"),
        ("solve_filter_only", "solve (filter-only arm)"))
MELT = "#ffb74d"          # melt body inside the nominal bounds
SPILL = "#ff5252"         # melt that has escaped the bounds
GHOST = "#5f6b7a"


SMOOTH_CELLS = 0.8       # display-only smoothing of the isosurface


def melt_body(phi, inside, ax_m):
    """Isosurface of phi = 0.9 with a per-face inside/outside classification.

    phi is piecewise linear off a P1 FEM field, so the raw isosurface carries
    visible element-boundary ringing. A 0.8 cell gaussian is applied BEFORE
    marching cubes for display only; it is stated on the figure. No metric in
    this campaign is computed from the smoothed field.
    """
    from scipy.ndimage import gaussian_filter
    phi = gaussian_filter(np.asarray(phi, float), SMOOTH_CELLS, mode="nearest")
    h = float(ax_m[1] - ax_m[0])
    verts, faces, _n, _v = measure.marching_cubes(phi, level=0.9,
                                                  spacing=(h, h, h))
    verts = verts + ax_m[0]
    tri = verts[faces]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300)
    idx = ((cen - ax_m[0]) / h).T
    ins = map_coordinates(inside.astype(float), idx, order=1,
                          mode="nearest") > 0.5
    return tri, nrm, ins


def shaded(nrm, ins, flip: float):
    light = np.array([0.55, 0.5, 0.68])
    light /= np.linalg.norm(light)
    lam = np.clip(flip * (nrm @ light), 0.0, 1.0)
    base = np.where(ins[:, None],
                    np.array(matplotlib.colors.to_rgb(MELT))[None, :],
                    np.array(matplotlib.colors.to_rgb(SPILL))[None, :])
    rgb = np.clip(base * (0.45 + 0.55 * lam)[:, None], 0.0, 1.0)
    return np.concatenate([rgb, np.ones((rgb.shape[0], 1))], axis=1)


def wire(ax, shape, half_mm, h_mm):
    if shape == "cube":
        CU.cube_wire(ax, half_mm)
    else:
        C.pyramid_wire(ax, half_mm, h_mm)


def body_panel(fig, rect, phi, inside, ax_m, shape, half_mm, h_mm, r):
    ax = fig.add_axes(rect, projection="3d")
    tri, nrm, ins = melt_body(phi, inside, ax_m)
    # orient: marching cubes normals point down-gradient, i.e. out of the body
    flip = -1.0
    ax.add_collection3d(Poly3DCollection(tri * MM, facecolors=shaded(nrm, ins, flip),
                                         edgecolors="none"))
    wire(ax, shape, half_mm, h_mm)
    st.dark_3d_axes(ax)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.line.set_color((0, 0, 0, 0))
        a.set_ticks([])
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1), zoom=1.28)
    ax.view_init(elev=C.ELEV, azim=C.AZIM)
    return int(ins.size), int((~ins).sum())


def nominal_outline(shape, half_mm, h_mm):
    """The ANALYTIC nominal cross-section on the vertical mid-plane.

    Drawn from the geometry, not as a contour of the voxelised mask: a mask
    contour puts a staircase on a shape whose whole point is that it has clean
    straight edges.
    """
    if shape == "cube":
        return ([-half_mm, half_mm, half_mm, -half_mm, -half_mm],
                [-half_mm, -half_mm, half_mm, half_mm, -half_mm])
    b2, hz = half_mm, h_mm / 2.0
    return ([-b2, b2, 0.0, -b2], [-hz, -hz, hz, -hz])


def section_panel(fig, rect, phi, phi_ref, inside, ax_m, r, show_ref: bool,
                  outline=None):
    """Vertical mid-plane cut: what the correction added, as filled area."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(st.BG)
    j = int(np.argmin(np.abs(ax_m)))
    X, Z = np.meshgrid(ax_m * MM, ax_m * MM, indexing="ij")
    p = phi[:, j, :]
    ins = inside[:, j, :].astype(float)
    ax.contourf(X, Z, np.where(ins > 0.5, p, np.nan), levels=[0.9, 10.0],
                colors=[MELT])
    ax.contourf(X, Z, np.where(ins <= 0.5, p, np.nan), levels=[0.9, 10.0],
                colors=[SPILL])
    if outline is None:
        ax.contour(X, Z, ins, levels=[0.5], colors=[st.ACCENT], linewidths=1.6)
    else:
        ax.plot(outline[0], outline[1], color=st.ACCENT, lw=1.8)
    if show_ref:
        ax.contour(X, Z, phi_ref[:, j, :], levels=[0.9], colors=["#ffffff"],
                   linewidths=1.4, linestyles="--")
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="cube")
    shape = ap.parse_args().shape

    z = np.load(RESULTS / f"vol_{shape}.npz")
    ax_m = np.asarray(z["axis_m"], float)
    inside = np.asarray(z["inside"], bool)
    half_mm = float(z["nominal_base_side_m"]) / 2.0 * MM
    h_mm = float(z["nominal_height_m"]) * MM
    r = 1.16 * (half_mm if shape == "cube" else h_mm / 2.0)
    doc = json.loads((RESULTS / f"phase_e_{shape}.json").read_text())
    uni = doc["arms"]["uniform_baseline"]
    phi = {a: C.phase_fraction_phi(z[f"T__{a}"]) for a, _ in ARMS}
    ref = phi["uniform_baseline"]

    fig = plt.figure(figsize=(15.0, 9.6), dpi=200)
    fig.patch.set_facecolor(st.BG)
    fig.text(0.035, 0.988, f"{shape}: the part that would form, with and "
             "without correction",
             fontsize=16, color=st.FG, ha="left", va="top")
    fig.text(0.035, 0.955, "solid body is melted material at each arm's own "
             "envelope stop; cyan wireframe is the part we asked for",
             fontsize=11.5, color=st.DIM, ha="left", va="top")

    W, H = 0.290, 0.400
    xs = (0.030, 0.345, 0.660)
    yt, yb = 0.490, 0.120

    for i, (arm, title) in enumerate(ARMS):
        rec = doc["arms"][arm]
        n_all, n_out = body_panel(fig, [xs[i], yt, W, H], phi[arm], inside,
                                  ax_m, shape, half_mm, h_mm, r)
        col = st.GOOD if arm == "solve_filter_only" else (
            st.FG if i == 0 else st.WARM)
        fig.text(xs[i] + W / 2, yt + H + 0.022, title, fontsize=15,
                 color=col, ha="center", va="baseline")
        d = 100.0 * (rec["J_asymmetric"] - uni["J_asymmetric"]) \
            / uni["J_asymmetric"]
        tag = "baseline" if i == 0 else ("%+.0f%% objective J" % d)
        fig.text(xs[i] + W / 2, yt - 0.006,
                 f"melted {100*rec['part_mean_phi']:.0f}% of the part"
                 f"      stop {rec['t_stop_s']:.0f} s",
                 fontsize=12, color=st.DIM, ha="center", va="top")
        fig.text(xs[i] + W / 2, yt - 0.038, tag, fontsize=13.5,
                 color=st.DIM if i == 0 else (st.GOOD if d < 0 else SPILL),
                 ha="center", va="top")

        section_panel(fig, [xs[i], yb, W, 0.300], phi[arm], ref, inside,
                      ax_m, r, show_ref=(i > 0),
                      outline=nominal_outline(shape, half_mm, h_mm))

    fig.text(0.5, 0.098, "vertical cut through the middle; the dashed line "
             "repeats where the melt reached with no correction, so amber "
             "beyond it is what the correction added",
             fontsize=11.5, color=st.DIM, ha="center", va="top")
    fig.text(0.985, 0.988, "amber   melted, inside the part",
             fontsize=11.5, color=MELT, ha="right", va="top")
    fig.text(0.985, 0.956, "red     melted, outside the part",
             fontsize=11.5, color=SPILL, ha="right", va="top")

    fig.text(0.5, 0.058,
             "melt body is phi >= 0.9 on a %d cube render grid; the JSON "
             "out-of-part figure integrates continuous phi and is a different "
             "quantity; surface lightly smoothed for display"
             % ax_m.size,
             fontsize=9.5, color=st.DIM, ha="center")
    fig.text(0.5, 0.030, "solved by 3-D adjoint (filter-only arm), "
             "budget-limited and still descending; simulation-only",
             fontsize=11, color=st.DIM, ha="center")

    out = RESULTS / f"fig_deck_{shape}_three_arms.png"
    fig.savefig(out, dpi=200, facecolor=st.BG)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
