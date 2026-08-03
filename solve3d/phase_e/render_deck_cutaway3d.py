"""Deck figure: 3-D quarter-cutaway pair, uniform vs the filter-only solve arm.

    .venv312/bin/python -m solve3d.phase_e.render_deck_cutaway3d

WHAT COLOURS THE SOLID (state this in the caption): MELT FRACTION phi, a real
full 3-D field. `field_<shape>_<arm>.npz` stores `T_read`, the complete nodal
temperature at that arm's own envelope read state, and phi is POINTWISE in T
(clip((T - 180)/10 + 0.5, 0, 1), forward.phase_fraction). So the colour is the
same quantity the shape metrics are scored on, evaluated on a 96^3 grid by
`sample_volume.py`. It is NOT the dopant map and not a proxy.

Style follows deck_figures_3d/fig1_dense_inside_bounds.py: quarter-cut solid
inside the cyan nominal wireframe, Space Mono dark. Differences: the solid is
the NOMINAL PART (so both panels show the identical object and only the colour
differs), and it carries a continuous colour field rather than a flat isosurface
colour. The phi = 0.9 melt front is drawn as a contour on the two cut faces.

Rendering only. No physics, no solve, no mesh work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates
from skimage import measure

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deck_figures_3d"))
import style3d as st                                    # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MM = 1000.0
ELEV, AZIM = 34.0, 32.0
FRONT = 0.9

# Colormap floor lifted off the background: plasma truncated at 0.22 so the
# coldest material is magenta, never near-black on the dark panel.
CMAP = LinearSegmentedColormap.from_list(
    "plasma_lifted", plt.get_cmap("plasma")(np.linspace(0.22, 1.0, 256)))


def phase_fraction_phi(T: np.ndarray) -> np.ndarray:
    """Identical to solve3d.gates.phase_fraction_phi / forward.phase_fraction."""
    return np.clip((np.asarray(T, float) - 180.0) / 10.0 + 0.5, 0.0, 1.0)


def cutaway_mask(inside: np.ndarray, ax_m: np.ndarray) -> np.ndarray:
    """Remove the x>0 & y>0 wedge so the camera looks into the solid."""
    wedge = (ax_m[:, None] > 0) & (ax_m[None, :] > 0)
    out = inside.astype(float).copy()
    out[wedge, :] = 0.0
    return out


def surface(mask_f: np.ndarray, h_m: float, ax0_m: float):
    """Cutaway surface triangles, plus inward-nudged sample points per face."""
    verts, faces, _n, _v = measure.marching_cubes(mask_f, level=0.5,
                                                  spacing=(h_m, h_m, h_m))
    verts = verts + ax0_m
    tri = verts[faces]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300)

    def frac_inside(sign):
        p = cen + sign * 0.9 * h_m * nrm
        idx = ((p - ax0_m) / h_m).T
        return map_coordinates(mask_f, idx, order=1, mode="nearest").mean()

    sign = 1.0 if frac_inside(1.0) > frac_inside(-1.0) else -1.0
    if max(frac_inside(1.0), frac_inside(-1.0)) < 0.6:
        raise RuntimeError("cannot orient faces into the solid")
    return tri, cen + sign * 0.9 * h_m * nrm, nrm * sign, verts, faces


def face_colors(phi_vol, sample_pts, ax0_m, h_m, normals, cmap=None,
                norm=None):
    """Per-face RGBA: the field through a colormap, lambertian-shaded.

    `cmap`/`norm` default to the melt-fraction pair used by the cutaway pair
    figure, so that figure renders bit-identically; the loop schematic passes
    its own for the mismatch and gradient stages.
    """
    cmap = CMAP if cmap is None else cmap
    idx = ((sample_pts - ax0_m) / h_m).T
    phi_f = map_coordinates(phi_vol, idx, order=1, mode="nearest")
    phi_f = np.clip(phi_f, 0.0, 1.0) if norm is None else norm(phi_f)
    rgba = cmap(phi_f)
    light = np.array([0.55, 0.5, 0.68])
    light = light / np.linalg.norm(light)
    lam = np.clip(-(normals @ light), 0.0, 1.0)     # normals point INTO solid
    shade = (0.55 + 0.45 * lam)[:, None]
    rgba[:, :3] = np.clip(rgba[:, :3] * shade, 0.0, 1.0)
    return rgba, phi_f


def pyramid_wire(ax, b2_mm, h_mm):
    """Nominal pyramid: square base + four slant edges to the apex."""
    c, z0, z1 = st.ACCENT, -h_mm / 2.0, h_mm / 2.0
    xs = [-b2_mm, b2_mm, b2_mm, -b2_mm, -b2_mm]
    ys = [-b2_mm, -b2_mm, b2_mm, b2_mm, -b2_mm]
    ax.plot(xs, ys, [z0] * 5, color=c, lw=1.3, alpha=0.95)
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot([sx * b2_mm, 0.0], [sy * b2_mm, 0.0], [z0, z1],
                    color=c, lw=1.3, alpha=0.95)


def front_on_cut_faces(ax, phi_vol, ax_m, off_mm=0.30):
    """phi = 0.9 contour drawn on the two exposed cut planes, nudged toward the
    camera so matplotlib's painter order cannot bury it in the surface."""
    n = ax_m.size
    i0 = int(np.argmin(np.abs(ax_m)))
    hi = np.arange(n) >= i0
    segs = []
    # plane x = 0, material at y > 0  -> (y, z), drawn at x = +off
    sl = phi_vol[i0][hi][:, :]
    for c in measure.find_contours(sl, FRONT):
        yy = np.interp(c[:, 0], np.arange(hi.sum()), ax_m[hi]) * MM
        zz = np.interp(c[:, 1], np.arange(n), ax_m) * MM
        segs.append((np.full_like(yy, off_mm), yy, zz))
    # plane y = 0, material at x > 0  -> (x, z), drawn at y = +off
    sl = phi_vol[:, i0][hi][:, :]
    for c in measure.find_contours(sl, FRONT):
        xx = np.interp(c[:, 0], np.arange(hi.sum()), ax_m[hi]) * MM
        zz = np.interp(c[:, 1], np.arange(n), ax_m) * MM
        segs.append((xx, np.full_like(xx, off_mm), zz))
    for a, b, c3 in segs:
        ax.plot(a, b, c3, color=st.GOOD, lw=2.0, alpha=0.95, zorder=10)
    return len(segs)


def panel(ax, phi_vol, ax_m, mask_f, title, mean_phi, iou, b_m, h_m_nom):
    h = float(ax_m[1] - ax_m[0])
    tri, spts, nrm, _v, _f = surface(mask_f, h, ax_m[0])
    rgba, _ = face_colors(phi_vol, spts, ax_m[0], h, nrm)
    ax.add_collection3d(Poly3DCollection(tri * MM, facecolors=rgba,
                                         edgecolors="none"))
    nseg = front_on_cut_faces(ax, phi_vol, ax_m)
    pyramid_wire(ax, b_m / 2.0 * MM, h_m_nom * MM)
    st.dark_3d_axes(ax)
    r = 12.2
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    # zoom fills the panel: matplotlib's 3-D axes otherwise leave ~25% margin,
    # which is what made the first render read as a small flat triangle
    ax.set_box_aspect((1, 1, 1), zoom=1.30)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title(title, fontsize=15, color=st.FG, pad=6)
    return nseg


def main() -> int:
    z = np.load(RESULTS / "vol_pyramid.npz")
    ax_m = np.asarray(z["axis_m"], float)
    inside = np.asarray(z["inside"], bool)
    mask_f = cutaway_mask(inside, ax_m)
    phi_u = phase_fraction_phi(z["T__uniform_baseline"])
    phi_s = phase_fraction_phi(z["T__solve_filter_only"])
    # outside the part the colour is never sampled, but keep it finite
    phi_u = np.where(inside, phi_u, 0.0)
    phi_s = np.where(inside, phi_s, 0.0)

    import json
    doc = json.loads((RESULTS / "phase_e_pyramid.json").read_text())
    u, s = doc["arms"]["uniform_baseline"], doc["arms"]["solve_filter_only"]

    fig = plt.figure(figsize=(13.6, 7.8), dpi=st.DPI)
    fig.patch.set_facecolor(st.BG)
    axL = fig.add_axes([0.02, 0.145, 0.46, 0.790], projection="3d")
    axR = fig.add_axes([0.47, 0.145, 0.46, 0.790], projection="3d")
    b_m = float(z["nominal_base_side_m"])
    hn_m = float(z["nominal_height_m"])
    n1 = panel(axL, phi_u, ax_m, mask_f, "uniform",
               u["part_mean_phi"], u["iou_phi0p9"], b_m, hn_m)
    n2 = panel(axR, phi_s, ax_m, mask_f, "solve (filter-only arm)",
               s["part_mean_phi"], s["iou_phi0p9"], b_m, hn_m)

    # colourbar on its own column, well clear of the legend text above it
    cax = fig.add_axes([0.945, 0.20, 0.016, 0.42])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                      cax=cax)
    cb.set_label("melt fraction phi", color=st.FG, fontsize=12, labelpad=-58)
    cb.ax.tick_params(colors=st.DIM, labelsize=10)
    cb.outline.set_edgecolor(st.DIM)

    for i, (txt, col) in enumerate((("quarter cutaway", st.DIM),
                                    ("cyan  nominal part", st.ACCENT),
                                    ("green  phi 0.9 front", st.GOOD))):
        fig.text(0.995, 0.92 - 0.033 * i, txt, fontsize=11, color=col,
                 ha="right")

    for xc, rec in ((0.25, u), (0.70, s)):
        fig.text(xc, 0.128,
                 f"mean phi {rec['part_mean_phi']:.2f}"
                 f"     melt IoU {rec['iou_phi0p9']:.3f}",
                 fontsize=12.5, color=st.FG, ha="center")

    fig.text(0.5, 0.072, "melt fraction on the full 3-D solve mesh, "
             "each arm read at its own envelope stop",
             fontsize=11, color=st.DIM, ha="center")
    fig.text(0.5, 0.032, "solved by 3-D adjoint (filter-only arm), "
             "budget-limited and still descending; simulation-only",
             fontsize=11, color=st.DIM, ha="center")

    out = RESULTS / "fig_deck_pyramid_cutaway3d.png"
    fig.savefig(out, dpi=st.DPI, facecolor=st.BG)
    print("wrote", out, "| front segments", n1, n2,
          "| in-part mean phi grid %.4f vs %.4f" %
          (phi_u[inside].mean(), phi_s[inside].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
