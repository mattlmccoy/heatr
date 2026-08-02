"""Figure 1: Dense inside the bounds.

3-D quarter-cutaway isosurfaces of the melt region at melt onset (phi >= 0.9
solid, phi >= 0.8 translucent shell) versus the nominal part wireframe, from
the committed Phase A heatr3d anchor fields. Circle: zero voxels melt outside
the part. Square: the known ~4 percent spill, drawn in the warning color. The anchors are full-height extrusions; a 20 mm mid slab is shown so
the geometry reads as an object (the melt-onset field is z-invariant, spread
across z planes 0.09 to 0.13 C, PHASE_A_REPORT C3).

Note (measured, not assumed): the square's out-of-part melt is NOT at the
corners. On this anchor grid it is a 1-2 voxel skin just outside the two
x-normal faces (x = +-10.3 to +-10.9 mm, |y| < 5.4 mm). The figure and its
caption say "face spill" for that reason.

Data: solve3d/results/anchor_heatr3d_circle_n96.npz and
      solve3d/results/anchor_heatr3d_square_n96.npz (T_phi90, part, x, y, z).
Rendering only; no physics is run.
"""
from __future__ import annotations

import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import style3d as st

MM = 1000.0
Z_HALF = 0.010          # mid-slab half height shown [m]


def load_anchor(name: str):
    z = np.load(st.REPO / "solve3d" / "results" / name)
    return (np.asarray(z["x"]), np.asarray(z["y"]), np.asarray(z["z"]),
            np.asarray(z["T_phi90"]), np.asarray(z["part"]))


def slab(phi: np.ndarray, zax: np.ndarray):
    keep = np.abs(zax) <= Z_HALF + 1e-12
    return phi[:, :, keep], zax[keep]


def quarter_cut(phi: np.ndarray, xax: np.ndarray, yax: np.ndarray) -> np.ndarray:
    """Zero the x>0 & y>0 wedge so marching cubes closes flat cut faces."""
    out = phi.copy()
    wedge = (xax[:, None] > 0) & (yax[None, :] > 0)
    out[wedge, :] = 0.0
    return out


def iso_mesh(phi_cut, x, y, z, level, color, alpha, shade=True):
    """Marching-cubes triangles + per-face RGBA for one isosurface."""
    h = float(x[1] - x[0])
    verts, faces, _, _ = measure.marching_cubes(phi_cut, level=level,
                                                spacing=(h, h, h))
    verts = (verts + np.array([x[0], y[0], z[0]])) * MM
    if shade:
        fc = st.shade_faces(verts, faces, color, light=(0.55, 0.5, 0.68))
        rgba = np.concatenate([fc, np.full((len(faces), 1), alpha)], axis=1)
    else:
        rgb = np.array(plt.matplotlib.colors.to_rgb(color))
        rgba = np.concatenate([np.tile(rgb, (len(faces), 1)),
                               np.full((len(faces), 1), alpha)], axis=1)
    return verts[faces], rgba


def square_wire(ax, half, z0, z1):
    c = st.ACCENT
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot([sx * half] * 2, [sy * half] * 2, [z0, z1],
                    color=c, lw=1.2, alpha=0.95)
    for zz in (z0, z1):
        xs = [-half, half, half, -half, -half]
        ys = [-half, -half, half, half, -half]
        ax.plot(xs, ys, [zz] * 5, color=c, lw=1.2, alpha=0.95)


def circle_wire(ax, r, z0, z1):
    c = st.ACCENT
    t = np.linspace(0, 2 * np.pi, 181)
    for zz in (z0, z1):
        ax.plot(r * np.cos(t), r * np.sin(t), [zz] * t.size,
                color=c, lw=1.2, alpha=0.95)
    for ang in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot([r * np.cos(ang)] * 2, [r * np.sin(ang)] * 2,
                [z0, z1], color=c, lw=0.5, alpha=0.45)


def spill_mesh(phi, part, x, y, z):
    """Out-of-part melt rendered as a warm isosurface sheet (its real extent,
    a 1-2 voxel skin outside the x-normal faces; no exaggeration)."""
    m = ((phi >= 0.9) & (~part)).astype(float)
    n = int(m.sum())
    if not n:
        return None, None, 0
    wedge = (x[:, None] > 0) & (y[None, :] > 0)
    m[wedge, :] = 0.0                        # respect the quarter cutaway
    tri, rgba = iso_mesh(m, x, y, z, 0.5, st.WARM, 1.0)
    return tri, rgba, n


def panel(ax, name, shape, label, note, note_color):
    x, y, z, T, part = load_anchor(name)
    phi = st.phase_fraction_phi(T)
    phi_s, z_s = slab(phi, z)
    part_s, _ = slab(part.astype(float), z)
    phic = quarter_cut(phi_s, x, y)
    tris, rgbas = [], []
    for tri, rgba in (iso_mesh(phic, x, y, z_s, 0.9, st.MELT, 1.0),
                      iso_mesh(phic, x, y, z_s, 0.8, st.SHELL, 0.14,
                               shade=False)):
        tris.append(tri)
        rgbas.append(rgba)
    tri_sp, rgba_sp, n_spill = spill_mesh(phi_s, part_s > 0.5, x, y, z_s)
    if tri_sp is not None:
        tris.append(tri_sp)
        rgbas.append(rgba_sp)
    ax.add_collection3d(Poly3DCollection(np.concatenate(tris),
                                         facecolors=np.concatenate(rgbas),
                                         edgecolors="none"))
    z0, z1 = -Z_HALF * MM, Z_HALF * MM
    if shape == "square":
        square_wire(ax, 10.0, z0, z1)
    else:
        circle_wire(ax, 10.0, z0, z1)
    st.dark_3d_axes(ax)
    r = 11.5
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=38)
    ax.set_title(label, fontsize=13.5, color=st.FG, pad=-10)
    ax.text2D(0.5, 0.015, note, transform=ax.transAxes, ha="center",
              va="top", fontsize=12, color=note_color, fontweight="bold")
    return n_spill


def main() -> None:
    fig = plt.figure(figsize=(12.6, 7.2), dpi=st.DPI)
    ax1 = fig.add_axes([0.00, 0.075, 0.50, 0.76], projection="3d")
    ax2 = fig.add_axes([0.48, 0.075, 0.50, 0.76], projection="3d")

    x, y, z, T, part = load_anchor("anchor_heatr3d_square_n96.npz")
    spill_frac = ((st.phase_fraction_phi(T) >= 0.9) & ~part).sum() / part.sum()

    n_c = panel(ax1, "anchor_heatr3d_circle_n96.npz", "circle", "circle, 20 mm",
                "zero melt outside the bounds", st.GOOD)
    n_s = panel(ax2, "anchor_heatr3d_square_n96.npz", "square", "square, 20 mm",
                f"face spill: {100*spill_frac:.1f}% of part volume", st.WARM)
    assert n_c == 0, "circle should have zero out-of-part melt voxels"
    assert n_s > 0, "square spill voxels expected"

    st.title_block(fig, "DENSE INSIDE THE BOUNDS",
                   "melt region at melt onset vs the nominal part (cyan wire), "
                   "quarter cutaway, 20 mm mid slab of the extrusion")
    fig.text(0.975, 0.022,
             "heatr3d anchor fields n=96  |  both engines agree on the square "
             "spill: 3.81 vs 3.85% on the shared gate grid",
             fontsize=8.5, color=st.DIM, ha="right")
    fig.text(0.975, 0.955, "solid  phi>=0.9 melt", fontsize=9.5, color=st.MELT, ha="right")
    fig.text(0.975, 0.925, "shell  phi>=0.8 melt", fontsize=9.5, color=st.SHELL, ha="right")
    fig.text(0.975, 0.895, "wire   nominal bounds", fontsize=9.5, color=st.ACCENT, ha="right")
    fig.text(0.975, 0.865, "red    melt outside part", fontsize=9.5, color=st.WARM, ha="right")

    out = st.OUT / "fig1_dense_inside_bounds.png"
    fig.savefig(out, dpi=st.DPI)
    print("wrote", out, "| circle spill voxels:", n_c,
          "| square spill voxels (slab):", n_s,
          "| square spill frac (full): %.4f" % spill_frac)


if __name__ == "__main__":
    main()
