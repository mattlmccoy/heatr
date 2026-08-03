"""Deck figure: the SOLVED pyramid, uniform vs solved (style: deck_figures_3d/style3d).

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.render_deck_pyramid

Rendering only; every number is read from the Phase E result JSONs.
Left: the solved dopant map through the build height, so the through-z
variation is visible at a glance. Right: what it bought.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "deck_figures_3d"))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.gridspec import GridSpec             # noqa: E402
from matplotlib.colors import ListedColormap         # noqa: E402

import style3d as S                                  # noqa: E402
from solve3d import gates as sg                      # noqa: E402
from solve3d.phase_e import geometry as geo          # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "fig_deck_pyramid_solved.png"
CMAP_S = ListedColormap(plt.cm.viridis(np.linspace(0.18, 1.0, 256)))
CMAP_PHI = ListedColormap(plt.cm.inferno(np.linspace(0.06, 1.0, 256)))


def _arms():
    return json.loads((RESULTS / "phase_e_pyramid.json").read_text())["arms"]


def _vertical_section(name: str, nx: int = 260, nz: int = 260):
    """Melt fraction on the x-z plane through the part centre.

    The mid-height horizontal slice is a poor deck panel for a tapered solid:
    at z = 0 the uniform arm has essentially no melt, so its panel renders
    black and the comparison reads as a broken figure rather than a result.
    A vertical section shows the whole build height at once, which is also the
    axis the solved map varies along."""
    from solve3d import forward as fwd
    from solve3d.phase_e import run as R
    tc = R.build_case("pyramid")
    T = np.asarray(np.load(RESULTS / f"field_pyramid_{name}.npz")["T_read"], float)
    W = fwd.functionspace(tc.msh, ("Lagrange", 1))
    Tf = fwd.fem.Function(W)
    Tf.x.array[:] = T.astype(fwd.dolfinx.default_scalar_type)
    half = geo.PYR_B_M / 2.0
    hz = geo.PYR_H_M / 2.0
    xs = np.linspace(-1.35 * half, 1.35 * half, nx)
    zs = np.linspace(-1.30 * hz, 1.30 * hz, nz)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    v, _ = fwd.eval_at(Tf, tc.msh, pts)
    return sg.phase_fraction_phi(v.reshape(nx, nz)), xs, zs


def _fieldz(name):
    z = np.load(RESULTS / f"fieldz_pyramid_{name}.npz")
    return np.asarray(z["T_eval"]), np.asarray(z["z_planes"])


def main() -> int:
    arms = _arms()
    uni, sol = arms["uniform_baseline"], arms["solve_filter_only"]
    m = np.load(RESULTS / "map_pyramid_solve_filter_only.npz")
    s_map = np.asarray(m["s_map"], float)
    cen = np.asarray(m["centroids"], float)
    vols = np.asarray(m["volumes"], float)

    Tu, zs = _fieldz("uniform_baseline")
    Ts, _ = _fieldz("solve_filter_only")
    mid = len(zs) // 2
    x, y, _, _ = sg.eval_grid_axes()
    ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]

    fig = plt.figure(figsize=(15.2, 7.9))
    fig.patch.set_facecolor(S.BG)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 0.52],
                  width_ratios=[1, 1, 1, 1, 1.45, 1.45],
                  left=0.035, right=0.985, top=0.90, bottom=0.115,
                  wspace=0.16, hspace=0.30)

    # ---------------- LEFT: the solved map through the build height --------
    half = geo.PYR_H_M / 2.0
    edges = np.linspace(-half, half, 5)
    vmin, vmax = float(s_map.min()), float(s_map.max())
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(S.BG)
        sel = (cen[:, 2] >= edges[i]) & (cen[:, 2] < edges[i + 1])
        sc = ax.scatter(cen[sel, 0] * 1e3, cen[sel, 1] * 1e3, c=s_map[sel],
                        s=7.5, cmap=CMAP_S, vmin=vmin, vmax=vmax, linewidths=0)
        mu = float(np.average(s_map[sel], weights=vols[sel]))
        ax.set_title(f"z {edges[i]*1e3:+.1f} to {edges[i+1]*1e3:+.1f} mm\n"
                     f"mean sat {mu:.2f}", fontsize=9.5, color=S.FG, pad=5)
        ax.set_aspect("equal")
        ax.set_xlim(-13, 13); ax.set_ylim(-13, 13)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2c3138")
    cax = fig.add_axes([0.045, 0.520, 0.265, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("dopant saturation (solved map)", color=S.FG, fontsize=9.5,
                 labelpad=2)
    cb.ax.tick_params(colors=S.DIM, labelsize=8.5)
    cb.outline.set_edgecolor("#2c3138")

    # z-profile: makes the through-z variation undeniable
    axp = fig.add_subplot(gs[1, 0:4])
    axp.set_facecolor(S.BG)
    nb = 26
    be = np.linspace(-half, half, nb + 1)
    zc, prof = [], []
    for a, b in zip(be, be[1:]):
        sel = (cen[:, 2] >= a) & (cen[:, 2] < b)
        if sel.sum() > 4:
            zc.append(0.5 * (a + b) * 1e3)
            prof.append(float(np.average(s_map[sel], weights=vols[sel])))
    axp.plot(zc, prof, color=S.MELT, lw=2.2)
    axp.axhline(1.0, color=S.DIM, lw=1.0, ls="--")
    axp.text(zc[0], 1.005, "uniform arm = 1.00 everywhere", color=S.DIM,
             fontsize=9, va="bottom")
    axp.set_xlabel("build height z (mm)", color=S.FG, fontsize=10)
    axp.set_ylabel("mean saturation", color=S.FG, fontsize=10)
    axp.tick_params(colors=S.DIM, labelsize=9)
    axp.set_ylim(min(prof) - 0.06, 1.06)
    for sp in axp.spines.values():
        sp.set_color("#2c3138")
    axp.grid(alpha=0.14, color=S.DIM)

    # ---------------- RIGHT: what it bought --------------------------------
    secs = {n: _vertical_section(n) for n in
            ("uniform_baseline", "solve_filter_only")}
    half = geo.PYR_B_M / 2.0
    hz = geo.PYR_H_M / 2.0
    out_x = np.array([-half, half, 0.0, -half]) * 1e3
    out_z = np.array([-hz, -hz, hz, -hz]) * 1e3
    for k, (nm, lab) in enumerate((("uniform_baseline", "uniform"),
                                   ("solve_filter_only", "SOLVED"))):
        ax = fig.add_subplot(gs[0, 4 + k])
        ax.set_facecolor(S.BG)
        phi, xs, zss = secs[nm]
        im = ax.imshow(phi.T, origin="lower", cmap=CMAP_PHI, vmin=0, vmax=1,
                       extent=[xs[0] * 1e3, xs[-1] * 1e3,
                               zss[0] * 1e3, zss[-1] * 1e3],
                       interpolation="bilinear", aspect="equal")
        ax.plot(out_x, out_z, color=S.ACCENT, lw=1.6)
        ax.contour(xs * 1e3, zss * 1e3, phi.T, levels=[0.9],
                   colors=S.GOOD, linewidths=1.5)
        ax.set_title(lab, fontsize=13, color=S.FG, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2c3138")
        if k == 0:
            ax.set_ylabel("build height", color=S.DIM, fontsize=10)
    cax2 = fig.add_axes([0.700, 0.520, 0.265, 0.016])
    cb2 = fig.colorbar(im, cax=cax2, orientation="horizontal")
    cb2.set_label("melt fraction", color=S.FG, fontsize=9.5, labelpad=2)
    cb2.ax.tick_params(colors=S.DIM, labelsize=8.5)
    cb2.outline.set_edgecolor("#2c3138")
    fig.text(0.8325, 0.443,
             "vertical section through the part centre\n"
             "cyan = nominal outline    green = melt front (phi 0.9)",
             color=S.DIM, fontsize=9, ha="center", va="top", linespacing=1.6)

    # numbers block: the ratio is folded into the value string, so no two
    # text objects can ever land on top of each other
    axn = fig.add_subplot(gs[1, 4:6])
    axn.set_facecolor(S.BG)
    axn.axis("off")
    iou_u, iou_s = uni["iou_phi0p9"], sol["iou_phi0p9"]
    rows = [
        ("mean melt fraction", f"{uni['part_mean_phi']:.2f}",
         f"{sol['part_mean_phi']:.2f}",
         f"x{sol['part_mean_phi']/uni['part_mean_phi']:.2f}"),
        ("melt-region IoU", f"{iou_u:.3f}", f"{iou_s:.3f}",
         f"x{iou_s/iou_u:.2f}"),
        ("melt-front distance", f"{uni['front_ssd_mm']:.2f} mm",
         f"{sol['front_ssd_mm']:.2f} mm",
         f"{(sol['front_ssd_mm']/uni['front_ssd_mm']-1)*100:+.0f}%"),
        ("objective J", f"{uni['J_asymmetric']:.2e}",
         f"{sol['J_asymmetric']:.2e}",
         f"{(sol['J_asymmetric']/uni['J_asymmetric']-1)*100:+.0f}%"),
    ]
    axn.text(0.52, 0.985, "uniform", color=S.DIM, fontsize=10.5, va="top",
             ha="right", transform=axn.transAxes)
    axn.text(0.86, 0.985, "solved", color=S.GOOD, fontsize=10.5, va="top",
             ha="right", transform=axn.transAxes)
    for i, (lab, a, b, r) in enumerate(rows):
        yy = 0.775 - i * 0.215
        axn.text(0.0, yy, lab, color=S.FG, fontsize=10.5, va="center",
                 transform=axn.transAxes)
        axn.text(0.52, yy, a, color=S.DIM, fontsize=10.5, va="center",
                 ha="right", transform=axn.transAxes)
        axn.text(0.575, yy, "to", color=S.DIM, fontsize=9.5, va="center",
                 ha="center", transform=axn.transAxes)
        axn.text(0.86, yy, b, color=S.GOOD, fontsize=10.5, va="center",
                 ha="right", transform=axn.transAxes)
        axn.text(1.0, yy, r, color=S.MELT, fontsize=10.5, va="center",
                 ha="right", transform=axn.transAxes)

    fig.text(0.5, 0.028,
             "solved by 3-D adjoint (filter-only arm), budget-limited, still "
             "descending; simulation-only",
             color=S.DIM, fontsize=9.5, ha="center", va="bottom")
    fig.savefig(OUT, dpi=S.DPI, facecolor=S.BG)
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
