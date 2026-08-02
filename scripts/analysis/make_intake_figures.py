#!/usr/bin/env python3
"""Figures and tables for the geometry generalization pass.

Three figures, all written to `fgm_solve_campaign/figs_intake/`:

  fig_intake_routes.png       the three intake routes (polygon, binary mask,
                              PNG mask) on the SAME geometry, with the measured
                              disagreement between their target indicators;
  fig_intake_classifier.png   the anisotropy spectrum and the actuator
                              recommendation on all eighteen library shapes,
                              with the five campaign-measured outcomes marked;
  fig_intake_novel_<name>.png the novel geometry end to end: chi, the
                              anisotropy spectrum, the kernels, the solved
                              4 bits per pixel map, and the melt region at the
                              arm's own stop against the nominal outline.

Run:  ./.venv312/bin/python scripts/analysis/make_intake_figures.py [name]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import geometry_actuator as ga        # noqa: E402
from adjoint2d import geometry_intake as gi          # noqa: E402
from adjoint2d import library_solve as lib           # noqa: E402
from novel_shapes import NOVEL                       # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
FIGS = REPO / "fgm_solve_campaign/figs_intake"
DPI = 180
MM = 1000.0


def _outline(ax, poly, **kw):
    p = np.vstack([poly, poly[:1]])
    ax.plot(p[:, 0] * MM, p[:, 1] * MM, **kw)


def _field(ax, f, x, y, title, cmap="magma", vmin=None, vmax=None):
    im = ax.imshow(f, origin="lower", cmap=cmap, interpolation="bilinear",
                   extent=[x[0] * MM, x[-1] * MM, y[0] * MM, y[-1] * MM],
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


# ---------------------------------------------------------------------------
# figure 1: the three routes agree
# ---------------------------------------------------------------------------

def figure_routes(name: str = "gear8") -> dict:
    from PIL import Image

    poly = NOVEL[name]()
    it_poly = gi.from_polygon(poly, grid=120, name=name)

    # Rasterize the same outline at 400 by 400 and re-import it as a MASK.
    n = 400
    half = 0.030
    xs = np.linspace(-half, half, n)
    ys = np.linspace(-half, half, n)
    from adjoint2d import geometry_contour as gc
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    mask = gc.point_in_polygon_evenodd(poly, xx.ravel(), yy.ravel()).reshape(n, n)
    pitch = float(xs[1] - xs[0])
    it_mask = gi.from_mask(mask, pixel_pitch_m=pitch, grid=120, name=name)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"route_input_{name}_mask.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(mask).astype(np.uint8) * 255, "L").save(png)
    it_png = gi.from_png(png, pixel_pitch_m=pitch, grid=120, name=name)

    a_poly = it_poly.info["area_chi_m2"]
    stats = {
        "polygon_area_m2": a_poly,
        "mask_area_m2": it_mask.info["area_chi_m2"],
        "png_area_m2": it_png.info["area_chi_m2"],
        "mask_vs_polygon_rel": (it_mask.info["area_chi_m2"] - a_poly) / a_poly,
        "png_vs_mask_max_cell_delta": float(np.max(np.abs(it_png.chi - it_mask.chi))),
        "mask_vs_polygon_max_cell_delta": float(np.max(np.abs(it_mask.chi - it_poly.chi))),
        "mask_pixels": int(mask.sum()),
        "mask_pitch_m": pitch,
    }

    fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.0))
    _field(axs[0], it_poly.chi, it_poly.x, it_poly.y, "route (b) polygon: chi", "viridis")
    _outline(axs[0], poly, color="w", lw=0.7)
    _field(axs[1], it_mask.chi, it_mask.x, it_mask.y,
           f"route (a) mask {n} x {n}: chi", "viridis")
    _field(axs[2], it_png.chi, it_png.x, it_png.y, "route (c) PNG mask: chi", "viridis")
    d = it_mask.chi - it_poly.chi
    lim = max(stats["mask_vs_polygon_max_cell_delta"], 1e-6)
    im = _field(axs[3], d, it_poly.x, it_poly.y,
                f"mask minus polygon (own scale, +/- {lim:.3f})\nmax cell "
                f"{stats['mask_vs_polygon_max_cell_delta']:.3f}, "
                f"area {stats['mask_vs_polygon_rel']*100:+.3f} pct", "coolwarm",
                vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axs[3], fraction=0.046)
    for a in axs:
        a.set_xlim(-16, 16)
        a.set_ylim(-16, 16)
    fig.suptitle(f"Geometry intake, three routes to the same target indicator "
                 f"({name}), grid 120", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_intake_routes.png", dpi=DPI)
    plt.close(fig)
    (OUT / "route_equivalence.json").write_text(json.dumps(stats, indent=2, default=float))
    return stats


# ---------------------------------------------------------------------------
# figure 2: the classifier on the eighteen library shapes
# ---------------------------------------------------------------------------

def figure_classifier() -> dict:
    rows = []
    for s in lib.SHAPES:
        p = OUT / f"{s}_intake.json"
        if not p.exists():
            continue
        rows.append(json.loads(p.read_text()))
    rows.sort(key=lambda r: r["recommendation"]["residual_anisotropy"])

    names = [r["shape"] for r in rows]
    a_static = [r["anisotropy"]["static"] for r in rows]
    a_best = [r["recommendation"]["residual_anisotropy"] for r in rows]
    known = [r.get("known_rotation_outcome") for r in rows]

    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    xi = np.arange(len(rows))
    ax.bar(xi - 0.2, a_static, width=0.4, color="0.72", label="static kernel")
    colors = {"MODE_SUFFICES": "#2c7bb6", "MAP_PLUS_MODE": "#fdae61",
              "PHYSICAL_LIMIT": "#d7191c"}
    ax.bar(xi + 0.2, a_best, width=0.4,
           color=[colors[r["recommendation"]["actuator_class"]] for r in rows],
           label="best mode (coloured by recommendation)")
    ax.axhline(ga.A_MODE_SUFFICES, color="k", ls="--", lw=1.0)
    ax.axhline(ga.A_PHYSICAL_LIMIT, color="k", ls=":", lw=1.0)
    ax.text(len(rows) - 0.4, ga.A_MODE_SUFFICES + 0.02, "MODE_SUFFICES band",
            ha="right", fontsize=8)
    ax.text(len(rows) - 0.4, ga.A_PHYSICAL_LIMIT + 0.02, "PHYSICAL_LIMIT band",
            ha="right", fontsize=8)
    for i, (k, r) in enumerate(zip(known, rows)):
        if k:
            ax.annotate("rotation WINS" if k == "rotation_wins" else "rotation FAILS",
                        (i + 0.2, a_best[i]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7,
                        color="#1a9641" if k == "rotation_wins" else "#d7191c",
                        rotation=90)
        ax.annotate(r["recommendation"]["mode"], (i + 0.2, 0.02), ha="center",
                    fontsize=6, rotation=90, color="w")
    ax.set_xticks(xi)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("residual azimuthal anisotropy (occupancy weighted)")
    ax.set_title("Actuator classifier on the eighteen library shapes, grid 120, "
                 "uniform dopant map. Labels mark the five outcomes the rotation "
                 "campaign MEASURED.", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(max(a_static), 1.35) * 1.15)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_intake_classifier.png", dpi=DPI)
    plt.close(fig)
    return {"n_shapes": len(rows)}


# ---------------------------------------------------------------------------
# figure 3: the novel geometry end to end
# ---------------------------------------------------------------------------

def figure_novel(name: str) -> dict:
    res = json.loads((OUT / f"{name}_novel.json").read_text())
    d = np.load(OUT / f"{name}_intake.npz")
    x, y, chi, pm = d["x"], d["y"], d["chi"], d["part_mask"]
    poly = d["polygon"]
    maps_path = OUT / f"{name}_maps.npz"
    has_solve = maps_path.exists()
    m = np.load(maps_path) if has_solve else None

    fig, axs = plt.subplots(2, 3, figsize=(15.0, 9.0))

    _field(axs[0, 0], chi, x, y,
           f"(a) target indicator chi, area fill\n{res['intake']['n_part_cells']} part "
           f"cells, {res['intake']['area_chi_m2']*1e6:.1f} mm^2", "viridis")
    _outline(axs[0, 0], poly, color="w", lw=0.8)

    ax = axs[0, 1]
    spec = res["anisotropy"]
    keys = list(spec)
    vals = [spec[k] for k in keys]
    cls = res["recommendation"]["actuator_class"]
    colors = {"MODE_SUFFICES": "#2c7bb6", "MAP_PLUS_MODE": "#fdae61",
              "PHYSICAL_LIMIT": "#d7191c"}
    bar_c = ["0.72" if k != res["recommendation"]["mode"] else colors[cls] for k in keys]
    ax.bar(np.arange(len(keys)), vals, color=bar_c)
    ax.axhline(ga.A_MODE_SUFFICES, color="k", ls="--", lw=1.0)
    ax.axhline(ga.A_PHYSICAL_LIMIT, color="k", ls=":", lw=1.0)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("residual anisotropy")
    ax.set_title(f"(b) anisotropy spectrum\nrecommendation {cls} via "
                 f"{res['recommendation']['mode']}", fontsize=9)

    k_static = d["kernel_static"]
    key_mode = f"kernel_{res['recommendation']['mode']}"
    k_mode = d[key_mode] if key_mode in d.files else k_static
    vmax = float(np.percentile(k_static[pm], 97.0))
    _field(axs[0, 2], np.where(pm, k_static, np.nan), x, y,
           "(c) static heating kernel Q_rf (state B)", "inferno", 0, vmax)
    _outline(axs[0, 2], poly, color="w", lw=0.6)

    _field(axs[1, 0], np.where(pm, k_mode, np.nan), x, y,
           f"(d) {res['recommendation']['mode']} averaged kernel", "inferno", 0, vmax)
    _outline(axs[1, 0], poly, color="w", lw=0.6)

    if has_solve:
        best = res["verdict"]["best_arm"]
        # the DELIVERABLE is always the 4 bits per pixel arm; show the one that
        # belongs to the best arm's mode
        mode_tag = best.replace("A_", "").replace("U_uniform_", "")
        mode_tag = mode_tag.replace("_cont", "").replace("_4bpp", "")
        key = f"{mode_tag}_4bpp"
        if key not in m.files:
            key = "static_4bpp"
        _field(axs[1, 1], np.where(pm, m[key], np.nan), x, y,
               f"(e) deliverable dopant map, 4 bits per pixel ({key})\nbest arm "
               f"overall {best}", "cividis", 0, 1)
        _outline(axs[1, 1], poly, color="w", lw=0.6)

        pkey = "phi_" + key
        phi = m[pkey] if pkey in m.files else m["phi_static_4bpp"]
        shown_arm = ("A_static_4bpp" if key == "static_4bpp"
                     else f"A_{key.replace('_4bpp', '')}_4bpp")
        ms = res["arms"].get(shown_arm, {})
        ax = axs[1, 2]
        im = _field(ax, phi, x, y,
                    f"(f) melt fraction at the deliverable's own stop, arm "
                    f"{shown_arm}\nIoU {ms.get('IoU', float('nan')):.4f}, J "
                    f"{ms.get('J', float('nan')):.2f} (grid 120, area-fill chi)",
                    "hot", 0, 1)
        _outline(ax, poly, color="#00e5ff", lw=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        for a in (axs[1, 1], axs[1, 2]):
            a.text(0.5, 0.5, "spectrum-only pass\n(no solve run)", ha="center",
                   va="center", fontsize=11)
            a.set_axis_off()

    for a in (axs[0, 0], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]):
        if a.has_data():
            a.set_xlim(-18, 18)
            a.set_ylim(-18, 18)
    sym = res["symmetry"]
    fig.suptitle(f"Novel geometry '{name}' end to end: intake, calibration "
                 f"({res['intake']['voltage_v']:.0f} V for 500 W/m), symmetry "
                 f"{sym['point_group']}, actuator recommendation, solve. Grid 120.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / f"fig_intake_novel_{name}.png", dpi=DPI)
    plt.close(fig)
    return {"figure": str(FIGS / f"fig_intake_novel_{name}.png")}


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["routes", "classifier", "gear8"]
    for w in which:
        if w == "routes":
            print("routes", json.dumps(figure_routes(), indent=2, default=float))
        elif w == "classifier":
            print("classifier", figure_classifier())
        else:
            print("novel", figure_novel(w))
