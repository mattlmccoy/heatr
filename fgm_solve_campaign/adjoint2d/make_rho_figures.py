"""Figures for the density-region solve.

  fig_rho_<shape>.png   one composite per shape. Row 1 is the dopant map of
      every arm; row 2 is the melt fraction at that arm's own PHI-STOP; row 3 is
      the normalized relative density at that arm's own RHO-STOP (the flat
      onset). The nominal part is cyan, the melt front and the densified front
      are dashed white. The question it answers at a glance is whether the two
      objectives want the same map and whether they want the same stop.

  fig_rho_census.png    the whole-library verdict. Panel A is the densified
      fraction of the part; panel B is the shape-versus-density tension (mean
      relative density read at the melt stop against read at the density stop);
      panel C is what the density-solved map costs on the melt objective; panel
      D is the two stop times.

READ/STOP CONVENTIONS drawn on every number: PHI-STOP is the argmin over that
arm's own trajectory of J_phi; RHO-STOP is the flat onset of J_rho. The melted
region is melt fraction >= 0.5; the densified region is normalized relative
density >= 0.5. Grid 120 x 120 throughout.

Run:
  ./.venv312/bin/python -m adjoint2d.make_rho_figures shape <shape> <out_rho> <figs_rho>
  ./.venv312/bin/python -m adjoint2d.make_rho_figures census <out_rho> <figs_rho>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import density_objective as dobj, shape_objective as so
from .library_solve import SHAPES, shape_config
from .make_figures import crop_box
from .pins import build_case, load_cfg
from .rho_solve import run_forward

DPI = 180

PANEL_ORDER = [
    ("U_uniform", "uniform s = 1\nno grading", False),
    ("HIST_best", "best stored historical mask\n4 bits per pixel, permittivity co-varying", True),
    ("PHI4_prev", "MELT-solved map, 4 bits per pixel\nthe previous library deliverable", False),
    ("RHO_4bpp", "DENSITY-solved map, 4 bits per pixel\nfiltered design, THE deliverable", False),
    ("RHO_4bpp_nofilter", "DENSITY-solved, 4 bits per pixel\nUNFILTERED control", False),
]


def _map_panel(ax, s, pm, extent, title):
    im = ax.imshow(np.where(pm, s, np.nan), origin="lower", extent=extent,
                   vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
    xs = np.linspace(extent[0], extent[1], pm.shape[1])
    ys = np.linspace(extent[2], extent[3], pm.shape[0])
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#ffffff", linewidths=1.0)
    ax.set_title(title, fontsize=7.0, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _field_panel(ax, f, pm, extent, title, cmap, front=0.5, vmin=0.0):
    im = ax.imshow(f, origin="lower", extent=extent, vmin=vmin, vmax=1.0,
                   cmap=cmap, interpolation="bilinear")
    xs = np.linspace(extent[0], extent[1], pm.shape[1])
    ys = np.linspace(extent[2], extent[3], pm.shape[0])
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.4)
    if np.nanmax(f) > front > np.nanmin(f):
        ax.contour(xs, ys, f, levels=[front], colors="#ffffff", linewidths=1.0,
                   linestyles="--")
    ax.set_title(title, fontsize=7.0, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def shape_figure(shape: str, outdir: Path, figdir: Path) -> Path:
    res = json.loads((outdir / f"{shape}.json").read_text())
    case = build_case(load_cfg(shape_config(shape)))
    pm = case.part_mask
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    with np.load(outdir / f"{shape}_maps.npz") as d:
        maps = {k: np.asarray(d[k], dtype=float) for k in d.files}

    cols = [(k, lab, maps[k], cov) for k, lab, cov in PANEL_ORDER
            if k in maps and k in res["arms"]]
    fig, axes = plt.subplots(3, len(cols), figsize=(2.85 * len(cols), 10.4))
    if len(cols) == 1:
        axes = axes.reshape(3, 1)
    im_map = im_phi = im_rho = None
    for j, (key, label, s, covary) in enumerate(cols):
        m = res["arms"][key]
        im_map = _map_panel(axes[0, j], s, pm, extent,
                            f"{label}\nmean s in part {float(np.mean(s[pm])):.3f}   "
                            f"roughness {m['map_roughness']:.4f}")
        tr = run_forward(case, s, eps_covary=covary)
        phi = so.phi_field(tr.T_at_end(int(m["t_stop_index"])), case)[0]
        rn = dobj.rho_norm(dobj.rho_effective(tr.rho_at_end(int(m["rho_stop_index"])), case), case)
        del tr
        im_phi = _field_panel(
            axes[1, j], phi, pm, extent,
            f"melt fraction at the PHI-STOP {m['t_stop_s']:.0f} s"
            f"{' (AT HORIZON)' if m['t_stop_at_horizon'] else ''}\n"
            f"J_phi {m['J']:.1f}     IoU {m['IoU']:.4f}\n"
            f"growth {m['bed_melt_pct_of_part']:.2f} %     "
            f"under {m['part_under_melt_pct']:.2f} %\n"
            f"mean relative density there {m['mean_rho_rel_part_at_phi_stop']:.3f}",
            "inferno")
        im_rho = _field_panel(
            axes[2, j], rn, pm, extent,
            f"normalized density at the RHO-STOP {m['rho_stop_s']:.0f} s"
            f"{' (AT HORIZON)' if m['rho_stop_at_horizon'] else ''}\n"
            f"J_rho {m['J_rho']:.1f}     densified fraction {m['IoU_rho']:.4f}\n"
            f"mean relative density {m['mean_rho_rel_part']:.3f}\n"
            f"melt IoU there {m['IoU_at_rho_stop']:.4f}, "
            f"growth {m['bed_melt_pct_of_part_at_rho_stop']:.1f} %",
            # The colour scale starts at 0.5, not 0: at the flat onset every
            # arm is above 0.5 almost everywhere, so a full-range scale would
            # render every panel a flat single colour and hide the residual
            # under-densification that J_rho is actually made of. The dashed
            # 0.5 contour still marks anything that falls below the floor.
            "cividis", vmin=0.5)
        for i in range(3):
            axes[i, j].set_xlim(box[0], box[1]); axes[i, j].set_ylim(box[2], box[3])
    fig.colorbar(im_map, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01,
                 label="binder saturation s")
    fig.colorbar(im_phi, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01,
                 label="melt fraction")
    fig.colorbar(im_rho, ax=axes[2, :].tolist(), fraction=0.02, pad=0.01,
                 label="normalized density, 0.50 to 1")
    v = res["verdict"]
    fig.suptitle(
        f"{shape}   density-region solve, grid 120 x 120, design filter sigma = "
        f"{res['sigma_cells']} cells, flat-onset tolerance {res['flat_tol']}.   "
        f"Best start: {res.get('best_start', 'n/a')}.   "
        f"Cyan is the nominal part; dashed white is the melt front (row 2) and the "
        f"densified front (row 3).\n"
        f"Deliverable mean relative density {v['mean_rho_at_rho_stop']:.3f} at its own "
        f"density stop against {v['mean_rho_at_phi_stop']:.3f} at its own melt stop.",
        fontsize=9.5)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"fig_rho_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def census_rows(outdir: Path) -> list[dict]:
    rows = []
    for sh in SHAPES:
        f = Path(outdir) / f"{sh}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        a = r["arms"]
        if "RHO_4bpp" not in a:
            continue
        d, u = a["RHO_4bpp"], a["U_uniform"]
        p4 = a.get("PHI4_prev")
        rows.append({
            "shape": sh,
            "rho_solved": d["mean_rho_rel_part"], "rho_solved_at_phi": d["mean_rho_rel_part_at_phi_stop"],
            "rho_uniform": u["mean_rho_rel_part"], "rho_uniform_at_phi": u["mean_rho_rel_part_at_phi_stop"],
            "rho_phi4": None if p4 is None else p4["mean_rho_rel_part"],
            "rho_phi4_at_phi": None if p4 is None else p4["mean_rho_rel_part_at_phi_stop"],
            "dens_solved": d["dense_frac_0p90"], "dens_uniform": u["dense_frac_0p90"],
            "dens_phi4": None if p4 is None else p4["dense_frac_0p90"],
            "half_solved": d["IoU_rho"], "half_uniform": u["IoU_rho"],
            "half_phi4": None if p4 is None else p4["IoU_rho"],
            "Jrho_solved": d["J_rho"], "Jrho_uniform": u["J_rho"],
            "Jrho_phi4": None if p4 is None else p4["J_rho"],
            "iou_solved": d["IoU"], "iou_uniform": u["IoU"],
            "iou_phi4": None if p4 is None else p4["IoU"],
            "Jphi_solved": d["J"], "Jphi_phi4": None if p4 is None else p4["J"],
            "t_phi": d["t_stop_s"], "t_rho": d["rho_stop_s"],
            "iou_at_rho_stop": d["IoU_at_rho_stop"],
            "rho_horizon": d["rho_stop_at_horizon"],
            "grad_death": r["verdict"]["gradient_death_any"],
        })
    return rows


def census_figure(outdir: Path, figdir: Path) -> Path:
    rows = sorted(census_rows(outdir), key=lambda r: r["dens_solved"])
    n = len(rows)
    lab = [r["shape"] for r in rows]
    yy = np.arange(n)
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 0.46 * n + 3.4))

    ax = axes[0]
    ax.barh(yy - 0.22, [r["dens_uniform"] for r in rows], height=0.2,
            color="#9e9e9e", label="uniform s = 1")
    ax.barh(yy, [r["dens_phi4"] or 0.0 for r in rows], height=0.2,
            color="#ef6c00", label="melt-solved 4 bits per pixel")
    ax.barh(yy + 0.22, [r["dens_solved"] for r in rows], height=0.2,
            color="#1565c0", label="density-solved 4 bits per pixel")
    ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("fraction of the part above normalized relative density 0.90,\n"
                  "at that arm's own density stop", fontsize=8)
    ax.set_title("A. densified region, read at the 0.90 level\n"
                 "(the 0.50 level is 1.00 on %d of %d arms and cannot discriminate)"
                 % (sum(1 for r in rows for v in (r["half_solved"], r["half_uniform"],
                                                  r["half_phi4"]) if v is not None and v >= 0.9999),
                    sum(1 for r in rows for v in (r["half_solved"], r["half_uniform"],
                                                  r["half_phi4"]) if v is not None)),
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.075),
              frameon=False)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    for k, r in enumerate(rows):
        ax.plot([r["rho_solved_at_phi"], r["rho_solved"]], [k, k], color="#1565c0", lw=1.2)
    ax.scatter([r["rho_solved_at_phi"] for r in rows], yy, s=26, color="#c62828",
               zorder=3, label="read at the MELT stop")
    ax.scatter([r["rho_solved"] for r in rows], yy, s=26, color="#1565c0",
               zorder=3, label="read at the DENSITY stop")
    ax.axvline(0.68, color="#455a64", ls=":", lw=1.2)
    ax.text(0.68, n - 0.2, " the 0.68 class of the\n triangle showcase",
            fontsize=7, color="#455a64", va="top")
    ax.set_yticks(yy); ax.set_yticklabels([]); ax.set_xlabel(
        "mean relative density in the part, density-solved map", fontsize=8)
    ax.set_title("B. the shape-versus-density tension", fontsize=10)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    for k, r in enumerate(rows):
        if r["iou_phi4"] is None:
            continue
        ax.plot([r["iou_phi4"], r["iou_solved"]], [k, k], color="#6d4c41", lw=1.2)
    ax.scatter([r["iou_phi4"] for r in rows if r["iou_phi4"] is not None],
               [k for k, r in enumerate(rows) if r["iou_phi4"] is not None],
               s=26, color="#ef6c00", zorder=3, label="melt-solved map")
    ax.scatter([r["iou_solved"] for r in rows], yy, s=26, color="#1565c0",
               zorder=3, label="density-solved map")
    ax.scatter([r["iou_at_rho_stop"] for r in rows], yy, s=22, marker="x",
               color="#c62828", zorder=3, label="density-solved, read at the DENSITY stop")
    ax.set_yticks(yy); ax.set_yticklabels([])
    ax.set_xlabel("melt-region intersection over union at the MELT stop", fontsize=8)
    ax.xaxis.set_label_coords(0.5, -0.035)
    ax.set_title("C. what the density objective costs on shape", fontsize=10)
    lo = min([r["iou_at_rho_stop"] for r in rows]
             + [r["iou_solved"] for r in rows]
             + [r["iou_phi4"] for r in rows if r["iou_phi4"] is not None])
    ax.set_xlim(lo - 0.04, 1.02)
    # Below the axes: every in-panel corner holds a marker on some shape, and a
    # legend that hides a data point is worse than one that costs a little room.
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.055), ncol=1,
              frameon=False)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[3]
    ax.barh(yy - 0.18, [r["t_phi"] for r in rows], height=0.34, color="#ef6c00",
            label="melt stop")
    ax.barh(yy + 0.18, [r["t_rho"] for r in rows], height=0.34, color="#1565c0",
            label="density stop (flat onset)")
    for k, r in enumerate(rows):
        if r["rho_horizon"]:
            ax.text(r["t_rho"], k + 0.18, " HORIZON", fontsize=6, va="center")
    ax.set_yticks(yy); ax.set_yticklabels([])
    ax.set_xlabel("read time, seconds, density-solved map", fontsize=8)
    ax.set_title("D. the two read states", fontsize=10)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(axis="x", alpha=0.3)

    n_dens = sum(1 for r in rows if r["dens_phi4"] is not None
                 and r["dens_solved"] > r["dens_phi4"])
    n_jrho = sum(1 for r in rows if r["Jrho_solved"] < (r["Jrho_phi4"] or np.inf))
    n_iou = sum(1 for r in rows if r["iou_solved"] > (r["iou_phi4"] or np.inf))
    fig.suptitle(
        f"Density-region objective across {n} shapes, grid 120 x 120, design filter "
        f"sigma = 1.5 cells, 4 bits per pixel, single printing pass.   "
        f"Density-solved beats melt-solved on J_rho on {n_jrho} of {n}, on the densified "
        f"fraction on {n_dens} of {n}, and on melt intersection over union on {n_iou} of {n}.\n"
        "Every number is read at the stated stop for that arm; no arm is read at a fixed time.",
        fontsize=10)
    figdir = Path(figdir); figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / "fig_rho_census.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    if sys.argv[1] == "shape":
        print(shape_figure(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])))
    elif sys.argv[1] == "census":
        print(census_figure(Path(sys.argv[2]), Path(sys.argv[3])))
    else:
        raise SystemExit("usage: shape <shape> <out> <figs> | census <out> <figs>")
