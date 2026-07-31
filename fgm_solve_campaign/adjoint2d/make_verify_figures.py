"""Figures for the verification and printability pass.

Two families.

  fig_print_<shape>.png   the solved dopant map continuous, at 4 bits per pixel
      and at 2 bits per pixel, side by side, with the melted region at each
      variant's OWN optimal stop underneath and the nominal part outline over
      it. The question the figure has to answer at a glance is whether the melt
      front moves when the map is put on the printer's level grid.

  fig_hist_<shape>.png    the ACTUAL stored historical 4-bits-per-pixel mask
      against the solved map, same engine, same configuration, each at its own
      optimal stop.

Run: ./.venv312/bin/python -m adjoint2d.make_verify_figures <out_verify> <figs>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import forward as fwd, shape_objective as so
from .make_figures import crop_box
from .pins import build_case, load_cfg
from .verify_hist import CFGD, WINNERS, load_stored_map

DPI = 180
PATIENCE = 250


def _map_panel(ax, s, pm, extent, vmax, title):
    im = ax.imshow(np.where(pm, s, np.nan), origin="lower", extent=extent,
                   vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest")
    ny, nx = pm.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#ffffff", linewidths=1.0)
    ax.set_title(title, fontsize=7.2, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _phi_panel(ax, phi, pm, extent, title):
    im = ax.imshow(phi, origin="lower", extent=extent, vmin=0.0, vmax=1.0,
                   cmap="inferno", interpolation="bilinear")
    ny, nx = pm.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.4)
    ax.contour(xs, ys, phi, levels=[0.5], colors="#ffffff", linewidths=1.0, linestyles="--")
    ax.set_title(title, fontsize=7.2, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _run(case, s, eps_covary=False):
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE,
                     eps_covary=eps_covary)
    st = so.optimal_stop(tr, case)
    phi, _ = so.phi_field(tr.T_at_end(st.index), case)
    return phi, st


def print_figure(shape: str, verify_dir: Path, figdir: Path) -> Path:
    cfg_name, _g, _n = WINNERS[shape]
    case = build_case(load_cfg(CFGD / f"{cfg_name}.yaml"))
    pm = case.part_mask
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    res = json.loads((verify_dir / "printability.json").read_text())[shape]["arms"]
    maps = np.load(verify_dir / f"{shape}_quantized_maps.npz")

    cols = [("A1_cont", "box [0, 1] continuous\nsingle pass, NOT printable"),
            ("A1_4bpp", "box [0, 1] at 4 bits per pixel\n16 levels, single pass"),
            ("A1_2bpp", "box [0, 1] at 2 bits per pixel\n4 levels, single pass"),
            ("A15_cont", "box [0, 1.5] continuous\ndouble pass, NOT printable"),
            ("A15_4bpp", "box [0, 1.5] at 4 bits per pixel\ndouble pass"),
            ("A15_2bpp", "box [0, 1.5] at 2 bits per pixel\ndouble pass")]
    cols = [c for c in cols if c[0] in maps.files]

    fig, axes = plt.subplots(2, len(cols), figsize=(2.75 * len(cols), 7.2))
    im_map = im_phi = None
    for j, (key, label) in enumerate(cols):
        s = np.asarray(maps[key], dtype=float)
        m = res[key]
        im_map = _map_panel(axes[0, j], s, pm, extent, 1.5,
                            f"{label}\nmean s in part {float(np.mean(s[pm])):.3f}, "
                            f"{m['census_n_levels_used']} levels used")
        phi, st = _run(case, s)
        im_phi = _phi_panel(
            axes[1, j], phi, pm, extent,
            f"melted region at own optimal stop {st.time_s:.0f} s\n"
            f"J {m['J']:.1f}     IoU {m['IoU']:.4f}\n"
            f"growth {m['bed_melt_pct_of_part']:.2f} %     "
            f"under {m['part_under_melt_pct']:.2f} %")
        for i in range(2):
            axes[i, j].set_xlim(box[0], box[1]); axes[i, j].set_ylim(box[2], box[3])
    fig.colorbar(im_map, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01,
                 label="binder saturation s")
    fig.colorbar(im_phi, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01,
                 label="melt fraction")
    fig.suptitle(f"{shape}: does the solved map survive the printer's bit depth? "
                 "Cyan is the nominal part, dashed white is the melt front.", fontsize=10)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"fig_print_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def hist_figure(shape: str, verify_dir: Path, figdir: Path) -> Path:
    cfg_name, gain, win_npz = WINNERS[shape]
    cfg = load_cfg(CFGD / f"{cfg_name}.yaml")
    case = build_case(cfg)
    pm = case.part_mask
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    hist = json.loads((verify_dir / "hist_arms.json").read_text())[shape]["arms"]
    pr = json.loads((verify_dir / "printability.json").read_text())[shape]["arms"]
    qmaps = np.load(verify_dir / f"{shape}_quantized_maps.npz")

    s_hist = load_stored_map(case, Path(win_npz), cfg)
    panels = [
        (np.ones(pm.shape), hist and None, "uniform s = 1", False, None),
        (s_hist, hist["hist_win_asstored_eps"],
         f"ACTUAL stored 4 bits per pixel mask\nwindow-selected gain m = {gain}\n"
         "historical injection, permittivity co-varying", True, None),
        (np.asarray(qmaps["A1_4bpp"], dtype=float), pr["A1_4bpp"],
         "solved map, box [0, 1], 4 bits per pixel\nsingle printing pass", False, None),
        (np.asarray(qmaps["A15_4bpp"], dtype=float), pr["A15_4bpp"],
         "solved map, box [0, 1.5], 4 bits per pixel\ndouble printing pass", False, None),
    ]
    uni = json.loads((Path(__file__).resolve().parent.parent / "out_shape" /
                      f"{shape}.json").read_text())["uniform"]
    panels[0] = (np.ones(pm.shape), uni, "uniform s = 1\nno grading", False, None)

    fig, axes = plt.subplots(2, len(panels), figsize=(2.9 * len(panels), 7.2))
    im_map = im_phi = None
    for j, (s, m, label, covary, _x) in enumerate(panels):
        im_map = _map_panel(axes[0, j], s, pm, extent, 1.5,
                            f"{label}\nmean s in part {float(np.mean(s[pm])):.3f}")
        phi, st = _run(case, s, eps_covary=covary)
        im_phi = _phi_panel(
            axes[1, j], phi, pm, extent,
            f"melted region at own optimal stop {st.time_s:.0f} s\n"
            f"J {m['J']:.1f}     IoU {m['IoU']:.4f}\n"
            f"growth {m['bed_melt_pct_of_part']:.2f} %     "
            f"under {m['part_under_melt_pct']:.2f} %")
        for i in range(2):
            axes[i, j].set_xlim(box[0], box[1]); axes[i, j].set_ylim(box[2], box[3])
    fig.colorbar(im_map, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01,
                 label="binder saturation s")
    fig.colorbar(im_phi, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01,
                 label="melt fraction")
    fig.suptitle(f"{shape}: solved printable map against the ACTUAL stored historical mask, "
                 "same engine, same configuration.", fontsize=10)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"fig_hist_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    vd = Path(sys.argv[1]).resolve()
    fd = Path(sys.argv[2]).resolve()
    for sh in WINNERS:
        print(print_figure(sh, vd, fd), flush=True)
        print(hist_figure(sh, vd, fd), flush=True)
