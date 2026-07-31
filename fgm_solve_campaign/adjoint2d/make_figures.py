"""Figures for the shape-fidelity solve.

One composite per shape: the dopant map of every arm on the top row, and the
melt-fraction field at that arm's own optimal stop on the bottom row with the
nominal part outline overlaid. The reader should be able to see, at a glance,
where each arm's melted region departs from the nominal shape.

Plus one figure with the J(t) curves of every arm on every shape, which is what
makes the stop-time choice legible.
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
from .pins import build_case, load_cfg

DPI = 180
ARMS = ("uniform", "heuristic", "heuristic_eps", "A1", "A15")
ARM_LABEL = {"uniform": "uniform s = 1",
             "heuristic": "heuristic gain, conductivity only",
             "heuristic_eps": "heuristic gain, permittivity co-varying",
             "A1": "adjoint, box [0, 1]", "A15": "adjoint, box [0, 1.5]"}
EPS_ARM = {"heuristic_eps": True}


def crop_box(part_mask, x, y, pad_frac: float = 0.45):
    """Bounding box of the part with a margin, in millimetres.

    The chamber is three times the part across, so plotting the whole domain
    leaves the result unreadable. The margin is wide enough to show melt that
    escapes into the bed, which is the growth term the objective penalizes.
    """
    rows = np.flatnonzero(part_mask.any(axis=1))
    cols = np.flatnonzero(part_mask.any(axis=0))
    x0, x1 = x[cols[0]] * 1e3, x[cols[-1]] * 1e3
    y0, y1 = y[rows[0]] * 1e3, y[rows[-1]] * 1e3
    px, py = pad_frac * (x1 - x0), pad_frac * (y1 - y0)
    pad = max(px, py)
    return (x0 - pad, x1 + pad, y0 - pad, y1 + pad)


def _panel_phi(ax, phi, part_mask, extent, title):
    im = ax.imshow(phi, origin="lower", extent=extent, vmin=0.0, vmax=1.0,
                   cmap="inferno", interpolation="bilinear")
    ny, nx = part_mask.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    ax.contour(xs, ys, part_mask.astype(float), levels=[0.5], colors="#00e5ff",
               linewidths=1.4)
    ax.contour(xs, ys, phi, levels=[0.5], colors="#ffffff", linewidths=1.0,
               linestyles="--")
    ax.set_title(title, fontsize=7.5, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _panel_map(ax, s, part_mask, extent, title):
    v = np.where(part_mask, s, np.nan)
    im = ax.imshow(v, origin="lower", extent=extent, vmin=0.0, vmax=1.5,
                   cmap="viridis", interpolation="nearest")
    ny, nx = part_mask.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    ax.contour(xs, ys, part_mask.astype(float), levels=[0.5], colors="#ffffff",
               linewidths=1.0)
    ax.set_title(title, fontsize=7.5, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def build_shape_figure(shape: str, cfg_path: str, outdir: Path, figdir: Path) -> Path:
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    data = np.load(outdir / f"{shape}_maps.npz")
    res = json.loads((outdir / f"{shape}.json").read_text())
    pm = case.part_mask
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)

    metrics = {"uniform": res["uniform"], "heuristic": res["H_sig"],
               "heuristic_eps": res["H_eps"],
               "A1": res["budgets"]["40"]["A1"], "A15": res["budgets"]["40"]["A15"]}

    box = crop_box(pm, case.x, case.y)
    fig, axes = plt.subplots(2, 5, figsize=(16.2, 7.4))
    im_map = im_phi = None
    for j, arm in enumerate(ARMS):
        key = "heuristic" if arm == "heuristic_eps" else arm
        if key not in data:
            for i in range(2):
                axes[i, j].axis("off")
            continue
        s = data[key]
        m = metrics[arm]
        covary = EPS_ARM.get(arm, False)
        im_map = _panel_map(axes[0, j], s, pm, extent,
                            f"{ARM_LABEL[arm]}\ndopant map, mean in part "
                            f"{float(np.mean(s[pm])):.3f}")
        tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=250,
                         eps_covary=covary)
        st = so.optimal_stop(tr, case)
        phi, _ = so.phi_field(tr.T_at_end(st.index), case)
        im_phi = _panel_phi(
            axes[1, j], phi, pm, extent,
            f"melt fraction at stop {st.time_s:.0f} s\n"
            f"J {m['J']:.0f}     IoU {m['IoU']:.3f}\n"
            f"growth {m['bed_melt_pct_of_part']:.1f} %     "
            f"under {m['part_under_melt_pct']:.1f} %")
        for i in range(2):
            axes[i, j].set_xlim(box[0], box[1])
            axes[i, j].set_ylim(box[2], box[3])
    fig.colorbar(im_map, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01,
                 label="binder saturation s")
    fig.colorbar(im_phi, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01,
                 label="melt fraction")
    fig.suptitle(f"{shape}: dopant map and melted region at each arm's own optimal stop. "
                 "Cyan is the nominal part outline, dashed white is the melt front.",
                 fontsize=10)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"fig_shape_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def build_jcurve_figure(shapes, cfgs, outdir: Path, figdir: Path) -> Path:
    fig, axes = plt.subplots(1, len(shapes), figsize=(4.0 * len(shapes), 3.4))
    if len(shapes) == 1:
        axes = [axes]
    for ax, shape, cfgp in zip(axes, shapes, cfgs):
        case = build_case(load_cfg(Path(cfgp).resolve()))
        data = np.load(outdir / f"{shape}_maps.npz")
        n_part = int(case.part_mask.sum())
        for arm, colour in zip(ARMS, ("#444444", "#d62728", "#ff7f0e",
                                      "#1f77b4", "#2ca02c")):
            key = "heuristic" if arm == "heuristic_eps" else arm
            if key not in data:
                continue
            tr = fwd.forward(case, data[key], stop_after_phi=None,
                             shape_stop_patience=250,
                             eps_covary=EPS_ARM.get(arm, False))
            jc = so.J_curve(tr, case) / n_part
            ax.plot(tr.time_s, jc, color=colour, lw=1.4, label=ARM_LABEL[arm])
            i = int(np.argmin(jc))
            ax.plot(tr.time_s[i], jc[i], "o", color=colour, ms=4)
        ax.set_title(shape, fontsize=9)
        ax.set_xlabel("time, s", fontsize=8)
        ax.set_ylabel("J per part cell", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Shape-fidelity objective against exposure time. The marker is each arm's "
                 "own optimal stop.", fontsize=10)
    fig.tight_layout()
    p = figdir / "fig_shape_Jcurves.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    outdir = Path(sys.argv[1]).resolve()
    figdir = Path(sys.argv[2]).resolve()
    specs = [a.split("=") for a in sys.argv[3:]]
    shapes = [s for s, _ in specs]
    cfgs = [c for _, c in specs]
    for s, c in specs:
        print(build_shape_figure(s, c, outdir, figdir), flush=True)
    print(build_jcurve_figure(shapes, cfgs, outdir, figdir))
