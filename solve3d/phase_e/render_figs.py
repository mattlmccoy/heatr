"""Phase E opener figures: solved vs uniform vs heuristic, on the SAME axes.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.render_figs --shape pyramid

Rendering only; no physics computed here. Every figure carries the S2 honesty
label, because on these cornered shapes the ABSOLUTE fidelity is ungated and
only the same-instrument comparison is meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import ListedColormap          # noqa: E402

from solve3d import gates as sg                       # noqa: E402
from solve3d.phase_e import geometry as geo           # noqa: E402
from solve3d.phase_e import rescore as rs             # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIGS = Path(__file__).resolve().parent / "figs"
DPI = 190
CMAP_T = ListedColormap(plt.cm.inferno(np.linspace(0.22, 1.0, 256)))
CMAP_S = ListedColormap(plt.cm.viridis(np.linspace(0.20, 1.0, 256)))
BG = "#12141a"
FG = "#e8e8ee"

HONESTY = ("S2: these are CORNERED shapes with an unresolved corner/apex "
           "singularity (escalated to S3). ABSOLUTE fidelity here is UNGATED -- "
           "what is meaningful is solved-vs-uniform at matched mesh and read.")


def _style(fig):
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for s in ax.spines.values():
            s.set_color("#3a3f4b")
        ax.title.set_color(FG)


def _label(fig, extra=""):
    fig.text(0.5, 0.012, HONESTY + ("  " + extra if extra else ""),
             ha="center", va="bottom", color="#9aa0ad", fontsize=6.4, wrap=True)


def _arms(shape: str):
    doc = json.loads((RESULTS / f"phase_e_{shape}.json").read_text())
    return doc["arms"], doc


def _field(shape: str, name: str):
    f = RESULTS / f"fieldz_{shape}_{name}.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    return np.asarray(z["T_eval"]), np.asarray(z["z_planes"])


def fig_comparison(shape: str) -> Path:
    """Solved vs uniform vs heuristic: melt fraction at three heights."""
    arms, _ = _arms(shape)
    names = [("uniform_baseline", "uniform"),
             ("heuristic_grading_law", "heuristic law (6c2aab9)"),
             ("solve_filter_only", "SOLVED (filter-only)")]
    names = [(k, t) for k, t in names if _field(shape, k)[0] is not None]
    _, zs = _field(shape, names[0][0])
    rows = [0, len(zs) // 2, len(zs) - 1]
    fig, axes = plt.subplots(len(rows), len(names),
                             figsize=(3.05 * len(names), 2.95 * len(rows)))
    axes = np.atleast_2d(axes)
    x, y, _, _ = sg.eval_grid_axes()
    ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    for ci, (key, title) in enumerate(names):
        Te, zz = _field(shape, key)
        for ri, zi in enumerate(rows):
            ax = axes[ri, ci]
            phi = sg.phase_fraction_phi(Te[zi])
            ax.imshow(phi.T, origin="lower", extent=ext, cmap=CMAP_T,
                      vmin=0, vmax=1, interpolation="bilinear")
            nom = geo.nominal_mask_2d(shape, z=float(zz[zi]))
            ax.contour(np.linspace(ext[0], ext[1], nom.shape[0]),
                       np.linspace(ext[2], ext[3], nom.shape[1]),
                       nom.T.astype(float), levels=[0.5],
                       colors="#7fe3ff", linewidths=0.9)
            if ri == 0:
                ax.set_title(title, fontsize=8.5, pad=6)
            if ci == 0:
                ax.set_ylabel(f"z = {zz[zi]*1e3:+.1f} mm", color=FG, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{shape}: melt fraction at the envelope stop "
                 f"(cyan = nominal outline)", color=FG, fontsize=11)
    _style(fig)
    _label(fig)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    out = FIGS / f"fig1_{shape}_comparison.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


def fig_difference(shape: str) -> Path:
    """What the solve changed, against each reference."""
    arms, _ = _arms(shape)
    Ts, zs = _field(shape, "solve_filter_only")
    Tu, _ = _field(shape, "uniform_baseline")
    Th, _ = _field(shape, "heuristic_grading_law")
    pairs = [(Tu, "solved - uniform")]
    if Th is not None:
        pairs.append((Th, "solved - heuristic"))
    rows = [0, len(zs) // 2, len(zs) - 1]
    fig, axes = plt.subplots(len(rows), len(pairs),
                             figsize=(3.2 * len(pairs), 2.95 * len(rows)))
    axes = np.atleast_2d(axes).reshape(len(rows), len(pairs))
    x, y, _, _ = sg.eval_grid_axes()
    ext = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    dmax = max(float(np.nanmax(np.abs(sg.phase_fraction_phi(Ts) -
                                      sg.phase_fraction_phi(o)))) for o, _ in pairs)
    dmax = max(dmax, 1e-6)
    for ci, (Toth, title) in enumerate(pairs):
        for ri, zi in enumerate(rows):
            ax = axes[ri, ci]
            d = sg.phase_fraction_phi(Ts[zi]) - sg.phase_fraction_phi(Toth[zi])
            im = ax.imshow(d.T, origin="lower", extent=ext, cmap="coolwarm",
                           vmin=-dmax, vmax=dmax, interpolation="bilinear")
            nom = geo.nominal_mask_2d(shape, z=float(zs[zi]))
            ax.contour(np.linspace(ext[0], ext[1], nom.shape[0]),
                       np.linspace(ext[2], ext[3], nom.shape[1]),
                       nom.T.astype(float), levels=[0.5],
                       colors="#22252c", linewidths=0.9)
            if ri == 0:
                ax.set_title(title, fontsize=8.5, pad=6)
            if ci == 0:
                ax.set_ylabel(f"z = {zs[zi]*1e3:+.1f} mm", color=FG, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02)
    cb.set_label("d(melt fraction)", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=7)
    fig.suptitle(f"{shape}: what the solve changed", color=FG, fontsize=11)
    _style(fig)
    _label(fig, "red = the solve melts MORE here; blue = less.")
    out = FIGS / f"fig2_{shape}_difference.png"
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_layer_stack(shape: str) -> Path:
    """The SOLVED dopant map, sliced the way a printer would lay it down."""
    z = np.load(RESULTS / f"map_{shape}_solve_filter_only.npz")
    s, c = np.asarray(z["s_map"], float), np.asarray(z["centroids"], float)
    half = geo.PYR_H_M / 2.0 if shape == "pyramid" else geo.CUBE_A_M / 2.0
    n = 6
    edges = np.linspace(-half, half, n + 1)
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.6))
    for i, ax in enumerate(axes):
        m = (c[:, 2] >= edges[i]) & (c[:, 2] < edges[i + 1])
        if m.sum() < 5:
            ax.axis("off"); continue
        sc = ax.scatter(c[m, 0] * 1e3, c[m, 1] * 1e3, c=s[m], s=5.5,
                        cmap=CMAP_S, vmin=float(s.min()), vmax=float(s.max()),
                        linewidths=0)
        ax.set_title(f"z {edges[i]*1e3:+.1f} .. {edges[i+1]*1e3:+.1f} mm",
                     fontsize=7.5, pad=4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("dopant saturation", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=7)
    fig.suptitle(f"{shape}: the SOLVED dopant map, layer by layer "
                 f"(mean {s.mean():.3f}, min {s.min():.3f}, max {s.max():.3f})",
                 color=FG, fontsize=10.5)
    _style(fig)
    _label(fig)
    out = FIGS / f"fig3_{shape}_layer_stack.png"
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    for f in (fig_comparison, fig_difference, fig_layer_stack):
        try:
            print("wrote", f(a.shape))
        except Exception as exc:
            print(f"SKIP {f.__name__}: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
