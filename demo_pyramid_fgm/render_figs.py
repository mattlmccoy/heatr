#!/usr/bin/env python3
"""Deck-grade renders for the pyramid FGM demo (style: deck_figures_3d/style3d).

Reads demo_pyramid_fgm/out_graded/fields.npz (+ out_uniform if present) and
writes PNGs into demo_pyramid_fgm/. Rendering only; no physics computed here.
Every figure carries the honesty label from grading.HONESTY_TEXT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "deck_figures_3d"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import style3d as S  # noqa: E402
from demo_pyramid_fgm.grading import HONESTY_TEXT, LAW_TEXT  # noqa: E402

DPI = 200
# Colormap floors clipped so nothing reads black on the dark background.
CMAP_HOT = ListedColormap(plt.cm.inferno(np.linspace(0.25, 1.0, 256)))
CMAP_RHO = ListedColormap(plt.cm.viridis(np.linspace(0.22, 1.0, 256)))


def _load(arm: str) -> dict | None:
    f = HERE / f"out_{arm}" / "fields.npz"
    if not f.exists():
        return None
    d = dict(np.load(f))
    d["part"] = d["part"].astype(bool)
    return d


def _honesty(fig, extra: str = "") -> None:
    txt = HONESTY_TEXT + ((" | " + extra) if extra else "")
    fig.text(0.985, 0.015, txt, fontsize=8.0, color=S.DIM, ha="right", va="bottom")


def _pyramid_edges(part: np.ndarray, h_m: float, L_m: float):
    """Nominal pyramid wireframe in voxel-index coords (base corners + apex)."""
    hw_mm = 11.624473
    ht_mm = 23.248947
    h_mm = h_m * 1e3
    off = (L_m * 1e3) / 2.0
    def gi(x, y, z):
        return ((x + off) / h_mm, (y + off) / h_mm, (z + off) / h_mm)
    z0, z1 = -ht_mm / 2.0, ht_mm / 2.0
    c = [gi(sx * hw_mm, sy * hw_mm, z0) for sx, sy in
         ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    apex = gi(0.0, 0.0, z1)
    edges = [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]
    edges += [(ci, apex) for ci in c]
    return edges


def fig_graded_all_axes(d: dict) -> None:
    part, sat = d["part"], d["sat"]
    n = part.shape[0]
    ii, jj, kk = np.indices(part.shape)
    cx = cy = n // 2
    keep = part & ~((ii >= cx) & (jj <= cy))          # quarter cut toward viewer
    vmin, vmax = 0.20, 1.00
    colors = np.zeros(part.shape + (4,))
    colors[..., :] = CMAP_HOT((sat - vmin) / (vmax - vmin))
    colors[..., 3] = 1.0

    fig = plt.figure(figsize=(9.2, 8.2), dpi=DPI)
    ax = fig.add_axes([0.02, 0.02, 0.9, 0.86], projection="3d")
    S.dark_3d_axes(ax)
    ax.voxels(keep, facecolors=colors, edgecolors=None, shade=True,
              lightsource=matplotlib.colors.LightSource(azdeg=210, altdeg=45))
    for a, b in _pyramid_edges(part, float(d["h"]), float(d["L"])):
        ax.plot(*zip(a, b), color=S.ACCENT, lw=1.1, alpha=0.85)
    lo = np.argwhere(part)
    pad = 2
    ax.set_xlim(lo[:, 0].min() - pad, lo[:, 0].max() + pad)
    ax.set_ylim(lo[:, 1].min() - pad, lo[:, 1].max() + pad)
    ax.set_zlim(lo[:, 2].min() - pad, lo[:, 2].max() + pad)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-55)

    S.title_block(fig, "Graded through every axis",
                  "quarter-cut dopant volume, pyramid, cyan = nominal CAD wireframe")
    fig.text(0.035, 0.055, LAW_TEXT, fontsize=8.5, color=S.DIM)
    sm = plt.cm.ScalarMappable(cmap=CMAP_HOT,
                               norm=matplotlib.colors.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.55)
    cb.set_label("dopant fraction sat", color=S.FG, fontsize=9)
    cb.ax.tick_params(colors=S.DIM, labelsize=8)
    cb.outline.set_edgecolor(S.DIM)
    _honesty(fig)
    fig.savefig(HERE / "fig1_graded_all_axes.png")
    plt.close(fig)


def _slice_xz(a: np.ndarray, part: np.ndarray):
    ny = part.shape[1]
    m = part[:, ny // 2, :]
    return np.where(m, a[:, ny // 2, :].astype(float), np.nan).T, m.T


def _crop(ax, mask_t: np.ndarray, pad: int = 3) -> None:
    """Zoom an XZ-slice axes to the part bounding box (mask already transposed)."""
    idx = np.argwhere(mask_t)
    ax.set_xlim(idx[:, 1].min() - pad, idx[:, 1].max() + pad)
    ax.set_ylim(idx[:, 0].min() - pad, idx[:, 0].max() + pad)


def fig_densification(d: dict) -> None:
    part = d["part"]
    rho, mrho = _slice_xz(d["rho_final"], part)
    T, _ = _slice_xz(d["T_phi90"], part)
    phi, _ = _slice_xz(d["phi_final"], part)
    h_mm = float(d["h"]) * 1e3

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 5.6), dpi=DPI)
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.14)
    panels = [
        (axs[0], rho, CMAP_RHO, "relative density (final)", None, None),
        (axs[1], T, CMAP_HOT, "temperature at phi=0.90 (C)", None, None),
    ]
    for ax, img, cmap, label, vmin, vmax in panels:
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.contour(np.nan_to_num(phi), levels=[0.5], colors=S.ACCENT,
                   linewidths=1.4)
        ax.contour(mrho.astype(float), levels=[0.5], colors=S.FG,
                   linewidths=0.7, alpha=0.7)
        ax.set_title(label, fontsize=10, color=S.FG, pad=8)
        _crop(ax, mrho)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(colors=S.DIM, labelsize=8)
        cb.outline.set_edgecolor(S.DIM)
    S.title_block(fig, "Densification of the graded pyramid",
                  "XZ mid-slice, apex up; cyan = melt front (phi = 0.5), "
                  f"white = part outline, voxel {h_mm:.2f} mm")
    _honesty(fig, "densify=True, enthalpy phase update")
    fig.savefig(HERE / "fig2_densification.png")
    plt.close(fig)


def fig_graded_vs_uniform(dg: dict, du: dict) -> None:
    rg, mg = _slice_xz(dg["rho_final"], dg["part"])
    ru, mu = _slice_xz(du["rho_final"], du["part"])
    pg, _ = _slice_xz(dg["phi_final"], dg["part"])
    pu, _ = _slice_xz(du["phi_final"], du["part"])
    vmin = np.nanmin([np.nanmin(rg), np.nanmin(ru)])
    vmax = np.nanmax([np.nanmax(rg), np.nanmax(ru)])

    def _std(d):
        m = d["part"]
        return float(d["rho_final"][m].std())

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 5.6), dpi=DPI)
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.10)
    for ax, img, phi, mk, name, sd in (
            (axs[0], ru, pu, mu, "uniform dopant", _std(du)),
            (axs[1], rg, pg, mg, "graded dopant (heuristic)", _std(dg))):
        im = ax.imshow(img, origin="lower", cmap=CMAP_RHO, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.contour(np.nan_to_num(phi), levels=[0.5], colors=S.ACCENT,
                   linewidths=1.4)
        ax.contour(mk.astype(float), levels=[0.5], colors=S.FG,
                   linewidths=0.7, alpha=0.7)
        ax.set_title(f"{name}   std(rho) = {sd:.4f}", fontsize=10,
                     color=S.FG, pad=8)
        _crop(ax, mk)
        ax.axis("off")
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, shrink=0.8)
    cb.set_label("relative density (final)", color=S.FG, fontsize=9)
    cb.ax.tick_params(colors=S.DIM, labelsize=8)
    cb.outline.set_edgecolor(S.DIM)
    S.title_block(fig, "Graded vs uniform dopant",
                  "final relative density, XZ mid-slice, shared color scale; "
                  "cyan = melt front (phi = 0.5)")
    _honesty(fig, "same grid, exposure, and physics in both arms")
    fig.savefig(HERE / "fig3_graded_vs_uniform.png")
    plt.close(fig)


def fig_layer_stack(d: dict) -> None:
    part, sat = d["part"], d["sat"]
    ks = np.where(part.any(axis=(0, 1)))[0]
    picks = np.unique(np.linspace(ks[0] + 1, ks[-1] - 2, 6).astype(int))
    h_mm = float(d["h"]) * 1e3
    idx = np.argwhere(part)
    x0, x1 = idx[:, 0].min() - 2, idx[:, 0].max() + 3
    y0, y1 = idx[:, 1].min() - 2, idx[:, 1].max() + 3
    fig, axs = plt.subplots(1, len(picks), figsize=(12.4, 3.1), dpi=DPI)
    fig.subplots_adjust(top=0.66, bottom=0.03, left=0.02, right=0.98, wspace=0.08)
    for ax, k in zip(axs, picks):
        img = np.where(part[:, :, k], sat[:, :, k].astype(float), np.nan).T
        ax.imshow(img, origin="lower", cmap=CMAP_HOT, vmin=0.20, vmax=1.00,
                  interpolation="nearest")
        ax.contour(part[:, :, k].T.astype(float), levels=[0.5], colors=S.ACCENT,
                   linewidths=0.8, alpha=0.8)
        z_mm = (k - ks[0]) * h_mm
        ax.set_title(f"z = {z_mm:.1f} mm", fontsize=9, color=S.FG, pad=5)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.axis("off")
    fig.text(0.02, 0.965, "Per-layer dopant rasters, base to apex",
             fontsize=15, color=S.FG, ha="left", va="top", fontweight="bold")
    fig.text(0.02, 0.83, "each printed layer gets a DIFFERENT raster: footprint "
             "shrinks, skin-to-core contrast persists, level drops toward the apex",
             fontsize=9.5, color=S.DIM, ha="left", va="top")
    _honesty(fig)
    fig.savefig(HERE / "fig4_layer_stack.png")
    plt.close(fig)


def main() -> None:
    dg = _load("graded")
    if dg is None:
        raise SystemExit("out_graded/fields.npz missing; run run_demo.py graded first")
    fig_graded_all_axes(dg)
    fig_layer_stack(dg)
    fig_densification(dg)
    du = _load("uniform")
    if du is not None:
        fig_graded_vs_uniform(dg, du)
    for f in sorted(HERE.glob("fig*.png")):
        print(f)


if __name__ == "__main__":
    main()
