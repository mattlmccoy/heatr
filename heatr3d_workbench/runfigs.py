"""Per-run figure set for workbench runs, in the dark deck style.

ADAPTED FROM demo_pyramid_fgm/render_figs.py and render_fig5_diff.py
(commits 4284d3a / 6c2aab9) at Matt's request 2026-08-03: "I want figures
like this to be made (without the titles) when I run HEATR3D". Differences
from the demo renderers, per that request:
  - NO headline title blocks (S.title_block is never called) - these are
    working outputs, not deck slides;
  - a small honest annotation line on every figure: engine version, grid,
    phase update, run kind (replaces the demo's grading-law honesty text);
  - on-figure numbers kept (melt heights, std(rho), max deltas);
  - colormap floors stay clipped per the 2026-08-03 legibility fix
    (inferno from 0.25, viridis from 0.20/0.22-class floors);
  - shape-agnostic: no pyramid wireframe; the part mask outline stands in.

Figures land in <run>/plots/ so the Study gallery and the Results tab's
single-walk media scan pick them up with no new scan paths (a few PNGs/run).

Solver-venv only (numpy + matplotlib + deck_figures_3d/style3d); the style3d
import mutates matplotlib rcParams, so it is imported lazily inside
render_run_figs and the caller invokes this LAST in the render pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SATURATION_PHI = 0.95     # a read is "pre-saturation" while phi_bar <= this


# --------------------------------------------------------------------------- #
# Pure logic (TDD'd)
# --------------------------------------------------------------------------- #
def choose_read_state(snaps: Optional[Sequence[Dict[str, Any]]]
                      ) -> Tuple[str, Optional[int]]:
    """Pick the non-saturating read: the LAST snapshot with phi_bar <= 0.95,
    else the final state. All-saturated snapshots fall back to final - the
    honest read, never a fabricated early state."""
    if not snaps:
        return "final", None
    best = None
    for s in snaps:
        pb = s.get("phi_bar")
        if pb is not None and float(pb) <= SATURATION_PHI:
            best = int(s["i"])
    return ("snapshot", best) if best is not None else ("final", None)


def check_baseline_grid(dims_a, h_a: float, dims_b, h_b: float
                        ) -> Tuple[bool, str]:
    """Baseline comparisons are only meaningful on the SHARED grid."""
    if tuple(dims_a) != tuple(dims_b):
        return False, (f"grid mismatch: {tuple(dims_a)} vs {tuple(dims_b)} - "
                       f"difference figures need the same grid")
    if abs(float(h_a) - float(h_b)) > 1e-9:
        return False, (f"cell spacing mismatch: {h_a} vs {h_b} mm - "
                       f"difference figures need the same spacing")
    return True, ""


def melt_height_mm(phi: np.ndarray, part: np.ndarray, h_mm: float,
                   thresh: float = 0.5) -> float:
    """Height from the part's base layer to the topmost melted layer, mm."""
    m = part.astype(bool)
    melted = m & (np.asarray(phi) >= thresh)
    if not melted.any() or not m.any():
        return 0.0
    ks = np.where(m.any(axis=(0, 1)))[0]
    km = np.where(melted.any(axis=(0, 1)))[0]
    return float((km.max() - ks[0] + 1) * h_mm)


def annotation_line(results: Dict[str, Any]) -> str:
    """The one-line honest annotation carried by every figure."""
    bits = [str(results.get("heatr3d_engine_version", "pre-workbench")),
            f"n={results.get('grid_n', '?')}",
            str(results.get("phase_update", "apparent_cp")),
            f"fgm={results.get('fgm', 'none')}"]
    if results.get("densify"):
        bits.append("densify")
    bits.append("display floors clipped")
    return " | ".join(bits)


def figure_plan(cfg: Dict[str, Any], has_sat: bool, has_rho: bool) -> List[str]:
    """Which figures this run gets (spec: FGM -> cutaway + layer stack;
    densify -> density/temperature pair; baseline named -> comparison pair)."""
    plan: List[str] = []
    if has_sat:
        plan += ["dopant_cutaway", "layer_stack"]
    if has_rho:
        plan.append("density_temperature")
    if cfg.get("baseline_run_id"):
        plan += ["vs_baseline", "what_changed"]
    return plan


# --------------------------------------------------------------------------- #
# Rendering (adapted demo code; viewed, not unit-tested)
# --------------------------------------------------------------------------- #
def _load_fields(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    f = run_dir / "fields.npz"
    if not f.exists():
        return None
    d = dict(np.load(f))
    d["part"] = d["part"].astype(bool)
    return d


def _slice_xz(a: np.ndarray, part: np.ndarray):
    ny = part.shape[1]
    m = part[:, ny // 2, :]
    return np.where(m, a[:, ny // 2, :].astype(float), np.nan).T, m.T


def _crop(ax, mask_t: np.ndarray, pad: int = 3) -> None:
    idx = np.argwhere(mask_t)
    if len(idx) == 0:
        return
    ax.set_xlim(idx[:, 1].min() - pad, idx[:, 1].max() + pad)
    ax.set_ylim(idx[:, 0].min() - pad, idx[:, 0].max() + pad)


def render_run_figs(run_dir: Path, cfg: Dict[str, Any],
                    results: Dict[str, Any]) -> List[str]:
    """Render the per-run figure set into <run>/plots/. Returns files written.
    Baseline artifacts are read from cfg['baseline_run_id']'s run dir next to
    this one; a grid mismatch SKIPS the comparison loudly (log + marker file),
    never renders a wrong figure."""
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root / "deck_figures_3d") not in sys.path:
        sys.path.insert(0, str(root / "deck_figures_3d"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LightSource, Normalize
    import style3d as S      # dark deck style; mutates rcParams (call LAST)

    from heatr3d_workbench.workbench_job import field_cmap  # clipped floors

    CMAP_HOT = ListedColormap(plt.cm.inferno(np.linspace(0.25, 1.0, 256)))
    CMAP_RHO = ListedColormap(plt.cm.viridis(np.linspace(0.22, 1.0, 256)))
    for cm in (CMAP_HOT, CMAP_RHO):
        cm.set_bad("#22252c")

    run_dir = Path(run_dir)
    pdir = run_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    d = _load_fields(run_dir)
    if d is None:
        return []
    part = d["part"]
    h_mm = float(d["h"]) * 1e3
    has_sat = d.get("sat") is not None and d["sat"].ndim == 3
    has_rho = d.get("rho_final") is not None and d["rho_final"].ndim == 3
    note = annotation_line(results)
    written: List[str] = []

    def _annot(fig, extra: str = "") -> None:
        fig.text(0.985, 0.015, note + ((" | " + extra) if extra else ""),
                 fontsize=8.0, color=S.DIM, ha="right", va="bottom")

    def _save(fig, name: str) -> None:
        fig.savefig(pdir / name, facecolor=S.BG)
        plt.close(fig)
        written.append(name)

    # read-state for temperature panels: last pre-saturation snapshot if any
    snaps_idx = None
    idx_file = run_dir / "snapshots" / "index.json"
    if idx_file.exists():
        try:
            raw = json.loads(idx_file.read_text())["snaps"]
            snaps_idx = []
            for s in raw:
                zz = np.load(run_dir / "snapshots" / f"snap_{s['i']:03d}.npz")
                phi_bar = float(zz["phi"][part].mean()) if part.any() else 1.0
                snaps_idx.append({"i": s["i"], "t_s": s["t_s"],
                                  "phi_bar": phi_bar})
        except (OSError, KeyError, json.JSONDecodeError) as e:
            logger.warning("snapshot index unreadable for figs: %s", e)
            snaps_idx = None
    read_kind, read_i = choose_read_state(snaps_idx)
    if read_kind == "snapshot":
        zz = np.load(run_dir / "snapshots" / f"snap_{read_i:03d}.npz")
        T_read = zz["T"]
        phi_read = zz["phi"]
        read_label = f"pre-saturation read t = {float(zz['t_s']):.0f} s"
    else:
        T_read, phi_read = d["T_phi90"], d["phi_final"]
        read_label = "final-state read"

    plan = figure_plan(cfg, has_sat, has_rho)

    # ---- fig: quarter-cut dopant volume (demo fig1, no wireframe/titles) ----
    if "dopant_cutaway" in plan:
        sat = d["sat"]
        n = part.shape[0]
        ii, jj, _ = np.indices(part.shape)
        keep = part & ~((ii >= n // 2) & (jj <= n // 2))
        vmin, vmax = float(np.nanmin(sat[part])), float(np.nanmax(sat[part]))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        colors = CMAP_HOT((sat - vmin) / (vmax - vmin))
        colors[..., 3] = 1.0
        fig = plt.figure(figsize=(7.4, 6.6), dpi=180)
        ax = fig.add_axes([0.02, 0.02, 0.9, 0.94], projection="3d")
        S.dark_3d_axes(ax)
        ax.voxels(keep, facecolors=colors, edgecolors=None, shade=True,
                  lightsource=LightSource(azdeg=210, altdeg=45))
        lo = np.argwhere(part)
        pad = 2
        ax.set_xlim(lo[:, 0].min() - pad, lo[:, 0].max() + pad)
        ax.set_ylim(lo[:, 1].min() - pad, lo[:, 1].max() + pad)
        ax.set_zlim(lo[:, 2].min() - pad, lo[:, 2].max() + pad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-55)
        sm = plt.cm.ScalarMappable(cmap=CMAP_HOT, norm=Normalize(vmin, vmax))
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.55)
        cb.set_label("dopant fraction sat", color=S.FG, fontsize=9)
        cb.ax.tick_params(colors=S.DIM, labelsize=8)
        cb.outline.set_edgecolor(S.DIM)
        _annot(fig, "quarter-cut dopant volume")
        _save(fig, "fig_dopant_cutaway.png")

    # ---- fig: per-layer dopant raster stack (demo fig4, no titles) ---------
    if "layer_stack" in plan:
        sat = d["sat"]
        ks = np.where(part.any(axis=(0, 1)))[0]
        picks = np.unique(np.linspace(ks[0] + 1, max(ks[0] + 1, ks[-1] - 2),
                                      6).astype(int))
        idx = np.argwhere(part)
        x0, x1 = idx[:, 0].min() - 2, idx[:, 0].max() + 3
        y0, y1 = idx[:, 1].min() - 2, idx[:, 1].max() + 3
        vmin, vmax = float(np.nanmin(sat[part])), float(np.nanmax(sat[part]))
        fig, axs = plt.subplots(1, len(picks), figsize=(12.4, 2.9), dpi=180)
        axs = np.atleast_1d(axs)
        fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98,
                            wspace=0.08)
        for ax, k in zip(axs, picks):
            img = np.where(part[:, :, k], sat[:, :, k].astype(float), np.nan).T
            ax.imshow(img, origin="lower", cmap=CMAP_HOT, vmin=vmin,
                      vmax=(vmax if vmax > vmin else vmin + 1e-9),
                      interpolation="nearest")
            ax.contour(part[:, :, k].T.astype(float), levels=[0.5],
                       colors=S.ACCENT, linewidths=0.8, alpha=0.8)
            ax.set_title(f"z = {(k - ks[0]) * h_mm:.1f} mm", fontsize=9,
                         color=S.FG, pad=4)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.axis("off")
        _annot(fig, "per-layer dopant rasters, base to top")
        _save(fig, "fig_layer_stack.png")

    # ---- fig: density + temperature slice pair (demo fig2, no titles) ------
    if "density_temperature" in plan:
        rho, mrho = _slice_xz(d["rho_final"], part)
        T, _ = _slice_xz(T_read, part)
        phi, _ = _slice_xz(phi_read, part)
        fig, axs = plt.subplots(1, 2, figsize=(10.6, 5.0), dpi=180)
        fig.subplots_adjust(top=0.90, bottom=0.10, wspace=0.14)
        for ax, img, cmap, label in (
                (axs[0], rho, CMAP_RHO, "relative density (final)"),
                (axs[1], T, CMAP_HOT, f"temperature (C), {read_label}")):
            im = ax.imshow(img, origin="lower", cmap=cmap,
                           interpolation="nearest")
            ax.contour(np.nan_to_num(phi), levels=[0.5], colors=S.ACCENT,
                       linewidths=1.4)
            ax.contour(mrho.astype(float), levels=[0.5], colors=S.FG,
                       linewidths=0.7, alpha=0.7)
            ax.set_title(label, fontsize=10, color=S.FG, pad=6)
            _crop(ax, mrho)
            ax.axis("off")
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cb.ax.tick_params(colors=S.DIM, labelsize=8)
            cb.outline.set_edgecolor(S.DIM)
        _annot(fig, f"XZ mid-slice, cyan = melt front, voxel {h_mm:.2f} mm")
        _save(fig, "fig_density_temperature.png")

    # ---- baseline comparison figures ---------------------------------------
    base_id = cfg.get("baseline_run_id")
    if base_id and ("vs_baseline" in plan):
        base_dir = run_dir.parent / str(base_id)
        b = _load_fields(base_dir)
        if b is None:
            logger.warning("baseline %s has no fields.npz; comparison skipped",
                           base_id)
            (pdir / "baseline_comparison_SKIPPED.txt").write_text(
                f"baseline {base_id}: fields.npz missing")
        else:
            ok, msg = check_baseline_grid(part.shape, h_mm,
                                          b["part"].shape,
                                          float(b["h"]) * 1e3)
            if not ok:
                logger.warning("baseline %s: %s", base_id, msg)
                (pdir / "baseline_comparison_SKIPPED.txt").write_text(
                    f"baseline {base_id}: {msg}")
            else:
                _render_baseline_figs(d, b, base_id, part, h_mm, has_rho,
                                      plt, S, CMAP_RHO, CMAP_HOT,
                                      _annot, _save, _slice_xz)

    logger.info("run figs written: %s", written)
    return written


def _render_baseline_figs(d, b, base_id, part, h_mm, has_rho, plt, S,
                          CMAP_RHO, CMAP_HOT, _annot, _save, _slice) -> None:
    """Side-by-side vs baseline (demo fig3) + difference panels (demo fig5)."""
    bpart = b["part"]
    field = "rho_final" if (has_rho and b["rho_final"].ndim == 3) else "T_phi90"
    cmap = CMAP_RHO if field == "rho_final" else CMAP_HOT
    fu, mu = _slice(b[field], bpart)
    fg_, mg = _slice(d[field], part)
    pu, _ = _slice(b["phi_final"], bpart)
    pg, _ = _slice(d["phi_final"], part)
    vmin = np.nanmin([np.nanmin(fu), np.nanmin(fg_)])
    vmax = np.nanmax([np.nanmax(fu), np.nanmax(fg_)])

    def _std(dd, mask):
        return float(np.asarray(dd[field], float)[mask].std())

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 5.0), dpi=180)
    fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.10)
    for ax, img, phi, mk, name, sd, mh in (
            (axs[0], fu, pu, mu, f"baseline ({base_id})", _std(b, bpart),
             melt_height_mm(b["phi_final"], bpart, h_mm)),
            (axs[1], fg_, pg, mg, "this run", _std(d, part),
             melt_height_mm(d["phi_final"], part, h_mm))):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.contour(np.nan_to_num(phi), levels=[0.5], colors=S.ACCENT,
                   linewidths=1.4)
        ax.contour(mk.astype(float), levels=[0.5], colors=S.FG,
                   linewidths=0.7, alpha=0.7)
        ax.set_title(f"{name}\nstd = {sd:.4f}   melt height = {mh:.1f} mm",
                     fontsize=10, color=S.FG, pad=6)
        _crop(ax, mk)
        ax.axis("off")
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02, shrink=0.8)
    cb.set_label(("relative density (final)" if field == "rho_final"
                  else "temperature at read (C)"), color=S.FG, fontsize=9)
    cb.ax.tick_params(colors=S.DIM, labelsize=8)
    cb.outline.set_edgecolor(S.DIM)
    _annot(fig, "same grid and physics in both arms; XZ mid-slice")
    _save(fig, "fig_vs_baseline.png")

    # difference panels (demo fig5): dQ, dT, drho on the shared grid
    union = part | bpart
    mid = part.shape[1] // 2
    mask = union[:, mid, :].T
    dT = (d["T_phi90"] - b["T_phi90"])[:, mid, :].T
    dQ = ((d["Qrf"] - b["Qrf"]) / max(float(b["Qrf"].max()), 1.0))[:, mid, :].T
    panels = [("delta Q_rf [% of baseline peak]", np.where(mask, dQ * 100, np.nan)),
              ("delta T [C]", np.where(mask, dT, np.nan))]
    if has_rho and b.get("rho_final") is not None and b["rho_final"].ndim == 3:
        dR = (d["rho_final"] - b["rho_final"])[:, mid, :].T
        panels.append(("delta relative density (final)",
                       np.where(mask, dR, np.nan)))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 5.6),
                             dpi=180)
    axes = np.atleast_1d(axes)
    for ax, (cbl, fld) in zip(axes, panels):
        v = float(np.nanmax(np.abs(fld))) or 1e-9
        cm = plt.get_cmap("RdBu_r").copy()
        cm.set_bad("#22252c")
        im = ax.imshow(fld, origin="lower", cmap=cm, vmin=-v, vmax=v,
                       interpolation="bilinear")
        _crop(ax, mask)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=8, colors=S.DIM, length=2)
        cb.outline.set_edgecolor(S.DIM)
        cb.set_label(cbl, fontsize=9, color=S.DIM)
    if len(panels) < 3:
        # top-left, clear of the bottom-right annotation line
        fig.text(0.015, 0.975, "delta rho not computed (a non-densify arm)",
                 fontsize=8.5, color=S.DIM, ha="left", va="top")
    _annot(fig, "this run minus baseline; XZ mid-slice; shared grid")
    _save(fig, "fig_what_changed.png")
