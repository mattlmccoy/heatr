#!/usr/bin/env python3
"""
make_rf_summary_v5.py
=====================
Generate canonical RFAM summary figures matching the rf_summary_v4.png style:
  • Large LEFT  panel: jet colormap (dark-blue→green→yellow→red) thermal + white
                       E-field streamlines + temperature annotations
  • Top CENTRE  panel: Blues colormap RF electric potential with equipotential
                       lines and electrode labels
  • Top RIGHT   panel: Full-chamber relative density with grey powder background,
                       blue→red colormap for part density, and φ = 0.10 / 0.50 /
                       0.90 melt contours (cyan dashed / yellow solid / red dashed)
  • Bottom CENTRE: Part temperature vs exposure time series
  • Bottom RIGHT : Melt fraction & densification vs exposure

Usage:
    # single directory
    python3 make_rf_summary_v5.py outputs_eqs/shape_circle

    # all shape_* directories under outputs_eqs/
    python3 make_rf_summary_v5.py --all

    # custom output filename
    python3 make_rf_summary_v5.py outputs_eqs/shape_circle --out rf_summary_v5.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import yaml
from scipy.ndimage import binary_erosion, gaussian_filter

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

T_MELT_C = 185.0   # Nylon 12 melt temperature

# ── custom density colormap ────────────────────────────────────────────────────
# Matches the canonical figure: grey (powder/outside) → blue → cyan → green →
# yellow → orange → red  (dense).  The grey band covers the sub-initial-density
# region so undoped powder reads clearly as "not part".
_density_cmap = mcolors.LinearSegmentedColormap.from_list(
    "rfam_density",
    [
        (0.00, "#888888"),   # 0.55  → grey (powder)
        (0.10, "#2255CC"),   # 0.605 → dark blue
        (0.30, "#22AADD"),   # 0.69  → cyan-blue
        (0.50, "#22BB55"),   # 0.775 → green
        (0.70, "#CCCC00"),   # 0.86  → yellow
        (0.85, "#FF8800"),   # 0.925 → orange
        (1.00, "#CC0000"),   # 1.00  → red (dense)
    ],
    N=256,
)


def _heatmap_limits(arr: np.ndarray) -> tuple[float, float]:
    """Return robust (vmin, vmax) matching the plotted heatmap data."""
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        pad = max(1e-9, abs(vmin) * 1e-6)
        return vmin - pad, vmax + pad
    return vmin, vmax


# ══════════════════════════════════════════════════════════════════════════════
def make_rf_summary_v5(out_dir: str,
                       save_name: str = "rf_summary_v5.png",
                       shape_name: str | None = None) -> str:
    """
    Load simulation data from *out_dir* and write a canonical summary figure.
    Returns the path to the saved PNG.
    """
    # ── load data ──────────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, "fields.npz")
    ts_path  = os.path.join(out_dir, "time_series.json")
    if not os.path.isfile(npz_path) or not os.path.isfile(ts_path):
        print(f"  ✗  Missing fields.npz or time_series.json in {out_dir}")
        return ""

    d  = np.load(npz_path, allow_pickle=True)
    ts = json.load(open(ts_path))
    used_cfg_path = os.path.join(out_dir, "used_config.yaml")
    used_cfg = {}
    if os.path.isfile(used_cfg_path):
        try:
            raw_cfg = yaml.safe_load(open(used_cfg_path))
            if isinstance(raw_cfg, dict):
                used_cfg = raw_cfg
        except Exception:
            used_cfg = {}

    x_mm = d["x"] * 1e3          # metres → mm,  shape (nx,)
    y_mm = d["y"] * 1e3          # metres → mm,  shape (ny,)
    V    = np.real(d["V"])        # (ny, nx)
    Ex_r = np.real(d["Ex"])       # (ny, nx)
    Ey_r = np.real(d["Ey"])       # (ny, nx)
    T    = d["T"]                 # (ny, nx)  °C
    phi  = d["phi"]               # (ny, nx)
    rho  = d["rho_rel"]           # (ny, nx)
    pm   = d["part_mask"]         # (ny, nx) bool

    # meshgrid for pcolormesh / contour – both shape (ny, nx)
    X, Y = np.meshgrid(x_mm, y_mm, indexing="xy")

    # ── Gaussian-smoothed display copies (removes staircase grid-cell artefacts) ──
    _sig_f = 1.2   # field smoothing radius in grid cells
    _sig_m = 1.0   # mask smoothing radius for smooth boundary contours
    T_plot    = gaussian_filter(T.astype(float),   sigma=_sig_f)
    V_plot    = gaussian_filter(V.astype(float),   sigma=0.7)
    pm_smooth = gaussian_filter(pm.astype(float),  sigma=_sig_m)

    # ── time series arrays ─────────────────────────────────────────────────
    t_arr   = np.array(ts["time_s"]) / 60.0
    T_mean  = np.array(ts["mean_T_part_c"])
    T_max   = np.array(ts["max_T_part_c"])
    phi_ts  = np.array(ts["mean_phi_part"])
    rho_ts  = np.array(ts["mean_rho_rel_part"])

    t_final_min = float(t_arr[-1])
    rho_final   = float(rho_ts[-1])

    # ── part geometry stats ────────────────────────────────────────────────
    iy_pts, ix_pts = np.where(pm)
    iy_ctr = int(round(iy_pts.mean()))
    ix_ctr = int(round(ix_pts.mean()))
    T_center = float(T[iy_ctr, ix_ctr])

    # hotspot: maximum T inside part
    T_flat = T.copy(); T_flat[~pm] = -1e9
    hot_iy, hot_ix = np.unravel_index(T_flat.argmax(), T.shape)
    T_hotspot = float(T[hot_iy, hot_ix])

    # cool boundary: minimum T on part boundary surface
    eroded = binary_erosion(pm, iterations=2)
    boundary_mask = pm & ~eroded
    T_bnd = T.copy(); T_bnd[~boundary_mask] = 1e9
    cool_iy, cool_ix = np.unravel_index(T_bnd.argmin(), T.shape)
    T_cool = float(T[cool_iy, cool_ix])

    # ── electrode voltages from data ───────────────────────────────────────
    elec_hi  = d["elec_hi"]
    elec_lo  = d["elec_lo"]
    V_hi_val = float(V[elec_hi].mean()) if elec_hi.any() else 860.0
    V_lo_val = float(V[elec_lo].mean()) if elec_lo.any() else   0.0
    iy_hi_ctr = float(np.where(elec_hi.any(axis=1))[0].mean()) if elec_hi.any() else (y_mm.size - 1)
    hi_is_top = (iy_hi_ctr > y_mm.size / 2)
    label_top    = (f"V = {V_hi_val:.0f} V  (top electrode)"
                    if hi_is_top else
                    f"V = {V_lo_val:.0f} V  (top electrode)")
    label_bottom = (f"V = {V_lo_val:.0f} V  (bottom electrode)"
                    if hi_is_top else
                    f"V = {V_hi_val:.0f} V  (bottom electrode)")

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17.0, 8.2), facecolor="white", dpi=150)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.05, right=0.97, top=0.89, bottom=0.09,
        hspace=0.40, wspace=0.32,
        width_ratios=[1.35, 1.0, 1.0],
    )
    ax_th  = fig.add_subplot(gs[:, 0])   # large left – thermal
    ax_pot = fig.add_subplot(gs[0, 1])   # top centre – potential
    ax_rho = fig.add_subplot(gs[0, 2])   # top right  – density
    ax_tts = fig.add_subplot(gs[1, 1])   # bot centre – T(t)
    ax_dts = fig.add_subplot(gs[1, 2])   # bot right  – densification

    # ══════════════════════════════════════════════════════════════════════
    # Panel A  :  Thermal + E-field streamlines  (jet: dark-blue→green→red)
    # ══════════════════════════════════════════════════════════════════════
    T_vmin, T_vmax = _heatmap_limits(T_plot)
    im_T = ax_th.pcolormesh(
        X, Y, T_plot, cmap="jet",
        vmin=T_vmin, vmax=T_vmax,
        shading="gouraud", rasterized=True,
    )
    # E-field streamlines in white
    sp_Ex = Ex_r.copy()
    sp_Ey = Ey_r.copy()
    sp_Ex[np.abs(sp_Ex) < 1.0] = 1.0   # prevent degenerate seeds
    try:
        ax_th.streamplot(
            x_mm, y_mm, sp_Ex, sp_Ey,
            color="white", density=0.6, linewidth=0.75,
            arrowsize=0.55, broken_streamlines=False,
            zorder=3,
        )
    except Exception:
        pass   # streamplot can fail for some degenerate grids

    # Part outline – gold (smoothed mask → clean curve on all shapes)
    ax_th.contour(X, Y, pm_smooth, levels=[0.5],
                  colors=["#FFD700"], linewidths=[1.8], zorder=4)

    # ── temperature annotations ─────────────────────────────────────────
    ann_kw = dict(
        xycoords="data", textcoords="offset points",
        fontsize=7.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
        zorder=6,
    )
    # Hotspot
    ax_th.annotate(
        f"{T_hotspot:.0f}°C\n(hotspot)",
        xy=(float(x_mm[hot_ix]), float(y_mm[hot_iy])),
        xytext=(22, 12), **ann_kw, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#B22222",
                  alpha=0.88, edgecolor="none"),
    )
    # Center
    ax_th.annotate(
        f"{T_center:.0f}°C\n(center)",
        xy=(float(x_mm[ix_ctr]), float(y_mm[iy_ctr])),
        xytext=(-36, -24), **ann_kw, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#7B3F00",
                  alpha=0.88, edgecolor="none"),
    )
    # Cool face
    ax_th.annotate(
        f"{T_cool:.0f}°C\n(cool face)",
        xy=(float(x_mm[cool_ix]), float(y_mm[cool_iy])),
        xytext=(-50, 12), **ann_kw, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#003399",
                  alpha=0.88, edgecolor="none"),
    )

    cb_T = plt.colorbar(im_T, ax=ax_th, shrink=0.82, pad=0.02)
    cb_T.set_label("Temperature [°C]", fontsize=8)
    cb_T.ax.tick_params(labelsize=7)

    ax_th.set_xlim(x_mm[0], x_mm[-1])
    ax_th.set_ylim(y_mm[0], y_mm[-1])
    ax_th.set_aspect("equal")
    ax_th.set_xlabel("x [mm]", fontsize=8)
    ax_th.set_ylabel("y [mm]", fontsize=8)
    _sname = shape_name or os.path.basename(out_dir).replace("shape_", "").replace("_", " ")
    ax_th.set_title(
        f"High-contrast thermal + E-field streamlines\n"
        f"t = {t_final_min:.0f} min  "
        f"(hotspot {T_hotspot:.0f}°C · cool-face {T_cool:.0f}°C)",
        fontsize=8.5, pad=4,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Panel B  :  RF Electric Potential  (Blues colormap)
    # ══════════════════════════════════════════════════════════════════════
    V_vmin, V_vmax = _heatmap_limits(V_plot)
    im_V = ax_pot.pcolormesh(
        X, Y, V_plot, cmap="Blues",
        vmin=V_vmin, vmax=V_vmax,
        shading="gouraud", rasterized=True,
    )
    ax_pot.contour(X, Y, V, levels=18,
                   colors="white", linewidths=0.45, alpha=0.65, zorder=3)
    ax_pot.contour(X, Y, pm_smooth, levels=[0.5],
                   colors=["white"], linewidths=[2.0], zorder=4)

    # Electrode labels inside plot
    ax_pot.text(
        float(np.mean(x_mm)), y_mm[-1] * 0.87,
        label_top,
        ha="center", va="top", fontsize=7, color="white", fontweight="bold",
        bbox=dict(facecolor="#00224e", alpha=0.80, pad=2, edgecolor="none"),
        zorder=5,
    )
    ax_pot.text(
        float(np.mean(x_mm)), y_mm[0] * 0.87,
        label_bottom,
        ha="center", va="bottom", fontsize=7, color="white", fontweight="bold",
        bbox=dict(facecolor="#00224e", alpha=0.80, pad=2, edgecolor="none"),
        zorder=5,
    )

    cb_V = plt.colorbar(im_V, ax=ax_pot, shrink=0.88, pad=0.02)
    cb_V.set_label("Electric potential V [V]", fontsize=7.5)
    cb_V.ax.tick_params(labelsize=6.5)
    ax_pot.set_aspect("equal")
    ax_pot.set_xlabel("x [mm]", fontsize=8)
    ax_pot.set_ylabel("y [mm]", fontsize=8)
    ax_pot.set_title(
        "RF Electric Potential field V\n"
        "(equipotential lines · doped region in white)",
        fontsize=8.5, pad=4,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Panel C  :  Relative density  (full chamber; grey powder; rfam_density cmap)
    #             Melt contours: φ=0.10 cyan dashed / φ=0.50 yellow / φ=0.90 red
    # ══════════════════════════════════════════════════════════════════════
    # Show full chamber; powder region in grey (ρ_initial = 0.55)
    rho_disp = rho.copy()
    rho_disp[~pm] = 0.55   # undoped powder → maps to grey band in colormap
    rho_plot = gaussian_filter(rho_disp.astype(float), sigma=_sig_f)
    rho_vmin, rho_vmax = _heatmap_limits(rho_plot)

    im_rho = ax_rho.pcolormesh(
        X, Y, rho_plot,
        cmap=_density_cmap,
        vmin=rho_vmin, vmax=rho_vmax,
        shading="gouraud", rasterized=True,
    )
    # Part outline – black bold (smoothed mask → clean curve)
    ax_rho.contour(X, Y, pm_smooth, levels=[0.5],
                   colors=["black"], linewidths=[1.8], zorder=4)

    # Melt fraction contours:
    # default is full-domain (physically allows melt fronts outside target due to conduction);
    # optional reporting override can clip to part for legacy style.
    rep_cfg = used_cfg.get("reporting", {}) if isinstance(used_cfg.get("reporting", {}), dict) else {}
    clip_melt_to_part = bool(rep_cfg.get("clip_melt_contours_to_part", False))
    phi_contours = np.where(pm, phi, 0.0) if clip_melt_to_part else phi
    try:
        ax_rho.contour(X, Y, phi_contours, levels=[0.10],
                       colors=["cyan"],    linewidths=[1.1],
                       linestyles=["--"], zorder=5)
    except Exception:
        pass
    try:
        ax_rho.contour(X, Y, phi_contours, levels=[0.50],
                       colors=["yellow"],  linewidths=[1.4],
                       linestyles=["-"],  zorder=5)
    except Exception:
        pass
    try:
        ax_rho.contour(X, Y, phi_contours, levels=[0.90],
                       colors=["red"],     linewidths=[1.1],
                       linestyles=["--"], zorder=5)
    except Exception:
        pass

    # Legend for melt contours
    from matplotlib.lines import Line2D as _L2D
    _legend_handles = [
        _L2D([0], [0], color="cyan",   lw=1.1, ls="--", label="φ=0.10 (melt onset)"),
        _L2D([0], [0], color="yellow", lw=1.4, ls="-",  label="φ=0.50 (half melt)"),
        _L2D([0], [0], color="red",    lw=1.1, ls="--", label="φ=0.90 (near full)"),
    ]
    ax_rho.legend(handles=_legend_handles, fontsize=6.5, loc="lower right",
                  framealpha=0.80, facecolor="#222222", labelcolor="white",
                  edgecolor="none")

    # Colorbar labels at actual plotted limits
    cb_rho = plt.colorbar(im_rho, ax=ax_rho, shrink=0.88, pad=0.02)
    cb_rho.set_label("Relative density ρ_rel", fontsize=7.5)
    cb_rho.ax.tick_params(labelsize=6.5)
    cb_rho.ax.text(1.4, 0.0,  f"powder\n{rho_vmin:.2f}",  transform=cb_rho.ax.transAxes,
                   fontsize=6, va="bottom", color="#888888")
    cb_rho.ax.text(1.4, 1.0,  f"dense\n{rho_vmax:.2f}",   transform=cb_rho.ax.transAxes,
                   fontsize=6, va="top",    color="#CC0000")

    ax_rho.set_xlim(x_mm[0], x_mm[-1])   # full chamber
    ax_rho.set_ylim(y_mm[0], y_mm[-1])
    ax_rho.set_aspect("equal")
    ax_rho.set_xlabel("x [mm]", fontsize=8)
    ax_rho.set_ylabel("y [mm]", fontsize=8)
    ax_rho.set_title(
        f"Relative density map at t = {t_final_min:.0f} min\n"
        f"(contours: φ=0.10/0.50/0.90 melt fraction · ρ_final={rho_final:.3f})",
        fontsize=8.5, pad=4,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Panel D  :  Temperature time series
    # ══════════════════════════════════════════════════════════════════════
    ax_tts.fill_between(t_arr, T_mean, T_max, alpha=0.13, color="orangered")
    ax_tts.plot(t_arr, T_mean, color="orangered", lw=1.8, label="Mean T (part)")
    ax_tts.plot(t_arr, T_max,  color="firebrick", lw=1.8, label="Max T (corners)")
    ax_tts.axhline(T_MELT_C, color="#555555", ls="--", lw=1.0,
                   label=f"Melt {T_MELT_C:.0f}°C")
    ax_tts.set_xlabel("Exposure time [min]", fontsize=8)
    ax_tts.set_ylabel("T [°C]", fontsize=8)
    ax_tts.set_title("Part temperature vs exposure", fontsize=8.5, pad=4)
    ax_tts.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax_tts.grid(True, alpha=0.25, lw=0.6)
    ax_tts.set_xlim(0, t_arr[-1])
    ax_tts.set_facecolor("white")

    # ══════════════════════════════════════════════════════════════════════
    # Panel E  :  Melt fraction & densification
    # ══════════════════════════════════════════════════════════════════════
    ax_dts2 = ax_dts.twinx()
    l1, = ax_dts.plot( t_arr, phi_ts, color="orangered", lw=1.8,
                       label="Melt fraction φ")
    l2, = ax_dts2.plot(t_arr, rho_ts, color="steelblue", lw=1.8, ls="--",
                       label="Mean ρ_rel")
    ax_dts.axhline(0.5, color="#999999", ls=":", lw=0.8)
    ax_dts.set_xlabel("Exposure time [min]", fontsize=8)
    ax_dts.set_ylabel("Melt fraction φ",  fontsize=8, color="orangered")
    ax_dts2.set_ylabel("Mean ρ_rel",       fontsize=8, color="steelblue")
    ax_dts.tick_params( axis="y", labelcolor="orangered", labelsize=7)
    ax_dts2.tick_params(axis="y", labelcolor="steelblue", labelsize=7)
    ax_dts.tick_params( axis="x", labelsize=7)
    ax_dts.set_title("Melt fraction & densification vs exposure", fontsize=8.5, pad=4)
    ax_dts.legend(handles=[l1, l2], fontsize=7, loc="center right", framealpha=0.9)
    ax_dts.grid(True, alpha=0.25, lw=0.6)
    ax_dts.set_xlim(0, t_arr[-1])
    ax_dts.set_ylim(-0.05, 1.10)
    ax_dts2.set_ylim(
        max(0.40, rho_ts.min() - 0.05),
        min(1.02, rho_ts.max() + 0.05),
    )
    ax_dts.set_facecolor("white")

    # ── Suptitle ───────────────────────────────────────────────────────────
    fig.suptitle(
        "RFAM Simulation — Uniform σ EQS · Nylon 12 · 500W generator · 2% coupling efficiency\n"
        f"Shape: {_sname}  ·  Left: thermal at {t_final_min:.0f} min  ·  "
        f"RF potential field  ·  Density at {t_final_min:.0f} min  ·  "
        f"{t_final_min:.0f}-min exposure sweep",
        fontsize=9.5, fontweight="bold", y=0.975,
    )

    # ── save ───────────────────────────────────────────────────────────────
    save_path = os.path.join(out_dir, save_name)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("out_dir", nargs="?", help="output directory to process")
    ap.add_argument("--all",  action="store_true",
                    help="process all outputs_eqs/shape_* directories")
    ap.add_argument("--out",  default="rf_summary_v5.png",
                    help="output filename within the directory (default: rf_summary_v5.png)")
    ap.add_argument("--base", default="outputs_eqs",
                    help="base directory for --all (default: outputs_eqs)")
    args = ap.parse_args()

    if args.all:
        base = args.base
        dirs = sorted(
            d for d in os.listdir(base)
            if d.startswith("shape_") and os.path.isdir(os.path.join(base, d))
        )
        print(f"Processing {len(dirs)} shape directories …")
        for d in dirs:
            make_rf_summary_v5(os.path.join(base, d), save_name=args.out)
    elif args.out_dir:
        make_rf_summary_v5(args.out_dir, save_name=args.out)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
