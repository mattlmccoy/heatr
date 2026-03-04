#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


def _out_paths(out_dir: Path, base_name: str, suffix: str, write_canonical: bool) -> list[Path]:
    base = Path(base_name)
    out: list[Path] = []
    if suffix:
        out.append(out_dir / f"{base.stem}.{suffix}{base.suffix}")
    else:
        out.append(out_dir / base_name)
    if write_canonical:
        canonical = out_dir / base_name
        if canonical not in out:
            out.append(canonical)
    return out


def _finite_bounds(arr: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return (vmin, vmax)


def _load_time_series(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            try:
                out[str(k)] = np.asarray(v, dtype=float)
            except Exception:
                continue
    return out


def _overlay_part_contour(ax: plt.Axes, x: np.ndarray, y: np.ndarray, part_mask: np.ndarray | None) -> None:
    if part_mask is None:
        return
    try:
        ax.contour(x, y, part_mask.astype(float), levels=[0.5], colors=["w"], linewidths=0.9)
    except Exception:
        return


def _render_thermal(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    temp_c: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    part_mask: np.ndarray | None,
    exposure_label: str,
    reporting_cfg: dict | None = None,
) -> None:
    rep = reporting_cfg or {}
    sig = 1.2
    t_disp = gaussian_filter(np.asarray(temp_c, dtype=float), sigma=sig)
    phi_disp = gaussian_filter(np.asarray(phi, dtype=float), sigma=sig)
    rho_disp = gaussian_filter(np.asarray(rho, dtype=float), sigma=sig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), dpi=180)
    imt = ax[0].imshow(
        t_disp, extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower", cmap="turbo", interpolation="bilinear",
    )
    ax[0].set_title("T final [C]")
    cbt = plt.colorbar(imt, ax=ax[0], shrink=0.84)
    _style_cbar(cbt, decimals=1)

    pm = np.asarray(part_mask, dtype=bool) if part_mask is not None else None
    phi_plot_mode = str(rep.get("phi_display_mode", "full")).strip().lower()
    if pm is not None and np.any(pm) and phi_plot_mode == "part_contrast":
        phi_part = np.asarray(phi[pm], dtype=float)
        phi_vmin = float(max(np.min(phi_part), 0.0))
        phi_vmax = float(min(1.0, max(np.percentile(phi_part, 99.0), phi_vmin + 1e-6)))
    else:
        phi_vmin, phi_vmax = 0.0, 1.0
    phi_show = np.where(pm, phi_disp, np.nan) if pm is not None else phi_disp
    imp = ax[1].imshow(
        phi_show, extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower", cmap="viridis", vmin=phi_vmin, vmax=phi_vmax, interpolation="bilinear",
    )
    ax[1].set_title("Melt fraction phi (part only)")
    cbp = plt.colorbar(imp, ax=ax[1], shrink=0.84)
    _style_cbar(cbp, decimals=2)

    rho_plot_mode = str(rep.get("rho_display_mode", "part_contrast")).strip().lower()
    if pm is not None and np.any(pm) and rho_plot_mode == "part_contrast":
        rho_part = np.asarray(rho[pm], dtype=float)
        rho_lo = float(np.min(rho_part))
        rho_hi = float(np.max(rho_part))
        if rho_hi - rho_lo < 0.02:
            rho_hi = min(1.0, rho_lo + 0.02)
        rho_vmin, rho_vmax = max(0.0, rho_lo), min(1.0, rho_hi)
    else:
        rho_vmin, rho_vmax = 0.0, 1.0
    imr = ax[2].imshow(
        rho_disp, extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower", cmap="magma", vmin=rho_vmin, vmax=rho_vmax, interpolation="bilinear",
    )
    ax[2].set_title("Relative density")
    cbr = plt.colorbar(imr, ax=ax[2], shrink=0.84)
    _style_cbar(cbr, decimals=2)

    for a in ax:
        _overlay_part_contour(a, x, y, part_mask)
        a.set_xlabel("x [mm]")
        a.set_ylabel("y [mm]")
        a.set_aspect("equal")
    fig.suptitle(exposure_label, fontsize=8, color="#555555", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _style_cbar(cb, decimals: int = 1) -> None:
    cb.formatter = plt.FuncFormatter(lambda val, _pos: f"{val:,.{decimals}f}")
    cb.update_ticks()


def _render_electric(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    emag: np.ndarray,
    qrf_wpm3: np.ndarray,
    v: np.ndarray,
    part_mask: np.ndarray | None,
    elec_hi: np.ndarray | None,
    elec_lo: np.ndarray | None,
    reporting_cfg: dict | None = None,
) -> None:
    rep = reporting_cfg or {}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), dpi=180)
    vmag = np.abs(v)
    e_kvpm = np.asarray(emag, dtype=float) / 1e3
    qrf_mwpm3 = np.asarray(qrf_wpm3, dtype=float) / 1e6

    q_part = np.asarray(qrf_mwpm3, dtype=float).ravel()
    if part_mask is not None:
        pm = np.asarray(part_mask, dtype=bool)
        if np.any(pm):
            q_part = qrf_mwpm3[pm]
    q_mode = str(rep.get("qrf_display_mode", "part_p99")).strip().lower()
    q_vmax_raw = rep.get("qrf_display_vmax_mw_per_m3", None)
    if q_vmax_raw is None:
        if q_mode == "part_p95":
            q_vmax = float(np.percentile(q_part, 95.0))
        elif q_mode == "global_max":
            q_vmax = float(np.max(qrf_mwpm3))
        else:
            q_vmax = float(np.percentile(q_part, 99.0))
    else:
        q_vmax = float(q_vmax_raw)
    q_vmax = max(q_vmax, 1e-6)
    q_disp = np.clip(qrf_mwpm3, 0.0, q_vmax)

    im0 = ax[0].imshow(
        vmag,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap="viridis",
        interpolation="bilinear",
    )
    ax[0].set_title("|V| [V]")
    cb0 = plt.colorbar(im0, ax=ax[0], shrink=0.84)
    _style_cbar(cb0, decimals=0)

    im1 = ax[1].imshow(
        e_kvpm,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap="turbo",
        interpolation="bilinear",
    )
    ax[1].set_title("|E| [kV/m]")
    cb1 = plt.colorbar(im1, ax=ax[1], shrink=0.84)
    _style_cbar(cb1, decimals=1)

    im2 = ax[2].imshow(
        q_disp,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=q_vmax,
        interpolation="bilinear",
    )
    ax[2].set_title(f"Qrf [MW/m^3] (clip 0-{q_vmax:.2f})")
    cb2 = plt.colorbar(im2, ax=ax[2], shrink=0.84)
    _style_cbar(cb2, decimals=2)

    ex_plot = np.asarray(np.real(ex), dtype=float).copy()
    ey_plot = np.asarray(np.real(ey), dtype=float).copy()
    if elec_hi is not None and elec_lo is not None:
        e_mask = np.asarray(elec_hi, dtype=bool) | np.asarray(elec_lo, dtype=bool)
        ex_plot[e_mask] = np.nan
        ey_plot[e_mask] = np.nan

    for a in ax[1:]:
        try:
            sps = a.streamplot(
                x, y, ex_plot, ey_plot,
                density=0.6, color="w", linewidth=0.9, arrowsize=0.6,
                minlength=0.06, maxlength=6.0, integration_direction="both", broken_streamlines=True, zorder=4,
            )
            if hasattr(sps, "lines"):
                sps.lines.set_alpha(0.92)
            if hasattr(sps, "arrows") and sps.arrows is not None:
                sps.arrows.set_alpha(0.92)
        except Exception:
            pass

    for a in ax:
        _overlay_part_contour(a, x, y, part_mask)
        if elec_hi is not None and elec_lo is not None:
            try:
                a.contour(x, y, np.asarray(elec_hi, dtype=float), levels=[0.5], colors=["#00ffcc"], linewidths=0.9)
                a.contour(x, y, np.asarray(elec_lo, dtype=float), levels=[0.5], colors=["#ff66cc"], linewidths=0.9)
            except Exception:
                pass
        a.set_xlabel("x [mm]")
        a.set_ylabel("y [mm]")
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_time_series(out_path: Path, ts: dict[str, np.ndarray]) -> bool:
    t = ts.get("time_s")
    if t is None or t.size == 0:
        return False
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=180)
    if "mean_T_part_c" in ts:
        ax[0, 0].plot(t, ts["mean_T_part_c"], "-", lw=1.6, label="mean T")
    if "max_T_part_c" in ts:
        ax[0, 0].plot(t, ts["max_T_part_c"], "-", lw=1.2, label="max T")
    ax[0, 0].set_title("Part temperature")
    ax[0, 0].set_xlabel("time [s]")
    ax[0, 0].set_ylabel("T [C]")
    ax[0, 0].grid(alpha=0.25)
    if ax[0, 0].lines:
        ax[0, 0].legend(loc="best", fontsize=8)

    ax_pw = ax[0, 1]
    ax_en = ax_pw.twinx()
    if "power_doped_W_per_m" in ts:
        ax_pw.plot(t, ts["power_doped_W_per_m"], "-", lw=1.6, color="tab:blue", label="Q_rf power")
    if "power_conv_loss_W_per_m" in ts:
        ax_pw.plot(t, ts["power_conv_loss_W_per_m"], "--", lw=1.2, color="tab:cyan", label="conv loss")
    if "energy_doped_J_per_m" in ts:
        ax_en.plot(t, ts["energy_doped_J_per_m"], "-", lw=1.2, color="tab:orange", label="cum. Q_rf")
    if "energy_stored_part_J_per_m" in ts:
        ax_en.plot(t, ts["energy_stored_part_J_per_m"], "-", lw=1.2, color="tab:red", label="stored")
    ax_pw.set_title("Power and cumulative energy")
    ax_pw.set_xlabel("time [s]")
    ax_pw.set_ylabel("Power [W/m depth]", color="tab:blue")
    ax_en.set_ylabel("Energy [J/m depth]", color="tab:orange")
    ax_pw.tick_params(axis="y", colors="tab:blue")
    ax_en.tick_params(axis="y", colors="tab:orange")
    h1, l1 = ax_pw.get_legend_handles_labels()
    h2, l2 = ax_en.get_legend_handles_labels()
    if h1 or h2:
        ax_pw.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax_pw.grid(alpha=0.25)

    if "ui_abs_part" in ts:
        ax[1, 0].plot(t, ts["ui_abs_part"], "-", lw=1.6, label="UI_abs")
    if "ui_rms_part" in ts:
        ax[1, 0].plot(t, ts["ui_rms_part"], "-", lw=1.2, label="UI_rms")
    ax[1, 0].set_title("Uniformity index")
    ax[1, 0].set_xlabel("time [s]")
    ax[1, 0].set_ylabel("UI")
    ax[1, 0].grid(alpha=0.25)
    if ax[1, 0].lines:
        ax[1, 0].legend(loc="best", fontsize=8)

    if "mean_phi_part" in ts:
        ax[1, 1].plot(t, ts["mean_phi_part"], "-", lw=1.6, label="phi")
    if "mean_rho_rel_part" in ts:
        ax[1, 1].plot(t, ts["mean_rho_rel_part"], "-", lw=1.2, label="rho_rel")
    ax[1, 1].set_title("Melt and densification")
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylabel("fraction")
    ax[1, 1].set_ylim(0.0, 1.0)
    ax[1, 1].grid(alpha=0.25)
    ax_r = ax[1, 1].twinx()
    if "mean_dens_rate_part_per_s" in ts:
        ax_r.plot(t, ts["mean_dens_rate_part_per_s"], "-", lw=1.0, color="tab:green", alpha=0.9)
        ax_r.set_ylabel("drho/dt [1/s]", color="tab:green")
        ax_r.tick_params(axis="y", colors="tab:green")
    if ax[1, 1].lines:
        ax[1, 1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _render_paper_style(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    temp_c: np.ndarray,
    qrf_wpm3: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    v: np.ndarray,
    part_mask: np.ndarray | None,
    elec_hi: np.ndarray | None,
    elec_lo: np.ndarray | None,
    ts: dict[str, np.ndarray],
    exposure_label: str,
) -> None:
    temp_disp = gaussian_filter(np.asarray(temp_c, dtype=float), sigma=1.2)
    qrf_mwpm3 = gaussian_filter(np.asarray(qrf_wpm3, dtype=float) / 1.0e6, sigma=1.2)
    tmin, tmax = _finite_bounds(temp_disp)
    qmin, qmax = _finite_bounds(qrf_mwpm3)
    fig, ax = plt.subplots(2, 2, figsize=(10.6, 8.8), dpi=180)
    ex_plot = np.asarray(np.real(ex), dtype=float).copy()
    ey_plot = np.asarray(np.real(ey), dtype=float).copy()
    if elec_hi is not None and elec_lo is not None:
        em = np.asarray(elec_hi, dtype=bool) | np.asarray(elec_lo, dtype=bool)
        ex_plot[em] = np.nan
        ey_plot[em] = np.nan
    def _overlay_stream(ax_obj: plt.Axes) -> None:
        try:
            sps = ax_obj.streamplot(
                x, y, ex_plot, ey_plot,
                density=0.6, color="w", linewidth=0.9, arrowsize=0.6,
                minlength=0.06, maxlength=6.0, integration_direction="both", broken_streamlines=True, zorder=4,
            )
            if hasattr(sps, "lines"):
                sps.lines.set_alpha(0.92)
            if hasattr(sps, "arrows") and sps.arrows is not None:
                sps.arrows.set_alpha(0.92)
        except Exception:
            pass

    im0 = ax[0, 0].imshow(
        temp_disp,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap="jet",
        vmin=tmin,
        vmax=tmax,
        interpolation="bilinear",
    )
    _overlay_part_contour(ax[0, 0], x, y, part_mask)
    _overlay_stream(ax[0, 0])
    ax[0, 0].set_title(f"Temperature [C] ({tmin:.1f}-{tmax:.1f})")
    ax[0, 0].set_aspect("equal")
    c0 = plt.colorbar(im0, ax=ax[0, 0], shrink=0.84)
    _style_cbar(c0, decimals=1)

    im1 = ax[0, 1].imshow(
        qrf_mwpm3,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap="jet",
        vmin=qmin,
        vmax=qmax,
        interpolation="bilinear",
    )
    _overlay_part_contour(ax[0, 1], x, y, part_mask)
    _overlay_stream(ax[0, 1])
    ax[0, 1].set_title(f"Qrf [MW/m^3] ({qmin:.2f}-{qmax:.2f})")
    ax[0, 1].set_aspect("equal")
    c1 = plt.colorbar(im1, ax=ax[0, 1], shrink=0.84)
    _style_cbar(c1, decimals=2)

    t = ts.get("time_s")
    if t is not None and t.size > 0:
        if "mean_T_part_c" in ts:
            ax[1, 0].plot(t, ts["mean_T_part_c"], "-", lw=1.6, label="mean T")
        if "max_T_part_c" in ts:
            ax[1, 0].plot(t, ts["max_T_part_c"], "-", lw=1.2, label="max T")
        if "ui_abs_part" in ts:
            ax[1, 0].plot(t, ts["ui_abs_part"], "-", lw=1.0, label="UI_abs")
        ax[1, 0].set_title("Thermal trajectory")
        ax[1, 0].set_xlabel("time [s]")
        ax[1, 0].grid(alpha=0.25)
        if ax[1, 0].lines:
            ax[1, 0].legend(loc="best", fontsize=8)

        if "mean_phi_part" in ts:
            ax[1, 1].plot(t, ts["mean_phi_part"], "-", lw=1.6, label="mean phi")
        if "mean_rho_rel_part" in ts:
            ax[1, 1].plot(t, ts["mean_rho_rel_part"], "-", lw=1.2, label="mean rho_rel")
        ax[1, 1].set_title("Melt and densification")
        ax[1, 1].set_xlabel("time [s]")
        ax[1, 1].set_ylim(0.0, 1.0)
        ax[1, 1].grid(alpha=0.25)
        if ax[1, 1].lines:
            ax[1, 1].legend(loc="upper left", fontsize=8)
        axr = ax[1, 1].twinx()
        if "energy_balance_residual_J_per_m" in ts:
            axr.plot(t, ts["energy_balance_residual_J_per_m"], "-", lw=1.0, color="tab:red", alpha=0.9)
            axr.set_ylabel("residual [J/m]", color="tab:red")
            axr.tick_params(axis="y", colors="tab:red")
    else:
        ax[1, 0].text(0.5, 0.5, "time_series.json missing", ha="center", va="center")
        ax[1, 1].text(0.5, 0.5, "time_series.json missing", ha="center", va="center")
        ax[1, 0].set_title("Thermal trajectory")
        ax[1, 1].set_title("Melt and densification")

    for a in [ax[0, 0], ax[0, 1]]:
        if elec_hi is not None and elec_lo is not None:
            try:
                a.contour(x, y, np.asarray(elec_hi, dtype=float), levels=[0.5], colors=["#00ffcc"], linewidths=0.9)
                a.contour(x, y, np.asarray(elec_lo, dtype=float), levels=[0.5], colors=["#ff66cc"], linewidths=0.9)
            except Exception:
                pass
        try:
            levels = np.linspace(float(np.min(np.real(v))), float(np.max(np.real(v))), 18)
            a.contour(x, y, np.real(v), levels=levels, colors="w", linewidths=0.5, alpha=0.45)
        except Exception:
            pass
        a.set_aspect("equal")
    fig.suptitle(exposure_label, fontsize=8, color="#555555", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill core report plots from existing fields/time-series artifacts.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Run output directory.")
    ap.add_argument("--output-suffix", type=str, default="", help="Optional suffix inserted before extension.")
    ap.add_argument("--write-canonical", action="store_true", help="Also overwrite canonical report filenames.")
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    fields_path = out_dir / "fields.npz"
    if not fields_path.exists():
        raise SystemExit("Missing fields.npz; cannot backfill core reports.")
    npz = np.load(fields_path, allow_pickle=True)
    x = np.asarray(npz["x"], dtype=float)
    y = np.asarray(npz["y"], dtype=float)
    temp_c = np.asarray(npz["T"], dtype=float)
    phi = np.asarray(npz["phi"], dtype=float) if "phi" in npz.files else np.zeros_like(temp_c)
    rho = np.asarray(npz["rho_rel"], dtype=float) if "rho_rel" in npz.files else np.zeros_like(temp_c)
    qrf = np.asarray(npz["Qrf"], dtype=float)
    emag = np.asarray(npz["E_mag"], dtype=float)
    ex = np.asarray(np.real(npz["Ex"]), dtype=float) if "Ex" in npz.files else np.zeros_like(emag)
    ey = np.asarray(np.real(npz["Ey"]), dtype=float) if "Ey" in npz.files else np.zeros_like(emag)
    v = np.asarray(npz["V"])
    part_mask = np.asarray(npz["part_mask"], dtype=bool) if "part_mask" in npz.files else None
    elec_hi = np.asarray(npz["elec_hi"], dtype=bool) if "elec_hi" in npz.files else None
    elec_lo = np.asarray(npz["elec_lo"], dtype=bool) if "elec_lo" in npz.files else None
    reporting_cfg = {}
    cfg_path = out_dir / "used_config.yaml"
    if cfg_path.exists():
        try:
            raw_cfg = yaml.safe_load(cfg_path.read_text())
            if isinstance(raw_cfg, dict) and isinstance(raw_cfg.get("reporting", {}), dict):
                reporting_cfg = dict(raw_cfg.get("reporting", {}))
        except Exception:
            reporting_cfg = {}
    ts = _load_time_series(out_dir / "time_series.json")
    t_series = ts.get("time_s")
    t_total_s = float(t_series[-1]) if isinstance(t_series, np.ndarray) and t_series.size > 0 else 0.0
    mm, ss = divmod(t_total_s, 60.0)
    exposure_label = f"Exposure: {t_total_s:.0f} s  ({int(mm)} min {ss:.0f} s)"
    suffix = str(args.output_suffix or "").strip()

    produced: list[Path] = []
    for p in _out_paths(out_dir, "thermal_fields_final.png", suffix, args.write_canonical):
        _render_thermal(p, x, y, temp_c, phi, rho, part_mask, exposure_label, reporting_cfg)
        produced.append(p)
    for p in _out_paths(out_dir, "electric_fields.png", suffix, args.write_canonical):
        _render_electric(p, x, y, ex, ey, emag, qrf, v, part_mask, elec_hi, elec_lo, reporting_cfg)
        produced.append(p)
    for p in _out_paths(out_dir, "paper_style_report.png", suffix, args.write_canonical):
        _render_paper_style(p, x, y, temp_c, qrf, ex, ey, v, part_mask, elec_hi, elec_lo, ts, exposure_label)
        produced.append(p)
    for p in _out_paths(out_dir, "time_series.png", suffix, args.write_canonical):
        if _render_time_series(p, ts):
            produced.append(p)

    names = ", ".join(pp.name for pp in produced)
    print(f"Backfilled core reports: {names}")


if __name__ == "__main__":
    main()
