#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _arr(d: dict[str, Any], key: str) -> np.ndarray:
    v = d.get(key, [])
    if not isinstance(v, list):
        return np.array([], dtype=float)
    return np.asarray([float(x) for x in v], dtype=float)


def _cross_time(t: np.ndarray, y: np.ndarray, thr: float) -> float | None:
    if len(t) == 0 or len(y) == 0:
        return None
    for i in range(len(y)):
        if y[i] >= thr:
            return float(t[i])
    return None


def _fmt_time_s(x: float | None) -> str:
    if x is None:
        return "not reached"
    return f"{x:.1f}s"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: np.asarray(z[k]) for k in z.files}


def main() -> None:
    p = argparse.ArgumentParser(description="Build square heating-stage alignment figure (baseline vs DSC-calibrated model).")
    p.add_argument("--baseline-dir", default="outputs_eqs/prewarp_square/verification")
    p.add_argument("--experimental-dir", default="outputs_eqs/_experimental/square_static_stage_b_ab-20260302-square1")
    p.add_argument("--output", default="outputs_eqs/_experimental_ab/screen-20260302-allstatic/figure_square_stage_alignment.png")
    args = p.parse_args()

    bdir = Path(args.baseline_dir).resolve()
    edir = Path(args.experimental_dir).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    bsum = _load_json(bdir / "summary.json")
    esum = _load_json(edir / "summary.json")
    bts = _load_json(bdir / "time_series.json")
    ets = _load_json(edir / "time_series.json")
    bcfg = _load_yaml(bdir / "used_config.yaml")
    ecfg = _load_yaml(edir / "used_config.yaml")
    bfields = _load_npz(bdir / "fields.npz")
    efields = _load_npz(edir / "fields.npz")

    tb = _arr(bts, "time_s")
    te = _arr(ets, "time_s")
    Tb = _arr(bts, "mean_T_part_c")
    Te = _arr(ets, "mean_T_part_c")
    phib = _arr(bts, "mean_phi_part")
    phie = _arr(ets, "mean_phi_part")
    rhob = _arr(bts, "mean_rho_rel_part")
    rhoe = _arr(ets, "mean_rho_rel_part")

    b_lo = float(bsum.get("phase_transition_lo_c", 175.0))
    b_hi = float(bsum.get("phase_transition_hi_c", 185.0))
    e_lo = float(esum.get("dsc_profile_melt_onset_c", esum.get("phase_transition_lo_c", 171.0)))
    e_pk = float(esum.get("dsc_profile_melt_peak_c", 180.8))
    e_hi = float(esum.get("dsc_profile_melt_end_c", esum.get("phase_transition_hi_c", 186.0)))

    b_t_on = _cross_time(tb, Tb, b_lo)
    b_t_off = _cross_time(tb, Tb, b_hi)
    e_t_on = _cross_time(te, Te, e_lo)
    e_t_pk = _cross_time(te, Te, e_pk)
    e_t_off = _cross_time(te, Te, e_hi)

    b_phase = bcfg.get("thermal", {}).get("phase_change", {}) if isinstance(bcfg.get("thermal", {}), dict) else {}
    e_phase = ecfg.get("phase_model", {}) if isinstance(ecfg.get("phase_model", {}), dict) else {}

    baseline_tech = "Baseline: COMSOL smoothed Heaviside apparent-heat-capacity transition"
    exp_tech = "Experimental: DSC-calibrated apparent-heat-capacity melt interval"
    bdens = bcfg.get("densification", {}) if isinstance(bcfg.get("densification", {}), dict) else {}
    baseline_cfg_txt = (
        f"{baseline_tech}; "
        f"t_pc={b_phase.get('t_pc_c', 180)}C, dT={b_phase.get('dt_pc_c', 10)}C, "
        f"densification={bdens.get('model', 'physics_dual')}."
    )
    exp_cfg_txt = (
        f"{exp_tech}; onset/peak/end={e_lo:.1f}/{e_pk:.1f}/{e_hi:.1f}C, "
        f"latent={float(esum.get('dsc_profile_lat_heat_j_per_kg', 101700.0))/1000.0:.1f} kJ/kg."
    )

    t_final = max(float(tb[-1]) if len(tb) else 0.0, float(te[-1]) if len(te) else 0.0)

    fig = plt.figure(figsize=(15.6, 9.6), dpi=180)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[2.05, 1.0],
        height_ratios=[1.70, 0.95, 1.18],
        hspace=0.40,
        wspace=0.22,
    )
    ax_temp = fig.add_subplot(gs[0, 0])
    ax_stage = fig.add_subplot(gs[1, 0])
    ax_prof = fig.add_subplot(gs[2, 0])
    ax_map_t = fig.add_subplot(gs[0, 1])
    ax_leg_ts = fig.add_subplot(gs[1, 1])
    ax_leg_out = fig.add_subplot(gs[2, 1])
    ax_leg_ts.axis("off")
    ax_leg_out.axis("off")

    # Panel 1: mean temperature traces + DSC/baseline windows.
    ax_temp.plot(tb, Tb, color="#1d3557", lw=2.1, label="Baseline mean T")
    ax_temp.plot(te, Te, color="#e76f51", lw=2.1, label="Experimental mean T")
    ax_temp.axhspan(e_lo, e_hi, color="#f4a261", alpha=0.22, label="DSC melt interval (exp)")
    ax_temp.axhline(e_pk, color="#f4a261", ls="--", lw=1.2, label="DSC melt peak")
    ax_temp.axhspan(b_lo, b_hi, color="#457b9d", alpha=0.13, label="Baseline phase interval")
    if b_t_on is not None:
        ax_temp.axvline(b_t_on, color="#1d3557", ls=":", lw=1.0)
    if b_t_off is not None:
        ax_temp.axvline(b_t_off, color="#1d3557", ls=":", lw=1.0)
    if e_t_on is not None:
        ax_temp.axvline(e_t_on, color="#e76f51", ls=":", lw=1.0)
    if e_t_off is not None:
        ax_temp.axvline(e_t_off, color="#e76f51", ls=":", lw=1.0)
    ax_temp.set_xlim(0, t_final)
    ax_temp.set_ylabel("Mean part temperature (C)")
    ax_temp.set_title("Mean Temperature with DSC and Baseline Transition Windows")
    ax_temp.grid(alpha=0.2)

    # Difference series for companion plots and external legends.
    n = min(len(tb), len(te), len(Tb), len(Te))
    td = te[:n] if n > 0 else np.array([], dtype=float)
    dT = (Te[:n] - Tb[:n]) if n > 0 else np.array([], dtype=float)

    # Panel 3: stage bars (explicitly marks solidification as conceptual/not simulated).
    ax_stage.set_xlim(0, t_final)
    ax_stage.set_ylim(-0.6, 1.6)
    ax_stage.set_yticks([1.0, 0.0])
    ax_stage.set_yticklabels(["Baseline", "Experimental"])
    ax_stage.set_title("Stage Map (Heating Window Only)", pad=4)

    def _bar(y: float, x0: float, x1: float, c: str, label: str, hatch: str | None = None):
        w = max(x1 - x0, 0.0)
        ax_stage.barh(y, w, left=x0, height=0.32, color=c, edgecolor="black", hatch=hatch, linewidth=0.6)

    # Baseline stages
    b_on = b_t_on or 0.0
    b_off = b_t_off or t_final
    _bar(1.0, 0.0, b_on, "#a8dadc", "Heating (solid)")
    _bar(1.0, b_on, b_off, "#457b9d", "Phase transition")
    _bar(1.0, b_off, t_final, "#1d3557", "Post-transition heating")

    # Experimental stages
    e_on = e_t_on or 0.0
    e_off = e_t_off or t_final
    _bar(0.0, 0.0, e_on, "#fef3c7", "Heating (solid)")
    _bar(0.0, e_on, e_off, "#f4a261", "DSC melt interval")
    _bar(0.0, e_off, t_final, "#e76f51", "Post-melt heating")

    e_pk_x = e_t_pk if e_t_pk is not None else 0.5 * (e_on + e_off)

    for x in [b_on, b_off, e_on, e_pk_x, e_off]:
        ax_stage.axvline(x, color="gray", ls=":", lw=0.8, alpha=0.65)

    # Temperature/time callouts for phase windows on the stage map.
    def _callout(x: float, y: float, txt: str, dx: float = 0.0, dy: float = 0.0):
        ax_stage.annotate(
            txt,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            textcoords="data",
            fontsize=7.2,
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#555"},
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#888", "alpha": 0.9},
        )

    xoff = 0.05 * max(t_final, 1.0)
    _callout(b_on, 1.14, f"Baseline onset\n{b_lo:.1f}C @ {_fmt_time_s(b_t_on)}", dx=-xoff, dy=0.20)
    _callout(b_off, 1.14, f"Baseline end\n{b_hi:.1f}C @ {_fmt_time_s(b_t_off)}", dx=+xoff, dy=0.20)
    _callout(e_on, 0.14, f"DSC onset\n{e_lo:.1f}C @ {_fmt_time_s(e_t_on)}", dx=-xoff, dy=-0.36)
    _callout(e_pk_x, 0.14, f"DSC peak\n{e_pk:.1f}C @ {_fmt_time_s(e_t_pk)}", dx=0.0, dy=-0.54)
    _callout(e_off, 0.14, f"DSC end\n{e_hi:.1f}C @ {_fmt_time_s(e_t_off)}", dx=+xoff, dy=-0.36)

    stage_legend = [
        Patch(facecolor="#a8dadc", edgecolor="black", label="Heating (baseline)"),
        Patch(facecolor="#457b9d", edgecolor="black", label="Baseline phase-transition window"),
        Patch(facecolor="#fef3c7", edgecolor="black", label="Heating (experimental)"),
        Patch(facecolor="#f4a261", edgecolor="black", label="DSC melt interval"),
    ]

    # Panel 4: profiles + differences in one clean panel.
    n = min(len(tb), len(te), len(Tb), len(Te), len(phib), len(phie), len(rhob), len(rhoe))
    td = te[:n] if n > 0 else np.array([], dtype=float)
    dphi = (phie[:n] - phib[:n]) if n > 0 else np.array([], dtype=float)
    drho = (rhoe[:n] - rhob[:n]) if n > 0 else np.array([], dtype=float)
    ax_prof.plot(tb, phib, color="#1d3557", lw=1.8, label="Baseline phi")
    ax_prof.plot(te, phie, color="#e76f51", lw=1.8, label="Experimental phi")
    ax_prof.plot(tb, rhob, color="#264653", lw=1.4, ls="--", label="Baseline rho_rel")
    ax_prof.plot(te, rhoe, color="#2a9d8f", lw=1.4, ls="--", label="Experimental rho_rel")
    ax_prof.set_xlim(0, t_final)
    ax_prof.set_ylim(0.0, 1.05)
    ax_prof.set_xlabel("Time (s)")
    ax_prof.set_ylabel("phi / rho_rel")
    ax_prof.set_title("Outputs Through Heating", pad=6)
    ax_prof.grid(alpha=0.2)

    # Right-column context maps from real simulated fields (experimental final state).
    x = np.asarray(efields["x"], dtype=float)
    y = np.asarray(efields["y"], dtype=float)
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    pmask = np.asarray(efields["part_mask"], dtype=bool)
    Tmap = np.where(pmask, np.asarray(efields["T"], dtype=float), np.nan)
    imt = ax_map_t.imshow(Tmap, origin="lower", extent=extent, cmap="turbo")
    ax_map_t.set_title("Experimental final thermal field (C)")
    ax_map_t.set_xlabel("x (m)")
    ax_map_t.set_ylabel("y (m)")
    fig.colorbar(imt, ax=ax_map_t, fraction=0.045, pad=0.03)

    # Figure annotations.
    fig.suptitle("HEATR Square: Heating/Transition Stage Alignment (Baseline vs DSC-Calibrated)", fontsize=12, y=0.989)
    fig.text(
        0.5,
        0.957,
        "Baseline: COMSOL AHC (smoothed Heaviside)  |  Experimental: DSC-calibrated AHC  |  "
        f"Crossings onset/end: {_fmt_time_s(b_t_on)}/{_fmt_time_s(b_t_off)} vs {_fmt_time_s(e_t_on)}/{_fmt_time_s(e_t_off)}",
        ha="center",
        va="top",
        fontsize=7.8,
    )
    fig.subplots_adjust(top=0.87)

    # Dedicated legend panels (outside all data axes).
    temp_handles = [
        Line2D([0], [0], color="#1d3557", lw=2.1, label="Baseline traces"),
        Line2D([0], [0], color="#e76f51", lw=2.1, label="Experimental traces"),
        Patch(facecolor="#457b9d", alpha=0.13, edgecolor="none", label="Baseline phase interval"),
        Patch(facecolor="#f4a261", alpha=0.22, edgecolor="none", label="DSC melt interval"),
        Line2D([0], [0], color="#f4a261", lw=1.2, ls="--", label="DSC melt peak"),
    ]
    prof_handles = [
        Line2D([0], [0], color="#1d3557", lw=1.8, label="Baseline phi"),
        Line2D([0], [0], color="#e76f51", lw=1.8, label="Experimental phi"),
        Line2D([0], [0], color="#264653", lw=1.4, ls="--", label="Baseline rho_rel"),
        Line2D([0], [0], color="#2a9d8f", lw=1.4, ls="--", label="Experimental rho_rel"),
    ]

    # Right-middle legend section: temperature + stage map semantics.
    ax_leg_ts.text(0.0, 1.0, "Section Legend: Temperature + Stage Map", ha="left", va="top", fontsize=9.2, fontweight="bold", transform=ax_leg_ts.transAxes)
    leg_a = ax_leg_ts.legend(handles=temp_handles, loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=7.5, frameon=False, ncol=1)
    ax_leg_ts.add_artist(leg_a)
    leg_b = ax_leg_ts.legend(handles=stage_legend, loc="upper left", bbox_to_anchor=(0.0, 0.30), fontsize=7.2, frameon=False, ncol=1)
    ax_leg_ts.add_artist(leg_b)

    # Right-bottom legend section: output traces.
    ax_leg_out.text(0.0, 1.0, "Section Legend: Outputs Through Heating", ha="left", va="top", fontsize=9.2, fontweight="bold", transform=ax_leg_out.transAxes)
    ax_leg_out.legend(handles=prof_handles, loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=7.8, frameon=False, ncol=1)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage-figure] wrote {out}")

    # Companion: difference-focused view to make subtle separation explicit.
    diff_out = out.with_name(out.stem + "_differences" + out.suffix)
    fig2 = plt.figure(figsize=(10.2, 6.8), dpi=180)
    gs2 = fig2.add_gridspec(2, 2, hspace=0.32, wspace=0.26, height_ratios=[1.05, 1.0])
    adt = fig2.add_subplot(gs2[0, 0])
    adp = fig2.add_subplot(gs2[0, 1])
    amdT = fig2.add_subplot(gs2[1, 0])
    amdr = fig2.add_subplot(gs2[1, 1])

    # Time-series deltas
    adt.plot(td, dT, color="#8d0801", lw=2.0)
    adt.axhline(0.0, color="black", lw=0.9)
    adt.set_xlim(0, t_final)
    adt.set_title("Delta Mean Temperature (Experimental - Baseline)")
    adt.set_xlabel("Time (s)")
    adt.set_ylabel("Delta T (C)")
    adt.grid(alpha=0.25)

    adp.plot(td, dphi, color="#1d3557", lw=2.0, ls="--", label="Delta phi")
    adp.plot(td, drho, color="#2a9d8f", lw=2.0, ls="-.", label="Delta rho_rel")
    adp.axhline(0.0, color="black", lw=0.9)
    adp.set_xlim(0, t_final)
    adp.set_title("Delta Melt Fraction / Delta Relative Density")
    adp.set_xlabel("Time (s)")
    adp.set_ylabel("Delta value")
    adp.legend(fontsize=8)
    adp.grid(alpha=0.25)

    # Spatial deltas on part domain
    bmask = np.asarray(bfields["part_mask"], dtype=bool)
    emask = np.asarray(efields["part_mask"], dtype=bool)
    cmask = np.logical_or(bmask, emask)
    dT_map = np.where(cmask, np.asarray(efields["T"], dtype=float) - np.asarray(bfields["T"], dtype=float), np.nan)
    dr_map = np.where(cmask, np.asarray(efields["rho_rel"], dtype=float) - np.asarray(bfields["rho_rel"], dtype=float), np.nan)
    vT = float(np.nanmax(np.abs(dT_map))) if np.isfinite(dT_map).any() else 1.0
    vr = float(np.nanmax(np.abs(dr_map))) if np.isfinite(dr_map).any() else 0.05
    vT = max(vT, 1e-3)
    vr = max(vr, 1e-4)
    imdT = amdT.imshow(dT_map, origin="lower", extent=extent, cmap="coolwarm", vmin=-vT, vmax=vT)
    imdr = amdr.imshow(dr_map, origin="lower", extent=extent, cmap="coolwarm", vmin=-vr, vmax=vr)
    amdT.set_title("Spatial Delta T (C)")
    amdr.set_title("Spatial Delta rho_rel")
    for a in (amdT, amdr):
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")
    cb1 = fig2.colorbar(imdT, ax=amdT, fraction=0.046, pad=0.03)
    cb2 = fig2.colorbar(imdr, ax=amdr, fraction=0.046, pad=0.03)
    cb1.set_label("C")
    cb2.set_label("rho_rel")

    fig2.suptitle("HEATR Square: Difference-Focused View (Experimental - Baseline)", fontsize=12, y=0.985)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(diff_out, bbox_inches="tight")
    plt.close(fig2)
    print(f"[stage-figure] wrote {diff_out}")


if __name__ == "__main__":
    main()
