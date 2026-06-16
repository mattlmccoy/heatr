#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG = ROOT / ".mplconfig"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

OVERLEAF = ROOT / "rfam_paper_overleaf"
FIG_DIR = OVERLEAF / "figures"
GEN_DIR = OVERLEAF / "generated"
OUT = ROOT / "outputs_eqs"

SINGLE_SUMMARY_PATH = (
    OUT
    / "runs"
    / "unknown"
    / "single"
    / "experimental"
    / "experimental_ab"
    / "screen-20260302-allstatic"
    / "geometry_performance_summary.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Invalid JSON list: {path}")
    return data


def _energy_residual_pct(summary: dict[str, Any]) -> float | None:
    e_in = summary.get("energy_doped_total_J_per_m")
    e_res = summary.get("energy_balance_residual_final_J_per_m")
    if e_in in (None, 0):
        return None
    return 100.0 * abs(float(e_res)) / max(abs(float(e_in)), 1e-9)


def _boundary_stats(fields_path: Path) -> dict[str, float]:
    f = np.load(fields_path)
    rho = np.asarray(f["rho_rel"], dtype=float)
    temp = np.asarray(f["T"], dtype=float)
    part_mask = np.asarray(f["part_mask"], dtype=bool)
    boundary = part_mask & ~binary_erosion(part_mask)
    rho_b = rho[boundary]
    return {
        "boundary_rho_std": float(np.std(rho_b)),
        "boundary_rho_min": float(np.min(rho_b)),
        "boundary_rho_max": float(np.max(rho_b)),
        "boundary_rho_min_over_max": float(np.min(rho_b) / max(np.max(rho_b), 1e-9)),
        "mean_rho": float(np.mean(rho[part_mask])),
        "mean_temp_c": float(np.mean(temp[part_mask])),
    }


def _plot_heatmap(ax: Any, fields_path: Path, title: str, subtitle: str) -> None:
    f = np.load(fields_path)
    rho = np.asarray(f["rho_rel"], dtype=float)
    part_mask = np.asarray(f["part_mask"], dtype=bool)
    x = np.asarray(f["x"], dtype=float) * 1e3
    y = np.asarray(f["y"], dtype=float) * 1e3
    rho_smooth = np.clip(gaussian_filter(rho, sigma=1.1), 0.0, 1.0)
    mask_smooth = gaussian_filter(part_mask.astype(float), sigma=1.0)

    cols = np.where(part_mask.any(axis=0))[0]
    rows = np.where(part_mask.any(axis=1))[0]
    pad = 7
    r0 = max(0, rows[0] - pad)
    r1 = min(rho.shape[0], rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad)
    c1 = min(rho.shape[1], cols[-1] + pad + 1)

    im = ax.imshow(
        rho_smooth[r0:r1, c0:c1],
        origin="lower",
        extent=[x[c0], x[c1 - 1], y[r0], y[r1 - 1]],
        cmap="plasma",
        vmin=0.55,
        vmax=1.0,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.contour(
        x[c0:c1],
        y[r0:r1],
        mask_smooth[r0:r1, c0:c1],
        levels=[0.5],
        colors="white",
        linewidths=1.1,
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight="bold")
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho_{rel}$")


def _write_manifest(manifest_rows: list[dict[str, Any]]) -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_json = GEN_DIR / "paper_refresh_manifest.json"
    manifest_csv = GEN_DIR / "paper_refresh_manifest.csv"
    manifest_md = GEN_DIR / "paper_refresh_manifest.md"

    payload = {
        "schema_version": 1,
        "single_source_dataset": str(SINGLE_SUMMARY_PATH),
        "rows": manifest_rows,
    }
    manifest_json.write_text(json.dumps(payload, indent=2))

    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    lines = [
        "# Paper Refresh Manifest",
        "",
        "| Label | Mode | Geometry | Physics | Exposure (s) | Residual % | Include | Source |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in manifest_rows:
        lines.append(
            "| {label} | {mode} | {geometry} | {physics_family} | {exposure_s} | {energy_residual_pct} | {include_in_paper} | `{source_path}` |".format(
                **row
            )
        )
    manifest_md.write_text("\n".join(lines))


def _build_single_section(single_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(single_summary["rows"])
    rows.sort(key=lambda r: str(r["geometry"]))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), dpi=180, constrained_layout=True)
    labels = [str(r["geometry"]) for r in rows]
    xs = np.arange(len(rows))
    exp_s = [float(r["recommended_exposure_s"]) for r in rows]
    rho = [float(r["baseline_rho"]) for r in rows]
    dsc = [float(r["dsc_mae_max_across_exposure"]) for r in rows]
    temp_rmse = [float(r["exp_nominal_temp_rmse_c"]) for r in rows]

    axes[0].bar(xs, exp_s, color="#355070")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Recommended exposure [s]")
    axes[0].set_title("A — Single-exposure recommendations across the 17-shape library", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(xs, rho, color="#6d597a", label=r"Baseline mean $\rho_{rel}$")
    ax2 = axes[1].twinx()
    ax2.plot(xs, dsc, "o-", color="#e56b6f", lw=1.8, ms=4.5, label="DSC MAE max")
    ax2.plot(xs, temp_rmse, "s--", color="#2a9d8f", lw=1.4, ms=4.0, label="Temp RMSE @ nominal")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel(r"Mean $\rho_{rel}$ at baseline exposure")
    ax2.set_ylabel("Error metric")
    axes[1].set_title("B — Baseline density and validation-error context for the chosen exposures", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)

    handles_1, labels_1 = axes[1].get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles_1 + handles_2, labels_1 + labels_2, fontsize=8, loc="upper left")

    fig.savefig(FIG_DIR / "fig_single_library_refresh.png", bbox_inches="tight")
    plt.close(fig)

    lines = [
        r"\begin{table*}[tb]",
        r"\centering",
        r"\caption{Single-exposure summary for the refreshed 17-shape library using the DSC-calibrated PA12 model family. The chosen exposure is the minimum sweep exposure retained in the reusable screening dataset that matched the baseline density target while keeping the diagnostic errors reported for transparency.}",
        r"\label{tab:single_library_refresh}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Geometry & Exp. [s] & Base. $\phi$ & Base. $\rho_{rel}$ & DSC MAE$_{max}$ & Temp RMSE$_{360}$ [C] & $\rho$ RMSE$_{360}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['geometry']} & "
            f"{float(row['recommended_exposure_s']):.0f} & "
            f"{float(row['baseline_phi']):.3f} & "
            f"{float(row['baseline_rho']):.3f} & "
            f"{float(row['dsc_mae_max_across_exposure']):.3f} & "
            f"{float(row['exp_nominal_temp_rmse_c']):.2f} & "
            f"{float(row['exp_nominal_rho_rmse']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (GEN_DIR / "paper_refresh_single_table.tex").write_text("\n".join(lines))
    return rows


def _build_turntable_section() -> list[dict[str, Any]]:
    cases = [
        {
            "geometry": "Square",
            "single": OUT / "runs/square/single/experimental/square_single_20260310",
            "turntable": OUT / "runs/square/turntable/experimental/square_tt_20260310_testing_TT_error",
            "schedule": "16x45deg over 360 s",
            "exposure_s": 360,
        },
        {
            "geometry": "Circle",
            "single": OUT / "runs/circle/single/experimental/circle_single_20260305",
            "turntable": OUT / "runs/circle/turntable/experimental/circle_tt_20260305",
            "schedule": "24x30deg over 360 s",
            "exposure_s": 360,
        },
        {
            "geometry": "Equilateral triangle",
            "single": OUT / "runs/equilateral_triangle/single/experimental/equilatera_single_20260310",
            "turntable": OUT / "runs/equilateral_triangle/turntable/experimental/equilatera_tt_20260307",
            "schedule": "16x45deg over 360 s",
            "exposure_s": 360,
        },
        {
            "geometry": "L-shape",
            "single": OUT / "runs/L_shape/single/experimental/lshape_single_20260304_12min",
            "turntable": OUT / "runs/L_shape/turntable/experimental/lshape_tt_20260304_12min",
            "schedule": "16x45deg over 720 s",
            "exposure_s": 720,
        },
    ]

    rows: list[dict[str, Any]] = []
    for case in cases:
        single_summary = _load_json(case["single"] / "summary.json")
        tt_summary = _load_json(case["turntable"] / "summary.json")
        single_bnd = _boundary_stats(case["single"] / "fields.npz")
        tt_bnd = _boundary_stats(case["turntable"] / "fields.npz")
        rows.append(
            {
                "geometry": case["geometry"],
                "schedule": case["schedule"],
                "exposure_s": case["exposure_s"],
                "single_boundary_std": single_bnd["boundary_rho_std"],
                "turntable_boundary_std": tt_bnd["boundary_rho_std"],
                "single_ratio": single_bnd["boundary_rho_min_over_max"],
                "turntable_ratio": tt_bnd["boundary_rho_min_over_max"],
                "single_energy_residual_pct": _energy_residual_pct(single_summary),
                "turntable_energy_residual_pct": _energy_residual_pct(tt_summary),
                "single_mean_rho": float(single_summary["mean_rho_rel_part_final"]),
                "turntable_mean_rho": float(tt_summary["mean_rho_rel_part_final"]),
            }
        )

    xs = np.arange(len(rows))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180, constrained_layout=True)
    axes[0].bar(xs - width / 2, [r["single_boundary_std"] for r in rows], width=width, color="#6d597a", label="Single exposure")
    axes[0].bar(xs + width / 2, [r["turntable_boundary_std"] for r in rows], width=width, color="#2a9d8f", label="Turntable")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([r["geometry"] for r in rows], rotation=20, ha="right")
    axes[0].set_ylabel(r"Boundary std of $\rho_{rel}$")
    axes[0].set_title("A — Boundary-density standard deviation", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(xs - width / 2, [r["single_ratio"] for r in rows], width=width, color="#6d597a", label="Single exposure")
    axes[1].bar(xs + width / 2, [r["turntable_ratio"] for r in rows], width=width, color="#2a9d8f", label="Turntable")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([r["geometry"] for r in rows], rotation=20, ha="right")
    axes[1].set_ylabel(r"Boundary min/max $\rho_{rel}$")
    axes[1].set_title("B — Boundary min/max ratio", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(FIG_DIR / "fig_turntable_core_refresh.png", bbox_inches="tight")
    plt.close(fig)

    lines = [
        r"\begin{table*}[tb]",
        r"\centering",
        r"\caption{Turntable benchmark set for the refreshed manuscript. Boundary metrics were recomputed from the saved fields. The schedules reported here match the reusable run artifacts currently available in the repository, so the exposure durations and discrete rotation sequences are geometry specific.}",
        r"\label{tab:turntable_core_refresh}",
        r"\small",
        r"\begin{tabular}{lcrrrrrr}",
        r"\toprule",
        r"Geometry & Exp. [s] & Schedule & Single std & Turntable std & Single min/max & Turntable min/max & Turntable residual [\%] \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['geometry']} & {row['exposure_s']:.0f} & {row['schedule']} & "
            f"{row['single_boundary_std']:.4f} & {row['turntable_boundary_std']:.4f} & "
            f"{row['single_ratio']:.3f} & {row['turntable_ratio']:.3f} & "
            f"{(row['turntable_energy_residual_pct'] or 0.0):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (GEN_DIR / "paper_refresh_turntable_table.tex").write_text("\n".join(lines))

    return rows


def _build_orientation_section() -> dict[str, Any]:
    run_dir = OUT / "runs/L_shape/orientation_optimizer/experimental/lshape_orient_20260304_6min"
    best = _load_json(run_dir / "orientation_best.json")
    summary = _load_json(run_dir / "summary.json")
    shutil.copy2(run_dir / "paper_style_report.png", FIG_DIR / "fig_orientation_lshape_refresh.png")

    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\caption{L-shape orientation-optimizer showcase used in the refreshed manuscript. This section is framed as the more practical static alternative to a physical turntable because it avoids breaking electrode/media contact during the run.}",
        r"\label{tab:orientation_refresh}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Best angle [deg] & {float(best['rotation_deg']):.1f} \\\\",
        f"Exposure [s] & {float(best['exposure_s']):.0f} \\\\",
        f"Mean $\\rho_{{rel}}$ & {float(best['mean_rho']):.3f} \\\\",
        f"Max temperature [C] & {float(best['max_temp_c']):.1f} \\\\",
        f"Temperature ceiling violation [C] & {float(best['temp_violation']):.1f} \\\\",
        f"Energy residual [\\%] & {(_energy_residual_pct(summary) or 0.0):.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (GEN_DIR / "paper_refresh_orientation_table.tex").write_text("\n".join(lines))
    return {
        "best_angle_deg": float(best["rotation_deg"]),
        "exposure_s": float(best["exposure_s"]),
        "mean_rho": float(best["mean_rho"]),
        "temp_violation_c": float(best["temp_violation"]),
        "energy_residual_pct": _energy_residual_pct(summary),
    }


def _build_antennae_section() -> dict[str, Any]:
    base_run = OUT / "runs/square/single/experimental/square_single_20260310"
    ant_run = OUT / "runs/unknown/single/experimental/experimental/square_single_20260303_antcal"
    ant_best = _load_json(ant_run / "antennae_calibration_best.json")
    ant_case = OUT / "runs/square/single/experimental/case_03_global"
    base_bnd = _boundary_stats(base_run / "fields.npz")
    ant_bnd = _boundary_stats(ant_case / "fields.npz")
    shutil.copy2(ant_run / "antennae_calibration_report.png", FIG_DIR / "fig_antennae_refresh.png")

    improvement_pct = 100.0 * (base_bnd["boundary_rho_std"] - ant_bnd["boundary_rho_std"]) / max(base_bnd["boundary_rho_std"], 1e-9)
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\caption{Antennae energy concentrator snapshot retained as a limited-result section. The current repository only contains a small square-study calibration set, so the refreshed manuscript treats these results as preliminary rather than as a mature design-of-experiments campaign.}",
        r"\label{tab:antennae_refresh}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Best candidate id & {int(ant_best['candidate_id'])} \\\\",
        f"Global antenna size [mm] & {float(ant_best['global_size_mm']):.1f} \\\\",
        f"Square baseline boundary std & {base_bnd['boundary_rho_std']:.4f} \\\\",
        f"Antennae case boundary std & {ant_bnd['boundary_rho_std']:.4f} \\\\",
        f"Boundary-std change [\\%] & {improvement_pct:+.1f} \\\\",
        f"Mean $\\rho_{{rel}}$ & {float(ant_best['mean_rho']):.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (GEN_DIR / "paper_refresh_antennae_table.tex").write_text("\n".join(lines))
    return {
        "candidate_id": int(ant_best["candidate_id"]),
        "global_size_mm": float(ant_best["global_size_mm"]),
        "base_boundary_std": base_bnd["boundary_rho_std"],
        "antennae_boundary_std": ant_bnd["boundary_rho_std"],
        "improvement_pct": improvement_pct,
        "mean_rho": float(ant_best["mean_rho"]),
    }


def _build_prewarp_section() -> dict[str, Any]:
    legacy_fig = FIG_DIR / "fig_prewarp_square.png"
    square_dir = OUT / "runs/square/single/baseline/prewarp_eval_square_L7mm"
    circle_dir = OUT / "runs/circle/single/baseline/prewarp_eval_circle_L5mm_v2"
    legacy_eval = _load_json(square_dir / "eval_summary.json")
    circle_eval = _load_json(circle_dir / "eval_summary.json")
    square_bnd = _boundary_stats(square_dir / "fields.npz")
    circle_bnd = _boundary_stats(circle_dir / "fields.npz")
    # Figure already lives in the Overleaf tree; keep it stable and only emit the updated table snippet.
    if not legacy_fig.exists():
        raise FileNotFoundError(legacy_fig)

    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\caption{Reduced legacy prewarp summary retained only to document the current prototype status. These runs were generated with the earlier baseline-model workflow and are therefore discussed separately from the refreshed DSC-calibrated result set.}",
        r"\label{tab:prewarp_refresh}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Case & RMSE [mm] & Boundary std \\",
        r"\midrule",
        f"Circle legacy prewarp & {float(circle_eval['rmse_mm']):.4f} & {circle_bnd['boundary_rho_std']:.4f} \\\\",
        f"Square legacy prewarp & {float(legacy_eval['rmse_mm']):.4f} & {square_bnd['boundary_rho_std']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (GEN_DIR / "paper_refresh_prewarp_table.tex").write_text("\n".join(lines))
    return {
        "square_rmse_mm": float(legacy_eval["rmse_mm"]),
        "circle_rmse_mm": float(circle_eval["rmse_mm"]),
        "square_boundary_std": square_bnd["boundary_rho_std"],
        "circle_boundary_std": circle_bnd["boundary_rho_std"],
    }


def _build_manifest_rows(
    single_rows: list[dict[str, Any]],
    turntable_rows: list[dict[str, Any]],
    orientation: dict[str, Any],
    prewarp: dict[str, Any],
    antennae: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in single_rows:
        manifest_rows.append(
            {
                "label": f"single-{row['geometry']}",
                "mode": "single",
                "geometry": row["geometry"],
                "physics_family": "experimental_pa12_hybrid",
                "exposure_s": int(float(row["recommended_exposure_s"])),
                "energy_residual_pct": "dataset-level",
                "include_in_paper": "yes",
                "source_path": str(SINGLE_SUMMARY_PATH),
            }
        )
    for row in turntable_rows:
        manifest_rows.append(
            {
                "label": f"turntable-{row['geometry']}",
                "mode": "turntable",
                "geometry": row["geometry"],
                "physics_family": "experimental_pa12_hybrid",
                "exposure_s": int(row["exposure_s"]),
                "energy_residual_pct": f"{(row['turntable_energy_residual_pct'] or 0.0):.2f}",
                "include_in_paper": "yes",
                "source_path": "generated from saved run directories",
            }
        )
    manifest_rows.extend(
        [
            {
                "label": "orientation-lshape",
                "mode": "orientation_optimizer",
                "geometry": "L-shape",
                "physics_family": "experimental_pa12_hybrid",
                "exposure_s": int(orientation["exposure_s"]),
                "energy_residual_pct": f"{(orientation['energy_residual_pct'] or 0.0):.2f}",
                "include_in_paper": "yes",
                "source_path": str(OUT / 'runs/L_shape/orientation_optimizer/experimental/lshape_orient_20260304_6min'),
            },
            {
                "label": "prewarp-legacy-square",
                "mode": "prewarp",
                "geometry": "Square",
                "physics_family": "baseline-legacy",
                "exposure_s": 360,
                "energy_residual_pct": "legacy",
                "include_in_paper": "limited",
                "source_path": str(OUT / 'runs/square/single/baseline/prewarp_eval_square_L7mm'),
            },
            {
                "label": "antennae-square",
                "mode": "antennae",
                "geometry": "Square",
                "physics_family": "experimental_pa12_hybrid",
                "exposure_s": 360,
                "energy_residual_pct": "see calibration run",
                "include_in_paper": "limited",
                "source_path": str(OUT / 'runs/unknown/single/experimental/experimental/square_single_20260303_antcal'),
            },
        ]
    )
    return manifest_rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    single_summary = _load_json(SINGLE_SUMMARY_PATH)
    single_rows = _build_single_section(single_summary)
    turntable_rows = _build_turntable_section()
    orientation = _build_orientation_section()
    prewarp = _build_prewarp_section()
    antennae = _build_antennae_section()

    manifest_rows = _build_manifest_rows(single_rows, turntable_rows, orientation, prewarp, antennae)
    _write_manifest(manifest_rows)

    summary = {
        "single_geometries": len(single_rows),
        "turntable_geometries": len(turntable_rows),
        "orientation": orientation,
        "prewarp": prewarp,
        "antennae": antennae,
    }
    (GEN_DIR / "paper_refresh_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[paper-refresh] wrote assets to {GEN_DIR}")
    print(f"[paper-refresh] wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
