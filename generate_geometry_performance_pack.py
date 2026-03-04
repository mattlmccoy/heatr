#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_eqs"


def _load_json(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    if not isinstance(d, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return d


def _baseline_summary(geometry: str) -> dict[str, Any]:
    if geometry == "H":
        path = OUT / "shape_H" / "summary.json"
    elif geometry == "L":
        path = OUT / "shapes" / "shape_L_shape" / "summary.json"
    elif geometry == "T":
        path = OUT / "shapes" / "shape_T_shape" / "summary.json"
    else:
        path = OUT / "shapes" / f"shape_{geometry}" / "summary.json"
    return _load_json(path)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _recommend_exposure(rows: list[dict[str, Any]], baseline_phi: float, baseline_rho: float) -> tuple[float, float]:
    best_t = rows[0]["exposure_s"]
    best_score = 1e99
    for r in rows:
        phi = r["mean_phi"]
        rho = r["mean_rho"]
        score = abs(phi - baseline_phi) / max(abs(baseline_phi), 1e-6) + abs(rho - baseline_rho) / max(abs(baseline_rho), 1e-6)
        if score < best_score:
            best_score = score
            best_t = r["exposure_s"]
    return best_t, best_score


def main() -> None:
    p = argparse.ArgumentParser(description="Generate per-geometry performance pack from static screen outputs.")
    p.add_argument("--report-dir", required=True, help="Path to static screen report dir")
    p.add_argument("--ignore", default="gt_logo", help="Comma-separated geometry ids to ignore")
    p.add_argument("--nominal-s", type=float, default=360.0)
    p.add_argument("--dsc-mae-max", type=float, default=0.20)
    args = p.parse_args()

    report_dir = Path(args.report_dir).resolve()
    rows_csv = report_dir / "static_geometry_screen_rows.csv"
    ignore = {x.strip() for x in args.ignore.split(",") if x.strip()}

    by_geom: dict[str, list[dict[str, Any]]] = {}
    with rows_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom = str(row["geometry"])
            if geom in ignore:
                continue
            by_geom.setdefault(geom, []).append(
                {
                    "geometry": geom,
                    "exposure_s": _as_float(row["exposure_s"]),
                    "mean_phi": _as_float(row["mean_phi"]),
                    "mean_rho": _as_float(row["mean_rho"]),
                    "dsc_phi_mae": _as_float(row["dsc_phi_mae"]),
                    "temp_rmse_c": _as_float(row["baseline_temp_rmse_c"], float("nan")),
                    "phi_iou_0p5": _as_float(row["baseline_phi_iou_0p5"], float("nan")),
                    "rho_rmse": _as_float(row["baseline_rho_rmse"], float("nan")),
                }
            )

    fig_dir = report_dir / "geometry_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for geom, rows in sorted(by_geom.items()):
        rows.sort(key=lambda x: x["exposure_s"])
        by_t = {int(round(r["exposure_s"])): r for r in rows}
        base = _baseline_summary(geom)
        b_phi = _as_float(base.get("mean_phi_part_final"))
        b_rho = _as_float(base.get("mean_rho_rel_part_final"))
        b_temp = _as_float(base.get("mean_T_part_final_c"))

        nominal = min(rows, key=lambda r: abs(r["exposure_s"] - args.nominal_s))
        rec_t, rec_score = _recommend_exposure(rows, b_phi, b_rho)
        delta_s = rec_t - args.nominal_s
        if delta_s > 1e-6:
            rec = f"+{int(delta_s)}s"
        elif delta_s < -1e-6:
            rec = f"{int(delta_s)}s"
        else:
            rec = "0s (same)"
        dsc_ok = all(r["dsc_phi_mae"] <= args.dsc_mae_max for r in rows)

        summary_rows.append(
            {
                "geometry": geom,
                "baseline_phi": b_phi,
                "baseline_rho": b_rho,
                "baseline_temp_c": b_temp,
                "exp_nominal_phi": nominal["mean_phi"],
                "exp_nominal_rho": nominal["mean_rho"],
                "exp_phi_360": by_t.get(360, {}).get("mean_phi", float("nan")),
                "exp_phi_420": by_t.get(420, {}).get("mean_phi", float("nan")),
                "exp_phi_480": by_t.get(480, {}).get("mean_phi", float("nan")),
                "exp_rho_360": by_t.get(360, {}).get("mean_rho", float("nan")),
                "exp_rho_420": by_t.get(420, {}).get("mean_rho", float("nan")),
                "exp_rho_480": by_t.get(480, {}).get("mean_rho", float("nan")),
                "exp_nominal_temp_rmse_c": nominal["temp_rmse_c"],
                "exp_nominal_phi_iou_0p5": nominal["phi_iou_0p5"],
                "exp_nominal_rho_rmse": nominal["rho_rmse"],
                "dsc_pass_all_exposures": dsc_ok,
                "dsc_mae_max_across_exposure": max(r["dsc_phi_mae"] for r in rows),
                "recommended_exposure_s": rec_t,
                "recommended_delta_s": delta_s,
                "recommendation": rec,
                "recommend_score": rec_score,
            }
        )

        # Geometry figure
        xs = [r["exposure_s"] for r in rows]
        ph = [r["mean_phi"] for r in rows]
        rh = [r["mean_rho"] for r in rows]
        dm = [r["dsc_phi_mae"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(12.8, 3.8), dpi=180)
        ax[0].plot(xs, ph, marker="o", lw=2.0, color="#1d3557", label="Experimental")
        ax[0].axhline(b_phi, ls="--", lw=1.0, color="#2a9d8f", label="Baseline @360s")
        ax[0].set_title(f"{geom}: Mean Melt Fraction")
        ax[0].set_xlabel("Exposure (s)")
        ax[0].set_ylabel("phi")
        ax[0].set_ylim(0.0, 1.05)
        ax[0].set_xlim(355, 485)
        ax[0].set_xticks([360, 420, 480])
        ax[0].legend(fontsize=7)

        ax[1].plot(xs, rh, marker="o", lw=2.0, color="#6d597a", label="Experimental")
        ax[1].axhline(b_rho, ls="--", lw=1.0, color="#e76f51", label="Baseline @360s")
        ax[1].set_title(f"{geom}: Mean Relative Density")
        ax[1].set_xlabel("Exposure (s)")
        ax[1].set_ylabel("rho_rel")
        ax[1].set_ylim(0.55, 1.02)
        ax[1].set_xlim(355, 485)
        ax[1].set_xticks([360, 420, 480])
        ax[1].legend(fontsize=7)

        ax[2].plot(xs, dm, marker="o", lw=2.0, color="#264653", label="DSC MAE")
        ax[2].axhline(args.dsc_mae_max, ls="--", lw=1.0, color="#e63946", label="Gate")
        ax[2].set_title(f"{geom}: DSC Alignment Error")
        ax[2].set_xlabel("Exposure (s)")
        ax[2].set_ylabel("MAE")
        # Adaptive scale for DSC MAE so low-error regimes are still interpretable.
        y_max = max(dm) if dm else 0.0
        if y_max <= 1e-6:
            ax[2].set_ylim(0.0, 1e-3)
        else:
            ax[2].set_ylim(0.0, y_max * 1.25)
        ax[2].set_xlim(355, 485)
        ax[2].set_xticks([360, 420, 480])
        ax[2].legend(fontsize=7)

        fig.suptitle(f"HEATR Geometry Performance: {geom} (recommended exposure: {int(rec_t)}s, {rec})", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{geom}_performance.png", bbox_inches="tight")
        plt.close(fig)

    # Summary CSV + markdown
    csv_out = report_dir / "geometry_performance_summary.csv"
    md_out = report_dir / "geometry_performance_summary.md"
    js_out = report_dir / "geometry_performance_summary.json"

    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    conf_pct = 100.0 * sum(1 for r in summary_rows if r["dsc_pass_all_exposures"]) / max(len(summary_rows), 1)
    data = {
        "geometries_included": len(summary_rows),
        "excluded_geometries": sorted(ignore),
        "dsc_pass_percent": conf_pct,
        "rows": summary_rows,
    }
    js_out.write_text(json.dumps(data, indent=2))

    lines = [
        "# Geometry Performance Summary (DSC vs Baseline)",
        "",
        "- This is sweep-based recommendation from `360/420/480s` runs; it is not the optimizer module.",
        f"- Excluded geometries: `{', '.join(sorted(ignore)) if ignore else '(none)'}`",
        f"- Geometry count: `{len(summary_rows)}`",
        f"- DSC-pass geometries: `{sum(1 for r in summary_rows if r['dsc_pass_all_exposures'])}/{len(summary_rows)}`",
        "",
        "| Geometry | DSC Pass | DSC MAE max | Baseline phi | Exp phi 360/420/480 | Baseline rho | Exp rho 360/420/480 | temp_rmse @360 | phi_iou @360 | rho_rmse @360 | Recommended exposure | Delta |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(summary_rows, key=lambda x: x["geometry"]):
        lines.append(
            "| {geometry} | {dsc_pass_all_exposures} | {dsc_mae_max_across_exposure:.4f} | {baseline_phi:.4f} | {exp_phi_360:.4f}/{exp_phi_420:.4f}/{exp_phi_480:.4f} | {baseline_rho:.4f} | {exp_rho_360:.4f}/{exp_rho_420:.4f}/{exp_rho_480:.4f} | {exp_nominal_temp_rmse_c:.4f} | {exp_nominal_phi_iou_0p5:.4f} | {exp_nominal_rho_rmse:.4f} | {recommended_exposure_s:.0f}s | {recommended_delta_s:+.0f}s |".format(
                **r
            )
        )
    lines.extend(
        [
            "",
            f"- CSV: `{csv_out}`",
            f"- JSON: `{js_out}`",
            f"- Per-geometry figures: `{fig_dir}`",
        ]
    )
    md_out.write_text("\n".join(lines))

    print(f"[pack] wrote {md_out}")
    print(f"[pack] wrote {csv_out}")
    print(f"[pack] wrote {js_out}")
    print(f"[pack] wrote figures in {fig_dir}")


if __name__ == "__main__":
    main()
