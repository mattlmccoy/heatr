#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
OUT_ROOT = ROOT / "outputs_eqs"
REPORT_ROOT = OUT_ROOT / "_experimental_ab"

THRESHOLDS = {
    "temp_rmse_c_max": 15.0,
    "phi_iou_0p5_min": 0.70,
    "rho_rmse_max": 0.10,
    "exp_phi_dsc_mae_max": 0.20,
}


@dataclass
class RunRow:
    config_name: str
    geometry: str
    exposure_s: float
    run_dir: str
    mean_t_c: float
    mean_phi: float
    mean_rho: float
    dsc_phi_mae: float
    baseline_temp_rmse_c: float
    baseline_phi_iou_0p5: float
    baseline_rho_rmse: float
    baseline_available: bool


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return obj


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in d.files}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def _phi_iou(phi_a: np.ndarray, phi_b: np.ndarray, thr: float = 0.5) -> float:
    a = np.asarray(phi_a, dtype=float) >= thr
    b = np.asarray(phi_b, dtype=float) >= thr
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return 1.0
    return float(inter / union)


def _phi_from_dsc(T: np.ndarray, onset_c: float, end_c: float) -> np.ndarray:
    width = max(float(end_c) - float(onset_c), 1e-9)
    return np.clip((np.asarray(T, dtype=float) - float(onset_c)) / width, 0.0, 1.0)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _inject_experimental(cfg: dict[str, Any], bucket: str) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg))
    out["physics_model"] = {
        "family": "experimental_pa12_hybrid",
        "experimental_enabled": True,
        "experimental_densification_enabled": True,
        "provenance_tag": bucket,
        "parameter_source": "literature+calibrated",
        "calibration_version": "exp-pa12-v1",
        "ab_bucket_id": bucket,
        "provenance_file": "configs/experimental_pa12_provenance.yaml",
        "dsc_profile_file": "configs/experimental_pa12_dsc_profile.yaml",
        "literature_refs": ["JA2022", "JA2022_VOL", "NAKAMURA_CLASS"],
    }
    out["phase_model"] = {
        "type": "apparent_heat_capacity_dsc",
        "use_dsc_profile": True,
        "cp_smoothing_strategy": "linear",
    }
    out["dens_model"] = {
        "type": "viscous_capillary_pa12",
        "viscosity_model": "arrhenius",
        "viscosity_params": {
            "eta_ref_pa_s": 8000.0,
            "eta_ref_temp_k": 458.15,
            "activation_energy_j_per_mol": 60000.0,
            "wlf_c1": 8.86,
            "wlf_c2_k": 101.6,
        },
        "surface_tension_n_per_m": 0.03,
        "particle_radius_m": 3.5e-5,
        "phi_exponent": 1.0,
        "phi_threshold": 0.02,
        "porosity_coupling_params": {"porosity_exponent": 1.0},
        "geom_factor": 0.05,
    }
    out["crystallization_model"] = {
        "enabled": False,
        "type": "nakamura",
        "params": {
            "k0_per_s": 0.05,
            "ea_j_per_mol": 42000.0,
            "exponent": 2.0,
            "liquid_suppression": 0.25,
        },
    }
    return out


def _baseline_dir_for_config(config_name: str) -> Path:
    stem = Path(config_name).stem  # e.g., shape_hexagon_6min
    base = stem.replace("_6min", "")
    if base == "shape_H":
        return OUT_ROOT / "shape_H"
    return OUT_ROOT / "shapes" / base


def _run(cfg_path: Path, out_dir: Path) -> int:
    cmd = ["python3", str(ROOT / "rfam_eqs_coupled.py"), "--config", str(cfg_path), "--output-dir", str(out_dir)]
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def _clean_output_dir(out_dir: Path) -> None:
    keep = {"summary.json", "time_series.json", "used_config.yaml"}
    for p in out_dir.iterdir():
        if p.name in keep:
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def _nondecreasing(xs: list[float], tol: float = 1e-6) -> bool:
    return all((b + tol) >= a for a, b in zip(xs, xs[1:]))


def main() -> None:
    p = argparse.ArgumentParser(description="Static geometry screen (baseline vs experimental + exposure sweep).")
    p.add_argument("--exposures-s", default="360,420,480", help="Comma-separated exposure times in seconds.")
    p.add_argument("--bucket", default="", help="Report bucket id")
    p.add_argument("--geometry", default="", help="Optional geometry filter (e.g., square).")
    p.add_argument("--limit", type=int, default=0, help="Optional geometry limit for quick tests.")
    args = p.parse_args()

    exposures = [float(x.strip()) for x in args.exposures_s.split(",") if x.strip()]
    exposures = sorted(set(exposures))
    if not exposures:
        raise ValueError("No exposures provided")
    nominal_exposure = exposures[0]
    bucket = args.bucket.strip() or f"screen-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    cfg_paths = sorted(CONFIG_DIR.glob("shape_*_6min.yaml"))
    geom_filter = args.geometry.strip().lower()
    if geom_filter:
        cfg_paths = [p for p in cfg_paths if p.stem.replace("shape_", "").replace("_6min", "").lower() == geom_filter]
    if args.limit > 0:
        cfg_paths = cfg_paths[: args.limit]

    scratch_root = Path("/tmp") / "heatr_static_screen" / bucket
    scratch_root.mkdir(parents=True, exist_ok=True)
    report_dir = REPORT_ROOT / bucket
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunRow] = []
    skipped: list[str] = []

    for cfg_path in cfg_paths:
        config_name = cfg_path.name
        baseline_dir = _baseline_dir_for_config(config_name)
        baseline_fields_path = baseline_dir / "fields.npz"
        baseline_has_fields = baseline_fields_path.exists()
        if not baseline_has_fields:
            skipped.append(f"{config_name}: missing baseline fields at {baseline_fields_path}")
            continue
        baseline_fields = _load_npz(baseline_fields_path)

        base_cfg = _load_yaml(cfg_path)
        geom = Path(config_name).stem.replace("shape_", "").replace("_6min", "")

        for exposure_s in exposures:
            cfg = _inject_experimental(base_cfg, bucket=bucket)
            thermal = cfg.get("thermal", {})
            if not isinstance(thermal, dict):
                thermal = {}
                cfg["thermal"] = thermal
            dt_s = float(thermal.get("dt_s", 0.5))
            thermal["n_steps"] = int(round(exposure_s / max(dt_s, 1e-9)))

            run_id = f"{Path(config_name).stem}_exp_{int(exposure_s)}s"
            out_dir = scratch_root / run_id
            gen_cfg = scratch_root / "configs" / f"{run_id}.yaml"
            _write_yaml(gen_cfg, cfg)
            rc = _run(gen_cfg, out_dir)
            if rc != 0:
                raise SystemExit(rc)

            summary = _load_json(out_dir / "summary.json")
            exp_fields = _load_npz(out_dir / "fields.npz")

            emask = np.asarray(exp_fields["part_mask"], dtype=bool)
            ephi = np.asarray(exp_fields["phi"], dtype=float)[emask]
            eT = np.asarray(exp_fields["T"], dtype=float)[emask]
            onset = float(summary.get("dsc_profile_melt_onset_c", summary.get("phase_transition_lo_c", 171.0)))
            end = float(summary.get("dsc_profile_melt_end_c", summary.get("phase_transition_hi_c", 186.0)))
            phi_dsc = _phi_from_dsc(eT, onset, end)
            dsc_phi_mae = _mae(ephi, phi_dsc)

            temp_rmse = float("nan")
            phi_iou = float("nan")
            rho_rmse = float("nan")
            if abs(exposure_s - nominal_exposure) < 1e-6:
                bmask = np.asarray(baseline_fields["part_mask"], dtype=bool)
                cmask = np.logical_or(bmask, emask)
                bT = np.asarray(baseline_fields["T"], dtype=float)[cmask]
                eT_u = np.asarray(exp_fields["T"], dtype=float)[cmask]
                bphi = np.asarray(baseline_fields["phi"], dtype=float)[cmask]
                ephi_u = np.asarray(exp_fields["phi"], dtype=float)[cmask]
                brho = np.asarray(baseline_fields["rho_rel"], dtype=float)[cmask]
                erho = np.asarray(exp_fields["rho_rel"], dtype=float)[cmask]
                temp_rmse = _rmse(eT_u, bT)
                phi_iou = _phi_iou(ephi_u, bphi, 0.5)
                rho_rmse = _rmse(erho, brho)

            rows.append(
                RunRow(
                    config_name=config_name,
                    geometry=geom,
                    exposure_s=exposure_s,
                    run_dir=str(out_dir),
                    mean_t_c=float(summary.get("mean_T_part_final_c", float("nan"))),
                    mean_phi=float(summary.get("mean_phi_part_final", float("nan"))),
                    mean_rho=float(summary.get("mean_rho_rel_part_final", float("nan"))),
                    dsc_phi_mae=dsc_phi_mae,
                    baseline_temp_rmse_c=temp_rmse,
                    baseline_phi_iou_0p5=phi_iou,
                    baseline_rho_rmse=rho_rmse,
                    baseline_available=baseline_has_fields,
                )
            )

            _clean_output_dir(out_dir)

    # Write row-level CSV.
    csv_path = report_dir / "static_geometry_screen_rows.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config_name",
                "geometry",
                "exposure_s",
                "run_dir",
                "mean_t_c",
                "mean_phi",
                "mean_rho",
                "dsc_phi_mae",
                "baseline_temp_rmse_c",
                "baseline_phi_iou_0p5",
                "baseline_rho_rmse",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.config_name,
                    r.geometry,
                    r.exposure_s,
                    r.run_dir,
                    r.mean_t_c,
                    r.mean_phi,
                    r.mean_rho,
                    r.dsc_phi_mae,
                    r.baseline_temp_rmse_c,
                    r.baseline_phi_iou_0p5,
                    r.baseline_rho_rmse,
                ]
            )

    # Per-geometry gates and confidence.
    per_geom: dict[str, list[RunRow]] = {}
    for r in rows:
        per_geom.setdefault(r.geometry, []).append(r)

    geom_results: list[dict[str, Any]] = []
    for geom, rs in sorted(per_geom.items()):
        rs = sorted(rs, key=lambda x: x.exposure_s)
        dsc_pass = all(r.dsc_phi_mae <= THRESHOLDS["exp_phi_dsc_mae_max"] for r in rs)
        phi_monotonic = _nondecreasing([r.mean_phi for r in rs])
        rho_monotonic = _nondecreasing([r.mean_rho for r in rs])
        monotonic_pass = phi_monotonic and rho_monotonic

        nominal = rs[0]
        baseline_pass = (
            nominal.baseline_temp_rmse_c <= THRESHOLDS["temp_rmse_c_max"]
            and nominal.baseline_phi_iou_0p5 >= THRESHOLDS["phi_iou_0p5_min"]
            and nominal.baseline_rho_rmse <= THRESHOLDS["rho_rmse_max"]
        )
        if not dsc_pass:
            status = "FAIL_DSC"
            conf = 0.0
        elif baseline_pass and monotonic_pass:
            status = "PASS_READY"
            conf = 1.0
        else:
            status = "PASS_RECAL_REQUIRED"
            conf = 0.75 + (0.15 if baseline_pass else 0.0) + (0.10 if monotonic_pass else 0.0)

        geom_results.append(
            {
                "geometry": geom,
                "status": status,
                "confidence": round(conf, 4),
                "dsc_pass_all_exposures": dsc_pass,
                "baseline_pass_nominal": baseline_pass,
                "phi_monotonic": phi_monotonic,
                "rho_monotonic": rho_monotonic,
                "nominal_temp_rmse_c": round(nominal.baseline_temp_rmse_c, 4),
                "nominal_phi_iou_0p5": round(nominal.baseline_phi_iou_0p5, 4),
                "nominal_rho_rmse": round(nominal.baseline_rho_rmse, 4),
            }
        )

    conf_pct = (100.0 * float(np.mean([g["confidence"] for g in geom_results]))) if geom_results else 0.0
    dsc_pass_n = sum(1 for g in geom_results if g["dsc_pass_all_exposures"])
    ready_n = sum(1 for g in geom_results if g["status"] == "PASS_READY")

    report = {
        "bucket": bucket,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "thresholds": THRESHOLDS,
        "exposures_s": exposures,
        "summary": {
            "geometries_evaluated": len(geom_results),
            "geometries_skipped": len(skipped),
            "dsc_pass_count": dsc_pass_n,
            "pass_ready_count": ready_n,
            "switch_confidence_percent": round(conf_pct, 2),
        },
        "geometry_results": geom_results,
        "skipped": skipped,
        "row_csv": str(csv_path),
        "scratch_root": str(scratch_root),
    }

    json_path = report_dir / "static_geometry_screen_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_path = report_dir / "static_geometry_screen_report.md"
    lines = [
        "# Static Geometry Screen Report (No Turntable)",
        "",
        f"- Bucket: `{bucket}`",
        f"- Exposures (s): `{', '.join(str(int(x)) for x in exposures)}`",
        f"- Geometries evaluated: `{len(geom_results)}`",
        f"- Geometries skipped: `{len(skipped)}`",
        f"- DSC pass count: `{dsc_pass_n}/{len(geom_results) if geom_results else 0}`",
        f"- PASS_READY count: `{ready_n}/{len(geom_results) if geom_results else 0}`",
        f"- Switch confidence: `{conf_pct:.2f}%`",
        "",
        "## Gates",
        "",
        "- `FAIL_DSC`: any exposure fails DSC melt-fraction alignment.",
        "- `PASS_READY`: DSC passes all exposures and nominal baseline metrics + monotonic sweep pass.",
        "- `PASS_RECAL_REQUIRED`: DSC passes but baseline or sweep monotonic gate needs recalibration.",
        "",
        "## Geometry Results",
        "",
        "| Geometry | Status | Confidence | DSC Pass | Baseline Pass (nominal) | Phi Monotonic | Rho Monotonic | temp_rmse | phi_iou | rho_rmse |",
        "|---|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for g in geom_results:
        lines.append(
            "| {geometry} | {status} | {confidence:.3f} | {dsc_pass_all_exposures} | {baseline_pass_nominal} | {phi_monotonic} | {rho_monotonic} | {nominal_temp_rmse_c:.4f} | {nominal_phi_iou_0p5:.4f} | {nominal_rho_rmse:.4f} |".format(
                **g
            )
        )
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for s in skipped:
            lines.append(f"- {s}")
    lines.append("")
    lines.append(f"- Row data: `{csv_path}`")
    lines.append(f"- JSON report: `{json_path}`")
    lines.append(f"- Scratch runs (trimmed): `{scratch_root}`")
    md_path.write_text("\n".join(lines))

    print(f"[screen] report: {json_path}")
    print(f"[screen] report: {md_path}")
    print(f"[screen] rows: {csv_path}")
    print(f"[screen] switch confidence: {conf_pct:.2f}%")


if __name__ == "__main__":
    main()
