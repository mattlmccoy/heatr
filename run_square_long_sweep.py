#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
OUT_ROOT = ROOT / "outputs_eqs" / "_experimental_ab"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return data


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


def _set_exposure(cfg: dict[str, Any], exposure_s: float) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg))
    thermal = out.get("thermal", {})
    if not isinstance(thermal, dict):
        thermal = {}
        out["thermal"] = thermal
    dt_s = float(thermal.get("dt_s", 0.5))
    thermal["n_steps"] = int(round(float(exposure_s) / max(dt_s, 1e-9)))
    return out


def _run(cfg_path: Path, out_dir: Path) -> int:
    cmd = ["python3", str(ROOT / "rfam_eqs_coupled.py"), "--config", str(cfg_path), "--output-dir", str(out_dir)]
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def _plot(rows: list[dict[str, Any]], out_png: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["model"], float(r["exposure_s"])))
    exps = sorted({float(r["exposure_s"]) for r in rows})

    def _series(model: str, key: str) -> list[float]:
        m = {float(r["exposure_s"]): float(r[key]) for r in rows if r["model"] == model}
        return [m.get(x, float("nan")) for x in exps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=180)
    for model, color in [("baseline", "#1d3557"), ("experimental", "#e76f51")]:
        ax1.plot(exps, _series(model, "mean_phi"), marker="o", lw=2.0, color=color, label=model)
        ax2.plot(exps, _series(model, "mean_rho"), marker="o", lw=2.0, color=color, label=model)
    ax1.set_title("Square Long Sweep: Mean Melt Fraction")
    ax2.set_title("Square Long Sweep: Mean Relative Density")
    for ax in (ax1, ax2):
        ax.set_xlabel("Exposure (s)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    ax1.set_ylabel("Mean phi")
    ax2.set_ylabel("Mean rho_rel")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run square-only long exposure sweep (baseline vs experimental).")
    p.add_argument("--exposures-s", default="360,480,600,720")
    p.add_argument("--bucket", default="")
    args = p.parse_args()

    exposures = sorted({float(x.strip()) for x in args.exposures_s.split(",") if x.strip()})
    if not exposures:
        raise ValueError("No exposures provided")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bucket = args.bucket.strip() or f"square-long-{stamp}"
    out_root = OUT_ROOT / bucket
    cfg_root = out_root / "configs"
    run_root = out_root / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(CONFIG_DIR / "rfam_prewarp_square.yaml")
    exp_cfg_base = _inject_experimental(base_cfg, bucket=bucket)

    rows: list[dict[str, Any]] = []
    for exposure_s in exposures:
        for model in ("baseline", "experimental"):
            cfg_model = base_cfg if model == "baseline" else exp_cfg_base
            cfg = _set_exposure(cfg_model, exposure_s)
            run_tag = f"{model}_{int(exposure_s)}s"
            cfg_path = cfg_root / f"{run_tag}.yaml"
            out_dir = run_root / run_tag
            _write_yaml(cfg_path, cfg)
            rc = _run(cfg_path, out_dir)
            if rc != 0:
                raise SystemExit(rc)
            s = _load_json(out_dir / "summary.json")
            rows.append(
                {
                    "model": model,
                    "exposure_s": exposure_s,
                    "run_dir": str(out_dir),
                    "mean_t_c": float(s.get("mean_T_part_final_c", float("nan"))),
                    "mean_phi": float(s.get("mean_phi_part_final", float("nan"))),
                    "mean_rho": float(s.get("mean_rho_rel_part_final", float("nan"))),
                }
            )

    csv_path = out_root / "square_long_sweep_rows.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "exposure_s", "run_dir", "mean_t_c", "mean_phi", "mean_rho"])
        for r in rows:
            w.writerow([r["model"], r["exposure_s"], r["run_dir"], r["mean_t_c"], r["mean_phi"], r["mean_rho"]])

    fig_path = out_root / "square_long_sweep_metrics.png"
    _plot(rows, fig_path)

    report = {
        "bucket": bucket,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "exposures_s": exposures,
        "rows_csv": str(csv_path),
        "figure": str(fig_path),
        "run_root": str(run_root),
    }
    (out_root / "square_long_sweep_report.json").write_text(json.dumps(report, indent=2))
    print(f"[square-long-sweep] rows: {csv_path}")
    print(f"[square-long-sweep] figure: {fig_path}")
    print(f"[square-long-sweep] run root: {run_root}")


if __name__ == "__main__":
    main()
