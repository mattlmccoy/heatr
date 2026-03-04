#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
GEN_DIR = CONFIG_DIR / "_gui_generated"
OUT_DIR = ROOT / "outputs_eqs" / "_experimental"
AB_DIR = ROOT / "outputs_eqs" / "_experimental_ab"
GEN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
AB_DIR.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    d = yaml.safe_load(path.read_text())
    if not isinstance(d, dict):
        raise ValueError(f"Invalid yaml: {path}")
    return d


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _inject_experimental(cfg: dict[str, Any], stage: str, bucket: str) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg))
    out["physics_model"] = {
        "family": "experimental_pa12_hybrid",
        "experimental_enabled": True,
        "experimental_densification_enabled": True,
        "provenance_tag": f"{bucket}-{stage}",
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
    if stage == "stage_a":
        # Stage A: lock densification to baseline-style behavior while calibrating phase+thermal.
        out["physics_model"]["experimental_densification_enabled"] = False
        out["densification"]["model"] = "arrhenius_phi"
    return out


def _run_case(cfg_path: Path, out_name: str) -> int:
    out_path = OUT_DIR / out_name
    cmd = ["python3", str(ROOT / "rfam_eqs_coupled.py"), "--config", str(cfg_path), "--output-dir", str(out_path)]
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def _run_cmd(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="Run staged experimental benchmark suite.")
    p.add_argument("--stage", choices=["stage_a", "stage_b"], default="stage_b")
    p.add_argument("--bucket", default="", help="A/B bucket id")
    p.add_argument("--skip-compare", action="store_true", help="Skip A/B comparison and confidence dashboard generation.")
    args = p.parse_args()

    bucket = args.bucket.strip() or f"ab-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # Square geometry is the primary benchmark anchor.
    cases = [
        ("rfam_prewarp_square.yaml", "square_static", "prewarp_square/verification", "Square static"),
        ("rfam_prewarp_square_turntable_4x.yaml", "square_turntable_4x", "prewarp_turntable_4x_eval", "Square turntable 4x"),
        ("shape_star_6min.yaml", "shape_star", "shapes/shape_star", "Shape star static"),
    ]
    manifest_cases: list[dict[str, str]] = []
    for cfg_name, tag, baseline_run, case_name in cases:
        base = _load_yaml(CONFIG_DIR / cfg_name)
        exp = _inject_experimental(base, args.stage, bucket)
        out_name = f"{tag}_{args.stage}_{bucket}"
        cfg_path = GEN_DIR / f"exp_{tag}_{args.stage}_{bucket}.yaml"
        _write_yaml(cfg_path, exp)
        rc = _run_case(cfg_path, out_name)
        if rc != 0:
            raise SystemExit(rc)
        manifest_cases.append(
            {
                "name": f"{case_name} ({args.stage})",
                "baseline_run": baseline_run,
                "experimental_run": f"_experimental/{out_name}",
            }
        )

    if not args.skip_compare:
        template = _load_yaml(CONFIG_DIR / "experimental_ab_manifest.yaml")
        thresholds = template.get("thresholds", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        manifest = {"thresholds": thresholds, "cases": manifest_cases}
        manifest_path = GEN_DIR / f"ab_manifest_{args.stage}_{bucket}.yaml"
        _write_yaml(manifest_path, manifest)

        out_dir = AB_DIR / f"{bucket}_{args.stage}"
        rc = _run_cmd(
            [
                "python3",
                str(ROOT / "experimental_ab_compare.py"),
                "--manifest",
                str(manifest_path),
                "--outputs-root",
                "outputs_eqs",
                "--out-dir",
                str(out_dir),
            ]
        )
        if rc != 0:
            raise SystemExit(rc)
        rc = _run_cmd(
            [
                "python3",
                str(ROOT / "physics_confidence_dashboard.py"),
                "--ab-report",
                str(out_dir / "report.json"),
            ]
        )
        if rc != 0:
            raise SystemExit(rc)

    print(f"[benchmark] completed stage={args.stage} bucket={bucket}")


if __name__ == "__main__":
    main()
