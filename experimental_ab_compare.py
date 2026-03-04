#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import re

import numpy as np
import yaml

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


@dataclass
class CaseResult:
    name: str
    baseline_run: str
    experimental_run: str
    metrics: dict[str, float]
    checks: dict[str, bool]
    passed: bool
    notes: list[str]
    plot_path: str


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in d.files}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


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


def _series(arr: dict[str, Any], key: str) -> np.ndarray:
    vals = arr.get(key, [])
    if not isinstance(vals, list):
        return np.array([], dtype=float)
    return np.asarray([float(v) for v in vals], dtype=float)


def _make_overlay_plot(
    case_name: str,
    ts_base: dict[str, Any],
    ts_exp: dict[str, Any],
    out_path: Path,
) -> None:
    t0 = _series(ts_base, "time_s")
    t1 = _series(ts_exp, "time_s")
    t = t0 if len(t0) > 0 else t1
    if len(t) == 0:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), dpi=170)
    ax[0].plot(t0, _series(ts_base, "mean_T_part_c"), label="baseline", lw=1.8)
    ax[0].plot(t1, _series(ts_exp, "mean_T_part_c"), label="experimental", lw=1.8)
    ax[0].set_title("Mean T (part)")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("C")
    ax[0].legend(fontsize=8)

    ax[1].plot(t0, _series(ts_base, "mean_phi_part"), label="baseline", lw=1.8)
    ax[1].plot(t1, _series(ts_exp, "mean_phi_part"), label="experimental", lw=1.8)
    ax[1].set_title("Mean melt fraction")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("phi")

    ax[2].plot(t0, _series(ts_base, "mean_rho_rel_part"), label="baseline", lw=1.8)
    ax[2].plot(t1, _series(ts_exp, "mean_rho_rel_part"), label="experimental", lw=1.8)
    ax[2].set_title("Mean relative density")
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("rho_rel")

    fig.suptitle(f"A/B Overlay: {case_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _compute_case(
    outputs_root: Path,
    case: dict[str, Any],
    thresholds: dict[str, float],
    out_plot_dir: Path,
) -> CaseResult:
    name = str(case["name"])
    baseline_run = str(case["baseline_run"])
    experimental_run = str(case["experimental_run"])

    bdir = outputs_root / baseline_run
    edir = outputs_root / experimental_run
    bfields = _load_npz(bdir / "fields.npz")
    efields = _load_npz(edir / "fields.npz")

    bmask = np.asarray(bfields["part_mask"], dtype=bool)
    emask = np.asarray(efields["part_mask"], dtype=bool)
    cmask = np.logical_or(bmask, emask)

    bT = np.asarray(bfields["T"], dtype=float)[cmask]
    eT = np.asarray(efields["T"], dtype=float)[cmask]
    bphi = np.asarray(bfields["phi"], dtype=float)[cmask]
    ephi = np.asarray(efields["phi"], dtype=float)[cmask]
    brho = np.asarray(bfields["rho_rel"], dtype=float)[cmask]
    erho = np.asarray(efields["rho_rel"], dtype=float)[cmask]

    bsum = _load_json(bdir / "summary.json")
    esum = _load_json(edir / "summary.json")
    bts = _load_json(bdir / "time_series.json")
    ets = _load_json(edir / "time_series.json")

    metrics = {
        "temp_rmse_c": _rmse(eT, bT),
        "phi_mae": _mae(ephi, bphi),
        "phi_iou_0p5": _phi_iou(ephi, bphi, 0.5),
        "rho_rmse": _rmse(erho, brho),
        "delta_mean_rho_final": float(esum.get("mean_rho_rel_part_final", np.nan))
        - float(bsum.get("mean_rho_rel_part_final", np.nan)),
        "delta_mean_phi_final": float(esum.get("mean_phi_part_final", np.nan))
        - float(bsum.get("mean_phi_part_final", np.nan)),
        "exp_frac_cells_dT_clipped_mean": float(ets.get("frac_cells_dT_clipped", [0.0])[-1])
        if isinstance(ets.get("frac_cells_dT_clipped"), list)
        else float(esum.get("frac_cells_dT_clipped_mean", 0.0)),
    }

    onset = float(esum.get("dsc_profile_melt_onset_c", esum.get("phase_transition_lo_c", 171.0)))
    end = float(esum.get("dsc_profile_melt_end_c", esum.get("phase_transition_hi_c", 186.0)))
    phi_dsc_exp = _phi_from_dsc(np.asarray(efields["T"], dtype=float)[cmask], onset, end)
    metrics["exp_phi_mean_model"] = float(np.mean(ephi))
    metrics["exp_phi_mean_dsc_implied"] = float(np.mean(phi_dsc_exp))
    metrics["exp_phi_dsc_mae"] = _mae(ephi, phi_dsc_exp)

    checks = {
        "temp_rmse_ok": metrics["temp_rmse_c"] <= float(thresholds.get("temp_rmse_c_max", 15.0)),
        "phi_iou_ok": metrics["phi_iou_0p5"] >= float(thresholds.get("phi_iou_0p5_min", 0.70)),
        "rho_rmse_ok": metrics["rho_rmse"] <= float(thresholds.get("rho_rmse_max", 0.10)),
        "density_shift_ok": abs(metrics["delta_mean_rho_final"]) <= float(thresholds.get("delta_mean_rho_final_abs_max", 0.08)),
        "phi_dsc_alignment_ok": metrics["exp_phi_dsc_mae"] <= float(thresholds.get("exp_phi_dsc_mae_max", 0.20)),
    }
    passed = all(checks.values())

    notes: list[str] = []
    if not checks["temp_rmse_ok"]:
        notes.append("Temperature field RMSE exceeds threshold.")
    if not checks["phi_iou_ok"]:
        notes.append("Melt-front overlap (IoU) below threshold.")
    if not checks["rho_rmse_ok"]:
        notes.append("Relative density RMSE exceeds threshold.")
    if not checks["density_shift_ok"]:
        notes.append("Mean final relative density shift too large.")
    if not checks["phi_dsc_alignment_ok"]:
        notes.append("Experimental melt fraction is inconsistent with DSC-implied melt window.")

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or "case"
    plot_name = f"{safe}_overlay.png"
    plot_path = out_plot_dir / plot_name
    _make_overlay_plot(name, bts, ets, plot_path)

    return CaseResult(
        name=name,
        baseline_run=baseline_run,
        experimental_run=experimental_run,
        metrics=metrics,
        checks=checks,
        passed=passed,
        notes=notes,
        plot_path=plot_name,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="A/B compare baseline vs experimental PA12 model runs.")
    p.add_argument("--manifest", required=True, help="YAML file with cases and thresholds")
    p.add_argument("--outputs-root", default="outputs_eqs", help="Outputs root directory")
    p.add_argument("--out-dir", default="", help="Destination for report artifacts")
    args = p.parse_args()

    manifest = _load_yaml(Path(args.manifest))
    thresholds = manifest.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must include non-empty cases list.")

    outputs_root = Path(args.outputs_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (outputs_root / "_experimental_ab" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        results.append(_compute_case(outputs_root, case, thresholds, plot_dir))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(Path(args.manifest).resolve()),
        "outputs_root": str(outputs_root),
        "summary": {
            "total_cases": len(results),
            "passed_cases": int(sum(1 for r in results if r.passed)),
            "failed_cases": int(sum(1 for r in results if not r.passed)),
        },
        "thresholds": thresholds,
        "cases": [
            {
                "name": r.name,
                "baseline_run": r.baseline_run,
                "experimental_run": r.experimental_run,
                "passed": r.passed,
                "metrics": r.metrics,
                "checks": r.checks,
                "notes": r.notes,
                "plot": f"plots/{r.plot_path}",
            }
            for r in results
        ],
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "baseline_run",
                "experimental_run",
                "passed",
                "temp_rmse_c",
                "phi_mae",
                "phi_iou_0p5",
                "rho_rmse",
                "delta_mean_rho_final",
                "delta_mean_phi_final",
                "exp_phi_mean_model",
                "exp_phi_mean_dsc_implied",
                "exp_phi_dsc_mae",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.name,
                    r.baseline_run,
                    r.experimental_run,
                    r.passed,
                    r.metrics["temp_rmse_c"],
                    r.metrics["phi_mae"],
                    r.metrics["phi_iou_0p5"],
                    r.metrics["rho_rmse"],
                    r.metrics["delta_mean_rho_final"],
                    r.metrics["delta_mean_phi_final"],
                    r.metrics["exp_phi_mean_model"],
                    r.metrics["exp_phi_mean_dsc_implied"],
                    r.metrics["exp_phi_dsc_mae"],
                ]
            )

    md_lines = [
        "# Experimental A/B Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Cases: {report['summary']['total_cases']}",
        f"- Passed: {report['summary']['passed_cases']}",
        f"- Failed: {report['summary']['failed_cases']}",
        "",
        "## Case Results",
    ]
    for r in results:
        md_lines.extend(
            [
                "",
                f"### {r.name}",
                f"- Baseline: `{r.baseline_run}`",
                f"- Experimental: `{r.experimental_run}`",
                f"- Passed: `{r.passed}`",
                f"- temp_rmse_c: `{r.metrics['temp_rmse_c']:.4f}`",
                f"- phi_iou_0p5: `{r.metrics['phi_iou_0p5']:.4f}`",
                f"- rho_rmse: `{r.metrics['rho_rmse']:.4f}`",
                f"- exp_phi_mean_model: `{r.metrics['exp_phi_mean_model']:.4f}`",
                f"- exp_phi_mean_dsc_implied: `{r.metrics['exp_phi_mean_dsc_implied']:.4f}`",
                f"- exp_phi_dsc_mae: `{r.metrics['exp_phi_dsc_mae']:.4f}`",
                f"- Plot: `plots/{r.plot_path}`",
            ]
        )
        if r.notes:
            md_lines.append("- Notes:")
            for note in r.notes:
                md_lines.append(f"  - {note}")

    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n")
    (out_dir / "physics_review_checklist.md").write_text(
        "\n".join(
            [
                "# Physics Review Checklist",
                "",
                "- [ ] Hotspot movement remains physically consistent under rotation.",
                "- [ ] Melt-front continuity is smooth and free of non-physical jumps.",
                "- [ ] Relative density remains bounded in [0,1] and monotonic where expected.",
                "- [ ] No unexplained clipping/convergence instability.",
                "- [ ] Experimental improvements are consistent across the benchmark matrix.",
            ]
        )
        + "\n"
    )
    print(f"[A/B] report written to: {out_dir}")


if __name__ == "__main__":
    main()
