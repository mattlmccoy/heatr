#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON structure: {path}")
    return data


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _status_for_case(case: dict[str, Any]) -> tuple[str, bool, bool, float]:
    checks = case.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    metrics = case.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    primary_dsc = bool(checks.get("phi_dsc_alignment_ok", False))
    secondary_keys = [k for k in checks.keys() if k != "phi_dsc_alignment_ok"]
    secondary_total = len(secondary_keys)
    secondary_passed = sum(1 for k in secondary_keys if bool(checks.get(k, False)))
    secondary_ratio = (secondary_passed / secondary_total) if secondary_total > 0 else 1.0

    if not primary_dsc:
        status = "FAIL_DSC"
        confidence = 0.0
    elif secondary_ratio >= 1.0:
        status = "PASS_DSC_AND_PROCESS"
        confidence = 1.0
    else:
        status = "PASS_DSC_RECAL_REQUIRED"
        confidence = 0.7 + 0.3 * secondary_ratio

    # Slight confidence penalty for larger melt-fraction mismatch inside pass band.
    phi_dsc_mae = _as_float(metrics.get("exp_phi_dsc_mae"), 0.0)
    confidence = max(0.0, min(1.0, confidence - 0.1 * phi_dsc_mae))
    return status, primary_dsc, secondary_ratio >= 1.0, confidence


def main() -> None:
    p = argparse.ArgumentParser(description="Build a DSC-first physics confidence dashboard from an A/B report.")
    p.add_argument("--ab-report", required=True, help="Path to experimental_ab_compare report.json")
    args = p.parse_args()

    report_path = Path(args.ab_report).resolve()
    report = _load_json(report_path)
    cases = report.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Expected 'cases' list in A/B report")

    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        status, primary_pass, secondary_all_pass, confidence = _status_for_case(case)
        metrics = case.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "name": str(case.get("name", "")),
                "status": status,
                "primary_dsc_pass": primary_pass,
                "secondary_all_pass": secondary_all_pass,
                "confidence_score": round(confidence, 4),
                "temp_rmse_c": _as_float(metrics.get("temp_rmse_c")),
                "phi_iou_0p5": _as_float(metrics.get("phi_iou_0p5")),
                "rho_rmse": _as_float(metrics.get("rho_rmse")),
                "delta_mean_rho_final": _as_float(metrics.get("delta_mean_rho_final")),
                "exp_phi_dsc_mae": _as_float(metrics.get("exp_phi_dsc_mae")),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: (r["primary_dsc_pass"], r["confidence_score"]), reverse=True)
    dsc_pass_count = sum(1 for r in rows if r["primary_dsc_pass"])
    full_pass_count = sum(1 for r in rows if r["status"] == "PASS_DSC_AND_PROCESS")

    dashboard = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_ab_report": str(report_path),
        "summary": {
            "total_cases": len(rows),
            "primary_dsc_pass_cases": dsc_pass_count,
            "full_process_pass_cases": full_pass_count,
            "dsc_first_decision_rule": "FAIL only when DSC-alignment fails; otherwise pass-with-recalibration if needed.",
        },
        "cases_ranked": rows_sorted,
    }

    out_dir = report_path.parent
    json_path = out_dir / "physics_confidence_dashboard.json"
    md_path = out_dir / "physics_confidence_dashboard.md"
    json_path.write_text(json.dumps(dashboard, indent=2))

    lines = [
        "# Physics Confidence Dashboard (DSC-first)",
        "",
        f"- Generated: {dashboard['generated_at']}",
        f"- Source A/B report: `{report_path}`",
        f"- Cases: {len(rows)}",
        f"- Primary DSC pass: {dsc_pass_count}/{len(rows)}",
        f"- Full process pass: {full_pass_count}/{len(rows)}",
        "",
        "## Decision Rule",
        "",
        "- Failure is defined by mismatch with DSC-grounded melting physics (`FAIL_DSC`).",
        "- If DSC passes but process metrics drift, status is `PASS_DSC_RECAL_REQUIRED`.",
        "",
        "## Ranked Cases",
        "",
        "| Case | Status | Confidence | temp_rmse_c | phi_iou_0p5 | rho_rmse | delta_mean_rho_final | exp_phi_dsc_mae |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted:
        lines.append(
            "| {name} | {status} | {confidence_score:.3f} | {temp_rmse_c:.4f} | {phi_iou_0p5:.4f} | {rho_rmse:.4f} | {delta_mean_rho_final:.4f} | {exp_phi_dsc_mae:.4f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `PASS_DSC_RECAL_REQUIRED` means the thermal phase behavior is physically plausible, but process setpoints likely need retuning.")
    lines.append("- This dashboard does not auto-promote any model.")
    md_path.write_text("\n".join(lines))

    print(f"[dashboard] wrote {json_path}")
    print(f"[dashboard] wrote {md_path}")


if __name__ == "__main__":
    main()
