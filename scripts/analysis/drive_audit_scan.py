#!/usr/bin/env python3
"""Scan every HEATR run under outputs_eqs and record its drive settings.

Measurement-only script (no new physics).  For each run directory containing a
``used_config.yaml`` it records the drive mode, seed/actual voltage, and the
absorbed power, then writes a CSV.

Absorbed power:
  * enforced-power runs: analytic, ``P_gen * eta`` (rfam_eqs_coupled.py:2291),
    cross-checked against ``summary.json:integrated_power_doped_W_per_m * depth``.
  * voltage-driven runs: read from ``summary.json:integrated_power_doped_W_per_m``
    (which is exactly sum(Qrf[doped])*dA) times ``effective_depth_m``.

Usage:
    python3 scripts/analysis/drive_audit_scan.py <root> <out_csv>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.drive_audit import enforced_absorbed_power_w, implied_coupling_pct

FIELDS = [
    "run_dir",
    "family",
    "shape",
    "drive_mode",
    "voltage_v",
    "freq_hz",
    "generator_power_w",
    "transfer_eff",
    "effective_depth_m",
    "chamber_y_m",
    "part_w_m",
    "part_h_m",
    "absorbed_w",
    "absorbed_w_analytic",
    "coupling_pct_at_500w",
    "gap_field_kv_per_m",
    "energy_resid_pct",
    "clip_frac",
]


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def scan_run(cfg_path: Path, root: Path) -> dict | None:
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
    except (yaml.YAMLError, OSError):
        return None
    if not isinstance(cfg, dict) or "electric" not in cfg:
        return None
    elec = cfg.get("electric") or {}
    geom = cfg.get("geometry") or {}
    part = geom.get("part") or {}

    run_dir = cfg_path.parent
    enforced = bool(elec.get("enforce_generator_power", False))
    gen_w = _f(elec.get("generator_power_w"), 0.0)
    eff = _f(elec.get("generator_transfer_efficiency", elec.get("transfer_efficiency")), 0.0)
    depth = _f(elec.get("effective_depth_m"), 1.0)
    volt = _f(elec.get("voltage_v"))

    absorbed = None
    resid_pct = None
    clip = None
    s_path = run_dir / "summary.json"
    if s_path.exists():
        try:
            s = json.loads(s_path.read_text())
            wpm = _f(s.get("integrated_power_doped_W_per_m"))
            if wpm is not None and depth:
                absorbed = wpm * depth
            e_in = _f(s.get("energy_doped_total_J_per_m"))
            e_res = _f(s.get("energy_balance_residual_final_J_per_m"))
            if e_in and e_res is not None and abs(e_in) > 1e-9:
                resid_pct = 100.0 * e_res / e_in
            clip = _f(s.get("frac_cells_dT_clipped_final"))
        except (json.JSONDecodeError, OSError):
            pass

    analytic = enforced_absorbed_power_w(gen_w, eff) if enforced else None
    cy = _f(geom.get("chamber_y"))
    gap_kv = (volt / cy / 1000.0) if (volt and cy) else None

    rel = str(run_dir.relative_to(root))
    return {
        "run_dir": rel,
        "family": rel.split("/")[0],
        "shape": part.get("shape"),
        "drive_mode": "enforced_power" if enforced else "voltage",
        "voltage_v": volt,
        "freq_hz": _f(elec.get("frequency_hz")),
        "generator_power_w": gen_w,
        "transfer_eff": eff,
        "effective_depth_m": depth,
        "chamber_y_m": cy,
        "part_w_m": _f(part.get("width")),
        "part_h_m": _f(part.get("height")),
        "absorbed_w": absorbed,
        "absorbed_w_analytic": analytic,
        "coupling_pct_at_500w": (implied_coupling_pct(absorbed, 500.0) if absorbed else None),
        "gap_field_kv_per_m": gap_kv,
        "energy_resid_pct": resid_pct,
        "clip_frac": clip,
    }


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2])
    rows = []
    for cfg_path in sorted(root.rglob("used_config.yaml")):
        r = scan_run(cfg_path, root)
        if r is not None:
            rows.append(r)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"scanned {len(rows)} runs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
