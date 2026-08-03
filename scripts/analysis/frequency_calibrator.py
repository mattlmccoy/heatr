#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

EPS0 = 8.8541878128e-12
MU0 = 4e-7 * math.pi


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def _compute(
    frequency_hz: float,
    sigma_s_per_m: float,
    eps_r: float,
    mu_r: float,
    part_depth_m: float | None,
) -> dict[str, float | str | None]:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be > 0")
    if sigma_s_per_m < 0:
        raise ValueError("sigma_s_per_m must be >= 0")
    if eps_r <= 0 or mu_r <= 0:
        raise ValueError("eps_r and mu_r must be > 0")
    if part_depth_m is not None and part_depth_m <= 0:
        raise ValueError("part_depth_m must be > 0")

    omega = 2.0 * math.pi * frequency_hz
    eps = EPS0 * eps_r
    mu = MU0 * mu_r
    tan_delta = sigma_s_per_m / max(omega * eps, 1e-30)
    alpha = omega * math.sqrt((mu * eps) / 2.0) * math.sqrt(max(math.sqrt(1.0 + tan_delta * tan_delta) - 1.0, 0.0))
    penetration_depth_dielectric_m = math.inf if alpha <= 0 else 1.0 / alpha
    skin_depth_conductor_m = (
        math.inf if sigma_s_per_m <= 0 else math.sqrt(2.0 / max(omega * mu * sigma_s_per_m, 1e-30))
    )

    suggested = penetration_depth_dielectric_m
    if not math.isfinite(suggested):
        suggested = skin_depth_conductor_m
    if not math.isfinite(suggested):
        suggested = 0.02
    suggested = max(float(suggested), 1e-4)
    if part_depth_m is not None:
        suggested = min(suggested, part_depth_m)

    regime = "conductive-loss-dominant" if tan_delta >= 1.0 else "dielectric-loss-dominant"
    return {
        "frequency_hz": float(frequency_hz),
        "omega_rad_per_s": float(omega),
        "sigma_s_per_m": float(sigma_s_per_m),
        "eps_r": float(eps_r),
        "mu_r": float(mu_r),
        "tan_delta": float(tan_delta),
        "regime": regime,
        "penetration_depth_dielectric_m": float(penetration_depth_dielectric_m),
        "skin_depth_conductor_m": float(skin_depth_conductor_m),
        "part_depth_cap_m": None if part_depth_m is None else float(part_depth_m),
        "suggested_effective_depth_m": float(suggested),
    }


def _set_nested(cfg: dict[str, Any], key_path: list[str], value: Any) -> None:
    cur: dict[str, Any] = cfg
    for k in key_path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[key_path[-1]] = value


def main() -> None:
    p = argparse.ArgumentParser(description="Compute frequency-dependent calibration values and generate YAML overrides.")
    p.add_argument("--frequency-hz", type=float, required=True)
    p.add_argument("--sigma-s-per-m", type=float, required=True, help="Doped-region conductivity used for calibration.")
    p.add_argument("--eps-r", type=float, required=True, help="Doped-region relative permittivity.")
    p.add_argument("--mu-r", type=float, default=1.0, help="Relative permeability.")
    p.add_argument("--part-depth-m", type=float, default=None, help="Optional depth cap for suggested effective_depth_m.")
    p.add_argument("--base-config", type=str, default="", help="Optional base YAML config to merge into.")
    p.add_argument("--output", type=str, default="", help="Output YAML path. If omitted, prints to stdout.")
    p.add_argument("--snippet-only", action="store_true", help="Emit only override snippet (no full config merge).")
    p.add_argument("--json-summary", action="store_true", help="Print JSON calibration summary to stderr.")
    args = p.parse_args()

    cal = _compute(
        frequency_hz=float(args.frequency_hz),
        sigma_s_per_m=float(args.sigma_s_per_m),
        eps_r=float(args.eps_r),
        mu_r=float(args.mu_r),
        part_depth_m=float(args.part_depth_m) if args.part_depth_m is not None else None,
    )

    snippet: dict[str, Any] = {
        "electric": {
            "frequency_hz": cal["frequency_hz"],
            "effective_depth_m": cal["suggested_effective_depth_m"],
        },
        "materials": {
            "doped": {
                "sigma_s_per_m": cal["sigma_s_per_m"],
                "eps_r": cal["eps_r"],
            }
        },
        "frequency_calibration": {
            **cal,
            "generated_at_iso": datetime.now().isoformat(timespec="seconds"),
        },
    }

    out_cfg: dict[str, Any]
    if args.snippet_only or not args.base_config:
        out_cfg = snippet
    else:
        base_path = Path(args.base_config).resolve()
        out_cfg = _load_yaml(base_path)
        _set_nested(out_cfg, ["electric", "frequency_hz"], cal["frequency_hz"])
        _set_nested(out_cfg, ["electric", "effective_depth_m"], cal["suggested_effective_depth_m"])
        _set_nested(out_cfg, ["materials", "doped", "sigma_s_per_m"], cal["sigma_s_per_m"])
        _set_nested(out_cfg, ["materials", "doped", "eps_r"], cal["eps_r"])
        out_cfg["frequency_calibration"] = {
            **cal,
            "generated_at_iso": datetime.now().isoformat(timespec="seconds"),
            "base_config": str(base_path),
        }

    raw = _dump_yaml(out_cfg)
    if args.output:
        op = Path(args.output).resolve()
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(raw)
        print(f"[frequency-calibrator] wrote {op}")
    else:
        print(raw, end="")

    if args.json_summary:
        print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()

