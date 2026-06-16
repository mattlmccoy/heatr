#!/usr/bin/env python3
"""
launch_square_lshape_integral_v2.py
=====================================
Corrected resubmission of square and L-shape integral FGM runs.

Fixes vs v1:
  1. ROTATION: uses base_config="configs/shape_L_shape_30deg.yaml" for the
     rotated L-shape run — this bakes rotation_deg=30.0 directly into the
     config template so it cannot be overwritten at runtime and requires no
     server restart.
  2. MAGNITUDE: m=0.7 caused T_max=398°C abort on L-shape. Root cause:
     iter0_ref range [145°C–350°C] → cold inner corner goes from 0.5 to
     0.85 saturation at m=0.7 → 70% more energy at same exposure → runaway.
     Fix: scale m proportional to baseline σ_T relative to circle:
       m_lshape = 0.7 × (σ_circle / σ_lshape) = 0.7 × (19.3/65.8) ≈ 0.20
     At m=0.20 the cold corner reaches 0.60 saturation (20% increase) — safe.
  3. SQUARE EXPOSURE: optimizer found t★=7 min where ρ̄=0.74 (active regime
     only). Server convergence note: "increase exposure to reach ρ̄≥0.82."
     Fix: ceiling → 12 min so optimizer finds t★ in near-final regime.

Runs submitted:
  Shape            rot    bpp  m     n    Exp ceiling   Name
  ────────────────────────────────────────────────────────────────────────
  square            0°     4   0.7   12   12 min        square_INTEGRAL_m007_n12_v2
  L_shape           0°     4   0.20  15   15 min        lshape_INTEGRAL_m020_n15
  L_shape          30°     4   0.20  15   12 min        lshape_INTEGRAL_m020_n15_rot30
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def _lshape_payload(output_name: str, m: float, n: int,
                    exposure_ceiling_min: float,
                    base_config: str | None = None) -> dict:
    """Payload for an L-shape fgm_iterate run."""
    p = {
        "mode":                   "fgm_iterate",
        "shape":                  "L_shape",
        "output_name":            output_name,
        # ── Integral mode ──────────────────────────────────────────────────
        "use_delta_correction":   True,
        "magnitude":              m,
        "magnitude_decay":        1.0,
        "min_magnitude":          0.0,
        "move_limit":             0.0,
        "sensitivity_filter_sigma": 0.0,
        "fgm_momentum":           0.0,
        # ── Core FGM ───────────────────────────────────────────────────────
        "bpp":                    4,
        "proxy_field":            "T_phi90",
        "invert":                 True,
        "dead_band":              0.05,
        "baseline_saturation":    0.5,
        # ── Convergence ────────────────────────────────────────────────────
        "n_iterations":           n,
        "convergence_sigma_T":    3.0,
        "melt_abort_frac":        0.50,
        # ── Exposure ───────────────────────────────────────────────────────
        "use_optimizer":          True,
        "exposure_minutes":       exposure_ceiling_min,
    }
    if base_config:
        p["base_config"] = base_config
    return p


def _square_payload(output_name: str, m: float, n: int,
                    exposure_ceiling_min: float) -> dict:
    """Payload for a square fgm_iterate run."""
    return {
        "mode":                   "fgm_iterate",
        "shape":                  "square",
        "output_name":            output_name,
        "use_delta_correction":   True,
        "magnitude":              m,
        "magnitude_decay":        1.0,
        "min_magnitude":          0.0,
        "move_limit":             0.0,
        "sensitivity_filter_sigma": 0.0,
        "fgm_momentum":           0.0,
        "bpp":                    4,
        "proxy_field":            "T_phi90",
        "invert":                 True,
        "dead_band":              0.05,
        "baseline_saturation":    0.5,
        "n_iterations":           n,
        "convergence_sigma_T":    3.0,
        "melt_abort_frac":        0.50,
        "use_optimizer":          True,
        "exposure_minutes":       exposure_ceiling_min,
    }


# ─── Runs to submit ──────────────────────────────────────────────────────────
# Each entry: (builder_fn, kwargs, label)
def _build_runs():
    return [
        # Square — negative/validation control; 12-min ceiling to reach near-final regime
        (
            _square_payload,
            dict(output_name="square_INTEGRAL_m007_n12_v2", m=0.7, n=12,
                 exposure_ceiling_min=12.0),
            "square  rot=0°  m=0.7  12-min ceiling",
        ),
        # L-shape unrotated — m scaled to prevent T_max abort
        (
            _lshape_payload,
            dict(output_name="lshape_INTEGRAL_m020_n15", m=0.20, n=15,
                 exposure_ceiling_min=15.0, base_config=None),
            "L_shape  rot=0°  m=0.20  15-min ceiling",
        ),
        # L-shape 30° rotation — uses custom base_config with rotation_deg=30 baked in
        (
            _lshape_payload,
            dict(output_name="lshape_INTEGRAL_m020_n15_rot30", m=0.20, n=15,
                 exposure_ceiling_min=12.0,
                 base_config="shape_L_shape_30deg.yaml"),
            "L_shape  rot=30°  m=0.20  12-min ceiling",
        ),
    ]


def submit_run(base_url: str, builder_fn, kwargs: dict, label: str,
               dry_run: bool = False) -> str:
    payload = builder_fn(**kwargs)
    url = f"{base_url}/api/tools/fgm-iterate"

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Submitting: {kwargs['output_name']}")
    print(f"  {label}")

    if dry_run:
        print(f"  Payload: {json.dumps(payload, indent=4)}")
        return "dry-run"

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            job_id = body.get("job_id", "?")
            print(f"  ✓ Queued  job_id={job_id}  status={body.get('status')}  "
                  f"queue_position={body.get('queue_position')}")
            return job_id
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        print(f"  ✗ HTTP {e.code}: {msg[:400]}", file=sys.stderr)
        return f"error-{e.code}"
    except Exception as ex:
        print(f"  ✗ {ex}", file=sys.stderr)
        return "error"


def main():
    ap = argparse.ArgumentParser(
        description="Submit corrected square + L-shape INTEGRAL runs to HEATR server"
    )
    ap.add_argument("--port",    type=int, default=8080)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    runs = _build_runs()

    print("HEATR Square + L-shape Integral Sweep — v2 (corrected)")
    print(f"Target: {base_url}")
    print(f"Runs: {len(runs)}")
    print("─" * 70)

    results = []
    for builder_fn, kwargs, label in runs:
        job_id = submit_run(base_url, builder_fn, kwargs, label, dry_run=args.dry_run)
        results.append((kwargs["output_name"], job_id))
        if not args.dry_run:
            time.sleep(0.5)

    print("\n" + "─" * 70)
    print("Summary:")
    for name, job_id in results:
        print(f"  {name:<48}  job={job_id}")

    print(f"\nMonitor: {base_url}")
    print()
    print("Fixes applied vs v1:")
    print("  [1] L-shape rotation: base_config bakes rotation_deg=30 in at template level")
    print("  [2] L-shape magnitude: 0.7→0.20 (scaled to σ_0=65°C; max correction +10% sat)")
    print("  [3] Square exposure ceiling: 7→12 min (let optimizer find ρ̄≥0.82 t★)")
    print()
    print("Expected outcomes:")
    print("  square_INTEGRAL_v2:       σ_T near 5.9°C baseline; key test = does")
    print("                            longer exposure allow improvement or confirm")
    print("                            square is self-regularizing?")
    print("  lshape_INTEGRAL_0°:       σ_T improvement from 65.8°C; no T_max abort;")
    print("                            Qrf_ratio=22× → borderline correctable")
    print("  lshape_INTEGRAL_30°:      σ_T improvement from ~55°C (30° baseline);")
    print("                            rotation reduces Qrf_ratio; combined rotation+FGM")


if __name__ == "__main__":
    main()
