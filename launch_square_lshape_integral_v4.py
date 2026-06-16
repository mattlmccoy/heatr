#!/usr/bin/env python3
"""
launch_square_lshape_integral_v4.py
=====================================
Corrected L-shape integral FGM runs — fixes vs v3:

  1. ROTATION: 30° → 140°.  140° is the empirically optimal orientation from
     the orientation sweep (lshape_orient_20260304_6min), NOT 30°.  At 140°:
     • σ_T baseline: ~41°C (vs ~47°C at 0°)
     • Qrf_ratio: 20.3× (vs 22.1× at 0°) — more correctable
     • Achieves phi=1.0 (full melt) at ≤12 min vs phi=0.786 at 8 min for 0°
     New base_config: shape_L_shape_140deg.yaml (rotation_deg: 140.0 baked in)

  2. ITER-0 T_MAX ABORT: Server now uses HARD_DECOMP_T_C=350°C as the abort
     threshold for manual-mode iter-0 checks, not the soft DAMAGE_T_C=245°C.
     L-shape at 7.5 min has T_max≈284°C — this is the melt/densification
     regime, NOT decomposition.  The 245°C check was blocking FGM from ever
     running on geometries whose hot corners are naturally in the melt regime.
     A warning is logged but the run proceeds.

     Requires server restart for the change to take effect.

Runs submitted (L-shape only — square is a confirmed negative control):
  Shape            rot    bpp   m     n    Exp (fixed)   Name
  ─────────────────────────────────────────────────────────────────────────
  L_shape           0°     4   0.20   15   7.5 min       lshape_INTEGRAL_m020_n15_v4
  L_shape         140°     4   0.20   15   7.0 min       lshape_INTEGRAL_m020_n15_rot140
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def _lshape_payload(output_name: str, m: float, n: int,
                    exposure_min: float,
                    base_config: str | None = None) -> dict:
    """Payload for an L-shape integral FGM run with fixed exposure time."""
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
        # ── Fixed exposure — bypasses phi=0.90 optimizer criterion ─────────
        "use_optimizer":          False,
        "exposure_minutes":       exposure_min,
    }
    if base_config:
        p["base_config"] = base_config
    return p


def _build_runs():
    return [
        # L-shape unrotated — 7.5 min fixed
        (
            _lshape_payload,
            dict(output_name="lshape_INTEGRAL_m020_n15_v4",
                 m=0.20, n=15,
                 exposure_min=7.5,
                 base_config=None),
            "L_shape  rot=0°   m=0.20  7.5 min fixed",
        ),
        # L-shape 140° rotation (empirically optimal) — 7.0 min fixed
        (
            _lshape_payload,
            dict(output_name="lshape_INTEGRAL_m020_n15_rot140",
                 m=0.20, n=15,
                 exposure_min=7.0,
                 base_config="shape_L_shape_140deg.yaml"),
            "L_shape  rot=140°  m=0.20  7.0 min fixed",
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
        description="Submit corrected L-shape INTEGRAL runs (v4) to HEATR server"
    )
    ap.add_argument("--port",    type=int, default=8080)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    runs = _build_runs()

    print("HEATR L-shape Integral Sweep — v4 (140° rotation + iter-0 abort fix)")
    print(f"Target: {base_url}")
    print(f"Runs:   {len(runs)}")
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
        print(f"  {name:<52}  job={job_id}")

    print(f"\nMonitor: {base_url}")
    print()
    print("Fixes applied vs v3:")
    print("  [1] Rotation corrected: 30° → 140° (empirically optimal from orientation sweep)")
    print("      Config: shape_L_shape_140deg.yaml (rotation_deg: 140.0 baked in)")
    print("  [2] Iter-0 abort threshold: DAMAGE_T_C=245°C → HARD_DECOMP_T_C=350°C")
    print("      PA12 at 280-300°C is in the melt/densification regime — not decomposing.")
    print("      FGM is exactly the tool to correct hot spots in this regime.")
    print("      (Server must be restarted for this fix to take effect.)")
    print()
    print("Expected outcomes:")
    print("  lshape_INTEGRAL_v4 (0°):   σ_T ~66°C baseline → target <20°C over 15 iters")
    print("  lshape_INTEGRAL_rot140:    σ_T ~41°C baseline → target <15°C (rotation+FGM)")


if __name__ == "__main__":
    main()
