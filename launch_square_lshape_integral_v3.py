#!/usr/bin/env python3
"""
launch_square_lshape_integral_v3.py
=====================================
Corrected L-shape integral FGM runs — fixes vs v2:

  1. EXPOSURE: L-shape exposure time now fixed at 7.5 min (unrotated) and
     7.0 min (rotated 30°).  The phi=0.90 optimizer criterion was causing
     ~12-min exposure because the reentrant inner corner is poorly coupled
     and drags down mean phi.  Using use_optimizer=False bypasses this and
     lets the FGM corrections (not longer time) fix the corner.

  2. FGM DISPLAY (not a simulation parameter, but a visual fix applied to
     fgm_generator.py before this run):  PNG previews now flip the image
     vertically so physical top appears at the top of the image.  The
     fields.npz arrays (sat_map, T) store row 0 = y=-30 mm = physical
     bottom; PIL without flipud puts that at the top → images appeared
     upside-down for asymmetric geometries like the rotated L-shape.

  3. SQUARE: confirmed negative control in v2 — sigma_T at T_phi90 already
     4.93 °C (only 2 °C above the 3 °C convergence threshold); FGM
     consistently worsens it to ~9.4 °C.  Not resubmitted here; the v2
     square result is the final paper data point for the square geometry.

Runs submitted:
  Shape            rot    bpp   m     n    Exp (fixed)   Name
  ─────────────────────────────────────────────────────────────────────────
  L_shape           0°     4   0.20   15   7.5 min       lshape_INTEGRAL_m020_n15_v3
  L_shape          30°     4   0.20   15   7.0 min       lshape_INTEGRAL_m020_n15_rot30_v3

Parameter rationale (unchanged from v2 except exposure):
  m = 0.20: scaled from circle optimal (m=0.7) by σ_circle/σ_lshape = 19.3/65.8 ≈ 0.20
  n = 15:   large σ_0 (~66 °C unrot, ~55 °C 30°) needs more iterations to accumulate
            corrections into the reentrant corner; early-stop at σ_T < 3 °C prevents
            overshoot
  bpp = 4:  16-level gradient for fine spatial control
  Adaptive damage ceiling: max(245, iter0_T_max × 1.15) — already in server code;
            prevents spurious abort if T_max naturally exceeds 245 °C (as it does for
            the L-shape even at 7.5 min)
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
        # ── Exposure: FIXED time, no optimizer ─────────────────────────────
        # The phi=0.90 optimizer criterion pushes L-shape to ~12 min because
        # the inner reentrant corner is poorly coupled and drags down mean phi.
        # FGM corrections — not longer exposure — are the right fix for the corner.
        "use_optimizer":          False,
        "exposure_minutes":       exposure_min,
    }
    if base_config:
        p["base_config"] = base_config
    return p


# ─── Runs to submit ──────────────────────────────────────────────────────────
def _build_runs():
    return [
        # L-shape unrotated — 7.5 min, no optimizer
        (
            _lshape_payload,
            dict(output_name="lshape_INTEGRAL_m020_n15_v3",
                 m=0.20, n=15,
                 exposure_min=7.5,
                 base_config=None),
            "L_shape  rot=0°  m=0.20  7.5 min fixed",
        ),
        # L-shape 30° rotation — 7.0 min, no optimizer
        (
            _lshape_payload,
            dict(output_name="lshape_INTEGRAL_m020_n15_rot30_v3",
                 m=0.20, n=15,
                 exposure_min=7.0,
                 base_config="shape_L_shape_30deg.yaml"),
            "L_shape  rot=30°  m=0.20  7.0 min fixed",
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
        description="Submit corrected L-shape INTEGRAL runs (v3) to HEATR server"
    )
    ap.add_argument("--port",    type=int, default=8080)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    runs = _build_runs()

    print("HEATR L-shape Integral Sweep — v3 (fixed exposure)")
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
    print("Fixes applied vs v2:")
    print("  [1] Exposure: use_optimizer=False; fixed 7.5 min (unrot) / 7.0 min (30°)")
    print("      Root cause: phi=0.90 criterion drove optimizer to ~12 min because the")
    print("      inner reentrant corner never fully sinters — FGM not more time is the fix.")
    print("  [2] FGM display: np.flipud() added to PNG saves in fgm_generator.py so")
    print("      physical top appears at image top (row 0 = y=-30mm = physical bottom).")
    print()
    print("Expected outcomes:")
    print("  lshape_INTEGRAL_v3 (0°):   iter-0 T_max ~250-280°C (not ~400°C); FGM applies")
    print("                              incremental corrections to inner corner; σ_T should")
    print("                              improve from ~66°C baseline toward <20°C.")
    print("  lshape_INTEGRAL_v3 (30°):  σ_0 already reduced by rotation (~55°C); combined")
    print("                              rotation + FGM expected to reach <15°C.")
    print()
    print("Square result (v2, final):")
    print("  square_INTEGRAL_v2:   σ_T 4.93°C → 9.4°C at iter-1 (worsened).")
    print("  Conclusion: square is a NEGATIVE CONTROL — FGM counterproductive for")
    print("  geometries already near-uniform (σ_0 only 2°C above convergence threshold).")


if __name__ == "__main__":
    main()
