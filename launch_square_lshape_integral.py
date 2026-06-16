#!/usr/bin/env python3
"""
launch_square_lshape_integral.py
=================================
Submits integral-mode FGM sweep runs for square and L-shape geometries
to the running HEATR GUI server via its REST API.

Runs submitted:
  Shape           rot   bpp  m     n    Name
  ─────────────────────────────────────────────────────────────────────────
  square           0°    4   0.7   12   square_INTEGRAL_m007_n12
  L_shape          0°    4   0.7   15   lshape_INTEGRAL_m007_n15
  L_shape        140°    4   0.7   15   lshape_INTEGRAL_m007_n15_rot140

Rationale for parameter choices
────────────────────────────────
SQUARE (negative/validation control):
• σ_0≈6.4°C — already near-uniform; all prior proportional runs worsened it.
• Qrf_ratio≈14.8× — falls in the "correctable" regime alongside circle (16.9×).
• Integral mode with fixed iter-0 normalization: corrections proportional to
  residual; small σ_0 → small corrections → should preserve low σ_T.
• n=12 to match circle's successful sweep; early-stop at σ_T<3°C will trigger
  quickly if the part stays uniform.
• Key question: does integral mode preserve σ_T≈6°C where proportional degraded it?

L-SHAPE UNROTATED (0°):
• σ_0≈47°C — largest baseline σ_T of all tested geometries; inner reentrant
  corner receives almost no direct RF energy.
• Qrf_ratio=22.1× — borderline between correctable and t★-confounded.
• phi=0.90 NOT reached at 8 min; barely at 12 min → use_optimizer + 15-min
  ceiling ensures the optimizer finds the phi=0.90 crossing (expected ~12 min).
• n=15: large σ_0 means more iterations needed to accumulate corrections to
  the reentrant corner; early-stop will prevent overshoot.
• Outcome: if integral reduces σ_T to <20°C, it is scientifically interesting;
  <10°C would be outstanding given the geometry's inherent asymmetry.

L-SHAPE ROTATED 140°:
• Orientation optimizer (lshape_orient_20260304_6min) found 140° optimal:
  reduces Qrf_ratio from 22.1× to 20.3×, sigma_T from 47° to 41°C, and
  achieves phi=1.0 (full sintering) at ≤12 min vs phi=0.786 for 0° at 8 min.
• User mentioned "30 degrees" improves uniformity; simulation data shows 140°
  is the empirically optimal orientation (orientation_optimizer result).
  Both 0° and 140° are submitted so the improvement from rotation + FGM
  can be compared to rotation alone and FGM alone.
• 10-min ceiling (vs 15 min for unrotated): 140° rotation enables phi=0.90
  crossing well before 12 min; 10 min gives the optimizer room to find it.
• n=15: same as unrotated; σ_0 is still large (41°C).

All runs:
• magnitude_decay=1.0 (integral mode; no decay maintains consistent step size)
• dead_band=0.05 (±5% of mean; suppresses noise corrections)
• melt_abort_frac=0.50 (raised from 0.15 default — L-shape physically has high
  frac_melt because the reentrant corner drives the mean up; using 0.15 would
  falsely abort. Square also has high frac_melt. 0.50 lets the run proceed.)
• convergence_sigma_T=3.0 (early-stop threshold)
• use_optimizer=True (find t★ = phi=0.90 crossing for each geometry)
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def _base_payload(shape: str, output_name: str, m: float, n: int,
                  bpp: int = 4, exposure_ceiling_min: float = 8.0,
                  rotation_deg: float | None = None,
                  melt_abort_frac: float = 0.50) -> dict:
    """Return the full payload dict for one fgm_iterate run."""
    payload = {
        "mode":                   "fgm_iterate",
        "shape":                  shape,
        "output_name":            output_name,
        # ── Integral mode ────────────────────────────────────────────────────
        "use_delta_correction":   True,
        "magnitude":              m,
        "magnitude_decay":        1.0,           # no decay (pure integral)
        "min_magnitude":          0.0,
        "move_limit":             0.0,
        "sensitivity_filter_sigma": 0.0,
        "fgm_momentum":           0.0,
        # ── Core FGM parameters ───────────────────────────────────────────────
        "bpp":                    bpp,
        "proxy_field":            "T_phi90",
        "invert":                 True,
        "dead_band":              0.05,
        "baseline_saturation":    0.5,
        # ── Convergence / stopping ────────────────────────────────────────────
        "n_iterations":           n,
        "convergence_sigma_T":    3.0,
        "melt_abort_frac":        melt_abort_frac,
        # ── Exposure time ─────────────────────────────────────────────────────
        "use_optimizer":          True,
        "exposure_minutes":       exposure_ceiling_min,
    }

    # Rotation via advanced override (part_rotation_deg key, added to
    # _apply_advanced_overrides in rfam_gui_server.py)
    if rotation_deg is not None and abs(rotation_deg) > 0.01:
        payload["advanced"] = {"part_rotation_deg": rotation_deg}

    return payload


# ─── Runs to submit ──────────────────────────────────────────────────────────
#  (shape, output_name, m, n, bpp, exposure_ceiling_min, rotation_deg, melt_abort_frac)
SWEEP_RUNS = [
    # Square — validation/negative control; expect σ_T to remain near baseline 6°C
    ("square",  "square_INTEGRAL_m007_n12",        0.7, 12, 4,  8.0, None,  0.50),
    # L-shape unrotated — large σ_0; needs 15-min ceiling for phi=0.90 crossing
    ("L_shape", "lshape_INTEGRAL_m007_n15",        0.7, 15, 4, 15.0, None,  0.50),
    # L-shape rotated 140° — orientation-optimized; faster sintering, lower Qrf ratio
    ("L_shape", "lshape_INTEGRAL_m007_n15_rot140", 0.7, 15, 4, 10.0, 140.0, 0.50),
]


def submit_run(base_url: str, run: tuple, dry_run: bool = False) -> str:
    shape, name, m, n, bpp, exp_ceil, rot, melt_ab = run
    payload = _base_payload(shape, name, m, n, bpp, exp_ceil, rot, melt_ab)
    url = f"{base_url}/api/tools/fgm-iterate"

    rot_str = f"rot={rot}°" if rot is not None else "rot=0°"
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Submitting: {name}")
    print(f"  shape={shape}  {rot_str}  m={m}  n={n}  bpp={bpp}  "
          f"exposure_ceiling={exp_ceil}min  melt_abort={melt_ab}")

    if dry_run:
        print(f"  Payload: {json.dumps(payload, indent=4)}")
        return "dry-run"

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"}
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
        description="Submit square + L-shape INTEGRAL runs to HEATR GUI server"
    )
    ap.add_argument("--port",    type=int, default=8080)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print payloads without submitting")
    ap.add_argument("--shapes",  nargs="*",
                    help="Limit to specific shapes (default: all)")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    filter_shapes = set(args.shapes) if args.shapes else None

    print("HEATR Square + L-shape Integral Sweep Launcher")
    print(f"Target: {base_url}")
    print(f"Runs to submit: {len(SWEEP_RUNS)}")
    print("─" * 70)

    results = []
    for run in SWEEP_RUNS:
        shape = run[0]
        if filter_shapes and shape not in filter_shapes:
            continue
        job_id = submit_run(base_url, run, dry_run=args.dry_run)
        results.append((run[1], job_id))
        if not args.dry_run:
            time.sleep(0.5)

    print("\n" + "─" * 70)
    print("Summary:")
    for name, job_id in results:
        print(f"  {name:<45}  job={job_id}")

    print(f"\nMonitor progress at: {base_url}")
    print()
    print("Expected results:")
    print("  square_INTEGRAL:          σ_T should stay near 6°C (negative control)")
    print("  lshape_INTEGRAL (0°):     expect σ_T improvement from ~47°C; "
          "goal <20°C")
    print("  lshape_INTEGRAL (140°):   expect σ_T improvement from ~42°C; "
          "goal <15°C (rotation+FGM synergy)")
    print()
    print("Wall-time estimates:")
    print("  Square: ~15-min iter-0 probe + ~6 min/iter × 12 iter ≈ 1.2 hr")
    print("  L-shape 0°:   ~20-min iter-0 probe + ~8 min/iter × 15 iter ≈ 2.2 hr")
    print("  L-shape 140°: ~15-min iter-0 probe + ~5 min/iter × 15 iter ≈ 1.5 hr")
    print("  Total: ~5 hr wall time")


if __name__ == "__main__":
    main()
