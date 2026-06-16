#!/usr/bin/env python3
"""
launch_integral_sweep.py
========================
Submits FGM integral-mode sweep runs for hexagon and diamond (and circle
variants) to the running HEATR GUI server via its REST API.

Usage:
    python launch_integral_sweep.py [--port 8080] [--dry-run]

Runs submitted:
  Geometry    bpp  m     n    Name
  ─────────────────────────────────────────────────────────────────────────
  hexagon      4   0.7   12   hexagon_INTEGRAL_m007_n12
  hexagon      4   0.5   15   hexagon_INTEGRAL_m005_n15   (bracketing)
  hexagon      4   0.8   12   hexagon_INTEGRAL_m008_n12   (bracketing)
  diamond      4   0.7   15   diamond_INTEGRAL_m007_n15
  diamond      4   0.5   15   diamond_INTEGRAL_m005_n15   (bracketing)
  diamond      4   0.8   15   diamond_INTEGRAL_m008_n15   (bracketing)
  circle       2   0.7   12   circle_INTEGRAL_m007_2bpp    (2bpp validation)

Rationale for parameter choices
────────────────────────────────
• m=0.7 is the empirical optimum for circle 4bpp (σ_0≈19°C).  We submit
  bracketing runs at m=0.5 and m=0.8 for each new geometry since the
  optimal m may shift with different σ_0 and EM coupling geometry.
• n=12 for hexagon (similar σ_0 to circle); n=15 for diamond (larger σ_0,
  may need more iterations to reach the quantization snap).
• magnitude_decay=1.0 (no decay) — integral mode's fixed normalization
  reference provides implicit amplitude reduction as residuals shrink.
• bpp=4 for all primary runs (16 levels enable the quantization snap events).
• use_optimizer=True so all runs share a consistent exposure time derived
  from iter-0 (φ=0.90 crossing).

All runs stop early if σ_T rises for 2 consecutive iterations.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

# ─── Geometry sizes (mm), matching existing HEATR defaults ─────────────────
SHAPE_SIZES = {
    "hexagon": {"size_mm": 30},
    "diamond": {"size_mm": 30},
    "circle":  {"size_mm": 30},
}

def _base_payload(shape: str, output_name: str, m: float, n: int, bpp: int = 4,
                  size_mm: float = 30.0) -> dict:
    """Return the full payload dict for one fgm_iterate run."""
    return {
        "mode":                   "fgm_iterate",
        "shape":                  shape,
        "size_mm":                size_mm,
        "output_name":            output_name,
        # ── Integral mode (empirically optimal) ──────────────────────────────
        "use_delta_correction":   True,          # integral accumulation
        "magnitude":              m,             # per-iteration step size
        "magnitude_decay":        1.0,           # no decay (integral mode)
        "min_magnitude":          0.0,           # unused with decay=1.0
        "move_limit":             0.0,           # pure delta, no OC-TO bounds
        "sensitivity_filter_sigma": 0.0,         # off for pure integral
        "fgm_momentum":           0.0,           # not applicable
        # ── Core FGM parameters ───────────────────────────────────────────────
        "bpp":                    bpp,
        "proxy_field":            "T_phi90",     # max-contrast sintering snapshot
        "invert":                 True,          # hot → low saturation
        "dead_band":              0.05,          # ±5% of mean = no-correction zone
        "baseline_saturation":    0.5,           # mid-scale default
        # ── Convergence / stopping ────────────────────────────────────────────
        "n_iterations":           n,
        "convergence_sigma_T":    3.0,           # hard stop (early-stop by rising σ_T hits first)
        "melt_abort_frac":        0.15,          # abort if iter-0 over-melts
        # ── Exposure time ─────────────────────────────────────────────────────
        "use_optimizer":          True,          # optimizer probe → ideal t* for all iters
        "exposure_minutes":       8.0,           # probe ceiling; ideal ~5–7 min for 30mm
    }

# ─── Runs to submit ──────────────────────────────────────────────────────────
SWEEP_RUNS = [
    # Primary hexagon runs
    ("hexagon", "hexagon_INTEGRAL_m007_n12", 0.7, 12, 4),
    ("hexagon", "hexagon_INTEGRAL_m005_n15", 0.5, 15, 4),
    ("hexagon", "hexagon_INTEGRAL_m008_n12", 0.8, 12, 4),
    # Primary diamond runs
    ("diamond", "diamond_INTEGRAL_m007_n15", 0.7, 15, 4),
    ("diamond", "diamond_INTEGRAL_m005_n15", 0.5, 15, 4),
    ("diamond", "diamond_INTEGRAL_m008_n15", 0.8, 15, 4),
    # 2bpp validation (circle only — already have 4bpp, want bpp comparison for integral)
    ("circle",  "circle_INTEGRAL_m007_2bpp", 0.7, 12, 2),
]


def submit_run(base_url: str, run: tuple, dry_run: bool = False) -> str:
    shape, name, m, n, bpp = run
    size_mm = SHAPE_SIZES[shape]["size_mm"]
    payload = _base_payload(shape, name, m, n, bpp, size_mm)
    url = f"{base_url}/api/tools/fgm-iterate"

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Submitting: {name}")
    print(f"  shape={shape}  m={m}  n={n}  bpp={bpp}  size={size_mm}mm")

    if dry_run:
        print(f"  Payload: {json.dumps(payload, indent=4)}")
        return "dry-run"

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
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
        print(f"  ✗ HTTP {e.code}: {msg[:200]}", file=sys.stderr)
        return f"error-{e.code}"
    except Exception as ex:
        print(f"  ✗ {ex}", file=sys.stderr)
        return f"error"


def main():
    ap = argparse.ArgumentParser(description="Submit INTEGRAL sweep runs to HEATR GUI server")
    ap.add_argument("--port",    type=int, default=8080)
    ap.add_argument("--host",    default="localhost")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print payloads without submitting")
    ap.add_argument("--shapes",  nargs="*",
                    help="Limit to specific shapes (default: all)")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    filter_shapes = set(args.shapes) if args.shapes else None

    print(f"HEATR Integral Sweep Launcher")
    print(f"Target: {base_url}")
    print(f"Runs to submit: {len(SWEEP_RUNS)}")
    print("─" * 60)

    results = []
    for run in SWEEP_RUNS:
        shape = run[0]
        if filter_shapes and shape not in filter_shapes:
            continue
        job_id = submit_run(base_url, run, dry_run=args.dry_run)
        results.append((run[1], job_id))
        if not args.dry_run:
            time.sleep(0.5)   # small gap between submissions

    print("\n─" * 60)
    print("Summary:")
    for name, job_id in results:
        print(f"  {name:<45}  job={job_id}")

    print(f"\nMonitor progress at: {base_url}")
    print("Each run takes ~4–7 min (iter-0 optimizer probe) + ~3–5 min per FGM iteration.")
    print("Estimated total wall time per run: 45–80 minutes.")
    print("Expected key result: hexagon_INTEGRAL_m007 should achieve σ_T < 5°C.")


if __name__ == "__main__":
    main()
