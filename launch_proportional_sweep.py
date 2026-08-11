#!/usr/bin/env python3
"""
launch_proportional_sweep.py
============================
Submits the constant-magnitude PROPORTIONAL FGM iteration sweep (circle) to the
running HEATR GUI server via its REST API.  Companion to launch_integral_sweep.py
and the stored circle_INTEGRAL_m0* runs.

Purpose: settle the fixed-point vs period-2 question for the proportional update
(fig_fgm_optimizer_characterization panels b/d/e).  The stored circle magnitude
sweeps are INTEGRAL mode only (use_delta_correction=true, iter0_ref bounds); the
constant-magnitude proportional sweep was never run.

Drive match (read from circle_INTEGRAL_m03_n30/*_iter0/used_config.yaml and
convergence.json, NOT guessed):
  20 mm circle, grid 120, 27.12 MHz, enforced generator power 500 W at 2%
  transfer efficiency (voltage_mode grounded, 860 V nominal), bpp=4,
  proxy T_phi90, dead_band 0.02, baseline_saturation 0.5, invert=True,
  use_optimizer=True with an 8.0 min iter-0 probe (integral sweeps all chose
  t* = 6.308 min from the identical iter-0; deterministic, so this reproduces
  the same exposure).

Proportional mode = use_delta_correction=False, magnitude_decay=1.0 (constant m),
momentum=0, move_limit=0, sensitivity_filter_sigma=0: the map is recomputed from
the current iterate's T_phi90 field every iteration with per-iteration adaptive
normalization (no iter0_ref anchoring; verified in fgm_generator.generate_fgm —
prior_sat_npz is ignored when use_delta_correction=False and momentum=0).

regression_threshold_multiplier=100 so all 30 iterations run (academic sweep;
the code comment recommends raising it for exhaustive runs).

Usage:
    python3 launch_proportional_sweep.py [--port 8080] [--dry-run]

Output dirs (per-iteration checkpointing by the server loop):
    outputs_eqs/runs/circle/fgm_iterate/circle_PROPORTIONAL_m0{3,5,07,08,09}_n30/
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

N_ITER = 30  # including iter-0 baseline → 29 proportional iterates


def _payload(output_name: str, m: float) -> dict:
    return {
        "mode":                   "fgm_iterate",
        "shape":                  "circle",           # default 20 mm (matches integral sweeps)
        "output_name":            output_name,
        # ── Proportional mode: constant magnitude, no accumulation ─────────
        "use_delta_correction":   False,
        "magnitude":              m,
        "magnitude_decay":        1.0,                # constant m every iteration
        "min_magnitude":          0.0,
        "move_limit":             0.0,
        "sensitivity_filter_sigma": 0.0,
        "fgm_momentum":           0.0,
        "use_hybrid":             False,
        # ── Core FGM parameters (match circle_INTEGRAL_m0* exactly) ────────
        "bpp":                    4,
        "proxy_field":            "T_phi90",
        "invert":                 True,
        "dead_band":              0.02,
        "baseline_saturation":    0.5,
        "thorough":               False,
        "regime_adaptive":        False,
        "use_ebc":                False,
        "perturbation_amplitude": 0.0,
        # ── Run length / stopping ───────────────────────────────────────────
        "n_iterations":           N_ITER,
        "convergence_sigma_T":    3.0,
        "melt_abort_frac":        0.15,
        "regression_threshold_multiplier": 100.0,     # run all iterations
        # ── Exposure (same optimizer probe as the integral sweeps) ─────────
        "use_optimizer":          True,
        "exposure_minutes":       8.0,
    }


SWEEP = [
    ("circle_PROPORTIONAL_m03_n30", 0.3),
    ("circle_PROPORTIONAL_m05_n30", 0.5),
    ("circle_PROPORTIONAL_m007_n30", 0.7),
    ("circle_PROPORTIONAL_m008_n30", 0.8),
    ("circle_PROPORTIONAL_m009_n30", 0.9),
]


def submit(base_url: str, name: str, m: float, dry_run: bool) -> str:
    payload = _payload(name, m)
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Submitting: {name}  (m={m}, n={N_ITER})")
    if dry_run:
        print(json.dumps(payload, indent=2))
        return "dry-run"
    req = urllib.request.Request(
        f"{base_url}/api/tools/fgm-iterate",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            print(f"  queued job_id={body.get('job_id')}  status={body.get('status')}"
                  f"  queue_position={body.get('queue_position')}")
            return str(body.get("job_id"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode(errors='replace')[:200]}", file=sys.stderr)
        return f"error-{e.code}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    base_url = f"http://{args.host}:{args.port}"
    results = []
    for name, m in SWEEP:
        results.append((name, submit(base_url, name, m, args.dry_run)))
        if not args.dry_run:
            time.sleep(0.5)
    print("\nSummary:")
    for name, jid in results:
        print(f"  {name:<35} job={jid}")


if __name__ == "__main__":
    main()
