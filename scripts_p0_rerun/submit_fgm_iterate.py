#!/usr/bin/env python3
"""Submit one INTEGRAL-mode fgm_iterate job to the HEATR GUI server and print job_id.

P0 re-run campaign helper. All physics knobs match the task spec:
INTEGRAL (use_delta_correction, move_limit=0, sensitivity_filter_sigma=0),
bpp=4, proxy=T_phi90, dead_band=0.05, baseline=0.5, n_iterations=12,
use_optimizer=true (Step A t* probe), use_ebc=true (EBC t* rescale, clamp 0.5-2.5).
"""
import argparse
import json
import sys
import urllib.request
import urllib.error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--output-name", required=True)
    ap.add_argument("--magnitude", type=float, required=True)
    ap.add_argument("--exposure-minutes", type=float, default=10.0,
                    help="optimizer probe ceiling (max) time")
    ap.add_argument("--n-iterations", type=int, default=12)
    ap.add_argument("--port", type=int, default=8080)
    # Equal-cross-sectional-area geometry scaling (GAP 1):
    # nominal_mm = base part width; size_mm = target width; lock_aspect scales H too.
    ap.add_argument("--geometry-size-mm", type=float, default=None)
    ap.add_argument("--geometry-nominal-mm", type=float, default=None)
    args = ap.parse_args()

    payload = {
        "mode": "fgm_iterate",
        "shape": args.shape,
        "output_name": args.output_name,
        # INTEGRAL mode
        "use_delta_correction": True,
        "magnitude": args.magnitude,
        "magnitude_decay": 1.0,
        "min_magnitude": 0.0,
        "move_limit": 0.0,
        "sensitivity_filter_sigma": 0.0,
        "fgm_momentum": 0.0,
        # core FGM
        "bpp": 4,
        "proxy_field": "T_phi90",
        "invert": True,
        "dead_band": 0.05,
        "baseline_saturation": 0.5,
        # convergence / stop
        "n_iterations": args.n_iterations,
        "convergence_sigma_T": 3.0,
        "melt_abort_frac": 0.15,
        # exposure / Step A
        "use_optimizer": True,
        "exposure_minutes": args.exposure_minutes,
        # EBC t* rescale (clamp 0.5-2.5)
        "use_ebc": True,
        "ebc_clamp_lo": 0.5,
        "ebc_clamp_hi": 2.5,
    }
    if args.geometry_size_mm is not None and args.geometry_nominal_mm is not None:
        payload["geometry_size_enabled"] = True
        payload["geometry_size_mm"] = args.geometry_size_mm
        payload["geometry_nominal_mm"] = args.geometry_nominal_mm
        payload["geometry_lock_aspect"] = True
    url = f"http://localhost:{args.port}/api/tools/fgm-iterate"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(body))


if __name__ == "__main__":
    main()
