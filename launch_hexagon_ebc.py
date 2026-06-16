#!/usr/bin/env python3
"""
launch_hexagon_ebc.py
=====================
Priority Experiment P2: Hexagon EBC-Integral (M3 — Energy-Budget-Conserving
Integral FGM iteration).

Background
----------
Hexagon is a "t★-confounded" geometry (Regime 3).  The FGM has very high
contrast (κ = Qrf_max/mean = 32.7×), meaning the FGM's heavy downweighting of
the hot corners removes disproportionate absorbed power:

  P_total = σ₀ × ∫ sat_map(x,y) × |∇φ(x,y)|² dA

When sat_map drops the hot-corner contribution from ~32.7× to ~0.5×, total power
falls by roughly 30–40%.  The solver tries to compensate at t★ but now runs
past the ideal sintering window, accumulating density in the corners while the
center under-sinters.  Over multiple iterations, mean_rho drifts progressively
lower (0.805→0.785 in hexagon_run_20260504_4bpp) and the part never converges.

EBC fix: after each FGM generation, compute mean(sat_map[part_mask]) and adjust
t★ proportionally so total absorbed energy remains constant:

  t★ᵢ = t★_base × (mean_sat₀ / mean_satᵢ)

This recalibration should stabilize mean_rho near the iter-0 baseline and allow
the quantization snap (the discrete σ_T reduction event when the FGM resolution
crosses an optimal threshold) to emerge at ~iter-6 to iter-9 as expected.

Expected outcome
----------------
  Without EBC: σ_T oscillates 18→15→13→... without clear convergence
  With EBC:    σ_T should snap downward at ~iter-6 to iter-9 → < 8°C
  (Hypothesis: EBC stabilizes mean_rho → quantization snap unlocks)

Runs submitted:
  Name                          ebc    m      n   bpp   t_ceil
  ──────────────────────────────────────────────────────────────
  hexagon_EBC_m007_n15          True   0.70   15   4    8.0
  hexagon_EBC_m005_n15          True   0.50   15   4    8.0  (bracketing)
  hexagon_EBC_m008_n15          True   0.80   15   4    8.0  (bracketing)
  hexagon_NOEBC_m007_n15        False  0.70   15   4    8.0  (matched control)

The matched control (NOEBC, same parameters otherwise) allows direct comparison
of whether EBC actually helps or if hexagon's problem is something else.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def _hexagon_ebc_payload(
    output_name: str,
    m: float,
    n: int,
    use_ebc: bool,
    exposure_min: float = 8.0,
    bpp: int = 4,
) -> dict:
    return {
        "mode":                   "fgm_iterate",
        "shape":                  "hexagon",
        "size_mm":                30.0,
        "output_name":            output_name,
        # ── Integral mode ──────────────────────────────────────────────────────
        "use_delta_correction":   True,
        "magnitude":              m,
        "magnitude_decay":        1.0,         # no decay (integral mode)
        "min_magnitude":          0.0,
        "move_limit":             0.0,
        "sensitivity_filter_sigma": 0.0,
        "fgm_momentum":           0.0,
        # ── EBC: Energy-Budget-Conserving exposure adjustment (M3) ────────────
        "use_ebc":                use_ebc,
        "ebc_clamp_lo":           0.50,        # t★ never < 50% of base
        "ebc_clamp_hi":           2.50,        # t★ never > 2.5× base
        # ── Core FGM ───────────────────────────────────────────────────────────
        "bpp":                    bpp,
        "proxy_field":            "T_phi90",
        "invert":                 True,
        "dead_band":              0.05,
        "baseline_saturation":    0.5,
        # ── Convergence ────────────────────────────────────────────────────────
        "n_iterations":           n,
        "convergence_sigma_T":    3.0,
        "melt_abort_frac":        0.15,
        # ── Exposure: optimizer finds t★ ──────────────────────────────────────
        "use_optimizer":          True,
        "exposure_minutes":       exposure_min,
        # ── Abort safety ──────────────────────────────────────────────────────
        "regression_threshold_multiplier": 10.0,  # lenient — allow many iters to observe snap
    }


def _build_runs() -> list[tuple[dict, str]]:
    return [
        # ── Primary EBC run ──────────────────────────────────────────────────
        (
            _hexagon_ebc_payload("hexagon_EBC_m007_n15",  m=0.70, n=15, use_ebc=True),
            "hexagon  EBC=True   m=0.70  n=15  [PRIMARY P2 — critical test of M3]",
        ),
        # ── Matched control (no EBC, same params) ────────────────────────────
        (
            _hexagon_ebc_payload("hexagon_NOEBC_m007_n15", m=0.70, n=15, use_ebc=False),
            "hexagon  EBC=False  m=0.70  n=15  [control — identical params, no EBC]",
        ),
        # ── Bracketing ───────────────────────────────────────────────────────
        (
            _hexagon_ebc_payload("hexagon_EBC_m005_n15",  m=0.50, n=15, use_ebc=True),
            "hexagon  EBC=True   m=0.50  n=15  [bracketing — lower magnitude]",
        ),
        (
            _hexagon_ebc_payload("hexagon_EBC_m008_n15",  m=0.80, n=15, use_ebc=True),
            "hexagon  EBC=True   m=0.80  n=15  [bracketing — higher magnitude]",
        ),
    ]


def submit_run(
    base_url: str,
    payload: dict,
    label: str,
    dry_run: bool = False,
) -> str:
    url = f"{base_url}/api/tools/fgm-iterate"
    name = payload["output_name"]

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Submitting: {name}")
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
            print(
                f"  ✓ Queued  job_id={job_id}  status={body.get('status')}  "
                f"queue_position={body.get('queue_position')}"
            )
            return job_id
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        print(f"  ✗ HTTP {e.code}: {msg[:400]}", file=sys.stderr)
        return f"error-{e.code}"
    except Exception as ex:
        print(f"  ✗ {ex}", file=sys.stderr)
        return "error"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit hexagon EBC-Integral runs (P2) to HEATR server"
    )
    ap.add_argument("--port",    type=int, default=8080)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        metavar="NAME",
        help="Submit only named runs",
    )
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    all_runs = _build_runs()

    if args.only:
        keep = set(args.only)
        runs = [(p, l) for p, l in all_runs if p["output_name"] in keep]
        if not runs:
            print(f"No runs matched --only {args.only}", file=sys.stderr)
            sys.exit(1)
    else:
        runs = all_runs

    print("HEATR Hexagon EBC-Integral Sweep — Priority Experiment P2 (M3)")
    print(f"Target : {base_url}")
    print(f"Runs   : {len(runs)}")
    print(f"Method : M3 (EBC-Integral — exposure adjusted to conserve total RF energy)")
    print(f"Purpose: Resolve hexagon t★ confound caused by high FGM contrast (κ=32.7×)")
    print("─" * 70)

    results = []
    for payload, label in runs:
        job_id = submit_run(base_url, payload, label, dry_run=args.dry_run)
        results.append((payload["output_name"], job_id))
        if not args.dry_run:
            time.sleep(0.5)

    print("\n" + "─" * 70)
    print("Summary:")
    for name, job_id in results:
        print(f"  {name:<40}  job={job_id}")

    print(f"\nMonitor: {base_url}")
    print()
    print("What to look for in results:")
    print("  1. mean_rho stability: EBC run should hold mean_rho ≈ 0.90 across all iters")
    print("     Non-EBC control will show progressive rho drift (0.805→0.785 pattern)")
    print("  2. Quantization snap: EBC run σ_T should drop sharply at iter ~6–9")
    print("     (snap = discrete σ_T reduction when FGM crosses optimal threshold)")
    print("  3. Best σ_T: EBC target < 8°C; non-EBC ceiling ~15°C")
    print()
    print("Decision criterion:")
    print("  IF EBC best_sigma_T < 10°C AND non-EBC best_sigma_T > 12°C:")
    print("    → EBC is the mechanism resolving the t★ confound for hexagon")
    print("    → Proceed to implement server-side EQS power probe for exact EBC")
    print("  IF both converge similarly:")
    print("    → t★ confound is not the primary failure mode for hexagon")
    print("    → Investigate whether FGM Responsiveness Probe (R < 0.55) explains it")
    print()
    print("Estimated wall time: ~90–120 min total (15 iters × ~6-7 min each × optimizer)")


if __name__ == "__main__":
    main()
