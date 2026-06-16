#!/usr/bin/env python3
"""
launch_lshape_turntable.py
==========================
Priority Experiment P1: L-shape turntable runs to evaluate M5 (static-to-turntable
rotation as the primary remedy for the EM Shadow Zone regime).

Physical rationale
------------------
The L-shape reentrant corner is geometrically screened in EQS: the inner concavity
creates a Laplace "shadow zone" where increasing local σ decreases |∇φ|², making
conventional FGM counterproductive.  Time-averaging the E-field over multiple
orientations destroys the persistent shadow zone, distributing Qrf uniformly.

Expected outcome
----------------
  L-shape 140°, 90°×4  →  σ_T < 15°C  (from 32.7°C baseline)  — P1 primary
  L-shape 140°, 45°×8  →  σ_T < 12°C  (finer angular coverage)  — P1 extended
  L-shape 0°,   90°×4  →  σ_T < 18°C  (unrotated control)
  L-shape 0°,   45°×8  →  σ_T < 15°C  (unrotated fine coverage)

Subsequent step (after P1 confirms turntable reduces σ_T):
  P4: FCT-FGM — run turntable + FGM on the time-averaged T field
  (expected further −20–30% σ_T reduction on top of turntable baseline)

Runs submitted:
  Name                            rot0    tt_deg  n_rot  t_min
  ──────────────────────────────────────────────────────────────
  lshape_tt_140deg_90x4           140°    90°      4     7.0
  lshape_tt_140deg_45x8           140°    45°      8     7.0
  lshape_tt_0deg_90x4               0°    90°      4     7.5
  lshape_tt_0deg_45x8               0°    45°      8     7.5

All runs go to /api/run with mode=turntable.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def _turntable_payload(
    output_name: str,
    rotation_deg: float,
    total_rotations: int,
    exposure_min: float,
    base_config: str | None = None,
) -> dict:
    """Build a turntable run payload for the L-shape geometry."""
    p = {
        "mode":                  "turntable",
        "shape":                 "L_shape",
        "output_name":           output_name,
        "exposure_minutes":      exposure_min,
        # Turntable parameters
        "turntable_rotation_deg":    rotation_deg,
        "turntable_total_rotations": total_rotations,
        # No rotation_interval_s → solver distributes evenly over exposure time
    }
    if base_config:
        p["base_config"] = base_config
    return p


def _build_runs() -> list[tuple[dict, str]]:
    """Return list of (payload, description) pairs."""
    return [
        # ── Primary: L-shape at optimal 140° with turntable ──────────────────
        (
            _turntable_payload(
                output_name     = "lshape_tt_140deg_90x4",
                rotation_deg    = 90.0,
                total_rotations = 4,
                exposure_min    = 7.0,
                base_config     = "shape_L_shape_140deg.yaml",
            ),
            "L_shape 140°  tt=90°×4  7.0 min   [PRIMARY P1 — optimal initial orientation]",
        ),
        (
            _turntable_payload(
                output_name     = "lshape_tt_140deg_45x8",
                rotation_deg    = 45.0,
                total_rotations = 8,
                exposure_min    = 7.0,
                base_config     = "shape_L_shape_140deg.yaml",
            ),
            "L_shape 140°  tt=45°×8  7.0 min   [P1 extended — finer angular coverage]",
        ),
        # ── Control: L-shape at 0° (unrotated) with turntable ────────────────
        (
            _turntable_payload(
                output_name     = "lshape_tt_0deg_90x4",
                rotation_deg    = 90.0,
                total_rotations = 4,
                exposure_min    = 7.5,
                base_config     = None,
            ),
            "L_shape   0°  tt=90°×4  7.5 min   [control — unrotated, coarse]",
        ),
        (
            _turntable_payload(
                output_name     = "lshape_tt_0deg_45x8",
                rotation_deg    = 45.0,
                total_rotations = 8,
                exposure_min    = 7.5,
                base_config     = None,
            ),
            "L_shape   0°  tt=45°×8  7.5 min   [control — unrotated, fine]",
        ),
    ]


def submit_run(
    base_url: str,
    payload: dict,
    label: str,
    dry_run: bool = False,
) -> str:
    url = f"{base_url}/api/run"
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
        description="Submit L-shape turntable runs (P1) to HEATR server"
    )
    ap.add_argument("--port",    type=int, default=8080)
    ap.add_argument("--host",    default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        metavar="NAME",
        help="Submit only runs whose output_name matches one of these strings",
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

    print("HEATR L-shape Turntable Sweep — Priority Experiment P1")
    print(f"Target : {base_url}")
    print(f"Runs   : {len(runs)}")
    print(f"Method : M5 (static-to-turntable rotation, no FGM)")
    print(f"Purpose: Eliminate EM shadow zone by time-averaging E-field orientation")
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
        print(f"  {name:<42}  job={job_id}")

    print(f"\nMonitor: {base_url}")
    print()
    print("Expected σ_T outcomes (vs single-orientation baselines):")
    print("  lshape_tt_140deg_90x4  →  < 15°C  (from σ_T0=32.7°C at 140°)")
    print("  lshape_tt_140deg_45x8  →  < 12°C  (finer coverage)")
    print("  lshape_tt_0deg_90x4    →  < 18°C  (from σ_T0=46.1°C at 0°)")
    print("  lshape_tt_0deg_45x8    →  < 15°C")
    print()
    print("Next step after P1:")
    print("  If turntable achieves σ_T < 15°C → run P4: FCT-FGM on turntable T field")
    print("  FCT-FGM: turntable sim → T_avg → FGM → new turntable sim with FGM applied")
    print("  Expected: additional −20–30% σ_T reduction")
    print()
    print("Estimated wall time: ~6–8 min per run (7.0–7.5 min exposure + overhead)")


if __name__ == "__main__":
    main()
