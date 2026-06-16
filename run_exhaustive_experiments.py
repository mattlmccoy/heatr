#!/usr/bin/env python3
"""
Standalone script to run all FGM exhaustive experiments without HTTP server.
Usage:  python3 run_exhaustive_experiments.py
Can be run while rfam_gui_server is also running (uses separate dirs).
"""
import sys
import os
import time
import json
from pathlib import Path
from typing import Any

# Add geo-prewarp to path so we can import the server functions directly
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Import after path is set
import rfam_gui_server as _srv

RUNS = [
    # (shape, output_name, bpp)
    ("circle",  "circle_hybrid_2bpp_exhaustive",  2),
    ("hexagon", "hexagon_hybrid_2bpp_exhaustive",  2),
    ("diamond", "diamond_hybrid_2bpp_exhaustive",  2),
    ("circle",  "circle_hybrid_4bpp",             4),
    ("hexagon", "hexagon_hybrid_4bpp",             4),
    ("diamond", "diamond_hybrid_4bpp",             4),
]

EXHAUSTIVE_PARAMS = {
    "use_optimizer":                True,
    "n_iterations":                 50,
    "proxy_field":                  "T_phi90",
    "magnitude":                    1.0,
    "magnitude_decay":              0.92,
    "dead_band":                    0.05,
    "perturbation_amplitude":       0.08,
    "stagnation_window":            4,
    "regression_threshold_multiplier": 2.0,
    "post_perturbation_grace":      6,
    "use_hybrid":                   True,
    "hybrid_phase2_magnitude":      0.4,
    "hybrid_phase2_move_limit":     0.15,
    "hybrid_phase2_sensitivity_sigma": 0.5,
}


def make_mock_job(job_id: str) -> dict[str, Any]:
    """Create a mock job dict compatible with _launch_fgm_iterate_mode."""
    import uuid, datetime as _dt
    log_path = _srv.LOGS_DIR / f"gui_job_{job_id}.log"
    return {
        "id":               job_id,
        "job_id":           job_id,        # alias
        "mode":             "fgm_iterate",
        "status":           "running",
        "created_at":       _dt.datetime.now().isoformat(),
        "started_at":       _dt.datetime.now().isoformat(),
        "ended_at":         None,
        "output_name":      job_id,
        "log_path":         str(log_path),
        "log_url":          f"/files/outputs_eqs/_logs/{log_path.name}",
        "error":            None,
        "config_resolution": [],
        "output_dirs":      [],
        "artifacts":        [],
        "progress_pct":     0.0,
        "progress_label":   "",
        "completed_runs":   0,
        "total_runs":       50,
        "queue_position":   None,
        "physics_snapshot": None,
    }


def set_progress_noop(job_id, completed_runs=0, total_runs=None,
                      progress_pct=None, progress_label=""):
    """Silent replacement for _set_job_progress — just print to stdout."""
    pct = progress_pct or (100 * completed_runs / total_runs if total_runs else 0)
    print(f"  [{job_id}] {pct:.0f}% — {progress_label}", flush=True)


def main():
    # Monkey-patch the progress setter so we get stdout output instead of
    # writing to the in-memory job store (which doesn't exist here).
    _srv._set_job_progress = set_progress_noop

    total = len(RUNS)
    for idx, (shape, name, bpp) in enumerate(RUNS, 1):
        payload = {**EXHAUSTIVE_PARAMS, "shape": shape, "output_name": name, "bpp": bpp}
        job_id  = f"direct_{name}_{int(time.time())}"
        job     = make_mock_job(job_id)

        print(f"\n{'='*70}")
        print(f"RUN {idx}/{total}: {name}  (bpp={bpp})")
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        try:
            _srv._launch_fgm_iterate_mode(payload, job)
            elapsed = time.time() - t0
            print(f"\n  ✓ DONE in {elapsed/60:.1f} min", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ✗ FAILED after {elapsed/60:.1f} min: {e}", flush=True)
            import traceback; traceback.print_exc()

    print("\n\nAll experiments complete.", flush=True)


if __name__ == "__main__":
    os.chdir(HERE)
    main()
