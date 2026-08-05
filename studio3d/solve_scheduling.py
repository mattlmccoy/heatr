"""Auto-solve scheduling for Express (spec 7e / TAMPER fix step 4).

The compute-schedule rules (agreed machine-wide 2026-08-03): a heavy solve
launches only when the 1-minute load average is under 20 AND fewer than two
heavy solve processes are alive. The Express chain polls this predicate via
the CLI below (a tracked, cancellable subprocess) before launching the
direct solve.

check_part_solvable is deliberately the SINGLE place the Studio asks "can
the direct solve take this part": today that is solve3d's own extrusion
detector; when the solve3d lane lands arbitrary-STL tet meshing this one
function widens to has_solve_mesh and nothing else changes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Tuple

import numpy as np

LOAD_LIMIT = 20.0
MAX_HEAVY = 2
HEAVY_PATTERNS = ("solve3d.phase_e.run", "solve3d.studio_solve")


def slot_available(load_1min: float, n_heavy: int) -> bool:
    """The machine-wide launch rule, pure and testable."""
    return load_1min < LOAD_LIMIT and n_heavy < MAX_HEAVY


def check_part_solvable(part: np.ndarray) -> Tuple[bool, str]:
    """Can the direct solve take this part? ONE widening point, no shape
    heuristics: solve3d's own detector decides.

    WIDENED 2026-08-05 (solve3d tranche-1 notify, their commit 8d7fe39):
    arbitrary-STL chamber tet meshing passed its equivalence gate
    (solve3d/results/stl_chamber_gate.json), so non-extrusions route
    through the STL chamber path. Meshability is NOT pre-checked here:
    build_mesh_from_stl inside the solve refuses loudly
    (SurfaceReconstructionError, including the meshed-volume-vs-STL-volume
    check) and that refusal surfaces in the Express timeline. Note from
    the solve3d lane: has_solve_mesh is necessary, not sufficient - a part
    can mesh perfectly and still produce a red-gate forward (the Tamper);
    the correction chain therefore never ships an artifact whose
    solved_label is false (studio3d/correction.py)."""
    from solve3d import studio_geom as sg

    det = sg.detect_extrusion(np.asarray(part, bool))
    if det["is_extruded"]:
        return True, "extrusion detected"
    try:
        from solve3d import stl_mesh  # noqa: F401  (availability probe)
    except ImportError as e:
        return False, ("direct solve unavailable: solve3d STL chamber "
                       f"meshing not importable ({e})")
    return True, ("STL chamber tet meshing (solve3d tranche 1); mesh build "
                  "may still refuse on surface-reconstruction volume "
                  "deviation, and a red-gate solve is never shipped")


def read_machine_state() -> Tuple[float, int]:
    """(1-min load, live heavy-solve count) from the running machine."""
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout
    load = float(up.rsplit("load averages:", 1)[-1].split()[0])
    pg = subprocess.run(["pgrep", "-f", "|".join(HEAVY_PATTERNS)],
                        capture_output=True, text=True).stdout
    n_heavy = len([ln for ln in pg.splitlines() if ln.strip()])
    return load, n_heavy


def wait_for_slot(poll_s: float = 60.0, max_wait_s: float = 6 * 3600) -> bool:
    """Block until the launch rule passes; progress lines for the log tail."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        load, n_heavy = read_machine_state()
        if slot_available(load, n_heavy):
            print(f"SOLVE_SLOT_FREE load={load:.1f} heavy={n_heavy}",
                  flush=True)
            return True
        print(f"SOLVE_SLOT_WAIT load={load:.1f} heavy={n_heavy} "
              f"(need load<{LOAD_LIMIT:.0f} and heavy<{MAX_HEAVY})",
              flush=True)
        time.sleep(poll_s)
    print("SOLVE_SLOT_TIMEOUT", flush=True)
    return False


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Express solve scheduling")
    ap.add_argument("--check-part", metavar="PART_NPZ",
                    help="print {solvable, reason} for a part npz")
    ap.add_argument("--wait-slot", action="store_true",
                    help="block until a heavy-solve slot is free")
    ap.add_argument("--poll-s", type=float, default=60.0)
    args = ap.parse_args()
    if args.check_part:
        with np.load(args.check_part) as d:
            part = np.asarray(d["part"], bool)
        ok, reason = check_part_solvable(part)
        print(json.dumps({"solvable": ok, "reason": reason}))
        return 0
    if args.wait_slot:
        return 0 if wait_for_slot(poll_s=args.poll_s) else 1
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
