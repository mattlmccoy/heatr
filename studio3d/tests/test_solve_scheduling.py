"""Auto-solve scheduling (spec 7e / Tamper fix step 4): the slot predicate
and the one-place solvability check."""
from __future__ import annotations

import numpy as np

from studio3d.solve_scheduling import check_part_solvable, slot_available


def test_slot_predicate_enforces_both_rules():
    assert slot_available(load_1min=5.0, n_heavy=0) is True
    assert slot_available(load_1min=5.0, n_heavy=1) is True
    assert slot_available(load_1min=5.0, n_heavy=2) is False   # slots full
    assert slot_available(load_1min=25.0, n_heavy=0) is False  # load too high
    assert slot_available(load_1min=19.9, n_heavy=1) is True


def test_solvable_check_is_the_single_widening_point():
    """WIDENED 2026-08-05 per the solve3d lane's tranche-1 notify (their
    commit 8d7fe39): arbitrary-STL chamber tet meshing passed its
    equivalence gate (solve3d/results/stl_chamber_gate.json), so
    non-extrusions are now solve-eligible through the STL chamber path.
    Meshability itself is decided by build_mesh_from_stl inside the solve
    (it refuses with SurfaceReconstructionError on volume deviation); this
    check only routes."""
    part = np.zeros((16, 16, 16), bool)
    part[6:10, 6:10, 4:12] = True                  # extrusion
    ok, reason = check_part_solvable(part)
    assert ok is True
    assert "extrusion" in reason
    pyramid = part.copy()
    pyramid[6:10, 6:10, 11] = False
    pyramid[7:9, 7:9, 11] = True                   # tapering top
    ok, reason = check_part_solvable(pyramid)
    assert ok is True
    assert "STL chamber" in reason
