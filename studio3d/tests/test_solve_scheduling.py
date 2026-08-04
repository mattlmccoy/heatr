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
    part = np.zeros((16, 16, 16), bool)
    part[6:10, 6:10, 4:12] = True                  # extrusion
    ok, reason = check_part_solvable(part)
    assert ok is True
    pyramid = part.copy()
    pyramid[6:10, 6:10, 11] = False
    pyramid[7:9, 7:9, 11] = True                   # tapering top
    ok, reason = check_part_solvable(pyramid)
    assert ok is False
    assert "Phase E tet meshing" in reason
