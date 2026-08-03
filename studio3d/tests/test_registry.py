"""Solved-map registry (spec section 6): geometry-hash matching only.

No shape heuristics: an imported part matches a solved artifact only when
its voxelized part mask hashes identically to the registered one. No hash
match, no solved map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from studio3d.registry import find_solved_map, part_hash

ROOT = Path(__file__).resolve().parents[2]
PHASE_C_INV = ROOT / "solve3d/results/phase_c_inversion_map.npz"


def _cylinder_part() -> np.ndarray:
    with np.load(PHASE_C_INV) as d:
        return d["part"].astype(bool)


def test_real_cylinder_part_matches_the_phase_c_artifact():
    entry = find_solved_map(_cylinder_part())
    assert entry is not None
    assert entry["engine"] == "solve3d_solved"
    assert "sim-only" in entry["trust_badge"]
    assert (ROOT / entry["artifact"]).exists()
    assert entry["form"] == "dg0"


def test_non_matching_part_returns_none():
    part = _cylinder_part().copy()
    part[0, 0, 0] = ~part[0, 0, 0]
    assert find_solved_map(part) is None


def test_part_hash_is_shape_sensitive():
    a = np.ones((4, 4, 4), bool)
    b = np.ones((8, 8, 8), bool)
    assert part_hash(a) != part_hash(b)
