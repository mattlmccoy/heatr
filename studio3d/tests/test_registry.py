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


def test_chamber_tag_is_part_of_the_match_key(tmp_path):
    """Adaptive chamber sizing (solve3d convention, Matt 2026-08-05): the
    cross-size fringing pattern shift is measured REAL (rel-L2 0.21 at 60 vs
    85 mm against a 0.037 remesh noise floor, chamber_field_check.json), so
    a ch060 solved map is the wrong answer for a ch085 job even for an
    identical part mask. Chamber size is part of the match key. Entries
    written before 2026-08-05 carry no tag and are all frozen-60 mm by
    construction, so a missing tag reads as 0.060."""
    import json
    import numpy as np
    from studio3d.registry import find_solved_map, part_hash

    part = np.zeros((8, 8, 8), bool)
    part[2:6, 2:6, 2:6] = True
    ph = part_hash(part)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"entries": [
        {"name": "grown", "part_sha256": ph, "chamber_m": 0.085},
        {"name": "legacy_frozen", "part_sha256": ph},
    ]}))
    hit = find_solved_map(part, registry_path=reg, chamber_m=0.060)
    assert hit is not None and hit["name"] == "legacy_frozen"
    hit = find_solved_map(part, registry_path=reg, chamber_m=0.085)
    assert hit is not None and hit["name"] == "grown"
    reg.write_text(json.dumps({"entries": [
        {"name": "grown", "part_sha256": ph, "chamber_m": 0.085}]}))
    assert find_solved_map(part, registry_path=reg, chamber_m=0.060) is None
