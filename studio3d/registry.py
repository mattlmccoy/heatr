"""Solved-map registry (spec section 6).

Maps EXACT voxelized part-mask hashes to solved 3-D dopant artifacts. This
is deliberately strict: no shape heuristics, no fuzzy matching. A part that
does not hash-match gets the deployable 2.5-D per-slice path instead, and
the registry says so by returning None.

Entries live in solved_registry.json next to this module. Today there is
exactly one solved artifact: the Phase C cylinder
(solve3d/PHASE_C_REPORT.md, solved_label true, sim-only trust).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REGISTRY_PATH = Path(__file__).resolve().parent / "solved_registry.json"


def part_hash(part: np.ndarray) -> str:
    """SHA-256 of the packed part mask, shape included."""
    a = np.ascontiguousarray(np.asarray(part, bool))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(np.packbits(a).tobytes())
    return h.hexdigest()


# Entries written before adaptive chamber sizing (solve3d convention,
# 2026-08-05) carry no chamber tag; every one of them was solved in the
# frozen 60 mm chamber, so a missing tag reads as 0.060.
LEGACY_CHAMBER_M = 0.060


def find_solved_map(part: np.ndarray,
                    registry_path: Path = REGISTRY_PATH,
                    chamber_m: float = LEGACY_CHAMBER_M
                    ) -> Optional[Dict[str, Any]]:
    """The registered solved artifact for this exact part mask AND chamber.

    Chamber size is part of the match key, not metadata: the cross-size
    fringing pattern shift is measured real (solve3d chamber_field_check,
    rel-L2 0.21 between 60 and 85 mm vs a 0.037 remesh noise floor), so a
    map solved in one chamber is the wrong answer in another even for an
    identical part mask.
    """
    if not registry_path.exists():
        return None
    reg = json.loads(registry_path.read_text())
    ph = part_hash(part)
    for entry in reg.get("entries", []):
        if entry.get("part_sha256") != ph:
            continue
        entry_ch = float(entry.get("chamber_m", LEGACY_CHAMBER_M))
        if abs(entry_ch - float(chamber_m)) < 1e-9:
            return dict(entry)
    return None
