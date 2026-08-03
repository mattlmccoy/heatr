"""STL intake gate: loud typed refusals (solver-venv side; trimesh needed).

Reuses shape_library_3d.validate for planarity/watertight/zero-volume and adds
the workbench's NEW self-intersection gate plus the chamber-fit check. Real
rejection fixtures come from the merged library (open_cylinder, flat_plane);
the self-intersecting fixture is constructed here (two interpenetrating
tetrahedra fused into one mesh with overlapping geometry and inconsistent
winding cannot be used - we build the standard self-intersection probe: a
single mesh whose faces pass through each other).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from heatr3d_workbench import intake

LIB = Path(__file__).resolve().parents[2] / "shape_library_3d" / "stl"


def _self_intersecting_mesh() -> trimesh.Trimesh:
    """A closed mesh whose surface passes through itself: two overlapping
    boxes concatenated as ONE mesh (each shell closed, so edge-watertightness
    holds per-face-component, but the combined surface self-intersects)."""
    a = trimesh.creation.box(extents=(10, 10, 10))
    b = trimesh.creation.box(extents=(10, 10, 10))
    b.apply_translation((5.0, 5.0, 5.0))
    m = trimesh.util.concatenate([a, b])
    return m


def test_accepts_library_sphere():
    v = intake.gate_stl(LIB / "sphere.stl")
    assert v["accepted"] is True
    assert v["volume_mm3"] == pytest.approx(4188.79, rel=0.01)


def test_refuses_open_cylinder_loudly():
    v = intake.gate_stl(LIB / "open_cylinder.stl")
    assert v["accepted"] is False
    assert v["error_type"] == "NonWatertightMeshError"
    assert "naked" in v["error"]


def test_refuses_flat_plane_loudly():
    v = intake.gate_stl(LIB / "flat_plane.stl")
    assert v["accepted"] is False
    assert v["error_type"] == "ZeroVolumeError"


def test_refuses_self_intersection():
    m = _self_intersecting_mesh()
    v = intake.gate_mesh(m)
    assert v["accepted"] is False
    assert v["error_type"] == "SelfIntersectingMeshError"


def test_refuses_oversize_for_chamber():
    m = trimesh.creation.box(extents=(80.0, 10.0, 10.0))   # 80 mm > 60 mm chamber
    v = intake.gate_mesh(m)
    assert v["accepted"] is False
    assert v["error_type"] == "ChamberFitError"
    assert "60" in v["error"]


def test_accept_reports_extents_and_volume():
    m = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    v = intake.gate_mesh(m)
    assert v["accepted"] is True
    assert v["extents_mm"] == pytest.approx([10.0, 10.0, 10.0])
    assert v["volume_mm3"] == pytest.approx(1000.0)
    assert v["watertight"] is True
