"""Intake refusal rules for the Grade and Print path (spec section 4).

A mesh is ACCEPTED only when it is watertight and free of self-intersections.
Refusals are loud and informative: they name the defect and its count.
Hole refusal happens at analyze time on the slice loops (existing has_holes
detection), not here.
"""
from __future__ import annotations

import json

import numpy as np
import trimesh

from studio3d.intake import intake_verdict


def _box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def _open_box() -> trimesh.Trimesh:
    b = _box()
    return trimesh.Trimesh(vertices=b.vertices, faces=b.faces[:-2],
                           process=False)


def _interpenetrating_boxes() -> trimesh.Trimesh:
    a = _box()
    b = _box()
    # generic-position offset: exactly symmetric offsets put every crossing
    # on a triangle edge, which the strict interior test excludes so that
    # exactly-touching shells are not refused
    b.apply_translation((0.4, 0.3, 0.35))
    return trimesh.util.concatenate([a, b])


def test_watertight_box_is_accepted():
    v = intake_verdict(_box())
    assert v["accepted"] is True
    assert v["checks"]["watertight"]["ok"] is True
    assert v["checks"]["self_intersection"]["ok"] is True
    assert v["checks"]["self_intersection"]["n_intersecting_pairs"] == 0


def test_open_mesh_is_refused_with_open_edge_count():
    v = intake_verdict(_open_box())
    assert v["accepted"] is False
    c = v["checks"]["watertight"]
    assert c["ok"] is False
    assert c["n_open_edges"] == 4
    assert "REFUSED" in v["error"]
    assert "watertight" in v["error"]
    assert "4" in v["error"]


def test_interpenetrating_shells_are_refused_as_self_intersecting():
    m = _interpenetrating_boxes()
    # each shell is closed, so watertightness alone would wrongly accept it
    assert m.is_watertight
    v = intake_verdict(m)
    assert v["accepted"] is False
    c = v["checks"]["self_intersection"]
    assert c["ok"] is False
    assert c["n_intersecting_pairs"] > 0
    assert "REFUSED" in v["error"]
    assert "self-intersect" in v["error"]


def test_verdict_is_json_serializable():
    for m in (_box(), _open_box(), _interpenetrating_boxes()):
        json.dumps(intake_verdict(m))


def test_sphere_has_no_false_positive_pairs():
    # adjacent triangles share vertices and must not count as intersections
    v = intake_verdict(trimesh.creation.icosphere(subdivisions=2))
    assert v["accepted"] is True
    assert v["checks"]["self_intersection"]["n_intersecting_pairs"] == 0


def test_oversize_part_is_refused_at_intake(tmp_path):
    """Chamber fit is an import-time fact (found live: a 100 mm calibration
    part sailed through intake and failed minutes later inside the Express
    densify stage with a raw traceback)."""
    import trimesh as tm
    p = tmp_path / "big.stl"
    tm.creation.box(extents=(100.0, 100.0, 50.0)).export(p)
    v = intake_verdict(tm.load_mesh(p))
    assert v["accepted"] is False
    assert v["checks"]["chamber_fit"]["ok"] is False
    assert "REFUSED" in v["error"] and "60 mm" in v["error"]
    assert "100" in v["error"]


def test_in_chamber_part_passes_the_fit_check():
    v = intake_verdict(_box())   # 1 mm box, far inside the chamber
    assert v["checks"]["chamber_fit"]["ok"] is True


def test_self_intersection_check_is_fast_on_real_size_meshes():
    """Perf gate (found live: a real STL hung intake for 20+ minutes).
    An 82k-triangle sphere must clear the check in seconds, not minutes."""
    import time
    from studio3d.intake import count_self_intersections
    m = trimesh.creation.icosphere(subdivisions=6)   # 81,920 triangles
    t0 = time.time()
    n = count_self_intersections(m)
    dt = time.time() - t0
    assert n == 0
    assert dt < 15.0, f"self-intersection check took {dt:.1f} s"
