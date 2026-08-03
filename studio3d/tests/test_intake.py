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
