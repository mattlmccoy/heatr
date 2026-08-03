"""server_module: endpoint assembly logic against REAL repo artifacts.

Stdlib-only module (runs in the GUI server interpreter). Tests read the real
shape library meta, the real captured run 8a16b9459f84, and the real
solve3d/results Phase C artifacts - fixtures are reality, not inventions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heatr3d_workbench import server_module as SM

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs_eqs" / "_heatr3d" / "8a16b9459f84"


def test_library_shapes_lists_14_with_tiers():
    shapes = SM.library_shapes()
    names = {s["name"] for s in shapes}
    assert "sphere" in names and "open_cylinder" in names
    assert len(shapes) == 14
    t3 = [s for s in shapes if s["tier"] == 3]
    assert {s["name"] for s in t3} == {"open_cylinder", "flat_plane"}
    for s in t3:
        assert s["loadable"] is False


@pytest.mark.skipif(not RUN.exists(), reason="captured run absent")
def test_run_detail_carries_results_flags_badges():
    d = SM.run_detail("8a16b9459f84")
    on_disk = json.loads((RUN / "results.json").read_text())
    assert d["results"]["sigma_T"] == on_disk["sigma_T"]   # assembled, not invented
    assert d["banner"] in ("ok", "warn", "fail")
    ids = {f["id"] for f in d["flags"]}
    assert "energy_gate" in ids
    # legacy run: energy gate must be not_recorded, never pass
    e = next(f for f in d["flags"] if f["id"] == "energy_gate")
    assert e["state"] == "not_recorded"
    assert d["badges"]["thermal"]["label"].startswith("exploratory")
    assert d["fieldmeta"]["dims"] == [32, 32, 32]
    # this captured run was backfilled with --rerender (2026-08-03), so all
    # three slice axes are available; a never-rerendered legacy run reads ["z"]
    assert set(d["slice_axes"]) == {"x", "y", "z"}


def test_run_detail_unknown_id_is_error_not_empty():
    d = SM.run_detail("nope")
    assert d.get("error")


def test_solved_cards_from_real_phase_c():
    cards = SM.solved_cards()
    assert len(cards) >= 4          # 3 solve arms + uniform + inversion(dropped)
    by = {c["arm"]: c for c in cards}
    scaled = by["solve_filter_only_asymmetric_scaled"]
    assert scaled["solved_label"] is True
    assert scaled["badge"]["level"] == "sim-only"
    assert scaled["deviation"] is True         # recorded-deviation arm
    drop = by["inversion_map"]
    assert drop["status"] == "DROPPED"
    # numbers are printed from JSON, never transcribed: the 10.67% margin
    # from PHASE_C_REPORT must emerge from the served values themselves.
    margin = 1.0 - scaled["J_asymmetric"] / by["uniform_baseline"]["J_asymmetric"]
    assert margin == pytest.approx(0.1067, abs=0.001)


def test_stl_route_serves_library_shape_and_rejects_traversal():
    r = SM.handle_get("/api/heatr3d/wb/stl", "shape=sphere")
    assert r[0] == 200 and r[1]["_serve_file"].endswith("shape_library_3d/stl/sphere.stl")
    assert SM.handle_get("/api/heatr3d/wb/stl", "shape=../../etc/passwd")[0] == 404
    assert SM.handle_get("/api/heatr3d/wb/stl", "shape=nope")[0] == 404


@pytest.mark.skipif(not (ROOT / "outputs_eqs/_heatr3d/5a4e465fde1b").exists(),
                    reason="cone run absent")
def test_stl_route_resolves_run_geometry():
    r = SM.handle_get("/api/heatr3d/wb/stl", "run=5a4e465fde1b")
    assert r[0] == 200 and r[1]["_serve_file"].endswith("stl/cone.stl")


def test_enqueue_validation_rejects_ceiling_and_bad_source(tmp_path):
    q = SM._queue_for_tests(tmp_path)
    ok, err = SM.validate_cfg({"source": "parametric", "shape": "sphere", "n": 96})
    assert ok
    ok, err = SM.validate_cfg({"source": "parametric", "shape": "sphere", "n": 128})
    assert not ok and "ceiling" in err
    ok, err = SM.validate_cfg({"source": "library", "library_shape": "open_cylinder", "n": 64})
    assert not ok and "not a loadable part" in err
    ok, err = SM.validate_cfg({"source": "junk", "n": 64})
    assert not ok
