#!/usr/bin/env python3
"""RED/GREEN tests for the server-side per-shape preset YAML files.

Deferred item (b) from HEATR_V2_ROLLOUT_NOTES.md: persist the per-shape
one-click standards as YAML files under a presets/ directory that the server
reads. Single source: the files are REGENERATED from
webui/static/fgm_shape_standards.json by scripts/build_shape_presets.py, and
the graphical user interface fetches them through GET /api/presets (with the
static JSON as fallback for a pre-preset server).

Contract:
  1. build_shape_presets.standards_to_presets(standards) returns one preset
     dict per shape carrying the shape standard block plus the shared
     standard-parameters block, stamped with its source.
  2. build_shape_presets.write_presets(presets, out_dir) writes one
     <shape>.yaml per shape; loading the directory back reproduces the data.
  3. rfam_gui_server._load_shape_presets(dir) returns the same structure the
     front end already consumes from fgm_shape_standards.json
     ({"standard_parameters": ..., "shapes": ...}), stamped
     source="presets_yaml"; a missing or empty directory returns None (the
     front end then falls back to the static JSON, promote never remove).

Run: ./.venv312/bin/python -m pytest test_presets_yaml.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

import rfam_gui_server as srv                      # noqa: E402
from build_shape_presets import standards_to_presets, write_presets  # noqa: E402

_STANDARDS = {
    "generated": "2026-08-01",
    "engine_version_floor": "2.0.0",
    "sources": ["HEATR_STANDARD_PARAMETERS.md"],
    "standard_parameters": {
        "grid_nx": 120, "grid_ny": 120, "enforce_generator_power": False,
        "proxy_field": "T_phi90", "bpp": 4, "note": "test note",
    },
    "shapes": {
        "square": {"v_cal_v": 2428.1732217858744, "quick_look_gain": 0.9033,
                   "quick_look_verdict": "BETTER", "solve_class": "MATCHED",
                   "solve_iou_4bpp": 0.9815950920245399, "actuator_note": None},
        "cross": {"v_cal_v": 2815.4, "quick_look_gain": 1.1855,
                  "quick_look_verdict": "BETTER", "solve_class": "NOT-RESCUED",
                  "solve_iou_4bpp": 0.675, "actuator_note": "dwell upgrade"},
    },
}


def test_standards_to_presets_one_per_shape() -> None:
    presets = standards_to_presets(_STANDARDS)
    assert set(presets) == {"square", "cross"}
    sq = presets["square"]
    assert sq["shape"] == "square"
    assert sq["shape_standard"]["v_cal_v"] == _STANDARDS["shapes"]["square"]["v_cal_v"]
    assert sq["standard_parameters"] == _STANDARDS["standard_parameters"]
    assert sq["generated_from"].endswith("fgm_shape_standards.json")
    assert sq["preset_version"] == 1


def test_write_presets_yaml_roundtrip(tmp_path: Path) -> None:
    presets = standards_to_presets(_STANDARDS)
    written = write_presets(presets, tmp_path)
    assert sorted(p.name for p in written) == ["cross.yaml", "square.yaml"]
    back = yaml.safe_load((tmp_path / "cross.yaml").read_text(encoding="utf-8"))
    assert back["shape_standard"]["quick_look_gain"] == 1.1855
    assert back["standard_parameters"]["proxy_field"] == "T_phi90"


def test_server_loader_matches_front_end_structure(tmp_path: Path) -> None:
    write_presets(standards_to_presets(_STANDARDS), tmp_path)
    loaded = srv._load_shape_presets(tmp_path)
    assert loaded is not None
    assert loaded["source"] == "presets_yaml"
    assert loaded["standard_parameters"]["grid_nx"] == 120
    assert loaded["shapes"]["square"]["v_cal_v"] == _STANDARDS["shapes"]["square"]["v_cal_v"]
    assert loaded["shapes"]["cross"]["solve_class"] == "NOT-RESCUED"


def test_server_loader_missing_or_empty_dir_returns_none(tmp_path: Path) -> None:
    assert srv._load_shape_presets(tmp_path / "nope") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert srv._load_shape_presets(empty) is None


def test_repo_presets_match_current_standards_json() -> None:
    """The generated presets/ directory in the repo is in sync with the JSON."""
    presets_dir = BASE / "presets"
    assert presets_dir.exists(), "run scripts/build_shape_presets.py to generate presets/"
    standards = json.loads(
        (BASE / "webui" / "static" / "fgm_shape_standards.json").read_text(encoding="utf-8"))
    loaded = srv._load_shape_presets(presets_dir)
    assert loaded is not None
    assert set(loaded["shapes"]) == set(standards["shapes"])
    for shp, std in standards["shapes"].items():
        assert loaded["shapes"][shp] == std, f"preset drift for {shp}"
    assert loaded["standard_parameters"] == standards["standard_parameters"]
