#!/usr/bin/env python3
"""RED/GREEN tests for the STUDIO ALPHA print package format and emitter.

The print package is the artifact the Meteor print workflow and the future
scheduler consume: one versioned directory (plus a zip) per planned part with
manifest.json, the Meteor-import raster PNG at the production printer DPI
(dots per inch) convention, the 4 bits per pixel (bpp) map npz, the turntable
program JSON (or an explicit static record), and the production verify summary.

Data-contract anchors (probed against reality, not invented):
  * production DPI is 720, the Meteor native default of fgm_generator.py:70
    (`dpi: int = 720`), resample path fgm_generator.py:588-606;
  * at grid 120 over the 60 mm chamber the level_map is 1715 x 1715, matching
    the stored production artifacts (SHAPE_LIBRARY_SOLVE_REPORT.md quantizer
    check against "1715 x 1715 level_map artifacts");
  * the real-map fixture is CAPTURED from the intake campaign
    (fgm_solve_campaign/out_intake/keyhole_maps.npz, key static_4bpp), never
    hand-authored;
  * the Meteor import PNG convention is fgm_generator.py:702-744 (preview is
    white = max ink flipped vertically; meteor import is the exact pixel
    inversion), and the production loader convention for a PNG is the
    /api/tools/import-fgm-png math in rfam_gui_server.py
    (level = round((255 - pixel) / 255 * max_level)).

Run: ./.venv312/bin/python -m pytest test_print_package.py -q
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import print_package as pp  # noqa: E402

FIXTURE = BASE / "fgm_solve_campaign" / "out_intake" / "keyhole_maps.npz"


@pytest.fixture(scope="module")
def keyhole_map():
    with np.load(FIXTURE) as d:
        return {"sat": np.asarray(d["static_4bpp"], dtype=np.float32),
                "x": np.asarray(d["x"], dtype=float),
                "y": np.asarray(d["y"], dtype=float),
                "part_mask": np.asarray(d["part_mask"], dtype=bool)}


def _minimal_manifest_kwargs():
    return dict(
        part_name="keyhole",
        source_geometry_sha256="ab" * 32,
        source_route="polygon",
        engine_version="2.0.0",
        chi_provenance={"grid": 120, "n_sub": 32,
                        "convention": "sub-cell area fill (chi_area)"},
        recommendation={"actuator_class": "MAP_PLUS_MODE", "mode": "continuous",
                        "rotation_recommended": True},
        gates={"energy_gate": {"run": True, "pass": True,
                               "residual_frac": 0.002},
               "gate_a_grid_holdout": {"run": False},
               "gate_b_sub_filter_blur": {"run": False}},
        expected_outcomes={"J": 8.08, "IoU": 0.9753, "grid": 120},
        power_settings={"rf_mode": "constant", "drive": "voltage",
                        "voltage_v": 2630.33,
                        "calibration": "auto-calibrated, 500 W/m uniform arm"},
        planned_arm="A_continuous_4bpp",
    )


# ---------------------------------------------------------------------------
# manifest schema
# ---------------------------------------------------------------------------

def test_schema_version_and_production_dpi_constants():
    assert pp.SCHEMA_VERSION == "1.0.0"
    assert pp.PRODUCTION_DPI == 720  # fgm_generator.py:70, Meteor native


def test_build_manifest_carries_every_required_block():
    m = pp.build_manifest(**_minimal_manifest_kwargs())
    assert m["schema_version"] == pp.SCHEMA_VERSION
    assert m["engine_version"] == "2.0.0"
    assert m["part"]["name"] == "keyhole"
    assert m["part"]["source_geometry_sha256"] == "ab" * 32
    assert m["chi_provenance"]["grid"] == 120
    assert m["classifier_recommendation"]["actuator_class"] == "MAP_PLUS_MODE"
    # the advisory disclaimer is mandatory and states the honest record
    disc = m["classifier_recommendation"]["advisory_disclaimer"]
    assert "advisory" in disc.lower()
    assert "1 of 2" in disc
    assert m["expected_outcomes"]["grid"] == 120
    assert "grid" in m["expected_outcomes"]["grid_qualifier"].lower()
    assert m["power_settings"]["rf_mode"] == "constant"
    assert m["scheduler"]["drop_folder"] == "packages/"
    assert "hardware" in m["scheduler"]["note"].lower()


def test_validate_manifest_flags_missing_blocks():
    m = pp.build_manifest(**_minimal_manifest_kwargs())
    assert pp.validate_manifest(m) == []
    bad = dict(m)
    bad.pop("power_settings")
    errs = pp.validate_manifest(bad)
    assert any("power_settings" in e for e in errs)


def test_geometry_hash_is_deterministic_and_sensitive():
    poly = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
    h1 = pp.geometry_hash([poly])
    h2 = pp.geometry_hash([poly.copy()])
    assert h1 == h2 and len(h1) == 64
    h3 = pp.geometry_hash([poly + 1e-6])
    assert h3 != h1


def test_package_name_scheme_is_stable():
    n = pp.package_name("keyhole", stamp="20260802-120000")
    assert n == "pkg_keyhole_20260802-120000"


# ---------------------------------------------------------------------------
# emitter, real captured map
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emitted(tmp_path_factory, keyhole_map):
    root = tmp_path_factory.mktemp("packages")
    program = {"actuator": "indexed turntable, constant radio-frequency power",
               "positions_deg": [0.0, 90.0], "cycle_time_s": 20.0,
               "moves": [{"angle_deg": 0.0, "duration_s": 10.0},
                         {"angle_deg": 90.0, "duration_s": 10.0}]}
    pkg_dir, zip_path = pp.emit_package(
        root, manifest_kwargs=_minimal_manifest_kwargs(),
        sat_map=keyhole_map["sat"], x=keyhole_map["x"], y=keyhole_map["y"],
        bpp=4, turntable_program=program,
        production_verify={"run": True, "J_production": 8.1,
                           "agrees": True},
        stamp="20260802-120000")
    return {"dir": pkg_dir, "zip": zip_path, "root": root}


def test_emit_package_writes_the_full_inventory(emitted):
    d = emitted["dir"]
    assert d.name == "pkg_keyhole_20260802-120000"
    for f in ("manifest.json", "raster_meteor_import.png",
              "raster_preview.png", "map_4bpp.npz",
              "turntable_program.json", "production_verify_summary.json"):
        assert (d / f).exists(), f
    assert emitted["zip"].exists()
    with zipfile.ZipFile(emitted["zip"]) as z:
        names = {Path(n).name for n in z.namelist()}
    assert "manifest.json" in names and "raster_meteor_import.png" in names


def test_manifest_on_disk_is_valid_and_lists_files_with_hashes(emitted):
    m = json.loads((emitted["dir"] / "manifest.json").read_text())
    assert pp.validate_manifest(m) == []
    listed = {f["name"] for f in m["files"]}
    assert "raster_meteor_import.png" in listed
    for f in m["files"]:
        assert len(f["sha256"]) == 64
        assert (emitted["dir"] / f["name"]).exists()
    assert m["raster"]["dpi"] == 720
    assert m["raster"]["bpp"] == 4
    assert m["turntable"]["mode"] == "program"


def test_raster_is_at_the_production_printer_dpi(emitted):
    from PIL import Image
    img = Image.open(emitted["dir"] / "raster_meteor_import.png")
    # grid 120 over 60 mm at 720 dots per inch: the 1715-class level_map
    assert img.size == (1715, 1715)


def test_meteor_png_round_trips_through_the_production_loader_convention(
        emitted, keyhole_map):
    """The real-data contract test.

    Meteor import PNG -> the /api/tools/import-fgm-png inversion math
    (rfam_gui_server.py: level = round((255 - pixel)/255 * max_level)) ->
    un-flip -> must equal the level_map stored in the package npz exactly,
    and resampling that level_map back to the simulation grid through the
    production loader convention (printability.load_level_map, the
    rfam_eqs_coupled.py:366-380 inverse) must reproduce the delivered map
    inside the part to within one quantization step.
    """
    from PIL import Image
    sys.path.insert(0, str(BASE / "fgm_solve_campaign"))
    from adjoint2d import printability as pq

    d = emitted["dir"]
    with np.load(d / "map_4bpp.npz", allow_pickle=True) as z:
        level_map = np.asarray(z["level_map"])
        bpp = int(z["bpp"])
        assert int(z["dpi"]) == 720
    max_val = (1 << bpp) - 1

    pix = np.asarray(Image.open(d / "raster_meteor_import.png").convert("L"),
                     dtype=np.uint8)
    recovered = np.round((255 - pix.astype(np.float32)) / 255.0 * max_val)
    recovered = np.clip(recovered, 0, max_val).astype(np.uint8)
    # the PNG pair is flipped vertically (image top = physical top)
    recovered = np.flipud(recovered)
    assert recovered.shape == level_map.shape
    assert np.array_equal(recovered, level_map)

    sat_back = pq.load_level_map(level_map, len(keyhole_map["x"]),
                                 len(keyhole_map["y"]), bpp=bpp)
    pm = keyhole_map["part_mask"]
    err = np.abs(sat_back[pm] - keyhole_map["sat"][pm])
    assert float(np.quantile(err, 0.99)) <= 1.0 / max_val + 1e-6


def test_static_plan_writes_an_explicit_no_turntable_record(
        tmp_path, keyhole_map):
    kw = _minimal_manifest_kwargs()
    kw["recommendation"] = {"actuator_class": "MAP_ONLY", "mode": "static",
                            "rotation_recommended": False}
    pkg_dir, _ = pp.emit_package(
        tmp_path, manifest_kwargs=kw, sat_map=keyhole_map["sat"],
        x=keyhole_map["x"], y=keyhole_map["y"], bpp=4,
        turntable_program=None, production_verify=None,
        stamp="20260802-120001")
    rec = json.loads((pkg_dir / "turntable_static.json").read_text())
    assert rec["mode"] == "static"
    assert "no turntable" in rec["statement"].lower()
    m = json.loads((pkg_dir / "manifest.json").read_text())
    assert m["turntable"]["mode"] == "static"
    assert pp.validate_manifest(m) == []
    # production verify not run must be stated, never silently absent
    assert m["production_verify"]["run"] is False
