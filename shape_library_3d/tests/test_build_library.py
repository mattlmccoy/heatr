"""build_library generates watertight STLs + measured metadata; public API enumerates parts."""
import pytest

from shape_library_3d.build_library import build
from shape_library_3d import iter_parts, load_part_stl
from shape_library_3d.constants import V_STAR_MM3


def test_build_writes_all_stls_and_measured_metadata(tmp_path):
    manifest = build(out_dir=tmp_path)
    assert len(manifest["shapes"]) == 14
    by_name = {e["name"]: e for e in manifest["shapes"]}
    # every Tier-1/2 part: watertight, at V*, STL on disk, genus present
    for name, e in by_name.items():
        if e["tier"] in (1, 2):
            assert e["is_watertight"] is True
            assert e["volume_err_frac"] < 1e-4
            assert e["actual_volume_mm3"] == pytest.approx(V_STAR_MM3, rel=1e-4)
            assert (tmp_path / "stl" / f"{name}.stl").exists()
            assert (tmp_path / "meta" / f"{name}.json").exists()
            assert isinstance(e["genus"], int)
            assert e["facet_area_mm2"]["mean"] > 0
            assert len(e["sha256"]) == 64
    # genus witnesses
    assert by_name["toroid"]["genus"] == 1
    assert by_name["pipe"]["genus"] == 1
    assert by_name["lattice"]["genus"] >= 1


def test_tier3_recorded_as_reject_with_reason(tmp_path):
    manifest = build(out_dir=tmp_path)
    rejects = [e for e in manifest["shapes"] if e["tier"] == 3]
    assert len(rejects) == 2
    for e in rejects:
        assert e["role"] == "reject"
        assert e["target_volume_mm3"] is None
        assert e["rejection_error"] in ("NonWatertightMeshError", "ZeroVolumeError")


def test_manifest_written_to_disk(tmp_path):
    build(out_dir=tmp_path)
    assert (tmp_path / "meta" / "library_manifest.json").exists()


def test_public_api_iter_parts_excludes_rejects():
    names = [name for name, _ in iter_parts()]
    assert "cube" in names and "sphere" in names
    assert "open_cylinder" not in names and "flat_plane" not in names
    assert len(names) == 12


def test_load_part_stl_roundtrips_a_normalized_part(tmp_path):
    build(out_dir=tmp_path)
    m = load_part_stl("cube", stl_dir=tmp_path / "stl")
    assert m.is_watertight
    assert m.volume == pytest.approx(V_STAR_MM3, rel=1e-3)
