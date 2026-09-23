"""End-to-end: real solved map + STL + densify march -> staged, pre-flighted job.

Exercises the operator path (stage_job.py + preflight.py in the MetPrint tools).
Skips when the MetPrint tools or the real artifacts are absent (e.g. CI).
"""
import json
from pathlib import Path

import pytest

from solve3d import stage_part as sp

REPO = Path(__file__).resolve().parents[2]
TOOLS = sp._default_meteor_tools()
# pages = round(STL height x measured densify factor / 0.2 mm layer)
#   cube    16.12 mm x 1.707 = 27.52 mm -> 138
#   pyramid 23.25 mm x 1.583 = 36.80 mm -> 184
CASES = [("cube", 1.707, 138, 24), ("pyramid", 1.583, 184, 4)]


@pytest.mark.parametrize("part,factor,pages,symmetry", CASES)
def test_stage_part_end_to_end(tmp_path, capsys, part, factor, pages, symmetry):
    m = REPO / f"solve3d/phase_e/results/map_{part}_solve_filter_only.npz"
    stl = REPO / f"shape_library_3d/stl/{part}.stl"
    fields = REPO / f"solve3d/results/densify_{part}/fields.npz"
    if not (TOOLS / "stage_job.py").is_file() or not all(
            p.exists() for p in (m, stl, fields)):
        pytest.skip("MetPrint tools or real artifacts not present")
    rc = sp.main(["--map", str(m), "--stl", str(stl), "--densify", str(fields),
                  "--hot-folder", str(tmp_path / "hf"), "--job-name", f"{part}_it",
                  "--work-dir", str(tmp_path / "work")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0, out
    assert out["staged_all_pass"] and out["preflight_ready"], out
    assert out["preflight_errors"] == []
    assert out["z_mode"] == "densify_summary"
    assert out["factor"] == pytest.approx(factor, abs=2e-3)
    assert out["print_layers"] == pages
    reg = out["registration"]
    assert reg["volume_rel_err"] < 0.03
    assert reg["extent_ok"] and reg["pose_ok"]
    # both parts are rotationally symmetric, so the report must SAY the pose is
    # verified only up to that symmetry -- never claim a unique pose
    assert reg["pose_symmetry_count"] == symmetry and reg["pose_unique"] is False

    # the staged job on disk carries provenance tying it to this densify run
    info = json.loads((Path(out["out_dir"]) / "job_info.json").read_text())
    z = info["provenance"]["z"]
    assert z["mode"] == "densify_summary"
    assert z["source"] == str(fields)
    assert info["layer_count"] == pages
    assert info["layer_height_mm"] == 0.2


def test_driver_refuses_a_wrongly_declared_pose(tmp_path, capsys):
    """The real pyramid map declared upside down must be refused before staging."""
    m = REPO / "solve3d/phase_e/results/map_pyramid_solve_filter_only.npz"
    stl = REPO / "shape_library_3d/stl/pyramid.stl"
    if not (TOOLS / "stage_job.py").is_file() or not (m.exists() and stl.exists()):
        pytest.skip("MetPrint tools or real artifacts not present")
    rc = sp.main(["--map", str(m), "--stl", str(stl), "--no-densification",
                  "--base", "max", "--hot-folder", str(tmp_path / "hf"),
                  "--job-name", "pyr_bad", "--work-dir", str(tmp_path / "work")])
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "hf").exists() or not any((tmp_path / "hf").iterdir())
