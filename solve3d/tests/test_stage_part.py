import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from solve3d import stage_part as sp


_RZ90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)   # 90 deg about z


def _pyramid_mesh(b=20.0, h=20.0, b_y=None):
    bx, by = b, (b if b_y is None else b_y)
    v = np.array([[-bx / 2, -by / 2, 0], [bx / 2, -by / 2, 0], [bx / 2, by / 2, 0],
                  [-bx / 2, by / 2, 0], [0, 0, h]], float)
    return trimesh.convex.convex_hull(v)


def _map_from_mesh(mesh, R=np.eye(3), cell_mm=0.5):
    """Synthetic solve3d map: cell centres on a regular grid inside the mesh, in a
    SOLVE frame centred on the solid's centre of mass and rotated so that applying
    R recovers the print pose (row form: c = (P - com) @ R). Metres, like real maps.
    The dopant rises with print-frame z so orientation errors are visible."""
    lo, hi = mesh.bounds
    ax = [np.arange(lo[i] + cell_mm / 2, hi[i], cell_mm) for i in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    P = P[mesh.contains(P)]
    c = (P - np.asarray(mesh.center_mass)) @ R
    s = np.clip((P[:, 2] - lo[2]) / (hi[2] - lo[2]), 0, 1)
    return {"centroids": c / 1e3, "volumes": np.full(len(P), (cell_mm / 1e3) ** 3),
            "s_map": s}


def _save_map(tmp_path, m, name="map.npz"):
    p = tmp_path / name
    np.savez(p, **m)
    return p


def test_rotations_are_proper_never_mirrors():
    for i, ax in enumerate("xyz"):
        for base in ("min", "max"):
            R = sp.rotation(ax, base)
            assert np.allclose(R @ R.T, np.eye(3))
            assert np.linalg.det(R) == pytest.approx(1.0)
            assert np.allclose(R @ np.eye(3)[i], [0, 0, 1 if base == "min" else -1])


def test_register_accepts_correct_pose(tmp_path):
    mesh = _pyramid_mesh()
    reg = sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(mesh))), mesh)
    rep = reg["report"]
    assert rep["volume_rel_err"] < 0.03
    assert rep["pose_ok"] is True
    w = np.full(len(reg["points_mm"]), 1.0)
    assert np.allclose(np.average(reg["points_mm"], 0, weights=w), mesh.center_mass, atol=0.5)


def test_register_refuses_upside_down(tmp_path):
    mesh = _pyramid_mesh()
    m = _map_from_mesh(mesh, R=sp.rotation("z", "max"))       # solved base-up
    with pytest.raises(sp.RegistrationError, match="upside down"):
        sp.register(sp.load_map(_save_map(tmp_path, m)), mesh)
    sp.register(sp.load_map(_save_map(tmp_path, m)), mesh, base="max")


def test_register_refuses_wrong_build_axis(tmp_path):
    mesh = _pyramid_mesh()
    m = _map_from_mesh(mesh, R=sp.rotation("y", "min"))       # solved with build +y
    with pytest.raises(sp.RegistrationError):
        sp.register(sp.load_map(_save_map(tmp_path, m)), mesh)
    sp.register(sp.load_map(_save_map(tmp_path, m)), mesh, build_axis="y")


def test_register_refuses_volume_mismatch(tmp_path):
    mesh = _pyramid_mesh()
    m = _map_from_mesh(mesh)
    m["volumes"] = m["volumes"] * 1.3
    with pytest.raises(sp.RegistrationError, match="volume"):
        sp.register(sp.load_map(_save_map(tmp_path, m)), mesh)


def test_load_map_refuses_non_dg0(tmp_path):
    p = tmp_path / "grid.npz"
    np.savez(p, sat_map=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="DG0"):
        sp.load_map(p)


def test_build_spec_is_staging_ready(tmp_path):
    mesh = _pyramid_mesh()
    stl = tmp_path / "pyr.stl"
    mesh.export(stl)
    spec = sp.build_spec(_save_map(tmp_path, _map_from_mesh(mesh)), stl, voxel_mm=1.0)
    sol, m = spec["SOLVE_cont"], spec["part_mask"]
    assert sol.ndim == 3 and sol.shape[1] == sol.shape[2]        # (nz, n, n)
    assert m[0].sum() > m[-1].sum()                                # base at k = 0
    assert spec["z_mm"] == pytest.approx(20.0)
    assert spec["domain_mm"] >= 20.0
    assert sol.shape[1] == round(spec["domain_mm"] / 1.0)
    assert np.all(sol[~m] == 0) and sol.min() >= 0 and sol.max() <= 1
    means = [sol[k][m[k]].mean() for k in range(sol.shape[0]) if m[k].any()]
    assert means[-1] > means[0]                    # dopant still rises with height


def test_build_spec_refuses_too_small_chamber(tmp_path):
    mesh = _pyramid_mesh()
    stl = tmp_path / "pyr.stl"
    mesh.export(stl)
    with pytest.raises(ValueError, match="chamber"):
        sp.build_spec(_save_map(tmp_path, _map_from_mesh(mesh)), stl,
                      voxel_mm=1.0, chamber_mm=10.0)


def test_write_spec_round_trips_for_the_stager(tmp_path):
    import json
    mesh = _pyramid_mesh()
    stl = tmp_path / "pyr.stl"
    mesh.export(stl)
    spec = sp.build_spec(_save_map(tmp_path, _map_from_mesh(mesh)), stl, voxel_mm=1.0)
    p = sp.write_spec(spec, tmp_path / "s.npz")
    d = np.load(p, allow_pickle=False)                 # no pickle needed to stage it
    assert str(d["proxy_field"]) == "solve"
    assert float(d["domain_mm"]) == pytest.approx(spec["domain_mm"])
    assert json.loads(str(d["registration"]))["pose_ok"] is True


def test_register_inverted_winding_uses_absolute_volume(tmp_path):
    mesh = _pyramid_mesh()
    m = _map_from_mesh(mesh)
    inv = _pyramid_mesh()
    inv.invert()
    assert inv.volume < 0                                  # trimesh volume is SIGNED
    reg = sp.register(sp.load_map(_save_map(tmp_path, m)), inv)
    assert reg["report"]["volume_stl_mm3"] > 0
    assert reg["report"]["volume_rel_err"] < 0.03
    # a global inversion leaves the centre of mass where it was
    assert np.allclose(inv.center_mass, mesh.center_mass, atol=1e-6)
    w = np.asarray(m["volumes"], float)
    assert np.allclose(np.average(reg["points_mm"], 0, weights=w), inv.center_mass,
                       atol=1e-6)


def test_register_inverted_winding_still_refuses_volume_mismatch(tmp_path):
    inv = _pyramid_mesh()
    inv.invert()
    m = _map_from_mesh(_pyramid_mesh())
    m["volumes"] = m["volumes"] * 0.4
    with pytest.raises(sp.RegistrationError, match="volume"):
        sp.register(sp.load_map(_save_map(tmp_path, m)), inv)


def test_register_refuses_inconsistent_winding(tmp_path):
    good = _pyramid_mesh()
    faces = np.array(good.faces)
    faces[0] = faces[0][::-1]
    bad = trimesh.Trimesh(vertices=np.array(good.vertices), faces=faces, process=False)
    assert not bad.is_winding_consistent
    with pytest.raises(sp.RegistrationError, match="winding"):
        sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(good))), bad)


def test_register_refuses_in_plane_rotation_of_rectangular_base(tmp_path):
    mesh = _pyramid_mesh(b=20.0, b_y=12.0, h=20.0)
    m = _map_from_mesh(mesh, R=_RZ90)                  # solved turned 90 deg on the bed
    with pytest.raises(sp.RegistrationError, match="rotated in plane"):
        sp.register(sp.load_map(_save_map(tmp_path, m)), mesh)
    ok = sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(mesh))), mesh)
    assert ok["report"]["pose_ok"] is True
    assert ok["report"]["pose_symmetry_count"] == 2   # identity + 180 deg about z


def test_register_refuses_in_plane_rotation_the_bbox_cannot_see(tmp_path):
    """A bar on the xy diagonal turned 90 deg keeps the SAME bounding box and zero
    per-axis skew; only the xy cross moment flips sign."""
    rz45 = trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1])
    mesh = trimesh.creation.box((20.0, 6.0, 6.0), transform=rz45)
    m = _map_from_mesh(mesh, R=_RZ90)
    with pytest.raises(sp.RegistrationError, match="rotated in plane"):
        sp.register(sp.load_map(_save_map(tmp_path, m)), mesh)
    assert sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(mesh))),
                       mesh)["report"]["pose_ok"] is True


def test_square_pyramid_correct_pose_reports_fourfold_symmetry(tmp_path):
    mesh = _pyramid_mesh()
    rep = sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(mesh))), mesh)["report"]
    assert rep["pose_ok"] is True
    assert rep["pose_symmetry_count"] == 4
    assert rep["pose_unique"] is False
    assert "4 axis rotations" in rep["pose_note"]
    assert rep["moment_err_2nd"] < sp.SEC_TOL and rep["moment_err_3rd"] < sp.THR_TOL


def test_square_pyramid_rotated_map_registers_but_is_not_unique(tmp_path):
    mesh = _pyramid_mesh()
    m = _map_from_mesh(mesh, R=_RZ90)                  # geometry genuinely cannot tell
    rep = sp.register(sp.load_map(_save_map(tmp_path, m)), mesh)["report"]
    assert rep["pose_ok"] is True
    assert rep["pose_unique"] is False
    assert rep["pose_symmetry_count"] == 4


def test_cube_reports_full_axis_symmetry(tmp_path):
    mesh = trimesh.creation.box((16.0, 16.0, 16.0))
    rep = sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(mesh))), mesh)["report"]
    assert rep["pose_ok"] is True
    assert rep["pose_symmetry_count"] == 24
    assert rep["pose_unique"] is False


def test_asymmetric_part_pose_is_unique(tmp_path):
    mesh = _pyramid_mesh(b=20.0, b_y=12.0, h=20.0)
    v = np.array(mesh.vertices)
    v[-1] = [4.0, 2.0, 20.0]                           # apex off-centre: no symmetry left
    mesh = trimesh.convex.convex_hull(v)
    rep = sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(mesh))), mesh)["report"]
    assert rep["pose_symmetry_count"] == 1
    assert rep["pose_unique"] is True
    assert rep["pose_note"] == "pose verified by shape moments"


def test_register_refuses_multi_body_stl(tmp_path):
    pyr = _pyramid_mesh()
    sliver = trimesh.creation.box((1.0, 1.0, 1.0))
    sliver.apply_translation([60.0, 0.0, 0.5])            # a stray body far away
    both = trimesh.util.concatenate([pyr, sliver])
    assert both.is_watertight and both.body_count == 2
    with pytest.raises(sp.RegistrationError, match="bodies"):
        sp.register(sp.load_map(_save_map(tmp_path, _map_from_mesh(pyr))), both)


def test_build_spec_canvas_centre_is_raw_triangle_vertex_mean(tmp_path):
    mesh = _pyramid_mesh(b=20.0, b_y=12.0)
    stl = tmp_path / "pyr.stl"
    mesh.export(stl)                                   # binary STL
    buf = stl.read_bytes()                             # parse it WITHOUT trimesh
    n = int(np.frombuffer(buf, "<u4", 1, 80)[0])
    rec = np.frombuffer(buf, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                       ("a", "<u2")]), n, 84)
    raw_mean = rec["v"].reshape(-1, 3).astype(float).mean(axis=0)
    spec = sp.build_spec(_save_map(tmp_path, _map_from_mesh(mesh)), stl, voxel_mm=1.0)
    assert spec["canvas_center_mm"] == pytest.approx(raw_mean[:2].tolist(), abs=1e-5)


def test_parse_preflight_valid_json():
    assert sp._parse_preflight(0, '[{"ready": true, "errors": [], "warnings": []}]',
                               "")["ready"] is True


def test_parse_preflight_non_json_is_not_ready():
    r = sp._parse_preflight(0, "Warning: blah\nnot json", "")
    assert r["ready"] is False and r["errors"] and r["warnings"] == []


def test_parse_preflight_failure_carries_stderr():
    r = sp._parse_preflight(1, "", "boom")
    assert r["ready"] is False
    assert any("boom" in e for e in r["errors"])


def _cli_setup(tmp_path, layer_height="0.2"):
    tools = tmp_path / "tools"
    tools.mkdir()
    for f in ("stage_job.py", "preflight.py"):
        (tools / f).write_text("")
    mesh = _pyramid_mesh()                             # square base: 4-fold symmetric
    stl = tmp_path / "pyr.stl"
    mesh.export(stl)
    argv = ["--map", str(_save_map(tmp_path, _map_from_mesh(mesh))), "--stl", str(stl),
            "--no-densification", "--hot-folder", str(tmp_path / "hot"),
            "--job-name", "j", "--voxel-mm", "1.0", "--work-dir", str(tmp_path / "w"),
            "--meteor-tools", str(tools), "--meteor-python", sys.executable]
    if layer_height is not None:
        argv += ["--layer-height", layer_height]
    return argv


def _fake_stager(tmp_path, monkeypatch, ready=True, seen=None):
    """Fake subprocess.run for stage_job + preflight. Like the real stage_job, the
    fake writes its job dir <stamp>_FGM_<job> under the --hot-folder it was GIVEN
    and reports that path; the fake preflight answers `ready`."""
    import json
    import subprocess

    def fake_run(cmd, **kw):
        if seen is not None:
            seen.append((cmd, kw))
        if "--report-json" in cmd:
            job = Path(cmd[cmd.index("--hot-folder") + 1]) / (
                "20260101_000000_FGM_" + cmd[cmd.index("--job-name") + 1])
            job.mkdir(parents=True)
            (job / "job_info.json").write_text("{}")
            rj = cmd[cmd.index("--report-json") + 1]
            with open(rj, "w") as fh:
                json.dump({"out_dir": str(job), "all_pass": True}, fh)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        res = {"ready": ready, "errors": [] if ready else ["bad page count"],
               "warnings": []}
        return subprocess.CompletedProcess(cmd, 0, json.dumps([res]), "")
    monkeypatch.setattr(sp.subprocess, "run", fake_run)


def test_cli_requires_layer_height(tmp_path, monkeypatch, capsys):
    """MetPrint prints one page per MACHINE layer and ignores our metadata, so the
    layer height must be the machine's setting (0.1-0.3 mm): never defaulted."""
    def fake_run(cmd, **kw):
        raise AssertionError("must not stage without an explicit --layer-height")
    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        sp.main(_cli_setup(tmp_path, layer_height=None))
    assert "--layer-height" in capsys.readouterr().err


def test_cli_timeout_fails_cleanly(tmp_path, monkeypatch, capsys):
    import subprocess

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    assert sp.main(_cli_setup(tmp_path)) == 1
    assert "STAGE FAILED (timeout" in capsys.readouterr().err


def test_cli_warns_when_pose_not_unique(tmp_path, monkeypatch, capsys):
    import json
    seen = []
    _fake_stager(tmp_path, monkeypatch, seen=seen)
    assert sp.main(_cli_setup(tmp_path)) == 0
    cap = capsys.readouterr()
    assert [kw.get("timeout") for _, kw in seen] == [1800, 1800]
    assert "WARNING: part is symmetric under 4 axis rotations" in cap.err
    out = json.loads(cap.out)
    assert "4 axis rotations" in out["pose_note"] and out["preflight_ready"] is True


def test_cli_hands_the_stager_absolute_paths(tmp_path, monkeypatch):
    """stage_job runs with cwd = the tools dir, so every path the driver passes it
    must be absolute. Regression: a relative --stl (as typed at a shell prompt)
    was forwarded verbatim and stage_job could not find it."""
    argv = _cli_setup(tmp_path)
    rel = [a.replace(str(tmp_path) + os.sep, "") for a in argv]   # user types relative paths
    monkeypatch.chdir(tmp_path)
    seen = []
    _fake_stager(tmp_path, monkeypatch, seen=seen)
    assert sp.main(rel) == 0
    stage_cmd = next(c for c, _ in seen if "--report-json" in c)
    assert Path(stage_cmd[2]).is_absolute()                          # the spec npz
    for flag in ("--stl", "--hot-folder", "--report-json"):
        assert Path(stage_cmd[stage_cmd.index(flag) + 1]).is_absolute(), flag
    assert Path(stage_cmd[stage_cmd.index("--stl") + 1]).is_file()


def _fields_from_mesh(tmp_path, mesh, h_mm=1.25, name="fields.npz"):
    """Synthetic densify-march fields.npz: the mesh voxelised on a coarse grid the
    way the march stores it -- part (nx, ny, nz) bool with z = build axis, voxel
    centres at (idx + 0.5) * h, h in metres. Placed at an arbitrary grid offset."""
    lo, hi = mesh.bounds
    n = [int(np.ceil((hi[i] - lo[i]) / h_mm)) + 4 for i in range(3)]
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1)
    P = (idx.reshape(-1, 3) + 0.5) * h_mm + (lo - 2.0 * h_mm)
    part = mesh.contains(P).reshape(n)
    p = tmp_path / name
    np.savez(p, part=part, h=np.float64(h_mm / 1e3), rho_final=np.full(n, 0.9))
    return p


def test_march_check_accepts_the_same_part(tmp_path):
    mesh = _pyramid_mesh()
    rep = sp.check_march_matches_stl(_fields_from_mesh(tmp_path, mesh), mesh)
    assert rep["moment_err_2nd"] < sp.MARCH_SEC_TOL
    assert rep["moment_err_3rd"] < sp.MARCH_THR_TOL
    assert rep["extent_ok"] is True
    assert rep["h_mm"] == pytest.approx(1.25)
    assert rep["march_extent_mm"][2] == pytest.approx(20.0, abs=2 * 1.25)


def test_march_check_refuses_an_equal_volume_box(tmp_path):
    """The real failure: the shape library is equal-VOLUME, so a cube march staged
    with a pyramid map + STL passes every volume check and prints too tall."""
    pyr = _pyramid_mesh()
    side = abs(pyr.volume) ** (1.0 / 3.0)
    box = trimesh.creation.box((side, side, side))
    assert abs(box.volume) == pytest.approx(abs(pyr.volume))
    with pytest.raises(sp.RegistrationError):
        sp.check_march_matches_stl(_fields_from_mesh(tmp_path, box), pyr)


def test_march_check_refuses_a_box_with_the_same_bounding_box(tmp_path):
    """Same extents, different shape: only the moments can tell."""
    pyr = _pyramid_mesh()
    box = trimesh.creation.box((20.0, 20.0, 20.0))
    with pytest.raises(sp.RegistrationError, match="moment"):
        sp.check_march_matches_stl(_fields_from_mesh(tmp_path, box), pyr)


def test_march_check_refuses_wrong_physical_size(tmp_path):
    """Normalised moments are scale-invariant; the physical extents are not."""
    pyr = _pyramid_mesh()
    big = _pyramid_mesh(b=26.0, h=26.0)                # same shape, 1.3x the size
    with pytest.raises(sp.RegistrationError, match="extent"):
        sp.check_march_matches_stl(_fields_from_mesh(tmp_path, big), pyr)


def test_march_check_refuses_an_upside_down_march(tmp_path):
    pyr = _pyramid_mesh()
    flipped = pyr.copy()
    flipped.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    with pytest.raises(sp.RegistrationError, match="moment"):
        sp.check_march_matches_stl(_fields_from_mesh(tmp_path, flipped), pyr)


def test_cli_refuses_a_densify_march_of_another_part(tmp_path, monkeypatch, capsys):
    def fake_run(cmd, **kw):
        raise AssertionError("must not stage with another part's densify march")
    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    side = abs(_pyramid_mesh().volume) ** (1.0 / 3.0)
    fields = _fields_from_mesh(tmp_path, trimesh.creation.box((side, side, side)))
    argv = _cli_setup(tmp_path)
    argv[argv.index("--no-densification")] = "--densify"
    argv.insert(argv.index("--densify") + 1, str(fields))
    assert sp.main(argv) == 1
    assert "REFUSED: densify march is not this part" in capsys.readouterr().err


def test_cli_reports_the_densify_match(tmp_path, monkeypatch, capsys):
    import json
    from solve3d import densify_summary as ds

    def fake_summary(fields_npz, out_json=None, xy_frac=0.04):
        Path(out_json).write_text("{}")
        return Path(out_json)
    monkeypatch.setattr(ds, "write_summary", fake_summary)
    _fake_stager(tmp_path, monkeypatch)
    argv = _cli_setup(tmp_path)
    argv[argv.index("--no-densification")] = "--densify"
    argv.insert(argv.index("--densify") + 1,
                str(_fields_from_mesh(tmp_path, _pyramid_mesh())))
    assert sp.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["densify_match"]["extent_ok"] is True
    assert out["densify_match"]["moment_err_2nd"] < sp.MARCH_SEC_TOL


_JOB = "20260101_000000_FGM_j"                         # what _fake_stager writes


def test_cli_stages_off_the_hot_folder_then_moves_a_ready_job_in(tmp_path, monkeypatch,
                                                                 capsys):
    """MetPrint must never see a half-written job: stage + preflight in
    <work>/staging, and only a READY job is moved (atomic rename) into the hot folder."""
    import json
    seen = []
    _fake_stager(tmp_path, monkeypatch, seen=seen)
    assert sp.main(_cli_setup(tmp_path)) == 0
    stage_cmd = next(c for c, _ in seen if "--report-json" in c)
    pre_cmd = next(c for c, _ in seen if "--report-json" not in c)
    staging = tmp_path / "w" / "staging"
    assert Path(stage_cmd[stage_cmd.index("--hot-folder") + 1]) == staging
    assert Path(pre_cmd[2]) == staging / _JOB             # preflight ran off the hot folder
    hot = tmp_path / "hot"
    assert (hot / _JOB / "job_info.json").is_file()
    assert not (staging / _JOB).exists()
    assert json.loads(capsys.readouterr().out)["out_dir"] == str(hot / _JOB)


def test_cli_never_leaves_a_rejected_job_in_the_hot_folder(tmp_path, monkeypatch, capsys):
    import json
    _fake_stager(tmp_path, monkeypatch, ready=False)
    assert sp.main(_cli_setup(tmp_path)) == 1
    hot = tmp_path / "hot"
    assert not hot.exists() or not any(hot.iterdir())
    rejected = tmp_path / "w" / "_rejected" / _JOB
    assert (rejected / "job_info.json").is_file()
    assert not (tmp_path / "w" / "staging" / _JOB).exists()
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert out["out_dir"] == str(rejected)
    assert out["preflight_ready"] is False and out["preflight_errors"]
    assert "bad page count" in cap.err


def test_cli_refuses_to_overwrite_a_job_already_in_the_hot_folder(tmp_path, monkeypatch,
                                                                  capsys):
    existing = tmp_path / "hot" / _JOB
    existing.mkdir(parents=True)
    (existing / "marker").write_text("theirs")
    _fake_stager(tmp_path, monkeypatch)
    assert sp.main(_cli_setup(tmp_path)) == 1
    assert (existing / "marker").read_text() == "theirs"
    assert sorted(p.name for p in existing.iterdir()) == ["marker"]
    assert "already exists" in capsys.readouterr().err


def test_cli_refuses_to_move_a_job_staged_outside_staging(tmp_path, monkeypatch, capsys):
    """Only ever move what WE staged: a report pointing elsewhere is not moved."""
    import json
    import subprocess
    elsewhere = tmp_path / "elsewhere" / _JOB
    elsewhere.mkdir(parents=True)

    def fake_run(cmd, **kw):
        if "--report-json" in cmd:
            with open(cmd[cmd.index("--report-json") + 1], "w") as fh:
                json.dump({"out_dir": str(elsewhere), "all_pass": True}, fh)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(
            cmd, 0, '[{"ready": true, "errors": [], "warnings": []}]', "")
    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    assert sp.main(_cli_setup(tmp_path)) == 1
    assert elsewhere.is_dir()
    hot = tmp_path / "hot"
    assert not hot.exists() or not any(hot.iterdir())
    assert "outside" in capsys.readouterr().err


def test_cli_requires_exactly_one_densification_choice():
    base = ["--map", "m", "--stl", "s", "--hot-folder", "h", "--job-name", "j",
            "--layer-height", "0.2"]
    with pytest.raises(SystemExit):
        sp.main(base)
    with pytest.raises(SystemExit):
        sp.main(base + ["--no-densification", "--densify-factor", "1.5"])


# --------------------------------------------------------------------------- #
# large parts: the voxel fill is one ray per column, not per voxel
# --------------------------------------------------------------------------- #
def _grid_for(mesh, vox, pad=1.25):
    lo, hi = mesh.bounds
    c = (lo + hi) / 2
    ch = float(np.max(hi[:2] - lo[:2])) * pad
    n = int(round(ch / vox))
    xs = c[0] - ch / 2 + (np.arange(n) + 0.5) * ch / n
    ys = c[1] - ch / 2 + (np.arange(n) + 0.5) * ch / n
    nz = int(np.ceil((hi[2] - lo[2]) / vox))
    zs = lo[2] + (np.arange(nz) + 0.5) * (hi[2] - lo[2]) / nz
    return xs, ys, zs


def _contains_grid(mesh, xs, ys, zs):
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return mesh.contains(P).reshape(len(zs), len(ys), len(xs))


def _annulus():
    return trimesh.creation.annulus(r_min=4.0, r_max=9.0, height=12.0)


@pytest.mark.parametrize("make", [
    lambda: _pyramid_mesh(),
    lambda: trimesh.creation.box((16.0, 16.0, 16.0)),          # faces on the grid
    lambda: trimesh.creation.box(
        (20.0, 6.0, 6.0), transform=trimesh.transformations.rotation_matrix(
            np.pi / 4, [0, 0, 1])),
    _annulus,                                                   # a hole: 4 hits a ray
    lambda: trimesh.creation.icosphere(subdivisions=2, radius=9.0),
])
def test_inside_grid_matches_per_voxel_contains(make):
    mesh = make()
    xs, ys, zs = _grid_for(mesh, 1.0)
    got = sp._inside_grid(mesh, xs, ys, zs)
    ref = _contains_grid(mesh, xs, ys, zs)
    assert got.shape == ref.shape == (len(zs), len(ys), len(xs))
    assert ref.sum() > 100
    # a voxel centre lying exactly on the surface is ambiguous either way
    assert (got != ref).sum() <= max(1, ref.sum() // 1000)


def test_inside_grid_settles_unpaired_columns_exactly(monkeypatch):
    """A column whose ray hits do not pair up must not be guessed: it is decided
    point by point with mesh.contains."""
    mesh = _pyramid_mesh()
    xs, ys, zs = _grid_for(mesh, 1.0)
    real = mesh.ray.intersects_location

    def drop_one_hit(origins, dirs, multiple_hits=True):
        loc, ray, tri = real(origins, dirs, multiple_hits=multiple_hits)
        keep = np.ones(len(ray), bool)
        keep[int(np.argmax(ray == ray[len(ray) // 2]))] = False   # unpair one column
        return loc[keep], ray[keep], tri[keep]
    monkeypatch.setattr(mesh.ray, "intersects_location", drop_one_hit)
    got = sp._inside_grid(mesh, xs, ys, zs)
    ref = _contains_grid(mesh, xs, ys, zs)
    assert (got != ref).sum() <= 1


def test_build_spec_large_part_never_tests_every_voxel(tmp_path, monkeypatch):
    """A 60 mm part at 0.5 mm is ~2.6 M canvas voxels. Per-voxel containment took
    minutes and ~GBs; the column fill must touch mesh.contains only for the few
    columns it cannot pair (here: none)."""
    mesh = _pyramid_mesh(b=60.0, h=40.0)
    stl = tmp_path / "big.stl"
    mesh.export(stl)
    m = _save_map(tmp_path, _map_from_mesh(mesh, cell_mm=2.0))
    calls = []
    real_contains = trimesh.Trimesh.contains

    def counting(self, points):
        calls.append(len(points))
        return real_contains(self, points)
    monkeypatch.setattr(trimesh.Trimesh, "contains", counting)
    spec = sp.build_spec(m, stl, voxel_mm=0.5)
    sol, part = spec["SOLVE_cont"], spec["part_mask"]
    assert part.shape == (80, 150, 150)
    assert sum(calls) < 0.1 * part.size, calls       # register() samples 40^3 itself
    vol = part.sum() * np.prod(spec["voxel_mm"])
    assert vol == pytest.approx(abs(mesh.volume), rel=0.03)
    assert np.all(sol[~part] == 0) and sol[part].max() <= 1.0


# --------------------------------------------------------------------------- #
# portability: interpreter, UTF-8 pipes, cross-volume delivery
# --------------------------------------------------------------------------- #
def test_meteor_python_flag_runs_both_tools(tmp_path, monkeypatch):
    seen = []
    _fake_stager(tmp_path, monkeypatch, seen=seen)
    argv = _cli_setup(tmp_path)
    argv[argv.index("--meteor-python") + 1] = "/opt/meteor/python"
    assert sp.main(argv) == 0
    assert [c[0] for c, _ in seen] == [str(Path("/opt/meteor/python"))] * 2


def test_meteor_python_env_then_venv_then_self(tmp_path, monkeypatch):
    monkeypatch.setenv("RFAM_METEOR_PYTHON", "/x/py")
    assert sp._default_meteor_python() == "/x/py"
    monkeypatch.delenv("RFAM_METEOR_PYTHON")
    monkeypatch.setattr(sp.Path, "home", classmethod(lambda cls: tmp_path))
    assert sp._default_meteor_python() == sys.executable      # no venv: this python
    venv = tmp_path / ".venvs" / "meteor-tools"
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert Path(sp._default_meteor_python()) == exe


def test_venv_python_finds_either_layout(tmp_path):
    win = tmp_path / "w" / "Scripts" / "python.exe"
    win.parent.mkdir(parents=True)
    win.write_text("")
    assert sp._venv_python(tmp_path / "w") == win               # a Windows-made venv
    posix = tmp_path / "p" / "bin" / "python"
    posix.parent.mkdir(parents=True)
    posix.write_text("")
    assert sp._venv_python(tmp_path / "p") == posix


def test_tools_run_with_utf8_pipes(tmp_path, monkeypatch):
    seen = []
    _fake_stager(tmp_path, monkeypatch, seen=seen)
    assert sp.main(_cli_setup(tmp_path)) == 0
    for _, kw in seen:
        assert kw["encoding"] == "utf-8" and kw["errors"] == "replace"
        assert kw["env"]["PYTHONIOENCODING"] == "utf-8"
        assert kw["cwd"] == str(tmp_path / "tools")


_FAKE_STAGE_JOB = r"""
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("spec"); ap.add_argument("--3d", action="store_true")
for f in ("--stl", "--chamber-mm", "--layer-height", "--hot-folder", "--job-name",
          "--report-json", "--densify-summary", "--z-densification"):
    ap.add_argument(f)
a = ap.parse_args()
assert os.path.isfile(a.spec) and os.path.isfile(a.stl)
job = os.path.join(a.hot_folder, "20260101_000000_FGM_" + a.job_name)
os.makedirs(os.path.join(job, "print_job"))
for k in range(3):
    open(os.path.join(job, "print_job", f"layer_{k:04d}.tif"), "wb").write(b"II*\0")
with open(os.path.join(job, "job_info.json"), "w", encoding="utf-8") as fh:
    json.dump({"layer_count": 3, "chamber_mm": float(a.chamber_mm)}, fh)
print(f"staged {a.job_name}: canvas {a.chamber_mm} mm ≥ part → 3 pages ±0.1 mm")
with open(a.report_json, "w", encoding="utf-8") as fh:
    json.dump({"out_dir": job, "all_pass": True, "print_layers": 3}, fh)
"""
_FAKE_PREFLIGHT = r"""
import json, os, sys
job = sys.argv[1]
ok = os.path.isfile(os.path.join(job, "job_info.json"))
print(json.dumps([{"ready": ok, "errors": [], "warnings": ["dose ≤ 7 ✓"]}],
                 ensure_ascii=False))
"""


def _real_tools(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "stage_job.py").write_text(_FAKE_STAGE_JOB, encoding="utf-8")
    (tools / "preflight.py").write_text(_FAKE_PREFLIGHT, encoding="utf-8")
    return tools


def test_real_child_processes_on_this_platform(tmp_path, monkeypatch, capsys):
    """No subprocess mock: the stand-in tools run as real child interpreters with
    cwd = the tools dir, print non-ASCII (which kills a cp1252 pipe on Windows
    without the UTF-8 env), and the job is moved into the hot folder whole."""
    import json
    argv = _cli_setup(tmp_path)
    _real_tools(tmp_path)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    assert sp.main(argv) == 0, capsys.readouterr().err
    out = json.loads(capsys.readouterr().out)
    job = tmp_path / "hot" / "20260101_000000_FGM_j"
    assert out["preflight_ready"] is True and Path(out["out_dir"]) == job.resolve()
    assert out["preflight_warnings"] == ["dose ≤ 7 ✓"]
    assert len(list((job / "print_job").iterdir())) == 3
    assert out["chamber_mm"] == json.loads((job / "job_info.json").read_text())[
        "chamber_mm"]


def _cross_volume_rename(monkeypatch, hot, err):
    """os.rename that refuses any move from outside `hot`'s volume stand-in: a
    rename INTO the hot folder only works from a sibling of it."""
    real = os.rename
    moves = []

    def rename(src, dst):
        src, dst = Path(src), Path(dst)
        moves.append((src, dst))
        if dst.parent == hot and src.parent.parent != hot.parent:
            raise err
        return real(src, dst)
    monkeypatch.setattr(sp.os, "rename", rename)
    return moves


@pytest.mark.parametrize("kind", ["posix", "windows"])
def test_cross_volume_delivery_never_exposes_a_partial_job(tmp_path, monkeypatch,
                                                           capsys, kind):
    """Work dir on one drive, hot folder on another (Windows C: vs a share, Linux
    /tmp vs /home): the rename is refused, and the job must be assembled beside
    the hot folder and renamed in, never copied file by file into it."""
    import errno
    import json
    hot = tmp_path / "hot"
    if kind == "posix":
        err = OSError(errno.EXDEV, "Invalid cross-device link")
    else:
        err = OSError(0, "The system cannot move the file to a different disk drive")
        err.winerror = 17
    _fake_stager(tmp_path, monkeypatch)
    copied_into = []
    real_copytree = sp.shutil.copytree

    def copytree(src, dst, *a, **k):
        copied_into.append(Path(dst))
        return real_copytree(src, dst, *a, **k)
    monkeypatch.setattr(sp.shutil, "copytree", copytree)
    moves = _cross_volume_rename(monkeypatch, hot, err)
    assert sp.main(_cli_setup(tmp_path)) == 0, capsys.readouterr().err
    assert all(hot not in d.parents for d in copied_into), copied_into
    final = [m for m in moves if m[1] == hot / _JOB]
    assert final[-1][0].parent.name == ".hot.incoming"
    assert (hot / _JOB / "job_info.json").is_file()
    assert not (tmp_path / ".hot.incoming").exists()
    assert not (tmp_path / "w" / "staging" / _JOB).exists()
    assert json.loads(capsys.readouterr().out)["out_dir"] == str(hot / _JOB)


def test_failed_move_keeps_the_job_and_says_so(tmp_path, monkeypatch, capsys):
    _fake_stager(tmp_path, monkeypatch)

    def rename(src, dst):
        raise PermissionError(13, "Access is denied")
    monkeypatch.setattr(sp.os, "rename", rename)
    assert sp.main(_cli_setup(tmp_path)) == 1
    assert (tmp_path / "w" / "staging" / _JOB / "job_info.json").is_file()
    assert "could not move the job" in capsys.readouterr().err


def test_same_path_is_case_and_link_tolerant(tmp_path):
    a = tmp_path / "Staging"
    a.mkdir()
    assert sp._same_path(a, tmp_path / "x" / ".." / "Staging")
    assert not sp._same_path(a, tmp_path)
