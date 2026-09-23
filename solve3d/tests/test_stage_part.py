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


def _cli_setup(tmp_path):
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
            "--meteor-tools", str(tools)]
    return argv


def test_cli_timeout_fails_cleanly(tmp_path, monkeypatch, capsys):
    import subprocess

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    assert sp.main(_cli_setup(tmp_path)) == 1
    assert "STAGE FAILED (timeout" in capsys.readouterr().err


def test_cli_warns_when_pose_not_unique(tmp_path, monkeypatch, capsys):
    import json
    import subprocess
    timeouts = []

    def fake_run(cmd, **kw):
        timeouts.append(kw.get("timeout"))
        if "--report-json" in cmd:
            rj = cmd[cmd.index("--report-json") + 1]
            with open(rj, "w") as fh:
                json.dump({"out_dir": str(tmp_path / "out"), "all_pass": True}, fh)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(
            cmd, 0, '[{"ready": true, "errors": [], "warnings": []}]', "")
    monkeypatch.setattr(sp.subprocess, "run", fake_run)
    assert sp.main(_cli_setup(tmp_path)) == 0
    cap = capsys.readouterr()
    assert timeouts == [1800, 1800]
    assert "WARNING: part is symmetric under 4 axis rotations" in cap.err
    out = json.loads(cap.out)
    assert "4 axis rotations" in out["pose_note"] and out["preflight_ready"] is True


def test_cli_requires_exactly_one_densification_choice():
    base = ["--map", "m", "--stl", "s", "--hot-folder", "h", "--job-name", "j"]
    with pytest.raises(SystemExit):
        sp.main(base)
    with pytest.raises(SystemExit):
        sp.main(base + ["--no-densification", "--densify-factor", "1.5"])
