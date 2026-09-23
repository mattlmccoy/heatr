import numpy as np
import pytest
import trimesh

from solve3d import stage_part as sp


def _pyramid_mesh(b=20.0, h=20.0):
    v = np.array([[-b / 2, -b / 2, 0], [b / 2, -b / 2, 0], [b / 2, b / 2, 0],
                  [-b / 2, b / 2, 0], [0, 0, h]], float)
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
    assert rep["orientation_checked"] and rep["orientation_ok"]
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
    assert json.loads(str(d["registration"]))["orientation_ok"] is True


def test_cli_requires_exactly_one_densification_choice():
    base = ["--map", "m", "--stl", "s", "--hot-folder", "h", "--job-name", "j"]
    with pytest.raises(SystemExit):
        sp.main(base)
    with pytest.raises(SystemExit):
        sp.main(base + ["--no-densification", "--densify-factor", "1.5"])
