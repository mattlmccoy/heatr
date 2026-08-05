"""Level 0 anisotropic affine shrinkage pre-compensation (spec section 1).

The pre-scale carries MATERIAL shrinkage only. The densification
consolidation model (heatr3d.shrinkage_factors -> studio3d.warped_mesh) owns
the powder-to-solid collapse and must never see these coefficients: the
double-counting guard is the named test at the bottom of this file.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import studio3d.precomp as P

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# config loading
# --------------------------------------------------------------------------- #
def test_loads_the_repo_root_config():
    cfg = P.load_precomp_config()
    assert cfg["schema_version"] == "1.0"
    assert cfg["s_xy"] == 0.030
    assert cfg["s_z_mat"] == 0.020
    assert cfg["material_only"] is True
    assert "pending P1" in cfg["applicability"]


def test_unsupported_schema_version_refused(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"schema_version": "9.9", "s_xy": 0.03,
                             "s_z_mat": 0.02, "material_only": True,
                             "applicability": "x"}))
    with pytest.raises(ValueError, match="schema_version"):
        P.load_precomp_config(p)


def test_material_only_false_refused(tmp_path):
    """material_only False would mean the coefficient includes consolidation,
    which the densify model already marches: refuse rather than double-count."""
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"schema_version": "1.0", "s_xy": 0.03,
                             "s_z_mat": 0.02, "material_only": False,
                             "applicability": "x"}))
    with pytest.raises(ValueError, match="material_only"):
        P.load_precomp_config(p)


def test_missing_config_refused_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.load_precomp_config(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# the compensation form
# --------------------------------------------------------------------------- #
def test_compensation_factors_are_the_inverse_form():
    cfg = P.load_precomp_config()
    f_xy, f_z = P.compensation_factors(cfg)
    assert f_xy == pytest.approx(1.0 / (1.0 - 0.030), rel=1e-15)
    assert f_z == pytest.approx(1.0 / (1.0 - 0.020), rel=1e-15)
    # the round trip is what the form is FOR: enlarge, then shrink by s,
    # and you land on nominal
    assert f_xy * (1.0 - 0.030) == pytest.approx(1.0, rel=1e-15)
    assert f_z * (1.0 - 0.020) == pytest.approx(1.0, rel=1e-15)


def test_compensation_factors_are_not_the_naive_one_plus_s():
    cfg = P.load_precomp_config()
    f_xy, _ = P.compensation_factors(cfg)
    assert f_xy != pytest.approx(1.030, rel=1e-9)     # 1+s is the wrong form


# --------------------------------------------------------------------------- #
# precompensate_stl
# --------------------------------------------------------------------------- #
def _box(path: Path, extents=(20.0, 20.0, 20.0), translate=(5.0, -3.0, 2.0)):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translate)
    m.export(path)
    return m


def test_precompensate_scales_each_axis_by_its_factor(tmp_path):
    src = tmp_path / "box.stl"
    _box(src)
    dst = tmp_path / "box_precomp.stl"
    rec = P.precompensate_stl(src, dst)
    out = trimesh.load_mesh(dst)
    f_xy, f_z = P.compensation_factors(P.load_precomp_config())
    # binary STL stores float32 vertices: 1e-6 relative is the file format's
    # own floor, not slack in the scaling
    assert out.extents[0] == pytest.approx(20.0 * f_xy, rel=1e-6)
    assert out.extents[1] == pytest.approx(20.0 * f_xy, rel=1e-6)
    assert out.extents[2] == pytest.approx(20.0 * f_z, rel=1e-6)
    assert rec["applied"] is True


def test_precompensate_is_about_the_centroid(tmp_path):
    """The anchor is the mesh centroid: the part does not translate."""
    src = tmp_path / "box.stl"
    m = _box(src, translate=(5.0, -3.0, 2.0))
    dst = tmp_path / "box_precomp.stl"
    P.precompensate_stl(src, dst)
    out = trimesh.load_mesh(dst)
    before = m.bounds.mean(axis=0)
    after = out.bounds.mean(axis=0)
    assert np.allclose(after, before, atol=1e-4)   # float32 STL floor


def test_sidecar_records_the_provenance_verbatim(tmp_path):
    src = tmp_path / "box.stl"
    _box(src)
    dst = tmp_path / "box_precomp.stl"
    rec = P.precompensate_stl(src, dst)
    side = Path(str(dst) + ".precomp.json")
    assert side.exists()
    on_disk = json.loads(side.read_text())
    assert on_disk == rec
    cfg = P.load_precomp_config()
    f_xy, f_z = P.compensation_factors(cfg)
    assert on_disk["applied"] is True
    assert on_disk["factors"] == {"f_xy": f_xy, "f_z": f_z}
    assert on_disk["coefficients"] == {"s_xy": cfg["s_xy"],
                                       "s_z_mat": cfg["s_z_mat"]}
    assert on_disk["bands"] == {"s_xy_band": cfg["s_xy_band"],
                                "s_z_mat_band": cfg["s_z_mat_band"]}
    # applicability VERBATIM: the borrowed-SLS caveat must travel with the file
    assert on_disk["applicability"] == cfg["applicability"]
    sha = hashlib.sha256(P.config_path().read_bytes()).hexdigest()
    assert on_disk["source_config_sha256"] == sha


def test_double_application_refused(tmp_path):
    """An input whose own sidecar says applied must never be scaled again."""
    src = tmp_path / "box.stl"
    _box(src)
    once = tmp_path / "once.stl"
    P.precompensate_stl(src, once)
    twice = tmp_path / "twice.stl"
    with pytest.raises(ValueError, match="already"):
        P.precompensate_stl(once, twice)
    assert not twice.exists()


def test_ensure_precompensated_reuses_the_same_file(tmp_path):
    src = tmp_path / "box.stl"
    _box(src)
    work = tmp_path / "grade"
    p1, r1 = P.ensure_precompensated(src, work)
    mtime = Path(p1).stat().st_mtime_ns
    p2, r2 = P.ensure_precompensated(src, work)
    assert p1 == p2 == str(work / "precomp" / "box.stl")
    assert Path(p2).stat().st_mtime_ns == mtime      # reused, not rebuilt
    assert r2["reused"] is True and r1["reused"] is False


def test_ensure_precompensated_rebuilds_when_the_config_changes(tmp_path,
                                                                monkeypatch):
    src = tmp_path / "box.stl"
    _box(src)
    work = tmp_path / "grade"
    P.ensure_precompensated(src, work)
    alt = tmp_path / "alt.json"
    cfg = dict(P.load_precomp_config())
    cfg["s_xy"] = 0.10
    alt.write_text(json.dumps(cfg))
    monkeypatch.setenv(P.CONFIG_ENV, str(alt))
    _, rec = P.ensure_precompensated(src, work)
    assert rec["reused"] is False
    assert rec["coefficients"]["s_xy"] == 0.10


def test_compensated_bbox_mm_reports_the_enlarged_size(tmp_path):
    src = tmp_path / "box.stl"
    _box(src, extents=(58.5, 10.0, 10.0))
    bbox = P.compensated_bbox_mm(src)
    f_xy, f_z = P.compensation_factors(P.load_precomp_config())
    assert bbox[0] == pytest.approx(58.5 * f_xy, rel=1e-9)
    assert bbox[2] == pytest.approx(10.0 * f_z, rel=1e-9)


# --------------------------------------------------------------------------- #
# DOUBLE-COUNTING GUARD (spec section 1, named test)
# --------------------------------------------------------------------------- #
CONSOLIDATION_SOURCES = (REPO_ROOT / "heatr3d.py",
                         REPO_ROOT / "studio3d" / "warped_mesh.py")


def test_consolidation_model_never_imports_precomp():
    """Import-graph assertion: the consolidation path must not reference the
    material-shrinkage layer at all (either direction of coupling would
    double-count the same physical contraction)."""
    for src in CONSOLIDATION_SOURCES:
        text = src.read_text()
        assert "precomp" not in text, f"{src.name} references precomp"
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any("precomp" in nm for nm in names), \
                f"{src.name} imports {names}"
    # and the consolidation entry points still exist (guard is not vacuous)
    import heatr3d as H
    from studio3d.warped_mesh import build_densified_meshes  # noqa: F401
    assert callable(H.shrinkage_factors)


def _densified_signature(monkeypatch=None):
    import heatr3d as H
    from studio3d.warped_mesh import build_densified_meshes
    n, w = 16, 6
    part = np.zeros((n, n, n), bool)
    lo = (n - w) // 2
    part[lo:lo + w, lo:lo + w, lo:lo + w] = True
    rng = np.random.default_rng(0)
    rho = np.where(part, 0.55 + 0.45 * rng.random(part.shape), 0.0)
    phi = np.where(part, 0.9, 0.0)
    grid = H.Grid(n=n)
    p = H.Params()
    solid, powder, info = build_densified_meshes(part, rho, phi, p, grid)
    return (solid.vertices.tobytes(), solid.faces.tobytes(),
            powder.vertices.tobytes(), json.dumps(info, sort_keys=True,
                                                  default=float))


def test_consolidation_output_is_bit_identical_regardless_of_precomp_config(
        tmp_path, monkeypatch):
    """Numerical guard: the densification model never reads the material
    coefficients, so its output cannot move when they do (or vanish)."""
    base = _densified_signature()

    alt = tmp_path / "alt.json"
    cfg = dict(P.load_precomp_config())
    cfg["s_xy"], cfg["s_z_mat"] = 0.25, 0.25       # absurd coefficients
    alt.write_text(json.dumps(cfg))
    monkeypatch.setenv(P.CONFIG_ENV, str(alt))
    assert P.load_precomp_config()["s_xy"] == 0.25   # the override IS live
    assert _densified_signature() == base

    monkeypatch.setenv(P.CONFIG_ENV, str(tmp_path / "does_not_exist.json"))
    with pytest.raises(FileNotFoundError):
        P.load_precomp_config()
    assert _densified_signature() == base
