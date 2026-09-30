import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from solve3d import stage_batch as sb
from solve3d import stage_part as sp
from solve3d.tests.test_stage_part import (_FAKE_PREFLIGHT, _FAKE_STAGE_JOB,
                                           _map_from_mesh, _pyramid_mesh)


def _part(root, name, mesh, map_scale=1.0):
    d = root / name
    d.mkdir(parents=True)
    mesh.export(d / f"{name}.stl")
    m = _map_from_mesh(mesh, cell_mm=1.0)
    m["volumes"] = m["volumes"] * map_scale
    np.savez(d / "map.npz", **m)


def _manifest(tmp_path, jobs, **defaults):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "stage_job.py").write_text(_FAKE_STAGE_JOB, encoding="utf-8")
    (tools / "preflight.py").write_text(_FAKE_PREFLIGHT, encoding="utf-8")
    base = {"hot_folder": "hot", "layer_height": 0.2, "voxel_mm": 1.0,
            "no_densification": True, "meteor_tools": "tools",
            "meteor_python": sys.executable}
    base.update(defaults)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"defaults": base, "jobs": jobs}), encoding="utf-8")
    return p


def _job(name, **kw):
    return {"job_name": name, "map": f"{name}/map.npz", "stl": f"{name}/{name}.stl",
            "work_dir": f"work/{name}", **kw}


def test_batch_stages_every_part_as_its_own_job(tmp_path, capsys):
    """Two geometries, one of them 2x the size in a bigger chamber, from one
    manifest with paths relative to the manifest (portable between machines)."""
    _part(tmp_path, "pyr", _pyramid_mesh())
    _part(tmp_path, "cube2x", trimesh.creation.box((32.0, 32.0, 32.0)))
    man = _manifest(tmp_path, [_job("pyr"), _job("cube2x", chamber_mm=48)])
    summary_path = tmp_path / "out" / "summary.json"
    rc = sb.main([str(man), "--summary", str(summary_path)])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    s = json.loads(cap.out)
    assert s == json.loads(summary_path.read_text(encoding="utf-8"))
    assert (s["n_jobs"], s["n_ready"]) == (2, 2)
    hot = tmp_path / "hot"
    assert sorted(p.name for p in hot.iterdir()) == [
        "20260101_000000_FGM_cube2x", "20260101_000000_FGM_pyr"]
    big = next(j for j in s["jobs"] if j["job"] == "cube2x")["result"]
    assert big["chamber_mm"] == 48.0
    info = json.loads((hot / "20260101_000000_FGM_cube2x" / "job_info.json").read_text())
    assert info["chamber_mm"] == 48.0


def test_batch_one_refused_part_does_not_stop_the_rest(tmp_path, capsys):
    _part(tmp_path, "bad", _pyramid_mesh(), map_scale=1.3)     # map is not this STL
    _part(tmp_path, "good", _pyramid_mesh())
    man = _manifest(tmp_path, [_job("bad"), _job("good")])
    assert sb.main([str(man)]) == 1
    cap = capsys.readouterr()
    s = json.loads(cap.out)
    assert [(j["job"], j["ready"]) for j in s["jobs"]] == [("bad", False), ("good", True)]
    assert "REFUSED" in cap.err
    assert [p.name for p in (tmp_path / "hot").iterdir()] == ["20260101_000000_FGM_good"]


def test_batch_only_runs_the_named_jobs(tmp_path, capsys):
    _part(tmp_path, "a", _pyramid_mesh())
    _part(tmp_path, "b", _pyramid_mesh())
    man = _manifest(tmp_path, [_job("a"), _job("b")])
    assert sb.main([str(man), "--only", "b"]) == 0
    assert [j["job"] for j in json.loads(capsys.readouterr().out)["jobs"]] == ["b"]
    assert sb.main([str(man), "--only", "zz"]) == 2


def test_batch_bad_entry_is_a_failed_job_not_a_crash(tmp_path, capsys):
    _part(tmp_path, "a", _pyramid_mesh())
    man = _manifest(tmp_path, [_job("a", layer_height=None), _job("a2", map="a/map.npz",
                                                                  stl="a/a.stl")])
    assert sb.main([str(man)]) == 1
    s = json.loads(capsys.readouterr().out)
    assert [(j["job"], j["ready"]) for j in s["jobs"]] == [("a", False), ("a2", True)]


@pytest.mark.parametrize("jobs,msg", [
    ([], "non-empty"),
    ([{"map": "m"}], "job_name"),
    ([{"job_name": "a"}, {"job_name": "a"}], "duplicate"),
    ([{"job_name": "a", "scale": 2}], "unknown keys"),
])
def test_manifest_is_validated_up_front(tmp_path, jobs, msg):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"jobs": jobs}))
    with pytest.raises(ValueError, match=msg):
        sb.load_manifest(p)


def test_manifest_paths_resolve_against_the_manifest_not_the_cwd(tmp_path, monkeypatch):
    sub = tmp_path / "proj"
    sub.mkdir()
    p = sub / "m.json"
    p.write_text(json.dumps({"defaults": {"hot_folder": "../hf"},
                             "jobs": [{"job_name": "a", "stl": "parts/a.stl",
                                       "map": str(tmp_path / "abs.npz")}]}))
    monkeypatch.chdir(tmp_path)
    (job,) = sb.load_manifest("proj/m.json")
    assert Path(job["stl"]) == (sub / "parts" / "a.stl").resolve()
    assert Path(job["hot_folder"]) == (tmp_path / "hf").resolve()
    assert Path(job["map"]) == tmp_path / "abs.npz"


def test_job_argv_maps_keys_to_stage_part_flags():
    argv = sb.job_argv({"job_name": "a", "no_densification": True, "chamber_mm": 50,
                        "densify": None, "base": "max", "layer_height": 0.2})
    assert argv == ["--job-name", "a", "--no-densification", "--chamber-mm", "50",
                    "--base", "max", "--layer-height", "0.2"]
    # every manifest key is a real stage_part option
    dests = {a.dest for a in sp._parser()._actions}
    assert sb._KNOWN <= dests
