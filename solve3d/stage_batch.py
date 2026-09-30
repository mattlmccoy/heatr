"""Stage several solved parts in one run, each as its own pre-flighted MetPrint job.

    python -m solve3d.stage_batch manifest.json
    python -m solve3d.stage_batch manifest.json --only cube pyramid --summary out.json

The manifest is JSON: shared `defaults` plus one entry per job. Keys are the
stage_part flags with underscores (job_name, map, stl, densify, densify_factor,
no_densification, hot_folder, layer_height, voxel_mm, chamber_mm, build_axis,
base, work_dir, meteor_tools, meteor_python); a job's own keys override the
defaults.

    {"defaults": {"hot_folder": "../Hot Folder", "layer_height": 0.2},
     "jobs": [{"job_name": "cube_1x", "map": "maps/cube.npz", "stl": "stl/cube.stl",
               "densify": "densify_cube/fields.npz"},
              {"job_name": "cube_2x", "map": "maps/cube_2x.npz", "stl": "stl/cube_2x.stl",
               "no_densification": true, "chamber_mm": 50}]}

Relative paths resolve against the manifest's own folder, so one manifest works on
the Mac, the lab Windows PC and Linux as long as it travels with its files. Each
job goes through exactly the same registration, densify and preflight gates as a
single stage_part run; one refused job never stops the rest. Every part stays its
own job, centred on its own canvas: stage_job refuses STL placement transforms
(an offset moves the outline but not the dopant field), so parts are not
co-placed on one bed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from solve3d import stage_part as sp

_PATH_KEYS = {"map", "stl", "densify", "hot_folder", "work_dir", "meteor_tools"}
_KNOWN = _PATH_KEYS | {"job_name", "densify_factor", "no_densification",
                       "layer_height", "voxel_mm", "chamber_mm", "build_axis", "base",
                       "meteor_python"}


def load_manifest(path) -> list:
    """Manifest -> one merged dict per job, with paths made absolute."""
    path = Path(path).expanduser().resolve()
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or not isinstance(d.get("jobs"), list) or not d["jobs"]:
        raise ValueError(f"{path}: needs a non-empty 'jobs' list")
    defaults = d.get("defaults") or {}
    jobs, names = [], set()
    for i, raw in enumerate(d["jobs"]):
        job = {**defaults, **raw}
        unknown = sorted(set(job) - _KNOWN)
        if unknown:
            raise ValueError(f"{path}: job {i}: unknown keys {unknown}")
        name = job.get("job_name")
        if not name:
            raise ValueError(f"{path}: job {i}: missing 'job_name'")
        if name in names:
            raise ValueError(f"{path}: duplicate job_name {name!r}")
        names.add(name)
        for k in _PATH_KEYS & set(job):
            p = Path(str(job[k])).expanduser()
            job[k] = str(p if p.is_absolute() else (path.parent / p).resolve())
        jobs.append(job)
    return jobs


def job_argv(job: dict) -> list:
    """A merged manifest entry -> the stage_part command line."""
    argv = []
    for k, v in job.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else [flag, str(v)]
    return argv


def run(manifest, only=None) -> tuple:
    jobs = load_manifest(manifest)
    if only:
        missing = sorted(set(only) - {j["job_name"] for j in jobs})
        if missing:
            raise ValueError(f"--only names not in the manifest: {missing}")
        jobs = [j for j in jobs if j["job_name"] in only]
    results = []
    for n, job in enumerate(jobs, 1):
        name = job["job_name"]
        print(f"[{n}/{len(jobs)}] staging {name}", file=sys.stderr, flush=True)
        try:
            rc, out = sp.stage(job_argv(job))
        except SystemExit as e:                 # argparse rejected this entry
            rc, out = (e.code if isinstance(e.code, int) else 2), None
        except Exception as e:                  # never let one part sink the batch
            print(f"STAGE FAILED ({name}): {type(e).__name__}: {e}", file=sys.stderr)
            rc, out = 1, None
        results.append({"job": name, "exit": rc, "ready": rc == 0,
                        "out_dir": (out or {}).get("out_dir"), "result": out})
    ok = all(r["ready"] for r in results)
    return (0 if ok else 1), {"manifest": str(Path(manifest).resolve()),
                              "n_jobs": len(results),
                              "n_ready": sum(r["ready"] for r in results),
                              "jobs": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--only", nargs="+", metavar="JOB_NAME",
                    help="stage just these jobs from the manifest")
    ap.add_argument("--summary", help="also write the batch summary JSON here")
    args = ap.parse_args(argv)
    try:
        rc, summary = run(args.manifest, args.only)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    text = json.dumps(summary, indent=2)
    if args.summary:
        out = Path(args.summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    for r in summary["jobs"]:
        print(f"  {'READY  ' if r['ready'] else 'FAILED '} {r['job']}"
              + (f" -> {r['out_dir']}" if r["out_dir"] else ""), file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
