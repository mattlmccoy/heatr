"""Capture the REAL Grade-and-Print job artifacts as test fixtures.

Data-contract rule: fixtures are CAPTURED from real jobs, never authored from
belief. This script copies the actual results.json pairs and recomputes the
field-derived statistics (out-of-part melt fraction, proxy spreads, proxy
saturation fractions, emitted-map zero fractions) from the actual fields.npz /
correction_sat.npz, recording the absolute source path and a sha256 of every
file it read so the fixture is auditable.

Run:
    OMP_NUM_THREADS=1 .venv312/bin/python \
        studio3d/tests/fixtures/capture_tamper_fixtures.py

Source jobs (uploads/<id>/grade/), read-only:
    feb850ec  Tamper   -- the harm case in TAMPER_DIAGNOSIS.md
    c474d787  AIRCOIL  -- second harm case (sigma_T tripled, clamp bound)
    d4d50045           -- third harm case, found during capture
    2ffdfc3c  tube     -- healthy-contrast proxy fixture (uncorrected arm)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

UPLOADS = Path("/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/"
               "mattmccoy-research/research/binderjet/software/meteor/tools/"
               "uploads")
HERE = Path(__file__).resolve().parent / "tamper"
JOBS = {"feb850ec": "Tamper", "c474d787": "AIRCOIL", "d4d50045": "job_d4d50045",
        "2ffdfc3c": "tube"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _atom_ratio(v: np.ndarray) -> float:
    """Occupancy of the MAX value divided by the typical occupancy of the
    other distinct values near the top.

    Grid-independent, unlike a bare frac-at-max: a smoothly varying field
    ties 1/(number of levels) of its voxels at the top whatever the grid, so
    frac_at_max alone reads 0.167 at n=16 and 0.046 at n=64 for the SAME
    field. The ratio is 1.0 for any smooth field at any grid, and large only
    when a clip or a stop target has piled an ATOM of voxels at one value.
    """
    vals, counts = np.unique(np.round(v, 12), return_counts=True)
    if len(vals) < 2:
        return float("inf")
    f = counts / counts.sum()
    others = f[-6:-1] if len(f) >= 6 else f[:-1]
    typ = float(np.median(others)) if len(others) else 0.0
    return float(f[-1] / typ) if typ > 0 else float("inf")


def _field_stats(fp: Path) -> dict:
    """Everything the gate and the degeneracy refusal need, recomputed from
    the real field arrays."""
    out: dict = {}
    with np.load(fp) as d:
        part = d["part"].astype(bool)
        out["grid_n"] = int(part.shape[0])
        out["n_part_voxels"] = int(part.sum())
        if "phi_final" in d:
            phi = np.asarray(d["phi_final"], float)
            if phi.shape == part.shape:
                # THE dense-iff-in-bounds hard-failure side: melt outside the
                # part is bed spill, and must never regress silently.
                out["out_of_part_melt_frac"] = float(phi[~part].mean())
                out["out_of_part_melted_voxel_frac"] = float(
                    (phi[~part] > 0.5).mean())
                out["in_part_melt_frac"] = float(phi[part].mean())
        for name in ("rho_final", "T_phi90"):
            if name not in d:
                continue
            a = np.asarray(d[name], float)
            if a.shape != part.shape:
                continue
            v = a[part]
            lo, hi = np.percentile(v, [2.0, 98.0])
            out[name] = {
                "p2": float(lo), "p98": float(hi), "spread": float(hi - lo),
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "std": float(v.std()),
                # the statistics that actually separate the harm cases
                "frac_at_max": float(np.mean(np.isclose(v, v.max(), rtol=0,
                                                        atol=1e-9))),
                "atom_ratio": _atom_ratio(v),
            }
    return out


def capture() -> dict:
    HERE.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "captured_by": "studio3d/tests/fixtures/capture_tamper_fixtures.py",
        "rule": ("fixtures CAPTURED from real jobs; statistics recomputed from "
                 "the real fields.npz, never authored"),
        "uploads_root": str(UPLOADS),
        "jobs": {},
    }
    for job, label in JOBS.items():
        g = UPLOADS / job / "grade" / "heatr3d"
        rec: dict = {"label": label, "source": str(g), "arms": {}}
        for arm in ("uncorrected", "corrected"):
            a: dict = {}
            rp = g / arm / "results.json"
            if rp.exists():
                a["results"] = json.loads(rp.read_text())
                a["results_sha256_16"] = _sha(rp)
                (HERE / f"{job}_{arm}_results.json").write_text(rp.read_text())
            else:
                # Recorded, not silently skipped: the tube job has NO
                # uncorrected results.json, so it cannot be a before/after pair.
                a["results"] = None
                a["results_absent"] = True
            fp = g / arm / "fields.npz"
            if fp.exists():
                a["fields"] = _field_stats(fp)
                a["fields_sha256_16"] = _sha(fp)
            rec["arms"][arm] = a
        pp = g / "correction_provenance.json"
        if pp.exists():
            rec["correction_provenance"] = json.loads(pp.read_text())
        sp = g / "correction_sat.npz"
        fu = g / "uncorrected" / "fields.npz"
        if sp.exists() and fu.exists():
            with np.load(fu) as d:
                part = d["part"].astype(bool)
            with np.load(sp) as s:
                sat = np.asarray(s["sat"], float)
            if sat.shape == part.shape:
                sv = sat[part]
                rec["emitted_sat"] = {
                    "zero_frac": float(np.mean(sv <= 1e-12)),
                    "mean": float(sv.mean()), "std": float(sv.std()),
                }
        manifest["jobs"][job] = rec
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                   default=float))
    return manifest


if __name__ == "__main__":
    m = capture()
    print(f"captured {len(m['jobs'])} jobs into {HERE}")
    for job, rec in m["jobs"].items():
        u = rec["arms"]["uncorrected"]
        c = rec["arms"]["corrected"]
        pair = (u.get("results") is not None) and (c.get("results") is not None)
        print(f"  {job} ({rec['label']}): complete pair={pair}"
              + ("" if pair else "   <-- NOT a before/after pair"))
