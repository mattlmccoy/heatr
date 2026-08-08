"""Versioned print package per the FROZEN schema 2.0.0 (spec 7b, e8600d1).

One package per planned print: directory + zip under packages/ (the interim
scheduler drop folder). Kept from the 1.0.0 seed: turntable record never
silently absent; production_verify record or explicit not-run; files list
with per-file sha256; pkg_<part>_<stamp> naming. 2.0.0 additions: plural
engine_versions (conditionally required by source_route), correction
provenance with the three-state transfer record, per-arm densify summaries,
exactly-one power settings source.

Rasters: the Studio's STL route rasterizes at PRINTER DPI through the
slicer + grade_tiff_stack production path (720 DPI, WhiteIsZero, graded
levels); the package carries that TIFF job. Never solve-grid rasters.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "2.0.0"
_TRANSFER_STATES = ("measured_and_passed", "measured_and_failed",
                    "transfer_not_applicable")
_REQUIRED_BLOCKS = (
    "schema_version", "created_utc", "engine_versions", "part",
    "correction_provenance", "densify_summary", "power_settings", "raster",
    "turntable", "production_verify", "scheduler", "files",
)

SCHEDULER_NOTE = (
    "packages/ is the scheduler drop folder for now; a real uploader is "
    "blocked on hardware. pkg_<part>_<stamp> naming is the contract.")

TURNTABLE_ADVISORY = (
    "ADVISORY. The per-build turntable recommendation aggregates per-layer "
    "anisotropy (dose-weighted default, worst-layer flagged) and z-coupling "
    "is real; heatr3d verification of the chosen program is the authority. "
    "Program GENERATION lands with the intake-API integration; this package "
    "records the operator's intent.")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _engine_versions() -> Dict[str, str]:
    """Content stamps: reproducible identifiers for what actually ran."""
    out = {}
    for name, rel in (("heatr3d", "heatr3d.py"),
                      ("studio3d", "studio3d/runner.py"),
                      ("stl_compensation_tool",
                       "stl_compensation_tool/pipeline.py")):
        p = ROOT / rel
        out[name] = _sha256_file(p)[:12] if p.exists() else "absent"
    return out


def package_name(part_name: str, stamp: Optional[str] = None) -> str:
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "_-") else "_"
                   for c in part_name)
    return f"pkg_{safe}_{stamp}"


def validate_manifest(m: Dict[str, Any]) -> list[str]:
    """Problems list; empty means valid against the frozen schema."""
    errs: list[str] = []
    for key in _REQUIRED_BLOCKS:
        if key not in m:
            errs.append(f"missing required block: {key}")
    if errs:
        return errs
    if m["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {m['schema_version']!r} is not "
                    f"{SCHEMA_VERSION!r}")
    part = m["part"]
    for key in ("name", "source_geometry_sha256", "source_route", "intake"):
        if key not in part:
            errs.append(f"part.{key} missing")
    if len(str(part.get("source_geometry_sha256", ""))) != 64:
        errs.append("part.source_geometry_sha256 is not a sha256 digest")
    pw = m["power_settings"]
    has_pd = "power_density_w_per_m3" in pw
    has_v = "voltage_v" in pw
    if has_pd == has_v:
        errs.append("power_settings must carry exactly one of "
                    "power_density_w_per_m3 or the voltage block")
    cp = m["correction_provenance"]
    if cp.get("transfer", {}).get("state") not in _TRANSFER_STATES:
        errs.append("correction_provenance.transfer.state must be one of "
                    f"{_TRANSFER_STATES} (absence never implies "
                    "not-applicable)")
    if "engine" not in cp:
        errs.append("correction_provenance.engine missing")
    tt = m["turntable"]
    if tt.get("mode") not in ("program", "static"):
        errs.append("turntable.mode must be 'program' or 'static'")
    if not tt.get("file"):
        errs.append("turntable.file missing (never silently absent)")
    if "run" not in m["production_verify"]:
        errs.append("production_verify.run missing (not-run must be "
                    "explicit)")
    for f in m["files"]:
        if len(str(f.get("sha256", ""))) != 64:
            errs.append(f"files entry {f.get('name')!r} lacks a sha256")
    return errs


def is_sendable(m: Dict[str, Any]) -> bool:
    """Hot-folder gate: verified rasters or no send (spec 7c)."""
    pv = m.get("production_verify", {})
    return bool(pv.get("run")) and bool(pv.get("gates_ok"))


def emit_package(packages_root: str | Path, *, grade_dir: str | Path,
                 mesh_path: str, part_name: str, tiff_job_dir: str | Path,
                 options: Dict[str, Any],
                 stamp: Optional[str] = None) -> Tuple[Path, Dict[str, Any]]:
    """Assemble one package directory + zip; returns (dir, manifest)."""
    grade_dir = Path(grade_dir)
    tiff_job_dir = Path(tiff_job_dir)
    root = Path(packages_root)
    pkg = root / package_name(part_name, stamp)
    pkg.mkdir(parents=True, exist_ok=False)

    intake = json.loads((grade_dir / "intake.json").read_text())
    prov = json.loads(
        (grade_dir / "heatr3d" / "correction_provenance.json").read_text())
    densify: Dict[str, Any] = {}
    for arm in ("uncorrected", "corrected"):
        rp = grade_dir / "heatr3d" / arm / "results.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            densify[arm] = {k: r.get(k) for k in (
                "engine", "trust_badge", "correction_engine", "grid_n",
                "sigma_T", "t_phi90_s", "sim_time_s", "stop_mean_rho",
                "rho_final_mean", "rho_final_std", "gates",
                "power_density_w_per_m3", "drive_recommended")}

    # the print power the arms ACTUALLY marched at (the recommended
    # ceiling-feasible drive when the solve gave one; nominal otherwise).
    # power_settings must record this, not a hardcoded nominal, or the
    # printed part cooks at a power the verification never simulated.
    NOMINAL_PD = 1.5915e6
    _c = densify.get("corrected", {})
    drive_pd = _c.get("power_density_w_per_m3") or NOMINAL_PD
    drive_recommended = bool(_c.get("drive_recommended"))

    # copy the graded TIFF job (the printer-DPI production rasters)
    job_dst = pkg / "print_job"
    shutil.copytree(tiff_job_dir, job_dst)
    job_info = json.loads((job_dst / "job_info.json").read_text()) \
        if (job_dst / "job_info.json").exists() else {}
    tiffs = sorted(q.name for q in job_dst.iterdir()
                   if q.suffix.lower() in (".tif", ".tiff"))

    # turntable: intent recorded, never fabricated, never absent
    tt_rec = {
        "mode": "static",
        "requested_intent": bool(options.get("turntable_intent", False)),
        "dwell_intent": bool(options.get("dwell_intent", False)),
        "advisory": True,
        "statement": ("Static exposure record. "
                      + ("Operator REQUESTED a turntable; " if
                         options.get("turntable_intent") else
                         "No turntable requested; ")
                      + TURNTABLE_ADVISORY),
    }
    tt_path = pkg / "turntable_static.json"
    tt_path.write_text(json.dumps(tt_rec, indent=2))

    pv = {"run": False,
          "statement": ("Production verify NOT run yet for this package; "
                        "send to the hot folder is blocked until a heatr3d "
                        "verification of THESE emitted rasters passes.")}
    (pkg / "production_verify_summary.json").write_text(
        json.dumps(pv, indent=2))

    files = [{"name": str(q.relative_to(pkg)), "sha256": _sha256_file(q),
              "bytes": q.stat().st_size}
             for q in sorted(pkg.rglob("*")) if q.is_file()]

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_versions": _engine_versions(),
        "part": {"name": part_name,
                 "source_geometry_sha256": _sha256_file(Path(mesh_path)),
                 "source_route": "stl_3d_studio",
                 "intake": intake},
        "correction_provenance": prov,
        "densify_summary": densify,
        "power_settings": {
            "power_density_w_per_m3": drive_pd,
            "drive_recommended": drive_recommended,
            "rf_mode": "constant",
            "note": ("heatr3d drive convention: fixed absorbed power "
                     "density; constant RF over the exposure. "
                     + ("recommended ceiling-feasible drive from the solve."
                        if drive_recommended else
                        "nominal drive (no ceiling-feasible drive recommended; "
                        "the ceiling gate is the backstop)."))},
        "raster": {
            "job_dir": "print_job", "n_tiffs": len(tiffs),
            "dpi": job_info.get("dpi", 720), "bpp": job_info.get("bpp", 4),
            "graded": job_info.get("graded"),
            "convention": ("Meteor production TIFFs: WhiteIsZero, black = "
                           "max ink, rasterized at printer DPI by the "
                           "slicer + grade_tiff_stack path; never "
                           "solve-grid rasters.")},
        "turntable": {"mode": "static", "file": tt_path.name,
                      "requested_intent": tt_rec["requested_intent"],
                      "dwell_intent": tt_rec["dwell_intent"],
                      "advisory": True},
        "production_verify": pv,
        "scheduler": {"drop_folder": "packages/", "note": SCHEDULER_NOTE},
        "files": files,
    }
    errs = validate_manifest(manifest)
    if errs:
        raise ValueError(f"refusing to emit an invalid manifest: {errs}")
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  default=float))
    shutil.make_archive(str(root / pkg.name), "zip", root_dir=root,
                        base_dir=pkg.name)
    return pkg, manifest
