"""STUDIO ALPHA print package: the versioned artifact the printer consumes.

One package per planned part, emitted into the top-level ``packages/``
directory (a directory plus a zip of it). ``packages/`` IS the scheduler drop
folder for now; a real uploader waits on hardware, and that decision is
recorded inside every manifest (``scheduler`` block).

Contents of a package directory:
  manifest.json                    schema below, validate_manifest checks it
  raster_meteor_import.png         the Meteor raster image processor import
                                   file at the PRODUCTION printer DPI (dots
                                   per inch) convention: 720 DPI, the Meteor
                                   native default of fgm_generator.py:70,
                                   resample path fgm_generator.py:588-606
                                   (at grid 120 over the 60 mm chamber this
                                   is the 1715 x 1715 level_map class)
  raster_preview.png               the human preview (white = max ink,
                                   image top = physical top)
  map_4bpp.npz                     the production fgm_generator npz format
                                   (level_map at printer DPI, sat_map at
                                   simulation resolution)
  turntable_program.json OR        the machine-readable turntable program,
  turntable_static.json            or an explicit "static, no turntable"
                                   record; never silently absent
  production_verify_summary.json   the real-engine verification summary, or
                                   an explicit not-run record

Acronyms on first use: DPI dots per inch; bpp bits per pixel; RF radio
frequency; FGM functionally graded material; IoU intersection over union.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent

SCHEMA_VERSION = "1.0.0"

# The production printer DPI convention. Source: fgm_generator.py:70
# (`dpi: int = 720`, "Target printer resolution in dots per inch (default
# 720, Meteor native)"), resampled at fgm_generator.py:588-606. The stored
# production artifacts at grid 120 over the 60 mm chamber are 1715 x 1715
# uint8 level maps (SHAPE_LIBRARY_SOLVE_REPORT.md quantizer pin).
PRODUCTION_DPI = 720

# The honest record, stamped into every manifest next to the recommendation.
ADVISORY_DISCLAIMER = (
    "ADVISORY, not a guarantee. The actuator classifier's bands were "
    "calibrated on the five measured rotation outcomes of the campaign "
    "(in-sample 5 of 5) and its end-to-end out-of-sample record on novel "
    "geometry is 1 of 2: it was right on the keyhole and wrong on the gear, "
    "where the solved static map beat the recommended rotation "
    "(GEOMETRY_GENERALIZATION_REPORT.md sections 5 and 6). The classifier "
    "answers 'does this actuator improve the heating', not 'does this "
    "actuator beat a solved map'. Treat the recommendation as a starting "
    "point and read the solved-arm numbers in expected_outcomes.")

GRID_QUALIFIER_TEMPLATE = (
    "All fidelity numbers are at grid {grid}. Grid-{grid} fidelity does not "
    "transfer to other grids unaided (SOLVE_ROBUSTNESS_VALIDATION.md); no "
    "SOLVED label applies unless Gate A (grid hold-out) and Gate B "
    "(sub-filter blur) both pass, and the gates block records whether they "
    "were run.")

SCHEDULER_NOTE = (
    "The top-level packages/ directory is the scheduler drop folder for now: "
    "a scheduler or printer-host uploader that consumes it is blocked on "
    "hardware availability, so deposit-and-collect is the interface. Do not "
    "rename or nest packages; the naming scheme pkg_<part>_<stamp> is the "
    "contract.")

_REQUIRED_BLOCKS = (
    "schema_version", "created_utc", "engine_version", "part",
    "chi_provenance", "classifier_recommendation", "gates",
    "expected_outcomes", "power_settings", "raster", "turntable",
    "production_verify", "scheduler", "files",
)


# ---------------------------------------------------------------------------
# pure logic: hash, naming, manifest
# ---------------------------------------------------------------------------

def geometry_hash(polys: Sequence[np.ndarray]) -> str:
    """SHA-256 of the source geometry: exact vertex bytes, in order."""
    h = hashlib.sha256()
    for p in polys:
        a = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def package_name(part_name: str, stamp: str | None = None) -> str:
    """Stable naming scheme: pkg_<part>_<YYYYMMDD-HHMMSS>."""
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in part_name)
    return f"pkg_{safe}_{stamp}"


def build_manifest(*, part_name: str, source_geometry_sha256: str,
                   source_route: str, engine_version: str,
                   chi_provenance: dict, recommendation: dict, gates: dict,
                   expected_outcomes: dict, power_settings: dict,
                   planned_arm: str, raster: dict | None = None,
                   turntable: dict | None = None,
                   production_verify: dict | None = None,
                   files: list[dict] | None = None,
                   extra: dict | None = None) -> dict:
    """Assemble a schema-1.0.0 manifest dict. Pure; no filesystem access."""
    rec = dict(recommendation)
    rec["advisory_disclaimer"] = ADVISORY_DISCLAIMER
    exp = dict(expected_outcomes)
    grid = exp.get("grid", "unknown")
    exp.setdefault("grid_qualifier", GRID_QUALIFIER_TEMPLATE.format(grid=grid))
    pw = dict(power_settings)
    pw.setdefault("rf_mode", "constant")
    pw.setdefault(
        "note", "Constant RF power over the exposure per the HEATR v2 "
                "standard (temporal power scheduling RETIRED, "
                "HEATR_V2_STANDARD.md); drive voltage from the automatic "
                "calibration to 500 W per metre on the uniform arm.")
    m: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_version": str(engine_version),
        "part": {"name": str(part_name),
                 "source_geometry_sha256": str(source_geometry_sha256),
                 "source_route": str(source_route)},
        "chi_provenance": dict(chi_provenance),
        "classifier_recommendation": rec,
        "gates": dict(gates),
        "expected_outcomes": exp,
        "planned_arm": str(planned_arm),
        "power_settings": pw,
        "raster": dict(raster or {}),
        "turntable": dict(turntable or {"mode": "static"}),
        "production_verify": dict(production_verify or {"run": False}),
        "scheduler": {"drop_folder": "packages/", "note": SCHEDULER_NOTE},
        "files": list(files or []),
    }
    if extra:
        m.update(extra)
    return m


def validate_manifest(manifest: dict) -> list[str]:
    """Return a list of problems; empty means valid."""
    errs: list[str] = []
    for key in _REQUIRED_BLOCKS:
        if key not in manifest:
            errs.append(f"missing required block: {key}")
    if errs:
        return errs
    if manifest["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {manifest['schema_version']!r} is not "
                    f"{SCHEMA_VERSION!r}")
    part = manifest["part"]
    for key in ("name", "source_geometry_sha256", "source_route"):
        if not part.get(key):
            errs.append(f"part.{key} is empty")
    if len(str(part.get("source_geometry_sha256", ""))) != 64:
        errs.append("part.source_geometry_sha256 is not a sha256 hex digest")
    if "advisory_disclaimer" not in manifest["classifier_recommendation"]:
        errs.append("classifier_recommendation.advisory_disclaimer missing")
    if "grid_qualifier" not in manifest["expected_outcomes"]:
        errs.append("expected_outcomes.grid_qualifier missing")
    if manifest["power_settings"].get("rf_mode") != "constant":
        errs.append("power_settings.rf_mode must be 'constant' "
                    "(HEATR v2 standard)")
    if "voltage_v" not in manifest["power_settings"]:
        errs.append("power_settings.voltage_v missing")
    if manifest["turntable"].get("mode") not in ("program", "static"):
        errs.append("turntable.mode must be 'program' or 'static'")
    if "run" not in manifest["production_verify"]:
        errs.append("production_verify.run missing (not-run must be "
                    "explicit, never silently absent)")
    for f in manifest["files"]:
        if len(str(f.get("sha256", ""))) != 64:
            errs.append(f"files entry {f.get('name')!r} lacks a sha256")
    return errs


# ---------------------------------------------------------------------------
# emitter
# ---------------------------------------------------------------------------

def _emit_npz_and_pngs(pkg_dir: Path, sat_map: np.ndarray, x: np.ndarray,
                       y: np.ndarray, bpp: int) -> dict:
    """Write map_4bpp.npz + the raster PNG pair; return the raster block."""
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import solve_fgm  # scripts/solve_fgm.py, the production npz + PNG emitter

    npz_path = pkg_dir / f"map_{int(bpp)}bpp.npz"
    solve_fgm.emit_production_npz(sat_map, x, y, npz_path, bpp=int(bpp),
                                  dpi=PRODUCTION_DPI)
    pngs = solve_fgm.emit_map_pngs(npz_path)
    # Package-stable names, independent of the npz stem.
    preview = pkg_dir / "raster_preview.png"
    meteor = pkg_dir / "raster_meteor_import.png"
    Path(pngs["png_path"]).rename(preview)
    Path(pngs["meteor_png_path"]).rename(meteor)
    with np.load(npz_path, allow_pickle=True) as d:
        shape_px = [int(s) for s in d["level_map"].shape]
    return {
        "file": meteor.name,
        "preview_file": preview.name,
        "npz_file": npz_path.name,
        "dpi": PRODUCTION_DPI,
        "bpp": int(bpp),
        "level_map_shape_px": shape_px,
        "convention": ("Meteor import: black = max ink, image top = physical "
                       "top; exact pixel inversion of the preview "
                       "(fgm_generator.py:702-744). Loader inverse: level = "
                       "round((255 - pixel) / 255 * (2^bpp - 1))."),
    }


def emit_package(root: str | Path, *, manifest_kwargs: dict,
                 sat_map: np.ndarray, x: np.ndarray, y: np.ndarray,
                 bpp: int = 4, turntable_program: dict | None = None,
                 production_verify: dict | None = None,
                 stamp: str | None = None) -> tuple[Path, Path]:
    """Emit one print package directory plus its zip; return both paths.

    ``manifest_kwargs`` are the pure ``build_manifest`` keyword arguments
    minus ``raster``, ``turntable``, ``production_verify`` and ``files``,
    which this emitter fills from what it actually wrote.
    """
    root = Path(root)
    pkg_dir = root / package_name(str(manifest_kwargs["part_name"]), stamp)
    pkg_dir.mkdir(parents=True, exist_ok=False)

    raster = _emit_npz_and_pngs(pkg_dir, np.asarray(sat_map), np.asarray(x),
                                np.asarray(y), int(bpp))

    if turntable_program is not None:
        tt_path = pkg_dir / "turntable_program.json"
        tt_path.write_text(json.dumps(turntable_program, indent=2,
                                      default=float))
        turntable = {"mode": "program", "file": tt_path.name,
                     "actuator": turntable_program.get("actuator",
                                                       "indexed turntable")}
    else:
        tt_path = pkg_dir / "turntable_static.json"
        record = {"mode": "static",
                  "statement": ("Static exposure: NO TURNTABLE is used for "
                                "this part. This record is written "
                                "explicitly so a missing program can never "
                                "be mistaken for a forgotten one.")}
        tt_path.write_text(json.dumps(record, indent=2))
        turntable = {"mode": "static", "file": tt_path.name}

    pv = dict(production_verify) if production_verify else {
        "run": False,
        "statement": ("Production verify was NOT run for this package; the "
                      "expected_outcomes numbers come from the planning "
                      "solve only.")}
    pv.setdefault("run", True)
    (pkg_dir / "production_verify_summary.json").write_text(
        json.dumps(pv, indent=2, default=float))

    files = [{"name": p.name, "sha256": sha256_file(p),
              "bytes": p.stat().st_size}
             for p in sorted(pkg_dir.iterdir()) if p.is_file()]
    manifest = build_manifest(**manifest_kwargs, raster=raster,
                              turntable=turntable, production_verify=pv,
                              files=files)
    errs = validate_manifest(manifest)
    if errs:
        raise ValueError(f"refusing to emit an invalid manifest: {errs}")
    (pkg_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float))

    zip_base = root / pkg_dir.name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip",
                                        root_dir=root,
                                        base_dir=pkg_dir.name))
    return pkg_dir, zip_path
