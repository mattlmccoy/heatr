"""Level 0 anisotropic affine shrinkage pre-compensation (spec section 1).

Scope, stated once and enforced by tests:

- This module carries MATERIAL shrinkage only -- the contraction of nylon 12
  itself on melt and recrystallization (SHRINKAGE_COEFFICIENTS_MEMO.md).
- DENSIFICATION CONSOLIDATION (powder -> solid) is the densify model's job
  (heatr3d.shrinkage_factors, studio3d.warped_mesh). That path must never see
  these coefficients; the double-counting guard lives in
  studio3d/tests/test_precomp.py.

The coefficients are read from the cross-lane config at the repo root
(shrinkage_precomp.json, schema agreed with the solve3d lane, read-only here).
They are SLS literature values borrowed for RFAM and are UNMEASURED for our
process: the config's applicability string travels verbatim into every sidecar
and every provenance record so no quoted dimension can lose the caveat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_NAME = "shrinkage_precomp.json"
CONFIG_ENV = "RFAM_SHRINKAGE_PRECOMP_CONFIG"
SUPPORTED_SCHEMA = ("1.0",)
SIDECAR_SUFFIX = ".precomp.json"
COMPENSATION_FORM = "f = 1 / (1 - s) per axis"


def config_path() -> Path:
    """Where the coefficients come from (env override for tests/campaigns)."""
    env = os.environ.get(CONFIG_ENV)
    return Path(env) if env else REPO_ROOT / CONFIG_NAME


def load_precomp_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Read the agreed cross-lane coefficient config, or REFUSE loudly.

    Refusals (never a silent default):
      - file missing                      -> FileNotFoundError
      - schema_version not supported      -> ValueError
      - material_only is not True         -> ValueError, because a coefficient
        that already contains powder consolidation would double-count the
        densify model's own collapse.
    """
    p = Path(path) if path is not None else config_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"shrinkage pre-compensation config not found at {p}; Level 0 "
            "coefficients are config-driven and are never defaulted in code")
    cfg = json.loads(p.read_text())
    ver = str(cfg.get("schema_version"))
    if ver not in SUPPORTED_SCHEMA:
        raise ValueError(
            f"unsupported schema_version {ver!r} in {p} (supported: "
            f"{list(SUPPORTED_SCHEMA)}); refusing to guess the field meaning")
    if cfg.get("material_only") is not True:
        raise ValueError(
            f"{p} does not declare material_only=true; the Level 0 pre-scale "
            "may carry MATERIAL shrinkage only -- densification consolidation "
            "is marched by the densify model and applying it here would "
            "double-count the same contraction")
    for k in ("s_xy", "s_z_mat", "applicability"):
        if k not in cfg:
            raise ValueError(f"{p} is missing required key {k!r}")
    cfg["_config_path"] = str(p)
    return cfg


def compensation_factors(cfg: Dict[str, Any]) -> Tuple[float, float]:
    """(f_xy, f_z) from the shrinkage coefficients.

    Form: f = 1 / (1 - s), the industry-standard compensation form
    (SHRINKAGE_COEFFICIENTS_MEMO.md section 2, "the industry-standard
    compensation form"). Why the inverse and not the naive 1 + s: shrinkage is
    defined MULTIPLICATIVELY on the built dimension, L_final = (1 - s) * L_built.
    To land on the nominal L_nom we must build L_built = L_nom / (1 - s). Using
    1 + s leaves a residual of s^2/(1 - s) (about 0.09 % at s = 3 %, i.e. 18 um
    on a 20 mm part) -- small, but a systematic undersize with no reason to
    accept it.

    s_xy applies to both in-plane axes (the literature X-Y difference is small
    relative to the coefficient band; memo section 3), s_z_mat to Z.
    """
    s_xy = float(cfg["s_xy"])
    s_z = float(cfg["s_z_mat"])
    for name, s in (("s_xy", s_xy), ("s_z_mat", s_z)):
        if not 0.0 <= s < 1.0:
            raise ValueError(f"{name}={s} outside [0, 1); a shrinkage "
                             "coefficient is a fraction of the built dimension")
    return 1.0 / (1.0 - s_xy), 1.0 / (1.0 - s_z)


def sidecar_path(mesh_path: str | Path) -> Path:
    return Path(str(mesh_path) + SIDECAR_SUFFIX)


def read_sidecar(mesh_path: str | Path) -> Dict[str, Any] | None:
    p = sidecar_path(mesh_path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"unreadable precomp sidecar {p}: {e}") from e


def _config_sha256(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest()


def compensated_bbox_mm(mesh_path: str | Path,
                        cfg: Dict[str, Any] | None = None) -> list[float]:
    """Bounding box of the part AS PRINTED (green, enlarged), in mm.

    This is the box the chamber-fit check must see: the machine has to fit the
    pre-compensated part, not the nominal CAD.
    """
    cfg = cfg or load_precomp_config()
    f_xy, f_z = compensation_factors(cfg)
    mesh = trimesh.load_mesh(str(mesh_path))
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], float)
    return [float(ext[0] * f_xy), float(ext[1] * f_xy), float(ext[2] * f_z)]


def precompensate_stl(src_path: str | Path, dst_path: str | Path,
                      cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Write the pre-compensated mesh + its provenance sidecar.

    Scales (f_xy, f_xy, f_z) about the MESH CENTROID so the part does not
    translate. REFUSES an input whose own sidecar already records applied=True
    (double application would enlarge by f^2).
    """
    src, dst = Path(src_path), Path(dst_path)
    prior = read_sidecar(src)
    if prior and prior.get("applied"):
        raise ValueError(
            f"{src} is already pre-compensated (sidecar {sidecar_path(src)} "
            f"records applied=True, factors {prior.get('factors')}); refusing "
            "to apply the shrinkage pre-scale twice")
    cfg = cfg or load_precomp_config()
    f_xy, f_z = compensation_factors(cfg)

    mesh = trimesh.load_mesh(str(src))
    bbox_before = (np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0]))
    anchor = np.asarray(mesh.centroid, float)
    scale = np.array([f_xy, f_xy, f_z], float)
    mesh.vertices = anchor + (np.asarray(mesh.vertices, float) - anchor) * scale
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)
    bbox_after = (np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0]))

    record: Dict[str, Any] = {
        "applied": True,
        "form": COMPENSATION_FORM,
        "factors": {"f_xy": f_xy, "f_z": f_z},
        "coefficients": {"s_xy": float(cfg["s_xy"]),
                         "s_z_mat": float(cfg["s_z_mat"])},
        "bands": {"s_xy_band": cfg.get("s_xy_band"),
                  "s_z_mat_band": cfg.get("s_z_mat_band")},
        "applicability": cfg["applicability"],
        "schema_version": cfg["schema_version"],
        "source_config": Path(cfg["_config_path"]).name,
        "source_config_sha256": _config_sha256(cfg),
        "anchor": "mesh centroid",
        "src": str(src),
        "dst": str(dst),
        "bbox_mm_nominal": [float(v) for v in bbox_before],
        "bbox_mm_compensated": [float(v) for v in bbox_after],
        "scope": ("MATERIAL shrinkage only; densification consolidation is "
                  "marched by the densify model and is NOT in this factor"),
    }
    sidecar_path(dst).write_text(json.dumps(record, indent=2))
    logger.info("pre-compensated %s -> %s (f_xy=%.6f f_z=%.6f)",
                src.name, dst, f_xy, f_z)
    return record


def ensure_precompensated(mesh_path: str | Path, work_dir: str | Path,
                          cfg: Dict[str, Any] | None = None
                          ) -> Tuple[str, Dict[str, Any]]:
    """(path, record) for <work_dir>/precomp/<meshname>, building it if needed.

    Reused only when the existing sidecar was written from the SAME config
    (sha256 match); a coefficient change rebuilds rather than silently serving
    a mesh compensated with retired numbers.
    """
    src = Path(mesh_path)
    cfg = cfg or load_precomp_config()
    dst = Path(work_dir) / "precomp" / src.name
    existing = read_sidecar(dst) if dst.is_file() else None
    if existing and existing.get("source_config_sha256") == _config_sha256(cfg):
        rec = dict(existing)
        rec["reused"] = True
        return str(dst), rec
    rec = precompensate_stl(src, dst, cfg)
    rec = dict(rec)
    rec["reused"] = False
    return str(dst), rec


CHAMBER_MM = 60.0


def prepare_mesh(mesh_path: str | Path, work_dir: str | Path,
                 enabled: bool = True, chamber_mm: float = CHAMBER_MM,
                 cfg: Dict[str, Any] | None = None
                 ) -> Tuple[str, Dict[str, Any]]:
    """The one entry every consumer calls: (mesh_to_use, provenance block).

    Default ON. With enabled=False the ORIGINAL mesh is returned and the
    provenance says enabled false explicitly -- absence and off are never
    confusable.

    Chamber-fit ORDERING: the fit check is applied to the COMPENSATED bbox,
    because the machine has to fit the green part as printed, not the nominal
    CAD. A part that fits at nominal and overflows after compensation is
    refused here, before any voxelization or march.
    """
    src = Path(mesh_path)
    if not enabled:
        return str(src), provenance_record(
            enabled=False, reason="precomp=False requested by the caller")
    cfg = cfg or load_precomp_config()
    nominal = np.asarray(trimesh.load_mesh(str(src)).extents, float)
    comp = compensated_bbox_mm(src, cfg)
    if any(c >= chamber_mm for c in comp):
        raise ValueError(
            "REFUSED: shrinkage pre-compensation enlarges the part beyond the "
            f"{chamber_mm:.0f} mm chamber. Nominal bbox "
            f"{[round(float(v), 1) for v in nominal]} mm fits, but the "
            "compensated (as-printed green) bbox is "
            f"{[round(float(v), 1) for v in comp]} mm. The printed part, not "
            "the CAD, has to fit: scale the part down or disable "
            "pre-compensation knowing the result will land undersize.")
    path, rec = ensure_precompensated(src, work_dir, cfg)
    prov = provenance_record(cfg, enabled=True)
    prov["mesh"] = path
    prov["reused"] = rec["reused"]
    prov["bbox_mm_nominal"] = [float(v) for v in nominal]
    prov["bbox_mm_compensated"] = comp
    return path, prov


def provenance_record(cfg: Dict[str, Any] | None = None,
                      enabled: bool = True,
                      reason: str | None = None) -> Dict[str, Any]:
    """The "shrinkage_precomp" block carried in results / provenance JSON."""
    if not enabled:
        rec: Dict[str, Any] = {"enabled": False}
        if reason:
            rec["reason"] = reason
        return rec
    cfg = cfg or load_precomp_config()
    f_xy, f_z = compensation_factors(cfg)
    return {"enabled": True, "form": COMPENSATION_FORM,
            "f_xy": f_xy, "f_z": f_z,
            "s_xy": float(cfg["s_xy"]), "s_z_mat": float(cfg["s_z_mat"]),
            "s_xy_band": cfg.get("s_xy_band"),
            "s_z_mat_band": cfg.get("s_z_mat_band"),
            "applicability": cfg["applicability"],
            "source_config_sha256": _config_sha256(cfg)}


def ui_summary(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """One-line display payload for the Studio Mesh card (numbers computed,
    never hardcoded in the UI)."""
    cfg = cfg or load_precomp_config()
    f_xy, f_z = compensation_factors(cfg)
    pct_xy = (f_xy - 1.0) * 100.0
    pct_z = (f_z - 1.0) * 100.0
    return {"enabled": True, "f_xy": f_xy, "f_z": f_z,
            "pct_xy": pct_xy, "pct_z": pct_z,
            "s_xy": float(cfg["s_xy"]), "s_z_mat": float(cfg["s_z_mat"]),
            "applicability": cfg["applicability"],
            "line": (f"shrinkage pre-comp ON: xy +{pct_xy:.2f}%, "
                     f"z +{pct_z:.2f}% (SLS literature values, RFAM "
                     "unmeasured - pending P1)")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Level 0 shrinkage pre-compensation")
    ap.add_argument("src", nargs="?", help="input mesh (omit for --info)")
    ap.add_argument("dst", nargs="?", help="output mesh")
    ap.add_argument("--info", action="store_true",
                    help="print the coefficient/factor summary as JSON")
    args = ap.parse_args()
    if args.info or not args.src:
        print(json.dumps(ui_summary()))
        return 0
    if not args.dst:
        ap.error("dst is required unless --info is given")
    print(json.dumps(precompensate_stl(args.src, args.dst)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
