"""Generate the shape library: watertight STLs + measured metadata + manifest.

Run as a module to (re)generate the committed artifacts:
    ../geo-prewarp/.venv312/bin/python -m shape_library_3d.build_library

Every metadata value is MEASURED from the final mesh (never hand-transcribed).
Tier-1/2 meshes are normalized to V* and re-asserted watertight before export;
Tier-3 meshes are exported unnormalized with their expected rejection recorded.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh

from shape_library_3d import generators as _gen
from shape_library_3d.constants import CHAMBER_L_MM, V_STAR_MM3, VOL_REL_TOL
from shape_library_3d.normalize import scale_to_volume
from shape_library_3d.registry import SHAPES, ShapeSpec
from shape_library_3d.validate import InvalidPartGeometryError, validate_part_mesh

logger = logging.getLogger(__name__)
_PKG = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _measure(mesh: trimesh.Trimesh, name: str, spec: ShapeSpec, scale_factor: float,
             stl_path: Path, rejection_error: Optional[str]) -> Dict:
    ext = np.asarray(mesh.extents, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    euler = int(mesh.euler_number)
    genus = (2 - euler) // 2
    vol = float(abs(mesh.volume))
    is_part = spec.tier in (1, 2)
    return {
        "name": name,
        "tier": spec.tier,
        "role": spec.role,
        "rf_characteristic": spec.rf_characteristic,
        "numerical_characteristic": spec.numerical_characteristic,
        "target_volume_mm3": spec.target_volume_mm3,
        "actual_volume_mm3": vol,
        "volume_err_frac": (abs(vol - V_STAR_MM3) / V_STAR_MM3) if is_part else None,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": euler,
        "genus": genus,
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "facet_area_mm2": {
            "min": float(areas.min()), "max": float(areas.max()),
            "mean": float(areas.mean()), "std": float(areas.std()),
        },
        "bbox_mm": [float(ext[0]), float(ext[1]), float(ext[2])],
        "realized_chamber_gap_mm": float((CHAMBER_L_MM - float(ext.max())) / 2.0),
        "scale_factor": float(scale_factor),
        "stl_path": f"stl/{name}.stl",
        "sha256": _sha256(stl_path),
        "rejection_error": rejection_error,
    }


def build(out_dir: Optional[Path] = None) -> Dict:
    """Generate every shape into ``out_dir`` (default: the package dir).

    Writes ``stl/<name>.stl`` and ``meta/<name>.json`` per shape plus
    ``meta/library_manifest.json``. Returns the manifest dict.
    """
    out_dir = Path(out_dir) if out_dir is not None else _PKG
    stl_dir = out_dir / "stl"
    meta_dir = out_dir / "meta"
    stl_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for name, spec in SHAPES.items():
        # Tier-3 meshes have zero/degenerate volume; trimesh's center_mass
        # divide warns harmlessly while we measure them. Silence locally.
        with np.errstate(divide="ignore", invalid="ignore"):
            entries.append(_build_one(name, spec, stl_dir, meta_dir))

    manifest = {
        "library": "shape_library_3d",
        "v_star_mm3": V_STAR_MM3,
        "chamber_l_mm": CHAMBER_L_MM,
        "n_shapes": len(entries),
        "shapes": entries,
    }
    (meta_dir / "library_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _build_one(name: str, spec: ShapeSpec, stl_dir: Path, meta_dir: Path) -> Dict:
    mesh = getattr(_gen, spec.generator)()
    scale_factor = 1.0
    rejection_error = None
    if spec.tier in (1, 2):
        _, scale_factor = scale_to_volume(mesh, V_STAR_MM3)
        validate_part_mesh(mesh)  # must not raise for a real part
        if not mesh.is_watertight:
            raise RuntimeError(f"{name}: not watertight after normalization")
        err = abs(mesh.volume - V_STAR_MM3) / V_STAR_MM3
        if err >= VOL_REL_TOL:
            raise RuntimeError(f"{name}: volume error {err:.2e} >= {VOL_REL_TOL}")
    else:
        try:
            validate_part_mesh(mesh)
            raise RuntimeError(f"{name}: Tier-3 mesh unexpectedly validated as a part")
        except InvalidPartGeometryError as e:
            rejection_error = type(e).__name__

    stl_path = stl_dir / f"{name}.stl"
    mesh.export(stl_path)
    entry = _measure(mesh, name, spec, scale_factor, stl_path, rejection_error)
    (meta_dir / f"{name}.json").write_text(json.dumps(entry, indent=2))
    logger.info("built %s (tier %d, genus %d)", name, spec.tier, entry["genus"])
    return entry


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = build()
    print(f"Built {manifest['n_shapes']} shapes into {_PKG}/stl and {_PKG}/meta")
    for e in manifest["shapes"]:
        tag = "REJECT" if e["tier"] == 3 else f"V*err {e['volume_err_frac']:.1e}"
        print(f"  [{e['tier']}] {e['name']:16s} genus={e['genus']:<2d} "
              f"faces={e['n_faces']:<6d} gap={e['realized_chamber_gap_mm']:.1f}mm  {tag}")


__all__ = ["build", "main"]

if __name__ == "__main__":
    main()
