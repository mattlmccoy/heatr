"""STL intake gate for the workbench (solver-venv side: trimesh + manifold3d).

Order of gates (each refusal is loud and typed, JSON-serializable):
  1. shape_library_3d.validate.validate_part_mesh:
     planarity -> watertightness (holes/naked edges) -> zero volume.
  2. NEW self-intersection gate: split the mesh into connected components and
     boolean-union them with the manifold engine; if the resolved solid volume
     differs from the raw signed volume beyond tolerance, the surface overlaps
     itself (the interpenetrating-shells class produced by bad CAD exports).
     Measured basis (2026-08-02): two overlapping 10 mm boxes read raw signed
     volume 2000 mm^3 vs resolved union 1875 mm^3, while trimesh's own
     is_watertight / is_volume / manifold3d status all read clean - hence this
     gate. LIMITATION (stated, not hidden): a single shell that passes through
     itself without changing the signed volume can evade this check; winding
     consistency is also required as a secondary signal.
  3. Chamber fit: axis-aligned extents must fit the 60 mm chamber.

Never imported by the GUI server (numpy); invoked via workbench_job --intake.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import trimesh

from shape_library_3d.validate import (
    InvalidPartGeometryError,
    validate_part_mesh,
)

logger = logging.getLogger(__name__)

CHAMBER_MM = 60.0
_SELF_INTERSECT_RTOL = 1e-3


class SelfIntersectingMeshError(InvalidPartGeometryError):
    """Mesh surface overlaps/passes through itself."""


class ChamberFitError(ValueError):
    """Part does not fit the 60 mm chamber."""


def _check_self_intersection(mesh: trimesh.Trimesh) -> None:
    if not mesh.is_winding_consistent:
        raise SelfIntersectingMeshError(
            "mesh winding is inconsistent; the surface orientation flips, "
            "which indicates self-intersection or flipped faces")
    parts = mesh.split()
    if len(parts) == 0:
        return
    try:
        resolved = trimesh.boolean.union(list(parts), engine="manifold")
        v_resolved = abs(float(resolved.volume))
    except (ValueError, RuntimeError) as e:
        raise SelfIntersectingMeshError(
            f"boolean resolution of the surface failed ({e}); the mesh cannot "
            f"be interpreted as a solid") from e
    v_raw = abs(float(mesh.volume))
    if v_raw <= 0:
        return  # zero-volume is caught earlier
    if abs(v_raw - v_resolved) / v_raw > _SELF_INTERSECT_RTOL:
        raise SelfIntersectingMeshError(
            f"surface overlaps itself: raw signed volume {v_raw:.1f} mm^3 vs "
            f"resolved solid volume {v_resolved:.1f} mm^3 "
            f"({abs(v_raw - v_resolved) / v_raw * 100:.1f}% discrepancy)")


def _check_chamber(mesh: trimesh.Trimesh) -> None:
    ext = np.asarray(mesh.extents, dtype=float)
    if float(ext.max()) > CHAMBER_MM:
        raise ChamberFitError(
            f"part extents {np.round(ext, 1).tolist()} mm do not fit the "
            f"{CHAMBER_MM:.0f} mm chamber. STL units are interpreted as "
            f"MILLIMETRES; a metres-unit export would read 1000x too large.")


def gate_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Run all intake gates on a loaded mesh -> verdict dict (JSON-safe)."""
    try:
        validate_part_mesh(mesh)
        _check_self_intersection(mesh)
        _check_chamber(mesh)
    except (InvalidPartGeometryError, ChamberFitError) as e:
        logger.warning("intake REFUSED: %s: %s", type(e).__name__, e)
        return {"accepted": False, "error_type": type(e).__name__,
                "error": str(e)}
    return {
        "accepted": True,
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(abs(mesh.volume)),
        "extents_mm": [float(v) for v in mesh.extents],
        "n_faces": int(len(mesh.faces)),
    }


def gate_stl(path: "str | Path") -> Dict[str, Any]:
    """Load an STL (mm units) and gate it; load failures are refusals too."""
    try:
        mesh = trimesh.load(str(path), force="mesh")
    except (ValueError, OSError) as e:
        return {"accepted": False, "error_type": "LoadError", "error": str(e)}
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return {"accepted": False, "error_type": "LoadError",
                "error": "file did not load as a triangle mesh"}
    return gate_mesh(mesh)
