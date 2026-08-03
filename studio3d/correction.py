"""Correction source selection and build (spec section 6).

Engine registry, never blended:
- "solve3d_solved": exact part-hash match in the solved registry; the DG0
  artifact transfers support-aware onto the voxel grid.
- "heatr_25d_perslice": the deployable fallback, the existing 2.5-D
  verification's dopant_volume.npz transferred support-aware.
Every provenance record names its engine, trust badge, and transfer state.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from studio3d import registry as reg
from studio3d.runner import voxelize_stl
from studio3d.transfer import dg0_to_voxel, stack_to_voxel

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

BADGE_25D = ("HEATR 2.5-D per-slice | deployable grading path | sim-only")


def build_correction(grade_dir: str | Path, mesh_path: str, n: int,
                     registry_path: Path = reg.REGISTRY_PATH
                     ) -> Dict[str, Any]:
    """Build the active correction volume for this mesh at grid n.

    Writes grade_dir/heatr3d/correction_sat.npz (sat + part) and
    correction_provenance.json; returns the provenance record.
    """
    grade_dir = Path(grade_dir)
    part = voxelize_stl(mesh_path, n)

    entry = reg.find_solved_map(part, registry_path=registry_path)
    if entry is not None:
        art = Path(entry["artifact"])
        if not art.is_absolute():
            art = ROOT / art
        with np.load(art) as d:
            rec = dg0_to_voxel(d["centroids"], d["s_map"], d["volumes"],
                               part, chamber_m=0.060)
        prov: Dict[str, Any] = {
            "engine": entry["engine"],
            "trust_badge": entry["trust_badge"],
            "artifact": str(entry["artifact"]),
            "registry_entry": entry["name"],
            "source": entry.get("source"),
        }
    else:
        dop = grade_dir / "heatr" / "dopant_volume.npz"
        if not dop.exists():
            raise FileNotFoundError(
                "no solved 3-D map matches this geometry and no 2.5-D "
                "dopant volume exists yet: run the HEATR 2.5-D "
                "verification first (it produces dopant_volume.npz, the "
                "deployable correction source)")
        with np.load(dop) as d:
            rec = stack_to_voxel(d["sat"], d["part_mask"].astype(bool),
                                 d["z_mm"], part,
                                 chamber_m=float(d["chamber_m"])
                                 if "chamber_m" in d.files else 0.060)
        prov = {
            "engine": "heatr_25d_perslice",
            "trust_badge": BADGE_25D,
            "artifact": str(dop),
        }

    prov["transfer"] = {k: v for k, v in rec.items() if k != "sat"}
    prov["grid_n"] = int(n)

    out = grade_dir / "heatr3d"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "correction_sat.npz",
                        sat=rec["sat"].astype(np.float32), part=part)
    (out / "correction_provenance.json").write_text(
        json.dumps(prov, indent=2, default=float))
    logger.info("correction built: engine=%s move=%.3e", prov["engine"],
                prov["transfer"]["dopant_mass_move_rel"])
    return prov
