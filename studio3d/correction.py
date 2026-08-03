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
BADGE_INVERSION = ("heatr3d native inversion | legacy heuristic (the direct "
                   "solve is the primary generator) | sim-only")


def _is_null_map(sat: np.ndarray, part: np.ndarray) -> bool:
    return bool(part.any()) and float(np.abs(sat[part] - 1.0).max()) < 1e-6


def _native_inversion(grade_dir: Path, part: np.ndarray):
    """Density-targeted make_fgm from the BEFORE arm's own fields.

    heatr3d's proportional-inverse rule on rho_final: under-densified
    regions get more dopant. Legacy heuristic, labeled as such; it exists
    so every part gets a real modulated correction even when the 2.5-D
    rulebook refuses the shape and no solved map matches."""
    from types import SimpleNamespace
    import heatr3d as H

    before = grade_dir / "heatr3d" / "uncorrected" / "fields.npz"
    if not before.exists():
        raise FileNotFoundError(
            "no correction source: no solved map matches, the 2.5-D map is "
            "absent or null, and the BEFORE arm has not run yet (its fields "
            "feed the native inversion). Run the BEFORE densification "
            "first.")
    with np.load(before) as d:
        part_b = d["part"].astype(bool)
        rho = np.asarray(d["rho_final"], float)
        T = np.asarray(d["T_phi90"], float)
    if part_b.shape != part.shape or not np.array_equal(part_b, part):
        raise ValueError("BEFORE arm part mask does not match this "
                         "voxelization; rerun the BEFORE arm at this grid")
    proxy = rho if rho.ndim == 3 else T
    res = SimpleNamespace(T_phi90=T, part=part_b)
    sat = H.make_fgm(res, magnitude=1.0, bpp=4, proxy=proxy)
    # in-part map; outside-part value never reaches the solver (gamma
    # blends part * sat), transfer not applicable: same grid, no resample
    rec = {"sat": np.where(part, sat, 0.0),
           "state": "transfer_not_applicable",
           "method": "native_inversion_same_grid",
           "dopant_mass_move_rel": 0.0, "gate": None}
    prov = {"engine": "heatr3d_native_inversion",
            "trust_badge": BADGE_INVERSION,
            "artifact": str(before),
            "proxy": "rho_final" if rho.ndim == 3 else "T_phi90"}
    return rec, prov


def _fallback_chain(grade_dir: Path, part: np.ndarray):
    """2.5-D per-slice if it modulates; else the native inversion."""
    dop = grade_dir / "heatr" / "dopant_volume.npz"
    if dop.exists():
        with np.load(dop) as d:
            rec = stack_to_voxel(d["sat"], d["part_mask"].astype(bool),
                                 d["z_mm"], part,
                                 chamber_m=float(d["chamber_m"])
                                 if "chamber_m" in d.files else 0.060)
        if not _is_null_map(rec["sat"], part):
            return rec, {"engine": "heatr_25d_perslice",
                         "trust_badge": BADGE_25D, "artifact": str(dop)}
    return _native_inversion(grade_dir, part)


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
        rec, prov = _fallback_chain(grade_dir, part)

    prov["transfer"] = {k: v for k, v in rec.items() if k != "sat"}
    prov["grid_n"] = int(n)
    # A map that never deviates from 1.0 in-part modulates nothing; the
    # AFTER arm will equal the BEFORE arm by construction. Say so loudly
    # (no false impression of a correction having been applied).
    max_dev = float(np.abs(rec["sat"][part] - 1.0).max()) if part.any() else 0.0
    prov["null_correction"] = bool(max_dev < 1e-6)
    if prov["null_correction"]:
        prov["null_note"] = (
            "correction is NULL for this part: the source map is 1.0 "
            "(unmodulated) everywhere in the part, so the corrected arm "
            "equals the uncorrected arm by construction")

    out = grade_dir / "heatr3d"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "correction_sat.npz",
                        sat=rec["sat"].astype(np.float32), part=part)
    (out / "correction_provenance.json").write_text(
        json.dumps(prov, indent=2, default=float))
    logger.info("correction built: engine=%s move=%.3e", prov["engine"],
                prov["transfer"]["dopant_mass_move_rel"])
    return prov
