"""Mac-native 3D printable-raster emitter.

Turns a solved 3D dopant map into the printable artifact set -- a per-layer stack
of 720-dpi ordered-dithered printer level maps + Meteor WhiteIsZero TIFFs +
job_info + a correction_stack.npz -- WITHOUT the Windows-only Meteor slicer. This
is the 3D analog of the 2D pure-numpy emit path (scripts/solve_fgm.emit_production_npz):
it reuses the TRUSTED 2D dither (adjoint2d.printability.printer_level_map, the same
Bayer-8 grade the production meteor_bridge uses) and the studio3d FEM->voxel
resampler (studio3d.transfer.dg0_to_voxel, with its 2% dopant-mass gate).

The slicer-free footprint = the part cross-section per layer; the grade multiplies
it by the dopant saturation. The MetPrint submission stays Windows-side (stage_job);
this module produces the files a MetPrint-style consumer reads. It runs in a pure-
numpy env (.venv312) -- no dolfinx.

Contract fields carried (2D-parity): dpi 720, bpp 4, grey_levels 8, layer_height,
per-layer part_mask (bool), z_mm, proxy_field="solve", engine_version. Frame:
centroids are re-centered to the part voxel frame before resampling (the map sits in
the gmsh outline frame; voxelize_stl does this centering in the Windows path).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# adjoint2d (the trusted 2D printability dither) lives under fgm_solve_campaign;
# make the module importable however it is invoked (mirrors the 2D lane's layout).
_CAMPAIGN = str(Path(__file__).resolve().parents[1] / "fgm_solve_campaign")
if _CAMPAIGN not in sys.path:
    sys.path.insert(0, _CAMPAIGN)
from adjoint2d import printability as pq          # noqa: E402

DPI_DEFAULT = 720
BPP_DEFAULT = 4
GREY_LEVELS_DEFAULT = 8


# --------------------------------------------------------------------------- #
# Pure transforms (unit-tested)
# --------------------------------------------------------------------------- #
def voxel_to_layer_stack(sat_vox: np.ndarray, part: np.ndarray, h: float):
    """(nx,ny,nz) voxel saturation + bool part mask + cell size h (m) ->
    (sat_stack, mask_stack, z_mm) in the Meteor (nz,ny,nx) print-layer convention
    over the OCCUPIED z-layers (bottom->top). sat=1 outside the part (undoped
    baseline). Mirrors studio3d.correction._finish (the (2,1,0) transpose + z_mm).
    """
    part = np.asarray(part, bool)
    sat_vox = np.asarray(sat_vox, float)
    zs = np.where(part.any(axis=(0, 1)))[0]
    mask_stack = np.transpose(part[:, :, zs], (2, 1, 0))
    sat_stack = np.transpose(sat_vox[:, :, zs], (2, 1, 0))
    sat_stack = np.where(mask_stack, sat_stack, 1.0).astype(np.float32)
    z_mm = (np.arange(len(zs)) + 0.5) * float(h) * 1e3
    return sat_stack, mask_stack, z_mm


def layer_to_levels(sat_layer: np.ndarray, mask_layer: np.ndarray, h: float, *,
                    dpi: int = DPI_DEFAULT, bpp: int = BPP_DEFAULT,
                    grey_levels: int = GREY_LEVELS_DEFAULT) -> np.ndarray:
    """One (ny,nx) saturation layer -> a printer-resolution ordered-dithered level
    map (uint8, 0..mv), 0 ink outside the part footprint. Reuses the trusted 2D
    adjoint2d.printability.printer_level_map (Bayer-8 dither, head cap 7)."""
    inked = np.where(np.asarray(mask_layer, bool), sat_layer, 0.0)
    return pq.printer_level_map(inked, dx_m=float(h), dy_m=float(h), dpi=int(dpi),
                                bpp=int(bpp), grey_levels=int(grey_levels),
                                dither="ordered")


def levels_to_whiteiszero(levels: np.ndarray, bpp: int = BPP_DEFAULT,
                          grey_levels: int = GREY_LEVELS_DEFAULT) -> np.ndarray:
    """Level map (0..mv) -> Meteor WhiteIsZero 8-bit gray: black (0) = max ink,
    white (255) = no ink, monotonic decreasing in level."""
    mv = pq.max_level(bpp, grey_levels)
    return (255.0 * (1.0 - np.asarray(levels, float) / mv)).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# IO orchestrator (end-to-end verified by a real run)
# --------------------------------------------------------------------------- #
def emit_printable_package(map_npz: str, part_npz: str, out_dir: str, *,
                           dpi: int = DPI_DEFAULT, bpp: int = BPP_DEFAULT,
                           grey_levels: int = GREY_LEVELS_DEFAULT,
                           engine_version: str = "mac_raster-1.0.0") -> dict:
    """Solved-map npz (FEM centroids+s_map+volumes) + part voxel npz (part,h,n) ->
    a printable package dir: correction_stack.npz + print_job/ (per-layer 720-dpi
    WhiteIsZero TIFFs + _ungraded/) + job_info.json + level_stack.npz. Returns a
    manifest dict. Pure-numpy; no Meteor slicer, no dolfinx."""
    from studio3d.transfer import dg0_to_voxel
    from PIL import Image

    out = Path(out_dir); (out / "print_job" / "_ungraded").mkdir(parents=True, exist_ok=True)
    m = np.load(map_npz, allow_pickle=True)
    centroids = m["centroids"].astype(float)
    s_map = m["s_map"].astype(float)
    volumes = m["volumes"].astype(float)
    pj = np.load(part_npz)
    part = pj["part"].astype(bool)
    n = int(pj["n"]); h = float(pj["h"]); chamber_m = n * h

    # frame: re-center the map centroids onto the part voxel frame (the map sits in
    # the gmsh outline frame; the voxel part is chamber-centered). voxelize_stl does
    # this in the Windows path; do it explicitly for the map-only entry.
    bb_c = 0.5 * (centroids.min(0) + centroids.max(0))
    rec = dg0_to_voxel(centroids - bb_c, s_map, volumes, part, chamber_m=chamber_m)
    sat_vox = rec["sat"]

    sat_stack, mask_stack, z_mm = voxel_to_layer_stack(sat_vox, part, h)
    nz, ny, nx = sat_stack.shape
    np.savez_compressed(out / "correction_stack.npz",
                        sat=sat_stack, part_mask=mask_stack, z_mm=z_mm,
                        chamber_m=chamber_m, proxy_field="solve")

    mv = float(pq.max_level(bpp, grey_levels))
    levels = []
    for k in range(nz):
        lv = layer_to_levels(sat_stack[k], mask_stack[k], h, dpi=dpi, bpp=bpp,
                             grey_levels=grey_levels)
        base = layer_to_levels(mask_stack[k].astype(float), mask_stack[k], h,
                               dpi=dpi, bpp=bpp, grey_levels=grey_levels)
        levels.append(lv)
        Image.fromarray(levels_to_whiteiszero(lv, bpp, grey_levels), "L").save(
            out / "print_job" / f"layer_{k:03d}.tif")
        Image.fromarray(levels_to_whiteiszero(base, bpp, grey_levels), "L").save(
            out / "print_job" / "_ungraded" / f"layer_{k:03d}.tif")
    levels = np.stack(levels).astype(np.uint8)
    np.savez_compressed(out / "level_stack.npz", levels=levels, z_mm=z_mm,
                        dpi=dpi, bpp=bpp, grey_levels=grey_levels)

    manifest = {
        "engine_version": engine_version, "proxy_field": "solve",
        "dpi": int(dpi), "bpp": int(bpp), "grey_levels": int(grey_levels),
        "layer_height_mm": float(h * 1e3), "layer_count": int(nz),
        "raster_px": [int(levels.shape[1]), int(levels.shape[2])],
        "chamber_m": float(chamber_m),
        "dopant_mass_move_rel": float(rec.get("dopant_mass_move_rel", float("nan"))),
        "dg0_state": str(rec.get("state", "")),
        "tiff_convention": "Meteor WhiteIsZero (black=max ink), levels 0..%d" % int(mv),
        "slicer": "mac_raster pure-numpy (no Meteor slicer); MetPrint submission is Windows-side",
    }
    (out / "print_job" / "job_info.json").write_text(json.dumps(manifest, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Emit a Mac-native 3D printable package "
                                             "from a solved dopant map (no Meteor slicer).")
    ap.add_argument("map_npz", help="solved map npz (centroids, s_map, volumes)")
    ap.add_argument("part_npz", help="part voxel npz (part, h, n)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dpi", type=int, default=DPI_DEFAULT)
    ap.add_argument("--bpp", type=int, default=BPP_DEFAULT)
    ap.add_argument("--grey-levels", type=int, default=GREY_LEVELS_DEFAULT)
    a = ap.parse_args()
    man = emit_printable_package(a.map_npz, a.part_npz, a.out_dir, dpi=a.dpi,
                                 bpp=a.bpp, grey_levels=a.grey_levels)
    print(json.dumps(man, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
