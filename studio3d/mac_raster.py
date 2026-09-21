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


def centered_axis_mm(n: int, h: float) -> np.ndarray:
    """The n cell-centered chamber-column coordinates in mm, centered about the
    origin (symmetric, half-cell inset from the edges), mirroring the 2D x_mm/y_mm
    placement contract (scripts/solve_fgm.py:409). This is the georeferencing a
    bed-placement consumer (the Windows slicer / stage_job) reads to position the
    part on the printer bed; the Mac raster canvas IS the chamber (package_verify
    derives chamber_m = raster_px / dpi), so these coords fully place it."""
    return ((np.arange(int(n)) + 0.5) * float(h) - 0.5 * int(n) * float(h)) * 1e3


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


# --------------------------------------------------------------------------- #
# IO orchestrator (end-to-end verified by a real run)
# --------------------------------------------------------------------------- #
def emit_printable_package(map_npz: str, part_npz: str, out_dir: str, *,
                           dpi: int = DPI_DEFAULT, bpp: int = BPP_DEFAULT,
                           grey_levels: int = GREY_LEVELS_DEFAULT,
                           engine_version: str = "mac_raster-1.0.0") -> dict:
    """Solved-map npz (FEM centroids+s_map+volumes) + part voxel npz (part,h,n) ->
    the printable inputs the SANCTIONED Meteor writer consumes:

      correction_stack.npz  -- (nz,ny,nx) voxel sat + mask + placement coords
      fgm_level_map.npz      -- level_map (nz,ny_px,nx_px) uint8 in [0, head ceiling
                                7] at printer resolution + bpp/dpi/width_mm/height_mm;
                                THE format software/meteor/tools/fgm_to_rip.
                                fgm_to_tiff_stack reads to emit 4bpp/LZW/WhiteIsZero
                                MetPrint TIFFs.
      print_job/*.tif        -- the real MetPrint TIFFs, IF meteor_rip is importable
                                (delegated to fgm_to_rip); else write them later with
                                `fgm_to_rip.fgm_to_tiff_stack(fgm_level_map.npz, ...)`.

    This module does NOT encode TIFFs itself -- the 4bpp 3-bit-in-nibble WhiteIsZero
    LZW format (MetPrint masks the 4th bit; quantize to the head's 7, not the
    container 15) lives in meteor_rip.py and must not be re-implemented here.
    Pure-numpy up to the level_map; no dolfinx, no Meteor slicer for the geometry
    (the slicer-free footprint is the voxel cross-section -- edges limited by the
    sim grid; crisp true-3D edges are stage_3d's slicer job)."""
    from studio3d.transfer import dg0_to_voxel

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    m = np.load(map_npz, allow_pickle=True)
    centroids = m["centroids"].astype(float)
    s_map = m["s_map"].astype(float)
    volumes = m["volumes"].astype(float)
    pj = np.load(part_npz)
    part = pj["part"].astype(bool)
    n = int(pj["n"]); h = float(pj["h"]); chamber_m = n * h
    job_name = Path(map_npz).stem

    # frame: re-center the map centroids onto the part voxel frame (the map sits in
    # the gmsh outline frame; the voxel part is chamber-centered). voxelize_stl does
    # this in the Windows path; do it explicitly for the map-only entry.
    bb_c = 0.5 * (centroids.min(0) + centroids.max(0))
    rec = dg0_to_voxel(centroids - bb_c, s_map, volumes, part, chamber_m=chamber_m)
    sat_vox = rec["sat"]

    sat_stack, mask_stack, z_mm = voxel_to_layer_stack(sat_vox, part, h)
    nz, ny, nx = sat_stack.shape
    x_mm = centered_axis_mm(nx, h)
    y_mm = centered_axis_mm(ny, h)
    np.savez_compressed(out / "correction_stack.npz",
                        sat=sat_stack, part_mask=mask_stack,
                        x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
                        width_mm=float(nx * h * 1e3), height_mm=float(ny * h * 1e3),
                        chamber_m=chamber_m, proxy_field="solve")

    # printer-resolution level_map (0..head ceiling 7), one page per z -> fgm_to_rip
    levels = np.stack([
        layer_to_levels(sat_stack[k], mask_stack[k], h, dpi=dpi, bpp=bpp,
                        grey_levels=grey_levels)
        for k in range(nz)]).astype(np.uint8)
    px_h, px_w = levels.shape[1], levels.shape[2]
    lm_path = out / "fgm_level_map.npz"
    np.savez_compressed(lm_path, level_map=levels, bpp=int(bpp), dpi=int(dpi),
                        grey_levels=int(grey_levels),
                        width_mm=float(px_w / dpi * 25.4),
                        height_mm=float(px_h / dpi * 25.4),
                        x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, proxy_field="solve")

    # delegate the real MetPrint TIFF write to meteor_rip via fgm_to_rip (sanctioned
    # 4bpp/LZW/WhiteIsZero writer, 3-bit head cap). Optional: absent tools -> the
    # level_map npz above is the handoff.
    tiff_writer, tiff_paths = "deferred (run fgm_to_rip on fgm_level_map.npz)", []
    try:
        _tools = Path(__file__).resolve().parents[3] / "software" / "meteor" / "tools"
        if str(_tools) not in sys.path:
            sys.path.insert(0, str(_tools))
        from fgm_to_rip import fgm_to_tiff_stack           # noqa: E402
        tiff_paths = fgm_to_tiff_stack(str(lm_path), str(out / "print_job"),
                                       n_layers=nz, job_name=job_name, bpp=bpp,
                                       dpi=dpi, compression="lzw")
        tiff_writer = "meteor_rip.fgm_to_tiff_stack (4bpp/LZW/WhiteIsZero)"
    except Exception as e:                                # tools not present here
        tiff_writer = f"deferred ({type(e).__name__}: run fgm_to_rip on fgm_level_map.npz)"

    manifest = {
        "engine_version": engine_version, "proxy_field": "solve",
        "dpi": int(dpi), "bpp": int(bpp), "grey_levels": int(grey_levels),
        "head_ceiling": int(pq.max_level(bpp, grey_levels)),
        "layer_height_mm": float(h * 1e3), "layer_count": int(nz),
        "raster_px": [int(px_h), int(px_w)],
        "printed_size_mm": [float(px_w / dpi * 25.4), float(px_h / dpi * 25.4)],
        "chamber_m": float(chamber_m),
        "level_map_npz": str(lm_path),
        "tiff_writer": tiff_writer,
        "tiff_count": len(tiff_paths),
        "dopant_mass_move_rel": float(rec.get("dopant_mass_move_rel", float("nan"))),
        "dg0_state": str(rec.get("state", "")),
        "note": "fgm_level_map.npz is the sanctioned meteor_rip/fgm_to_rip input; the "
                "4bpp 3-bit-in-nibble WhiteIsZero LZW encoding is meteor_rip's, not here.",
    }
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
