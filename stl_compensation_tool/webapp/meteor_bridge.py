"""Bridge between the grading pipeline and the Meteor RIP & Slice tool.

Pure logic here is unit-tested (test_meteor_bridge.py). The Meteor imports
(slicer, meteor_rip) are resolved lazily from the configured tools directory
so the pure functions stay importable without the Meteor environment.

Frames (probed from real code, see test docstring):
- Print canvas: pixels[r, c], r=0 is TOP, y_mm = (height_px - r) / ppm.
- Dopant grid: sat[iy, ix], x/y ascending, part centered at (0, 0), cell
  pitch = chamber / (n - 1) with the canonical 60 mm / 120 grid.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates

CHAMBER_M = 0.060          # canonical HEATR chamber (configs/shape_circle_6min)
DOPANT_N = 120


def layer_to_slice_index(layer_idx: int, layer_height_mm: float,
                         z_slices_mm: np.ndarray) -> int:
    """Nearest analysis-slice index for a print layer (mid-layer z), clamped."""
    z = (layer_idx + 0.5) * layer_height_mm
    return int(np.argmin(np.abs(np.asarray(z_slices_mm, float) - z)))


def decode_tiff_gray_to_levels(gray: np.ndarray, bpp: int) -> np.ndarray:
    """Invert the Meteor TIFF encoding (black = max ink): PIL reads the
    written TIFF as 8-bit gray = 255 * (1 - level / max_val). Probed
    2026-07-30 against meteor_rip.write_tiff round-trip."""
    max_val = (1 << int(bpp)) - 1
    lv = np.rint((255.0 - np.asarray(gray, np.float64)) / 255.0 * max_val)
    return np.clip(lv, 0, max_val).astype(np.uint8)


# 8x8 Bayer matrix: deterministic ordered-dither thresholds in [0, 1).
# Built recursively from the 2x2 base; classic index construction.
def _bayer8() -> np.ndarray:
    m = np.array([[0, 2], [3, 1]], float)
    for _ in range(2):
        m = np.block([[4 * m + 0, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


_BAYER8 = _bayer8()


def apply_sat_to_levels(levels: np.ndarray, sat: np.ndarray, bpp: int,
                        dither: str | None = "ordered") -> np.ndarray:
    """Scale a rasterized ink-level image by a continuous saturation map.

    Preserves the shell/infill structure rasterize_zones produced (0 stays 0);
    clips to the bpp ceiling so sat > 1 (per-node two-sided maps) cannot
    overflow the TIFF encoding.

    dither="ordered" (default, spec 7b amendment 2026-08-03): Bayer 8x8
    ordered dithering of the fractional level, so the LOCAL AVERAGE printed
    dose tracks the continuous map instead of carrying a uniform up-to-half-
    level rounding bias. Deterministic (no random numbers); a zero level
    stays exactly zero. dither=None reproduces the legacy per-pixel round.
    """
    max_val = (1 << int(bpp)) - 1
    desired = levels.astype(np.float64) * np.asarray(sat, np.float64)
    if dither is None:
        out = np.rint(desired)
    elif dither == "ordered":
        h, w = desired.shape
        thresh = np.tile(_BAYER8,
                         (h // 8 + 1, w // 8 + 1))[:h, :w]
        base = np.floor(desired)
        frac = desired - base
        out = base + (frac > thresh)
    else:
        raise ValueError(f"unknown dither mode {dither!r}")
    out = np.clip(out, 0, max_val).astype(np.uint8)
    out[levels == 0] = 0
    return out


def sample_sat_on_canvas(sat: np.ndarray, canvas_w_px: int, canvas_h_px: int,
                         ppm: float, part_center_canvas_mm: tuple,
                         chamber_m: float = CHAMBER_M) -> np.ndarray:
    """Resample a dopant-grid sat map onto the print canvas pixel frame.

    Physical mapping: canvas position (mm) minus the part center (mm) equals
    the dopant-frame position (part centered at dopant origin). Outside the
    dopant grid the edge value is extended (background sat 1.0 in practice).
    """
    sat = np.asarray(sat, float)
    n_y, n_x = sat.shape
    cx_mm, cy_mm = part_center_canvas_mm
    r = np.arange(canvas_h_px)
    c = np.arange(canvas_w_px)
    cc, rr = np.meshgrid(c, r)
    x_mm = cc / ppm - cx_mm                      # dopant-frame x, mm
    y_mm = (canvas_h_px - rr) / ppm - cy_mm      # y-flip: row 0 = top = +y
    half_mm = chamber_m * 1e3 / 2.0
    ix = (x_mm + half_mm) / (chamber_m * 1e3) * (n_x - 1)
    iy = (y_mm + half_mm) / (chamber_m * 1e3) * (n_y - 1)
    return map_coordinates(sat, [iy, ix], order=1, mode="nearest")


# ---------------------------------------------------------------------------
# Meteor tool integration (lazy imports; not unit-tested, exercised by the
# end-to-end run against the real slicer)
# ---------------------------------------------------------------------------

def grade_tiff_stack(tiff_paths: list, dopant_volume_npz: str | Path,
                     layer_height_mm: float, bpp: int, dpi: int,
                     compression: str, meteor_dir: str | Path,
                     center_offset_m: tuple = (0.0, 0.0)) -> dict:
    """Apply the per-slice dopant maps to an already-sliced TIFF stack, in
    place. Alignment: the analysis frame centered the part on its mid-slice
    centroid; on the canvas the same point is the mid-layer ink centroid.

    Refuses stacks produced with scan_rotation != 0 (caller must check):
    the sat map is defined in the unrotated frame.
    """
    add_meteor_path(meteor_dir)
    from PIL import Image
    from meteor_rip import write_tiff

    vol = np.load(dopant_volume_npz, allow_pickle=False)
    sat_vol = vol["sat"]
    z_slices = np.asarray(vol["z_mm"], float)
    chamber = float(vol["chamber_m"]) if "chamber_m" in vol.files else CHAMBER_M
    ppm = dpi / 25.4

    # part center on canvas from the mid-layer ink centroid (mm, y-up frame)
    mid = tiff_paths[len(tiff_paths) // 2]
    arr = np.array(Image.open(mid))
    lv = decode_tiff_gray_to_levels(arr, bpp)
    rr, cc = np.nonzero(lv)
    if len(rr) == 0:
        raise RuntimeError("mid-layer TIFF has no ink; cannot align grading")
    h_px = lv.shape[0]
    cx_mm = cc.mean() / ppm - center_offset_m[0] * 1e3
    cy_mm = (h_px - rr.mean()) / ppm - center_offset_m[1] * 1e3

    graded = 0
    sat_stats = []
    for k, tp in enumerate(tiff_paths):
        arr = np.array(Image.open(tp))
        lv = decode_tiff_gray_to_levels(arr, bpp)
        si = layer_to_slice_index(k, layer_height_mm, z_slices)
        sat_canvas = sample_sat_on_canvas(
            sat_vol[si], canvas_w_px=lv.shape[1], canvas_h_px=lv.shape[0],
            ppm=ppm, part_center_canvas_mm=(cx_mm, cy_mm), chamber_m=chamber)
        out = apply_sat_to_levels(lv, sat_canvas, bpp)
        write_tiff(str(tp), out, bpp, dpi, compression)
        graded += 1
        ink = lv > 0
        if ink.any():
            sat_stats.append(float(sat_canvas[ink].mean()))
    return {"layers_graded": graded,
            "part_center_canvas_mm": [cx_mm, cy_mm],
            "mean_sat_over_ink": (float(np.mean(sat_stats)) if sat_stats else None)}


def add_meteor_path(meteor_dir: str | Path) -> None:
    p = str(Path(meteor_dir))
    if p not in sys.path:
        sys.path.insert(0, p)


def analysis_polygons(mesh_path: str, meteor_dir: str | Path,
                      layer_mm: float, params: Optional[dict] = None) -> list:
    """Slice a mesh with the METEOR slicer into analysis polygons (metres).

    Returns the same slice-record list slicer_cli.py produced, so the
    geo-prewarp pipeline consumes it unchanged: largest exterior loop per z,
    centered on the part centroid (HEATR wants the part at the origin),
    n_loops / has_holes honesty flags.
    """
    add_meteor_path(meteor_dir)
    import slicer as msl  # Meteor slicer

    tris = msl.load_mesh(mesh_path)
    p = dict(msl.DEFAULT_PARAMS)
    p.update(params or {})
    tris = msl.apply_transform(tris, p)
    z_min = float(tris[:, :, 2].min())
    z_max = float(tris[:, :, 2].max())
    out = []
    for z in np.arange(z_min + layer_mm / 2, z_max, layer_mm):
        segs = msl.slice_at_z(tris, float(z))
        geom = msl.chain_to_polygons(segs)           # shapely (Multi)Polygon
        if geom is None or geom.is_empty:
            continue
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        largest = max(polys, key=lambda q: q.area)
        ext = np.asarray(largest.exterior.coords)[:-1]    # mm
        # faithful display outlines: EVERY ring (islands + holes), simplified
        # with topology-preserving tolerance instead of naive decimation
        tol = max(0.05, float(np.sqrt(largest.area)) / 400.0)   # mm
        loops = []
        for q in polys:
            qs = q.simplify(tol, preserve_topology=True)
            loops.append({"pts": np.asarray(qs.exterior.coords).round(3).tolist(),
                          "hole": False})
            for ring in qs.interiors:
                loops.append({"pts": np.asarray(ring.coords).round(3).tolist(),
                              "hole": True})
        out.append({
            "z_mm": float(z - z_min),
            "polygon_m": (ext / 1000.0).tolist(),
            "loops_mm": loops,
            "n_loops": int(len(polys)),
            "has_holes": bool(any(len(q.interiors) for q in polys)),
            "area_mm2_all_loops": float(sum(q.area for q in polys)),
        })
    # center the stack on the mid-height slice's centroid so HEATR sees the
    # part at the origin (analysis frame), remembering the offset for later
    if out:
        mid = out[len(out) // 2]
        c = np.asarray(mid["polygon_m"]).mean(axis=0)
        for s in out:
            s["polygon_m"] = (np.asarray(s["polygon_m"]) - c).tolist()
            s["center_offset_m"] = [float(c[0]), float(c[1])]
    return out
