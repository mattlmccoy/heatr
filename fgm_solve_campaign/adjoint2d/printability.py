"""Printer quantization, in the SAME convention as the production path.

The printer rasterizes binder saturation at 2 bits per pixel (4 levels) or
4 bits per pixel (16 levels). There is no continuous grading in hardware, so a
solved continuous dopant map is not by itself a printable artifact. This module
puts the solved maps through the production quantizer so the fidelity numbers
are quoted for something the machine can actually lay down.

Production references, read and reproduced (not re-derived):

  `fgm_generator.py:583-586`
      n_levels = 1 << bpp ; max_val = n_levels - 1
      level_map_sim = np.round(sat_scaled * max_val).astype(np.uint8)

  `fgm_generator.py:593-606`
      px_m = 25.4e-3 / dpi ; zx = dx_m / px_m ; zy = dy_m / px_m
      sat_dpi = clip(zoom(sat_scaled, (zy, zx), order=1), 0, 1)
      level_map_dpi = clip(round(sat_dpi * max_val), 0, max_val).astype(uint8)

  `rfam_eqs_coupled.py:366-380` (the loader that inverts the round trip)
      sat = level_map.astype(float32) / max_val
      sat = zoom(sat, (ny_sim/ny_dpi, nx_sim/nx_dpi), order=1)   when needed
      sat = clip(sat, 0, 1).astype(float32)

Two named conventions:

  SIM-RESOLUTION quantization  `quantize_levels`, the value grid k/max_val at
      the simulation grid. This is what the printer can address per pixel with
      no resampling, and it is the clean statement of the bit-depth cost.

  PRINTER ROUND TRIP  `printer_round_trip`, the full up-sample to printer dots
      per inch, quantize, and resample back that the production pipeline
      performs. It adds a resampling blur on top of the bit-depth cost.

Saturation above 1.0 is a SECOND printing pass, not a clipped value. The ink is
25 weight percent carbon black in isopropyl alcohol and one pass saturates at
s = 1. `sat_max` therefore names the number of passes allowed: `sat_max = 1.0`
is single pass and clips, `sat_max = 1.5` admits the double-pass region while
keeping the same 1/max_val level quantum.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

INCH_M = 25.4e-3


def max_level(bpp: int) -> float:
    if int(bpp) not in (2, 4):
        raise ValueError(f"bpp must be 2 or 4, got {bpp!r}")
    return float((1 << int(bpp)) - 1)


def quantize_levels(sat: np.ndarray, bpp: int = 4, sat_max: float = 1.0) -> np.ndarray:
    """Snap saturation to the printer's level grid k/max_val, clipped at sat_max."""
    mv = max_level(bpp)
    a = np.clip(np.asarray(sat, dtype=float), 0.0, float(sat_max))
    return np.round(a * mv) / mv


def quantize_in_part(sat: np.ndarray, part_mask: np.ndarray, bpp: int = 4,
                     sat_max: float = 1.0, outside: float = 1.0) -> np.ndarray:
    """Quantize the design domain only; hold the nominal value outside the part.

    The design variable lives on the part. Outside it the prototype convention
    is s = 1 so that an arm changes the dopant map and nothing else (the
    sub-pixel geometry fill of boundary cells is preserved). Quantizing the
    outside would change the geometry, not the print.
    """
    q = quantize_levels(sat, bpp=bpp, sat_max=sat_max)
    return np.where(np.asarray(part_mask, dtype=bool), q, float(outside))


def printer_level_map(sat: np.ndarray, dx_m: float, dy_m: float, dpi: int = 720,
                      bpp: int = 4) -> np.ndarray:
    """The printer-resolution integer level map, `fgm_generator.py:593-606`."""
    mv = max_level(bpp)
    px_m = INCH_M / float(dpi)
    zx = float(dx_m) / px_m
    zy = float(dy_m) / px_m
    a = np.asarray(sat)
    if abs(zx - 1.0) > 0.01 or abs(zy - 1.0) > 0.01:
        dpi_map = zoom(a, (zy, zx), order=1)
        dpi_map = np.clip(dpi_map, 0.0, 1.0)
    else:
        dpi_map = a
    return np.clip(np.round(dpi_map * mv).astype(np.uint8), 0, mv).astype(np.uint8)


def load_level_map(level_map: np.ndarray, nx_sim: int, ny_sim: int,
                   bpp: int = 4) -> np.ndarray:
    """The production loader's inverse, `rfam_eqs_coupled.py:366-380`."""
    mv = np.float32(max_level(bpp))
    sat = np.asarray(level_map).astype(np.float32) / mv
    zy = ny_sim / sat.shape[0]
    zx = nx_sim / sat.shape[1]
    if abs(zy - 1.0) > 0.01 or abs(zx - 1.0) > 0.01:
        sat = zoom(sat, (zy, zx), order=1)
    return np.clip(sat, 0.0, 1.0).astype(np.float32)


def printer_round_trip(sat: np.ndarray, x: np.ndarray, y: np.ndarray,
                       bpp: int = 4, dpi: int = 720) -> np.ndarray:
    """Up-sample to printer resolution, quantize, resample back to the grid."""
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    lm = printer_level_map(sat, dx_m=dx, dy_m=dy, dpi=dpi, bpp=bpp)
    return np.asarray(load_level_map(lm, len(x), len(y), bpp=bpp), dtype=float)


def write_printed_map(sat: np.ndarray, x: np.ndarray, y: np.ndarray,
                      path: str | Path, bpp: int = 4, dpi: int = 720) -> np.ndarray:
    """Write an npz the production loader accepts; return what it will read back."""
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    lm = printer_level_map(sat, dx_m=dx, dy_m=dy, dpi=dpi, bpp=bpp)
    np.savez_compressed(
        Path(path), level_map=lm, sat_map=np.asarray(sat, dtype=np.float32),
        x_mm=np.asarray(x, dtype=float) * 1000.0,
        y_mm=np.asarray(y, dtype=float) * 1000.0,
        bpp=np.array(int(bpp), dtype=np.int32),
        n_levels=np.array(1 << int(bpp), dtype=np.int32),
        dpi=np.array(int(dpi), dtype=np.int32),
    )
    return np.asarray(load_level_map(lm, len(x), len(y), bpp=bpp), dtype=float)


def level_census(sat: np.ndarray, part_mask: np.ndarray, bpp: int) -> dict:
    """How many distinct printer levels the map actually uses inside the part."""
    mv = max_level(bpp)
    v = np.asarray(sat, dtype=float)[np.asarray(part_mask, dtype=bool)]
    lv = np.round(v * mv).astype(int)
    return {"n_levels_used": int(np.unique(lv).size),
            "level_min": int(lv.min()), "level_max": int(lv.max()),
            "frac_above_single_pass": float(np.mean(v > 1.0))}
