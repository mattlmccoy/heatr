"""Robustness helpers: grid transfer and rim smoothing of a solved dopant map.

Two independent perturbations of a solved map, both applied OUTSIDE the solve so
that nothing is re-optimized in the design variable:

  GRID TRANSFER   the map is solved on the 120 x 120 grid and re-scored on a
      160 x 160 grid. The resampling convention is the production
      map-injection one, `rfam_eqs_coupled.py:374-380`:

          zy = ny_sim / sat.shape[0] ; zx = nx_sim / sat.shape[1]
          if abs(zy - 1) > 0.01 or abs(zx - 1) > 0.01:
              sat = scipy.ndimage.zoom(sat, (zy, zx), order=1)
          sat = clip(sat, 0, 1)

      The same call appears in the direct-map branch at
      `rfam_eqs_coupled.py:335-340`. Bilinear, then clipped; no smoothing, no
      area weighting. It is reproduced here rather than re-derived.

  RIM SMOOTHING   the continuous solved map is blurred by a Gaussian of 1 or 2
      cells before it is quantized. The blur is a NORMALIZED CONVOLUTION over
      the part only,

          out = gaussian(s * chi) / gaussian(chi)   on the part,

      so the nominal saturation held outside the part (the prototype convention,
      s = 1) cannot bleed inward and change the answer. Outside the part the map
      is held at the nominal value, exactly as `printability.quantize_in_part`
      does, so an arm changes the dopant map and not the sub-pixel geometry fill.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

ZOOM_DEADBAND = 0.01   # rfam_eqs_coupled.py:378


def resample_map(sat: np.ndarray, ny: int, nx: int, sat_max: float = 1.0) -> np.ndarray:
    """Resample to (ny, nx) in the production map-injection convention."""
    a = np.asarray(sat, dtype=float)
    zy = float(ny) / a.shape[0]
    zx = float(nx) / a.shape[1]
    if abs(zy - 1.0) > ZOOM_DEADBAND or abs(zx - 1.0) > ZOOM_DEADBAND:
        a = zoom(a, (zy, zx), order=1)
    return np.clip(a, 0.0, float(sat_max))


def resample_mask(mask: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Nearest-neighbour transfer of a boolean mask (diagnostic use only)."""
    m = np.asarray(mask, dtype=float)
    zy = float(ny) / m.shape[0]
    zx = float(nx) / m.shape[1]
    if abs(zy - 1.0) > ZOOM_DEADBAND or abs(zx - 1.0) > ZOOM_DEADBAND:
        m = zoom(m, (zy, zx), order=0)
    return m > 0.5


def smooth_in_part(sat: np.ndarray, part_mask: np.ndarray, sigma_cells: float,
                   outside: float = 1.0, sat_max: float = 1.0) -> np.ndarray:
    """Part-masked Gaussian blur of the continuous map, held at `outside` outside."""
    if float(sigma_cells) < 0.0:
        raise ValueError(f"sigma_cells must be non-negative, got {sigma_cells!r}")
    a = np.asarray(sat, dtype=float)
    pm = np.asarray(part_mask, dtype=bool)
    sig = float(sigma_cells)
    if sig == 0.0:
        sm = a
    else:
        chi = pm.astype(float)
        num = gaussian_filter(a * chi, sig, mode="constant", cval=0.0)
        den = gaussian_filter(chi, sig, mode="constant", cval=0.0)
        sm = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return np.where(pm, np.clip(sm, 0.0, float(sat_max)), float(outside))


def recalibrated_voltage(v0: float, p_measured: float, p_target: float = 500.0) -> float:
    """The drive voltage that moves absorbed power from `p_measured` to `p_target`.

    The electro-quasi-static solve is linear in the applied potential and the
    conductivity and permittivity fields do not depend on it, so the absorbed
    power is exactly quadratic in the drive: P = c * V^2. One rescale is exact,
    no iteration. This reproduces the campaign's own calibration convention (the
    drive voltage is chosen so the UNIFORM arm absorbs 500 W per metre of depth)
    at a grid where that calibration no longer holds.
    """
    if float(p_measured) <= 0.0:
        raise ValueError(f"p_measured must be positive, got {p_measured!r}")
    return float(v0) * float(np.sqrt(float(p_target) / float(p_measured)))
