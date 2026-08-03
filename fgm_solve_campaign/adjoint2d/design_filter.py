"""Physical-length design filter, applied INSIDE the solve.

`SOLVE_ROBUSTNESS_VALIDATION.md` Task B measured that the previously solved
dopant maps carry load-bearing structure at the single-cell scale: a one-cell
part-masked Gaussian blur of the solved continuous map costs between +37.3
percent (triangle) and +891.9 percent (circle) of the shape objective on five
of six shapes. Its Section 10 names the fix: filter the design variable rather
than the final map, so structure below the filter length is not expressible at
all and cannot be bought.

The map is

    s = F(v) = gaussian(v * chi) / gaussian(chi)      inside the part
    s = outside                                        outside the part

a NORMALIZED CONVOLUTION over the part only, exactly the convention of
`robust.smooth_in_part`, with two differences that matter for a solve:

  * there is no clip. A normalized convolution is a convex combination of
    in-part values, so v in [lo, hi] gives s in [lo, hi] automatically. The box
    on the design variable is therefore the box on the injected map, exactly,
    and no clip subgradient enters the chain rule. (`test_box_is_preserved`.)
  * the transpose is provided. F is linear in v at fixed masks, and with the
    nominal outside value taken as an affine offset that carries no design
    sensitivity, F^T g = chi * gaussian(chi * g / gaussian(chi)). A Gaussian
    correlation with zero padding is self-adjoint because its kernel is
    symmetric, which is what makes that expression exact and is checked by the
    dot-product identity in `test_design_filter.py`.

RADIUS CONVENTION. `sigma_cells` is the Gaussian standard deviation in grid
cells, which is what `scipy.ndimage.gaussian_filter` takes and what Task B
swept. At the pinned geometry one cell is 0.5 mm at grid 120, so sigma = 1.5
cells is a 0.75 mm design length. It is a PHYSICAL length: quoted in cells only
because the whole campaign runs at one grid.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_SIGMA_CELLS = 1.5
_DEN_FLOOR = 1e-12


def _weights(part_mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    chi = np.asarray(part_mask, dtype=bool).astype(float)
    return gaussian_filter(chi, float(sigma_cells), mode="constant", cval=0.0)


def apply_filter(v: np.ndarray, part_mask: np.ndarray, sigma_cells: float,
                 outside: float = 1.0) -> np.ndarray:
    """The filtered design map s = F(v), held at `outside` outside the part."""
    if float(sigma_cells) < 0.0:
        raise ValueError(f"sigma_cells must be non-negative, got {sigma_cells!r}")
    a = np.asarray(v, dtype=float)
    pm = np.asarray(part_mask, dtype=bool)
    sig = float(sigma_cells)
    if sig == 0.0:
        sm = a
    else:
        num = gaussian_filter(a * pm.astype(float), sig, mode="constant", cval=0.0)
        den = _weights(pm, sig)
        sm = np.divide(num, den, out=np.zeros_like(num), where=den > _DEN_FLOOR)
    return np.where(pm, sm, float(outside))


def filter_vjp(g_s: np.ndarray, part_mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    """dJ/dv given dJ/ds. Supported on the part, zero outside."""
    if float(sigma_cells) < 0.0:
        raise ValueError(f"sigma_cells must be non-negative, got {sigma_cells!r}")
    g = np.asarray(g_s, dtype=float)
    pm = np.asarray(part_mask, dtype=bool)
    sig = float(sigma_cells)
    if sig == 0.0:
        return np.where(pm, g, 0.0)
    den = _weights(pm, sig)
    inner = np.divide(np.where(pm, g, 0.0), den,
                      out=np.zeros_like(den), where=den > _DEN_FLOOR)
    return np.where(pm, gaussian_filter(inner, sig, mode="constant", cval=0.0), 0.0)


def roughness_in_part(s: np.ndarray, part_mask: np.ndarray) -> float:
    """Mean absolute first difference across in-part faces, a filter diagnostic."""
    a = np.asarray(s, dtype=float)
    pm = np.asarray(part_mask, dtype=bool)
    vals = []
    fx = pm[:, 1:] & pm[:, :-1]
    fy = pm[1:, :] & pm[:-1, :]
    if fx.any():
        vals.append(np.abs(a[:, 1:] - a[:, :-1])[fx])
    if fy.any():
        vals.append(np.abs(a[1:, :] - a[:-1, :])[fy])
    if not vals:
        return 0.0
    return float(np.mean(np.concatenate(vals)))
