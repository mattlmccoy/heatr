"""Gate S4 library: measured FLIR field extraction, data-derived registration,
and the metric operators applied IDENTICALLY to measured and predicted fields.

Nothing here changes heatr3d or the FLIR pipeline; the decoders
(extract_flir_seq / triage_flir_archive) are imported read-only.

Registration and metric definitions are pre-registered in README.md sections 4-5.
"""
from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, label, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_flir_seq import Radiometry, counts_to_celsius  # noqa: E402
from flir_archive_paths import find as find_archive_file  # noqa: E402
from triage_flir_archive import SeqLayout, _read_frame  # noqa: E402

logger = logging.getLogger(__name__)

PART_MM = 40.0        # inferred part footprint (FE.m mirrored), README section 2
E60_HFOV_DEG = 25.0   # FLIR E60 + FOL18 horizontal field of view
E60_WIDTH_PX = 320


# --------------------------------------------------------------------------- #
# measured fields
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rect:
    """Minimum-area rotated rectangle of a mask, in pixel coordinates."""
    cx: float
    cy: float
    angle_deg: float
    side_u: float
    side_v: float


def optical_mm_per_px(distance_m: float) -> float:
    """Independent scale cross-check from the camera optics (not used to register)."""
    half = np.tan(np.deg2rad(E60_HFOV_DEG / 2.0)) * distance_m * 1000.0
    return 2.0 * half / E60_WIDTH_PX


def read_seq(name: str, n_samples: int = 240, emissivity: float | None = None
             ) -> Dict[str, np.ndarray]:
    """Decode `n_samples` evenly spaced frames of an archive .seq to deg C.

    emissivity=None uses each frame's as-recorded value; a float overrides it
    (the one-sided uncertainty band of README section 6.3).
    """
    path = find_archive_file(name if name.endswith(".seq") else name + ".seq")
    size = path.stat().st_size
    with open(path, "rb") as fh:
        lay = SeqLayout(fh, size)
        idxs = sorted(set(np.linspace(0, lay.n_frames - 1, n_samples).round().astype(int)))
        frames, times = [], []
        rad0 = None
        for i in idxs:
            rad, counts = _read_frame(fh, lay, i)
            if emissivity is not None:
                rad = dataclasses.replace(rad, emissivity=float(emissivity))
            if rad0 is None:
                rad0 = rad
            frames.append(counts_to_celsius(counts, rad).astype(np.float32))
            times.append(rad.time_s + rad.time_ms / 1000.0)
    t = np.array(times) - times[0]
    return {
        "t_s": t,
        "frames": np.array(frames),
        "idx": np.array(idxs),
        "n_frames": lay.n_frames,
        "emissivity": float(rad0.emissivity),
        "distance_m": float(rad0.distance_m),
        "t_refl_c": float(rad0.t_refl_k - 273.15),
        "path": str(path),
    }


def largest_component_roi(field: np.ndarray, ambient: float, frac: float = 0.5,
                          median_size: int = 3) -> np.ndarray:
    """README section 4 steps 2-3: threshold at ambient + frac*(Tmax-ambient),
    keep the largest connected component."""
    sm = median_filter(np.asarray(field, dtype=float), size=median_size)
    thresh = ambient + frac * (float(np.nanmax(sm)) - ambient)
    mask = sm >= thresh
    lab, n = label(mask)
    if n == 0:
        raise ValueError("no pixels above the ROI threshold")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    roi = lab == int(np.argmax(sizes))
    frag = roi.sum() / max(int(mask.sum()), 1)
    if frag < 0.6:
        # The threshold set broke into pieces (e.g. a hollow/corner-only hot
        # region). Registration on one fragment would be wrong; make it loud.
        logger.warning("ROI FRAGMENTED: largest component holds only %.0f%% of the "
                       "above-threshold pixels; registration may be invalid.",
                       100.0 * frag)
    return roi


def min_area_rect(mask: np.ndarray, angle_step_deg: float = 0.25) -> Rect:
    """Minimum-area rotated bounding rectangle by exhaustive rotation search over
    the mask's boundary points (0-90 deg is sufficient: rectangles repeat)."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size < 4:
        raise ValueError("mask too small for a rectangle fit")
    pts = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    best = None
    for a_deg in np.arange(0.0, 90.0, angle_step_deg):
        a = np.deg2rad(a_deg)
        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = pts @ rot.T
        lo, hi = q.min(axis=0), q.max(axis=0)
        # +1 px: a pixel centre span of N-1 covers N pixels of area
        w, h = (hi - lo) + 1.0
        area = w * h
        if best is None or area < best[0]:
            cq = 0.5 * (lo + hi)
            c = rot.T @ cq
            best = (area, a_deg, w, h, float(c[0]), float(c[1]))
    _, a_deg, w, h, cx, cy = best
    return Rect(cx=cx, cy=cy, angle_deg=float(a_deg), side_u=float(w), side_v=float(h))


SPAN = 0.90   # sampled fraction of the fitted footprint; see DEVIATION note below


def resample_unit_square(field: np.ndarray, rect: Rect, ng: int = 64,
                         smooth_cells: float = 1.0, span: float = SPAN) -> np.ndarray:
    """Sample `field` on the part-relative grid defined by `rect`
    (README section 4 step 4). Bilinear; out-of-frame -> nearest edge.

    DEVIATION from README section 4 (pre-registered as u,v in [-0.5, 0.5]):
    sampling stops at +-0.45 (span=0.90). Sampling exactly at +-0.5 lands on the
    material boundary, where one sub-pixel of registration error swings a sample
    between part (~200 C) and powder (~20 C); that made the correlation a
    measure of edge alignment rather than of pattern. The cut is applied
    IDENTICALLY to measured and predicted fields and still contains the corner
    hot spots (+-18 mm of the 20 mm half-width). Recorded in S4_GATE_REPORT.md.
    """
    f = np.asarray(field, dtype=float)
    ny, nx = f.shape
    g = np.linspace(-span / 2.0, span / 2.0, ng)
    U, V = np.meshgrid(g, g, indexing="ij")
    a = np.deg2rad(rect.angle_deg)
    # rect frame -> pixel frame (inverse of the rotation used in min_area_rect)
    du, dv = U * rect.side_u, V * rect.side_v
    px = rect.cx + du * np.cos(a) - dv * np.sin(a)
    py = rect.cy + du * np.sin(a) + dv * np.cos(a)
    x0 = np.clip(np.floor(px).astype(int), 0, nx - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, ny - 2)
    tx = np.clip(px - x0, 0.0, 1.0)
    ty = np.clip(py - y0, 0.0, 1.0)
    out = ((1 - tx) * (1 - ty) * f[y0, x0] + tx * (1 - ty) * f[y0, x0 + 1]
           + (1 - tx) * ty * f[y0 + 1, x0] + tx * ty * f[y0 + 1, x0 + 1])
    if smooth_cells > 0:
        out = gaussian_filter(out, sigma=smooth_cells)
    return out


# --------------------------------------------------------------------------- #
# metric operators (identical on measured and predicted)
# --------------------------------------------------------------------------- #
def normalize_rise(field: np.ndarray, ambient: float) -> np.ndarray:
    """(T - T_ambient) / mean(T - T_ambient): removes any residual scale error so
    the metric sees pattern only."""
    r = np.asarray(field, dtype=float) - ambient
    m = float(np.mean(r))
    return r / m if abs(m) > 1e-12 else r


def pattern_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 0 else float("nan")


def corner_edge_contrast(field: np.ndarray) -> float:
    """5x5 block corner-minus-edge-mid contrast, the X-pattern discriminator of
    triage_flir_archive._spatial_signature, applied to a part-relative field."""
    f = np.asarray(field, dtype=float)
    n, m = f.shape

    def blk(i: int, j: int) -> float:
        return float(np.nanmean(f[i * n // 5:(i + 1) * n // 5, j * m // 5:(j + 1) * m // 5]))

    corners = np.mean([blk(0, 0), blk(0, 4), blk(4, 0), blk(4, 4)])
    edges = np.mean([blk(0, 2), blk(2, 0), blk(2, 4), blk(4, 2)])
    return float(corners - edges)


def _hot_set(field: np.ndarray, top_frac: float) -> np.ndarray:
    """The top `top_frac` of pixels by RANK (not by quantile threshold, which
    degenerates when the field has large flat regions)."""
    f = np.asarray(field, dtype=float)
    k = max(1, int(round(top_frac * f.size)))
    flat = np.argsort(f, axis=None)[::-1][:k]
    out = np.zeros(f.size, bool)
    out[flat] = True
    return out.reshape(f.shape)


def chamfer_mm(measured: np.ndarray, predicted: np.ndarray, part_mm: float = PART_MM,
               top_frac: float = 0.10) -> float:
    """Mean nearest-neighbour distance from the measured hot set to the predicted
    hot set, in mm on the part (README M3). Multiplicity-aware: a 4-corner hot
    pattern matched by a 4-corner hot pattern scores 0."""
    a = _hot_set(measured, top_frac)
    b = _hot_set(predicted, top_frac)
    ng = a.shape[0]
    step_mm = part_mm * SPAN / (ng - 1)
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    if ya.size == 0 or yb.size == 0:
        return float("nan")
    d = np.sqrt((ya[:, None] - yb[None, :]) ** 2 + (xa[:, None] - xb[None, :]) ** 2)
    return float(d.min(axis=1).mean() * step_mm)


def argmax_offset_mm(measured: np.ndarray, predicted: np.ndarray,
                     part_mm: float = PART_MM) -> float:
    a = np.asarray(measured, dtype=float)
    b = np.asarray(predicted, dtype=float)
    ia = np.unravel_index(np.argmax(a), a.shape)
    ib = np.unravel_index(np.argmax(b), b.shape)
    step_mm = part_mm * SPAN / (a.shape[0] - 1)
    return float(np.hypot(ia[0] - ib[0], ia[1] - ib[1]) * step_mm)


# --------------------------------------------------------------------------- #
# curve helpers
# --------------------------------------------------------------------------- #
def normalized_rise_curve(t: np.ndarray, y: np.ndarray, ambient: float) -> np.ndarray:
    r = np.asarray(y, dtype=float) - ambient
    denom = r[-1] if abs(r[-1]) > 1e-9 else 1.0
    return r / denom


def time_at_fraction(t: np.ndarray, theta: np.ndarray, frac: float) -> float:
    """First crossing time of a normalized rise curve (linear interpolation)."""
    th = np.asarray(theta, dtype=float)
    idx = np.nonzero(th >= frac)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1, y0, y1 = t[i - 1], t[i], th[i - 1], th[i]
    return float(t0 + (frac - y0) * (t1 - t0) / max(y1 - y0, 1e-12))


def time_at_value(t: np.ndarray, y: np.ndarray, value: float) -> float:
    yy = np.asarray(y, dtype=float)
    idx = np.nonzero(yy >= value)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (value - yy[i - 1]) * (t[i] - t[i - 1])
                 / max(yy[i] - yy[i - 1], 1e-12))
