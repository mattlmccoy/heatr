"""The control arm: the one-shot proportional-inverse map with a calibrated gain.

This reproduces the step-2 method (`FGM_CALIBRATED_CONTROL_REPORT.md`) at
simulation resolution: Gaussian smoothing of the proxy, percentile contrast
stretch inside the part, inversion, magnitude scaling about a baseline
saturation, a dead band, and a clip to [0, 1]. It is injected through the SAME
two-sided per-node hook the adjoint arm uses, so the two arms differ only in
how the map is chosen.

Two deliberate deviations from the production `fgm_generator.generate_fgm`,
both stated in the report:
  * the printer-resolution up/down sampling round trip is skipped (it is a
    printing-pipeline artifact and only adds resampling noise at 120 x 120);
  * quantization is available but off by default, so the primary comparison is
    continuous-versus-continuous.

Also holds the shared budget accounting and selection rules, which are pure
arithmetic and unit-tested.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def proportional_inverse_map(
    proxy: np.ndarray,
    part_mask: np.ndarray,
    magnitude: float,
    baseline: float = 0.5,
    dead_band: float = 0.05,
    smoothing_sigma: float = 1.5,
    clip_percentile: tuple[float, float] = (2.0, 98.0),
    outside: float = 1.0,
) -> np.ndarray:
    raw = np.asarray(proxy, dtype=np.float64)
    raw_s = gaussian_filter(raw, sigma=float(smoothing_sigma)) if smoothing_sigma > 0 else raw
    inside = raw_s[part_mask]
    lo = float(np.percentile(inside, clip_percentile[0]))
    hi = float(np.percentile(inside, clip_percentile[1]))
    span = max(hi - lo, 1e-12)
    norm = np.clip((raw_s - lo) / span, 0.0, 1.0)
    sat_raw = 1.0 - norm
    sat = baseline + float(magnitude) * (sat_raw - baseline)
    sat = np.clip(sat, 0.0, 1.0)
    if dead_band > 0.0:
        mean_norm = float(np.mean(norm[part_mask]))
        in_db = (np.abs(norm - mean_norm) <= float(dead_band)) & part_mask
        sat = np.where(in_db, baseline, sat)
    # Outside the part the saturation is held at 1, the nominal value. The
    # design domain is the part; sub-pixel boundary cells (geometry fill
    # between 0 and 1) must keep their nominal fill, otherwise the arm
    # silently changes the geometry as well as the dopant. Measured: with 0
    # outside instead of 1, the uniform triangle reference moved from 23.06 C
    # to 39.37 C in the permittivity-co-varying channel.
    return np.where(part_mask, sat, float(outside))


def quantize(sat: np.ndarray, bpp: int = 4) -> np.ndarray:
    max_val = float((1 << int(bpp)) - 1)
    return np.clip(np.round(np.asarray(sat) * max_val), 0.0, max_val) / max_val


def golden_section(f: Callable[[float], float], a: float, b: float,
                   n_evals: int) -> tuple[list[float], list[float]]:
    """Golden-section minimisation with a fixed evaluation budget.

    Returns the evaluation points and values in the order they were taken, so
    a caller can read off the running selection at any smaller budget.
    """
    xs: list[float] = []
    ys: list[float] = []

    def ev(x: float) -> float:
        xs.append(float(x))
        y = float(f(x))
        ys.append(y)
        return y

    lo, hi = float(a), float(b)
    if n_evals <= 0:
        return xs, ys
    c = hi - GOLDEN * (hi - lo)
    d = lo + GOLDEN * (hi - lo)
    fc = ev(c)
    if len(xs) >= n_evals:
        return xs, ys
    fd = ev(d)
    while len(xs) < n_evals:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - GOLDEN * (hi - lo)
            fc = ev(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + GOLDEN * (hi - lo)
            fd = ev(d)
    return xs, ys


def select_on_fit(rows: list[dict]) -> dict | None:
    """Pre-registered selection: argmin of the FIT metric among FEASIBLE rows.

    Selecting on the hold-out would make the reported number in-sample. The
    feasibility filter is what stops the L_shape pathology (a map that never
    melts scoring a spectacular fake uniformity).
    """
    feasible = [r for r in rows if r.get("feasible", False)]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["fit"])


def infeasible_rank(met: dict, offset: float = 1.0e4, k_phi: float = 1.0e4,
                    t_pc_c: float = 180.0) -> float:
    """Ordering for gains that never reach the melt-onset read state.

    An infeasible gain must never win, but it must still be ORDERED, otherwise
    the golden-section bracketing compares equal penalties and walks away from
    the feasible region. Measured failure: on the cross shape every gain gave
    mean part melt fraction exactly 0, the melt-shortfall term tied, and the
    search ran to the top of the domain instead of the bottom. The secondary
    term (how close the part got to the phase-change temperature) breaks that
    tie in the physically correct direction.
    """
    shortfall = max(0.0, 0.90 - float(met.get("max_phi_bar", met.get("final_phi_bar", 0.0))))
    heat_gap = max(0.0, float(t_pc_c) - float(met.get("max_mean_T_part_c", 0.0)))
    return offset + k_phi * shortfall + heat_gap


def select_on_holdout(rows: list[dict]) -> dict | None:
    """In-sample selection, used only for the arms that are LABELLED in-sample.

    Kept as a separate named function so no arm can drift into selecting on its
    own reported metric by accident.
    """
    feasible = [r for r in rows if r.get("feasible", False)]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["holdout"])


def forward_equivalents(n_forward: int, n_gradient: int, ratio: float) -> float:
    """Cost in forward-solve equivalents. `ratio` is the MEASURED adjoint cost."""
    return float(n_forward) + float(n_gradient) * float(ratio)


def max_gradient_evals(budget: float, ratio: float) -> int:
    """How many objective-plus-gradient evaluations fit in `budget`."""
    per = 1.0 + float(ratio)
    return int(math.floor(float(budget) / per))
