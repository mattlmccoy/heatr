"""Symmetry analysis of an imported geometry, from its target indicator alone.

WHAT IT DECIDES, AND WHY THE SOLVE STACK NEEDS IT.

  * The GAUGE. `DWELL_SCHEDULE_REPORT.md` Section 2 measured that the part-frame
    heating at theta and at theta + 180 degrees is the same field to 4e-13
    relative, because the grounded parallel-plate drive is invariant under a
    half turn of the whole system. Orientation is therefore only ever defined
    modulo 180 degrees, for EVERY geometry. `gauge_reduce` applies that rule
    universally instead of per shape, and it is what halves an emitted
    turntable program (150 moves to 75 on the cross).
  * The INDEXING ORDER. `CONTINUOUS_ROTATION_REPORT.md` Section 7 measured that
    symmetry-matched indexing beats the finest available rotation, by factors
    of 2.8 (cross) and 2.1 (square) on J. Choosing the indexing needs the
    part's rotational order, which until now was known by hand.
  * The CANDIDATE SET. The dwell campaign used eight positions at 45 degrees
    for every shape and named that as a limit (`DWELL_SCHEDULE_REPORT.md`
    limit 4). With the order known, the candidate set can be reduced to the
    orientations that are genuinely distinct: under the part's own rotational
    symmetry AND under the half-turn gauge.

THE MEASUREMENT. Rotational symmetry is read off the ANGULAR AUTOCORRELATION of
chi about its own area centroid: for a trial angle alpha,

    mismatch(alpha) = sum |chi - R_alpha chi| / (2 sum chi)

which is 0 for a perfect symmetry and 1 for a rotation that moves the part
entirely off itself. The rotation is a bilinear resampling about the centroid,
so a rotation that is not a multiple of 90 degrees carries a smoothing floor.
That floor was MEASURED on the library rather than assumed: exact symmetries at
non-multiples of 90 degrees score 0.0082 (circle) to 0.0241 (five-pointed star),
and the smallest NON-symmetry scores 0.0351 (octagon at 120 degrees). The
acceptance threshold `SYMMETRY_TOL = 0.030` sits in that gap. It is a measured
threshold with a factor of 1.2 of margin on the tight side (the star), and that
margin is the honest limit of this detector: a part whose symmetry is broken by
less than roughly 3 percent of its area will be reported as symmetric.

MIRRORS are read the same way, reflecting about a line through the centroid at
angle beta and scanning beta over [0, 180).

THE WHOLE-GROUP ACCEPTANCE RULE, and the octagon regression it fixes. An order
N is accepted only when EVERY power of its generator is a symmetry, that is when

    max over k = 1 .. N-1 of mismatch(k * 360 / N) < tol,

and not when the generator 360 / N alone passes. MEASURED root cause: a
near-circular part scores a small defect at almost every trial angle, so a
generator that happens to fall near a multiple of the part's TRUE period passes
by accident. On the octagon (true period 45 degrees) the generators of orders 7,
9 and 10 are 51.43, 40 and 36 degrees, which sit 6.4, 5 and 9 degrees off a true
symmetry and score 0.0201, 0.0169 and 0.0261, all under the 0.030 threshold,
while the true non-symmetries those groups also require (102.9, 120 and 72
degrees) score 0.033 to 0.038, above it. The single-generator rule therefore
reported the octagon as 10-fold; the whole-group rule reports 8. It is not a
sampling-density problem and not a raster-staircase harmonic: the defect curve
is correct, the acceptance predicate was reading one point of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import affine_transform

__all__ = ["SymmetryReport", "analyze", "rotation_mismatch", "mirror_mismatch",
           "group_mismatch",
           "gauge_reduce", "candidate_span_deg", "candidate_angles",
           "indexing_orders", "SYMMETRY_TOL"]

# Measured on the eighteen-shape library; see the module docstring.
SYMMETRY_TOL = 0.030
MAX_ORDER = 12
GENERIC_ANGLES_DEG = (17.0, 23.0, 41.0)     # non-divisors, for the circle test


@dataclass(frozen=True)
class SymmetryReport:
    rotational_order: int
    continuous_rotational: bool
    mirror_axes_deg: tuple[float, ...]
    centroid_rc: tuple[float, float]
    diagnostics: dict = field(default_factory=dict)

    @property
    def point_group(self) -> str:
        n = int(self.rotational_order)
        if self.continuous_rotational:
            return "O(2)" if self.mirror_axes_deg else "SO(2)"
        return f"C{n}v" if self.mirror_axes_deg else f"C{n}"

    def as_json(self) -> dict:
        return {"rotational_order": int(self.rotational_order),
                "continuous_rotational": bool(self.continuous_rotational),
                "mirror_axes_deg": [float(a) for a in self.mirror_axes_deg],
                "point_group": self.point_group,
                "centroid_row_col": [float(self.centroid_rc[0]),
                                     float(self.centroid_rc[1])],
                "diagnostics": self.diagnostics}


# ---------------------------------------------------------------------------
# the two mismatch measures
# ---------------------------------------------------------------------------

def _centroid_rc(chi: np.ndarray) -> tuple[float, float]:
    c = np.asarray(chi, dtype=float)
    tot = float(c.sum())
    if tot <= 0.0:
        raise ValueError("chi is empty; there is no geometry to analyse")
    ny, nx = c.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    return (float((c * jj).sum() / tot), float((c * ii).sum() / tot))


def _apply_linear(chi: np.ndarray, M: np.ndarray,
                  centre: tuple[float, float]) -> np.ndarray:
    c = np.asarray(chi, dtype=float)
    ctr = np.asarray(centre, dtype=float)
    off = ctr - M @ ctr
    return affine_transform(c, M, offset=off, order=1, mode="constant", cval=0.0)


def _rot_matrix(deg: float) -> np.ndarray:
    """Row-column rotation matrix for `scipy.ndimage.affine_transform`.

    The matrix maps OUTPUT coordinates to INPUT coordinates, so it is the
    inverse rotation; the sign convention is irrelevant to a symmetry test but
    is fixed here so the reported mirror axis angles are in the usual
    mathematical frame (measured from the +x axis, counter-clockwise).
    """
    a = math.radians(float(deg))
    ca, sa = math.cos(a), math.sin(a)
    q = float(deg) / 90.0
    if abs(q - round(q)) < 1e-12:
        ca, sa = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][int(round(q)) % 4]
    return np.array([[ca, sa], [-sa, ca]], dtype=float)


def rotation_mismatch(chi: np.ndarray, deg: float,
                      centre: tuple[float, float] | None = None) -> float:
    """Normalized angular autocorrelation defect at one trial angle."""
    c = np.asarray(chi, dtype=float)
    if abs(float(deg) % 360.0) < 1e-12:
        return 0.0
    ctr = _centroid_rc(c) if centre is None else centre
    r = _apply_linear(c, _rot_matrix(deg), ctr)
    return float(np.sum(np.abs(c - r))) / (2.0 * float(c.sum()))


def mirror_mismatch(chi: np.ndarray, axis_deg: float,
                    centre: tuple[float, float] | None = None) -> float:
    """Defect under reflection in the line through the centroid at `axis_deg`."""
    c = np.asarray(chi, dtype=float)
    ctr = _centroid_rc(c) if centre is None else centre
    a = math.radians(float(axis_deg))
    ca, sa = math.cos(2 * a), math.sin(2 * a)
    # Reflection in the row-column frame: rows are y, columns are x, so the
    # x-axis reflection is the row flip.
    M = np.array([[-ca, sa], [sa, ca]], dtype=float)
    r = _apply_linear(c, M, ctr)
    return float(np.sum(np.abs(c - r))) / (2.0 * float(c.sum()))


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _cluster_minima(vals: np.ndarray, angles: np.ndarray,
                    tol: float) -> tuple[float, ...]:
    """One representative angle per contiguous run of accepted trial angles."""
    ok = vals < tol
    if not ok.any():
        return ()
    n = len(ok)
    start = 0
    if ok.all():
        return (float(angles[int(np.argmin(vals))]),)
    while ok[start]:                 # rotate so index 0 is not accepted
        start += 1
    idx = [(start + k) % n for k in range(n)]
    out: list[float] = []
    run: list[int] = []
    for k in idx:
        if ok[k]:
            run.append(k)
        elif run:
            best = min(run, key=lambda j: vals[j])
            out.append(float(angles[best]))
            run = []
    if run:
        out.append(float(angles[min(run, key=lambda j: vals[j])]))
    return tuple(sorted(out))


def group_mismatch(chi: np.ndarray, order: int,
                   centre: tuple[float, float] | None = None,
                   cache: dict | None = None) -> float:
    """The WORST defect over the whole cyclic group C_order, not just its generator.

    `max over k = 1 .. N-1 of mismatch(k * 360 / N)`. This is the quantity an
    order has to clear: a cyclic group is a symmetry of the part only when every
    one of its elements is, and testing only the generator is what reported the
    octagon as 10-fold (see the module docstring).
    """
    c = np.asarray(chi, dtype=float)
    n = int(order)
    if n <= 1:
        return 0.0
    ctr = _centroid_rc(c) if centre is None else centre
    worst = 0.0
    for k in range(1, n):
        ang = round((k * 360.0 / n) % 360.0, 9)
        if cache is not None and ang in cache:
            m = cache[ang]
        else:
            m = rotation_mismatch(c, ang, ctr)
            if cache is not None:
                cache[ang] = m
        worst = max(worst, m)
    return float(worst)


def analyze(chi: np.ndarray, tol: float = SYMMETRY_TOL,
            max_order: int = MAX_ORDER,
            mirror_step_deg: float = 1.0) -> SymmetryReport:
    """Rotational order, continuity flag and mirror axes of one indicator field."""
    c = np.asarray(chi, dtype=float)
    ctr = _centroid_rc(c)
    gen = {N: rotation_mismatch(c, 360.0 / N, ctr)
           for N in range(2, int(max_order) + 1)}
    ang_cache: dict = {}
    orders = {N: group_mismatch(c, N, ctr, ang_cache)
              for N in range(2, int(max_order) + 1)}
    accepted = [N for N, m in orders.items() if m < tol]
    order = max(accepted) if accepted else 1
    generic = {a: rotation_mismatch(c, a, ctr) for a in GENERIC_ANGLES_DEG}
    continuous = all(v < tol for v in generic.values())

    betas = np.arange(0.0, 180.0, float(mirror_step_deg))
    mvals = np.array([mirror_mismatch(c, b, ctr) for b in betas])
    axes = _cluster_minima(mvals, betas, tol)
    if continuous and axes:
        axes = (float(betas[int(np.argmin(mvals))]),)

    return SymmetryReport(
        rotational_order=int(order),
        continuous_rotational=bool(continuous),
        mirror_axes_deg=axes,
        centroid_rc=ctr,
        diagnostics={
            "tested_orders": {str(k): float(v) for k, v in orders.items()},
            "tested_orders_generator_only": {str(k): float(v)
                                             for k, v in gen.items()},
            "acceptance_rule": "whole cyclic group: max over k = 1 .. N-1 of "
                               "mismatch(k * 360 / N) < tol",
            "generic_angle_mismatch": {str(k): float(v) for k, v in generic.items()},
            "mirror_min_mismatch": float(mvals.min()),
            "tol": float(tol),
        })


# ---------------------------------------------------------------------------
# the gauge and the candidate set
# ---------------------------------------------------------------------------

def gauge_reduce(angles_deg) -> np.ndarray:
    """Fold orientations by the measured half-turn gauge, then sort and dedup.

    `DWELL_SCHEDULE_REPORT.md` Section 2: theta and theta + 180 degrees are the
    SAME part-frame heating to 4e-13 relative, so the eight-position candidate
    set carries only four distinct patterns. Applying this here means no
    downstream layer has to remember it.
    """
    a = np.asarray(angles_deg, dtype=float).ravel() % 180.0
    a[np.abs(a - 180.0) < 1e-9] = 0.0
    out = np.unique(np.round(a, 9))
    return out


def candidate_span_deg(order: int) -> float:
    """The interval of genuinely distinct orientations, in degrees.

    Distinct under the part's own C_n rotations AND under the half-turn gauge.
    For EVEN n the half turn is already in C_n, so the span is 360 / n. For ODD
    n it is not, and the group generated by both is C_2n, so the span halves to
    180 / n.
    """
    n = max(int(order), 1)
    return 360.0 / n if n % 2 == 0 else 180.0 / n


def candidate_angles(order: int | None = None, step_deg: float = 15.0,
                     report: SymmetryReport | None = None) -> np.ndarray:
    """The symmetry-reduced, gauge-reduced candidate orientations."""
    if report is not None:
        if report.continuous_rotational:
            return np.array([0.0])
        order = report.rotational_order
    if order is None:
        raise ValueError("give an order or a report")
    span = candidate_span_deg(int(order))
    step = float(step_deg)
    if step <= 0.0:
        raise ValueError(f"step_deg must be positive, got {step_deg!r}")
    n = max(int(math.floor(span / step - 1e-9)) + 1, 1)
    return np.arange(n, dtype=float) * step


def indexing_orders(order: int) -> tuple[int, ...]:
    """The indexing multiplicities compatible with the part's own symmetry.

    An N-fold indexing schedule visits the orientations 360 k / N. It leaves the
    part mask invariant exactly when N divides the part's rotational order, so
    those are the schedules for which the mode-averaged kernel carries the
    part's own symmetry. Order 1 admits none, which is the honest statement that
    an asymmetric part has no matched indexing schedule.
    """
    n = int(order)
    if n <= 1:
        return ()
    return tuple(d for d in range(2, n + 1) if n % d == 0)
