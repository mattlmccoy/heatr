"""Pure logic for the gain-calibrated one-shot FGM control.

Two independent pieces, both free of any solver dependency so they can be unit tested:

1. `calibrate_gain` -- a budgeted one-dimensional minimization of a fit metric over the
   scalar FGM (functionally graded material) gain `magnitude`. It warm-starts from an
   already-evaluated grid, expands the bracket geometrically when the grid argmin sits on
   an endpoint (the usual case: the stored 4-point grid stops at m = 0.85 while the fit
   metric is still falling), then refines with parabolic interpolation inside a bracketing
   triple, falling back to golden section when the parabola is unusable.

2. `select_on_holdout` -- the selection rule. The gain is chosen by the FIT metric only.
   The hold-out metric is reported, never used to choose. Feasibility reads `melt_reached`
   (the EXISTENCE of the hold-out read state), never the hold-out value.

Objective callables may return `None` to mean "this solve produced no usable fit score";
such points are recorded but never selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence

GOLDEN = 1.6180339887498949
_DUP_TOL = 1e-3          # two gains closer than this are the same design point
_PARABOLA_EPS = 1e-12


@dataclass(frozen=True)
class Bracket:
    """A candidate bracket for the minimum, derived from evaluated points."""

    lo: float
    hi: float
    needs_expansion: bool
    direction: int          # +1 expand above `hi`, -1 expand below `lo`, 0 bracketed


@dataclass(frozen=True)
class GainCandidate:
    """One evaluated gain: its fit score, its hold-out score, and feasibility."""

    gain: float
    fit_score: Optional[float]
    holdout_score: Optional[float]
    melt_reached: bool


@dataclass(frozen=True)
class SearchResult:
    evaluated: Dict[float, Optional[float]]
    best_gain: Optional[float]
    best_fit: Optional[float]
    n_new_evals: int
    n_total_evals: int
    new_gains: tuple[float, ...] = ()
    stop_reason: str = ""


@dataclass(frozen=True)
class Selection:
    selected: Optional[GainCandidate]
    unconstrained: Optional[GainCandidate]
    reported_holdout: Optional[float]
    n_infeasible: int
    status: str
    infeasible_gains: tuple[float, ...] = field(default=())


# ---------------------------------------------------------------------------
# bracketing
# ---------------------------------------------------------------------------

def _scored(grid: Mapping[float, Optional[float]]) -> list[tuple[float, float]]:
    pts = [(float(g), float(v)) for g, v in grid.items() if v is not None]
    pts.sort(key=lambda p: p[0])
    return pts


def bracket_from_grid(grid: Mapping[float, Optional[float]],
                      domain: tuple[float, float]) -> Bracket:
    """Bracket the minimum of an evaluated grid.

    Interior argmin -> the two neighbours bracket it. Endpoint argmin -> not bracketed;
    `needs_expansion` is True and `direction` says which way to step out.
    """
    pts = _scored(grid)
    if len(pts) < 3:
        raise ValueError(
            f"bracket_from_grid needs at least 3 scored points, got {len(pts)}"
        )
    lo_d, hi_d = domain
    i_best = min(range(len(pts)), key=lambda i: pts[i][1])

    if i_best == 0:
        return Bracket(lo=pts[0][0], hi=pts[1][0],
                       needs_expansion=pts[0][0] > lo_d, direction=-1)
    if i_best == len(pts) - 1:
        return Bracket(lo=pts[-2][0], hi=pts[-1][0],
                       needs_expansion=pts[-1][0] < hi_d, direction=+1)
    return Bracket(lo=pts[i_best - 1][0], hi=pts[i_best + 1][0],
                   needs_expansion=False, direction=0)


def next_expansion_point(x_prev: float, x_end: float,
                         domain: tuple[float, float]) -> Optional[float]:
    """One geometric step out past `x_end`, away from `x_prev`, clipped to `domain`.

    Returns None when `x_end` is already at the domain edge in that direction.
    """
    lo_d, hi_d = domain
    step = GOLDEN * (x_end - x_prev)
    x_new = x_end + step
    if step > 0.0:
        if x_end >= hi_d - _DUP_TOL:
            return None
        return min(x_new, hi_d)
    if x_end <= lo_d + _DUP_TOL:
        return None
    return max(x_new, lo_d)


# ---------------------------------------------------------------------------
# refinement probe
# ---------------------------------------------------------------------------

def _parabolic_vertex(p0: tuple[float, float], p1: tuple[float, float],
                      p2: tuple[float, float]) -> Optional[float]:
    """Vertex of the parabola through three points, or None if degenerate."""
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    d1, d2 = x1 - x0, x1 - x2
    num = d1 * d1 * (y1 - y2) - d2 * d2 * (y1 - y0)
    den = 2.0 * (d1 * (y1 - y2) - d2 * (y1 - y0))
    if abs(den) < _PARABOLA_EPS:
        return None
    return x1 - num / den


def _refine_probe(grid: Mapping[float, Optional[float]], br: Bracket) -> Optional[float]:
    """Next probe inside a bracketing triple: parabolic if usable, else golden section."""
    pts = _scored(grid)
    inside = [p for p in pts if br.lo <= p[0] <= br.hi]
    if len(inside) < 2:
        # Two points are enough for a golden step; fewer is not.
        return None
    i_best = min(range(len(inside)), key=lambda i: inside[i][1])
    x_b = inside[i_best][0]

    if 0 < i_best < len(inside) - 1:
        v = _parabolic_vertex(inside[i_best - 1], inside[i_best], inside[i_best + 1])
        if v is not None and br.lo + _DUP_TOL < v < br.hi - _DUP_TOL:
            if all(abs(v - x) > _DUP_TOL for x, _ in pts):
                return v

    # golden section into the larger sub-interval
    left, right = x_b - br.lo, br.hi - x_b
    probe = (x_b - left / GOLDEN) if left > right else (x_b + right / GOLDEN)
    if all(abs(probe - x) > _DUP_TOL for x, _ in pts):
        return probe
    return None


# ---------------------------------------------------------------------------
# the budgeted search
# ---------------------------------------------------------------------------

def calibrate_gain(f: Callable[[float], Optional[float]],
                   warm_start: Mapping[float, Optional[float]],
                   domain: tuple[float, float],
                   max_new_evals: int) -> SearchResult:
    """Minimize `f` over the scalar gain, spending at most `max_new_evals` new calls."""
    grid: Dict[float, Optional[float]] = {float(g): v for g, v in warm_start.items()}
    n_warm = len(grid)
    new_gains: list[float] = []
    stop = "budget_exhausted"

    while len(new_gains) < max_new_evals:
        try:
            br = bracket_from_grid(grid, domain)
        except ValueError:
            stop = "too_few_scored_points"
            break

        if br.needs_expansion:
            if br.direction > 0:
                probe = next_expansion_point(br.lo, br.hi, domain)
            else:
                probe = next_expansion_point(br.hi, br.lo, domain)
            if probe is None:
                stop = "domain_edge_reached"
                break
        else:
            probe = _refine_probe(grid, br)
            if probe is None:
                stop = "converged_no_new_probe"
                break

        if any(abs(probe - x) <= _DUP_TOL for x in grid):
            stop = "probe_duplicates_existing_point"
            break

        grid[probe] = f(probe)
        new_gains.append(probe)

    scored = _scored(grid)
    if scored:
        best_gain, best_fit = min(scored, key=lambda p: p[1])
    else:
        best_gain, best_fit = None, None

    return SearchResult(
        evaluated=grid,
        best_gain=best_gain,
        best_fit=best_fit,
        n_new_evals=len(new_gains),
        n_total_evals=n_warm + len(new_gains),
        new_gains=tuple(new_gains),
        stop_reason=stop,
    )


# ---------------------------------------------------------------------------
# unseeded, multimodality-robust search (v2)
# ---------------------------------------------------------------------------

def coarse_scan_gains(domain: tuple[float, float], n_points: int) -> list[float]:
    """Log-spaced scan points spanning the whole domain, endpoints included.

    Log rather than linear spacing because the useful gains span more than two
    decades (shapes optimize anywhere from m ~ 0.3 to m > 2.4) and a linear grid
    wastes most of its points above m = 3 where the map is already two-level.
    """
    lo, hi = domain
    if lo <= 0.0:
        raise ValueError(f"coarse_scan_gains needs a positive lower bound, got {lo}")
    if hi <= lo:
        raise ValueError(f"empty domain ({lo}, {hi})")
    if n_points < 3:
        raise ValueError(f"n_points must be at least 3, got {n_points}")
    ratio = (hi / lo) ** (1.0 / (n_points - 1))
    return [lo * ratio ** i for i in range(n_points)]


def calibrate_gain_unseeded(f: Callable[[float], Optional[float]],
                            domain: tuple[float, float],
                            n_coarse: int,
                            max_refine: int) -> SearchResult:
    """Coarse scan across the FULL domain, then refine around the best bracket.

    Unlike `calibrate_gain`, this takes no warm start, so it cannot be trapped on
    one side of a local maximum. The triangle fit metric is measured to be bimodal
    (local maximum m = 0.638, global minimum m = 2.402), which is exactly the case
    a warm-started bracket search gets wrong.
    """
    scan = coarse_scan_gains(domain, n_coarse)
    warm: Dict[float, Optional[float]] = {g: f(g) for g in scan}
    refined = calibrate_gain(f, warm_start=warm, domain=domain,
                             max_new_evals=max_refine)
    n_new = len(scan) + refined.n_new_evals
    return SearchResult(
        evaluated=refined.evaluated,
        best_gain=refined.best_gain,
        best_fit=refined.best_fit,
        n_new_evals=n_new,
        n_total_evals=n_new,
        new_gains=tuple(scan) + refined.new_gains,
        stop_reason=refined.stop_reason,
    )


# ---------------------------------------------------------------------------
# hold-out selection
# ---------------------------------------------------------------------------

def select_on_holdout(candidates: Sequence[GainCandidate]) -> Selection:
    """Choose the gain on the fit metric; report the hold-out metric of that gain.

    Feasibility uses `melt_reached` only, so the hold-out VALUE never enters selection.
    A candidate with no fit score cannot be selected.
    """
    if not candidates:
        raise ValueError("select_on_holdout requires at least one candidate")

    with_fit = [c for c in candidates if c.fit_score is not None]
    feasible = [c for c in with_fit if c.melt_reached and c.holdout_score is not None]
    infeasible = [c for c in with_fit if c not in feasible]

    unconstrained = min(with_fit, key=lambda c: c.fit_score) if with_fit else None
    selected = min(feasible, key=lambda c: c.fit_score) if feasible else None

    return Selection(
        selected=selected,
        unconstrained=unconstrained,
        reported_holdout=selected.holdout_score if selected else None,
        n_infeasible=len(infeasible),
        status="OK" if selected else "NOT_REACHED",
        infeasible_gains=tuple(c.gain for c in infeasible),
    )
