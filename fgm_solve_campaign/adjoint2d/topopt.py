"""Topology-optimization parameterization of the dopant design variable.

THE PROBLEM THIS FIXES. `SOLVE_ROBUSTNESS_VALIDATION.md` Section 4 measured
that the unfiltered solved dopant maps are grid sculptures: a one-cell blur
costs +37.3 to +891.9 percent of the melt-region objective on five of six
shapes. `MULTISTART_REPORT.md` Section 6 added the physical-length design
filter alone and got a partial fix: the square's one-cell blur cost fell from
+152.7 percent to +9.9 percent, but its grid hold-out fidelity only moved from
intersection over union 0.7767 to 0.8063 against 0.9681 in grid. Filtering
removes sub-radius structure from the DESIGN VARIABLE but leaves the map
continuous-valued, so the solve can still buy fidelity with a fine-grained
grey-scale pattern that the resampler does not preserve.

THE PARAMETERIZATION. The standard topology-optimization chain, three stages:

    v  --F-->  v_f  --P_beta-->  s

  F        a smoothing filter of PHYSICAL radius R, the normalized convolution
           over the part of `design_filter.apply_filter`. Its transpose is
           exact (`tests/test_design_filter.py`) and it preserves the box
           without a clip, so no clip subgradient enters the chain.
  P_beta   the smoothed-Heaviside projection, the tanh form of Wang, Lazarov
           and Sigmund, threshold eta, sharpness beta:

               P(u) = [tanh(beta*eta) + tanh(beta*(u - eta))]
                      / [tanh(beta*eta) + tanh(beta*(1 - eta))]

           At eta = 0.5 it fixes 0, 0.5 and 1 exactly, is monotone, and maps
           [0, 1] onto [0, 1], so the box on v is again the box on s with no
           clip anywhere in the chain. As beta grows it drives interior values
           toward 0 or 1, which is what makes the surviving structure a
           resolved feature of size R rather than a grey-scale texture.

THE RADIUS, AND WHY IT IS 1.0 MILLIMETRE. Frozen at `FILTER_RADIUS_M`, a
LENGTH, converted to a cell count per grid by `sigma_cells_for`. Two candidate
scales were considered and only one binds.

  * The printer's dopant edge scale is roughly 50 to 100 micrometres. At grid
    120 on the 60 mm domain the cell size is 504 micrometres, five to ten times
    coarser, and at grid 160 it is 377 micrometres. The printer scale is finer
    than any cell of any grid the solve runs on, so it can never be the
    binding constraint on what the solve is allowed to express. It is NOT the
    radius.
  * The binding scale is solver convergence. The cell-count evidence in
    `MULTISTART_REPORT.md` is that sigma = 1.5 cells (0.75 mm at grid 120) is
    not enough: the filtered square still lost 0.16 intersection-over-union
    points across the grid hold-out. The next resolvable step up is two cells,
    1.0 mm at grid 120, which is the smallest length that the 160 grid also
    resolves with more than two cells (2.65) so that the same physical feature
    is representable on both grids with room to spare.

  1.0 mm is therefore a solver-convergence radius, not a process radius, and
  that is stated wherever it is quoted. It was NOT swept; see the report's
  assumptions.

THE CONTINUATION SCHEDULE. `BETA_SCHEDULE` = (1, 2, 4, 8, 16), each stage
restarting L-BFGS-B from the previous stage's iterate. Doubling is used rather
than a larger jump because the optimizer's curvature memory is discarded at
each restart, so a stage must be able to re-converge inside its own share of
the budget; a 4x jump moves the objective further than four evaluations can
recover on this budget. beta = 1 is nearly the unprojected filtered problem, so
stage 1 reproduces the previously gated arm and the continuation is an addition
to it rather than a replacement. beta = 16 is the stopping point because at
eta = 0.5 it already maps 0.65 to above 0.95 and 0.35 to below 0.05, so the
map is effectively binary at the design level, and pushing further only
steepens the gradient without changing the geometry. The schedule is FROZEN,
not swept.
"""
from __future__ import annotations

import numpy as np

from . import design_filter as df

# The frozen conventions. Cited in `FROZEN_CONVENTIONS_2D.md`.
FILTER_RADIUS_M = 1.0e-3
BETA_SCHEDULE: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
ETA = 0.5
BOX = (0.0, 1.0)


# ---------------------------------------------------------------------------
# the radius, as a length
# ---------------------------------------------------------------------------

def sigma_cells_for(radius_m: float, dx: float) -> float:
    """Cell count of a PHYSICAL filter radius on a grid of cell size `dx`.

    This is the whole of the resample consistency claim: the solve, the gate
    and every re-score convert the same length through their own cell size, so
    two grids filter the same physical feature.
    """
    r = float(radius_m)
    if r < 0.0:
        raise ValueError(f"radius_m must be non-negative, got {radius_m!r}")
    if float(dx) <= 0.0:
        raise ValueError(f"dx must be positive, got {dx!r}")
    return r / float(dx)


# ---------------------------------------------------------------------------
# the projection
# ---------------------------------------------------------------------------

def _check_eta(eta: float) -> float:
    e = float(eta)
    if not (0.0 < e < 1.0):
        raise ValueError(f"eta must lie strictly inside (0, 1), got {eta!r}")
    return e


def project(u, beta: float, eta: float = ETA):
    """Smoothed-Heaviside projection. `beta` <= 0 is the exact identity."""
    e = _check_eta(eta)
    a = np.asarray(u, dtype=float)
    b = float(beta)
    if b <= 0.0:
        return a
    den = np.tanh(b * e) + np.tanh(b * (1.0 - e))
    return (np.tanh(b * e) + np.tanh(b * (a - e))) / den


def project_deriv(u, beta: float, eta: float = ETA):
    """dP/du. `beta` <= 0 gives ones, matching the identity branch."""
    e = _check_eta(eta)
    a = np.asarray(u, dtype=float)
    b = float(beta)
    if b <= 0.0:
        return np.ones_like(a)
    den = np.tanh(b * e) + np.tanh(b * (1.0 - e))
    return b * (1.0 - np.tanh(b * (a - e)) ** 2) / den


# ---------------------------------------------------------------------------
# the composed map and its derivatives
# ---------------------------------------------------------------------------

def filtered(v: np.ndarray, part_mask: np.ndarray, dx: float, radius_m: float,
             outside: float = 1.0) -> np.ndarray:
    """v_f = F(v), the intermediate the projection reads."""
    return df.apply_filter(v, part_mask, sigma_cells_for(radius_m, dx), outside=outside)


def design_to_map(v: np.ndarray, part_mask: np.ndarray, dx: float,
                  radius_m: float, beta: float, eta: float = ETA,
                  outside: float = 1.0) -> np.ndarray:
    """s = P_beta(F(v)) inside the part, held at `outside` outside it.

    At `beta` <= 0 this is BIT-IDENTICAL to `design_filter.apply_filter`, which
    is the flag-off identity that keeps the new channel from perturbing the
    previously gated filtered path.
    """
    vf = filtered(v, part_mask, dx, radius_m, outside=outside)
    if float(beta) <= 0.0:
        return vf
    pm = np.asarray(part_mask, dtype=bool)
    return np.where(pm, project(vf, beta, eta), float(outside))


def design_jvp(d: np.ndarray, v: np.ndarray, part_mask: np.ndarray, dx: float,
               radius_m: float, beta: float, eta: float = ETA) -> np.ndarray:
    """dS(v)[d], the directional derivative of the composed map.

    The nominal outside value carries no design sensitivity, so the
    linearization of the filter is taken with `outside = 0`.
    """
    pm = np.asarray(part_mask, dtype=bool)
    sig = sigma_cells_for(radius_m, dx)
    fd_ = df.apply_filter(d, pm, sig, outside=0.0)
    if float(beta) <= 0.0:
        return np.where(pm, fd_, 0.0)
    vf = filtered(v, pm, dx, radius_m)
    return np.where(pm, project_deriv(vf, beta, eta) * fd_, 0.0)


def design_vjp(g_s: np.ndarray, v: np.ndarray, part_mask: np.ndarray, dx: float,
               radius_m: float, beta: float, eta: float = ETA) -> np.ndarray:
    """dJ/dv given dJ/ds. Supported on the part, zero outside.

    Exactly F^T applied to the projection derivative times the incoming
    sensitivity, which `test_design_vjp_is_the_exact_transpose_of_the_
    linearized_map` checks against `design_jvp` by the dot-product identity.
    """
    pm = np.asarray(part_mask, dtype=bool)
    g = np.asarray(g_s, dtype=float)
    if float(beta) <= 0.0:
        h = np.where(pm, g, 0.0)
    else:
        vf = filtered(v, pm, dx, radius_m)
        h = np.where(pm, g * project_deriv(vf, beta, eta), 0.0)
    return df.filter_vjp(h, pm, sigma_cells_for(radius_m, dx))


# ---------------------------------------------------------------------------
# the continuation budget split
# ---------------------------------------------------------------------------

def stage_split(pool: int, n_stages: int) -> tuple[int, ...]:
    """Deterministic split of a gradient-evaluation pool over the beta stages.

    The remainder goes to the EARLIEST stages. Justification, and it is a
    convention rather than a measurement: at low beta the landscape is
    smoothest and the map is furthest from its final geometry, so an
    evaluation there moves the design more than the same evaluation at beta 16,
    where the projection has already committed most cells to a rail. When the
    pool is smaller than the number of stages the LATE stages are dropped
    entirely rather than every stage being starved to zero, because a stage
    that gets no evaluation cannot re-converge after its beta jump and would
    hand the next stage a design that is worse than the one it inherited.
    """
    p = int(pool)
    k = int(n_stages)
    if p < 0:
        raise ValueError(f"pool must be non-negative, got {pool!r}")
    if k <= 0:
        raise ValueError(f"n_stages must be positive, got {n_stages!r}")
    base, rem = divmod(p, k)
    if base == 0:
        return tuple(1 if i < rem else 0 for i in range(k))
    return tuple(base + 1 if i < rem else base for i in range(k))
