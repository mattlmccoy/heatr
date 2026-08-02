"""The method of moving asymptotes (MMA, Svanberg 1987), box constraints only.

WHY THIS MODULE EXISTS. `TOPOPT_REPORT.md` Section 4.3 measured that with three
evaluations per continuation stage, L-BFGS-B (limited-memory
Broyden-Fletcher-Goldfarb-Shanno with box constraints) spends most of a stage
on line-search trial points that are worse than the incumbent, and its
curvature memory is discarded at every stage restart. That report's own
diagnosis is that the projection's measured in-grid cost is "an optimizer
artifact as much as a physics one", and `FROZEN_CONVENTIONS_2D.md` Section 9
item 4 names MMA as not implemented. This module implements it so the question
can be settled at a matched budget.

Two properties make MMA the field standard for topology optimization and both
are the reason it is the right instrument here:

  * ONE function-and-gradient evaluation per design update, no line search, so
    a budget of N evaluations buys N design updates rather than N line-search
    trial points;
  * the state that adapts the step is the pair of ASYMPTOTES, which are carried
    across iterations and therefore across a continuation stage boundary. The
    method does not have to relearn anything when beta jumps.

--------------------------------------------------------------------------
THE SIMPLIFICATIONS, STATED IN FULL, BECAUSE THIS IS NOT FULL MMA
--------------------------------------------------------------------------

Svanberg's MMA solves, at each iterate x^k,

    min_x  f0~(x) + a0 z + sum_i (c_i y_i + d_i y_i^2 / 2)
    s.t.   fi~(x) - a_i z - y_i <= 0,   alpha <= x <= beta,  y, z >= 0

where f0~ and fi~ are separable convex rational approximations. The design
problem here has NO constraint other than the box: no volume fraction, no
compliance limit, no stress limit. That removes m = 0 constraints, and with
them the artificial variables y and z and the whole dual iteration.

Consequences, each one a real simplification of the published method:

  1. **The dual is zero dimensional, not one dimensional.** With m = 0 there is
     no multiplier at all. The subproblem
     min sum_j [ p0_j/(U_j - x_j) + q0_j/(x_j - L_j) ] over the box is
     separable and each scalar term is strictly convex on (L_j, U_j), so the
     minimizer is available in closed form,
         x*_j = (sqrt(p0_j) L_j + sqrt(q0_j) U_j) / (sqrt(p0_j) + sqrt(q0_j)),
     clamped to [alpha_j, beta_j]. No Newton iteration, no line search, no
     dual gradient. This is the entire subproblem solver.
  2. **No y, z, a0, a_i, c_i, d_i.** They exist only to guarantee a feasible
     subproblem when constraints can conflict. With only a box, every
     subproblem is trivially feasible.
  3. **No globally convergent GCMMA outer loop.** The 1987 method is used, not
     the 2002 conservative variant, so there is no inner loop that inflates the
     approximation until it is conservative. The practical consequence is that
     MMA here is NOT a descent method: an iterate can be worse than its
     predecessor. The solve driver therefore keeps the best iterate, exactly as
     the L-BFGS-B driver already does. Pinned by
     `test_the_iterate_can_overshoot_so_the_caller_must_keep_the_best_iterate`.
  4. **The step tracks the asymptote distance, not the gradient magnitude.**
     This follows from 1 and is worth stating because it is the most
     surprising property of the box-only case. For any variable whose gradient
     is well above the raa0 floor, sqrt(p0_j)/sqrt(q0_j) is close to the fixed
     ratio sqrt(1.001/0.001) = 31.6 and the closed form lands about 94 percent
     of the way toward the asymptote on the descent side, whatever the
     gradient's size. MMA here behaves as a per-variable adaptive trust region
     driven by the SIGN of the gradient and by oscillation history. In full
     MMA the constraint multipliers reintroduce magnitude sensitivity; with no
     constraints they cannot. Pinned by
     `test_the_step_size_tracks_the_asymptote_distance_not_the_gradient_magnitude`.
  5. **The asymptote lower clamp floors the achievable accuracy** at roughly
     `asy_bound_lo` times the box span, again because the step is proportional
     to the asymptote distance. At Svanberg's default 0.01 that is one percent
     of the span, which is the same order as the change tolerance conventional
     topology-optimization codes stop on. Pinned by
     `test_the_default_asymptote_clamp_floors_the_accuracy_at_one_percent_of_the_span`.

--------------------------------------------------------------------------
THE METHOD AS IMPLEMENTED
--------------------------------------------------------------------------

Span `sp_j = xmax_j - xmin_j`. At iteration k with iterate x^k:

  asymptotes, k <= 2:   L = x^k - s0 sp,  U = x^k + s0 sp
  asymptotes, k >= 3:   gamma_j = 0.7 if (x^k_j - x^{k-1}_j)(x^{k-1}_j - x^{k-2}_j) < 0
                                  1.2 if > 0
                                  1.0 if = 0
                        L_j = x^k_j - gamma_j (x^{k-1}_j - L^{k-1}_j)
                        U_j = x^k_j + gamma_j (U^{k-1}_j - x^{k-1}_j)
  clamp each distance into [asy_bound_lo, asy_bound_hi] * sp

  move limits:  alpha_j = max(xmin_j, L_j + albefa (x^k_j - L_j), x^k_j - move sp_j)
                beta_j  = min(xmax_j, U_j - albefa (U_j - x^k_j), x^k_j + move sp_j)

  approximation, with g_j = df0/dx_j and r = raa0 / sp_j:
                p0_j = (U_j - x^k_j)^2 (1.001 max(g_j, 0) + 0.001 max(-g_j, 0) + r)
                q0_j = (x^k_j - L_j)^2 (0.001 max(g_j, 0) + 1.001 max(-g_j, 0) + r)

  closed form:  x^{k+1}_j = clip( (sqrt(p0) L + sqrt(q0) U)/(sqrt(p0)+sqrt(q0)),
                                  alpha_j, beta_j )

FROZEN HYPERPARAMETERS. Svanberg's published defaults are used unchanged for
asy_init, asy_decr, asy_incr, albefa, raa0 and the asymptote bounds. The only
value that is not Svanberg's is the move limit, set to 0.2, the standard
density move limit of the topology-optimization literature (Andreassen et al.,
the 88-line code). These were frozen BEFORE any solve was run and were NOT
tuned on any result; the choice is recorded in `MMA_RETEST_REPORT.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class MMAConfig:
    """Immutable MMA hyperparameters. See the module docstring for provenance."""

    asy_init: float = 0.5        # Svanberg: initial asymptote distance / span
    asy_decr: float = 0.7        # Svanberg: contraction on an oscillation
    asy_incr: float = 1.2        # Svanberg: expansion on monotone progress
    asy_bound_lo: float = 0.01   # Svanberg: min asymptote distance / span
    asy_bound_hi: float = 10.0   # Svanberg: max asymptote distance / span
    albefa: float = 0.1          # Svanberg: fraction of the asymptote gap kept free
    raa0: float = 1e-5           # Svanberg: the convexity floor
    move: float = 0.2            # topology-optimization standard move limit / span


@dataclass(frozen=True)
class SubProblem:
    """One MMA subproblem: the point it was built at, and everything it needs."""

    x: np.ndarray
    L: np.ndarray
    U: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    p0: np.ndarray
    q0: np.ndarray
    grad: np.ndarray


def _as_vector(value: ArrayLike, n: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or a length-{n} vector, "
                         f"got shape {a.shape}")
    return a


class MMA:
    """Box-constrained MMA driver. One `step` per function-and-gradient call.

    The object owns exactly the state the method needs across iterations: the
    current iterate, the two previous iterates, and the two asymptotes. That
    state is what carries across a continuation stage boundary, which is the
    property this module was written to test.
    """

    def __init__(self, x0: np.ndarray, lower: ArrayLike, upper: ArrayLike,
                 cfg: MMAConfig = MMAConfig()) -> None:
        x = np.asarray(x0, dtype=float).ravel().copy()
        if x.size == 0:
            raise ValueError("x0 must have at least one entry")
        if not np.all(np.isfinite(x)):
            raise ValueError("x0 must be finite")
        n = x.size
        lo = _as_vector(lower, n, "lower")
        hi = _as_vector(upper, n, "upper")
        if np.any(hi <= lo):
            raise ValueError("upper must exceed lower in every coordinate")
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12):
            raise ValueError("x0 must lie inside the box; it is not clipped "
                             "silently because a start outside the box is "
                             "almost always a caller error")
        if not (0.0 < float(cfg.move)):
            raise ValueError(f"move must be positive, got {cfg.move!r}")
        if not (0.0 < float(cfg.albefa) < 1.0):
            raise ValueError(f"albefa must lie in (0, 1), got {cfg.albefa!r}")
        if not (0.0 <= float(cfg.asy_bound_lo) < float(cfg.asy_bound_hi)):
            raise ValueError("asy_bound_lo must be non-negative and below "
                             "asy_bound_hi")

        self.cfg = cfg
        self.xmin = lo
        self.xmax = hi
        self.span = hi - lo
        self.x = np.clip(x, lo, hi)
        self.xold1: Optional[np.ndarray] = None
        self.xold2: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._U: Optional[np.ndarray] = None
        self.iteration = 0
        self.gamma_last = np.ones(n)
        self.last_sub: Optional[SubProblem] = None

    # -- asymptotes ---------------------------------------------------------

    def _asymptotes(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        if self.iteration < 2 or self.xold1 is None or self.xold2 is None:
            self.gamma_last = np.ones_like(self.x)
            low = self.x - c.asy_init * self.span
            upp = self.x + c.asy_init * self.span
        else:
            moved = (self.x - self.xold1) * (self.xold1 - self.xold2)
            gamma = np.ones_like(self.x)
            gamma[moved < 0.0] = c.asy_decr
            gamma[moved > 0.0] = c.asy_incr
            self.gamma_last = gamma
            low = self.x - gamma * (self.xold1 - self._L)
            upp = self.x + gamma * (self._U - self.xold1)
        d_lo = np.clip(self.x - low, c.asy_bound_lo * self.span,
                       c.asy_bound_hi * self.span)
        d_hi = np.clip(upp - self.x, c.asy_bound_lo * self.span,
                       c.asy_bound_hi * self.span)
        return self.x - d_lo, self.x + d_hi

    # -- the subproblem -----------------------------------------------------

    def build_subproblem(self, grad: np.ndarray) -> SubProblem:
        """Move the asymptotes and build the separable convex approximation."""
        g = np.asarray(grad, dtype=float).ravel()
        if g.shape != self.x.shape:
            raise ValueError(f"gradient must have shape {self.x.shape}, "
                             f"got {g.shape}")
        if not np.all(np.isfinite(g)):
            raise ValueError("gradient must be finite; a non-finite entry would "
                             "poison the asymptote history for every later step")
        c = self.cfg
        low, upp = self._asymptotes()
        alpha = np.maximum.reduce([self.xmin,
                                   low + c.albefa * (self.x - low),
                                   self.x - c.move * self.span])
        beta = np.minimum.reduce([self.xmax,
                                  upp - c.albefa * (upp - self.x),
                                  self.x + c.move * self.span])
        # alpha <= beta always holds for albefa < 1 and move > 0, but a
        # degenerate box could still cross them; keep the interval non-empty.
        beta = np.maximum(beta, alpha)
        gp = np.maximum(g, 0.0)
        gm = np.maximum(-g, 0.0)
        r = c.raa0 / self.span
        p0 = (upp - self.x) ** 2 * (1.001 * gp + 0.001 * gm + r)
        q0 = (self.x - low) ** 2 * (0.001 * gp + 1.001 * gm + r)
        return SubProblem(x=self.x.copy(), L=low, U=upp, alpha=alpha, beta=beta,
                          p0=p0, q0=q0, grad=g)

    @staticmethod
    def solve_subproblem(sub: SubProblem) -> np.ndarray:
        """Closed-form minimizer of the separable model on [alpha, beta].

        Derivation: d/dx [p/(U-x) + q/(x-L)] = p/(U-x)^2 - q/(x-L)^2 = 0 gives
        sqrt(p) (x - L) = sqrt(q) (U - x). Both terms are convex on (L, U), so
        the stationary point is the unconstrained minimizer and clipping into
        the box interval gives the constrained one.
        """
        sp = np.sqrt(sub.p0)
        sq = np.sqrt(sub.q0)
        den = sp + sq
        # den == 0 only if p0 = q0 = 0, i.e. a zero gradient with raa0 = 0.
        # The model is then flat and the honest answer is not to move.
        safe = den > 0.0
        x_new = np.where(safe, (sp * sub.L + sq * sub.U) / np.where(safe, den, 1.0),
                         sub.x)
        return np.clip(x_new, sub.alpha, sub.beta)

    # -- one iteration ------------------------------------------------------

    def step(self, grad: np.ndarray) -> np.ndarray:
        """One design update from one gradient. Returns the NEW iterate."""
        sub = self.build_subproblem(grad)
        x_new = self.solve_subproblem(sub)
        self._L, self._U = sub.L, sub.U
        self.last_sub = sub
        self.xold2 = self.xold1
        self.xold1 = self.x
        self.x = x_new
        self.iteration += 1
        return self.x.copy()

    # -- reporting ----------------------------------------------------------

    @property
    def L(self) -> Optional[np.ndarray]:
        """The lower asymptotes of the LAST subproblem built."""
        return self._L

    @property
    def U(self) -> Optional[np.ndarray]:
        """The upper asymptotes of the LAST subproblem built."""
        return self._U

    def state(self) -> dict:
        """A JSON-safe snapshot, for the solve's per-iteration record."""
        s = self.last_sub
        return {
            "iteration": int(self.iteration),
            "gamma_frac_contracted": float(np.mean(
                self.gamma_last == self.cfg.asy_decr)),
            "gamma_frac_expanded": float(np.mean(
                self.gamma_last == self.cfg.asy_incr)),
            "asy_dist_lo_mean": None if s is None else float(np.mean(s.x - s.L)),
            "asy_dist_hi_mean": None if s is None else float(np.mean(s.U - s.x)),
            "asy_dist_lo_min": None if s is None else float(np.min(s.x - s.L)),
            "asy_dist_hi_min": None if s is None else float(np.min(s.U - s.x)),
            "frac_at_move_limit": None if s is None else float(np.mean(
                np.isclose(np.abs(self.x - s.x), self.cfg.move * self.span,
                           rtol=1e-9, atol=1e-12))),
            "frac_at_box_lo": float(np.mean(np.isclose(self.x, self.xmin))),
            "frac_at_box_hi": float(np.mean(np.isclose(self.x, self.xmax))),
            "step_rms": None if s is None else float(
                np.sqrt(np.mean((self.x - s.x) ** 2))),
            "step_max_abs": None if s is None else float(
                np.max(np.abs(self.x - s.x))),
        }
