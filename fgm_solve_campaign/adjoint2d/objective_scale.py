"""The 1/|g0| objective rescale applied once at solve start.

WHAT IT DOES. Before the first optimizer step, evaluate the gradient at the
start point, take its Euclidean norm |g0|, and from then on hand the optimizer
J / |g0| and g / |g0| instead of J and g.

WHY THAT IS SAFE. Dividing an objective and its gradient by the SAME positive
constant is a pure reparameterization of the objective's units. It cannot move
a minimizer, it cannot rotate a descent direction, and it cannot reorder two
candidate designs. The solve therefore keeps reporting RAW J values (this module
scales only what the optimizer sees), and the deliverable of a well-scaled
problem is unchanged.

WHY IT IS NEEDED. The upper-rail stall class. `scipy.optimize.minimize` with
method "L-BFGS-B" (limited-memory Broyden-Fletcher-Goldfarb-Shanno with box
constraints) terminates when

    (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol

The `max(..., 1)` means that for an objective whose magnitude sits well below 1
the nominally RELATIVE test is an ABSOLUTE one, so the solve declares success
after zero or one iteration with the design still pinned where it started. For
the production full-depth start that is the UPPER RAIL of the box [0, 1], which
is why the failure presents as "the solve returned uniform saturation 1".
Confirmed independently in the two-dimensional gear8 solve and in the
three-dimensional port lane's Phase C.

WHAT IT IS NOT. It is not preconditioning: it is one scalar, it does not touch
the metric of the design space, and it does nothing about ill conditioning
BETWEEN design variables. It only removes the units of the objective from the
optimizer's convergence test and from the method of moving asymptotes' fixed
convexity floor raa0.

GUARDS. If |g0| is zero or non-finite the factor falls back to exactly 1.0 with
a stated reason rather than producing an infinity or a silent no-op, because a
zero start gradient means the start point is already stationary (or the seed is
dead) and that is a fact the run record must carry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESCALE_DEFAULT = True


@dataclass(frozen=True)
class ObjectiveRescale:
    """The one constant the optimizer sees the objective through.

    Attributes:
        factor: the positive multiplier applied to both J and g.
        g0_norm: the measured Euclidean norm of the start gradient.
        applied: False when the rescale was disabled or guarded away.
        reason: the stated reason, always populated, never empty.
    """

    factor: float
    g0_norm: float
    applied: bool
    reason: str

    @classmethod
    def from_start_gradient(cls, g0, enabled: bool = RESCALE_DEFAULT
                            ) -> "ObjectiveRescale":
        """Build the rescale from the gradient at the solve's start point."""
        g = np.asarray(g0, dtype=float).ravel()
        n = float(np.linalg.norm(g)) if g.size else 0.0
        if not enabled:
            return cls(factor=1.0, g0_norm=n, applied=False,
                       reason="objective rescale disabled by configuration "
                              "(objective_rescale: false); the optimizer sees "
                              "raw J, the v2.0.x behaviour")
        if not np.isfinite(n):
            return cls(factor=1.0, g0_norm=float("nan"), applied=False,
                       reason="start gradient is non-finite; rescale fell back "
                              "to the identity and the solve is suspect")
        if n <= 0.0:
            return cls(factor=1.0, g0_norm=n, applied=False,
                       reason="start gradient norm is exactly zero (stationary "
                              "start or a dead objective seed); rescale fell "
                              "back to the identity")
        return cls(factor=1.0 / n, g0_norm=n, applied=True,
                   reason=f"1/|g0| objective rescale applied at solve start, "
                          f"|g0| = {n:.6e}; pure reparameterization, the "
                          f"minimizer is unchanged and reported J stays raw")

    def apply(self, J: float, g) -> tuple[float, np.ndarray]:
        """Scale one objective-and-gradient pair for the optimizer."""
        k = float(self.factor)
        return float(J) * k, np.asarray(g, dtype=float) * k

    def as_dict(self) -> dict:
        return {"applied": bool(self.applied), "factor": float(self.factor),
                "g0_norm": float(self.g0_norm), "reason": str(self.reason)}
