"""Pure logic for the multi-start / warm-start melt-objective solve.

No forward run appears in this module. It holds the budget arithmetic, the
early-kill rule, the two deterministic non-uniform starts, and the evaluation
cache that makes the probe-to-continuation handover free. Every function here
is unit tested in `tests/test_multistart.py`.

THE BUDGET SPLIT, stated once.

The whole solve is given `BUDGET_FORWARD_EQUIVALENTS` forward-equivalents per
shape, ACROSS all starts, with the adjoint-to-forward cost `ratio` measured on
that shape. One objective-plus-gradient evaluation costs `1 + ratio`
forward-equivalents, so the pool is

    n_total = floor(budget / (1 + ratio))                (`control.max_gradient_evals`)

evaluations. The pool is spent in two phases:

  PROBE        every start gets `probe_evals(n_total, n_starts, N_PROBE)`
               evaluations. The first of those is the objective at the start
               point itself, so a probe of 2 buys one descent step.
  CONTINUATION the starts that survive the kill rule restart from their probe's
               LAST iterate and split the remaining pool equally.

THE EARLY-KILL RULE, stated once and applied without exception.

After the probe, let J_k be the best J_phi start k reached and J* the smallest
of them. Start k SURVIVES when

    J_k <= J* + KILL_MARGIN * |J*|        and        rank(J_k) < MAX_KEEP

with the rank taken in ascending J_k and exact ties broken by the order the
starts were declared. The leader always survives. A killed start spends nothing
further and its probe rows are kept and reported.

THE CONTINUATION RESTART IS NOT FREE, and that is stated rather than hidden.
`scipy.optimize.minimize` cannot be resumed, so a survivor's continuation is a
FRESH L-BFGS-B from the probe's last iterate. The limited-memory curvature
accumulated during the probe is discarded. With N_PROBE = 2 that memory holds
at most one secant pair, so the loss is small, but it is not zero and it was not
measured. What IS free is the repeated evaluation: the continuation's first
objective call lands on exactly the vector the probe last evaluated, the forward
is deterministic (`DENSITY_OBJECTIVE_LIBRARY_REPORT.md` Section 2.3 measured a
repeat difference of exactly 0.000e+00), and `EvalCache` returns it without
running anything and without charging the budget.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

BUDGET_FORWARD_EQUIVALENTS = 40.0
N_PROBE = 2          # gradient evaluations per start in the probe phase
KILL_MARGIN = 0.10   # a start survives if it is within 10 percent of the leader
MAX_KEEP = 2         # at most this many starts continue


# ---------------------------------------------------------------------------
# budget arithmetic
# ---------------------------------------------------------------------------

def probe_evals(n_total: int, n_starts: int, n_probe: int = N_PROBE) -> int:
    """Gradient evaluations each start gets in the probe phase.

    Never more than an even share of the pool, and never fewer than one, so a
    start always at least reports the objective at its own start point.
    """
    if int(n_starts) <= 0:
        raise ValueError(f"n_starts must be positive, got {n_starts!r}")
    share = int(n_total) // int(n_starts)
    return max(1, min(int(n_probe), share))


def continuation_evals(n_total: int, n_spent: int, n_survivors: int) -> int:
    """Additional evaluations each survivor gets, floor-divided, never negative."""
    if int(n_survivors) <= 0:
        return 0
    return max(0, (int(n_total) - int(n_spent)) // int(n_survivors))


# ---------------------------------------------------------------------------
# the early-kill rule
# ---------------------------------------------------------------------------

def survivors(best_J: Mapping[str, float | None], margin: float = KILL_MARGIN,
              max_keep: int = MAX_KEEP) -> list[str]:
    """Which starts continue past the probe. Ordered best first.

    A start with no result (None) never survives. The comparison is absolute in
    |J*| rather than a ratio so a leader at exactly zero does not divide by
    zero; J_phi is a sum of squares and is non-negative, so the two readings
    agree everywhere else.
    """
    order = list(best_J.keys())
    live = [(k, float(best_J[k])) for k in order if best_J[k] is not None]
    if not live:
        return []
    live.sort(key=lambda kv: (kv[1], order.index(kv[0])))
    j_star = live[0][1]
    cut = j_star + float(margin) * abs(j_star)
    keep = [k for i, (k, j) in enumerate(live)
            if i < int(max_keep) and (i == 0 or j <= cut)]
    return keep


def best_final(results: Mapping[str, dict | None], key: str = "J") -> str | None:
    """The start whose final row has the lowest `key`. None if there is none."""
    order = list(results.keys())
    live = [(k, float(results[k][key])) for k in order if results[k] is not None]
    if not live:
        return None
    live.sort(key=lambda kv: (kv[1], order.index(kv[0])))
    return live[0][0]


# ---------------------------------------------------------------------------
# deterministic starts
# ---------------------------------------------------------------------------

def start_from_map(s: np.ndarray, part_mask: np.ndarray,
                   box: tuple[float, float]) -> np.ndarray:
    """Project a stored dopant map into a design variable inside `box`.

    In the part the value is clipped into the box. Outside the part the nominal
    saturation 1 is held, the prototype convention, so a start changes the
    dopant map and not the sub-pixel geometry fill of boundary cells.
    """
    a = np.asarray(s, dtype=float)
    pm = np.asarray(part_mask, dtype=bool)
    return np.where(pm, np.clip(a, float(box[0]), float(box[1])), 1.0)


def perturbed_start(pi_map: np.ndarray, part_mask: np.ndarray) -> np.ndarray:
    """The deterministic perturbed start: the MIDPOINT of uniform and `pi_map`.

    The task's phrase is "uniform plus the proportional-inverse map at gain
    0.5". Read literally as a sum it is degenerate: the proportional-inverse map
    at gain 0.5 lies in [0.25, 0.75] inside the part, so 1 + that is above 1
    everywhere and clipping to the box returns the uniform start exactly,
    duplicating start (a). The midpoint is the non-degenerate reading of the
    same intent, it is seedless, and it is bit-reproducible on a resume.
    """
    a = np.asarray(pi_map, dtype=float)
    pm = np.asarray(part_mask, dtype=bool)
    mid = 0.5 * (1.0 + a)
    return np.where(pm, np.clip(mid, 0.0, 1.0), 1.0)


# ---------------------------------------------------------------------------
# the evaluation cache
# ---------------------------------------------------------------------------

class EvalCache:
    """Exact-vector memo of (objective, gradient), keyed on the raw bytes.

    Only a BIT-IDENTICAL design vector hits. The forward is deterministic, so a
    hit is the same numbers the forward would have produced, not an
    approximation.
    """

    def __init__(self) -> None:
        self._d: dict[bytes, tuple[float, np.ndarray]] = {}
        self.n_hit = 0
        self.n_miss = 0

    @staticmethod
    def _key(v: Sequence[float] | np.ndarray) -> bytes:
        return np.ascontiguousarray(np.asarray(v, dtype=float)).tobytes()

    def get(self, v) -> tuple[float, np.ndarray] | None:
        hit = self._d.get(self._key(v))
        if hit is None:
            self.n_miss += 1
            return None
        self.n_hit += 1
        return hit

    def put(self, v, value: tuple[float, np.ndarray]) -> None:
        self._d[self._key(v)] = (float(value[0]), np.asarray(value[1], dtype=float))

    def __len__(self) -> int:
        return len(self._d)
