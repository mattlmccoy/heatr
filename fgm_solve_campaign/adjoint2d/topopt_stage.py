"""One continuation stage, run under a choice of optimizer.

WHY THIS EXISTS. `TOPOPT_REPORT.md` Section 4.3 and `FROZEN_CONVENTIONS_2D.md`
Section 9 item 4 both record the same open question: the measured in-grid cost
of the smoothed-Heaviside projection may be an artifact of the OPTIMIZER rather
than of the projection. With three evaluations per beta stage, L-BFGS-B
(limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints) spends
most of a stage on line-search trial points that are worse than the incumbent,
and its curvature memory is discarded at every stage restart.

To settle that, two optimizers have to run against the IDENTICAL objective and
the IDENTICAL budget accounting. This module is the seam that makes that true
by construction: both optimizers consume one `evaluate` callable and one
integer budget, and neither can see anything else.

THE THREE MODES.

  `lbfgsb`        the production recipe. One `scipy.optimize.minimize` call per
                  beta stage, budget enforced by raising `StopIteration` from
                  the objective, curvature memory discarded at each stage
                  boundary. Unchanged from `topopt_solve` before this pass.
  `mma`           the method of moving asymptotes (`mma.MMA`). One design
                  update per evaluation, no line search, and the asymptote
                  state is owned by the CALLER so it survives the stage
                  boundary. This is the mechanics fix.
  `lbfgsb_carry`  L-BFGS-B run ONCE across the whole pool, with beta switched
                  inside the objective at the stage boundaries. This is the
                  cheap alternative arm: scipy exposes no way to seed the
                  curvature memory of a new call, so the only way to carry it
                  is never to end the call. The driver, not this module,
                  arranges the beta switching; from here it is a single stage
                  whose objective happens to change.

The budget is counted the same way in every mode: one unit per call of
`evaluate`. The stage returns the BEST evaluated design point, never the last,
because MMA without the globally convergent variant is not a descent method
(see `mma.py` simplification 3).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from . import mma as mma_mod

OPTIMIZERS = ("lbfgsb", "mma", "lbfgsb_carry")

EvaluateFn = Callable[[np.ndarray], tuple[float, np.ndarray]]


class StageRunner:
    """Runs one budgeted stage of one optimizer against one evaluate callable.

    `evaluate(vec) -> (J, grad)` is called at most `n_new` times per `run`.
    Every call is logged with its design point so the caller can recover the
    best iterate and so the two optimizers' trajectories can be compared
    directly.
    """

    def __init__(self, evaluate: EvaluateFn, box: tuple[float, float],
                 optimizer: str = "lbfgsb",
                 mma_state: Optional[mma_mod.MMA] = None) -> None:
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {optimizer!r}; expected one of "
                             f"{OPTIMIZERS}")
        self.evaluate = evaluate
        self.box = (float(box[0]), float(box[1]))
        if self.box[1] <= self.box[0]:
            raise ValueError(f"box must be increasing, got {box!r}")
        self.optimizer = optimizer
        self.mma_state = mma_state
        self.rows: list[dict] = []
        self.points: dict[int, np.ndarray] = {}
        self.n_used = 0
        self.n_restarts = 0

    # -- the shared evaluation seam -----------------------------------------

    def _call(self, vec: np.ndarray, n_new: int) -> tuple[float, np.ndarray]:
        if self.n_used >= int(n_new):
            raise StopIteration
        v = np.clip(np.asarray(vec, dtype=float).ravel(), self.box[0], self.box[1])
        J, g = self.evaluate(v)
        self.n_used += 1
        self.points[self.n_used] = v.copy()
        self.rows.append({"eval_index": self.n_used, "J": float(J)})
        return float(J), np.asarray(g, dtype=float).ravel()

    # -- the three modes ----------------------------------------------------

    def _one_lbfgsb_call(self, v0: np.ndarray, n_new: int) -> None:
        def fun(vec):
            J, g = self._call(vec, n_new)
            return J, g
        try:
            minimize(fun, v0, jac=True, method="L-BFGS-B",
                     bounds=[self.box] * v0.size,
                     options={"maxiter": 10_000, "maxfun": 10_000,
                              "ftol": 1e-16, "gtol": 1e-16})
        except StopIteration:
            pass

    def _run_lbfgsb(self, v0: np.ndarray, n_new: int) -> None:
        """The production recipe: exactly ONE scipy call per stage.

        If scipy declares convergence before the stage budget is spent, the
        stage simply ends. That is the behaviour of every result in
        `TOPOPT_REPORT.md` and it is preserved here unchanged, verified
        bit-identical on all six shapes of the filter-only arm.
        """
        self._one_lbfgsb_call(v0, n_new)

    def _run_lbfgsb_carry(self, v0: np.ndarray, n_new: int) -> None:
        """One call if possible, re-entered from the current point if not.

        MEASURED REASON, not an assumption. On the triangle at 80
        forward-equivalents this arm's single scipy call declared convergence
        after 12 of 32 allowed evaluations, at beta 2, leaving the beta 4, 8
        and 16 stages with no evaluation at all
        (`logs_mma/triangle_lbfgsb_carry_b80.log`, first attempt). An arm that
        never reaches beta 16 cannot be compared with arms that do.

        Each re-entry DISCARDS the curvature memory, which is exactly what this
        arm was built to avoid, so the count is reported and quoted with every
        number this arm produces. The re-entries only happen where scipy had
        already given up, which is where the memory was stale in any case.
        """
        self.n_restarts = 0
        x = np.asarray(v0, dtype=float).copy()
        while self.n_used < int(n_new):
            before = self.n_used
            self._one_lbfgsb_call(x, n_new)
            if self.n_used == before:
                break                       # cannot make progress; do not spin
            if self.rows:
                b = min(self.rows, key=lambda r: r["J"])
                x = self.points[b["eval_index"]].copy()
            if self.n_used < int(n_new):
                self.n_restarts += 1

    def _run_mma(self, v0: np.ndarray, n_new: int) -> None:
        state = self.mma_state
        if state is None:
            raise ValueError(
                "optimizer 'mma' needs an mma.MMA state object owned by the "
                "caller. It is required rather than created here so that a "
                "stage boundary cannot silently reset the asymptotes, which is "
                "the property this arm exists to test.")
        state.x = np.clip(np.asarray(v0, dtype=float).ravel(),
                          state.xmin, state.xmax)
        x = state.x
        for _ in range(int(n_new)):
            try:
                _J, g = self._call(x, n_new)
            except StopIteration:
                break
            x = state.step(g)

    # -- entry point --------------------------------------------------------

    def run(self, v_start: np.ndarray, n_new: int) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        v0 = np.clip(np.asarray(v_start, dtype=float).ravel(), self.box[0], self.box[1])
        info: dict = {"optimizer": self.optimizer, "n_new_allowed": int(n_new)}
        if int(n_new) > 0:
            if self.optimizer == "mma":
                self._run_mma(v0, int(n_new))
            elif self.optimizer == "lbfgsb_carry":
                self._run_lbfgsb_carry(v0, int(n_new))
            else:
                self._run_lbfgsb(v0, int(n_new))
        info["n_new_used"] = self.n_used
        info["n_optimizer_restarts"] = int(self.n_restarts)
        info["wall_s"] = time.perf_counter() - t0
        if self.rows:
            b = min(self.rows, key=lambda r: r["J"])
            info["best_J"] = b["J"]
            info["best_eval_index"] = b["eval_index"]
            info["first_J"] = self.rows[0]["J"]
            out = self.points[b["eval_index"]]
        else:
            info["best_J"] = None
            info["best_eval_index"] = None
            out = v0
        if self.optimizer == "mma" and self.mma_state is not None:
            info["mma_state"] = self.mma_state.state()
        return out.copy(), info
