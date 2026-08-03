"""Resumable L-BFGS-B driving: an interruption costs ONE iteration, not the run.

Written after a laptop shutdown killed a pyramid solve five evaluations into a
twelve-evaluation budget, losing about two hours because the arm only wrote its
result after the loop finished.

WHAT IS AND IS NOT PRESERVED, stated plainly because it bounds what a resume
means. The checkpoint stores the design vector, the best-so-far, the objective
scale and the full history. It does NOT store L-BFGS-B's limited-memory
curvature approximation, which scipy does not expose. So a resume is a WARM
RESTART from the best iterate with the remaining budget, and the curvature
memory is rebuilt. That is exactly the cost the frozen 2-D conventions already
accept at every beta-continuation stage ("the optimizer's curvature memory is
discarded at each restart"), and each stage there likewise keeps its own best
iterate. A resumed run is therefore not bit-identical to an uninterrupted one;
it is budget-honest and never worse than the checkpoint it resumed from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class _Stop(Exception):
    pass


class SolveCheckpoint:
    def __init__(self, path, budget: int):
        self.path = Path(path) if path is not None else None
        self.budget = int(budget)

    def load(self) -> dict | None:
        if self.path is None or not self.path.exists():
            return None
        z = np.load(self.path, allow_pickle=False)
        return {"n": int(z["n"]), "best_J": float(z["best_J"]),
                "best_v": np.asarray(z["best_v"], dtype=float),
                "scale": float(z["scale"]),
                "hist": json.loads(str(z["hist"]))}

    def save(self, n: int, best_J: float, best_v: np.ndarray, scale: float,
             hist: list) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, n=int(n), best_J=float(best_J),
                            best_v=np.asarray(best_v, dtype=float),
                            scale=float(scale), hist=json.dumps(hist))
        tmp.replace(self.path)          # atomic: never a half-written checkpoint

    @property
    def done(self) -> bool:
        st = self.load()
        return bool(st is not None and st["n"] >= self.budget)


def run_with_checkpoint(fg, v0: np.ndarray, budget: int, path,
                        bounds: tuple[float, float] = (0.0, 1.0),
                        scale_first_step: bool = True,
                        stop_after: int | None = None,
                        on_eval=None) -> dict:
    """L-BFGS-B with a checkpoint written after EVERY objective evaluation.

    `fg(v) -> (J, grad)` in UNSCALED units. `stop_after` is a test hook that
    simulates an interruption.
    """
    from scipy.optimize import minimize

    ck = SolveCheckpoint(path, budget)
    st0 = ck.load()
    if st0 is not None and st0["n"] >= budget:
        return {"already_complete": True, "evals_used": st0["n"],
                "best_J": st0["best_J"], "best_v": st0["best_v"],
                "hist": st0["hist"], "scale": st0["scale"],
                "resumed_from_eval": st0["n"], "interrupted": False}

    start_n = 0 if st0 is None else st0["n"]
    v_start = np.asarray(v0, float) if st0 is None else st0["best_v"]
    s = {"n": start_n,
         "best_J": np.inf if st0 is None else st0["best_J"],
         "best_v": np.asarray(v_start, float).copy(),
         "scale": 1.0 if st0 is None else st0["scale"],
         "hist": [] if st0 is None else list(st0["hist"]),
         "interrupted": False}

    def wrapped(v):
        if s["n"] >= budget:
            raise _Stop()
        if stop_after is not None and (s["n"] - start_n) >= stop_after:
            s["interrupted"] = True
            raise _Stop()
        J, g = fg(v)
        s["n"] += 1
        if scale_first_step and s["n"] == 1:
            gn = float(np.linalg.norm(g))
            s["scale"] = (1.0 / gn) if gn > 0 else 1.0
        if J < s["best_J"]:
            s["best_J"], s["best_v"] = float(J), np.asarray(v, float).copy()
        rec = {"eval": s["n"], "J": float(J),
               "grad_norm": float(np.linalg.norm(g))}
        if on_eval is not None:
            rec.update(on_eval(v, J, g) or {})
        s["hist"].append(rec)
        ck.save(s["n"], s["best_J"], s["best_v"], s["scale"], s["hist"])
        return J * s["scale"], g * s["scale"]

    status = "budget_exhausted"
    try:
        r = minimize(wrapped, np.asarray(v_start, float), jac=True,
                     method="L-BFGS-B", bounds=[bounds] * len(v_start),
                     options={"ftol": 1e-16, "gtol": 1e-16, "maxiter": 10000})
        status = f"converged:{r.message}"
    except _Stop:
        pass
    return {"already_complete": False, "evals_used": s["n"],
            "best_J": s["best_J"], "best_v": s["best_v"], "hist": s["hist"],
            "scale": s["scale"], "status": status,
            "resumed_from_eval": start_n, "interrupted": s["interrupted"]}
