"""Asymmetric dwell scheduling: the pure logic, with no physics in it.

THE ACTUATOR. The part sits on a turntable that can be COMMANDED TO INDEXED
POSITIONS and held there. The design variables are the fraction of the total
exposure spent at each position of a fixed candidate set. Radio-frequency
generator power is constant throughout; the only thing being scheduled is where
the part is pointing and for how long.

THE EXECUTION MODEL, and why it makes the design variable a set of weights.
The schedule is executed as REPEATED SHORT CYCLES through the kept positions,
each cycle allocating time in proportion to the dwell fractions. If the cycle
time is short compared with the part's thermal times, every material point sees
the time average of the heating, which is the WEIGHTED angle average

    Q_eff(x) = sum_k w_k * Q_k(x),        w_k >= 0,   sum_k w_k = 1

with Q_k the part-frame heating at position k. That is the same quasi-static
argument the continuous-rotation pass used, with uniform weights replaced by
free ones, so the dwell fractions ARE the weights and the physical dwell
durations are d_k = w_k * exposure. The cycle time is a stated parameter of the
program, not a fitted one, and the error of the quasi-static step is measured
against the true rotating engine rather than assumed.

TWO PARAMETERIZATIONS, both here, used for different jobs.

  * `softmax_weights` is what the OPTIMIZER moves. It is smooth, keeps the
    weights strictly positive and exactly normalized for any real vector, and
    therefore needs no constraint handling and introduces no kink into the
    finite-difference gate. Its one cost is that a weight can only reach zero
    in the limit, so an exactly-zero dwell is never produced; the reported
    programs quantize to the step grid and a position below half a step drops
    out there.
  * `project_to_simplex` is the Euclidean projection onto the simplex. It is
    used to SNAP a weight vector onto the feasible set (for reporting, for
    comparing against an externally supplied vector, and as the non-smooth
    alternative parameterization). It is not used inside the gated gradient
    chain, precisely because it is non-smooth at the faces of the simplex.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# simplex projection and softmax
# ---------------------------------------------------------------------------

def project_to_simplex(v: Sequence[float], total: float = 1.0) -> np.ndarray:
    """Euclidean projection of `v` onto {w >= 0, sum w = total}.

    The standard sort-and-threshold algorithm: with the coordinates sorted
    descending, the projection is `max(v - tau, 0)` for the unique `tau` that
    makes the sum equal `total`.
    """
    a = np.asarray(v, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("project_to_simplex needs at least one coordinate")
    t = float(total)
    if t <= 0.0:
        raise ValueError(f"total must be positive, got {total!r}")
    u = np.sort(a)[::-1]
    css = np.cumsum(u) - t
    idx = np.arange(1, a.size + 1, dtype=float)
    cond = u - css / idx > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    tau = css[rho] / float(rho + 1)
    return np.maximum(a - tau, 0.0)


def softmax_weights(z: Sequence[float]) -> np.ndarray:
    """`w_k = exp(z_k) / sum_j exp(z_j)`, shifted for overflow safety."""
    a = np.asarray(z, dtype=float).ravel()
    e = np.exp(a - np.max(a))
    return e / float(np.sum(e))


def softmax_vjp(w: np.ndarray, g_w: np.ndarray) -> np.ndarray:
    """`dJ/dz` given `dJ/dw` and the weights: `w * (g - <w, g>)`."""
    a = np.asarray(w, dtype=float).ravel()
    g = np.asarray(g_w, dtype=float).ravel()
    if a.shape != g.shape:
        raise ValueError(f"shape mismatch {a.shape} against {g.shape}")
    return a * (g - float(np.dot(a, g)))


def softmax_logits_for(w: Sequence[float], floor: float = 1e-6) -> np.ndarray:
    """Logits whose softmax is `w`, centred at zero. Inverse of `softmax_weights`."""
    a = np.maximum(np.asarray(w, dtype=float).ravel(), float(floor))
    z = np.log(a / float(np.sum(a)))
    return z - float(np.mean(z))


# ---------------------------------------------------------------------------
# the turntable program
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TurntableProgram:
    """A machine-readable indexed-dwell program.

    `moves` is the ordered list the machine executes, one entry per hold:
    `position_deg` the commanded turntable angle, `dwell_s` how long to hold it,
    `move_at_s` the time from exposure start at which the move to that position
    is commanded (so the hold runs from `move_at_s` to `move_at_s + dwell_s`).
    Move duration itself is not modelled and is stated as a limit.
    """

    moves: tuple[dict, ...]
    kept_positions_deg: tuple[float, ...]
    steps_per_position: tuple[int, ...]
    realized_weights: np.ndarray
    requested_weights: np.ndarray
    cycle_time_s: float
    total_s: float
    dt_s: float
    n_cycles: float
    steps_per_position_total: tuple[int, ...] = ()
    cycle_slot_plan: tuple[tuple[int, ...], ...] = ()
    allocation_rule: str = "carry-forward largest remainder across cycles"

    def as_json(self) -> dict:
        return {
            "actuator": "indexed turntable, constant radio-frequency power",
            "cycle_time_s": self.cycle_time_s,
            "total_exposure_s": self.total_s,
            "control_step_s": self.dt_s,
            "n_cycles": self.n_cycles,
            "positions_deg": list(self.kept_positions_deg),
            "steps_per_position_per_cycle": list(self.steps_per_position),
            "steps_per_position_over_exposure": list(self.steps_per_position_total),
            "allocation_rule": self.allocation_rule,
            "realized_dwell_fraction": [float(x) for x in self.realized_weights],
            "requested_dwell_fraction": [float(x) for x in self.requested_weights],
            "total_dwell_s_per_position": [
                float(w * self.total_s) for w in self.realized_weights],
            "n_moves": len(self.moves),
            "moves": [dict(m) for m in self.moves],
        }


def largest_remainder(weights: Sequence[float], n_slots: int) -> np.ndarray:
    """Apportion `n_slots` integer slots in proportion to `weights`.

    Hamilton's largest-remainder rule: floor the exact quotas, then hand the
    leftover slots to the largest fractional remainders. Deterministic ties go
    to the earlier position.
    """
    w = np.asarray(weights, dtype=float).ravel()
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    tot = float(np.sum(w))
    if tot <= 0.0:
        raise ValueError("weights must not be all zero")
    exact = w / tot * int(n_slots)
    base = np.floor(exact).astype(int)
    left = int(n_slots) - int(base.sum())
    if left > 0:
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:left]] += 1
    return base


def carry_forward_slots(frac: Sequence[float], n_slots: int,
                        n_total: int) -> np.ndarray:
    """Per-cycle integer slot allocations that carry their rounding forward.

    THE BUG THIS EXISTS TO FIX. Applying largest-remainder INSIDE one cycle and
    then repeating that same allocation every cycle turns a per-cycle rounding
    into a permanent bias. With `n_slots` 40 control steps and 12 equally
    weighted positions the per-cycle quota is 3.333 steps, largest remainder
    gives 4/4/4/4 then 3 eight times, and repeating it realizes dwell fractions
    0.100 and 0.075 against a design 1/12 = 0.08333 forever. That is the
    keyhole's emitted program (`ROTATING_HOLDOUT_REPORT.md` Section 5) and it
    cost 73.2 percent of J at grid 120 with no physics involved.

    THE RULE. Largest remainder is applied to a running DEBT, the exact quota
    accumulated so far minus the slots already handed out. A position that was
    rounded up this cycle carries a negative debt into the next one and is
    rounded down there, so the leftover steps rotate. Each cycle still receives
    exactly its own slot count, every allocation is non-negative, and the
    cumulative allocation tracks the exact quota to within one control step
    over the whole exposure.

    Returns an integer array of shape `(n_cycles, k)`. The final row is a
    partial cycle when `n_total` is not a multiple of `n_slots`.
    """
    f = np.asarray(frac, dtype=float).ravel()
    if np.any(f < 0.0):
        raise ValueError("weights must be non-negative")
    tot = float(np.sum(f))
    if tot <= 0.0:
        raise ValueError("weights must not be all zero")
    f = f / tot
    ns, nt = int(n_slots), int(n_total)
    if ns < 1:
        raise ValueError(f"n_slots must be at least 1, got {n_slots!r}")
    rows: list[np.ndarray] = []
    debt = np.zeros(f.size)
    done = 0
    while done < nt:
        chunk = min(ns, nt - done)
        debt = debt + f * chunk
        base = np.floor(debt).astype(int)
        # A position rounded up in an earlier cycle can carry a negative debt;
        # its floor is then negative, which is not a dwell. Clamp, then repair
        # the cycle total deterministically by the same remainder ordering.
        np.maximum(base, 0, out=base)
        rem = debt - base
        left = chunk - int(base.sum())
        if left > 0:
            order = np.argsort(-rem, kind="stable")
            base[order[:left]] += 1
        elif left < 0:
            order = np.argsort(rem, kind="stable")
            for j in order:
                if left == 0:
                    break
                take = min(int(base[j]), -left)
                base[j] -= take
                left += take
            if left != 0:                                   # pragma: no cover
                raise RuntimeError(
                    f"could not fit {chunk} control steps into the cycle "
                    f"allocation; {left} left over")
        rows.append(base)
        debt = debt - base
        done += chunk
    return np.asarray(rows, dtype=int)


def cycle_program(weights: Sequence[float], angles_deg: Sequence[float],
                  cycle_time_s: float, total_s: float, dt_s: float,
                  drop_below: float = 0.5) -> TurntableProgram:
    """Turn dwell fractions into the ordered list of holds the machine runs.

    A position whose per-cycle allotment rounds below `drop_below` control steps
    is DROPPED, and the remaining positions are re-apportioned over the whole
    cycle. That is what makes "park nowhere near this orientation" expressible
    as a program even though the softmax never returns exactly zero.

    THE ALLOCATION IS DIVISOR AWARE. The leftover control steps of a cycle that
    does not divide evenly across the kept positions ROTATE from cycle to cycle
    (`carry_forward_slots`), so the dwell fraction the machine actually realizes
    over the exposure matches the design to within one control step even when
    the cycle length is not a multiple of the position count. Before this the
    same per-cycle rounding was repeated every cycle and became permanent; the
    keyhole's twelve-position program realized 0.100 and 0.075 instead of
    1/12 and paid 73.2 percent of J for it (`ROTATING_HOLDOUT_REPORT.md`
    Section 5).
    """
    w_req = np.asarray(weights, dtype=float).ravel()
    ang = np.asarray(angles_deg, dtype=float).ravel()
    if w_req.shape != ang.shape:
        raise ValueError(f"{w_req.size} weights against {ang.size} angles")
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s!r}")
    n_slots = int(round(float(cycle_time_s) / dt))
    if n_slots < 1:
        raise ValueError(f"cycle_time_s {cycle_time_s!r} is under one control step")

    frac = w_req / max(float(np.sum(w_req)), 1e-300)
    keep = frac * n_slots >= float(drop_below)
    if not keep.any():
        keep = frac >= frac.max()
    if int(keep.sum()) > n_slots:
        raise ValueError(
            f"cycle of {n_slots} control steps cannot give each of "
            f"{int(keep.sum())} kept positions a step; lengthen cycle_time_s")

    total = float(total_s)
    n_total = int(round(total / dt))
    kept_ang = ang[keep]
    kept_frac = frac[keep]
    plan = carry_forward_slots(kept_frac, n_slots, n_total)
    # A position that survived the keep test but wins no control step ANYWHERE
    # in the exposure would be a silent no-op in the program; drop it and
    # re-apportion. The test is over the exposure, not over one cycle, because
    # under the carry-forward rule a position legitimately sits out some cycles.
    while np.any(plan.sum(axis=0) == 0) and int(np.sum(plan.sum(axis=0) > 0)) >= 1:
        alive = plan.sum(axis=0) > 0
        kept_ang = kept_ang[alive]
        kept_frac = kept_frac[alive]
        plan = carry_forward_slots(kept_frac, n_slots, n_total)

    totals = plan.sum(axis=0)
    slots = plan[0]
    realized = totals / float(max(n_total, 1))
    n_cycles = total / (n_slots * dt)

    moves: list[dict] = []
    t = 0.0
    if len(kept_ang) == 1:
        moves.append({"position_deg": float(kept_ang[0]), "dwell_s": total,
                      "move_at_s": 0.0})
    else:
        for row in plan:
            for i in range(len(kept_ang)):
                if t >= total - 1e-9:
                    break
                hold = min(int(row[i]) * dt, total - t)
                if hold <= 1e-9:
                    continue
                if moves and moves[-1]["position_deg"] == float(kept_ang[i]):
                    moves[-1]["dwell_s"] += float(hold)   # no null move
                else:
                    moves.append({"position_deg": float(kept_ang[i]),
                                  "dwell_s": float(hold), "move_at_s": float(t)})
                t += hold
        if moves and t < total - 1e-9:
            moves[-1]["dwell_s"] += total - t              # sub-step remainder
    return TurntableProgram(
        moves=tuple(moves), kept_positions_deg=tuple(float(a) for a in kept_ang),
        steps_per_position=tuple(int(s) for s in slots),
        realized_weights=realized, requested_weights=w_req,
        cycle_time_s=float(n_slots * dt), total_s=total, dt_s=dt,
        n_cycles=float(n_cycles),
        steps_per_position_total=tuple(int(s) for s in totals),
        cycle_slot_plan=tuple(tuple(int(v) for v in r) for r in plan))


def expand_to_full_weights(prog: TurntableProgram,
                           angles_deg: Sequence[float]) -> np.ndarray:
    """The realized dwell fractions on the FULL candidate set, dropped ones zero."""
    ang = np.asarray(angles_deg, dtype=float).ravel()
    out = np.zeros(ang.size)
    for a, w in zip(prog.kept_positions_deg, prog.realized_weights):
        j = int(np.argmin(np.abs(ang - a)))
        out[j] = float(w)
    return out


def dwell_asymmetry(weights: Sequence[float]) -> dict:
    """How far a dwell vector is from equal, in three readings."""
    w = np.asarray(weights, dtype=float).ravel()
    k = w.size
    eq = 1.0 / k
    p = np.clip(w, 1e-300, None)
    ent = float(-np.sum(p * np.log(p)) / np.log(k)) if k > 1 else 1.0
    return {
        "max_over_equal": float(np.max(w) / eq),
        "min_over_equal": float(np.min(w) / eq),
        "max_minus_min": float(np.max(w) - np.min(w)),
        "l1_from_equal": float(np.sum(np.abs(w - eq))),
        "normalized_entropy": ent,
        "effective_positions": float(np.exp(-np.sum(p * np.log(p)))),
    }
