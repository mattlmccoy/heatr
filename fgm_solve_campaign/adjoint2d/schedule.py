"""The temporal power-scheduling actuator p(t).

The RF generator can be switched on and off in a scheduled pattern during the
cure and its power can be fluctuated continuously. Both are real hardware
knobs. This module holds the pure logic of the control parameterization; the
physics injection lives in `forward.py` and the gradient in `adjoint.py`.

PARAMETERIZATION. p is piecewise constant on `n_seg` segments of the NOMINAL
horizon `n_steps` (the configured outer-step count, not the realized length of
a truncated march). Fixing the segment boundaries to the nominal horizon is
what makes the design variable stable: if the boundaries moved with the
realized march length, the same p vector would mean different things at
different points of the optimization and the gradient would be wrong.

Segment k owns outer steps `ceil(k*n_steps/n_seg) .. ceil((k+1)*n_steps/n_seg)`,
which is the exact inverse of `segment_index(it) = it*n_seg // n_steps`. When
`n_steps` is not divisible by `n_seg` the segments differ in length by one
step, so every time-weighted quantity (the duty cycle above all) is weighted by
segment LENGTH and never by segment count.

INJECTION CONVENTION, stated once and tested. p multiplies POWER. The
electro-quasi-static problem is linear in the potential at fixed material
properties, so scaling the drive voltage by sqrt(p) scales the field by sqrt(p)
and the volumetric heating Q_rf = 0.5 * power_factor * sigma * |E|^2 by exactly
p. Multiplying Q_rf by p is therefore not an approximation of a voltage change,
it IS the voltage change V -> V*sqrt(p), with no re-solve needed. The cap
`max_qrf` is applied AFTER the scaling, which is where it would act on a real
re-solve.
"""
from __future__ import annotations

import math

import numpy as np

NOMINAL_LEVEL = 1.0
BINARY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# segment mapping
# ---------------------------------------------------------------------------

def segment_index(it: int, n_steps: int, n_seg: int) -> int:
    """Which segment outer step `it` belongs to, clamped past the horizon."""
    k = (int(it) * int(n_seg)) // int(n_steps)
    return min(max(k, 0), int(n_seg) - 1)


def segment_bounds(n_steps: int, n_seg: int) -> list[tuple[int, int]]:
    """[(lo, hi)] outer-step ranges, half open, partitioning [0, n_steps)."""
    n_steps, n_seg = int(n_steps), int(n_seg)
    edges = [math.ceil(k * n_steps / n_seg) for k in range(n_seg + 1)]
    return [(edges[k], edges[k + 1]) for k in range(n_seg)]


def expand(p_seg: np.ndarray, n_steps: int, n_seg: int) -> np.ndarray:
    """Per-outer-step power scale from the segment vector."""
    p = np.asarray(p_seg, dtype=float).ravel()
    if p.size != int(n_seg):
        raise ValueError(f"expected {n_seg} segments, got {p.size}")
    out = np.empty(int(n_steps), dtype=float)
    for k, (lo, hi) in enumerate(segment_bounds(n_steps, n_seg)):
        out[lo:hi] = p[k]
    return out


def accumulate_to_segments(per_step: np.ndarray, n_steps: int, n_seg: int) -> np.ndarray:
    """Adjoint of `expand`: segment-wise sums of a per-outer-step quantity.

    `per_step` may be SHORTER than `n_steps` when the march stopped early; the
    missing steps contribute exactly zero.
    """
    v = np.asarray(per_step, dtype=float).ravel()
    out = np.zeros(int(n_seg), dtype=float)
    for k, (lo, hi) in enumerate(segment_bounds(n_steps, n_seg)):
        if lo >= v.size:
            break
        out[k] = float(np.sum(v[lo:min(hi, v.size)]))
    return out


# ---------------------------------------------------------------------------
# schedule classes
# ---------------------------------------------------------------------------

def round_binary(p_seg: np.ndarray, threshold: float = BINARY_THRESHOLD) -> np.ndarray:
    """Round a continuous relaxation onto the binary on/off set {0, 1}."""
    return (np.asarray(p_seg, dtype=float) >= float(threshold)).astype(float)


def duty_cycle(p_seg: np.ndarray, n_steps: int, n_seg: int) -> float:
    """Time-weighted mean power scale. For a binary schedule this is the
    fraction of the horizon the generator is ON."""
    p = np.asarray(p_seg, dtype=float).ravel()
    tot = 0.0
    for k, (lo, hi) in enumerate(segment_bounds(n_steps, n_seg)):
        tot += (hi - lo) * p[k]
    return float(tot / float(n_steps))


def n_switches(p_seg: np.ndarray) -> int:
    """How many times the level changes between adjacent segments."""
    p = np.asarray(p_seg, dtype=float).ravel()
    return int(np.sum(p[1:] != p[:-1]))


def instructions(p_seg: np.ndarray, n_steps: int, n_seg: int, dt_s: float,
                 merge: bool = False) -> list[dict]:
    """The schedule as generator instructions: segment start, end and level."""
    p = np.asarray(p_seg, dtype=float).ravel()
    rows = [{"segment": k, "step_lo": lo, "step_hi": hi,
             "t_start_s": lo * float(dt_s), "t_end_s": hi * float(dt_s),
             "level": float(p[k])}
            for k, (lo, hi) in enumerate(segment_bounds(n_steps, n_seg))]
    if not merge:
        return rows
    out: list[dict] = []
    for r in rows:
        if out and out[-1]["level"] == r["level"]:
            out[-1] = dict(out[-1], step_hi=r["step_hi"], t_end_s=r["t_end_s"],
                           segment=out[-1]["segment"])
        else:
            out.append(dict(r))
    return out


# ---------------------------------------------------------------------------
# budget bookkeeping for the alternating block solve
# ---------------------------------------------------------------------------

def block_budget(n_evals: int, n_blocks: int) -> list[int]:
    """Split an evaluation budget across alternating blocks as evenly as
    possible, spending every evaluation and never silently capping."""
    n, b = int(n_evals), int(n_blocks)
    base, rem = divmod(max(n, 0), max(b, 1))
    return [base + (1 if i < rem else 0) for i in range(b)]
