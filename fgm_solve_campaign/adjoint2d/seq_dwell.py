"""SEQUENTIAL dwell scheduling: the pure logic, with no physics in it.

WHY THIS IS A DIFFERENT ACTUATOR FROM `dwell.py`. The cycled dwell of
`dwell.py` runs repeated SHORT cycles through the kept positions. If the cycle
is short against the thermal times, every material point sees the dwell
weighted TIME AVERAGE of the heating, and the design collapses to a weight
vector on the simplex. Measured consequence, `DWELL_SCHEDULE_REPORT.md`
Section 1: given full freedom on the L_shape the cycled optimizer put all the
exposure at one position, that is, it degenerated to a STATIC arm.

A SEQUENTIAL schedule is not a time average. It holds ONE orientation long
enough to drive one limb through melt, then moves and holds another. Melting is
PATH DEPENDENT: the latent heat and the melt state are carried in the
temperature field, so a limb driven through melt in phase one is still hot and
still melted while phase two heats the other limb. The design variables are the
ORDER of the positions and the DURATIONS of the holds, not a weight vector.

THE PARAMETERIZATION, and why the durations are free rather than on a simplex.
A segment list (a_1, d_1), ..., (a_K, d_K) with d_k >= 0. The machine holds
a_K after the program ends, exactly as `dwell_march.program_step_positions`
already does, so the LAST duration is inert and the free variables are the K-1
switch times. The stop time is chosen afterwards as the argmin of the objective
along the trajectory, the same convention as every other arm in the campaign,
so total exposure is not a design variable here and no simplex constraint is
needed. That removes the softmax the cycled pass had to introduce and with it
the "a weight can only reach zero in the limit" caveat.

THE OVERLAP MATRIX is the whole of this module. An outer step of length dt can
straddle a switch, so the heating injected at step i is the EXACT time average
over that step

    Q_i = sum_k f_ik Q_(a_k),    f_ik = |[i dt, (i+1) dt) n [c_(k-1), c_k)| / dt

with c_k the cumulative switch times. `f` is continuous and piecewise linear in
the durations, with kinks only where a switch time crosses a control-step edge,
so the derivative below is exact almost everywhere and a finite difference at
an epsilon smaller than the distance to the nearest step edge sees no kink at
all. Nothing here is smoothed and no regularizer is introduced.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# the segment-to-step overlap matrix and its vector-Jacobian product
# ---------------------------------------------------------------------------

def switch_times(durations: Sequence[float]) -> np.ndarray:
    """Cumulative switch times c_1 .. c_K. c_K is never used (the hold)."""
    d = np.asarray(durations, dtype=float).ravel()
    if d.size == 0:
        raise ValueError("a sequential schedule needs at least one segment")
    if np.any(d < 0.0):
        raise ValueError(f"segment durations must be non-negative, got {d!r}")
    return np.cumsum(d)


def step_mix(durations: Sequence[float], dt_s: float, n_steps: int) -> np.ndarray:
    """`f[i, k]`, the fraction of outer step i spent in segment k.

    The last segment HOLDS past the end of the program, which is what a machine
    that has finished its program and not been told to stop would do and what
    keeps a horizon longer than the program well defined.
    """
    c = switch_times(durations)
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s!r}")
    n = int(n_steps)
    k = c.size
    t0 = np.arange(n, dtype=float) * dt
    t1 = t0 + dt
    lo = np.concatenate(([0.0], c[:-1]))
    hi = c.copy()
    hi[-1] = max(float(hi[-1]), n * dt) + dt        # the hold
    f = np.empty((n, k), dtype=float)
    for j in range(k):
        f[:, j] = np.clip(np.minimum(hi[j], t1) - np.maximum(lo[j], t0),
                          0.0, None) / dt
    return f


def mix_vjp(g_mix: np.ndarray, durations: Sequence[float], dt_s: float,
            n_steps: int) -> np.ndarray:
    """`dJ/d(durations)` given `dJ/df`.

    Moving switch time c_k by delta hands delta of exposure from segment k+1
    back to segment k, and it does so entirely inside the ONE control step that
    contains c_k. So dJ/dc_k = (g[i, k] - g[i, k+1]) / dt at that step, and the
    duration derivative is the cumulative sum of the switch derivatives.
    """
    g = np.asarray(g_mix, dtype=float)
    c = switch_times(durations)
    dt = float(dt_s)
    n = int(n_steps)
    k = c.size
    if g.shape != (n, k):
        raise ValueError(f"g_mix has shape {g.shape}, expected {(n, k)}")
    g_c = np.zeros(k)
    for j in range(k - 1):                       # c_K is the hold: no effect
        i = int(np.floor(c[j] / dt))
        if 0 <= i < n:
            g_c[j] = (g[i, j] - g[i, j + 1]) / dt
    # d(c_j)/d(d_m) = 1 for j >= m
    return np.cumsum(g_c[::-1])[::-1].copy()


def snap_durations(durations: Sequence[float], dt_s: float) -> np.ndarray:
    """Quantize hold durations to the control step the machine can express."""
    d = np.asarray(durations, dtype=float).ravel()
    return np.round(d / float(dt_s)) * float(dt_s)


# ---------------------------------------------------------------------------
# ordering, by enumeration
# ---------------------------------------------------------------------------

def segment_orders(candidate_idx: Sequence[int], n_segments: int) -> list[tuple]:
    """Every ORDERED choice of `n_segments` distinct candidate positions.

    Order is a design variable and it is solved by enumeration, not by a
    gradient: it is a discrete variable and for two to four segments the
    enumeration is cheap. Repeats are excluded because two ADJACENT segments at
    the same position are one longer segment, and a non-adjacent revisit is a
    strictly larger space that is named as a limit rather than searched.
    """
    return list(itertools.permutations([int(i) for i in candidate_idx],
                                       int(n_segments)))


# ---------------------------------------------------------------------------
# the machine-readable program
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequentialProgram:
    """An ordered turntable program: go there, hold that long, then move on."""

    moves: tuple[dict, ...]
    positions_deg: tuple[float, ...]
    durations_s: tuple[float, ...]
    horizon_s: float
    dt_s: float

    def as_json(self) -> dict:
        return {
            "actuator": ("indexed turntable, SEQUENTIAL holds, constant "
                         "radio-frequency power"),
            "schedule_kind": "sequential",
            "control_step_s": self.dt_s,
            "horizon_s": self.horizon_s,
            "positions_deg": list(self.positions_deg),
            "durations_s": list(self.durations_s),
            "n_moves": len(self.moves),
            "moves": [dict(m) for m in self.moves],
        }


def sequential_program(angles_deg: Sequence[float], seg_idx: Sequence[int],
                       durations: Sequence[float], dt_s: float,
                       horizon_s: float, snap: bool = False
                       ) -> SequentialProgram:
    """Turn (position order, hold durations) into the list the machine runs."""
    ang = np.asarray(angles_deg, dtype=float).ravel()
    idx = [int(i) for i in seg_idx]
    d = np.asarray(durations, dtype=float).ravel()
    if len(idx) != d.size:
        raise ValueError(f"{len(idx)} segments against {d.size} durations")
    if snap:
        d = snap_durations(d, dt_s)
    # merge adjacent repeats: two holds at the same position are one hold
    m_idx: list[int] = []
    m_dur: list[float] = []
    for j, dd in zip(idx, d):
        if m_idx and m_idx[-1] == j:
            m_dur[-1] += float(dd)
        else:
            m_idx.append(j)
            m_dur.append(float(dd))
    moves: list[dict] = []
    t = 0.0
    for n, (j, dd) in enumerate(zip(m_idx, m_dur)):
        if t >= float(horizon_s) - 1e-9:
            break
        last = (n == len(m_idx) - 1)
        hold = (float(horizon_s) - t) if last else min(float(dd),
                                                       float(horizon_s) - t)
        moves.append({"position_deg": float(ang[j]), "dwell_s": float(hold),
                      "move_at_s": float(t)})
        t += hold
    if moves and t < float(horizon_s) - 1e-9:
        moves[-1]["dwell_s"] += float(horizon_s) - t
    return SequentialProgram(
        moves=tuple(moves),
        positions_deg=tuple(float(ang[j]) for j in m_idx),
        durations_s=tuple(float(x) for x in m_dur),
        horizon_s=float(horizon_s), dt_s=float(dt_s))
