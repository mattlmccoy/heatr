"""Prefix-sharing screen over SEQUENTIAL two-segment schedules.

WHY A SEPARATE LEAN MARCH. Every two-segment schedule that starts at the same
orientation shares its whole first phase. Marching that phase once and
branching from stored states at each candidate switch time turns an
`n_switch x n_second_angle` grid from that many full marches into ONE full
march plus the tails, which is what makes a directed scan affordable inside the
stated budget.

WHAT IT DOES AND DOES NOT MEASURE. It records the objective curve, the melt
field at the argmin and the peak temperature. It does NOT run the energy
bookkeeping, so no arm is SCORED here: the screen only chooses which schedules
are worth a full `seq_dwell_march.sequential_forward` run, and every number
that reaches the report comes from that full run with all the standing gates
on. Switch times on this grid are exact multiples of the control step, so the
lean march and the full march agree bit for bit (the degeneracy gate in
`tests/test_seq_dwell_march.py`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from adjoint2d import forward as fwd
from adjoint2d import shape_objective as so
from adjoint2d.dwell_kernel import DwellKernel


def _j_of(T: np.ndarray, case, chi: np.ndarray) -> float:
    p = case.pins
    d = np.clip((T - p.t_pc_c) / p.dt_pc_c + 0.5, 0.0, 1.0) - chi
    return float(np.sum(d * d))


def _leg(kern: DwellKernel, angle_idx: int, T: np.ndarray, rho: np.ndarray,
         i0: int, i1: int, snapshot_at: Sequence[int] = ()) -> dict:
    """March steps [i0, i1) at ONE position. Returns curve, end state, snapshots."""
    case = kern.case0
    p = case.pins
    ui = p.update_interval
    chi = case.part_mask.astype(float)
    want = set(int(i) for i in snapshot_at)
    js: list[float] = []
    snaps: dict[int, tuple] = {}
    best = (np.inf, i0, None)
    for it in range(int(i0), int(i1)):
        if it in want:
            # the state to branch from, AND the running best over the prefix,
            # so a branch whose argmin lands before its switch never has to be
            # re-marched to recover its melt field
            snaps[it] = (T.copy(), rho.copy(), best)
        use_b = ui > 0 and it >= ui
        Q = kern._Qk_b[angle_idx] if use_b else kern._Qk_a[angle_idx]
        for _ in range(p.n_substeps):
            T, rho, _phi, _c = fwd.substep(T, rho, Q, case, keep_cache=False)
        j = _j_of(T, case, chi)
        js.append(j)
        if j < best[0]:
            best = (j, it, T.copy())
    if int(i1) in want:
        snaps[int(i1)] = (T.copy(), rho.copy(), best)
    return {"J": np.asarray(js), "T": T, "rho": rho, "snaps": snaps, "best": best}


def initial_state(kern: DwellKernel) -> tuple[np.ndarray, np.ndarray]:
    case = kern.case0
    p = case.pins
    T = np.full(case.part_mask.shape, p.ambient_c, dtype=float)
    rho = np.zeros(case.part_mask.shape, dtype=float)
    rho[case.part_mask] = p.rho_rel_init
    return T, rho


def screen_two_segment(kern: DwellKernel, a1: int, a2_list: Sequence[int],
                       switch_steps: Sequence[int], n_steps: int,
                       log=print) -> list[dict]:
    """Every (switch time, second position) branch off ONE first-phase march."""
    case = kern.case0
    chi = case.part_mask.astype(float)
    T0, rho0 = initial_state(kern)
    sw = sorted(int(x) for x in switch_steps)
    head = _leg(kern, a1, T0, rho0, 0, int(n_steps), snapshot_at=sw)
    rows = []
    for i_sw in sw:
        Ts, rs, head_best = head["snaps"][i_sw]
        for a2 in a2_list:
            tail = _leg(kern, int(a2), Ts.copy(), rs.copy(), i_sw, int(n_steps))
            curve = np.concatenate([head["J"][:i_sw], tail["J"]])
            i_best = int(np.argmin(curve))
            T_best = head_best[2] if i_best < i_sw else tail["best"][2]
            i_post = int(tail["best"][1])
            rows.append({
                "a1": int(a1), "a2": int(a2), "switch_step": i_sw,
                "switch_s": i_sw * case.pins.dt,
                "J": float(curve[i_best]), "stop_step": i_best,
                "stop_s": (i_best + 1) * case.pins.dt,
                "at_horizon": bool(i_best == len(curve) - 1),
                "J_post_switch": float(curve[i_post]),
                "stop_post_switch_s": (i_post + 1) * case.pins.dt,
                "stop_is_in_phase_2": bool(i_best >= i_sw),
                "T_at_stop": T_best, "J_curve": curve,
            })
            log(f"    a1 {a1} -> a2 {a2}  switch {i_sw * case.pins.dt:6.1f} s : "
                f"J {curve[i_best]:8.2f} at {(i_best + 1) * case.pins.dt:6.1f} s"
                f"{'  (phase 2)' if i_best >= i_sw else '  (phase 1, COLLAPSED)'}"
                f"  J_post {curve[i_post]:8.2f}")
    return rows


def screen_three_segment(kern: DwellKernel, a1: int, sw1_steps: Sequence[int],
                         a2: int, sw2_steps: Sequence[int],
                         a3_list: Sequence[int], n_steps: int,
                         log=print) -> list[dict]:
    """Three segments, sharing BOTH the first phase and each second phase."""
    case = kern.case0
    dt = case.pins.dt
    T0, rho0 = initial_state(kern)
    sw1 = sorted(int(x) for x in sw1_steps)
    sw2 = sorted(int(x) for x in sw2_steps)
    head = _leg(kern, int(a1), T0, rho0, 0, max(sw1), snapshot_at=sw1)
    rows = []
    for i1 in sw1:
        Ts, rs, best1 = head["snaps"][i1]
        want2 = [i for i in sw2 if i > i1]
        if not want2:
            continue
        mid = _leg(kern, int(a2), Ts.copy(), rs.copy(), i1, max(want2),
                   snapshot_at=want2)
        for i2 in want2:
            Tm, rm, best2 = mid["snaps"][i2]
            for a3 in a3_list:
                tail = _leg(kern, int(a3), Tm.copy(), rm.copy(), i2, int(n_steps))
                curve = np.concatenate([head["J"][:i1], mid["J"][:i2 - i1],
                                        tail["J"]])
                cands = [best1, best2, tail["best"]]
                cands = [c for c in cands if c[2] is not None]
                jb, ib, Tb = min(cands, key=lambda c: c[0])
                rows.append({
                    "a1": int(a1), "a2": int(a2), "a3": int(a3),
                    "switch1_step": i1, "switch2_step": i2,
                    "switch1_s": i1 * dt, "switch2_s": i2 * dt,
                    "J": float(jb), "stop_step": int(ib),
                    "stop_s": (int(ib) + 1) * dt,
                    "at_horizon": bool(int(ib) == len(curve) - 1),
                    "J_post_switch": float(tail["best"][0]),
                    "stop_post_switch_s": (int(tail["best"][1]) + 1) * dt,
                    "stop_is_in_phase_3": bool(int(ib) >= i2),
                    "T_at_stop": Tb, "J_curve": curve})
                log(f"    {a1} -> {a2} -> {a3}  switches {i1 * dt:6.1f}, "
                    f"{i2 * dt:6.1f} s : J {jb:8.2f} at {(int(ib) + 1) * dt:6.1f} s"
                    f"{'  (phase 3)' if int(ib) >= i2 else ''}")
    return rows


def static_curve(kern: DwellKernel, angle_idx: int, n_steps: int) -> dict:
    T0, rho0 = initial_state(kern)
    return _leg(kern, int(angle_idx), T0, rho0, 0, int(n_steps))
