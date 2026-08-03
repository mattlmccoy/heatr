"""Finite-difference gate for the SEQUENTIAL dwell gradients.

Nothing in the sequential campaign may be optimized before this passes. Three
layers, so a failure localizes instead of being guessed at:

  Q0  the MAP gradient through the sequential march, UNFILTERED. At a
      one-segment schedule the sequential march IS the single-position march,
      so this layer re-gates an already gated chain inside the new code path.
      Run at a genuine multi-segment schedule, so it also gates the per-step,
      per-angle accumulation that the sequential march introduces.
  Q1  dJ/d(durations), the new object. Probes are single segments and a random
      direction over the free segments. The LAST duration is excluded: the
      machine holds the last position past the end of the program, so that
      derivative is identically zero and its relative error would be 0/0.
  Q2  the MAP gradient at the same schedule WITH the 1.0 mm physical-length
      design filter, which is the gradient the co-solved arm uses.

READ STATE. Fixed read index, taken as the argmin of J_phi on the base run.
`DWELL_SCHEDULE_REPORT.md` Section 3.4 established that a read state deep into
full melt is NOT Lipschitz (the densification driving term is `(1 - phi)**0.8`)
and that gating there is meaningless; the base run here uses the production
horizon and the interior argmin, and `at_horizon` is recorded.

SWITCH TIMES AND THE ONE KINK. The overlap matrix is piecewise linear in the
durations with kinks exactly where a switch time crosses a control-step edge.
The gate durations are chosen so that every switch sits strictly inside a step
and at least 0.01 s from either edge, which is four orders of magnitude larger
than the biggest epsilon in the sweep, so no probe crosses a kink. That is a
stated property of the gate point, checked in `_switch_clearance`.

Pass standard: 1e-6 preferred, 1e-5 is the campaign's documented subgradient
standard (`FROZEN_CONVENTIONS_2D.md` Section 5 item 5).

Run:
  ./.venv312/bin/python -m adjoint2d.gate_seq_dwell <shape> <out.json> [n_steps]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import design_filter as df, gradops
from . import library_solve as lib
from . import seq_dwell as sq
from . import seq_dwell_march as sqm
from . import shape_objective as so
from . import topopt
from .dwell_kernel import DwellKernel
from .gate_dwell import _finish, _log, _sweep
from .gate_ms import gradient_direction
from .gate_rho import _probe_dirs, default_v
from .pins import load_cfg

N_STEPS_GATE = 900
# Only the angles the gate schedule actually visits are built, because every
# objective evaluation re-solves the electro-quasi-static problem at EVERY
# angle in the kernel and the sweep makes hundreds of those calls.
CANDIDATE_ANGLES = np.array([0.0, 90.0, 135.0])
# A deliberately non-degenerate three-segment schedule: three DIFFERENT
# positions, every switch strictly inside a control step.
GATE_SEGMENTS = (0, 1, 2)                       # 0, 90 and 135 degrees
GATE_DURATIONS = (123.3, 141.15, 200.0)         # switches at 123.3 s and 264.45 s


def _switch_clearance(durations, dt: float) -> float:
    """Distance from the nearest switch time to the nearest control-step edge."""
    c = sq.switch_times(durations)[:-1]
    r = np.mod(c, dt)
    return float(np.min(np.minimum(r, dt - r))) if r.size else float("inf")


def _forward(kern, s, durations, n_steps, keep=False):
    kern.averaged_Q(s)
    return sqm.sequential_forward(kern, GATE_SEGMENTS, durations, n_steps,
                                  keep_checkpoints=keep)


# ---------------------------------------------------------------------------
# Q0 and Q2, the map layers
# ---------------------------------------------------------------------------

def gate_map(kern: DwellKernel, name: str, ops, v0, durations, read_index: int,
             sigma_cells: float) -> dict:
    case = kern.case0
    pm = case.part_mask
    filt = float(sigma_cells) > 0.0

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells) if filt else np.where(pm, v, 1.0)

    def J_of(v):
        tr = _forward(kern, to_map(v), durations, N_STEPS_GATE)
        return so.shape_J_and_seed(tr.T_at_end(min(read_index, tr.n_outer - 1)), case)[0]

    s0 = to_map(v0)
    tr0 = _forward(kern, s0, durations, N_STEPS_GATE, keep=True)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    g_s, _gd = sqm.sequential_gradients(kern, s0, tr0, {i0: seed}, grad_ops=ops)
    g = df.filter_vjp(g_s, pm, sigma_cells) if filt else np.where(pm, g_s, 0.0)

    out = {"layer": name, "variable": "dopant map v", "J0": float(J0),
           "read_index": int(i0), "sigma_cells": float(sigma_cells),
           "segments": [int(x) for x in GATE_SEGMENTS],
           "durations_s": [float(x) for x in durations],
           "grad_norm": float(np.linalg.norm(g[pm])), "probes": {}}
    probes = list(_probe_dirs(pm, g, sigma_cells=sigma_cells))
    probes.append(("gradient_direction", gradient_direction(pm, g), None))
    for pname, d, cell in probes:
        out["probes"][pname] = _sweep(J_of, v0, d, float(np.sum(g * d)))
        out["probes"][pname]["cell"] = None if cell is None else [int(c) for c in cell]
    return _finish(out)


# ---------------------------------------------------------------------------
# Q1, the duration layer
# ---------------------------------------------------------------------------

def gate_durations(kern: DwellKernel, name: str, s0, durations,
                   read_index: int) -> dict:
    case = kern.case0
    d0 = np.asarray(durations, dtype=float)
    n_free = d0.size - 1                     # the last hold is inert

    def J_of(d):
        tr = _forward(kern, s0, d, N_STEPS_GATE)
        return so.shape_J_and_seed(tr.T_at_end(min(read_index, tr.n_outer - 1)), case)[0]

    tr0 = _forward(kern, s0, d0, N_STEPS_GATE, keep=True)
    i0 = min(int(read_index), tr0.n_outer - 1)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(i0), case)
    _gs, gd = sqm.sequential_gradients(kern, s0, tr0, {i0: seed})

    rng = np.random.default_rng(17)
    e_max = np.zeros(d0.size)
    e_max[int(np.argmax(np.abs(gd[:n_free])))] = 1.0
    e_rnd = np.zeros(d0.size)
    e_rnd[int(rng.integers(n_free))] = 1.0
    u = np.zeros(d0.size)
    u[:n_free] = rng.standard_normal(n_free)
    gdir = np.zeros(d0.size)
    gdir[:n_free] = gd[:n_free]
    probes = (("max_sensitivity_duration", e_max),
              ("random_duration", e_rnd),
              ("random_direction", u / np.linalg.norm(u)),
              ("gradient_direction", gdir / max(np.linalg.norm(gdir), 1e-30)))
    out = {"layer": name, "variable": "segment durations d", "J0": float(J0),
           "read_index": int(i0),
           "segments": [int(x) for x in GATE_SEGMENTS],
           "durations_s": [float(x) for x in d0],
           "switch_clearance_s": _switch_clearance(d0, case.pins.dt),
           "dJ_dd": [float(x) for x in gd],
           "grad_norm": float(np.linalg.norm(gd[:n_free])), "probes": {}}
    for pname, d in probes:
        out["probes"][pname] = _sweep(J_of, d0, d, float(np.dot(gd, d)))
    out["dJ_dd_last_is_zero"] = bool(abs(float(gd[-1])) < 1e-30)
    return _finish(out)


# ---------------------------------------------------------------------------

def stop_index_stability(kern, s0, durations, eps: float = 1e-3) -> dict:
    case = kern.case0
    d0 = np.asarray(durations, dtype=float)

    def stop(d):
        return so.optimal_stop(_forward(kern, s0, d, N_STEPS_GATE), case).index

    base = stop(d0)
    out = {"eps": float(eps), "base_index": int(base), "moved": False, "probes": {}}
    for k in range(d0.size - 1):
        dp, dm = d0.copy(), d0.copy()
        dp[k] += eps
        dm[k] -= eps
        ip, im = stop(dp), stop(dm)
        out["probes"][f"duration_{k}"] = {
            "plus": int(ip), "minus": int(im),
            "moved": bool(ip != base or im != base)}
        out["moved"] = out["moved"] or out["probes"][f"duration_{k}"]["moved"]
    return out


def main(shape: str, out_path: str, n_steps: int = N_STEPS_GATE) -> dict:
    t0 = time.perf_counter()
    global N_STEPS_GATE
    N_STEPS_GATE = int(n_steps)
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    kern = DwellKernel.build(cfg, angles=CANDIDATE_ANGLES)
    case = kern.case0
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)
    kern.set_weights(np.full(CANDIDATE_ANGLES.size, 1.0 / CANDIDATE_ANGLES.size))

    v0 = default_v(case)
    s0 = df.apply_filter(v0, pm, sigma_cells)
    d0 = np.asarray(GATE_DURATIONS, dtype=float)

    base = _forward(kern, np.where(pm, v0, 1.0), d0, N_STEPS_GATE)
    st = so.optimal_stop(base, case)

    res = {"shape": shape, "config": str(cfg_path),
           "objective": "J_phi = sum over the WHOLE domain of (phi - chi_part)^2",
           "kernel": "SEQUENTIAL hold schedule, part frame, no interpolation",
           "angles_deg": [float(a) for a in CANDIDATE_ANGLES],
           "segments_angle_index": [int(x) for x in GATE_SEGMENTS],
           "segments_angle_deg": [float(CANDIDATE_ANGLES[i]) for i in GATE_SEGMENTS],
           "durations_s": [float(x) for x in d0],
           "switch_clearance_s": _switch_clearance(d0, case.pins.dt),
           "n_steps_gate": N_STEPS_GATE,
           "filter_radius_m": topopt.FILTER_RADIUS_M,
           "sigma_cells": float(sigma_cells),
           "base": {"read_index": int(st.index), "read_time_s": st.time_s,
                    "J_phi": st.J, "at_horizon": st.at_horizon},
           "layers": []}
    print(f"[{shape}] gate point: segments "
          f"{[float(CANDIDATE_ANGLES[i]) for i in GATE_SEGMENTS]} deg, durations "
          f"{list(d0)} s, switch clearance "
          f"{res['switch_clearance_s']:.4f} s, read index {st.index} "
          f"(horizon = {st.at_horizon})", flush=True)

    r = gate_map(kern, "Q0_map_sequential_unfiltered", ops, v0, d0, st.index, 0.0)
    res["layers"].append(r)
    _log(r)
    r = gate_durations(kern, "Q1_segment_durations", s0, d0, st.index)
    res["layers"].append(r)
    _log(r)
    r = gate_map(kern, "Q2_map_sequential_filtered", ops, v0, d0, st.index,
                 float(sigma_cells))
    res["layers"].append(r)
    _log(r)

    res["stop_index_stability"] = stop_index_stability(kern, s0, d0)
    print(f"stop index stability: base {res['stop_index_stability']['base_index']}, "
          f"moved = {res['stop_index_stability']['moved']}", flush=True)

    degenerate = [l["layer"] for l in res["layers"] if l["grad_norm"] == 0.0]
    res["degenerate_layers"] = degenerate
    res["ALL_GATES_PASS"] = (not degenerate) and all(l["PASS"] for l in res["layers"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = (not degenerate) and all(
        l["PASS_subgradient"] for l in res["layers"])
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] ALL_GATES_PASS at 1e-6 = {res['ALL_GATES_PASS']}, at the 1e-5 "
          f"subgradient standard = {res['ALL_GATES_PASS_SUBGRADIENT']}, "
          f"wall {res['wall_s']:.1f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else N_STEPS_GATE)
