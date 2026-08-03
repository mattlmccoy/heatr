"""S2 Task 2: the electrode-gauge question, and the plumbing to test it.

THE QUESTION. heatr3d.solve_eqs_3d puts its Dirichlet rows on the CELL CENTRES
of the first and last y layers, so the imposed potential drop spans (n-1)h and
not L. At a fixed physical chamber that gap is GRID DEPENDENT, so the imposed
FIELD is v_lo * n / ((n-1) L) -- it drifts with n and only reaches v_lo/L in
the limit. Recorded as a FINDING in test_heatr3d_s1.py
(test_eqs_uniform_medium_is_parallel_plate): "the effective plate gap is
(n-1)h, not L ... the bias is grid dependent (0.5 % at n=200) ... it is not
corrected here".

THE TWO ARMS.
  cell_centred_current  the shipped convention, v_lo imposed across (n-1)h
  face_gauge            v_lo rescaled by (n-1)/n so the imposed field is
                        exactly v_lo/L on every grid

THE OBSERVABLE, and why the obvious one does not work. Total absorbed power
CANNOT discriminate: compute_qrf_3d renormalizes Q to
power_density_w_per_m3 * V_doped, so the total is pinned by construction and is
identical under both gauges. The discriminating observable is the RAW integral
0.5 * sigma * |E|^2 dV BEFORE that renormalization, which scales as
(v / gap)^2 and therefore should drift like (n/(n-1))^2 under the shipped gauge
and be flat under the face gauge.

NOTHING HERE EDITS heatr3d.py. The face gauge is applied by rescaling v_lo in a
Params copy, which is exactly equivalent to moving the electrodes for a
Laplace problem (the solve is linear in the drive). If the decision goes
against the shipped convention, the flag-gated option lands in heatr3d.py in a
separate step and the DEFAULT FLIP GOES TO MATT -- the EQS-02 precedent.
"""
from __future__ import annotations

import dataclasses

import numpy as np

import heatr3d

ARMS = ("cell_centred_current", "face_gauge")


def gauge_params(p: heatr3d.Params, n: int, arm: str) -> heatr3d.Params:
    """Params for one gauge arm. `cell_centred_current` returns the input
    UNCHANGED, so the flag-off path is bit-identical by construction."""
    if arm not in ARMS:
        raise ValueError(f"unknown gauge arm {arm!r}; expected one of {ARMS}")
    if arm == "cell_centred_current":
        return p
    return dataclasses.replace(p, v_lo=p.v_lo * (n - 1) / n)


def imposed_field_v_per_m(p: heatr3d.Params, grid: heatr3d.Grid,
                          arm: str) -> float:
    """The field the arm actually imposes across the cell-centred gap."""
    q = gauge_params(p, grid.n, arm)
    return float(q.v_lo / ((grid.n - 1) * grid.h))


def raw_absorbed_power_w(V: np.ndarray, gamma: np.ndarray, grid: heatr3d.Grid,
                         part: np.ndarray) -> float:
    """integral 0.5 * Re(gamma) * |E|^2 dV over the part, BEFORE heatr3d's
    fixed-power renormalization, using heatr3d's own masked-gradient stencil so
    the number is the corrected-default convention and not a second one."""
    Ex, Ey, Ez = heatr3d._masked_grad_3d(V, part, grid.h)
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey) + Ez * np.conj(Ez))
    q = 0.5 * np.real(gamma * e2)
    q = np.clip(np.nan_to_num(q), 0.0, None)
    q[~part] = 0.0
    return float(q.sum() * grid.dV)


def run_arm(shape_fn, n: int, arm: str, p: heatr3d.Params | None = None) -> dict:
    """One EQS-only solve for one gauge arm at one grid."""
    import time
    p = p or heatr3d.Params()
    grid = heatr3d.Grid(n=n)
    part = shape_fn(grid)
    q = gauge_params(p, n, arm)
    gamma = heatr3d.build_gamma(part, q)
    t0 = time.perf_counter()
    V = heatr3d.solve_eqs_3d(gamma, grid, q)
    wall = time.perf_counter() - t0
    raw = raw_absorbed_power_w(V, gamma, grid, part)
    v_part = float(part.sum() * grid.dV)
    return {"arm": arm, "n": n, "h_m": grid.h,
            "v_lo_applied": float(q.v_lo),
            "gap_m": float((n - 1) * grid.h),
            "imposed_field_v_per_m": imposed_field_v_per_m(p, grid, arm),
            "raw_absorbed_power_w": raw,
            "raw_power_density_w_per_m3": raw / v_part,
            "part_volume_m3": v_part,
            "n_voxels_in_part": int(part.sum()),
            "wall_eqs_s": wall}


def decide(grids=(48, 64, 96), shape: str = "circle") -> dict:
    """Run both arms across the grid ladder and apply the PRE-REGISTERED
    decision rule: the gauge whose raw pre-renormalization absorbed power is
    grid-invariant wins."""
    import json
    from pathlib import Path

    from heatr3d_s2 import harness

    def shape_fn(grid):
        return harness.make_part(grid, shape)

    runs = {a: [run_arm(shape_fn, n, a) for n in grids] for a in ARMS}
    summary = {}
    for a, rs in runs.items():
        d = [r["raw_power_density_w_per_m3"] for r in rs]
        drift = (max(d) - min(d)) / min(d)
        # successive relative changes, coarse -> fine
        succ = [abs(b - a2) / a2 for a2, b in zip(d, d[1:])]
        summary[a] = {"raw_power_density_w_per_m3": d,
                      "total_drift_rel": float(drift),
                      "successive_rel_changes": [float(s) for s in succ],
                      "finest_pair_rel_change": float(succ[-1]) if succ else None}
    winner = min(summary, key=lambda a: summary[a]["total_drift_rel"])
    predicted = [((n / (n - 1)) ** 2) for n in grids]
    pred_drift = (max(predicted) - min(predicted)) / min(predicted)
    doc = {
        "what": "S2 Task 2 electrode-gauge decision",
        "shape": shape, "grids": list(grids), "eqs_only": True,
        "observable": "raw absorbed power density BEFORE heatr3d's fixed-power "
                      "renormalization",
        "why_not_total_power": "compute_qrf_3d pins the total by construction, "
                               "so it is identical under both arms and cannot "
                               "discriminate",
        "decision_rule": "smallest grid drift of the observable wins",
        "runs": runs, "summary": summary, "winner": winner,
        "shipped_arm": "cell_centred_current",
        "winner_is_the_shipped_arm": winner == "cell_centred_current",
        "predicted_cell_centred_drift_rel": float(pred_drift),
        "predicted_from": "(n/(n-1))^2, the ratio of the imposed field squared "
                          "at the cell-centred gap (n-1)h against the face gap L",
        "default_flip": "NOT PERFORMED. If the winner is not the shipped arm, "
                        "the flag-gated option and the default flip go to Matt "
                        "with this evidence -- the EQS-02 precedent. The "
                        "executing agent never flips a default.",
    }
    out = Path(__file__).resolve().parent / "results" / "gauge_decision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    return doc
