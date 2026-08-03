#!/usr/bin/env python3
"""LEVEL 1 of the continuous-rotation campaign: the averaged-kernel solve.

Solve the dopant map against a forward whose radio-frequency heating is the
ROTATIONALLY AVERAGED kernel in the part frame (`adjoint2d.rot_kernel`). This
is the quasi-static limit of a turntable: fast rotation compared with the
thermal time constants, so every material point sees the angle-average of the
heating it would receive at each orientation. The approximation is not asserted
here; Level 2 measures its error against the real rotating engine.

Objective and stop convention, carried on every number:

    J_phi(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

with chi the part mask at zero degrees, which under continuous rotation is the
part frame and therefore the frame the nominal target lives in. t_stop = argmin
of J_phi over that arm's own trajectory on a 1500-step horizon (dt 0.5 s, 750 s)
with early truncation 250 steps past the running minimum; a minimum on the last
stored step is flagged HORIZON and makes that J a bound. The melted region for
intersection over union (IoU), growth and under-melt is phi >= 0.5. GRID 120.

Recipe: physical-length design filter at sigma = 1.5 cells (MANDATORY), box
[0, 1], L-BFGS-B on the finite-difference gated filtered averaged-kernel
gradient (`adjoint2d.gate_rot`). Conductivity channel only, which is the
deployment-safe one.

Run:
  ./.venv312/bin/python scripts/analysis/run_rot_avg_solve.py <shape> [step_deg] [n_evals]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import design_filter as df, energy_gate as eg, gradops   # noqa: E402
from adjoint2d import library_solve as lib                              # noqa: E402
from adjoint2d import printability as pq                                # noqa: E402
from adjoint2d import shape_objective as so                             # noqa: E402
from adjoint2d.pins import load_cfg                                     # noqa: E402
from adjoint2d.rot_frame import averaging_angles                        # noqa: E402
from adjoint2d.rot_kernel import AveragedKernel, quasistatic_numbers    # noqa: E402

SIGMA_CELLS = df.DEFAULT_SIGMA_CELLS
BOX = (0.0, 1.0)
N_STEPS = 1500
PATIENCE = lib.PATIENCE
N_EVALS = 16                 # the campaign's 40 forward-equivalent depth budget
STEP_DEG = 15.0              # the near-continuous turntable schedule's own step
SHAPES = ("T_shape", "L_shape", "cross", "star", "square")
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
OUT_MS = REPO / "fgm_solve_campaign/out_ms"
OUT_ROOT = REPO / "fgm_solve_campaign/out_rot"


def best_known_zero_map(shape: str, log) -> tuple[np.ndarray, dict]:
    """The lowest-J continuous zero-degree conductivity map on record."""
    best = None
    for arm, npz, key, jp in (("A1_cont", OUT_LIB / f"{shape}_maps.npz", "A1_cont",
                               OUT_LIB / f"{shape}.json"),
                              ("MS_cont", OUT_MS / f"{shape}_maps.npz", "MS_cont",
                               OUT_MS / f"{shape}.json")):
        if not (npz.exists() and jp.exists()):
            continue
        arms = json.loads(jp.read_text()).get("arms", {})
        if arm not in arms:
            continue
        J = float(arms[arm]["J"])
        if best is None or J < best[0]:
            best = (J, arm, npz, key, jp, float(arms[arm]["IoU"]))
    if best is None:
        raise FileNotFoundError(f"no stored zero-degree map for {shape}")
    J, arm, npz, key, jp, iou = best
    log(f"warm-start source: {arm} from {npz.name} (static J {J:.2f}, IoU {iou:.4f})")
    return np.asarray(np.load(npz)[key], dtype=float), {
        "arm": arm, "npz": str(npz), "key": key, "source_json": str(jp),
        "static_J_at_zero_deg": J, "static_IoU_at_zero_deg": iou}


def score(kern: AveragedKernel, s: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    case = kern.case0
    tr = kern.forward(s, keep_checkpoints=False, n_steps=N_STEPS,
                      shape_stop_patience=PATIENCE)
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    T_stop = tr.T_at_end(i)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["mean_rho_rel_part_at_end"] = float(tr.mean_rho_rel_part[tr.n_outer - 1])
    m["max_T_at_stop_c"] = float(np.max(T_stop))
    m["over_ceiling_250c"] = bool(m["max_T_at_stop_c"] > 250.0)
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_std_in_part"] = float(np.std(s[case.part_mask]))
    m["map_roughness"] = df.roughness_in_part(s, case.part_mask)
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    phi_stop, _ = so.phi_field(T_stop, case)
    jc = so.J_curve(tr, case)
    del tr
    return m, phi_stop, jc


def solve_start(kern, ops, v0, n_evals: int, log, name: str,
                sigma_cells: float | None = None):
    """`sigma_cells` defaults to the grid-120 width; pass it to hold the design
    filter at a fixed PHYSICAL length when the grid changes."""
    sigma = SIGMA_CELLS if sigma_cells is None else float(sigma_cells)
    case = kern.case0
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
    rows: list[dict] = []
    store: dict[int, np.ndarray] = {}

    def unpack(vec):
        v = np.ones(pm.shape)
        v.ravel()[idx] = vec
        return v

    def fun(vec):
        if len(rows) >= int(n_evals):
            raise StopIteration
        v = unpack(vec)
        s = df.apply_filter(v, pm, sigma)
        tr = kern.forward(s, keep_checkpoints=True, n_steps=N_STEPS,
                          shape_stop_patience=PATIENCE)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        g_s = kern.gradient(s, tr, {st.index: seed}, grad_ops=ops)
        g = df.filter_vjp(g_s, pm, sigma)
        rows.append({"eval_index": len(rows) + 1, "start": name, "J": float(J),
                     "t_stop_index": int(st.index), "t_stop_s": float(st.time_s),
                     "t_stop_at_horizon": bool(st.at_horizon),
                     "grad_norm": float(np.linalg.norm(g[pm]))})
        store[rows[-1]["eval_index"]] = v.copy()
        del tr
        return float(J), g.ravel()[idx].astype(float)

    v_start = np.clip(np.asarray(v0, dtype=float).ravel()[idx], BOX[0], BOX[1])
    t0 = time.perf_counter()
    try:
        minimize(fun, v_start, jac=True, method="L-BFGS-B", bounds=[BOX] * len(idx),
                 options={"maxiter": 10_000, "maxfun": 10_000,
                          "ftol": 1e-16, "gtol": 1e-16})
    except StopIteration:
        pass
    if rows:
        b = min(rows, key=lambda r: r["J"])
        log(f"  start/{name}: {len(rows)} evaluations, J {rows[0]['J']:.2f} -> "
            f"{b['J']:.2f}, {time.perf_counter() - t0:.0f} s")
    return rows, store


def main(shape: str, step_deg: float = STEP_DEG, n_evals: int = N_EVALS) -> dict:
    if shape not in SHAPES:
        raise ValueError(f"{shape!r} not in {SHAPES}")
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{shape}/rotavg] {msg}", flush=True)

    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    # A non-default averaging step gets its own file so the 15-degree
    # deliverable is never silently overwritten.
    sfx = "" if abs(float(step_deg) - STEP_DEG) < 1e-9 else f"_step{int(round(step_deg))}"
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    angles = averaging_angles(float(step_deg))
    kern = AveragedKernel.build(cfg, angles=angles)
    case = kern.case0
    ops = gradops.gradient_matrices(case.x, case.y)
    pm = case.part_mask
    log(f"kernel over {len(angles)} angles, step {step_deg} deg, "
        f"{int(pm.sum())} part cells")

    # the averaged kernel itself, uniform map, for the figures and the physics read
    Qa_u, Qb_u = kern.averaged_Q(np.ones(pm.shape))

    m_u, phi_u, jc_u = score(kern, np.ones(pm.shape))
    m_u["arm"] = "AVG_uniform"
    log(f"  uniform under the averaged kernel: J {m_u['J']:.2f}  IoU {m_u['IoU']:.4f}  "
        f"stop {m_u['t_stop_s']:.1f} s{' HORIZON' if m_u['t_stop_at_horizon'] else ''}")

    sat_warm, warm_meta = best_known_zero_map(shape, log)
    starts = {"cold": np.ones(pm.shape),
              "warm": np.where(pm, np.clip(sat_warm, BOX[0], BOX[1]), 1.0)}
    per_start = {}
    for name, v0 in starts.items():
        rows, store = solve_start(kern, ops, v0, n_evals, log, name)
        if not rows:
            log(f"  start {name}: NO EVALUATIONS, skipped loudly")
            continue
        b = min(rows, key=lambda r: r["J"])
        per_start[name] = {"rows": rows, "best_J": float(b["J"]),
                           "v": store[int(b["eval_index"])]}
    if not per_start:
        raise RuntimeError(f"{shape}: no start produced an evaluation")
    winner = min(per_start, key=lambda k: per_start[k]["best_J"])
    v_best = per_start[winner]["v"]
    s_cont = df.apply_filter(v_best, pm, SIGMA_CELLS)
    s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)

    m_c, phi_c, jc_c = score(kern, s_cont)
    m_c["arm"] = "AVG_cont"
    m_q, phi_q, jc_q = score(kern, s_q)
    m_q["arm"] = "AVG_4bpp"
    m_q.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm, bpp=4).items()})

    # the stored static zero-degree map, scored ON THE AVERAGED KERNEL, so the
    # cost of not re-solving for rotation is visible before any engine run
    s_static = np.where(pm, np.clip(sat_warm, BOX[0], BOX[1]), 1.0)
    m_s, phi_s, jc_s = score(kern, s_static)
    m_s["arm"] = "AVG_static0deg_map"

    for m in (m_c, m_q, m_s):
        log(f"  {m['arm']:20s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
            f"grow {m['bed_melt_pct_of_part']:5.2f}%  under {m['part_under_melt_pct']:5.2f}%  "
            f"rho {m['mean_rho_rel_part_at_stop']:.4f}  P {m['P_abs_W_per_m']:6.1f} W/m  "
            f"stop {m['t_stop_s']:6.1f} s{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
            f"maxT {m['max_T_at_stop_c']:.1f} C"
            f"{'  CEILING' if m['over_ceiling_250c'] else ''}  "
            f"Egate {'PASS' if m['energy_gate']['PASS'] else 'FAIL'}")

    np.savez_compressed(
        out_dir / f"{shape}_rotavg{sfx}_maps.npz",
        sat_cont=s_cont.astype(np.float32), sat_4bpp=s_q.astype(np.float32),
        sat_static0=s_static.astype(np.float32), v_best=v_best.astype(np.float32),
        part_mask=pm, Q_avg_uniform_A=Qa_u.astype(np.float32),
        Q_avg_uniform_B=Qb_u.astype(np.float32),
        phi_cont=phi_c.astype(np.float32), phi_4bpp=phi_q.astype(np.float32),
        phi_uniform=phi_u.astype(np.float32), phi_static0=phi_s.astype(np.float32),
        J_curve_cont=jc_c.astype(np.float32), J_curve_uniform=jc_u.astype(np.float32),
        J_curve_static0=jc_s.astype(np.float32), x=case.x, y=case.y)

    result = {
        "shape": shape, "config": str(cfg_path), "channel": "conductivity only",
        "kernel": "rotationally averaged in the part frame (quasi-static limit)",
        "angles_deg": [float(a) for a in angles], "n_angles": int(len(angles)),
        "step_deg": float(step_deg), "grid": 120, "box": list(BOX),
        "sigma_cells": float(SIGMA_CELLS), "n_steps": N_STEPS, "patience": PATIENCE,
        "n_gradient_evals_per_start": int(n_evals),
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "warm_start": warm_meta, "winner_start": winner,
        "start_best_J": {k: v["best_J"] for k, v in per_start.items()},
        "n_evals_by_start": {k: len(v["rows"]) for k, v in per_start.items()},
        "rows_by_start": {k: v["rows"] for k, v in per_start.items()},
        "quasistatic": quasistatic_numbers(case, rotation_period_s=float("nan")),
        "arms": {m["arm"]: m for m in (m_u, m_c, m_q, m_s)},
        "stop_convention": ("t_stop = argmin of J_phi over the arm's own trajectory; "
                            "1500-step horizon (750 s), patience 250; melted region "
                            "phi >= 0.5; grid 120"),
        "wall_s": time.perf_counter() - t0,
    }
    (out_dir / f"{shape}_rotavg{sfx}.json").write_text(json.dumps(result, indent=1, default=float))
    log(f"DONE winner start {winner}, AVG_cont J {m_c['J']:.2f}, "
        f"AVG_4bpp J {m_q['J']:.2f}, wall {result['wall_s']:.0f} s")
    return result


if __name__ == "__main__":
    main(sys.argv[1],
         float(sys.argv[2]) if len(sys.argv) > 2 else STEP_DEG,
         int(sys.argv[3]) if len(sys.argv) > 3 else N_EVALS)
