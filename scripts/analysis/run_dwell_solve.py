#!/usr/bin/env python3
"""ASYMMETRIC DWELL SCHEDULING: solve the dopant map and the dwell times jointly.

THE ACTUATOR. A turntable that can be commanded to a fixed set of INDEXED
POSITIONS and held at each one. Radio-frequency generator power is constant.
The design variables are the fraction of the exposure spent at each position,
solved jointly with the dopant map.

THE CANDIDATE SET is the eight multiples of 45 degrees, the SAME set for every
shape. This is wider than a symmetry-reduced set on purpose. Reducing the set
by the part's symmetry would only be exact if the dopant map carried that
symmetry too, and the map is free (the argument
`rot_frame.averaging_angles` already makes for the averaging set). Keeping the
full set also makes the regression control sharper: on a four-fold symmetric
shape the solver is free to put dwell on the 45-degree positions and the
prediction is that it does not.

THE EXECUTION MODEL. Repeated short cycles through the kept positions,
allocating time within each cycle in proportion to the dwell fractions, at a
stated CYCLE TIME. In the limit of a short cycle the heating a material point
sees is the dwell-weighted angle average, which is what the solve optimizes
(`adjoint2d.dwell_kernel`), and the finite-cycle error is MEASURED against a
time-resolved execution of the actual program (`adjoint2d.dwell_march`).

Objective and stop convention, carried on every number:

    J_phi(s, w, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi(x))^2

with chi the part mask in the PART frame, which is the frame the design lives
in and the frame the part never leaves. t_stop = argmin of J_phi over that
arm's own trajectory on a 1500-step horizon (dt 0.5 s, 750 s), early truncation
250 steps past the running minimum; a minimum on the last stored step is
flagged HORIZON and makes that J a BOUND. Melted region phi >= 0.5. GRID 120.

Recipe: filter radius 1.0 mm as a PHYSICAL length (FROZEN_CONVENTIONS_2D.md
Section 1.2), box [0, 1] on the map, softmax on the dwell fractions, L-BFGS-B
in ALTERNATING blocks on the finite-difference gated gradients
(`adjoint2d.gate_dwell`). Conductivity channel only.

Run:
  ./.venv312/bin/python scripts/analysis/run_dwell_solve.py <shape> [n_evals]
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

from adjoint2d import design_filter as df, dwell, energy_gate as eg    # noqa: E402
from adjoint2d import dwell_march as dmarch, gradops                   # noqa: E402
from adjoint2d import library_solve as lib, printability as pq         # noqa: E402
from adjoint2d import shape_objective as so, topopt                    # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                         # noqa: E402
from adjoint2d.pins import load_cfg                                    # noqa: E402

CANDIDATE_ANGLES = np.arange(8, dtype=float) * 45.0
BOX = (0.0, 1.0)
N_STEPS = 1500
DT_S = 0.5
PATIENCE = lib.PATIENCE
N_EVALS = 16                    # the campaign's 40 forward-equivalent budget
CYCLE_TIME_S = 20.0             # 40 control steps; 37.5 cycles over the horizon
CYCLE_SWEEP_S = (10.0, 20.0, 40.0, 80.0, 160.0)
SHAPES = ("cross", "square", "T_shape", "L_shape")
OUT = REPO / "fgm_solve_campaign/out_dwell"
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
OUT_MS = REPO / "fgm_solve_campaign/out_ms"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(tr, case, s, w) -> tuple[dict, np.ndarray, np.ndarray]:
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    T_stop = tr.T_at_end(i)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["mean_rho_rel_part_at_end"] = float(tr.mean_rho_rel_part[tr.n_outer - 1])
    m["max_T_at_stop_c"] = float(np.max(T_stop))
    m["max_T_part_at_stop_c"] = float(np.max(T_stop[case.part_mask]))
    m["over_ceiling_250c"] = bool(m["max_T_at_stop_c"] > 250.0)
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["n_outer"] = tr.n_outer
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["map_roughness"] = df.roughness_in_part(s, case.part_mask)
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    m["dwell_weights"] = [float(x) for x in w]
    m.update({f"dwell_{k}": v for k, v in dwell.dwell_asymmetry(w).items()})
    phi_stop, _ = so.phi_field(T_stop, case)
    jc = so.J_curve(tr, case)
    return m, phi_stop, jc


def log_row(log, m):
    log(f"  {m['arm']:22s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
        f"grow {m['bed_melt_pct_of_part']:5.2f}%  under {m['part_under_melt_pct']:5.2f}%  "
        f"rho {m['mean_rho_rel_part_at_stop']:.4f}  P {m['P_abs_W_per_m']:6.1f} W/m  "
        f"stop {m['t_stop_s']:6.1f} s{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
        f"maxT {m['max_T_at_stop_c']:6.1f} C{'  CEILING' if m['over_ceiling_250c'] else ''}  "
        f"Egate {'PASS' if m['energy_gate']['PASS'] else 'FAIL'}  "
        f"asym {m['dwell_max_over_equal']:.2f}x")


# ---------------------------------------------------------------------------
# the two blocks
# ---------------------------------------------------------------------------

def run_forward(kern, s, keep=True):
    return kern.forward(s, keep_checkpoints=keep, n_steps=N_STEPS,
                        shape_stop_patience=PATIENCE)


def eval_point(kern, ops, v, z, pm, sigma_cells):
    """One forward and ONE reverse march; both gradients come out of it."""
    kern.set_weights(dwell.softmax_weights(z))
    s = df.apply_filter(v, pm, sigma_cells)
    tr = run_forward(kern, s)
    st = so.optimal_stop(tr, kern.case0)
    J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), kern.case0)
    g_s, g_w = kern.both_gradients(s, tr, {st.index: seed}, grad_ops=ops)
    g_v = df.filter_vjp(g_s, pm, sigma_cells)
    g_z = dwell.softmax_vjp(dwell.softmax_weights(z), g_w)
    del tr
    return float(J), g_v, g_z, int(st.index), bool(st.at_horizon)


def block(kern, ops, state, pm, sigma_cells, which: str, n_evals: int,
          rows: list, log) -> None:
    """One alternating block: move only `which` ('map' or 'dwell')."""
    idx = np.flatnonzero(pm.ravel())
    n0 = len(rows)

    def fun(x):
        if len(rows) - n0 >= int(n_evals):
            raise StopIteration
        if which == "map":
            v = np.ones(pm.shape)
            v.ravel()[idx] = x
            z = state["z"]
        else:
            v = state["v"]
            z = np.asarray(x, dtype=float)
        J, g_v, g_z, si, hz = eval_point(kern, ops, v, z, pm, sigma_cells)
        rows.append({"eval_index": len(rows) + 1, "block": which, "J": J,
                     "t_stop_index": si, "t_stop_at_horizon": hz,
                     "v": v.copy(), "z": np.array(z, dtype=float),
                     "weights": [float(t) for t in dwell.softmax_weights(z)]})
        g = g_v.ravel()[idx] if which == "map" else g_z
        return J, np.asarray(g, dtype=float)

    if which == "map":
        x0 = np.clip(state["v"].ravel()[idx], BOX[0], BOX[1])
        bounds = [BOX] * len(idx)
    else:
        x0 = np.array(state["z"], dtype=float)
        # A box on the logits, wide enough that the softmax can reach 1e-4 on a
        # position and tight enough that L-BFGS-B cannot walk to infinity while
        # the objective is flat.
        bounds = [(-6.0, 6.0)] * x0.size
    t0 = time.perf_counter()
    try:
        minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 10_000, "maxfun": 10_000,
                          "ftol": 1e-16, "gtol": 1e-16})
    except StopIteration:
        pass
    mine = rows[n0:]
    if not mine:
        log(f"  block {which}: NO EVALUATIONS")
        return
    b = min(mine, key=lambda r: r["J"])
    if b["J"] <= min(r["J"] for r in rows):
        state["v"], state["z"] = b["v"], b["z"]
    log(f"  block {which:5s}: {len(mine)} evals, J {mine[0]['J']:.2f} -> "
        f"{b['J']:.2f}, {time.perf_counter() - t0:.0f} s")


def solve(kern, ops, pm, sigma_cells, plan, v0, z0, log):
    state = {"v": np.array(v0, dtype=float), "z": np.array(z0, dtype=float)}
    rows: list[dict] = []
    for which, n in plan:
        block(kern, ops, state, pm, sigma_cells, which, n, rows, log)
    best = min(rows, key=lambda r: r["J"])
    return best, rows


# ---------------------------------------------------------------------------
# programs and the time-resolved check
# ---------------------------------------------------------------------------

def time_resolved(kern, case, s, w, cycle_s, total_s=N_STEPS * DT_S):
    prog = dwell.cycle_program(w, CANDIDATE_ANGLES, cycle_time_s=cycle_s,
                               total_s=total_s, dt_s=DT_S)
    kern.set_weights(w)
    kern.averaged_Q(s)
    idx = dmarch.program_step_positions(prog, CANDIDATE_ANGLES, DT_S, N_STEPS)
    tr = dmarch.program_forward(kern, idx, N_STEPS)
    return prog, tr


def best_known_zero_map(shape: str):
    best = None
    for arm, npz, jp in (("A1_cont", OUT_LIB / f"{shape}_maps.npz", OUT_LIB / f"{shape}.json"),
                         ("MS_cont", OUT_MS / f"{shape}_maps.npz", OUT_MS / f"{shape}.json")):
        if not (npz.exists() and jp.exists()):
            continue
        arms = json.loads(jp.read_text()).get("arms", {})
        if arm not in arms:
            continue
        J = float(arms[arm]["J"])
        if best is None or J < best[0]:
            best = (J, arm, npz)
    if best is None:
        return None, None
    J, arm, npz = best
    return np.clip(np.asarray(np.load(npz)[arm], dtype=float), 0.0, 1.0), {
        "arm": arm, "npz": str(npz), "static_J_at_zero_deg": J}


# ---------------------------------------------------------------------------

def main(shape: str, n_evals: int = N_EVALS) -> dict:
    if shape not in SHAPES:
        raise ValueError(f"{shape!r} not in {SHAPES}")
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{shape}/dwell] {msg}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    kern = DwellKernel.build(cfg, angles=CANDIDATE_ANGLES)
    case = kern.case0
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)
    k = len(CANDIDATE_ANGLES)
    w_eq = np.full(k, 1.0 / k)
    z_eq = np.zeros(k)
    log(f"{k} candidate positions {list(CANDIDATE_ANGLES)}, {int(pm.sum())} part "
        f"cells, filter {topopt.FILTER_RADIUS_M * 1e3:.1f} mm = {sigma_cells:.3f} cells")

    arms: dict[str, dict] = {}
    fields: dict[str, np.ndarray] = {}
    programs: dict[str, dict] = {}

    def record(name, tr, s, w, extra=None):
        m, phi, jc = score(tr, case, s, w)
        m["arm"] = name
        m.update(extra or {})
        arms[name] = m
        fields[f"phi_{name}"] = phi.astype(np.float32)
        fields[f"J_curve_{name}"] = jc.astype(np.float32)
        fields[f"sat_{name}"] = np.asarray(s, dtype=np.float32)
        log_row(log, m)
        return m

    # -- reference arms ------------------------------------------------------
    s_unif = np.ones(pm.shape)
    kern.set_weights(w_eq)
    record("D_uniform_equal", run_forward(kern, s_unif, keep=False), s_unif, w_eq,
           {"note": "uniform dopant, equal dwells: the indexed-rotation reference"})

    sat0, meta0 = best_known_zero_map(shape)
    if sat0 is not None:
        s0 = np.where(pm, sat0, 1.0)
        kern.set_weights(w_eq)
        record("D_static0map_equal", run_forward(kern, s0, keep=False), s0, w_eq,
               {"note": "best stored zero-degree static map, equal dwells",
                "map_source": meta0})

    # -- CONTROL: equal dwells, map solved -----------------------------------
    log(f"CONTROL solve: equal dwells fixed, map only, {n_evals} gradient evaluations")
    b_eq, rows_eq = solve(kern, ops, pm, sigma_cells, [("map", n_evals)],
                          np.ones(pm.shape), z_eq, log)
    s_eq = df.apply_filter(b_eq["v"], pm, sigma_cells)
    kern.set_weights(w_eq)
    record("D_map_equal", run_forward(kern, s_eq, keep=False), s_eq, w_eq,
           {"note": "equal dwells, map solved: the symmetric-indexing control"})

    # -- DELIVERABLE: joint (map, dwell), same budget ------------------------
    plan = [("dwell", n_evals // 4), ("map", n_evals // 4),
            ("dwell", n_evals // 4), ("map", n_evals - 3 * (n_evals // 4))]
    log(f"JOINT solve, cold, alternating blocks {plan}")
    b_j, rows_j = solve(kern, ops, pm, sigma_cells, plan, np.ones(pm.shape), z_eq, log)
    s_j = df.apply_filter(b_j["v"], pm, sigma_cells)
    w_j = dwell.softmax_weights(b_j["z"])
    kern.set_weights(w_j)
    record("D_joint_cold", run_forward(kern, s_j, keep=False), s_j, w_j,
           {"note": f"joint (map, dwell), cold, {n_evals} gradient evaluations"})

    # -- DELIVERABLE: joint warm-started from the control's map --------------
    log(f"JOINT solve, WARM from the equal-dwell map, alternating blocks {plan} "
        f"(this arm has spent {2 * n_evals} gradient evaluations in total and is "
        f"labelled as such)")
    b_w, rows_w = solve(kern, ops, pm, sigma_cells, plan, b_eq["v"], z_eq, log)
    s_w = df.apply_filter(b_w["v"], pm, sigma_cells)
    w_w = dwell.softmax_weights(b_w["z"])
    kern.set_weights(w_w)
    record("D_joint_warm", run_forward(kern, s_w, keep=False), s_w, w_w,
           {"note": f"joint (map, dwell) warm-started from D_map_equal's map; "
                    f"{2 * n_evals} gradient evaluations in total"})

    # -- pick the deliverable, quantize both channels ------------------------
    cand = {n: arms[n]["J"] for n in ("D_joint_cold", "D_joint_warm")}
    winner = min(cand, key=cand.get)
    s_win = s_j if winner == "D_joint_cold" else s_w
    w_win = w_j if winner == "D_joint_cold" else w_w
    log(f"deliverable start = {winner} (J {cand[winner]:.2f})")

    s_q = pq.quantize_in_part(s_win, pm, bpp=4, sat_max=1.0)
    kern.set_weights(w_win)
    record("D_joint_4bpp", run_forward(kern, s_q, keep=False), s_q, w_win,
           {"note": "the deliverable map at 4 bits per pixel", "source_arm": winner})

    # the dwell vector the MACHINE can execute: quantized to the control step
    prog = dwell.cycle_program(w_win, CANDIDATE_ANGLES, cycle_time_s=CYCLE_TIME_S,
                               total_s=N_STEPS * DT_S, dt_s=DT_S)
    w_exec = dwell.expand_to_full_weights(prog, CANDIDATE_ANGLES)
    kern.set_weights(w_exec)
    record("D_program_4bpp", run_forward(kern, s_q, keep=False), s_q, w_exec,
           {"note": "4 bits per pixel map, dwell fractions quantized to the "
                    "executable cycle program (quasi-static score)",
            "source_arm": winner})

    # -- the time-resolved check, and the cycle-time sweep -------------------
    log("time-resolved execution of the program (part frame, no interpolation)")
    cyc_rows = []
    for cyc in CYCLE_SWEEP_S:
        p_c, tr_c = time_resolved(kern, case, s_q, w_exec, cyc)
        m = so.full_metrics(tr_c, case)
        i = int(m["t_stop_index"])
        m["energy_gate"] = eg.gate_from_trajectory(tr_c, i)
        m["max_T_at_stop_c"] = float(np.max(tr_c.T_at_end(i)))
        m["over_ceiling_250c"] = bool(m["max_T_at_stop_c"] > 250.0)
        m["mean_rho_rel_part_at_stop"] = float(tr_c.mean_rho_rel_part[i])
        m["P_abs_W_per_m"] = tr_c.P_abs_B
        m["cycle_time_s"] = float(p_c.cycle_time_s)
        m["n_moves"] = len(p_c.moves)
        m["quasi_static_J"] = float(arms["D_program_4bpp"]["J"])
        m["quasi_static_error_pct"] = 100.0 * (m["J"] - m["quasi_static_J"]) / max(
            m["quasi_static_J"], 1e-12)
        cyc_rows.append(m)
        log(f"  cycle {p_c.cycle_time_s:6.1f} s ({len(p_c.moves):4d} moves): "
            f"J {m['J']:8.2f}  IoU {m['IoU']:.4f}  vs quasi-static "
            f"{m['quasi_static_error_pct']:+6.2f}%  "
            f"Egate {'PASS' if m['energy_gate']['PASS'] else 'FAIL'}")
        if abs(cyc - CYCLE_TIME_S) < 1e-9:
            phi_tr, _ = so.phi_field(tr_c.T_at_end(i), case)
            fields["phi_D_timeresolved"] = phi_tr.astype(np.float32)
            fields["J_curve_D_timeresolved"] = so.J_curve(tr_c, case).astype(np.float32)
            m2 = dict(m)
            m2["arm"] = "D_timeresolved"
            m2["dwell_weights"] = [float(x) for x in w_exec]
            arms["D_timeresolved"] = m2

    # the same check on the EQUAL-dwell control, so the comparison is like for like
    s_eq_q = pq.quantize_in_part(s_eq, pm, bpp=4, sat_max=1.0)
    kern.set_weights(w_eq)
    record("D_map_equal_4bpp", run_forward(kern, s_eq_q, keep=False), s_eq_q, w_eq,
           {"note": "the equal-dwell control at 4 bits per pixel"})
    prog_eq, tr_eq = time_resolved(kern, case, s_eq_q, w_eq, CYCLE_TIME_S)
    m_eq = so.full_metrics(tr_eq, case)
    i_eq = int(m_eq["t_stop_index"])
    m_eq["arm"] = "D_timeresolved_equal"
    m_eq["energy_gate"] = eg.gate_from_trajectory(tr_eq, i_eq)
    m_eq["max_T_at_stop_c"] = float(np.max(tr_eq.T_at_end(i_eq)))
    m_eq["over_ceiling_250c"] = bool(m_eq["max_T_at_stop_c"] > 250.0)
    m_eq["mean_rho_rel_part_at_stop"] = float(tr_eq.mean_rho_rel_part[i_eq])
    m_eq["P_abs_W_per_m"] = tr_eq.P_abs_B
    m_eq["cycle_time_s"] = float(prog_eq.cycle_time_s)
    m_eq["dwell_weights"] = [float(x) for x in w_eq]
    arms["D_timeresolved_equal"] = m_eq
    phi_eq, _ = so.phi_field(tr_eq.T_at_end(i_eq), case)
    fields["phi_D_timeresolved_equal"] = phi_eq.astype(np.float32)
    fields["J_curve_D_timeresolved_equal"] = so.J_curve(tr_eq, case).astype(np.float32)
    log(f"  equal-dwell control, time resolved at {CYCLE_TIME_S:.0f} s cycle: "
        f"J {m_eq['J']:.2f}  IoU {m_eq['IoU']:.4f}")

    # -- machine-readable turntable programs ---------------------------------
    for tag, ww, stop_s in (("deliverable", w_exec, arms["D_timeresolved"]["t_stop_s"]),
                            ("equal_dwell_control", w_eq,
                             arms["D_timeresolved_equal"]["t_stop_s"])):
        pr = dwell.cycle_program(ww, CANDIDATE_ANGLES, cycle_time_s=CYCLE_TIME_S,
                                 total_s=N_STEPS * DT_S, dt_s=DT_S)
        j = pr.as_json()
        j["shape"] = shape
        j["arm"] = tag
        j["candidate_positions_deg"] = [float(a) for a in CANDIDATE_ANGLES]
        j["recommended_stop_s"] = float(stop_s)
        j["dopant_map_npz"] = f"{shape}_dwell_maps.npz"
        j["dopant_map_key"] = ("sat_D_joint_4bpp" if tag == "deliverable"
                               else "sat_D_map_equal_4bpp")
        j["rf_program"] = {"mode": "constant", "relative_power": 1.0,
                           "voltage_v": float(cfg["electric"]["voltage_v"])}
        programs[tag] = j
        (OUT / f"{shape}_turntable_{tag}.json").write_text(json.dumps(j, indent=1))

    np.savez_compressed(OUT / f"{shape}_dwell_maps.npz", part_mask=pm,
                        x=case.x, y=case.y, **fields)
    result = {
        "shape": shape, "config": str(cfg_path), "channel": "conductivity only",
        "actuator": "indexed turntable dwell schedule, constant RF power",
        "candidate_angles_deg": [float(a) for a in CANDIDATE_ANGLES],
        "grid": 120, "n_steps": N_STEPS, "dt_s": DT_S, "patience": PATIENCE,
        "filter_radius_m": topopt.FILTER_RADIUS_M, "sigma_cells": float(sigma_cells),
        "box": list(BOX), "parameterization": "softmax on the dwell fractions",
        "cycle_time_s": CYCLE_TIME_S, "cycle_sweep_s": list(CYCLE_SWEEP_S),
        "n_gradient_evals_per_solve": int(n_evals),
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "deliverable_source_arm": winner,
        "arms": arms, "cycle_sweep": cyc_rows,
        "solve_rows": {"control_equal": [{k: v for k, v in r.items()
                                          if k not in ("v", "z")} for r in rows_eq],
                       "joint_cold": [{k: v for k, v in r.items()
                                       if k not in ("v", "z")} for r in rows_j],
                       "joint_warm": [{k: v for k, v in r.items()
                                       if k not in ("v", "z")} for r in rows_w]},
        "programs": programs,
        "stop_convention": ("t_stop = argmin of J_phi over the arm's own "
                            "trajectory; 1500-step horizon (750 s), patience 250; "
                            "melted region phi >= 0.5; grid 120; part frame"),
        "wall_s": time.perf_counter() - t0,
    }
    (OUT / f"{shape}_dwell.json").write_text(json.dumps(result, indent=1, default=float))
    log(f"DONE wall {result['wall_s']:.0f} s")
    return result


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else N_EVALS)
