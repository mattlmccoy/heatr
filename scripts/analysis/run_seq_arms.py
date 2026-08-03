#!/usr/bin/env python3
"""SEQUENTIAL DWELL, step 3: the scored arms, each at its own optimal stop.

Arms, all on the same objective and the same stop convention:

  c  S_static_best      the best STATIC orientation, the arm on record
  c' S_static_best_map  the same with the stored dopant map of the previous
                        pass, which is that pass's L_shape deliverable
  d  S_cycled_equal     the CYCLED dwell arm, executed time-resolved at the
                        20 s cycle, for contrast with the sequential one
  a  S_seq_uniform      UNIFORM dopant map plus the sequential schedule, with
                        the switch times refined on the finite-difference gated
                        duration gradient. This is the sub-hypothesis on its own
  b  S_seq_cosolved     the dopant map co-solved with the switch times
  b' S_seq_cosolved_4bpp  that map on the printer's 16-level grid

Objective, carried on every number:

    J_phi(t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with chi_part the binary part mask in the PART frame. t_stop = argmin of J_phi
over that arm's OWN trajectory on a 1500-step horizon (dt 0.5 s, 750 s). The
shape early stop is DISABLED on every sequential arm, because a schedule whose
objective falls in phase one, rises, and falls again in phase two would be
truncated by it before phase two ever ran. Grid 120.

Run:
  ./.venv312/bin/python scripts/analysis/run_seq_arms.py <shape>
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
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import design_filter as df, dwell, energy_gate as eg   # noqa: E402
from adjoint2d import dwell_march as dmarch, gradops                  # noqa: E402
from adjoint2d import library_solve as lib, printability as pq        # noqa: E402
from adjoint2d import seq_dwell as sq, seq_dwell_march as sqm         # noqa: E402
from adjoint2d import shape_objective as so, topopt                   # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                        # noqa: E402
from adjoint2d.pins import load_cfg                                   # noqa: E402
from run_seq_probe import DT_S, N_STEPS, limb_masks, limb_report      # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_seq"
OUT_DWELL = REPO / "fgm_solve_campaign/out_dwell"
BOX = (0.0, 1.0)
CYCLE_ANGLES = np.arange(8, dtype=float) * 45.0
CYCLE_TIME_S = 20.0

CFG = {
    "L_shape": {"angles_deg": [0.0, 90.0, 105.0, 135.0], "static_best_deg": 135.0},
    "T_shape": {"angles_deg": [0.0, 45.0, 90.0, 135.0], "static_best_deg": 90.0},
}


# ---------------------------------------------------------------------------

def score(tr, case, s, wide, narrow, extra=None) -> tuple[dict, np.ndarray, np.ndarray]:
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    T = tr.T_at_end(i)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["mean_rho_rel_part_at_end"] = float(tr.mean_rho_rel_part[tr.n_outer - 1])
    m["max_T_at_stop_c"] = float(np.max(T))
    m["max_T_part_at_stop_c"] = float(np.max(T[case.part_mask]))
    # the ceiling is a PROCESS limit, so it is read over the whole exposure up
    # to the stop, not only at the stop
    m["max_T_upto_stop_c"] = (float(np.max([np.max(x) for x in tr.ckpt_T[:i + 1]]))
                              if tr.ckpt_T else m["max_T_at_stop_c"])
    m["max_T_upto_stop_c"] = max(m["max_T_upto_stop_c"], m["max_T_at_stop_c"])
    m["over_ceiling_250c"] = bool(m["max_T_upto_stop_c"] > 250.0)
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["n_outer"] = tr.n_outer
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    m.update(limb_report(T, case, wide, narrow))
    m.update(extra or {})
    phi, _ = so.phi_field(T, case)
    return m, phi, so.J_curve(tr, case)


def log_row(shape, m):
    print(f"[{shape}] {m['arm']:22s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
          f"grow {m['bed_melt_pct_of_part']:5.2f}%  under {m['part_under_melt_pct']:5.2f}%  "
          f"wide {m['wide_melted_pct']:6.2f}%  narrow {m['narrow_melted_pct']:6.2f}%  "
          f"rho {m['mean_rho_rel_part_at_stop']:.4f}  P {m['P_abs_W_per_m']:6.1f} W/m  "
          f"stop {m['t_stop_s']:6.1f} s{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
          f"maxT {m['max_T_upto_stop_c']:6.1f} C{'  CEILING' if m['over_ceiling_250c'] else ''}  "
          f"Egate {'PASS' if m['energy_gate']['PASS'] else 'FAIL'} "
          f"({m['energy_gate']['rel_residual_at_index'] * 100:.2f}%)", flush=True)


def seq_forward(kern, s, seg, dur, keep=False):
    kern.averaged_Q(s)
    return sqm.sequential_forward(kern, seg, dur, N_STEPS, keep_checkpoints=keep)


# ---------------------------------------------------------------------------
# the two solve blocks
# ---------------------------------------------------------------------------

def eval_point(kern, ops, v, dur, seg, pm, sigma_cells):
    """One forward and ONE reverse march; both gradients come out of it."""
    s = df.apply_filter(v, pm, sigma_cells)
    tr = seq_forward(kern, s, seg, dur, keep=True)
    st = so.optimal_stop(tr, kern.case0)
    J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), kern.case0)
    g_s, g_d = sqm.sequential_gradients(kern, s, tr, {st.index: seed}, grad_ops=ops)
    g_v = df.filter_vjp(g_s, pm, sigma_cells)
    del tr
    return float(J), g_v, g_d, int(st.index), bool(st.at_horizon)


def block(kern, ops, state, seg, pm, sigma_cells, which, n_evals, rows, log):
    idx = np.flatnonzero(pm.ravel())
    n_free = len(seg) - 1
    n0 = len(rows)

    def fun(x):
        if len(rows) - n0 >= int(n_evals):
            raise StopIteration
        if which == "map":
            v = np.ones(pm.shape)
            v.ravel()[idx] = x
            dur = state["dur"]
        else:
            v = state["v"]
            dur = np.concatenate([np.asarray(x, dtype=float), state["dur"][n_free:]])
        J, g_v, g_d, si, hz = eval_point(kern, ops, v, dur, seg, pm, sigma_cells)
        rows.append({"eval_index": len(rows) + 1, "block": which, "J": J,
                     "t_stop_index": si, "t_stop_at_horizon": hz,
                     "v": v.copy(), "dur": np.asarray(dur, dtype=float).copy()})
        g = g_v.ravel()[idx] if which == "map" else g_d[:n_free]
        return J, np.asarray(g, dtype=float)

    if which == "map":
        x0 = np.clip(state["v"].ravel()[idx], BOX[0], BOX[1])
        bounds = [BOX] * len(idx)
    else:
        x0 = np.asarray(state["dur"][:n_free], dtype=float)
        bounds = [(1.0, N_STEPS * DT_S)] * n_free
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
        state["v"], state["dur"] = b["v"], b["dur"]
    log(f"  block {which:8s}: {len(mine)} evals, J {mine[0]['J']:.2f} -> {b['J']:.2f}, "
        f"durations {np.round(b['dur'], 2).tolist()}, "
        f"{time.perf_counter() - t0:.0f} s")


# ---------------------------------------------------------------------------

def best_from_screens(shape: str) -> dict:
    """The best two-segment and three-segment schedule the screens found."""
    rows2, rows3 = [], []
    for name in (f"{shape}_screen.json", f"{shape}_screen2.json"):
        p = OUT / name
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        rows2 += j.get("rows", []) + j.get("two_segment", [])
        rows3 += j.get("three_segment", [])
    out = {}
    if rows2:
        r = min(rows2, key=lambda x: x["J"])
        out["two"] = {"angles_deg": [r["a1_deg"], r["a2_deg"]],
                      "durations_s": [r["switch_s"], N_STEPS * DT_S - r["switch_s"]],
                      "screen_J": r["J"], "screen_IoU": r["IoU"]}
    if rows3:
        r = min(rows3, key=lambda x: x["J"])
        out["three"] = {"angles_deg": [r["a1_deg"], r["a2_deg"], r["a3_deg"]],
                        "durations_s": [r["switch1_s"], r["switch2_s"] - r["switch1_s"],
                                        N_STEPS * DT_S - r["switch2_s"]],
                        "screen_J": r["J"], "screen_IoU": r["IoU"]}
    return out


def main(shape: str, n_evals: int = 8) -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    conf = CFG[shape]
    angles = np.asarray(conf["angles_deg"], dtype=float)
    cfg = load_cfg(lib.shape_config(shape))
    kern = DwellKernel.build(cfg, angles=angles)
    case = kern.case0
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)
    wide, narrow = limb_masks(pm)
    idx_of = {float(a): j for j, a in enumerate(angles)}
    s_unif = np.ones(pm.shape)
    kern.set_weights(np.full(angles.size, 1.0 / angles.size))

    arms: dict[str, dict] = {}
    fields: dict[str, np.ndarray] = {}

    def record(name, tr, s, extra=None):
        m, phi, jc = score(tr, case, s, wide, narrow, extra)
        m["arm"] = name
        arms[name] = m
        fields[f"phi_{name}"] = phi.astype(np.float32)
        fields[f"J_curve_{name}"] = jc.astype(np.float32)
        fields[f"sat_{name}"] = np.asarray(s, dtype=np.float32)
        log_row(shape, m)
        return m

    # -- c: the best static orientation --------------------------------------
    j_stat = idx_of[conf["static_best_deg"]]
    record("S_static_best_uniform",
           seq_forward(kern, s_unif, [j_stat], [N_STEPS * DT_S]), s_unif,
           {"note": f"static at {conf['static_best_deg']:.0f} deg, uniform map",
            "schedule": "static", "angles_deg": [conf["static_best_deg"]]})

    npz = OUT_DWELL / f"{shape}_dwell_maps.npz"
    if npz.exists():
        z = np.load(npz)
        key = "sat_D_joint_4bpp" if "sat_D_joint_4bpp" in z else None
        if key:
            s_prev = np.clip(np.asarray(z[key], dtype=float), 0.0, 1.0)
            s_prev = np.where(pm, s_prev, 1.0)
            record("S_static_best_prevmap",
                   seq_forward(kern, s_prev, [j_stat], [N_STEPS * DT_S]),
                   s_prev,
                   {"note": "static at the best angle with the PREVIOUS pass's "
                            "4 bits-per-pixel dopant map",
                    "schedule": "static", "map_source": str(npz) + f"::{key}"})

    # -- d: the cycled-dwell arm, time resolved ------------------------------
    kc = DwellKernel.build(cfg, angles=CYCLE_ANGLES)
    w_eq = np.full(CYCLE_ANGLES.size, 1.0 / CYCLE_ANGLES.size)
    kc.set_weights(w_eq)
    kc.averaged_Q(s_unif)
    prog_c = dwell.cycle_program(w_eq, CYCLE_ANGLES, cycle_time_s=CYCLE_TIME_S,
                                 total_s=N_STEPS * DT_S, dt_s=DT_S)
    idx_c = dmarch.program_step_positions(prog_c, CYCLE_ANGLES, DT_S, N_STEPS)
    tr_c = dmarch.program_forward(kc, idx_c, N_STEPS)
    m, phi, jc = score(tr_c, kc.case0, s_unif, wide, narrow,
                       {"note": "CYCLED equal dwell over the eight 45-degree "
                                "positions, executed time-resolved at the 20 s "
                                "cycle, uniform map",
                        "schedule": "cycled", "cycle_time_s": CYCLE_TIME_S})
    m["arm"] = "S_cycled_equal_uniform"
    arms[m["arm"]] = m
    fields[f"phi_{m['arm']}"] = phi.astype(np.float32)
    fields[f"J_curve_{m['arm']}"] = jc.astype(np.float32)
    log_row(shape, m)
    del kc, tr_c

    # -- a: uniform map plus the sequential schedule -------------------------
    best = best_from_screens(shape)
    if not best:
        raise RuntimeError(f"no screen results for {shape}; run run_seq_screen first")
    plans = {}
    for tag in ("two", "three"):
        if tag not in best:
            continue
        seg = [idx_of[a] for a in best[tag]["angles_deg"]]
        dur = np.asarray(best[tag]["durations_s"], dtype=float)
        name = f"S_seq_uniform_{tag}"
        record(name, seq_forward(kern, s_unif, seg, dur), s_unif,
               {"note": f"UNIFORM map, {len(seg)}-segment sequential schedule "
                        f"straight off the screen grid",
                "schedule": "sequential", "segments_deg": best[tag]["angles_deg"],
                "durations_s": [float(x) for x in dur]})
        if tag != "two":
            # The screen showed the three-segment argmin sitting BEFORE the
            # third switch, so the third hold is inert. It is recorded as
            # measured and not refined; spending the budget on it would be
            # optimizing a variable the objective does not see.
            arms[name]["note"] += ("; NOT refined: the argmin sits before the "
                                   "third switch, so the third hold is inert")
            continue
        # refine the switch times on the gated duration gradient
        state = {"v": np.ones(pm.shape), "dur": dur.copy()}
        rows: list[dict] = []
        block(kern, ops, state, seg, pm, sigma_cells, "duration", n_evals, rows, print)
        dur_r = state["dur"]
        record(f"S_seq_uniform_{tag}_refined",
               seq_forward(kern, s_unif, seg, dur_r), s_unif,
               {"note": f"UNIFORM map, switch times refined on the gated "
                        f"duration gradient, {len(rows)} gradient evaluations",
                "schedule": "sequential", "segments_deg": best[tag]["angles_deg"],
                "durations_s": [float(x) for x in dur_r],
                "n_gradient_evals": len(rows),
                "duration_trace": [{"J": r["J"], "dur": r["dur"].tolist()}
                                   for r in rows]})
        plans[tag] = {"seg": seg, "dur": dur_r,
                      "angles_deg": best[tag]["angles_deg"]}

    # -- b: co-solve the dopant map with the switch times --------------------
    tag = min(plans, key=lambda t: arms[f"S_seq_uniform_{t}_refined"]["J"])
    seg, dur = plans[tag]["seg"], plans[tag]["dur"]
    print(f"[{shape}] co-solve starts from the {tag}-segment refined schedule "
          f"{plans[tag]['angles_deg']} at {np.round(dur, 2).tolist()} s", flush=True)
    state = {"v": np.ones(pm.shape), "dur": np.asarray(dur, dtype=float).copy()}
    rows_j: list[dict] = []
    for which, n in (("map", n_evals), ("duration", max(n_evals // 2, 2)),
                     ("map", n_evals)):
        block(kern, ops, state, seg, pm, sigma_cells, which, n, rows_j, print)
    s_co = df.apply_filter(state["v"], pm, sigma_cells)
    record("S_seq_cosolved", seq_forward(kern, s_co, seg, state["dur"]),
           s_co,
           {"note": f"dopant map co-solved with the switch times, "
                    f"{len(rows_j)} gradient evaluations, filter "
                    f"{topopt.FILTER_RADIUS_M * 1e3:.1f} mm, box {BOX}",
            "schedule": "sequential", "segments_deg": plans[tag]["angles_deg"],
            "durations_s": [float(x) for x in state["dur"]],
            "n_gradient_evals": len(rows_j)})
    s_q = pq.quantize_in_part(s_co, pm, bpp=4, sat_max=1.0)
    record("S_seq_cosolved_4bpp",
           seq_forward(kern, s_q, seg, state["dur"]), s_q,
           {"note": "the co-solved map at 4 bits per pixel, the deliverable",
            "schedule": "sequential", "segments_deg": plans[tag]["angles_deg"],
            "durations_s": [float(x) for x in state["dur"]]})

    # -- phase-by-phase melt snapshots, the story the figure has to show -----
    # The claim under test is that the limb melted in phase one STAYS melted
    # while phase two heats the other limb. That is a statement about the melt
    # field at intermediate times, so those fields are stored here rather than
    # inferred from the endpoints.
    snap_arms = {"seq": (seg, state["dur"], s_q),
                 "static": ([j_stat], [N_STEPS * DT_S], s_unif)}
    snap_times = {}
    for tag_s, (sg, dr, ss) in snap_arms.items():
        kern.averaged_Q(ss)
        tr_s = sqm.sequential_forward(kern, sg, dr, N_STEPS)
        i_stop = int(so.optimal_stop(tr_s, case).index)
        sw = float(np.cumsum(np.asarray(dr, dtype=float))[0])
        picks = sorted({int(round(0.5 * sw / DT_S)), int(round(sw / DT_S)) - 1,
                        int(round(sw / DT_S)) + int(0.34 * (i_stop - sw / DT_S)),
                        int(round(sw / DT_S)) + int(0.67 * (i_stop - sw / DT_S)),
                        i_stop})
        picks = [p for p in picks if 0 <= p <= i_stop]
        for p_i in picks:
            phi_p, _ = so.phi_field(tr_s.T_at_end(p_i), case)
            fields[f"snap_{tag_s}_{p_i}"] = phi_p.astype(np.float32)
        snap_times[tag_s] = {"steps": picks,
                             "times_s": [(p + 1) * DT_S for p in picks],
                             "switch_s": sw, "stop_step": i_stop,
                             "stop_s": (i_stop + 1) * DT_S,
                             "segments_deg": [float(angles[a]) for a in sg]}
        del tr_s
    fields["part_mask"] = pm.astype(np.uint8)
    fields["x"] = np.asarray(case.x, dtype=np.float64)
    fields["y"] = np.asarray(case.y, dtype=np.float64)
    fields["wide_limb"] = wide.astype(np.uint8)
    fields["narrow_limb"] = narrow.astype(np.uint8)

    # -- the machine-readable turntable programs -----------------------------
    programs = {}
    for name in arms:
        a = arms[name]
        if a.get("schedule") != "sequential":
            continue
        p = sq.sequential_program(a["segments_deg"], list(range(len(a["segments_deg"]))),
                                  a["durations_s"], DT_S, N_STEPS * DT_S, snap=True)
        j = p.as_json()
        j.update({"shape": shape, "arm": name, "grid": 120,
                  "recommended_stop_s": a["t_stop_s"],
                  "J_phi_at_stop": a["J"], "IoU_at_stop": a["IoU"],
                  "dopant_map": (f"{shape}_seq_maps.npz::sat_{name}"),
                  "radio_frequency_program": "constant, calibrated drive voltage",
                  "caveat": ("grid 120 only; not verified on the production "
                             "engine, which cannot express an unequal dwell")})
        programs[name] = j
        (OUT / f"{shape}_turntable_{name}.json").write_text(
            json.dumps(j, indent=2, default=float))

    out = {"shape": shape, "angles_deg": [float(a) for a in angles],
           "n_steps": N_STEPS, "dt_s": DT_S, "early_stop": "DISABLED",
           "filter_radius_m": topopt.FILTER_RADIUS_M,
           "sigma_cells": float(sigma_cells),
           "screen_best": best, "arms": arms, "programs": programs,
           "snapshots": snap_times,
           "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_arms.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(OUT / f"{shape}_seq_maps.npz", **fields)
    print(f"[{shape}] wrote {OUT / f'{shape}_arms.json'} in {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L_shape",
         int(sys.argv[2]) if len(sys.argv) > 2 else 8)
