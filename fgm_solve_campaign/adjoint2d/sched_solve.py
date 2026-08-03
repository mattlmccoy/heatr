"""Temporal power scheduling on the shapes the library census left NOT RESCUED.

One shape per invocation. Every arm is scored under the SAME objective

    J(s, p, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

at its OWN J-stop, t_stop = argmin over that arm's own trajectory of J.

STOP AND HORIZON CONVENTIONS, stated once.
  * The shape-fidelity early stop of the library run (`shape_stop_patience`) is
    DISABLED here on every arm, baselines included. A schedule with a long OFF
    stretch makes J rise, which would trip the patience counter and truncate
    the march before a later ON stretch could act; the optimizer would then see
    zero gradient on every segment past the truncation. Every march therefore
    runs the full horizon. This makes the numbers here slightly different from
    the library report even for identical maps, and the re-scored baseline is
    reported rather than the library number being carried across.
  * The horizon is the configuration's 1500 outer steps except on the cross,
    where the library run's J-minimum sat ON the 1500-step horizon and its J
    was therefore a bound. The cross is run to 2500 steps so its optimum is
    interior. Cross numbers here are NOT comparable to cross numbers in
    `SHAPE_LIBRARY_SOLVE_REPORT.md`.

ARMS
  U_uniform        s = 1, no schedule. The do-nothing reference.
  BASE_map4bpp     the library's deliverable solved map (A1_4bpp), no schedule,
                   re-scored under the conventions above. THE baseline.
  SCHED_only       s = 1 plus an optimized continuous schedule. Isolates what
                   scheduling buys with NO dopant grading at all.
  CO_cont          co-optimized map (box [0, 1]) and schedule (box [0, 1.5]).
  CO_4bpp          that map quantized to 4 bits per pixel, schedule re-optimized
                   continuously. THE deliverable arm.
  BIN_relax        the same 4-bit map with the schedule re-optimized inside the
                   BINARY box [0, 1] (the continuous relaxation of on/off).
  BIN_round        that relaxation rounded onto {0, 1}. The rounding loss is
                   reported as a labelled number, never hidden.

Run:
  ./.venv312/bin/python -m adjoint2d.sched_solve <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import adjoint, control, energy_gate as eg, forward as fwd, gradops
from . import printability as pq
from . import schedule as sch
from . import shape_objective as so
from .library_solve import shape_config
from .pins import build_case, load_cfg

SHAPES = ("cross", "T_shape", "L_shape", "star")
HORIZON = {"cross": 2500, "T_shape": 1500, "L_shape": 1500, "star": 1500}
# Schedule WINDOW, in outer steps. The 16 segments are laid over this window,
# not over the whole march; past it the generator HOLDS the last segment level.
# The window is 1.5x the later of the uniform-map and the library-baseline
# J-optimal stop, MEASURED once per shape before any optimization (uniform /
# baseline stops: cross 439 / 1505, T_shape 460 / 433, L_shape 526 / 525,
# star 284 / 327) and rounded up. Laying the segments over the full march
# instead leaves 10 to 13 of the 16 segments entirely after the stop with
# exactly zero gradient, which is not a control resolution the optimizer can
# use. The post-window hold term is part of dJ/dp[-1] and is FD-gated.
WINDOW = {"cross": 2250, "T_shape": 700, "L_shape": 800, "star": 500}
N_SEG_DEFAULT = 16
P_BOX_CONT = (0.0, 1.5)
P_BOX_BINARY = (0.0, 1.0)
S_BOX = (0.0, 1.0)
BUDGET_SCHED_ONLY = 20.0
BUDGET_CO = 40.0
BUDGET_RESCHED = 15.0
BLOCK_ORDER = ("p", "s", "p", "s")
LIB_MAPS = Path(__file__).resolve().parent.parent / "out_lib"


# ---------------------------------------------------------------------------
# pure logic (unit tested)
# ---------------------------------------------------------------------------

def rounding_loss(J_relaxed: float, J_rounded: float) -> dict:
    """How much the binary rounding costs, in absolute and relative J."""
    a, b = float(J_relaxed), float(J_rounded)
    return {"J_relaxed": a, "J_rounded": b, "delta_J": b - a,
            "rel_loss": (b - a) / max(abs(a), 1e-30)}


def iso_j_hold_gain(j_curve, rho_curve, stop_index: int, tol: float = 0.02) -> dict:
    """How much densification a HOLD past the stop buys at iso shape fidelity.

    The triangle showcase found mean rho only 0.684 at the shape-optimal stop:
    densification lags the melt front. This asks the direct question. Starting
    at the J-optimal stop, walk FORWARD while J stays inside a band of `tol`
    around its value at the stop, and report the densest time reached.

    The walk is contiguous on purpose. A later time inside the band that is
    separated from the stop by an excursion outside it is NOT reachable by
    holding, because the march passes through the excursion.
    """
    j = np.asarray(j_curve, dtype=float)
    r = np.asarray(rho_curve, dtype=float)
    i0 = int(stop_index)
    limit = j[i0] * (1.0 + float(tol)) if j[i0] >= 0 else j[i0] * (1.0 - float(tol))
    i = i0
    while i + 1 < j.size and j[i + 1] <= limit:
        i += 1
    return {"index": int(i), "rho": float(r[i]), "rho_at_stop": float(r[i0]),
            "d_rho": float(r[i] - r[i0]), "extra_steps": int(i - i0),
            "J_at_stop": float(j[i0]), "J_at_hold_end": float(j[i]),
            "J_band_tol": float(tol)}


def rescue_verdict(iou_base: float, iou_arm: float, tol: float = 0.02) -> str:
    """Did scheduling move the shape? Comparative, with a 2 IoU point tie band."""
    d = float(iou_arm) - float(iou_base)
    if d >= tol:
        return "SCHEDULING HELPS"
    if d <= -tol:
        return "SCHEDULING HURTS"
    return "NULL, scheduling buys nothing"


# ---------------------------------------------------------------------------
# forward / scoring with a schedule
# ---------------------------------------------------------------------------

def run_forward(case, s, p, n_seg, horizon, checkpoints=False, window=None):
    return fwd.forward(case, s, keep_checkpoints=checkpoints, stop_after_phi=None,
                       shape_stop_patience=None, n_steps=horizon,
                       p_seg=p, n_seg=n_seg,
                       p_horizon=horizon if window is None else int(window))


def score(case, s, p, n_seg, horizon, window=None) -> dict:
    win = int(horizon if window is None else window)
    tr = run_forward(case, s, p, n_seg, horizon, window=win)
    m = so.full_metrics(tr, case)
    m["P_abs_B_full_power_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["horizon"] = int(horizon)
    m["schedule_window_steps"] = win
    # Densification readouts. rho is NOT in the objective; it is reported so
    # the place-then-hold question can be answered with numbers.
    m["mean_rho_part_at_stop"] = float(tr.mean_rho_rel_part[m["t_stop_index"]])
    m["mean_rho_part_at_end"] = float(tr.mean_rho_rel_part[-1])
    m["rho_curve"] = [float(v) for v in tr.mean_rho_rel_part]
    m["J_curve"] = [float(v) for v in so.J_curve(tr, case)]
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    m["energy_gate"] = eg.gate_from_trajectory(tr, m["t_stop_index"])
    m["dose_J_per_m_at_stop"] = float(tr.energy_in_J_per_m[m["t_stop_index"]])
    if p is None:
        m["p_seg"] = None
        m["duty_cycle"] = 1.0
        m["n_switches"] = 0
        m["structure"] = {"structure": "NO SCHEDULE", "n_active_segments": 0}
    else:
        m["p_seg"] = [float(v) for v in np.asarray(p).ravel()]
        m["duty_cycle"] = sch.duty_cycle(p, win, n_seg)
        m["n_switches"] = sch.n_switches(p)
        m["schedule_instructions"] = sch.instructions(p, win, n_seg,
                                                      case.pins.dt, merge=True)
        m["structure"] = sch.place_then_hold(p, win, n_seg, m["t_stop_index"])
    return m


# ---------------------------------------------------------------------------
# alternating block solve
# ---------------------------------------------------------------------------

class Budget(Exception):
    pass


def joint_solve(case, ops, n_seg, horizon, s0, p0, n_evals: int,
                s_box=S_BOX, p_box=P_BOX_CONT,
                blocks=BLOCK_ORDER,
                window=None) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    """Alternating L-BFGS-B on the dopant map and the power schedule.

    Alternating blocks rather than one joint vector, deliberately: the two
    design variables differ by three orders of magnitude in gradient scale
    (dJ/dp_k sums thousands of cells, dJ/ds_i is one cell), and L-BFGS-B is not
    invariant to that. Alternating removes the need for an invented scaling
    constant. Both gradients still come out of ONE backward sweep, so a block
    that only moves p pays for the map gradient it discards; that cost is real
    and it is inside the budget.
    """
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
    s_cur = np.array(s0, dtype=float, copy=True)
    p_cur = np.array(p0, dtype=float, copy=True)
    rows: list[dict] = []
    best = {"J": np.inf}
    best_s, best_p = s_cur.copy(), p_cur.copy()

    def evaluate(s, p):
        nonlocal best, best_s, best_p
        if len(rows) >= n_evals:
            raise Budget
        tr = run_forward(case, s, p, n_seg, horizon, checkpoints=True, window=window)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        gs, gp = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops,
                                  with_schedule=True)
        row = {"eval_index": len(rows) + 1, "J": float(J),
               "t_stop_index": int(st.index), "at_horizon": bool(st.at_horizon),
               "duty_cycle": sch.duty_cycle(p, int(horizon if window is None else window), n_seg)}
        rows.append(row)
        if J < best["J"]:
            best = row
            best_s, best_p = s.copy(), p.copy()
        return float(J), gs, gp

    shares = sch.block_budget(n_evals, len(blocks))
    for which, share in zip(blocks, shares):
        if share <= 0:
            continue
        start = len(rows)

        if which == "p":
            def fun_p(v):
                if len(rows) - start >= share:
                    raise Budget
                J, _gs, gp = evaluate(s_cur, v)
                return J, gp.astype(float)
            try:
                minimize(fun_p, p_cur, jac=True, method="L-BFGS-B",
                         bounds=[p_box] * n_seg,
                         options={"maxiter": 10_000, "maxfun": share,
                                  "ftol": 1e-16, "gtol": 1e-16})
            except Budget:
                pass
            p_cur = best_p.copy()
            s_cur = best_s.copy()
        else:
            def fun_s(v):
                if len(rows) - start >= share:
                    raise Budget
                s = np.ones(pm.shape)
                s.ravel()[idx] = v
                J, gs, _gp = evaluate(s, p_cur)
                return J, gs.ravel()[idx].astype(float)
            try:
                minimize(fun_s, s_cur.ravel()[idx], jac=True, method="L-BFGS-B",
                         bounds=[s_box] * len(idx),
                         options={"maxiter": 10_000, "maxfun": share,
                                  "ftol": 1e-16, "gtol": 1e-16})
            except Budget:
                pass
            p_cur = best_p.copy()
            s_cur = best_s.copy()

    return best, best_s, best_p, rows


def schedule_only_solve(case, ops, n_seg, horizon, s_fixed, p0, n_evals, p_box,
                        window=None):
    """Optimize the schedule alone, with the dopant map held fixed."""
    return joint_solve(case, ops, n_seg, horizon, s_fixed, p0, n_evals,
                       p_box=p_box, blocks=("p",), window=window)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(shape: str, outdir: str, n_seg: int = N_SEG_DEFAULT,
         window: int | None = None, tag: str = "") -> dict:
    if shape not in SHAPES:
        raise ValueError(f"{shape!r} is not in the NOT RESCUED set {SHAPES}")
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cfg_path = shape_config(shape)
    case = build_case(load_cfg(cfg_path))
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    horizon = HORIZON[shape]
    win = int(WINDOW[shape] if window is None else window)
    ones = np.ones(n_seg)

    res: dict = {
        "shape": shape, "config": str(cfg_path), "n_seg": n_seg,
        "tag": tag,
        "horizon_steps": horizon, "horizon_s": horizon * case.pins.dt,
        "schedule_window_steps": win,
        "schedule_window_s": win * case.pins.dt,
        "segment_length_steps": horizon / n_seg if win == horizon else win / n_seg,
        "window_convention": (
            "the 16 segments span the WINDOW; past it the generator holds the "
            "last segment level, and that hold is inside dJ/dp[-1] and FD-gated"),
        "dt_s": case.pins.dt, "n_part_cells": case.n_part,
        "p_box_continuous": list(P_BOX_CONT), "p_box_binary": list(P_BOX_BINARY),
        "early_stop": "DISABLED on every arm; every march runs the full horizon",
        "stop_convention": "t_stop = argmin over the arm's own trajectory of J",
        "injection_convention": (
            "p multiplies POWER; equivalently the drive voltage is V*sqrt(p). "
            "The max_qrf cap is applied after the scaling."),
        "arms": {}, "budgets": {}, "cost": {},
    }
    maps: dict[str, np.ndarray] = {"part_mask": pm, "x": case.x, "y": case.y}

    # --- measured cost model ------------------------------------------------
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = run_forward(case, s_u, ones, n_seg, horizon, checkpoints=True, window=win)
    t_b = time.perf_counter()
    st_u = so.optimal_stop(tr_u, case)
    _J, seed = so.shape_J_and_seed(tr_u.T_at_end(st_u.index), case)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed}, grad_ops=ops,
                     with_schedule=True)
    t_d = time.perf_counter()
    ratio = (t_d - t_c) / max(t_b - t_a, 1e-9)
    res["cost"] = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c, "ratio": ratio}
    n_sched = control.max_gradient_evals(BUDGET_SCHED_ONLY, ratio)
    n_co = control.max_gradient_evals(BUDGET_CO, ratio)
    n_re = control.max_gradient_evals(BUDGET_RESCHED, ratio)
    res["budgets"] = {
        "sched_only_forward_equivalents": BUDGET_SCHED_ONLY, "sched_only_evals": n_sched,
        "co_forward_equivalents": BUDGET_CO, "co_evals": n_co,
        "resched_forward_equivalents": BUDGET_RESCHED, "resched_evals": n_re,
        "block_order": list(BLOCK_ORDER),
    }
    print(f"[{shape}] horizon {horizon} ratio {ratio:.3f} evals sched {n_sched} "
          f"co {n_co} resched {n_re}", flush=True)

    def record(tag, s, p, extra=None):
        m = score(case, s, p, n_seg, horizon, window=win)
        m["arm"] = tag
        m["iso_J_hold"] = iso_j_hold_gain(m["J_curve"], m["rho_curve"],
                                          m["t_stop_index"])
        if extra:
            m.update(extra)
        res["arms"][tag] = m
        maps[f"map_{tag}"] = s
        print(f"[{shape}] {tag:14s} J {m['J']:9.2f} IoU {m['IoU']:.4f} "
              f"under {m['part_under_melt_pct']:6.2f}% grow {m['bed_melt_pct_of_part']:5.2f}% "
              f"stop {m['t_stop_s']:7.1f}s duty {m['duty_cycle']:.3f} "
              f"rho {m['mean_rho_part_at_stop']:.3f} "
              f"{m['structure']['structure'][:15]:15s} "
              f"Eres {m['energy_gate']['rel_residual_at_index']*100:.2f}%", flush=True)
        return m

    # --- references ---------------------------------------------------------
    record("U_uniform", s_u, None)
    lib = np.load(LIB_MAPS / f"{shape}_maps.npz")
    s_base = np.asarray(lib["A1_4bpp"], dtype=float)
    record("BASE_map4bpp", s_base, None,
           {"source": str(LIB_MAPS / f'{shape}_maps.npz') + " key A1_4bpp"})

    # --- (ii) schedule alone, uniform map -----------------------------------
    _b, _s, p_sched, rows_s = schedule_only_solve(
        case, ops, n_seg, horizon, s_u, ones, n_sched, P_BOX_CONT, window=win)
    res["rows_SCHED_only"] = rows_s
    record("SCHED_only", s_u, p_sched)

    # --- (iii) co-optimized map and schedule --------------------------------
    _b, s_co, p_co, rows_co = joint_solve(
        case, ops, n_seg, horizon, s_u, ones, n_co, p_box=P_BOX_CONT, window=win)
    res["rows_CO"] = rows_co
    record("CO_cont", s_co, p_co)

    s_co4 = pq.quantize_in_part(s_co, pm, bpp=4, sat_max=1.0)
    _b, _s, p_co4, rows_r = schedule_only_solve(
        case, ops, n_seg, horizon, s_co4, p_co, n_re, P_BOX_CONT, window=win)
    res["rows_CO_4bpp_resched"] = rows_r
    record("CO_4bpp", s_co4, p_co4)

    # --- (iv) binary on/off -------------------------------------------------
    _b, _s, p_bin_relax, rows_b = schedule_only_solve(
        case, ops, n_seg, horizon, s_co4, np.clip(p_co4, *P_BOX_BINARY),
        n_re, P_BOX_BINARY, window=win)
    res["rows_BIN"] = rows_b
    m_relax = record("BIN_relax", s_co4, p_bin_relax)
    p_bin = sch.round_binary(p_bin_relax)
    m_round = record("BIN_round", s_co4, p_bin)
    res["binary_rounding"] = rounding_loss(m_relax["J"], m_round["J"])

    # --- verdict ------------------------------------------------------------
    base = res["arms"]["BASE_map4bpp"]
    res["verdict"] = {
        "baseline_arm": "BASE_map4bpp",
        "baseline_J": base["J"], "baseline_IoU": base["IoU"],
        "schedule_alone_vs_uniform_dIoU":
            res["arms"]["SCHED_only"]["IoU"] - res["arms"]["U_uniform"]["IoU"],
        "deliverable_arm": "CO_4bpp",
        "deliverable_J": res["arms"]["CO_4bpp"]["J"],
        "deliverable_IoU": res["arms"]["CO_4bpp"]["IoU"],
        "dIoU_vs_baseline": res["arms"]["CO_4bpp"]["IoU"] - base["IoU"],
        "dJ_rel_vs_baseline": (base["J"] - res["arms"]["CO_4bpp"]["J"]) / max(abs(base["J"]), 1e-30),
        "class": rescue_verdict(base["IoU"], res["arms"]["CO_4bpp"]["IoU"]),
        "binary_dIoU_vs_continuous":
            res["arms"]["BIN_round"]["IoU"] - res["arms"]["CO_4bpp"]["IoU"],
        "deliverable_mean_rho_at_stop": res["arms"]["CO_4bpp"]["mean_rho_part_at_stop"],
        "baseline_mean_rho_at_stop": base["mean_rho_part_at_stop"],
        "deliverable_structure": res["arms"]["CO_4bpp"]["structure"]["structure"],
        "structure_by_arm": {k: v["structure"]["structure"]
                             for k, v in res["arms"].items()},
        "mean_rho_by_arm": {k: v["mean_rho_part_at_stop"]
                            for k, v in res["arms"].items()},
        "iso_J_hold_by_arm": {k: v["iso_J_hold"] for k, v in res["arms"].items()},
    }
    res["energy_gate_violations"] = [
        a for a, m in res["arms"].items() if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t0

    suffix = f"_{tag}" if tag else ""
    np.savez_compressed(out / f"{shape}{suffix}_sched_maps.npz", **maps,
                        p_SCHED_only=p_sched, p_CO_cont=p_co, p_CO_4bpp=p_co4,
                        p_BIN_relax=p_bin_relax, p_BIN_round=p_bin)
    (out / f"{shape}{suffix}_sched.json").write_text(
        json.dumps(res, indent=2, default=float))
    v = res["verdict"]
    print(f"[{shape}] VERDICT {v['class']}  dIoU {v['dIoU_vs_baseline']:+.4f}  "
          f"dJ {v['dJ_rel_vs_baseline']*100:+.1f}%  wall {res['wall_s']:.0f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else N_SEG_DEFAULT,
         int(sys.argv[4]) if len(sys.argv) > 4 else None,
         sys.argv[5] if len(sys.argv) > 5 else "")
