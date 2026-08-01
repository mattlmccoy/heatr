"""Melt-objective solve in the PERMITTIVITY-CO-VARYING dopant channel.

One shape per invocation. The objective is unchanged:

    J_phi(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

read at that arm's OWN J-stop, t_stop = argmin over its own stored trajectory,
with `t_stop_at_horizon` flagged whenever the minimum sits on the last stored
step (which makes that arm's J a bound). The melted region for intersection
over union (IoU), growth and under-melt is phi >= 0.5. GRID 120 x 120
throughout, and `SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute
fidelity at this grid does not transfer to 160, so every IoU carries that
qualifier.

WHAT CHANGES against `ms_solve.py`, and it is exactly one thing: the ACTUATOR.

    conductivity-only (previous arms, `sat_map_npz_direct` hook)
        sigma = sigma_v + (fill*s)*(sigma_d0 - sigma_v)
        eps_r = eps_v   +  fill    *(eps_d   - eps_v)

    permittivity co-varying (THIS module, `saturation_map_npz` hook)
        sigma = sigma_v + (fill*s)*(sigma_d0 - sigma_v)
        eps_r = eps_v   + (fill*s)*(eps_d   - eps_v)

Every stored historical dopant map was scored in the second channel, and the
solve had only the first. `VERIFICATION_PRINTABILITY_REPORT.md` Section 3.1
measured the size of that gap: the same historical map on the cross scores
J = 343.27 with permittivity co-varying and J = 1036.00 (nothing melts at all)
with conductivity only. This module removes the confound.

RECIPE, the production one from `MULTISTART_REPORT.md`: filtered FULL-DEPTH
single-start solves, no budget split, physical-length design filter on the
design variable at sigma = 1.5 cells, box [0, 1].

TWO STARTS, each run at the FULL budget, and the better kept:

  cold  uniform saturation 1
  warm  the best stored historical 4-bits-per-pixel dopant map for this shape
        as `out_lib/<shape>.json` selected it, in the boundary convention that
        won there, loaded through the PRODUCTION loader and clipped into the
        box. NOW the start and the solve share an actuator, so start `warm`'s
        first evaluation is the historical arm's map passed through the design
        filter, in the historical arm's own channel.

BUDGET ACCOUNTING, stated because it is a deviation worth naming. Each start
gets the full 40 forward-equivalents, so a shape costs 80 in total.
`MULTISTART_REPORT.md` Section 4.2 measured that splitting one 40-equivalent
budget across starts costs more depth than the breadth is worth on 12 of 18
shapes, so splitting is not the right control here. The budget-matched arm is
therefore reported SEPARATELY: `EPS_cold_4bpp` used exactly 40
forward-equivalents and is the arm directly comparable to the 40-equivalent
`out_lib` and `out_ms` arms. `EPS_best_4bpp` is the better of the two starts
and carries an 80-equivalent price tag that is quoted every time it is.

Run:
  ./.venv312/bin/python -m adjoint2d.eps_solve <shape> <outdir> [budget] [sigma_cells]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import adjoint, control as ctl, design_filter as df
from . import energy_gate as eg, forward as fwd, gradops
from . import library_solve as lib
from . import multistart as ms
from . import printability as pq
from . import shape_objective as so
from .ms_solve import budget_ratio
from .pins import build_case, load_cfg
from .verify_hist import load_stored_map

BUDGET_FORWARD_EQUIVALENTS = 40.0
SIGMA_CELLS = df.DEFAULT_SIGMA_CELLS
BOX = (0.0, 1.0)
OUT_LIB = Path(__file__).resolve().parents[1] / "out_lib"
OUT_MS = Path(__file__).resolve().parents[1] / "out_ms"
EPS_COVARY = True


# ---------------------------------------------------------------------------
# forward and scoring, both in the permittivity channel
# ---------------------------------------------------------------------------

def run_forward(case, s, checkpoints: bool = False, eps_covary: bool = EPS_COVARY):
    return fwd.forward(case, s, keep_checkpoints=checkpoints, stop_after_phi=None,
                       shape_stop_patience=lib.PATIENCE, eps_covary=eps_covary)


def score_trajectory(case, tr, s: np.ndarray, eps_covary: bool = EPS_COVARY) -> dict:
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["mean_rho_rel_part_at_end"] = float(tr.mean_rho_rel_part[tr.n_outer - 1])
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["eps_covary"] = bool(eps_covary)
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_std_in_part"] = float(np.std(s[case.part_mask]))
    m["sat_min_in_part"] = float(np.min(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    m["map_roughness"] = df.roughness_in_part(s, case.part_mask)
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    return m


def score(case, s: np.ndarray, eps_covary: bool = EPS_COVARY) -> dict:
    t0 = time.perf_counter()
    tr = run_forward(case, s, eps_covary=eps_covary)
    m = score_trajectory(case, tr, s, eps_covary)
    m["wall_s"] = time.perf_counter() - t0
    del tr
    return m


# ---------------------------------------------------------------------------
# one full-depth start
# ---------------------------------------------------------------------------

def solve_start(case, ops, v0: np.ndarray, n_evals: int, sigma_cells: float,
                log, name: str) -> tuple[list[dict], dict[int, np.ndarray]]:
    """L-BFGS-B on dJ_phi/dv, `n_evals` objective-plus-gradient evaluations."""
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
        s = df.apply_filter(v, pm, sigma_cells)
        tr = run_forward(case, s, checkpoints=True)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        g_s = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops,
                               eps_covary=EPS_COVARY)
        g = df.filter_vjp(g_s, pm, sigma_cells)
        row = {"eval_index": len(rows) + 1, "start": name, "J": float(J),
               "t_stop_index": int(st.index), "t_stop_s": float(st.time_s),
               "t_stop_at_horizon": bool(st.at_horizon),
               "grad_max_abs": float(np.max(np.abs(g[pm]))),
               "grad_norm": float(np.linalg.norm(g[pm])),
               "IoU": float(so.region_metrics(tr.T_at_end(st.index), case)["IoU"])}
        rows.append(row)
        store[row["eval_index"]] = v.copy()
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
            f"{b['J']:.2f}, IoU {b['IoU']:.4f}, {time.perf_counter() - t0:.0f} s")
    return rows, store


# ---------------------------------------------------------------------------
# the warm start
# ---------------------------------------------------------------------------

def warm_start(case, cfg, shape: str, log) -> tuple[np.ndarray | None, dict]:
    p = OUT_LIB / f"{shape}.json"
    if not p.exists():
        log("start warm: SKIPPED LOUDLY, no out_lib record for this shape")
        return None, {}
    j = json.loads(p.read_text())
    h = j.get("arms", {}).get("HIST_best")
    if not (h and h.get("map_npz")):
        log("start warm: SKIPPED LOUDLY, no stored historical arm in out_lib")
        return None, {}
    s_h = load_stored_map(case, Path(h["map_npz"]), cfg)
    if h.get("convention") == "outside1":
        s_h = np.where(case.part_mask, s_h, 1.0)
    meta = {"map_npz": h["map_npz"], "convention": h.get("convention"),
            "source_arm": j.get("best_hist_arm"),
            "library_J_phi": h.get("J"), "library_IoU": h.get("IoU"),
            "library_eps_covary": h.get("eps_covary"),
            "note": "the historical arm was scored in the permittivity-co-varying "
                    "channel and THIS SOLVE NOW ACTUATES THE SAME CHANNEL, so the "
                    "start and the solve are actuator matched for the first time"}
    return ms.start_from_map(s_h, case.part_mask, BOX), meta


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(shape: str, outdir: str, budget: float = BUDGET_FORWARD_EQUIVALENTS,
         sigma_cells: float = SIGMA_CELLS) -> dict:
    if shape not in lib.SHAPES:
        raise ValueError(f"{shape!r} is not in the standardized library; "
                         + (lib.GT_LOGO_SKIP_REASON if shape == "gt_logo" else ""))
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    def log(msg):
        print(f"[{shape}] {msg}", flush=True)

    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)

    res: dict = {
        "shape": shape, "config": str(cfg_path), "mode": "eps_covary_full_depth",
        "channel": "permittivity co-varying (fgm_feedback.saturation_map_npz); "
                   "sigma AND eps_r both blend with fill_frac*s",
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "n_part_cells": case.n_part, "n_grid": int(pm.shape[0]),
        "budget_forward_equivalents_per_start": float(budget),
        "sigma_cells": float(sigma_cells), "box": list(BOX),
        "stop_convention": "t_stop = argmin over the arm's own trajectory of J_phi; "
                           "at_horizon flagged; melted region phi >= 0.5",
        "arms": {}, "starts": {},
    }
    maps_store: dict[str, np.ndarray] = {"part_mask": pm.astype(np.uint8)}

    # --- cost model; the uniform arm comes free -----------------------------
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = run_forward(case, s_u, checkpoints=True)
    t_b = time.perf_counter()
    st_u = so.optimal_stop(tr_u, case)
    _J, seed = so.shape_J_and_seed(tr_u.T_at_end(st_u.index), case)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed}, grad_ops=ops,
                     eps_covary=EPS_COVARY)
    t_d = time.perf_counter()
    ratio, ratio_info = budget_ratio(shape, (t_d - t_c) / max(t_b - t_a, 1e-9), log)
    n_evals = ctl.max_gradient_evals(float(budget), ratio)
    m_u = score_trajectory(case, tr_u, s_u)
    m_u["arm"] = "U_uniform"
    res["arms"]["U_uniform"] = m_u
    maps_store["U_uniform"] = s_u
    del tr_u
    log(f"cost: forward {t_b - t_a:.1f} s, adjoint {t_d - t_c:.1f} s, ratio {ratio:.2f} "
        f"({ratio_info['source']}), pool {n_evals} gradient evaluations PER START")
    log(f"U_uniform J {m_u['J']:.2f} IoU {m_u['IoU']:.4f} P {m_u['P_abs_W_per_m']:.1f}")

    res["cost"] = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c, "ratio": ratio,
                   "ratio_info": ratio_info, "n_gradient_evals_per_start": n_evals}

    # The uniform map is the SAME in both channels (s = 1 makes fill*s = fill),
    # so this arm must reproduce the conductivity-only campaigns exactly. That
    # is checked, not asserted.
    ref: dict = {}
    p_lib = OUT_LIB / f"{shape}.json"
    if p_lib.exists():
        j = json.loads(p_lib.read_text())
        for a in ("U_uniform", "HIST_best", "A1_cont", "A1_4bpp"):
            if a in j.get("arms", {}):
                ref[a] = j["arms"][a]
        ref["_source"] = str(p_lib)
        ref["_best_hist_arm"] = j.get("best_hist_arm")
    p_ms = OUT_MS / f"{shape}.json"
    if p_ms.exists():
        jm = json.loads(p_ms.read_text())
        ref["MS_4bpp"] = jm.get("arms", {}).get("MS_4bpp")
        ref["_ms_source"] = str(p_ms)
    p_ctl = OUT_MS / f"{shape}_control_cold.json"
    if p_ctl.exists():
        jc = json.loads(p_ctl.read_text())
        ref["MS_control_cold_4bpp"] = jc.get("arms", {}).get("MS_4bpp")
        ref["_ms_control_source"] = str(p_ctl)
    res["reference"] = ref
    if "U_uniform" in ref:
        res["uniform_channel_invariance_check"] = {
            "J_here_eps_channel": m_u["J"], "J_out_lib_sigma_channel": ref["U_uniform"]["J"],
            "abs_diff": abs(m_u["J"] - float(ref["U_uniform"]["J"])),
            "statement": "s = 1 gives fill*s = fill, so the uniform arm must be "
                         "identical in both channels"}
        log(f"uniform channel invariance: J {m_u['J']:.6f} against out_lib "
            f"{float(ref['U_uniform']['J']):.6f}")

    # --- the two full-depth starts -------------------------------------------
    starts: dict[str, np.ndarray] = {"cold": np.ones(pm.shape)}
    v_warm, warm_meta = warm_start(case, cfg, shape, log)
    if v_warm is not None:
        starts["warm"] = v_warm
    res["start_meta"] = {"warm": warm_meta,
                         "cold": {"rule": "uniform saturation 1"}}

    best_by_start: dict[str, dict] = {}
    for name, v0 in starts.items():
        rows, store = solve_start(case, ops, v0, n_evals, sigma_cells, log, name)
        if not rows:
            log(f"start {name}: NO EVALUATIONS, skipped loudly")
            continue
        b = min(rows, key=lambda r: r["J"])
        v_best = store[int(b["eval_index"])]
        s_cont = df.apply_filter(v_best, pm, sigma_cells)
        s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)
        m_c = score(case, s_cont)
        m_c.update({"arm": f"EPS_{name}_cont", "start": name})
        m_q = score(case, s_q)
        m_q.update({"arm": f"EPS_{name}_4bpp", "start": name})
        m_q.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm, bpp=4).items()})
        res["arms"][f"EPS_{name}_cont"] = m_c
        res["arms"][f"EPS_{name}_4bpp"] = m_q
        maps_store[f"EPS_{name}_cont"] = s_cont
        maps_store[f"EPS_{name}_4bpp"] = s_q
        maps_store[f"EPS_{name}_v"] = v_best
        res["starts"][name] = {"rows": rows, "n_evals": len(rows),
                               "spent_forward_equivalents":
                                   ctl.forward_equivalents(len(rows), len(rows), ratio)}
        best_by_start[name] = m_q
        log(f"EPS_{name}_4bpp J {m_q['J']:.2f} IoU {m_q['IoU']:.4f} "
            f"grow {m_q['bed_melt_pct_of_part']:.2f}% under {m_q['part_under_melt_pct']:.2f}% "
            f"rho@stop {m_q['mean_rho_rel_part_at_stop']:.4f} "
            f"P {m_q['P_abs_W_per_m']:.1f} stop {m_q['t_stop_s']:.1f} s"
            f"{' HORIZON' if m_q['t_stop_at_horizon'] else ''}")

    if not best_by_start:
        raise RuntimeError(f"{shape}: no start produced an evaluation")
    winner = min(best_by_start, key=lambda k: best_by_start[k]["J"])
    res["winner_start"] = winner
    res["arms"]["EPS_best_4bpp"] = dict(best_by_start[winner], arm="EPS_best_4bpp")
    res["arms"]["EPS_best_cont"] = dict(res["arms"][f"EPS_{winner}_cont"],
                                        arm="EPS_best_cont")
    maps_store["EPS_best_4bpp"] = maps_store[f"EPS_{winner}_4bpp"]
    maps_store["EPS_best_cont"] = maps_store[f"EPS_{winner}_cont"]
    res["budget_note"] = {
        "per_start_forward_equivalents": float(budget),
        "n_starts_run": len(best_by_start),
        "total_forward_equivalents": float(budget) * len(best_by_start),
        "budget_matched_arm": "EPS_cold_4bpp",
        "statement": "EPS_cold_4bpp used the same 40 forward-equivalents the out_lib "
                     "and out_ms arms used and is the like-for-like comparison; "
                     "EPS_best_4bpp is the better of two full-depth starts and cost "
                     "twice that"}

    # --- verdict -------------------------------------------------------------
    cmp: dict = {}
    for arm_name in ("EPS_cold_4bpp", "EPS_best_4bpp"):
        d = res["arms"].get(arm_name)
        if d is None:
            continue
        cmp[arm_name] = {}
        for bname, r in (("U_uniform_here", m_u), ("HIST_best", ref.get("HIST_best")),
                         ("LIB_A1_4bpp", ref.get("A1_4bpp")),
                         ("MS_4bpp", ref.get("MS_4bpp")),
                         ("MS_control_cold_4bpp", ref.get("MS_control_cold_4bpp"))):
            if r is None:
                continue
            cmp[arm_name][bname] = {
                "beats_on_J": bool(d["J"] < float(r["J"])),
                "beats_on_IoU": bool(d["IoU"] > float(r["IoU"])),
                "dJ_rel": (float(r["J"]) - d["J"]) / max(abs(float(r["J"])), 1e-30),
                "dIoU": d["IoU"] - float(r["IoU"])}
    dbest = res["arms"]["EPS_best_4bpp"]
    res["verdict"] = {
        "deliverable_arm": "EPS_best_4bpp", "winner_start": winner,
        "class": (lib.classify(dbest["IoU"], dbest["J"], float(ref["HIST_best"]["IoU"]),
                               float(ref["HIST_best"]["J"]))
                  if "HIST_best" in ref else None),
        "class_budget_matched": (
            lib.classify(res["arms"]["EPS_cold_4bpp"]["IoU"],
                         res["arms"]["EPS_cold_4bpp"]["J"],
                         float(ref["HIST_best"]["IoU"]), float(ref["HIST_best"]["J"]))
            if ("HIST_best" in ref and "EPS_cold_4bpp" in res["arms"]) else None),
        "reaches_nominal_at_grid_120": bool(dbest["IoU"] >= lib.SOLVED_IOU),
        "against": cmp,
    }
    res["energy_gate_violations"] = [a for a, m in res["arms"].items()
                                     if not m["energy_gate"]["PASS"]]
    res["stop_at_horizon_arms"] = [a for a, m in res["arms"].items()
                                   if m["t_stop_at_horizon"]]
    res["wall_s"] = time.perf_counter() - t_start

    np.savez_compressed(out / f"{shape}_maps.npz", x=case.x, y=case.y, **maps_store)
    (out / f"{shape}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done: winner {winner}, class {res['verdict']['class']}, "
        f"wall {res['wall_s']:.0f} s, energy gate violations "
        f"{res['energy_gate_violations'] or 'none'}, horizon arms "
        f"{res['stop_at_horizon_arms'] or 'none'}")
    return res


if __name__ == "__main__":
    _shape = sys.argv[1]
    _outdir = sys.argv[2]
    _budget = float(sys.argv[3]) if len(sys.argv) > 3 else BUDGET_FORWARD_EQUIVALENTS
    _sigma = float(sys.argv[4]) if len(sys.argv) > 4 else SIGMA_CELLS
    main(_shape, _outdir, _budget, _sigma)
