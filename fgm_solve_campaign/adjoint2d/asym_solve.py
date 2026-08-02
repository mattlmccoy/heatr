"""Solve the DENSE-IF-AND-ONLY-IF-IN-BOUNDS objective, both optimizers.

One shape per invocation. Every arm is read at ITS OWN J_asym stop, which is
the argmin over that arm's own stored trajectory (the out-of-bounds term rises
with time, the in-bounds deficit falls, so the argmin is interior and the
envelope theorem applies). The melt-region objective J_phi is also scored at
ITS own argmin on every arm, so the two objectives can be compared without
either being read at the other's stop by accident.

THE ARMS.

  U_uniform     uniform saturation s = 1, the control.
  PHI4_prev     the existing melt-region-solved 4-bits-per-pixel map of the
                library campaign (`out_lib/<shape>_maps.npz`, key `A1_4bpp`),
                RE-SCORED under J_asym at its own J_asym stop. This is the
                question "does the old objective already satisfy the new
                specification".
  ASYM_lbfgsb   solved on J_asym with L-BFGS-B (limited-memory
                Broyden-Fletcher-Goldfarb-Shanno with box constraints), the
                production recipe: filtered design variable at the 1.0 mm
                physical length, box [0, 1], single cold start from uniform.
  ASYM_mma      the same objective, the same budget, the same start, solved
                with the method of moving asymptotes (`mma.py`).
  *_4bpp        each solved continuous map quantized to 16 levels inside the
                part and re-run through the real forward. THE deliverable.

THE TRADE CURVE IS FREE, and that is a property of the objective rather than a
shortcut. J_asym = w_out * J_out + w_in * J_in with both parts read at the SAME
index, so sweeping the out-of-bounds price w_out over a stored pair of curves
recovers the exact stop and the exact objective at every price from ONE forward
run. `trade_curve` does that and `test_asym_solve_logic.py` pins the identity.

Run:
  ./.venv312/bin/python -m adjoint2d.asym_solve <shape> <outdir> [budget]
                                                [floor] [w_out] [tag]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, asym_objective as ao, control, design_filter as df
from . import energy_gate as eg, forward as fwd, gradops
from . import library_solve as lib, mma as mma_mod
from . import printability as pq
from . import shape_objective as so
from . import topopt_stage as ts
from .pins import build_case, load_cfg

BUDGET_FORWARD_EQUIVALENTS = 40.0
SIGMA_CELLS = df.DEFAULT_SIGMA_CELLS
OPTIMIZERS = ("lbfgsb", "mma")
TRADE_W_OUT = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
GROWTH_TOL_PCT = 1.0        # the acceptance rule, stated once
FLOOR_FRAC_TOL = 0.95
OUT_LIB = Path(__file__).resolve().parents[1] / "out_lib"


# ---------------------------------------------------------------------------
# pure logic (unit tested)
# ---------------------------------------------------------------------------

def better_of(a: dict | None, b: dict | None) -> dict | None:
    """The lower-J_asym of two records; either may be missing."""
    if a is None:
        return b
    if b is None:
        return a
    return b if float(b["J_asym"]) < float(a["J_asym"]) else a


def spec_verdict(growth_pct: float, frac_at_or_above_floor: float,
                 growth_tol_pct: float = GROWTH_TOL_PCT,
                 floor_frac_tol: float = FLOOR_FRAC_TOL) -> dict:
    """DENSE IF AND ONLY IF IN BOUNDS, as a pass/fail with explicit tolerances.

    The specification has two halves and an arm has to clear both: essentially
    no melt outside the nominal bounds, and essentially every in-bounds cell at
    or above the density floor. The tolerances are conventions and are carried
    in the returned record so no table can quote a PASS without them.
    """
    no_growth = bool(float(growth_pct) <= float(growth_tol_pct))
    in_ok = bool(float(frac_at_or_above_floor) >= float(floor_frac_tol))
    return {"no_growth_ok": no_growth, "in_bounds_ok": in_ok,
            "PASS": bool(no_growth and in_ok),
            "growth_tol_pct": float(growth_tol_pct),
            "floor_frac_tol": float(floor_frac_tol)}


def trade_curve(J_out_curve, J_in_curve, w_list=TRADE_W_OUT) -> list[dict]:
    """The exchange-rate sweep, recovered exactly from one forward run."""
    jo = np.asarray(J_out_curve, dtype=float)
    ji = np.asarray(J_in_curve, dtype=float)
    if jo.shape != ji.shape:
        raise ValueError(f"curve length mismatch: {jo.shape} against {ji.shape}")
    rows = []
    for w in w_list:
        tot = float(w) * jo + ji
        i = int(np.argmin(tot))
        rows.append({"w_out": float(w), "index": i, "J_asym": float(tot[i]),
                     "J_out_unweighted": float(jo[i]), "J_in": float(ji[i]),
                     "at_horizon": bool(i == tot.size - 1)})
    return rows


# ---------------------------------------------------------------------------
# forward and scoring
# ---------------------------------------------------------------------------

def run_forward(case, s, eps_covary: bool = False):
    """Full horizon, both checkpoint sets, no early stop.

    The shape-fidelity early stop is DISABLED on every arm: it truncates the
    march on the melt objective, and the J_asym argmin sits LATER than the melt
    stop whenever the density term still has anything to gain.
    """
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=None, eps_covary=eps_covary)


def score_asym(case, s, floor: float, w_out: float, w_in: float,
               eps_covary: bool = False, w_list=TRADE_W_OUT) -> dict:
    """One forward run, read at the J_asym stop, plus the J_phi cross read."""
    pm = np.asarray(case.part_mask, dtype=bool)
    tr = run_forward(case, s, eps_covary=eps_covary)
    m = ao.full_metrics(tr, case, floor=floor, w_out=w_out, w_in=w_in)
    st = ao.asym_stop(tr, case, floor=floor, w_out=w_out, w_in=w_in)

    # the melt-region objective at ITS own stop, for the cross comparison
    sp = so.optimal_stop(tr, case)
    m["J_phi"] = float(sp.J)
    m["phi_stop_index"] = int(sp.index)
    m["phi_stop_s"] = float(sp.time_s)
    m["phi_stop_at_horizon"] = bool(sp.at_horizon)
    m["stop_gap_steps"] = int(st.index - sp.index)
    for k, v in so.region_metrics(tr.T_at_end(sp.index), case).items():
        m[f"{k}_at_phi_stop"] = v
    m["J_asym_at_phi_stop"] = float(ao.J_and_seeds(
        tr.T_at_end(sp.index), tr.rho_at_end(sp.index), case,
        floor=floor, w_out=w_out, w_in=w_in)[0])
    m["mean_rho_rel_part_at_phi_stop"] = float(np.mean(tr.rho_at_end(sp.index)[pm]))
    m["J_phi_at_asym_stop"] = float(so.shape_J_and_seed(tr.T_at_end(st.index), case)[0])

    m["trade_curve"] = trade_curve(st.J_out_curve, st.J_in_curve, w_list)
    m["spec"] = spec_verdict(m["growth_pct"], m["frac_part_at_or_above_floor"])
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = int(tr.n_outer)
    m["eps_covary"] = bool(eps_covary)
    m["sat_mean_in_part"] = float(np.mean(s[pm]))
    m["sat_std_in_part"] = float(np.std(s[pm]))
    m["sat_min_in_part"] = float(np.min(s[pm]))
    m["map_roughness"] = df.roughness_in_part(s, pm)
    m["energy_gate_at_asym_stop"] = eg.gate_from_trajectory(tr, st.index)
    m["energy_gate_at_phi_stop"] = eg.gate_from_trajectory(tr, sp.index)
    m["J_asym_curve_stride10"] = [float(x) for x in st.J_curve[::10]]
    m["J_out_curve_stride10"] = [float(x) for x in st.J_out_curve[::10]]
    m["J_in_curve_stride10"] = [float(x) for x in st.J_in_curve[::10]]
    del tr
    return m


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

def solve_asym(case, ops, n_evals: int, box, v_init: np.ndarray,
               sigma_cells: float, floor: float, w_out: float, w_in: float,
               optimizer: str = "lbfgsb") -> tuple[list[dict], dict, np.ndarray, dict]:
    """One budgeted solve under one optimizer, against one evaluate callable.

    Both optimizers consume the SAME closure through `topopt_stage.StageRunner`,
    which is what makes the comparison a comparison of optimizers rather than of
    objectives.
    """
    pm = np.asarray(case.part_mask, dtype=bool)
    idx = np.flatnonzero(pm.ravel())
    rows: list[dict] = []

    def unpack(vec):
        v = np.ones(pm.shape)
        v.ravel()[idx] = vec
        return v

    def evaluate(vec):
        v = unpack(vec)
        s = df.apply_filter(v, pm, sigma_cells)
        tr = run_forward(case, s)
        st = ao.asym_stop(tr, case, floor=floor, w_out=w_out, w_in=w_in)
        J, sT, sr, _p = ao.J_and_seeds(tr.T_at_end(st.index), tr.rho_at_end(st.index),
                                       case, floor=floor, w_out=w_out, w_in=w_in)
        g_s = adjoint.gradient(case, s, tr, {st.index: sT}, grad_ops=ops,
                               seeds_rho={st.index: sr})
        g = df.filter_vjp(g_s, pm, sigma_cells)
        h = ao.deficit(tr.rho_at_end(st.index), case, floor)
        rows.append({"eval_index": len(rows) + 1, "J_asym": float(J),
                     "J_out": float(st.J_out), "J_in": float(st.J_in),
                     "stop_index": int(st.index), "stop_s": float(st.time_s),
                     "at_horizon": bool(st.at_horizon),
                     "in_term_dead": bool(st.in_term_dead),
                     "hinge_active_frac": float(np.mean(h[pm] > 0.0)),
                     "grad_max_abs": float(np.max(np.abs(g[pm]))),
                     "grad_norm": float(np.linalg.norm(g[pm])),
                     "mean_rho_rel_part": float(np.mean(tr.rho_at_end(st.index)[pm]))})
        del tr
        return float(J), g.ravel()[idx].astype(float)

    v0 = np.clip(np.asarray(v_init, dtype=float).ravel()[idx], box[0], box[1])
    state = (mma_mod.MMA(v0, box[0], box[1]) if optimizer == "mma" else None)
    runner = ts.StageRunner(evaluate, box, optimizer=optimizer, mma_state=state)
    t0 = time.perf_counter()
    _vend, info = runner.run(v0, int(n_evals))
    info = dict(info)
    info["wall_s"] = time.perf_counter() - t0
    info["n_evals_used"] = len(rows)
    if not rows:
        return rows, info, np.ones(pm.shape), {}
    best = min(rows, key=lambda r: r["J_asym"])
    v_best = unpack(runner.points[best["eval_index"]])
    n_after = max(len(rows) - 1, 0)
    improved = sum(1 for k in range(1, len(rows))
                   if rows[k]["J_asym"] < min(r["J_asym"] for r in rows[:k]))
    info["frac_followups_improving"] = (improved / n_after) if n_after else float("nan")
    if state is not None:
        info["mma_config"] = state.cfg.__dict__
    return rows, info, v_best, best


# ---------------------------------------------------------------------------
# reference maps
# ---------------------------------------------------------------------------

def previous_phi_map(shape: str) -> np.ndarray | None:
    """The melt-region-solved 4-bits-per-pixel deliverable of the library pass."""
    p = OUT_LIB / f"{shape}_maps.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        if "A1_4bpp" not in d.files:
            return None
        return np.asarray(d["A1_4bpp"], dtype=float)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(shape: str, outdir: str, budget: float = BUDGET_FORWARD_EQUIVALENTS,
         floor: float = ao.FLOOR_RHO_REL_DEFAULT, w_out: float = ao.W_OUT_DEFAULT,
         tag: str = "", w_in: float = ao.W_IN_DEFAULT,
         sigma_cells: float = SIGMA_CELLS,
         optimizers: tuple[str, ...] = OPTIMIZERS) -> dict:
    if shape not in lib.SHAPES:
        raise ValueError(f"{shape!r} is not in the standardized library")
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{shape}{tag}"
    t_start = time.perf_counter()

    def log(msg):
        print(f"[{stem}] {msg}", flush=True)

    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = np.asarray(case.part_mask, dtype=bool)
    ops = gradops.gradient_matrices(case.x, case.y)

    res: dict = {
        "shape": shape, "tag": tag, "config": str(cfg_path),
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "n_part_cells": int(case.n_part), "n_steps": int(case.pins.n_steps),
        "grid": [int(case.part_mask.shape[0]), int(case.part_mask.shape[1])],
        "budget_forward_equivalents_per_optimizer": float(budget),
        "sigma_cells": float(sigma_cells),
        "floor_rho_rel": float(floor), "w_out": float(w_out), "w_in": float(w_in),
        "rho_floor": ao.rho_floor(case),
        "stop_convention": {
            "asym": "argmin over the arm's own stored trajectory of J_asym",
            "phi": "argmin over the arm's own stored trajectory of J_phi",
            "shape_early_stop": "DISABLED on every arm (full horizon)"},
        "arms": {}, "solves": {},
    }
    maps_store: dict[str, np.ndarray] = {"part_mask": pm.astype(np.uint8)}

    # --- cost model, measured on this shape --------------------------------
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = run_forward(case, s_u)
    t_b = time.perf_counter()
    st_u = ao.asym_stop(tr_u, case, floor=floor, w_out=w_out, w_in=w_in)
    _J, sT, sr, _p = ao.J_and_seeds(tr_u.T_at_end(st_u.index),
                                    tr_u.rho_at_end(st_u.index), case,
                                    floor=floor, w_out=w_out, w_in=w_in)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: sT}, grad_ops=ops,
                     seeds_rho={st_u.index: sr})
    t_d = time.perf_counter()
    del tr_u
    ratio = (t_d - t_c) / max(t_b - t_a, 1e-9)
    n_each = control.max_gradient_evals(float(budget), ratio)
    res["cost"] = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c, "ratio": ratio,
                   "n_gradient_evals_per_optimizer": n_each,
                   "spent_forward_equivalents_per_optimizer":
                       control.forward_equivalents(n_each, n_each, ratio)}
    log(f"cost: forward {t_b - t_a:.1f} s, adjoint {t_d - t_c:.1f} s, "
        f"ratio {ratio:.2f}, {n_each} gradient evaluations per optimizer")

    # --- control arms -------------------------------------------------------
    m_u = score_asym(case, s_u, floor, w_out, w_in)
    m_u["arm"] = "U_uniform"
    res["arms"]["U_uniform"] = m_u
    maps_store["U_uniform"] = s_u
    log(f"U_uniform J_asym {m_u['J_asym']:.5f} (out {m_u['J_asym_out']:.5f} + "
        f"in {m_u['J_asym_in']:.5f}) stop {m_u['asym_stop_s']:.0f} s IoU "
        f"{m_u['IoU']:.4f} growth {m_u['growth_pct']:.2f}% rho "
        f"{m_u['mean_rho_rel_part']:.4f} above-floor "
        f"{m_u['frac_part_at_or_above_floor']:.3f}")

    s_phi4 = previous_phi_map(shape)
    if s_phi4 is not None:
        m_p = score_asym(case, s_phi4, floor, w_out, w_in)
        m_p["arm"] = "PHI4_prev"
        res["arms"]["PHI4_prev"] = m_p
        maps_store["PHI4_prev"] = s_phi4
        log(f"PHI4_prev J_asym {m_p['J_asym']:.5f} IoU {m_p['IoU']:.4f} growth "
            f"{m_p['growth_pct']:.2f}% rho {m_p['mean_rho_rel_part']:.4f} "
            f"above-floor {m_p['frac_part_at_or_above_floor']:.3f} "
            f"spec {m_p['spec']['PASS']}")
    else:
        log("PHI4_prev: SKIPPED, no library melt-solved map for this shape")

    # --- the solves ---------------------------------------------------------
    for opt in optimizers:
        rows, info, v_best, best = solve_asym(case, ops, n_each, (0.0, 1.0),
                                              np.ones(pm.shape), sigma_cells,
                                              floor, w_out, w_in, optimizer=opt)
        res["solves"][opt] = {"info": info, "rows": rows}
        if not rows:
            log(f"{opt}: no evaluations completed")
            continue
        log(f"{opt}: {len(rows)} evals, J_asym {rows[0]['J_asym']:.5f} -> "
            f"{best['J_asym']:.5f}, stop {rows[0]['stop_s']:.0f} -> "
            f"{best['stop_s']:.0f} s, grad_max {rows[0]['grad_max_abs']:.3e} -> "
            f"{rows[-1]['grad_max_abs']:.3e}, "
            f"{info.get('frac_followups_improving', float('nan')):.3f} of follow-ups "
            f"improved, {info['wall_s']:.0f} s")
        s_cont = df.apply_filter(v_best, pm, sigma_cells)
        m_c = score_asym(case, s_cont, floor, w_out, w_in)
        m_c["arm"] = f"ASYM_{opt}_cont"
        res["arms"][f"ASYM_{opt}_cont"] = m_c
        maps_store[f"ASYM_{opt}_cont"] = s_cont
        maps_store[f"ASYM_{opt}_v"] = v_best

        s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)
        m_q = score_asym(case, s_q, floor, w_out, w_in)
        m_q["arm"] = f"ASYM_{opt}_4bpp"
        m_q.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm, bpp=4).items()})
        res["arms"][f"ASYM_{opt}_4bpp"] = m_q
        maps_store[f"ASYM_{opt}_4bpp"] = s_q
        log(f"ASYM_{opt}_4bpp J_asym {m_q['J_asym']:.5f} (out {m_q['J_asym_out']:.5f} "
            f"+ in {m_q['J_asym_in']:.5f}) stop {m_q['asym_stop_s']:.0f} s IoU "
            f"{m_q['IoU']:.4f} growth {m_q['growth_pct']:.2f}% under "
            f"{m_q['under_pct']:.2f}% rho {m_q['mean_rho_rel_part']:.4f} "
            f"above-floor {m_q['frac_part_at_or_above_floor']:.3f} "
            f"spec {m_q['spec']['PASS']}")

    # --- verdict ------------------------------------------------------------
    deliverables = [a for a in res["arms"] if a.endswith("_4bpp")]
    best_arm = min(deliverables, key=lambda a: res["arms"][a]["J_asym"]) \
        if deliverables else None
    ref = res["arms"]["U_uniform"]
    res["verdict"] = {
        "best_deliverable": best_arm,
        "J_asym_by_arm": {a: res["arms"][a]["J_asym"] for a in res["arms"]},
        "beats_uniform": {a: bool(res["arms"][a]["J_asym"] < ref["J_asym"])
                          for a in res["arms"]},
        "spec_pass_by_arm": {a: res["arms"][a]["spec"]["PASS"] for a in res["arms"]},
        "any_stop_at_horizon": [a for a, m in res["arms"].items()
                                if m["asym_stop_at_horizon"]],
        "any_in_term_dead": [a for a, m in res["arms"].items()
                             if m["asym_in_term_dead"]],
        "any_stop_first_step": [a for a, m in res["arms"].items()
                                if m["asym_stop_is_first_step"]],
    }
    res["energy_gate_violations"] = [
        a for a, m in res["arms"].items()
        if not m["energy_gate_at_asym_stop"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t_start

    np.savez_compressed(out / f"{stem}_maps.npz", x=case.x, y=case.y, **maps_store)
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done, wall {res['wall_s']:.0f} s, best {best_arm}, energy gate violations "
        f"{res['energy_gate_violations'] or 'none'}")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else BUDGET_FORWARD_EQUIVALENTS,
         float(sys.argv[4]) if len(sys.argv) > 4 else ao.FLOOR_RHO_REL_DEFAULT,
         float(sys.argv[5]) if len(sys.argv) > 5 else ao.W_OUT_DEFAULT,
         sys.argv[6] if len(sys.argv) > 6 else "")
