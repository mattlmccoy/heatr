#!/usr/bin/env python3
"""END-TO-END ACCEPTANCE TEST of the geometry generalization layer.

One geometry that is NOT in the eighteen-shape standardized library goes in as
a bare vertex list and the whole pipeline runs on it with no hand-coded
knowledge of the shape anywhere:

  intake  ->  drive calibration  ->  symmetry  ->  anisotropy spectrum  ->
  actuator recommendation  ->  filtered static solve (production recipe,
  40 forward-equivalents)  ->  the recommended mode's co-solve if the
  classifier says rotation helps  ->  4 bits per pixel deliverable  ->
  turntable program.

CONVENTIONS, carried on every number below and in the emitted JSON.
Objective J(s, t_stop) = sum over the WHOLE domain of (phi - chi)^2 with chi
the grid-independent sub-cell AREA FILL indicator (`adjoint2d.chi_area`), so J
is NOT numerically comparable to any J quoted against the binary raster in
`out_lib` or `out_rot`; `J_raster_chi` is reported alongside for that
comparison. t_stop = argmin of J over that arm's OWN stored trajectory, with
`t_stop_at_horizon` flagged, which makes that arm's J a bound. The melted
region for intersection over union (IoU) is phi >= 0.5. GRID 120 throughout:
`SOLVE_ROBUSTNESS_VALIDATION.md` established that grid-120 fidelity does not
transfer to grid 160, so every fidelity number here is a property of the method
AT GRID 120. Absorbed power is calibrated to 500 watts per metre on the UNIFORM
arm only; solved arms are NOT dose matched. Conductivity channel only.

Run:
  ./.venv312/bin/python scripts/analysis/run_intake_novel.py gear8 [budget]
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

from adjoint2d import adjoint, control as ctl, design_filter as df   # noqa: E402
from adjoint2d import dwell, energy_gate as eg, forward as fwd       # noqa: E402
from adjoint2d import geometry_actuator as ga                        # noqa: E402
from adjoint2d import geometry_calibrate as gcal                     # noqa: E402
from adjoint2d import geometry_intake as gi                          # noqa: E402
from adjoint2d import geometry_symmetry as gs                        # noqa: E402
from adjoint2d import gradops, library_solve as lib                  # noqa: E402
from adjoint2d import printability as pq, topopt                     # noqa: E402
from adjoint2d import topopt_objective as tobj                       # noqa: E402
from adjoint2d.pins import build_case                                # noqa: E402
from adjoint2d.rot_kernel import AveragedKernel                      # noqa: E402
from novel_shapes import NOVEL                                       # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
GRID = 120
FILTER_RADIUS_M = 1.0e-3          # FROZEN_CONVENTIONS_2D section 1.2
BOX = (0.0, 1.0)
BUDGET = lib.BUDGET_FORWARD_EQUIVALENTS
PATIENCE = lib.PATIENCE
CYCLE_TIME_S = 20.0               # the dwell campaign's cycle


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_static(case, chi, s):
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE)
    m = tobj.full_metrics(tr, case, chi)
    i = int(m["t_stop_index"])
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    m["J_raster_chi"] = tobj.optimal_stop(tr, case, case.part_mask.astype(float)).J
    m["max_T_at_stop_c"] = float(np.max(tr.T_at_end(i)))
    m["frac_qrf_cap"] = float(tr.frac_qrf_cap)
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    phi = tobj.phi_of(tr.T_at_end(i), case)
    del tr
    return m, phi


def score_mode(kern, chi, s):
    case = kern.case0
    tr = kern.forward(s, keep_checkpoints=False, shape_stop_patience=PATIENCE)
    m = tobj.full_metrics(tr, case, chi)
    i = int(m["t_stop_index"])
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    m["J_raster_chi"] = tobj.optimal_stop(tr, case, case.part_mask.astype(float)).J
    m["max_T_at_stop_c"] = float(np.max(tr.T_at_end(i)))
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    phi = tobj.phi_of(tr.T_at_end(i), case)
    del tr
    return m, phi


# ---------------------------------------------------------------------------
# the two solves, same recipe, different forward
# ---------------------------------------------------------------------------

def _lbfgs(fun, v0, idx, pool):
    try:
        minimize(fun, np.clip(v0.ravel()[idx], *BOX), jac=True, method="L-BFGS-B",
                 bounds=[BOX] * len(idx),
                 options={"maxiter": 10_000, "maxfun": 10_000,
                          "ftol": 1e-16, "gtol": 1e-16})
    except StopIteration:
        pass


def solve_static(case, chi, ops, sigma_cells, pool, log):
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
    rows, store = [], {}

    def fun(vec):
        if len(rows) >= pool:
            raise StopIteration
        v = np.ones(pm.shape)
        v.ravel()[idx] = vec
        s = df.apply_filter(v, pm, sigma_cells)
        tr = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                         shape_stop_patience=PATIENCE)
        st = tobj.optimal_stop(tr, case, chi)
        J, seed = tobj.J_and_seed(tr.T_at_end(st.index), case, chi)
        g = df.filter_vjp(adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops),
                          pm, sigma_cells)
        del tr
        rows.append({"eval_index": len(rows) + 1, "J": float(J),
                     "t_stop_index": int(st.index),
                     "t_stop_at_horizon": bool(st.at_horizon)})
        store[rows[-1]["eval_index"]] = v.copy()
        log(f"    static eval {rows[-1]['eval_index']:2d}/{pool} J {J:.2f}")
        return float(J), g.ravel()[idx].astype(float)

    _lbfgs(fun, np.ones(pm.shape), idx, pool)
    best = min(rows, key=lambda r: r["J"])
    v_best = store[best["eval_index"]]
    return df.apply_filter(v_best, pm, sigma_cells), rows, v_best


def solve_mode(kern, chi, ops, sigma_cells, pool, log, tag, starts):
    """The mode co-solve, one L-BFGS-B run per start, better kept.

    TWO STARTS AND WHY. Measured on the gear in this pass: under a rotation
    mode the uniform design v = 1 can be a CONSTRAINED stationary point, every
    gradient component negative (more dopant everywhere would help) while every
    component is already pinned at the upper rail of the box, so L-BFGS-B stops
    after one evaluation and the co-solve returns the uniform arm. That is the
    optimizer behaving correctly, not a bug, and it is exactly why
    `run_rot_avg_solve.py` ran a cold start AND a warm start from the best
    known static map. Here the warm start is the map the static arm of this
    same pipeline just solved, which is the only strong prior an imported
    geometry has.
    """
    case = kern.case0
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
    all_rows: list[dict] = []
    best_v, best_J = None, np.inf

    for start_name, v0 in starts.items():
        rows, store = [], {}

        def fun(vec, rows=rows, store=store, start_name=start_name):
            if len(rows) >= pool:
                raise StopIteration
            v = np.ones(pm.shape)
            v.ravel()[idx] = vec
            s = df.apply_filter(v, pm, sigma_cells)
            tr = kern.forward(s, keep_checkpoints=True, shape_stop_patience=PATIENCE)
            st = tobj.optimal_stop(tr, case, chi)
            J, seed = tobj.J_and_seed(tr.T_at_end(st.index), case, chi)
            g = df.filter_vjp(kern.gradient(s, tr, {st.index: seed}, grad_ops=ops),
                              pm, sigma_cells)
            del tr
            rows.append({"eval_index": len(rows) + 1, "start": start_name,
                         "J": float(J), "t_stop_index": int(st.index),
                         "t_stop_at_horizon": bool(st.at_horizon)})
            store[rows[-1]["eval_index"]] = v.copy()
            log(f"    {tag}/{start_name} eval {rows[-1]['eval_index']:2d}/{pool} "
                f"J {J:.2f}")
            return float(J), g.ravel()[idx].astype(float)

        _lbfgs(fun, np.asarray(v0, dtype=float), idx, pool)
        if not rows:
            log(f"    {tag}/{start_name}: NO EVALUATIONS, skipped loudly")
            continue
        b = min(rows, key=lambda r: r["J"])
        log(f"    {tag}/{start_name}: {len(rows)} evaluations, "
            f"J {rows[0]['J']:.2f} -> {b['J']:.2f}")
        all_rows.extend(rows)
        if b["J"] < best_J:
            best_J, best_v = b["J"], store[b["eval_index"]]
    if best_v is None:
        raise RuntimeError(f"{tag}: no start produced an evaluation")
    return df.apply_filter(best_v, pm, sigma_cells), all_rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(name: str, budget: float = BUDGET, spectrum_only: bool = False) -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{name}] {msg}", flush=True)

    if name not in NOVEL:
        raise SystemExit(f"{name!r} is not a novel shape; have {sorted(NOVEL)}")
    poly = NOVEL[name]()
    log(f"intake: {len(poly)} vertices, route 'polygon'")

    it = gi.from_polygon(poly, grid=GRID, name=name)
    it = gcal.calibrate_intake(it)
    cal = it.info["calibration"]
    log(f"calibration: {cal['voltage_v']:.2f} V in {cal['n_electro_quasi_static_solves']} "
        f"solves, verified {cal['p_verified_w_per_m']:.4f} W/m against target "
        f"{cal['p_target_w_per_m']:.1f}")

    rep = gs.analyze(it.chi)
    log(f"symmetry: order {rep.rotational_order} ({rep.point_group}), "
        f"{len(rep.mirror_axes_deg)} mirror axes, candidate span "
        f"{gs.candidate_span_deg(rep.rotational_order):.1f} deg")

    spec = ga.spectrum_from_cfg(it.cfg, rep.rotational_order, log=log)
    rec = ga.recommend(spec["residual"], rep.rotational_order)
    log(f"RECOMMENDATION: {rec.actuator_class} via mode {rec.mode} "
        f"(residual {rec.residual:.4f}, static {rec.static_residual:.4f}, "
        f"reduction {rec.reduction_factor:.2f}, rotation "
        f"{'RECOMMENDED' if rec.rotation_recommended else 'not recommended'})")

    res = {
        "shape": name, "grid": GRID, "budget_forward_equivalents": float(budget),
        "intake": {k: v for k, v in it.info.items() if k != "raster_vs_area"},
        "raster_vs_area": it.info["raster_vs_area"],
        "symmetry": rep.as_json(),
        "anisotropy": {k: float(v) for k, v in spec["residual"].items()},
        "recommendation": rec.as_json(),
        "conventions": {
            "objective": "J = sum over the whole domain of (phi - chi_area)^2",
            "stop": "argmin of J over the arm's own trajectory; horizon flagged",
            "grid_qualifier": "grid 120; grid-120 fidelity does not transfer to 160",
            "dose": "uniform arm calibrated to 500 W/m; solved arms NOT dose matched",
            "channel": "conductivity only (deployable)",
        },
        "arms": {},
    }
    np.savez_compressed(OUT / f"{name}_intake.npz", chi=it.chi,
                        part_mask=it.part_mask, x=it.x, y=it.y, polygon=poly,
                        **{f"kernel_{k}": v for k, v in spec["kernels"].items()})
    if spectrum_only:
        res["wall_s"] = time.perf_counter() - t0
        (OUT / f"{name}_novel.json").write_text(json.dumps(res, indent=2, default=float))
        log(f"spectrum-only pass done in {res['wall_s']:.0f} s")
        return res

    case = build_case(it.cfg)
    chi = it.chi
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = topopt.sigma_cells_for(FILTER_RADIUS_M, case.dx)
    log(f"filter radius 1.00 mm = {sigma_cells:.3f} cells at grid {GRID}")

    # cost model, and the uniform arm
    s_u = np.ones(pm.shape)
    ta = time.perf_counter()
    tr_u = fwd.forward(case, s_u, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=PATIENCE)
    tb = time.perf_counter()
    st_u = tobj.optimal_stop(tr_u, case, chi)
    _J, seed_u = tobj.J_and_seed(tr_u.T_at_end(st_u.index), case, chi)
    tc = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed_u}, grad_ops=ops)
    td = time.perf_counter()
    del tr_u
    ratio = (td - tc) / max(tb - ta, 1e-9)
    pool = max(int(ctl.max_gradient_evals(float(budget), ratio)), 1)
    res["cost"] = {"forward_s": tb - ta, "adjoint_s": td - tc, "ratio": ratio,
                   "pool_gradient_evals": pool}
    log(f"pool {pool} gradient evaluations (adjoint-to-forward ratio {ratio:.2f})")

    m_u, phi_u = score_static(case, chi, s_u)
    m_u["arm"] = "U_uniform_static"
    res["arms"]["U_uniform_static"] = m_u
    log(f"  U_uniform_static J {m_u['J']:.2f} IoU {m_u['IoU']:.4f} "
        f"P {m_u['P_abs_W_per_m']:.1f} W/m")

    s_cont, rows, v_static = solve_static(case, chi, ops, sigma_cells, pool, log)
    res["static_rows"] = rows
    m_c, _ = score_static(case, chi, s_cont)
    m_c["arm"] = "A_static_cont"
    res["arms"]["A_static_cont"] = m_c
    s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)
    m_q, phi_q = score_static(case, chi, s_q)
    m_q["arm"] = "A_static_4bpp"
    res["arms"]["A_static_4bpp"] = m_q
    log(f"  A_static_4bpp J {m_q['J']:.2f} IoU {m_q['IoU']:.4f} "
        f"under {m_q['part_under_melt_pct']:.2f}%")

    maps = {"uniform": s_u, "static_cont": s_cont, "static_4bpp": s_q,
            "phi_uniform": phi_u, "phi_static_4bpp": phi_q}

    # --- the recommended mode, only if the classifier asked for it -----------
    if rec.rotation_recommended:
        # The recommended mode, plus any INDEXING mode whose residual is within
        # NEAR_TIE of it. The rotation campaign measured that symmetry-matched
        # indexing beats the finest rotation in the engine, so a near tie in
        # the spectrum is worth spending the arm on rather than resolving on
        # the fourth decimal place.
        NEAR_TIE = 1.15
        best_a = spec["residual"][rec.mode]
        modes = [rec.mode] + [k for k, v in spec["residual"].items()
                              if k.startswith("index") and k != rec.mode
                              and v <= NEAR_TIE * best_a]
        res["modes_co_solved"] = modes
        programs = {}
        for mode in modes:
            ang = ga.mode_angles(mode)
            log(f"  mode co-solve: {mode}, {len(ang)} angles")
            kern = AveragedKernel.build(it.cfg, angles=ang)
            m_mu, phi_mu = score_mode(kern, chi, s_u)
            m_mu["arm"] = f"U_uniform_{mode}"
            res["arms"][m_mu["arm"]] = m_mu
            log(f"  U_uniform_{mode} J {m_mu['J']:.2f} IoU {m_mu['IoU']:.4f} "
                f"P {m_mu['P_abs_W_per_m']:.1f} W/m")
            starts = {"cold": np.ones(pm.shape), "warm_static": v_static}
            s_mc, rows_m = solve_mode(kern, chi, ops, sigma_cells, pool, log,
                                      mode, starts)
            res.setdefault("mode_rows", {})[mode] = rows_m
            m_mc, _ = score_mode(kern, chi, s_mc)
            m_mc["arm"] = f"A_{mode}_cont"
            res["arms"][m_mc["arm"]] = m_mc
            s_mq = pq.quantize_in_part(s_mc, pm, bpp=4, sat_max=1.0)
            m_mq, phi_mq = score_mode(kern, chi, s_mq)
            m_mq["arm"] = f"A_{mode}_4bpp"
            res["arms"][m_mq["arm"]] = m_mq
            log(f"  A_{mode}_4bpp J {m_mq['J']:.2f} IoU {m_mq['IoU']:.4f} "
                f"under {m_mq['part_under_melt_pct']:.2f}%")
            maps.update({f"{mode}_cont": s_mc, f"{mode}_4bpp": s_mq,
                         f"phi_uniform_{mode}": phi_mu,
                         f"phi_{mode}_4bpp": phi_mq})
            del kern

            # the machine-readable program, on the gauge-reduced distinct positions
            distinct = gs.gauge_reduce(ang)
            w = np.full(len(distinct), 1.0 / len(distinct))
            total_s = float(m_mq["t_stop_s"])
            prog = dwell.cycle_program(w, distinct, cycle_time_s=CYCLE_TIME_S,
                                       total_s=total_s, dt_s=float(case.pins.dt))
            programs[mode] = prog.as_json()
            programs[mode]["gauge_note"] = (
                "positions are the HALF-TURN-DISTINCT ones: theta and theta + "
                "180 degrees are the same part-frame heating to 4e-13 relative "
                "(DWELL_SCHEDULE_REPORT section 2), so the literal program is "
                "halved without changing the physics")
            log(f"  turntable program ({mode}): {len(prog.moves)} moves over "
                f"{total_s:.1f} s at {len(distinct)} distinct positions "
                f"{list(distinct)}")
        res["turntable_programs"] = programs

    # --- did the prediction hold? ---------------------------------------------
    base = res["arms"]["A_static_4bpp"]
    best_name = min(res["arms"], key=lambda k: res["arms"][k]["J"])
    res["verdict"] = {
        "predicted_class": rec.actuator_class,
        "predicted_mode": rec.mode,
        "rotation_recommended": bool(rec.rotation_recommended),
        "best_arm": best_name,
        "best_J": float(res["arms"][best_name]["J"]),
        "best_IoU": float(res["arms"][best_name]["IoU"]),
        "static_4bpp_J": float(base["J"]),
        "static_4bpp_IoU": float(base["IoU"]),
        "solved_class_at_grid_120": bool(res["arms"][best_name]["IoU"] >= lib.SOLVED_IOU),
        "prediction_held": bool(
            (not best_name.startswith("A_static") and not best_name.startswith("U_uniform_static"))
            if rec.rotation_recommended else best_name.startswith("A_static")),
        "note_on_prediction": (
            "the prediction is that the RECOMMENDED actuator class produces the "
            "best arm; when rotation is recommended, any rotating arm counts, "
            "because the near-tie indexing arms are part of the same "
            "recommendation"),
    }
    res["energy_gate_violations"] = [k for k, m in res["arms"].items()
                                     if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(OUT / f"{name}_maps.npz", part_mask=pm, x=case.x, y=case.y,
                        chi=chi, **maps)
    (OUT / f"{name}_novel.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"VERDICT best arm {best_name} J {res['verdict']['best_J']:.2f} "
        f"IoU {res['verdict']['best_IoU']:.4f}; prediction "
        f"{'HELD' if res['verdict']['prediction_held'] else 'DID NOT HOLD'}; "
        f"wall {res['wall_s']:.0f} s")
    return res


if __name__ == "__main__":
    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _name = _args[0] if _args else "gear8"
    _budget = float(_args[1]) if len(_args) > 1 else BUDGET
    _spec = "--spectrum-only" in sys.argv
    main(_name, _budget, spectrum_only=_spec)
