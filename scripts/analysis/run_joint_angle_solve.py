#!/usr/bin/env python3
"""JOINT per-angle dopant-map re-solve on the stranded shapes.

The orientation sweep (`ORIENTATION_OPTIMIZATION_REPORT.md`) scanned angles with
the dopant map solved once at zero degrees and rigidly rotated; its honest limit
2 names that as a lower bound on what orientation plus grading can do. This
driver RE-SOLVES the map at every angle, which turns the scan into a joint
optimization over orientation and dopant map, and answers whether the best angle
moves.

Objective and stop convention, carried on every number:

    J_phi(s, theta, t_stop) = sum over the WHOLE domain of
                              (phi(x, t_stop) - chi_part(x; theta))^2

with chi_part rasterized by the production engine at
`geometry.part.rotation_deg = theta`, so the nominal target co-rotates with the
part. t_stop = argmin of J_phi over that arm's OWN stored trajectory on a
1500-step horizon (dt 0.5 s, 750 s) with early truncation 250 steps past the
running minimum. `at_horizon` is flagged and makes that arm's J a bound. The
melted region for intersection over union (IoU), growth and under-melt is
phi >= 0.5. GRID 120 x 120 throughout.

Recipe, the production one: physical-length design filter on the design variable
at sigma = 1.5 cells (MANDATORY), box [0, 1], L-BFGS-B on the finite-difference
gated filtered gradient.

Two starts per angle, each at the full per-start budget, better kept:
  cold  uniform saturation 1
  warm  the best known zero-degree map for that shape in this actuator channel,
        rotated into the rotated part frame by the production convention
        (`joint_angle_lib.rotated_warm_start`, red-first tested).

Actuators:
  sigma  conductivity only. The DEPLOYMENT-SAFE primary arm, because
         `EPS_CHANNEL_REPORT.md` Section 11 leaves open whether the dopant moves
         relative permittivity at all in the real binder.
  eps    conductivity and relative permittivity co-varying. MODEL ONLY.

Run:
  ./.venv312/bin/python scripts/analysis/run_joint_angle_solve.py <shape> <actuator> [budget]
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts"))

from adjoint2d import adjoint, control as ctl, design_filter as df   # noqa: E402
from adjoint2d import energy_gate as eg, forward as fwd, gradops     # noqa: E402
from adjoint2d import library_solve as lib                           # noqa: E402
from adjoint2d import printability as pq                             # noqa: E402
from adjoint2d import shape_objective as so                          # noqa: E402
from adjoint2d.pins import build_case, load_cfg                      # noqa: E402

from analysis.joint_angle_lib import (ANGLES, SYMMETRY_PERIOD_DEG,   # noqa: E402
                                      angle_delta_deg, argmin_angle,
                                      best_row, exceeds_ceiling,
                                      rotated_warm_start)

BUDGET_FORWARD_EQUIVALENTS = 15.0     # PER START; two starts, so ~30 per angle
SIGMA_CELLS = df.DEFAULT_SIGMA_CELLS
BOX = (0.0, 1.0)
N_STEPS = 1500
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
OUT_MS = REPO / "fgm_solve_campaign/out_ms"
OUT_EPS = REPO / "fgm_solve_campaign/out_eps"
SWEEP = REPO / "outputs_eqs/orientation_optimization"
OUT_ROOT = REPO / "fgm_solve_campaign/out_joint"

# The fixed-map sweep's best angle per shape, from ORIENTATION_OPTIMIZATION_REPORT.md
# Section 3 (the graded arm, which is the arm this campaign re-solves).
SWEEP_BEST_GRADED_DEG = {"T_shape": 90.0, "L_shape": 135.0, "cross": 30.0,
                         "star": 0.0}

# Uniform-arm reference at zero degrees, the real-data driver gate.
REF_UNIFORM_AT_ZERO = {"T_shape": 614.26, "L_shape": 538.42,
                       "cross": 471.82, "star": 192.58}


# ---------------------------------------------------------------------------
# the best known zero-degree map, read from the stored campaigns
# ---------------------------------------------------------------------------

def best_known_zero_map(shape: str, actuator: str, log) -> tuple[np.ndarray, dict]:
    """The lowest-J continuous zero-degree map available in this channel.

    Conductivity-only candidates are the single-start library solve
    (`out_lib/<shape>_maps.npz:A1_cont`, box [0, 1]) and the filtered
    multi-start solve (`out_ms/<shape>_maps.npz:MS_cont`). The permittivity
    candidate is `out_eps/<shape>_maps.npz:EPS_best_cont`. The box [0, 1.5]
    arms are excluded because this campaign's box is [0, 1].
    """
    cands: list[tuple[str, Path, str, str]] = []
    if actuator == "sigma":
        cands = [("A1_cont", OUT_LIB / f"{shape}_maps.npz", "A1_cont",
                  str(OUT_LIB / f"{shape}.json")),
                 ("MS_cont", OUT_MS / f"{shape}_maps.npz", "MS_cont",
                  str(OUT_MS / f"{shape}.json"))]
    else:
        cands = [("EPS_best_cont", OUT_EPS / f"{shape}_maps.npz", "EPS_best_cont",
                  str(OUT_EPS / f"{shape}.json"))]
    best = None
    for arm, npz, key, jpath in cands:
        p = Path(jpath)
        if not (npz.exists() and p.exists()):
            continue
        arms = json.loads(p.read_text()).get("arms", {})
        if arm not in arms:
            continue
        J = float(arms[arm]["J"])
        if best is None or J < best[0]:
            best = (J, arm, npz, key, jpath, float(arms[arm]["IoU"]))
    if best is None:
        raise FileNotFoundError(f"no stored zero-degree map for {shape}/{actuator}")
    J, arm, npz, key, jpath, iou = best
    sat = np.asarray(np.load(npz)[key], dtype=float)
    meta = {"arm": arm, "npz": str(npz), "key": key, "source_json": jpath,
            "J_at_zero_deg": J, "IoU_at_zero_deg": iou}
    log(f"warm-start source: {arm} from {npz.name} (J {J:.2f}, IoU {iou:.4f} at 0 deg)")
    return sat, meta


def budget_ratio(shape: str) -> tuple[float, dict]:
    """The adjoint-to-forward cost ratio, taken from the library campaign.

    `MULTISTART_REPORT.md` Section 3.3 measured that timing one forward and one
    adjoint on a loaded machine is unreliable (a ratio of 12.91 against a true
    1.11 under contention), so the budget conversion uses the per-shape ratio the
    library campaign recorded. That also makes the evaluation pool deterministic.
    """
    r = float(json.loads((OUT_LIB / f"{shape}.json").read_text())["cost"]["ratio"])
    return r, {"source": "out_lib cost.ratio", "value": r}


# ---------------------------------------------------------------------------
# forward, scoring
# ---------------------------------------------------------------------------

def run_forward(case, s, eps_covary: bool, checkpoints: bool = False):
    return fwd.forward(case, s, keep_checkpoints=checkpoints, stop_after_phi=None,
                       shape_stop_patience=lib.PATIENCE, n_steps=N_STEPS,
                       eps_covary=eps_covary)


def score_trajectory(case, tr, s: np.ndarray, eps_covary: bool) -> dict:
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    T_stop = tr.T_at_end(i)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["mean_rho_rel_part_at_end"] = float(tr.mean_rho_rel_part[tr.n_outer - 1])
    m["max_T_at_stop_c"] = float(np.max(T_stop))
    m["over_ceiling_250c"] = exceeds_ceiling(m["max_T_at_stop_c"])
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["eps_covary"] = bool(eps_covary)
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_std_in_part"] = float(np.std(s[case.part_mask]))
    m["map_roughness"] = df.roughness_in_part(s, case.part_mask)
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    return m


def score(case, s: np.ndarray, eps_covary: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    tr = run_forward(case, s, eps_covary)
    m = score_trajectory(case, tr, s, eps_covary)
    T_stop = tr.T_at_end(int(m["t_stop_index"]))
    phi_stop, _ = so.phi_field(T_stop, case)
    j_curve = so.J_curve(tr, case)
    del tr
    return m, phi_stop, j_curve


# ---------------------------------------------------------------------------
# one full-depth start at one angle
# ---------------------------------------------------------------------------

def solve_start(case, ops, v0, n_evals: int, eps_covary: bool, log, name: str):
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
        s = df.apply_filter(v, pm, SIGMA_CELLS)
        tr = run_forward(case, s, eps_covary, checkpoints=True)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        g_s = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops,
                               eps_covary=eps_covary)
        g = df.filter_vjp(g_s, pm, SIGMA_CELLS)
        row = {"eval_index": len(rows) + 1, "start": name, "J": float(J),
               "t_stop_index": int(st.index), "t_stop_s": float(st.time_s),
               "t_stop_at_horizon": bool(st.at_horizon),
               "grad_norm": float(np.linalg.norm(g[pm]))}
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
        b = best_row(rows)
        log(f"    start/{name}: {len(rows)} evaluations, J {rows[0]['J']:.2f} -> "
            f"{b['J']:.2f}, {time.perf_counter() - t0:.0f} s")
    return rows, store


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_angle(shape, actuator, cfg0, angle, sat_warm, ops_cache, n_evals,
              eps_covary, out_dir, log) -> dict:
    t_angle = time.perf_counter()
    cfg = copy.deepcopy(cfg0)
    cfg["geometry"]["part"]["rotation_deg"] = float(angle)
    case = build_case(cfg)
    pm = case.part_mask
    key = (pm.shape[0], pm.shape[1])
    if key not in ops_cache:
        ops_cache[key] = gradops.gradient_matrices(case.x, case.y)
    ops = ops_cache[key]

    row: dict = {"shape": shape, "actuator": actuator, "angle_deg": float(angle),
                 "n_part_cells": int(pm.sum())}

    # Real-data driver gate at zero degrees: the uniform arm must reproduce the
    # library campaign's stored number. One forward, only at angle 0.
    if float(angle) == 0.0:
        m_u, _phi, _jc = score(case, np.ones(pm.shape), eps_covary)
        ref = REF_UNIFORM_AT_ZERO[shape]
        row["gate_uniform_at_zero"] = {
            "J_here": m_u["J"], "J_reference": ref,
            "rel_diff": abs(m_u["J"] - ref) / ref,
            "note": "s = 1 makes fill*s = fill, so the uniform arm is identical "
                    "in both actuator channels and must match out_lib"}
        log(f"  gate uniform@0: J {m_u['J']:.2f} vs reference {ref} "
            f"(rel {row['gate_uniform_at_zero']['rel_diff']:.2e})")

    starts = {"cold": np.ones(pm.shape),
              "warm": rotated_warm_start(sat_warm, angle, pm, BOX)}
    per_start: dict[str, dict] = {}
    for name, v0 in starts.items():
        rows, store = solve_start(case, ops, v0, n_evals, eps_covary, log, name)
        if not rows:
            log(f"    start {name}: NO EVALUATIONS, skipped loudly")
            continue
        b = best_row(rows)
        per_start[name] = {"rows": rows, "best_J": float(b["J"]),
                           "v": store[int(b["eval_index"])]}

    if not per_start:
        raise RuntimeError(f"{shape} {angle}: no start produced an evaluation")
    winner = min(per_start, key=lambda k: per_start[k]["best_J"])
    v_best = per_start[winner]["v"]
    s_cont = df.apply_filter(v_best, pm, SIGMA_CELLS)
    s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)

    m_c, phi_c, jc_c = score(case, s_cont, eps_covary)
    m_c["arm"] = "JOINT_cont"
    m_q, phi_q, jc_q = score(case, s_q, eps_covary)
    m_q["arm"] = "JOINT_4bpp"
    m_q.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm, bpp=4).items()})

    row.update({
        "winner_start": winner,
        "start_best_J": {k: v["best_J"] for k, v in per_start.items()},
        "n_evals_by_start": {k: len(v["rows"]) for k, v in per_start.items()},
        "rows_by_start": {k: v["rows"] for k, v in per_start.items()},
        "JOINT_cont": m_c, "JOINT_4bpp": m_q,
        # the headline arm, matching the fixed-map sweep's quantized graded arm
        "J": m_q["J"], "IoU": m_q["IoU"],
        "wall_s": time.perf_counter() - t_angle,
    })

    tag = f"ang{angle:07.2f}".replace(".", "p")
    np.savez_compressed(
        out_dir / "fields" / f"{tag}.npz",
        phi_cont=phi_c.astype(np.float32), phi_4bpp=phi_q.astype(np.float32),
        part_mask=pm, sat_cont=s_cont.astype(np.float32),
        sat_4bpp=s_q.astype(np.float32), v_best=v_best.astype(np.float32),
        J_curve_cont=jc_c.astype(np.float32), J_curve_4bpp=jc_q.astype(np.float32),
        angle_deg=float(angle), x=case.x, y=case.y)

    log(f"  ang {angle:6.1f}: winner {winner:4s}  J_4bpp {m_q['J']:8.2f}  "
        f"IoU {m_q['IoU']:.4f}  grow {m_q['bed_melt_pct_of_part']:5.2f}%  "
        f"under {m_q['part_under_melt_pct']:5.2f}%  rho {m_q['mean_rho_rel_part_at_stop']:.4f}  "
        f"P {m_q['P_abs_W_per_m']:6.1f} W/m  stop {m_q['t_stop_s']:6.1f} s"
        f"{' HORIZON' if m_q['t_stop_at_horizon'] else ''}  "
        f"maxT {m_q['max_T_at_stop_c']:.1f} C"
        f"{'  CEILING' if m_q['over_ceiling_250c'] else ''}  "
        f"[{row['wall_s']:.0f} s]")
    return row


def main(shape: str, actuator: str = "sigma",
         budget: float = BUDGET_FORWARD_EQUIVALENTS) -> dict:
    if shape not in ANGLES:
        raise ValueError(f"{shape!r} is not one of the stranded shapes {list(ANGLES)}")
    if actuator not in ("sigma", "eps"):
        raise ValueError(f"actuator must be 'sigma' or 'eps', got {actuator!r}")
    eps_covary = actuator == "eps"
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{shape}/{actuator}] {msg}", flush=True)

    out_dir = OUT_ROOT / f"{shape}_{actuator}"
    (out_dir / "fields").mkdir(parents=True, exist_ok=True)

    cfg_path = lib.shape_config(shape)
    cfg0 = load_cfg(cfg_path)
    ratio, ratio_info = budget_ratio(shape)
    n_evals = ctl.max_gradient_evals(float(budget), ratio)
    sat_warm, warm_meta = best_known_zero_map(shape, actuator, log)
    log(f"budget {budget} forward-equivalents per start, ratio {ratio:.3f} "
        f"-> {n_evals} gradient evaluations per start, 2 starts per angle")

    rows = [run_angle(shape, actuator, cfg0, a, sat_warm, {}, n_evals,
                      eps_covary, out_dir, log) for a in ANGLES[shape]]

    period = SYMMETRY_PERIOD_DEG[shape]
    joint_best = argmin_angle(rows)
    fixed_best = SWEEP_BEST_GRADED_DEG[shape]
    by_angle = {float(r["angle_deg"]): r for r in rows}

    result = {
        "shape": shape, "actuator": actuator,
        "actuator_note": ("conductivity only; the deployment-safe channel given the "
                          "open eps_r material question in EPS_CHANNEL_REPORT.md "
                          "Section 11" if actuator == "sigma" else
                          "conductivity AND relative permittivity co-varying; "
                          "MODEL ONLY, printability unresolved"),
        "config": str(cfg_path),
        "voltage_v": float(cfg0["electric"]["voltage_v"]),
        "grid": 120, "box": list(BOX), "sigma_cells": float(SIGMA_CELLS),
        "budget_forward_equivalents_per_start": float(budget),
        "n_gradient_evals_per_start": n_evals, "ratio": ratio,
        "ratio_info": ratio_info, "warm_start": warm_meta,
        "angles_deg": list(ANGLES[shape]),
        "symmetry_period_deg": period,
        "stop_convention": ("t_stop = argmin of J_phi over the arm's own trajectory; "
                            "1500-step horizon (750 s), patience 250; "
                            "melted region phi >= 0.5; grid 120"),
        "headline_arm": "JOINT_4bpp",
        "joint_best_angle_deg": joint_best,
        "fixed_map_sweep_best_angle_deg": fixed_best,
        "angle_move_deg": angle_delta_deg(joint_best, fixed_best, period),
        "prediction_confirmed": bool(abs(angle_delta_deg(joint_best, fixed_best,
                                                         period)) > 1e-9),
        "J_at_joint_best": by_angle[joint_best]["J"],
        "IoU_at_joint_best": by_angle[joint_best]["IoU"],
        "J_at_fixed_best_angle": by_angle.get(fixed_best, {}).get("J"),
        "ceiling_flags": [r["angle_deg"] for r in rows
                          if r["JOINT_4bpp"]["over_ceiling_250c"]],
        "horizon_arms": [r["angle_deg"] for r in rows
                         if r["JOINT_4bpp"]["t_stop_at_horizon"]],
        "energy_gate_violations": [r["angle_deg"] for r in rows
                                   if not r["JOINT_4bpp"]["energy_gate"]["PASS"]],
        "rows": rows,
        "wall_s": time.perf_counter() - t0,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=1, default=float))
    log(f"DONE joint best {joint_best} deg (fixed-map sweep best {fixed_best} deg, "
        f"move {result['angle_move_deg']:+.1f} deg), J {result['J_at_joint_best']:.2f}, "
        f"IoU {result['IoU_at_joint_best']:.4f}, wall {result['wall_s']:.0f} s")
    return result


def refine(shape: str, actuator: str = "sigma", budget: float = 40.0,
           top_k: int = 2) -> dict:
    """Depth check: re-solve the best few angles at a larger budget.

    A 15 forward-equivalent scan can move an argmin by itself, so the headline
    is only safe if the ranking survives depth. This re-runs the `top_k` angles
    of the scan at `budget` forward-equivalents per start and reports whether
    the joint argmin changes.
    """
    eps_covary = actuator == "eps"
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{shape}/{actuator}/refine] {msg}", flush=True)

    scan_dir = OUT_ROOT / f"{shape}_{actuator}"
    scan = json.loads((scan_dir / "results.json").read_text())
    ranked = sorted(scan["rows"], key=lambda r: float(r["J"]))
    angles = sorted(float(r["angle_deg"]) for r in ranked[:int(top_k)])
    # The fixed-map sweep's own best angle is always included, because the
    # headline compares against it.
    fixed_best = SWEEP_BEST_GRADED_DEG[shape]
    if fixed_best not in angles:
        angles = sorted(angles + [fixed_best])

    out_dir = OUT_ROOT / f"{shape}_{actuator}_refine"
    (out_dir / "fields").mkdir(parents=True, exist_ok=True)
    cfg_path = lib.shape_config(shape)
    cfg0 = load_cfg(cfg_path)
    ratio, ratio_info = budget_ratio(shape)
    n_evals = ctl.max_gradient_evals(float(budget), ratio)
    sat_warm, warm_meta = best_known_zero_map(shape, actuator, log)
    log(f"refining angles {angles} at {budget} forward-equivalents per start "
        f"-> {n_evals} gradient evaluations per start")

    rows = [run_angle(shape, actuator, cfg0, a, sat_warm, {}, n_evals,
                      eps_covary, out_dir, log) for a in angles]
    period = SYMMETRY_PERIOD_DEG[shape]
    refined_best = argmin_angle(rows)
    scan_best = float(scan["joint_best_angle_deg"])
    by_angle = {float(r["angle_deg"]): r for r in rows}
    result = {
        "shape": shape, "actuator": actuator, "mode": "refine",
        "budget_forward_equivalents_per_start": float(budget),
        "n_gradient_evals_per_start": n_evals, "ratio": ratio,
        "ratio_info": ratio_info, "warm_start": warm_meta,
        "angles_refined_deg": angles, "top_k": int(top_k),
        "scan_best_angle_deg": scan_best,
        "refined_best_angle_deg": refined_best,
        "argmin_survives_depth": bool(abs(angle_delta_deg(
            refined_best, scan_best, period)) < 1e-9),
        "fixed_map_sweep_best_angle_deg": fixed_best,
        "angle_move_deg": angle_delta_deg(refined_best, fixed_best, period),
        "J_at_refined_best": by_angle[refined_best]["J"],
        "IoU_at_refined_best": by_angle[refined_best]["IoU"],
        "ceiling_flags": [r["angle_deg"] for r in rows
                          if r["JOINT_4bpp"]["over_ceiling_250c"]],
        "horizon_arms": [r["angle_deg"] for r in rows
                         if r["JOINT_4bpp"]["t_stop_at_horizon"]],
        "energy_gate_violations": [r["angle_deg"] for r in rows
                                   if not r["JOINT_4bpp"]["energy_gate"]["PASS"]],
        "rows": rows, "wall_s": time.perf_counter() - t0,
    }
    (out_dir / "results_refine.json").write_text(
        json.dumps(result, indent=1, default=float))
    log(f"DONE refine: best {refined_best} deg (scan said {scan_best}), "
        f"survives depth {result['argmin_survives_depth']}, "
        f"J {result['J_at_refined_best']:.2f}, IoU {result['IoU_at_refined_best']:.4f}, "
        f"wall {result['wall_s']:.0f} s")
    return result


if __name__ == "__main__":
    if sys.argv[1] == "refine":
        _shape = sys.argv[2]
        _act = sys.argv[3] if len(sys.argv) > 3 else "sigma"
        _budget = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
        _k = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        refine(_shape, _act, _budget, _k)
    else:
        _shape = sys.argv[1]
        _act = sys.argv[2] if len(sys.argv) > 2 else "sigma"
        _budget = (float(sys.argv[3]) if len(sys.argv) > 3
                   else BUDGET_FORWARD_EQUIVALENTS)
        main(_shape, _act, _budget)
