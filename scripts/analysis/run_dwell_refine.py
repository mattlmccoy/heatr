#!/usr/bin/env python3
"""Refinement arm: fix the discovered dwell schedule, then solve the map deeper.

WHY THIS ARM EXISTS, stated as the budget accounting it is. The joint solve
alternates blocks, so at a 16 gradient-evaluation budget the map receives only
8 of them while the equal-dwell control receives all 16. That makes the joint
arm's map SHALLOWER than the control's, which biases the comparison AGAINST the
dwell schedule. This arm removes that bias in the only honest direction
available: it fixes the dwell at the schedule the joint solve found and spends
a further 16 gradient evaluations on the map alone.

It is therefore NOT a 40 forward-equivalent arm. It is labelled with its total
spend everywhere it is quoted, and the equal-budget joint arm is kept as the
number that answers "what does 40 forward-equivalents buy".

Run:
  ./.venv312/bin/python scripts/analysis/run_dwell_refine.py <shape> [n_evals]
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

from adjoint2d import design_filter as df, dwell, energy_gate as eg   # noqa: E402
from adjoint2d import dwell_march as dmarch, gradops                  # noqa: E402
from adjoint2d import library_solve as lib, printability as pq        # noqa: E402
from adjoint2d import shape_objective as so, topopt                   # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                        # noqa: E402
from adjoint2d.pins import load_cfg                                   # noqa: E402

N_STEPS = 1500
DT_S = 0.5
BOX = (0.0, 1.0)
PATIENCE = lib.PATIENCE
OUT = REPO / "fgm_solve_campaign/out_dwell"


def _library_zero_map(shape: str, pm) -> np.ndarray:
    best = None
    for arm, npz, jp in (("A1_cont", REPO / f"fgm_solve_campaign/out_lib/{shape}_maps.npz",
                          REPO / f"fgm_solve_campaign/out_lib/{shape}.json"),
                         ("MS_cont", REPO / f"fgm_solve_campaign/out_ms/{shape}_maps.npz",
                          REPO / f"fgm_solve_campaign/out_ms/{shape}.json")):
        if not (npz.exists() and jp.exists()):
            continue
        a = json.loads(jp.read_text()).get("arms", {})
        if arm not in a:
            continue
        if best is None or float(a[arm]["J"]) < best[0]:
            best = (float(a[arm]["J"]), npz, arm)
    if best is None:
        raise FileNotFoundError(f"no stored zero-degree map for {shape}")
    m = np.clip(np.asarray(np.load(best[1])[best[2]], dtype=float), 0.0, 1.0)
    return np.where(pm, m, 1.0)


def main(shape: str, n_evals: int = 16, start: str = "warm",
         sigma_override: float | None = None, dwell_mode: str = "discovered") -> dict:
    t0 = time.perf_counter()

    def log(m):
        print(f"[{shape}/refine] {m}", flush=True)

    js = OUT / f"{shape}_dwell.json"
    r = json.loads(js.read_text())
    d = np.load(OUT / f"{shape}_dwell_maps.npz")
    ang = np.asarray(r["candidate_angles_deg"], float)
    cycle_s = float(r["cycle_time_s"])
    cfg = load_cfg(lib.shape_config(shape))
    kern = DwellKernel.build(cfg, angles=ang)
    case = kern.case0
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = (topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)
                   if sigma_override is None else float(sigma_override))
    tag_sfx = f"_{dwell_mode}_{start}" + ("" if sigma_override is None
                                          else f"_sig{sigma_override:g}")

    if dwell_mode == "equal":
        w = np.full(len(ang), 1.0 / len(ang))
    else:
        w = np.asarray(r["arms"]["D_program_4bpp"]["dwell_weights"], float)
    w = w / w.sum()
    kern.set_weights(w)
    log(f"dwell fixed at {np.round(w, 4).tolist()}")

    # warm start from the joint arm's own design variable, recovered by
    # inverting the filter is not possible, so start from its FILTERED map,
    # which is a legal design point and whose first evaluation is that arm's
    # own score to within the filter's idempotency.
    if start == "cold":
        v0 = np.ones(pm.shape)
    elif start == "lib":
        # The start that decided the continuous-rotation pass's cross result:
        # the best stored ZERO-DEGREE library map. MEASURED there, the cold
        # start stalls at 126.80 and this one reaches 34.99 on the same shape,
        # same kernel, same budget (`out_rot/cross_rotavg_step90.json`).
        v0 = _library_zero_map(shape, pm)
    else:
        v0 = np.asarray(d[f"sat_{r['deliverable_source_arm']}"], dtype=float)
    log(f"start {start}, filter sigma {sigma_cells:.3f} cells")
    idx = np.flatnonzero(pm.ravel())
    rows: list[dict] = []
    store: dict[int, np.ndarray] = {}

    def fun(x):
        if len(rows) >= int(n_evals):
            raise StopIteration
        v = np.ones(pm.shape)
        v.ravel()[idx] = x
        s = df.apply_filter(v, pm, sigma_cells)
        tr = kern.forward(s, keep_checkpoints=True, n_steps=N_STEPS,
                          shape_stop_patience=PATIENCE)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        g_s, _gw = kern.both_gradients(s, tr, {st.index: seed}, grad_ops=ops)
        g = df.filter_vjp(g_s, pm, sigma_cells)
        rows.append({"eval_index": len(rows) + 1, "J": float(J),
                     "t_stop_index": int(st.index)})
        store[len(rows)] = v.copy()
        del tr
        return float(J), g.ravel()[idx].astype(float)

    try:
        minimize(fun, np.clip(v0.ravel()[idx], *BOX), jac=True, method="L-BFGS-B",
                 bounds=[BOX] * len(idx), options={"maxiter": 10_000,
                                                   "maxfun": 10_000,
                                                   "ftol": 1e-16, "gtol": 1e-16})
    except StopIteration:
        pass
    b = min(rows, key=lambda x: x["J"])
    log(f"  {len(rows)} evals, J {rows[0]['J']:.2f} -> {b['J']:.2f}")
    s_ref = df.apply_filter(store[b["eval_index"]], pm, sigma_cells)
    s_q = pq.quantize_in_part(s_ref, pm, bpp=4, sat_max=1.0)

    def sc(s, tag, time_resolved=False):
        kern.set_weights(w)
        if time_resolved:
            prog = dwell.cycle_program(w, ang, cycle_time_s=cycle_s,
                                       total_s=N_STEPS * DT_S, dt_s=DT_S)
            kern.averaged_Q(s)
            ii = dmarch.program_step_positions(prog, ang, DT_S, N_STEPS)
            tr = dmarch.program_forward(kern, ii, N_STEPS)
        else:
            tr = kern.forward(s, keep_checkpoints=False, n_steps=N_STEPS,
                              shape_stop_patience=PATIENCE)
        m = so.full_metrics(tr, case)
        i = int(m["t_stop_index"])
        m["arm"] = tag
        m["energy_gate"] = eg.gate_from_trajectory(tr, i)
        m["max_T_at_stop_c"] = float(np.max(tr.T_at_end(i)))
        m["over_ceiling_250c"] = bool(m["max_T_at_stop_c"] > 250.0)
        m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
        m["P_abs_W_per_m"] = tr.P_abs_B
        m["dwell_weights"] = [float(v) for v in w]
        m.update({f"dwell_{k}": v for k, v in dwell.dwell_asymmetry(w).items()})
        phi, _ = so.phi_field(tr.T_at_end(i), case)
        log(f"  {tag:24s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  grow "
            f"{m['bed_melt_pct_of_part']:5.2f}%  under {m['part_under_melt_pct']:5.2f}%  "
            f"rho {m['mean_rho_rel_part_at_stop']:.4f}  P {m['P_abs_W_per_m']:6.1f} W/m  "
            f"stop {m['t_stop_s']:6.1f} s  maxT {m['max_T_at_stop_c']:6.1f} C"
            f"{'  CEILING' if m['over_ceiling_250c'] else ''}  "
            f"Egate {'PASS' if m['energy_gate']['PASS'] else 'FAIL'}")
        return m, phi, so.J_curve(tr, case)

    new = {}
    fields = {}
    for s, tag, tres in ((s_ref, f"D_refined_cont{tag_sfx}", False),
                         (s_q, f"D_refined_4bpp{tag_sfx}", False),
                         (s_q, f"D_refined_timeresolved{tag_sfx}", True)):
        m, phi, jc = sc(s, tag, tres)
        new[tag] = m
        fields[f"phi_{tag}"] = phi.astype(np.float32)
        fields[f"J_curve_{tag}"] = jc.astype(np.float32)
    fields[f"sat_D_refined_4bpp{tag_sfx}"] = s_q.astype(np.float32)
    fields[f"sat_D_refined_cont{tag_sfx}"] = s_ref.astype(np.float32)

    old = dict(np.load(OUT / f"{shape}_dwell_maps.npz"))
    old.update(fields)
    np.savez_compressed(OUT / f"{shape}_dwell_maps.npz", **old)
    r["arms"].update(new)
    r.setdefault("refine_runs", {})[f"refine{tag_sfx}"] = {
        "start": start, "sigma_cells": float(sigma_cells),
        "n_gradient_evals": int(n_evals), "rows": rows}
    r["refine"] = {"n_gradient_evals": int(n_evals), "rows": rows,
                   "total_gradient_evals_for_this_arm":
                       int(r["n_gradient_evals_per_solve"]) * 2 + int(n_evals),
                   "note": "dwell held at the joint solve's schedule, map solved deeper"}
    r["wall_s_refine"] = time.perf_counter() - t0
    js.write_text(json.dumps(r, indent=1, default=float))
    log(f"DONE wall {r['wall_s_refine']:.0f} s")
    return r


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16,
         sys.argv[3] if len(sys.argv) > 3 else "warm",
         float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None,
         sys.argv[5] if len(sys.argv) > 5 else "discovered")
