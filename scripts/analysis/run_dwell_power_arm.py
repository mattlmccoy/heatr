#!/usr/bin/env python3
"""The ONE labelled secondary arm: does p(t) add anything on top of the dwell?

Power scheduling is deprioritized. This arm exists because the cross is the one
shape whose optimized power schedule previously produced a real generator OFF
period whose benefit SURVIVED a dose-matched control
(`TEMPORAL_SCHEDULING_REPORT.md` Sections 5 and 8, +0.0413 intersection over
union from temporal structure and +0.0000 from dose). The narrow question here
is whether that gain is still there once the turntable dwell schedule has been
optimized, or whether the dwell schedule has already taken it.

Arms, all on the deliverable dwell schedule and its 4-bits-per-pixel map:

  P0_dwell_only     the dwell deliverable with the generator at nominal power.
  P1_power_only     p(t) optimized with the dwell FIXED.
  P2_joint          p(t) and the dwell co-optimized, alternating blocks.
  P3_dose_control   the SAME map and dwell at a CONSTANT power equal to the
                    winning arm's time-weighted duty cycle, which delivers the
                    same integrated dose with no temporal structure. The gap
                    between the winning arm and this one is the part of the
                    gain that is structure rather than dose.

Run:
  ./.venv312/bin/python scripts/analysis/run_dwell_power_arm.py [shape] [n_evals]
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

from adjoint2d import dwell, dwell_power as dp, energy_gate as eg     # noqa: E402
from adjoint2d import gradops, library_solve as lib, schedule as sch  # noqa: E402
from adjoint2d import shape_objective as so                          # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                       # noqa: E402
from adjoint2d.pins import load_cfg                                  # noqa: E402

N_STEPS = 1500
DT_S = 0.5
N_SEG = 12
P_BOX = (0.0, 1.5)
Z_BOX = (-6.0, 6.0)
OUT = REPO / "fgm_solve_campaign/out_dwell"


def score(kern, case, s, p_seg, w, tag):
    pf = sch.expand_full(p_seg, N_STEPS, N_STEPS, N_SEG)
    kern.set_weights(w)
    tr = dp.scheduled_forward(kern, s, pf, n_steps=N_STEPS)
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    m["arm"] = tag
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    m["max_T_at_stop_c"] = float(np.max(tr.T_at_end(i)))
    m["over_ceiling_250c"] = bool(m["max_T_at_stop_c"] > 250.0)
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["duty_cycle"] = sch.duty_cycle(p_seg, N_STEPS, N_SEG)
    m["n_switches"] = sch.n_switches(np.round(p_seg, 6))
    m["p_seg"] = [float(v) for v in p_seg]
    m["structure"] = sch.place_then_hold(p_seg, N_STEPS, N_SEG, i)
    m["dwell_weights"] = [float(v) for v in w]
    m["instructions"] = sch.instructions(p_seg, N_STEPS, N_SEG, DT_S, merge=True)
    phi, _ = so.phi_field(tr.T_at_end(i), case)
    return m, phi, so.J_curve(tr, case)


def block(kern, case, ops, s, state, which, n_evals, rows, log):
    n0 = len(rows)

    def fun(xv):
        if len(rows) - n0 >= int(n_evals):
            raise StopIteration
        p_seg = np.asarray(xv, float) if which == "power" else state["p"]
        z = state["z"] if which == "power" else np.asarray(xv, float)
        w = dwell.softmax_weights(z)
        kern.set_weights(w)
        pf = sch.expand_full(p_seg, N_STEPS, N_STEPS, N_SEG)
        tr = dp.scheduled_forward(kern, s, pf, n_steps=N_STEPS, keep_checkpoints=True)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        _gs, gw, gp = dp.scheduled_gradients(kern, s, tr, {st.index: seed}, pf,
                                             n_seg=N_SEG, grad_ops=ops,
                                             n_window=N_STEPS)
        rows.append({"eval_index": len(rows) + 1, "block": which, "J": float(J),
                     "p": np.array(p_seg, float), "z": np.array(z, float),
                     "t_stop_index": int(st.index)})
        g = gp if which == "power" else dwell.softmax_vjp(w, gw)
        del tr
        return float(J), np.asarray(g, float)

    x0 = state["p"] if which == "power" else state["z"]
    bounds = [P_BOX] * N_SEG if which == "power" else [Z_BOX] * state["z"].size
    t0 = time.perf_counter()
    try:
        minimize(fun, np.array(x0, float), jac=True, method="L-BFGS-B",
                 bounds=bounds, options={"maxiter": 10_000, "maxfun": 10_000,
                                         "ftol": 1e-16, "gtol": 1e-16})
    except StopIteration:
        pass
    mine = rows[n0:]
    if not mine:
        return
    b = min(mine, key=lambda r: r["J"])
    if b["J"] <= min(r["J"] for r in rows):
        state["p"], state["z"] = b["p"], b["z"]
    log(f"  block {which:6s}: {len(mine)} evals, J {mine[0]['J']:.2f} -> "
        f"{b['J']:.2f}, {time.perf_counter() - t0:.0f} s")


def main(shape: str = "cross", n_evals: int = 12) -> dict:
    t0 = time.perf_counter()

    def log(m):
        print(f"[{shape}/dwellpower] {m}", flush=True)

    js = OUT / f"{shape}_dwell.json"
    npz = OUT / f"{shape}_dwell_maps.npz"
    if not (js.exists() and npz.exists()):
        raise FileNotFoundError(f"run the dwell solve for {shape} first")
    r = json.loads(js.read_text())
    d = np.load(npz)
    ang = np.asarray(r["candidate_angles_deg"], float)
    cfg = load_cfg(lib.shape_config(shape))
    kern = DwellKernel.build(cfg, angles=ang)
    case = kern.case0
    ops = gradops.gradient_matrices(case.x, case.y)
    # the best DELIVERABLE map on record for this shape, by quasi-static J
    cands = [(v["J"], k) for k, v in r["arms"].items()
             if "4bpp" in k and f"sat_{k}" in d and "equal" not in k]
    _J, key = min(cands)
    s = np.asarray(d[f"sat_{key}"], dtype=float)
    log(f"map = {key} (quasi-static J {_J:.2f})")
    w0 = np.asarray(r["arms"][key]["dwell_weights"], float)
    w0 = w0 / w0.sum()
    z0 = dwell.softmax_logits_for(w0)
    p_one = np.ones(N_SEG)

    arms, fields = {}, {}

    def record(m, phi, jc):
        arms[m["arm"]] = m
        fields[f"phi_{m['arm']}"] = phi.astype(np.float32)
        fields[f"J_curve_{m['arm']}"] = jc.astype(np.float32)
        log(f"  {m['arm']:16s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  duty "
            f"{m['duty_cycle']:.3f}  switches {m['n_switches']:2d}  "
            f"rho {m['mean_rho_rel_part_at_stop']:.4f}  stop {m['t_stop_s']:6.1f} s  "
            f"maxT {m['max_T_at_stop_c']:6.1f} C  structure {m['structure']['structure']}"
            f"  Egate {'PASS' if m['energy_gate']['PASS'] else 'FAIL'}")

    record(*score(kern, case, s, p_one, w0, "P0_dwell_only"))

    log(f"P1: power schedule alone, {n_evals} gradient evaluations, "
        f"{N_SEG} segments over {N_STEPS * DT_S:.0f} s, box {P_BOX}")
    st1 = {"p": p_one.copy(), "z": z0.copy()}
    rows1: list[dict] = []
    block(kern, case, ops, s, st1, "power", n_evals, rows1, log)
    record(*score(kern, case, s, st1["p"], dwell.softmax_weights(st1["z"]),
                  "P1_power_only"))

    log(f"P2: power and dwell co-optimized, alternating blocks")
    st2 = {"p": p_one.copy(), "z": z0.copy()}
    rows2: list[dict] = []
    for which, n in (("power", n_evals // 2), ("dwell", n_evals // 4),
                     ("power", n_evals - n_evals // 2 - n_evals // 4)):
        block(kern, case, ops, s, st2, which, n, rows2, log)
    w2 = dwell.softmax_weights(st2["z"])
    record(*score(kern, case, s, st2["p"], w2, "P2_joint"))

    best = min(("P1_power_only", "P2_joint"), key=lambda k: arms[k]["J"])
    duty = arms[best]["duty_cycle"]
    w_best = np.asarray(arms[best]["dwell_weights"], float)
    log(f"P3: dose-matched control at constant p = {duty:.4f} on {best}'s "
        f"map and dwell")
    record(*score(kern, case, s, np.full(N_SEG, duty), w_best, "P3_dose_control"))

    struct = arms[best]["IoU"] - arms["P3_dose_control"]["IoU"]
    dose = arms["P3_dose_control"]["IoU"] - arms["P0_dwell_only"]["IoU"]
    log(f"DECOMPOSITION on {best}: total {arms[best]['IoU'] - arms['P0_dwell_only']['IoU']:+.4f} "
        f"IoU = structure {struct:+.4f} + dose {dose:+.4f}")

    np.savez_compressed(OUT / f"{shape}_dwellpower_fields.npz",
                        part_mask=case.part_mask, x=case.x, y=case.y, **fields)
    out = {"shape": shape, "n_seg": N_SEG, "n_steps": N_STEPS, "dt_s": DT_S,
           "power_box": list(P_BOX), "candidate_angles_deg": [float(a) for a in ang],
           "base_dwell_weights": [float(v) for v in w0],
           "n_gradient_evals": int(n_evals), "best_arm": best,
           "decomposition_IoU": {"total": arms[best]["IoU"] - arms["P0_dwell_only"]["IoU"],
                                 "structure": struct, "dose": dose},
           "arms": arms, "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_dwellpower.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"DONE wall {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cross",
         int(sys.argv[2]) if len(sys.argv) > 2 else 12)
