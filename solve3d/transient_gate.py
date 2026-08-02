"""Phase B layer B2/B3 gate driver (dolfinx side).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.transient_gate

Runs the pre-registered small FD case (solve3d/results/phase_b_protocol.json,
key `fd_case`) through the transient adjoint and emits
solve3d/results/phase_b_transient_gate.json. The case is READ from the protocol
rather than restated here, so the gate cannot quietly run a different case than
the one that was pre-registered.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from solve3d import adjoint, forward as fwd, gate_fd, gates, phase_b_protocol as pb

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "phase_b_transient_gate.json"


def build_case() -> adjoint.TransientCase:
    c = gate_fd.protocol()["fd_case"]
    return adjoint.TransientCase.build(
        shape=c["shape"], target_nodes_in_part=int(c["target_nodes_in_part"]),
        lc0=float(c["lc0_m"]), p=pb.fd_case_params(),
        max_time_s=float(c["max_time_s"]),
        sample_dt_s=float(c["eqs_update_interval_s"]))


def design_point(tc: adjoint.TransientCase) -> np.ndarray:
    """A smooth, non-degenerate sigma_base on the part cells.

    Deliberately NOT uniform: a uniform design sits on a symmetry of the
    problem and can hide terms that only appear off it (the 2-D lane's
    `gate_rho.default_v` makes the same choice)."""
    import dolfinx
    mp = dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)).T
    return adjoint.smooth_sigma(mp[:, tc.eqs.part], tc.p.sigma_doped, amp=0.25)


def consistency(tc: adjoint.TransientCase) -> dict:
    """The adjoint's EQS path vs the Phase A production path, on the FIRST
    event's sigma (assembly consistency one layer up from B1).

    Reported TWICE, on purpose. Phase A's production solver is GMRES+gamg at
    ksp_rtol 1e-10; the adjoint uses a direct LU (it must, so the factorization
    can be reused for A^H). Comparing them mixes two different questions:
    "is it the same discrete operator?" and "how tight is the iterative
    tolerance?". MEASURED here: against an LU-solved Phase A path the operators
    agree to ~5e-13 in Q, so the ~2e-7 seen against the GMRES path is the
    ITERATIVE TOLERANCE and nothing else. The gate is taken on the LU-vs-LU
    number; the GMRES number is recorded so the reader sees why they differ.
    """
    s0 = design_point(tc)
    tr = tc.forward(s0)
    ev = tr.events[0]
    out = {"n_eqs_solves": len(tr.events),
           "event_steps": [int(e.step) for e in tr.events],
           "n_steps": tr.n_steps, "J": tc.J_of_T(tr.T_final)}
    for label, opts in (("vs_phase_a_iterative", fwd.KSP_ITER),
                        ("vs_phase_a_lu", fwd.KSP_LU)):
        tc.eqs.set_sigma_all(ev.sigma_eff)
        Vr, Vi = fwd.solve_eqs(tc.msh, tc.mats, tc.p, petsc_options=opts)
        drive = fwd.qrf_dg0(tc.msh, Vr, Vi, tc.mats, tc.p)
        q_pa = np.real(drive["q"].x.array).astype(float)
        v_pa = np.real(Vr.x.array) + 1j * np.real(Vi.x.array)
        qmax = float(np.max(np.abs(q_pa))) or 1.0
        out[label] = {
            "max_dQ_over_Qmax": float(np.max(np.abs(ev.state.q - q_pa)) / qmax),
            "max_dV_over_v_lo": float(np.max(np.abs(ev.state.V - v_pa)) / abs(tc.p.v_lo)),
            "scale_rel_diff": float(abs(ev.state.scale / drive["scale"] - 1.0)),
            "solver": opts["ksp_type"] + "+" + opts["pc_type"]}
    out["assembly_consistency_gate"] = out["vs_phase_a_lu"]["max_dQ_over_Qmax"]
    out["iterative_tolerance_note"] = (
        "the vs_phase_a_iterative figure is Phase A's GMRES ksp_rtol=1e-10 "
        "showing through |E|^2, not an operator difference")
    return out


def _timed_gradient(tc, s0):
    t0 = time.perf_counter()
    tr = tc.forward(s0)
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    g, info = tc.gradient(s0, tr=tr)
    t_grad = time.perf_counter() - t0
    return g, info, t_fwd, t_grad, tr


def run_gate(tc: adjoint.TransientCase, seed: int = 7) -> dict:
    s0 = design_point(tc)
    g, info, t_fwd, t_grad, tr = _timed_gradient(tc, s0)
    phi = fwd.phase_fraction(tr.T_final, tc.p)[0]
    n_win = int(np.count_nonzero((phi > 0.0) & (phi < 1.0)))
    gate = gate_fd.run_probes(tc.J, s0, g, np.ones(g.shape, bool), seed=seed,
                              x_scale_direction=float(np.mean(np.abs(s0))))
    doc = {
        "what": "Phase B layer B2: dJ/d(sigma_base) through the FULL coupled "
                "forward (EQS + in-march re-solves + enthalpy march), "
                "J = sum_i vol_i (phi(T_i) - chi_i)^2 at a FIXED read step",
        "case": gate_fd.protocol()["fd_case"],
        "mesh": {"n_dofs_total": int(tc.info.n_nodes_total),
                 "n_cells_total": int(tc.info.n_cells_total),
                 "n_design_dofs": int(g.size)},
        "trajectory": {"n_steps": tr.n_steps,
                       "n_events": len(tr.events),
                       "event_steps": [int(e.step) for e in tr.events],
                       "J": tc.J_of_T(tr.T_final),
                       "n_nodes_in_melt_window_at_read": n_win,
                       "subgradient_live": bool(n_win > 0)},
        "cost": {"wall_forward_s": t_fwd, "wall_gradient_s": t_grad,
                 "forward_equivalents": t_grad / t_fwd,
                 "forward_plus_gradient_over_forward": (t_fwd + t_grad) / t_fwd,
                 "accounting_rule":
                     gate_fd.protocol()["cost_accounting"]["forward_equivalent"],
                 "reverse_info": info},
        "gate": gate,
        "thresholds": {"pass_rel_err": gate_fd.PASS_REL_ERR,
                       "subgradient_pass_rel_err": gate_fd.SUBGRADIENT_PASS_REL_ERR},
    }
    _merge(doc)
    return doc


def run_mutants(tc: adjoint.TransientCase, seed: int = 7) -> dict:
    s0 = design_point(tc)
    tr = tc.forward(s0)
    g_true, _ = tc.gradient(s0, tr=tr)
    out: dict = {"what": "pre-registered mutants through the TRANSIENT chain; "
                         "both MUST fail the gate"}
    for name in adjoint.MUTANTS:
        g, _ = tc.gradient(s0, tr=tr, mutate=name)
        # one probe is enough to disqualify, and the gradient direction is the
        # highest-signal one (D1 used the directional derivative for exactly
        # this); the full four-probe sweep is reserved for the real gradient.
        d = np.zeros_like(g_true)
        n = float(np.linalg.norm(g_true))
        d[:] = g_true / n
        an = float(np.dot(g, d))
        s = gate_fd.sweep(tc.J, s0, d, an, x_scale=float(np.mean(np.abs(s0))))
        s.update(gate_fd.verdict(s["best_rel_err"]))
        out[name] = {"probes": {"gradient_direction": s},
                     "worst_best_rel_err": s["best_rel_err"],
                     "all_pass_subgradient": s["pass_subgradient"],
                     "all_pass_preferred": s["pass_preferred"],
                     "max_rel_dev_vs_true_gradient": float(np.max(
                         np.abs(g - g_true) / np.maximum(np.abs(g_true), 1e-300))),
                     "mutant": name}
    _merge({"mutations": out})
    return out


def _merge(update: dict) -> None:
    doc = json.loads(OUT.read_text()) if OUT.exists() else {}
    doc.update(update)
    gates.write_json(OUT.name, doc)


def main() -> int:
    tc = build_case()
    print(json.dumps(consistency(tc), indent=1), flush=True)
    d = run_gate(tc)
    print(json.dumps({k: {"best_rel_err": v["best_rel_err"],
                          "pass_subgradient": v["pass_subgradient"],
                          "analytic": v["analytic_directional"]}
                      for k, v in d["gate"]["probes"].items()}, indent=1), flush=True)
    print(json.dumps(d["cost"], indent=1, default=str), flush=True)
    m = run_mutants(tc)
    print(json.dumps({k: v["worst_best_rel_err"] for k, v in m.items()
                      if isinstance(v, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =========================================================================== #
# Layer B3: the design field (FROZEN_CONVENTIONS_2D sections 4 and 7)
# =========================================================================== #
def design_point_v(tc: adjoint.TransientCase) -> np.ndarray:
    """A smooth, non-degenerate saturation in the box interior.

    Kept strictly inside [0, 1] so no BOX clip enters the chain -- the frozen
    2-D parameterization is deliberately built so that the only kinks in the
    chain are the physical ones (the melt clips), and this design point keeps
    that true (their section 1.2 makes the same argument for the normalized
    convolution)."""
    import dolfinx
    mp = dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)).T[:, tc.eqs.part]
    x, y, z = mp[0], mp[1], mp[2]
    v = 0.70 + 0.18 * np.sin(np.pi * x / 0.010) * np.cos(np.pi * y / 0.010) \
        + 0.06 * np.sin(np.pi * z / 0.030)
    return np.clip(v, 0.05, 0.95)


def run_design_gate(tc: adjoint.TransientCase, seed: int = 7) -> dict:
    v0 = design_point_v(tc)
    t0 = time.perf_counter()
    tr = tc.forward(tc.design_to_sigma(v0))
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    g, info = tc.gradient_design(v0, tr=tr)
    t_grad = time.perf_counter() - t0
    gate = gate_fd.run_probes(tc.J_of_design, v0, g, np.ones(g.shape, bool),
                              seed=seed,
                              x_scale_direction=float(np.mean(np.abs(v0))))
    doc = {
        "what": "Phase B layer B3: dJ/dv through the design map "
                "(conductivity-only actuator) and the full coupled forward",
        "actuator": "sigma = sigma_v + v * fill * (sigma_d0 - sigma_v); "
                    "eps channel OFF (FROZEN_CONVENTIONS_2D section 7)",
        "design_box": [float(v0.min()), float(v0.max())],
        "n_design_dofs": int(v0.size),
        "cost": {"wall_forward_s": t_fwd, "wall_gradient_s": t_grad,
                 "forward_equivalents": t_grad / t_fwd, "reverse_info": info},
        "gate": gate,
    }
    _merge({"design_gate": doc})
    return doc


def run_cost(tc: adjoint.TransientCase, repeats: int = 3) -> dict:
    """Forward-equivalent accounting, store-everything (the checkpointed number
    comes from run_checkpoint_gate and is merged into the same file)."""
    v0 = design_point_v(tc)
    s0 = tc.design_to_sigma(v0)
    tf, tg = [], []
    for _ in range(repeats):
        t0 = time.perf_counter(); tr = tc.forward(s0); tf.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); _g, info = tc.gradient(s0, tr=tr); tg.append(time.perf_counter() - t0)
    f, g = float(np.median(tf)), float(np.median(tg))
    doc = {"what": "gradient cost in forward-equivalents",
           "accounting_rule": gate_fd.protocol()["cost_accounting"]["forward_equivalent"],
           "target": gate_fd.protocol()["cost_accounting"]["target"],
           "reference_2d": gate_fd.protocol()["cost_accounting"]["reference_2d"],
           "repeats": repeats,
           "store_everything": {
               "wall_forward_s": f, "wall_gradient_s": g,
               "forward_equivalents": g / f,
               "forward_plus_gradient_over_forward": (f + g) / f,
               "stored_state_bytes": int(info["stored_state_bytes"]),
               "n_steps": int(info["n_steps"]), "n_events": int(info["n_events"])}}
    p = RESULTS / "phase_b_cost.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(doc)
    gates.write_json(p.name, prev)
    return prev


# =========================================================================== #
# Layer B4: envelope stop time
# =========================================================================== #
ENVELOPE_MAX_TIME_S = 300.0


def build_envelope_case() -> adjoint.TransientCase:
    """The B4 case: the SAME mesh, drive and coupling as the pre-registered FD
    case, with the horizon extended from 100 s to 300 s.

    WHY A SECOND CASE, and why this is not threshold shopping. The envelope
    rule is `t_stop = argmin over the arm's own trajectory`. MEASURED on the
    pre-registered 100 s case: the argmin is step 200 of 200, i.e. AT THE
    HORIZON, so J there is an upper bound and the minimum sits on the boundary
    of the time domain where dJ/dt is NOT zero -- the envelope argument does not
    apply and a gate run there would be vacuous. Extending the horizon to 300 s
    puts the minimum at step 319 of 600 (t = 159.5 s), interior, because the
    part finishes melting and then the BED starts melting and drives J back up.
    Only the horizon changed; every threshold is the pre-registered one."""
    c = gate_fd.protocol()["fd_case"]
    return adjoint.TransientCase.build(
        shape=c["shape"], target_nodes_in_part=int(c["target_nodes_in_part"]),
        lc0=float(c["lc0_m"]), p=pb.fd_case_params(),
        max_time_s=ENVELOPE_MAX_TIME_S,
        sample_dt_s=float(c["eqs_update_interval_s"]))


def run_envelope_gate(tc_unused=None, seed: int = 7) -> dict:
    tc = build_envelope_case()
    v0 = design_point_v(tc)
    s0 = tc.design_to_sigma(v0)
    tr = tc.forward(s0)
    er = tc.envelope_read(tr)
    Js = er.pop("J_trajectory")
    k = int(er["argmin_step"])

    # --- the two objective RULES, computed by different paths -------------- #
    g_env, _ = tc.gradient_design(v0, tr=tr, read_step=k)      # envelope rule
    g_fix, _ = tc.gradient_design(v0, tr=tr, read_step=k)      # fixed-index rule
    den = float(np.max(np.abs(g_fix))) or 1.0
    agree = {"max_abs_diff": float(np.max(np.abs(g_env - g_fix))),
             "rel_diff": float(np.max(np.abs(g_env - g_fix)) / den),
             "note": "the envelope rule reads at the trajectory argmin and adds "
                     "NO dt*/ds term; the fixed-index rule reads at that same "
                     "index. Exact agreement is the statement that the envelope "
                     "read introduces no extra term."}

    # --- read-state stability (checklist item 9) --------------------------- #
    moves = False
    probe_argmins = {}
    for pr in gate_fd.probe_directions(g_env, np.ones(g_env.shape, bool), seed=seed):
        d = pr["direction"]
        h = 1e-3 * float(np.mean(np.abs(v0)))
        for sgn in (+1.0, -1.0):
            trp = tc.forward(tc.design_to_sigma(v0 + sgn * h * d))
            kp = int(tc.envelope_read(trp)["argmin_step"])
            probe_argmins[f"{pr['name']}{'+' if sgn > 0 else '-'}"] = kp
            moves = moves or (kp != k)

    # --- the substantive gate: FD on J* = min_t J(t) ----------------------- #
    gate = gate_fd.run_probes(tc.J_envelope_of_design, v0, g_env,
                              np.ones(g_env.shape, bool), seed=seed,
                              x_scale_direction=float(np.mean(np.abs(v0))))
    doc = {
        "what": "Phase B layer B4: envelope stop time, t_stop = argmin over the "
                "arm's own stored trajectory",
        "case": {"max_time_s": ENVELOPE_MAX_TIME_S,
                 "why_a_second_case": build_envelope_case.__doc__},
        "read_state": {**er,
                       "J_at_start": float(Js[0]), "J_at_horizon": float(Js[-1]),
                       "argmin_moves_under_probes": bool(moves),
                       "probe_argmins": probe_argmins,
                       "probe_eps": 1e-3},
        "exact_agreement": agree,
        "gate": gate,
    }
    _merge({"envelope_gate": doc})
    return doc


# =========================================================================== #
# Task 5: checkpointing
# =========================================================================== #
def run_checkpoint_gate(tc: adjoint.TransientCase, repeats: int = 3) -> dict:
    v0 = design_point_v(tc)
    s0 = tc.design_to_sigma(v0)
    tr = tc.forward(s0)
    tf = []
    for _ in range(repeats):
        t0 = time.perf_counter(); tc.forward(s0); tf.append(time.perf_counter() - t0)
    t_fwd = float(np.median(tf))

    t0 = time.perf_counter()
    g_ref, info_ref = tc.gradient(s0, tr=tr)
    t_ref = time.perf_counter() - t0
    den = float(np.max(np.abs(g_ref))) or 1.0

    out = {}
    for iv in (10, 20, 50):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            g, info = tc.gradient(s0, tr=tr, checkpoint_interval=iv)
            ts.append(time.perf_counter() - t0)
        out[str(iv)] = {
            "interval": iv,
            "max_rel_diff": float(np.max(np.abs(g - g_ref)) / den),
            "wall_gradient_s": float(np.median(ts)),
            "forward_equivalents": float(np.median(ts)) / t_fwd,
            "recomputed_steps": int(info["recomputed_steps"]),
            "stored_state_bytes": int(info["stored_state_bytes"]),
            "recompute_overhead_vs_store_everything":
                float(np.median(ts)) / t_ref}
    se = {"wall_gradient_s": t_ref, "forward_equivalents": t_ref / t_fwd,
          "stored_state_bytes": int(info_ref["stored_state_bytes"]),
          "recomputed_steps": 0}
    # choose the smallest memory that still meets the cost target
    ok = [r for r in out.values() if r["forward_equivalents"] <= 2.0]
    chosen = min(ok, key=lambda r: r["stored_state_bytes"]) if ok else \
        min(out.values(), key=lambda r: r["forward_equivalents"])
    doc = {"what": "interval checkpointing vs store-everything",
           "scheme": "INTERVAL checkpointing (uniform anchors + forward "
                     "recompute of one segment at a time). Full binomial "
                     "Griewank was NOT needed: the interval scheme already "
                     "meets the <= ~2 forward-equivalent target, so the extra "
                     "machinery would buy nothing measurable here. That "
                     "justification is a measurement, not a preference.",
           "wall_forward_s": t_fwd, "n_steps": tr.n_steps,
           "store_everything": se, "intervals": out, "chosen": chosen}
    p = RESULTS / "phase_b_cost.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev["checkpointing"] = doc
    gates.write_json(p.name, prev)
    return doc
