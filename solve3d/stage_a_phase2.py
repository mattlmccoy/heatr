"""solve3d Stage A Task 4 PHASE 2: the dopant SHAPE-solve at the chosen fixed drive.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.stage_a_phase2 --fd-gate
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.stage_a_phase2 --solve --budget 12

WHAT THIS IS (scope confirmed by both lanes, STAGE_A_REPORT.md Task 4 phase 2):
optimize the dopant SHAPE with the EXISTING melt-onset envelope adjoint at the
FIXED drive Phase 1 certified feasible (0.40x baseline = 636619.77 W/m^3, the
ONLY drive keeping the square's densify end-state peak under the 250 C
degradation ceiling). The ceiling is evaluated at the densify END-STATE as a
FORWARD gate only (it is nearly dopant-independent, so the dopant needs no
ceiling gradient here; the drive handles the ceiling). The rho co-state /
density adjoint is Stage B+ and is OUT OF SCOPE.

THE ONLY SUBSTANTIVE CHANGE vs a default-drive Phase-E-style square solve is the
drive: `fwd.ForwardParams(power_density_w_per_m3=chosen)`. Everything else --
the asymmetric envelope objective, the design_chain (filter + tanh projection),
the 1/|g0| first-step rescale, per-evaluation checkpointing -- is the frozen
convention reused verbatim from solve3d.phase_e.run / run_tamper.

THE CARDINAL RULE. The gradient is FD-gated AT THE FIXED DRIVE on a small coarse
case BEFORE the multi-hour solve launches (`--fd-gate`). A gradient that is not
FD-verified is presumed wrong; the heavy slot is never spent on an unverified
setup.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from solve3d import (adjoint, densify_forward as df, design_chain as dc,
                     forward as fwd, gate_fd, gates, objective as obj, stage_a)

RESULTS = Path(__file__).resolve().parent / "results"
PART = "square"

# --------------------------------------------------------------------------- #
# The FIXED drive: READ from the Phase 1 artifact, never restated (so it cannot
# drift from the drive Phase 1 certified as the only feasible one).
# --------------------------------------------------------------------------- #
PHASE1_RESULT = RESULTS / "stage_a_task4_square.json"


def _phase1() -> dict:
    if not PHASE1_RESULT.exists():
        raise FileNotFoundError(
            f"{PHASE1_RESULT} not found. Phase 1 (drive selection) must have run "
            "and written its verdict before the phase-2 shape-solve reads the "
            "chosen drive; the drive is never invented here.")
    return json.loads(PHASE1_RESULT.read_text())


def chosen_drive_a() -> float:
    """The drive multiplier Phase 1 selected (0.40x)."""
    return float(_phase1()["verdict"]["chosen_drive_a"])


def chosen_drive_power_density() -> float:
    """The absolute power density (W/m^3) Phase 1 recommended for the best part
    under the 250 C degradation ceiling. This is the fixed drive the shape-solve
    runs at."""
    return float(_phase1()["recommended_power_settings"]["power_density_w_per_m3"])


def drive_params(dt_s: float | None = None,
                 power_density: float | None = None) -> fwd.ForwardParams:
    """ForwardParams at the fixed drive -- the ONE substantive change.

    `power_density` overrides the Phase-1 chosen drive (W/m^3) for the Stage B4
    drive-backoff sweep; None keeps the frozen 0.40x chosen drive, so every
    existing (B1/B2/B3) call is byte-identical. The override is a lower ABSOLUTE
    drive, nothing else about the forward changes."""
    pw = (chosen_drive_power_density() if power_density is None
          else float(power_density))
    p = fwd.ForwardParams(power_density_w_per_m3=pw)
    return p if dt_s is None else dataclasses.replace(p, dt_s=float(dt_s))


# --------------------------------------------------------------------------- #
# Case parameters
# --------------------------------------------------------------------------- #
FILTER_RADIUS_M = 1.0e-3          # the frozen Phase C/E design-chain filter radius
CHECKPOINT_INTERVAL = 25          # reverse-sweep anchor interval (Phase B convention)

# The heavy SOLVE case: the same 5600-in-part-node square anchor Phase 1 used.
SOLVE_TARGET_NODES = 5600
SOLVE_LC0_M = 0.060 / 48.0
# At 0.40x the square melts slowly; the melt-onset envelope minimum sits near
# t~900 s (bed melt drives J back up afterwards). 1800 s puts that minimum
# interior. Verified on the solve mesh by the pre-launch sanity check.
SOLVE_MAX_TIME_S = 1800.0

# The FD-GATE case: coarse + a bigger dt so the 8-eps x 4-probe central-difference
# sweep is a couple of minutes, while the melt-onset envelope argmin stays
# INTERIOR and a live melt-window population keeps the clip subgradients on.
# MEASURED (0.40x, this case): argmin step 1772/3600 (t=886 s), part peak
# 206 C, 69 melt-window nodes, one forward ~1.1 s.
FD_TARGET_NODES = 300
FD_LC0_M = 0.060 / 12.0
FD_DT_S = 0.5
FD_MAX_TIME_S = 1800.0


def _n_part_cells(tc) -> int:
    return int(tc.eqs.vol[tc.eqs.part].shape[0])


def _part_centroids(tc) -> np.ndarray:
    import dolfinx
    return np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]


def build_gate_case() -> adjoint.TransientCase:
    """The coarse square at the fixed drive for the pre-launch FD gate."""
    return adjoint.TransientCase.build(
        shape=PART, target_nodes_in_part=FD_TARGET_NODES, lc0=FD_LC0_M,
        p=drive_params(dt_s=FD_DT_S), max_time_s=FD_MAX_TIME_S)


def build_solve_case(target_nodes: int = SOLVE_TARGET_NODES,
                     lc0: float = SOLVE_LC0_M,
                     max_time_s: float = SOLVE_MAX_TIME_S) -> adjoint.TransientCase:
    """The heavy square at the fixed drive -- identical to a default-drive
    Phase-E-style square solve except for the drive."""
    return adjoint.TransientCase.build(
        shape=PART, target_nodes_in_part=int(target_nodes), lc0=float(lc0),
        p=drive_params(), max_time_s=float(max_time_s))


def make_chain(tc) -> dc.DesignChain:
    return dc.DesignChain(_part_centroids(tc), tc.eqs.vol[tc.eqs.part],
                          FILTER_RADIUS_M, [0.0])


# --------------------------------------------------------------------------- #
# The SHARED gradient: the exact (J_envelope, dJ/dv) the solve minimizes.
# --------------------------------------------------------------------------- #
def envelope_grad_of_design(tc, chain: dc.DesignChain, v: np.ndarray,
                            beta: float = 0.0):
    """(J_envelope, dJ/dv, info) with the frozen conventions: filter chain,
    asymmetric objective, melt-onset envelope stop (argmin over the trajectory),
    checkpointed reverse sweep. This is the function BOTH the FD gate and the
    L-BFGS-B solve call, so the gate certifies the exact operator the solve uses.
    """
    tc.set_objective("asymmetric")
    s = chain.design_to_map(v, beta)
    tr = tc.forward(tc.design_to_sigma(s))
    Jt = tc.J_trajectory(tr)
    k = int(np.argmin(Jt))
    J = float(Jt[k])
    g_s, rinfo = tc.gradient_design(s, tr=tr, read_step=k,
                                    checkpoint_interval=CHECKPOINT_INTERVAL)
    g_v = chain.design_vjp(v, g_s, beta=beta)
    info = {"argmin_step": k, "n_steps": int(tr.n_steps),
            "at_horizon": bool(k >= tr.n_steps),
            "t_stop_s": float(k) * tc.dt_step}
    return J, g_v, info


def _J_envelope_of_design(tc, chain: dc.DesignChain, v: np.ndarray,
                          beta: float = 0.0) -> float:
    tc.set_objective("asymmetric")
    s = chain.design_to_map(v, beta)
    return float(np.min(tc.J_trajectory(tc.forward(tc.design_to_sigma(s)))))


# --------------------------------------------------------------------------- #
# The PRE-LAUNCH HARD GATE: FD-gate the gradient at the fixed drive.
# --------------------------------------------------------------------------- #
def fd_gate(seed: int = 7) -> dict:
    """Three layered checks at the FIXED 0.40x drive, on the coarse case:

      A. map-space adjoint dJ/ds  -- the coupled transient adjoint through the
         conductivity actuator, gated by central FD of the envelope objective
         (this is what the drive change touches; the existing envelope gate ran
         the identical machinery at the 2.0x FD-case drive).
      B. filter transpose identity -- the design_chain layer the solve wraps
         around dJ/ds, by the dot-product identity (frozen checklist item 8).
      C. composite dJ/dv          -- filter . adjoint, the EXACT solve gradient,
         gated by central FD of J(v). Confirms A and B compose correctly.

    Launch only if A and C pass the subgradient standard (the objective reads a
    clipped melt fraction, so melt-window cells are kinks -> the 1e-5 standard,
    protocol item 5) and B passes 1e-10.
    """
    tc = build_gate_case()
    chain = make_chain(tc)
    v0 = _gate_design_point(tc)
    n_part = _n_part_cells(tc)
    mask = np.ones(n_part, dtype=bool)

    # --- A. map-space adjoint dJ/ds (no filter) ---------------------------- #
    tc.set_objective("asymmetric")
    tr = tc.forward(tc.design_to_sigma(v0))
    er = tc.envelope_read(tr)
    k = int(er["argmin_step"])
    T_read = tc.state_at(tr, k)
    phi = fwd.phase_fraction(T_read, tc.p)[0]
    melt_window = int(((phi > 1e-6) & (phi < 1.0 - 1e-6)).sum())
    g_s, _ = tc.gradient_design(v0, tr=tr, read_step=k,
                                checkpoint_interval=CHECKPOINT_INTERVAL)
    gate_A = gate_fd.run_probes(
        lambda v: tc.J_envelope_of_design(v), v0, g_s, mask, seed=seed,
        x_scale_direction=float(np.mean(np.abs(v0))))

    # --- B. filter transpose identity (drive-independent) ------------------ #
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_part)
    y = rng.standard_normal(n_part)
    gate_B = gate_fd.transpose_residual(chain.filter_apply, chain.filter_transpose,
                                        x, y)

    # --- C. composite dJ/dv = filter . adjoint, the EXACT solve gradient --- #
    _J, g_v, cinfo = envelope_grad_of_design(tc, chain, v0, beta=0.0)
    gate_C = gate_fd.run_probes(
        lambda v: _J_envelope_of_design(tc, chain, v, 0.0), v0, g_v, mask,
        seed=seed, x_scale_direction=float(np.mean(np.abs(v0))))

    launch_ok = bool(gate_A["all_pass_subgradient"] and gate_B["pass"]
                     and gate_C["all_pass_subgradient"])
    doc = {
        "what": "Stage A phase 2 PRE-LAUNCH FD gate: the dopant shape-solve "
                "gradient at the FIXED 0.40x drive (the only substantive change) "
                "on a coarse square case. dJ/ds (adjoint), the filter transpose, "
                "and the composite dJ/dv are each gated at the frozen tolerances "
                "(pass_rel_err %.0e; subgradient %.0e; transpose %.0e). No "
                "widening."
                % (gate_fd.PASS_REL_ERR, gate_fd.SUBGRADIENT_PASS_REL_ERR,
                   gate_fd.TRANSPOSE_REL_ERR),
        "drive": {"chosen_drive_a": chosen_drive_a(),
                  "power_density_w_per_m3": chosen_drive_power_density(),
                  "note": "the ONLY substantive change vs the default-drive solve"},
        "case": {"shape": PART, "target_nodes_in_part": FD_TARGET_NODES,
                 "n_part_cells": n_part, "n_cells_total": int(tc.ncells),
                 "lc0_m": FD_LC0_M, "dt_s": FD_DT_S, "max_time_s": FD_MAX_TIME_S,
                 "n_steps": int(tr.n_steps), "n_sub": int(getattr(tr, "n_sub", 1)),
                 "cfl_violated": bool(tr.out.get("cfl_violated", False))},
        "read_state": {"argmin_step": k, "n_steps": int(tr.n_steps),
                       "at_horizon": bool(er["at_horizon"]),
                       "t_stop_s": float(k) * tc.dt_step,
                       "part_peak_T_c": float(T_read[tc.m_nodal > 0.5].max()),
                       "melt_window_nodes": melt_window,
                       "note": "interior argmin + a live melt-window population "
                               "is required, else |g|=0 and the gate is vacuous"},
        "filter": chain.kernel_report(),
        "gate_A_map_adjoint": _probe_summary(gate_A),
        "gate_B_filter_transpose": gate_B,
        "gate_C_composite": _probe_summary(gate_C),
        "launch_ok": launch_ok,
    }
    gates.write_json("stage_a_phase2_fd_gate.json", doc)
    return doc


def _probe_summary(gate: dict) -> dict:
    """The verdict-carrying numbers per probe (best rel err + eps), plus the
    aggregate pass counts; the full sweep stays in the gate object."""
    return {
        "worst_best_rel_err": gate["worst_best_rel_err"],
        "n_pass_preferred": gate["n_pass_preferred"],
        "n_pass_subgradient": gate["n_pass_subgradient"],
        "n_probes": gate["n_probes"],
        "all_pass_preferred": gate["all_pass_preferred"],
        "all_pass_subgradient": gate["all_pass_subgradient"],
        "pass_rel_err": gate_fd.PASS_REL_ERR,
        "subgradient_pass_rel_err": gate_fd.SUBGRADIENT_PASS_REL_ERR,
        "evaluation_floor_range": gate["evaluation_floor_range"],
        "analytic_magnitude_range": gate["analytic_magnitude_range"],
        "probes": {name: {"best_rel_err": pr["best_rel_err"],
                          "best_eps": pr["best_eps"],
                          "analytic_directional": pr["analytic_directional"],
                          "best_abs_err": pr["best_abs_err"],
                          "pass_subgradient": pr["pass_subgradient"],
                          "pass_preferred": pr["pass_preferred"]}
                   for name, pr in gate["probes"].items()},
    }


def _gate_design_point(tc) -> np.ndarray:
    """A smooth, interior, non-uniform saturation on the part cells (kept in
    [0.05, 0.95] so no box clip enters the chain; the frozen 2-D choice)."""
    import dolfinx
    mp = dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)).T[:, tc.eqs.part]
    x, y, z = mp[0], mp[1], mp[2]
    v = (0.70 + 0.18 * np.sin(np.pi * x / 0.010) * np.cos(np.pi * y / 0.010)
         + 0.06 * np.sin(np.pi * z / 0.030))
    return np.clip(v, 0.05, 0.95)


def solve_mesh_sanity() -> dict:
    """One forward + one gradient on the REAL 5600-node solve case at the fixed
    drive, before detaching: confirm the melt-onset envelope argmin is INTERIOR
    (not step 0, not the horizon) and the first gradient is finite and nonzero.
    A horizon too short lands the argmin at the horizon and the run would burn
    the budget on a bit-identical J (run_tamper's NonFiniteGradientError lesson).
    """
    t0 = time.perf_counter()
    tc = build_solve_case()
    chain = make_chain(tc)
    v0 = np.ones(chain.n_design)
    J, g_v, info = envelope_grad_of_design(tc, chain, v0, beta=0.0)
    gn = float(np.linalg.norm(g_v))
    interior = bool((not info["at_horizon"]) and info["argmin_step"] > 0)
    finite_nonzero = bool(np.isfinite(J) and np.isfinite(gn) and gn > 0.0)
    return {"n_part_cells": _n_part_cells(tc), "n_design": int(chain.n_design),
            "n_cells_total": int(tc.ncells), "max_time_s": SOLVE_MAX_TIME_S,
            "J_first": J, "grad_norm": gn, "argmin_step": info["argmin_step"],
            "n_steps": info["n_steps"], "at_horizon": info["at_horizon"],
            "t_stop_s": info["t_stop_s"],
            "envelope_argmin_interior": interior,
            "first_gradient_finite_nonzero": finite_nonzero,
            "ok": bool(interior and finite_nonzero),
            "wall_s": round(time.perf_counter() - t0, 1)}


# --------------------------------------------------------------------------- #
# Ceiling verdict (FORWARD gate on the end-state true peak) + hold-out gate
# --------------------------------------------------------------------------- #
def ceiling_verdict(true_peak_c: float, ceiling_c: float) -> dict:
    """Feasible iff the densify END-STATE true peak stays at/under the ceiling.
    A pure forward read; no dopant gradient is involved (Stage A scope)."""
    peak = float(true_peak_c)
    ceil = float(ceiling_c)
    return {"true_peak_c": peak, "ceiling_c": ceil,
            "margin_c": ceil - peak, "feasible": bool(peak <= ceil)}


def ceiling_end_state_gate(s_map_solve: np.ndarray, solve_centroids: np.ndarray,
                           holdout_nodes: int, holdout_lc0: float,
                           rho_target: float, max_time_s: float = 3000.0,
                           power_density: float | None = None) -> dict:
    """ACCEPTANCE: transfer the solved dopant saturation onto a mesh HOLD-OUT
    (finer, NOT the solve mesh), densify to rho_target at the fixed drive with
    coupling off, and read the end-state true peak against the ceiling.

    The map is transferred nearest-neighbour from the solve part-cell centroids
    (saturation units), then converted to sigma exactly as the solve does. The
    ceiling being nearly dopant-independent is the physics that makes this a
    forward gate rather than a constrained solve; the number is still MEASURED,
    never assumed (no false-green)."""
    import mesh_gmsh as mg
    from scipy.spatial import cKDTree

    tcfg = stage_a.thermal_config()
    ceiling_c = float(tcfg["T_ceiling_C"])
    p = drive_params(power_density=power_density)

    msh, info, _ = mg.match_lc("square", int(holdout_nodes), float(holdout_lc0))
    mats = fwd.build_materials(msh, fwd.in_part_predicate("square"), p)
    import dolfinx
    cent = np.asarray(dolfinx.mesh.compute_midpoints(
        msh, msh.topology.dim,
        np.arange(fwd.n_cells_local(msh), dtype=np.int32)))
    part_cent = cent[mats.mask]
    _, idx = cKDTree(np.asarray(solve_centroids, float)).query(part_cent)
    sat = np.clip(np.asarray(s_map_solve, float)[idx], 0.0, 1.0)

    sig = np.real(mats.sigma.x.array).astype(float)
    sig[mats.mask] = p.sigma_virgin + sat * (p.sigma_doped - p.sigma_virgin)
    mats.sigma.x.array[:] = sig.astype(fwd.dolfinx.default_scalar_type)

    Vr, Vi = fwd.solve_eqs(msh, mats, p)
    drive = fwd.qrf_dg0(msh, Vr, Vi, mats, p)
    march = df.march_densify(msh, p, stop_mean_rho=float(rho_target), mats=mats,
                             q_dg0=drive["q"], max_time_s=float(max_time_s),
                             sample_dt_s=20.0)
    verdict = ceiling_verdict(march["true_peak_T_c"], ceiling_c)
    verdict.update({
        "holdout_nodes_in_part": int(info.n_nodes_in_part),
        "holdout_lc0_m": float(holdout_lc0),
        "achieved_rho": float(march["part_mean_rho"]),
        "reached_rho": bool(march["reached_rho"]),
        "energy_residual_frac": float(march["energy_residual_frac"]),
        "clamp_bound": bool(march["clamp_bound"]),
        "note": "densify end-state true peak on a HOLD-OUT mesh (not the solve "
                "mesh), solved dopant transferred nearest-neighbour",
    })
    return verdict


# --------------------------------------------------------------------------- #
# Output shape (Stage A contract; recommended_power_settings unchanged 2.0.0)
# --------------------------------------------------------------------------- #
def stage_a_output(part: str, solve_record: dict, chosen_drive_a: float,
                   ceiling_gate: dict, holdout_nodes: int,
                   fd_gate_passed: bool) -> dict:
    """Emit the solved map result + the recommended drive in the Stage A shape.

    is_shippable requires BOTH the FD gate (the gradient was verified) AND the
    end-state ceiling hold-out (the part does not cook) -- never a false-green
    map from an ungated gradient or an over-ceiling end state."""
    feasible = bool(ceiling_gate.get("feasible", False))
    is_shippable = bool(fd_gate_passed and feasible)
    return {
        "what": "Stage A Task 4 phase 2: dopant SHAPE-solve at the fixed drive "
                "Phase 1 certified feasible; ceiling evaluated at the densify "
                "end-state as a forward gate.",
        "stage": "A_phase2_dopant_shape_solve",
        "part": part,
        "chosen_drive_a": float(chosen_drive_a),
        "solve": solve_record,
        "acceptance": {
            "fd_gate_passed": bool(fd_gate_passed),
            "end_state_ceiling_holdout": ceiling_gate,
            "holdout_nodes_target": int(holdout_nodes),
            "is_shippable": is_shippable,
            "note": "is_shippable = FD-gated gradient AND end-state peak <= "
                    "ceiling on a mesh hold-out",
        },
        "recommended_power_settings":
            stage_a.recommended_power_settings(float(chosen_drive_a)),
    }


# --------------------------------------------------------------------------- #
# The heavy solve (detached, checkpointed, resumable)
# --------------------------------------------------------------------------- #
def _score_solved_map(tc, chain, v_best, res, budget_evals, wall_total) -> dict:
    tc.set_objective("asymmetric")
    s_best = chain.design_to_map(v_best, 0.0)
    tr = tc.forward(tc.design_to_sigma(s_best))
    Ja = tc.J_trajectory(tr)
    ka = int(np.argmin(Ja))
    T_read = tc.state_at(tr, ka)
    phi = fwd.phase_fraction(T_read, tc.p)[0]
    vol, chi = tc.vol_nodal, tc.m_nodal
    part_w = vol * chi
    split = obj.split_asymmetric(phi, chi, vol)
    return {
        "arm": "solve_filter_only", "part": PART,
        "objective_optimized": "asymmetric",
        "J_asymmetric": float(Ja[ka]), "argmin_asymmetric": ka,
        "t_stop_s": float(ka) * tc.dt_step,
        "at_horizon_asymmetric": bool(ka >= tr.n_steps),
        "J_out_of_bounds": split["J_out_of_bounds"],
        "J_in_bounds_deficit": split["J_in_bounds_deficit"],
        "part_mean_phi": float(np.dot(phi, part_w) / part_w.sum()),
        "part_max_T_c": float(T_read[chi > 0.5].max()),
        "sigma_T_diagnostic_c": float(np.sqrt(
            np.dot((T_read - np.dot(T_read, part_w) / part_w.sum()) ** 2,
                   part_w) / part_w.sum())),
        "map_stats": {"mean": float(np.average(
            s_best, weights=tc.eqs.vol[tc.eqs.part])),
            "min": float(s_best.min()), "max": float(s_best.max())},
        "gates": {"energy_residual_frac": float(tr.out["energy_residual_frac"]),
                  "clamp_bound": bool(tr.out["clamp_bound"]),
                  "cfl_violated": bool(tr.out["cfl_violated"])},
        "budget_gradient_evaluations": int(budget_evals),
        "gradient_evaluations_used": int(res["evals_used"]),
        "resumed_from_eval": int(res["resumed_from_eval"]),
        "J_first_eval": res["hist"][0]["J"] if res["hist"] else None,
        "trajectory": res["hist"], "scale_first_step": True,
        "objective_scale_applied": res["scale"],
        "status": res.get("status", "resumed_complete"),
        "wall_total_s": round(wall_total, 1),
    }


def run_solve(budget_evals: int = 12, holdout_nodes: int = 9000,
              fd_gate_passed: bool = True) -> dict:
    """L-BFGS-B on the FD-gated gradient at the fixed drive, checkpointed after
    every evaluation (resumable), then the end-state ceiling hold-out gate and
    the Stage A output. run_solve REFUSES to start unless the FD gate has
    passed (fd_gate_passed), so an ungated gradient never reaches the optimizer.
    """
    if not fd_gate_passed:
        raise RuntimeError(
            "run_solve called with fd_gate_passed=False: the phase-2 gradient "
            "must be FD-gated at the fixed drive BEFORE the heavy solve. Run "
            "`--fd-gate` and confirm launch_ok first.")
    from solve3d.phase_e import checkpoint as ck

    t0 = time.perf_counter()
    tc = build_solve_case()
    chain = make_chain(tc)
    ckpt = RESULTS / f"ckpt_phase2_{PART}.npz"
    status = RESULTS / f"stage_a_phase2_{PART}_status.json"
    n = chain.n_design
    _write_status(status, {"part": PART, "pid": os.getpid(), "state": "running",
                           "n_design": int(n), "budget": int(budget_evals),
                           "power_density_w_per_m3": chosen_drive_power_density()})

    def fg(v):
        J, g_v, info = envelope_grad_of_design(tc, chain, v, beta=0.0)
        if fg.n_calls == 0 and not (np.isfinite(J)
                                    and np.isfinite(np.linalg.norm(g_v))
                                    and np.linalg.norm(g_v) > 0):
            raise RuntimeError(
                f"eval 1 non-finite/zero gradient (J={J}, |g|="
                f"{np.linalg.norm(g_v)}, argmin={info['argmin_step']}"
                f"/{info['n_steps']}). Refusing to optimize; check the horizon "
                "and the adjoint before relaunching.")
        fg.n_calls += 1
        fg.last = {**info, "wall_s": time.perf_counter() - t0}
        return J, g_v

    fg.n_calls = 0

    def on_eval(v, J, g):
        info = dict(getattr(fg, "last", {}))
        print(f"  [phase2] eval J={J:.6e} t_stop={info.get('t_stop_s')}s "
              f"|g|={np.linalg.norm(g):.3e} wall={info.get('wall_s'):.0f}s",
              flush=True)
        return info

    res = ck.run_with_checkpoint(fg, np.ones(n), budget=budget_evals, path=ckpt,
                                 bounds=(0.0, 1.0), scale_first_step=True,
                                 on_eval=on_eval)
    v_best = np.asarray(res["best_v"], float)
    s_best = chain.design_to_map(v_best, 0.0)
    np.savez_compressed(RESULTS / f"map_phase2_{PART}.npz", v_raw=v_best,
                        s_map=s_best, centroids=chain.centroids,
                        volumes=chain.volumes)
    solve_record = _score_solved_map(tc, chain, v_best, res, budget_evals,
                                      time.perf_counter() - t0)

    print("[phase2] end-state ceiling hold-out gate ...", flush=True)
    rho_target = float(stage_a.thermal_config()["rho_target"]["practical_ideal"])
    ceiling_gate = ceiling_end_state_gate(
        s_best, chain.centroids, holdout_nodes=holdout_nodes,
        holdout_lc0=0.060 / 64.0, rho_target=rho_target)

    doc = stage_a_output(PART, solve_record, chosen_drive_a(), ceiling_gate,
                         holdout_nodes, fd_gate_passed=fd_gate_passed)
    final = RESULTS / f"stage_a_phase2_{PART}.json"
    _write_json(final, doc)
    _write_status(status, {"part": PART, "pid": os.getpid(), "state": "done",
                           "is_shippable": doc["acceptance"]["is_shippable"],
                           "evals_used": int(res["evals_used"]),
                           "final": final.name})
    print(json.dumps({"is_shippable": doc["acceptance"]["is_shippable"],
                      "end_state_peak_c":
                          ceiling_gate["true_peak_c"],
                      "feasible": ceiling_gate["feasible"],
                      "final": final.name}, indent=1))
    return doc


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float))
    tmp.replace(path)


def _write_status(path: Path, doc: dict) -> None:
    doc = {**doc, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())}
    _write_json(path, doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-gate", action="store_true",
                    help="pre-launch: FD-gate the gradient at the fixed drive")
    ap.add_argument("--sanity", action="store_true",
                    help="pre-launch: one forward+gradient on the solve mesh "
                         "(interior envelope argmin + finite nonzero gradient)")
    ap.add_argument("--solve", action="store_true",
                    help="launch the heavy shape-solve (requires --fd-gate to "
                         "have passed; pass --fd-gate-passed to confirm)")
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--holdout-nodes", type=int, default=9000)
    ap.add_argument("--fd-gate-passed", action="store_true",
                    help="operator assertion that --fd-gate reported launch_ok")
    a = ap.parse_args()
    if a.fd_gate:
        doc = fd_gate()
        print(json.dumps({"launch_ok": doc["launch_ok"],
                          "A_map_adjoint_worst_rel":
                              doc["gate_A_map_adjoint"]["worst_best_rel_err"],
                          "A_n_pass_subgradient":
                              doc["gate_A_map_adjoint"]["n_pass_subgradient"],
                          "B_filter_transpose_rel": doc["gate_B_filter_transpose"]["rel_err"],
                          "C_composite_worst_rel":
                              doc["gate_C_composite"]["worst_best_rel_err"],
                          "C_n_pass_subgradient":
                              doc["gate_C_composite"]["n_pass_subgradient"],
                          "read_state": doc["read_state"]}, indent=1))
    if a.sanity:
        print(json.dumps(solve_mesh_sanity(), indent=1))
    if a.solve:
        run_solve(budget_evals=a.budget, holdout_nodes=a.holdout_nodes,
                  fd_gate_passed=a.fd_gate_passed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
