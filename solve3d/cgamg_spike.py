"""Task 3 CG/AMG spike: benchmark + the binding FD re-gate, recorded to JSON.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      heatr3d_d1_spike/env/bin/python -m solve3d.cgamg_spike

Protocol: solve3d/results/cgamg_protocol.json, pre-registered and committed
(091863d) BEFORE any iterative solver code existed. This module measures; it
does not decide anything the protocol did not already bind.

Two arms, and only one of them decides:
  * field agreement and wall-clock speedup are RECORDED at every rtol;
  * the FD re-gate is the BINDING criterion. The direct path solves to ~1e-19
    relative residual, so an iterative solve at 1e-10 is nine orders looser,
    and central differences are exactly where that surfaces. A run can look
    perfect in the fields and still have lost the gradient.

Keeping the direct path is a valid outcome and is reported as a result, not as
a failure to be worked around.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from solve3d import adjoint, gate_fd, precomp
from solve3d.phase_e import run as R

RESULTS = Path(__file__).resolve().parent / "results"
RTOLS = (1e-10, 1e-12, 1e-14)
# FD-gate arms actually RUN. The pre-registered selection rule is "loosest
# rtol passing both gates", so if the loosest passes it IS the answer and the
# tighter ones are implied; 1e-14 is kept as the tight control. Each arm is a
# real ~39 minute sweep (8 epsilons x 4 probes x 2 solves on 24784 complex
# dofs), so running arms whose outcome the rule cannot use would be waste.
FD_RTOLS = (1e-10, 1e-14)
BENCH_LEVELS = ((2.0e-3, "coarse"), (1.2e-3, "mid"))
REPEATS = 3


def _bench_level(lc: float, name: str) -> dict:
    """Direct vs iterative on ONE assembled operator and one machine state."""
    tc = R.build_case("cube", lc_part=lc,
                      precomp_coeffs=precomp.ShrinkageL0(0.0, 0.0))
    e = tc.eqs
    e.set_solver("direct")
    wd = []
    st_d = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        st_d = e.solve_state()
        wd.append(time.perf_counter() - t0)
    v_d, q_d = e.Vfun.x.array.copy(), st_d.q.copy()
    rec = {"lc_part_m": lc, "n_dofs": int(v_d.size), "n_cells": int(tc.ncells),
           "direct": {"wall_best_s": min(wd), "wall_median_s": sorted(wd)[1],
                      "res_norm": float(st_d.res_norm)},
           "iterative": {}}
    for rt in RTOLS:
        e.set_solver("iterative", rtol=rt)
        wi = []
        st_i = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            st_i = e.solve_state()
            wi.append(time.perf_counter() - t0)
        v_i, q_i = e.Vfun.x.array.copy(), st_i.q.copy()
        rec["iterative"]["%g" % rt] = {
            "wall_best_s": min(wi), "wall_median_s": sorted(wi)[1],
            "speedup_vs_direct_best": min(wd) / min(wi),
            "res_norm": float(st_i.res_norm),
            "V_rel_l2_vs_direct":
                float(np.linalg.norm(v_i - v_d) / np.linalg.norm(v_d)),
            "Q_rel_l2_vs_direct":
                float(np.linalg.norm(q_i - q_d) / np.linalg.norm(q_d))}
    e.set_solver("direct")
    return rec


def _fd_gate(kind: str, rtol: float | None) -> dict:
    """The pre-registered probe, with the chosen solver in the loop."""
    case = adjoint.SteadyCase.build(shape="circle",
                                    target_nodes_in_part=23040, lc0=0.0009375)
    if kind == "iterative":
        case.eqs.set_solver("iterative", rtol=rtol)
    # reuse=False is MANDATORY here. run_fd_gate(reuse=True) returns the
    # cached phase_b_steady_gate.json artifact WITHOUT SOLVING, which on the
    # first pass of this spike produced four arms with wall_s = 0.0 and
    # rel_errs identical to twelve significant figures: a false green that
    # measured the solver swap not at all.
    t0 = time.perf_counter()
    doc = case.run_fd_gate(reuse=False)
    wall = time.perf_counter() - t0
    if wall < 1.0:
        raise RuntimeError(
            f"FD gate returned in {wall:.3f}s: it did not actually solve, so "
            f"this arm measures nothing")
    g = doc["gate"]
    probes = {k: float(v["best_rel_err"]) for k, v in g["probes"].items()}
    return {"solver": kind, "rtol": rtol, "wall_s": wall,
            "all_pass_subgradient": bool(g["all_pass_subgradient"]),
            "n_pass_subgradient": int(g["n_pass_subgradient"]),
            "n_pass_preferred": int(g["n_pass_preferred"]),
            "n_probes": int(g["n_probes"]),
            "worst_best_rel_err": max(probes.values()),
            "probes": probes}


def main() -> int:
    out = {
        "what": "Task 3 CG/AMG spike RESULTS",
        "protocol": "solve3d/results/cgamg_protocol.json (committed 091863d)",
        "solver": ("GMRES + GAMG (PETSc). CG rejected a priori: the EQS "
                   "operator is complex SYMMETRIC, not Hermitian positive "
                   "definite, so CG has no convergence guarantee here."),
        "env": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        "benchmarks": {}, "fd_regate": {},
    }
    for lc, name in BENCH_LEVELS:
        out["benchmarks"][name] = _bench_level(lc, name)
        print(name, json.dumps(out["benchmarks"][name]["iterative"]),
              flush=True)

    out["fd_regate"]["direct_control"] = _fd_gate("direct", None)
    print("direct_control", out["fd_regate"]["direct_control"]
          ["all_pass_subgradient"],
          "worst %.3e" % out["fd_regate"]["direct_control"]
          ["worst_best_rel_err"], flush=True)
    for rt in FD_RTOLS:
        k = "iterative_%g" % rt
        out["fd_regate"][k] = _fd_gate("iterative", rt)
        print(k, out["fd_regate"][k]["all_pass_subgradient"],
              "worst %.3e" % out["fd_regate"][k]["worst_best_rel_err"],
              flush=True)

    out["thresholds"] = {
        "subgradient_pass_rel_err": gate_fd.SUBGRADIENT_PASS_REL_ERR,
        "pass_rel_err": gate_fd.PASS_REL_ERR}

    passing = [k for k, v in out["fd_regate"].items()
               if k != "direct_control" and v["all_pass_subgradient"]]
    out["verdict"] = {
        "fd_gate_is_binding": True,
        "control_passed": out["fd_regate"]["direct_control"]
        ["all_pass_subgradient"],
        "rtols_passing_fd_gate": passing,
        "selected_rtol": (max(float(k.split("_")[1]) for k in passing)
                          if passing else None),
        "fd_arms_run": list(FD_RTOLS),
        "selection_rule": ("loosest rtol passing both gates (pre-registered); "
                           "tighter rtols than the loosest passing one are "
                           "implied and not run"),
        "recommendation": ("adopt GMRES+GAMG at the selected rtol behind the "
                           "flag" if passing else
                           "KEEP THE DIRECT PATH: no tested rtol preserved "
                           "the pre-registered gradient standard"),
    }
    p = RESULTS / "cgamg_results.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print("WROTE", p)
    print("VERDICT", json.dumps(out["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
