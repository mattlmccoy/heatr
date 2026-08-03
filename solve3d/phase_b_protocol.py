"""Phase B Task 0: PRE-REGISTER the FD/subgradient gate protocol.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_b_protocol

Emits solve3d/results/phase_b_protocol.json BEFORE any adjoint code exists, so
the thresholds cannot be chosen after seeing a gradient. Everything numeric here
is the FROZEN 2-D convention (FROZEN_CONVENTIONS_2D.md, their commit b04e356),
ported verbatim WITH its citation. Nothing in this file is a new threshold.

The one thing that is measured rather than quoted is the small FD case, and it
is measured because the plan requires evidence, not assertion, that the case
crosses the melt window and triggers at least two EQS re-solves.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from solve3d import forward, gates

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "phase_b_protocol.json"

# --------------------------------------------------------------------------- #
# The FD case (values pinned here; the MEASURED properties are filled by
# measure_case() below and must satisfy the Task-0 conditions)
# --------------------------------------------------------------------------- #
FD_CASE = {
    "shape": "circle",
    "target_nodes_in_part": 2500,
    "lc0_m": 0.0026,
    "dt_s": 0.5,
    "power_density_multiplier": 2.0,
    "eqs_update_interval_s": 25.0,
    "sigma_temp_coeff_per_K": -0.002,
    "max_time_s": 100.0,
    "phi_target": 2.0,          # never stop early; fixed read at the horizon
    "objective": "J = sum_i vol_i * (phi(T_i) - chi_i)^2 over ALL nodes",
}


def fd_case_params() -> forward.ForwardParams:
    base = forward.ForwardParams()
    return dataclasses.replace(
        base,
        dt_s=FD_CASE["dt_s"],
        power_density_w_per_m3=base.power_density_w_per_m3
        * FD_CASE["power_density_multiplier"],
        eqs_update_interval_s=FD_CASE["eqs_update_interval_s"],
        sigma_temp_coeff_per_K=FD_CASE["sigma_temp_coeff_per_K"],
    )


def measure_case() -> dict:
    p = fd_case_params()
    t0 = time.perf_counter()
    out = forward.run_forward(FD_CASE["shape"], FD_CASE["target_nodes_in_part"],
                              FD_CASE["lc0_m"], p=p,
                              max_time_s=FD_CASE["max_time_s"],
                              phi_target=FD_CASE["phi_target"],
                              sample_dt_s=FD_CASE["eqs_update_interval_s"])
    wall = time.perf_counter() - t0
    phi = forward.phase_fraction(out["T_phi90"], p)[0]
    w_part = out["vol_nodal"] * out["m_nodal"]
    in_window = int(np.count_nonzero((phi > 0.0) & (phi < 1.0)))
    return {
        "wall_forward_s": wall,
        "n_dofs_total": int(out["n_dofs_total"]),
        "n_cells_total": int(out["n_cells_total"]),
        "n_nodes_in_part": int(out["n_nodes_in_part"]),
        "n_eqs_solves": int(out["n_eqs_solves"]),
        "resolve_times_s": [float(t) for t in out["resolve_times_s"]],
        "n_substeps_used": int(out["n_substeps_used"]),
        "dt_stable_s": float(out["dt_stable_s"]),
        "n_march_steps": int(FD_CASE["max_time_s"] / FD_CASE["dt_s"]),
        "part_mean_phi": float(np.dot(phi, w_part) / w_part.sum()),
        "n_nodes_in_melt_window": in_window,
        "n_nodes_total": int(phi.size),
        "T_max_c": float(out["T_max_c"]),
        "energy_residual_frac": float(out["energy_residual_frac"]),
        "clamp_bound": bool(out["clamp_bound"]),
    }


CHECKLIST = [
    {"n": 1, "item": "L0 bit identity: the forward reproduces a stored run in "
                     "the same channel to the last bit, and any NEW channel is "
                     "bit-identical to the old one when its flag is off",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 1",
     "status": "ported",
     "how": "Phase A's forward is frozen; any forward.py hook added for state "
            "recording is proven bit-identical when off "
            "(test_adjoint_transient.py::test_recording_flag_off_is_bit_identical)"},
    {"n": 2, "item": "Layered bisect: add ONE thing per layer so a failure "
                     "localizes",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 2",
     "status": "ported",
     "how": "B1 steady EQS adjoint (J of Q_rf only) -> B2 + transient reverse "
            "march at fixed read time -> B3 + design-field composition -> "
            "B4 + envelope stop time -> B5 + checkpointing"},
    {"n": 3, "item": "Three-probe protocol per layer, plus two: max-sensitivity "
                     "cell, fixed pseudo-random in-part cell, random unit "
                     "direction; plus a filter-smooth direction when a filter is "
                     "in the chain, plus the gradient direction",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 3",
     "status": "ported",
     "how": "max_sensitivity_cell, random_cell, random_direction, "
            "gradient_direction. The filter-smooth probe is Phase C: there is "
            "no filter in the Phase B chain (see checklist item 8)"},
    {"n": 4, "item": "Central differences, epsilon swept over eight values, "
                     "expect a V-shaped relative error bottoming between 1e-6 "
                     "and 1e-7",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 4 (gate_rho.py:47-52)",
     "status": "ported"},
    {"n": 5, "item": "Pass standard 1e-6 preferred, 1e-5 is the campaign "
                     "SUBGRADIENT standard; report the count at both",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 5 "
                 "(gate_rho.PASS_REL_ERR / SUBGRADIENT_PASS_REL_ERR)",
     "status": "ported",
     "how": "the objective reads a CLIPPED melt fraction, so melt-window cells "
            "are kinks and a smooth-function standard is not available; the "
            "measured case keeps a large in-window population on purpose"},
    {"n": 6, "item": "Measure the evaluation floor, do not assume it: in the "
                     "roundoff-dominated tail the central-difference error is "
                     "floor/(2 eps), so 2*eps*abs_err at the two smallest "
                     "epsilons estimates the objective's absolute floor",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 6",
     "status": "ported",
     "reference_2d_measurement": "3.2e-12 to 1.0e-10 across 40 probes"},
    {"n": 7, "item": "Interpret a relative-error failure against the analytic "
                     "magnitude; report the scatter of ABSOLUTE error against "
                     "analytic magnitude, not only the relative number",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 7",
     "status": "ported"},
    {"n": 8, "item": "Filter and projection transpose exactness by the "
                     "dot-product identity, threshold 1e-10",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 8",
     "status": "ported",
     "how": "no filter/projection exists in Phase B (that chain is Phase C), "
            "but the discipline is applied to the linear operators Phase B DOES "
            "introduce: the P1 cell-average operator used for the sigma(T) "
            "coupling and its scatter transpose, checked by the same "
            "dot-product identity at the same 1e-10 threshold"},
    {"n": 9, "item": "Read-state stability: report whether the objective's "
                     "argmin index moves under the probe perturbations rather "
                     "than assuming the envelope theorem covers it",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 9",
     "status": "ported",
     "how": "Task 2 reads at a FIXED index so the question is deferred; Task 4 "
            "introduces the envelope read and reports argmin movement"},
    {"n": 10, "item": "Flag-off bit identity for every new channel",
     "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 10",
     "status": "ported"},
]

MUTATION_TESTS = [
    {"name": "renorm_frozen",
     "what": "treat the fixed-power renormalization scale as a constant",
     "must": "FAIL the FD gate",
     "d1_precedent": "heatr3d_d1_spike/results.json task5.mutations: directional "
                     "rel err 0.017159885771654493, up to 1.4983795363711947 "
                     "(150%) on an individual dof"},
    {"name": "adjoint_dropped",
     "what": "keep only the explicit partial, no adjoint solve",
     "must": "FAIL the FD gate",
     "d1_precedent": "heatr3d_d1_spike/results.json task5.mutations: directional "
                     "rel err 0.46174542475102587"},
]


def build() -> dict:
    measured = measure_case()
    doc = {
        "what": "Phase B FD/subgradient gate protocol, PRE-REGISTERED before "
                "any adjoint code. Thresholds are the frozen 2-D ones, ported "
                "verbatim with citations.",
        "source_of_thresholds": "FROZEN_CONVENTIONS_2D.md (repo root, commit "
                                "b04e356), section 5",
        "semantic_template": "fgm_solve_campaign/adjoint2d/ (READ ONLY): "
                             "adjoint.substep_vjp / reverse_march / eqs_vjp, "
                             "and gate_rho.py for the probe protocol",
        "fd": {"scheme": "central",
               "epsilons": [1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8],
               "citation": "FROZEN_CONVENTIONS_2D.md section 5 item 4 "
                           "(adjoint2d/gate_rho.py:47-52)",
               "expectation": "V-shaped relative error bottoming between 1e-6 "
                              "and 1e-7"},
        "thresholds": {
            "pass_rel_err": 1e-6,
            "subgradient_pass_rel_err": 1e-5,
            "transpose_rel_err": 1e-10,
            "citations": {
                "pass_rel_err": "FROZEN_CONVENTIONS_2D.md section 5 item 5 "
                                "(gate_rho.PASS_REL_ERR)",
                "subgradient_pass_rel_err": "FROZEN_CONVENTIONS_2D.md section 5 "
                                            "item 5 (gate_rho.SUBGRADIENT_PASS_REL_ERR)",
                "transpose_rel_err": "FROZEN_CONVENTIONS_2D.md section 5 item 8 "
                                     "(2-D measured 4.19e-16 worst)",
            },
            "no_widening_rule": "if a 3-D gate needs a different threshold, "
                                "MEASURE the reason and record it; never widen "
                                "to pass",
        },
        "probes": ["max_sensitivity_cell", "random_cell", "random_direction",
                   "gradient_direction"],
        "probe_citation": "FROZEN_CONVENTIONS_2D.md section 5 item 3 "
                          "(adjoint2d/gate_rho._probe_dirs)",
        "checklist": CHECKLIST,
        "mutation_tests": MUTATION_TESTS,
        "conventions": {
            "actuator": "conductivity_only",
            "eps_channel": "OFF",
            "outside_part_saturation": 1.0,
            "citations": {
                "actuator": "FROZEN_CONVENTIONS_2D.md section 7 -- conductivity "
                            "only is the DEPLOYABLE channel; the permittivity "
                            "channel is MODEL ONLY and must default OFF",
                "outside_part_saturation": "FROZEN_CONVENTIONS_2D.md section 4 "
                                           "-- outside the part the saturation "
                                           "is held at 1.0, the nominal value",
            },
            "chi": "the DG0 doped indicator. NOTE the 2-D lane's sub-cell "
                   "area-fill chi (their section 2) is NOT needed here: the "
                   "mesh CONFORMS to the part boundary, so every cell lies "
                   "wholly inside or wholly outside and the indicator is exact "
                   "with no partial cells to rasterize. The nodal form used by "
                   "the objective is the exact volume fraction m_i, which sums "
                   "to the exact part volume.",
        },
        "cost_accounting": {
            "forward_equivalent": "wall time of ONE gradient evaluation "
                                  "(reverse march + all adjoint solves, "
                                  "EXCLUDING the forward that produced the "
                                  "trajectory) divided by the wall time of one "
                                  "forward on the same case and machine",
            "also_reported": "(forward + gradient) / forward, the cost of a "
                             "gradient from scratch",
            "target": "<= ~2 forward-equivalents CHECKPOINTED",
            "reference_2d": "1.4-1.7 store-everything under 2-D memory "
                            "conditions; 3-D must checkpoint",
            "timing_caveat": "FROZEN_CONVENTIONS_2D.md section 6 records that "
                             "timing a single forward/adjoint on a loaded "
                             "machine gave a ratio of 12.91 against the "
                             "campaign's 1.11 for the same calls; cost is "
                             "therefore measured on an otherwise idle machine "
                             "and the load state is recorded",
        },
        "fd_case": {
            **FD_CASE,
            "measured": measured,
            "justification":
                "Chosen to make central differences affordable while exercising "
                "every Phase B code path. MEASURED: the forward costs "
                f"{measured['wall_forward_s']:.2f} s, so the 8-epsilon x 4-probe "
                "central-difference sweep is ~2 minutes per layer. It performs "
                f"{measured['n_eqs_solves']} EQS solves (1 pre-loop + "
                f"{len(measured['resolve_times_s'])} in-march re-solves at "
                f"{measured['resolve_times_s']} s), so the sigma(T) coupling VJP "
                "and the adjoint restart at re-solve boundaries are both "
                "exercised more than once. The part-mean melt fraction is "
                f"{measured['part_mean_phi']:.4f} with "
                f"{measured['n_nodes_in_melt_window']} of "
                f"{measured['n_nodes_total']} nodes strictly inside the melt "
                "window, so the melt front sits INSIDE the part: the "
                "clip subgradients are live (which is the whole reason the 1e-5 "
                "subgradient standard exists) rather than the objective being "
                "saturated at 0 or 1 everywhere. Deviations from the Phase A "
                "anchor are deliberate and named: a coarse mesh, dt_s = 0.5 s "
                "(still CFL-stable, dt_stable = "
                f"{measured['dt_stable_s']:.3f} s, n_substeps = "
                f"{measured['n_substeps_used']}), and a 2x power density so melt "
                "onset arrives in 100 s. The gate validates the DISCRETE "
                "gradient of the DISCRETE forward, so a coarser discretization "
                "is a valid gate case; it is NOT a physics claim and no Phase A "
                "number is restated from it.",
            "representativeness_limits":
                "single shape (circle), single mesh, coupling ON only. The "
                "coupling-OFF path is covered by the same code with the "
                "schedule inert, and that inertness is a flag-off bit-identity "
                "check, not an FD gate.",
        },
    }
    gates.write_json(OUT.name, doc)
    return doc


def main() -> int:
    doc = build()
    print(json.dumps(doc["fd_case"]["measured"], indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
