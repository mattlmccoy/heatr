"""Phase C Task 0: PRE-REGISTER the first-solve campaign.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_c_prereg

Emits solve3d/results/phase_c_preregistration.json BEFORE any objective or
solve code exists. Budgets, arms, objective weightings and acceptance bands are
fixed here so none of them can be chosen after seeing a result.

Everything numeric is either (a) a frozen 2-D convention quoted with its
citation, (b) a spread ALREADY MEASURED and recorded in
solve3d/results/phase_a_shape_gate.json, or (c) a cost MEASURED at solve scale
and pasted in with the command that produced it. Nothing is a fresh guess.
"""
from __future__ import annotations

import json
from pathlib import Path

from solve3d import gates

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "phase_c_preregistration.json"
SAFETY = 1.5

# --------------------------------------------------------------------------- #
# Cost, MEASURED at solve scale before this file was written.
# Command (spike env, OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1):
#   TransientCase.build('circle', 23040, 0.0009375, ForwardParams(),
#                       max_time_s=500.0, sample_dt_s=50.0)
#   forward, then gradient(read_step=argmin, checkpoint_interval=25)
# --------------------------------------------------------------------------- #
MEASURED = {
    "mesh": "phase_a_coarse (target 23040 in-part nodes)",
    "n_cells_total": 143892,
    "n_nodes_total": 24784,
    "dt_s": 0.05,
    "max_time_s": 500.0,
    "n_march_steps": 10000,
    "n_eqs_solves": 1,
    "wall_forward_s": 135.6,
    "wall_gradient_s": 315.8,
    "gradient_forward_equivalents": 2.328,
    "per_gradient_eval_forward_equivalents": 3.328,
    "checkpoint_interval": 25,
    "envelope_argmin_step": 7279,
    "envelope_argmin_t_s": 364.0,
    "envelope_at_horizon": False,
    "J_symmetric_at_start": 1.76380e-05,
    "J_symmetric_at_argmin": 9.85503e-08,
    "J_symmetric_at_horizon": 4.40057e-06,
    "note_cost_did_not_transfer":
        "Phase B measured 1.774 gradient forward-equivalents on its small FD "
        "case; at solve scale the same code measures 2.328, ABOVE the Phase B "
        "<= ~2 target. This is exactly the non-transfer FROZEN_CONVENTIONS_2D "
        "section 6 warns about (their 12.91 vs 1.11 on the star shape), which "
        "is why the budget below is converted with the SOLVE-SCALE number and "
        "not with Phase B's. Recorded as a finding, not smoothed over.",
}


def _phase_a_spread(metric: str) -> float:
    d = json.loads((RESULTS / "phase_a_shape_gate.json").read_text())
    return float(d["self_spread_detail"]["circle"]["dolfinx_coarse_vs_mid"][metric])


def build() -> dict:
    n_grad = int(40 // MEASURED["per_gradient_eval_forward_equivalents"])
    holdout_metrics = ("jaccard_dist_phi0p8", "jaccard_dist_phi0p9",
                       "front_ssd_mm", "in_part_absdiff_phi0p8",
                       "in_part_absdiff_phi0p9", "bed_melt_absdiff_phi0p8",
                       "bed_melt_absdiff_phi0p9")
    doc = {
        "what": "Phase C first-solve pre-registration (cylinder null). Frozen "
                "BEFORE any objective or solve code.",
        "case": {
            "shape": "circle",
            "geometry": "Phase A anchor: extruded circle d = 20 mm, full height, "
                        "60 mm chamber",
            "coupling": "OFF (eqs_update_interval_s = 0)",
            "why_coupling_off":
                "the heatr3d FGM benefit re-run this campaign is compared "
                "against ran every march through run(qrf_override=...), which "
                "skips the in-march EQS entirely. Coupling OFF is the "
                "like-for-like setting; the coupled path is Phase B-gated and "
                "available but would make the inversion comparison "
                "actuator-mismatched.",
            "drive": "the Phase A forward's fixed-power renormalization "
                     "convention. Drive reconciliation is Phase D and is NOT "
                     "done here (spec section 5).",
            "stop_rule": "envelope: t_stop = argmin over the arm's OWN stored "
                         "trajectory (Phase B layer B4 machinery, exact "
                         "agreement 0.0). at_horizon is flagged, and an arm "
                         "whose minimum sits on the last step reports J as an "
                         "UPPER BOUND.",
        },
        "budget": {
            "forward_equivalents_per_arm": 40,
            "citation": "FROZEN_CONVENTIONS_2D.md section 6 "
                        "(library_solve.BUDGET_FORWARD_EQUIVALENTS)",
            "measured_at_solve_scale": MEASURED,
            "gradient_evaluations_per_arm": n_grad,
            "conversion": "40 / per_gradient_eval_forward_equivalents "
                          f"= 40 / {MEASURED['per_gradient_eval_forward_equivalents']} "
                          f"-> {n_grad} gradient evaluations",
            "wall_estimate_min_per_solve_arm":
                n_grad * (MEASURED["wall_forward_s"] + MEASURED["wall_gradient_s"]) / 60.0,
            "unrun_arms_policy":
                "Arms run in the pre-registered priority order. Any arm not "
                "reached within the session's compute is recorded in the "
                "results JSON with status NOT_RUN and its reason. An unrun arm "
                "is never silently dropped and never back-filled with an "
                "estimate.",
        },
        "design_chain": {
            "filter_radius_m": 1.0e-3,
            "filter_citation": "FROZEN_CONVENTIONS_2D.md section 1.2 -- a "
                               "PHYSICAL length, solver-convergence justified, "
                               "explicitly NOT the printer's 50-100 um edge "
                               "scale (which is finer than any solve cell and "
                               "would be equivalent to no filter)",
            "projection": {"eta": 0.5, "beta_schedule": [1, 2, 4, 8, 16],
                           "citation": "FROZEN_CONVENTIONS_2D.md section 1.3"},
            "actuator": "conductivity only; outside-part saturation 1.0 "
                        "(sections 7 and 4)",
            "box": [0.0, 1.0],
            "start": "single full-depth cold start from uniform saturation 1.0 "
                     "(section 6)",
            "arms_note": "MMA_RETEST_REPORT.md (their eaf3aec) settles the "
                         "shape of this chain: filter-only is the in-grid "
                         "production recipe, projection is the robustness arm. "
                         "BOTH are run, and filter-only is the primary.",
        },
        "objective": {
            "source": "spec commit d298c6d, Phase C OBJECTIVE REFINEMENT "
                      "(Matt, 2026-08-01)",
            "read": "envelope stop time; nodal lumped quadrature, chi_i = the "
                    "exact nodal volume fraction m_i",
            "symmetric_control": {
                "formula": "J_sym = sum_i vol_i * (phi_i - chi_i)^2",
                "why": "the Phase B objective and the 2-D lane's functional; "
                       "run in EVERY arm so this campaign stays comparable "
                       "across lanes even though it is not the primary."},
            "asymmetric": {
                "formula":
                    "J_asym = W_out * sum_i vol_i * max(phi_i - chi_i, 0)^2 "
                    "+ W_in * sum_i vol_i * max(phi_floor*chi_i - phi_i, 0)^2",
                "phi_floor": 0.85,
                "phi_floor_why": "the middle of Matt's stated 80-90 % density "
                                 "band; in-bounds melt above the floor carries "
                                 "NO penalty, which is what makes the in-bounds "
                                 "side soft",
                "w_out_over_w_in": 10.0,
                "sensitivity_arm_w_out_over_w_in": 3.0,
                "not_tuned": True,
                "justification":
                    "The hinge already encodes 'soft' on the in-bounds side, so "
                    "the ratio only has to express 'hard versus soft'. One "
                    "order of magnitude is the coarsest non-trivial encoding of "
                    "that and deliberately does NOT imply a calibrated "
                    "trade-off, which we have not measured and have no data to "
                    "measure. The 3x sensitivity arm exists to show whether any "
                    "ranking depends on the choice; if the ranking flips "
                    "between 10x and 3x, the conclusion is that the ratio "
                    "matters and must be calibrated, and that will be reported "
                    "as the finding rather than resolved by picking one.",
                "normalization": "W_in = 1, W_out = the ratio; J values are "
                                 "therefore comparable within a weighting arm "
                                 "and NOT across weighting arms.",
            },
            "sigma_T": "REPORTED DIAGNOSTIC ONLY, never the objective "
                       "(spec, and the Phase A close-out verdict rule).",
        },
        "arms": [
            {"priority": 1, "name": "uniform_baseline",
             "design": "v = 1.0 everywhere in the part",
             "cost_forward_equivalents": 1,
             "role": "the reference every delta is measured against, and the "
                     "arm whose OWN mesh move bounds the hold-out band"},
            {"priority": 2, "name": "solve_filter_only",
             "design": "L-BFGS-B on the filtered design, beta = 0 (no "
                       "projection); the PRIMARY solve arm",
             "cost_forward_equivalents": 40},
            {"priority": 3, "name": "inversion_map",
             "design": "the corrected-design (masked-physics) inversion map, "
                       "heatr3d.make_fgm(magnitude=1.0, baseline=0.5, bpp=2)",
             "cost_forward_equivalents": 1,
             "provenance": {
                 "report": "heatr3d_eqs02_rerank/FGM_BENEFIT_RERUN.md "
                           "(commit ea4fe01), arm D, cylinder",
                 "recorded_result": "+0.3 % sigma_T vs the masked baseline -- "
                                    "the cylinder NULL this campaign exists to "
                                    "test against",
                 "artifact_status": "not_saved_regenerated_from_make_fgm",
                 "artifact_search":
                     "heatr3d_eqs02_rerank/ contains only JSON shards and "
                     "reports; no .npz dopant map was written by "
                     "run_fgm_rerun.py (checked: no savez call for the map). "
                     "The map is however EXACTLY reproducible, because "
                     "make_fgm is a pure function of the masked-baseline "
                     "T_phi90 and the part mask, both of which this repo "
                     "already stores in "
                     "solve3d/results/anchor_heatr3d_circle_n{64,96}.npz.",
                 "reproduction_check": {
                     "target_mean_sat": 0.4074,
                     "source": "FGM_BENEFIT_RERUN.md table 'How much did the "
                               "map itself change?', cylinder, "
                               "'mean sat (masked-designed)', at n = 64",
                     "tolerance_rel": 1e-3,
                     "why": "regenerating at n=64 must reproduce the recorded "
                            "mean saturation; that is what proves this arm is "
                            "the published null's map and not a lookalike"},
                 "transfer": {
                     "method": "trilinear_then_clip",
                     "citation": "the 3-D analogue of the 2-D production "
                                 "transfer robust.resample_map (bilinear then "
                                 "clip, reproducing rfam_eqs_coupled.py:374-380), "
                                 "FROZEN_CONVENTIONS_2D.md section 8 Gate A",
                     "reported": "mean saturation and total dopant before and "
                                 "after transfer, so the transfer loss is "
                                 "visible rather than assumed small"},
                 "drop_condition":
                     "If the n=64 regeneration does not reproduce mean sat "
                     "0.4074 to 1e-3 relative, OR the trilinear transfer moves "
                     "total in-part dopant by more than 2 %, this arm is "
                     "DROPPED and the reason recorded. It is never replaced by "
                     "an approximation presented as the published map."}},
            {"priority": 4, "name": "solve_projection_beta_continuation",
             "design": "L-BFGS-B with the smoothed-Heaviside projection, "
                       "eta = 0.5, beta continuation 1/2/4/8/16, budget split "
                       "by topopt.stage_split",
             "cost_forward_equivalents": 40,
             "role": "the robustness arm (MMA_RETEST_REPORT.md)"},
            {"priority": 5, "name": "solve_filter_only_w3",
             "design": "same as solve_filter_only with the asymmetric weight "
                       "ratio at 3x instead of 10x",
             "cost_forward_equivalents": 40,
             "role": "the pre-registered weight SENSITIVITY arm"},
        ],
        "acceptance": {
            "mesh_holdout": {
                "solve_mesh": "phase_a_coarse",
                "score_mesh": "phase_a_mid",
                "what": "re-run the FORWARD (not the solve) for the delivered "
                        "map on the finer mesh and score it there",
                "band_rule":
                    "band(metric) = 1.5 * (the dolfinx coarse-vs-mid self-spread "
                    "of that metric already measured in Phase A). RULE STATED "
                    "BEFORE THE NUMBERS WERE READ. The coarse-vs-mid pair is "
                    "used because it is exactly the solve-mesh-to-score-mesh "
                    "step, and 1.5 is the same safety factor Phase A and the "
                    "close-out used. The band is the FORWARD's own "
                    "discretization move: a map that changes by more than that "
                    "changed because of map transfer, not physics.",
                "band_source": "solve3d/results/phase_a_shape_gate.json",
                "band_source_key": "self_spread_detail.circle.dolfinx_coarse_vs_mid",
                "safety_factor": SAFETY,
                "bands": {m: SAFETY * _phase_a_spread(m) for m in holdout_metrics},
                "bands_deferred": {
                    "J_rel": {
                        "rule": "1.5 * |J_mid - J_coarse| / |J_coarse| measured "
                                "on the UNIFORM arm, i.e. the forward's own "
                                "mesh move with no design involved",
                        "measured_from": "uniform_arm_own_mid_move",
                        "when": "measured in Task 4 BEFORE any solved map is "
                                "scored on the fine mesh",
                        "why_deferred": "Phase A never computed J, so there is "
                                        "no recorded spread to quote; the rule "
                                        "is frozen here and only the number "
                                        "waits"},
                    "also_reported": "the uniform arm's own move is reported "
                                     "beside every arm's move, per "
                                     "FROZEN_CONVENTIONS_2D section 8 Gate A"},
            },
            "smoothing_robustness": {
                "what": "blur the delivered map by a part-masked normalized "
                        "convolution at a radius strictly BELOW the filter "
                        "radius, and re-score",
                "perturbation_radius_m": 0.5e-3,
                "filter_radius_m": 1.0e-3,
                "tolerance_rel_J": 0.10,
                "citation": "FROZEN_CONVENTIONS_2D.md section 8 Gate B "
                            "(less than 10 percent change in J)",
                "also_reported": "the radius AT the filter length, as context "
                                 "and explicitly NOT part of the gate",
            },
            "solved_label_rule": {
                "all_of": [
                    "beats the uniform baseline on the PRIMARY objective "
                    "(asymmetric, 10x) by more than the in-grid noise band",
                    "passes the mesh hold-out on every banded metric",
                    "passes the smoothing-robustness tolerance",
                ],
                "citation": "FROZEN_CONVENTIONS_2D.md section 8 -- a map does "
                            "not get a SOLVED label unless Gate A and Gate B "
                            "both pass. On their own pass NO 2-D map earned it.",
                "null_result_is_a_valid_finding": True,
                "null_statement":
                    "If no solve arm beats uniform beyond the noise band, that "
                    "is the RESULT, not a failure of the port. The corrected "
                    "cylinder field is already nearly uniform (D1: interior "
                    "coefficient of variation 0.0119 and falling under "
                    "refinement), and the inversion heuristic's own answer on "
                    "this shape is +0.3 %, i.e. nothing. A solve that also "
                    "finds nothing would be evidence that there is nothing to "
                    "find on this shape, which is a physically meaningful "
                    "statement about the cylinder null and NOT evidence about "
                    "the method on shapes that do have structure.",
            },
        },
        "not_covered": [
            "one shape only (extruded circle); the library campaign is Phase E",
            "no permittivity channel (Phase D)",
            "no drive reconciliation (Phase D)",
            "no densification coupling in the forward",
            "Phase A's converged ~1.6 s circle t90 cross-family offset remains "
            "an S3 question and is not revisited",
        ],
    }
    gates.write_json(OUT.name, doc)
    return doc


def main() -> int:
    d = build()
    print(json.dumps({"budget": d["budget"]["gradient_evaluations_per_arm"],
                      "wall_min_per_solve_arm":
                          round(d["budget"]["wall_estimate_min_per_solve_arm"], 1),
                      "holdout_bands": d["acceptance"]["mesh_holdout"]["bands"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
