"""Phase E opener: PRE-REGISTER the first library cases (pyramid + cube).

    ./.venv312/bin/python -m solve3d.phase_e.prereg

Committed BEFORE any solve, the same discipline Phase C used.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "phase_e_preregistration.json"

CUBE_SIDE_MM = 16.119919540164695
PYR_BASE_MM = 23.248947030192525          # square base side == height, apex up
TARGET_VOL_MM3 = 4188.790204786391        # equal-volume to the 20 mm sphere


def build() -> dict:
    doc = {
        "what": "Phase E opener pre-registration: the SOLVED pyramid, plus a "
                "cube arm. First cases of the library campaign, run with the "
                "full Phase C discipline.",
        "instrument": {
            "engine": "solve3d",
            "conforming_mesh": True,
            "why_not_heatr3d":
                "S2 measured that heatr3d masks at CELL CENTRES, so an "
                "AXIS-ALIGNED boundary snaps coherently and the voxelized part "
                "is a different SIZE at each grid -- effective half-width error "
                "up to +3.12 % on a 20 mm part, and the square's apparent "
                "non-convergence was that artifact rather than the solver "
                "(heatr3d_s2/S2_GATE_REPORT.md, close-out C2). The CUBE is the "
                "maximally axis-aligned shape in the library, so the voxel "
                "engine is the WORST available instrument for it. solve3d "
                "meshes the boundary conformally, which removes the staircase "
                "and the commensurability question entirely, and it is also the "
                "only engine with the Phase B gated gradient the solve needs.",
            "cost": "solve3d also costs more per forward; that is accepted "
                    "rather than traded against correctness here."},
        "cases": {
            "pyramid": {
                "source_stl": "shape_library_3d/stl/pyramid.stl",
                "role": "stressor -- apex singularity PLUS sharp edges and flat "
                        "facets (shape_library_3d/meta/pyramid.json)",
                "construction": "built analytically in gmsh OCC as a square "
                                "pyramid, apex up, base side = height = "
                                f"{PYR_BASE_MM} mm, NOT imported from the STL "
                                "mesh; the construction is then VERIFIED "
                                "against the STL's recorded volume",
                "base_side_mm": PYR_BASE_MM, "height_mm": PYR_BASE_MM,
                "target_volume_mm3": TARGET_VOL_MM3,
                "geometry_verification": {
                    "against": "shape_library_3d/meta/pyramid.json "
                               "actual_volume_mm3",
                    "tolerance_rel": 1e-9,
                    "why": "an analytic OCC solid is exact and cheap to mesh, "
                           "while an STL import carries the tessellation into "
                           "the solve; the volume check is what proves the "
                           "analytic solid IS the library shape"}},
            "cube": {
                "source_stl": "shape_library_3d/stl/cube.stl",
                "role": "control -- flat-faced orthogonal baseline",
                "construction": "gmsh OCC box, side "
                                f"{CUBE_SIDE_MM} mm, centred",
                "side_mm": CUBE_SIDE_MM,
                "target_volume_mm3": TARGET_VOL_MM3,
                "geometry_verification": {
                    "against": "shape_library_3d/meta/cube.json "
                               "actual_volume_mm3",
                    "tolerance_rel": 1e-9,
                    "why": "same"}}},
        "conventions": {
            "note": "frozen conventions carried verbatim from Phase C; none is "
                    "re-derived here",
            "filter_radius_m": 1.0e-3,
            "filter": "explicit normalized-convolution matrix, volume-weighted, "
                      "3-sigma truncation",
            "chi": "sub-cell VOLUME fill; MUST pass the shared cross-lane fill "
                   "contract on these shapes before any solve "
                   "(fgm_solve_campaign/adjoint2d/tests/fill_contract.py)",
            "objective_primary": "asymmetric, W_out/W_in = 10, phi_floor 0.85",
            "phi_floor": 0.85, "w_out_over_w_in": 10.0,
            "symmetric_control_scored": True,
            "budget_forward_equivalents_per_arm": 40,
            "scale_first_step": True,
            "scale_first_step_why":
                "now the CONVENTION, not a deviation: Phase C measured that the "
                "uniform cold start sits on the upper box rail and L-BFGS-B's "
                "O(1/|g|) first step collapses there, stalling the solve after "
                "2 of 12 evaluations. Rescaling J by the constant 1/|g0| is a "
                "pure reparameterization that cannot move the minimizer.",
            "envelope_stop": True,
            "start": "single full-depth cold start, uniform saturation 1.0",
            "actuator": "conductivity only; eps channel OFF (Phase D)"},
        "arms": [
            {"priority": 1, "name": "uniform_baseline", "shapes": ["pyramid", "cube"],
             "cost_forward_equivalents": 1},
            {"priority": 2, "name": "solve_filter_only",
             "shapes": ["pyramid", "cube"], "cost_forward_equivalents": 40,
             "beta": 0.0, "role": "the PRIMARY solve arm"},
            {"priority": 3, "name": "heuristic_grading_law", "shapes": ["pyramid"],
             "cost_forward_equivalents": 1,
             "provenance": {
                 "commit": "6c2aab9 (demo_pyramid_fgm, strong variant)",
                 "law": "sat = clip(0.95*(0.35+0.65*d_hat)*(1-0.45*z_hat), 0.20, 1.00); "
                        "the STRONG variant hardens the constants",
                 "status": "HAND-CONSTRUCTED and physically motivated, NOT "
                           "solved or optimized -- its own module says so",
                 "transfer": "evaluate the closed-form law directly at the FEM "
                             "cell centroids (it is an analytic function of "
                             "depth and height), which avoids the voxel-to-mesh "
                             "resampling loss that dropped the Phase C "
                             "inversion arm",
                 "transfer_drop_condition_rel": 0.02,
                 "never_approximated":
                     "if the transferred map's total in-part dopant moves by "
                     "more than 2 % against the law evaluated on a fine "
                     "reference grid, the arm is DROPPED with the reason "
                     "recorded, never replaced by an approximation presented "
                     "as the published law -- the Phase C inversion-arm "
                     "precedent"}},
            {"priority": 4, "name": "solve_projection_beta_continuation",
             "shapes": ["pyramid"], "cost_forward_equivalents": 40,
             "optional": True,
             "role": "robustness arm; ONE shape only, and only if budget allows "
                     "after everything above completes"}],
        "acceptance": {
            "bands_are_per_shape": True,
            "band_rule": "1.5 x the measured self-spread of that shape, same "
                         "rule and safety factor as Phase A, its close-out and "
                         "Phase C. Bands are MEASURED in this campaign, not "
                         "inherited from the circle/square anchors, because "
                         "these are different geometries.",
            "mesh_holdout": {
                "what": "solve on the coarse mesh, re-run the FORWARD (not the "
                        "solve) on the refined mesh and score there",
                "band_source": "measured_in_campaign",
                "band_rule": "1.5 x the UNIFORM arm's own coarse-to-fine move "
                             "for each metric, measured before any solved map "
                             "is scored on the fine mesh"},
            "smoothing": {
                "perturbation_radius_m": 0.5e-3, "tolerance_rel_J": 0.10,
                "citation": "FROZEN_CONVENTIONS_2D.md section 8 Gate B"},
            "solved_label_rule": {
                "all_of": ["beats uniform on the PRIMARY objective beyond the "
                           "in-grid noise band",
                           "passes the mesh hold-out on every banded metric",
                           "passes smoothing robustness"],
                "null_result_is_a_valid_finding": True}},
        "honesty_constraint_from_s2": {
            "absolute_fidelity_is_ungated": True,
            "statement":
                "BOTH cases are CORNERED shapes and the pyramid additionally has "
                "an APEX. S2 measured that heatr3d's melt-region geometry does "
                "not settle at a reentrant corner even with geometry held fixed "
                "(lshape front SSD 0.33342 mm on the commensurate pair, above "
                "its 0.25 mm ceiling, with the corner power peak growing "
                "2.8125 -> 3.3788 -> 3.7135). solve3d's conforming mesh removes "
                "the staircase but does NOT remove a physical field "
                "singularity. Therefore no ABSOLUTE shape-fidelity number on "
                "these shapes is gated by anything, and none will be quoted as "
                "if it were.",
            "what_is_meaningful":
                "the solved-vs-uniform comparison at MATCHED mesh and matched "
                "read state (a same-instrument comparison, where the "
                "singularity affects both arms identically and cancels), and "
                "the mesh hold-out gate RESULT itself",
            "apex_holdout_may_fail_and_that_is_informative": True,
            "apex_note":
                "if the hold-out fails at the apex, that is a finding about "
                "solving on singular geometry -- it says the delivered map is "
                "tuned to a mesh-dependent feature -- and is NOT a campaign "
                "bug to be worked around.",
            "escalation": "unchanged from S2: the apex/reentrant singularity "
                          "question belongs to S3's COMSOL anchor. Phase E "
                          "does not attempt to settle it.",
            "sigma_T": "diagnostic only, never verdict-carrying"},
        "deliverables": [
            "solve3d/phase_e/results/*.json",
            "solve3d/phase_e/PHASE_E_OPENER_REPORT.md",
            "per-shape comparison figures: solved vs uniform vs heuristic, "
            "difference panels, and a layer stack of the SOLVED map, adapted "
            "from demo_pyramid_fgm/render_figs.py; every PNG viewed before "
            "delivery (the F7 sliver lesson)"],
        "partial_completion_policy": {
            "unrun": "any arm not reached is recorded NOT_RUN with its reason; "
                     "never back-filled, never extrapolated",
            "order": "arms run in the pre-registered priority order; prefer "
                     "finishing fewer arms FULLY (with their acceptance gates) "
                     "over starting all of them"},
        "out_of_scope": [
            "the rest of the library (Phase E proper)",
            "eps_r channel and drive reconciliation (Phase D)",
            "the apex/reentrant singularity adjudication (S3)",
            "any dissertation edit; dissertation_materials is READ-ONLY"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    return doc


def main() -> int:
    d = build()
    print(json.dumps({"cases": list(d["cases"]),
                      "instrument": d["instrument"]["engine"],
                      "arms": [a["name"] for a in d["arms"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
