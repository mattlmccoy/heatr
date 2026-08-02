"""STUDIO ALPHA plan-card pure logic (no solver imports, unit testable).

The Import & Plan flow runs geometry intake, drive calibration, symmetry
analysis and the actuator anisotropy spectrum, then renders a PLAN CARD.
This module holds the pure assembly and classification logic so the card the
graphical user interface renders and the card the command-line drivers embed
in the print package manifest can never drift apart.

Difficulty bands reuse the actuator classifier's calibrated thresholds
(fgm_solve_campaign/adjoint2d/geometry_actuator.py: A_MODE_SUFFICES = 0.50,
A_PHYSICAL_LIMIT = 0.80) applied to the BEST mode's residual anisotropy.
Those thresholds were calibrated on five measured rotation outcomes and the
classifier's out-of-sample record is 1 of 2, so everything here is framed as
advisory (print_package.ADVISORY_DISCLAIMER is the single source of that
text).
"""
from __future__ import annotations

from typing import Any

from print_package import ADVISORY_DISCLAIMER, GRID_QUALIFIER_TEMPLATE

# Calibrated classifier thresholds, mirrored by value and pinned by test
# against geometry_actuator so a drift is caught loudly.
A_MODE_SUFFICES = 0.50
A_PHYSICAL_LIMIT = 0.80

SMOKE_BUDGET_THRESHOLD = 20.0   # below this, scripts/solve_fgm.py labels a
                                # run smoke-test class (solve_fgm.py:496-499)


def difficulty_class(best_mode_residual: float) -> dict[str, Any]:
    """Predicted difficulty from the best mode's residual anisotropy."""
    a = float(best_mode_residual)
    basis = ("Bands are the actuator classifier's thresholds calibrated on "
             "the campaign's five measured rotation outcomes "
             "(A_MODE_SUFFICES 0.50, A_PHYSICAL_LIMIT 0.80, "
             "GEOMETRY_GENERALIZATION_REPORT.md section 5.2); the "
             "out-of-sample Spearman correlation between residual "
             "anisotropy and achieved fidelity is -0.79 (n = 18).")
    if a < A_MODE_SUFFICES:
        klass, meaning = "EXPECTED_TRACTABLE", (
            "the best actuator mode leaves little heating anisotropy; the "
            "campaign's winners sit in this band")
    elif a < A_PHYSICAL_LIMIT:
        klass, meaning = "HARD_ACTUATOR_LOAD_BEARING", (
            "real anisotropy remains after the best mode; expect the solved "
            "dopant map to be load bearing rather than a finish")
    else:
        klass, meaning = "AT_THE_PHYSICAL_LIMIT", (
            "the campaign's failures sit in this band; rotation and grading "
            "may not rescue this geometry")
    return {"klass": klass, "best_mode_residual": a, "meaning": meaning,
            "basis": basis}


def budget_label(budget_forward_equivalents: float) -> dict[str, Any]:
    """Smoke-class labelling, the scripts/solve_fgm.py convention."""
    b = float(budget_forward_equivalents)
    smoke = b < SMOKE_BUDGET_THRESHOLD
    label = (f"budget {b:g} forward-equivalents"
             + (" (REDUCED BUDGET: smoke-test class, not a quality solve)"
                if smoke else " (campaign standard class)"))
    return {"budget_forward_equivalents": b, "smoke": smoke, "label": label}


def plan_card(*, intake_info: dict, symmetry: dict, anisotropy: dict,
              recommendation: dict, grid: int) -> dict[str, Any]:
    """Assemble the plan card from the analysis stage outputs.

    All inputs are the JSON-safe dicts the intake pipeline already emits
    (`Intake.info`, `SymmetryReport.as_json()`, the spectrum residual dict,
    `Recommendation.as_json()`), so this is a pure re-arrangement with the
    advisory framing and difficulty class added.
    """
    rec = dict(recommendation)
    rec["advisory_disclaimer"] = ADVISORY_DISCLAIMER
    best_mode = rec.get("mode")
    best_residual = float(rec.get("residual_anisotropy",
                                  anisotropy.get(str(best_mode), 1.0)))
    mirrors = symmetry.get("mirror_axes_deg", [])
    return {
        "part": {
            "name": intake_info.get("name", "imported"),
            "source_route": intake_info.get("source", "unknown"),
            "n_vertices": intake_info.get("n_vertices"),
            "n_part_cells": intake_info.get("n_part_cells"),
            "area_mm2": (float(intake_info["area_chi_m2"]) * 1e6
                         if "area_chi_m2" in intake_info else None),
        },
        "symmetry": {
            "rotational_order": symmetry.get("rotational_order"),
            "point_group": symmetry.get("point_group"),
            "n_mirror_axes": len(mirrors),
            "mirror_axes_deg": mirrors,
            "continuous": symmetry.get("continuous", False),
        },
        "anisotropy_per_mode": {k: float(v) for k, v in anisotropy.items()},
        "recommendation": rec,
        "difficulty": difficulty_class(best_residual),
        "power": {
            "rf_mode": "constant",
            "voltage_v": float(intake_info.get("voltage_v", float("nan"))),
            "voltage_is_calibrated": bool(
                intake_info.get("voltage_is_calibrated", False)),
            "calibration": ("automatic drive calibration to 500 W per metre "
                            "absorbed on the uniform arm "
                            "(geometry_calibrate)"),
        },
        "grid": int(grid),
        "grid_qualifier": GRID_QUALIFIER_TEMPLATE.format(grid=int(grid)),
    }


def user_facing_refusal(raw_message: str) -> dict[str, Any]:
    """Turn an IntakeError message into a clear user-facing refusal."""
    low = str(raw_message).lower()
    if "self-intersect" in low or "crosses edge" in low:
        reason = ("The outline self-intersects (two edges cross). The fill "
                  "rule would silently reinterpret the crossing as a notch "
                  "or hole and the pipeline would plan a part you did not "
                  "draw.")
        todo = ("Fix the outline so no two edges cross, then re-import. "
                "The raw message below names the offending edge pair.")
    elif "hole" in low or "interior loop" in low:
        reason = ("The geometry contains a hole (an interior loop). The "
                  "production fill combines parts by a maximum rule and "
                  "cannot express a hole; an annulus would come back a "
                  "filled disc.")
        todo = ("Remove the hole or split the part into hole-free pieces "
                "and import them one at a time.")
    elif "disjoint components" in low or "multi-part" in low:
        reason = ("The import contains multiple disjoint parts. The "
                  "rotation and dwell conventions are established for a "
                  "single part only.")
        todo = "Import the parts one at a time."
    elif "chamber" in low:
        reason = "The imported geometry does not fit inside the chamber."
        todo = "Scale the part down or enlarge the chamber setting."
    else:
        reason = str(raw_message)
        todo = "Fix the reported problem in the source geometry and retry."
    return {"refused": True, "reason": reason, "what_to_do": todo,
            "raw_message": str(raw_message)}
