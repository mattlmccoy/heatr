"""Adaptive chamber sizing for solve3d: the 3-D analog of the 2.5-D rule.

DECISION AND AUTHORITY. Matt, 2026-08-05: grow the chamber. The frozen 60 mm
box cannot melt the Tamper, and the reason is geometric rather than numerical
(solve3d/results/tamper_chamber_fit.json).

HIS PHYSICS RATIONALE, recorded because it is what licenses the change:

  * At 27.12 MHz the free-space wavelength is about 11 m, against parts of
    tens of millimetres. The rig is deep in the QUASI-STATIC regime, so
    growing the chamber and the electrode gap does not change the material's
    RF characteristics. In this model that is true BY CONSTRUCTION as well as
    physically: the forward solves an electro-quasi-static problem
    (Laplace/current-continuity for a complex potential), which carries no
    wavelength at all. Growing the box cannot introduce a propagation effect
    the formulation does not contain.
  * The drive is POWER-DENSITY normalized to the part volume
    (forward.qrf_dg0 renormalizes Q so that integral(Q dV) =
    power_density_w_per_m3 * V_part), so a bigger box does not dilute the
    watts the part receives. Growing the chamber changes where the heat GOES
    (further to the cold wall), which is exactly the intended effect.

WHAT THE RATIONALE DOES NOT COVER, stated because it bounds the claim: the
fringing FIELD GEOMETRY does change with the gap even in quasi-statics. That
is why solve3d/make_chamber_field_check.py measures the in-part field pattern
across a chamber sweep instead of assuming it is invariant.

HARDWARE IS NOT BOUND BY THIS. Growing the simulated chamber is a modelling
decision. The physical implication -- electrode gap, matching network, and
whether the rig can drive a larger gap at all -- is P-gate territory and is
NOT decided here.

MARGIN, pre-registered from the measurement rather than from taste:

    part      powder to wall    plateau estimate    melts
    pyramid       18.38 mm           378.5 C         yes
    tamper         7.80 mm           154.5 C         no

The margin must clear the pyramid's 18.38 mm, the smallest gap this project
has measured to melt. The 2.5-D lane already froze 20 mm per side
(stl_compensation_tool/pipeline_logic.py, GAP_M), which clears it and makes
the two lanes ONE convention. 20 mm is therefore adopted, not invented, and
solve3d/tests/test_chamber.py reads their constant so the two cannot drift.

CHAMBER-TAGGED IDENTIFIERS are the 2.5-D lesson made structural. A mesh or run
built at one chamber size must never be silently comparable with one built at
another, so the size travels IN the identifier: `ch085`, not a footnote.

REPRODUCTION. The frozen 60 mm chamber stays available by explicit override,
and a spec built that way says so AND flags that its margin is below the
pre-registered gap. Every pre-2026-08-05 solve3d artifact was built at 60 mm.
"""
from __future__ import annotations

import math

# The powder margin on every side. Adopted from the 2.5-D lane's GAP_M; see
# the module docstring for the measurement that sets the lower bound.
GAP_M = 0.020

# Never smaller than the frozen chamber: adaptive sizing may only GROW.
FLOOR_M = 0.060

# The smallest powder-to-wall gap this project has measured to melt (the
# library pyramid). GAP_M must clear it; a test asserts that.
MEASURED_MELTING_MARGIN_M = 0.01838


def chamber_for_bbox(span_x_m: float, span_y_m: float,
                     span_z_m: float | None = None) -> float:
    """Square chamber side for a part bounding box: span + 2 x GAP_M per axis,
    never below FLOOR_M, rounded UP to a whole millimetre.

    Rounding is up, never to nearest: rounding down would eat into the
    pre-registered margin, which is the one quantity this function exists to
    guarantee.
    """
    spans = [span_x_m, span_y_m] + ([] if span_z_m is None else [span_z_m])
    for s in spans:
        if not (float(s) > 0.0) or not math.isfinite(float(s)):
            raise ValueError(
                f"chamber_for_bbox: span {s!r} is not a positive finite "
                "length; refusing rather than returning a chamber whose wall "
                "would cut the part")
    need = max(float(s) + 2.0 * GAP_M for s in spans)
    L = max(FLOOR_M, need)
    return float(math.ceil(L * 1e3 - 1e-9) / 1e3)


def chamber_tag(L_m: float) -> str:
    """`ch085` for an 85 mm chamber: the size, in the identifier."""
    return f"ch{int(round(float(L_m) * 1e3)):03d}"


def run_id(name: str, L_m: float) -> str:
    """A run/mesh identifier that cannot be confused across chamber sizes."""
    return f"{name}_{chamber_tag(L_m)}"


def chamber_spec(span_x_m: float, span_y_m: float,
                 span_z_m: float | None = None,
                 L_override: float | None = None) -> dict:
    """The chamber a run used, and why -- the record that travels with it.

    `L_override` reproduces a fixed chamber (60 mm for every pre-2026-08-05
    artifact). An override that leaves less than the pre-registered margin is
    FLAGGED rather than silently accepted, because a short margin is exactly
    what made the Tamper unmeltable.
    """
    spans = [float(span_x_m), float(span_y_m)]
    if span_z_m is not None:
        spans.append(float(span_z_m))
    adaptive = chamber_for_bbox(*spans)
    L = adaptive if L_override is None else float(L_override)
    governing = max(spans)
    margin = (L - governing) / 2.0
    return {
        "L_m": float(L),
        "mode": "adaptive" if L_override is None else "frozen_override",
        "tag": chamber_tag(L),
        "gap_m": GAP_M,
        "floor_m": FLOOR_M,
        "adaptive_would_be_m": float(adaptive),
        "governing_span_m": float(governing),
        "spans_m": spans,
        "margin_actual_m": float(margin),
        "margin_below_preregistered": bool(margin < GAP_M - 1e-9),
        "measured_melting_margin_m": MEASURED_MELTING_MARGIN_M,
        "gap_source": ("stl_compensation_tool/pipeline_logic.py GAP_M; "
                       "adopted, not invented"),
        "rationale": (
            "27.12 MHz gives an ~11 m wavelength against parts of tens of mm, "
            "so the rig is deep in the quasi-static regime and growing the "
            "chamber does not change the material's RF characteristics; in "
            "this model that also holds by construction, because the forward "
            "is electro-quasi-static and carries no wavelength. The drive is "
            "power-density normalized to part volume, so a larger box does "
            "not dilute the watts the part absorbs. Fringing field GEOMETRY "
            "does still change with the gap, which is why the chamber sweep "
            "measures the in-part pattern rather than assuming invariance."),
        "hardware_note": (
            "modelling decision only; electrode gap and matching network are "
            "P-gate territory and are not bound by this"),
        "authority": "Matt, 2026-08-05",
    }


def preregistration() -> dict:
    """What was fixed BEFORE the grown-chamber runs were made."""
    return {
        "decision": "grow the chamber adaptively with the part bounding box",
        "rule": "L = ceil_mm(max(FLOOR_M, max_axis_span + 2 * GAP_M))",
        "gap_m": GAP_M,
        "floor_m": FLOOR_M,
        "gap_evidence": {
            "pyramid_powder_to_wall_m": 0.01838,
            "pyramid_plateau_estimate_c": 378.5,
            "pyramid_melts": True,
            "tamper_powder_to_wall_m": 0.00780,
            "tamper_plateau_estimate_c": 154.5,
            "tamper_melts": False,
            "melt_onset_c": 180.0,
            "source": "solve3d/results/tamper_chamber_fit.json"},
        "identifiers": "chamber size is embedded in every run/mesh id",
        "reproduction": ("pass L_override=0.060 for any pre-2026-08-05 "
                         "artifact; the adaptive default does NOT reproduce "
                         "them, including the Phase E pyramid and cube"),
        "insurance_check": ("solve3d/make_chamber_field_check.py measures the "
                            "in-part field pattern across a chamber sweep "
                            "before grown-chamber solves are trusted"),
        "authority": "Matt, 2026-08-05",
    }
