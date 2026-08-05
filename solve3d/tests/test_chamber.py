"""Adaptive chamber sizing: the 3-D analog of the 2.5-D chamber_for_bbox rule.

WHY THIS EXISTS. The frozen 60 mm chamber cannot melt the Tamper. Not because
the mesh is wrong -- every geometric gate on it passes -- but because the part
spans 44.4 mm of it and sits 7.8 mm from a fixed-temperature wall, so at the
frozen power density it plateaus at ~154 C against a 180 C melt onset
(solve3d/results/tamper_chamber_fit.json).

MARGIN PRE-REGISTRATION, from the measurement rather than from taste:

    part      powder to wall    plateau est    melts
    pyramid       18.38 mm        378.5 C       yes
    tamper         7.80 mm        154.5 C       no

So the margin must be at least the pyramid's 18.4 mm. The 2.5-D lane already
froze 20 mm per side (stl_compensation_tool/pipeline_logic.py GAP_M), which
clears that bar and makes the two lanes one convention rather than two. 20 mm
is therefore adopted, not invented, and a test reads THEIR constant so the
lanes cannot drift apart silently.

CHAMBER-TAGGED IDENTIFIERS are the 2.5-D lesson made structural: a run or mesh
built at one chamber size must never be comparable-by-accident with one built
at another, so the size is IN the identifier.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from solve3d import chamber as ch

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# The sizing rule
# --------------------------------------------------------------------------- #
def test_a_small_part_keeps_the_frozen_sixty_millimetre_chamber():
    """The floor is the frozen convention; adaptive sizing may only GROW."""
    assert ch.chamber_for_bbox(0.020, 0.020, 0.010) == pytest.approx(0.060)


def test_the_library_pyramid_still_gets_the_frozen_chamber():
    """23.2 mm + 2x20 mm = 63.2 mm... which is ABOVE the 60 mm floor, so the
    pyramid grows too. Stated explicitly because it means Phase E's recorded
    pyramid numbers are NOT reproduced by the adaptive default -- they need
    the frozen size passed explicitly."""
    got = ch.chamber_for_bbox(0.0232489, 0.0232489, 0.0232489)
    assert got == pytest.approx(0.064)
    assert got > 0.060


def test_the_tamper_grows_the_chamber_to_clear_the_margin():
    """44.4 mm + 2x20 mm = 84.4 mm, rounded up to a whole millimetre."""
    got = ch.chamber_for_bbox(0.0444, 0.0443349, 0.011)
    assert got == pytest.approx(0.085)


def test_the_governing_span_is_the_largest_axis():
    """A part long in z must not be sized by its x span."""
    assert ch.chamber_for_bbox(0.010, 0.010, 0.070) == pytest.approx(0.110)


def test_the_chamber_is_rounded_up_never_down():
    """Rounding down would eat into the pre-registered margin."""
    got = ch.chamber_for_bbox(0.0301, 0.010, 0.010)
    assert got >= 0.0301 + 2 * ch.GAP_M
    assert (got * 1e3) == pytest.approx(round(got * 1e3))


def test_the_resulting_margin_is_never_below_the_pre_registered_gap():
    for span in (0.005, 0.0232489, 0.0444, 0.070, 0.0999):
        L = ch.chamber_for_bbox(span, span, span)
        assert (L - span) / 2.0 >= ch.GAP_M - 1e-9, span


def test_a_part_that_cannot_be_given_the_margin_is_refused():
    """Refusing beats silently returning a chamber whose wall cuts the part."""
    with pytest.raises(ValueError):
        ch.chamber_for_bbox(0.010, 0.010, -0.001)


# --------------------------------------------------------------------------- #
# The margin is adopted from the 2.5-D lane, not invented here
# --------------------------------------------------------------------------- #
def test_the_margin_matches_the_two_and_a_half_d_lane_constant():
    src = (ROOT / "stl_compensation_tool" / "pipeline_logic.py").read_text()
    m = re.search(r"^GAP_M\s*=\s*([0-9.]+)", src, re.M)
    assert m, "the 2.5-D lane no longer defines GAP_M"
    assert float(m.group(1)) == pytest.approx(ch.GAP_M), (
        "solve3d and the 2.5-D lane disagree on the powder gap")


def test_the_margin_clears_the_measured_melting_margin():
    """18.38 mm is the pyramid's powder-to-wall gap, the smallest gap this
    project has MEASURED to melt. The pre-registered margin must clear it."""
    assert ch.GAP_M >= 0.01838


def test_the_floor_is_the_frozen_chamber():
    from solve3d import forward as fwd
    assert ch.FLOOR_M == pytest.approx(fwd.L_DOMAIN)


# --------------------------------------------------------------------------- #
# Chamber-tagged identifiers (the 2.5-D lesson)
# --------------------------------------------------------------------------- #
def test_the_tag_carries_the_chamber_size_in_millimetres():
    assert ch.chamber_tag(0.060) == "ch060"
    assert ch.chamber_tag(0.085) == "ch085"
    assert ch.chamber_tag(0.110) == "ch110"


def test_two_different_chambers_never_share_a_tag():
    assert ch.chamber_tag(0.060) != ch.chamber_tag(0.085)


def test_the_run_id_embeds_the_tag_so_sizes_cannot_mix_silently():
    rid = ch.run_id("tamper", 0.085)
    assert "ch085" in rid
    assert ch.run_id("tamper", 0.060) != rid


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_the_spec_records_why_this_size_and_that_it_is_adaptive():
    s = ch.chamber_spec(0.0444, 0.0443349, 0.011)
    assert s["L_m"] == pytest.approx(0.085)
    assert s["mode"] == "adaptive"
    assert s["gap_m"] == pytest.approx(ch.GAP_M)
    assert s["margin_actual_m"] == pytest.approx((0.085 - 0.0444) / 2.0)
    assert s["tag"] == "ch085"
    assert "quasi-static" in s["rationale"]
    assert s["floor_m"] == pytest.approx(0.060)


def test_the_frozen_chamber_stays_available_for_reproduction():
    s = ch.chamber_spec(0.0444, 0.0443349, 0.011, L_override=0.060)
    assert s["L_m"] == pytest.approx(0.060)
    assert s["mode"] == "frozen_override"
    assert s["tag"] == "ch060"
    # and it must SAY the margin is short, not hide it
    assert s["margin_actual_m"] < ch.GAP_M
    assert s["margin_below_preregistered"] is True


# --------------------------------------------------------------------------- #
# The frame must reach the SOLVER, not just the mesher
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_the_transient_case_honours_a_non_default_chamber():
    """The bug this pins: adjoint.TransientCase hardcoded fwd.L_DOMAIN for the
    convective top facet, and SteadyEqs was constructed without L. On an 85 mm
    mesh the electrode dofs are at z = +-42.5 mm, so looking at +-30 mm found
    NONE, the Dirichlet rows were never set, the RHS was identically zero and
    the solve died on `r.norm() / b.norm()`.

    It failed loudly, which is the only reason it is a bug report and not a
    silently wrong answer. Both ends of the frame are now threaded.
    """
    import numpy as np
    from solve3d import adjoint, forward as fwd, stl_mesh
    from solve3d.phase_e import run_tamper as rt

    tc, info = rt.build_case(lc_part=4.0e-3, max_time_s=1.0,
                             precomp_coeffs=None)
    assert info.L_chamber_m == pytest.approx(0.085)
    assert tc.L == pytest.approx(0.085)
    # the electrodes must actually exist in this frame
    lo, hi = fwd._electrode_dofs(tc.eqs.W, tc.msh, tc.L)
    assert lo.size > 0 and hi.size > 0
    # and the EQS solve must produce a non-trivial field
    st = tc.eqs.solve_state()
    assert float(np.abs(st.q).max()) > 0.0


def test_the_transient_case_default_frame_is_unchanged():
    """The knob is inert: no L argument still means the frozen chamber."""
    from solve3d import adjoint, forward as fwd
    import inspect
    sig = inspect.signature(adjoint.TransientCase.__init__)
    assert sig.parameters["L"].default == pytest.approx(fwd.L_DOMAIN)
