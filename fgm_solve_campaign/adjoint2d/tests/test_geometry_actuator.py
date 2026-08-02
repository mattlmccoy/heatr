"""Red-first tests for the anisotropy spectrum and the actuator recommendation.

THE CLAIM BEING MECHANIZED. `CONTINUOUS_ROTATION_REPORT.md` Section 4 measured
that the residual azimuthal anisotropy of the part-frame heating kernel RANKS
the outcomes of the rotation campaign exactly: the shapes whose averaged kernel
came out nearly annular (star, square, cross) are the shapes rotation rescued,
and the two that stayed strongly anisotropic (T_shape, L_shape) are the two it
did not. Section 1 of that report states the mechanism in one line: a radial
kernel melts a rounded blob, which reproduces a nearly radially symmetric part
and cannot reproduce a high aspect-ratio non-convex one.

That was a hand reading of five shapes. This module turns it into a classifier
that runs on ANY imported geometry, so the pipeline can tell a user which
actuator to spend money on before the solve.

HONESTY ABOUT THE CALIBRATION. The numbers quoted in that report (star 0.0205,
square 0.0216, cross 0.0389, T_shape 0.2193, L_shape 0.2840) came from a script
that is NOT in the repository; `kernel_anisotropy.json` survives but nothing
that writes it does, and four documented reconstructions of the stated
definition all failed to reproduce it (they also return an undefined value on
the L_shape, whose part does not cover the rotation centre, so the stored
definition cannot be the one the prose describes). The metric here is therefore
a re-derivation with its own definition, and the thresholds are re-calibrated
against the campaign's OUTCOMES on those same five shapes rather than against
its numbers. The outcomes are the ground truth; the numbers were the proxy.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import geometry_actuator as ga


def _fields(n: int = 61):
    jj, ii = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    r = np.hypot(jj - c, ii - c)
    th = np.arctan2(jj - c, ii - c)
    disc = r <= 0.45 * n
    return r, th, disc


# --- the metric ----------------------------------------------------------------

def test_a_perfectly_annular_kernel_scores_essentially_zero():
    r, th, disc = _fields()
    K = np.where(disc, np.exp(-(r / 12.0) ** 2), 0.0)
    a = ga.azimuthal_anisotropy(K, disc, ga.reference_angles())
    assert a < 0.02, f"an annular kernel should score near zero, got {a:.4f}"


def test_a_two_lobed_kernel_scores_high():
    r, th, disc = _fields()
    K = np.where(disc, np.exp(-(r / 12.0) ** 2) * (1.0 + 0.9 * np.cos(2 * th)), 0.0)
    a = ga.azimuthal_anisotropy(K, disc, ga.reference_angles())
    assert a > 0.2, f"a two-lobed kernel should score high, got {a:.4f}"


def test_the_metric_is_invariant_to_the_kernel_scale():
    """The recommendation must not change when the drive voltage does."""
    r, th, disc = _fields()
    K = np.where(disc, np.exp(-(r / 12.0) ** 2) * (1.0 + 0.6 * np.cos(4 * th)), 0.0)
    ang = ga.reference_angles()
    a1 = ga.azimuthal_anisotropy(K, disc, ang)
    a2 = ga.azimuthal_anisotropy(7.3 * K, disc, ang)
    assert a1 == pytest.approx(a2, rel=1e-12)


def test_a_four_fold_kernel_is_invariant_under_its_own_quarter_turns():
    r, th, disc = _fields()
    K = np.where(disc, np.exp(-(r / 12.0) ** 2) * (1.0 + 0.6 * np.cos(4 * th)), 0.0)
    a = ga.azimuthal_anisotropy(K, disc, np.array([0.0, 90.0, 180.0, 270.0]))
    assert a < 0.02, f"a four-fold kernel is quarter-turn invariant, got {a:.4f}"


# --- the classifier -------------------------------------------------------------

def test_the_calibration_table_is_the_five_campaign_shapes():
    names = {c["shape"] for c in ga.CALIBRATION}
    assert names == {"star", "square", "cross", "T_shape", "L_shape"}
    for c in ga.CALIBRATION:
        assert c["outcome"] in {"rotation_wins", "rotation_fails"}
        assert "source" in c and c["source"]


def test_the_classifier_reproduces_every_calibration_point():
    """The bands must agree with the campaign's own measured outcomes."""
    for c in ga.CALIBRATION:
        cls = ga.classify_residual(c["A_measured"])
        won = cls in {"MODE_SUFFICES", "MAP_PLUS_MODE"}
        assert won == (c["outcome"] == "rotation_wins"), (
            f"{c['shape']}: A {c['A_measured']:.3f} classified {cls} against "
            f"the campaign outcome {c['outcome']}")


def test_the_bands_are_ordered_and_labelled():
    assert ga.classify_residual(0.10) == "MODE_SUFFICES"
    assert ga.classify_residual(0.65) == "MAP_PLUS_MODE"
    assert ga.classify_residual(1.50) == "PHYSICAL_LIMIT"
    assert ga.A_MODE_SUFFICES < ga.A_PHYSICAL_LIMIT


def test_recommend_prefers_the_lowest_residual_mode():
    spec = {"static": 0.90, "continuous": 0.55, "index4": 0.31}
    rec = ga.recommend(spec, rotational_order=4)
    assert rec.mode == "index4"
    assert rec.actuator_class == "MODE_SUFFICES"
    assert rec.rotation_recommended is True


def test_recommend_says_sequential_dwell_when_no_mode_helps():
    spec = {"static": 1.20, "continuous": 1.10}
    rec = ga.recommend(spec, rotational_order=1)
    assert rec.actuator_class == "PHYSICAL_LIMIT"
    assert "sequential dwell" in rec.advice.lower()
    assert rec.rotation_recommended is False


def test_recommend_keeps_the_static_arm_when_rotation_does_not_reduce_anisotropy():
    spec = {"static": 0.42, "continuous": 0.41}
    rec = ga.recommend(spec, rotational_order=1)
    assert rec.mode == "static"
    assert rec.rotation_recommended is False


def test_recommend_reports_the_reduction_factor_it_used():
    rec = ga.recommend({"static": 0.80, "continuous": 0.40}, rotational_order=2)
    assert rec.reduction_factor == pytest.approx(2.0)
    assert rec.rotation_recommended is True


def test_recommend_refuses_a_spectrum_without_a_static_arm():
    with pytest.raises(ValueError, match="static"):
        ga.recommend({"continuous": 0.3}, rotational_order=4)


# --- the mode set ----------------------------------------------------------------

def test_mode_names_cover_static_continuous_and_the_divisor_indexings():
    assert ga.mode_names(rotational_order=6) == (
        "static", "continuous", "index2", "index3", "index6")
    assert ga.mode_names(rotational_order=1) == ("static", "continuous")
    assert ga.mode_names(rotational_order=5) == ("static", "continuous", "index5")


def test_mode_angles_are_the_indexed_positions():
    assert np.allclose(ga.mode_angles("index4"), [0.0, 90.0, 180.0, 270.0])
    assert np.allclose(ga.mode_angles("index3"), [0.0, 120.0, 240.0])
    assert np.allclose(ga.mode_angles("static"), [0.0])
    assert len(ga.mode_angles("continuous")) == 24


# --- CLASSIFIER VERSION 2: predict against the SOLVED arm, not the uniform arm ---
#
# WHY THERE IS A VERSION 2. GEOMETRY_GENERALIZATION_REPORT Section 6.2 measured
# the version-1 classifier's one honest miss: on a novel eight-tooth gear it
# called MODE_SUFFICES via continuous rotation, rotation did exactly what the
# anisotropy predicted (uniform-arm J 207.89 -> 138.59, a 33 percent cut at 8
# percent less absorbed power), and the SOLVED STATIC MAP still won (132.36 /
# IoU 0.8458 against 138.59 / 0.8273). Version 1 answers "does this actuator
# improve the heating"; the user's question is "does this actuator beat a solved
# map". Version 2 measures each mode's residual anisotropy with a MAP INJECTED
# rather than with the uniform map, so the recommendation is made against the
# competitor the pipeline actually has.

def test_the_zero_solve_stand_in_is_the_proportional_inverse_of_the_static_field():
    """The free variant: no gradient solve, one map built from the static kernel.

    It must be the production `control.proportional_inverse_map` and not a
    second implementation of the same rule, must stay inside the box, and must
    hold the nominal saturation 1 outside the part so it cannot silently change
    the geometry.
    """
    from adjoint2d import control as ctl
    r, th, disc = _fields()
    Q = np.where(disc, np.exp(-(r / 12.0) ** 2) * (1.0 + 0.6 * np.cos(2 * th)), 0.0)
    s = ga.prop_inverse_stand_in(Q, disc)
    assert s.shape == Q.shape
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert np.all(s[~disc] == 1.0), "outside the part the map is nominal 1"
    ref = ctl.proportional_inverse_map(
        Q, disc, magnitude=ga.PROP_INVERSE_MAGNITUDE,
        baseline=ga.PROP_INVERSE_BASELINE)
    assert np.array_equal(s, ref), "must delegate, not re-derive"


def test_the_stand_in_is_anti_correlated_with_the_heating_it_inverts():
    r, th, disc = _fields()
    Q = np.where(disc, np.exp(-(r / 12.0) ** 2) * (1.0 + 0.6 * np.cos(2 * th)), 0.0)
    s = ga.prop_inverse_stand_in(Q, disc)
    c = np.corrcoef(Q[disc], s[disc])[0, 1]
    assert c < -0.8, f"the inverse must oppose the heating, got r = {c:.3f}"


def test_the_version_2_calibration_table_is_the_seven_measured_points():
    """Five rotation-campaign shapes plus the two novel end-to-end geometries.

    Every one of the seven is a MEASURED comparison of the best rotating arm
    against a SOLVED STATIC arm, which is the comparison version 2 predicts.
    `CONTINUOUS_ROTATION_REPORT.md` Section 1's "best static arm" column is the
    joint campaign's winning angle plus its winning MAP, so those five already
    carry the right ground truth; the two novel geometries add the only
    out-of-sample points that exist.
    """
    names = {c["shape"] for c in ga.CALIBRATION_V2}
    assert names == {"square", "cross", "star", "keyhole", "T_shape",
                     "L_shape", "gear8"}
    for c in ga.CALIBRATION_V2:
        assert c["outcome"] in {"rotation_wins", "rotation_fails"}
        assert c["source"]
        assert "A2_static" in c and "A2_best" in c and "best_mode" in c


def test_version_2_reproduces_every_calibration_point_on_the_free_variant():
    for c in ga.CALIBRATION_V2:
        spec = {"static": c["A2_static"], c["best_mode"]: c["A2_best"]}
        rec = ga.recommend_v2(spec, rotational_order=c["rotational_order"])
        assert rec.rotation_recommended == (c["outcome"] == "rotation_wins"), (
            f"{c['shape']}: reduction {c['A2_static'] / c['A2_best']:.3f}, "
            f"class {rec.actuator_class}, outcome {c['outcome']}")


def test_version_2_calls_the_gear_MAP_SUFFICES_which_is_the_version_1_miss():
    """The decisive test. Version 1 said MODE_SUFFICES and the solved static map won."""
    g = [c for c in ga.CALIBRATION_V2 if c["shape"] == "gear8"][0]
    rec = ga.recommend_v2({"static": g["A2_static"], "continuous": g["A2_best"]},
                          rotational_order=8)
    assert rec.rotation_recommended is False
    assert rec.actuator_class == "MAP_SUFFICES"
    assert "map" in rec.advice.lower()


def test_version_2_keeps_the_keyhole_on_the_rotating_side():
    k = [c for c in ga.CALIBRATION_V2 if c["shape"] == "keyhole"][0]
    rec = ga.recommend_v2({"static": k["A2_static"], "continuous": k["A2_best"]},
                          rotational_order=1)
    assert rec.rotation_recommended is True
    assert rec.mode == "continuous"


def test_version_2_still_calls_the_two_stranded_shapes_a_physical_limit():
    for name in ("T_shape", "L_shape"):
        c = [k for k in ga.CALIBRATION_V2 if k["shape"] == name][0]
        rec = ga.recommend_v2({"static": c["A2_static"], c["best_mode"]: c["A2_best"]},
                              rotational_order=c["rotational_order"])
        assert rec.actuator_class == "PHYSICAL_LIMIT"
        assert rec.rotation_recommended is False
        assert "sequential dwell" in rec.advice.lower()


def test_the_version_2_bands_are_ordered_and_the_map_band_is_named():
    assert ga.classify_v2(0.20, rotation_recommended=False) == "MAP_SUFFICES"
    assert ga.classify_v2(0.20, rotation_recommended=True) == "MODE_SUFFICES"
    assert ga.classify_v2(0.65, rotation_recommended=True) == "MAP_PLUS_MODE"
    assert ga.classify_v2(1.50, rotation_recommended=True) == "PHYSICAL_LIMIT"
    assert ga.classify_v2(1.50, rotation_recommended=False) == "PHYSICAL_LIMIT"
    assert ga.A2_MODE_SUFFICES < ga.A2_PHYSICAL_LIMIT


def test_version_2_records_which_injected_map_it_was_measured_on():
    """A recommendation read off a different basis must not be silently mixed."""
    rec = ga.recommend_v2({"static": 0.80, "continuous": 0.40},
                          rotational_order=2, basis="solved")
    assert rec.basis == "solved"
    assert rec.as_json()["basis"] == "solved"
    assert ga.DEFAULT_V2_BASIS == "prop_inverse"


def test_version_2_refuses_a_spectrum_without_a_static_arm():
    with pytest.raises(ValueError, match="static"):
        ga.recommend_v2({"continuous": 0.3}, rotational_order=4)
