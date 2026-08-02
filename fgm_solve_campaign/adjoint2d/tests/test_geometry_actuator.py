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
