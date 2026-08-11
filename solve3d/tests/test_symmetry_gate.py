"""Tests for the symmetry-consistency primitive (solve3d/symmetry_gate.py).

The PRIMITIVE only: given a solved map, node coords, weights, and an explicit
symmetry group (list of coord->coord ops), what fraction of the map's
(vol-weighted) variance survives projection onto the group-symmetric subspace?
A budget-limited solve that fit discretization-frame noise scores LOW here even
when the peak/density gates pass (the Phase C cylinder was 0.25).

GROUP DETECTION (part INTERSECT field INTERSECT objective INTERSECT
convection-BC symmetry) and the is_sendable WIRING are deliberately NOT here --
they conform to the campaign lane's SYMMETRY_GATE_REPORT.md when it lands. This
file pins only the spec-independent linear-algebra core.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import symmetry_gate as sg

IDENT = lambda p: p
XMIRROR = lambda p: p * np.array([-1.0, 1.0, 1.0])

# 4 nodes on the x-axis, symmetric about 0: -2 <-> 2, -1 <-> 1 under x-mirror.
COORDS = np.array([[-2.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
W = np.ones(4)


def _frac(s, group, weights=W):
    return sg.symmetric_variance_fraction(
        np.asarray(s, float), COORDS, weights, group)["fraction"]


def test_fully_symmetric_map_scores_one():
    # s(-x) == s(x) -> entirely in the symmetric subspace
    assert _frac([10.0, 4.0, 4.0, 10.0], [IDENT, XMIRROR]) == pytest.approx(1.0)


def test_pure_antisymmetric_map_scores_zero():
    # mean-zero, s(-x) == -s(x) -> the symmetric component is identically 0
    assert _frac([3.0, 1.0, -1.0, -3.0], [IDENT, XMIRROR]) == pytest.approx(0.0, abs=1e-12)


def test_known_symmetric_plus_antisymmetric_split():
    # s = sym[10,4,4,10] + anti[3,1,-1,-3]; sym var = 9, total var = 14 -> 9/14
    assert _frac([13.0, 5.0, 3.0, 7.0], [IDENT, XMIRROR]) == pytest.approx(9.0 / 14.0)


def test_identity_only_group_is_trivially_symmetric():
    # the trivial group projects to identity -> every map is "symmetric"
    assert _frac([13.0, 5.0, 3.0, 7.0], [IDENT]) == pytest.approx(1.0)


def test_weights_are_respected():
    # 6 nodes x=[-3,-2,-1,1,2,3]; symmetric part [9,4,1,1,4,9] carries variance on
    # BOTH pairs, the antisymmetry [5,0,0,0,0,-5] lives ONLY on the |x|=3 pair.
    # s = sym + anti = [14,4,1,1,4,4]. Down-weighting the |x|=3 nodes removes the
    # only antisymmetric content while the inner pairs still carry symmetric
    # variance -> the fraction rises toward 1. (Uniform weights ~0.566.)
    coords6 = np.array([[-3.0, 0, 0], [-2.0, 0, 0], [-1.0, 0, 0],
                        [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    s = np.array([14.0, 4.0, 1.0, 1.0, 4.0, 4.0])
    grp = [IDENT, XMIRROR]
    lo = sg.symmetric_variance_fraction(s, coords6, np.ones(6), grp)["fraction"]
    w_down = np.array([1e-3, 1.0, 1.0, 1.0, 1.0, 1e-3])
    hi = sg.symmetric_variance_fraction(s, coords6, w_down, grp)["fraction"]
    assert lo == pytest.approx(0.566, abs=1e-2)
    assert hi > lo
    assert hi == pytest.approx(1.0, abs=1e-2)


def test_reports_max_match_distance_zero_on_exact_symmetric_mesh():
    out = sg.symmetric_variance_fraction(
        np.array([10.0, 4.0, 4.0, 10.0]), COORDS, W, [IDENT, XMIRROR])
    assert out["max_match_dist"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= out["fraction"] <= 1.0


def test_verdict_pass_fail_at_threshold():
    # convenience: PASS iff fraction >= threshold (default 0.8)
    assert sg.symmetry_verdict(0.912) == "PASS"
    assert sg.symmetry_verdict(0.354) == "FAIL"
    assert sg.symmetry_verdict(0.80) == "PASS"
