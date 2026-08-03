"""S2 Task 2: gauge plumbing."""
from __future__ import annotations

import numpy as np
import pytest

import heatr3d
from heatr3d_s2 import gauge


def test_flag_off_returns_the_shipped_params_object_unchanged():
    """The flag-off path must be bit-identical BY CONSTRUCTION, not by
    numerical luck: the current arm returns the very same Params instance."""
    p = heatr3d.Params()
    assert gauge.gauge_params(p, 64, "cell_centred_current") is p


def test_face_gauge_rescales_the_drive_by_exactly_n_minus_1_over_n():
    p = heatr3d.Params()
    for n in (48, 64, 96):
        q = gauge.gauge_params(p, n, "face_gauge")
        assert q.v_lo == pytest.approx(p.v_lo * (n - 1) / n)
        assert q.v_hi == p.v_hi


def test_face_gauge_imposes_a_grid_invariant_field_and_the_current_one_does_not():
    """The whole point, stated as a test: the shipped gauge's imposed field
    drifts with n; the face gauge's does not."""
    p = heatr3d.Params()
    fields = {}
    for arm in gauge.ARMS:
        fields[arm] = [gauge.imposed_field_v_per_m(p, heatr3d.Grid(n=n), arm)
                       for n in (48, 64, 96)]
    face = fields["face_gauge"]
    assert max(face) - min(face) < 1e-9 * face[0]
    assert all(abs(f - p.v_lo / 0.060) < 1e-9 for f in face)
    cur = fields["cell_centred_current"]
    assert (max(cur) - min(cur)) / min(cur) > 0.01     # a real, measurable drift


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown gauge arm"):
        gauge.gauge_params(heatr3d.Params(), 64, "whatever")


def test_raw_power_is_quadratic_in_the_drive():
    """Sanity on the observable: the EQS solve is linear in the drive, so the
    raw absorbed power must scale as v^2. If this fails, the observable cannot
    be used to compare gauges."""
    import dataclasses
    grid = heatr3d.Grid(n=24)
    part = heatr3d.make_geometry(grid, "cylinder", diam=0.020)
    p1 = heatr3d.Params()
    p2 = dataclasses.replace(p1, v_lo=2.0 * p1.v_lo)
    out = []
    for p in (p1, p2):
        gam = heatr3d.build_gamma(part, p)
        V = heatr3d.solve_eqs_3d(gam, grid, p)
        out.append(gauge.raw_absorbed_power_w(V, gam, grid, part))
    assert out[1] / out[0] == pytest.approx(4.0, rel=1e-6)
