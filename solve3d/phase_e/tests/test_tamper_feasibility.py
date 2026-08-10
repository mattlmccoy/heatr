"""Gate the Tamper feasibility estimate's phi law against the real forward.

The estimate's whole credibility rests on its phi law being the SAME melt
fraction the solver scores with. If they drift, the below-floor numbers are
meaningless. So at drive a=1.0 (identity scaling) phi_at_drive MUST equal
solve3d.forward.phase_fraction elementwise.
"""
import numpy as np

from solve3d import forward as fwd
from solve3d.phase_e import tamper_feasibility as tf


def test_phi_law_matches_forward_phase_fraction_at_drive_one():
    p = fwd.ForwardParams()
    # the estimate hard-codes the melt band; it must equal the ForwardParams one
    assert p.t_pc_c == tf.TPC_C
    assert p.dt_pc_c == tf.DTPC_C
    rng = np.random.default_rng(0)
    T = 50.0 + 300.0 * rng.random(500)          # 50..350 C
    got = tf.phi_at_drive(T, 1.0)               # identity scaling
    want = fwd.phase_fraction(T, p)[0]
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_below_floor_monotone_decreasing_in_drive():
    T = np.array([80.0, 150.0, 175.0, 183.0, 200.0])
    w = np.ones_like(T)
    fr = [tf.below_floor_fraction(T, w, a) for a in (0.6, 1.0, 1.5, 2.0)]
    assert all(fr[i] >= fr[i + 1] for i in range(len(fr) - 1))


def test_peak_scales_and_is_identity_at_drive_one():
    assert tf.peak_at_drive(281.88, 1.0) == 281.88
    assert tf.peak_at_drive(281.88, 0.0) == tf.TAMB_C
    # linear in drive
    assert abs(tf.peak_at_drive(281.88, 2.0) - (50.0 + 2.0 * (281.88 - 50.0))) < 1e-9
