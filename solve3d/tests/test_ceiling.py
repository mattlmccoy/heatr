"""Stage A Task 2: the thermal-ceiling observable (KS peak + TRUE max).

The pure-numpy tests run anywhere; test_ks_tracks_on_a_real_densify_march needs
the spike env (it marches the forward).
"""
from __future__ import annotations

import numpy as np

from solve3d import ceiling


# --------------------------------------------------------------------------- #
# true max, and KS as a lower bound tracking it
# --------------------------------------------------------------------------- #
def test_true_max_is_the_hard_max_over_the_mask():
    T = np.array([100.0, 250.0, 260.0, 10.0])
    mask = np.array([True, True, True, False])       # excludes the cold 10.0
    pk = ceiling.peak_temp(T, mask=mask)
    assert pk["true_max_c"] == 260.0
    assert pk["n_selected"] == 3


def test_ks_is_a_lower_bound_and_tracks_true_max_within_band():
    rng = np.random.default_rng(1)
    # a realistic in-part peak field: a plateau near 235 C with a hot core
    T = np.concatenate([rng.normal(235.0, 4.0, 4900),
                        rng.normal(248.0, 1.0, 100)])
    pk = ceiling.peak_temp(T)
    assert pk["ks_aggregate_c"] <= pk["true_max_c"]      # lower bound
    assert pk["surrogate_is_lower_bound"]
    assert pk["gap_rel"] <= 0.05                         # pre-registered band


def test_higher_sharpness_shrinks_the_gap():
    rng = np.random.default_rng(2)
    T = rng.normal(240.0, 6.0, 3000)
    lo = ceiling.peak_temp(T, rho=0.5)["gap_c"]
    hi = ceiling.peak_temp(T, rho=4.0)["gap_c"]
    assert hi < lo


# --------------------------------------------------------------------------- #
# false-green guard: the ceiling verdict is on the TRUE max, never the surrogate
# --------------------------------------------------------------------------- #
def test_ceiling_ok_is_on_true_max_not_the_ks_surrogate():
    # bulk sits at 240 C (under the 250 C ceiling); one hot spike at 254 C, so
    # the true peak is OVER the ceiling but the volume-mean KS lower bound sits
    # UNDER it -- exactly the false green this guard forbids.
    T = np.concatenate([np.full(4000, 240.0), np.array([254.0])])
    pk = ceiling.peak_temp(T)
    status = ceiling.ceiling_status(pk["true_max_c"], 250.0, warn_c=235.0)
    # the TRUE peak is over the ceiling -> NOT ok
    assert status["true_max_c"] == 254.0
    assert status["T_ceiling_ok"] is False
    assert status["in_warning_band"] is True
    # the KS surrogate sits BELOW the ceiling -- reading the ceiling off the
    # surrogate would falsely certify a part that is actually over the ceiling
    assert pk["ks_aggregate_c"] < 250.0 < pk["true_max_c"]
    # ceiling_status has no way to be fed the surrogate (no ks argument)
    import inspect
    assert "ks" not in inspect.signature(ceiling.ceiling_status).parameters


def test_ceiling_status_ok_when_true_max_under_ceiling():
    s = ceiling.ceiling_status(243.0, 250.0, warn_c=235.0)
    assert s["T_ceiling_ok"] is True
    assert s["over_by_c"] < 0
    assert s["in_warning_band"] is True         # 243 >= 235


# --------------------------------------------------------------------------- #
# melt completeness, reported separately
# --------------------------------------------------------------------------- #
def test_melt_completeness_is_min_in_part_peak_vs_onset():
    Tpk = np.array([190.0, 200.0, 170.0, 999.0])     # one node never melted
    mask = np.array([True, True, True, False])
    c = ceiling.melt_completeness(Tpk, mask, 185.0)
    assert c["min_in_part_peak_c"] == 170.0
    assert c["complete"] is False
    assert c["n_unmelted"] == 1

    Tpk2 = np.array([190.0, 200.0, 188.0])
    mask2 = np.array([True, True, True])
    c2 = ceiling.melt_completeness(Tpk2, mask2, 185.0)
    assert c2["complete"] is True
    assert c2["n_unmelted"] == 0
