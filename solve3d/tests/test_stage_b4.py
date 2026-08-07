"""Stage B4 tests: the lower-drive, headroom-margin AL parameterization so the
shaped map passes BOTH engines' ceiling gates (cross-engine is_sendable).

B4 is a small parameterization increment on B3 (same FD-gated gradient, same AL
machinery); the two coupled changes are (1) an EFFECTIVE ceiling T_eff = 250 -
Delta_headroom that the restoration shift targets, and (2) a backed-off drive so
the uniform dolfinx peak sits a few C UNDER T_eff. These pure-logic tests pin the
target split (no false-green: the 235 is only the AL target; is_shippable/
is_sendable are judged vs the REAL 250 ceiling) and the drive->power mapping.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_stage_b4.py -x -q
"""
import numpy as np

from solve3d import stage_a, stage_b4 as b4


# --------------------------------------------------------------------------- #
# Task 1: pure-logic target split (no physics) -- the no-false-green crux
# --------------------------------------------------------------------------- #
def test_t_ceiling_eff_is_ceiling_minus_headroom():
    # T_eff = 250 - 15 = 235; the restoration shift targets THIS, not 250.
    assert b4.t_ceiling_eff(ceiling=250.0, delta_headroom=15.0) == 235.0
    assert b4.t_ceiling_eff(ceiling=250.0, delta_headroom=0.0) == 250.0


def test_b4_targets_split_al_target_from_real_ceiling():
    # The AL/restoration shift drives MY dolfinx peak to T_eff (235); is_shippable
    # and is_sendable are STILL judged against the REAL 250 ceiling. Conflating
    # them would be a false-green (certifying vs the soft 235 target).
    t = b4.b4_targets(ceiling=250.0, delta_headroom=15.0)
    assert t["real_ceiling_c"] == 250.0          # what is_shippable/is_sendable use
    assert t["t_ceiling_eff_c"] == 235.0         # what the restoration shift targets
    assert t["delta_headroom_c"] == 15.0
    assert t["t_ceiling_eff_c"] < t["real_ceiling_c"]


def test_b4_shippable_reads_true_peak_vs_real_ceiling_not_teff():
    # A dolfinx peak at ~235 (driven there by the T_eff=235 target) is trivially
    # is_shippable vs the REAL 250 ceiling. The framing must read the TRUE peak vs
    # 250 -- NOT vs 235 -- else a peak of 240 would falsely read "over the (235)
    # target" and block a map that clears 250. The real gate is the Studio heatr3d
    # <= 250 verify (separate, cross-engine), not this dolfinx-side arbiter.
    from solve3d import stage_b
    v = stage_b.shippable_verdict(true_peak_c=235.0, ks_peak_c=230.0,
                                  ceiling_c=250.0, fd_gate_passed=True)
    assert v["is_shippable"] is True
    assert v["over_by_c"] == 235.0 - 250.0       # vs 250, negative (under)


# --------------------------------------------------------------------------- #
# Task 2: the drive -> power-density mapping (reuses stage_a baseline)
# --------------------------------------------------------------------------- #
def test_power_density_for_drive_a_is_a_times_baseline():
    baseline = float(
        stage_a.recommended_power_settings(1.0)["power_density_w_per_m3"])
    for a in (0.34, 0.36, 0.38, 0.40):
        pw = b4.power_density_for_drive_a(a)
        assert abs(pw - a * baseline) < 1e-6 * baseline
    # a backed-off drive is strictly lower power than 0.40x
    assert b4.power_density_for_drive_a(0.36) < b4.power_density_for_drive_a(0.40)


def test_drive_probe_pick_selects_in_band():
    # pure selection logic: pick the candidate whose uniform peak lands in the
    # target band (~228-232 C), preferring the highest such drive (most headroom
    # to shape UP toward T_eff=235 while staying feasible).
    peaks = {0.34: 224.0, 0.36: 229.5, 0.38: 234.0}
    pick = b4.pick_backed_off_drive(peaks, band=(228.0, 232.0))
    assert pick["drive_a"] == 0.36
    assert pick["uniform_peak_c"] == 229.5
    # if two are in band, prefer the higher drive (more shaping headroom)
    peaks2 = {0.34: 228.5, 0.36: 231.0, 0.38: 236.0}
    pick2 = b4.pick_backed_off_drive(peaks2, band=(228.0, 232.0))
    assert pick2["drive_a"] == 0.36
