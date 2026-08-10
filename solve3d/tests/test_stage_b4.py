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
import pytest

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


def test_drive_probe_pick_is_highest_drive_under_t_eff():
    # CORRECTED rule (2026-08-10): the AL shapes the peak UP from the uniform
    # start, so the uniform peak MUST be under T_eff for the AL to reach it. Pick
    # the HIGHEST drive whose uniform peak <= T_eff (most throughput, still
    # feasible). Over-T_eff drives are rejected outright.
    peaks = {0.55: 226.79, 0.60: 242.0}
    pick = b4.pick_backed_off_drive(peaks, t_eff=235.0)
    assert pick["drive_a"] == 0.55           # 0.60 (242) is over T_eff -> out
    assert pick["under_t_eff"] is True
    assert pick["drive_limited"] is False
    assert pick["shaping_room_c"] == pytest.approx(235.0 - 226.79)


def test_drive_probe_pick_rejects_over_t_eff_even_when_closer_cylinder_bug():
    # THE cylinder regression: 0.60x=236.54 sits CLOSER to any band centre than
    # 0.55x=222.78, but it is OVER T_eff (235). The old closest-to-centre
    # fallback wrongly picked it; the corrected rule must pick 0.55x.
    peaks = {0.55: 222.78, 0.60: 236.54}
    pick = b4.pick_backed_off_drive(peaks, t_eff=235.0)
    assert pick["drive_a"] == 0.55
    assert pick["under_t_eff"] is True


def test_drive_probe_pick_takes_the_highest_of_several_under_t_eff():
    peaks = {0.34: 224.0, 0.36: 229.5, 0.38: 234.0}
    pick = b4.pick_backed_off_drive(peaks, t_eff=235.0)
    assert pick["drive_a"] == 0.38           # all under 235 -> highest wins
    assert pick["shaping_room_c"] == pytest.approx(1.0)


def test_drive_probe_pick_honest_null_when_no_drive_is_under_t_eff():
    # drive-limited: every candidate cooks over T_eff. Report the least-over one
    # as EVIDENCE only, flagged infeasible -- never a silent over-ceiling pick.
    peaks = {0.60: 236.54, 0.65: 245.0}
    pick = b4.pick_backed_off_drive(peaks, t_eff=235.0)
    assert pick["drive_limited"] is True
    assert pick["under_t_eff"] is False
    assert pick["drive_a"] == 0.60           # least-over, as evidence
    assert pick["shaping_room_c"] < 0.0


def test_cli_delta_ema_flows_into_run_solve_al_b4(monkeypatch):
    """Sharp shapes (cone) oscillated at the default DELTA_EMA=0.5; the launch
    command must be able to lower it. --delta-ema must reach run_solve_al_b4."""
    import sys
    captured = {}

    def fake_solve(**kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(b4, "run_solve_al_b4", fake_solve)
    monkeypatch.setattr(sys, "argv",
                        ["stage_b4", "--solve", "--shape", "cone",
                         "--drive-a", "0.57", "--delta-ema", "0.3"])
    b4.main()
    assert captured["shape"] == "cone"
    assert captured["drive_a"] == 0.57
    assert captured["delta_ema"] == 0.3


def test_run_solve_al_b4_default_delta_ema_is_the_frozen_value():
    import inspect
    from solve3d import stage_b3 as b3
    sig = inspect.signature(b4.run_solve_al_b4)
    assert sig.parameters["delta_ema"].default == b3.DELTA_EMA


def test_finalize_drive_probe_writes_canonical_launch_fields(tmp_path,
                                                             monkeypatch):
    """The canonical JSON the campaign + launch commands read must carry
    chosen_drive_a, shaping_room_c, drive_limited, grid, t_eff_c and
    all_candidates -- and reject the over-T_eff drive (cylinder measured peaks)."""
    monkeypatch.setattr(b4, "RESULTS", tmp_path)
    records = [
        {"drive_a": 0.55, "power_density_w_per_m3": 875352.2,
         "uniform_true_peak_c": 222.78, "under_t_eff": True,
         "margin_to_t_eff_c": 12.22},
        {"drive_a": 0.60, "power_density_w_per_m3": 954929.7,
         "uniform_true_peak_c": 236.54, "under_t_eff": False,
         "margin_to_t_eff_c": -1.54},
    ]
    doc = b4._finalize_drive_probe("cylinder", records, t_eff=235.0,
                                   ceiling_c=250.0, delta_headroom=15.0,
                                   band=(228.0, 232.0))
    assert doc["chosen_drive_a"] == 0.55          # NOT the over-T_eff 0.60
    assert doc["drive_limited"] is False
    assert doc["shaping_room_c"] == pytest.approx(235.0 - 222.78)
    assert doc["t_eff_c"] == 235.0
    assert len(doc["all_candidates"]) == 2
    import json
    written = json.loads((tmp_path / "stage_b4_drive_probe_cylinder.json")
                         .read_text())
    assert written["chosen_drive_a"] == 0.55
