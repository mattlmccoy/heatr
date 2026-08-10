"""Contract tests for the studio_solve ceiling-feasible-drive PRODUCER.

The Studio's Grade-and-Print consumer (studio3d/densify_job.recommended_drive,
studio3d/package_verify.verify_package) reads the PINNED fields written into
studio_solve_results.json:

  recommended_power_density_w_per_m3  absolute W/m3 = frac * baseline, or NULL
  recommended_drive_frac              the chosen a (e.g. 0.34), or NULL
  chamber_tag                         the chamber the drive is valid in
  ceiling_c, thermal_config_path      provenance (the shared 250 C source)
  recommended_drive_reason            a string; on honest-null says why

These tests pin the CONTRACT and the honest-null (the false-green refusal): a
drive-limited part emits NULL power + a reason, never a cooking drive. They use
pure logic and a stubbed drive-selection probe -- no heavy solve, no mesh.
"""
from __future__ import annotations

import pytest

from solve3d import studio_solve as ss

# baseline the Studio hardcodes (studio3d/package.py:193); a=1.0 is exactly this.
BASELINE = 1_591_549.4309189534
TCFG_PATH = "solve3d/thermal_config.json"


def _rec(peaks):
    return ss.select_recommended_drive(
        peaks, baseline=BASELINE, ceiling_c=250.0, chamber_tag="ch060",
        thermal_config_path=TCFG_PATH, rho_target=0.98)


def _feasible():
    return {
        0.30: {"true_peak_c": 210.0, "reached_rho": True, "achieved_rho": 0.98},
        0.34: {"true_peak_c": 235.0, "reached_rho": True, "achieved_rho": 0.98},
        0.38: {"true_peak_c": 248.0, "reached_rho": True, "achieved_rho": 0.98},
        0.42: {"true_peak_c": 262.0, "reached_rho": True, "achieved_rho": 0.99},
    }


def _all_over_ceiling():
    return {
        0.34: {"true_peak_c": 255.0, "reached_rho": True, "achieved_rho": 0.98},
        0.38: {"true_peak_c": 268.0, "reached_rho": True, "achieved_rho": 0.99},
    }


def _never_densifies():
    return {
        0.20: {"true_peak_c": 180.0, "reached_rho": False, "achieved_rho": 0.55},
        0.24: {"true_peak_c": 195.0, "reached_rho": False, "achieved_rho": 0.70},
    }


# ---- (d) the field is absolute W/m3 = frac * baseline -------------------- #
def test_recommended_power_is_frac_times_baseline():
    rec = _rec({0.34: {"true_peak_c": 230.0, "reached_rho": True,
                       "achieved_rho": 0.98}})
    assert rec["recommended_drive_frac"] == 0.34
    assert rec["recommended_power_density_w_per_m3"] is not None
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.34 * BASELINE)


def test_picks_highest_feasible_drive():
    rec = _rec(_feasible())
    # 0.42 is over ceiling; 0.38 is the highest feasible (most throughput)
    assert rec["recommended_drive_frac"] == 0.38
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.38 * BASELINE)


# ---- (a) full contract present with correct types when feasible --------- #
def test_feasible_contract_fields_and_types():
    rec = _rec(_feasible())
    assert isinstance(rec["recommended_power_density_w_per_m3"], float)
    assert isinstance(rec["recommended_drive_frac"], float)
    assert isinstance(rec["chamber_tag"], str)
    assert isinstance(rec["ceiling_c"], float)
    assert isinstance(rec["thermal_config_path"], str)
    assert isinstance(rec["recommended_drive_reason"], str)


# ---- (b) honest-null: NULL power + a reason, never a cooking drive ------- #
def test_honest_null_when_every_densifying_drive_is_over_ceiling():
    rec = _rec(_all_over_ceiling())
    assert rec["recommended_power_density_w_per_m3"] is None
    assert rec["recommended_drive_frac"] is None
    assert "drive_limited" in rec["recommended_drive_reason"]
    assert "250" in rec["recommended_drive_reason"]


def test_honest_null_when_no_drive_reaches_rho_target():
    rec = _rec(_never_densifies())
    assert rec["recommended_power_density_w_per_m3"] is None
    assert rec["recommended_drive_frac"] is None
    assert "drive_limited" in rec["recommended_drive_reason"]


def test_reason_is_string_and_never_a_cooking_drive_on_null():
    rec = _rec(_all_over_ceiling())
    assert isinstance(rec["recommended_drive_reason"], str)
    # the false-green refusal: null must not smuggle a power number through
    assert rec["recommended_power_density_w_per_m3"] is None


# ---- (c) chamber_tag + provenance ride through BOTH paths --------------- #
@pytest.mark.parametrize("peaks", [_feasible(), _all_over_ceiling(),
                                   _never_densifies()])
def test_chamber_and_provenance_ride_through(peaks):
    rec = _rec(peaks)
    assert rec["chamber_tag"] == "ch060"
    assert rec["ceiling_c"] == 250.0
    assert rec["thermal_config_path"] == TCFG_PATH


# ---- wiring: recommended_drive_for_part with a stubbed probe ------------- #
def test_recommended_drive_for_part_wires_probe_and_selects():
    def cool_ramp(drive_a, **kw):
        # peak rises with drive; 0.30/0.34/0.38 feasible, 0.42 over ceiling
        return {"drive_a": drive_a,
                "power_density_w_per_m3": drive_a * BASELINE,
                "true_peak_c": 210.0 + (drive_a - 0.30) * 430.0,
                "reached_rho": True, "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=None, rings=None, z_lo=0.0, z_hi=0.0,
        candidates=(0.30, 0.34, 0.38, 0.42), baseline=BASELINE, ceiling_c=250.0,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH, rho_target=0.98,
        max_time_s=1.0, peak_probe=cool_ramp)
    assert rec["recommended_drive_frac"] == 0.38
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.38 * BASELINE)
    assert len(rec["candidates"]) == 4


def test_recommended_drive_for_part_honest_null_all_hot():
    def hot(drive_a, **kw):
        return {"drive_a": drive_a,
                "power_density_w_per_m3": drive_a * BASELINE,
                "true_peak_c": 300.0, "reached_rho": True, "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=None, rings=None, z_lo=0.0, z_hi=0.0,
        candidates=(0.30, 0.34, 0.38), baseline=BASELINE, ceiling_c=250.0,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH, rho_target=0.98,
        max_time_s=1.0, peak_probe=hot)
    assert rec["recommended_power_density_w_per_m3"] is None
    assert "drive_limited" in rec["recommended_drive_reason"]


# ---- the emission merge into studio_solve_results.json ------------------ #
def test_merge_recommended_drive_folds_pinned_fields_into_doc():
    doc = {"solved_label": True}
    rec = _rec(_feasible())
    ss._merge_recommended_drive(doc, rec)
    for k in ("recommended_power_density_w_per_m3", "recommended_drive_frac",
              "chamber_tag", "ceiling_c", "thermal_config_path",
              "recommended_drive_reason"):
        assert k in doc, k
    assert doc["recommended_power_density_w_per_m3"] == \
        rec["recommended_power_density_w_per_m3"]
    # the full selection record is carried nested for provenance
    assert doc["recommended_drive"]["candidates"] == rec["candidates"]


def test_merge_carries_null_on_honest_null():
    doc = {}
    rec = _rec(_all_over_ceiling())
    ss._merge_recommended_drive(doc, rec)
    assert doc["recommended_power_density_w_per_m3"] is None
    assert doc["recommended_drive_frac"] is None
    assert isinstance(doc["recommended_drive_reason"], str)


# ---- the module baseline agrees with Stage A (a=1.0 baseline) ----------- #
def test_module_baseline_matches_stage_a():
    from solve3d import stage_a
    got = stage_a.recommended_power_settings(1.0)["power_density_w_per_m3"]
    assert ss.DRIVE_BASELINE_W_PER_M3 == pytest.approx(got)
