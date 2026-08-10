"""Contract tests for the studio_solve ceiling-feasible-drive PRODUCER.

The Studio's Grade-and-Print consumer (studio3d/densify_job.recommended_drive,
studio3d/package_verify.verify_package) reads the PINNED fields written into
studio_solve_results.json:

  recommended_power_density_w_per_m3  absolute W/m3 = frac * baseline, or NULL
  recommended_drive_frac              the chosen a (e.g. 0.34), or NULL
  chamber_tag                         the chamber the drive is valid in
  ceiling_c, thermal_config_path      provenance (the shared 250 C source)
  recommended_drive_reason            a string; on honest-null says why

Feasibility is judged against the EFFECTIVE ceiling T_eff = ceiling_c -
DELTA_HEADROOM_C (15 C), NOT the raw ceiling: the deployed job runs a SHAPED
dopant map at the recommended drive, and the shape-optimal dopant RELOCATES the
peak +11-13 C (the B4 finding). A drive whose UNIFORM peak is 248 C sits under
250 but over T_eff=235, so the shaped map would land ~261 C and fail the
cross-engine is_sendable gate -- it is NOT feasible.

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
    # T_eff = 250 - 15 = 235. Feasible: 0.30 (210) and 0.34 (235). 0.38 (248)
    # is under 250 but OVER T_eff, so NOT feasible; 0.42 (262) is over both.
    # Highest T_eff-feasible drive is 0.34.
    assert rec["recommended_drive_frac"] == 0.34
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.34 * BASELINE)


def test_uniform_peak_under_ceiling_but_over_t_eff_is_not_feasible():
    """REGRESSION (the B4 bug): a uniform peak of 248 C is under the raw 250
    ceiling but over T_eff=235. Judging against 250 would recommend it and ship a
    map that relocates to ~261 C. It must be REFUSED -> honest-null."""
    rec = _rec({0.38: {"true_peak_c": 248.0, "reached_rho": True,
                       "achieved_rho": 0.98}})
    assert rec["recommended_power_density_w_per_m3"] is None
    assert rec["recommended_drive_frac"] is None
    assert "drive_limited" in rec["recommended_drive_reason"]


def test_t_eff_fallback_is_ceiling_minus_headroom_when_no_t_warning():
    """No explicit t_eff_c (older config path) -> T_eff = ceiling - 15."""
    rec = _rec(_feasible())
    assert rec["delta_headroom_c"] == 15.0
    assert rec["t_eff_c"] == pytest.approx(250.0 - 15.0)
    assert rec["ceiling_c"] == 250.0            # the REAL ceiling still carried
    assert rec["t_eff_source"].startswith("ceiling_minus")
    assert "T_eff" in rec["recommended_drive_reason"]
    assert "235" in rec["recommended_drive_reason"]


def test_explicit_t_eff_is_used_and_cites_t_warning():
    """When t_eff_c is passed (from thermal_config.T_warning_C), it is the
    selection target and the provenance/reason cite T_warning_C."""
    rec = ss.select_recommended_drive(
        _feasible(), baseline=BASELINE, ceiling_c=250.0, chamber_tag="ch060",
        thermal_config_path=TCFG_PATH, rho_target=0.98, t_eff_c=235.0)
    assert rec["t_eff_c"] == 235.0
    assert rec["ceiling_c"] == 250.0
    assert rec["delta_headroom_c"] == pytest.approx(15.0)
    assert rec["t_eff_source"] == "thermal_config.T_warning_C"
    assert "T_warning_C" in rec["recommended_drive_reason"]
    # feasibility unchanged: 248 still over 235 -> 0.34 wins
    assert rec["recommended_drive_frac"] == 0.34


def test_recommended_drive_for_part_reads_t_warning_from_thermal_config():
    """The wiring reads T_warning_C from the SAME thermal_config.json the Studio
    verify reads T_ceiling_C from -- single source, no drift."""
    from solve3d import stage_a
    tcfg = stage_a.thermal_config()
    assert tcfg["T_warning_C"] == 235.0         # the shared source value

    def cool(drive_a, **kw):
        return {"drive_a": drive_a, "power_density_w_per_m3": drive_a * BASELINE,
                "true_peak_c": 210.0 + (drive_a - 0.30) * 430.0,
                "reached_rho": True, "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=None, rings=None, z_lo=0.0, z_hi=0.0,
        candidates=(0.30, 0.34, 0.38, 0.42), baseline=BASELINE,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH,
        max_time_s=1.0, peak_probe=cool)          # ceiling_c/rho from config
    assert rec["ceiling_c"] == 250.0
    assert rec["t_eff_c"] == 235.0
    assert rec["t_eff_source"] == "thermal_config.T_warning_C"
    assert "T_warning_C" in rec["recommended_drive_reason"]


def test_delta_headroom_is_single_source_with_stage_b4():
    from solve3d import stage_b4
    assert ss.DELTA_HEADROOM_C == stage_b4.DELTA_HEADROOM_C
    # and the config's T_warning_C equals ceiling - the B4 headroom
    from solve3d import stage_a
    tcfg = stage_a.thermal_config()
    assert tcfg["T_warning_C"] == pytest.approx(
        tcfg["T_ceiling_C"] - stage_b4.DELTA_HEADROOM_C)


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
        # peak rises with drive: 0.30->210, 0.34->227.2, 0.38->244.4, 0.42->261.6
        # T_eff=235 -> feasible are 0.30 and 0.34; highest feasible is 0.34
        return {"drive_a": drive_a,
                "power_density_w_per_m3": drive_a * BASELINE,
                "true_peak_c": 210.0 + (drive_a - 0.30) * 430.0,
                "reached_rho": True, "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=None, rings=None, z_lo=0.0, z_hi=0.0,
        candidates=(0.30, 0.34, 0.38, 0.42), baseline=BASELINE, ceiling_c=250.0,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH, rho_target=0.98,
        max_time_s=1.0, peak_probe=cool_ramp)
    assert rec["recommended_drive_frac"] == 0.34
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.34 * BASELINE)
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


def test_recommended_drive_for_part_calls_probe_with_real_signature():
    """Regression for the acceptance-test bug: the ladder call site must invoke
    the probe so a stub matching the REAL _uniform_end_state_peak signature
    (msh first, drive_a keyword-reachable) works -- not only a (drive_a, **kw)
    stub. The old positional call `probe(float(a), msh=msh, ...)` bound float(a)
    to msh AND passed msh=msh -> TypeError: multiple values for 'msh'. The unit
    stubs hid it (drive-first); the first real --ceiling-drive run surfaced it."""
    seen = []

    def real_sig_probe(msh, rings, z_lo, z_hi, drive_a, *, baseline,
                       rho_target, max_time_s, sample_dt_s):
        seen.append(float(drive_a))
        return {"drive_a": float(drive_a),
                "power_density_w_per_m3": float(drive_a) * BASELINE,
                "true_peak_c": 210.0 + (float(drive_a) - 0.30) * 430.0,
                "reached_rho": True, "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=object(), rings=None, z_lo=0.0, z_hi=0.0,
        candidates=(0.30, 0.34, 0.38, 0.42), baseline=BASELINE,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH,
        max_time_s=1.0, peak_probe=real_sig_probe)
    assert seen == [0.30, 0.34, 0.38, 0.42]        # every candidate reached the probe
    assert rec["recommended_drive_frac"] == 0.34   # highest feasible under T_eff=235


def test_widened_ladder_reaches_compact_shape_feasible_drive():
    """Regression for the live-cube honest-null: a compact shape whose feasible
    drive is ABOVE the old 0.42x cap (cube/pyramid ~0.57-0.585x) must now be
    reachable, not honest-nulled. Uses the DEFAULT (widened) candidate ladder."""
    def compact(msh, rings, z_lo, z_hi, drive_a, *, baseline, rho_target,
                max_time_s, sample_dt_s):
        # uniform peak under T_eff=235 up to 0.58x, over at 0.66x; densifies throughout
        peak = 233.0 + (float(drive_a) - 0.58) * 100.0
        return {"drive_a": float(drive_a),
                "power_density_w_per_m3": float(drive_a) * BASELINE,
                "true_peak_c": peak, "reached_rho": True, "achieved_rho": 0.98}
    rec = ss.recommended_drive_for_part(
        msh=object(), rings=None, z_lo=0.0, z_hi=0.0,   # default (widened) candidates
        baseline=BASELINE, chamber_tag="ch060", thermal_config_path=TCFG_PATH,
        max_time_s=1.0, peak_probe=compact)
    assert rec["recommended_drive_frac"] == 0.58                 # highest feasible, above old 0.42 cap
    assert rec["recommended_power_density_w_per_m3"] is not None  # NOT honest-null
