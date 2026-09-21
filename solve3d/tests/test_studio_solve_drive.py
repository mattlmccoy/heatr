"""Contract tests for the studio_solve ceiling-feasible-drive PRODUCER.

The Studio's Grade-and-Print consumer (studio3d/densify_job.recommended_drive,
studio3d/package_verify.verify_package) reads the PINNED fields written into
studio_solve_results.json:

  recommended_power_density_w_per_m3  absolute W/m3 = frac * baseline, or NULL
  recommended_drive_frac              the chosen a (e.g. 0.58), or NULL
  chamber_tag                         the chamber the drive is valid in
  ceiling_c, thermal_config_path      provenance (the shared 250 C source)
  recommended_drive_reason            a string; on honest-null says why

POLICY (2026-08-10, approved by Matt after the cube false-null). The cheap
UNIFORM drive sweep only BRACKETS the densification onset, gated against the
REAL degradation ceiling T_ceiling_C = 250 C -- NOT the warning band. A drive is
FEASIBLE iff it densifies (`reached_rho`) AND its uniform peak <= the real
ceiling. The HIGHEST feasible drive is handed to the heavy shaped solve, which
targets T_eff = ceiling - cross-engine reserve (235 C) BY CONSTRUCTION, and the
cross-engine `is_sendable` gate (shaped peak vs 250 on BOTH engines) is the FINAL
arbiter; if it fails, the Studio lane backs off down the emitted candidate
ladder.

WHY THE CHANGE. The old rule judged the UNIFORM peak against T_eff=235, baking
in the SQUARE's +11-13 C upward relocation as if universal. It false-nulled
COMPACT shapes whose dopant is ceiling-neutral: the cube densifies at 0.58x with
uniform peak 243.6 C -- under the real 250 ceiling, over 235 -- and its SHAPED
map ships (235.08 dolfinx / 246.9 heatr3d, both < 250, verify_stage_b4_cube).
The relocation sign is shape-dependent (square +11-13 up, cube ~0), so a fixed
uniform<=235 reserve is wrong; shaped-is_sendable<=250 is the honest arbiter.

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
    # 0.38 (248) is UNDER the real 250 ceiling -> feasible under the new policy;
    # 0.42 (262) is over 250 -> not. Highest feasible is 0.38.
    return {
        0.30: {"true_peak_c": 210.0, "reached_rho": True, "achieved_rho": 0.98},
        0.34: {"true_peak_c": 235.0, "reached_rho": True, "achieved_rho": 0.98},
        0.38: {"true_peak_c": 248.0, "reached_rho": True, "achieved_rho": 0.98},
        0.42: {"true_peak_c": 262.0, "reached_rho": True, "achieved_rho": 0.99},
    }


def _cube_like():
    """The real cube data: densifies throughout, uniform peak crosses the REAL
    250 ceiling between 0.58x (243.6, ships) and 0.66x (251.7, over)."""
    return {
        0.42: {"true_peak_c": 200.4, "reached_rho": True, "achieved_rho": 0.71},
        0.50: {"true_peak_c": 221.1, "reached_rho": True, "achieved_rho": 0.89},
        0.58: {"true_peak_c": 243.6, "reached_rho": True, "achieved_rho": 0.98},
        0.66: {"true_peak_c": 251.7, "reached_rho": True, "achieved_rho": 0.98},
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


def test_picks_highest_densifying_drive_under_real_ceiling():
    """NEW POLICY: gate the densification bracket against the REAL 250 ceiling.
    Feasible: 0.30 (210), 0.34 (235), 0.38 (248) -- all densify AND under 250.
    0.42 (262) is over 250 -> not. Highest feasible is 0.38 (was 0.34 under the
    old uniform<=235 rule, which is the false-null this fixes)."""
    rec = _rec(_feasible())
    assert rec["recommended_drive_frac"] == 0.38
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.38 * BASELINE)


def test_uniform_peak_under_real_ceiling_is_feasible():
    """REGRESSION for the cube false-null: a uniform peak of 248 C is under the
    real 250 ceiling. The OLD rule judged it against T_eff=235 and honest-nulled;
    the NEW rule brackets against 250, so 0.38x is feasible and handed to the
    shaped solve (is_sendable arbitrates). It must NOT honest-null."""
    rec = _rec({0.38: {"true_peak_c": 248.0, "reached_rho": True,
                       "achieved_rho": 0.98}})
    assert rec["recommended_drive_frac"] == 0.38
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.38 * BASELINE)


def test_uniform_peak_over_real_ceiling_is_not_feasible():
    """A uniform peak OVER the real 250 ceiling is over-driven even before
    grading -> honest-null (the false-green refusal still bites at the real
    ceiling)."""
    rec = _rec({0.42: {"true_peak_c": 262.0, "reached_rho": True,
                       "achieved_rho": 0.99}})
    assert rec["recommended_power_density_w_per_m3"] is None
    assert rec["recommended_drive_frac"] is None
    assert "drive_limited" in rec["recommended_drive_reason"]
    assert "250" in rec["recommended_drive_reason"]


def test_cube_like_part_ships_at_058x_not_honest_nulled():
    """The exact cube regression: densifies throughout, uniform peak 243.6 C at
    0.58x (under 250, over 235) and 251.7 C at 0.66x (over 250). The OLD rule
    honest-nulled the whole part (243.6 > 235). The NEW rule recommends 0.58x
    (highest densifying under the real ceiling)."""
    rec = _rec(_cube_like())
    assert rec["recommended_drive_frac"] == 0.58
    assert rec["recommended_power_density_w_per_m3"] == pytest.approx(0.58 * BASELINE)


def test_reason_documents_is_sendable_as_final_arbiter():
    """The feasible reason must tell the downstream consumer that the shaped
    solve + cross-engine is_sendable gate is the FINAL arbiter (so it verifies
    and backs off), not that the uniform peak alone certifies the drive."""
    rec = _rec(_cube_like())
    reason = rec["recommended_drive_reason"].lower()
    assert "is_sendable" in reason
    assert "arbiter" in reason or "final" in reason


def test_t_eff_carried_as_shaped_solve_target_not_picker_gate():
    """t_eff_c (235) is STILL carried, but now as the SHAPED-SOLVE TARGET handed
    downstream, not the picker's feasibility gate. The picker gates on the real
    ceiling (250); the candidates record BOTH flags for transparency."""
    rec = _rec(_feasible())
    assert rec["ceiling_c"] == 250.0                 # the picker gate
    assert rec["t_eff_c"] == pytest.approx(235.0)     # the shaped-solve target
    assert rec["delta_headroom_c"] == pytest.approx(15.0)
    # each candidate carries both the ceiling test (the gate) and the T_eff info
    for c in rec["candidates"]:
        assert "under_ceiling" in c and "under_t_eff" in c
        assert c["feasible"] == bool(c["reached_rho"] and c["under_ceiling"])


def test_explicit_t_eff_carried_and_cites_t_warning():
    """When t_eff_c is passed (from thermal_config.T_warning_C) it is carried as
    the shaped-solve target and cited as T_warning_C; feasibility still gates on
    the real 250 ceiling, so 0.38 wins."""
    rec = ss.select_recommended_drive(
        _feasible(), baseline=BASELINE, ceiling_c=250.0, chamber_tag="ch060",
        thermal_config_path=TCFG_PATH, rho_target=0.98, t_eff_c=235.0)
    assert rec["t_eff_c"] == 235.0
    assert rec["ceiling_c"] == 250.0
    assert rec["delta_headroom_c"] == pytest.approx(15.0)
    assert rec["t_eff_source"] == "thermal_config.T_warning_C"
    assert "T_warning_C" in rec["recommended_drive_reason"]
    assert rec["recommended_drive_frac"] == 0.38     # gated on 250, not 235


def test_recommended_drive_for_part_reads_t_warning_from_thermal_config():
    """The wiring reads T_warning_C from the SAME thermal_config.json the Studio
    verify reads T_ceiling_C from -- single source, no drift. The shaped-solve
    target is carried; the drive is gated on the real ceiling."""
    from solve3d import stage_a
    tcfg = stage_a.thermal_config()
    assert tcfg["T_warning_C"] == 235.0         # the shared source value

    def ramp(drive_a, **kw):
        # 0.30->210, 0.34->227.2, 0.38->244.4, 0.42->261.6 ; under 250 up to 0.38
        return {"drive_a": drive_a, "power_density_w_per_m3": drive_a * BASELINE,
                "true_peak_c": 210.0 + (drive_a - 0.30) * 430.0,
                "reached_rho": True, "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=None, rings=None, z_lo=0.0, z_hi=0.0,
        candidates=(0.30, 0.34, 0.38, 0.42), baseline=BASELINE,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH,
        max_time_s=1.0, peak_probe=ramp)          # ceiling_c/rho from config
    assert rec["ceiling_c"] == 250.0
    assert rec["t_eff_c"] == 235.0
    assert rec["t_eff_source"] == "thermal_config.T_warning_C"
    assert "T_warning_C" in rec["recommended_drive_reason"]
    assert rec["recommended_drive_frac"] == 0.38    # gated on the real ceiling


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
    def ramp(drive_a, **kw):
        # 0.30->210, 0.34->227.2, 0.38->244.4, 0.42->261.6 ; under 250 up to 0.38
        return {"drive_a": drive_a,
                "power_density_w_per_m3": drive_a * BASELINE,
                "true_peak_c": 210.0 + (drive_a - 0.30) * 430.0,
                "reached_rho": True, "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=None, rings=None, z_lo=0.0, z_hi=0.0,
        candidates=(0.30, 0.34, 0.38, 0.42), baseline=BASELINE, ceiling_c=250.0,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH, rho_target=0.98,
        max_time_s=1.0, peak_probe=ramp)
    assert rec["recommended_drive_frac"] == 0.38    # highest densifying under 250
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
    assert rec["recommended_drive_frac"] == 0.38   # highest densifying under 250


def test_cube_like_ship_via_wiring_default_candidates():
    """Regression for the live-cube honest-null via the full wiring on the
    DEFAULT candidate ladder: a compact shape that densifies throughout with
    uniform peak 243.6 C at 0.58x (under 250) and 251.7 C at 0.66x (over 250)
    must recommend 0.58x, not honest-null."""
    def cube(msh, rings, z_lo, z_hi, drive_a, *, baseline, rho_target,
             max_time_s, sample_dt_s):
        table = {0.26: 146.2, 0.34: 180.8, 0.42: 200.4, 0.50: 221.1,
                 0.58: 243.6, 0.66: 251.7}
        peak = table.get(round(float(drive_a), 2), 300.0)
        return {"drive_a": float(drive_a),
                "power_density_w_per_m3": float(drive_a) * BASELINE,
                "true_peak_c": peak, "reached_rho": bool(drive_a >= 0.50),
                "achieved_rho": 0.98 if drive_a >= 0.58 else 0.80}
    rec = ss.recommended_drive_for_part(
        msh=object(), rings=None, z_lo=0.0, z_hi=0.0,   # default (widened) candidates
        baseline=BASELINE, chamber_tag="ch060", thermal_config_path=TCFG_PATH,
        max_time_s=1.0, peak_probe=cube)
    assert rec["recommended_drive_frac"] == 0.58                 # highest densifying under 250
    assert rec["recommended_power_density_w_per_m3"] is not None  # NOT honest-null


# ---- (B) adaptive drive probe: find the highest feasible drive in ~3 evals - #
# The fixed 6-point ladder is ~37% of the producer's wall. The adaptive probe
# secants toward the ceiling crossing so the library runs in fewer forwards.
# SAFETY: the adaptive search only chooses WHICH drives to evaluate;
# select_recommended_drive still arbitrates the pick, so an imperfect search can
# only under-drive (slower print), never over-drive (unsafe).
def _linear_probe(base=100.0, slope=300.0, onset=0.30):
    """peak(a) = base + slope*a ; densifies for a >= onset. Records calls.
    Default crossing peak==250 at a = (250-100)/300 = 0.50."""
    calls = []

    def probe(a):
        calls.append(round(float(a), 3))
        return {"drive_a": float(a), "power_density_w_per_m3": float(a) * BASELINE,
                "true_peak_c": base + slope * float(a),
                "reached_rho": bool(a >= onset),
                "achieved_rho": 0.98 if a >= onset else 0.70}
    probe.calls = calls
    return probe


def _sel(peaks):
    return ss.select_recommended_drive(
        peaks, baseline=BASELINE, ceiling_c=250.0, chamber_tag="ch060",
        thermal_config_path=TCFG_PATH, rho_target=0.98)


def test_adaptive_converges_near_ceiling_crossing_within_budget():
    probe = _linear_probe()                       # crossing at a=0.50
    peaks = ss.adaptive_drive_peaks(
        probe, ceiling_c=250.0, a_min=0.26, a_max=0.66, max_evals=4)
    assert len(probe.calls) <= 4                   # budget respected
    rec = _sel(peaks)
    # finds a feasible drive near the 0.50 crossing, NOT stuck at a_min
    assert rec["recommended_drive_frac"] is not None
    assert 0.44 <= rec["recommended_drive_frac"] <= 0.50


def test_adaptive_whole_range_feasible_returns_top_of_range():
    probe = _linear_probe(base=180.0, slope=100.0)  # peak(0.66)=246 < 250, all under
    peaks = ss.adaptive_drive_peaks(
        probe, ceiling_c=250.0, a_min=0.26, a_max=0.66, max_evals=4)
    rec = _sel(peaks)
    assert rec["recommended_drive_frac"] == 0.66     # highest, whole range feasible
    assert len(probe.calls) <= 4


def test_adaptive_min_drive_over_ceiling_honest_nulls():
    probe = _linear_probe(base=100.0, slope=800.0)  # peak(0.26)=308 > 250 already
    peaks = ss.adaptive_drive_peaks(
        probe, ceiling_c=250.0, a_min=0.26, a_max=0.66, max_evals=4)
    rec = _sel(peaks)
    assert rec["recommended_drive_frac"] is None      # fabricates no feasibility
    assert len(probe.calls) <= 4


def test_adaptive_uses_fewer_evals_than_fixed_ladder():
    probe = _linear_probe()
    ss.adaptive_drive_peaks(
        probe, ceiling_c=250.0, a_min=0.26, a_max=0.66, max_evals=4)
    assert len(probe.calls) < len(ss.CEILING_DRIVE_CANDIDATES)   # < 6


def test_recommended_drive_for_part_adaptive_probes_few_and_selects():
    calls = []

    def ramp(msh, rings, z_lo, z_hi, drive_a, *, baseline, rho_target,
             max_time_s, sample_dt_s):
        calls.append(round(float(drive_a), 3))
        return {"drive_a": float(drive_a),
                "power_density_w_per_m3": float(drive_a) * BASELINE,
                "true_peak_c": 100.0 + 300.0 * float(drive_a),   # crossing 0.50
                "reached_rho": bool(drive_a >= 0.30), "achieved_rho": 0.98}

    rec = ss.recommended_drive_for_part(
        msh=object(), rings=None, z_lo=0.0, z_hi=0.0, baseline=BASELINE,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH, max_time_s=1.0,
        peak_probe=ramp, adaptive=True, max_evals=4)
    assert len(calls) <= 4                             # fewer than the 6-ladder
    assert rec["recommended_drive_frac"] is not None
    assert 0.44 <= rec["recommended_drive_frac"] <= 0.50


# ---- (e) over-driven fallback: the grading opportunity ------------------- #
# When no UNIFORM drive both densifies AND stays under the ceiling, but some drive
# DOES densify (by busting the ceiling), that is exactly the case grading exists
# for. select_recommended_drive keeps honest-nulling the RECOMMENDED (path-A)
# fields but additively emits an `over_driven_fallback` = the LOWEST densifying
# drive (least bust -> easiest for the shaped AL to pull under the ceiling). The
# ceiling_restore path consumes it; path A ignores it. A cold part -> None.
def _cube_like_all_busting():
    """The REAL vol_cube run: 0.26x cold, 0.34-0.66x all densify but bust 250
    (uniform peaks 261..293). The old fallback picked 0.34x (least bust, +11C),
    which starves grading -> near-uniform map -> fails the symmetry gate. The fix
    picks the drive whose overshoot is closest to the grading-target (+35C default
    = the proven stage_b4 0.58x regime)."""
    return {
        0.26: {"true_peak_c": 220.0, "reached_rho": False, "achieved_rho": 0.84},
        0.34: {"true_peak_c": 261.0, "reached_rho": True, "achieved_rho": 0.98},
        0.42: {"true_peak_c": 273.0, "reached_rho": True, "achieved_rho": 0.98},
        0.50: {"true_peak_c": 281.0, "reached_rho": True, "achieved_rho": 0.98},
        0.58: {"true_peak_c": 288.0, "reached_rho": True, "achieved_rho": 0.98},
        0.66: {"true_peak_c": 293.0, "reached_rho": True, "achieved_rho": 0.98},
    }


def test_over_driven_fallback_picks_meaningful_grading_drive_not_lowest():
    """The fallback must NOT pick the lowest densifying drive (starves grading);
    it picks the drive whose uniform overshoot is closest to the +35C grading
    target = 0.58x (288C, +38) for the cube, the proven strong-grading regime."""
    rec = _rec(_cube_like_all_busting())
    assert rec["recommended_power_density_w_per_m3"] is None   # path A unchanged
    fb = rec["over_driven_fallback"]
    assert fb is not None
    assert fb["drive_frac"] != pytest.approx(0.34)             # NOT the starved-grading min
    assert fb["drive_frac"] == pytest.approx(0.58)             # closest to +35C overshoot
    assert fb["true_peak_c"] == pytest.approx(288.0)


def test_over_driven_fallback_target_overshoot_is_tunable():
    """A smaller grading target picks a cooler drive (closest to that overshoot)."""
    rec = ss.select_recommended_drive(
        _cube_like_all_busting(), baseline=BASELINE, ceiling_c=250.0,
        chamber_tag="ch060", thermal_config_path=TCFG_PATH, rho_target=0.98,
        grading_target_overshoot_c=10.0)
    fb = rec["over_driven_fallback"]
    assert fb["drive_frac"] == pytest.approx(0.34)             # 261C = +11, closest to +10


def test_over_driven_case_emits_most_grading_available_fallback():
    rec = _rec(_all_over_ceiling())          # 0.34/255 and 0.38/268 both densify
    assert rec["recommended_power_density_w_per_m3"] is None   # path A unchanged
    fb = rec["over_driven_fallback"]
    assert fb is not None
    # neither reaches +35C overshoot; the closest (most grading available) is 0.38 (+18)
    assert fb["drive_frac"] == pytest.approx(0.38)
    assert fb["power_density_w_per_m3"] == pytest.approx(0.38 * BASELINE)
    assert fb["true_peak_c"] == pytest.approx(268.0)


def test_cold_part_has_no_over_driven_fallback():
    rec = _rec(_never_densifies())
    assert rec["recommended_power_density_w_per_m3"] is None
    assert rec["over_driven_fallback"] is None


def test_feasible_case_carries_no_fallback():
    rec = _rec(_feasible())                    # a feasible drive exists
    assert rec["recommended_power_density_w_per_m3"] is not None
    assert rec.get("over_driven_fallback") is None


# ---- (f) drive gate on the mesh-stable KS peak (fix A) --------------------- #
# The raw true-MAX peak is mesh-sensitive (corner EQS singularity), which can
# false-null a drive whose real, solve-relevant peak is under the ceiling. Gate
# drive SELECTION on the mesh-stable KS-aggregate peak when the probe supplies
# it; still record true_peak_c for the downstream is_sendable safety gate (which
# is unchanged). Absent ks_peak_c -> fall back to true_peak (byte-compatible).
def test_drive_gate_uses_ks_peak_when_present():
    peaks = {0.50: {"true_peak_c": 274.0, "ks_peak_c": 248.0,
                    "reached_rho": True, "achieved_rho": 0.98}}
    rec = _rec(peaks)
    assert rec["recommended_drive_frac"] == 0.50            # KS 248 < 250 -> feasible
    c = rec["candidates"][0]
    assert c["true_peak_c"] == 274.0 and c["ks_peak_c"] == 248.0   # true peak kept


def test_drive_gate_falls_back_to_true_peak_without_ks():
    assert _rec({0.50: {"true_peak_c": 248.0, "reached_rho": True,
                        "achieved_rho": 0.98}})["recommended_drive_frac"] == 0.50
    assert _rec({0.50: {"true_peak_c": 274.0, "reached_rho": True,
                        "achieved_rho": 0.98}})["recommended_drive_frac"] is None


def test_drive_gate_ks_under_true_over_still_records_true_for_sendable():
    # a KS-feasible but true-busting drive: recommended (KS gate) BUT the true
    # peak rides in the candidate so is_sendable can still refuse the shaped map.
    rec = _rec({0.50: {"true_peak_c": 260.0, "ks_peak_c": 249.0,
                       "reached_rho": True, "achieved_rho": 0.98}})
    assert rec["recommended_drive_frac"] == 0.50
    assert rec["candidates"][0]["true_peak_c"] == 260.0
