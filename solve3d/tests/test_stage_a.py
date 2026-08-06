"""Stage A Task 3: best-part drive selection logic.

Pure-logic tests (no dolfinx). The end-to-end sweep is the CLI driver
`python -m solve3d.stage_a --demo-sweep` -> solve3d/results/stage_a_drive_sweep.json.
"""
from __future__ import annotations

from solve3d import stage_a

# thresholds mirror the pre-registration defaults
CEILING = 250.0
RHO_FLOOR = 0.90
RHO_IDEAL = 0.98
W_D, W_S = 0.5, 0.5
TIE = 0.01


def _sel(records):
    return stage_a.select_from_sweep(records, CEILING, RHO_FLOOR, RHO_IDEAL,
                                     W_D, W_S, TIE)


def test_over_ceiling_drives_are_rejected():
    records = [
        {"drive_a": 1.0, "true_peak_c": 281.0, "achieved_rho": 0.98, "shape_iou": 0.99},
        {"drive_a": 0.6, "true_peak_c": 243.0, "achieved_rho": 0.98, "shape_iou": 0.97},
    ]
    out = _sel(records)
    assert out["verdict"] == "ok"
    # the 1.0x drive cooks over the ceiling -> never chosen despite top shape
    assert out["chosen_drive_a"] == 0.6
    assert out["n_feasible"] == 1


def test_best_part_not_fastest_drive():
    """Two feasible drives both reach the density floor. The HOTTER/faster 0.9x
    has worse shape; the cooler 0.6x has the best part. Best-part wins -- NOT the
    largest-under-ceiling drive."""
    records = [
        {"drive_a": 0.9, "true_peak_c": 248.0, "achieved_rho": 0.98, "shape_iou": 0.80},
        {"drive_a": 0.6, "true_peak_c": 240.0, "achieved_rho": 0.98, "shape_iou": 0.98},
        {"drive_a": 0.4, "true_peak_c": 225.0, "achieved_rho": 0.88, "shape_iou": 0.99},
    ]
    out = _sel(records)
    assert out["verdict"] == "ok"
    assert out["chosen_drive_a"] == 0.6         # best Q, not the fastest 0.9
    assert out["chosen"]["Q"] > 0.0
    # 0.4x is cooler and better shape but UNDER the density floor -> not valid
    assert out["n_valid"] == 2


def test_quality_tie_breaks_to_the_cooler_drive():
    records = [
        {"drive_a": 0.8, "true_peak_c": 246.0, "achieved_rho": 0.98, "shape_iou": 0.95},
        {"drive_a": 0.6, "true_peak_c": 238.0, "achieved_rho": 0.98, "shape_iou": 0.95},
    ]
    out = _sel(records)
    assert out["verdict"] == "ok"
    assert out["chosen_drive_a"] == 0.6         # identical Q -> cooler wins
    assert out["tie_broken_by_cooler"] is True


def test_honest_null_when_no_feasible_drive_reaches_target():
    """Every drive that stays under the ceiling is under-dense, and every drive
    that reaches density cooks over the ceiling. The verdict is the honest null
    -- no map is returned."""
    records = [
        {"drive_a": 1.0, "true_peak_c": 281.0, "achieved_rho": 0.98, "shape_iou": 0.99},
        {"drive_a": 0.5, "true_peak_c": 240.0, "achieved_rho": 0.82, "shape_iou": 0.90},
        {"drive_a": 0.3, "true_peak_c": 210.0, "achieved_rho": 0.70, "shape_iou": 0.85},
    ]
    out = _sel(records)
    assert out["verdict"] == "cannot_make_under_ceiling"
    assert out["chosen_drive_a"] is None
    # the best feasible density is reported as EVIDENCE, not as a solution
    assert out["best_feasible_density"]["drive_a"] == 0.5
    assert out["best_feasible_density"]["achieved_rho"] == 0.82


def test_recommended_power_settings_writes_only_the_one_field():
    """The output populates ONLY power_density_w_per_m3 (no schema change, no new
    keys): a=1.0 is the baseline Studio hardcodes (1.5915e6), value scales
    linearly, and rf_mode is deliberately NOT written (it defers to the Stage C
    2.1.0 bump; 'constant' carries no information today)."""
    base = stage_a.recommended_power_settings(1.0)
    assert abs(base["power_density_w_per_m3"] - 1.5915e6) / 1.5915e6 < 1e-4
    assert "rf_mode" not in base           # zero-key-change property protected
    # the only non-provenance key is the field itself
    assert [k for k in base if not k.startswith("_")] == ["power_density_w_per_m3"]
    assert base["_stage_a_provenance"]["field_path"] == \
        "power_settings.power_density_w_per_m3"
    half = stage_a.recommended_power_settings(0.55)
    assert abs(half["power_density_w_per_m3"]
               - 0.55 * base["power_density_w_per_m3"]) < 1.0


def test_shared_thermal_config_is_the_single_ceiling_source():
    """The shared config both lanes read carries the cited PA12 values and its
    ceiling agrees with gates.T_CEILING_C (which mirrors the Studio runner), so
    the cross-engine verify judges a drive against the identical number."""
    from solve3d import gates
    tc = stage_a.thermal_config()
    assert tc["T_ceiling_C"] == 250.0 == gates.T_CEILING_C
    assert tc["T_warning_C"] == 235.0
    assert tc["T_melt_onset_C"] == 185.0
    assert tc["rho_target"]["floor"] == 0.90
    assert tc["rho_target"]["practical_ideal"] == 0.98
    assert "c25eb5c" in tc["source"]


def test_score_drive_is_monotone_and_clipped():
    lo = stage_a.score_drive(0.80, 0.5, RHO_IDEAL, W_D, W_S)
    hi = stage_a.score_drive(0.98, 0.9, RHO_IDEAL, W_D, W_S)
    assert hi["Q"] > lo["Q"]
    # density completeness clips at 1.0 (rho above the ideal is not extra credit)
    full = stage_a.score_drive(1.10, 1.0, RHO_IDEAL, W_D, W_S)
    assert full["q_density"] == 1.0
