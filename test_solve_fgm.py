"""Tests for the shape-fidelity SOLVE entry point (scripts/solve_fgm.py).

Covers the pure logic first (red-green test-driven development):
  1. parsing of the new ``fgm_solve`` config block, and
  2. the production-format map emitter, round-tripped through the REAL
     production loader (``rfam_eqs_coupled._FgmFeedback.from_config``), which
     is the data-contract test: the emitted npz must inject without error and
     decode to the quantized saturation the solve delivered.

Production recipe under test (MULTISTART_REPORT.md / TOPOPT_REPORT.md verdicts,
FROZEN_CONVENTIONS_2D.md): filtered full-depth single start, 1.0 mm physical
filter radius, no smoothed-Heaviside projection by default (beta = 0),
conductivity-only channel, 4 bits per pixel output via the production
quantizer.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from scripts.solve_fgm import FgmSolveConfig, parse_fgm_solve_block


# ---------------------------------------------------------------------------
# 1. fgm_solve config block parsing
# ---------------------------------------------------------------------------

class TestParseFgmSolveBlock:
    def test_defaults_from_empty_config(self):
        sc = parse_fgm_solve_block({})
        assert sc.budget_forward_equivalents == 40.0
        assert sc.filter_radius_mm == 1.0
        assert sc.warm_start == "auto"
        assert sc.eps_channel_model_only is False
        assert sc.bpp == 4

    def test_defaults_when_block_absent_key_present_elsewhere(self):
        cfg = {"electric": {"voltage_v": 3600.0}}
        sc = parse_fgm_solve_block(cfg)
        assert sc.budget_forward_equivalents == 40.0

    def test_explicit_values(self):
        cfg = {"fgm_solve": {
            "budget_forward_equivalents": 8.0,
            "filter_radius_mm": 1.5,
            "warm_start": "cold",
            "eps_channel_model_only": True,
        }}
        sc = parse_fgm_solve_block(cfg)
        assert sc.budget_forward_equivalents == 8.0
        assert sc.filter_radius_mm == 1.5
        assert sc.warm_start == "cold"
        assert sc.eps_channel_model_only is True

    def test_warm_start_path_is_kept_verbatim(self):
        cfg = {"fgm_solve": {"warm_start": "outputs_eqs/some/map_m0p5.npz"}}
        sc = parse_fgm_solve_block(cfg)
        assert sc.warm_start == "outputs_eqs/some/map_m0p5.npz"

    def test_config_is_immutable(self):
        sc = parse_fgm_solve_block({})
        with pytest.raises(dataclasses.FrozenInstanceError):
            sc.bpp = 2  # type: ignore[misc]

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError, match="budget"):
            parse_fgm_solve_block({"fgm_solve": {"budget_forward_equivalents": 0}})

    def test_rejects_non_positive_radius(self):
        with pytest.raises(ValueError, match="filter_radius_mm"):
            parse_fgm_solve_block({"fgm_solve": {"filter_radius_mm": -1.0}})

    def test_rejects_unknown_keys_loudly(self):
        with pytest.raises(ValueError, match="unknown fgm_solve key"):
            parse_fgm_solve_block({"fgm_solve": {"warmstart": "cold"}})

    def test_rejects_bad_bpp(self):
        with pytest.raises(ValueError, match="bpp"):
            parse_fgm_solve_block({"fgm_solve": {"bpp": 3}})

    def test_rejects_non_dict_block(self):
        with pytest.raises(ValueError, match="fgm_solve"):
            parse_fgm_solve_block({"fgm_solve": [1, 2]})

    def test_cli_budget_override(self):
        sc = parse_fgm_solve_block({"fgm_solve": {"budget_forward_equivalents": 40.0}},
                                   budget_override=8.0)
        assert sc.budget_forward_equivalents == 8.0

    def test_is_frozen_dataclass_type(self):
        assert dataclasses.is_dataclass(FgmSolveConfig)


# ---------------------------------------------------------------------------
# 1b. v2.1.0 config keys: stop rule, optimizer policy, objective rescale
# ---------------------------------------------------------------------------

class TestV210SolveKeys:
    """The three adopted-by-verdict decisions, each flag-visible.

    Defaults change behaviour, which is why v2.1.0 is a MINOR bump with the
    change stated plainly: stop_rule 'j_phi' restores the v2.0.x read state.
    """

    def test_stop_rule_defaults_to_j_asym(self):
        sc = parse_fgm_solve_block({})
        assert sc.stop_rule == "j_asym"

    def test_the_stop_rule_prices_default_to_the_reports_recommended_values(self):
        # DENSE_IFF_INBOUNDS_REPORT.md Section 6 (w_out 2 to 3) and Section 7
        # (floor 0.85 relative density).
        sc = parse_fgm_solve_block({})
        assert sc.w_out == 2.0
        assert sc.density_floor_rho_rel == 0.85

    def test_j_phi_restores_the_legacy_read_state(self):
        sc = parse_fgm_solve_block({"fgm_solve": {"stop_rule": "j_phi"}})
        assert sc.stop_rule == "j_phi"

    def test_an_unknown_stop_rule_is_rejected(self):
        with pytest.raises(ValueError, match="stop_rule"):
            parse_fgm_solve_block({"fgm_solve": {"stop_rule": "melt_onset"}})

    def test_a_non_positive_out_of_bounds_price_is_rejected(self):
        with pytest.raises(ValueError, match="w_out"):
            parse_fgm_solve_block({"fgm_solve": {"w_out": 0.0}})

    def test_a_floor_outside_the_stated_band_is_rejected(self):
        for bad in (0.4, 1.0, 1.5):
            with pytest.raises(ValueError, match="density_floor_rho_rel"):
                parse_fgm_solve_block({"fgm_solve": {"density_floor_rho_rel": bad}})

    def test_optimizer_defaults_to_auto_the_per_class_policy(self):
        sc = parse_fgm_solve_block({})
        assert sc.optimizer == "auto"

    def test_optimizer_override_is_respected_either_way(self):
        assert parse_fgm_solve_block(
            {"fgm_solve": {"optimizer": "mma"}}).optimizer == "mma"
        assert parse_fgm_solve_block(
            {"fgm_solve": {"optimizer": "lbfgsb"}}).optimizer == "lbfgsb"

    def test_an_unknown_optimizer_is_rejected(self):
        with pytest.raises(ValueError, match="optimizer"):
            parse_fgm_solve_block({"fgm_solve": {"optimizer": "newton"}})

    def test_objective_rescale_defaults_on(self):
        sc = parse_fgm_solve_block({})
        assert sc.objective_rescale is True

    def test_objective_rescale_can_be_turned_off(self):
        sc = parse_fgm_solve_block({"fgm_solve": {"objective_rescale": False}})
        assert sc.objective_rescale is False

    def test_the_map_solve_objective_is_the_melt_objective_and_is_not_a_knob(self):
        """The verdict: the melt objective drives the MAP, J_asym owns the STOP.

        There is deliberately no config key that switches the map driver, so a
        run cannot silently become a different experiment.
        """
        sc = parse_fgm_solve_block({})
        assert sc.map_objective == "j_phi"
        with pytest.raises(ValueError, match="unknown fgm_solve key"):
            parse_fgm_solve_block({"fgm_solve": {"map_objective": "j_asym"}})


# ---------------------------------------------------------------------------
# 1c. resolve-at-deployment-grid guidance
# ---------------------------------------------------------------------------

class TestPlanGridTransfer:
    """The keyhole map-transfer verdict, wired as guidance.

    HOLDOUT_FOLLOWUP_REPORT.md Section 1 Verdict 2: on the keyhole the grid-120
    class loss was MAP TRANSFER, and a map SOLVED natively at grid 160 returned
    the arm to the solved class (IoU 0.9716 against the transferred map's
    0.9447). Section 3.1: the design filter width is a PHYSICAL length of
    0.75 mm, and holding the CELL count instead would let a finer solve buy
    fidelity with finer features.
    """

    def test_matching_grids_are_a_no_op(self):
        from scripts.solve_fgm import plan_grid_transfer
        p = plan_grid_transfer(map_grid=120, config_grid=120,
                               resolve_native=False, filter_radius_mm=1.0)
        assert p["mismatch"] is False
        assert p["action"] == "use_map_at_its_native_grid"
        assert p["warm_start_allowed"] is True
        assert p["warning"] is None

    def test_a_mismatch_without_the_flag_warns_loudly_and_starts_cold(self):
        from scripts.solve_fgm import plan_grid_transfer
        p = plan_grid_transfer(map_grid=120, config_grid=160,
                               resolve_native=False, filter_radius_mm=1.0)
        assert p["mismatch"] is True
        assert p["action"] == "cold_start_with_warning"
        assert p["warm_start_allowed"] is False
        assert "120" in p["warning"] and "160" in p["warning"]
        assert "--resolve-native" in p["warning"]

    def test_the_flag_re_solves_at_the_requested_grid_from_the_resampled_map(self):
        from scripts.solve_fgm import plan_grid_transfer
        p = plan_grid_transfer(map_grid=120, config_grid=160,
                               resolve_native=True, filter_radius_mm=1.0)
        assert p["mismatch"] is True
        assert p["action"] == "resolve_native"
        assert p["warm_start_allowed"] is True
        assert p["warning"] is None
        assert "start" in p["note"].lower()

    def test_the_filter_radius_is_recorded_as_held_in_millimetres(self):
        from scripts.solve_fgm import plan_grid_transfer
        for native in (True, False):
            p = plan_grid_transfer(map_grid=120, config_grid=160,
                                   resolve_native=native, filter_radius_mm=1.25)
            assert p["filter_radius_mm"] == 1.25
            assert p["filter_radius_held_in"] == "mm"

    def test_stored_map_grid_reads_the_simulation_resolution_map(self, tmp_path,
                                                                 synthetic_case):
        from scripts.solve_fgm import emit_production_npz, stored_map_grid
        x, y, _pm, sat = synthetic_case
        p = emit_production_npz(sat, x, y, tmp_path / "m.npz", bpp=4,
                                run_name="unit")
        # sat_map is at SIMULATION resolution (60 here), level_map at printer
        # resolution; the grid must come from the former.
        assert stored_map_grid(p) == 60

    def test_stored_map_grid_is_none_without_a_sat_map(self, tmp_path):
        import numpy as np
        from scripts.solve_fgm import stored_map_grid
        p = tmp_path / "printer_only.npz"
        np.savez_compressed(p, level_map=np.zeros((720, 720), dtype=np.uint8))
        assert stored_map_grid(p) is None

    def test_stored_map_grid_is_none_for_an_unreadable_file(self, tmp_path):
        from scripts.solve_fgm import stored_map_grid
        p = tmp_path / "not_an_npz.npz"
        p.write_text("this is not an npz")
        assert stored_map_grid(p) is None

    def test_an_unknown_map_grid_is_reported_not_guessed(self):
        from scripts.solve_fgm import plan_grid_transfer
        p = plan_grid_transfer(map_grid=None, config_grid=160,
                               resolve_native=False, filter_radius_mm=1.0)
        assert p["mismatch"] is None
        assert p["action"] == "unknown_map_grid"
        assert p["warm_start_allowed"] is True
        assert "could not be read" in p["warning"]


# ---------------------------------------------------------------------------
# 2. production map emitter: key set + REAL production-loader round trip
# ---------------------------------------------------------------------------

# The exact key set np.savez_compressed writes at fgm_generator.py:679-696.
FGM_GENERATOR_NPZ_KEYS = {
    "level_map", "sat_map", "x_mm", "y_mm", "width_mm", "height_mm",
    "bpp", "n_levels", "magnitude", "baseline_saturation", "dead_band",
    "proxy_field", "invert", "dpi",
}


@pytest.fixture()
def synthetic_case():
    """A small simulation grid with a centered circular part mask."""
    np = pytest.importorskip("numpy")
    n = 60
    dx = 0.5e-3  # 0.5 mm cells: forces a real zoom to 720 dots per inch
    x = np.arange(n) * dx
    y = np.arange(n) * dx
    xx, yy = np.meshgrid(x - x.mean(), y - y.mean())
    part_mask = (xx**2 + yy**2) <= (10e-3) ** 2
    sat = np.where(part_mask, 0.5 + 0.4 * (xx / xx.max()), 1.0)
    sat = np.clip(sat, 0.0, 1.0)
    return x, y, part_mask, sat


class TestEmitProductionNpz:
    def test_key_set_matches_fgm_generator_emitter(self, tmp_path, synthetic_case):
        import numpy as np
        from scripts.solve_fgm import emit_production_npz
        x, y, part_mask, sat = synthetic_case
        p = emit_production_npz(sat, x, y, tmp_path / "m.npz", bpp=4,
                                run_name="unit")
        with np.load(p, allow_pickle=True) as d:
            assert set(d.files) == FGM_GENERATOR_NPZ_KEYS
            assert d["level_map"].dtype == np.uint8
            assert int(d["level_map"].max()) <= 15
            assert int(d["bpp"]) == 4
            assert int(d["n_levels"]) == 16
            # Defaults chosen so the resimulate injection path applies NO
            # rescale (rfam_eqs_coupled.py:383-388 rescales only when
            # magnitude != 1.0 or baseline != 0.5).
            assert float(d["magnitude"]) == 1.0
            assert float(d["baseline_saturation"]) == 0.5

    def test_real_production_loader_reads_emitted_npz(self, tmp_path,
                                                      synthetic_case):
        """Data contract: the REAL injection path must decode the emitted map.

        Uses rfam_eqs_coupled._FgmFeedback.from_config, the exact function the
        engine calls at run start (rfam_eqs_coupled.py:302-398), not a copy.
        """
        import numpy as np
        import rfam_eqs_coupled as rfam
        from fgm_solve_campaign.adjoint2d import printability as pq
        from scripts.solve_fgm import emit_production_npz

        x, y, part_mask, sat = synthetic_case
        s_q = pq.quantize_in_part(sat, part_mask, bpp=4, sat_max=1.0)
        p = emit_production_npz(s_q, x, y, tmp_path / "m.npz", bpp=4,
                                run_name="unit")
        cfg = {"fgm_feedback": {"enabled": True,
                                "saturation_map_npz": str(p)}}
        fb = rfam._FgmFeedback.from_config(cfg, x, y, part_mask)
        assert fb.enabled
        assert fb.sat_map.shape == (len(y), len(x))
        # The loader must recover exactly the printer round trip of the
        # quantized map (printability.load_level_map reproduces the loader,
        # proven in its module docstring; here we check the real one).
        with np.load(p) as d:
            expected = pq.load_level_map(d["level_map"], len(x), len(y), bpp=4)
        assert np.allclose(fb.sat_map, expected, atol=1e-6)
        # And the round trip must stay within one printer level quantum of
        # the delivered map deep inside the part (edges blur under resample).
        from scipy.ndimage import binary_erosion
        core = binary_erosion(part_mask, iterations=3)
        assert float(np.max(np.abs(fb.sat_map[core] - s_q[core]))) <= 1.0 / 15.0


# ---------------------------------------------------------------------------
# 3. production verification pass: config builder (pure logic)
# ---------------------------------------------------------------------------

def _base_cfg() -> dict:
    """A minimal calibrated-config skeleton with the keys the builder touches."""
    return {
        "geometry": {"part": {"shape": "ellipse"}, "grid_nx": 120},
        "electric": {"voltage_v": 3005.5},
        "thermal": {"dt_s": 0.5, "n_steps": 1500, "ambient_c": 23.0},
        "fgm_solve": {"budget_forward_equivalents": 8.0},
    }


class TestBuildProductionVerifyConfig:
    def test_n_steps_set_from_stop_time(self):
        from scripts.solve_fgm import build_production_verify_config
        out = build_production_verify_config(_base_cfg(), "/tmp/m.npz", 270.0)
        # The showcase precedent: stop 270 s at dt 0.5 s ran n_steps 540
        # (outputs_eqs/fgm_solve_showcase/triangle_solved_A1_4bpp_stop270.yaml).
        assert out["thermal"]["n_steps"] == 540

    def test_n_steps_rounds_and_floors_at_one(self):
        from scripts.solve_fgm import build_production_verify_config
        out = build_production_verify_config(_base_cfg(), "/tmp/m.npz", 0.1)
        assert out["thermal"]["n_steps"] == 1

    def test_fgm_feedback_direct_block(self):
        from scripts.solve_fgm import build_production_verify_config
        out = build_production_verify_config(_base_cfg(), "/tmp/m.npz", 270.0)
        fb = out["fgm_feedback"]
        assert fb["enabled"] is True
        assert fb["sat_map_npz_direct"] == "/tmp/m.npz"
        assert fb["sat_max"] == 1.0
        assert fb["iterate"] is False

    def test_fgm_solve_block_is_stripped(self):
        from scripts.solve_fgm import build_production_verify_config
        out = build_production_verify_config(_base_cfg(), "/tmp/m.npz", 270.0)
        assert "fgm_solve" not in out

    def test_input_config_is_not_mutated(self):
        from scripts.solve_fgm import build_production_verify_config
        cfg = _base_cfg()
        import copy
        before = copy.deepcopy(cfg)
        build_production_verify_config(cfg, "/tmp/m.npz", 270.0)
        assert cfg == before

    def test_missing_dt_raises(self):
        from scripts.solve_fgm import build_production_verify_config
        cfg = _base_cfg()
        del cfg["thermal"]["dt_s"]
        with pytest.raises(ValueError, match="dt_s"):
            build_production_verify_config(cfg, "/tmp/m.npz", 270.0)

    def test_other_sections_carried_verbatim(self):
        from scripts.solve_fgm import build_production_verify_config
        cfg = _base_cfg()
        out = build_production_verify_config(cfg, "/tmp/m.npz", 270.0)
        assert out["electric"] == cfg["electric"]
        assert out["geometry"] == cfg["geometry"]


# ---------------------------------------------------------------------------
# 4. FGM map figure artifacts (the fgm_generator preview convention)
# ---------------------------------------------------------------------------

class TestEmitMapPngs:
    def test_preview_and_meteor_pngs(self, tmp_path, synthetic_case):
        """Match the fgm_generator.py PNG convention (lines 702-744):

        preview: white = max ink, flipped so physical top is image top;
        meteor import: exact pixel inversion of the preview.
        """
        import numpy as np
        from PIL import Image
        from scripts.solve_fgm import emit_map_pngs, emit_production_npz

        x, y, part_mask, sat = synthetic_case
        p = emit_production_npz(sat, x, y, tmp_path / "m.npz", bpp=4,
                                run_name="unit")
        paths = emit_map_pngs(p)
        prev = Path(paths["png_path"])
        met = Path(paths["meteor_png_path"])
        assert prev.name == "m_preview.png" and prev.exists()
        assert met.name == "m_meteor_import.png" and met.exists()

        with np.load(p) as d:
            lm = d["level_map"]
        a_prev = np.asarray(Image.open(prev))
        a_met = np.asarray(Image.open(met))
        assert a_prev.shape == lm.shape
        expected = np.flipud(lm.astype(np.float32) * (255.0 / 15.0)).astype(np.uint8)
        assert np.array_equal(a_prev, expected)
        assert np.array_equal(a_met, 255 - a_prev)


# ---------------------------------------------------------------------------
# 5. the standard-suite inventory constant (used by the post-run check)
# ---------------------------------------------------------------------------

STORED_V20X_RUN = (
    Path(__file__).parent / "outputs_eqs" / "runs" / "ellipse" / "fgm_solve"
    / "ellipse_fragment_smoke_20260802")


@pytest.mark.slow
def test_j_phi_legacy_path_reproduces_a_stored_v2_0_x_run_stop_exactly():
    """The flag-off identity claim, proven against a REAL stored v2.0.1 run.

    Takes the delivered 4-bits-per-pixel map of
    ``outputs_eqs/runs/ellipse/fgm_solve/ellipse_fragment_smoke_20260802``
    (its ``production_verify.engine_version`` reads 2.0.1), re-runs the same
    forward with the current code, and reads it under ``stop_rule = "j_phi"``.
    The stop index, the stop time and the intersection over union must
    reproduce the stored numbers EXACTLY. The melt-region objective is asserted
    to 1e-12 relative rather than bitwise, and the reason is measured rather
    than assumed: ``solve_maps.npz`` archives the delivered map in float32,
    which perturbs it by up to 2.8e-08 in saturation. Feeding the archived
    float32 map straight in reproduces J to 7.4e-10 relative; restoring the
    exact 4-bits-per-pixel levels (k / 15, which is what the run itself
    marched) reproduces it to 1.1e-13 relative, that is to the accumulation
    noise of a 1500-step double-precision march. The archive's storage
    precision is the whole of the difference.

    Marked slow: one full grid-120 forward march.
    """
    import json

    import numpy as np

    from scripts.solve_fgm import _ensure_import_paths
    _ensure_import_paths()
    from adjoint2d import chi_area, forward as fwd, library_solve as lib
    from adjoint2d import stop_rule as srule, topopt_objective as tobj
    from adjoint2d.pins import build_case, load_cfg

    stored = json.loads((STORED_V20X_RUN / "results.json").read_text())
    assert stored["production_verify"]["engine_version"] == "2.0.1"
    arm = stored["arms"]["SOLVE_4bpp"]

    cfg = load_cfg(Path(stored["config"]))
    case = build_case(cfg)
    chi, _info = chi_area.chi_from_cfg(cfg, case.x, case.y)
    with np.load(STORED_V20X_RUN / "solve_maps.npz") as d:
        s32 = np.asarray(d["SOLVE_4bpp"], dtype=float)
    # restore the exact 16 printable levels the run actually marched
    s_q = np.where(case.part_mask, np.round(s32 * 15.0) / 15.0, 1.0)
    assert float(np.max(np.abs(s32 - s_q))) < 1e-7  # float32 storage only

    tr = fwd.forward(case, s_q, keep_checkpoints=True, stop_after_phi=None,
                     shape_stop_patience=lib.PATIENCE, eps_covary=False)
    rec = srule.dual_stop(tr, case, chi, rule="j_phi")
    iou = float(tobj.metrics(tr.T_at_end(int(rec["index"])), case, chi)["IoU"])

    assert int(rec["index"]) == int(arm["t_stop_index"])
    assert float(rec["time_s"]) == float(arm["t_stop_s"])
    assert iou == float(arm["IoU"])
    assert abs(float(rec["J_phi_at_stop"]) - float(arm["J"])) \
        <= 1e-12 * abs(float(arm["J"]))

    # and the SAME trajectory carries the v2.1.0 stop, later, as designed
    assert rec["j_asym_stop"]["index"] >= rec["j_phi_stop"]["index"]


def test_production_suite_files_match_showcase_inventory():
    """The file set the verification pass asserts is the showcase set
    (outputs_eqs/fgm_solve_showcase/triangle_A1_4bpp_stop270/, a real
    v2.0.0 engine run of a solved map)."""
    from scripts.solve_fgm import PRODUCTION_SUITE_FILES
    assert PRODUCTION_SUITE_FILES == {
        "electric_fields.png", "thermal_fields_final.png", "rf_summary_v5.png",
        "paper_style_report.png", "validation_report.png", "time_series.png",
        "time_series.json", "fields.npz", "summary.json", "used_config.yaml",
        "density_evolution.gif", "electric_field_evolution.gif",
        "thermal_evolution.gif", "report_manifest.json",
    }
