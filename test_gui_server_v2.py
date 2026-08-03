#!/usr/bin/env python3
"""RED/GREEN tests for the HEATR GUI server v2 promotion pass.

Contract:
  1. _summary_excerpt passes through engine_version (absent = pre-v2 handled
     by the front end), the dual read-state sigma_T keys, and an
     energy_err_pct derived from the summary's residual/dose (works for
     pre-v2 summaries too, which carry the raw J-per-m keys).
  2. _configure_turntable accepts a dwell program: payload key
     turntable_program_json becomes turntable.program_json with
     corotate_dopant and corotate_eps_geometry defaulting ON in program mode
     (dielectric-ghost decision, ENGINE_DWELL_SUPPORT_NOTES.md section 9).
     Legacy fixed-step payloads produce the exact pre-v2 config block.

Run: ./.venv312/bin/python -m pytest test_gui_server_v2.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import rfam_gui_server as srv  # noqa: E402


def test_summary_excerpt_passes_engine_version_and_dual_read_state():
    summary = {
        "engine_version": "2.0.0",
        "sigma_T_heating_peak_c": 11.05,
        "sigma_T_melt_onset_c": 4.93,
        "sigma_T_melt_reached": True,
        "max_T_part_final_c": 200.0,
    }
    ex = srv._summary_excerpt(summary)
    assert ex["engine_version"] == "2.0.0"
    assert abs(ex["sigma_T_heating_peak_c"] - 11.05) < 1e-9
    assert abs(ex["sigma_T_melt_onset_c"] - 4.93) < 1e-9
    assert ex["sigma_T_melt_reached"] is True


def test_summary_excerpt_omits_version_when_absent():
    ex = srv._summary_excerpt({"max_T_part_final_c": 100.0})
    assert "engine_version" not in ex  # front end renders "pre-v2"


def test_summary_excerpt_energy_err_pct_v2_key():
    ex = srv._summary_excerpt({"energy_residual_frac_final": 0.0123})
    assert abs(ex["energy_err_pct"] - 1.23) < 1e-9


def test_summary_excerpt_energy_err_pct_from_pre_v2_keys():
    ex = srv._summary_excerpt({
        "energy_balance_residual_final_J_per_m": -50.0,
        "energy_doped_total_J_per_m": 10000.0,
    })
    assert abs(ex["energy_err_pct"] - 0.5) < 1e-9


def test_configure_turntable_program_mode_defaults_corotation_on():
    cfg: dict = {}
    payload = {
        "turntable_program_json":
            "fgm_solve_campaign/out_dwell/cross_turntable_deliverable.json",
    }
    srv._configure_turntable(cfg, payload)
    tt = cfg["turntable"]
    assert tt["enabled"] is True
    assert tt["program_json"].endswith("cross_turntable_deliverable.json")
    assert tt["corotate_dopant"] is True
    assert tt["corotate_eps_geometry"] is True
    assert "rotation_deg" not in tt  # program mode, not fixed-step


def test_configure_turntable_program_mode_flags_can_be_disabled():
    cfg: dict = {}
    payload = {
        "turntable_program_json":
            "fgm_solve_campaign/out_dwell/T_shape_turntable_deliverable.json",
        "turntable_corotate_dopant": False,
        "turntable_corotate_eps": False,
    }
    srv._configure_turntable(cfg, payload)
    tt = cfg["turntable"]
    assert tt["corotate_dopant"] is False
    assert tt["corotate_eps_geometry"] is False


def test_configure_turntable_rejects_missing_program_file():
    cfg: dict = {}
    payload = {"turntable_program_json": "does/not/exist.json"}
    try:
        srv._configure_turntable(cfg, payload)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a missing program file")


def test_configure_turntable_legacy_payload_is_bit_identical():
    cfg: dict = {}
    srv._configure_turntable(cfg, {
        "turntable_rotation_deg": 15.0,
        "turntable_total_rotations": 48,
        "turntable_interval_s": 9.0,
    })
    assert cfg["turntable"] == {
        "enabled": True,
        "rotation_deg": 15.0,
        "total_rotations": 48,
        "rotation_interval_s": 9.0,
    }


if __name__ == "__main__":
    test_summary_excerpt_passes_engine_version_and_dual_read_state()
    print("PASS")
