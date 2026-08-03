#!/usr/bin/env python3
"""RED/GREEN tests for HEATR 2-D engine versioning (v2.0.0).

Contract:
  1. rfam_eqs_coupled exposes ENGINE_VERSION (semver string) and
     ENGINE_VERSION_NAME (human label) as module constants.
  2. stamped_config_echo(cfg) returns a COPY of cfg carrying engine_version /
     engine_version_name without mutating the input (used for the
     used_config.yaml echo written by save_outputs).
  3. dual_read_state_from_hist(hist) reproduces the standalone extractor
     outputs_eqs/geometry_dual_readstate/dual_readstate.py on the same series
     (sigma_T = ui_rms_part * (mean_T_part_c - 23); heating-peak = max over
     phi_bar < 0.90; melt-onset = first phi_bar >= 0.90).
  4. A real (tiny) run_sim summary carries engine_version, engine_version_name,
     the dual read-state keys, and energy_residual_frac_final.

Run: ./.venv312/bin/python -m pytest test_engine_version.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import re
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import rfam_eqs_coupled  # noqa: E402

CFG = (BASE / "outputs_eqs" / "fgm_dosecheck" / "square_baseline_voltage"
       / "used_config.yaml")


def _load_reference_extractor():
    """Import the campaign's standalone extractor as the oracle (real code,
    not a re-typed copy)."""
    p = BASE / "outputs_eqs" / "geometry_dual_readstate" / "dual_readstate.py"
    spec = importlib.util.spec_from_file_location("dual_readstate_ref", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_engine_version_constants_exist_and_are_semver():
    v = rfam_eqs_coupled.ENGINE_VERSION
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), v
    # v2.1.0: MINOR bump. New modes and read states behind config keys, with
    # the previous behaviour reachable by flag (fgm_solve.stop_rule: j_phi).
    assert v == "2.1.0"
    name = rfam_eqs_coupled.ENGINE_VERSION_NAME
    assert "2.1" in name


def test_the_changelog_carries_a_real_v2_1_0_entry_not_a_planned_one():
    """The version record and the changelog must not disagree.

    A "Planned v2.1.0" heading while ENGINE_VERSION already reads 2.1.0 is the
    exact state this test exists to make impossible.
    """
    text = (BASE / "CHANGELOG_ENGINE.md").read_text()
    assert "## v2.1.0" in text
    assert "Planned v2.1.0" not in text
    # the behaviour change has to be stated where a reader will meet it
    assert "stop_rule" in text
    assert "j_phi" in text


def test_stamped_config_echo_adds_version_without_mutating_input():
    cfg = {"geometry": {"grid_nx": 40}}
    before = copy.deepcopy(cfg)
    echo = rfam_eqs_coupled.stamped_config_echo(cfg)
    assert echo["engine_version"] == rfam_eqs_coupled.ENGINE_VERSION
    assert echo["engine_version_name"] == rfam_eqs_coupled.ENGINE_VERSION_NAME
    assert echo["geometry"] == {"grid_nx": 40}
    assert cfg == before, "input config must not be mutated"


def test_dual_read_state_matches_the_standalone_extractor():
    ref = _load_reference_extractor()
    # Synthetic series: heats, peaks, then melts at index 4.
    hist = {
        "ui_rms_part": [0.10, 0.30, 0.25, 0.20, 0.15, 0.10],
        "mean_T_part_c": [40.0, 120.0, 150.0, 170.0, 180.0, 182.0],
        "mean_phi_part": [0.0, 0.10, 0.50, 0.80, 0.95, 1.00],
    }
    got = rfam_eqs_coupled.dual_read_state_from_hist(hist)
    want = ref.extract_read_states({
        "ui_rms_part": hist["ui_rms_part"],
        "mean_T_part_c": hist["mean_T_part_c"],
        "mean_phi_part": hist["mean_phi_part"],
    })
    assert abs(got["sigma_T_heating_peak_c"] - want["heating_peak_sigma_T"]) < 1e-12
    assert got["sigma_T_heating_peak_idx"] == want["heating_peak_idx"]
    assert abs(got["sigma_T_melt_onset_c"] - want["melt_onset_sigma_T"]) < 1e-12
    assert got["sigma_T_melt_onset_idx"] == want["melt_onset_idx"]
    assert got["sigma_T_melt_reached"] is True


def test_dual_read_state_handles_never_melting():
    hist = {
        "ui_rms_part": [0.1, 0.2],
        "mean_T_part_c": [40.0, 60.0],
        "mean_phi_part": [0.0, 0.1],
    }
    got = rfam_eqs_coupled.dual_read_state_from_hist(hist)
    assert got["sigma_T_melt_reached"] is False
    assert got["sigma_T_melt_onset_c"] is None
    assert got["sigma_T_heating_peak_c"] is not None


def _tiny_cfg() -> dict:
    cfg = copy.deepcopy(yaml.safe_load(CFG.read_text()))
    cfg["geometry"]["grid_nx"] = 40
    cfg["geometry"]["grid_ny"] = 40
    cfg["thermal"]["n_steps"] = 12
    cfg["electric"]["solver_steps"] = 300
    return cfg


def test_run_sim_summary_is_version_stamped():
    _state, summary, _hist, *_rest = rfam_eqs_coupled.run_sim(_tiny_cfg())
    assert summary["engine_version"] == rfam_eqs_coupled.ENGINE_VERSION
    assert summary["engine_version_name"] == rfam_eqs_coupled.ENGINE_VERSION_NAME
    # Dual read-state keys are present (values may be None pre-melt).
    for k in ("sigma_T_heating_peak_c", "sigma_T_melt_onset_c",
              "sigma_T_melt_reached"):
        assert k in summary, f"summary missing {k}"
    assert "energy_residual_frac_final" in summary


if __name__ == "__main__":
    test_engine_version_constants_exist_and_are_semver()
    print("PASS")
