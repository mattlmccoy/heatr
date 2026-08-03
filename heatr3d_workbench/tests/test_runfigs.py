"""Per-run figure set (adapted from demo_pyramid_fgm, Matt 2026-08-03).

TDD covers the testable logic: read-state selection, baseline grid matching,
melt-height measurement, the annotation line (no headline titles), and the
figure-set selection per run kind. Renders themselves are verified by viewing.
"""
from __future__ import annotations

import numpy as np
import pytest

from heatr3d_workbench import runfigs as RF


# ---- read-state selection --------------------------------------------------

def test_read_state_prefers_last_presaturation_snapshot():
    snaps = [{"i": 0, "phi_bar": 0.10, "t_s": 100.0},
             {"i": 1, "phi_bar": 0.80, "t_s": 500.0},
             {"i": 2, "phi_bar": 0.97, "t_s": 900.0}]
    kind, idx = RF.choose_read_state(snaps)
    assert kind == "snapshot" and idx == 1        # last with phi_bar <= 0.95


def test_read_state_falls_back_to_final():
    assert RF.choose_read_state(None) == ("final", None)
    assert RF.choose_read_state([]) == ("final", None)
    # all saturated -> final is the honest read, not a fake early state
    snaps = [{"i": 0, "phi_bar": 0.99, "t_s": 900.0}]
    assert RF.choose_read_state(snaps) == ("final", None)


# ---- baseline grid matching ------------------------------------------------

def test_baseline_grid_match_pass_and_fail():
    ok, msg = RF.check_baseline_grid((32, 32, 32), 1.875, (32, 32, 32), 1.875)
    assert ok
    ok, msg = RF.check_baseline_grid((32, 32, 32), 1.875, (48, 48, 48), 1.25)
    assert not ok and "grid" in msg
    ok, msg = RF.check_baseline_grid((32, 32, 32), 1.875, (32, 32, 32), 1.0)
    assert not ok and "spacing" in msg


# ---- melt height -----------------------------------------------------------

def test_melt_height_mm():
    part = np.zeros((8, 8, 10), bool); part[2:6, 2:6, 1:9] = True
    phi = np.zeros(part.shape); phi[2:6, 2:6, 1:5] = 1.0   # melts 4 layers up
    h_mm = 2.0
    # melted top layer k=4, part base k=1 -> (4 - 1 + 1) * 2 mm = 8 mm
    assert RF.melt_height_mm(phi, part, h_mm) == pytest.approx(8.0)
    assert RF.melt_height_mm(np.zeros(part.shape), part, h_mm) == 0.0


# ---- annotation line (working outputs: no headline titles) -----------------

def test_annotation_line_contents():
    line = RF.annotation_line({"grid_n": 32, "phase_update": "enthalpy",
                               "heatr3d_engine_version": "heatr3d-4d51c55f",
                               "fgm": "melt", "densify": True})
    for token in ("heatr3d-4d51c55f", "n=32", "enthalpy", "fgm=melt", "densify"):
        assert token in line
    assert "\n" not in line


# ---- figure-set selection per run kind ------------------------------------

def test_figure_plan_per_run_kind():
    fgm_dens = RF.figure_plan({"fgm": "melt", "densify": True,
                               "baseline_run_id": "abc123"},
                              has_sat=True, has_rho=True)
    assert {"dopant_cutaway", "layer_stack", "density_temperature",
            "vs_baseline", "what_changed"} <= set(fgm_dens)
    baseline_only = RF.figure_plan({"fgm": "none", "densify": False},
                                   has_sat=False, has_rho=False)
    assert "dopant_cutaway" not in baseline_only
    assert "density_temperature" not in baseline_only
    assert "vs_baseline" not in baseline_only
