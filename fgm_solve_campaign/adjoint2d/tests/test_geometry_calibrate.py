"""Red-first tests for the automatic drive calibration of an imported part.

THE CONVENTION BEING AUTOMATED. `FROZEN_CONVENTIONS_2D.md` Section 3: the drive
voltage is the per-shape calibrated `electric.voltage_v` chosen so that the
UNIFORM arm absorbs 500 watts per metre of depth at grid 120. For the eighteen
library shapes that number was found once and stored in
`outputs_eqs/fgm_calibrated_control/configs/<shape>_m*.yaml`. An imported part
has no stored voltage, so the intake has to find it, and it must find the same
number the campaign would have.

WHAT "THE UNIFORM ARM ABSORBS 500 W/m" MEANS, EXACTLY. Measured in this pass on
six stored configurations: at the stored voltage the state-B absorbed power
(the electrical state the march runs in from the first `update_interval` tick
onwards) is 500.00 W/m on every one of them, while the state-A power is 450 to
475 W/m. The convention is therefore state B, summed over the doped mask.

WHY IT COSTS ONE SOLVE. The electro-quasi-static problem is linear in the
applied potential and the conductivity and permittivity fields do not depend on
it, so absorbed power is exactly quadratic in the drive and one solve plus one
closed-form rescale is exact. `robust.recalibrated_voltage` already encodes
that rescale for the grid hold-out; this module reuses it rather than deriving
it a second time. The second solve is a VERIFICATION, not an iteration.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import geometry_calibrate as gcal, geometry_intake as gi
from adjoint2d.library_solve import shape_config
from adjoint2d.pins import load_cfg

TARGET = 500.0


@pytest.mark.parametrize("shape", ["star", "square"])
def test_calibration_reproduces_the_stored_library_voltage(shape):
    cfg = load_cfg(shape_config(shape))
    stored = float(cfg["electric"]["voltage_v"])
    cal = gcal.calibrate_drive(cfg, target_w_per_m=TARGET, start_voltage_v=1000.0)
    rel = abs(cal.voltage_v - stored) / stored
    assert rel < 0.01, (f"{shape}: calibrated {cal.voltage_v:.2f} V against the "
                        f"stored {stored:.2f} V, {rel*100:.3f} percent apart")


def test_calibration_costs_at_most_two_electro_quasi_static_solves():
    cfg = load_cfg(shape_config("square"))
    cal = gcal.calibrate_drive(cfg, target_w_per_m=TARGET, start_voltage_v=800.0)
    assert cal.n_solves <= 2
    cheap = gcal.calibrate_drive(cfg, target_w_per_m=TARGET, verify=False)
    assert cheap.n_solves == 1


def test_the_verification_solve_hits_the_target():
    cfg = load_cfg(shape_config("square"))
    cal = gcal.calibrate_drive(cfg, target_w_per_m=TARGET, start_voltage_v=137.0)
    assert cal.p_verified_w_per_m == pytest.approx(TARGET, rel=1e-9)


def test_absorbed_power_is_exactly_quadratic_in_the_drive():
    """The claim that makes one solve enough, checked rather than asserted."""
    cfg = load_cfg(shape_config("square"))
    p1 = gcal.uniform_absorbed_power(cfg, voltage_v=1000.0)
    p2 = gcal.uniform_absorbed_power(cfg, voltage_v=3000.0)
    assert p2 / p1 == pytest.approx(9.0, rel=1e-9)


def test_calibration_is_independent_of_the_starting_voltage():
    cfg = load_cfg(shape_config("cross"))
    a = gcal.calibrate_drive(cfg, target_w_per_m=TARGET, start_voltage_v=50.0)
    b = gcal.calibrate_drive(cfg, target_w_per_m=TARGET, start_voltage_v=9000.0)
    assert a.voltage_v == pytest.approx(b.voltage_v, rel=1e-9)


def test_calibration_runs_on_an_imported_polygon_and_writes_the_config():
    """The point of the whole module: a geometry with no stored voltage."""
    from adjoint2d.tests import fill_contract as fc

    it = gi.from_polygon(fc.rotated_rect_polygon(0.022, 0.012, 27.0), grid=120)
    assert it.info["voltage_is_calibrated"] is False
    it2 = gcal.calibrate_intake(it, target_w_per_m=TARGET)
    assert it2.info["voltage_is_calibrated"] is True
    p = gcal.uniform_absorbed_power(it2.cfg)
    assert p == pytest.approx(TARGET, rel=1e-6)
    assert float(it2.cfg["electric"]["voltage_v"]) == pytest.approx(
        it2.info["calibration"]["voltage_v"], rel=1e-12)


def test_a_target_that_is_not_positive_is_refused():
    cfg = load_cfg(shape_config("square"))
    with pytest.raises(ValueError, match="positive"):
        gcal.calibrate_drive(cfg, target_w_per_m=0.0)
