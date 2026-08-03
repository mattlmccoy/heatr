"""The temporal power scale p(t) inside the differentiable forward march.

Three contracts, written RED first:

1. p identically 1 reproduces the no-schedule march BIT FOR BIT, so the whole
   library campaign's numbers are unaffected by the new argument existing.
2. A constant p = c is EXACTLY the drive voltage V -> V*sqrt(c). This is the
   injection convention and it is verified against a real re-solve at the
   scaled voltage rather than asserted.
3. The `max_qrf` cap acts AFTER the scaling, which is where it would act on a
   real re-solve.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from adjoint2d import forward as fwd
from adjoint2d.library_solve import shape_config
from adjoint2d.pins import build_case, load_cfg


def _case(shape: str = "star"):
    return build_case(load_cfg(shape_config(shape)))


def test_unit_schedule_is_bit_identical_to_no_schedule():
    case = _case()
    s = np.ones(case.part_mask.shape)
    a = fwd.forward(case, s, stop_after_phi=None, n_steps=40)
    b = fwd.forward(case, s, stop_after_phi=None, n_steps=40,
                    p_seg=np.ones(8), n_seg=8, p_horizon=40)
    assert np.array_equal(a.T_final, b.T_final)
    assert np.array_equal(a.rho_final, b.rho_final)
    assert np.array_equal(a.energy_in_J_per_m, b.energy_in_J_per_m)


def test_constant_power_scale_equals_a_sqrt_voltage_change():
    case = _case()
    c = 0.64                       # sqrt(c) = 0.8, an exactly representable ratio
    s = np.ones(case.part_mask.shape)
    scaled = dataclasses.replace(case)
    scaled.pins = dataclasses.replace(case.pins, v_hi=case.pins.v_hi * np.sqrt(c))

    a = fwd.forward(case, s, stop_after_phi=None, n_steps=40,
                    p_seg=np.full(4, c), n_seg=4, p_horizon=40)
    b = fwd.forward(scaled, s, stop_after_phi=None, n_steps=40)
    assert a.T_final == pytest.approx(b.T_final, rel=1e-11, abs=1e-11)
    assert float(a.energy_in_J_per_m[-1]) == pytest.approx(
        float(b.energy_in_J_per_m[-1]), rel=1e-11)


def test_power_scale_multiplies_the_absorbed_power_linearly():
    case = _case()
    s = np.ones(case.part_mask.shape)
    a = fwd.forward(case, s, stop_after_phi=None, n_steps=10)
    half = fwd.forward(case, s, stop_after_phi=None, n_steps=10,
                       p_seg=np.full(2, 0.5), n_seg=2, p_horizon=10)
    assert float(half.energy_in_J_per_m[0]) == pytest.approx(
        0.5 * float(a.energy_in_J_per_m[0]), rel=1e-12)


def test_zero_power_segment_injects_no_dose_and_lets_the_part_cool():
    case = _case()
    s = np.ones(case.part_mask.shape)
    p = np.array([1.0, 0.0])
    tr = fwd.forward(case, s, stop_after_phi=None, n_steps=40, p_seg=p, n_seg=2,
                     p_horizon=40)
    e = tr.energy_in_J_per_m
    # dose stops accumulating in the OFF segment
    assert float(e[-1]) == pytest.approx(float(e[19]), rel=1e-12)
    # and the part is cooler at the end of the OFF segment than at its start
    assert float(np.mean(tr.T_final[case.part_mask])) < float(
        np.mean(tr.ckpt_T[20][case.part_mask]))


def test_cap_is_applied_after_the_scaling():
    case = _case()
    s = np.ones(case.part_mask.shape)
    # Pin the cap just above the unscaled peak so that p = 2 must clip.
    peak = float(np.max(fwd.solve_electric(
        case, fwd.sigma_state_a(case, s, False), fwd.eps_field(case, s)).Qrf_raw))
    capped = dataclasses.replace(case)
    capped.pins = dataclasses.replace(case.pins, max_qrf=peak)
    tr = fwd.forward(capped, s, stop_after_phi=None, n_steps=4,
                     p_seg=np.full(1, 2.0), n_seg=1, p_horizon=4)
    unscaled = fwd.forward(capped, s, stop_after_phi=None, n_steps=4)
    # If the cap acted BEFORE the scale, the dose would be exactly 2x. It acts
    # after, so it must be strictly less.
    assert float(tr.energy_in_J_per_m[0]) < 2.0 * float(unscaled.energy_in_J_per_m[0])
    assert float(tr.energy_in_J_per_m[0]) > float(unscaled.energy_in_J_per_m[0])
