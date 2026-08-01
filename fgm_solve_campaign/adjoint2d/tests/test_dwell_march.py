"""Red-first tests for the TIME-RESOLVED dwell program march.

The dwell solve is done against the quasi-static weighted average. This module
is the verification layer: it executes the actual cycle program step by step in
the PART frame, switching the heating between the stored per-position fields at
the programmed times. Because the part never moves in this frame there is no
interpolation anywhere, so the only difference from the quasi-static forward is
the one thing being tested, the finite cycle time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
for p in (str(HERE.parents[2]), str(HERE.parents[3])):
    if p not in sys.path:
        sys.path.insert(0, p)

from adjoint2d import dwell, library_solve as lib           # noqa: E402
from adjoint2d import dwell_march as dm                     # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel              # noqa: E402
from adjoint2d.pins import load_cfg                         # noqa: E402
from adjoint2d.rot_kernel import AveragedKernel             # noqa: E402

ANG4 = (0.0, 90.0, 180.0, 270.0)


# ---------------------------------------------------------------------------
# 1. the step schedule, pure logic
# ---------------------------------------------------------------------------

def test_step_positions_has_one_entry_per_outer_step():
    prog = dwell.cycle_program([0.5, 0.25, 0.25, 0.0], ANG4, cycle_time_s=8.0,
                               total_s=50.0, dt_s=0.5)
    idx = dm.program_step_positions(prog, ANG4, dt_s=0.5, n_steps=100)
    assert idx.shape == (100,)
    assert set(np.unique(idx)) <= {0, 1, 2}


def test_step_positions_spends_time_in_proportion_to_the_realized_weights():
    prog = dwell.cycle_program([0.5, 0.25, 0.25], (0.0, 90.0, 180.0),
                               cycle_time_s=8.0, total_s=400.0, dt_s=0.5)
    idx = dm.program_step_positions(prog, (0.0, 90.0, 180.0), dt_s=0.5, n_steps=800)
    got = np.array([np.mean(idx == k) for k in range(3)])
    assert np.allclose(got, prog.realized_weights, atol=0.02)


def test_step_positions_starts_at_the_first_commanded_position():
    prog = dwell.cycle_program([0.0, 1.0], (0.0, 45.0), cycle_time_s=8.0,
                               total_s=50.0, dt_s=0.5)
    idx = dm.program_step_positions(prog, (0.0, 45.0), dt_s=0.5, n_steps=20)
    assert np.all(idx == 1)


def test_step_positions_holds_the_last_position_past_the_program_end():
    """A march longer than the program keeps the last commanded position."""
    prog = dwell.cycle_program([0.5, 0.5], (0.0, 90.0), cycle_time_s=8.0,
                               total_s=16.0, dt_s=0.5)
    idx = dm.program_step_positions(prog, (0.0, 90.0), dt_s=0.5, n_steps=40)
    assert np.all(idx[32:] == idx[31])


# ---------------------------------------------------------------------------
# 2. the march
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cross_cfg():
    return load_cfg(lib.shape_config("cross"))


def test_a_program_that_never_moves_reproduces_the_single_angle_forward(cross_cfg):
    """DEGENERACY GATE against the already-gated kernel, bit for bit."""
    kern = DwellKernel.build(cross_cfg, angles=np.array([0.0, 45.0]))
    single = AveragedKernel.build(cross_cfg, angles=np.array([45.0]))
    case = kern.case0
    s = np.where(case.part_mask, 0.75, 1.0)

    ref = single.forward(s, n_steps=30)
    kern.set_weights(np.array([0.0, 1.0]))
    kern.averaged_Q(s)
    got = dm.program_forward(kern, np.ones(30, dtype=int), n_steps=30)
    assert np.array_equal(ref.T_final, got.T_final)
    assert np.array_equal(ref.rho_final, got.rho_final)


def test_the_march_at_a_one_step_cycle_is_close_to_the_quasi_static_average(cross_cfg):
    """CHARACTERIZATION, with the measured gap pinned.

    The quasi-static weighted average is the limit of the program march as the
    cycle time goes to zero. At the fastest cycle the 0.5 s control step allows
    (one step per position) the two must already agree closely. The bound below
    is the MEASURED gap plus headroom, not a derived tolerance.
    """
    ang = np.array(ANG4)
    kern = DwellKernel.build(cross_cfg, angles=ang)
    case = kern.case0
    s = np.where(case.part_mask, 0.85, 1.0)
    kern.set_weights(np.full(4, 0.25))
    qs = kern.forward(s, n_steps=200)
    idx = np.arange(200) % 4
    tr = dm.program_forward(kern, idx, n_steps=200)
    rel = abs(float(np.mean(tr.T_final[case.part_mask]))
              - float(np.mean(qs.T_final[case.part_mask]))) / float(
        np.mean(qs.T_final[case.part_mask]))
    assert rel < 1e-3


def test_a_slow_cycle_departs_from_the_quasi_static_average_more_than_a_fast_one(cross_cfg):
    """The approximation error must grow with the cycle time, or it is not the
    quasi-static error being measured."""
    ang = np.array(ANG4)
    kern = DwellKernel.build(cross_cfg, angles=ang)
    case = kern.case0
    pm = case.part_mask
    s = np.where(pm, 0.85, 1.0)
    kern.set_weights(np.full(4, 0.25))
    qs = kern.forward(s, n_steps=200)
    ref = qs.T_final[pm]

    def gap(steps_per_pos):
        idx = (np.arange(200) // steps_per_pos) % 4
        tr = dm.program_forward(kern, idx, n_steps=200)
        return float(np.max(np.abs(tr.T_final[pm] - ref)))

    assert gap(1) < gap(10) < gap(50)


def test_program_forward_rejects_a_kernel_whose_forward_has_not_been_run(cross_cfg):
    kern = DwellKernel.build(cross_cfg, angles=np.array(ANG4))
    with pytest.raises(RuntimeError):
        dm.program_forward(kern, np.zeros(10, dtype=int), n_steps=10)
