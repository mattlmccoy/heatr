"""Red-first tests for the SEQUENTIAL dwell march and its two gradients.

The march is the part-frame execution vehicle of `dwell_march.py`, generalized
so that a control step may STRADDLE a switch: the heating injected at that step
is the exact time average over the step. Two degeneracy gates pin it to code
that is already finite-difference gated:

  * one segment for the whole horizon must reproduce the single-position march
    BIT FOR BIT, and its map gradient must reproduce `DwellKernel`'s at one-hot
    weights;
  * durations that are exact multiples of the control step must reproduce
    `dwell_march.program_forward` on the same position sequence BIT FOR BIT.

Then the new object, dJ/d(durations), against a central difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from adjoint2d import dwell_march as dm                     # noqa: E402
from adjoint2d import gradops, library_solve as lib         # noqa: E402
from adjoint2d import seq_dwell as sq                       # noqa: E402
from adjoint2d import seq_dwell_march as sqm                # noqa: E402
from adjoint2d import shape_objective as so                 # noqa: E402
from adjoint2d.dwell import TurntableProgram                # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel              # noqa: E402
from adjoint2d.pins import load_cfg                         # noqa: E402

ANGLES = np.array([0.0, 90.0])
N = 40
DT = 0.5


@pytest.fixture(scope="module")
def kern():
    cfg = load_cfg(lib.shape_config("L_shape"))
    return DwellKernel.build(cfg, angles=ANGLES)


@pytest.fixture(scope="module")
def smap(kern):
    rng = np.random.default_rng(3)
    pm = kern.case0.part_mask
    return np.where(pm, 0.4 + 0.4 * rng.random(pm.shape), 1.0)


def test_one_segment_reproduces_the_single_position_march_bit_for_bit(kern, smap):
    kern.set_weights(np.array([0.0, 1.0]))
    tr_ref = kern.forward(smap, keep_checkpoints=True, n_steps=N)
    tr = sqm.sequential_forward(kern, [1], [N * DT], N, keep_checkpoints=True)
    assert np.array_equal(tr.T_final, tr_ref.T_final)
    assert np.array_equal(tr.rho_final, tr_ref.rho_final)
    assert np.array_equal(tr.phi_final, tr_ref.phi_final)


def test_one_segment_map_gradient_matches_the_dwell_kernel_at_one_hot_weights(kern, smap):
    ops = gradops.gradient_matrices(kern.case0.x, kern.case0.y)
    kern.set_weights(np.array([0.0, 1.0]))
    tr_ref = kern.forward(smap, keep_checkpoints=True, n_steps=N)
    _J, seed = so.shape_J_and_seed(tr_ref.T_at_end(N - 1), kern.case0)
    g_ref, _gw = kern.both_gradients(smap, tr_ref, {N - 1: seed}, grad_ops=ops)

    tr = sqm.sequential_forward(kern, [1], [N * DT], N, keep_checkpoints=True)
    g_s, g_d = sqm.sequential_gradients(kern, smap, tr, {N - 1: seed}, grad_ops=ops)
    rel = np.max(np.abs(g_s - g_ref)) / max(np.max(np.abs(g_ref)), 1e-30)
    assert rel < 1e-13
    assert g_d.shape == (1,)
    assert g_d[0] == pytest.approx(0.0, abs=1e-30)


def test_step_aligned_durations_reproduce_program_forward_bit_for_bit(kern, smap):
    kern.set_weights(np.array([0.5, 0.5]))
    kern.averaged_Q(smap)
    prog = TurntableProgram(
        moves=({"position_deg": 0.0, "dwell_s": 6.0, "move_at_s": 0.0},
               {"position_deg": 90.0, "dwell_s": 14.0, "move_at_s": 6.0}),
        kept_positions_deg=(0.0, 90.0), steps_per_position=(12, 28),
        realized_weights=np.array([0.3, 0.7]),
        requested_weights=np.array([0.3, 0.7]),
        cycle_time_s=20.0, total_s=20.0, dt_s=DT, n_cycles=1.0)
    idx = dm.program_step_positions(prog, ANGLES, DT, N)
    tr_ref = dm.program_forward(kern, idx, N)
    tr = sqm.sequential_forward(kern, [0, 1], [6.0, 14.0], N)
    assert np.array_equal(tr.T_final, tr_ref.T_final)
    assert np.array_equal(tr.rho_final, tr_ref.rho_final)


def test_duration_gradient_against_a_central_difference(kern, smap):
    ops = gradops.gradient_matrices(kern.case0.x, kern.case0.y)
    kern.set_weights(np.array([0.5, 0.5]))
    kern.averaged_Q(smap)
    seg = [0, 1, 0]
    d0 = np.array([5.3, 6.15, 8.55])

    def J_of(d):
        kern.averaged_Q(smap)
        t = sqm.sequential_forward(kern, seg, d, N)
        return so.shape_J_and_seed(t.T_at_end(N - 1), kern.case0)[0]

    kern.averaged_Q(smap)
    tr = sqm.sequential_forward(kern, seg, d0, N, keep_checkpoints=True)
    _J, seed = so.shape_J_and_seed(tr.T_at_end(N - 1), kern.case0)
    _gs, g_d = sqm.sequential_gradients(kern, smap, tr, {N - 1: seed}, grad_ops=ops)

    for k in (0, 1):
        best = np.inf
        for e in (1e-3, 1e-4, 1e-5, 1e-6):
            dp, dmn = d0.copy(), d0.copy()
            dp[k] += e
            dmn[k] -= e
            fd = (J_of(dp) - J_of(dmn)) / (2 * e)
            best = min(best, abs(fd - g_d[k]) / max(abs(g_d[k]), 1e-30))
        assert best < 1e-6, f"segment {k}: best relative error {best:.3e}"


def test_sequential_gradients_refuse_a_stale_map(kern, smap):
    kern.set_weights(np.array([0.5, 0.5]))
    kern.averaged_Q(smap)
    tr = sqm.sequential_forward(kern, [0, 1], [5.0, 15.0], N, keep_checkpoints=True)
    _J, seed = so.shape_J_and_seed(tr.T_at_end(N - 1), kern.case0)
    with pytest.raises(RuntimeError):
        sqm.sequential_gradients(kern, smap * 0.5, tr, {N - 1: seed})


def test_sequential_forward_needs_the_per_position_fields(kern, smap):
    k2 = DwellKernel.build(load_cfg(lib.shape_config("L_shape")), angles=ANGLES)
    with pytest.raises(RuntimeError):
        sqm.sequential_forward(k2, [0], [20.0], N)


def test_mix_matrix_is_carried_on_the_trajectory(kern, smap):
    kern.set_weights(np.array([0.5, 0.5]))
    kern.averaged_Q(smap)
    tr = sqm.sequential_forward(kern, [0, 1], [6.0, 14.0], N)
    assert tr.seq_mix.shape == (N, 2)
    assert np.allclose(tr.seq_mix.sum(axis=1), 1.0)
    assert np.allclose(tr.seq_segment_fractions,
                       sq.step_mix([6.0, 14.0], DT, N).sum(axis=0) / N)
    # the per-ANGLE realized fractions keep `dwell_march`'s meaning
    assert tr.realized_weights_executed.shape == (2,)
    assert float(np.sum(tr.realized_weights_executed)) == pytest.approx(1.0)
