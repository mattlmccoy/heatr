"""RED-first tests for reading the relative-density state at an outer-step end.

The melt-region objective reads temperature through `Trajectory.T_at_end`. The
density-region objective needs the same accessor for relative density. The
checkpoint convention is identical: `ckpt_rho[n]` is the state at the START of
outer step n, so the state at the END of step n is `ckpt_rho[n + 1]`, and the
last step's end state is `rho_final`.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import forward as fwd


def _traj(n=4):
    ck_T = [np.full((2, 2), float(i)) for i in range(n)]
    ck_rho = [np.full((2, 2), 0.55 + 0.05 * i) for i in range(n)]
    return fwd.Trajectory(
        time_s=np.arange(1, n + 1) * 0.5,
        mean_T_part_c=np.zeros(n), ui_rms_part=np.zeros(n),
        mean_phi_part=np.zeros(n), mean_rho_rel_part=np.zeros(n),
        sigma_T=np.zeros(n), n_outer=n, stopped_early=False,
        T_final=np.full((2, 2), float(n)),
        rho_final=np.full((2, 2), 0.55 + 0.05 * n),
        phi_final=np.zeros((2, 2)),
        P_abs_A=0.0, P_abs_B=0.0, frac_dT_clipped_max=0.0,
        frac_temp_cap_max=0.0, frac_qrf_cap=0.0,
        ckpt_T=ck_T, ckpt_rho=ck_rho,
    )


def test_rho_at_end_of_the_last_step_is_the_final_state():
    tr = _traj(4)
    assert np.allclose(tr.rho_at_end(3), 0.55 + 0.05 * 4)


def test_rho_at_end_of_an_interior_step_is_the_next_checkpoint():
    tr = _traj(4)
    assert np.allclose(tr.rho_at_end(1), 0.55 + 0.05 * 2)


def test_rho_at_end_raises_without_checkpoints():
    tr = _traj(4)
    tr.ckpt_rho = []
    with pytest.raises(RuntimeError):
        tr.rho_at_end(1)


def test_rho_at_end_matches_the_temperature_accessor_convention():
    tr = _traj(5)
    for n in range(5):
        # both accessors must index the same step end
        assert tr.T_at_end(n)[0, 0] == pytest.approx(n + 1)
        assert tr.rho_at_end(n)[0, 0] == pytest.approx(0.55 + 0.05 * (n + 1))
