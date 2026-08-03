"""Phase B Task 2: the transient adjoint through the FULL coupled forward.

RUNS IN THE SPIKE ENV:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_adjoint_transient.py

Layer B2 of the pre-registered bisect: B1 (steady EQS) must already be green.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import gate_fd


@pytest.fixture(scope="module")
def tcase():
    from solve3d import transient_gate
    return transient_gate.build_case()


def test_recording_flag_off_is_bit_identical():
    """Protocol checklist item 10. The Phase B recording hook must not perturb
    the Phase A forward by one bit when it is off."""
    import numpy as np

    from solve3d import forward as fwd

    p = fwd.ForwardParams(dt_s=0.5, conv_h=5.0)
    msh = fwd.box_mesh(8)
    half = 0.010

    def in_part(mp):
        return (np.abs(mp[0]) <= half) & (np.abs(mp[1]) <= half)

    kw = dict(in_part=in_part, q_uniform=2.0e6, max_time_s=20.0,
              phi_target=2.0)
    a = fwd.march_enthalpy(msh, p, **kw)
    rec: dict = {}
    b = fwd.march_enthalpy(msh, p, record=rec, **kw)
    assert float(np.max(np.abs(a["T"] - b["T"]))) == 0.0
    assert a["energy_residual_frac"] == b["energy_residual_frac"]
    assert len(rec["T_steps"]) == 40
    # the recorded step-0 state IS the initial condition
    assert float(np.max(np.abs(rec["T_steps"][0] - a["T0"]))) == 0.0


def test_cell_average_transpose_is_exact(tcase):
    """Protocol checklist item 8, redirected to the linear operator Phase B
    actually introduces: the P1 cell-average feeding the sigma(T) coupling and
    its scatter transpose. Threshold 1e-10 (the 2-D filter transpose measured
    4.19e-16)."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(tcase.vol_nodal.size)
    y = rng.standard_normal(tcase.ncells)
    out = gate_fd.transpose_residual(tcase.cell_avg, tcase.cell_avg_T, x, y)
    assert out["pass"], out


def test_recorded_trajectory_reproduces_the_phase_a_drive(tcase):
    """The adjoint's forward (LU EQS path) must produce the same drive as the
    Phase A production path -- assembly consistency, one layer up."""
    from solve3d import transient_gate
    out = transient_gate.consistency(tcase)
    # The gate is the LU-vs-LU comparison: same operator, same solver class.
    assert out["assembly_consistency_gate"] < 1e-10, out
    # And the GMRES comparison is recorded, not gated: it measures Phase A's
    # ksp_rtol=1e-10 showing through |E|^2, which is a property of the
    # production solver rather than of the adjoint.
    assert out["vs_phase_a_iterative"]["max_dQ_over_Qmax"] < 1e-5, out
    assert out["n_eqs_solves"] >= 3, out


def test_reverse_march_fd_gate(tcase):
    """The gate: dJ/d(sigma_base) through EQS -> in-march re-solves -> enthalpy
    march, J = sum_i vol_i (phi - chi)^2 at a FIXED read step."""
    from solve3d import transient_gate
    doc = transient_gate.run_gate(tcase)
    g = doc["gate"]
    assert g["all_pass_subgradient"], {
        k: v["best_rel_err"] for k, v in g["probes"].items()}


def test_transient_mutants_fail_the_gate(tcase):
    from solve3d import transient_gate
    doc = transient_gate.run_mutants(tcase)
    for name in ("renorm_frozen", "adjoint_dropped"):
        assert not doc[name]["all_pass_subgradient"], (name, doc[name])
