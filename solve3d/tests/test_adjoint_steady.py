"""Phase B Task 1: the steady EQS adjoint, re-gated on a Phase A circle mesh.

RUNS IN THE SPIKE ENV:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_adjoint_steady.py

Layer B1 of the pre-registered bisect: J is a function of Q_rf ONLY, with no
thermal march, so a failure here cannot be blamed on the transient.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import gate_fd


@pytest.fixture(scope="module")
def case():
    from solve3d import adjoint
    # The Phase A circle mesh at its coarse refinement level (23040 in-part
    # nodes = the heatr3d n=64 unknown count), which is also the size class D1
    # gated its adjoint on.
    return adjoint.SteadyCase.build(shape="circle", target_nodes_in_part=23040,
                                    lc0=0.0009375)


def test_adjoint_forward_matches_the_phase_a_forward(case):
    """Assembly consistency: forward and adjoint MUST share one discrete
    operator. The adjoint carries its own LU-factorized solve (the adjoint
    reuses the factorization), so its Q_rf is checked against the Phase A
    production path before any gradient is trusted."""
    out = case.consistency_vs_phase_a()
    assert out["assembly_consistency_gate"] < 1e-10, out
    assert out["vs_phase_a_lu"]["max_dV_over_v_lo"] < 1e-10, out
    assert out["vs_phase_a_lu"]["scale_rel_diff"] < 1e-10, out
    # recorded, not gated -- Phase A's GMRES ksp_rtol showing through |E|^2
    assert out["vs_phase_a_iterative"]["max_dQ_over_Qmax"] < 1e-5, out


def test_renormalization_pins_the_part_mean_of_q(case):
    """The fixed-power renormalization makes the part mean of Q_rf a CONSTANT
    of the problem, independent of sigma. D1 measured this identity to 2.2e-16;
    it is what makes dQbar/ds = 0 an output rather than an assumption."""
    st = case.forward(case.s0)
    assert abs(st.qbar / case.power_density - 1.0) < 1e-12
    assert st.clip_active is False, "a live Q clip would raise a subgradient question"


def test_dj_dsigma_fd_gate(case):
    """The gate: four pre-registered probes, eight epsilons, central
    differences, thresholds read from the protocol JSON."""
    doc = case.run_fd_gate()
    assert doc["consistency"]["assembly_consistency_gate"] < 1e-10
    # THE frozen standard (FROZEN_CONVENTIONS_2D.md section 5 item 5): 1e-5 is
    # the campaign pass, 1e-6 is preferred, and the COUNT AT BOTH is reported.
    assert doc["gate"]["all_pass_subgradient"], {
        k: v["best_rel_err"] for k, v in doc["gate"]["probes"].items()}
    g = doc["gate"]
    assert g["n_pass_subgradient"] == g["n_probes"]
    assert g["n_pass_preferred"] >= 3, g["n_pass_preferred"]


def test_the_preferred_standard_miss_is_a_measured_floor_artifact(case):
    """Checklist item 7: interpret a relative-error miss against the ANALYTIC
    magnitude, and item 6: measure the evaluation floor.

    One probe (random_cell) reads 1.42e-06, i.e. 1.42x the PREFERRED 1e-6
    standard while clearing the frozen 1e-5 one. This test pins the evidence
    that it is an FD noise floor rather than a wrong gradient -- the same
    signature D1 recorded (`task5.gate.fd_noise_floor_note`):

      * the ABSOLUTE error is nearly flat across probes while the analytic
        derivative spans orders of magnitude, so the relative error is just
        1/|analytic|;
      * the probe that misses is the one with the SMALLEST derivative;
      * the floor-free probe -- the gradient direction, whose derivative is the
        largest available -- reads ~1e-10.
    """
    g = case.run_fd_gate()["gate"]
    abs_lo, abs_hi = g["abs_err_range"]
    an_lo, an_hi = g["analytic_magnitude_range"]
    # absolute error is flat compared with how far the analytic magnitude moves
    assert (abs_hi / abs_lo) < 100.0, g["abs_err_range"]
    assert (an_hi / an_lo) > 1000.0, g["analytic_magnitude_range"]
    assert (an_hi / an_lo) > 10.0 * (abs_hi / abs_lo)
    # the miss is the smallest-derivative probe
    probes = g["probes"]
    worst = max(probes, key=lambda k: probes[k]["best_rel_err"])
    smallest = min(probes, key=lambda k: abs(probes[k]["analytic_directional"]))
    assert worst == smallest, (worst, smallest)
    # and the floor-free probe is clean by four orders
    assert probes["gradient_direction"]["best_rel_err"] < 1e-9


def test_mutants_fail_the_gate(case):
    """A gate that the mutants pass is not a gate (D1 pattern, pre-registered
    in the protocol)."""
    doc = case.run_mutation_tests()
    for name in ("renorm_frozen", "adjoint_dropped"):
        m = doc[name]
        assert not m["all_pass_subgradient"], (name, m)
        assert m["worst_best_rel_err"] > gate_fd.SUBGRADIENT_PASS_REL_ERR, (name, m)
