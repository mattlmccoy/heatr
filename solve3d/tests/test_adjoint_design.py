"""Phase B Tasks 3 and 4: the design-field chain, and envelope stop-time.

RUNS IN THE SPIKE ENV.
Layers B3 (design composition) and B4 (envelope read) of the bisect.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import gate_fd


@pytest.fixture(scope="module")
def tcase():
    from solve3d import transient_gate
    return transient_gate.build_case()


# --------------------------------------------------------------------- B3 --
def test_design_map_matches_the_frozen_2d_actuator_convention(tcase):
    """FROZEN_CONVENTIONS_2D.md section 7: conductivity only,
    sigma = sigma_v + sat * fill * (sigma_d0 - sigma_v); and section 4: outside
    the part the saturation is held at 1.0 (inert on a conforming mesh, where
    fill is exactly 0 or 1 and there are no partial cells -- checked, not
    assumed)."""
    p = tcase.p
    v = np.full(tcase.eqs.part.size, 0.5)
    sig = tcase.design_to_sigma(v)
    assert np.allclose(sig, p.sigma_virgin + 0.5 * (p.sigma_doped - p.sigma_virgin))
    # box endpoints land on the physical endpoints
    assert np.allclose(tcase.design_to_sigma(np.zeros_like(v)), p.sigma_virgin)
    assert np.allclose(tcase.design_to_sigma(np.ones_like(v)), p.sigma_doped)
    # the conforming mesh really has no partial cells
    d = np.real(tcase.mats.doped.x.array)
    assert np.all((d == 0.0) | (d == 1.0))


def test_design_transpose_is_exact(tcase):
    """Checklist item 8 on the design map: <J(dv), y> == <dv, J^T y>."""
    rng = np.random.default_rng(21)
    n = tcase.eqs.part.size
    out = gate_fd.transpose_residual(tcase.design_jvp, tcase.design_vjp,
                                     rng.standard_normal(n),
                                     rng.standard_normal(n))
    assert out["pass"], out


def test_dj_dv_fd_gate(tcase):
    """B3 gate: dJ/dv through the design map and the full coupled forward."""
    from solve3d import transient_gate
    doc = transient_gate.run_design_gate(tcase)
    assert doc["gate"]["all_pass_subgradient"], {
        k: v["best_rel_err"] for k, v in doc["gate"]["probes"].items()}


def test_gradient_cost_is_within_the_target(tcase):
    """Cost target from the protocol: <= ~2 forward-equivalents."""
    from solve3d import transient_gate
    doc = transient_gate.run_cost(tcase)
    assert doc["store_everything"]["forward_equivalents"] <= 2.0, doc


# --------------------------------------------------------------------- B4 --
def test_envelope_read_is_interior_and_stable(tcase):
    """Checklist item 9: report whether the argmin moves under the probe
    perturbations rather than assuming the envelope theorem covers it. The
    2-D lane flags `at_horizon` because a minimum on the last stored step makes
    J an upper bound; the same flag is required here."""
    from solve3d import transient_gate
    doc = transient_gate.run_envelope_gate(tcase)
    assert doc["read_state"]["at_horizon"] is False, doc["read_state"]
    assert doc["read_state"]["argmin_moves_under_probes"] is False, doc["read_state"]


def test_envelope_gradient_equals_the_fixed_read_gradient_exactly(tcase):
    """The ported S1-vs-S2 exact-agreement gate: the envelope rule (argmin over
    the trajectory) and the fixed-index rule evaluated AT that argmin must
    produce the same gradient with NO dt*/ds term. The 2-D lane verified 0.0
    relative difference; anything else means the two rules are not the same
    objective."""
    from solve3d import transient_gate
    doc = transient_gate.run_envelope_gate(tcase)
    assert doc["exact_agreement"]["max_abs_diff"] == 0.0, doc["exact_agreement"]
    assert doc["exact_agreement"]["rel_diff"] == 0.0, doc["exact_agreement"]


def test_envelope_objective_fd_gate(tcase):
    """The substantive envelope check: finite differences of J* = min_t J(t)
    include any dt*/ds effect automatically. If the analytic gradient, which
    OMITS that term, still matches FD, the envelope theorem is verified
    numerically rather than argued."""
    from solve3d import transient_gate
    doc = transient_gate.run_envelope_gate(tcase)
    assert doc["gate"]["all_pass_subgradient"], {
        k: v["best_rel_err"] for k, v in doc["gate"]["probes"].items()}
