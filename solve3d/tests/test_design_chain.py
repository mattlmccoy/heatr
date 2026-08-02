"""Phase C Task 2: the design chain -- explicit normalized-convolution filter
plus the smoothed-Heaviside projection, with every transpose gated.

RUNS IN THE SPIKE ENV.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import gate_fd


@pytest.fixture(scope="module")
def tcase():
    from solve3d import transient_gate
    return transient_gate.build_case()


@pytest.fixture(scope="module")
def chain(tcase):
    from solve3d import design_chain
    return design_chain.DesignChain.build(tcase)


# ------------------------------------------------------------------ filter --
def test_filter_is_a_convex_combination_so_the_box_is_preserved(chain):
    """FROZEN_CONVENTIONS_2D section 1.2: a NORMALIZED convolution, so v in
    [0,1] gives s in [0,1] automatically and NO clip subgradient enters the
    chain rule. Their test_box_is_preserved, ported."""
    rng = np.random.default_rng(0)
    v = rng.random(chain.n_design)
    s = chain.filter_apply(v)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s.min() >= v.min() - 1e-12 and s.max() <= v.max() + 1e-12


def test_filter_reproduces_a_uniform_field_exactly(chain):
    """Partition of unity: row weights sum to 1, so a constant passes through
    untouched. Without this the filter would dim the design."""
    for c in (0.0, 0.37, 1.0):
        s = chain.filter_apply(np.full(chain.n_design, c))
        assert np.allclose(s, c, rtol=0, atol=1e-12)


def test_filter_matrix_is_not_symmetric(chain):
    """Evidence that the transpose gate below is testing something. Row
    normalization and volume weighting both break symmetry, which is exactly
    why the explicit matrix was chosen over a self-adjoint Helmholtz filter
    whose transpose check would be trivially true."""
    W = chain.W
    asym = abs(W - W.T)
    assert asym.max() > 1e-6, "a symmetric filter makes the transpose gate vacuous"


def test_filter_transpose_is_exact(chain):
    rng = np.random.default_rng(1)
    out = gate_fd.transpose_residual(
        chain.filter_apply, chain.filter_transpose,
        rng.standard_normal(chain.n_design), rng.standard_normal(chain.n_design))
    assert out["rel_err"] <= 1e-13, out


def test_filter_radius_is_physical_not_cellwise(chain):
    """The radius is a LENGTH (1.0 mm), so the measured smoothing length must
    match it regardless of the cell size."""
    m = chain.kernel_report()
    assert m["radius_m"] == 1.0e-3
    assert abs(m["measured_std_m"] / m["radius_m"] - 1.0) < 0.15, m
    assert m["mean_neighbours"] > 8, m


# -------------------------------------------------------------- projection --
def test_projection_fixes_the_endpoints_and_the_threshold(chain):
    from solve3d import design_chain as dc
    for beta in (1.0, 16.0):
        for u, want in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
            assert dc.project(np.array([u]), beta)[0] == pytest.approx(want, abs=1e-12)


def test_projection_is_monotone_and_maps_the_box_into_itself(chain):
    from solve3d import design_chain as dc
    u = np.linspace(0.0, 1.0, 257)
    for beta in (1.0, 16.0):
        p = dc.project(u, beta)
        assert np.all(np.diff(p) > 0)
        assert p.min() >= 0.0 and p.max() <= 1.0


def test_projection_derivative_matches_central_differences(chain):
    from solve3d import design_chain as dc
    rng = np.random.default_rng(2)
    u = rng.random(500)
    for beta in (1.0, 16.0):
        d = dc.project_prime(u, beta)
        h = 1e-7
        fd = (dc.project(u + h, beta) - dc.project(u - h, beta)) / (2 * h)
        assert np.max(np.abs(d - fd)) / np.max(np.abs(d)) < 1e-6, beta


def test_beta_zero_is_bit_identical_to_the_filter_alone(chain):
    """FROZEN_CONVENTIONS_2D section 1.3 flag-off identity: the port must keep
    an equivalent switch, and it must be BIT-identical."""
    rng = np.random.default_rng(3)
    v = rng.random(chain.n_design)
    assert float(np.max(np.abs(chain.design_to_map(v, beta=0.0)
                               - chain.filter_apply(v)))) == 0.0


# ---------------------------------------------------------------- composed --
@pytest.mark.parametrize("beta", [0.0, 1.0, 16.0])
def test_composed_transpose_is_exact_at_every_beta(chain, beta):
    """FROZEN_CONVENTIONS_2D section 5 item 8: the dot-product identity
    <dJ/dv, d> == <dJ/ds, dS(v)[d]> at ANY design point. This is the bisect
    that separates a chain-rule error from a property of the forward."""
    rng = np.random.default_rng(4)
    v = rng.random(chain.n_design)
    out = gate_fd.transpose_residual(
        lambda d: chain.design_jvp(v, d, beta=beta),
        lambda g: chain.design_vjp(v, g, beta=beta),
        rng.standard_normal(chain.n_design), rng.standard_normal(chain.n_design))
    assert out["rel_err"] <= 1e-13, (beta, out)


def test_composed_dj_dv_fd_gate(tcase, chain):
    """The composed gradient through filter (+ projection) and the full coupled
    forward, re-gated with the Phase B protocol before ANY solve runs."""
    from solve3d import phase_c_run
    doc = phase_c_run.run_chain_gate(tcase, chain)
    for beta_key, g in doc["gates"].items():
        assert g["all_pass_subgradient"], (beta_key, {
            k: v["best_rel_err"] for k, v in g["probes"].items()})
