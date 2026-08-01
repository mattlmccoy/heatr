"""The PERMITTIVITY channel of the dopant actuator, written RED first.

Every historical dopant map co-varies relative permittivity with saturation:
the production loader `fgm_feedback.saturation_map_npz` multiplies the geometry
fill fraction by the saturation map and blends BOTH conductivity and relative
permittivity with that product (`rfam_eqs_coupled.py:2527-2532`). The two-sided
per-node hook `sat_map_npz_direct` pins permittivity to geometry fill instead
(`rfam_eqs_coupled.py:342`, `eps_geometry_only = True`), which is the channel the
solve has actuated so far.

These tests cover, in order:

  1. the forward permittivity field in the co-varying channel equals the
     production expression exactly;
  2. the electro-quasi-static (EQS) vector-Jacobian product in permittivity
     reproduces a brute-force central difference of the absorbed-power field;
  3. asking for the permittivity channel leaves the conductivity sensitivity
     BIT-IDENTICAL, so the flag-off path cannot drift;
  4. `adjoint.gradient(..., eps_covary=False)` is bit-identical to the call
     without the keyword, on the real engine;
  5. `adjoint.gradient(..., eps_covary=True)` actually moves the gradient, so
     test 4 is not vacuous.

Acronyms: EQS = electro-quasi-static; VJP = vector-Jacobian product.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import adjoint, forward as fwd, gradops, shape_objective as so
from adjoint2d.library_solve import shape_config
from adjoint2d.pins import EPS0, build_case, load_cfg

from test_adjoint_pieces import _MiniCase, _state

N_STEPS = 950


# ---------------------------------------------------------------------------
# 1. the forward field
# ---------------------------------------------------------------------------

def test_eps_field_covary_matches_the_production_effective_fill_expression():
    case = _mini_with_pins()
    s = np.full(case.part_mask.shape, 0.73)
    s[case.part_mask] = np.linspace(0.2, 1.0, int(case.part_mask.sum()))
    got = fwd.eps_field(case, s, covary=True)
    p = case.pins
    want = p.eps_v + (case.fill_frac * s) * (p.eps_d - p.eps_v)
    assert np.array_equal(got, want)
    # and the pinned channel ignores s entirely
    pinned = fwd.eps_field(case, s, covary=False)
    assert np.array_equal(pinned, p.eps_v + case.fill_frac * (p.eps_d - p.eps_v))
    assert not np.array_equal(got, pinned)


def _mini_with_pins():
    case = _MiniCase()
    case.fill_frac = case.part_mask.astype(float)

    class P2:
        omega = 2.0 * np.pi * 27.12e6
        power_factor = 1.0
        max_qrf = 1.0e11
        v_hi = 1500.0
        v_lo = 0.0
        zero_qrf_outside_doped = True
        eps_v = 2.0
        eps_d = 20.0
        sigma_v = 1e-8
        sigma_d0 = 0.04

    case.pins = P2()
    return case


# ---------------------------------------------------------------------------
# 2 and 3. the EQS vector-Jacobian product in permittivity
# ---------------------------------------------------------------------------

def test_eqs_vjp_eps_matches_brute_force_finite_difference():
    rng = np.random.default_rng(11)
    case = _MiniCase()
    ny, nx = case.part_mask.shape
    eps0 = np.where(case.part_mask, 20.0, 2.0).astype(float)
    eps0 = eps0 * (1.0 + 0.15 * rng.standard_normal((ny, nx)) * case.part_mask)
    sigma0 = np.where(case.part_mask, 0.04, 1e-8)
    gQ = np.zeros((ny, nx))
    gQ[case.doped_mask] = rng.standard_normal(int(case.doped_mask.sum()))

    Gx, Gy = gradops.gradient_matrices(case.x, case.y)
    st = _state(case, sigma0, eps0)
    _dJ_dsig, dJ_deps = adjoint.eqs_vjp(case, st, gQ, Gx, Gy, with_eps=True)

    d = np.zeros((ny, nx))
    d[case.part_mask] = rng.standard_normal(int(case.part_mask.sum()))
    d *= 20.0
    ana = float(np.sum(dJ_deps * d))
    assert abs(ana) > 0.0

    best = np.inf
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        jp = float(np.sum(gQ * _state(case, sigma0, eps0 + h * d).Qrf))
        jm = float(np.sum(gQ * _state(case, sigma0, eps0 - h * d).Qrf))
        fd = (jp - jm) / (2 * h)
        best = min(best, abs(fd - ana) / max(abs(ana), 1e-30))
    assert best < 1e-7, f"best relative error {best:.3e}"


def test_asking_for_the_eps_channel_leaves_the_sigma_sensitivity_bit_identical():
    rng = np.random.default_rng(3)
    case = _MiniCase()
    ny, nx = case.part_mask.shape
    eps_r = np.where(case.part_mask, 20.0, 2.0)
    sigma0 = np.where(case.part_mask, 0.04, 1e-8)
    sigma0 = sigma0 * (1.0 + 0.20 * rng.standard_normal((ny, nx)) * case.part_mask)
    gQ = np.zeros((ny, nx))
    gQ[case.doped_mask] = rng.standard_normal(int(case.doped_mask.sum()))
    Gx, Gy = gradops.gradient_matrices(case.x, case.y)
    st = _state(case, sigma0, eps_r)

    a = adjoint.eqs_vjp(case, st, gQ, Gx, Gy)
    b, _eps = adjoint.eqs_vjp(case, st, gQ, Gx, Gy, with_eps=True)
    assert np.max(np.abs(a)) > 0.0
    assert np.array_equal(a, b)


def test_eps_sensitivity_scales_with_omega_eps0_as_the_gamma_chain_requires():
    """d gamma / d eps_r = 1j * omega * EPS0, so at omega = 0 the channel is dead."""
    rng = np.random.default_rng(5)
    case = _MiniCase()
    ny, nx = case.part_mask.shape
    eps_r = np.where(case.part_mask, 20.0, 2.0).astype(float)
    sigma0 = np.where(case.part_mask, 0.04, 1e-8)
    gQ = np.zeros((ny, nx))
    gQ[case.doped_mask] = rng.standard_normal(int(case.doped_mask.sum()))
    Gx, Gy = gradops.gradient_matrices(case.x, case.y)

    st = _state(case, sigma0, eps_r)
    _s, deps = adjoint.eqs_vjp(case, st, gQ, Gx, Gy, with_eps=True)
    assert np.max(np.abs(deps)) > 0.0

    case.pins.omega = 0.0
    st0 = _state(case, sigma0, eps_r)
    _s0, deps0 = adjoint.eqs_vjp(case, st0, gQ, Gx, Gy, with_eps=True)
    assert np.max(np.abs(deps0)) == 0.0
    assert EPS0 > 0.0


# ---------------------------------------------------------------------------
# 4 and 5. the full chain on the real engine
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_case():
    return build_case(load_cfg(shape_config("square")))


def _grad(case, s, **kw):
    """Gradient at the J-stop of a march long enough to have melted.

    N_STEPS is chosen so the melt-fraction ramp is populated: with nothing
    melted the objective seed is identically zero and any bit-identity check
    would be vacuous, which is what a shorter march produced while this test
    was being written.
    """
    tr = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                     n_steps=N_STEPS, eps_covary=kw.pop("eps_covary_forward", False))
    idx = int(so.optimal_stop(tr, case).index)
    seed = so.shape_J_and_seed(tr.T_at_end(idx), case)[1]
    assert np.max(np.abs(seed)) > 0.0, "read state has an empty phase ramp"
    ops = gradops.gradient_matrices(case.x, case.y)
    return adjoint.gradient(case, s, tr, {idx: seed}, grad_ops=ops, **kw)


def test_gradient_with_eps_covary_off_is_bit_identical_to_the_old_call(real_case):
    s = np.full(real_case.part_mask.shape, 0.9)
    a = _grad(real_case, s)
    b = _grad(real_case, s, eps_covary=False)
    assert np.max(np.abs(a)) > 0.0
    assert np.array_equal(a, b)


def test_gradient_with_eps_covary_on_moves_the_gradient(real_case):
    s = np.full(real_case.part_mask.shape, 0.9)
    off = _grad(real_case, s, eps_covary_forward=True)
    on = _grad(real_case, s, eps_covary_forward=True, eps_covary=True)
    assert not np.array_equal(off, on)
    rel = np.max(np.abs(on - off)) / max(np.max(np.abs(off)), 1e-30)
    assert rel > 1e-6, f"eps channel contributes only {rel:.3e} of the sigma channel"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
