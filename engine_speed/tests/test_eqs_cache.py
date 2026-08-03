"""RED-first gate for the EQS factorization/solution cache.

THE CATASTROPHIC FAILURE MODE IS A FALSE CACHE HIT: returning a stale field for
a system that actually changed. These tests are weighted accordingly -- the
MISS tests and the poisoned-key mutation tests are the point, not the hit test.

Sizes are kept small (n=20/24) so the suite stays runnable; n=24 (N=13824)
exercises heatr3d's DIRECT path and ``iterative=True`` is forced where the
ILU path needs covering, because the auto threshold (N > 50_000) would
otherwise need n >= 37 and a much slower test.
"""
from __future__ import annotations

import numpy as np
import pytest

import heatr3d as h3

from engine_speed.eqs_cache import (EqsCache, assemble_eqs,
                                    scale_invariance_deviation,
                                    solve_eqs_3d_cached)

P = h3.Params()


def _setup(n=24, diam=0.020, sat_seed=None):
    g = h3.Grid(n=n)
    part = h3.make_geometry(g, "square", diam=diam, zspan=0.020)
    sat = None
    if sat_seed is not None:
        rng = np.random.default_rng(sat_seed)
        sat = np.zeros(part.shape)
        sat[part] = rng.uniform(0.3, 1.0, int(part.sum()))
    gamma = h3.build_gamma(part, P, sat)
    return g, part, gamma


# --------------------------------------------------------------------------- #
# assembly contract
# --------------------------------------------------------------------------- #
def test_assembly_reproduces_heatr3d_solution_exactly():
    """The ported assembly must be the SAME linear system heatr3d builds --
    verified end-to-end, since heatr3d never exposes its A and b."""
    import scipy.sparse.linalg as spla
    g, part, gamma = _setup(20)
    A, b = assemble_eqs(gamma, g, P)
    mine = spla.spsolve(A, b).reshape(gamma.shape)
    ref = h3.solve_eqs_3d(gamma, g, P)
    assert np.array_equal(ref, mine)


# --------------------------------------------------------------------------- #
# hits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("iterative", [None, True])
def test_repeat_solve_hits_and_is_bit_identical(iterative):
    g, part, gamma = _setup(24)
    c = EqsCache()
    ref = h3.solve_eqs_3d(gamma, g, P, iterative=iterative)
    v1 = solve_eqs_3d_cached(gamma, g, P, iterative=iterative, cache=c)
    v2 = solve_eqs_3d_cached(gamma, g, P, iterative=iterative, cache=c)
    assert np.array_equal(ref, v1), "miss path diverged from heatr3d"
    assert np.array_equal(ref, v2), "HIT path returned a different field"
    assert c.stats["solution_hits"] == 1
    assert c.stats["solution_misses"] == 1


def test_hit_returns_an_independent_copy():
    """A caller mutating the returned field must not poison the cache."""
    g, part, gamma = _setup(20)
    c = EqsCache()
    v1 = solve_eqs_3d_cached(gamma, g, P, cache=c)
    v1[0, 0, 0] = 12345.0 + 0j
    v2 = solve_eqs_3d_cached(gamma, g, P, cache=c)
    assert v2[0, 0, 0] != 12345.0


# --------------------------------------------------------------------------- #
# misses -- the safety-critical half
# --------------------------------------------------------------------------- #
def test_changed_sat_misses():
    g, part, gamma_a = _setup(20)
    _, _, gamma_b = _setup(20, sat_seed=7)
    c = EqsCache()
    va = solve_eqs_3d_cached(gamma_a, g, P, cache=c)
    vb = solve_eqs_3d_cached(gamma_b, g, P, cache=c)
    assert c.stats["solution_hits"] == 0, "different sat produced a FALSE HIT"
    assert not np.array_equal(va, vb)
    assert np.array_equal(vb, h3.solve_eqs_3d(gamma_b, g, P))


def test_changed_grid_n_misses():
    ga, _, gamma_a = _setup(20)
    gb, _, gamma_b = _setup(24)
    c = EqsCache()
    solve_eqs_3d_cached(gamma_a, ga, P, cache=c)
    vb = solve_eqs_3d_cached(gamma_b, gb, P, cache=c)
    assert c.stats["solution_hits"] == 0, "different n produced a FALSE HIT"
    assert np.array_equal(vb, h3.solve_eqs_3d(gamma_b, gb, P))


def test_changed_grid_spacing_at_same_n_misses():
    """Same n, different chamber size L -> different h -> different matrix.
    Hashing the mask alone would MISS this; hashing gamma alone would too."""
    ga = h3.Grid(n=20, L=0.060)
    gb = h3.Grid(n=20, L=0.075)
    part = h3.make_geometry(ga, "square", diam=0.020, zspan=0.020)
    gamma = h3.build_gamma(part, P, None)
    c = EqsCache()
    solve_eqs_3d_cached(gamma, ga, P, cache=c)
    vb = solve_eqs_3d_cached(gamma, gb, P, cache=c)
    assert c.stats["solution_hits"] == 0, "different h produced a FALSE HIT"
    assert np.array_equal(vb, h3.solve_eqs_3d(gamma, gb, P))


@pytest.mark.parametrize("field,newval", [("v_lo", 3600.0), ("v_hi", 25.0)])
def test_changed_electrode_voltage_misses_the_solution_cache(field, newval):
    from dataclasses import replace
    g, part, gamma = _setup(20)
    p2 = replace(P, **{field: newval})
    c = EqsCache()
    solve_eqs_3d_cached(gamma, g, P, cache=c)
    v2 = solve_eqs_3d_cached(gamma, g, p2, cache=c)
    assert c.stats["solution_hits"] == 0, f"changed {field} produced a FALSE HIT"
    assert np.array_equal(v2, h3.solve_eqs_3d(gamma, g, p2))


# --------------------------------------------------------------------------- #
# POISONED KEY: a one-ULP perturbation of any assembly input must miss
# --------------------------------------------------------------------------- #
def test_one_ulp_gamma_mutation_misses():
    g, part, gamma = _setup(20)
    c = EqsCache()
    solve_eqs_3d_cached(gamma, g, P, cache=c)
    poisoned = gamma.copy()
    tgt = tuple(x // 2 for x in gamma.shape)
    poisoned[tgt] = complex(np.nextafter(poisoned[tgt].real, np.inf),
                            poisoned[tgt].imag)
    assert poisoned[tgt] != gamma[tgt]
    v = solve_eqs_3d_cached(poisoned, g, P, cache=c)
    assert c.stats["solution_hits"] == 0, "1-ULP gamma change produced a FALSE HIT"
    assert np.array_equal(v, h3.solve_eqs_3d(poisoned, g, P))


def test_one_ulp_mutation_in_the_imaginary_part_misses():
    """eps_r enters gamma only through the imaginary part; a cache that hashed
    only sigma would silently reuse the wrong field for an eps sweep."""
    g, part, gamma = _setup(20)
    c = EqsCache()
    solve_eqs_3d_cached(gamma, g, P, cache=c)
    poisoned = gamma.copy()
    tgt = tuple(x // 2 for x in gamma.shape)
    poisoned[tgt] = complex(poisoned[tgt].real,
                            np.nextafter(poisoned[tgt].imag, np.inf))
    solve_eqs_3d_cached(poisoned, g, P, cache=c)
    assert c.stats["solution_hits"] == 0, "1-ULP eps change produced a FALSE HIT"


def test_solver_source_change_invalidates_the_cache():
    """The key must fold in a fingerprint of heatr3d's own EQS source, so a
    future edit to solve_eqs_3d/_harmonic cannot be served from a stale entry."""
    from engine_speed import eqs_cache as ec
    g, part, gamma = _setup(20)
    c = EqsCache()
    solve_eqs_3d_cached(gamma, g, P, cache=c)
    real = ec.solver_fingerprint()
    try:
        ec._FINGERPRINT_OVERRIDE = "pretend-heatr3d-eqs-was-edited"
        solve_eqs_3d_cached(gamma, g, P, cache=c)
    finally:
        ec._FINGERPRINT_OVERRIDE = None
    assert c.stats["solution_hits"] == 0, (
        "a changed heatr3d EQS fingerprint produced a FALSE HIT")
    assert ec.solver_fingerprint() == real


# --------------------------------------------------------------------------- #
# factorization reuse (ILU path only -- see report for why the direct path
# cannot reuse a factorization bit-identically)
# --------------------------------------------------------------------------- #
def test_voltage_sweep_reuses_the_factorization_bit_identically():
    from dataclasses import replace
    g, part, gamma = _setup(24)
    c = EqsCache()
    v1 = solve_eqs_3d_cached(gamma, g, P, iterative=True, cache=c)
    p2 = replace(P, v_lo=3600.0)
    v2 = solve_eqs_3d_cached(gamma, g, p2, iterative=True, cache=c)
    assert c.stats["solution_hits"] == 0
    assert c.stats["factorization_hits"] == 1, "ILU was rebuilt for the same matrix"
    assert np.array_equal(v1, h3.solve_eqs_3d(gamma, g, P, iterative=True))
    assert np.array_equal(v2, h3.solve_eqs_3d(gamma, g, p2, iterative=True))


def test_changed_gamma_does_not_reuse_the_factorization():
    g, part, gamma_a = _setup(20)
    _, _, gamma_b = _setup(20, sat_seed=3)
    c = EqsCache()
    solve_eqs_3d_cached(gamma_a, g, P, iterative=True, cache=c)
    vb = solve_eqs_3d_cached(gamma_b, g, P, iterative=True, cache=c)
    assert c.stats["factorization_hits"] == 0, "FALSE factorization reuse"
    assert np.array_equal(vb, h3.solve_eqs_3d(gamma_b, g, P, iterative=True))


# --------------------------------------------------------------------------- #
# the renorm / scale-invariance question
# --------------------------------------------------------------------------- #
def test_scale_invariance_is_NOT_reliably_bit_identical():
    """The load-bearing claim: gamma -> c*gamma cannot be used as a cache-hit
    rule, because the exact-arithmetic invariance does not survive floating
    point. One counterexample is enough to forbid it, and there are many.

    (The invariance itself is real -- see scale_invariance_deviation's
    docstring for the derivation -- which is exactly what makes this a trap.)"""
    counterexamples = []
    for n in (20, 24):
        g, part, gamma = _setup(n)
        for c in (2.0, 1000.0):
            d = scale_invariance_deviation(gamma, g, P, part, c)
            if not (d["V_bit_identical"] and d["Qrf_bit_identical"]):
                counterexamples.append((n, c, d["Qrf_max_rel_dev"]))
    assert counterexamples, (
        "expected at least one (n, c) where the renorm shortcut is not "
        "bit-identical; if this ever passes cleanly, re-measure before "
        "trusting the shortcut")


def test_scale_invariance_sometimes_IS_exact_which_is_why_it_is_a_trap():
    """A shortcut that is exact on a small test grid and inexact on a bigger
    one is worse than one that always fails: it would pass a cheap gate and
    then corrupt production. Pinned so nobody 're-discovers' the n=20 result
    and concludes the shortcut is safe."""
    g20, part20, gamma20 = _setup(20)
    g24, part24, gamma24 = _setup(24)
    d20 = scale_invariance_deviation(gamma20, g20, P, part20, 2.0)
    d24 = scale_invariance_deviation(gamma24, g24, P, part24, 2.0)
    assert d20["V_bit_identical"] != d24["V_bit_identical"], (
        "the n-dependence of the c=2 result changed; re-measure and update the "
        "report before relying on any renorm shortcut")


def test_scale_invariance_error_grows_with_the_scale_factor():
    g, part, gamma = _setup(24)
    d2 = scale_invariance_deviation(gamma, g, P, part, 2.0)
    d1k = scale_invariance_deviation(gamma, g, P, part, 1000.0)
    assert d1k["Qrf_max_rel_dev"] > d2["Qrf_max_rel_dev"]


def test_s4_zero_coefficient_resolve_is_an_exact_cache_hit():
    """The engine lane's named reference case. With sigma_temp_coeff_per_K and
    sigma_density_coeff both 0, apply_sigma_coupling returns gamma bit-for-bit,
    so every scheduled S4 re-solve is an EXACT cache hit -- no renorm trick
    needed."""
    g, part, gamma = _setup(20)
    T = np.full(part.shape, 137.0)
    rho = np.full(part.shape, 0.71)
    gamma_c = h3.apply_sigma_coupling(gamma, part, T, rho, P)
    assert np.array_equal(gamma, gamma_c), "a=b=0 coupling was not bit-for-bit"
    c = EqsCache()
    solve_eqs_3d_cached(gamma, g, P, cache=c)
    solve_eqs_3d_cached(gamma_c, g, P, cache=c)
    assert c.stats["solution_hits"] == 1


def test_s4_nonzero_coefficient_resolve_is_not_a_scalar_rescale():
    """With nonzero coefficients the coupling scales ONLY Re(gamma), and by a
    spatially varying factor, so it is not c*gamma for any scalar c -- the
    scale-invariance argument cannot rescue it even in exact arithmetic."""
    from dataclasses import replace
    p2 = replace(P, sigma_temp_coeff_per_K=0.002, sigma_density_coeff=0.6)
    g, part, gamma = _setup(20)
    T = np.full(part.shape, 23.0)
    T[part] = np.linspace(100.0, 200.0, int(part.sum()))
    rho = np.full(part.shape, p2.rho_rel)
    gc = h3.apply_sigma_coupling(gamma, part, T, rho, p2)
    assert not np.array_equal(gamma, gc)
    assert np.array_equal(np.imag(gamma), np.imag(gc)), \
        "coupling touched the displacement part; the analysis needs revisiting"
    ratio = np.real(gc)[part] / np.real(gamma)[part]
    assert ratio.max() - ratio.min() > 1e-6, "expected a spatially varying factor"


# --------------------------------------------------------------------------- #
# end-to-end: the cache must not change a single bit of a marched run
# --------------------------------------------------------------------------- #
def test_march_with_cache_is_bit_identical_to_heatr3d():
    from engine_speed.march_fast import march_fast
    g = h3.Grid(n=20)
    part = h3.make_geometry(g, "square", diam=0.020, zspan=0.020)
    kw = dict(max_time_s=5.0, phi_target=0.90, densify=True, stop_mean_rho=0.98)
    ref = h3.run(g, part, P, **kw)
    c = EqsCache()
    a = march_fast(g, part, P, eqs_cache=c, **kw)
    b = march_fast(g, part, P, eqs_cache=c, **kw)      # this one HITS
    assert c.stats["solution_hits"] == 1
    for name in ("T_final", "T_phi90", "phi_final", "Qrf", "rho_final"):
        assert np.array_equal(getattr(ref, name), getattr(a, name)), name
        assert np.array_equal(getattr(ref, name), getattr(b, name)), f"{name} (hit)"
    assert ref.phi_hist == a.phi_hist == b.phi_hist
    assert ref.energy_residual_frac == a.energy_residual_frac == b.energy_residual_frac
