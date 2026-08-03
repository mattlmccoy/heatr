"""The ported FD/subgradient gate machinery, exercised on ANALYTIC functions.

Testing the gate itself before pointing it at a solver is checklist item 2
(layered bisect): if the gate is wrong, every downstream verdict is wrong.
Runs in either environment (pure numpy).
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import gate_fd


def test_thresholds_come_from_the_pre_registered_protocol():
    """The gate must not carry its own copy of the numbers."""
    pr = gate_fd.protocol()
    assert gate_fd.EPSILONS == tuple(pr["fd"]["epsilons"])
    assert gate_fd.PASS_REL_ERR == pr["thresholds"]["pass_rel_err"]
    assert gate_fd.SUBGRADIENT_PASS_REL_ERR == \
        pr["thresholds"]["subgradient_pass_rel_err"]


def test_sweep_recovers_an_exact_gradient_to_the_preferred_standard():
    """Smooth quadratic, exact analytic gradient: the V must bottom below the
    1e-6 preferred standard."""
    rng = np.random.default_rng(0)
    n = 40
    A = rng.random(n) + 1.0
    x0 = rng.random(n) + 1.0

    def J(x):
        return float(np.dot(A, (x - 0.3) ** 2))

    grad = 2.0 * A * (x0 - 0.3)
    d = np.zeros(n)
    d[7] = 1.0
    out = gate_fd.sweep(J, x0, d, float(np.dot(grad, d)), x_scale=abs(x0[7]))
    assert out["best_rel_err"] < gate_fd.PASS_REL_ERR, out["best_rel_err"]
    assert out["best_eps"] in gate_fd.EPSILONS


def test_sweep_rejects_a_deliberately_wrong_gradient():
    """A gate that a wrong gradient passes is not a gate."""
    rng = np.random.default_rng(1)
    n = 20
    A = rng.random(n) + 1.0
    x0 = rng.random(n) + 1.0

    def J(x):
        return float(np.dot(A, (x - 0.3) ** 2))

    bad = 2.0 * A * (x0 - 0.3) * 1.05          # 5 % wrong
    d = np.zeros(n)
    d[3] = 1.0
    out = gate_fd.sweep(J, x0, d, float(np.dot(bad, d)), x_scale=abs(x0[3]))
    assert out["best_rel_err"] > gate_fd.SUBGRADIENT_PASS_REL_ERR


def test_evaluation_floor_estimate_recovers_double_precision_on_a_clean_function():
    """Checklist item 6: floor ~ 2*eps*abs_err in the roundoff tail.

    With no injected floor the objective's own floor is double-precision
    representation, |J| * 2.2e-16, and the estimator must land within two orders
    of it. (First attempt at this test injected a quantizer that was a no-op --
    J/q happened to be exactly integral -- and the estimator correctly reported
    the double-precision floor instead. The test was wrong, not the gate.)"""
    rng = np.random.default_rng(5)
    n = 10
    x0 = 1.0 + rng.random(n)

    def J(x):
        return float(np.sum((x - 0.3) ** 2))

    grad = 2.0 * (x0 - 0.3)
    d = np.zeros(n)
    d[0] = 1.0
    out = gate_fd.sweep(J, x0, d, float(np.dot(grad, d)), x_scale=abs(x0[0]))
    expected = abs(J(x0)) * 2.2e-16
    est = out["evaluation_floor_estimate"]
    assert expected / 100.0 < est < expected * 100.0, (est, expected)


def test_evaluation_floor_estimate_tracks_a_floor_that_actually_bites():
    """Same estimator, with a quantization coarse enough to dominate the
    central difference at the smallest epsilons."""
    rng = np.random.default_rng(6)
    n = 10
    x0 = 1.0 + rng.random(n)
    q = 1e-6

    def J(x):
        return float(np.round(np.sum((x - 0.3) ** 2) / q) * q)

    grad = 2.0 * (x0 - 0.3)
    d = np.zeros(n)
    d[0] = 1.0
    out = gate_fd.sweep(J, x0, d, float(np.dot(grad, d)), x_scale=abs(x0[0]))
    est = out["evaluation_floor_estimate"]
    assert q / 20.0 < est < q * 20.0, (est, q)
    # A coarse floor does not break the gate, it MOVES the bottom of the V to a
    # larger epsilon -- which is exactly why the sweep exists instead of a
    # single hard-coded step. (Asserting the gate must fail here was wrong: at
    # eps 1e-3 the finite-difference signal still dwarfs the quantization.)
    assert out["best_eps"] >= 1e-5, out["best_eps"]


def test_probe_directions_are_the_four_pre_registered_probes():
    rng = np.random.default_rng(2)
    mask = np.zeros(30, bool)
    mask[5:25] = True
    g = np.zeros(30)
    g[mask] = rng.standard_normal(20)
    g[11] = 100.0                                # the max-sensitivity cell
    probes = gate_fd.probe_directions(g, mask, seed=7)
    assert [p["name"] for p in probes] == [
        "max_sensitivity_cell", "random_cell", "random_direction",
        "gradient_direction"]
    assert probes[0]["index"] == 11
    for p in probes:
        assert not np.any(p["direction"][~mask]), p["name"]
        assert np.isclose(np.linalg.norm(p["direction"]), 1.0)


def test_probe_single_cell_directions_are_unit_vectors_inside_the_mask():
    mask = np.zeros(12, bool)
    mask[2:9] = True
    g = np.zeros(12)
    g[mask] = np.arange(7, dtype=float) + 1.0
    probes = gate_fd.probe_directions(g, mask, seed=3)
    d0 = probes[0]["direction"]
    assert int(np.count_nonzero(d0)) == 1
    assert mask[int(np.argmax(np.abs(d0)))]


def test_verdict_labels_pass_at_both_standards():
    r = gate_fd.verdict(3e-7)
    assert r["pass_preferred"] and r["pass_subgradient"]
    r = gate_fd.verdict(3e-6)
    assert not r["pass_preferred"] and r["pass_subgradient"]
    r = gate_fd.verdict(3e-4)
    assert not r["pass_preferred"] and not r["pass_subgradient"]
