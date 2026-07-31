"""RED-first tests for the pure logic of the control arm and the shared
budget/selection rules. No physics here; every one of these is arithmetic that
can silently invert a result if it is wrong."""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import control


def _mask(ny=8, nx=8):
    m = np.zeros((ny, nx), dtype=bool)
    m[2:6, 2:6] = True
    return m


def test_map_at_unit_magnitude_inverts_the_proxy():
    m = _mask()
    proxy = np.zeros(m.shape)
    proxy[m] = np.linspace(0.0, 1.0, int(m.sum()))
    sat = control.proportional_inverse_map(proxy, m, magnitude=1.0,
                                           baseline=0.5, dead_band=0.0,
                                           smoothing_sigma=0.0)
    hottest = np.argmax(proxy[m])
    coldest = np.argmin(proxy[m])
    assert sat[m][hottest] < sat[m][coldest]
    # Outside the part the saturation is held at its nominal value 1, so the
    # arm changes the dopant map and nothing else. Setting it to 0 there also
    # strips the sub-pixel geometry fill from the permittivity field.
    assert np.all(sat[~m] == 1.0)


def test_zero_magnitude_gives_flat_baseline_not_uniform_dopant():
    """The m -> 0 limit is a part uniformly doped at the baseline, NOT the
    uniform s = 1 baseline. The step-2 report names this as the reason the
    line search has no do-nothing option."""
    m = _mask()
    proxy = np.zeros(m.shape)
    proxy[m] = np.linspace(0.0, 1.0, int(m.sum()))
    sat = control.proportional_inverse_map(proxy, m, magnitude=0.0,
                                           baseline=0.5, dead_band=0.0,
                                           smoothing_sigma=0.0)
    assert np.allclose(sat[m], 0.5)


def test_dead_band_holds_mid_cells_at_baseline():
    m = _mask()
    proxy = np.zeros(m.shape)
    proxy[m] = np.linspace(0.0, 1.0, int(m.sum()))
    sat = control.proportional_inverse_map(proxy, m, magnitude=1.0,
                                           baseline=0.5, dead_band=0.20,
                                           smoothing_sigma=0.0)
    n_at_baseline = int(np.sum(np.isclose(sat[m], 0.5)))
    assert n_at_baseline > 0


def test_quantize_to_4bpp_snaps_to_sixteen_levels():
    sat = np.linspace(0.0, 1.0, 101)
    q = control.quantize(sat, bpp=4)
    assert np.all(np.isclose(q * 15.0, np.round(q * 15.0)))
    assert q.max() <= 1.0 and q.min() >= 0.0


def test_golden_section_finds_the_minimum_of_a_unimodal_function():
    calls = []

    def f(m):
        calls.append(m)
        return (m - 1.3) ** 2 + 0.5

    xs, ys = control.golden_section(f, 0.05, 2.5, n_evals=12)
    assert abs(xs[int(np.argmin(ys))] - 1.3) < 0.02
    assert len(calls) == 12


def test_selection_rule_ignores_infeasible_even_when_fit_is_lower():
    rows = [
        {"tag": "a", "feasible": True, "fit": 5.0, "holdout": 4.0},
        {"tag": "b", "feasible": False, "fit": 1.0, "holdout": 0.1},
        {"tag": "c", "feasible": True, "fit": 3.0, "holdout": 2.5},
    ]
    sel = control.select_on_fit(rows)
    assert sel["tag"] == "c"
    assert control.select_on_fit([rows[1]]) is None


def test_budget_accounting_charges_the_adjoint_its_measured_ratio():
    assert control.forward_equivalents(n_forward=3, n_gradient=0, ratio=0.9) == pytest.approx(3.0)
    assert control.forward_equivalents(n_forward=3, n_gradient=3, ratio=0.9) == pytest.approx(5.7)


def test_budget_cap_counts_evaluations_that_fit():
    assert control.max_gradient_evals(budget=5.0, ratio=0.9) == 2
    assert control.max_gradient_evals(budget=15.0, ratio=0.9) == 7
    assert control.max_gradient_evals(budget=40.0, ratio=0.9) == 21


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
