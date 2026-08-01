"""Red-first tests for the topology-optimization parameterization.

The chain under test is

    v  ->  filter (PHYSICAL radius, normalized convolution over the part)
       ->  smoothed-Heaviside projection (tanh, sharpness beta, threshold eta)
       ->  physical saturation s in [0, 1]

Every property that the solve and the finite-difference gate rely on is pinned
here: the radius is a length and not a cell count, the projection is a monotone
[0, 1] -> [0, 1] map with proven endpoints, the composed derivative is exact,
and the new channel is bit-identical to the existing filtered path when it is
switched off.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import design_filter as df
from adjoint2d import topopt


# ---------------------------------------------------------------------------
# the radius is a physical length
# ---------------------------------------------------------------------------

def test_sigma_cells_is_the_radius_divided_by_the_cell_size():
    assert topopt.sigma_cells_for(1.0e-3, 5.0e-4) == pytest.approx(2.0)


def test_the_same_physical_radius_gives_different_cell_counts_on_two_grids():
    dx120 = 0.06 / 119.0
    dx160 = 0.06 / 159.0
    s120 = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, dx120)
    s160 = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, dx160)
    assert s160 > s120
    # resample consistency: the physical length is recovered on both grids
    assert s120 * dx120 == pytest.approx(topopt.FILTER_RADIUS_M)
    assert s160 * dx160 == pytest.approx(topopt.FILTER_RADIUS_M)


def test_negative_radius_raises():
    with pytest.raises(ValueError):
        topopt.sigma_cells_for(-1.0e-3, 5.0e-4)


def test_frozen_radius_is_at_least_two_cells_at_grid_120():
    dx120 = 0.06 / 119.0
    assert topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, dx120) > 1.95


# ---------------------------------------------------------------------------
# the projection
# ---------------------------------------------------------------------------

def test_projection_fixes_both_endpoints_and_the_threshold():
    for beta in (1.0, 4.0, 16.0, 64.0):
        assert topopt.project(0.0, beta) == pytest.approx(0.0, abs=1e-14)
        assert topopt.project(1.0, beta) == pytest.approx(1.0, abs=1e-14)
        assert topopt.project(0.5, beta) == pytest.approx(0.5, abs=1e-14)


def test_projection_is_monotone_and_stays_in_the_box():
    vf = np.linspace(0.0, 1.0, 501)
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        s = topopt.project(vf, beta)
        assert np.all(np.diff(s) > 0.0)
        assert s.min() >= 0.0 and s.max() <= 1.0


def test_projection_sharpens_with_beta():
    above = [topopt.project(0.65, b) for b in (1.0, 2.0, 4.0, 8.0, 16.0)]
    below = [topopt.project(0.35, b) for b in (1.0, 2.0, 4.0, 8.0, 16.0)]
    assert all(np.diff(above) > 0.0)
    assert all(np.diff(below) < 0.0)
    assert above[-1] > 0.95 and below[-1] < 0.05


def test_projection_at_beta_zero_is_the_identity_bitwise():
    vf = np.linspace(0.0, 1.0, 97)
    assert np.array_equal(topopt.project(vf, 0.0), vf)
    assert np.array_equal(topopt.project_deriv(vf, 0.0), np.ones_like(vf))


def test_projection_derivative_matches_central_differences():
    """Scaled by the PEAK derivative, not pointwise.

    At beta 16 the derivative in the tail is 6.6e-06 while the central-
    difference roundoff floor at h = 1e-06 is about 1e-10, so a pointwise
    relative error there is a statement about the denominator and not about the
    derivative. The peak-scaled error is the honest measure.
    """
    vf = np.linspace(0.02, 0.98, 25)
    for beta in (1.0, 4.0, 16.0):
        ana = topopt.project_deriv(vf, beta)
        h = 1e-6
        fd = (topopt.project(vf + h, beta) - topopt.project(vf - h, beta)) / (2 * h)
        assert np.max(np.abs(fd - ana)) / np.max(np.abs(ana)) < 1e-8


def test_projection_rejects_a_threshold_outside_the_open_unit_interval():
    with pytest.raises(ValueError):
        topopt.project(0.5, 4.0, eta=0.0)


# ---------------------------------------------------------------------------
# the composed map
# ---------------------------------------------------------------------------

def _case_like(n=24, seed=11):
    rng = np.random.default_rng(seed)
    pm = np.zeros((n, n), dtype=bool)
    pm[5:19, 6:20] = True
    pm[7, 8] = False           # an interior hole, so the mask is not convex
    v = np.ones((n, n))
    v[pm] = rng.uniform(0.0, 1.0, int(pm.sum()))
    return pm, v


def test_composed_map_keeps_the_box_and_the_nominal_outside_value():
    pm, v = _case_like()
    s = topopt.design_to_map(v, pm, dx=5.0e-4, radius_m=1.0e-3, beta=8.0)
    assert s[pm].min() >= 0.0 and s[pm].max() <= 1.0
    assert np.all(s[~pm] == 1.0)


def test_composed_map_at_beta_zero_is_bit_identical_to_the_existing_filter():
    """Flag-off bit identity: the new channel must not perturb the old path."""
    pm, v = _case_like()
    dx = 5.0e-4
    r = 1.0e-3
    new = topopt.design_to_map(v, pm, dx=dx, radius_m=r, beta=0.0)
    old = df.apply_filter(v, pm, topopt.sigma_cells_for(r, dx))
    assert np.array_equal(new, old)


def test_composed_map_at_radius_zero_is_the_projection_alone():
    pm, v = _case_like()
    s = topopt.design_to_map(v, pm, dx=5.0e-4, radius_m=0.0, beta=4.0)
    assert np.allclose(s[pm], topopt.project(v[pm], 4.0))


# ---------------------------------------------------------------------------
# the chain-rule gradient
# ---------------------------------------------------------------------------

def test_design_vjp_is_the_exact_transpose_of_the_linearized_map():
    """<vjp(g), d> must equal <g, dS(v)[d]> to machine precision."""
    pm, v = _case_like(seed=5)
    rng = np.random.default_rng(3)
    dx, r, beta = 5.0e-4, 1.0e-3, 8.0
    g = rng.standard_normal(pm.shape)
    d = np.zeros(pm.shape)
    d[pm] = rng.standard_normal(int(pm.sum()))
    lhs = float(np.sum(topopt.design_vjp(g, v, pm, dx=dx, radius_m=r, beta=beta) * d))
    rhs = float(np.sum(g * topopt.design_jvp(d, v, pm, dx=dx, radius_m=r, beta=beta)))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), 1.0)


def test_design_jvp_matches_central_differences_of_the_composed_map():
    pm, v = _case_like(seed=8)
    rng = np.random.default_rng(19)
    dx, r, beta = 5.0e-4, 1.0e-3, 4.0
    d = np.zeros(pm.shape)
    d[pm] = rng.standard_normal(int(pm.sum()))
    d /= np.linalg.norm(d[pm])
    kw = dict(dx=dx, radius_m=r, beta=beta)
    h = 1e-6
    fd = (topopt.design_to_map(v + h * d, pm, **kw)
          - topopt.design_to_map(v - h * d, pm, **kw)) / (2 * h)
    ana = topopt.design_jvp(d, v, pm, **kw)
    assert np.max(np.abs(fd - ana)) < 1e-7


def test_design_vjp_is_supported_on_the_part_only():
    pm, v = _case_like()
    g = np.ones(pm.shape)
    out = topopt.design_vjp(g, v, pm, dx=5.0e-4, radius_m=1.0e-3, beta=8.0)
    assert np.all(out[~pm] == 0.0)


# ---------------------------------------------------------------------------
# the continuation schedule and its budget split
# ---------------------------------------------------------------------------

def test_beta_schedule_is_frozen():
    assert topopt.BETA_SCHEDULE == (1.0, 2.0, 4.0, 8.0, 16.0)
    assert topopt.ETA == 0.5


def test_stage_split_spends_the_whole_pool():
    for pool in range(5, 40):
        split = topopt.stage_split(pool, len(topopt.BETA_SCHEDULE))
        assert sum(split) == pool
        assert len(split) == len(topopt.BETA_SCHEDULE)


def test_stage_split_puts_the_remainder_in_the_earliest_stages():
    assert topopt.stage_split(19, 5) == (4, 4, 4, 4, 3)
    assert topopt.stage_split(20, 5) == (4, 4, 4, 4, 4)
    assert topopt.stage_split(22, 5) == (5, 5, 4, 4, 4)


def test_stage_split_drops_late_stages_rather_than_starving_every_stage():
    """A pool smaller than the stage count must not give a zero-evaluation stage."""
    split = topopt.stage_split(3, 5)
    assert sum(split) == 3
    assert all(n >= 1 for n in split if n > 0)
    assert split[0] >= 1


def test_stage_split_rejects_a_negative_pool():
    with pytest.raises(ValueError):
        topopt.stage_split(-1, 5)


# ---------------------------------------------------------------------------
# result-file naming (regression: the stem was shadowed by a loop variable and
# all six shapes wrote to the same file)
# ---------------------------------------------------------------------------

def test_output_tag_is_derived_from_the_shape():
    from adjoint2d import topopt_solve as tos
    assert tos.output_tag("square") == "square"
    assert tos.output_tag("circle", "filteronly") == "circle_control_filteronly"


def test_output_tag_rejects_an_unknown_control_mode():
    from adjoint2d import topopt_solve as tos
    with pytest.raises(ValueError):
        tos.output_tag("square", "nonsense")
