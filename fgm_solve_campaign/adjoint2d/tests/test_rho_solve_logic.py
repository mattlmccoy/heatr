"""RED-first tests for the pure logic of the density solve.

The gradient-death assertion is the operational half of the saturation guard:
the flat-onset read state keeps the objective off its saturated plateau, and
this rule catches the case where it saturates anyway, so the run stops and says
so instead of carving structure that buys nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import rho_solve as rs


def test_a_healthy_gradient_is_not_death():
    assert rs.gradient_death(1.0e-2, 1.0e-2) is False


def test_a_gradient_six_orders_below_the_first_iterate_is_death():
    assert rs.gradient_death(1.0e-9, 1.0e-2) is True


def test_the_boundary_is_inclusive_at_the_relative_floor():
    assert rs.gradient_death(1.0e-8, 1.0e-2, rel=1e-6) is True
    assert rs.gradient_death(1.1e-8, 1.0e-2, rel=1e-6) is False


def test_an_exactly_zero_gradient_is_death_even_on_the_first_iterate():
    assert rs.gradient_death(0.0, 0.0) is True


def test_the_absolute_floor_catches_a_run_that_starts_dead():
    """A first iterate that is already numerically zero must not set a floor of
    zero and then declare every later iterate healthy."""
    assert rs.gradient_death(1.0e-40, 1.0e-40) is True


def test_a_missing_first_iterate_falls_back_to_the_absolute_floor():
    assert rs.gradient_death(1.0e-3, None) is False
    assert rs.gradient_death(0.0, None) is True


def test_better_of_two_starts_picks_the_lower_objective():
    a = {"start": "cold", "J_rho": 12.0}
    b = {"start": "warm", "J_rho": 9.5}
    assert rs.better_start(a, b) is b
    assert rs.better_start(b, a) is b


def test_better_of_two_starts_tolerates_a_missing_arm():
    a = {"start": "cold", "J_rho": 12.0}
    assert rs.better_start(a, None) is a
    assert rs.better_start(None, a) is a
    assert rs.better_start(None, None) is None
