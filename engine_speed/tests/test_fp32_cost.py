"""Tests for engine_speed.fp32_cost.

The module measures what the march would cost in float32 -- the only precision
an Apple-silicon GPU offers (see march_metal.py). Two things must hold for the
measurement to mean anything:

  1. FIDELITY. The float64 reduced march must reproduce heatr3d's own float64
     march on the same configuration. A precision-cost number from a kernel that
     is not the real kernel is worthless.
  2. The float32 arm must be measured, not asserted. These tests pin that the
     deviation is reported and that it lands far above the parity floor the
     numba gate holds.
"""
from __future__ import annotations

import numpy as np
import pytest

import heatr3d as h3

from engine_speed import fp32_cost as fc
from engine_speed.gate import FLOOR_RTOL


@pytest.fixture(scope="module")
def cfg():
    return fc.build_probe_config(n=24, n_steps=40)


def test_float64_reduced_march_matches_heatr3d(cfg):
    """FIDELITY GATE: the stand-in kernel is the real kernel."""
    rec = fc.fidelity_vs_heatr3d(cfg)
    assert rec["n_substeps"] >= 1
    assert rec["max_rel_dev_T"] <= 1e-12, rec
    assert rec["clamp_counts_match"] is True, rec


def test_float32_deviation_is_far_above_the_parity_floor(cfg):
    rec = fc.dtype_deviation(cfg)
    assert rec["max_rel_dev_T"] > 1e4 * FLOOR_RTOL, rec
    # float32 has ~7 decimal digits; a march that accumulates should be at or
    # above single-precision epsilon, not at the float64 floor.
    assert rec["max_rel_dev_T"] >= 1e-8, rec


def test_deviation_record_reports_the_discrete_decisions(cfg):
    rec = fc.dtype_deviation(cfg)
    for key in ("max_rel_dev_T", "max_rel_dev_phi", "n_enthalpy_branch_flips",
                "n_dT_clip_f64", "n_dT_clip_f32", "n_temp_clip_f64",
                "n_temp_clip_f32", "n_cells", "n_steps", "grid_n",
                # the melt-onset substep index is the reported quantity a
                # precision change is most likely to move: it is a threshold
                # crossing, so a 1e-6 field shift can move it a whole substep.
                "phi90_substep_f64", "phi90_substep_f32"):
        assert key in rec, key
    import json
    assert json.loads(json.dumps(rec)) == rec


def test_reduced_march_is_deterministic(cfg):
    a = fc.reduced_march(cfg, dtype=np.float64)
    b = fc.reduced_march(cfg, dtype=np.float64)
    assert np.array_equal(a["T"], b["T"])
