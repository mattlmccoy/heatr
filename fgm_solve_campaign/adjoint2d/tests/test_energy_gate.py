"""Tests for the standing energy-residual gate.

`HEATR_STANDARD_PARAMETERS.md` names `|energy residual| / integrated dose` as the
standing gate on every solve, and `VERIFICATION_PRINTABILITY_REPORT.md`
Section 2.2 records that it was NOT wired into the prototype. These tests are the
red-first specification of wiring it in.

Two levels:

1. a pure-logic test of the ratio itself;
2. a CAPTURED-REAL-DATA contract test: the prototype's per-outer-step energy
   bookkeeping must reproduce the production engine's
   `hist["energy_balance_residual_J_per_m"]`, which is built at
   `rfam_eqs_coupled.py:3159` (incremental stored energy at beginning-of-outer-step
   material properties) and `rfam_eqs_coupled.py:3217`
   (`residual = e_in - e_out - e_stored`). Nothing is invented here; the
   reference numbers come from running the production engine.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from adjoint2d import energy_gate as eg
from adjoint2d import forward as fwd
from adjoint2d.pins import build_case, load_cfg
from adjoint2d.prod import rfam

MAIN = Path("/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research"
            "/research/binderjet/code/geo-prewarp")
CFG = MAIN / "outputs_eqs/geometry_dual_readstate/runs/square/baseline/used_config.yaml"


def test_relative_residual_is_residual_over_dose():
    e_in = np.array([0.0, 100.0, 200.0])
    e_out = np.array([0.0, 10.0, 20.0])
    e_stored = np.array([0.0, 85.0, 175.0])
    got = eg.relative_residual(e_in, e_out, e_stored)
    # step 0: dose 0 -> denominator floored at 1.0 J/m, residual 0
    # step 1: |100 - 10 - 85| / 100 = 0.05
    # step 2: |200 - 20 - 175| / 200 = 0.025
    assert got == pytest.approx([0.0, 0.05, 0.025])


def test_relative_residual_uses_absolute_value():
    got = eg.relative_residual(np.array([100.0]), np.array([10.0]), np.array([95.0]))
    assert got == pytest.approx([0.05])


def test_gate_verdict_flags_the_five_percent_threshold():
    assert eg.gate_verdict(0.049)["PASS"] is True
    assert eg.gate_verdict(0.050)["PASS"] is False
    assert eg.gate_verdict(0.31)["PASS"] is False


@pytest.mark.skipif(not CFG.exists(), reason="stored production config not present")
def test_forward_reproduces_production_energy_residual():
    """Contract test against the REAL production engine, not an invented fixture."""
    cfg = load_cfg(CFG)
    cfg["thermal"]["n_steps"] = 60
    case = build_case(cfg)
    s = np.ones(case.part_mask.shape, dtype=np.float64)

    cfg_run = json.loads(json.dumps(cfg, default=float))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _state, _summary, hist = rfam.run_sim(cfg_run)[:3]

    tr = fwd.forward(case, s, float32_sat=False, stop_after_phi=None, n_steps=60)

    ref_in = np.asarray(hist["energy_doped_J_per_m"], dtype=float)
    ref_out = (np.asarray(hist["energy_conv_loss_J_per_m"], dtype=float)
               + np.asarray(hist["energy_z_loss_J_per_m"], dtype=float))
    ref_res = np.asarray(hist["energy_balance_residual_J_per_m"], dtype=float)

    assert tr.energy_in_J_per_m.shape == ref_in.shape
    # 1e-9 relative is bit-level agreement for a 1e5 J/m quantity accumulated
    # over 60 outer steps; the prototype and the production engine differ only
    # in loop structure, not in arithmetic.
    assert np.max(np.abs(tr.energy_in_J_per_m - ref_in)) / max(abs(ref_in[-1]), 1.0) < 1e-12
    assert np.max(np.abs(tr.energy_out_J_per_m - ref_out)) / max(abs(ref_in[-1]), 1.0) < 1e-12
    got_res = tr.energy_in_J_per_m - tr.energy_out_J_per_m - tr.energy_stored_J_per_m
    assert np.max(np.abs(got_res - ref_res)) / max(abs(ref_in[-1]), 1.0) < 1e-12


@pytest.mark.skipif(not CFG.exists(), reason="stored production config not present")
def test_uniform_square_passes_the_five_percent_gate_early_in_the_march():
    cfg = load_cfg(CFG)
    cfg["thermal"]["n_steps"] = 60
    case = build_case(cfg)
    tr = fwd.forward(case, np.ones(case.part_mask.shape), stop_after_phi=None, n_steps=60)
    g = eg.gate_from_trajectory(tr, index=tr.n_outer - 1)
    assert g["rel_residual_at_index"] < 0.05
    assert g["PASS"] is True
