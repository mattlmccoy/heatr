"""solve3d Phase A analytic benchmarks (S1 ports) + environment smoke test.

RUNS IN THE SPIKE ENV (dolfinx 0.11 complex build):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_forward_analytic.py
"""
from __future__ import annotations

import numpy as np


def test_env_imports():
    """Task 0: solve3d.forward imports dolfinx (through jit_fix) and exposes the
    Phase A public API."""
    from solve3d import forward

    assert forward.DOLFINX_VERSION.startswith("0.")
    assert forward.IS_COMPLEX is True, "Phase A requires the complex scalar build"
    assert hasattr(forward, "ForwardParams")
    assert hasattr(forward, "run_forward")
    p = forward.ForwardParams()
    # mirrors heatr3d.Params defaults that Phase A depends on
    assert p.phase_update == "enthalpy"
    assert p.eqs_update_interval_s == 0.0
    assert p.sigma_temp_coeff_per_K == 0.0
    assert p.freq_hz == 27.12e6
    assert p.dt_s == 0.05
