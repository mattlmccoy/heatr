"""P0b (heatr3d revalidation): the melt-onset read state must be LOUD when it
falls back to the final timestep because mean melt fraction phi_bar never
crossed 0.90 within the horizon.

Behavior under test (heatr3d.py, end of run()):
    if T_phi90 is None:
        T_phi90 = T.copy()
That fallback must (a) keep numerical behavior identical, and (b) emit a
WARNING containing 'MELT-ONSET FALLBACK' so no downstream consumer can mistake
the final-step read for the melt-onset metric.

Run: ./.venv312/bin/python -m pytest test_heatr3d_melt_fallback.py -q
"""
from __future__ import annotations

import logging

import numpy as np

import heatr3d as H


def _tiny_never_reaches():
    """A short-horizon run guaranteed not to reach phi_bar=0.90 (1 s of heating)."""
    grid = H.Grid(n=12)
    p = H.Params()
    part = H.make_geometry(grid, "sphere", diam=0.020, zspan=0.020)
    res = H.run(grid, part, p, max_time_s=1.0)
    return part, res


def test_fallback_emits_loud_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="heatr3d"):
        part, res = _tiny_never_reaches()
    assert res.reached is False
    msgs = [r.getMessage() for r in caplog.records]
    assert any("MELT-ONSET FALLBACK" in m for m in msgs), (
        "phi_bar never crossed 0.90 but no MELT-ONSET FALLBACK warning was "
        f"logged; warnings seen: {msgs}")


def test_fallback_numerics_unchanged(caplog):
    """The flag is instrumentation only: T_phi90 is still the final T field,
    t_phi90_s is still NaN, sigma_T is still std over part voxels."""
    with caplog.at_level(logging.WARNING, logger="heatr3d"):
        part, res = _tiny_never_reaches()
    assert np.isnan(res.t_phi90_s)
    assert np.isfinite(res.sigma_T)
    assert res.sigma_T == float(res.T_phi90[part].std())
