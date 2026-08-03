#!/usr/bin/env python3
"""Tests for the thin two-sided continuous-sigma hook in rfam_eqs_coupled.

The per-node tuning driver needs to inject a CONTINUOUS, per-cell sigma map
that can exceed the uniform baseline (sat = sigma/sigma_d0 > 1) without the
bpp quantization or the [0,1] clamp used by the printable-FGM path.  It must
also keep eps_r geometry-only (Allison's part eps_r=20 is fixed; only sigma
varies), i.e. sat>1 must NOT inflate permittivity.

Run: ./.venv312/bin/python -m pytest test_pernode_hook.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfam_eqs_coupled import _FgmFeedback


def _write_sat(tmp_path: Path, sat: np.ndarray) -> str:
    p = tmp_path / "sat_direct.npz"
    np.savez_compressed(p, sat_map=sat.astype(np.float32))
    return str(p)


def test_direct_hook_allows_sat_above_one(tmp_path):
    """A continuous sat map with values >1 is loaded un-clipped (up to sat_max)."""
    n = 10
    x = np.linspace(0, 0.06, n)
    y = np.linspace(0, 0.06, n)
    pm = np.ones((n, n), dtype=bool)
    sat = np.full((n, n), 1.4, dtype=np.float32)   # sigma = 1.4*sigma_d0

    cfg = {
        "fgm_feedback": {
            "enabled": True,
            "sat_map_npz_direct": _write_sat(tmp_path, sat),
            "sat_max": 1.5,
        }
    }
    fb = _FgmFeedback.from_config(cfg, x, y, pm)
    assert fb.enabled
    assert fb.sat_map.max() == pytest.approx(1.4, abs=1e-4)   # NOT clipped to 1.0
    assert getattr(fb, "eps_geometry_only", False) is True


def test_direct_hook_clamps_to_sat_max(tmp_path):
    """sat above sat_max is clamped to sat_max; below 0 clamped to 0."""
    n = 8
    x = np.linspace(0, 0.06, n)
    y = np.linspace(0, 0.06, n)
    pm = np.ones((n, n), dtype=bool)
    sat = np.full((n, n), 1.4, dtype=np.float32)
    sat[0, 0] = -0.2

    cfg = {
        "fgm_feedback": {
            "enabled": True,
            "sat_map_npz_direct": _write_sat(tmp_path, sat),
            "sat_max": 1.2,
        }
    }
    fb = _FgmFeedback.from_config(cfg, x, y, pm)
    assert fb.sat_map.max() == pytest.approx(1.2, abs=1e-4)
    assert fb.sat_map.min() == pytest.approx(0.0, abs=1e-6)


def test_direct_hook_sigma_at_mask_two_sided(tmp_path):
    """sigma_at_mask returns sigma_d0*sat, exceeding sigma_d0 where sat>1."""
    n = 6
    x = np.linspace(0, 0.06, n)
    y = np.linspace(0, 0.06, n)
    pm = np.ones((n, n), dtype=bool)
    sat = np.full((n, n), 1.25, dtype=np.float32)   # -> sigma = 0.05

    cfg = {
        "fgm_feedback": {
            "enabled": True,
            "sat_map_npz_direct": _write_sat(tmp_path, sat),
            "sat_max": 1.5,
        }
    }
    fb = _FgmFeedback.from_config(cfg, x, y, pm)
    sigma_d0 = 0.04
    zeros = np.zeros((n, n))
    s = fb.sigma_at_mask(pm, sigma_d0, zeros, np.full((n, n), 0.55),
                         0.0, 0.0, 23.0, 0.55)
    assert np.allclose(s, sigma_d0 * 1.25)   # = 0.05, ABOVE sigma_d0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
