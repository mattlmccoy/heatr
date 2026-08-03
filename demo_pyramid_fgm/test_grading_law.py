"""Red-green test for the heuristic 3-axis grading law (demo_pyramid_fgm.grading).

The law must: stay inside [0.20, 1.00] on part voxels, be exactly 0 outside the
part, and vary along ALL of x, y, and z (the demo's whole point).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_pyramid_fgm.grading import graded_sat, graded_sat_strong  # noqa: E402


def _toy_pyramid(n: int = 24) -> np.ndarray:
    """Solid square pyramid mask, apex +z, centered in a cubic grid."""
    part = np.zeros((n, n, n), dtype=bool)
    base = n // 8
    top = n - n // 8
    half0 = n // 3
    for k in range(base, top):
        f = 1.0 - (k - base) / max(top - base - 1, 1)
        half = max(int(round(half0 * f)), 0)
        if half == 0:
            part[n // 2, n // 2, k] = True
        else:
            part[n // 2 - half:n // 2 + half, n // 2 - half:n // 2 + half, k] = True
    return part


def test_bounds_and_outside_zero() -> None:
    part = _toy_pyramid()
    sat = graded_sat(part)
    assert sat.shape == part.shape
    assert np.all(sat[~part] == 0.0)
    inside = sat[part]
    assert inside.min() >= 0.20 - 1e-12
    assert inside.max() <= 1.00 + 1e-12


def test_varies_along_every_axis() -> None:
    part = _toy_pyramid()
    sat = graded_sat(part)
    for ax in range(3):
        prof = []
        for i in range(part.shape[ax]):
            sl = np.take(part, i, axis=ax)
            if sl.any():
                prof.append(float(np.take(sat, i, axis=ax)[sl].mean()))
        prof = np.asarray(prof)
        assert np.ptp(prof) > 0.02, f"no variation along axis {ax}"


def test_core_higher_than_skin_and_base_higher_than_apex() -> None:
    part = _toy_pyramid()
    sat = graded_sat(part)
    n = part.shape[0]
    ks = np.where(part.any(axis=(0, 1)))[0]
    k_lo = ks[len(ks) // 4]
    core = sat[n // 2, n // 2, k_lo]
    row = np.where(part[:, n // 2, k_lo])[0]
    skin = sat[row[0], n // 2, k_lo]
    assert core > skin, "core should carry more dopant than the skin"
    zprof = [float(sat[:, :, k][part[:, :, k]].mean()) for k in ks]
    assert zprof[0] > zprof[-1], "base should carry more dopant than the apex"


def test_strong_law_bounds_and_outside_zero() -> None:
    part = _toy_pyramid()
    sat = graded_sat_strong(part)
    assert np.all(sat[~part] == 0.0)
    inside = sat[part]
    assert inside.min() >= 0.10 - 1e-12
    assert inside.max() <= 1.00 + 1e-12


def test_strong_law_has_more_contrast_than_mild() -> None:
    part = _toy_pyramid()
    mild = graded_sat(part)[part]
    strong = graded_sat_strong(part)[part]
    assert np.ptp(strong) > np.ptp(mild), "strong law must span a wider range"
    ks = np.where(part.any(axis=(0, 1)))[0]

    def apex_over_base(sat3):
        base = float(sat3[:, :, ks[0]][part[:, :, ks[0]]].mean())
        apex = float(sat3[:, :, ks[-1]][part[:, :, ks[-1]]].mean())
        return apex / base

    s3 = graded_sat_strong(part)
    m3 = graded_sat(part)
    assert apex_over_base(s3) < apex_over_base(m3), \
        "strong law must cut the apex harder relative to the base"


if __name__ == "__main__":
    test_bounds_and_outside_zero()
    test_varies_along_every_axis()
    test_core_higher_than_skin_and_base_higher_than_apex()
    test_strong_law_bounds_and_outside_zero()
    test_strong_law_has_more_contrast_than_mild()
    print("ALL PASS")
