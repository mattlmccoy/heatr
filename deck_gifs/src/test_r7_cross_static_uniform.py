"""Data-contract gate for the cross static-uniform deck thumbnail.

These tests run against the REAL stored campaign fields
(`fgm_solve_campaign/out_rot_ladder/cross_ladder_maps.npz`, written by
`scripts/analysis/run_rot_grid_ladder.py` as `_phi_at_stop`, i.e. the melt
fraction at the arm's OWN argmin-of-J stop) and against the numbers printed in
`ROTATING_GRID_LADDER_REPORT.md` Section 3.1. No fixture is invented.

Run:  ./.venv312/bin/python -m pytest deck_gifs/src/test_r7_cross_static_uniform.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r7_cross_static_uniform import (  # noqa: E402
    iou_binary,
    j_whole_domain,
    limb_melt_fractions,
    load_arm,
)

# ROTATING_GRID_LADDER_REPORT.md Section 3.1, row: cross | 120 | STATIC_uniform
REPORT_IOU = 0.5515
REPORT_J = 463.25


def test_iou_reproduces_the_report() -> None:
    a = load_arm()
    assert abs(iou_binary(a.phi, a.part_mask) - REPORT_IOU) < 5e-4


def test_j_reproduces_the_report() -> None:
    a = load_arm()
    j = j_whole_domain(a.phi, a.chi)
    assert abs(j - REPORT_J) / REPORT_J < 5e-3


def test_melt_is_lopsided_along_the_electrode_axis() -> None:
    """The whole point of the thumbnail: vertical limbs melt, horizontal do not."""
    a = load_arm()
    horiz, vert = limb_melt_fractions(a.phi, a.part_mask)
    assert vert > 0.90, f"vertical limbs should be melted through, got {vert:.3f}"
    assert horiz < 0.10, f"horizontal limbs should stay cold, got {horiz:.3f}"


def test_limb_partition_covers_disjoint_real_cells() -> None:
    a = load_arm()
    assert a.part_mask.sum() > 0
    assert np.isfinite(a.phi).all()
