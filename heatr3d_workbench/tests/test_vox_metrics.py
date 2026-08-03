"""Voxel shape metrics: IoU / out-of-bounds melt / in-part melt / front distance.

Definitions ported from solve3d/shape_metrics.py (iou returns NaN on empty
union, never 1.0). Analytic fixtures first; then a captured-real-data check
against outputs_eqs/_heatr3d/8a16b9459f84/fields.npz when present.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from heatr3d_workbench import vox_metrics as VM

RUN = Path(__file__).resolve().parents[2] / "outputs_eqs" / "_heatr3d" / "8a16b9459f84"


def test_iou_identity_and_disjoint():
    a = np.zeros((4, 4, 4), bool); a[1:3, 1:3, 1:3] = True
    assert VM.iou(a, a) == 1.0
    b = np.zeros_like(a); b[0, 0, 0] = True
    assert VM.iou(a, b) == 0.0


def test_iou_empty_vs_empty_is_nan_not_one():
    z = np.zeros((3, 3, 3), bool)
    assert math.isnan(VM.iou(z, z))


def test_out_of_part_fraction():
    part = np.zeros((4, 4, 4), bool); part[:2] = True          # 32 voxels
    melt = part.copy(); melt[3, 0, 0] = True                    # 1 spill voxel
    assert VM.out_of_part_fraction(melt, part) == pytest.approx(1 / 32)
    assert VM.out_of_part_fraction(part, part) == 0.0


def test_in_part_melt_fraction():
    part = np.zeros((4, 4, 4), bool); part[:2] = True
    melt = np.zeros_like(part); melt[0] = True                  # half the part
    assert VM.in_part_melt_fraction(melt, part) == pytest.approx(0.5)


def test_shape_metrics_dict_contract():
    part = np.zeros((6, 6, 6), bool); part[1:5, 1:5, 1:5] = True
    phi = np.where(part, 0.95, 0.0)
    m = VM.shape_metrics(phi, part)
    for k in ("iou_phi80", "iou_phi90", "oob_melt_frac_phi80",
              "oob_melt_frac_phi90", "in_part_melt_frac_phi90"):
        assert k in m, k
    assert m["iou_phi90"] == 1.0
    assert m["oob_melt_frac_phi90"] == 0.0


def test_shape_metrics_no_melt_is_nan_iou_and_zero_oob():
    part = np.zeros((5, 5, 5), bool); part[2, 2, 2] = True
    phi = np.zeros((5, 5, 5))
    m = VM.shape_metrics(phi, part)
    assert m["iou_phi90"] == 0.0            # union nonempty (part exists)
    assert m["in_part_melt_frac_phi90"] == 0.0


def test_front_distance_zero_for_identical_masks():
    part = np.zeros((8, 8, 8), bool); part[2:6, 2:6, 2:6] = True
    phi = np.where(part, 1.0, 0.0)
    m = VM.shape_metrics(phi, part, h_mm=1.0)
    assert m["front_dist_phi90_mm"] == pytest.approx(0.0)


@pytest.mark.skipif(not (RUN / "fields.npz").exists(), reason="captured run absent")
def test_real_captured_run_metrics_are_finite_and_sane():
    z = np.load(RUN / "fields.npz")
    part = z["part"].astype(bool)
    phi = z["phi_final"].astype(float)
    m = VM.shape_metrics(phi, part, h_mm=float(z["h"]) * 1e3)
    assert 0.0 <= m["iou_phi90"] <= 1.0
    assert 0.0 <= m["in_part_melt_frac_phi90"] <= 1.0
    assert m["oob_melt_frac_phi90"] >= 0.0
