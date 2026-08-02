"""Red-first tests for the arbitrary-geometry intake.

The requirement is that every feature the campaign developed (the target
indicator, the calibrated drive, the filtered solve, the dwell schedule) works
on ANY imported geometry and not only on the eighteen library shapes. That
starts here: three intake routes (binary mask array, polygon vertex list,
portable network graphics (PNG) mask file) that all land on the SAME two
objects the solve stack needs, the grid-independent sub-cell area-fill target
indicator chi and the production part mask, at a requested grid.

The analytic anchors are a circle (area known in closed form) and a rotated
rectangle (area is rotation invariant), and they are asserted through
`tests/fill_contract.py` so the three-dimensional lane can re-run the identical
statements against its volume fill.
"""
from __future__ import annotations

import numpy as np
import pytest

from adjoint2d import chi_area, geometry_intake as gi
from adjoint2d.tests import fill_contract as fc

HALF = 0.030
GRID = 120


def _axes(n: int = GRID):
    return np.linspace(-HALF, HALF, n), np.linspace(-HALF, HALF, n)


# --- the fill contract, run against the two-dimensional implementation -------

def test_fill_contract_circle_area():
    x, y = _axes()
    info = fc.assert_circle_area(gi.area_fill, x, y)
    assert abs(info["rel_error"]) < 1.0e-3


def test_fill_contract_rotated_rectangle_area():
    x, y = _axes()
    fc.assert_rotated_rect_area(gi.area_fill, x, y)


def test_fill_contract_winding_invariance():
    x, y = _axes()
    fc.assert_winding_invariance(gi.area_fill, x, y)


def test_fill_contract_grid_independence_120_to_160():
    fc.assert_grid_independence(gi.area_fill, _axes(120), _axes(160))


def test_area_fill_delegates_to_the_production_sampler_bit_for_bit():
    """One implementation only. A second copy of the fill would drift."""
    x, y = _axes()
    poly = fc.circle_polygon(0.010, 360)
    a = gi.area_fill(poly, x, y)
    b = chi_area.area_fill_poly(poly, x, y)
    assert np.array_equal(a, b)


# --- route (b): polygon vertex list ------------------------------------------

def test_intake_from_polygon_matches_the_production_part_mask():
    """chi and the part mask must come from the SAME geometry.

    The intake emits a production config; building the case from that config
    must reproduce the intake's own part mask exactly, or the solve would run
    against a different part than the target indicator describes.
    """
    from adjoint2d.pins import build_case

    poly = fc.circle_polygon(0.011, 220)
    it = gi.from_polygon(poly, grid=GRID)
    case = build_case(it.cfg)
    assert np.array_equal(case.part_mask, it.part_mask)
    assert it.chi.shape == (GRID, GRID)
    assert it.chi.min() >= 0.0 and it.chi.max() <= 1.0


def test_intake_from_polygon_area_is_the_closed_form_area():
    poly = fc.circle_polygon(0.011, 720)
    it = gi.from_polygon(poly, grid=GRID)
    dA = float(it.x[1] - it.x[0]) * float(it.y[1] - it.y[0])
    area = float(it.chi.sum()) * dA
    exact = fc.inscribed_ngon_area(0.011, 720)
    assert area == pytest.approx(exact, rel=1e-3)


def test_intake_refuses_a_polygon_that_leaves_the_chamber():
    poly = fc.circle_polygon(0.040, 64)
    with pytest.raises(gi.IntakeError, match="chamber"):
        gi.from_polygon(poly, grid=GRID)


# --- route (a): binary mask array ---------------------------------------------

def test_intake_from_mask_recovers_the_pixel_area():
    """A pixel block imported at a known pitch keeps its area."""
    m = np.zeros((80, 80), dtype=bool)
    m[20:60, 25:55] = True
    pitch = 2.5e-4
    it = gi.from_mask(m, pixel_pitch_m=pitch, grid=GRID)
    dA = float(it.x[1] - it.x[0]) * float(it.y[1] - it.y[0])
    area = float(it.chi.sum()) * dA
    assert area == pytest.approx(float(m.sum()) * pitch * pitch, rel=2e-3)


def test_intake_from_mask_can_be_scaled_by_target_width():
    m = np.zeros((60, 60), dtype=bool)
    m[10:50, 20:40] = True          # 40 rows by 20 columns
    it = gi.from_mask(m, part_width_m=0.020, grid=GRID)
    poly = it.geometry.polygons[0]
    assert (poly[:, 0].max() - poly[:, 0].min()) == pytest.approx(0.020, rel=1e-9)
    # the aspect ratio of the pixel region is preserved
    assert (poly[:, 1].max() - poly[:, 1].min()) == pytest.approx(0.040, rel=1e-9)


def test_intake_from_mask_is_centred_on_the_requested_centre():
    m = np.zeros((40, 40), dtype=bool)
    m[2:10, 2:10] = True            # deliberately off-centre in its own array
    it = gi.from_mask(m, part_width_m=0.010, grid=GRID)
    poly = it.geometry.polygons[0]
    assert 0.5 * (poly[:, 0].min() + poly[:, 0].max()) == pytest.approx(0.0, abs=1e-12)
    assert 0.5 * (poly[:, 1].min() + poly[:, 1].max()) == pytest.approx(0.0, abs=1e-12)


def test_intake_from_mask_requires_exactly_one_scale_argument():
    m = np.ones((8, 8), dtype=bool)
    with pytest.raises(gi.IntakeError, match="pixel_pitch_m|part_width_m"):
        gi.from_mask(m, grid=GRID)


# --- route (c): PNG mask file --------------------------------------------------

def test_intake_from_png_round_trips_a_written_mask(tmp_path):
    from PIL import Image

    m = np.zeros((64, 64), dtype=bool)
    m[16:48, 8:56] = True
    p = tmp_path / "mask.png"
    Image.fromarray(np.flipud(m).astype(np.uint8) * 255, "L").save(p)
    it = gi.from_png(p, part_width_m=0.024, grid=GRID)
    poly = it.geometry.polygons[0]
    assert (poly[:, 0].max() - poly[:, 0].min()) == pytest.approx(0.024, rel=1e-9)
    assert (poly[:, 1].max() - poly[:, 1].min()) == pytest.approx(0.016, rel=1e-9)


def test_png_row_order_is_flipped_so_the_image_top_is_the_physical_top(tmp_path):
    """PNG row 0 is the image TOP; field row 0 is the physical BOTTOM.

    `scripts/solve_fgm.emit_map_pngs` flips on the way out, so the intake must
    flip on the way in or an imported mask comes in upside down.
    """
    from PIL import Image

    m = np.zeros((32, 32), dtype=bool)
    m[20:30, 10:24] = True          # a block near the image TOP
    p = tmp_path / "top.png"
    Image.fromarray(np.flipud(m).astype(np.uint8) * 255, "L").save(p)
    it = gi.from_png(p, pixel_pitch_m=8.0e-4, grid=GRID, centre=False)
    poly = it.geometry.polygons[0]
    assert poly[:, 1].mean() > 0.0, "the imported block landed in the wrong half"


# --- the config the intake emits ------------------------------------------------

def test_emitted_config_is_a_polygon_part_the_engine_understands():
    poly = fc.rotated_rect_polygon(0.020, 0.010, 23.0)
    it = gi.from_polygon(poly, grid=GRID)
    part = it.cfg["geometry"]["part"]
    assert part["shape"] == "polygon"
    assert np.allclose(np.asarray(part["polygon_points"], dtype=float)[:len(poly)],
                       poly, atol=1e-15)
    assert it.cfg["geometry"]["grid_nx"] == GRID


def test_emitted_config_carries_the_frozen_drive_conventions():
    """FROZEN_CONVENTIONS_2D section 3: grounded mode, no generator-power branch."""
    it = gi.from_polygon(fc.circle_polygon(0.010, 128), grid=GRID)
    elec = it.cfg["electric"]
    assert elec["voltage_mode"] == "grounded"
    assert elec["enforce_generator_power"] is False
    assert "voltage_hi_v" not in elec and "voltage_lo_v" not in elec


def test_chi_from_the_emitted_config_equals_the_intake_chi():
    """The intake's chi must be what `chi_area.chi_from_cfg` would build."""
    poly = fc.rotated_rect_polygon(0.018, 0.009, 41.0)
    it = gi.from_polygon(poly, grid=GRID)
    chi2, _info = chi_area.chi_from_cfg(it.cfg, it.x, it.y)
    assert np.allclose(chi2, it.chi, atol=1e-15)


def test_the_layer_aggregation_contract_is_documented_in_the_api():
    """The layer-wise consumers need the aggregation rule stated, not implied."""
    text = gi.LAYER_AGGREGATION_CONTRACT.lower()
    for token in ("dose-weighted", "worst-layer", "per-build", "heuristic"):
        assert token in text, f"the aggregation contract does not state {token!r}"
    assert "dose-weighted" in gi.__doc__.lower()


# --- self-intersection, found by using the tool on a real novel outline --------

def test_a_self_intersecting_outline_is_refused_loudly():
    """A crossed outline is silently REINTERPRETED by the even-odd rule.

    Found in this pass: a keyhole built as a head arc followed by stem corners
    crossed itself where the two joined, and the even-odd fill turned the
    overlap into a notch cut out of the head. The intake accepted it and solved
    a part the user did not draw. Refusing is the only safe behaviour, because
    the reinterpretation is deterministic, plausible looking and wrong.
    """
    bowtie = np.array([[-0.008, -0.006], [0.008, 0.006],
                       [0.008, -0.006], [-0.008, 0.006]])
    with pytest.raises(gi.IntakeError, match="self-intersect"):
        gi.from_polygon(bowtie, grid=GRID)


def test_a_clean_outline_is_not_flagged_as_self_intersecting():
    for poly in (fc.circle_polygon(0.010, 360),
                 fc.rotated_rect_polygon(0.020, 0.008, 13.0)):
        gi.from_polygon(poly, grid=GRID)          # must not raise


def test_touching_vertices_are_not_a_crossing():
    """Two edges that share an endpoint are not an intersection."""
    star = np.array([[0.0, 0.010], [0.003, 0.003], [0.010, 0.0], [0.003, -0.003],
                     [0.0, -0.010], [-0.003, -0.003], [-0.010, 0.0], [-0.003, 0.003]])
    gi.from_polygon(star, grid=GRID)              # must not raise
