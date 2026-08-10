"""Pure-logic tests for the shape-library campaign fidelity pre-gate.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_shape_campaign.py -x -q

These cover the DECISION layer (grid selection + transfer_limited flag) and the
probe-field construction -- no dolfinx, no trimesh, runnable in either env. The
mesh/voxel bridge is validated separately against the PINNED pyramid reality
(n64=2.26 / n80=1.59 / n96=3.41 percent) in the harness driver, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import shape_campaign as sc


# --------------------------------------------------------------------------- #
# probe fields
# --------------------------------------------------------------------------- #
def _cube_centroids(n: int = 6) -> np.ndarray:
    g = np.linspace(-0.01, 0.01, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def test_probe_fields_are_the_named_basis_normalised_to_unit_range():
    fields = sc.build_probe_fields(_cube_centroids())
    assert set(fields) == {"xramp", "yramp", "zramp", "radial"}
    for name, f in fields.items():
        assert f.shape == (216,)
        assert abs(float(f.min()) - 0.0) < 1e-12, name
        assert abs(float(f.max()) - 1.0) < 1e-12, name


def test_axis_ramps_track_their_own_coordinate():
    c = _cube_centroids()
    fields = sc.build_probe_fields(c)
    # xramp must be monotone in x (rank correlation = 1 by construction)
    order = np.argsort(c[:, 0])
    assert np.all(np.diff(fields["xramp"][order]) >= -1e-12)


def test_uniform_centroid_field_would_be_degenerate_guarded():
    # a single-plane (all z equal) still normalises without div-by-zero
    c = _cube_centroids()
    c[:, 2] = 0.0
    fields = sc.build_probe_fields(c)
    assert np.all(np.isfinite(fields["zramp"]))
    assert float(fields["zramp"].max()) == 0.0  # no variation -> all zeros


# --------------------------------------------------------------------------- #
# worst-axis metric (radial excluded by design)
# --------------------------------------------------------------------------- #
def test_worst_axis_move_is_max_over_xyz_and_ignores_radial():
    moves = {"xramp": 0.012, "yramp": 0.006, "zramp": 0.045, "radial": 0.99}
    assert sc.worst_axis_move(moves) == pytest.approx(0.045)


# --------------------------------------------------------------------------- #
# grid selection + transfer_limited flag
# --------------------------------------------------------------------------- #
def _pyramid_like() -> dict:
    # worst-axis numbers MEASURED on the library pyramid geometry:
    #   n64 worst 0.0123, n80 worst 0.0058, n96 worst 0.0457
    return {
        64: {"xramp": 0.0001, "yramp": 0.0011, "zramp": 0.0123, "radial": 0.031},
        80: {"xramp": 0.0009, "yramp": 0.0058, "zramp": 0.0013, "radial": 0.029},
        96: {"xramp": 0.0019, "yramp": 0.0070, "zramp": 0.0457, "radial": 0.023},
    }


def test_selection_picks_the_argmin_worst_axis_grid_pyramid():
    r = sc.select_faithful_grid(_pyramid_like(), gate=0.02)
    assert r["chosen_grid"] == 80          # lowest worst-axis, the faithful grid
    assert r["faithful"] is True
    assert r["transfer_limited"] is False
    assert r["chosen_worst_axis_move"] == pytest.approx(0.0058)
    # non-monotone in n reproduced: n64 and n96 both worse than n80
    assert r["per_grid_worst_axis"][64] > r["per_grid_worst_axis"][80]
    assert r["per_grid_worst_axis"][96] > r["per_grid_worst_axis"][80]


def test_selection_flags_transfer_limited_when_no_grid_is_under_gate():
    bad = {
        64: {"xramp": 0.03, "yramp": 0.04, "zramp": 0.05, "radial": 0.06},
        80: {"xramp": 0.031, "yramp": 0.028, "zramp": 0.033, "radial": 0.05},
        96: {"xramp": 0.05, "yramp": 0.06, "zramp": 0.09, "radial": 0.11},
    }
    r = sc.select_faithful_grid(bad, gate=0.02)
    assert r["transfer_limited"] is True
    assert r["faithful"] is False
    # it still reports the LEAST-BAD grid as evidence (n80 here, 0.033)
    assert r["chosen_grid"] == 80
    assert r["chosen_worst_axis_move"] == pytest.approx(0.033)


def test_selection_ties_break_to_the_coarser_grid_for_compute():
    tie = {
        64: {"xramp": 0.010, "yramp": 0.010, "zramp": 0.010, "radial": 0.02},
        80: {"xramp": 0.010, "yramp": 0.010, "zramp": 0.010, "radial": 0.02},
    }
    r = sc.select_faithful_grid(tie, gate=0.02)
    assert r["chosen_grid"] == 64          # cheaper mesh wins an exact tie


def test_gate_threshold_is_strict_less_than():
    exact = {64: {"xramp": 0.02, "yramp": 0.0, "zramp": 0.0, "radial": 0.0}}
    r = sc.select_faithful_grid(exact, gate=0.02)
    assert r["faithful"] is False          # 2.00% is NOT under the 2% gate
    assert r["transfer_limited"] is True


# --------------------------------------------------------------------------- #
# reality-pin: the non-gating measurement must equal the production transfer's
# reported value on the PINNED pyramid (n64=2.26 / n80=1.59 / n96=3.41 percent,
# from verify_stage_b4_pyramid_heatr3d.json). Needs trimesh + the solved map, so
# it runs only in the .venv312 env and skips in the spike env.
# --------------------------------------------------------------------------- #
def test_dg0_voxel_mass_move_reproduces_pinned_pyramid():
    trimesh = pytest.importorskip("trimesh")     # noqa: F841 (.venv312 only)
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    mp = root / "solve3d" / "results" / "map_stage_b4_pyramid.npz"
    stl = root / "shape_library_3d" / "stl" / "pyramid.stl"
    if not mp.exists():
        pytest.skip("pyramid solved map not present")
    from studio3d.runner import voxelize_stl

    d = np.load(mp)
    cent, vals, vols = d["centroids"], d["s_map"], d["volumes"]
    pinned = {64: 0.0226, 80: 0.0159, 96: 0.0341}
    for n, want in pinned.items():
        part = voxelize_stl(str(stl), n, chamber_m=0.060)
        got = sc._dg0_voxel_mass_move(cent, vals, vols, part, 0.060)
        assert abs(got - want) < 5e-4, (n, got, want)
