"""Tests for the shape-fidelity SOLVE entry point (scripts/solve_fgm.py).

Covers the pure logic first (red-green test-driven development):
  1. parsing of the new ``fgm_solve`` config block, and
  2. the production-format map emitter, round-tripped through the REAL
     production loader (``rfam_eqs_coupled._FgmFeedback.from_config``), which
     is the data-contract test: the emitted npz must inject without error and
     decode to the quantized saturation the solve delivered.

Production recipe under test (MULTISTART_REPORT.md / TOPOPT_REPORT.md verdicts,
FROZEN_CONVENTIONS_2D.md): filtered full-depth single start, 1.0 mm physical
filter radius, no smoothed-Heaviside projection by default (beta = 0),
conductivity-only channel, 4 bits per pixel output via the production
quantizer.
"""
from __future__ import annotations

import dataclasses

import pytest

from scripts.solve_fgm import FgmSolveConfig, parse_fgm_solve_block


# ---------------------------------------------------------------------------
# 1. fgm_solve config block parsing
# ---------------------------------------------------------------------------

class TestParseFgmSolveBlock:
    def test_defaults_from_empty_config(self):
        sc = parse_fgm_solve_block({})
        assert sc.budget_forward_equivalents == 40.0
        assert sc.filter_radius_mm == 1.0
        assert sc.warm_start == "auto"
        assert sc.eps_channel_model_only is False
        assert sc.bpp == 4

    def test_defaults_when_block_absent_key_present_elsewhere(self):
        cfg = {"electric": {"voltage_v": 3600.0}}
        sc = parse_fgm_solve_block(cfg)
        assert sc.budget_forward_equivalents == 40.0

    def test_explicit_values(self):
        cfg = {"fgm_solve": {
            "budget_forward_equivalents": 8.0,
            "filter_radius_mm": 1.5,
            "warm_start": "cold",
            "eps_channel_model_only": True,
        }}
        sc = parse_fgm_solve_block(cfg)
        assert sc.budget_forward_equivalents == 8.0
        assert sc.filter_radius_mm == 1.5
        assert sc.warm_start == "cold"
        assert sc.eps_channel_model_only is True

    def test_warm_start_path_is_kept_verbatim(self):
        cfg = {"fgm_solve": {"warm_start": "outputs_eqs/some/map_m0p5.npz"}}
        sc = parse_fgm_solve_block(cfg)
        assert sc.warm_start == "outputs_eqs/some/map_m0p5.npz"

    def test_config_is_immutable(self):
        sc = parse_fgm_solve_block({})
        with pytest.raises(dataclasses.FrozenInstanceError):
            sc.bpp = 2  # type: ignore[misc]

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError, match="budget"):
            parse_fgm_solve_block({"fgm_solve": {"budget_forward_equivalents": 0}})

    def test_rejects_non_positive_radius(self):
        with pytest.raises(ValueError, match="filter_radius_mm"):
            parse_fgm_solve_block({"fgm_solve": {"filter_radius_mm": -1.0}})

    def test_rejects_unknown_keys_loudly(self):
        with pytest.raises(ValueError, match="unknown fgm_solve key"):
            parse_fgm_solve_block({"fgm_solve": {"warmstart": "cold"}})

    def test_rejects_bad_bpp(self):
        with pytest.raises(ValueError, match="bpp"):
            parse_fgm_solve_block({"fgm_solve": {"bpp": 3}})

    def test_rejects_non_dict_block(self):
        with pytest.raises(ValueError, match="fgm_solve"):
            parse_fgm_solve_block({"fgm_solve": [1, 2]})

    def test_cli_budget_override(self):
        sc = parse_fgm_solve_block({"fgm_solve": {"budget_forward_equivalents": 40.0}},
                                   budget_override=8.0)
        assert sc.budget_forward_equivalents == 8.0

    def test_is_frozen_dataclass_type(self):
        assert dataclasses.is_dataclass(FgmSolveConfig)


# ---------------------------------------------------------------------------
# 2. production map emitter: key set + REAL production-loader round trip
# ---------------------------------------------------------------------------

# The exact key set np.savez_compressed writes at fgm_generator.py:679-696.
FGM_GENERATOR_NPZ_KEYS = {
    "level_map", "sat_map", "x_mm", "y_mm", "width_mm", "height_mm",
    "bpp", "n_levels", "magnitude", "baseline_saturation", "dead_band",
    "proxy_field", "invert", "dpi",
}


@pytest.fixture()
def synthetic_case():
    """A small simulation grid with a centered circular part mask."""
    np = pytest.importorskip("numpy")
    n = 60
    dx = 0.5e-3  # 0.5 mm cells: forces a real zoom to 720 dots per inch
    x = np.arange(n) * dx
    y = np.arange(n) * dx
    xx, yy = np.meshgrid(x - x.mean(), y - y.mean())
    part_mask = (xx**2 + yy**2) <= (10e-3) ** 2
    sat = np.where(part_mask, 0.5 + 0.4 * (xx / xx.max()), 1.0)
    sat = np.clip(sat, 0.0, 1.0)
    return x, y, part_mask, sat


class TestEmitProductionNpz:
    def test_key_set_matches_fgm_generator_emitter(self, tmp_path, synthetic_case):
        import numpy as np
        from scripts.solve_fgm import emit_production_npz
        x, y, part_mask, sat = synthetic_case
        p = emit_production_npz(sat, x, y, tmp_path / "m.npz", bpp=4,
                                run_name="unit")
        with np.load(p, allow_pickle=True) as d:
            assert set(d.files) == FGM_GENERATOR_NPZ_KEYS
            assert d["level_map"].dtype == np.uint8
            assert int(d["level_map"].max()) <= 15
            assert int(d["bpp"]) == 4
            assert int(d["n_levels"]) == 16
            # Defaults chosen so the resimulate injection path applies NO
            # rescale (rfam_eqs_coupled.py:383-388 rescales only when
            # magnitude != 1.0 or baseline != 0.5).
            assert float(d["magnitude"]) == 1.0
            assert float(d["baseline_saturation"]) == 0.5

    def test_real_production_loader_reads_emitted_npz(self, tmp_path,
                                                      synthetic_case):
        """Data contract: the REAL injection path must decode the emitted map.

        Uses rfam_eqs_coupled._FgmFeedback.from_config, the exact function the
        engine calls at run start (rfam_eqs_coupled.py:302-398), not a copy.
        """
        import numpy as np
        import rfam_eqs_coupled as rfam
        from fgm_solve_campaign.adjoint2d import printability as pq
        from scripts.solve_fgm import emit_production_npz

        x, y, part_mask, sat = synthetic_case
        s_q = pq.quantize_in_part(sat, part_mask, bpp=4, sat_max=1.0)
        p = emit_production_npz(s_q, x, y, tmp_path / "m.npz", bpp=4,
                                run_name="unit")
        cfg = {"fgm_feedback": {"enabled": True,
                                "saturation_map_npz": str(p)}}
        fb = rfam._FgmFeedback.from_config(cfg, x, y, part_mask)
        assert fb.enabled
        assert fb.sat_map.shape == (len(y), len(x))
        # The loader must recover exactly the printer round trip of the
        # quantized map (printability.load_level_map reproduces the loader,
        # proven in its module docstring; here we check the real one).
        with np.load(p) as d:
            expected = pq.load_level_map(d["level_map"], len(x), len(y), bpp=4)
        assert np.allclose(fb.sat_map, expected, atol=1e-6)
        # And the round trip must stay within one printer level quantum of
        # the delivered map deep inside the part (edges blur under resample).
        from scipy.ndimage import binary_erosion
        core = binary_erosion(part_mask, iterations=3)
        assert float(np.max(np.abs(fb.sat_map[core] - s_q[core]))) <= 1.0 / 15.0
