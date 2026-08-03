"""workbench_job: config -> Params mapping, ceilings, part building, segments.

The chained-segment equivalence gate (Tier 2 time snapshots) runs the REAL
solver at n=16 and must reproduce the single-march final temperature field.
"""
from __future__ import annotations

import numpy as np
import pytest

from heatr3d_workbench import workbench_job as WJ


def test_params_default_is_enthalpy_standard():
    p = WJ.params_from_cfg({})
    assert p.phase_update == "enthalpy"


def test_params_legacy_reveal():
    p = WJ.params_from_cfg({"phase_update": "apparent_cp"})
    assert p.phase_update == "apparent_cp"


def test_params_coupling_knobs_pass_through():
    p = WJ.params_from_cfg({"eqs_update_interval_s": 30.0,
                            "sigma_temp_coeff_per_K": -0.002})
    assert p.eqs_update_interval_s == 30.0
    assert p.sigma_temp_coeff_per_K == -0.002


def test_grid_ceiling_enforced():
    assert WJ.check_grid_n(96) == 96
    with pytest.raises(SystemExit, match="ceiling"):
        WJ.check_grid_n(128)          # EQS-only mode deferred; full runs capped
    with pytest.raises(SystemExit, match="ceiling"):
        WJ.check_grid_n(200)


def test_segment_plan():
    plan = WJ.segment_plan(total_s=100.0, k=4)
    assert plan == [(0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0)]
    assert WJ.segment_plan(100.0, 1) == [(0.0, 100.0)]


def test_build_part_library_shape():
    import heatr3d as H
    grid = H.Grid(n=24)
    part = WJ.build_part(grid, {"source": "library", "library_shape": "cube"})
    assert part.shape == (24, 24, 24)
    assert part.sum() > 0


def test_build_part_rejects_tier3_library_shape():
    import heatr3d as H
    grid = H.Grid(n=16)
    with pytest.raises(SystemExit, match="not a loadable part"):
        WJ.build_part(grid, {"source": "library", "library_shape": "open_cylinder"})


def test_engine_version_string():
    v = WJ.engine_version()
    assert v.startswith("heatr3d-")
    assert len(v) > len("heatr3d-")


def test_chained_segments_reproduce_single_march_final_T():
    """Tier 2 gate: K chained run() segments (T0_override + t_start_s) must
    reproduce the single march's final T field. n=16, short horizon."""
    import heatr3d as H
    grid = H.Grid(n=16)
    part = H.make_geometry(grid, "sphere", diam=0.028, zspan=0.028)
    p = H.Params(phase_update="enthalpy")
    single = H.run(grid, part, p, max_time_s=20.0, verbose=False)
    T_single = single.T_final

    T_prev = None
    snaps = []
    for (t0, t1) in WJ.segment_plan(20.0, 4):
        r = WJ.run_segment(grid, part, p, t0, t1, T_prev)
        T_prev = r.T_final
        snaps.append((t1, r.T_final.copy()))
    assert len(snaps) == 4
    dT = np.abs(T_prev - T_single)
    assert float(dT.max()) < 1e-6, f"chained-vs-single max |dT| = {dT.max()}"
