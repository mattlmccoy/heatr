"""Gate cases for the engine_speed parity suite.

Each case is a fully specified heatr3d configuration (Grid, Params, part mask,
optional sat map, run kwargs) that BOTH heatr3d.run and
engine_speed.march_fast.march_fast are driven with. The cases deliberately
include guard-TRIGGERING configurations, not only benign ones:

  * ``cfl_substep``   -- dt_s is pushed past CFL_SAFETY * dt_stable_thermal so
    Params.enforce_cfl actually engages the THM-03 powder-bed substepping
    (n_substeps_used > 1). Asserted to FIRE in both engines.
  * ``clamp_hot``     -- the absorbed-power density is raised until the THM-01
    per-step dT cap and/or the THM-02 temp_min/temp_max clamp BIND
    (clamp_bound True). Asserted to FIRE in both engines.

A gate that passes only where the guards are dormant is not evidence about the
guards, so those two cases carry an explicit ``must_fire`` field that the gate
harness checks before it looks at any field deviation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

import heatr3d as h3


@dataclass(frozen=True)
class Case:
    name: str
    grid: h3.Grid
    params: h3.Params
    part: np.ndarray
    sat: np.ndarray | None
    run_kwargs: dict[str, Any]
    # "cfl": n_substeps_used must be > 1; "clamp": clamp_bound must be True.
    must_fire: str | None = None
    description: str = ""


_BASE = h3.Params(phase_update="enthalpy")


def _tube_mask(grid: h3.Grid, r_out: float = 0.010, r_in: float = 0.005,
               zspan: float = 0.020) -> np.ndarray:
    """Hollow cylinder (a real hole, so the part mask is not simply connected
    in the (x,y) plane): tests the stencil at an INTERNAL as well as an external
    boundary, which a solid cube never exercises."""
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    return (r <= r_out) & (r >= r_in) & (np.abs(Z) <= zspan / 2.0)


def _graded_sat(grid: h3.Grid, part: np.ndarray) -> np.ndarray:
    """Smooth in-plane dopant-saturation gradient in [0.25, 1.0], zero outside
    the part -- the shape an FGM map takes, so gamma is spatially graded and the
    property blend inside the part is non-uniform at every voxel."""
    X, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    g = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(60.0 * X) * np.cos(45.0 * Y))
    sat = np.clip(g, 0.0, 1.0)
    sat[~part] = 0.0
    return sat


def build_cases(n: int = 32) -> list[Case]:
    grid = h3.Grid(n=n)
    cube = h3.make_geometry(grid, "square", diam=0.020, zspan=0.020)
    tube = _tube_mask(grid)
    densify_kw = dict(max_time_s=60.0, phi_target=0.90, densify=True,
                      stop_mean_rho=0.98)
    cases = [
        Case("uniform_cube", grid, _BASE, cube, None, dict(densify_kw),
             description="solid 20 mm square prism, uniform dopant, enthalpy scheme"),
        Case("graded_sat_cube", grid, _BASE, cube, _graded_sat(grid, cube),
             dict(densify_kw),
             description="same cube with a graded FGM saturation map"),
        Case("tube", grid, _BASE, tube, None, dict(densify_kw),
             description="hollow cylinder: internal + external mask boundaries"),
        # ---- guard-TRIGGERING cases ------------------------------------ #
        Case("cfl_substep", grid, replace(_BASE, dt_s=3.0), cube, None,
             dict(densify_kw), must_fire="cfl",
             description=("dt_s=3.0 s > CFL_SAFETY*dt_stable(n=32)=1.406 s, so "
                          "THM-03 substepping MUST engage (n_sub=3)")),
        Case("clamp_hot", grid,
             replace(_BASE,
                     power_density_w_per_m3=100.0 * _BASE.power_density_w_per_m3),
             cube, None, dict(densify_kw), must_fire="clamp",
             description=("100x absorbed-power density so the THM-01 dT cap and "
                          "the THM-02 temp_max clamp BIND")),
        # ---- branch-coverage cases (paths the five above never enter) ---- #
        Case("melt_onset_break", grid,
             replace(_BASE,
                     power_density_w_per_m3=20.0 * _BASE.power_density_w_per_m3),
             cube, None,
             dict(max_time_s=60.0, phi_target=0.90, densify=False,
                  stop_mean_rho=None),
             description=("densify=False + 20x power: mean phi crosses "
                          "phi_target, so the melt-onset read, t_phi90_s and "
                          "the append-then-break exit are all exercised")),
        Case("apparent_cp", grid, replace(_BASE, phase_update="apparent_cp"),
             cube, None, dict(densify_kw),
             description=("legacy apparent-cp phase update (the other supported "
                          "phase_update branch)")),
        Case("hot_top_convection", grid, _BASE, cube, None,
             dict(densify_kw, T0_override=_hot_top_field(grid)),
             description=("T0_override puts a 250 C band on the y_max electrode "
                          "plane so the top-face convection term -- and with it "
                          "energy_loss_j -- is O(1) instead of ~5e-4 J")),
    ]
    return cases


def _hot_top_field(grid: h3.Grid, peak_c: float = 250.0) -> np.ndarray:
    """Initial temperature field with a hot slab against the open top face.

    Without this, T on the y_max plane never leaves preheat_c and the
    convection sink integrates to ~5e-4 J over the whole run, i.e. the q_conv
    path is present but numerically dormant. Here it is O(1) J."""
    _, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    y_hi = grid.y[-1]
    band = np.exp(-((Y - y_hi) / (0.20 * grid.L)) ** 2)
    return 23.0 + (peak_c - 23.0) * band


CASE_BUILDERS: dict[str, Callable[[], list[Case]]] = {
    "n32": lambda: build_cases(32),
}
