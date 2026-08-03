"""GRID HOLD-OUT for ROTATING, INDEXED and DWELL-SCHEDULED arms.

`robust.py` and `robust_run.py` move a STATIC solved dopant map from the grid
it was solved on (120) to a hold-out grid (160) and re-score it. They cannot
touch any arm whose actuator turns the part, and the arms the dissertation
bolds as SOLVED are mostly of that class. This module is the missing half.

WHAT IS HELD OUT AND WHAT IS NOT. The dopant map is the only thing carried
across the grid. Everything else is REGENERATED at the hold-out grid:

  * the part mask, from the production domain builder at the new grid;
  * the target indicator chi, as the sub-cell AREA FILL of `chi_area`, which is
    grid independent to the sampling error it quotes. The binary raster target
    is NOT used as the primary objective here, because its own area moves with
    the grid and that confound is exactly what
    `SOLVE_ROBUSTNESS_VALIDATION.md` Section 3.2 could not separate. The binary
    raster J and the binary-raster intersection over union are reported
    alongside every arm, so the dissertation's grid-120 numbers stay quotable;
  * the per-angle rotated cases of the kernel, one production domain build per
    sampled turntable position at the new grid;
  * the drive voltage, recalibrated at the new grid by the campaign's own
    convention (the UNIFORM arm absorbs 500 watts per metre of depth in
    electrical state B, `FROZEN_CONVENTIONS_2D` Section 3), by one
    electro-quasi-static solve and one exact quadratic rescale, verified by a
    second solve.

The stored turntable PROGRAM is a wall-clock object (a list of
`{position_deg, dwell_s, move_at_s}`) and is carried across unchanged. That is
the point: the machine would run the same program whatever the simulation grid,
so the program is part of the arm, not part of the discretization.

HOW THE ARM IS EXECUTED. Through `dwell_march.program_forward`, in the PART
frame, which never interpolates a field and therefore carries none of the
rotation remap error the production engine's lab-frame turntable carries
(`CONTINUOUS_ROTATION_REPORT.md` Section 6.1). The per-position part-frame
heating fields come from `DwellKernel.averaged_Q`, so the design map is
co-rotated at every position: the map is rotated into the lab frame at that
angle, the electro-quasi-static problem is solved there against the part
rasterized at that angle, and the heating is rotated back. Co-rotation is not
optional; Section 6.3 of that report measured that WITHOUT it a graded map is
inert under rotation.

WHAT THIS IS NOT. It is not the production engine, and it does not re-solve
anything. Every arm here is a FORWARD run with only the stop re-optimized.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import chi_area, energy_gate as eg, forward as fwd
from . import geometry_calibrate as gcal
from . import printability as pq
from . import robust as rb
from . import topopt_objective as tobj
from .dwell_kernel import DwellKernel
from .dwell_march import program_forward
from .pins import build_case
from .rot_kernel import AveragedKernel

__all__ = [
    "transfer_map", "index_program_positions", "moves_to_positions",
    "recalibrate_at_grid", "cfg_at_grid", "build_rot_case", "RotCase",
    "score_program", "score_quasistatic", "score_static",
]

TARGET_W_PER_M = 500.0
BPP = 4
PATIENCE = 250


# ---------------------------------------------------------------------------
# 1. the map transfer, part frame
# ---------------------------------------------------------------------------

def transfer_map(sat_solved: np.ndarray, part_mask: np.ndarray,
                 bpp: int | None = BPP, outside: float = 1.0) -> np.ndarray:
    """Move a solved PART-FRAME map onto the hold-out grid's part mask.

    Resampling is the production map-injection convention reproduced in
    `robust.resample_map` (`scipy.ndimage.zoom` order 1, then clip, with the
    production 1 percent dead band). Outside the part the nominal saturation is
    held at `outside`, which is the convention `rot_kernel.lab_map` assumes when
    it rotates the map into the lab frame, so the transfer changes the dopant
    map and never the sub-pixel geometry fill of a boundary cell. Inside the
    part the resampled map is re-quantized onto the printer's level grid,
    because a bilinearly resampled 4-bits-per-pixel map is no longer on it.
    """
    pm = np.asarray(part_mask, dtype=bool)
    s = rb.resample_map(np.asarray(sat_solved, dtype=float),
                        pm.shape[0], pm.shape[1])
    if bpp is None:
        return np.where(pm, s, float(outside))
    return pq.quantize_in_part(s, pm, bpp=int(bpp), sat_max=1.0, outside=outside)


# ---------------------------------------------------------------------------
# 2. the program, which the grid must not touch
# ---------------------------------------------------------------------------

def index_program_positions(n_positions: int, interval_s: float, dt_s: float,
                            n_steps: int) -> np.ndarray:
    """Per-outer-step position index of a fixed-interval indexing schedule.

    The part advances one position every `interval_s` seconds and holds. This
    is `CONTINUOUS_ROTATION_REPORT.md` Section 7's actuator: a quarter turn
    every 2.0 s at the pinned 0.5 s outer step is four outer steps per position.
    """
    k = int(round(float(interval_s) / float(dt_s)))
    if k < 1:
        raise ValueError(
            f"the indexing interval {interval_s} s is shorter than one outer "
            f"step {dt_s} s; the program cannot be resolved")
    if abs(float(interval_s) / float(dt_s) - k) > 1e-9:
        raise ValueError(
            f"the indexing interval {interval_s} s is not an integer number of "
            f"{dt_s} s outer steps")
    return (np.arange(int(n_steps)) // k) % int(n_positions)


def moves_to_positions(moves: Sequence[dict], angles_deg: Sequence[float],
                       dt_s: float, n_steps: int,
                       tol_deg: float = 1e-6) -> np.ndarray:
    """Expand a stored `{position_deg, dwell_s, move_at_s}` program.

    `dwell_march.program_step_positions` snaps each commanded position to the
    NEAREST candidate angle silently. That is the wrong behaviour for a hold-out
    whose whole job is to execute the stored program exactly, so this refuses a
    position that is not in the angle set.
    """
    ang = np.asarray(angles_deg, dtype=float).ravel()
    dt = float(dt_s)
    out = np.zeros(int(n_steps), dtype=int)
    last = 0
    end_idx = 0
    for m in moves:
        p = float(m["position_deg"]) % 360.0
        d = np.abs(((ang % 360.0) - p + 180.0) % 360.0 - 180.0)
        j = int(np.argmin(d))
        if d[j] > float(tol_deg):
            raise ValueError(
                f"commanded position {m['position_deg']} degrees is not in the "
                f"candidate angle set {list(ang)}; nearest is {ang[j]} degrees, "
                f"off by {d[j]} degrees")
        i0 = int(round(float(m["move_at_s"]) / dt))
        i1 = int(round((float(m["move_at_s"]) + float(m["dwell_s"])) / dt))
        if i0 >= int(n_steps):
            break
        out[i0:min(i1, int(n_steps))] = j
        last = j
        end_idx = max(end_idx, min(i1, int(n_steps)))
    if end_idx < int(n_steps):
        out[end_idx:] = last
    return out


# ---------------------------------------------------------------------------
# 3. the case at the hold-out grid
# ---------------------------------------------------------------------------

def cfg_at_grid(cfg: dict, n_grid: int) -> dict:
    c = copy.deepcopy(cfg)
    c["geometry"]["grid_nx"] = int(n_grid)
    c["geometry"]["grid_ny"] = int(n_grid)
    return c


def recalibrate_at_grid(cfg: dict, n_grid: int,
                        target_w_per_m: float = TARGET_W_PER_M) -> dict:
    """The drive at which the UNIFORM arm absorbs the target power AT `n_grid`.

    One electro-quasi-static solve at the pinned drive, one exact quadratic
    rescale, one verification solve. No march, so this costs a few seconds even
    at grid 160.
    """
    c = cfg_at_grid(cfg, n_grid)
    cal = gcal.calibrate_drive(c, target_w_per_m=float(target_w_per_m),
                               verify=True)
    out = cal.as_json()
    out.update({"n_grid": int(n_grid),
                "voltage_v_pinned": float(cfg["electric"]["voltage_v"]),
                "p_at_pinned_v_w_per_m": float(cal.p_at_start_w_per_m)})
    return out


@dataclass
class RotCase:
    """Everything an arm needs at one grid, built once and shared by every arm."""

    n_grid: int
    cfg: dict
    case: object
    chi: np.ndarray
    chi_info: dict
    calibration: dict
    kernel: DwellKernel
    angles_deg: np.ndarray
    raster_vs_area: dict

    @property
    def part_mask(self) -> np.ndarray:
        return self.case.part_mask


def build_rot_case(cfg: dict, n_grid: int, angles_deg: Sequence[float],
                   recalibrate: bool = True,
                   target_w_per_m: float = TARGET_W_PER_M) -> RotCase:
    """Regenerate the part mask, chi, the drive and the per-angle kernel."""
    c = cfg_at_grid(cfg, n_grid)
    cal = (recalibrate_at_grid(cfg, n_grid, target_w_per_m) if recalibrate
           else {"voltage_v": float(cfg["electric"]["voltage_v"]),
                 "n_grid": int(n_grid), "recalibrated": False})
    c["electric"]["voltage_v"] = float(cal["voltage_v"])
    case = build_case(c)
    chi, info = chi_area.chi_from_cfg(c, case.x, case.y)
    kern = DwellKernel.build(c, angles=np.asarray(angles_deg, dtype=float))
    return RotCase(
        n_grid=int(n_grid), cfg=c, case=case, chi=np.asarray(chi, dtype=float),
        chi_info=info, calibration=cal, kernel=kern,
        angles_deg=np.asarray(angles_deg, dtype=float),
        raster_vs_area=chi_area.raster_vs_area_delta(case.part_mask, chi,
                                                     case.dx, case.dy))


# ---------------------------------------------------------------------------
# 4. scoring
# ---------------------------------------------------------------------------

def _finish(tr, case, chi: np.ndarray, s: np.ndarray, t0: float) -> dict:
    m = tobj.full_metrics(tr, case, chi)
    i = int(m["t_stop_index"])
    m["J_raster_chi"] = tobj.optimal_stop(
        tr, case, case.part_mask.astype(float)).J
    m["IoU_at_raster_stop"] = float(tobj.metrics(
        tr.T_at_end(int(tobj.optimal_stop(
            tr, case, case.part_mask.astype(float)).index)), case, chi)["IoU"])
    m["P_abs_W_per_m"] = float(tr.P_abs_B)
    m["P_abs_A_W_per_m"] = float(tr.P_abs_A)
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    m["max_T_at_stop_c"] = float(np.max(tr.T_at_end(i)))
    m["over_ceiling_250c"] = bool(m["max_T_at_stop_c"] > 250.0)
    m["frac_dT_clipped_max"] = float(tr.frac_dT_clipped_max)
    m["frac_temp_cap_max"] = float(tr.frac_temp_cap_max)
    m["n_outer"] = int(tr.n_outer)
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_min_in_part"] = float(np.min(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    m["wall_s"] = time.perf_counter() - t0
    # The melt field AT THIS ARM'S OWN STOP, for the figures. Popped by the
    # driver before the result is serialized; it is not part of the record.
    m["_phi_at_stop"] = tobj.phi_of(tr.T_at_end(i), case)
    return m


def score_program(rc: RotCase, s: np.ndarray, pos_index: np.ndarray,
                  n_steps: int | None = None) -> dict:
    """TIME-RESOLVED execution of a turntable program, part frame.

    `DwellKernel.averaged_Q` is called first because `program_forward` needs the
    per-position part-frame heating fields; the weighted average it also returns
    is not used by the march, which selects a stored field per outer step.
    """
    t0 = time.perf_counter()
    n = int(rc.case.pins.n_steps if n_steps is None else n_steps)
    rc.kernel.set_weights(np.full(rc.kernel.n_angles, 1.0 / rc.kernel.n_angles))
    rc.kernel.averaged_Q(np.asarray(s, dtype=float))
    tr = program_forward(rc.kernel, np.asarray(pos_index, dtype=int), n,
                         shape_stop_patience=PATIENCE)
    m = _finish(tr, rc.case, rc.chi, s, t0)
    m["executed_dwell_fraction"] = [float(v) for v in tr.realized_weights_executed]
    m["execution"] = "time-resolved program, part frame, no field interpolation"
    return m


def score_quasistatic(rc: RotCase, s: np.ndarray, weights=None) -> dict:
    """The weighted angle-average limit, the model the arms were SOLVED against."""
    t0 = time.perf_counter()
    if weights is None:
        kern = AveragedKernel(case0=rc.kernel.case0, angles=rc.kernel.angles,
                              per_angle=rc.kernel.per_angle)
        tr = kern.forward(np.asarray(s, dtype=float),
                          shape_stop_patience=PATIENCE)
    else:
        rc.kernel.set_weights(weights)
        tr = rc.kernel.forward(np.asarray(s, dtype=float),
                               shape_stop_patience=PATIENCE)
    m = _finish(tr, rc.case, rc.chi, s, t0)
    m["execution"] = "quasi-static angle average (infinitely fast cycle)"
    return m


def score_static(rc: RotCase, s: np.ndarray, eps_covary: bool = False) -> dict:
    """The non-rotating comparator on the SAME grid, chi and drive."""
    t0 = time.perf_counter()
    tr = fwd.forward(rc.case, np.asarray(s, dtype=float), stop_after_phi=None,
                     shape_stop_patience=PATIENCE, eps_covary=eps_covary)
    m = _finish(tr, rc.case, rc.chi, s, t0)
    m["execution"] = "static, part at 0 degrees, no rotation"
    m["eps_covary"] = bool(eps_covary)
    return m


def log_row(tag: str, name: str, m: dict) -> None:
    print(f"[{tag}] {name:26s} J {m['J']:9.2f}  IoU {m['IoU']:.4f}  "
          f"IoUa {m['IoU_area']:.4f}  grow {m['bed_melt_pct_of_part']:6.2f}  "
          f"under {m['part_under_melt_pct']:6.2f}  "
          f"stop {m['t_stop_index']:4d} ({m['t_stop_s']:6.1f} s)"
          f"{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
          f"P {m['P_abs_W_per_m']:6.1f}  "
          f"Eres {m['energy_gate']['rel_residual_at_index'] * 100:.2f}%  "
          f"Jrast {m['J_raster_chi']:9.2f}  {m['wall_s']:.1f} s", flush=True)
