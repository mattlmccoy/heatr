"""Automatic drive calibration for an imported geometry.

THE CONVENTION. `FROZEN_CONVENTIONS_2D.md` Section 3 pins the drive to the
per-shape voltage at which the UNIFORM dopant arm absorbs 500 watts per metre
of depth at grid 120. The eighteen library shapes carry that number in their
stored configurations; an imported part does not, so the intake has to produce
it, and it has to produce the same number the campaign's own calibration would.

WHAT THE CONVENTION MEANS, MEASURED NOT ASSUMED. At the stored voltages the
state-B absorbed power (the electrical state the thermal march runs in from the
first `update_interval` tick onwards, which is the state most of the exposure
sees) is 500.00 W/m on all six configurations checked in this pass, while the
state-A power sits at 450 to 475 W/m. So the calibration target is state B,
summed over the doped mask, with the uniform saturation map.

WHY ONE SOLVE IS ENOUGH. The electro-quasi-static problem is linear in the
applied potential, and the conductivity and permittivity fields do not depend
on it, so the absorbed power is exactly quadratic in the drive:

    v_target = v_0 * sqrt(P_target / P(v_0))

That is `robust.recalibrated_voltage`, already in use for the grid hold-out,
and it is reused here rather than re-derived. The quadratic claim is not taken
on faith: `test_geometry_calibrate.py` measures P(3 kV) / P(1 kV) = 9.000000.
The second solve this module runs by default is a VERIFICATION of the result,
not an iteration towards it, and it can be switched off.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from . import forward as fwd, robust
from .pins import build_case

__all__ = ["Calibration", "uniform_absorbed_power", "calibrate_drive",
           "calibrate_intake", "TARGET_W_PER_M"]

TARGET_W_PER_M = 500.0


@dataclass(frozen=True)
class Calibration:
    voltage_v: float
    start_voltage_v: float
    p_at_start_w_per_m: float
    p_target_w_per_m: float
    p_verified_w_per_m: float | None
    n_solves: int
    electrical_state: str = "B"

    def as_json(self) -> dict:
        return {"voltage_v": float(self.voltage_v),
                "start_voltage_v": float(self.start_voltage_v),
                "p_at_start_w_per_m": float(self.p_at_start_w_per_m),
                "p_target_w_per_m": float(self.p_target_w_per_m),
                "p_verified_w_per_m": (None if self.p_verified_w_per_m is None
                                       else float(self.p_verified_w_per_m)),
                "n_electro_quasi_static_solves": int(self.n_solves),
                "electrical_state": self.electrical_state,
                "convention": "uniform dopant arm absorbs the target power, "
                              "state B, summed over the doped mask "
                              "(FROZEN_CONVENTIONS_2D section 3)"}


def _power_state_b(case) -> float:
    """Absorbed power of the UNIFORM arm in electrical state B, W per metre."""
    s = np.ones(case.part_mask.shape)
    eps = fwd.eps_field(case)
    sig_b, _inrange = fwd.sigma_state_b(case, s)
    st = fwd.solve_electric(case, sig_b, eps)
    return float(np.sum(st.Qrf[case.doped_mask]) * case.dA)


def uniform_absorbed_power(cfg: dict, voltage_v: float | None = None) -> float:
    """One electro-quasi-static solve; the uniform arm's absorbed power."""
    c = copy.deepcopy(cfg)
    if voltage_v is not None:
        c.setdefault("electric", {})["voltage_v"] = float(voltage_v)
    return _power_state_b(build_case(c))


def calibrate_drive(cfg: dict, target_w_per_m: float = TARGET_W_PER_M,
                    start_voltage_v: float | None = None,
                    verify: bool = True) -> Calibration:
    """The voltage at which the uniform arm absorbs the target power."""
    tgt = float(target_w_per_m)
    if tgt <= 0.0:
        raise ValueError(f"the calibration target must be positive, got {target_w_per_m!r}")
    v0 = (float(cfg["electric"]["voltage_v"]) if start_voltage_v is None
          else float(start_voltage_v))
    if v0 <= 0.0:
        raise ValueError(f"the starting voltage must be positive, got {v0!r}")
    p0 = uniform_absorbed_power(cfg, voltage_v=v0)
    if not np.isfinite(p0) or p0 <= 0.0:
        raise RuntimeError(
            f"the uniform arm absorbs {p0!r} W/m at {v0:.1f} V; the geometry or "
            "the material blocks are degenerate and no calibration exists")
    v = float(robust.recalibrated_voltage(v0, p0, tgt))
    p_ver = uniform_absorbed_power(cfg, voltage_v=v) if verify else None
    return Calibration(voltage_v=v, start_voltage_v=v0, p_at_start_w_per_m=p0,
                       p_target_w_per_m=tgt, p_verified_w_per_m=p_ver,
                       n_solves=2 if verify else 1)


def calibrate_intake(intake, target_w_per_m: float = TARGET_W_PER_M,
                     verify: bool = True):
    """Calibrate an `Intake` in place-by-copy, returning the calibrated one.

    The returned `Intake` carries the calibrated voltage in its config, so the
    solve, the rotation kernels and any engine run downstream all use the same
    drive. chi and the part mask do not depend on the drive and are reused
    unchanged, which is also the check that this step cannot move the geometry.
    """
    from .geometry_intake import Intake

    cal = calibrate_drive(intake.cfg, target_w_per_m=target_w_per_m, verify=verify)
    cfg = copy.deepcopy(intake.cfg)
    cfg["electric"]["voltage_v"] = float(cal.voltage_v)
    info = dict(intake.info)
    info["voltage_v"] = float(cal.voltage_v)
    info["voltage_is_calibrated"] = True
    info["calibration"] = cal.as_json()
    return Intake(geometry=intake.geometry, x=intake.x, y=intake.y,
                  chi=intake.chi, part_mask=intake.part_mask, cfg=cfg, info=info)
