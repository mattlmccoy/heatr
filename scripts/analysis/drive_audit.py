"""Drive-realizability accounting for HEATR / heatr3d runs.

Pure functions that convert recorded solver output into absorbed watts and the
coupling efficiency a 400-500 W generator would need to deliver them.

HEATR is 2.5D: Q_rf is solved on an (nx, ny) plane and the out-of-page extent is
``electric.effective_depth_m``.  Absorbed power is therefore

    P_abs [W] = sum(Q_rf[mask]) * dA * depth

which is exactly ``summary.json:integrated_power_doped_W_per_m * effective_depth_m``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "absorbed_power_w",
    "enforced_absorbed_power_w",
    "implied_coupling_pct",
    "gap_field_v_per_m",
]


def absorbed_power_w(
    qrf: np.ndarray,
    dA: float,
    mask: np.ndarray,
    effective_depth_m: float,
) -> float:
    """Total RF power absorbed in ``mask`` [W].

    Args:
        qrf: volumetric RF source [W/m^3] on the 2.5D plane.
        dA: cell area [m^2].
        mask: boolean selector (normally the doped mask).
        effective_depth_m: out-of-page extent of the 2.5D slab [m].

    Returns:
        Absorbed power in watts.
    """
    return float(np.sum(np.asarray(qrf)[np.asarray(mask, dtype=bool)]) * dA * effective_depth_m)


def enforced_absorbed_power_w(generator_power_w: float, transfer_efficiency: float) -> float:
    """Absorbed watts in enforced-generator-power mode.

    The solver renormalizes Q_rf to ``P_gen * eta / depth`` W per metre of depth
    (rfam_eqs_coupled.py:2291), so the total absorbed power is analytic.
    """
    return float(generator_power_w) * float(transfer_efficiency)


def implied_coupling_pct(absorbed_w: float, generator_power_w: float) -> float:
    """Coupling efficiency [%] implied if the generator delivered ``generator_power_w``."""
    if generator_power_w <= 0.0:
        raise ValueError("generator_power_w must be positive")
    return 100.0 * float(absorbed_w) / float(generator_power_w)


def gap_field_v_per_m(voltage_v: float, gap_m: float) -> float:
    """Nominal uniform electrode-gap field [V/m]."""
    if gap_m <= 0.0:
        raise ValueError("gap_m must be positive")
    return float(voltage_v) / float(gap_m)
