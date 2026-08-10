"""Pure, testable core of the Tamper drive-feasibility estimate.

FIRST-ORDER model (pre-phase-change linear scaling of temperature rise):
    T(a)   = Tamb + a*(T_read - Tamb)
    phi(a) = clip((T(a) - (t_pc - dt_pc/2)) / dt_pc, 0, 1)   # melt band
    peak(a)= Tamb + a*(peak1 - Tamb)

This is a PRIOR, not the verdict (ignores latent heat -> optimistic on fusion;
ignores stop-time re-optimization and the single-cell/mesh peak question). The
verdict comes from the transient drive sweep. Kept as a module so the phi law
is gated against solve3d.forward.phase_fraction (data contract).
"""
from __future__ import annotations

import numpy as np

TAMB_C = 50.0
TPC_C = 180.0
DTPC_C = 10.0
FLOOR = 0.85


def phi_at_drive(T_read: np.ndarray, a: float,
                 tamb: float = TAMB_C, tpc: float = TPC_C,
                 dtpc: float = DTPC_C) -> np.ndarray:
    """Melt fraction after scaling the temperature RISE by drive multiplier a."""
    T = tamb + a * (np.asarray(T_read, float) - tamb)
    return np.clip((T - tpc) / dtpc + 0.5, 0.0, 1.0)


def below_floor_fraction(T_read: np.ndarray, w: np.ndarray, a: float,
                         floor: float = FLOOR) -> float:
    """Volume-weighted fraction of nodes below the density floor at drive a."""
    phi = phi_at_drive(T_read, a)
    w = np.asarray(w, float)
    return float(w[phi < floor].sum() / w.sum())


def peak_at_drive(peak1_c: float, a: float, tamb: float = TAMB_C) -> float:
    return tamb + a * (peak1_c - tamb)
