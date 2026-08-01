"""Pure bookkeeping for the FGM four-arm rerun (no physics, no I/O).

Tested by test_fgm_arms.py. Kept separate from run_fgm_rerun.py so the arithmetic
that produces the reported percentages is unit-tested rather than inlined in a
script that takes an hour to execute.
"""
from __future__ import annotations

import math

# A march whose |energy_residual_frac| exceeds this is reported as failed.
# The rerank campaign's clean marches sat at ~1e-13.
ENERGY_RESIDUAL_TOL = 1e-6

# Below this magnitude a sigma_T change is reported as "null" rather than as a
# benefit or a penalty. 1 % of sigma_T is ~0.2-0.4 C on these shapes.
DEFAULT_NULL_BAND_PCT = 1.0


def benefit_pct(sigma_new: float, sigma_ref: float) -> float:
    """Signed percent change of sigma_T. Negative == more uniform == a benefit."""
    if not sigma_ref > 0.0:
        raise ValueError(f"reference sigma_T must be positive, got {sigma_ref!r}")
    return 100.0 * (float(sigma_new) - float(sigma_ref)) / float(sigma_ref)


def sign_verdict(published_pct: float, measured_pct: float,
                 band_pct: float = DEFAULT_NULL_BAND_PCT) -> str:
    """Does the SIGN of a published benefit survive re-measurement?

    Returns "survives" (same sign, magnitude outside the null band), "reverses"
    (opposite sign, outside the band) or "null" (|measured| inside the band, i.e.
    no effect either way).
    """
    if abs(measured_pct) < band_pct:
        return "null"
    same = (measured_pct < 0.0) == (published_pct < 0.0)
    return "survives" if same else "reverses"


def gate_pass(gates: dict) -> bool:
    """The standing gates on a single thermal march."""
    resid = float(gates["energy_residual_frac"])
    if math.isnan(resid) or abs(resid) > ENERGY_RESIDUAL_TOL:
        return False
    return (bool(gates["reached_phi90"])
            and not bool(gates["clamp_bound"])
            and not bool(gates["cfl_violated"]))
