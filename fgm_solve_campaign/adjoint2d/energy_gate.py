"""The standing energy-residual gate of `HEATR_STANDARD_PARAMETERS.md`.

The standard names `|energy residual| / integrated dose` as the gate that must be
reported on every solve, because it catches the silent melt-onset instability
that leaves the clip fractions clean. `VERIFICATION_PRINTABILITY_REPORT.md`
Section 2.2 recorded that the prototype did not report it. This module wires it
in.

Definition, reproduced from the production engine rather than re-derived:

  `rfam_eqs_coupled.py:3210-3217`
      e_in  = cumulative sum of p_doped * dt          (integrated dose)
      e_out = cumulative convective loss + depth loss
      residual = e_in - e_out - e_stored
  `rfam_eqs_coupled.py:3149-3163`
      e_stored is tracked INCREMENTALLY with beginning-of-outer-step material
      properties, sensible plus latent.

The reported quantity is `|residual| / max(|e_in|, 1 J/m)`, and the standing
threshold is the 5 percent of `test_energy_balance.py::test_c_full_run_residual`.
"""
from __future__ import annotations

import numpy as np

THRESHOLD = 0.05
DOSE_FLOOR_J_PER_M = 1.0


def relative_residual(e_in, e_out, e_stored) -> np.ndarray:
    """|e_in - e_out - e_stored| / max(|e_in|, 1 J/m), elementwise."""
    a = np.asarray(e_in, dtype=float)
    b = np.asarray(e_out, dtype=float)
    c = np.asarray(e_stored, dtype=float)
    return np.abs(a - b - c) / np.maximum(np.abs(a), DOSE_FLOOR_J_PER_M)


def gate_verdict(rel: float, threshold: float = THRESHOLD) -> dict:
    """PASS when the relative residual is strictly below the threshold."""
    r = float(rel)
    return {"rel_residual": r, "threshold": float(threshold),
            "PASS": bool(r < float(threshold))}


def gate_from_trajectory(tr, index: int, threshold: float = THRESHOLD) -> dict:
    """The gate evaluated AT the arm's own stop index, plus the running maximum.

    The stop index matters: every J, IoU, growth and under-melt number in this
    campaign is read at t_stop = argmin J, so the residual that could taint that
    number is the residual at that same index. The running maximum over the
    march up to the stop is also reported, because an instability that blows up
    and then partially self-corrects would otherwise hide.
    """
    rel = relative_residual(tr.energy_in_J_per_m, tr.energy_out_J_per_m,
                            tr.energy_stored_J_per_m)
    i = int(index)
    at = float(rel[i])
    upto = float(np.max(rel[: i + 1]))
    out = gate_verdict(at, threshold)
    out["rel_residual_at_index"] = at
    out["rel_residual_max_upto_index"] = upto
    out["rel_residual_max_whole_march"] = float(np.max(rel))
    out["index"] = i
    out["energy_in_J_per_m_at_index"] = float(tr.energy_in_J_per_m[i])
    out["energy_residual_J_per_m_at_index"] = float(
        tr.energy_in_J_per_m[i] - tr.energy_out_J_per_m[i] - tr.energy_stored_J_per_m[i])
    out["PASS_max_upto_index"] = bool(upto < float(threshold))
    return out
