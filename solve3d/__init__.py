"""solve3d -- dolfinx forward/adjoint stack for solved volumetric dopant fields.

Phase A scope (docs/superpowers/specs/2026-07-31-solve-port-3d-design.md sec 5):
forward parity only. `forward.py` holds the complex EQS lift (from
heatr3d_d1_spike/eqs_common.py conventions, corrected/masked Q only) plus the
transient enthalpy thermal-phase march whose semantics are ported from
heatr3d.py `phase_update="enthalpy"`.

NOT in Phase A: adjoint, objective, regularization, eps_r channel, Studio.
"""
from __future__ import annotations

__all__ = ["forward", "gates", "cases"]
