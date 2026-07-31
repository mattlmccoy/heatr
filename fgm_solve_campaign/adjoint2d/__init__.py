"""Finite-difference-gated 2-D adjoint prototype for RFAM dopant design.

RFAM = radio-frequency additive manufacturing. FGM = functionally graded
material. EQS = electro-quasi-static.

Layout
------
prod      import shim for the production engine `rfam_eqs_coupled`
eqs       vectorized EQS assembly + factorization (forward and transpose)
pins      configuration pins extracted from a production YAML config
forward   differentiable forward march, gated bit-identical to run_sim
adjoint   reverse-mode gradient of the read-state objectives
objective sigma_T definitions and read-state (fixed horizon / melt-onset) logic
control   the one-shot proportional-inverse map and its gain line search
"""

__all__ = [
    "prod",
    "eqs",
    "pins",
    "forward",
    "adjoint",
    "objective",
    "control",
]
