"""Two-sided per-node dopant actuator: let the design saturation exceed 1.0 to
BOOST conductivity in a starved region, not only PULL it below 1.0.

WHY (solve3d/phase_e/TAMPER_STUDY.md). The Tamper fails from a ~3x radial
deposited-power gradient: the RF field couples into the outer rim while the core
starves at <0.6x mean power. The one-sided actuator caps the saturation at 1.0
(= sigma_doped, the nominal single-pass loading), so it can only PULL dopant OUT
of the rim; it CANNOT feed the starved core above baseline. Two-sided grading
raises the per-node cap so the core can be boosted above the nominal loading,
directly flattening the spatial power gradient a scalar drive cannot touch.

THE CAP IS PHYSICAL, NOT INVENTED. The saturation s scales the nominal doped ink
dose. s=1.0 is the single-pass nominal loading (sigma_doped = 0.04 S/m). s>1.0 is
realized as MULTIPLE ink passes -- the rasterizer already reads "sat>1 => double
pass" (stl_compensation_tool/webapp/meteor_bridge.py, apply_sat_to_levels clips a
sat-scaled level to the bit-depth ceiling). MAX_SAT_DEFAULT_TWO_SIDED = 2.0 is
exactly ONE extra full pass (a double dose). Three independent reasons this cap
is defensible rather than arbitrary:

  1. PRINTING REALIZABILITY. sat multiplies rasterized ink levels; a double dose
     is the concrete, conservative multi-pass print operation. Higher caps
     (triple pass) are a parameter, not the default.
  2. CONDUCTIVITY GUARD. sigma(sat=2) = 0.08 S/m stays far under the sigma-
     coupling numerical clip (SIGMA_COUPLING_CLIP_HI * sigma_doped = 1.0 S/m), so
     a boosted node never rides the clip -- the subgradient stays live and the
     two-sided gradient is FD-gatable (test_two_sided.py).
  3. DOPANT PHYSICS. sigma_doped is already over-critical (> sigma* ~= 0.030 S/m,
     the FGM over-critical branch); above the percolation optimum the sigma gain
     per added dose is sublinear, so 2x is a defensible upper actuation rather
     than an unbounded knob.

THE DEFAULT IS ONE-SIDED. MAX_SAT_ONE_SIDED = 1.0 reproduces the existing box
exactly, so every B1-B4 result (square / cube / pyramid de-risk) stays
byte-identical unless two-sided is explicitly opted in via max_sat > 1.0.
"""
from __future__ import annotations

MAX_SAT_ONE_SIDED: float = 1.0            # default: saturation capped at nominal doped
MAX_SAT_DEFAULT_TWO_SIDED: float = 2.0    # one extra full ink pass (double dose)


def design_bounds(max_sat: float = MAX_SAT_ONE_SIDED) -> tuple[float, float]:
    """The L-BFGS-B box for the design saturation: lower 0.0 (fully virgin),
    upper `max_sat`. max_sat = 1.0 reproduces the one-sided box exactly."""
    max_sat = float(max_sat)
    if max_sat < MAX_SAT_ONE_SIDED:
        raise ValueError(
            f"max_sat must be >= {MAX_SAT_ONE_SIDED} (the one-sided floor); "
            f"pulling dopant below baseline is the lower bound 0.0, not the cap. "
            f"got {max_sat}")
    return (0.0, max_sat)


def is_two_sided(max_sat: float) -> bool:
    """True when the cap opens the above-baseline (boost) branch."""
    return float(max_sat) > MAX_SAT_ONE_SIDED
