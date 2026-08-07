"""Continuous premix baseline dopant material law for the 2-D RFAM model.

Mirrors heatr3d.build_gamma premix semantics (heatr3d.py:591-624). premix_frac=0.0
returns the reference inline blend BIT-FOR-BIT. Premix is a swept forward parameter,
not a design variable, so the existing adjoint (which differentiates only w.r.t. the
printed dopant map) is unchanged -- no new gradient path.

See docs/superpowers/specs/2026-08-07-premix-baseline-dopant-2d-design.md
"""
from __future__ import annotations

import numpy as np

# Nominal label for the doped endpoint (rfam_eqs_coupled.py:333). NOT a measured
# value -- the doped sigma is an effective composite (~0.04 S/m operating point,
# RFAM Paper v1.3 p.542). Used only to label the premix_frac display axis.
PREMIX_WTPCT_FULL = 25.0


def apply_premix(blend_sigma, blend_eps, *, sigma_v, sigma_d0, eps_v, eps_d,
                 premix_frac=0.0, premix_budget="floor_added"):
    """Return (sigma, eps_r) fields from blend fractions + endpoints + premix params.

    blend_sigma / blend_eps: array-like fill/saturation blend (part ~= 1, bed = 0;
        may exceed 1 in two-sided per-node mode).
    premix_frac=0.0 -> reference inline blend, bit-for-bit.
    """
    f = float(premix_frac)
    if f < 0.0:
        raise ValueError("premix_frac must be >= 0")
    bs = np.asarray(blend_sigma, dtype=float)
    be = np.asarray(blend_eps, dtype=float)
    if f == 0.0:
        # identity path: EXACT reference formula, same op order -> bit-identical
        sigma = sigma_v + bs * (sigma_d0 - sigma_v)
        eps_r = eps_v + be * (eps_d - eps_v)
        return sigma, eps_r
    # premix ON: raise the whole domain to a premixed background, then re-apply
    # the jetted increment inside the part per the dopant-budget variant.
    sigma_premix = sigma_v + f * (sigma_d0 - sigma_v)
    eps_premix = eps_v + f * (eps_d - eps_v)
    if premix_budget == "floor_added":
        s_span = sigma_d0 - sigma_v      # full jet increment on top of premix
        e_span = eps_d - eps_v
    elif premix_budget == "budget_fixed":
        s_span = sigma_d0 - sigma_premix  # jet brings premix up to sigma_doped
        e_span = eps_d - eps_premix
    else:
        raise ValueError(f"unknown premix_budget {premix_budget!r}")
    sigma = sigma_premix + bs * s_span
    eps_r = eps_premix + be * e_span
    return sigma, eps_r
