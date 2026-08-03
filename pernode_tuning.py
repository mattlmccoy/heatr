#!/usr/bin/env python3
"""Per-node adaptive-gain absolute-target conductivity tuning law.

This is a faithful, pure-function port of Allison's ``Tune_Conductivity.m``
(JaredFiles .../Square/), summarised in
``outputs_eqs/jared_ir_exp1/COMSOL_ARCHIVE_ANALYSIS.md``.

Per node ``i`` and iteration ``k``::

    sigma_i(k+1) = clip( sigma_i(k) + K1_i * (Tt - T_i(k)) / maxDiff,
                         sigma_min, sigma_max )

where

    maxDiff = max_i |Tt - T_i(k)|          (largest absolute error at iter k)

and the per-node gain ``K1_i`` is HALVED whenever node ``i``'s error
``(Tt - T_i)`` changes sign between consecutive iterations (overshoot damping).
The loop is converged when ``maxDiff <= eps``.

TWO-SIDED actuation: ``sigma`` may rise above the uniform baseline (0.04 S/m)
up to ``sigma_max`` OR fall to ``sigma_min`` (default 0).  This is the crucial
difference from HEATR's one-sided saturation actuator (sat <= 1 can only lower
sigma from sigma0).

The function is deliberately free of any forward-model coupling so it can be
finite-difference / unit tested in isolation.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PernodeUpdate(NamedTuple):
    """Result of one per-node tuning step.

    Attributes
    ----------
    sigma : np.ndarray
        Updated, clamped per-node conductivity (S/m).
    K1 : np.ndarray
        Per-node gain after sign-flip halving (carry to next iteration).
    sign : np.ndarray
        Sign of the current error ``(Tt - T_i)`` (carry as ``sign_prev`` next).
    max_diff : float
        ``max_i |Tt - T_i|`` at this iteration.
    converged : bool
        ``max_diff <= eps``.
    """

    sigma: np.ndarray
    K1: np.ndarray
    sign: np.ndarray
    max_diff: float
    converged: bool


def pernode_sigma_update(
    sigma: np.ndarray,
    T: np.ndarray,
    Tt: float,
    K1: np.ndarray | float,
    sign_prev: np.ndarray,
    sigma_max: float,
    *,
    sigma_min: float = 0.0,
    eps: float = 5.0,
    gain_halving: float = 0.5,
) -> PernodeUpdate:
    """Apply one iteration of the per-node adaptive-gain absolute-target law.

    Parameters
    ----------
    sigma : array_like
        Current per-node conductivity (S/m).
    T : array_like
        Current per-node temperature (deg C), same shape as ``sigma``.
    Tt : float
        Absolute target temperature (deg C).
    K1 : array_like or float
        Current per-node gain (S/m).  A scalar is broadcast to ``sigma``'s shape.
    sign_prev : array_like
        Sign of the previous iteration's error, same shape as ``sigma``.
        Use zeros on the first iteration (no history -> no halving).
    sigma_max : float
        Upper clamp on conductivity (S/m).  Allison's Smax = 0.0425.
    sigma_min : float, optional
        Lower clamp (default 0.0).
    eps : float, optional
        Convergence tolerance on ``maxDiff`` (default 5.0 deg C).
    gain_halving : float, optional
        Multiplier applied to K1 on a sign flip (default 0.5).

    Returns
    -------
    PernodeUpdate
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    sign_prev = np.asarray(sign_prev, dtype=float)
    if np.ndim(K1) == 0:
        K1 = np.full(sigma.shape, float(K1), dtype=float)
    else:
        K1 = np.asarray(K1, dtype=float)

    err = float(Tt) - T
    max_diff = float(np.max(np.abs(err))) if err.size else 0.0

    sign = np.sign(err)
    # Sign flip: current and previous signs are both non-zero and opposite.
    flipped = (sign * sign_prev) < 0.0
    K1_new = np.where(flipped, K1 * gain_halving, K1)

    denom = max_diff if max_diff > 0.0 else 1.0
    sigma_new = sigma + K1_new * (err / denom)
    sigma_new = np.clip(sigma_new, sigma_min, sigma_max)

    converged = bool(max_diff <= eps)
    return PernodeUpdate(sigma_new, K1_new, sign, max_diff, converged)
