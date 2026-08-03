"""The DWELL-WEIGHTED heating kernel and its two gradients.

`rot_kernel.AveragedKernel` averages the per-position heating with equal weight
1/M, which is a turntable that spends the same time at every orientation. This
module makes those weights the DESIGN VARIABLE:

    Q_avg(s, w) = sum_k w_k * R_(-theta_k) [ Q_rf( R_(theta_k) s ; theta_k ) ]

with w on the simplex. Everything else is inherited unchanged, so at uniform
weights this class must be BIT IDENTICAL to the already finite-difference-gated
`AveragedKernel`. That degeneracy is a red-first test
(`tests/test_dwell.py::test_dwell_kernel_at_uniform_weights_is_bit_identical_to_the_averaged_kernel`),
not a claim.

THE TWO GRADIENTS COME OUT OF ONE BACKWARD SWEEP. The reverse march produces
`dJ/dQ_avg` for each of the two electrical states. From there:

    dJ/dw_k = <dJ/dQ_avg_A, Q_k_A> + <dJ/dQ_avg_B, Q_k_B>
    dJ/ds   = sum_k w_k * (the per-position electro-quasi-static adjoint chain)

The weight gradient is a plain inner product against the stored per-position
part-frame heating, which is the angle-segment analogue of the time-segment
inner product `schedule.accumulate_to_segments` uses for the power schedule.
It costs nothing beyond the sweep that the map gradient already pays for.

WHAT IS NOT MODELLED HERE. The move between positions is instantaneous and the
heating switches discontinuously; the quasi-static step assumes the cycle time
is short against the thermal times. Both are measured against the true rotating
engine rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import adjoint as adj, gradops
from . import forward as fwd
from .rot_kernel import AveragedKernel, _AngleState


@dataclass
class DwellKernel(AveragedKernel):
    """`AveragedKernel` with free, normalized, non-negative angle weights."""

    weights: np.ndarray = field(default=None, repr=False)
    _Qk_a: list = field(default_factory=list, repr=False)
    _Qk_b: list = field(default_factory=list, repr=False)
    _cache_w: np.ndarray = field(default=None, repr=False)

    # -- the design variable ------------------------------------------------

    def set_weights(self, w) -> np.ndarray:
        a = np.asarray(w, dtype=float).ravel()
        if a.size != self.n_angles:
            raise ValueError(f"{a.size} weights against {self.n_angles} angles")
        if np.any(a < 0.0):
            raise ValueError(f"dwell weights must be non-negative, got {a!r}")
        if abs(float(np.sum(a)) - 1.0) > 1e-9:
            raise ValueError(f"dwell weights must sum to 1, got {float(np.sum(a))!r}")
        self.weights = a
        return a

    def _w(self) -> np.ndarray:
        if self.weights is None:
            return np.full(self.n_angles, 1.0 / float(self.n_angles))
        return self.weights

    # -- forward ------------------------------------------------------------

    def averaged_Q(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted part-frame heating, and the per-position pieces it is made of.

        The per-position part-frame heating arrays are stored because they are
        exactly the vectors the weight gradient contracts against, so the
        weight gradient is free once the forward has run.
        """
        w = self._w()
        shape = self.case0.part_mask.shape
        Qa = np.zeros(shape)
        Qb = np.zeros(shape)
        self._Qk_a, self._Qk_b = [], []
        for wk, a in zip(w, self.per_angle):
            qa, qb = self._per_angle_Q(a, s)
            self._Qk_a.append(qa)
            self._Qk_b.append(qb)
            Qa += wk * qa
            Qb += wk * qb
        self._cache_s = np.array(s, dtype=float, copy=True)
        self._cache_w = np.array(w, dtype=float, copy=True)
        return Qa, Qb

    def _per_angle_Q(self, a: _AngleState, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One position's part-frame heating, both electrical states.

        Line for line `AveragedKernel.averaged_Q`'s loop body with the 1/M
        dropped, so the per-position electrical state caching the adjoint reads
        is set up identically.
        """
        s_lab = self.lab_map(a, s)
        eps = fwd.eps_field(a.case)
        a.st_a = fwd.solve_electric(a.case, fwd.sigma_state_a(a.case, s_lab, False), eps)
        sig_b, a.inrange_b = fwd.sigma_state_b(a.case, s_lab)
        a.st_b = fwd.solve_electric(a.case, sig_b, eps)
        p = a.case.pins
        a.active_a = (a.case.doped_mask & (a.st_a.Qrf_raw > 0.0)
                      & (a.st_a.Qrf_raw < p.max_qrf))
        a.active_b = (a.case.doped_mask & (a.st_b.Qrf_raw > 0.0)
                      & (a.st_b.Qrf_raw < p.max_qrf))
        return (a.R_to_part.apply(a.st_a.Qrf, outside=0.0),
                a.R_to_part.apply(a.st_b.Qrf, outside=0.0))

    # -- gradients ----------------------------------------------------------

    def _check_cache(self, s: np.ndarray) -> None:
        if self._cache_s is None or not np.array_equal(
                self._cache_s, np.asarray(s, dtype=float)):
            raise RuntimeError(
                "DwellKernel gradients must be called against the map and the "
                "weights of the most recent forward; the cached per-position "
                "electrical states belong to a different design point.")
        if self._cache_w is None or not np.array_equal(self._cache_w, self._w()):
            raise RuntimeError(
                "the dwell weights changed after the forward; re-run the "
                "forward before asking for a gradient.")

    def weight_gradient(self, tr: fwd.Trajectory, seeds: dict[int, np.ndarray],
                        seeds_rho: dict[int, np.ndarray] | None = None
                        ) -> np.ndarray:
        """`dJ/dw_k`, the angle-segment inner products of the SAME sweep."""
        if not self._Qk_a:
            raise RuntimeError("weight_gradient needs a forward run first")
        gQ_a, gQ_b = self._reverse_march(tr, seeds, seeds_rho)
        return np.array([float(np.sum(gQ_a * qa) + np.sum(gQ_b * qb))
                         for qa, qb in zip(self._Qk_a, self._Qk_b)])

    def both_gradients(self, s: np.ndarray, tr: fwd.Trajectory,
                       seeds: dict[int, np.ndarray], grad_ops=None,
                       seeds_rho: dict[int, np.ndarray] | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
        """(dJ/ds, dJ/dw) from ONE reverse march. The production call."""
        self._check_cache(s)
        case = self.case0
        p = case.pins
        Gx, Gy = (grad_ops if grad_ops is not None
                  else gradops.gradient_matrices(case.x, case.y))
        gQ_a, gQ_b = self._reverse_march(tr, seeds, seeds_rho)
        gw = np.array([float(np.sum(gQ_a * qa) + np.sum(gQ_b * qb))
                       for qa, qb in zip(self._Qk_a, self._Qk_b)])
        w = self._w()
        ds = np.zeros(case.part_mask.shape)
        for wk, a in zip(w, self.per_angle):
            if wk == 0.0:
                continue
            pm_j = a.case.part_mask
            for gQ_part, st, active, is_b in ((gQ_a, a.st_a, a.active_a, False),
                                              (gQ_b, a.st_b, a.active_b, True)):
                if not np.any(gQ_part):
                    continue
                g_lab = a.R_to_part.apply_T(gQ_part) * wk
                dsig = adj.eqs_vjp(a.case, st, g_lab * active, Gx, Gy, pre_masked=True)
                if is_b:
                    g_slab = dsig * a.inrange_b * p.sigma_d0
                else:
                    g_slab = dsig * a.case.fill_frac * (p.sigma_d0 - p.sigma_v)
                ds += a.R_to_lab.apply_T(np.where(pm_j, g_slab, 0.0))
        return ds, gw

    def gradient(self, s: np.ndarray, tr: fwd.Trajectory,
                 seeds: dict[int, np.ndarray], grad_ops=None,
                 seeds_rho: dict[int, np.ndarray] | None = None) -> np.ndarray:
        """`dJ/ds` alone, same signature as the inherited kernel."""
        return self.both_gradients(s, tr, seeds, grad_ops, seeds_rho)[0]
