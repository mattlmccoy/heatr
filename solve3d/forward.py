"""solve3d Phase A forward model: complex EQS + enthalpy thermal-phase march.

RUNS IN THE SPIKE ENV ONLY (dolfinx 0.11 complex build):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 heatr3d_d1_spike/env/bin/python ...

`jit_fix` MUST be imported and applied before dolfinx (Dropbox-path FFCx shim);
that is done at the top of this module, exactly as heatr3d_d1_spike/eqs_common.py
does it.

=============================================================================
PORTED SEMANTICS (source anchors read at heatr3d.py commit 699ed79)
=============================================================================
* gamma = sigma + 1j*omega*EPS0*eps_r          -- heatr3d.build_gamma (binary path)
* div(gamma grad V) = 0, V=v_lo on y_min, v_hi on y_max, Neumann elsewhere
                                               -- heatr3d.solve_eqs_3d
* Q = 0.5*Re(gamma |E|^2) = 0.5*sigma|E|^2, clipped >= 0, zeroed outside the
  part, then renormalized so integral(Q dV) = power_density_w_per_m3 * V_part
                                               -- heatr3d.compute_qrf_3d
  ONLY the corrected (part-confined) drive exists here. heatr3d's
  qrf_gradient="legacy" cross-interface np.gradient has no analogue and is
  forbidden by construction (spec sec 4). On a conforming FEM mesh grad(V) is
  cell-wise constant per element and elements never straddle the interface, so
  the DG0 Q is a part-confined quantity by construction -- this is precisely
  what heatr3d's `qrf_gradient="masked"` mode approximates on voxels.
* H(T) piecewise linear, latent plateau ramped over the melt window; exact
  inversion                                    -- heatr3d.enthalpy_from_T /
                                                  heatr3d.T_from_enthalpy
* sigma(T, rho) coupling + clip bounds         -- heatr3d.apply_sigma_coupling
* in-march EQS re-solve schedule on ABSOLUTE time, fixed-power renormalization
  on every re-solve, drift-tolerance skip      -- heatr3d.run (S4-COUPLING)

DELIBERATE DEVIATIONS (named, not hidden):
1. Electrode gauge. heatr3d's cell-centred Dirichlet rows span L-h; the FEM
   planes sit on the true faces and span L. This is an O(h/L) scale factor on
   E (1.6 % at n=64) that CANCELS out of Q because both engines renormalize to
   the same fixed absorbed power (heatr3d_d1_spike/metrics.unit_mean note).
2. Face harmonic averaging of gamma / k has no FEM analogue; the Galerkin form
   uses element-wise (DG0) coefficients. The two agree in the continuum limit.
3. Part geometry is CONFORMING (exact cylinder / prism), not a voxel staircase.
   Part volumes therefore differ from heatr3d's by the staircase error, which
   changes the absolute power target (p_target = power_density * V_part). This
   is recorded in every parity artifact, never silently absorbed.
4. The thermal march is mass-LUMPED explicit Euler on nodal volumetric
   enthalpy -- the FEM analogue of heatr3d's explicit cell-centred FV update.
   Lumping is what makes the pointwise H->T inversion (heatr3d's exact
   piecewise-linear inverse) well defined node-by-node.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_SPIKE = Path(__file__).resolve().parents[1] / "heatr3d_d1_spike"
if str(_SPIKE) not in sys.path:
    sys.path.insert(0, str(_SPIKE))

import jit_fix                       # noqa: E402  MUST precede dolfinx
JIT_FIX = jit_fix.apply()

import numpy as np                   # noqa: E402
import dolfinx                       # noqa: E402

DOLFINX_VERSION = dolfinx.__version__
IS_COMPLEX = bool(np.issubdtype(dolfinx.default_scalar_type, np.complexfloating))
SCALAR_PATH = "complex" if IS_COMPLEX else "real_split"

EPS0 = 8.8541878128e-12              # heatr3d.EPS0, verbatim
R_GAS = 8.314                        # heatr3d.R_GAS, verbatim (densification only)


@dataclass(frozen=True)
class ForwardParams:
    """The heatr3d.Params fields Phase A depends on, with heatr3d's defaults.

    Frozen (coding-style rule: immutable config). Every value below is copied
    verbatim from heatr3d.Params; `phase_update` is pinned to "enthalpy"
    because Phase A only ports the energy-conserving scheme (the legacy
    apparent_cp path is explicitly out of scope, spec sec 5 / plan Task 3).
    """
    # --- drive / EQS ---
    freq_hz: float = 27.12e6
    v_lo: float = 860.0
    v_hi: float = 0.0
    sigma_doped: float = 0.04
    eps_doped: float = 20.0
    sigma_virgin: float = 1e-8
    eps_virgin: float = 2.0
    power_density_w_per_m3: float = 10.0 / (np.pi * 0.010 ** 2 * 0.020)
    # --- powder thermal ---
    k_powder: float = 0.197
    rho_powder: float = 490.0
    cp_powder: float = 1072.0
    # --- melt (doped solid -> liquid) ---
    k_solid: float = 0.10
    k_liquid: float = 0.26
    rho_solid: float = 460.0
    rho_liquid: float = 1010.0
    cp_solid: float = 2500.0
    cp_liquid: float = 3279.0
    latent_j_per_kg: float = 96700.0
    t_pc_c: float = 180.0
    dt_pc_c: float = 10.0
    rho_rel: float = 0.55
    ambient_c: float = 23.0
    preheat_c: float = 23.0
    conv_h: float = 5.0              # top face (y = +L/2) only
    # --- numerics ---
    dt_s: float = 0.05
    max_dt_step_c: float = 10.0
    temp_min_c: float = -50.0
    temp_max_c: float = 600.0
    phase_update: str = "enthalpy"   # pinned; see class docstring
    enforce_cfl: bool = True
    # --- S4-COUPLING (heatr3d 699ed79); all inert at these defaults ---
    eqs_update_interval_s: float = 0.0
    sigma_temp_coeff_per_K: float = 0.0
    sigma_density_coeff: float = 0.0
    sigma_ref_temp_c: float = 23.0
    eqs_resolve_drift_rtol: float = 0.0

    def __post_init__(self) -> None:
        if self.phase_update != "enthalpy":
            raise ValueError(
                "solve3d Phase A only implements phase_update='enthalpy'; "
                f"got {self.phase_update!r}")

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq_hz


def run_forward(*args, **kwargs):
    """Coupled EQS + enthalpy-march forward. Wired in plan Task 4."""
    raise NotImplementedError(
        "run_forward is wired in Phase A Task 4 (coupled-forward parity). "
        "Task 2 provides solve_eqs / qrf_dg0; Task 3 provides march_enthalpy.")
