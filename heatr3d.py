#!/usr/bin/env python3
"""heatr3d.py -- SELF-CONTAINED SYNCED COPY of the canonical 3-D HEATR solver.

CFG-01 / REPRO-04 (audit hygiene): the canonical implementation lives in
dissertation_materials/analysis-3dfgm/heatr3d.py. This file is a VERBATIM full
copy of that module (same physics, same public API), kept here so geo-prewarp
(heatr3d_job.py and friends) imports a real in-repo module and a fresh standalone
clone of the `heatr` repo works with no cross-repo path dependency.

Provenance: copied from the canonical file whose SHA-256 is recorded in
SYNCED_FROM_SHA256 below. A non-fatal drift check (at the bottom of this file)
re-hashes the canonical copy IF it is reachable on this machine and logs a WARNING
when the two have diverged -- so silent drift between the copies is visible to the
maintainer, while clones that lack the canonical path are unaffected.

To re-sync after a deliberate canonical change: re-copy the canonical body here and
update SYNCED_FROM_SHA256 to the new canonical hash. See
audit_fixes_out/hygiene/RESULT.md for the bit-identical-output verification.

Override: set HEATR3D_CANONICAL_PATH to the canonical heatr3d.py if its default
relative location moves; leave it unset (or absent) to skip the drift check.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import distance_transform_edt

EPS0 = 8.8541878128e-12

# Module logger for clamp diagnostics (THM-01/02). Emits WARNINGs only when a
# numerical limiter actually BINDS; dormant in the dissertation production config,
# so output stays bit-identical (verified with np.array_equal).
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Material / process parameters (shape_circle_6min.yaml)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    freq_hz: float = 27.12e6
    v_lo: float = 860.0          # bottom plate (y_min)
    v_hi: float = 0.0            # top plate (y_max)
    sigma_doped: float = 0.04
    eps_doped: float = 20.0
    sigma_virgin: float = 1e-8
    eps_virgin: float = 2.0
    # Absorbed-power DENSITY reference (W/m^3): 10 W into the 2-D reference part
    # (20 mm circle x 20 mm depth = 6.283e-6 m^3). Total power per run is this
    # density times the doped volume, so heating rate per volume is constant and
    # sigma_T is comparable across part sizes (mirrors the 2-D study's t* norm).
    power_density_w_per_m3: float = 10.0 / (np.pi * 0.010 ** 2 * 0.020)
    # powder thermal
    k_powder: float = 0.197
    rho_powder: float = 490.0
    cp_powder: float = 1072.0
    # melt (doped solid->liquid)
    k_solid: float = 0.10
    k_liquid: float = 0.26
    rho_solid: float = 460.0
    rho_liquid: float = 1010.0
    cp_solid: float = 2500.0
    cp_liquid: float = 3279.0
    latent_j_per_kg: float = 96700.0
    t_pc_c: float = 180.0
    dt_pc_c: float = 10.0
    rho_rel: float = 0.55        # held constant (see header)
    ambient_c: float = 23.0
    preheat_c: float = 23.0      # bed preheat temp (real SLS holds powder near melt);
                                 # only doped region absorbs RF, so undoped stays sub-melt
    conv_h: float = 5.0          # top face only
    # numerics
    dt_s: float = 0.05
    max_dt_step_c: float = 10.0
    temp_min_c: float = -50.0
    temp_max_c: float = 600.0
    # S1: phase-update scheme. "apparent_cp" = legacy pointwise dphi in cp_eff
    # (bit-for-bit historical behavior; can skip the latent barrier when one
    # step crosses the melt window). "enthalpy" = exact piecewise-linear
    # enthalpy inversion (energy-conserving by construction; S1 fix).
    phase_update: str = "apparent_cp"
    # S1b / THM-03: explicit-conduction stability. True (default) = run() AUTO-
    # SUBSTEPS the thermal update whenever dt_s exceeds CFL_SAFETY * dt_stable
    # (see dt_stable_thermal), so a fine grid can never march an unstable
    # conduction step. Inert wherever the criterion is already satisfied
    # (n <= 169 at dt_s = 0.05 s, L = 0.060 m), i.e. every historical heatr3d
    # result. False = legacy single-step behavior, with Result.cfl_violated
    # latched and a warning logged when the criterion is broken.
    enforce_cfl: bool = True
    # densification (physics_dual; ported from shape_circle_6min.yaml)
    dens_k0_ss: float = 0.005
    dens_ea_ss: float = 48000.0
    dens_phi_solid_exp: float = 0.8
    dens_phi_threshold: float = 0.01
    dens_phi_liq_exp: float = 1.0
    dens_geom_factor: float = 0.05
    dens_surface_tension: float = 0.03
    dens_particle_radius_m: float = 3.5e-5
    dens_eta_ref_pa_s: float = 8.0e3
    dens_eta_ref_temp_k: float = 458.15
    dens_eta_activation: float = 6.0e4
    dens_rho_exp: float = 1.0
    dens_max_drho_rate: float = 0.04   # per second (= 0.02 per 0.5 s step in 2-D)
    # ---- S4-COUPLING (2026-08-01): in-march EQS re-solve with sigma(T, rho) ----
    # Motivation: heatr3d_s4_flir/S4_GATE_REPORT.md sec 4.2 lists "Q_rf is frozen"
    # and "no sigma(T)/sigma(rho_rel) feedback" as the top two candidate
    # mechanisms for the late-time topology miss. All four knobs below default to
    # the legacy values, and eqs_update_interval_s == 0.0 is the MASTER SWITCH:
    # with it at 0 the coefficients are completely inert and every historical
    # result is reproduced bit-for-bit (test_coupling_defaults_are_bit_for_bit_legacy).
    #
    # eqs_update_interval_s: simulated seconds between EQS re-solves. 0.0 = never
    #   re-solve (legacy: one solve before the march, held for the whole run).
    #   The schedule is on ABSOLUTE time (see run(t_start_s=...)), so a chained
    #   segmented march keeps one global schedule.
    # sigma_temp_coeff_per_K / sigma_density_coeff / sigma_ref_temp_c: the
    #   sigma(T, rho_rel) law, ported from the 2-D solver (see
    #   apply_sigma_coupling). The reference density is p.rho_rel (the initial
    #   relative density), mirroring the 2-D solver's rho_rel_init.
    # eqs_resolve_drift_rtol: D1/EQS-01 cost control. A re-solve is a FULL solve
    #   at the certified grid ceiling (n=96 -> ~26 s at the S4 grid, 322 s cubic),
    #   so when the max sigma drift since the last solve is below this relative
    #   tolerance the solve is SKIPPED and the drive is updated pointwise
    #   (Q *= sigma/sigma_last, then re-renormalized). 0.0 = never skip.
    #   Ported concept: rfam_eqs_coupled.py electric.eqs_adaptive_rtol (l. 2901-2907,
    #   3065-3082).
    eqs_update_interval_s: float = 0.0
    sigma_temp_coeff_per_K: float = 0.0
    sigma_density_coeff: float = 0.0
    sigma_ref_temp_c: float = 23.0
    eqs_resolve_drift_rtol: float = 0.0


# Universal gas constant [J/(mol*K)]. Kept at 8.314 (NOT the full 8.31446261815324
# used in rfam_eqs_coupled.py) DELIBERATELY. R_GAS feeds the Arrhenius densification
# kinetics (densify_rate / viscosity below), so changing it shifts committed
# densification/shrinkage/FGM results. The difference is ~0.005%, physically immaterial
# (the activation-energy uncertainty dwarfs it by orders of magnitude), so it is held
# fixed to avoid a pointless re-baseline of published figures. Audit 2026-06: keep as-is,
# documented (see heatr_audit/findings/hygiene + REAUDIT_SUMMARY). Do not "correct" without
# a deliberate, reviewed re-baseline.
R_GAS = 8.314


@dataclass
class Grid:
    n: int = 48                  # cells per axis
    L: float = 0.060             # chamber size (m), cubic
    def __post_init__(self):
        self.nx = self.ny = self.nz = self.n
        self.h = self.L / self.n
        c = (np.arange(self.n) + 0.5) * self.h - self.L / 2.0
        self.x = self.y = self.z = c           # centered coords
        self.dV = self.h ** 3


# --------------------------------------------------------------------------- #
# S1b / THM-03: explicit-conduction stability criterion
# (docs/superpowers/plans/s1-findings.md section 13.3)
# --------------------------------------------------------------------------- #
# Safety factor on the explicit 6-neighbour bound dt < h^2/(6 alpha). 0.9 keeps a
# 10 % margin from the marginal-stability point (where the amplification factor
# of the checkerboard mode is exactly -1 and the scheme merely oscillates).
CFL_SAFETY = 0.9


def alpha_max_thermal(p: Params) -> float:
    """Largest thermal diffusivity k/(rho cp) present in the domain [m^2/s].

    The BINDING medium is the POWDER BED (3.750381e-07), not the melt: the
    Task-3 note used alpha_liquid = 7.85e-08 and wrongly cleared the CFL. The
    powder is ~98 % of the voxels.

        powder  0.197/(490*1072)   = 3.750381e-07   <- binding
        solid   0.10 /(460*2500)   = 8.695652e-08
        liquid  0.26 /(1010*3279)  = 7.850700e-08

    The max over the three PURE materials also bounds every blended state the
    solver constructs: inside the part cp >= cp_solid = 2500 and rho >= 460, so
    even with the largest conductivity (k_liquid = 0.26) the blend cannot exceed
    0.26/(460*2500) = 2.26e-07 < 3.75e-07.

    NOT covered (documented, not silently assumed): the `heatsink_kgain` hook in
    run() multiplies k in lattice voxels, which raises the local alpha above this
    bound. Runs using it must set their own dt or accept the substepping this
    bound implies for the unmodified materials."""
    return max(p.k_powder / (p.rho_powder * p.cp_powder),
               p.k_solid / (p.rho_solid * p.cp_solid),
               p.k_liquid / (p.rho_liquid * p.cp_liquid))


def dt_stable_thermal(grid: Grid, p: Params) -> float:
    """Explicit 6-neighbour conduction stability limit dt < h^2/(6 alpha_max) [s].
    At L = 0.060 m: 1.5623 s at n=32, 0.1736 s at n=96, 0.0400 s at n=200."""
    return grid.h ** 2 / (6.0 * alpha_max_thermal(p))


def cfl_substeps(grid: Grid, p: Params) -> int:
    """Number of thermal substeps per p.dt_s needed to satisfy
    dt_sub <= CFL_SAFETY * dt_stable_thermal. 1 whenever the step is already
    stable -- which is every historical heatr3d configuration (n <= 169 at
    dt_s = 0.05 s, L = 0.060 m), so the default path is unchanged."""
    dt_limit = CFL_SAFETY * dt_stable_thermal(grid, p)
    if p.dt_s <= dt_limit:
        return 1
    return int(np.ceil(p.dt_s / dt_limit))


# --------------------------------------------------------------------------- #
# Geometry: part_mask(x,y,z). Cross-section in (x,y); may vary with z.
# --------------------------------------------------------------------------- #
def make_geometry(grid: Grid, shape: str, diam: float = 0.020,
                  zspan: float = 0.020) -> np.ndarray:
    """Return a 3-D boolean part mask of given shape, centered.
    shape: 'cylinder' (extruded circle, constant in z -> 2-D validation),
           'sphere', 'cone' (radius shrinks with z), 'dumbbell',
           'square'/'lshape'/'cross' (prewarp-study cross-sections in the (x,y)
           field plane, extruded as prisms along z over +-zspan/2).

    Geometry convention reminder: y = electrode/field axis (plates at y_min/y_max),
    x = lateral, z = build axis. The prewarp-study shapes are 2-D cross-sections in
    the (x,y) plane (so the RF field axis y is the same axis the simplified
    ilt_shape model saw), extruded uniformly along the build axis z."""
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    r_xy = np.sqrt(X ** 2 + Y ** 2)
    R = diam / 2.0
    # prewarp-study prisms: cross-section in (x,y), constant over the z slab.
    # `diam` sets the overall in-plane footprint (full width of the bounding box),
    # comparable to the cylinder/sphere parts. zspan sets the prism height.
    in_z = np.abs(Z) <= zspan / 2.0
    half = diam / 2.0                          # half the bounding-box width
    if shape == "square":
        # solid square, full bounding box (12/28 of grid in the study -> here `diam`)
        return (np.abs(X) <= half) & (np.abs(Y) <= half) & in_z
    if shape == "lshape":
        # L: vertical + horizontal arm sharing the lower-left corner. Arm length =
        # `diam`, arm thickness = ~5/12 of arm (study: arm 12, thick 5). Origin at
        # the lower-left so the part is centred on the L bounding box.
        thick = diam * (5.0 / 12.0)
        x0, y0 = -half, -half                  # lower-left corner of the L bbox
        vert = (X >= x0) & (X <= x0 + thick) & (Y >= y0) & (Y <= y0 + diam)
        horiz = (Y >= y0) & (Y <= y0 + thick) & (X >= x0) & (X <= x0 + diam)
        return (vert | horiz) & in_z
    if shape == "cross":
        # plus/cross: two perpendicular bars (study arm 9, thick 4 of 28).
        # arm half-length = `diam`/2, bar half-thickness = ~4/9 of that.
        thick = half * (4.0 / 9.0)
        vert = (np.abs(X) <= thick) & (np.abs(Y) <= half)
        horiz = (np.abs(Y) <= thick) & (np.abs(X) <= half)
        return (vert | horiz) & in_z
    if shape == "cylinder":
        # circular x-y cross-section, spans the full z extent (-> no z gradient)
        return r_xy <= R
    if shape == "sphere":
        return (X ** 2 + Y ** 2 + Z ** 2) <= R ** 2
    if shape == "cone":
        # base radius R at z=-zspan/2, tip at z=+zspan/2
        zr = np.clip((zspan / 2 - Z) / zspan, 0.0, 1.0)   # 1 at base, 0 at tip
        return (r_xy <= R * zr) & (np.abs(Z) <= zspan / 2)
    if shape == "dumbbell":
        # two lobes (radius R) joined by a thin neck (radius R/2.5) along z
        lobe = ((X ** 2 + Y ** 2 + (np.abs(Z) - zspan / 2) ** 2) <= (R) ** 2)
        neck = (r_xy <= R / 2.5) & (np.abs(Z) <= zspan / 2)
        return lobe | neck
    raise ValueError(f"unknown shape {shape}")


# --------------------------------------------------------------------------- #
# 3-D EQS solve: div(gamma grad V) = 0
# --------------------------------------------------------------------------- #
def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = a + b
    out = np.where(np.abs(den) > 1e-30, 2.0 * a * b / np.where(den == 0, 1, den),
                   0.5 * (a + b))
    return out


# S1b / EQS-01 (docs/superpowers/plans/s1-findings.md 13.2): largest system size
# for which the DIRECT complex LU fallback (spla.spsolve -> SuperLU zgstrf) is
# considered usable. Above this, the direct solve is not merely slow: at
# N = 8.0e6 (n=200) it SIGSEGVs the interpreter (reproduced 3/3 on a 34 GB
# machine) after spilu itself fails to allocate. A segfault cannot be handled by
# a caller and kills a whole batch campaign, so solve_eqs_3d now raises a
# MemoryError instead of entering it. 2e6 unknowns ~ n=126 on a cubic grid.
EQS_DIRECT_MAX_UNKNOWNS = 2_000_000

# Practical grid ceilings MEASURED for this solver (findings 13.1/13.2, same
# machine): full pipeline (EQS + thermal) completes at n=96 (N=8.85e5, EQS 322 s,
# 2.21 GB); n=128 (N=2.10e6) did not finish an EQS solve in 30 min. Quoted in the
# EQS-01 error message so a caller learns the supported envelope, not just that
# it failed.
EQS_MAX_GRID_FULL_PHYSICS = 96
EQS_MAX_GRID_EQS_ONLY = 128


def _direct_lu_memory_estimate_gb(N: int) -> float:
    """Order-of-magnitude memory needed by a DIRECT complex LU of the 7-point
    3-D Laplacian on an N-unknown grid.

    Nested-dissection fill-in for a 3-D box grid is O(N^(4/3)) nonzeros; each
    complex128 nonzero costs 16 B of value plus ~4 B of index. This is an
    ESTIMATE for the error message (it is not calibrated against SuperLU's
    actual COLAMD ordering), stated as such wherever it is printed."""
    return 20.0 * float(N) ** (4.0 / 3.0) / 1024.0 ** 3


def solve_eqs_3d(gamma: np.ndarray, grid: Grid, p: Params,
                 iterative: bool | None = None) -> np.ndarray:
    """Solve div(gamma grad V)=0. Electrodes: y_min plane = v_lo, y_max = v_hi.
    Neumann (no-flux) on x and z walls. Returns complex V (nx,ny,nz)."""
    nx, ny, nz = gamma.shape
    h2 = grid.h ** 2
    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)
    elec_lo = np.zeros(gamma.shape, bool); elec_lo[:, 0, :] = True
    elec_hi = np.zeros(gamma.shape, bool); elec_hi[:, -1, :] = True
    dirich = elec_lo | elec_hi

    rows = [idx[dirich]]
    cols = [idx[dirich]]
    vals = [np.ones(int(dirich.sum()), np.complex128)]
    b = np.zeros(N, np.complex128)
    b[idx[elec_lo]] = p.v_lo
    b[idx[elec_hi]] = p.v_hi

    diag = np.zeros(gamma.shape, np.complex128)
    # six face directions
    for ax in range(3):
        for s in (-1, +1):
            g_nb = np.roll(gamma, -s, axis=ax)
            gf = _harmonic(gamma, g_nb) / h2
            # mask faces that fall off the domain (no wrap; Neumann)
            valid = np.ones(gamma.shape, bool)
            sl = [slice(None)] * 3
            sl[ax] = (-1 if s == +1 else 0)
            valid[tuple(sl)] = False
            w = np.where(valid, gf, 0.0)
            # interior (non-Dirichlet) equations only
            interior = ~dirich
            src = interior & valid
            i_lin = idx[src]
            nb_lin = np.roll(idx, -s, axis=ax)[src]
            diag[src] += w[src]
            # if neighbor is Dirichlet, move to RHS; else off-diagonal
            nb_dir = np.roll(dirich, -s, axis=ax)[src]
            wv = w[src]
            # off-diagonal for non-Dirichlet neighbors
            on = ~nb_dir
            rows.append(i_lin[on]); cols.append(nb_lin[on]); vals.append(-wv[on])
            # Dirichlet neighbor contributes to b
            nb_is_lo = np.roll(elec_lo, -s, axis=ax)[src]
            bd = np.where(np.roll(elec_lo, -s, axis=ax)[src], p.v_lo, p.v_hi)
            np.add.at(b, i_lin[nb_dir], (wv * bd)[nb_dir])
    # diagonal for interior
    interior = ~dirich
    rows.append(idx[interior]); cols.append(idx[interior]); vals.append(diag[interior])

    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    A = A + sp.eye(N, format="csr", dtype=np.complex128) * 1e-18
    # Auto: direct (spsolve) is fastest+robust for small grids; for large grids the
    # 3-D LU fill-in explodes, so use ILU-preconditioned BiCGSTAB. Direct fallback.
    use_iter = (N > 50_000) if iterative is None else bool(iterative)   # n>=~37 -> iterative
    V = None
    iter_failure: BaseException | str | None = None
    if use_iter:
        try:
            ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=12)
            M = spla.LinearOperator(A.shape, ilu.solve, dtype=np.complex128)
            V, info = spla.bicgstab(A, b, rtol=1e-8, atol=0.0, maxiter=2000, M=M)
            if info != 0 or not np.all(np.isfinite(V)):
                V = None       # fall back to direct
                iter_failure = f"BiCGSTAB did not converge (info={info})"
        except Exception as exc:                       # ILU build/solve failed
            V = None
            iter_failure = exc
    if V is None:
        # S1b / EQS-01: the direct fallback is only armed for systems where a
        # direct complex LU is actually feasible. Above EQS_DIRECT_MAX_UNKNOWNS
        # it SIGSEGVs inside SuperLU (findings 13.2) -- an unhandleable failure
        # mode -- so raise a MemoryError the caller can catch instead.
        if N > EQS_DIRECT_MAX_UNKNOWNS:
            raise MemoryError(
                f"EQS solve failed and the direct fallback is not usable at this "
                f"size: N={N} complex unknowns ({nx}x{ny}x{nz} grid) exceeds the "
                f"direct-solve limit EQS_DIRECT_MAX_UNKNOWNS={EQS_DIRECT_MAX_UNKNOWNS}. "
                f"A direct complex LU here needs ~{_direct_lu_memory_estimate_gb(N):.3g} GB "
                f"(order-of-magnitude estimate: 20 B x N^(4/3) fill-in) and is known to "
                f"SIGSEGV inside SuperLU rather than raise. Iterative-path failure: "
                f"{iter_failure!r}. Largest grids this solver is measured to support "
                f"today: n={EQS_MAX_GRID_FULL_PHYSICS} for the full EQS+thermal pipeline, "
                f"n={EQS_MAX_GRID_EQS_ONLY} for an EQS-only solve. Reduce the grid, or "
                f"wait for the scalable large-N EQS work (decision point D1).")
        V = spla.spsolve(A, b)
    if not np.all(np.isfinite(V)):
        raise RuntimeError("EQS solve produced non-finite values")
    return V.reshape(nx, ny, nz)


# --------------------------------------------------------------------------- #
# EQS-02 (2026-07-31): the Q_rf gradient stencil, and the default flip
#
# compute_qrf_3d used to form E = -np.gradient(V) over the WHOLE domain and only
# afterwards zero Q outside the part, so the outermost in-part voxel was centrally
# differenced against an OUTSIDE voxel -- across the material interface, where
# grad V jumps by the conductivity contrast (sigma_doped/sigma_virgin = 4e6).
# Q ~ |E|^2 squares that jump, and the fixed-power renormalization then rescales
# the whole field down: no energy is invented, energy is MOVED from the part
# interior into a one-voxel surface skin.
#
# Evidence (heatr3d_d1_spike/EQS02_IMPACT.md, results.json["eqs02_impact"],
# ["task2"], ["task3"], n=64, L=60 mm, extruded circle d=20 mm and square 20 mm):
#   * 73.7 % of absorbed power sat in a 31-33 % volume band; mask-confined it is
#     31.9 % (circle) / 37.1 % (square) -- roughly volume-proportional.
#   * interior mean Q rises 2.59x (circle) / 2.39x (square) under the correction,
#     while the interior PATTERN is unchanged (rel L2 1.7e-16 / 2.0e-16).
#   * max/mean drops 12.25 -> 1.86 (circle), 19.15 -> 2.20 (square); the legacy
#     corner peak keeps growing with refinement (19.2 -> 29.1 -> 38.7 at
#     n=64/96/128) while the mask-confined one grows slowly (2.20 -> 2.71 -> 3.17)
#     -- most of the apparent 3-D grid dependence was this artifact.
#   * against the independent conforming-FEM reference (Task 2), the mask-confined
#     field differs by 10.8 % in unit-mean pattern L2 and the legacy field by
#     90.4 %; in the interior the two post-processings are identical.
#   * thermal topology INVERTS: surface-minus-interior mean T goes from +8.6 C /
#     +18.0 C (legacy) to -35.3 C / -30.5 C (corrected); sigma_T moves -23.1 %
#     (circle) but +10.9 % (square), i.e. geometry-dependently.
#
# DELIBERATE DEFAULT FLIP, signed off knowing it changes exploratory-labeled
# published numbers (surface/interior attribution reverses): qrf_gradient defaults
# to "masked". qrf_gradient="legacy" reproduces the pre-fix field bit-for-bit
# (test_qrf_legacy_flag_reproduces_the_pre_fix_field_bit_for_bit) for historical
# reproduction. Total absorbed power is identical under both.
#
# NOT fixed here: this is only the post-processing gradient. Any error in V itself
# (harmonic face averaging at the staircase boundary, the cell-centred electrode
# gauge) is present in both modes.
# --------------------------------------------------------------------------- #
QRF_GRADIENT_MODES = ("masked", "legacy")


def _masked_grad_3d(V: np.ndarray, mask: np.ndarray, h: float):
    """E = -grad V with a stencil that NEVER crosses the mask edge.

    Port of heatr3d_d1_spike/metrics.masked_grad_nd (ndim=3), the D1 spike's
    reference implementation; kept elementwise-identical to it so the corrected
    drive is exactly the field EQS02_IMPACT.md measured (pinned by
    test_qrf_default_is_the_part_confined_gradient_and_splits_power_by_volume).

    Second-order central difference wherever BOTH neighbours along an axis are
    inside the mask; one-sided (exact for a linear field) where only one is; zero
    where neither is, and zero outside the mask. Works for real or complex V (V is
    complex in the EQS solve)."""
    V = np.asarray(V)
    mask = np.asarray(mask, dtype=bool)
    if V.ndim != 3 or mask.shape != V.shape:
        raise ValueError("_masked_grad_3d: expected 3-D V and a matching mask")
    out = []
    for ax in range(3):
        g = np.zeros(V.shape, dtype=V.dtype if np.iscomplexobj(V) else float)
        fwd_ok = np.zeros(V.shape, bool)
        bwd_ok = np.zeros(V.shape, bool)
        s_lo, s_hi = [slice(None)] * 3, [slice(None)] * 3
        s_lo[ax] = slice(0, -1)
        s_hi[ax] = slice(1, None)
        s_lo, s_hi = tuple(s_lo), tuple(s_hi)
        fwd_ok[s_lo] = mask[s_lo] & mask[s_hi]        # i -> i+1 usable
        bwd_ok[s_hi] = mask[s_lo] & mask[s_hi]        # i -> i-1 usable
        dfwd = np.zeros_like(g)
        dbwd = np.zeros_like(g)
        dfwd[s_lo] = (V[s_hi] - V[s_lo]) / h
        dbwd[s_hi] = (V[s_hi] - V[s_lo]) / h
        both = fwd_ok & bwd_ok
        g = np.where(both, 0.5 * (dfwd + dbwd),
                     np.where(fwd_ok, dfwd, np.where(bwd_ok, dbwd, 0.0)))
        g = np.where(mask, g, 0.0)
        out.append(-g)
    return tuple(out)


def compute_qrf_3d(V: np.ndarray, gamma: np.ndarray, grid: Grid, p: Params,
                   doped: np.ndarray, premix: bool = False,
                   qrf_gradient: str = "masked") -> np.ndarray:
    """Volumetric RF heating Qrf = 0.5*Re(gamma|E|^2), renormalized to a fixed total
    absorbed power (enforce_generator_power).

    premix=False (default): Qrf is zeroed OUTSIDE the doped region (jet-only: only
    the printed part absorbs) and the fixed total power target is
    power_density * doped_volume.

    premix=True: the premixed conductive bed absorbs RF too, so Qrf is kept
    EVERYWHERE (not zeroed outside the part). The SAME total power target as the
    premix-off case for this geometry (power_density * doped_volume) is enforced over
    the whole domain, so premix REDISTRIBUTES a FIXED total absorbed power between the
    bed and the part -- it does not invent energy.

    qrf_gradient (EQS-02; see the module note above this function for the full
    evidence, heatr3d_d1_spike/EQS02_IMPACT.md):
      * "masked" (NEW DEFAULT, 2026-07-31): E is formed with a stencil confined to
        the doped region (_masked_grad_3d), so no difference is ever taken across
        the material interface. This is a DELIBERATE DEFAULT FLIP -- it changes
        previously published (exploratory-labeled) heatr3d numbers. Surface-vs-
        interior attribution of absorbed dose REVERSES (73.7 % of power in a
        31-33 % volume band becomes ~volume-proportional), peak/mean ratios fall
        by 6.6-8.7x, and sigma_T moves geometry-dependently (-23 % circle, +11 %
        square). Total absorbed power is unchanged.
      * "legacy": the pre-fix whole-domain np.gradient, retained VERBATIM so any
        historical result can be reproduced bit-for-bit.
    In premix mode the same non-crossing rule is applied to the bed as well (the
    stencil is confined to the doped region and, separately, to its complement), so
    the bed still absorbs; the two regions never difference into each other."""
    if qrf_gradient not in QRF_GRADIENT_MODES:
        raise ValueError(f"unknown qrf_gradient {qrf_gradient!r}; "
                         f"expected one of {QRF_GRADIENT_MODES}")
    if qrf_gradient == "legacy":
        Ex, Ey, Ez = np.gradient(V, grid.h, edge_order=1)
        Ex, Ey, Ez = -Ex, -Ey, -Ez
    else:
        Ex, Ey, Ez = _masked_grad_3d(V, doped, grid.h)
        if premix:
            bx, by, bz = _masked_grad_3d(V, ~np.asarray(doped, dtype=bool),
                                         grid.h)
            Ex, Ey, Ez = Ex + bx, Ey + by, Ez + bz
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey) + Ez * np.conj(Ez))
    Q = 0.5 * np.real(gamma * e2)
    Q = np.clip(np.nan_to_num(Q), 0.0, None)
    if not premix:
        Q[~doped] = 0.0
    # Fixed total power target = density reference * doped volume (SAME for premix on
    # or off at a given geometry, so premix redistributes a fixed total power).
    p_target = p.power_density_w_per_m3 * (int(doped.sum()) * grid.dV)
    p_now = Q.sum() * grid.dV
    if p_now > 1e-18:
        Q *= p_target / p_now            # enforce integral(Q dV) = p_target
    return Q


# --------------------------------------------------------------------------- #
# Materials helpers (gamma + thermal property fields)
# --------------------------------------------------------------------------- #
def _signed_distance_m(part: np.ndarray, h: float) -> np.ndarray:
    """Signed distance (meters) to the binary part surface.

    d > 0 outside the part, d < 0 inside. Magnitude = Euclidean distance to the
    nearest boundary, computed from the two EDTs (inside-to-edge, outside-to-edge).
    A 0.5*h shift centers the zero level on the voxel-face boundary so a single
    solid voxel's surface sits half a voxel from its center, matching the binary
    mask footprint. (Ported verbatim from the audited prototype edge_reg.py.)"""
    part = np.asarray(part, dtype=bool)
    d_out = distance_transform_edt(~part)         # voxels: outside -> nearest inside
    d_in = distance_transform_edt(part)           # voxels: inside  -> nearest outside
    d_vox = np.where(part, -(d_in - 0.5), (d_out - 0.5))
    return d_vox * h


def _material_fraction(part: np.ndarray, h: float, edge_width_m: float) -> np.ndarray:
    """Smooth material indicator m in [0,1]: erf blur of the binary mask over a
    FIXED PHYSICAL width edge_width_m (meters), independent of grid spacing h.

        m(x) = 0.5 * (1 + erf( -d(x) / (sqrt(2) * edge_width_m) ))

    m -> 1 deep inside, 0 deep outside, crossing 0.5 at the nominal boundary.
    edge_width_m <= 0 returns the exact binary indicator part.astype(float)."""
    if edge_width_m <= 0.0:
        return part.astype(float)
    d = _signed_distance_m(part, h)
    from scipy.special import erf as _erf_arr
    return 0.5 * (1.0 + _erf_arr(-d / (np.sqrt(2.0) * edge_width_m)))


def build_gamma(part: np.ndarray, p: Params, sat: np.ndarray | None = None,
                edge_width_m: float = 0.0, h: float | None = None,
                premix_frac: float = 0.0,
                premix_budget: str = "floor_added") -> np.ndarray:
    """gamma = sigma + j*omega*eps0*eps_r from the part mask.

    edge_width_m (meters): optional FIXED-PHYSICAL-WIDTH erf regularization of the
    material boundary (GRID-01/02 fix). edge_width_m=0.0 (default) reproduces the
    original binary behavior BIT-FOR-BIT. edge_width_m>0 requires h (grid spacing,
    meters) and diffuses the sigma/eps boundary over edge_width_m so the same
    physical edge is resolved at every grid. Total absorbed power is renorm-conserved
    downstream in compute_qrf_3d, so regularization changes only the field shape
    (removes the non-physical edge singularity), not the energy.

    premix_frac (PREMIX MODE; default 0.0 == OFF, bit-for-bit original behavior):
    a uniform PREMIXED dopant in the powder, expressed as a FRACTION of the full
    doped conductivity. premix_frac=f raises the WHOLE bed to a finite conductivity
        sigma_premix = f * (sigma_doped - sigma_virgin) + sigma_virgin
    and correspondingly raises the bed permittivity
        eps_premix   = eps_virgin + f * (eps_doped - eps_virgin)
    so the EQS gamma is finite everywhere and the field REDISTRIBUTES (current now
    flows through the conductive bed, not only the part). This is a genuine field
    coupling, not a background heat source. The masked (printed) region depends on
    the dopant-budget variant:
      * premix_budget='floor_added' (variant A, PRIMARY/real use): masked =
        sigma_premix + FULL jetted increment (= sigma_premix + blend*(sigma_doped -
        sigma_virgin)). Premix BOOSTS the mask; total dopant rises.
      * premix_budget='budget_fixed' (variant B, uniformity sweep): masked sigma is
        held at sigma_doped inside the part (= sigma_premix + blend*(sigma_doped -
        sigma_premix)); premix trades against jetted conc so total dopant is ~const.
    premix_frac=0.0 reduces both variants to the original binary/regularized path.
    """
    omega = 2.0 * np.pi * p.freq_hz
    # ---- resolve premixed background conductivity / permittivity ----
    f = float(premix_frac)
    if f < 0.0:
        raise ValueError("premix_frac must be >= 0")
    sigma_premix = p.sigma_virgin + f * (p.sigma_doped - p.sigma_virgin)
    eps_premix = p.eps_virgin + f * (p.eps_doped - p.eps_virgin)
    # masked-region span depends on the dopant budget (only matters when f>0)
    if premix_budget == "floor_added":
        s_span = p.sigma_doped - p.sigma_virgin   # full jet increment on top of premix
        e_span = p.eps_doped - p.eps_virgin
    elif premix_budget == "budget_fixed":
        s_span = p.sigma_doped - sigma_premix     # jet brings premix up to sigma_doped
        e_span = p.eps_doped - eps_premix
    else:
        raise ValueError(f"unknown premix_budget {premix_budget!r}")

    if edge_width_m <= 0.0:
        # ---- binary blend (premix_frac=0 -> bit-for-bit original) ----
        blend = part.astype(float) if sat is None else (part.astype(float) * sat)
        s = sigma_premix + blend * s_span
        e = eps_premix + blend * e_span
        return s + 1j * omega * EPS0 * e
    # ---- regularized path: smooth fractional blend over a fixed physical width ----
    if h is None:
        raise ValueError("build_gamma: edge_width_m>0 requires grid spacing h (m)")
    m = _material_fraction(part, h, edge_width_m)
    blend = m if sat is None else (m * sat)
    s = sigma_premix + blend * s_span
    e = eps_premix + blend * e_span
    return s + 1j * omega * EPS0 * e


# --------------------------------------------------------------------------- #
# S4-COUPLING: sigma(T, rho_rel) feedback, ported from the 2-D solver
#
# SOURCE (read 2026-08-01): rfam_eqs_coupled.py, _FGMFeedback.sigma_at_mask,
# lines 418-449:
#
#     base = (sigma_d0
#             * (1.0 + sigma_temp_coeff * (T_field[mask] - sigma_ref_temp))
#             * (1.0 + sigma_density_coeff * (rho_field[mask] - rho_rel_init)))
#     ... nan_to_num(base, nan=sigma_d0, posinf=25*sigma_d0, neginf=1e-4*sigma_d0)
#
# and its call site, lines 3038-3046 (the periodic `update_interval` re-solve):
#
#     sigma[:, :] = sigma_v
#     sigma[part_mask] = np.clip(fb.sigma_at_mask(...), 1e-4*sigma_d0, 25*sigma_d0)
#
# so the clip bounds below (1e-4 and 25 x sigma_doped) are the 2-D convention
# verbatim. Reference values for the coefficients exist only in ARCHIVED 2-D
# configs (configs/_archive_old/rfam_eqs_comsol_mimic.yaml l. 83-85:
# sigma_temp_coeff_per_K 0.002, sigma_density_coeff 0.6, sigma_ref_temp_c 23.0);
# every CURRENT config (e.g. configs/shape_circle_6min.yaml) sets both to 0.0.
# There is therefore NO validated nonzero value in this repo -- any nonzero use
# is exploratory and must be labelled as such.
#
# TWO DELIBERATE DIFFERENCES from the 2-D port, both documented rather than hidden:
#  1. The factor multiplies the LOCAL baseline sigma (which already carries the
#     edge_width_m erf blend, the premix floor and the FGM `sat` map) instead of
#     the flat scalar sigma_doped. With edge_width_m=0, premix_frac=0 and
#     sat=None -- the S4 configuration -- sigma_local == sigma_doped inside the
#     part, so the two forms are identical there.
#  2. Only the REAL (conduction) part of gamma is coupled. The 2-D solver
#     likewise leaves eps_r untouched by T/rho (it only rebuilds eps_r for an FGM
#     saturation change), so permittivity feedback remains NOT modelled -- a named
#     simplification, not an omission.
# --------------------------------------------------------------------------- #
SIGMA_COUPLING_CLIP_LO = 1e-4      # x p.sigma_doped   (2-D: rfam_eqs_coupled l. 3045)
SIGMA_COUPLING_CLIP_HI = 25.0      # x p.sigma_doped


def apply_sigma_coupling(gamma: np.ndarray, part: np.ndarray, T: np.ndarray,
                         rho_rel: np.ndarray, p: Params) -> np.ndarray:
    """Return gamma with the in-part conductivity rescaled by the sigma(T, rho)
    law (see the module note above).

        sigma_eff = clip( sigma_local * (1 + a (T - T_ref)) * (1 + b (rho - rho_ref)),
                          1e-4 * sigma_doped, 25 * sigma_doped )        inside part
        sigma_eff = sigma_local                                          outside

    a = p.sigma_temp_coeff_per_K, b = p.sigma_density_coeff,
    T_ref = p.sigma_ref_temp_c, rho_ref = p.rho_rel (initial relative density).
    The imaginary (displacement) part of gamma is returned unchanged. With both
    coefficients 0 the factor is exactly 1.0, so gamma is returned bit-for-bit.
    """
    part = np.asarray(part, dtype=bool)
    f = ((1.0 + p.sigma_temp_coeff_per_K * (np.asarray(T, dtype=float) - p.sigma_ref_temp_c))
         * (1.0 + p.sigma_density_coeff * (np.asarray(rho_rel, dtype=float) - p.rho_rel)))
    s0 = np.real(gamma)
    s = np.nan_to_num(s0 * f, nan=p.sigma_doped,
                      posinf=SIGMA_COUPLING_CLIP_HI * p.sigma_doped,
                      neginf=SIGMA_COUPLING_CLIP_LO * p.sigma_doped)
    s = np.clip(s, SIGMA_COUPLING_CLIP_LO * p.sigma_doped,
                SIGMA_COUPLING_CLIP_HI * p.sigma_doped)
    return np.where(part, s, s0) + 1j * np.imag(gamma)


def densify_rate(T: np.ndarray, phi: np.ndarray, rho_rel: np.ndarray,
                 p: Params) -> np.ndarray:
    """physics_dual densification rate d(rho_rel)/dt (>=0). Solid-state Arrhenius
    creep + liquid viscous-capillary flow, both gated by available porosity."""
    Tk = np.maximum(np.array(T, dtype=float, copy=True) + 273.15, 1.0)
    rho_term = np.power(np.clip(1.0 - rho_rel, 0.0, 1.0), p.dens_rho_exp)
    # solid-state
    kss = p.dens_k0_ss * np.exp(-p.dens_ea_ss / (R_GAS * Tk))
    ss_drive = np.power(np.clip(1.0 - phi, 0.0, 1.0), p.dens_phi_solid_exp)
    # liquid viscous-capillary
    eta = p.dens_eta_ref_pa_s * np.exp(
        p.dens_eta_activation / R_GAS * (1.0 / Tk - 1.0 / p.dens_eta_ref_temp_k))
    kliq = p.dens_geom_factor * p.dens_surface_tension / (
        np.maximum(eta, 1e-12) * p.dens_particle_radius_m)
    phi_act = np.clip((phi - p.dens_phi_threshold) / max(1.0 - p.dens_phi_threshold, 1e-9), 0.0, 1.0)
    liq_drive = np.power(phi_act, p.dens_phi_liq_exp)
    return (kss * ss_drive + kliq * liq_drive) * rho_term


def phase_fraction(T: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    # Defensive copy: NumPy 2.2 on Python 3.14 can elide the (T - c) temporary
    # into T's own buffer (in-place mutation side effect). Copy keeps T immutable.
    Te = np.array(T, dtype=float, copy=True)
    arg = (Te - p.t_pc_c) / p.dt_pc_c
    phi = np.clip(arg + 0.5, 0.0, 1.0)
    dphi = np.where(np.abs(arg) <= 0.5, 1.0 / p.dt_pc_c, 0.0)
    return phi, dphi


def enthalpy_from_T(T, rho_cp, rho_L, p: Params):
    """Volumetric enthalpy H(T) [J/m^3], piecewise linear: sensible slope
    rho_cp everywhere plus the latent plateau rho_L ramped linearly across
    the melt window [t_pc - dt_pc/2, t_pc + dt_pc/2]."""
    T = np.asarray(T, dtype=np.float64)
    lo = p.t_pc_c - p.dt_pc_c / 2.0
    frac = np.clip((T - lo) / p.dt_pc_c, 0.0, 1.0)
    return rho_cp * T + rho_L * frac


def T_from_enthalpy(H, rho_cp, rho_L, p: Params):
    """Exact inverse of enthalpy_from_T for scalar-per-voxel rho_cp/rho_L."""
    H = np.asarray(H, dtype=np.float64)
    lo = p.t_pc_c - p.dt_pc_c / 2.0
    H_lo = rho_cp * lo
    H_hi = rho_cp * (lo + p.dt_pc_c) + rho_L
    T_below = H / rho_cp
    # in-window: H = rho_cp*T + rho_L*(T-lo)/dt_pc
    T_window = (H + rho_L * lo / p.dt_pc_c) / (rho_cp + rho_L / p.dt_pc_c)
    T_above = (H - rho_L) / rho_cp
    return np.where(H <= H_lo, T_below,
                    np.where(H >= H_hi, T_above, T_window))


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    sigma_T: float
    T_phi90: np.ndarray
    part: np.ndarray
    Qrf: np.ndarray
    phi_final: np.ndarray
    t_phi90_s: float
    reached: bool
    phi_hist: list = field(default_factory=list)
    T_max_c: float = 0.0
    rho_final: np.ndarray | None = None     # final relative density (densify runs)
    exposure_s: float = 0.0
    # THM-01/02 provenance flag: True iff the per-step dT cap or the
    # temp_min/temp_max clamp BOUND on at least one cell at any step. False in the
    # dissertation production config (both limiters dormant). When True, numbers
    # were silently altered by a numerical limiter -- treat as suspect.
    clamp_bound: bool = False
    # S1 energy audit (Joules over the whole domain, cumulative over the run):
    # in = RF deposited; stored = sensible + latent actually banked in the
    # temperature/phase state; loss = convection + powder-loss + heatsink sinks.
    # residual_frac = (in - stored - loss) / max(in, 1e-30). The standing S1
    # gate: |residual_frac| must be small; a blow-up shows up here first.
    energy_in_j: float = 0.0
    energy_stored_j: float = 0.0
    energy_loss_j: float = 0.0
    energy_residual_frac: float = 0.0
    # Final-timestep temperature field [C], i.e. T at loop exit. Distinct from
    # T_phi90 (the melt-onset read, or the final field when phi_target was never
    # reached). Needed by the S1 analytic benchmarks, which compare the whole
    # final field against a closed-form solution.
    T_final: np.ndarray | None = None
    # S1b / THM-03 provenance. n_substeps_used = thermal substeps per p.dt_s that
    # run() actually took (1 = the legacy single step, i.e. every historical
    # configuration). cfl_violated = True iff the steps ACTUALLY marched broke
    # dt <= CFL_SAFETY * dt_stable_thermal -- only possible with
    # Params.enforce_cfl=False, since the default substeps the violation away.
    # When True, the explicit conduction update was unstable and the numbers are
    # not trustworthy (the THM-01 dT clamp may be hiding the divergence).
    n_substeps_used: int = 1
    cfl_violated: bool = False
    # S4-COUPLING census. n_eqs_solves = FULL EQS solves this call performed
    # (1 = the legacy single pre-loop solve; 0 when qrf_override bypassed it).
    # n_eqs_resolves_skipped = scheduled re-solves that the drift tolerance
    # turned into a cheap pointwise Q rescale instead of a solve.
    n_eqs_solves: int = 1
    n_eqs_resolves_skipped: int = 0


# --------------------------------------------------------------------------- #
# Main solve
# --------------------------------------------------------------------------- #
def _sched_mult(mean_phi: float, schedule) -> float:
    """RF-power multiplier for a fast->soak schedule = [(until_mean_phi, mult), ...].
    Returns the first segment's mult whose threshold the current mean melt fraction
    has not yet passed; last mult thereafter. None/empty => constant 1.0."""
    if not schedule:
        return 1.0
    for thr, m in schedule:
        if mean_phi < thr:
            return float(m)
    return float(schedule[-1][1])


def run(grid: Grid, part: np.ndarray, p: Params, sat: np.ndarray | None = None,
        max_time_s: float = 1500.0, phi_target: float = 0.90,
        densify: bool = False, stop_mean_rho: float | None = None,
        power_schedule=None, verbose: bool = False,
        heatsink_field: np.ndarray | None = None,
        heatsink_h: float = 0.0, heatsink_kgain: float = 0.0,
        edge_width_m: float = 0.0,
        eps_perturb_field: np.ndarray | None = None,
        eps_perturb_value: float = 0.0,
        qrf_override: np.ndarray | None = None,
        powder_loss_mode: str | None = None,
        powder_loss_coeff: float = 0.0,
        powder_path_len_m: float = 0.0,
        powder_loss_region: str = "part",
        premix_frac: float = 0.0,
        premix_budget: str = "floor_added",
        T0_override: np.ndarray | None = None,
        qrf_gradient: str = "masked",
        t_start_s: float = 0.0) -> Result:
    """Coupled 3-D solve. If densify=False (default): stop at the phi=0.90 crossing
    and report sigma_T (relative density held constant). If densify=True: evolve
    relative density (physics_dual) and run the FULL exposure, capturing T_phi90 in
    passing and returning the final density field for the shrinkage analysis.

    edge_width_m (meters): optional fixed-physical-width edge regularization of the
    EQS material boundary (GRID-01/02 fix). 0.0 (default) = original binary behavior.

    qrf_override (W/m^3, shape == part.shape): VALIDATION HOOK (default None ==
    original behavior, bit-for-bit). When provided, REPLACES the internally computed
    Qrf field and SKIPS the EQS solve entirely, so the thermal march is driven by an
    external heating field. Isolates the thermal solver from the EQS for the COMSOL
    audit. The override is used verbatim (no P_abs renorm) so the caller controls the
    absorbed power; it is still zeroed outside the part for safety.

    powder_loss_mode (default None == OFF, bit-for-bit original behavior): a
    PHYSICALLY-MOTIVATED 3-D heat-loss BC, the 3-D analog of the 2.5D solver's
    depth_correction series-resistance loss. The heatr3d domain only resolves a
    small powder buffer around the part with insulating (Neumann) outer walls plus
    top-face convection, so it under-sinks heat and over-predicts late-time T (see
    HEATR3D_COMSOL_validation.md: +90C @60min vs COMSOL). In reality the part is
    embedded in a large powder bed that conducts heat radially to the chamber wall
    at ambient. We model that unresolved sink as a volumetric Newton cooling term
    in the PART region:
        q_loss(x) = h_eff * (T(x) - T_ambient)      [W/m^3]
    subtracted from the energy balance numerator alongside Qrf and q_conv.

    h_eff is anchored in powder conduction physics, not an arbitrary fudge. A part
    voxel loses heat by conducting through a powder shell of characteristic path
    length L_path (powder thickness from part surface to the cold chamber boundary)
    over the powder conductivity k_powder. A series-resistance estimate of the
    volumetric loss coefficient for a body of half-thickness a surrounded by powder is
        h_eff ~ k_powder / (a * L_path)              [W/(m^3 K)]
    (units: W/(m.K) / m^2 = W/(m^3 K)). Two ways to set it:
      * powder_loss_mode='conduction': supply powder_path_len_m (L_path) and h_eff is
        computed from k_powder, the part half-thickness a (cube-root of part volume /2),
        and L_path. Physics-first path.
      * powder_loss_mode='coeff': supply powder_loss_coeff directly as h_eff
        [W/(m^3 K)] (used for calibration sweeps / reporting the value that
        reproduces the COMSOL plateau).
    Default (mode=None) adds NOTHING to the numerator, so sigma_T / T_phi90 / Qrf
    are bit-for-bit identical to the pre-audit baseline (verified with np.array_equal).

    premix_frac / premix_budget (PREMIX MODE; default premix_frac=0.0 == OFF,
    bit-for-bit original): a uniform premixed dopant raises the WHOLE bed conductivity
    so it ENTERS the EQS gamma everywhere and the field redistributes (conductive bed
    carries current and absorbs RF). When premix_frac>0, Qrf is computed everywhere
    (not zeroed outside the part) and the SAME fixed total absorbed power is enforced,
    so premix redistributes a fixed power between bed and part. See build_gamma for the
    floor_added (masked=premix+jet) vs budget_fixed (masked held at sigma_doped)
    variants.

    qrf_gradient (EQS-02, default "masked" since 2026-07-31): passed straight to
    compute_qrf_3d; see its docstring and the module note above it. "masked" is the
    corrected part-confined stencil (a DELIBERATE default flip that changes
    previously published numbers); "legacy" reproduces the pre-fix cross-interface
    np.gradient bit-for-bit. Inert when qrf_override is supplied (no EQS solve).

    t_start_s (S4-COUPLING; default 0.0 == original behavior, bit-for-bit): the
    ABSOLUTE simulated time at which this call's march begins. Only used to place
    the Params.eqs_update_interval_s re-solve schedule, so a march chained through
    T0_override keeps ONE global schedule instead of restarting it in every
    segment. Inert when eqs_update_interval_s == 0."""
    # ---- resolve the volumetric powder-loss coefficient h_eff [W/(m^3 K)] ----
    h_eff_loss = 0.0
    if powder_loss_mode is not None:
        if powder_loss_mode == "coeff":
            h_eff_loss = float(powder_loss_coeff)
        elif powder_loss_mode == "conduction":
            if powder_path_len_m <= 0.0:
                raise ValueError("powder_loss_mode='conduction' requires powder_path_len_m>0")
            part_vol = float(int(np.asarray(part).sum()) * grid.dV)
            a_half = 0.5 * part_vol ** (1.0 / 3.0)        # part half-thickness (m)
            h_eff_loss = p.k_powder / max(a_half * powder_path_len_m, 1e-30)
        else:
            raise ValueError(f"unknown powder_loss_mode {powder_loss_mode!r}")
    premix_on = float(premix_frac) > 0.0
    # T0_override (VALIDATION HOOK; default None == original behavior,
    # bit-for-bit): replaces the uniform preheat initial condition so an
    # analytic initial field (e.g. a Fourier mode) can be marched.
    # (S4-COUPLING moved this block ABOVE the EQS solve -- pure allocation, no
    # arithmetic change -- because the coupled gamma needs the initial T/rho.)
    if T0_override is not None:
        T = np.array(T0_override, dtype=np.float64, copy=True)
        if T.shape != part.shape:
            raise ValueError("T0_override shape must match part.shape")
    else:
        T = np.full(part.shape, p.preheat_c, dtype=np.float64)   # bed preheat
    rho_rel = np.full(part.shape, p.rho_rel, dtype=np.float64)   # evolving density field

    # ---- S4-COUPLING: is the in-march EQS re-solve armed for this call? ----
    _resolve_dt = float(p.eqs_update_interval_s)
    _coupling_on = _resolve_dt > 0.0 and qrf_override is None
    if _resolve_dt > 0.0 and qrf_override is not None:
        logger.warning(
            "S4-COUPLING: eqs_update_interval_s=%.4g s requested but qrf_override "
            "was supplied, so there is no EQS to re-solve. Coupling is INERT for "
            "this call.", _resolve_dt)
    n_eqs_solves = 0
    n_eqs_skipped = 0
    gamma = None
    _sigma_at_last_solve = None
    # fixed absorbed-power target; identical to compute_qrf_3d's internal target
    _p_target = p.power_density_w_per_m3 * (int(np.asarray(part).sum()) * grid.dV)

    if qrf_override is None:
        gamma = build_gamma(part, p, sat, edge_width_m=edge_width_m, h=grid.h,
                            premix_frac=premix_frac, premix_budget=premix_budget)
        # M6 EM-perturbation hook (inert when eps_perturb_field is None or value == 0):
        # a real ceramic heat-sink lattice perturbs the EQS field through its relative
        # permittivity. Add eps_perturb_value to eps_r in the lattice voxels and rebuild
        # gamma's imaginary (displacement) part. The thermal-k path is untouched.
        if eps_perturb_field is not None and eps_perturb_value != 0.0:
            _omega = 2.0 * np.pi * p.freq_hz
            _deps = eps_perturb_value * np.asarray(eps_perturb_field, dtype=float)
            gamma = gamma + 1j * _omega * EPS0 * _deps
        _gamma_base = gamma
        if _coupling_on:
            # EQS-01 cost guard: every re-solve is a FULL solve. Say so, with the
            # expected count, before spending the time.
            _n_exp = int(max(0.0, max_time_s) / _resolve_dt)
            logger.warning(
                "S4-COUPLING ARMED: sigma(T,rho) feedback with an EQS re-solve every "
                "%.4g s -> up to %d ADDITIONAL full EQS solves in this %.4g s segment "
                "(grid %s, N=%d unknowns; EQS-01 certified ceiling n=%d for the full "
                "pipeline). Set eqs_resolve_drift_rtol > 0 to skip low-drift re-solves.",
                _resolve_dt, _n_exp, max_time_s, "x".join(str(s) for s in part.shape),
                int(np.prod(part.shape)), EQS_MAX_GRID_FULL_PHYSICS)
            gamma = apply_sigma_coupling(_gamma_base, part, T, rho_rel, p)
            _sigma_at_last_solve = np.real(gamma).copy()
        V = solve_eqs_3d(gamma, grid, p)
        n_eqs_solves = 1
        Qrf = compute_qrf_3d(V, gamma, grid, p, part, premix=premix_on,
                             qrf_gradient=qrf_gradient)
    else:
        Qrf = np.array(qrf_override, dtype=np.float64, copy=True)
        if Qrf.shape != part.shape:
            raise ValueError("qrf_override shape must match part.shape")
        Qrf[~part] = 0.0

    # Re-solve schedule on ABSOLUTE time: the next multiple of the interval
    # STRICTLY greater than t_start_s (the solve at t_start_s itself is the
    # pre-loop one above, so a chained segmented march never double-solves).
    _next_resolve_t = float("inf")
    if _coupling_on:
        _next_resolve_t = _resolve_dt * (np.floor(t_start_s / _resolve_dt + 1e-9) + 1.0)
    nsteps = int(max_time_s / p.dt_s)
    top = (slice(None), -1, slice(None))     # open top face (y max)
    h = grid.h
    # ---- S1b / THM-03: explicit-conduction stability (findings 13.3) ----
    # The powder bed sets alpha_max, so dt_s = 0.05 s is unstable for n > 169 at
    # L = 0.060 m. With enforce_cfl (default) the thermal update is SUBSTEPPED so
    # each sub-step satisfies the bound; Qrf (and therefore the EQS solve) is held
    # fixed across the substeps of one dt_s, which is exact here because Qrf is
    # computed once per run. n_sub == 1 for every historical configuration, and
    # dt_sub is then p.dt_s exactly, so the default path is arithmetically
    # unchanged.
    dt_stable = dt_stable_thermal(grid, p)
    n_sub = cfl_substeps(grid, p) if p.enforce_cfl else 1
    dt_sub = p.dt_s / n_sub
    cfl_violated = dt_sub > CFL_SAFETY * dt_stable
    if cfl_violated:
        logger.warning(
            "THM-03 CFL VIOLATION: dt_s=%.4g s exceeds %.2f * h^2/(6 alpha_max) "
            "= %.4g s (h=%.4g m, alpha_max=%.4e m^2/s, grid n=%d). The explicit "
            "conduction update is UNSTABLE: the checkerboard mode grows by "
            "|1 - 12 alpha dt/h^2| = %.4f per step and the THM-01 dT clamp may "
            "hide it. Running anyway because Params.enforce_cfl is False; set it "
            "True to auto-substep (n_sub would be %d).",
            p.dt_s, CFL_SAFETY, CFL_SAFETY * dt_stable, h, alpha_max_thermal(p),
            grid.n, abs(1.0 - 2.0 * p.dt_s / dt_stable), cfl_substeps(grid, p))
    elif n_sub > 1:
        logger.info(
            "THM-03: substepping the thermal update %d x (dt_sub=%.4g s) to "
            "satisfy dt <= %.2f * h^2/(6 alpha_max) = %.4g s at n=%d.",
            n_sub, dt_sub, CFL_SAFETY, CFL_SAFETY * dt_stable, grid.n)
    drho_cap = p.dens_max_drho_rate * dt_sub

    # Empty-part-mask guard (S1 benchmarks): a part-free domain is a legitimate
    # pure-conduction / pure-powder configuration, but every part-masked
    # reduction below (phi[part].mean(), T[part].max/std) is a zero-size
    # reduction on it -- .max() raises. INERT for any non-empty mask (the only
    # case that ever ran before: an empty mask crashed), so legacy arithmetic is
    # unchanged; part-free runs now report mean_phi=0.0 and sigma_T/T_max_c=nan.
    _has_part = bool(np.asarray(part).any())
    T_phi90 = None; reached = False; t90 = float("nan"); phi_hist = []
    clamp_bound = False   # THM-01/02 manifest flag (latched if a clamp binds)
    # ---- S1 energy audit state (read-only w.r.t. the solve) ----
    dV = grid.dV
    e_in = 0.0
    e_loss = 0.0
    e_stored_acc = 0.0
    # THM-03: the march is over nsteps * n_sub SUBSTEPS of dt_sub. With n_sub=1
    # (every historical configuration) this is exactly the original loop:
    # it == the step index, isub == 0, dt_sub == p.dt_s.
    for _it_sub in range(nsteps * n_sub):
        it, isub = divmod(_it_sub, n_sub)
        # ---- S4-COUPLING: scheduled EQS re-solve (inert unless armed) -------- #
        # Placed at the TOP of a full step, before any thermal work, so the drive
        # used by step `it` is the one belonging to time t_start_s + it*dt_s.
        if _coupling_on and isub == 0 and (t_start_s + it * p.dt_s) >= _next_resolve_t - 1e-12:
            while (t_start_s + it * p.dt_s) >= _next_resolve_t - 1e-12:
                _next_resolve_t += _resolve_dt
            gamma_new = apply_sigma_coupling(_gamma_base, part, T, rho_rel, p)
            sigma_new = np.real(gamma_new)
            # D1 adaptive skip (EQS-01 cost control): if the conductivity has
            # barely moved since the last FULL solve, do not pay for another one;
            # rescale the frozen field pointwise and re-renormalize the power.
            # Ported from rfam_eqs_coupled.py lines 3065-3082.
            _skip = False
            if p.eqs_resolve_drift_rtol > 0.0 and _sigma_at_last_solve is not None:
                _drift = float(np.max(np.abs(sigma_new - _sigma_at_last_solve))) / max(
                    p.sigma_doped, 1e-30)
                _skip = _drift < p.eqs_resolve_drift_rtol
            if _skip:
                with np.errstate(divide="ignore", invalid="ignore"):
                    _ratio = np.where(_sigma_at_last_solve > 0.0,
                                      sigma_new / _sigma_at_last_solve, 1.0)
                Q_new = np.clip(np.nan_to_num(Qrf * _ratio), 0.0, None)
                if not premix_on:
                    Q_new[~part] = 0.0
                _p_now = Q_new.sum() * dV
                if _p_now > 1e-18:
                    Q_new *= _p_target / _p_now
                Qrf = Q_new
                n_eqs_skipped += 1
            else:
                gamma = gamma_new
                V = solve_eqs_3d(gamma, grid, p)
                Qrf = compute_qrf_3d(V, gamma, grid, p, part, premix=premix_on,
                                     qrf_gradient=qrf_gradient)
                _sigma_at_last_solve = sigma_new.copy()
                n_eqs_solves += 1
        phi, dphi = phase_fraction(T, p)
        pmult = _sched_mult(float(phi[part].mean()) if _has_part else 0.0,
                            power_schedule)
        # property fields (density-dependent solid props, per voxel)
        rho_s_eff = p.rho_powder + rho_rel * (p.rho_solid - p.rho_powder)
        k_s_eff = p.k_powder + rho_rel * (p.k_solid - p.k_powder)
        rho = np.full(part.shape, p.rho_powder)
        k = np.full(part.shape, p.k_powder)
        cp = np.full(part.shape, p.cp_powder)
        rho[part] = (1 - phi[part]) * rho_s_eff[part] + phi[part] * p.rho_liquid
        k[part] = (1 - phi[part]) * k_s_eff[part] + phi[part] * p.k_liquid
        cp[part] = (1 - phi[part]) * p.cp_solid + phi[part] * p.cp_liquid
        cp_eff = cp + p.latent_j_per_kg * dphi

        # IDEA 4: patterned heat-sink k-gain (inert when heatsink_field is None).
        # High-k channels where the lattice is present (NOT re-masked to part).
        if heatsink_field is not None and heatsink_kgain != 0.0:
            k = k * (1.0 + heatsink_kgain * np.asarray(heatsink_field, dtype=float))

        # div(k grad T): harmonic face conductivity, 6 faces
        div = np.zeros(part.shape)
        for ax in range(3):
            for s in (-1, +1):
                k_nb = np.roll(k, -s, axis=ax)
                T_nb = np.roll(T, -s, axis=ax)
                kf = _harmonic(k, k_nb)
                flux = kf * (T_nb - T) / (h * h)
                sl = [slice(None)] * 3
                sl[ax] = (-1 if s == +1 else 0)
                flux[tuple(sl)] = 0.0          # Neumann walls
                div += flux

        q_conv = np.zeros(part.shape)
        q_conv[top] = p.conv_h * (np.array(T[top], copy=True) - p.preheat_c) / h

        # Defensive: build the source in a fresh buffer so NumPy's in-place
        # temporary elision cannot corrupt the persistent Qrf array.
        num = np.array(div, copy=True)
        num += (Qrf * pmult) if pmult != 1.0 else Qrf
        num -= q_conv
        # 3-D powder heat-loss BC (default OFF; inert when h_eff_loss == 0.0).
        # Volumetric Newton sink representing conduction through the surrounding
        # powder bed to the chamber ambient (analog of 2.5D depth_correction).
        # Subtracted from the energy numerator [W/m^3].
        #   powder_loss_region (default 'part', bit-for-bit original): where the
        #   sink acts. 'part' = part voxels only (original part-conduction model,
        #   for solid-part long-exposure studies). 'bed' = powder voxels only
        #   (~part). 'all' = every voxel (the PHYSICAL premix case: when the bed
        #   is conductive premixed powder it heats up AND conducts to the chamber
        #   walls everywhere, so the loss must act on the powder bed, not only the
        #   part -- otherwise the bed-overheating selectivity test is meaningless).
        if h_eff_loss != 0.0:
            q_loss = np.zeros(part.shape)
            if powder_loss_region == "part":
                q_loss[part] = h_eff_loss * (T[part] - p.ambient_c)
            elif powder_loss_region == "bed":
                bed = ~part
                q_loss[bed] = h_eff_loss * (T[bed] - p.ambient_c)
            elif powder_loss_region == "all":
                q_loss = h_eff_loss * (T - p.ambient_c)
            else:
                raise ValueError(f"unknown powder_loss_region {powder_loss_region!r}")
            num -= q_loss
        # IDEA 4: patterned heat-sink cold-loss to ambient (energy-removing).
        if heatsink_field is not None and heatsink_h != 0.0:
            num -= heatsink_h * np.asarray(heatsink_field, dtype=float) * (T - p.ambient_c)
        # ---- S1 energy audit (per step, before the dT clamp) ----
        e_in += float((Qrf * (pmult if pmult != 1.0 else 1.0)).sum()) * dV * dt_sub
        e_loss += float(q_conv.sum()) * dV * dt_sub
        if h_eff_loss != 0.0:
            e_loss += float(q_loss.sum()) * dV * dt_sub
        if heatsink_field is not None and heatsink_h != 0.0:
            e_loss += float((heatsink_h * np.asarray(heatsink_field, dtype=float)
                             * (T - p.ambient_c)).sum()) * dV * dt_sub
        if p.phase_update == "enthalpy":
            # Energy-conserving update: deposit num*dt into volumetric
            # enthalpy and invert exactly. Latent uses the SOLID density
            # basis for a consistent H(T) (audit uses the same basis).
            rho_cp_map = rho * cp                     # sensible slope, J/(m^3 K)
            rho_L_map = np.zeros(part.shape)
            rho_L_map[part] = rho_s_eff[part] * p.latent_j_per_kg
            H = enthalpy_from_T(T, rho_cp_map, rho_L_map, p)
            H = H + dt_sub * np.nan_to_num(num)
            T_new = T_from_enthalpy(H, rho_cp_map, rho_L_map, p)
            dT_raw = T_new - T
            dT = np.clip(dT_raw, -p.max_dt_step_c, p.max_dt_step_c)
        else:
            dTdt = num / np.maximum(rho * cp_eff, 1e-9)
            dT_raw = dt_sub * np.nan_to_num(dTdt)
            dT = np.clip(dT_raw, -p.max_dt_step_c, p.max_dt_step_c)
        # THM-01 per-step dT-cap binding diagnostic (clamp arithmetic above is
        # UNCHANGED -> output bit-identical; cap is dormant in production).
        _n_dT_clip = int(np.count_nonzero(np.abs(dT_raw) > p.max_dt_step_c))
        if _n_dT_clip > 0:
            _frac = _n_dT_clip / dT_raw.size
            clamp_bound = True
            logger.warning(
                "THM-01 per-step dT cap BOUND at it=%d: %.4f%% of cells clipped to "
                "+-%.3f C (max raw |dT|=%.3f C). The explicit limiter is altering "
                "the physics; results at these cells are non-physical.",
                it, 100.0 * _frac, p.max_dt_step_c, float(np.max(np.abs(dT_raw))),
            )
        T_cand = np.array(T, copy=True) + dT
        # THM-02 temp_min/temp_max clamp binding diagnostic (clamp UNCHANGED).
        _n_temp_clip = int(np.count_nonzero((T_cand > p.temp_max_c) | (T_cand < p.temp_min_c)))
        if _n_temp_clip > 0:
            clamp_bound = True
            logger.warning(
                "THM-02 temperature clamp BOUND at it=%d: %.4f%% of cells hit "
                "[temp_min=%.1f, temp_max=%.1f] C. The clamp is masking a "
                "runaway/instability; results at these cells are non-physical.",
                it, 100.0 * _n_temp_clip / T_cand.size, p.temp_min_c, p.temp_max_c,
            )
        T_prev = T
        T = np.clip(T_cand, p.temp_min_c, p.temp_max_c)

        phi_now = phase_fraction(T, p)[0]
        # ---- S1 audit v2: bank stored energy with THIS step's property maps ----
        # v1 booked the whole run with initial-state solid properties and drifted
        # to +0.38 on a HEALTHY molten run (the solver blends cp -> cp_liquid and
        # rho -> rho_liquid once phi > 0). Banking per step with the same rho, cp,
        # rho_s_eff maps the update used makes the gate meaningful AT melt, which
        # is where the instability lives.
        # DEVIATION from the plan snippet (which banks `dT`): this banks the
        # ACTUAL applied change T - T_prev, i.e. AFTER the temp_min/temp_max
        # clamp. Banking `dT` would credit energy into a voxel pinned at
        # temp_max_c and so erase the saturation surplus that
        # test_legacy_phase_update_skips_latent_on_window_crossing pins
        # (resid > 0.10). The audit must show clamp-destroyed energy, not hide it.
        e_stored_acc += float((rho * cp * (T - T_prev)).sum()) * dV
        e_stored_acc += float((rho_s_eff[part] * p.latent_j_per_kg
                               * (phi_now[part] - phi[part])).sum()) * dV
        if densify:
            drho = np.clip(dt_sub * densify_rate(T, phi_now, rho_rel, p), 0.0, drho_cap)
            rho_new = np.array(rho_rel, copy=True)
            rho_new[part] = np.clip(rho_rel[part] + drho[part], 0.0, 1.0)
            rho_rel = rho_new

        mean_phi = float(phi_now[part].mean()) if _has_part else 0.0
        # THM-03: phi_hist stays ONE ENTRY PER dt_s STEP (its historical meaning),
        # so it is appended on the last substep of a step -- or on whichever
        # substep exits the loop, matching the original append-then-break order.
        if not reached and mean_phi >= phi_target:
            # sub-step resolved melt-onset time; == (it+1)*p.dt_s when n_sub == 1
            T_phi90 = T.copy(); reached = True
            t90 = (it + (isub + 1) / n_sub) * p.dt_s
            if not densify:
                phi_hist.append(mean_phi)
                break
        # densify runs stop at a target MEAN relative density (realistic process
        # stop) so baseline vs FGM are compared at matched densification, not at
        # over-exposed saturation where all non-uniformity is erased.
        if densify and stop_mean_rho is not None and float(rho_rel[part].mean()) >= stop_mean_rho:
            phi_hist.append(mean_phi)
            break
        if isub < n_sub - 1:
            continue                       # more substeps before this step ends
        phi_hist.append(mean_phi)
        if verbose and it % 200 == 0:
            print(f"  t={it*p.dt_s:6.1f}s  Tmax={(T[part].max() if _has_part else float('nan')):6.1f}  phi={mean_phi:.3f}  "
                  f"rho={(rho_rel[part].mean() if _has_part else float('nan')):.3f}")

    # ---- S1 energy audit: stored energy accumulated per step (v2) ----
    # Sensible + latent were banked inside the loop with each step's own
    # property maps (see the audit v2 block above), so the books follow the
    # solver's property blending instead of a fixed initial-state model.
    e_stored = e_stored_acc
    e_resid_frac = (e_in - e_stored - e_loss) / max(e_in, 1e-30)

    if T_phi90 is None:
        # P0b instrumentation (2026-07-30): the melt-onset read state never
        # existed in this run (mean phi never crossed phi_target within
        # max_time_s), so every T_phi90-derived metric below (sigma_T, T_max_c,
        # phi_final) is a FINAL-TIMESTEP read, not a melt-onset read. Numerical
        # behavior is unchanged (T_phi90 = final T, exactly as before); this
        # warning only makes the fallback loud. Consumers must check
        # Result.reached before quoting any of these as melt-onset values.
        logger.warning(
            "MELT-ONSET FALLBACK: mean melt fraction never crossed "
            "phi_target=%.2f within max_time_s=%.1f s (final mean phi=%.4f). "
            "sigma_T / T_phi90 / T_max_c are FINAL-TIMESTEP reads, not "
            "melt-onset reads. Check Result.reached before quoting.",
            phi_target, max_time_s, (phi_hist[-1] if phi_hist else float("nan")),
        )
        T_phi90 = T.copy()
    sigma_T = float(T_phi90[part].std()) if _has_part else float("nan")
    # ---- S1 standing conservation gate (always on, one line per solve) ----
    # Spec (Gate S1): "standing energy-conservation gate on every heatr3d solve
    # (|residual| / integrated dose), printed in every run summary." Healthy
    # runs read |residual_frac| <~ 1e-2 (machine precision on the enthalpy
    # scheme); a melt-onset blow-up shows up here first, usually together with
    # CLAMP-BOUND (THM-01/02 limiter latched).
    print(f"  [s1-energy] in={e_in:.1f} J stored={e_stored:.1f} J "
          f"loss={e_loss:.1f} J residual_frac={e_resid_frac:+.4f}"
          f"{'  CLAMP-BOUND' if clamp_bound else ''}")
    return Result(sigma_T=sigma_T, T_phi90=T_phi90, part=part, Qrf=Qrf,
                  phi_final=phase_fraction(T_phi90, p)[0], t_phi90_s=t90,
                  reached=reached, phi_hist=phi_hist, T_max_c=(float(T_phi90[part].max()) if _has_part
                           else float("nan")),
                  rho_final=(rho_rel if densify else None),
                  exposure_s=(nsteps * p.dt_s if densify else t90),
                  clamp_bound=clamp_bound,
                  energy_in_j=e_in,
                  energy_stored_j=e_stored,
                  energy_loss_j=e_loss,
                  energy_residual_frac=e_resid_frac,
                  T_final=T.copy(),
                  n_substeps_used=n_sub,
                  cfl_violated=cfl_violated,
                  n_eqs_solves=n_eqs_solves,
                  n_eqs_resolves_skipped=n_eqs_skipped)


def make_fgm(res: Result, magnitude: float = 1.0, baseline: float = 0.5,
             bpp: int = 2, proxy: np.ndarray | None = None) -> np.ndarray:
    """Per-voxel 3-D FGM by the proportional inverse rule (high proxy -> less
    dopant). Default proxy is the T_phi90 melt-onset field. Pass proxy=rho_final
    for a DENSIFICATION-targeted map (regions that ended denser get less dopant,
    under-densified regions get more), which flattens the final density directly."""
    P = np.array(res.T_phi90 if proxy is None else proxy, dtype=float, copy=True)
    m = res.part
    lo, hi = np.percentile(P[m], [2.0, 98.0])
    norm = np.clip((P - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    sat = baseline + magnitude * ((1.0 - norm) - baseline)
    sat = np.clip(sat, 0.0, 1.0)
    levels = (1 << bpp) - 1
    sat = np.round(sat * levels) / levels          # quantize to bpp
    sat = np.array(sat, copy=True)
    sat[~m] = 0.0
    return sat


def shrinkage_factors(S_V: np.ndarray, xy_frac: float = 0.04):
    """SINGLE source of truth for the anisotropic sintering shrinkage law.

    Given a local VOLUME shrink ratio S_V = rho_green / rho_final (<=1; a densified
    voxel shrinks), partition the log-volume between in-plane (xy) and through-
    thickness (z) directions with one free knob ``xy_frac``:

        lambda_xy = S_V ** xy_frac                  (in-plane linear shrink)
        lambda_z  = S_V / lambda_xy ** 2            (through-thickness linear shrink)

    so that lambda_xy**2 * lambda_z = S_V exactly (volume conserving for any knob).

    Physical meaning of ``xy_frac`` (the SAME family, one uncalibrated parameter):
        * 0.0   = fully in-plane-constrained: lambda_xy == 1, ALL shrink goes into
                  z. Rigid powder-bed / fully constrained substrate limit.
        * 0.04  = strong lateral constraint (CURRENT default assumption; powder-bed
                  friction holds the footprint nearly rigid, z collapses). NOT yet
                  calibrated against measured sinter data.
        * 0.10  = moderate lateral constraint.
        * 1/3   = free isotropic sinter: lambda_xy == lambda_z == S_V**(1/3). The
                  classic isotropic-shrinkage law (no directional constraint).

    Returns (lambda_xy, lambda_z), both clipped to (1e-6, 1.0). This is the ONLY
    place the law is defined; all call sites (shrinkage_analysis, the shrinkage-
    magnitude measure, the combination prewarp study) MUST route through here so a
    single xy_frac controls every reported shrink/prewarp magnitude.
    """
    S_V = np.clip(np.asarray(S_V, float), 1e-6, 1.0)
    lam_xy = np.power(S_V, xy_frac)
    lam_z = np.clip(S_V / np.maximum(lam_xy ** 2, 1e-9), 1e-6, 1.0)
    return lam_xy, lam_z


def shrinkage_analysis(res: Result, p: Params, h: float, xy_frac: float = 0.04,
                       layer_thickness_mm: float = 0.10,
                       target_final_height_mm: float | None = None) -> dict:
    """Anisotropic (Z-dominant) sintering shrinkage from the final density field.

    Local volume shrink S_V = rho_green / rho_final (mass conservation). The law
    (lambda_xy, lambda_z) is supplied by ``shrinkage_factors`` (single source of
    truth), with ``xy_frac`` the lateral-constraint knob. Then:
      * per-(x,y) column final height H = sum_z h*lambda_z  (Lagrangian Z compaction)
      * warpage = scatter of effective column shrink vs the uniform-shrink ideal
      * inverse layer count = green layers needed so the part compacts to target.
    Returns scalar metrics plus the H(x,y) height map and lambda_z field.
    """
    m = res.part
    rho_f = np.clip(np.asarray(res.rho_final, float), 1e-6, 1.0)
    S_V = np.clip(p.rho_rel / rho_f, 1e-6, 1.0)        # <=1 (densified -> shrinks)
    lam_xy, lam_z = shrinkage_factors(S_V, xy_frac=xy_frac)  # single source of truth
    lam_z = np.where(m, lam_z, 0.0)

    # per-(x,y) column compaction along z (axis=2)
    nominal_col = m.sum(axis=2)                          # voxel count per column
    H_final = (lam_z * m).sum(axis=2) * h               # final height per column (m)
    cols = nominal_col > 0
    eff_lam_z = np.zeros_like(H_final)
    eff_lam_z[cols] = H_final[cols] / (nominal_col[cols] * h)   # effective column lambda_z

    lz_mean = float(lam_z[m].mean())
    lz_xy_mean = float(lam_xy[m].mean())
    z_shrink_pct = 100.0 * (1.0 - lz_mean)
    xy_shrink_pct = 100.0 * (1.0 - lz_xy_mean)
    # warpage: column-to-column scatter of effective Z shrink (uniform -> 0)
    warp_std_pct = 100.0 * float(eff_lam_z[cols].std())
    warp_range_pct = 100.0 * float(eff_lam_z[cols].max() - eff_lam_z[cols].min())

    # inverse: green layers to reach a target final height
    layer_mm = layer_thickness_mm
    nominal_h_mm = float(nominal_col.max()) * h * 1e3    # green build height (mm)
    final_h_mm = float(H_final.max()) * 1e3
    if target_final_height_mm is None:
        target_final_height_mm = final_h_mm
    green_h_needed_mm = target_final_height_mm / max(lz_mean, 1e-6)
    green_layers = int(np.ceil(green_h_needed_mm / layer_mm))
    final_layers = int(np.ceil(target_final_height_mm / layer_mm))
    extra_layers = green_layers - final_layers

    return {
        "rho_final_mean": round(float(rho_f[m].mean()), 4),
        "rho_final_std": round(float(rho_f[m].std()), 4),
        "z_shrink_pct": round(z_shrink_pct, 2),
        "xy_shrink_pct": round(xy_shrink_pct, 2),
        "warp_std_pct": round(warp_std_pct, 3),
        "warp_range_pct": round(warp_range_pct, 2),
        "nominal_height_mm": round(nominal_h_mm, 2),
        "final_height_mm": round(final_h_mm, 2),
        "target_final_height_mm": round(float(target_final_height_mm), 2),
        "layer_thickness_mm": layer_mm,
        "green_layers": green_layers,
        "final_layers": final_layers,
        "extra_layers": extra_layers,
        "layer_multiplier": round(1.0 / max(lz_mean, 1e-6), 3),
        "_H_final": H_final, "_eff_lam_z": eff_lam_z, "_cols": cols, "_lam_z": lam_z,
    }


def sinter_metrics(res: Result, phi_thresh: float = 0.5) -> dict:
    """Compare the sintered body (melt fraction >= thresh at T_phi90) to the
    nominal CAD part. Quantifies under-sintering / shape deviation."""
    sintered = (res.phi_final >= phi_thresh) & res.part
    nominal = res.part
    inter = int((sintered & nominal).sum())
    vs, vn = int(sintered.sum()), int(nominal.sum())
    dice = 2.0 * inter / max(vs + vn, 1)
    return {"nominal_vox": vn, "sintered_vox": vs,
            "sintered_frac": vs / max(vn, 1), "dice": dice,
            "unsintered_vox": vn - inter}


if __name__ == "__main__":
    t0 = time.time()
    g = Grid(n=40)
    part = make_geometry(g, "cylinder", diam=0.020)
    print(f"cylinder part voxels={part.sum()}  grid={g.n}^3  h={g.h*1e3:.2f}mm")
    r = run(g, part, Params(), verbose=True)
    print(f"sigma_T={r.sigma_T:.2f} C  T_phi90 reached={r.reached} @ {r.t_phi90_s:.1f}s  "
          f"Tmax={r.T_max_c:.1f}  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# CFG-01 drift check (non-fatal). This module is the source of truth at import
# time; the block below only WARNS if a reachable canonical copy has diverged
# from the snapshot this file was synced from. It never raises and never alters
# behavior, so production output is unaffected and fresh clones (no canonical
# path) silently skip it.
# ---------------------------------------------------------------------------
SYNCED_FROM_SHA256 = "4d51c55fa1e81aff880dc19fd2e5bbdc4a2f82bb0588ead08f1fce37d2260dfb"


def _check_canonical_drift() -> None:
    import hashlib as _hashlib
    import os as _os
    from pathlib import Path as _Path

    env = _os.environ.get("HEATR3D_CANONICAL_PATH")
    if env:
        cand = _Path(env).expanduser()
    else:
        cand = (
            _Path(__file__).resolve().parent
            / "../../../dissertation_materials/analysis-3dfgm/heatr3d.py"
        )
    try:
        cand = cand.resolve()
        if not cand.is_file() or cand == _Path(__file__).resolve():
            return
        actual = _hashlib.sha256(cand.read_bytes()).hexdigest()
    except OSError:
        return
    if actual != SYNCED_FROM_SHA256:
        logger.warning(
            "CFG-01: canonical heatr3d.py at %s has DRIFTED from this synced copy "
            "(canonical=%s, synced_from=%s). Re-sync geo-prewarp/heatr3d.py and "
            "update SYNCED_FROM_SHA256 after reviewing the canonical change.",
            cand, actual, SYNCED_FROM_SHA256,
        )


try:
    _check_canonical_drift()
except Exception:  # never let the drift check break import
    pass
