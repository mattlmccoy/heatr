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
    if use_iter:
        try:
            ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=12)
            M = spla.LinearOperator(A.shape, ilu.solve, dtype=np.complex128)
            V, info = spla.bicgstab(A, b, rtol=1e-8, atol=0.0, maxiter=2000, M=M)
            if info != 0 or not np.all(np.isfinite(V)):
                V = None       # fall back to direct
        except Exception:
            V = None
    if V is None:
        V = spla.spsolve(A, b)
    if not np.all(np.isfinite(V)):
        raise RuntimeError("EQS solve produced non-finite values")
    return V.reshape(nx, ny, nz)


def compute_qrf_3d(V: np.ndarray, gamma: np.ndarray, grid: Grid, p: Params,
                   doped: np.ndarray, premix: bool = False) -> np.ndarray:
    """Volumetric RF heating Qrf = 0.5*Re(gamma|E|^2), renormalized to a fixed total
    absorbed power (enforce_generator_power).

    premix=False (default, bit-for-bit original): Qrf is zeroed OUTSIDE the doped
    region (jet-only: only the printed part absorbs) and the fixed total power target
    is power_density * doped_volume.

    premix=True: the premixed conductive bed absorbs RF too, so Qrf is kept
    EVERYWHERE (not zeroed outside the part). The SAME total power target as the
    premix-off case for this geometry (power_density * doped_volume) is enforced over
    the whole domain, so premix REDISTRIBUTES a FIXED total absorbed power between the
    bed and the part -- it does not invent energy."""
    Ex, Ey, Ez = np.gradient(V, grid.h, edge_order=1)
    Ex, Ey, Ez = -Ex, -Ey, -Ez
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
        premix_budget: str = "floor_added") -> Result:
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
    variants."""
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
        V = solve_eqs_3d(gamma, grid, p)
        Qrf = compute_qrf_3d(V, gamma, grid, p, part, premix=premix_on)
    else:
        Qrf = np.array(qrf_override, dtype=np.float64, copy=True)
        if Qrf.shape != part.shape:
            raise ValueError("qrf_override shape must match part.shape")
        Qrf[~part] = 0.0

    T = np.full(part.shape, p.preheat_c, dtype=np.float64)       # bed preheat
    rho_rel = np.full(part.shape, p.rho_rel, dtype=np.float64)   # evolving density field
    nsteps = int(max_time_s / p.dt_s)
    top = (slice(None), -1, slice(None))     # open top face (y max)
    h = grid.h
    drho_cap = p.dens_max_drho_rate * p.dt_s

    T_phi90 = None; reached = False; t90 = float("nan"); phi_hist = []
    clamp_bound = False   # THM-01/02 manifest flag (latched if a clamp binds)
    for it in range(nsteps):
        phi, dphi = phase_fraction(T, p)
        pmult = _sched_mult(float(phi[part].mean()), power_schedule)
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
        dTdt = num / np.maximum(rho * cp_eff, 1e-9)
        dT_raw = p.dt_s * np.nan_to_num(dTdt)
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
        T = np.clip(T_cand, p.temp_min_c, p.temp_max_c)

        phi_now = phase_fraction(T, p)[0]
        if densify:
            drho = np.clip(p.dt_s * densify_rate(T, phi_now, rho_rel, p), 0.0, drho_cap)
            rho_new = np.array(rho_rel, copy=True)
            rho_new[part] = np.clip(rho_rel[part] + drho[part], 0.0, 1.0)
            rho_rel = rho_new

        mean_phi = float(phi_now[part].mean())
        phi_hist.append(mean_phi)
        if not reached and mean_phi >= phi_target:
            T_phi90 = T.copy(); reached = True; t90 = (it + 1) * p.dt_s
            if not densify:
                break
        # densify runs stop at a target MEAN relative density (realistic process
        # stop) so baseline vs FGM are compared at matched densification, not at
        # over-exposed saturation where all non-uniformity is erased.
        if densify and stop_mean_rho is not None and float(rho_rel[part].mean()) >= stop_mean_rho:
            break
        if verbose and it % 200 == 0:
            print(f"  t={it*p.dt_s:6.1f}s  Tmax={T[part].max():6.1f}  phi={mean_phi:.3f}  "
                  f"rho={rho_rel[part].mean():.3f}")

    if T_phi90 is None:
        T_phi90 = T.copy()
    sigma_T = float(T_phi90[part].std())
    return Result(sigma_T=sigma_T, T_phi90=T_phi90, part=part, Qrf=Qrf,
                  phi_final=phase_fraction(T_phi90, p)[0], t_phi90_s=t90,
                  reached=reached, phi_hist=phi_hist, T_max_c=float(T_phi90[part].max()),
                  rho_final=(rho_rel if densify else None),
                  exposure_s=(nsteps * p.dt_s if densify else t90),
                  clamp_bound=clamp_bound)


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
SYNCED_FROM_SHA256 = "5243529f819e302de2cd063dac1646a181914ef96d5561a72493461cb1985ea7"


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
