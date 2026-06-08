#!/usr/bin/env python3
"""heatr3d.py -- a genuine 3-D coupled EQS -> RF-heating -> thermal -> melt solver.

A faithful 3-D port of the 2-D HEATR physics (rfam_eqs_coupled.py). Unlike the
stacked-2-D pilot, this solves the real 3-D electroquasistatic field and 3-D heat
equation, so z-conduction and out-of-plane field coupling are resolved rather than
approximated. Used to study how a part whose cross-section VARIES along the build
axis sinters, and how the sintered body deviates from the nominal CAD shape.

Physics (matched to shape_circle_6min.yaml):
  EQS:     div(gamma grad V) = 0,  gamma = sigma + j*omega*eps0*eps_r
           Dirichlet electrodes on the two y-planes; Neumann (insulating) elsewhere;
           harmonic face averaging for high-contrast media.
  Heating: Qrf = 0.5 * Re(gamma |E|^2), >=0, zero outside doped; scaled so
           integral(Qrf dV) = P_abs = generator_power * transfer_efficiency.
  Thermal: dT/dt = (div(k grad T) + Qrf - q_conv)/(rho * cp_eff),
           cp_eff = cp + L*dphi/dT (apparent heat capacity),
           phi = clip((T - Tpc)/dTpc + 0.5, 0, 1)  (COMSOL Heaviside).
           Convection h on the open top (y-max) face only.

Geometry convention: y = electrode/field axis (plates at y_min/y_max);
x = lateral; z = build axis (cross-section may vary with z).

Simplification vs 2-D HEATR: relative density held at its initial value (the
baseline config has zero sigma-density and sigma-temperature coupling, so gamma is
constant and the EQS field is solved ONCE per configuration). Stated in the writeup.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0 = 8.8541878128e-12


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
           'sphere', 'cone' (radius shrinks with z), 'dumbbell'."""
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    r_xy = np.sqrt(X ** 2 + Y ** 2)
    R = diam / 2.0
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


def solve_eqs_3d(gamma: np.ndarray, grid: Grid, p: Params) -> np.ndarray:
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
    V = spla.spsolve(A, b)
    if not np.all(np.isfinite(V)):
        raise RuntimeError("EQS solve produced non-finite values")
    return V.reshape(nx, ny, nz)


def compute_qrf_3d(V: np.ndarray, gamma: np.ndarray, grid: Grid, p: Params,
                   doped: np.ndarray) -> np.ndarray:
    Ex, Ey, Ez = np.gradient(V, grid.h, edge_order=1)
    Ex, Ey, Ez = -Ex, -Ey, -Ez
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey) + Ez * np.conj(Ez))
    Q = 0.5 * np.real(gamma * e2)
    Q = np.clip(np.nan_to_num(Q), 0.0, None)
    Q[~doped] = 0.0
    # Target total power = density reference * doped volume (constant power density).
    p_target = p.power_density_w_per_m3 * (int(doped.sum()) * grid.dV)
    p_now = Q.sum() * grid.dV
    if p_now > 1e-18:
        Q *= p_target / p_now            # enforce integral(Q dV) = p_target
    return Q


# --------------------------------------------------------------------------- #
# Materials helpers (gamma + thermal property fields)
# --------------------------------------------------------------------------- #
def build_gamma(part: np.ndarray, p: Params, sat: np.ndarray | None = None) -> np.ndarray:
    omega = 2.0 * np.pi * p.freq_hz
    s = np.full(part.shape, 0.0)
    e = np.full(part.shape, p.eps_virgin)
    blend = part.astype(float) if sat is None else (part.astype(float) * sat)
    s = p.sigma_virgin + blend * (p.sigma_doped - p.sigma_virgin)
    e = p.eps_virgin + blend * (p.eps_doped - p.eps_virgin)
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


# --------------------------------------------------------------------------- #
# Main solve
# --------------------------------------------------------------------------- #
def run(grid: Grid, part: np.ndarray, p: Params, sat: np.ndarray | None = None,
        max_time_s: float = 1500.0, phi_target: float = 0.90,
        densify: bool = False, stop_mean_rho: float | None = None,
        verbose: bool = False) -> Result:
    """Coupled 3-D solve. If densify=False (default): stop at the phi=0.90 crossing
    and report sigma_T (relative density held constant). If densify=True: evolve
    relative density (physics_dual) and run the FULL exposure, capturing T_phi90 in
    passing and returning the final density field for the shrinkage analysis."""
    gamma = build_gamma(part, p, sat)
    V = solve_eqs_3d(gamma, grid, p)
    Qrf = compute_qrf_3d(V, gamma, grid, p, part)

    T = np.full(part.shape, p.ambient_c, dtype=np.float64)
    rho_rel = np.full(part.shape, p.rho_rel, dtype=np.float64)   # evolving density field
    nsteps = int(max_time_s / p.dt_s)
    top = (slice(None), -1, slice(None))     # open top face (y max)
    h = grid.h
    drho_cap = p.dens_max_drho_rate * p.dt_s

    T_phi90 = None; reached = False; t90 = float("nan"); phi_hist = []
    for it in range(nsteps):
        phi, dphi = phase_fraction(T, p)
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
        q_conv[top] = p.conv_h * (np.array(T[top], copy=True) - p.ambient_c) / h

        # Defensive: build the source in a fresh buffer so NumPy's in-place
        # temporary elision cannot corrupt the persistent Qrf array.
        num = np.array(div, copy=True)
        num += Qrf
        num -= q_conv
        dTdt = num / np.maximum(rho * cp_eff, 1e-9)
        dT = np.clip(p.dt_s * np.nan_to_num(dTdt), -p.max_dt_step_c, p.max_dt_step_c)
        T = np.clip(np.array(T, copy=True) + dT, p.temp_min_c, p.temp_max_c)

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
                  exposure_s=(nsteps * p.dt_s if densify else t90))


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


def shrinkage_analysis(res: Result, p: Params, h: float, xy_frac: float = 0.04,
                       layer_thickness_mm: float = 0.10,
                       target_final_height_mm: float | None = None) -> dict:
    """Anisotropic (Z-dominant) sintering shrinkage from the final density field.

    Local volume shrink S_V = rho_green / rho_final (mass conservation). Partition
    the log-volume anisotropically: lambda_xy = S_V**xy_frac (minor),
    lambda_z = S_V / lambda_xy**2 (dominant), so XY stays nearly rigid (powder-bed
    constraint) and most shrink collapses into Z. Then:
      * per-(x,y) column final height H = sum_z h*lambda_z  (Lagrangian Z compaction)
      * warpage = scatter of effective column shrink vs the uniform-shrink ideal
      * inverse layer count = green layers needed so the part compacts to target.
    Returns scalar metrics plus the H(x,y) height map and lambda_z field.
    """
    m = res.part
    rho_f = np.clip(np.asarray(res.rho_final, float), 1e-6, 1.0)
    S_V = np.clip(p.rho_rel / rho_f, 1e-6, 1.0)        # <=1 (densified -> shrinks)
    lam_xy = np.power(S_V, xy_frac)
    lam_z = np.clip(S_V / np.maximum(lam_xy ** 2, 1e-9), 1e-6, 1.0)
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
