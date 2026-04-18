#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter
try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional dependency guard
    Image = None
    ImageDraw = None

from shapes import make_shape, rotate

EPS0 = 8.8541878128e-12
R_GAS = 8.31446261815324


@dataclass
class SimState:
    x: np.ndarray
    y: np.ndarray
    part_mask: np.ndarray
    doped_mask: np.ndarray
    elec_hi: np.ndarray
    elec_lo: np.ndarray
    V: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    E_mag: np.ndarray
    Qrf: np.ndarray
    T: np.ndarray
    phi: np.ndarray
    rho_rel: np.ndarray
    x_cryst: np.ndarray | None = None
    part_id_mask: np.ndarray | None = None


def _copy_state(state: "SimState") -> "SimState":
    """Deep-copy all array fields of a SimState for snapshot use."""
    return SimState(
        x=state.x.copy(),
        y=state.y.copy(),
        part_mask=state.part_mask.copy(),
        doped_mask=state.doped_mask.copy(),
        elec_hi=state.elec_hi.copy(),
        elec_lo=state.elec_lo.copy(),
        V=state.V.copy(),
        Ex=state.Ex.copy(),
        Ey=state.Ey.copy(),
        E_mag=state.E_mag.copy(),
        Qrf=state.Qrf.copy(),
        T=state.T.copy(),
        phi=state.phi.copy(),
        rho_rel=state.rho_rel.copy(),
        x_cryst=(None if state.x_cryst is None else state.x_cryst.copy()),
        part_id_mask=(None if state.part_id_mask is None else state.part_id_mask.copy()),
    )


def _pick_snapshot_steps(n_steps: int, max_frames: int, min_frames: int = 12) -> list[int]:
    if n_steps <= 0:
        return []
    n_hi = min(max_frames, n_steps)
    n = max(2, min(n_steps, max(min_frames, n_hi)))
    idx = np.linspace(0, n_steps - 1, num=n, dtype=int)
    return sorted(set(int(i) for i in idx))


def _capture_animation_snapshot(
    *,
    step_idx: int,
    time_s: float,
    cum_angle_deg: float | None,
    E_mag: np.ndarray,
    T: np.ndarray,
    rho_rel: np.ndarray,
    part_mask: np.ndarray,
) -> dict:
    return {
        "step_idx": int(step_idx),
        "time_s": float(time_s),
        "cum_angle_deg": None if cum_angle_deg is None else float(cum_angle_deg),
        "E_mag": np.asarray(E_mag, dtype=float).copy(),
        "T": np.asarray(T, dtype=float).copy(),
        "rho_rel": np.asarray(rho_rel, dtype=float).copy(),
        "part_mask": np.asarray(part_mask, dtype=bool).copy(),
    }


@dataclass
class PhaseConfig:
    model: str
    solidus_c: float
    liquidus_c: float
    t_pc_c: float
    dt_pc_c: float
    smooth_shape: str
    tanh_beta: float
    dsc_onset_c: float | None = None
    dsc_peak_c: float | None = None
    dsc_end_c: float | None = None
    cp_smoothing_strategy: str = "linear"


@dataclass
class ThermalParams:
    dx: float
    dy: float
    dt: float
    max_dt_step_c: float
    temp_min_c: float
    temp_max_c: float
    phase_cfg: PhaseConfig
    latent_heat: float
    ambient_c: float
    convection_model: str
    h_const: float
    conv_plate_distance_m: float
    conv_chimney_height_m: float
    conv_pressure_pa: float
    conv_external_temp_k: float
    rho_powder: float
    k_powder: float
    cp_powder: float
    rho_solid: float
    rho_liquid: float
    k_solid: float
    k_liquid: float
    cp_solid: float
    cp_liquid: float
    dens_k0: float
    dens_ea: float
    dens_model: str
    dens_phi_exponent: float
    dens_rho_exponent: float
    dens_max_delta_per_step: float
    dens_surface_tension_n_per_m: float
    dens_particle_radius_m: float
    dens_eta_ref_pa_s: float
    dens_eta_ref_temp_k: float
    dens_eta_activation_j_per_mol: float
    dens_geom_factor: float
    dens_k0_ss: float
    dens_ea_ss: float
    dens_k0_liq: float
    dens_ea_liq: float
    dens_phi_threshold: float
    dens_phi_solid_exponent: float
    dens_phi_liq_exponent: float
    dens_liquid_rate_mode: str
    dA: float
    depth_correction_enabled: bool = False
    depth_correction_part_depth_m: float = 0.010
    depth_correction_chamber_depth_m: float = 0.016
    physics_model_family: str = "baseline"
    experimental_viscosity_model: str = "arrhenius"
    experimental_viscosity_ref_pa_s: float = 8.0e3
    experimental_viscosity_ref_temp_k: float = 458.15
    experimental_viscosity_activation_j_per_mol: float = 6.0e4
    experimental_wlf_c1: float = 8.86
    experimental_wlf_c2_k: float = 101.6
    experimental_phi_exponent: float = 1.0
    experimental_porosity_exponent: float = 1.0
    experimental_phi_threshold: float = 0.02
    crystallization_enabled: bool = False
    crystallization_model: str = "nakamura"
    crystallization_k0_per_s: float = 0.05
    crystallization_ea_j_per_mol: float = 4.2e4
    crystallization_exponent: float = 2.0
    crystallization_liquid_suppression: float = 0.25
    debug: bool = False


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {path}")
    return cfg


def _load_provenance_table(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid provenance table: {path}")
    params = data.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError(f"Invalid provenance table parameters block: {path}")
    return data


def _validate_experimental_provenance(
    provenance: dict,
    required_params: list[str],
) -> None:
    params = provenance.get("parameters", {})
    missing = [p for p in required_params if p not in params]
    if missing:
        raise ValueError(
            "Experimental model requires provenance entries for: "
            + ", ".join(missing)
        )
    for key in required_params:
        item = params.get(key, {})
        if not isinstance(item, dict):
            raise ValueError(f"Invalid provenance entry for {key}")
        for field in ("source", "page_eq", "confidence"):
            if field not in item:
                raise ValueError(f"Provenance entry {key} missing field '{field}'")


def _load_dsc_profile(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid DSC profile: {path}")
    for key in ("melt_onset_c", "melt_peak_c", "melt_end_c", "lat_heat_j_per_kg"):
        if key not in data:
            raise ValueError(f"DSC profile missing '{key}': {path}")
    onset = float(data["melt_onset_c"])
    peak = float(data["melt_peak_c"])
    end = float(data["melt_end_c"])
    if not (onset < peak < end):
        raise ValueError(f"DSC profile invalid interval (need onset < peak < end): {path}")
    return data


def oriented_rect_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    center: np.ndarray,
    t_hat: np.ndarray,
    n_hat: np.ndarray,
    length: float,
    thickness: float,
) -> np.ndarray:
    # Force scalar axis components explicitly. On Python 3.14 we observed
    # inconsistent vector-component reads from direct array indexing in this
    # hot path; scalar extraction keeps the rectangle placement deterministic.
    cx = float(center[0])
    cy = float(center[1])
    tx = float(t_hat[0])
    ty = float(t_hat[1])
    nx = float(n_hat[0])
    ny = float(n_hat[1])
    dx = xx - cx
    dy = yy - cy
    u = dx * tx + dy * ty
    v = dx * nx + dy * ny
    return (np.abs(u) <= 0.5 * length) & (np.abs(v) <= 0.5 * thickness)


def polygon_mask(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = MplPath(poly).contains_points(pts)
    return mask.reshape(xx.shape)


def _subpixel_fill_fraction(poly_or_fn, x: np.ndarray, y: np.ndarray,
                             n_sub: int = 8) -> np.ndarray:
    """
    Sub-pixel fill fraction for each grid cell, in [0.0, 1.0].

    Samples n_sub×n_sub evenly-spaced interior points per cell and counts what
    fraction lies inside the shape.  This creates a smooth, anti-aliased
    material boundary that eliminates staircase-corner E-field artefacts in the
    EQS solve for shapes with edges that are not grid-aligned.

    poly_or_fn : numpy polygon (N×2) OR a callable(xx, yy) → bool ndarray
    Returns float array of shape (ny, nx).
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    offsets = np.linspace(-0.5 + 0.5 / n_sub, 0.5 - 0.5 / n_sub, n_sub)
    total = np.zeros((len(y), len(x)), dtype=float)
    if callable(poly_or_fn):
        fn = poly_or_fn
        for dox in offsets:
            for doy in offsets:
                xx, yy = np.meshgrid(x + dox * dx, y + doy * dy, indexing="xy")
                total += fn(xx, yy).astype(float)
    else:
        path = MplPath(poly_or_fn)
        for dox in offsets:
            for doy in offsets:
                xx, yy = np.meshgrid(x + dox * dx, y + doy * dy, indexing="xy")
                pts = np.column_stack([xx.ravel(), yy.ravel()])
                inside = path.contains_points(pts).reshape(len(y), len(x))
                total += inside.astype(float)
    return total / float(n_sub * n_sub)


def build_electrodes(
    x: np.ndarray,
    y: np.ndarray,
    part_poly: np.ndarray,
    spacing: float,
    length: float,
    thickness: float,
    angle_deg: float,
    center_x: float,
    center_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_deg)
    t_hat = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    n_hat = np.array([-math.sin(theta), math.cos(theta)], dtype=float)

    p = part_poly @ n_hat
    extent_n = float(np.max(p) - np.min(p))
    gap = extent_n + 2.0 * spacing
    c = np.array([center_x, center_y], dtype=float)
    c_hi = c + 0.5 * gap * n_hat
    c_lo = c - 0.5 * gap * n_hat

    def _rect_poly(center: np.ndarray) -> np.ndarray:
        hL = 0.5 * float(length)
        hT = 0.5 * float(thickness)
        return np.array(
            [
                center - hL * t_hat - hT * n_hat,
                center + hL * t_hat - hT * n_hat,
                center + hL * t_hat + hT * n_hat,
                center - hL * t_hat + hT * n_hat,
            ],
            dtype=float,
        )

    hi = polygon_mask(_rect_poly(c_hi), x, y)
    lo = polygon_mask(_rect_poly(c_lo), x, y)
    if not np.any(hi) or not np.any(lo):
        raise ValueError("Electrodes outside domain. Increase chamber size or reduce spacing/length.")
    return hi, lo


def build_boundary_electrodes(
    x: np.ndarray,
    y: np.ndarray,
    hi_boundary: str,
    lo_boundary: str,
    hi_span_min: float | None = None,
    hi_span_max: float | None = None,
    lo_span_min: float | None = None,
    lo_span_max: float | None = None,
    boundary_cells: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    ny = y.size
    nx = x.size
    hi = np.zeros((ny, nx), dtype=bool)
    lo = np.zeros((ny, nx), dtype=bool)

    def _span_mask(vals: np.ndarray, vmin: float | None, vmax: float | None) -> np.ndarray:
        m = np.ones_like(vals, dtype=bool)
        if vmin is not None:
            m &= vals >= float(vmin)
        if vmax is not None:
            m &= vals <= float(vmax)
        return m

    def _apply(mask: np.ndarray, boundary: str, vmin: float | None, vmax: float | None) -> None:
        b = str(boundary).strip().lower()
        n = max(int(boundary_cells), 1)
        if b in {"top", "bottom"}:
            sm = _span_mask(x, vmin, vmax)
            rows = slice(ny - n, ny) if b == "top" else slice(0, n)
            mask[rows, sm] = True
        elif b in {"left", "right"}:
            sm = _span_mask(y, vmin, vmax)
            cols = slice(nx - n, nx) if b == "right" else slice(0, n)
            mask[sm, cols] = True
        else:
            raise ValueError(f"Unsupported boundary electrode label: {boundary}")

    _apply(hi, hi_boundary, hi_span_min, hi_span_max)
    _apply(lo, lo_boundary, lo_span_min, lo_span_max)
    if not np.any(hi) or not np.any(lo):
        raise ValueError("Boundary electrode selection produced empty mask.")
    return hi, lo


_TURNTABLE_ANGLE_MAP: dict[int, tuple[str, str]] = {
    0:   ("top",    "bottom"),   # E-field vertical   (Y-axis)
    90:  ("left",   "right"),    # E-field horizontal (X-axis)
    180: ("bottom", "top"),      # E-field vertical   (inverted)
    270: ("right",  "left"),     # E-field horizontal (inverted)
}


def _electrode_for_angle(
    angle_deg: int,
    x: np.ndarray,
    y: np.ndarray,
    ep_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (elec_hi, elec_lo) masks for a given 90°-multiple rotation angle.

    Rotating the part by *angle_deg* relative to fixed electrodes is equivalent to
    keeping the part geometry fixed and changing which chamber walls carry the voltage
    boundary condition. This covers the 90°-increment turntable use-case with zero
    geometry re-rasterization.

    # TODO (future continuous turntable):
    # 1. Extract geometry builder into _build_part_mask(shape_cfg, angle_deg, x, y)
    # 2. At each rotation event: call _build_part_mask(new_angle), update state.part_mask
    # 3. Map arbitrary angle θ to diagonal electrode projections
    # 4. Use fine increments (e.g. 5° every N steps) for microwave-style continuous rotation
    """
    norm = int(angle_deg) % 360
    if norm not in _TURNTABLE_ANGLE_MAP:
        raise ValueError(
            f"Turntable angle must be a multiple of 90° (0/90/180/270). Got {angle_deg}°."
        )
    hi_b, lo_b = _TURNTABLE_ANGLE_MAP[norm]
    return build_boundary_electrodes(
        x, y, hi_b, lo_b,
        hi_span_min=ep_cfg.get("hi_span_min"),
        hi_span_max=ep_cfg.get("hi_span_max"),
        lo_span_min=ep_cfg.get("lo_span_min"),
        lo_span_max=ep_cfg.get("lo_span_max"),
        boundary_cells=int(ep_cfg.get("boundary_cells", 1)),
    )


def solve_eqs_complex(
    gamma: np.ndarray,
    elec_hi: np.ndarray,
    elec_lo: np.ndarray,
    v_hi: float,
    v_lo: float,
    dx: float,
    dy: float,
    steps: int,
    tol: float,
    relaxation: float = 0.45,
) -> np.ndarray:
    # Build a sparse linear system for:
    #   div(gamma grad(V)) = 0
    # with Dirichlet electrodes and insulating (zero-normal-flux) outer boundaries.
    # steps/tol/relaxation are kept in signature for compatibility with prior interface.
    ny, nx = gamma.shape
    n = ny * nx
    dirichlet = elec_hi | elec_lo

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    b = np.zeros(n, dtype=np.complex128)

    def lin(i: int, j: int) -> int:
        return i * nx + j

    for i in range(ny):
        for j in range(nx):
            k = lin(i, j)
            if elec_hi[i, j]:
                rows.append(k)
                cols.append(k)
                vals.append(1.0 + 0.0j)
                b[k] = complex(v_hi)
                continue
            if elec_lo[i, j]:
                rows.append(k)
                cols.append(k)
                vals.append(1.0 + 0.0j)
                b[k] = complex(v_lo)
                continue

            neighbors: list[tuple[int, int]] = []
            if i > 0:
                neighbors.append((i - 1, j))
            if i < ny - 1:
                neighbors.append((i + 1, j))
            if j > 0:
                neighbors.append((i, j - 1))
            if j < nx - 1:
                neighbors.append((i, j + 1))

            g0 = gamma[i, j]
            inv_dx2 = 1.0 / max(dx * dx, 1e-24)
            inv_dy2 = 1.0 / max(dy * dy, 1e-24)

            diag = 0.0 + 0.0j
            rhs = 0.0 + 0.0j
            for ni, nj in neighbors:
                g1 = complex(gamma[ni, nj])
                # Use harmonic face averaging for discontinuous media. Arithmetic
                # averaging can overestimate interfacial flux across high-contrast
                # regions and exaggerate corner localization.
                den = g0 + g1
                if abs(den) > 1e-30:
                    gf = complex(2.0 * g0 * g1 / den)
                else:
                    gf = complex(0.5 * (g0 + g1))
                if not np.isfinite(gf.real) or not np.isfinite(gf.imag):
                    gf = 1e-9 + 0.0j
                w = gf * (inv_dy2 if ni != i else inv_dx2)
                diag += w
                if dirichlet[ni, nj]:
                    rhs += w * complex(v_hi if elec_hi[ni, nj] else v_lo)
                else:
                    rows.append(k)
                    cols.append(lin(ni, nj))
                    vals.append(-w)

            rows.append(k)
            cols.append(k)
            vals.append(diag)
            b[k] = rhs

    A = sparse.csr_matrix((np.asarray(vals, dtype=np.complex128), (rows, cols)), shape=(n, n))
    # Tiny diagonal regularization improves conditioning in nearly-insulating regions.
    A = A + sparse.eye(n, format="csr", dtype=np.complex128) * (1e-18 + 0.0j)
    v_flat = spla.spsolve(A, b)
    if np.any(~np.isfinite(v_flat)):
        raise RuntimeError("EQS solve produced non-finite values.")
    return v_flat.reshape((ny, nx))


def compute_electric_fields(V: np.ndarray, x: np.ndarray, y: np.ndarray, gamma: np.ndarray, power_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Defensive copy: avoid any in-place buffer reuse side effects on caller-owned arrays.
    g = np.array(gamma, dtype=np.complex128, copy=True)
    dVdy, dVdx = np.gradient(V, y, x, edge_order=1)
    Ex = -dVdx
    Ey = -dVdy
    e2_complex = Ex * np.conj(Ex) + Ey * np.conj(Ey)
    Qrf = 0.5 * power_factor * np.real(g * e2_complex)
    Qrf = np.nan_to_num(Qrf, nan=0.0, posinf=0.0, neginf=0.0)
    # Resistive heating is non-negative; small negatives can appear from discretization noise.
    Qrf = np.maximum(Qrf, 0.0)
    e2 = np.real(e2_complex)
    e2 = np.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0)
    E_mag = np.sqrt(np.maximum(e2, 0.0))
    return Ex, Ey, E_mag, Qrf


def phase_fraction(T: np.ndarray, phase_cfg: PhaseConfig) -> tuple[np.ndarray, np.ndarray]:
    # Defensive copy: NumPy may reuse buffers in chained ufunc expressions in some
    # environments; keep phase evaluation side-effect free on caller arrays.
    T_eval = np.array(T, dtype=float, copy=True)
    model = str(phase_cfg.model).strip().lower()

    if model in {"apparent_heat_capacity_dsc", "dsc_apparent_cp"}:
        onset = float(
            phase_cfg.dsc_onset_c
            if phase_cfg.dsc_onset_c is not None
            else (phase_cfg.t_pc_c - 0.5 * max(phase_cfg.dt_pc_c, 1e-9))
        )
        end = float(
            phase_cfg.dsc_end_c
            if phase_cfg.dsc_end_c is not None
            else (phase_cfg.t_pc_c + 0.5 * max(phase_cfg.dt_pc_c, 1e-9))
        )
        peak = float(
            phase_cfg.dsc_peak_c
            if phase_cfg.dsc_peak_c is not None
            else 0.5 * (onset + end)
        )
        if end <= onset:
            end = onset + 1e-6
        width = max(end - onset, 1e-9)
        smooth = str(phase_cfg.cp_smoothing_strategy or "linear").strip().lower()

        if smooth == "tanh":
            w = max(0.20 * width, 1e-9)
            z = (T_eval - peak) / w
            phi = 0.5 * (1.0 + np.tanh(z))
            dphi_dT = 0.5 * (1.0 / np.cosh(z) ** 2) / w
            phi = np.where(T_eval <= onset, 0.0, np.where(T_eval >= end, 1.0, phi))
            dphi_dT = np.where((T_eval >= onset) & (T_eval <= end), dphi_dT, 0.0)
        else:
            # Default DSC apparent-cp style melt interval mapping.
            phi = np.clip((T_eval - onset) / width, 0.0, 1.0)
            dphi_dT = np.where((T_eval >= onset) & (T_eval <= end), 1.0 / width, 0.0)
        return phi, dphi_dT

    if model in {"comsol_heaviside", "heaviside"}:
        # COMSOL phase model uses a smooth Heaviside centered at T_pc with width dT_pc.
        t_pc = float(phase_cfg.t_pc_c)
        dT_pc = max(float(phase_cfg.dt_pc_c), 1e-9)
        smooth = str(phase_cfg.smooth_shape).strip().lower()
        arg = (T_eval - t_pc) / dT_pc

        if smooth == "tanh":
            beta = max(float(phase_cfg.tanh_beta), 1e-6)
            z = beta * arg
            phi = 0.5 * (1.0 + np.tanh(z))
            dphi_dT = 0.5 * beta * (1.0 / np.cosh(z) ** 2) / dT_pc
        else:
            # Piecewise-linear approximation around T_pc over interval dT_pc.
            phi = np.clip(arg + 0.5, 0.0, 1.0)
            dphi_dT = np.where(np.abs(arg) <= 0.5, 1.0 / dT_pc, 0.0)
        return phi, dphi_dT

    # Default linear window between solidus and liquidus.
    T_solidus = float(phase_cfg.solidus_c)
    T_liquidus = float(phase_cfg.liquidus_c)
    dT = max(T_liquidus - T_solidus, 1e-9)
    raw = (T_eval - T_solidus) / dT
    phi = np.clip(raw, 0.0, 1.0)
    dphi_dT = np.where((T_eval >= T_solidus) & (T_eval <= T_liquidus), 1.0 / dT, 0.0)
    return phi, dphi_dT


def diffusion_divergence(T: np.ndarray, k: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ny, nx = T.shape
    fx = np.zeros((ny, nx + 1), dtype=float)
    fy = np.zeros((ny + 1, nx), dtype=float)

    kx = 0.5 * (k[:, 1:] + k[:, :-1])
    ky = 0.5 * (k[1:, :] + k[:-1, :])

    fx[:, 1:nx] = -kx * (T[:, 1:] - T[:, :-1]) / dx
    fy[1:ny, :] = -ky * (T[1:, :] - T[:-1, :]) / dy

    # Boundary fluxes are zero here; convection handled as sink term.
    # q = -k grad(T). Heat equation needs -div(q) = div(k grad(T)).
    # The raw divergence of q would be anti-diffusive in the update.
    div_k_gradT = -((fx[:, 1:] - fx[:, :-1]) / dx + (fy[1:, :] - fy[:-1, :]) / dy)
    return div_k_gradT


def boundary_exposure_coeff(shape: tuple[int, int], dx: float, dy: float, boundaries: list[str] | None = None) -> np.ndarray:
    ny, nx = shape
    expo = np.zeros(shape, dtype=float)
    if boundaries is None:
        boundaries = ["left", "right", "bottom", "top"]
    bset = {str(b).strip().lower() for b in boundaries}
    if "left" in bset:
        expo[:, 0] += 1.0 / dx
    if "right" in bset:
        expo[:, -1] += 1.0 / dx
    if "bottom" in bset:
        expo[0, :] += 1.0 / dy
    if "top" in bset:
        expo[-1, :] += 1.0 / dy
    return expo


def air_mu_sutherland(Tk: np.ndarray | float) -> np.ndarray:
    T = np.asarray(Tk, dtype=float)
    T0 = 273.15
    mu0 = 1.716e-5
    S = 110.4
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)


def air_k_simple(Tk: np.ndarray | float) -> np.ndarray:
    # Simple smooth fit around ambient-to-moderate temperatures.
    T = np.asarray(Tk, dtype=float)
    return 0.0241 * (T / 273.15) ** 0.9


def chimney_natural_convection_h(
    T_surface_c: np.ndarray,
    T_ext_k: float,
    plate_distance_m: float,
    chimney_height_m: float,
    pressure_pa: float,
) -> np.ndarray:
    # Mirrors the COMSOL report expression form for "Internal natural convection,
    # narrow chimney, parallel plates":
    #   h = k * Ra / (24 * H)
    # with Ra based on air properties at film temperature.
    g = 9.81
    R_air = 287.05
    cp = 1006.0

    Ts = np.asarray(T_surface_c, dtype=float) + 273.15
    Text = float(T_ext_k)
    Tfilm = 0.5 * (Ts + Text)
    Tfilm = np.clip(Tfilm, 200.0, 1200.0)

    rho = pressure_pa / (R_air * Tfilm)
    k_air = air_k_simple(Tfilm)
    mu = air_mu_sutherland(Tfilm)
    drho_dT_abs = rho / np.maximum(Tfilm, 1e-9)

    L = max(float(plate_distance_m), 1e-9)
    H = max(float(chimney_height_m), 1e-9)
    dT = np.abs(Ts - Text)
    Ra = (L**3) * g * drho_dT_abs * rho * cp * dT / np.maximum(k_air * mu, 1e-18)
    h = k_air * Ra / (24.0 * H)
    return np.clip(h, 0.0, 1e4)


def _shape_kwargs_from_part(part: dict) -> dict:
    out: dict = {"n_circle_pts": int(part.get("n_circle_pts", 220))}
    for _k in (
        "svg_path", "image_path", "thickness", "inner_radius_ratio",
        "n_points", "corner_radius", "curve_pts", "path_index",
        "threshold", "polygon_points",
    ):
        if _k in part:
            out[_k] = part[_k]
    return out


def _single_part_mask_and_fill(
    x: np.ndarray,
    y: np.ndarray,
    part: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = str(part["shape"])
    width = float(part["width"])
    height = float(part.get("height", width))
    rot_deg = float(part.get("rotation_deg", 0.0))
    cx = float(part.get("center_x", 0.0))
    cy = float(part.get("center_y", 0.0))
    poly = make_shape(shape, width, height, **_shape_kwargs_from_part(part))
    poly = rotate(poly, math.radians(rot_deg))
    poly[:, 0] += cx
    poly[:, 1] += cy

    _s = shape.strip().lower()
    if _s in {"circle", "disk", "cylinder", "sphere"}:
        _xx, _yy = np.meshgrid(x, y, indexing="xy")
        r = width / 2.0
        p_mask = ((_xx - cx) ** 2 + (_yy - cy) ** 2) <= r ** 2
    elif _s in {"ellipse", "oval"}:
        _xx, _yy = np.meshgrid(x, y, indexing="xy")
        a = width / 2.0
        b = height / 2.0
        if rot_deg == 0.0:
            p_mask = ((_xx - cx) ** 2 / a ** 2 + (_yy - cy) ** 2 / b ** 2) <= 1.0
        else:
            _cos_r = math.cos(-math.radians(rot_deg))
            _sin_r = math.sin(-math.radians(rot_deg))
            _dx = _xx - cx
            _dy = _yy - cy
            _u = _dx * _cos_r - _dy * _sin_r
            _v = _dx * _sin_r + _dy * _cos_r
            p_mask = (_u ** 2 / a ** 2 + _v ** 2 / b ** 2) <= 1.0
    else:
        p_mask = polygon_mask(poly, x, y)

    _N_SUB = 8
    if _s in {"circle", "disk", "cylinder", "sphere"}:
        r_fill = width / 2.0
        fill_frac = _subpixel_fill_fraction(
            lambda xx, yy, _cx=cx, _cy=cy, _r=r_fill:
                (xx - _cx) ** 2 + (yy - _cy) ** 2 <= _r ** 2,
            x, y, n_sub=_N_SUB,
        )
    elif _s in {"ellipse", "oval"}:
        a_fill = width / 2.0
        b_fill = height / 2.0
        if rot_deg == 0.0:
            fill_frac = _subpixel_fill_fraction(
                lambda xx, yy, _cx=cx, _cy=cy, _a=a_fill, _b=b_fill:
                    (xx - _cx) ** 2 / _a ** 2 + (yy - _cy) ** 2 / _b ** 2 <= 1.0,
                x, y, n_sub=_N_SUB,
            )
        else:
            _cos_f = math.cos(-math.radians(rot_deg))
            _sin_f = math.sin(-math.radians(rot_deg))
            fill_frac = _subpixel_fill_fraction(
                lambda xx, yy, _cx=cx, _cy=cy, _a=a_fill, _b=b_fill, _c=_cos_f, _s=_sin_f: (
                    ((xx - _cx) * _c - (yy - _cy) * _s) ** 2 / _a ** 2
                    + ((xx - _cx) * _s + (yy - _cy) * _c) ** 2 / _b ** 2 <= 1.0
                ),
                x, y, n_sub=_N_SUB,
            )
    else:
        fill_frac = _subpixel_fill_fraction(poly, x, y, n_sub=_N_SUB)
    return poly, p_mask, fill_frac


def _apply_shell_to_mask(
    x: np.ndarray,
    y: np.ndarray,
    part: dict,
    outer_mask: np.ndarray,
    outer_fill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shell_cfg = part.get("shell", {}) if isinstance(part.get("shell", {}), dict) else {}
    if not shell_cfg:
        return outer_mask, outer_fill
    if not bool(shell_cfg.get("enabled", False)):
        return outer_mask, outer_fill
    wall_mm = float(shell_cfg.get("wall_thickness_mm", 0.0))
    if wall_mm <= 0.0:
        return outer_mask, outer_fill
    method = str(shell_cfg.get("method", "offset_inward")).strip().lower()
    if method != "offset_inward":
        return outer_mask, outer_fill

    shape = str(part.get("shape", "")).strip().lower()
    width = float(part.get("width", 0.0))
    height = float(part.get("height", width))
    inset = 2.0 * wall_mm * 1e-3
    if inset <= 0.0 or inset >= min(width, height):
        return outer_mask, outer_fill

    inner_part = dict(part)
    inner_part["width"] = width - inset
    inner_part["height"] = height - inset
    if shape not in {"circle", "disk", "cylinder", "sphere", "square", "rect", "rectangle"}:
        print(f"  [shell] shape '{shape}' not supported in v1; using solid mask")
        return outer_mask, outer_fill

    _, inner_mask, inner_fill = _single_part_mask_and_fill(x, y, inner_part)
    shell_mask = outer_mask & (~inner_mask)
    shell_fill = np.clip(outer_fill - inner_fill, 0.0, 1.0)
    return shell_mask, shell_fill


def _antennae_cfg_from_root(cfg: dict) -> dict:
    ant = cfg.get("antennae", {}) if isinstance(cfg.get("antennae", {}), dict) else {}
    spk = cfg.get("spike", {}) if isinstance(cfg.get("spike", {}), dict) else {}
    out = dict(ant)
    for k, v in spk.items():
        out.setdefault(str(k), v)
    return out


def _antennae_enabled(cfg: dict) -> bool:
    ant = _antennae_cfg_from_root(cfg)
    return bool(ant.get("enabled", False))


def _apply_antennae_to_mask(
    x: np.ndarray,
    y: np.ndarray,
    part: dict,
    base_mask: np.ndarray,
    base_fill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inst_raw = part.get("antennae_instances", [])
    if not isinstance(inst_raw, list) or not inst_raw:
        return base_mask, base_fill
    out_mask = np.array(base_mask, copy=True, dtype=bool)
    out_fill = np.array(base_fill, copy=True, dtype=float)
    for inst in inst_raw:
        if not isinstance(inst, dict):
            continue
        try:
            cx = float(inst.get("center_x", 0.0))
            cy = float(inst.get("center_y", 0.0))
            size_mm = float(inst.get("size_mm", 0.0))
        except Exception:
            continue
        r = max(size_mm * 1e-3, 0.0)
        if r <= 0.0:
            continue
        xx, yy = np.meshgrid(x, y, indexing="xy")
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        if not np.any(m):
            continue
        f = _subpixel_fill_fraction(
            lambda xq, yq, _cx=cx, _cy=cy, _r=r: (xq - _cx) ** 2 + (yq - _cy) ** 2 <= _r ** 2,
            x,
            y,
            n_sub=8,
        )
        out_mask |= m
        out_fill = np.maximum(out_fill, f)
    return out_mask, out_fill


def _resolve_antennae_instances(cfg: dict) -> dict:
    ant = _antennae_cfg_from_root(cfg)
    if not bool(ant.get("enabled", False)):
        return {"enabled": False, "instances": []}
    geom = cfg.get("geometry", {}) if isinstance(cfg.get("geometry", {}), dict) else {}
    if bool(geom.get("_antennae_resolved", False)):
        rows: list[dict] = []
        parts = _parts_from_geometry(geom) if isinstance(geom, dict) else []
        for pid, p in enumerate(parts, start=1):
            insts = p.get("antennae_instances", []) if isinstance(p, dict) else []
            if not isinstance(insts, list):
                continue
            for inst in insts:
                if isinstance(inst, dict):
                    rr = dict(inst)
                    rr["part_id"] = int(pid)
                    rows.append(rr)
        return {"enabled": True, "instances": rows}

    base = copy.deepcopy(cfg)
    base_geom = base.setdefault("geometry", {})
    base_geom.pop("_antennae_resolved", None)
    for p in _parts_from_geometry(base_geom):
        if isinstance(p, dict):
            p.pop("antennae_instances", None)

    x, y, _, _, doped_mask, elec_hi, elec_lo, fill_frac, part_id_mask, _ = make_domain(base)
    dx = float(base_geom["chamber_x"]) / max(int(base_geom["grid_nx"]) - 1, 1)
    dy = float(base_geom["chamber_y"]) / max(int(base_geom["grid_ny"]) - 1, 1)
    dA = dx * dy
    elec = base["electric"]
    mats = base["materials"]
    sigma_v = max(float(mats["virgin"]["sigma_s_per_m"]), 1e-8)
    eps_v = float(mats["virgin"]["eps_r"])
    sigma_d0 = float(mats["doped"]["sigma_s_per_m"])
    eps_d = float(mats["doped"]["eps_r"])
    sigma = sigma_v + fill_frac * (sigma_d0 - sigma_v)
    eps_r = eps_v + fill_frac * (eps_d - eps_v)
    omega = 2.0 * math.pi * float(elec["frequency_hz"])
    v0 = float(elec["voltage_v"])
    mode = str(elec.get("voltage_mode", "centered")).strip().lower()
    if mode in {"grounded", "single_ended"}:
        v_hi, v_lo = v0, 0.0
    else:
        v_hi, v_lo = 0.5 * v0, -0.5 * v0
    _, _, _, _, _, qrf = solve_electric_state(
        sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo, dx, dy, elec, x, y,
        float(elec.get("power_factor", 1.0)),
        float(elec.get("max_qrf_w_per_m3", 1.0e11)),
    )
    if bool(elec.get("zero_qrf_outside_doped", True)):
        qrf = np.where(doped_mask, qrf, 0.0)

    size_mode = str(ant.get("size_mode", "global")).strip().lower()
    global_size_mm = float(ant.get("global_size_mm", 1.0))
    auto = ant.get("auto_size", {}) if isinstance(ant.get("auto_size", {}), dict) else {}
    auto_min_mm = float(auto.get("min_mm", 0.5))
    auto_max_mm = float(auto.get("max_mm", 2.0))
    q_lo_pct = float(auto.get("qrf_percentile_low", 10.0))
    q_hi_pct = float(auto.get("qrf_percentile_high", 40.0))
    plc = ant.get("placement", {}) if isinstance(ant.get("placement", {}), dict) else {}
    max_n = int(plc.get("max_antennae_per_part", 4))
    min_spacing = float(plc.get("min_anchor_spacing_mm", 2.0)) * 1e-3
    edge_margin = float(plc.get("edge_margin_mm", 1.0)) * 1e-3

    yy, xx = np.meshgrid(y, x, indexing="ij")
    parts = _parts_from_geometry(cfg["geometry"])
    signed_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    selected_rows: list[dict] = []

    for pid in sorted(int(v) for v in np.unique(part_id_mask) if int(v) > 0):
        m = part_id_mask == pid
        if not np.any(m):
            continue
        edge = m & (~binary_erosion(m))
        if not np.any(edge):
            continue
        q_edge = qrf[edge]
        if q_edge.size == 0:
            continue
        q_lo = float(np.percentile(q_edge, q_lo_pct))
        q_hi = float(np.percentile(q_edge, q_hi_pct))
        denom = max(q_hi - q_lo, 1e-12)
        idxs = np.argwhere(edge)
        idxs = sorted(idxs, key=lambda ij: float(qrf[int(ij[0]), int(ij[1])]))
        part_cfg = parts[pid - 1] if pid - 1 < len(parts) else {}
        cx0 = float(part_cfg.get("center_x", 0.0)) if isinstance(part_cfg, dict) else 0.0
        cy0 = float(part_cfg.get("center_y", 0.0)) if isinstance(part_cfg, dict) else 0.0
        signed = distance_transform_edt(m) - distance_transform_edt(~m)
        gy, gx = np.gradient(signed)
        signed_cache[pid] = (signed, gx, gy)
        picked_xy: list[tuple[float, float]] = []
        for iy, ix in idxs:
            qv = float(qrf[int(iy), int(ix)])
            if qv > q_hi:
                continue
            px = float(x[int(ix)])
            py = float(y[int(iy)])
            if abs(px) > (0.5 * float(base_geom["chamber_x"]) - edge_margin):
                continue
            if abs(py) > (0.5 * float(base_geom["chamber_y"]) - edge_margin):
                continue
            if any(math.hypot(px - sx, py - sy) < min_spacing for sx, sy in picked_xy):
                continue
            deficit = float(np.clip((q_hi - qv) / denom, 0.0, 1.0))
            if size_mode == "auto":
                size_mm = float(np.clip(auto_min_mm + deficit * (auto_max_mm - auto_min_mm), auto_min_mm, auto_max_mm))
            else:
                size_mm = float(max(global_size_mm, 0.05))
            nxv = -float(gx[int(iy), int(ix)])
            nyv = -float(gy[int(iy), int(ix)])
            nn = math.hypot(nxv, nyv)
            if nn < 1e-12:
                nxv = px - cx0
                nyv = py - cy0
                nn = math.hypot(nxv, nyv)
            if nn < 1e-12:
                nxv, nyv, nn = 1.0, 0.0, 1.0
            nxv /= nn
            nyv /= nn
            offset = 0.55 * size_mm * 1e-3
            acx = float(px + nxv * offset)
            acy = float(py + nyv * offset)
            picked_xy.append((px, py))
            selected_rows.append({
                "part_id": int(pid),
                "anchor_x": px,
                "anchor_y": py,
                "center_x": acx,
                "center_y": acy,
                "size_mm": float(size_mm),
                "qrf": qv,
                "qrf_deficit": deficit,
                "size_mode": size_mode,
                "power_w_per_m": float(np.sum(qrf[m]) * dA),
            })
            if len(picked_xy) >= max_n:
                break

    for pid, p in enumerate(parts, start=1):
        if not isinstance(p, dict):
            continue
        rows = [r for r in selected_rows if int(r.get("part_id", -1)) == pid]
        p["antennae_instances"] = [
            {
                "center_x": float(r["center_x"]),
                "center_y": float(r["center_y"]),
                "size_mm": float(r["size_mm"]),
                "anchor_x": float(r["anchor_x"]),
                "anchor_y": float(r["anchor_y"]),
                "qrf_deficit": float(r["qrf_deficit"]),
            }
            for r in rows
        ]
    cfg["geometry"]["_antennae_resolved"] = True
    cfg["geometry"]["_antennae_summary"] = {"instances": selected_rows}
    return {"enabled": True, "instances": selected_rows}


def _parts_from_geometry(geom: dict) -> list[dict]:
    parts_raw = geom.get("parts", None)
    if isinstance(parts_raw, list) and parts_raw:
        return [p for p in parts_raw if isinstance(p, dict)]
    part = geom.get("part", {})
    if not isinstance(part, dict):
        raise ValueError("geometry.part must be a mapping")
    return [part]


def make_domain(
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    geom = cfg["geometry"]
    nx = int(geom["grid_nx"])
    ny = int(geom["grid_ny"])
    wx = float(geom["chamber_x"])
    wy = float(geom["chamber_y"])

    x = np.linspace(-0.5 * wx, 0.5 * wx, nx)
    y = np.linspace(-0.5 * wy, 0.5 * wy, ny)

    parts = _parts_from_geometry(geom)
    if not parts:
        raise ValueError("No geometry part definition found")
    boolean_cfg = geom.get("boolean", {}) if isinstance(geom.get("boolean", {}), dict) else {}
    boolean_mode = str(boolean_cfg.get("mode", "union")).strip().lower()
    if boolean_mode != "union":
        raise ValueError("Only geometry.boolean.mode='union' is supported in v1")

    global_shell = geom.get("shell", {}) if isinstance(geom.get("shell", {}), dict) else {}
    p_mask = np.zeros((ny, nx), dtype=bool)
    d_mask = np.zeros((ny, nx), dtype=bool)
    fill_frac = np.zeros((ny, nx), dtype=float)
    part_id_mask = np.zeros((ny, nx), dtype=np.int32)
    part_polys: list[np.ndarray] = []
    for idx, p in enumerate(parts, start=1):
        part_cfg = dict(p)
        if global_shell and "shell" not in part_cfg:
            part_cfg["shell"] = dict(global_shell)
        poly_i, mask_i, fill_i = _single_part_mask_and_fill(x, y, part_cfg)
        mask_i, fill_i = _apply_shell_to_mask(x, y, part_cfg, mask_i, fill_i)
        mask_i, fill_i = _apply_antennae_to_mask(x, y, part_cfg, mask_i, fill_i)
        if not np.any(mask_i):
            continue
        part_polys.append(poly_i)
        p_mask |= mask_i
        d_mask |= mask_i
        fill_frac = np.maximum(fill_frac, fill_i)
        part_id_mask[mask_i] = idx

    elec = cfg["electrodes"]
    mode = str(elec.get("mode", "plates")).strip().lower()
    part_poly_ref = part_polys[0] if part_polys else np.array([[-1e-3, -1e-3], [1e-3, -1e-3], [1e-3, 1e-3], [-1e-3, 1e-3]])
    if mode == "boundary":
        hi, lo = build_boundary_electrodes(
            x=x,
            y=y,
            hi_boundary=str(elec.get("hi_boundary", "top")),
            lo_boundary=str(elec.get("lo_boundary", "bottom")),
            hi_span_min=elec.get("hi_span_min", None),
            hi_span_max=elec.get("hi_span_max", None),
            lo_span_min=elec.get("lo_span_min", None),
            lo_span_max=elec.get("lo_span_max", None),
            boundary_cells=int(elec.get("boundary_cells", 1)),
        )
    else:
        hi, lo = build_electrodes(
            x=x,
            y=y,
            part_poly=part_poly_ref,
            spacing=float(elec["spacing"]),
            length=float(elec["length"]),
            thickness=float(elec["thickness"]),
            angle_deg=float(elec.get("angle_deg", 0.0)),
            center_x=float(elec.get("center_x", 0.0)),
            center_y=float(elec.get("center_y", 0.0)),
        )

    return x, y, part_poly_ref, p_mask, d_mask, hi, lo, fill_frac, part_id_mask, part_polys


def _build_rotated_part_mask(
    geom_cfg: dict,
    extra_rot_deg: float,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-rasterize the part mask at (original_rotation_deg + extra_rot_deg).

    Keeps electrodes fixed (physically correct turntable model) and only
    updates the part/doped masks and sub-pixel fill fraction.

    Returns
    -------
    part_mask  : bool (ny, nx)
    doped_mask : bool (ny, nx)  — same as part_mask for v1 whole-part doping
    fill_frac  : float (ny, nx) — sub-pixel fill fraction for blended EQS boundary
    """
    import copy as _copy

    def _rot_xy(cx: float, cy: float, deg: float) -> tuple[float, float]:
        rr = math.radians(float(deg))
        c = math.cos(rr)
        s = math.sin(rr)
        return float(c * cx - s * cy), float(s * cx + c * cy)

    cfg_rot = _copy.deepcopy(geom_cfg)
    parts = cfg_rot.get("parts", None)
    if isinstance(parts, list) and parts:
        p_mask_r = np.zeros((len(y), len(x)), dtype=bool)
        d_mask_r = np.zeros_like(p_mask_r)
        fill_frac_r = np.zeros((len(y), len(x)), dtype=float)
        part_id_mask = np.zeros_like(p_mask_r, dtype=np.int32)
        global_shell = cfg_rot.get("shell", {}) if isinstance(cfg_rot.get("shell", {}), dict) else {}
        for idx, part in enumerate(parts, start=1):
            if not isinstance(part, dict):
                continue
            p_cfg = dict(part)
            if global_shell and "shell" not in p_cfg:
                p_cfg["shell"] = dict(global_shell)
            cx0 = float(p_cfg.get("center_x", 0.0))
            cy0 = float(p_cfg.get("center_y", 0.0))
            cxr, cyr = _rot_xy(cx0, cy0, extra_rot_deg)
            p_cfg["center_x"] = cxr
            p_cfg["center_y"] = cyr
            p_cfg["rotation_deg"] = float(p_cfg.get("rotation_deg", 0.0)) + float(extra_rot_deg)
            insts = p_cfg.get("antennae_instances", [])
            if isinstance(insts, list) and insts:
                new_insts = []
                for inst in insts:
                    if not isinstance(inst, dict):
                        continue
                    icx = float(inst.get("center_x", 0.0))
                    icy = float(inst.get("center_y", 0.0))
                    icxr, icyr = _rot_xy(icx, icy, extra_rot_deg)
                    ii = dict(inst)
                    ii["center_x"] = float(icxr)
                    ii["center_y"] = float(icyr)
                    new_insts.append(ii)
                p_cfg["antennae_instances"] = new_insts
            _, pm_i, ff_i = _single_part_mask_and_fill(x, y, p_cfg)
            pm_i, ff_i = _apply_shell_to_mask(x, y, p_cfg, pm_i, ff_i)
            pm_i, ff_i = _apply_antennae_to_mask(x, y, p_cfg, pm_i, ff_i)
            if not np.any(pm_i):
                continue
            p_mask_r |= pm_i
            d_mask_r |= pm_i
            fill_frac_r = np.maximum(fill_frac_r, ff_i)
            part_id_mask[pm_i] = idx
        return p_mask_r, d_mask_r, fill_frac_r, part_id_mask

    part_cfg = cfg_rot.get("part", {}) if isinstance(cfg_rot.get("part", {}), dict) else {}
    cx0 = float(part_cfg.get("center_x", 0.0))
    cy0 = float(part_cfg.get("center_y", 0.0))
    cxr, cyr = _rot_xy(cx0, cy0, extra_rot_deg)
    part_cfg["center_x"] = cxr
    part_cfg["center_y"] = cyr
    part_cfg["rotation_deg"] = float(part_cfg.get("rotation_deg", 0.0)) + float(extra_rot_deg)
    insts = part_cfg.get("antennae_instances", [])
    if isinstance(insts, list) and insts:
        new_insts = []
        for inst in insts:
            if not isinstance(inst, dict):
                continue
            icx = float(inst.get("center_x", 0.0))
            icy = float(inst.get("center_y", 0.0))
            icxr, icyr = _rot_xy(icx, icy, extra_rot_deg)
            ii = dict(inst)
            ii["center_x"] = float(icxr)
            ii["center_y"] = float(icyr)
            new_insts.append(ii)
        part_cfg["antennae_instances"] = new_insts
    _, p_mask_r, fill_frac_r = _single_part_mask_and_fill(x, y, part_cfg)
    p_mask_r, fill_frac_r = _apply_shell_to_mask(x, y, part_cfg, p_mask_r, fill_frac_r)
    p_mask_r, fill_frac_r = _apply_antennae_to_mask(x, y, part_cfg, p_mask_r, fill_frac_r)
    d_mask_r = p_mask_r.copy()
    part_id_mask = np.zeros_like(p_mask_r, dtype=np.int32)
    part_id_mask[p_mask_r] = 1
    return p_mask_r, d_mask_r, fill_frac_r, part_id_mask


def solve_electric_state(
    sigma: np.ndarray,
    eps_r: np.ndarray,
    omega: float,
    elec_hi: np.ndarray,
    elec_lo: np.ndarray,
    v_hi: float,
    v_lo: float,
    dx: float,
    dy: float,
    elec_cfg: dict,
    x: np.ndarray,
    y: np.ndarray,
    power_factor: float,
    max_qrf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gamma = np.asarray(sigma, dtype=float) + 1j * omega * EPS0 * np.asarray(eps_r, dtype=float)
    V = solve_eqs_complex(
        gamma=gamma,
        elec_hi=elec_hi,
        elec_lo=elec_lo,
        v_hi=v_hi,
        v_lo=v_lo,
        dx=dx,
        dy=dy,
        steps=int(elec_cfg["solver_steps"]),
        tol=float(elec_cfg.get("solver_tol", 1e-8)),
        relaxation=float(elec_cfg.get("solver_relaxation", 0.45)),
    )
    Ex, Ey, E_mag, Qrf = compute_electric_fields(V, x, y, gamma, power_factor=power_factor)
    Qrf = np.clip(np.asarray(Qrf, dtype=float), 0.0, max_qrf)
    return gamma, V, Ex, Ey, E_mag, Qrf


def enforce_generator_power(
    Qrf: np.ndarray,
    doped_mask: np.ndarray,
    dA: float,
    target_power_w_per_m: float | None,
    max_qrf: float,
    max_iter: int = 4,
) -> tuple[np.ndarray, float, float]:
    if target_power_w_per_m is None:
        p_now = float(np.sum(Qrf[doped_mask]) * dA)
        return Qrf, p_now, 1.0
    target = float(target_power_w_per_m)
    if target <= 0.0:
        p_now = float(np.sum(Qrf[doped_mask]) * dA)
        return Qrf, p_now, 1.0

    q = np.array(Qrf, dtype=float, copy=True)
    p0 = float(np.sum(q[doped_mask]) * dA)
    if p0 <= 1e-18:
        return q, p0, 1.0

    total_scale = 1.0
    for _ in range(max(1, int(max_iter))):
        p_now = float(np.sum(q[doped_mask]) * dA)
        if p_now <= 1e-18:
            break
        corr = target / p_now
        total_scale *= corr
        q = np.clip(q * corr, 0.0, max_qrf)
        p_new = float(np.sum(q[doped_mask]) * dA)
        if p_new <= 1e-18:
            break
        if abs(p_new - target) / max(target, 1e-9) < 1e-3:
            return q, p_new, total_scale

    p_final = float(np.sum(q[doped_mask]) * dA)
    return q, p_final, total_scale


def thermal_step(
    T: np.ndarray,
    rho_rel: np.ndarray,
    Qrf: np.ndarray,
    part_mask: np.ndarray,
    expo: np.ndarray,
    params: ThermalParams,
    x_cryst: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, float]]:
    phi_part, dphi_dT = phase_fraction(T, params.phase_cfg)
    # Defensive scalar extraction to avoid interpreter-level key-load artifacts
    # observed on some Python 3.14 builds.
    rho_powder = float(params.rho_powder)
    k_powder = float(params.k_powder)
    cp_powder = float(params.cp_powder)
    rho_solid = float(params.rho_solid)
    rho_liquid = float(params.rho_liquid)
    k_solid = float(params.k_solid)
    k_liquid = float(params.k_liquid)
    cp_solid = float(params.cp_solid)
    cp_liquid = float(params.cp_liquid)
    latent_heat = float(params.latent_heat)
    ambient_c = float(params.ambient_c)
    dt = float(params.dt)
    dA = float(params.dA)
    dx = float(params.dx)
    dy = float(params.dy)
    temp_min_c = float(params.temp_min_c)
    temp_max_c = float(params.temp_max_c)
    convection_model = str(params.convection_model).strip().lower()
    h_const = float(params.h_const)
    conv_external_temp_k = float(params.conv_external_temp_k)
    conv_plate_distance_m = float(params.conv_plate_distance_m)
    conv_chimney_height_m = float(params.conv_chimney_height_m)
    conv_pressure_pa = float(params.conv_pressure_pa)
    dens_phi_exponent = float(params.dens_phi_exponent)
    dens_rho_exponent = float(params.dens_rho_exponent)
    dens_max_delta_per_step = float(params.dens_max_delta_per_step)

    rho = np.full_like(T, rho_powder)
    k = np.full_like(T, k_powder)
    cp_base = np.full_like(T, cp_powder)

    rho_s_eff = rho_powder + rho_rel * (rho_solid - rho_powder)
    k_s_eff = k_powder + rho_rel * (k_solid - k_powder)

    rho_local = (1.0 - phi_part) * rho_s_eff + phi_part * rho_liquid
    k_local = (1.0 - phi_part) * k_s_eff + phi_part * k_liquid
    cp_local = (1.0 - phi_part) * cp_solid + phi_part * cp_liquid

    rho[part_mask] = rho_local[part_mask]
    k[part_mask] = k_local[part_mask]
    cp_base[part_mask] = cp_local[part_mask]
    cp_eff = cp_base + latent_heat * dphi_dT

    div_term = diffusion_divergence(T, k, dx, dy)
    if convection_model == "natural_chimney_parallel_plates":
        h_field = chimney_natural_convection_h(
            T,
            conv_external_temp_k,
            conv_plate_distance_m,
            conv_chimney_height_m,
            conv_pressure_pa,
        )
    else:
        h_field = h_const
    q_conv = h_field * expo * (T - ambient_c)

    # Z-direction thermal loss for 2D depth correction (accounts for finite
    # part depth in a 3D chamber).  Approximates conduction loss through
    # powder above/below the part using a series-resistance slab model.
    q_z_loss = np.zeros_like(T)
    if params.depth_correction_enabled:
        Lp = 0.5 * float(params.depth_correction_part_depth_m)
        Lc = 0.5 * float(params.depth_correction_chamber_depth_m)
        Lpowder = max(Lc - Lp, 1e-6)
        # Series thermal resistance: part half-depth + powder layer
        # scaled by conductivity ratio
        k_ratio = np.where(k > 1e-12, k_powder / k, 1.0)
        L_eff = Lp + Lpowder * k_ratio
        q_z_loss_field = k * (T - ambient_c) * (np.pi**2 / (4.0 * L_eff**2))
        q_z_loss[part_mask] = q_z_loss_field[part_mask]

    dTdt = (div_term + Qrf - q_conv - q_z_loss) / np.maximum(rho * cp_eff, 1e-9)
    dTdt = np.nan_to_num(dTdt, nan=0.0, posinf=0.0, neginf=0.0)

    dT_step_raw = np.asarray(dt * dTdt, dtype=float)
    dT_step = np.array(dT_step_raw, copy=True)
    dT_cap = float(params.max_dt_step_c)
    clip_enabled = np.isfinite(dT_cap) and (dT_cap > 0.0)
    if clip_enabled:
        dT_cap_abs = abs(dT_cap)
        # Use clip() instead of chained where() for numerical robustness and simpler semantics.
        dT_step = np.clip(dT_step, -dT_cap_abs, dT_cap_abs)
    else:
        dT_cap_abs = float("inf")
    T_new = np.clip(np.asarray(T, dtype=float) + dT_step, temp_min_c, temp_max_c)
    # Keep an immutable copy for return; use a separate working copy for phase/density.
    T_return = np.array(T_new, copy=True)
    T_work = np.array(T_new, copy=True)
    if bool(params.debug):
        print(
            "DEBUG thermal_step",
            "T_in",
            float(np.min(T)),
            float(np.max(T)),
            "Qrf",
            float(np.min(Qrf)),
            float(np.max(Qrf)),
            "dT_cap",
            float(dT_cap_abs if np.isfinite(dT_cap_abs) else 0.0),
            "dT_raw",
            float(np.min(dT_step_raw)),
            float(np.max(dT_step_raw)),
            "n_gt_cap",
            int(np.sum(dT_step_raw > dT_cap_abs)) if np.isfinite(dT_cap_abs) else 0,
            "n_lt_cap",
            int(np.sum(dT_step_raw < -dT_cap_abs)) if np.isfinite(dT_cap_abs) else 0,
            "dT_clipped",
            float(np.min(dT_step)),
            float(np.max(dT_step)),
            "T_new",
            float(np.min(T_return)),
            float(np.max(T_return)),
        )

    phi_out, _ = phase_fraction(T_work, params.phase_cfg)
    phi_out = np.array(phi_out, dtype=float, copy=True)
    phi_out = np.nan_to_num(phi_out, nan=0.0, posinf=1.0, neginf=0.0)
    # Robust clamp (avoid rare clip-path corruption seen on some runtimes).
    phi_out = np.where(phi_out < 0.0, 0.0, np.where(phi_out > 1.0, 1.0, phi_out))
    Tk = np.maximum(np.array(T_work, dtype=float, copy=True) + 273.15, 1.0)
    one_minus_rho = np.clip(1.0 - rho_rel, 0.0, 1.0)
    # Optional non-isothermal crystallization kinetics for experimental bucket.
    x_cryst_new: np.ndarray | None = None
    phi_for_dens = np.array(phi_out, dtype=float, copy=True)
    if params.crystallization_enabled:
        if x_cryst is None:
            x_cryst_prev = np.zeros_like(T_work)
        else:
            x_cryst_prev = np.clip(np.asarray(x_cryst, dtype=float), 0.0, 1.0)
        cool_rate = np.maximum((np.asarray(T, dtype=float) - T_work) / max(dt, 1e-9), 0.0)
        cryst_model = str(params.crystallization_model).strip().lower()
        k_cr = float(params.crystallization_k0_per_s) * np.exp(
            -float(params.crystallization_ea_j_per_mol) / (R_GAS * Tk)
        )
        n_cr = max(float(params.crystallization_exponent), 1e-6)
        if cryst_model in {"avrami_nonisothermal", "avrami"}:
            rate_cr = k_cr * np.power(np.clip(1.0 - x_cryst_prev, 0.0, 1.0), n_cr)
        else:
            # Nakamura-like cooling-weighted growth form.
            rate_cr = k_cr * np.power(np.clip(1.0 - x_cryst_prev, 0.0, 1.0), n_cr) * (
                cool_rate / (cool_rate + 1.0)
            )
        x_cryst_new = np.clip(x_cryst_prev + dt * rate_cr, 0.0, 1.0)
        liq_supp = np.clip(float(params.crystallization_liquid_suppression), 0.0, 1.0)
        phi_for_dens = np.clip(phi_out * (1.0 - liq_supp * x_cryst_new), 0.0, 1.0)

    phi_term = np.power(phi_for_dens, dens_phi_exponent)
    rho_term = np.power(one_minus_rho, dens_rho_exponent)

    dens_model = str(params.dens_model).strip().lower()
    kdens_ss = np.zeros_like(T_work)
    kdens_liq = np.zeros_like(T_work)

    if dens_model in {"experimental_viscous_capillary_pa12"}:
        visc_model = str(params.experimental_viscosity_model).strip().lower()
        if visc_model == "wlf":
            t_ref = max(float(params.experimental_viscosity_ref_temp_k), 1.0)
            c1 = float(params.experimental_wlf_c1)
            c2 = float(params.experimental_wlf_c2_k)
            dtk = Tk - t_ref
            denom = np.where(np.abs(c2 + dtk) < 1e-6, np.sign(c2 + dtk) * 1e-6, c2 + dtk)
            log10_shift = -c1 * dtk / denom
            eta = max(float(params.experimental_viscosity_ref_pa_s), 1e-12) * np.power(10.0, log10_shift)
        else:
            eta_ref = max(float(params.experimental_viscosity_ref_pa_s), 1e-12)
            eta_ea = max(float(params.experimental_viscosity_activation_j_per_mol), 0.0)
            t_ref = max(float(params.experimental_viscosity_ref_temp_k), 1.0)
            eta = eta_ref * np.exp(eta_ea / R_GAS * (1.0 / Tk - 1.0 / t_ref))

        kdens_liq = float(params.dens_geom_factor) * float(params.dens_surface_tension_n_per_m) / (
            np.maximum(eta, 1e-12) * max(float(params.dens_particle_radius_m), 1e-9)
        )
        phi_thr = float(params.experimental_phi_threshold)
        phi_act = np.clip((phi_for_dens - phi_thr) / max(1.0 - phi_thr, 1e-9), 0.0, 1.0)
        phi_drive = np.power(phi_act, float(params.experimental_phi_exponent))
        porosity = np.clip(1.0 - rho_rel, 0.0, 1.0)
        porosity_drive = np.power(porosity, float(params.experimental_porosity_exponent))
        rate = kdens_liq * phi_drive * porosity_drive
    elif dens_model in {"physics_dual", "dual_mechanism"}:
        # Two-mechanism kinetics:
        # 1) solid-state diffusion/creep (active below full melt),
        # 2) liquid-assisted capillary flow (active once local melt fraction is present).
        dens_k0_ss = float(params.dens_k0_ss)
        dens_ea_ss = float(params.dens_ea_ss)
        dens_phi_solid_exponent = float(params.dens_phi_solid_exponent)
        dens_phi_threshold = float(params.dens_phi_threshold)
        dens_phi_liq_exponent = float(params.dens_phi_liq_exponent)
        dens_liquid_rate_mode = str(params.dens_liquid_rate_mode).strip().lower()
        dens_k0_liq = float(params.dens_k0_liq)
        dens_ea_liq = float(params.dens_ea_liq)
        dens_eta_ref_pa_s = float(params.dens_eta_ref_pa_s)
        dens_eta_activation_j_per_mol = float(params.dens_eta_activation_j_per_mol)
        dens_eta_ref_temp_k = float(params.dens_eta_ref_temp_k)
        dens_geom_factor = float(params.dens_geom_factor)
        dens_surface_tension_n_per_m = float(params.dens_surface_tension_n_per_m)
        dens_particle_radius_m = float(params.dens_particle_radius_m)

        kdens_ss = dens_k0_ss * np.exp(-dens_ea_ss / (R_GAS * Tk))
        ss_drive = np.power(np.clip(1.0 - phi_out, 0.0, 1.0), dens_phi_solid_exponent)

        phi_thr = dens_phi_threshold
        phi_act = np.clip((phi_out - phi_thr) / max(1.0 - phi_thr, 1e-9), 0.0, 1.0)
        liq_drive = np.power(phi_act, dens_phi_liq_exponent)

        liq_mode = dens_liquid_rate_mode
        if liq_mode in {"viscous_capillary", "capillary_viscous"}:
            eta_ref = max(dens_eta_ref_pa_s, 1e-9)
            eta_ea = max(dens_eta_activation_j_per_mol, 0.0)
            t_ref = max(dens_eta_ref_temp_k, 1.0)
            eta = eta_ref * np.exp(eta_ea / R_GAS * (1.0 / Tk - 1.0 / t_ref))
            kdens_liq = dens_geom_factor * dens_surface_tension_n_per_m / (
                np.maximum(eta, 1e-12) * max(dens_particle_radius_m, 1e-9)
            )
            # Optional Arrhenius penalty for the liquid mechanism (can be 0).
            if dens_ea_liq > 0.0:
                kdens_liq = kdens_liq * np.exp(-dens_ea_liq / (R_GAS * Tk))
        else:
            kdens_liq = dens_k0_liq * np.exp(-dens_ea_liq / (R_GAS * Tk))

        rate = (kdens_ss * ss_drive + kdens_liq * liq_drive) * rho_term
    elif dens_model in {"viscous_capillary", "capillary_viscous"}:
        # Physics-motivated densification: capillary pressure over melt viscosity.
        # kdens ~ C * gamma / (eta(T) * r_particle), eta(T) from Arrhenius law.
        eta_ref = max(float(params.dens_eta_ref_pa_s), 1e-9)
        eta_ea = max(float(params.dens_eta_activation_j_per_mol), 0.0)
        t_ref = max(float(params.dens_eta_ref_temp_k), 1.0)
        eta = eta_ref * np.exp(eta_ea / R_GAS * (1.0 / Tk - 1.0 / t_ref))
        kdens_liq = float(params.dens_geom_factor) * float(params.dens_surface_tension_n_per_m) / (
            np.maximum(eta, 1e-12) * max(float(params.dens_particle_radius_m), 1e-9)
        )
        rate = kdens_liq * phi_term * rho_term
    else:
        kdens_ss = float(params.dens_k0) * np.exp(-float(params.dens_ea) / (R_GAS * Tk))
        rate = kdens_ss * phi_term * rho_term

    drho = dt * rate
    drho = np.clip(drho, 0.0, dens_max_delta_per_step)

    rho_rel_new = np.array(rho_rel, copy=True)
    rho_rel_new[part_mask] = np.clip(rho_rel_new[part_mask] + drho[part_mask], 0.0, 1.0)
    rho_rel_new = np.nan_to_num(rho_rel_new, nan=0.0, posinf=1.0, neginf=0.0)
    rho_rel_new = np.where(rho_rel_new < 0.0, 0.0, np.where(rho_rel_new > 1.0, 1.0, rho_rel_new))
    if bool(params.debug):
        print(
            "DEBUG thermal_step return",
            "T_out",
            float(np.min(T_return)),
            float(np.max(T_return)),
            "phi_out",
            float(np.min(phi_out)),
            float(np.max(phi_out)),
            "rho_rel_new",
            float(np.min(rho_rel_new)),
            float(np.max(rho_rel_new)),
        )
    max_dT_raw = float(np.max(np.abs(dT_step_raw)))
    max_dT_used = float(min(max_dT_raw, dT_cap_abs)) if np.isfinite(dT_cap_abs) else max_dT_raw
    diag = {
        "p_qrf_gen_w_per_m": float(np.sum(np.maximum(Qrf, 0.0)) * dA),
        "p_conv_loss_w_per_m": float(np.sum(np.maximum(q_conv, 0.0)) * dA),
        "mean_dens_rate_part_per_s": float(np.mean(rate[part_mask])),
        "mean_kdens_ss_part_per_s": float(np.mean(kdens_ss[part_mask])),
        "mean_kdens_liq_part_per_s": float(np.mean(kdens_liq[part_mask])),
        "mean_phi_part": float(np.mean(phi_out[part_mask])),
        "max_dT_raw_c": max_dT_raw,
        "max_dT_used_c": max_dT_used,
        "frac_cells_dT_clipped": float(
            np.mean(np.abs(dT_step_raw) > dT_cap_abs) if np.isfinite(dT_cap_abs) else 0.0
        ),
    }
    if x_cryst_new is not None:
        diag["mean_x_cryst_part"] = float(np.mean(x_cryst_new[part_mask]))
    return T_return, rho_rel_new, phi_out, x_cryst_new, diag


def part_energy_per_depth(T: np.ndarray, rho_rel: np.ndarray, part_mask: np.ndarray, params: dict) -> float:
    # Stored thermal energy per meter depth in the part:
    # sensible + latent, referenced to ambient temperature.
    T_eval = np.array(T, dtype=float, copy=True)
    rho_eval = np.array(rho_rel, dtype=float, copy=True)
    mask = np.array(part_mask, dtype=bool, copy=False)
    rho_powder = float(params.rho_powder)
    rho_solid = float(params.rho_solid)
    rho_liquid = float(params.rho_liquid)
    cp_solid = float(params.cp_solid)
    cp_liquid = float(params.cp_liquid)
    ambient_c = float(params.ambient_c)
    latent_heat = float(params.latent_heat)
    dA = float(params.dA)

    # Robust scalar accumulation in-part only.
    # This avoids intermittent vectorized-sum corruption observed on some Python 3.14
    # builds for this workflow.
    ii, jj = np.where(mask)
    phase_model = str(params.phase_cfg.model).strip().lower()
    t_pc = float(params.phase_cfg.t_pc_c)
    dt_pc = max(float(params.phase_cfg.dt_pc_c), 1e-9)
    t_solidus = float(params.phase_cfg.solidus_c)
    t_liquidus = float(params.phase_cfg.liquidus_c)
    dT_lin = max(t_liquidus - t_solidus, 1e-9)
    smooth_shape = str(params.phase_cfg.smooth_shape).strip().lower()
    tanh_beta = max(float(params.phase_cfg.tanh_beta), 1e-6)

    total = 0.0
    for i, j in zip(ii.tolist(), jj.tolist()):
        t_ij = float(T_eval[i, j])
        if phase_model in {"comsol_heaviside", "heaviside"}:
            arg = (t_ij - t_pc) / dt_pc
            if smooth_shape == "tanh":
                ph = 0.5 * (1.0 + math.tanh(tanh_beta * arg))
            else:
                ph = min(max(arg + 0.5, 0.0), 1.0)
        else:
            ph = min(max((t_ij - t_solidus) / dT_lin, 0.0), 1.0)
        rr = float(rho_eval[i, j])
        rs = rho_powder + rr * (rho_solid - rho_powder)
        rloc = (1.0 - ph) * rs + ph * rho_liquid
        cploc = (1.0 - ph) * cp_solid + ph * cp_liquid
        dT_ij = max(t_ij - ambient_c, 0.0)
        total += rloc * cploc * dT_ij + rloc * latent_heat * ph
    energy = total * dA
    return float(max(energy, 0.0))


def run_sim(cfg: dict) -> tuple[SimState, dict, dict[str, list[float]]]:
    antennae_resolved = {"enabled": False, "instances": []}
    if _antennae_enabled(cfg):
        try:
            antennae_resolved = _resolve_antennae_instances(cfg)
        except Exception as ant_err:
            print(f"  [antennae] resolve skipped: {ant_err}")
            antennae_resolved = {"enabled": False, "instances": []}
    x, y, _, part_mask, doped_mask, elec_hi, elec_lo, fill_frac, part_id_mask, _part_polys = make_domain(cfg)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    elec = cfg["electric"]
    therm = cfg["thermal"]
    dens = cfg["densification"]
    mat = cfg["materials"]
    physics_cfg = cfg.get("physics_model", {}) if isinstance(cfg.get("physics_model", {}), dict) else {}
    physics_model_family = str(physics_cfg.get("family", "baseline")).strip().lower() or "baseline"
    if physics_model_family not in {"baseline", "experimental_pa12_hybrid"}:
        raise ValueError(f"Unsupported physics_model.family: {physics_model_family}")
    experimental_enabled = bool(physics_cfg.get("experimental_enabled", False))
    experimental_densification_enabled = bool(physics_cfg.get("experimental_densification_enabled", True))
    if physics_model_family != "baseline" and not experimental_enabled:
        raise ValueError(
            "physics_model.experimental_enabled must be true when physics_model.family != baseline"
        )
    provenance_tag = str(physics_cfg.get("provenance_tag", "")).strip()
    parameter_source = str(physics_cfg.get("parameter_source", "baseline")).strip()
    calibration_version = str(physics_cfg.get("calibration_version", "baseline")).strip()
    ab_bucket_id = str(physics_cfg.get("ab_bucket_id", "")).strip()
    literature_refs = physics_cfg.get("literature_refs", [])
    if not isinstance(literature_refs, list):
        literature_refs = []

    ny, nx = part_mask.shape
    dA = dx * dy
    f_hz = float(elec["frequency_hz"])
    omega = 2.0 * math.pi * f_hz
    n_steps = int(therm["n_steps"])
    dt = float(therm["dt_s"])
    update_interval = max(0, int(elec.get("update_interval", 1)))
    power_factor = float(elec.get("power_factor", 1.0))
    max_qrf = float(elec.get("max_qrf_w_per_m3", 2.0e9))
    zero_qrf_outside_doped = bool(elec.get("zero_qrf_outside_doped", True))
    qrf_file_npy = str(elec.get("qrf_file_npy", "")).strip()
    fixed_qrf_mode = bool(qrf_file_npy)
    enforce_gen_power = bool(elec.get("enforce_generator_power", False))
    gen_power_w = float(elec.get("generator_power_w", 0.0))
    gen_eff = float(elec.get("generator_transfer_efficiency", elec.get("transfer_efficiency", 1.0)))
    effective_depth_m = max(float(elec.get("effective_depth_m", 1.0)), 1e-9)
    target_power_w_per_m = None
    if enforce_gen_power and gen_power_w > 0.0:
        target_power_w_per_m = gen_power_w * gen_eff / effective_depth_m
    max_dt_step_c = float(therm.get("max_deltaT_per_step_c", 3.0))
    temp_min_c = float(therm.get("min_temp_c", -50.0))
    temp_max_c = float(therm.get("max_temp_c", 450.0))

    v0 = float(elec["voltage_v"])
    mode = str(elec.get("voltage_mode", "centered")).strip().lower()
    if "voltage_hi_v" in elec or "voltage_lo_v" in elec:
        v_hi = float(elec.get("voltage_hi_v", v0))
        v_lo = float(elec.get("voltage_lo_v", 0.0))
    elif mode == "grounded":
        v_hi = v0
        v_lo = 0.0
    else:
        v_hi = 0.5 * v0
        v_lo = -0.5 * v0

    sigma_v = max(float(mat["virgin"]["sigma_s_per_m"]), 1e-8)
    eps_v = float(mat["virgin"]["eps_r"])
    sigma_d0 = float(mat["doped"]["sigma_s_per_m"])
    eps_d = float(mat["doped"]["eps_r"])
    sigma_temp_coeff = float(mat["doped"].get("sigma_temp_coeff_per_K", 0.0))
    sigma_density_coeff = float(mat["doped"].get("sigma_density_coeff", 0.0))
    sigma_ref_temp = float(mat["doped"].get("sigma_ref_temp_c", 20.0))
    rho_rel_init = float(dens.get("rho_rel_initial", 0.55))

    phase_cfg = therm["phase_change"]
    exp_phase_cfg = cfg.get("phase_model", {}) if isinstance(cfg.get("phase_model", {}), dict) else {}
    exp_dens_cfg = cfg.get("dens_model", {}) if isinstance(cfg.get("dens_model", {}), dict) else {}
    exp_cr_cfg = cfg.get("crystallization_model", {}) if isinstance(cfg.get("crystallization_model", {}), dict) else {}
    provenance_table: dict | None = None
    dsc_profile: dict | None = None
    if physics_model_family == "experimental_pa12_hybrid":
        if str(exp_phase_cfg.get("type", "")).strip().lower() not in {"apparent_heat_capacity_dsc"}:
            raise ValueError("experimental_pa12_hybrid requires phase_model.type = apparent_heat_capacity_dsc")
        if str(exp_dens_cfg.get("type", "")).strip().lower() not in {"viscous_capillary_pa12"}:
            raise ValueError("experimental_pa12_hybrid requires dens_model.type = viscous_capillary_pa12")
        prov_path_raw = str(physics_cfg.get("provenance_file", "configs/experimental_pa12_provenance.yaml")).strip()
        prov_path = Path(prov_path_raw)
        if not prov_path.is_absolute():
            prov_path = Path(__file__).resolve().parent / prov_path
        provenance_table = _load_provenance_table(prov_path)
        req_prov = [
            "phase_model.melt_onset_c",
            "phase_model.melt_peak_c",
            "phase_model.melt_end_c",
            "phase_model.lat_heat_j_per_kg",
            "dens_model.surface_tension_n_per_m",
            "dens_model.particle_radius_m",
            "dens_model.viscosity_model",
        ]
        if bool(exp_cr_cfg.get("enabled", False)):
            req_prov.extend(
                [
                    "crystallization_model.k0_per_s",
                    "crystallization_model.ea_j_per_mol",
                    "crystallization_model.exponent",
                ]
            )
        _validate_experimental_provenance(provenance_table, req_prov)
        dsc_profile_raw = str(physics_cfg.get("dsc_profile_file", "configs/experimental_pa12_dsc_profile.yaml")).strip()
        dsc_profile_path = Path(dsc_profile_raw)
        if not dsc_profile_path.is_absolute():
            dsc_profile_path = Path(__file__).resolve().parent / dsc_profile_path
        dsc_profile = _load_dsc_profile(dsc_profile_path)
    if physics_model_family == "experimental_pa12_hybrid":
        use_dsc_profile = bool(exp_phase_cfg.get("use_dsc_profile", True))
        if use_dsc_profile and dsc_profile is not None:
            onset = float(dsc_profile["melt_onset_c"])
            peak = float(dsc_profile["melt_peak_c"])
            end = float(dsc_profile["melt_end_c"])
            latent_heat = float(dsc_profile["lat_heat_j_per_kg"])
        else:
            onset = float(exp_phase_cfg.get("melt_onset_c", 120.0))
            peak = float(exp_phase_cfg.get("melt_peak_c", 150.0))
            end = float(exp_phase_cfg.get("melt_end_c", 180.0))
            latent_heat = float(exp_phase_cfg.get("lat_heat_j_per_kg", phase_cfg.get("latent_heat_j_per_kg", 96700.0)))
        if not (onset < peak < end):
            raise ValueError("phase_model melt temperatures must satisfy onset < peak < end")
        phase_user = PhaseConfig(
            model="apparent_heat_capacity_dsc",
            solidus_c=onset,
            liquidus_c=end,
            t_pc_c=peak,
            dt_pc_c=max(end - onset, 1e-9),
            smooth_shape=str(exp_phase_cfg.get("cp_smoothing_strategy", "linear")),
            tanh_beta=float(phase_cfg.get("tanh_beta", 8.0)),
            dsc_onset_c=onset,
            dsc_peak_c=peak,
            dsc_end_c=end,
            cp_smoothing_strategy=str(exp_phase_cfg.get("cp_smoothing_strategy", "linear")),
        )
    else:
        phase_model = str(phase_cfg.get("model", "linear_window")).strip().lower()
        if phase_model in {"comsol_heaviside", "heaviside"}:
            if "t_pc_c" in phase_cfg and "dt_pc_c" in phase_cfg:
                phase_user = PhaseConfig(
                    model="comsol_heaviside",
                    solidus_c=0.0,
                    liquidus_c=0.0,
                    t_pc_c=float(phase_cfg["t_pc_c"]),
                    dt_pc_c=float(phase_cfg["dt_pc_c"]),
                    smooth_shape=str(phase_cfg.get("smooth_shape", "linear")),
                    tanh_beta=float(phase_cfg.get("tanh_beta", 8.0)),
                )
            else:
                # Backward compatibility when only solidus/liquidus are provided.
                t_s = float(phase_cfg["solidus_c"])
                t_l = float(phase_cfg["liquidus_c"])
                phase_user = PhaseConfig(
                    model="comsol_heaviside",
                    solidus_c=t_s,
                    liquidus_c=t_l,
                    t_pc_c=0.5 * (t_s + t_l),
                    dt_pc_c=max(t_l - t_s, 1e-9),
                    smooth_shape=str(phase_cfg.get("smooth_shape", "linear")),
                    tanh_beta=float(phase_cfg.get("tanh_beta", 8.0)),
                )
        else:
            if "solidus_c" in phase_cfg and "liquidus_c" in phase_cfg:
                t_s = float(phase_cfg["solidus_c"])
                t_l = float(phase_cfg["liquidus_c"])
                phase_user = PhaseConfig(
                    model="linear_window",
                    solidus_c=t_s,
                    liquidus_c=t_l,
                    t_pc_c=0.5 * (t_s + t_l),
                    dt_pc_c=max(t_l - t_s, 1e-9),
                    smooth_shape=str(phase_cfg.get("smooth_shape", "linear")),
                    tanh_beta=float(phase_cfg.get("tanh_beta", 8.0)),
                )
            else:
                t_pc = float(phase_cfg["t_pc_c"])
                dT_pc = float(phase_cfg["dt_pc_c"])
                phase_user = PhaseConfig(
                    model="linear_window",
                    solidus_c=t_pc - 0.5 * dT_pc,
                    liquidus_c=t_pc + 0.5 * dT_pc,
                    t_pc_c=t_pc,
                    dt_pc_c=max(dT_pc, 1e-9),
                    smooth_shape=str(phase_cfg.get("smooth_shape", "linear")),
                    tanh_beta=float(phase_cfg.get("tanh_beta", 8.0)),
                )
        latent_heat = float(phase_cfg["latent_heat_j_per_kg"])
    ambient_c = float(therm["ambient_c"])

    convection_model = str(therm.get("convection_model", "constant")).strip().lower()
    h_const = float(therm.get("convection_h_w_per_m2k", 25.0))
    conv_plate_distance_m = float(therm.get("conv_plate_distance_m", 0.01))
    conv_chimney_height_m = float(therm.get("conv_chimney_height_m", 0.01))
    conv_pressure_pa = float(therm.get("conv_pressure_pa", 1.0133e5))
    conv_external_temp_k = float(therm.get("conv_external_temp_k", ambient_c + 273.15))

    powder = mat["powder"]
    doped = mat["doped"]
    rho_powder = float(powder["rho_solid_kg_per_m3"])
    k_powder = float(powder["k_solid_w_per_mk"])
    cp_powder = float(powder["cp_solid_j_per_kgk"])
    rho_solid = float(doped["rho_solid_kg_per_m3"])
    rho_liquid = float(doped["rho_liquid_kg_per_m3"])
    k_solid = float(doped["k_solid_w_per_mk"])
    k_liquid = float(doped["k_liquid_w_per_mk"])
    cp_solid = float(doped["cp_solid_j_per_kgk"])
    cp_liquid = float(doped["cp_liquid_j_per_kgk"])

    dens_k0 = float(dens["k0_per_s"])
    dens_ea = float(dens["activation_energy_j_per_mol"])
    dens_model = str(dens.get("model", "arrhenius_phi")).strip().lower()
    if physics_model_family == "experimental_pa12_hybrid" and experimental_densification_enabled:
        dens_model = "experimental_viscous_capillary_pa12"
    dens_phi_exponent = float(dens.get("phi_exponent", 1.0))
    dens_rho_exponent = float(dens.get("rho_exponent", 1.0))
    dens_max_delta_per_step = float(dens.get("max_delta_per_step", 0.05))
    dens_surface_tension_n_per_m = float(dens.get("surface_tension_n_per_m", 0.03))
    dens_particle_radius_m = float(dens.get("particle_radius_m", 35e-6))
    dens_eta_ref_pa_s = float(dens.get("eta_ref_pa_s", 8.0e3))
    dens_eta_ref_temp_k = float(dens.get("eta_ref_temp_k", 458.15))
    dens_eta_activation_j_per_mol = float(dens.get("eta_activation_j_per_mol", 6.0e4))
    dens_geom_factor = float(dens.get("geom_factor", 0.05))
    exp_viscosity_model = str(exp_dens_cfg.get("viscosity_model", "arrhenius")).strip().lower()
    exp_visc_params = exp_dens_cfg.get("viscosity_params", {}) if isinstance(exp_dens_cfg.get("viscosity_params", {}), dict) else {}
    exp_visc_ref_pa_s = float(exp_visc_params.get("eta_ref_pa_s", dens_eta_ref_pa_s))
    exp_visc_ref_temp_k = float(exp_visc_params.get("eta_ref_temp_k", dens_eta_ref_temp_k))
    exp_visc_ea = float(exp_visc_params.get("activation_energy_j_per_mol", dens_eta_activation_j_per_mol))
    exp_wlf_c1 = float(exp_visc_params.get("wlf_c1", 8.86))
    exp_wlf_c2 = float(exp_visc_params.get("wlf_c2_k", 101.6))
    exp_phi_exp = float(exp_dens_cfg.get("phi_exponent", dens_phi_exponent))
    exp_porosity_cfg = exp_dens_cfg.get("porosity_coupling_params", {})
    if not isinstance(exp_porosity_cfg, dict):
        exp_porosity_cfg = {}
    exp_porosity_exp = float(exp_porosity_cfg.get("porosity_exponent", dens_rho_exponent))
    exp_phi_threshold = float(exp_dens_cfg.get("phi_threshold", dens.get("phi_threshold", 0.02)))
    if physics_model_family == "experimental_pa12_hybrid":
        dens_surface_tension_n_per_m = float(exp_dens_cfg.get("surface_tension_n_per_m", dens_surface_tension_n_per_m))
        dens_particle_radius_m = float(exp_dens_cfg.get("particle_radius_m", dens_particle_radius_m))
        dens_geom_factor = float(exp_dens_cfg.get("geom_factor", dens_geom_factor))

    crystallization_enabled = bool(exp_cr_cfg.get("enabled", False)) and physics_model_family == "experimental_pa12_hybrid"
    crystallization_model = str(exp_cr_cfg.get("type", "nakamura")).strip().lower()
    crystallization_params = exp_cr_cfg.get("params", {}) if isinstance(exp_cr_cfg.get("params", {}), dict) else {}
    crystallization_k0 = float(crystallization_params.get("k0_per_s", 0.05))
    crystallization_ea = float(crystallization_params.get("ea_j_per_mol", 4.2e4))
    crystallization_exp = float(crystallization_params.get("exponent", 2.0))
    crystallization_liq_supp = float(crystallization_params.get("liquid_suppression", 0.25))

    depth_cfg = therm.get("depth_correction", {})
    depth_correction_enabled = bool(depth_cfg.get("enabled", False))
    depth_correction_part_depth_m = float(depth_cfg.get("part_depth_m", 0.010))
    depth_correction_chamber_depth_m = float(depth_cfg.get("chamber_depth_m", 0.016))

    conv_boundaries_raw = therm.get("convective_boundaries", ["left", "right", "bottom", "top"])
    if conv_boundaries_raw is None:
        conv_boundaries = []
    elif isinstance(conv_boundaries_raw, list):
        conv_boundaries = [str(v) for v in conv_boundaries_raw]
    else:
        conv_boundaries = [str(conv_boundaries_raw)]
    expo = boundary_exposure_coeff((ny, nx), dx, dy, conv_boundaries)

    # Use sub-pixel fill fraction for anti-aliased EQS boundary.
    # Interior cells (fill=1) → doped values; exterior (fill=0) → virgin values;
    # boundary cells → smooth linear blend.  Binary part_mask is preserved for
    # thermal and densification physics (phase transitions need sharp boundaries).
    sigma = sigma_v + fill_frac * (sigma_d0 - sigma_v)
    eps_r = eps_v  + fill_frac * (eps_d  - eps_v)

    # --- Optional surface sigma model ---
    # When sigma_profile == "surface", the EQS concentrates conductivity in a
    # thin ring of cells at the part boundary (matching the COMSOL Tuned_Sigma
    # profile that is highest at the outer X/Y faces of the part).
    # Interior part cells are assigned a low sigma_interior_s_per_m.
    # This causes the EQS to produce the observed shell-heating Q_rf pattern.
    _sigma_profile = str(mat.get("doped", {}).get("sigma_profile", "uniform")).lower()
    if _sigma_profile == "surface" and part_mask.any():
        from scipy.ndimage import binary_erosion as _be
        _n_layers = int(mat["doped"].get("sigma_surface_layers", 1))
        _sig_int = float(mat["doped"].get(
            "sigma_interior_s_per_m", max(sigma_d0 * 0.01, 1e-6)
        ))
        _part_interior = _be(part_mask, iterations=_n_layers)
        # Surface ring keeps sigma_d0; interior cells get the reduced value.
        sigma[_part_interior] = _sig_int
        _n_surf = int((part_mask & ~_part_interior).sum())
        _n_int  = int(_part_interior.sum())
        print(
            f"  [surface sigma] layers={_n_layers}, "
            f"sigma_surf={sigma_d0:.4f} S/m, sigma_int={_sig_int:.4g} S/m, "
            f"surface={_n_surf} cells ({100*_n_surf/max(_n_surf+_n_int,1):.1f}%), "
            f"interior={_n_int} cells"
        )

    if fixed_qrf_mode:
        # Load Q_rf directly from a COMSOL export (e.g. qrf_2d_map.npy).
        # The EQS solve is bypassed; V/E fields are set to zero placeholders.
        from scipy.interpolate import RegularGridInterpolator as _RGI
        _qrf_raw = np.load(qrf_file_npy)
        _meta_path = Path(qrf_file_npy).parent / (Path(qrf_file_npy).stem + "_meta.json")
        if _meta_path.exists():
            import json as _json
            _m = _json.loads(_meta_path.read_text(encoding="utf-8"))
            _nx_s = int(_m["grid_nx"])
            _ny_s = int(_m["grid_ny"])
            _x_s = np.linspace(float(_m["x_min"]), float(_m["x_max"]), _nx_s)
            _y_s = np.linspace(float(_m["y_min"]), float(_m["y_max"]), _ny_s)
        else:
            _ny_s, _nx_s = _qrf_raw.shape
            _x_s = np.linspace(-0.5 * float(cfg["geometry"]["chamber_x"]),
                                0.5 * float(cfg["geometry"]["chamber_x"]), _nx_s)
            _y_s = np.linspace(-0.5 * float(cfg["geometry"]["chamber_y"]),
                                0.5 * float(cfg["geometry"]["chamber_y"]), _ny_s)
        _interp = _RGI((_y_s, _x_s), _qrf_raw, method="linear",
                       bounds_error=False, fill_value=0.0)
        _XX, _YY = np.meshgrid(x, y)
        _pts = np.column_stack([_YY.ravel(), _XX.ravel()])
        Qrf = np.maximum(_interp(_pts).reshape(ny, nx), 0.0)
        del _qrf_raw, _interp, _XX, _YY, _pts
        gamma = (np.full((ny, nx), sigma_v, dtype=np.complex128)
                 + 1j * omega * EPS0 * np.full((ny, nx), eps_v, dtype=float))
        gamma[part_mask] = sigma_d0 + 1j * omega * EPS0 * eps_d
        V = np.zeros((ny, nx), dtype=np.complex128)
        Ex = np.zeros((ny, nx), dtype=float)
        Ey = np.zeros((ny, nx), dtype=float)
        E_mag = np.zeros((ny, nx), dtype=float)
        qrf_scale_applied = 1.0
        if zero_qrf_outside_doped:
            Qrf = np.where(doped_mask, Qrf, 0.0)
        p_doped_eff_w_per_m = float(np.sum(Qrf[doped_mask]) * dA)
        # --- Power scaling for fixed_qrf_mode ---
        # target_power_w: total 3D RF power [W] absorbed in the part.
        # Converts to per-metre-depth using part Z-depth, then scales Q_rf
        # using the same enforce_generator_power() path as the EQS mode.
        _tp3d_w = float(elec.get("target_power_w", 0.0))
        if _tp3d_w > 0.0:
            _pz = float(depth_cfg.get("part_depth_m", depth_correction_part_depth_m))
            target_power_w_per_m = _tp3d_w / max(_pz, 1e-9)
        if target_power_w_per_m is not None:
            Qrf, p_doped_eff_w_per_m, qrf_scale_applied = enforce_generator_power(
                Qrf, doped_mask, dA, target_power_w_per_m, max_qrf
            )
            print(f"  [fixed_qrf] target_power_w={_tp3d_w:.1f} W → "
                  f"scale={qrf_scale_applied:.2f}×, "
                  f"absorbed={p_doped_eff_w_per_m:.1f} W/m depth "
                  f"({p_doped_eff_w_per_m * float(depth_cfg.get('part_depth_m', depth_correction_part_depth_m)):.1f} W total)")
    else:
        gamma, V, Ex, Ey, E_mag, Qrf = solve_electric_state(
            sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo, dx, dy, elec, x, y, power_factor, max_qrf
        )
        Qrf, p_doped_eff_w_per_m, qrf_scale_applied = enforce_generator_power(
            Qrf,
            doped_mask,
            dA,
            target_power_w_per_m,
            max_qrf,
        )
        if zero_qrf_outside_doped:
            Qrf = np.where(doped_mask, Qrf, 0.0)

    T = np.full((ny, nx), ambient_c, dtype=float)
    rho_rel = np.zeros((ny, nx), dtype=float)
    rho_rel[part_mask] = rho_rel_init
    phi = np.zeros_like(T)
    x_cryst = np.zeros_like(T) if crystallization_enabled else None

    hist: dict[str, list[float]] = {
        "time_s": [],
        "mean_T_part_c": [],
        "max_T_part_c": [],
        "ui_abs_part": [],
        "ui_rms_part": [],
        "mean_phi_part": [],
        "mean_rho_rel_part": [],
        "power_doped_W_per_m": [],
        "energy_doped_J_per_m": [],
        "power_conv_loss_W_per_m": [],
        "energy_conv_loss_J_per_m": [],
        "energy_stored_part_J_per_m": [],
        "energy_balance_residual_J_per_m": [],
        "mean_dens_rate_part_per_s": [],
        "mean_kdens_ss_part_per_s": [],
        "mean_kdens_liq_part_per_s": [],
        "mean_x_cryst_part": [],
        "max_dT_raw_c": [],
        "max_dT_used_c": [],
        "frac_cells_dT_clipped": [],
        "qrf_scale_applied": [],
        "target_power_doped_W_per_m": [],
    }

    # Explicit diffusion step stability guard (2D FTCS-style limit).
    k_max = max(k_powder, k_solid, k_liquid, 1e-12)
    rho_cp_min = max(
        min(
            rho_powder * cp_powder,
            rho_solid * cp_solid,
            rho_liquid * cp_liquid,
        ),
        1e-9,
    )
    dt_stable = 0.24 * (min(dx, dy) ** 2) * rho_cp_min / k_max
    n_substeps = max(1, int(math.ceil(dt / max(dt_stable, 1e-12))))
    dt_sub = dt / float(n_substeps)

    therm_params = ThermalParams(
        dx=dx,
        dy=dy,
        dt=dt_sub,
        # Treat this as a per-substep numerical limiter (not a global-step cap),
        # otherwise the limiter can dominate physics as substepping changes.
        max_dt_step_c=max_dt_step_c,
        temp_min_c=temp_min_c,
        temp_max_c=temp_max_c,
        phase_cfg=phase_user,
        latent_heat=latent_heat,
        ambient_c=ambient_c,
        convection_model=convection_model,
        h_const=h_const,
        conv_plate_distance_m=conv_plate_distance_m,
        conv_chimney_height_m=conv_chimney_height_m,
        conv_pressure_pa=conv_pressure_pa,
        conv_external_temp_k=conv_external_temp_k,
        rho_powder=rho_powder,
        k_powder=k_powder,
        cp_powder=cp_powder,
        rho_solid=rho_solid,
        rho_liquid=rho_liquid,
        k_solid=k_solid,
        k_liquid=k_liquid,
        cp_solid=cp_solid,
        cp_liquid=cp_liquid,
        dens_k0=dens_k0,
        dens_ea=dens_ea,
        dens_model=dens_model,
        dens_phi_exponent=dens_phi_exponent,
        dens_rho_exponent=dens_rho_exponent,
        dens_max_delta_per_step=dens_max_delta_per_step / float(n_substeps),
        dens_surface_tension_n_per_m=dens_surface_tension_n_per_m,
        dens_particle_radius_m=dens_particle_radius_m,
        dens_eta_ref_pa_s=dens_eta_ref_pa_s,
        dens_eta_ref_temp_k=dens_eta_ref_temp_k,
        dens_eta_activation_j_per_mol=dens_eta_activation_j_per_mol,
        dens_geom_factor=dens_geom_factor,
        dens_k0_ss=float(dens.get("k0_ss_per_s", max(0.1 * dens_k0, 1e-12))),
        dens_ea_ss=float(dens.get("activation_energy_ss_j_per_mol", max(0.6 * dens_ea, 0.0))),
        dens_k0_liq=float(dens.get("k0_liq_per_s", max(dens_k0, 1e-12))),
        dens_ea_liq=float(dens.get("activation_energy_liq_j_per_mol", 0.0)),
        dens_phi_threshold=float(dens.get("phi_threshold", 0.02)),
        dens_phi_solid_exponent=float(dens.get("phi_solid_exponent", 1.0)),
        dens_phi_liq_exponent=float(dens.get("phi_liq_exponent", 1.0)),
        dens_liquid_rate_mode=str(dens.get("liquid_rate_mode", "viscous_capillary")),
        dA=dA,
        depth_correction_enabled=depth_correction_enabled,
        depth_correction_part_depth_m=depth_correction_part_depth_m,
        depth_correction_chamber_depth_m=depth_correction_chamber_depth_m,
        physics_model_family=physics_model_family,
        experimental_viscosity_model=exp_viscosity_model,
        experimental_viscosity_ref_pa_s=exp_visc_ref_pa_s,
        experimental_viscosity_ref_temp_k=exp_visc_ref_temp_k,
        experimental_viscosity_activation_j_per_mol=exp_visc_ea,
        experimental_wlf_c1=exp_wlf_c1,
        experimental_wlf_c2_k=exp_wlf_c2,
        experimental_phi_exponent=exp_phi_exp,
        experimental_porosity_exponent=exp_porosity_exp,
        experimental_phi_threshold=exp_phi_threshold,
        crystallization_enabled=crystallization_enabled,
        crystallization_model=crystallization_model,
        crystallization_k0_per_s=crystallization_k0,
        crystallization_ea_j_per_mol=crystallization_ea,
        crystallization_exponent=crystallization_exp,
        crystallization_liquid_suppression=crystallization_liq_supp,
    )

    # ── Turntable mode setup ─────────────────────────────────────────────────────
    # New (Mar 2026): time-based rotation at arbitrary angles.
    # Electrodes stay fixed; part geometry is re-rasterized at each rotation event.
    # Backward-compatible: old `phases`-based config still works.
    tt_cfg = cfg.get("turntable", {})
    tt_enabled = bool(tt_cfg.get("enabled", False)) and not fixed_qrf_mode
    tt_rotation_steps: list[int] = []          # step indices where rotations occurred
    tt_phase_durations: list[tuple[float, float]] = []   # (cum_angle_deg, elapsed_s)

    # Time-based mode fields
    tt_current_rot   = 0.0        # cumulative rotation applied [degrees]
    tt_event_steps: list[int] = []  # pre-computed step indices for each rotation

    if tt_enabled:
        _tt_phases_legacy = tt_cfg.get("phases", [])
        if _tt_phases_legacy:
            # ── Legacy phases-based mode (backward compat) ───────────────────────
            # Convert phases to time-based events by equally spacing them.
            # Only uses angle_deg from each phase; ignores until_phi_mean.
            _tt_angles = [float(ph["angle_deg"]) for ph in _tt_phases_legacy[1:]]
            _n_evts    = len(_tt_angles)
            _interval  = max(1, round(n_steps / max(_n_evts, 1)))
            tt_event_steps = [_interval * (i + 1) for i in range(_n_evts)]
            _tt_rotation_deg_seq = _tt_angles   # list of absolute angles → convert to deltas
            _tt_delta_seq = [_tt_angles[0]]
            for _i in range(1, len(_tt_angles)):
                _tt_delta_seq.append(_tt_angles[_i] - _tt_angles[_i - 1])
            tt_rotation_deg_list: list[float] = _tt_delta_seq
            print(f"  [turntable] legacy-phases → time-based: {len(_n_evts)} events "
                  f"every {_interval*dt:.1f}s")
        else:
            # ── New time-based mode ───────────────────────────────────────────────
            _rot_deg   = float(tt_cfg.get("rotation_deg", 90.0))
            _n_rots    = int(tt_cfg.get("total_rotations", 1))
            # Default interval places rotations evenly WITHIN the run:
            #   interval = total_time / (total_rotations + 1)
            # so for 1 rotation in 360s: interval=180s → rotate at t=180s (midpoint)
            # for 4 rotations in 360s: interval=72s → rotate at t=72, 144, 216, 288s
            # This avoids placing events at t=0 or t=total_time.
            # User can override with rotation_interval_s for explicit control.
            _default_interval_s = n_steps * dt / (_n_rots + 1)
            _interval_s = float(tt_cfg.get("rotation_interval_s", _default_interval_s))
            _interval_steps = max(1, round(_interval_s / dt))
            tt_event_steps = [_interval_steps * (i + 1) for i in range(_n_rots)]
            tt_rotation_deg_list = [_rot_deg] * _n_rots
            print(f"  [turntable] time-based: {_rot_deg}° every {_interval_s:.1f}s "
                  f"× {_n_rots} = {_rot_deg * _n_rots:.0f}° total")

    # ── Exposure-time optimizer setup ────────────────────────────────────────────
    opt_cfg = cfg.get("optimizer", {})
    opt_enabled = bool(opt_cfg.get("enabled", False))
    opt_phi_thresholds = sorted(float(v) for v in opt_cfg.get("phi_snapshots", [0.50, 0.75, 0.90, 0.95]))
    opt_temp_ceiling = float(opt_cfg.get("temp_ceiling_c", 250.0))
    opt_highlight_phi = float(opt_cfg.get("highlight_phi", 0.90))
    opt_remaining: set[float] = set(opt_phi_thresholds) if opt_enabled else set()
    opt_snapshots: dict[object, tuple] = {}   # key → (SimState copy, hist copy, time_s)
    opt_prev_snap: tuple | None = None
    opt_maxdens_saved = False

    # ── Animation snapshot setup (all runs + turntable events) ──────────────────
    rep_cfg = cfg.get("reporting", {}) if isinstance(cfg.get("reporting", {}), dict) else {}
    gif_max_frames = int(rep_cfg.get("gif_max_frames", 36))
    gif_min_frames = int(rep_cfg.get("gif_min_frames", 14))
    gif_step_targets = set(
        _pick_snapshot_steps(n_steps, max(gif_max_frames, 2), min_frames=max(gif_min_frames, 2))
    )
    gif_snapshots_all: list[dict] = []
    gif_snapshots_turntable: list[dict] = []
    gif_snapshots_all.append(
        _capture_animation_snapshot(
            step_idx=-1,
            time_s=0.0,
            cum_angle_deg=(tt_current_rot if tt_enabled else None),
            E_mag=E_mag,
            T=T,
            rho_rel=rho_rel,
            part_mask=part_mask,
        )
    )
    if tt_enabled:
        gif_snapshots_turntable.append(
            _capture_animation_snapshot(
                step_idx=-1,
                time_s=0.0,
                cum_angle_deg=tt_current_rot,
                E_mag=E_mag,
                T=T,
                rho_rel=rho_rel,
                part_mask=part_mask,
            )
        )

    for it in range(n_steps):
        # ── Turntable rotation event check (time-based) ──────────────────────────
        if tt_enabled and tt_event_steps and (it + 1) == tt_event_steps[0]:
            _delta_deg = tt_rotation_deg_list.pop(0)
            tt_event_steps.pop(0)
            tt_current_rot += _delta_deg
            tt_rotation_steps.append(it)
            t_now_tt = (it + 1) * dt
            # Re-rasterize part at new cumulative rotation (electrodes stay fixed)
            new_part_mask, new_doped_mask, _new_fill, new_part_id_mask = _build_rotated_part_mask(
                cfg["geometry"], tt_current_rot, x, y
            )
            # Transfer thermal/density state using SPATIAL ROTATION REMAPPING.
            # For each new part cell, find where it came from in the old frame
            # via the inverse rotation — preserves spatial structure (e.g. corner
            # hot-spots, face gradients) rather than erasing it with mean-fill.
            from scipy.ndimage import map_coordinates as _map_coords
            _theta_rad = np.radians(-_delta_deg)   # inverse rotation angle
            _cos_t, _sin_t = np.cos(_theta_rad), np.sin(_theta_rad)
            # Build source coordinate arrays for every cell in the rotated grid
            _X_g, _Y_g = np.meshgrid(x, y, indexing="xy")  # (ny, nx)
            _cx_g = float(x.mean())
            _cy_g = float(y.mean())
            _X_src = _cx_g + _cos_t * (_X_g - _cx_g) - _sin_t * (_Y_g - _cy_g)
            _Y_src = _cy_g + _sin_t * (_X_g - _cx_g) + _cos_t * (_Y_g - _cy_g)
            # Convert source coords to fractional grid indices
            _i_src = (_X_src - x[0]) / (x[1] - x[0])   # column index (float)
            _j_src = (_Y_src - y[0]) / (y[1] - y[0])   # row index (float)
            # Interpolate old fields at rotated source positions for new part cells.
            # Important: use mask-normalized interpolation so old powder (outside part)
            # does not dilute/erase the rotated part state.
            _coords = np.array([_j_src.ravel(), _i_src.ravel()])
            _old_part = np.asarray(part_mask, dtype=bool)
            _w = _map_coords(_old_part.astype(float), _coords, order=1, mode="constant", cval=0.0).reshape(T.shape)
            _w_nn = _w > 1e-8

            def _remap_field(
                _field: np.ndarray,
                _outside_fill: float,
            ) -> np.ndarray:
                _num = _map_coords(
                    np.where(_old_part, _field, 0.0), _coords, order=1, mode="constant", cval=0.0
                ).reshape(_field.shape)
                _nn = _map_coords(_field, _coords, order=0, mode="nearest").reshape(_field.shape)
                _out = np.full_like(_field, float(_outside_fill), dtype=float)
                _good = new_part_mask & _w_nn
                _bad = new_part_mask & (~_w_nn)
                _out[_good] = _num[_good] / _w[_good]
                _out[_bad] = _nn[_bad]
                return _out

            # Rotate the full thermal field with the geometry to preserve
            # thermal coherence frame-to-frame around the moving part.
            T_new = _map_coords(T, _coords, order=1, mode="nearest").reshape(T.shape)
            rho_new = _remap_field(rho_rel, 0.0)
            phi_new = _remap_field(phi, 0.0)
            T_new = np.nan_to_num(T_new, nan=ambient_c, posinf=temp_max_c, neginf=temp_min_c)
            T_new = np.clip(T_new, temp_min_c, temp_max_c)
            rho_new = np.clip(rho_new, 0.0, 1.0)
            phi_new = np.clip(phi_new, 0.0, 1.0)
            T, rho_rel, phi = T_new, rho_new, phi_new
            if x_cryst is not None:
                x_cryst = np.clip(_remap_field(x_cryst, 0.0), 0.0, 1.0)
            part_mask, doped_mask = new_part_mask, new_doped_mask
            part_id_mask = new_part_id_mask
            # Recompute sigma from rotated thermal/density fields for consistency.
            sigma[:, :] = sigma_v
            sigma[new_part_mask] = np.clip(
                sigma_d0 * (1.0 + sigma_temp_coeff * (T[new_part_mask] - sigma_ref_temp))
                * (1.0 + sigma_density_coeff * (rho_rel[new_part_mask] - rho_rel_init)),
                1e-4 * sigma_d0, 25.0 * sigma_d0,
            )
            tt_phase_durations.append((tt_current_rot, t_now_tt))
            # Re-solve EQS with new part geometry (electrodes unchanged)
            gamma, V, Ex, Ey, E_mag, Qrf = solve_electric_state(
                sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo,
                dx, dy, elec, x, y, power_factor, max_qrf
            )
            Qrf, p_doped_eff_w_per_m, qrf_scale_applied = enforce_generator_power(
                Qrf, doped_mask, dA, target_power_w_per_m, max_qrf
            )
            if zero_qrf_outside_doped:
                Qrf = np.where(doped_mask, Qrf, 0.0)
            print(f"  [turntable] rotated by {_delta_deg:+.1f}° → cumulative {tt_current_rot:.1f}° "
                  f"at t={t_now_tt:.1f}s  "
                  f"(φ̄={float(np.mean(phi[part_mask])):.3f}  ρ̄={float(np.mean(rho_rel[part_mask])):.3f})")
            gif_snapshots_turntable.append(
                _capture_animation_snapshot(
                    step_idx=it,
                    time_s=t_now_tt,
                    cum_angle_deg=tt_current_rot,
                    E_mag=E_mag,
                    T=T,
                    rho_rel=rho_rel,
                    part_mask=part_mask,
                )
            )

        if not fixed_qrf_mode and update_interval > 0 and it > 0 and (it % update_interval == 0):
            sigma_part = sigma_d0 * (1.0 + sigma_temp_coeff * (T - sigma_ref_temp)) * (1.0 + sigma_density_coeff * (rho_rel - rho_rel_init))
            sigma_part = np.nan_to_num(sigma_part, nan=sigma_d0, posinf=25.0 * sigma_d0, neginf=1e-4 * sigma_d0)
            sigma[:, :] = sigma_v
            sigma[part_mask] = np.clip(sigma_part[part_mask], 1e-4 * sigma_d0, 25.0 * sigma_d0)
            # Re-apply surface sigma model if active (surface cells keep sigma_d0
            # but interior cells are reduced, preserving shell-heating pattern).
            if _sigma_profile == "surface" and part_mask.any():
                sigma[_part_interior] = np.clip(
                    sigma_part[_part_interior], 1e-4 * _sig_int, 25.0 * _sig_int
                )
            T_keep = T.copy()
            rho_keep = rho_rel.copy()
            phi_keep = phi.copy()
            gamma, V, Ex, Ey, E_mag, Qrf = solve_electric_state(
                sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo, dx, dy, elec, x, y, power_factor, max_qrf
            )
            Qrf, p_doped_eff_w_per_m, qrf_scale_applied = enforce_generator_power(
                Qrf,
                doped_mask,
                dA,
                target_power_w_per_m,
                max_qrf,
            )
            if zero_qrf_outside_doped:
                Qrf = np.where(doped_mask, Qrf, 0.0)
            T = T_keep
            rho_rel = rho_keep
            phi = phi_keep

        p_conv_acc = 0.0
        p_qrf_acc = 0.0
        dens_rate_acc = 0.0
        kss_acc = 0.0
        kliq_acc = 0.0
        max_dt_raw_acc = 0.0
        max_dt_used_acc = 0.0
        frac_dt_clip_acc = 0.0
        x_cr_acc = 0.0
        for _ in range(n_substeps):
            T, rho_rel, phi, x_cryst_new, diag_step = thermal_step(
                T, rho_rel, Qrf, part_mask, expo, therm_params, x_cryst=x_cryst
            )
            if x_cryst_new is not None:
                x_cryst = x_cryst_new
            T = np.nan_to_num(T, nan=ambient_c, posinf=temp_max_c, neginf=temp_min_c)
            T = np.where(T < temp_min_c, temp_min_c, np.where(T > temp_max_c, temp_max_c, T))
            phi = np.nan_to_num(phi, nan=0.0, posinf=1.0, neginf=0.0)
            phi = np.where(phi < 0.0, 0.0, np.where(phi > 1.0, 1.0, phi))
            rho_rel = np.nan_to_num(rho_rel, nan=0.0, posinf=1.0, neginf=0.0)
            rho_rel = np.where(rho_rel < 0.0, 0.0, np.where(rho_rel > 1.0, 1.0, rho_rel))
            p_conv_acc += diag_step["p_conv_loss_w_per_m"]
            p_qrf_acc += diag_step["p_qrf_gen_w_per_m"]
            dens_rate_acc += diag_step["mean_dens_rate_part_per_s"]
            kss_acc += diag_step["mean_kdens_ss_part_per_s"]
            kliq_acc += diag_step["mean_kdens_liq_part_per_s"]
            max_dt_raw_acc = max(max_dt_raw_acc, float(diag_step.get("max_dT_raw_c", 0.0)))
            max_dt_used_acc = max(max_dt_used_acc, float(diag_step.get("max_dT_used_c", 0.0)))
            frac_dt_clip_acc += float(diag_step.get("frac_cells_dT_clipped", 0.0))
            x_cr_acc += float(diag_step.get("mean_x_cryst_part", 0.0))

        hist["time_s"].append((it + 1) * dt)
        hist["mean_T_part_c"].append(float(np.mean(T[part_mask])))
        hist["max_T_part_c"].append(float(np.max(T[part_mask])))
        t_part_now = T[part_mask]
        t_avg = float(np.mean(t_part_now))
        denom = max(t_avg - ambient_c, 1e-9)
        hist["ui_abs_part"].append(float(np.mean(np.abs(t_part_now - t_avg)) / denom))
        hist["ui_rms_part"].append(float(np.sqrt(np.mean((t_part_now - t_avg) ** 2)) / denom))
        hist["mean_phi_part"].append(float(np.mean(phi[part_mask])))
        hist["mean_rho_rel_part"].append(float(np.mean(rho_rel[part_mask])))
        p_doped = float(np.sum(Qrf[doped_mask]) * dA)
        hist["power_doped_W_per_m"].append(p_doped)
        hist["qrf_scale_applied"].append(float(qrf_scale_applied))
        hist["target_power_doped_W_per_m"].append(
            float(target_power_w_per_m) if target_power_w_per_m is not None else p_doped
        )
        prev_energy = hist["energy_doped_J_per_m"][-1] if hist["energy_doped_J_per_m"] else 0.0
        hist["energy_doped_J_per_m"].append(prev_energy + p_doped * dt)
        p_conv_mean = p_conv_acc / float(n_substeps)
        hist["power_conv_loss_W_per_m"].append(p_conv_mean)
        prev_conv_energy = hist["energy_conv_loss_J_per_m"][-1] if hist["energy_conv_loss_J_per_m"] else 0.0
        hist["energy_conv_loss_J_per_m"].append(prev_conv_energy + p_conv_mean * dt)
        # Energy bookkeeping (stored sensible + latent energy in the part).
        # Keep this in a helper to reduce in-loop local expression complexity.
        e_stored = part_energy_per_depth(T, rho_rel, part_mask, therm_params)
        hist["energy_stored_part_J_per_m"].append(e_stored)
        e_in = hist["energy_doped_J_per_m"][-1]
        e_out = hist["energy_conv_loss_J_per_m"][-1]
        hist["energy_balance_residual_J_per_m"].append(e_in - e_out - e_stored)
        hist["mean_dens_rate_part_per_s"].append(dens_rate_acc / float(n_substeps))
        hist["mean_kdens_ss_part_per_s"].append(kss_acc / float(n_substeps))
        hist["mean_kdens_liq_part_per_s"].append(kliq_acc / float(n_substeps))
        hist["mean_x_cryst_part"].append(x_cr_acc / float(n_substeps))
        hist["max_dT_raw_c"].append(max_dt_raw_acc)
        hist["max_dT_used_c"].append(max_dt_used_acc)
        hist["frac_cells_dT_clipped"].append(frac_dt_clip_acc / float(n_substeps))

        if it in gif_step_targets:
            _snap = _capture_animation_snapshot(
                step_idx=it,
                time_s=(it + 1) * dt,
                cum_angle_deg=(tt_current_rot if tt_enabled else None),
                E_mag=E_mag,
                T=T,
                rho_rel=rho_rel,
                part_mask=part_mask,
            )
            gif_snapshots_all.append(_snap)
            if tt_enabled:
                gif_snapshots_turntable.append(_snap)

        # ── Optimizer snapshot logic ─────────────────────────────────────────────
        if opt_enabled:
            phi_now_opt = float(np.mean(phi[part_mask])) if part_mask.any() else 0.0
            T_max_now_opt = float(np.max(T[part_mask])) if part_mask.any() else 0.0
            t_now_s = (it + 1) * dt

            # φ threshold snapshots
            for thr in list(opt_remaining):
                if phi_now_opt >= thr:
                    _state_snap = _copy_state(SimState(
                        x=x, y=y, part_mask=part_mask, doped_mask=doped_mask,
                        elec_hi=elec_hi, elec_lo=elec_lo,
                        V=V, Ex=Ex, Ey=Ey, E_mag=E_mag, Qrf=Qrf,
                        T=T, phi=phi, rho_rel=rho_rel,
                        part_id_mask=part_id_mask,
                    ))
                    opt_snapshots[thr] = (
                        _state_snap,
                        {k: list(v) for k, v in hist.items()},
                        t_now_s,
                    )
                    opt_remaining.discard(thr)
                    print(f"  [optimizer] Snapshot @ φ_mean≥{thr:.2f}: t={t_now_s:.1f}s "
                          f"T̄={float(np.mean(T[part_mask])):.1f}°C  "
                          f"ρ_rel={float(np.mean(rho_rel[part_mask])):.3f}")

            # Track state just before T_max exceeds ceiling (max density snapshot)
            if not opt_maxdens_saved:
                if T_max_now_opt >= opt_temp_ceiling:
                    if opt_prev_snap is not None:
                        opt_snapshots["max_density"] = opt_prev_snap
                        print(f"  [optimizer] Max-density snapshot saved "
                              f"(T_max crossed {opt_temp_ceiling:.0f}°C at t={t_now_s:.1f}s)")
                    opt_maxdens_saved = True
                else:
                    _state_prev = _copy_state(SimState(
                        x=x, y=y, part_mask=part_mask, doped_mask=doped_mask,
                        elec_hi=elec_hi, elec_lo=elec_lo,
                        V=V, Ex=Ex, Ey=Ey, E_mag=E_mag, Qrf=Qrf,
                        T=T, phi=phi, rho_rel=rho_rel,
                        part_id_mask=part_id_mask,
                    ))
                    opt_prev_snap = (
                        _state_prev,
                        {k: list(v) for k, v in hist.items()},
                        t_now_s,
                    )

    if not fixed_qrf_mode:
        gamma, V, Ex, Ey, E_mag, Qrf = solve_electric_state(
            sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo, dx, dy, elec, x, y, power_factor, max_qrf
        )
        Qrf, p_doped_eff_w_per_m, qrf_scale_applied = enforce_generator_power(
            Qrf,
            doped_mask,
            dA,
            target_power_w_per_m,
            max_qrf,
        )
        if zero_qrf_outside_doped:
            Qrf = np.where(doped_mask, Qrf, 0.0)

    state = SimState(
        x=x,
        y=y,
        part_mask=part_mask,
        doped_mask=doped_mask,
        elec_hi=elec_hi,
        elec_lo=elec_lo,
        V=V,
        Ex=Ex,
        Ey=Ey,
        E_mag=E_mag,
        Qrf=Qrf,
        T=T,
        phi=phi,
        rho_rel=rho_rel,
        x_cryst=x_cryst,
        part_id_mask=part_id_mask,
    )

    t_part = T[part_mask]
    qrf_all = np.asarray(Qrf, dtype=float)
    qrf_part = qrf_all[part_mask]
    part_stats: list[dict[str, float]] = []
    part_ids = sorted(int(v) for v in np.unique(part_id_mask) if int(v) > 0)
    for pid in part_ids:
        m = part_id_mask == pid
        if not np.any(m):
            continue
        t_i = T[m]
        rho_i = rho_rel[m]
        q_i = qrf_all[m]
        part_stats.append(
            {
                "part_id": int(pid),
                "cell_count": int(np.sum(m)),
                "mean_T_c": float(np.mean(t_i)),
                "max_T_c": float(np.max(t_i)),
                "mean_rho_rel": float(np.mean(rho_i)),
                "min_rho_rel": float(np.min(rho_i)),
                "std_rho_rel": float(np.std(rho_i)),
                "mean_qrf_w_per_m3": float(np.mean(q_i)),
                "p10_qrf_w_per_m3": float(np.percentile(q_i, 10.0)),
            }
        )
    part_mean_rhos = [s["mean_rho_rel"] for s in part_stats]
    part_min_rhos = [s["min_rho_rel"] for s in part_stats]
    if phase_user.model in {"comsol_heaviside", "heaviside"}:
        trans_lo_c = float(phase_user.t_pc_c - 0.5 * phase_user.dt_pc_c)
        trans_hi_c = float(phase_user.t_pc_c + 0.5 * phase_user.dt_pc_c)
        melt_ref_c = float(phase_user.t_pc_c)
    else:
        trans_lo_c = float(phase_user.solidus_c)
        trans_hi_c = float(phase_user.liquidus_c)
        melt_ref_c = float(phase_user.liquidus_c)

    summary = {
        "grid": {"nx": int(nx), "ny": int(ny), "dx_m": dx, "dy_m": dy},
        "thermal_substeps_per_step": int(n_substeps),
        "thermal_dt_sub_s": float(dt_sub),
        "integrated_power_doped_W_per_m": float(np.sum(Qrf[doped_mask]) * dA),
        "target_power_doped_W_per_m": float(target_power_w_per_m) if target_power_w_per_m is not None else None,
        "qrf_scale_applied_final": float(qrf_scale_applied),
        "generator_power_w": float(gen_power_w) if enforce_gen_power else None,
        "generator_transfer_efficiency": float(gen_eff) if enforce_gen_power else None,
        "effective_depth_m": float(effective_depth_m) if enforce_gen_power else None,
        "energy_doped_total_J_per_m": float(hist["energy_doped_J_per_m"][-1] if hist["energy_doped_J_per_m"] else 0.0),
        "mean_T_part_final_c": float(np.mean(t_part)),
        "max_T_part_final_c": float(np.max(t_part)),
        "p95_T_part_final_c": float(np.percentile(t_part, 95.0)),
        "p99_T_part_final_c": float(np.percentile(t_part, 99.0)),
        "phase_transition_lo_c": float(trans_lo_c),
        "phase_transition_hi_c": float(trans_hi_c),
        "melt_reference_c": float(melt_ref_c),
        "frac_part_ge_transition_lo": float(np.mean(t_part >= trans_lo_c)),
        "frac_part_ge_transition_hi": float(np.mean(t_part >= trans_hi_c)),
        "frac_part_ge_melt_ref": float(np.mean(t_part >= melt_ref_c)),
        "qrf_global_min_w_per_m3": float(np.min(qrf_all)),
        "qrf_global_mean_w_per_m3": float(np.mean(qrf_all)),
        "qrf_global_max_w_per_m3": float(np.max(qrf_all)),
        "qrf_part_min_w_per_m3": float(np.min(qrf_part)),
        "qrf_part_mean_w_per_m3": float(np.mean(qrf_part)),
        "qrf_part_p95_w_per_m3": float(np.percentile(qrf_part, 95.0)),
        "qrf_part_p99_w_per_m3": float(np.percentile(qrf_part, 99.0)),
        "qrf_part_max_w_per_m3": float(np.max(qrf_part)),
        "frac_part_at_qrf_cap_final": float(np.mean(qrf_part >= (0.999 * max_qrf))),
        "frac_part_at_temp_cap": float(np.mean(t_part >= (temp_max_c - 1e-9))),
        "ui_abs_part_final": float(hist["ui_abs_part"][-1] if hist["ui_abs_part"] else 0.0),
        "ui_rms_part_final": float(hist["ui_rms_part"][-1] if hist["ui_rms_part"] else 0.0),
        "mean_phi_part_final": float(np.mean(phi[part_mask])),
        "frac_part_phi_gt_0p1_final": float(np.mean(phi[part_mask] > 0.1)),
        "frac_part_phi_gt_0p5_final": float(np.mean(phi[part_mask] > 0.5)),
        "mean_rho_rel_part_final": float(np.mean(rho_rel[part_mask])),
        "energy_conv_loss_total_J_per_m": float(hist["energy_conv_loss_J_per_m"][-1] if hist["energy_conv_loss_J_per_m"] else 0.0),
        "energy_stored_part_final_J_per_m": float(hist["energy_stored_part_J_per_m"][-1] if hist["energy_stored_part_J_per_m"] else 0.0),
        "energy_balance_residual_final_J_per_m": float(
            hist["energy_balance_residual_J_per_m"][-1] if hist["energy_balance_residual_J_per_m"] else 0.0
        ),
        "mean_dens_rate_part_final_per_s": float(
            hist["mean_dens_rate_part_per_s"][-1] if hist["mean_dens_rate_part_per_s"] else 0.0
        ),
        "mean_kdens_ss_part_final_per_s": float(
            hist["mean_kdens_ss_part_per_s"][-1] if hist["mean_kdens_ss_part_per_s"] else 0.0
        ),
        "mean_kdens_liq_part_final_per_s": float(
            hist["mean_kdens_liq_part_per_s"][-1] if hist["mean_kdens_liq_part_per_s"] else 0.0
        ),
        "mean_x_cryst_part_final": float(hist["mean_x_cryst_part"][-1] if hist["mean_x_cryst_part"] else 0.0),
        "max_dT_raw_final_c": float(hist["max_dT_raw_c"][-1] if hist["max_dT_raw_c"] else 0.0),
        "max_dT_used_final_c": float(hist["max_dT_used_c"][-1] if hist["max_dT_used_c"] else 0.0),
        "frac_cells_dT_clipped_final": float(
            hist["frac_cells_dT_clipped"][-1] if hist["frac_cells_dT_clipped"] else 0.0
        ),
        "frac_cells_dT_clipped_mean": float(np.mean(hist["frac_cells_dT_clipped"])) if hist["frac_cells_dT_clipped"] else 0.0,
        "time_final_s": float(hist["time_s"][-1] if hist["time_s"] else 0.0),
        "model_family": physics_model_family,
        "parameter_source": parameter_source,
        "literature_refs": [str(v) for v in literature_refs],
        "calibration_version": calibration_version,
        "ab_bucket_id": ab_bucket_id,
        "provenance_tag": provenance_tag,
        "part_stats": part_stats,
        "parts_count": int(len(part_stats)),
        "inter_part_mean_rho_std": float(np.std(part_mean_rhos)) if part_mean_rhos else 0.0,
        "inter_part_mean_rho_minmax_ratio": float(min(part_mean_rhos) / max(max(part_mean_rhos), 1e-9)) if part_mean_rhos else 0.0,
        "inter_part_min_rho_std": float(np.std(part_min_rhos)) if part_min_rhos else 0.0,
        "antennae_enabled": bool(antennae_resolved.get("enabled", False)),
        "antennae_count": int(len(antennae_resolved.get("instances", []))),
        "antennae_instances": antennae_resolved.get("instances", []),
    }
    if provenance_table is not None:
        summary["provenance_file"] = str(
            Path(physics_cfg.get("provenance_file", "configs/experimental_pa12_provenance.yaml"))
        )
        summary["provenance_version"] = str(provenance_table.get("version", "unknown"))
    if dsc_profile is not None:
        summary["dsc_profile_file"] = str(Path(physics_cfg.get("dsc_profile_file", "configs/experimental_pa12_dsc_profile.yaml")))
        summary["dsc_profile_version"] = str(dsc_profile.get("version", "unknown"))
        summary["dsc_profile_melt_onset_c"] = float(dsc_profile["melt_onset_c"])
        summary["dsc_profile_melt_peak_c"] = float(dsc_profile["melt_peak_c"])
        summary["dsc_profile_melt_end_c"] = float(dsc_profile["melt_end_c"])
        summary["dsc_profile_lat_heat_j_per_kg"] = float(dsc_profile["lat_heat_j_per_kg"])

    # ── Turntable post-loop: add rotation schedule to summary ────────────────────
    if tt_enabled and tt_rotation_steps:
        summary["turntable_rotations"] = [
            {"cum_angle_deg": round(a, 2), "event_time_s": round(e, 3), "event_time_min": round(e / 60.0, 4)}
            for a, e in tt_phase_durations
        ]
        summary["turntable_total_rotation_deg"] = float(tt_current_rot)
        print(f"  [turntable] ── Rotation Schedule {'─'*25}")
        for cum_ang, evt_s in tt_phase_durations:
            mm, ss = divmod(evt_s, 60)
            print(f"    cum {cum_ang:+.1f}°  at t={evt_s:7.1f} s  ({int(mm)} min {ss:.0f} s)")
        print(f"  [turntable] total rotation: {tt_current_rot:.1f}°  over {n_steps * dt:.0f} s")

    # ── Build optimizer output bundle ─────────────────────────────────────────────
    opt_data = {
        "enabled": opt_enabled,
        "phi_thresholds": opt_phi_thresholds,
        "temp_ceiling_c": opt_temp_ceiling,
        "highlight_phi": opt_highlight_phi,
        "snapshots": opt_snapshots,
        "turntable_rotation_steps": tt_rotation_steps,
        "gif_snapshots_all": gif_snapshots_all,
        "gif_snapshots_turntable": gif_snapshots_turntable,
        "time_final_s": float(hist["time_s"][-1] if hist["time_s"] else 0.0),
    }

    return state, summary, hist, tt_rotation_steps, opt_data


def generate_optimizer_report(
    cfg: dict,
    opt_data: dict,
    hist_full: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Generate optimizer_report.png: time series with optimal-time annotations
    plus 2 rows of field snapshot panels (melt criterion + max-density criterion)."""
    if not opt_data.get("enabled", False):
        return
    snaps = opt_data.get("snapshots", {})
    if not snaps:
        print("  [optimizer] No snapshots captured — skipping report.")
        return

    highlight_phi  = float(opt_data.get("highlight_phi", 0.90))
    temp_ceiling   = float(opt_data.get("temp_ceiling_c", 250.0))
    tt_rot_steps   = opt_data.get("turntable_rotation_steps", [])

    rep   = cfg.get("reporting", {})
    t_arr = np.array(hist_full["time_s"], dtype=float)
    dt    = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 0.5

    # ── Identify the two key snapshots ──────────────────────────────────────────
    melt_key = None
    for thr in sorted(snaps.keys(), key=lambda k: (isinstance(k, str), k)):
        if isinstance(thr, float) and thr >= highlight_phi * 0.999:
            melt_key = thr
            break
    if melt_key is None:
        avail_float = [k for k in snaps if isinstance(k, float)]
        if avail_float:
            melt_key = min(avail_float, key=lambda k: abs(k - highlight_phi))

    maxd_key = "max_density" if "max_density" in snaps else None

    def _snap_fields(key):
        if key is None or key not in snaps:
            return None
        st, _h, ts = snaps[key]
        sig_f, sig_m = 1.2, 1.0
        T_d    = gaussian_filter(st.T.astype(float),         sigma=sig_f)
        phi_d  = gaussian_filter(st.phi.astype(float),       sigma=sig_f)
        rho_d  = gaussian_filter(st.rho_rel.astype(float),   sigma=sig_f)
        pm_d   = gaussian_filter(st.part_mask.astype(float), sigma=sig_m)
        phi_mean = float(np.mean(st.phi[st.part_mask]))    if st.part_mask.any() else 0.0
        rho_mean = float(np.mean(st.rho_rel[st.part_mask])) if st.part_mask.any() else 0.0
        T_mean   = float(np.mean(st.T[st.part_mask]))      if st.part_mask.any() else 0.0
        T_max_v  = float(np.max(st.T[st.part_mask]))       if st.part_mask.any() else 0.0
        mm_s, ss_s = divmod(ts, 60)
        label = (f"t={ts:.0f}s ({int(mm_s)}m{ss_s:.0f}s)  "
                 f"T̄={T_mean:.0f}°C  T_max={T_max_v:.0f}°C\n"
                 f"φ={phi_mean:.3f}  ρ_rel={rho_mean:.3f}")
        return {
            "state": st, "T_d": T_d, "phi_d": phi_d, "rho_d": rho_d,
            "pm_d": pm_d, "t_s": ts, "label": label, "T_max": T_max_v,
        }

    snap_melt = _snap_fields(melt_key)
    snap_maxd = _snap_fields(maxd_key)

    # ── Recommendation logic ─────────────────────────────────────────────────────
    if snap_melt is None and snap_maxd is None:
        recommendation = None
        rec_reason = ""
    elif snap_melt is None:
        recommendation = "max_density"
        rec_reason = "Melt snapshot unavailable — max-density criterion is the only option"
    elif snap_maxd is None:
        recommendation = "melt"
        rec_reason = (f"T_max never reached {temp_ceiling:.0f}°C during run — "
                      f"melt criterion is the practical limit")
    else:
        delta_s = snap_maxd["t_s"] - snap_melt["t_s"]
        if delta_s >= 30.0:
            recommendation = "max_density"
            rec_reason = (f"+{delta_s:.0f}s of additional densification available "
                          f"before T_max exceeds {temp_ceiling:.0f}°C — safe to extend")
        else:
            recommendation = "melt"
            rec_reason = (f"Only {delta_s:.0f}s margin before thermal ceiling — "
                          f"stop at melt criterion for safety")

    rec_snap = (snap_melt if recommendation == "melt" else snap_maxd) if recommendation else None

    # ── stdout recommendations ───────────────────────────────────────────────────
    print(f"  [optimizer] ── Exposure Time Recommendations {'─'*18}")
    if snap_melt is not None:
        mm, ss = divmod(snap_melt["t_s"], 60)
        melt_tag = " ◆ Melt criterion" + (" ★" if recommendation == "melt" else "")
        print(f"    {melt_tag}  (φ_mean ≥ {melt_key:.2f}):  "
              f"{snap_melt['t_s']:.1f} s  ({int(mm)} min {ss:.0f} s)")
    if snap_maxd is not None:
        mm, ss = divmod(snap_maxd["t_s"], 60)
        maxd_tag = " ▲ Max-density ceiling" + (" ★" if recommendation == "max_density" else "")
        print(f"    {maxd_tag}  (T_max < {temp_ceiling:.0f}°C):  "
              f"{snap_maxd['t_s']:.1f} s  ({int(mm)} min {ss:.0f} s)")
    if rec_snap is not None:
        mm_r, ss_r = divmod(rec_snap["t_s"], 60)
        print(f"  [optimizer] ★ Recommendation: {recommendation.replace('_',' ')} criterion — "
              f"{rec_snap['t_s']:.1f} s ({int(mm_r)} min {ss_r:.0f} s)")
        print(f"              {rec_reason}")

    # ── Build figure ─────────────────────────────────────────────────────────────
    n_snap_rows = int(snap_melt is not None) + int(snap_maxd is not None)
    fig_h = 4.8 + 3.4 * n_snap_rows
    fig, axes = plt.subplots(
        1 + n_snap_rows, 3,
        figsize=(13.0, fig_h),
        gridspec_kw={"height_ratios": [2.4] + [1.0] * n_snap_rows},
    )

    # Merge top row into single axes for time series
    for a in axes[0, 1:]:
        a.remove()
    ax_ts_ax = fig.add_subplot(axes[0, 0].get_gridspec()[0, :])
    axes[0, 0].remove()

    # ── Time series curves ────────────────────────────────────────────────────────
    color_T   = "#d62728"
    color_phi = "#1f77b4"
    color_rho = "#2ca02c"
    ln1, = ax_ts_ax.plot(t_arr, hist_full["mean_T_part_c"], color=color_T, lw=1.8, label="T̄ part [°C]")
    ln2, = ax_ts_ax.plot(t_arr, hist_full["max_T_part_c"],  color=color_T, lw=1.0, ls="--", label="T_max [°C]")
    ax_ts_ax.axhline(temp_ceiling, color=color_T, lw=0.8, ls=":", alpha=0.55,
                     label=f"T ceiling ({temp_ceiling:.0f}°C)")
    ax_ts_ax.set_ylabel("Temperature [°C]", color=color_T)
    ax_ts_ax.tick_params(axis="y", colors=color_T)

    ax_ts2 = ax_ts_ax.twinx()
    ln3, = ax_ts2.plot(t_arr, hist_full["mean_phi_part"],     color=color_phi, lw=1.6, label="φ_mean")
    ln4, = ax_ts2.plot(t_arr, hist_full["mean_rho_rel_part"], color=color_rho, lw=1.6, label="ρ_rel_mean")
    ax_ts2.set_ylabel("Melt fraction / Rel. density", color="#555555")
    ax_ts2.set_ylim(0.0, 1.1)

    # ── Vertical event lines (Fix 2: labels at BOTTOM with white bbox) ───────────
    y_bot = ax_ts_ax.get_ylim()[0]

    # Non-highlight φ threshold lines (minor, gray)
    for thr, (_, _, t_thr) in sorted(
        [(k, v) for k, v in snaps.items()
         if isinstance(k, float) and abs(k - highlight_phi) > 0.001],
        key=lambda kv: kv[0],
    ):
        ax_ts_ax.axvline(t_thr, color="#bbbbbb", lw=0.8, ls="--", alpha=0.7, zorder=2)
        ax_ts_ax.text(
            t_thr, y_bot, f"φ≥{thr:.2f}\n{t_thr:.0f}s",
            fontsize=6.5, ha="center", va="bottom", color="#888888", rotation=90,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=3,
        )

    # Melt criterion line (gold, ◆ label)
    if snap_melt is not None:
        t_m = snap_melt["t_s"]
        is_rec_m = (recommendation == "melt")
        ax_ts_ax.axvline(t_m, color="#FFD700", lw=2.0, ls="--", alpha=0.95, zorder=4)
        if is_rec_m:
            ax_ts_ax.axvspan(t_m - dt, t_m + dt, alpha=0.12, color="#FFD700", zorder=1)
        ax_ts_ax.text(
            t_m, y_bot,
            f"◆ Melt  {t_m:.0f}s" + (" ★" if is_rec_m else ""),
            fontsize=7, ha="center", va="bottom", color="#b8860b", rotation=90,
            fontweight="bold" if is_rec_m else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FFD700",
                      alpha=0.9, lw=1.0 if is_rec_m else 0.0),
            zorder=5,
        )

    # Max-density line (red, ▲ label)
    if snap_maxd is not None:
        t_d = snap_maxd["t_s"]
        is_rec_d = (recommendation == "max_density")
        ax_ts_ax.axvline(t_d, color="#e84040", lw=2.0, ls="-.", alpha=0.95, zorder=4)
        if is_rec_d:
            ax_ts_ax.axvspan(t_d - dt, t_d + dt, alpha=0.10, color="#e84040", zorder=1)
        ax_ts_ax.text(
            t_d, y_bot,
            f"▲ Max ρ  {t_d:.0f}s" + (" ★" if is_rec_d else ""),
            fontsize=7, ha="center", va="bottom", color="#b00000", rotation=90,
            fontweight="bold" if is_rec_d else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e84040",
                      alpha=0.9, lw=1.0 if is_rec_d else 0.0),
            zorder=5,
        )

    # Turntable rotation events
    for rot_step in tt_rot_steps:
        t_rot = rot_step * dt
        ax_ts_ax.axvline(t_rot, color="#9467bd", lw=0.9, ls=":", alpha=0.7, zorder=2)
        ax_ts_ax.text(
            t_rot, y_bot, f"↺ {t_rot:.0f}s",
            fontsize=6.5, ha="center", va="bottom", color="#7b4fa0", rotation=90,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            zorder=3,
        )

    # ── Recommendation annotation box on figure ──────────────────────────────────
    if rec_snap is not None:
        mm_r, ss_r = divmod(rec_snap["t_s"], 60)
        rec_label_str = "Melt criterion" if recommendation == "melt" else "Max-density ceiling"
        ax_ts_ax.annotate(
            f"★ Recommended: {rec_label_str}\n"
            f"   {rec_snap['t_s']:.0f} s  ({int(mm_r)} min {ss_r:.0f} s)\n"
            f"   {rec_reason}",
            xy=(rec_snap["t_s"], ax_ts_ax.get_ylim()[1]),
            xycoords="data",
            xytext=(0.97, 0.97), textcoords="axes fraction",
            fontsize=8, fontweight="bold", color="#1a1a1a",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fffde7", ec="#f0c040", lw=1.4),
            arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1.2),
        )

    ax_ts_ax.set_xlabel("time [s]")
    ax_ts_ax.set_title("Exposure Time Optimizer — Time Series", fontsize=11, fontweight="bold")
    ax_ts_ax.grid(alpha=0.18)
    all_lines = [ln1, ln2, ln3, ln4]
    ax_ts_ax.legend(all_lines, [l.get_label() for l in all_lines],
                    loc="upper left", fontsize=8, framealpha=0.85)

    # ── Dynamic T colorbar: scaled to actual max across snapshots (Fix 4c) ────────
    all_T_maxes = []
    for key in [melt_key, maxd_key]:
        if key is not None and key in snaps:
            st_snap, _, _ = snaps[key]
            if st_snap.part_mask.any():
                all_T_maxes.append(float(np.max(st_snap.T[st_snap.part_mask])))
    T_vmin = float(cfg.get("thermal", {}).get("ambient_c", 23.0))
    if all_T_maxes:
        T_vmax = math.ceil(max(all_T_maxes) * 1.02 / 10.0) * 10.0  # round up to nearest 10°C
    else:
        T_vmax = float(rep.get("temp_cbar_max_c", 220.0))

    # ── Snapshot rows ─────────────────────────────────────────────────────────────
    snap_row_data = []
    if snap_melt is not None:
        star_m = " ★" if recommendation == "melt" else ""
        snap_row_data.append((snap_melt,
                               f"◆ Melt criterion{star_m}  (φ_mean ≥ {melt_key:.2f})"))
    if snap_maxd is not None:
        star_d = " ★" if recommendation == "max_density" else ""
        snap_row_data.append((snap_maxd,
                               f"▲ Max-density ceiling{star_d}  (T_max < {temp_ceiling:.0f}°C)"))

    for row_i, (snap, row_title) in enumerate(snap_row_data):
        row = row_i + 1
        st  = snap["state"]
        ext = [st.x[0], st.x[-1], st.y[0], st.y[-1]]
        kw  = dict(extent=ext, origin="lower", interpolation="bilinear", aspect="equal")

        im_T = axes[row, 0].imshow(snap["T_d"],   cmap="jet",    vmin=T_vmin, vmax=T_vmax, **kw)
        axes[row, 0].contour(st.x, st.y, snap["pm_d"], levels=[0.5], colors=["w"], linewidths=0.9)
        axes[row, 0].set_title(f"T [°C]  |  {snap['label']}", fontsize=7)
        plt.colorbar(im_T, ax=axes[row, 0], shrink=0.82)

        im_phi = axes[row, 1].imshow(snap["phi_d"], cmap="plasma", vmin=0.0, vmax=1.0, **kw)
        axes[row, 1].contour(st.x, st.y, snap["pm_d"], levels=[0.5], colors=["w"], linewidths=0.9)
        axes[row, 1].set_title(f"Melt fraction φ  |  {snap['label']}", fontsize=7)
        plt.colorbar(im_phi, ax=axes[row, 1], shrink=0.82)

        im_rho = axes[row, 2].imshow(snap["rho_d"], cmap="viridis", vmin=0.5, vmax=1.0, **kw)
        axes[row, 2].contour(st.x, st.y, snap["pm_d"], levels=[0.5], colors=["w"], linewidths=0.9)
        axes[row, 2].set_title(f"Rel. density ρ  |  {snap['label']}", fontsize=7)
        plt.colorbar(im_rho, ax=axes[row, 2], shrink=0.82)

        # Row label on y-axis of first column
        axes[row, 0].set_ylabel(row_title, fontsize=8, fontweight="bold")

    fig.tight_layout()
    save_path = output_dir / "optimizer_report.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [optimizer] Report saved: {save_path}")


def _stack_limits(snaps: list[dict], key: str) -> tuple[float, float]:
    vals = [np.asarray(s[key], dtype=float) for s in snaps if key in s]
    if not vals:
        return 0.0, 1.0
    finite_chunks = [v[np.isfinite(v)] for v in vals if np.isfinite(v).any()]
    if not finite_chunks:
        return 0.0, 1.0
    arr = np.concatenate(finite_chunks)
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if math.isclose(vmin, vmax):
        pad = max(1e-9, abs(vmin) * 1e-6)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _density_gif_limits(snaps: list[dict], rho_rel_initial: float) -> tuple[float, float]:
    part_vals: list[np.ndarray] = []
    for s in snaps:
        if "rho_rel" not in s or "part_mask" not in s:
            continue
        rho = np.asarray(s["rho_rel"], dtype=float)
        pm = np.asarray(s["part_mask"], dtype=bool)
        vals = rho[pm]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            part_vals.append(vals)
    if not part_vals:
        return float(rho_rel_initial), min(1.0, float(rho_rel_initial) + 0.05)
    allv = np.concatenate(part_vals)
    vmin = float(rho_rel_initial)
    vmax = float(np.percentile(allv, 99.5))
    vmax = min(1.0, max(vmax, vmin + 0.03))
    return vmin, vmax


def _render_animation_frame(
    *,
    snap: dict,
    x: np.ndarray,
    y: np.ndarray,
    key: str,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    outside_fill: float | None = None,
    apply_part_mask: bool = False,
) -> Image.Image:
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.8))
    pm_bool = np.asarray(snap["part_mask"], dtype=bool)
    data_raw = np.asarray(snap[key], dtype=float)
    if apply_part_mask:
        if outside_fill is None:
            outside_fill = 0.0
        data_raw = np.where(pm_bool, data_raw, float(outside_fill))
    data = gaussian_filter(data_raw, sigma=1.0)
    pm = gaussian_filter(pm_bool.astype(float), sigma=0.8)
    im = ax.imshow(
        data,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.contour(x, y, pm, levels=[0.5], colors=["w"], linewidths=0.9)
    t_s = float(snap.get("time_s", 0.0))
    rot = snap.get("cum_angle_deg", None)
    meta = f"t={t_s:.1f} s"
    if rot is not None:
        meta += f"  |  cum rot={float(rot):.1f}°"
    ax.set_title(f"{title}\n{meta}", fontsize=9)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, shrink=0.84)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("P", palette=Image.ADAPTIVE).copy()
    buf.close()
    return img


def _write_animation_gif(
    *,
    snaps: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    key: str,
    title: str,
    cmap: str,
    out_path: Path,
    frame_duration_s: float,
    outside_fill: float | None = None,
    apply_part_mask: bool = False,
    fixed_limits: tuple[float, float] | None = None,
) -> None:
    if len(snaps) < 2:
        return
    if fixed_limits is None:
        vmin, vmax = _stack_limits(snaps, key)
    else:
        vmin, vmax = float(fixed_limits[0]), float(fixed_limits[1])
    frames = [
        _render_animation_frame(
            snap=s, x=x, y=y, key=key, title=title, cmap=cmap, vmin=vmin, vmax=vmax,
            outside_fill=outside_fill,
            apply_part_mask=apply_part_mask,
        )
        for s in snaps
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(40, int(frame_duration_s * 1000.0)),
        loop=0,
        disposal=2,  # clear previous frame; prevents ghost artifacts from GIF disposal
        optimize=False,
    )
    print(f"Saved GIF: {out_path}")


def _dedupe_and_sort_snaps(snaps: list[dict]) -> list[dict]:
    by_time: dict[float, dict] = {}
    for s in snaps:
        t = float(s.get("time_s", 0.0))
        by_time[t] = s
    return [by_time[t] for t in sorted(by_time.keys())]


def _generate_animation_gifs(
    *,
    cfg: dict,
    state: SimState,
    output_dir: Path,
    opt_data: dict | None,
) -> None:
    if Image is None:
        print("  [gif] Pillow not available; skipping GIF generation.")
        return
    if not opt_data:
        return
    rep = cfg.get("reporting", {}) if isinstance(cfg.get("reporting", {}), dict) else {}
    tt_cfg = cfg.get("turntable", {}) if isinstance(cfg.get("turntable", {}), dict) else {}
    is_turntable_run = bool(tt_cfg.get("enabled", False))
    frame_duration_s = float(rep.get("gif_frame_duration_s", 0.6))
    ambient_c = float(cfg.get("thermal", {}).get("ambient_c", 23.0))
    rho_rel_initial = float(cfg.get("densification", {}).get("rho_rel_initial", 0.55))

    snaps_all = list(opt_data.get("gif_snapshots_all", []))
    snaps_tt = list(opt_data.get("gif_snapshots_turntable", []))

    # Ensure final state is present for both collections.
    final_snap = _capture_animation_snapshot(
        step_idx=-2,
        time_s=float(opt_data.get("time_final_s", 0.0)),
        cum_angle_deg=snaps_tt[-1].get("cum_angle_deg") if snaps_tt else None,
        E_mag=state.E_mag,
        T=state.T,
        rho_rel=state.rho_rel,
        part_mask=state.part_mask,
    )
    if (not snaps_all) or (float(snaps_all[-1].get("time_s", -1.0)) != float(final_snap["time_s"])):
        snaps_all.append(final_snap)
    if is_turntable_run and snaps_tt:
        if float(snaps_tt[-1].get("time_s", -1.0)) != float(final_snap["time_s"]):
            snaps_tt.append(final_snap)
    snaps_all = _dedupe_and_sort_snaps(snaps_all)
    if is_turntable_run:
        snaps_tt = _dedupe_and_sort_snaps(snaps_tt)
        if len(snaps_tt) < 4:
            # If turntable event list is sparse (e.g. 1 rotation), fall back to timeline snapshots.
            snaps_tt = snaps_all

    # Turntable runs: generate only turntable-labeled GIFs to avoid duplication.
    if is_turntable_run:
        rho_limits_tt = _density_gif_limits(snaps_tt, rho_rel_initial)
        _write_animation_gif(
            snaps=snaps_tt, x=state.x, y=state.y, key="E_mag",
            title="Turntable Rotation: Electric Field", cmap="turbo",
            out_path=output_dir / "turntable_electric.gif",
            frame_duration_s=frame_duration_s,
        )
        _write_animation_gif(
            snaps=snaps_tt, x=state.x, y=state.y, key="T",
            title="Turntable Rotation: Temperature", cmap="turbo",
            out_path=output_dir / "turntable_thermal.gif",
            frame_duration_s=frame_duration_s,
        )
        _write_animation_gif(
            snaps=snaps_tt, x=state.x, y=state.y, key="rho_rel",
            title="Turntable Rotation: Relative Density", cmap="magma",
            out_path=output_dir / "turntable_density.gif",
            frame_duration_s=frame_duration_s,
            outside_fill=rho_rel_initial,
            apply_part_mask=True,
            fixed_limits=rho_limits_tt,
        )
        return

    # Non-turntable runs: generic evolution GIFs.
    rho_limits_all = _density_gif_limits(snaps_all, rho_rel_initial)
    _write_animation_gif(
        snaps=snaps_all, x=state.x, y=state.y, key="E_mag",
        title="Electric Field Magnitude Evolution", cmap="turbo",
        out_path=output_dir / "electric_field_evolution.gif",
        frame_duration_s=frame_duration_s,
    )
    _write_animation_gif(
        snaps=snaps_all, x=state.x, y=state.y, key="T",
        title="Temperature Evolution", cmap="turbo",
        out_path=output_dir / "thermal_evolution.gif",
        frame_duration_s=frame_duration_s,
    )
    _write_animation_gif(
        snaps=snaps_all, x=state.x, y=state.y, key="rho_rel",
        title="Relative Density Evolution", cmap="magma",
        out_path=output_dir / "density_evolution.gif",
        frame_duration_s=frame_duration_s,
        outside_fill=rho_rel_initial,
        apply_part_mask=True,
        fixed_limits=rho_limits_all,
    )


def save_outputs(cfg: dict, state: SimState, summary: dict, hist: dict[str, list[float]], output_dir: Path,
                 tt_rotation_steps: list | None = None, opt_data: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "used_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    ant_rows = summary.get("antennae_instances", []) if isinstance(summary, dict) else []
    if isinstance(ant_rows, list) and ant_rows:
        (output_dir / "antennae_summary.json").write_text(
            json.dumps(
                {
                    "enabled": bool(summary.get("antennae_enabled", False)),
                    "count": int(summary.get("antennae_count", len(ant_rows))),
                    "instances": ant_rows,
                },
                indent=2,
            )
        )

    np.savez_compressed(
        output_dir / "fields.npz",
        x=state.x,
        y=state.y,
        V=state.V,
        Ex=state.Ex,
        Ey=state.Ey,
        E_mag=state.E_mag,
        Qrf=state.Qrf,
        T=state.T,
        phi=state.phi,
        rho_rel=state.rho_rel,
        part_mask=state.part_mask,
        doped_mask=state.doped_mask,
        part_id_mask=(state.part_id_mask if state.part_id_mask is not None else np.zeros_like(state.part_mask, dtype=np.int32)),
        elec_hi=state.elec_hi,
        elec_lo=state.elec_lo,
    )

    # ── Gaussian-smoothed display copies (removes grid-cell staircase artefacts) ──
    _sig, _sig_m = 1.2, 1.0
    T_disp     = gaussian_filter(state.T.astype(float),          sigma=_sig)
    E_disp     = gaussian_filter(state.E_mag.astype(float),       sigma=_sig)
    Qrf_disp   = gaussian_filter(state.Qrf.astype(float),         sigma=_sig)
    V_abs_disp = gaussian_filter(np.abs(state.V).astype(float),   sigma=0.7)
    phi_disp   = gaussian_filter(state.phi.astype(float),          sigma=_sig)
    rho_disp   = gaussian_filter(state.rho_rel.astype(float),      sigma=_sig)
    pm_disp    = gaussian_filter(state.part_mask.astype(float),    sigma=_sig_m)
    hi_disp    = gaussian_filter(state.elec_hi.astype(float),      sigma=_sig_m)
    lo_disp    = gaussian_filter(state.elec_lo.astype(float),      sigma=_sig_m)

    rep = cfg.get("reporting", {}) if isinstance(cfg, dict) else {}
    stream_density = float(rep.get("e_stream_density", 0.6))
    stream_lw = float(rep.get("e_stream_linewidth", 0.9))
    stream_arrow = float(rep.get("e_stream_arrowsize", 0.6))
    stream_alpha = float(rep.get("e_stream_alpha", 0.92))
    stream_color = str(rep.get("e_stream_color", "w"))
    stream_under_color = str(rep.get("e_stream_under_color", "#101010"))
    stream_dual_tone = bool(rep.get("e_stream_dual_tone", True))
    stream_seed_mode = str(rep.get("e_stream_seed_mode", "auto")).strip().lower()
    stream_seed_count = int(rep.get("e_stream_seed_count", 33))
    show_equipotential = bool(rep.get("show_equipotential_overlay", False))

    ex_plot = np.array(np.real(state.Ex), dtype=float, copy=True)
    ey_plot = np.array(np.real(state.Ey), dtype=float, copy=True)
    mask_elec = state.elec_hi | state.elec_lo
    ex_plot[mask_elec] = np.nan
    ey_plot[mask_elec] = np.nan

    start_points = None
    if stream_seed_mode in {"hi_electrode", "hv_electrode"}:
        i_hi, j_hi = np.where(state.elec_hi)
        i_lo, _ = np.where(state.elec_lo)
        if i_hi.size > 1 and i_lo.size > 0:
            x0 = float(state.x[np.min(j_hi)])
            x1 = float(state.x[np.max(j_hi)])
            y_hi = float(np.mean(state.y[i_hi]))
            y_lo = float(np.mean(state.y[i_lo]))
            nseed = max(stream_seed_count, 8)
            xs = np.linspace(x0, x1, nseed)
            # Shift just inside the domain toward the low electrode.
            inward = -0.65 * (state.y[1] - state.y[0]) if y_hi > y_lo else 0.65 * (state.y[1] - state.y[0])
            ys = np.full_like(xs, y_hi + inward)
            start_points = np.column_stack([xs, ys])

    def _draw_stream(ax_obj, color: str, lw: float, alpha: float) -> None:
        kwargs = dict(
            density=stream_density,
            color=color,
            linewidth=lw,
            arrowsize=stream_arrow,
            minlength=0.06,
            maxlength=6.0,
            integration_direction="both",
            broken_streamlines=True,
            zorder=4,
        )
        seeded_attempt = start_points is not None
        if seeded_attempt:
            kwargs["start_points"] = start_points
        sps = ax_obj.streamplot(state.x, state.y, ex_plot, ey_plot, **kwargs)
        segs = len(sps.lines.get_segments()) if hasattr(sps.lines, "get_segments") else 0
        # Seeded mode can fail when points land on near-singular/low-gradient cells;
        # fall back to auto seeding so field lines are always visible.
        if seeded_attempt and segs == 0:
            kwargs.pop("start_points", None)
            sps = ax_obj.streamplot(state.x, state.y, ex_plot, ey_plot, **kwargs)
        sps.lines.set_alpha(alpha)
        if hasattr(sps, "arrows") and sps.arrows is not None:
            sps.arrows.set_alpha(alpha)

    def _overlay_e_stream(ax_obj) -> None:
        if stream_dual_tone:
            _draw_stream(ax_obj, stream_under_color, 1.65 * stream_lw, max(0.45, 0.8 * stream_alpha))
        _draw_stream(ax_obj, stream_color, stream_lw, stream_alpha)

    def _style_cbar(cb, decimals: int = 1) -> None:
        cb.formatter = FuncFormatter(lambda val, _pos: f"{val:,.{decimals}f}")
        cb.update_ticks()

    # Electric fields figure.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    v_plot = np.abs(state.V)
    e_plot_kvpm = state.E_mag / 1e3
    q_plot_mwpm3 = state.Qrf / 1e6
    q_part_mwpm3 = q_plot_mwpm3[state.part_mask]
    qrf_display_mode = str(rep.get("qrf_display_mode", "part_p99")).strip().lower()
    qrf_display_vmax = rep.get("qrf_display_vmax_mw_per_m3", None)
    if qrf_display_vmax is None:
        if qrf_display_mode == "part_p95":
            qrf_display_vmax = float(np.percentile(q_part_mwpm3, 95.0))
        elif qrf_display_mode == "global_max":
            qrf_display_vmax = float(np.max(q_plot_mwpm3))
        else:
            qrf_display_vmax = float(np.percentile(q_part_mwpm3, 99.0))
    else:
        qrf_display_vmax = float(qrf_display_vmax)
    qrf_display_vmax = max(qrf_display_vmax, 1e-6)
    q_plot_disp = gaussian_filter(
        np.clip(q_plot_mwpm3, 0.0, qrf_display_vmax).astype(float), sigma=_sig
    )

    im0 = ax[0].imshow(V_abs_disp, extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]], origin="lower", cmap="viridis", interpolation="bilinear")
    ax[0].set_title("|V| [V]")
    cb0 = plt.colorbar(im0, ax=ax[0], shrink=0.84)
    _style_cbar(cb0, decimals=0)

    im1 = ax[1].imshow(E_disp / 1e3, extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]], origin="lower", cmap="turbo", interpolation="bilinear")
    ax[1].set_title("|E| [kV/m]")
    cb1 = plt.colorbar(im1, ax=ax[1], shrink=0.84)
    _style_cbar(cb1, decimals=1)

    im2 = ax[2].imshow(
        q_plot_disp,
        extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]],
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=qrf_display_vmax,
        interpolation="bilinear",
    )
    ax[2].set_title(f"Qrf [MW/m^3] (clip 0-{qrf_display_vmax:.2f})")
    cb2 = plt.colorbar(im2, ax=ax[2], shrink=0.84)
    _style_cbar(cb2, decimals=2)

    # Overlay electric-field streamlines to show field propagation and distortion.
    _overlay_e_stream(ax[1])
    _overlay_e_stream(ax[2])

    # Optional equipotential overlay for diagnostics.
    n_contours = int(rep.get("potential_contours", 18))
    levels = np.linspace(float(np.min(np.real(state.V))), float(np.max(np.real(state.V))), max(n_contours, 3))
    if show_equipotential:
        for a in ax[1:]:
            a.contour(state.x, state.y, np.real(state.V), levels=levels, colors="w", linewidths=0.55, alpha=0.55)

    for a in ax:
        a.contour(state.x, state.y, pm_disp, levels=[0.5], colors=["w"], linewidths=1.0)
        a.contour(state.x, state.y, hi_disp, levels=[0.5], colors=["#00ffcc"], linewidths=0.9)
        a.contour(state.x, state.y, lo_disp, levels=[0.5], colors=["#ff66cc"], linewidths=0.9)
        a.set_aspect("equal")
    _t_total_s = float(hist["time_s"][-1]) if hist["time_s"] else 0.0
    _mm, _ss = divmod(_t_total_s, 60)
    _exp_label = f"Exposure: {_t_total_s:.0f} s  ({int(_mm)} min {_ss:.0f} s)"
    fig.suptitle(_exp_label, fontsize=8, color="#555555", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "electric_fields.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Thermal + state figure.
    fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4.3))
    imt = ax2[0].imshow(T_disp, extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]], origin="lower", cmap="turbo", interpolation="bilinear")
    ax2[0].set_title("T final [C]")
    cbt = plt.colorbar(imt, ax=ax2[0], shrink=0.84)
    _style_cbar(cbt, decimals=1)

    phi_part = np.asarray(state.phi[state.part_mask], dtype=float)
    rho_part = np.asarray(state.rho_rel[state.part_mask], dtype=float)
    phi_plot_mode = str(rep.get("phi_display_mode", "full")).strip().lower()
    rho_plot_mode = str(rep.get("rho_display_mode", "part_contrast")).strip().lower()

    if phi_plot_mode == "part_contrast":
        phi_vmin = float(max(np.min(phi_part), 0.0))
        phi_vmax = float(min(1.0, max(np.percentile(phi_part, 99.0), phi_vmin + 1e-6)))
    else:
        phi_vmin, phi_vmax = 0.0, 1.0
    phi_display = np.where(state.part_mask, phi_disp, np.nan)
    imp = ax2[1].imshow(
        phi_display,
        extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]],
        origin="lower",
        cmap="viridis",
        vmin=phi_vmin,
        vmax=phi_vmax,
        interpolation="bilinear",
    )
    ax2[1].set_title("Melt fraction phi (part only)")
    cbp = plt.colorbar(imp, ax=ax2[1], shrink=0.84)
    _style_cbar(cbp, decimals=2)

    if rho_plot_mode == "part_contrast":
        rho_lo = float(np.min(rho_part))
        rho_hi = float(np.max(rho_part))
        if rho_hi - rho_lo < 0.02:
            rho_hi = min(1.0, rho_lo + 0.02)
        rho_vmin, rho_vmax = max(0.0, rho_lo), min(1.0, rho_hi)
    else:
        rho_vmin, rho_vmax = 0.0, 1.0
    imr = ax2[2].imshow(
        rho_disp,
        extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]],
        origin="lower",
        cmap="magma",
        vmin=rho_vmin,
        vmax=rho_vmax,
        interpolation="bilinear",
    )
    ax2[2].set_title("Relative density")
    cbr = plt.colorbar(imr, ax=ax2[2], shrink=0.84)
    _style_cbar(cbr, decimals=2)

    for a in ax2:
        a.contour(state.x, state.y, pm_disp, levels=[0.5], colors=["w"], linewidths=1.0)
        a.set_aspect("equal")
    fig2.suptitle(_exp_label, fontsize=8, color="#555555", y=1.01)
    fig2.tight_layout()
    fig2.savefig(output_dir / "thermal_fields_final.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Time series.
    t = np.array(hist["time_s"], dtype=float)
    fig3, ax3 = plt.subplots(2, 2, figsize=(10.5, 6.8))
    ax3[0, 0].plot(t, hist["mean_T_part_c"], "-", lw=1.8)
    ax3[0, 0].plot(t, hist["max_T_part_c"], "-", lw=1.2)
    ax3[0, 0].set_title("Part temperature")
    ax3[0, 0].set_xlabel("time [s]")
    ax3[0, 0].set_ylabel("T [C]")
    ax3[0, 0].legend(["mean", "max"], loc="best")
    ax3[0, 0].grid(alpha=0.25)

    # Dual Y-axis: power (W/m) on left, cumulative energy (J/m) on right.
    # Formerly plotted on one axis, making power (~500 W/m) invisible next to
    # cumulative energy (~180 000 J/m).
    _ax_pw = ax3[0, 1]
    _ax_en = ax3[0, 1].twinx()
    _ax_pw.plot(t, hist["power_doped_W_per_m"],     "-",  lw=1.8, color="tab:blue",   label="Q_rf power")
    _ax_pw.plot(t, hist["power_conv_loss_W_per_m"], "--", lw=1.2, color="tab:cyan",   label="conv loss")
    _ax_en.plot(t, hist["energy_doped_J_per_m"],    "-",  lw=1.2, color="tab:orange", label="cum. Q_rf")
    _ax_en.plot(t, hist["energy_stored_part_J_per_m"], "-", lw=1.2, color="tab:red",  label="stored")
    _ax_pw.set_title("Power and cumulative energy")
    _ax_pw.set_xlabel("time [s]")
    _ax_pw.set_ylabel("Power [W/m depth]", color="tab:blue")
    _ax_en.set_ylabel("Energy [J/m depth]", color="tab:orange")
    _ax_pw.tick_params(axis="y", colors="tab:blue")
    _ax_en.tick_params(axis="y", colors="tab:orange")
    _lines_pw, _labs_pw = _ax_pw.get_legend_handles_labels()
    _lines_en, _labs_en = _ax_en.get_legend_handles_labels()
    _ax_pw.legend(_lines_pw + _lines_en, _labs_pw + _labs_en, loc="center right", fontsize=8)
    _ax_pw.grid(alpha=0.25)

    ax3[1, 0].plot(t, hist["ui_abs_part"], "-", lw=1.8)
    ax3[1, 0].plot(t, hist["ui_rms_part"], "-", lw=1.2)
    ax3[1, 0].set_title("Uniformity index")
    ax3[1, 0].set_xlabel("time [s]")
    ax3[1, 0].set_ylabel("UI")
    ax3[1, 0].legend(["UI_abs", "UI_rms"], loc="best")
    ax3[1, 0].grid(alpha=0.25)

    ax3[1, 1].plot(t, hist["mean_phi_part"], "-", lw=1.8)
    ax3[1, 1].plot(t, hist["mean_rho_rel_part"], "-", lw=1.2)
    ax3[1, 1].set_title("Melt and densification")
    ax3[1, 1].set_xlabel("time [s]")
    ax3[1, 1].set_ylabel("fraction")
    ax3[1, 1].set_ylim(0.0, 1.0)
    ax3[1, 1].grid(alpha=0.25)
    ax3r = ax3[1, 1].twinx()
    ax3r.plot(t, hist["mean_dens_rate_part_per_s"], "-", lw=1.0, color="tab:green", alpha=0.9)
    ax3r.set_ylabel("drho/dt [1/s]", color="tab:green")
    ax3r.tick_params(axis="y", colors="tab:green")
    ax3[1, 1].legend(["mean phi", "mean rho_rel"], loc="upper left")

    # Turntable rotation event markers (vertical dashed lines, purple)
    if tt_rotation_steps:
        t_vals = np.array(hist["time_s"], dtype=float)
        dt_ts  = float(t_vals[1] - t_vals[0]) if len(t_vals) > 1 else 0.5
        for rs in tt_rotation_steps:
            t_rot = rs * dt_ts
            for axr in ax3.flat:
                axr.axvline(t_rot, color="#9467bd", lw=0.9, ls=":", alpha=0.7)
            ax3[0, 0].text(t_rot, ax3[0, 0].get_ylim()[1],
                           "↺", fontsize=9, ha="center", va="top", color="#9467bd")

    fig3.tight_layout()
    fig3.savefig(output_dir / "time_series.png", dpi=180)
    plt.close(fig3)

    # Persist the time-series dict as JSON for post-processing / sweep analysis.
    import json as _json_ts
    _ts_path = output_dir / "time_series.json"
    with _ts_path.open("w", encoding="utf-8") as _fh_ts:
        _json_ts.dump(
            {k: [float(v) for v in vals] for k, vals in hist.items()},
            _fh_ts,
            indent=2,
        )
    print(f"Saved time series: {_ts_path}")

    # Compact "paper-style" summary figure.
    fig4, ax4 = plt.subplots(2, 2, figsize=(10.6, 8.8))
    # Colorbars must match the displayed map range for contextual correctness.
    _t_finite = T_disp[np.isfinite(T_disp)]
    if _t_finite.size:
        _t_vmin = float(np.min(_t_finite))
        _t_vmax = float(np.max(_t_finite))
    else:
        _t_vmin, _t_vmax = 0.0, 1.0
    if math.isclose(_t_vmin, _t_vmax):
        _t_vmax = _t_vmin + 1e-6

    _q_finite = q_plot_disp[np.isfinite(q_plot_disp)]
    if _q_finite.size:
        _q_vmin = float(np.min(_q_finite))
        _q_vmax = float(np.max(_q_finite))
    else:
        _q_vmin, _q_vmax = 0.0, 1.0
    if math.isclose(_q_vmin, _q_vmax):
        _q_vmax = _q_vmin + 1e-6

    imrpt = ax4[0, 0].imshow(
        T_disp,
        extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]],
        origin="lower",
        cmap="jet",
        vmin=_t_vmin,
        vmax=_t_vmax,
        interpolation="bilinear",
    )
    ax4[0, 0].set_title(f"Temperature [C] with E-field streamlines ({_t_vmin:.1f}-{_t_vmax:.1f})")
    _overlay_e_stream(ax4[0, 0])
    if show_equipotential:
        ax4[0, 0].contour(state.x, state.y, np.real(state.V), levels=levels, colors="w", linewidths=0.5, alpha=0.45)
    cbrpt = plt.colorbar(imrpt, ax=ax4[0, 0], shrink=0.84)
    _style_cbar(cbrpt, decimals=1)

    imq = ax4[0, 1].imshow(
        q_plot_disp,
        extent=[state.x[0], state.x[-1], state.y[0], state.y[-1]],
        origin="lower",
        cmap="jet",
        vmin=_q_vmin,
        vmax=_q_vmax,
        interpolation="bilinear",
    )
    ax4[0, 1].set_title(f"Qrf [MW/m^3] ({_q_vmin:.2f}-{_q_vmax:.2f})")
    _overlay_e_stream(ax4[0, 1])
    cbq = plt.colorbar(imq, ax=ax4[0, 1], shrink=0.84)
    _style_cbar(cbq, decimals=2)

    ax4[1, 0].plot(t, hist["mean_T_part_c"], "-", lw=1.8, label="mean T")
    ax4[1, 0].plot(t, hist["max_T_part_c"], "-", lw=1.2, label="max T")
    ax4[1, 0].plot(t, hist["ui_abs_part"], "-", lw=1.2, label="UI_abs")
    ax4[1, 0].set_title("Thermal trajectory")
    ax4[1, 0].set_xlabel("time [s]")
    ax4[1, 0].grid(alpha=0.25)
    ax4[1, 0].legend(loc="best")

    ax4[1, 1].plot(t, hist["mean_phi_part"], "-", lw=1.8, label="mean phi")
    ax4[1, 1].plot(t, hist["mean_rho_rel_part"], "-", lw=1.2, label="mean rho_rel")
    ax4[1, 1].set_title("Melt, density, energy balance")
    ax4[1, 1].set_xlabel("time [s]")
    ax4[1, 1].set_ylim(0.0, 1.0)
    ax4[1, 1].grid(alpha=0.25)
    ax4[1, 1].legend(loc="upper left")
    ax4r = ax4[1, 1].twinx()
    ax4r.plot(t, hist["energy_balance_residual_J_per_m"], "-", lw=1.0, color="tab:red", alpha=0.9, label="Ebal residual")
    ax4r.set_ylabel("residual [J/m]", color="tab:red")
    ax4r.tick_params(axis="y", colors="tab:red")

    for a in [ax4[0, 0], ax4[0, 1]]:
        a.contour(state.x, state.y, pm_disp, levels=[0.5], colors=["w"], linewidths=1.0)
        a.contour(state.x, state.y, hi_disp, levels=[0.5], colors=["#00ffcc"], linewidths=0.9)
        a.contour(state.x, state.y, lo_disp, levels=[0.5], colors=["#ff66cc"], linewidths=0.9)
        a.set_aspect("equal")

    fig4.suptitle(_exp_label, fontsize=8, color="#555555", y=1.01)
    fig4.tight_layout()
    fig4.savefig(output_dir / "paper_style_report.png", dpi=180, bbox_inches="tight")
    plt.close(fig4)

    # Animated evolution outputs (all runs + turntable rotation events).
    _generate_animation_gifs(cfg=cfg, state=state, output_dir=output_dir, opt_data=opt_data)

    # Always attempt consolidated summary figure output for every run mode.
    try:
        import make_rf_summary_v5 as _v5mod
        _v5mod.make_rf_summary_v5(str(output_dir))
    except Exception as _v5_err:
        print(f"  [rf_summary_v5] Skipped: {_v5_err}")

    # Persist per-run report manifest used by generalized backfill tooling.
    try:
        def _run_type_from_cfg(_cfg: dict) -> str:
            if bool(_cfg.get("orientation_optimizer", {}).get("enabled", False)):
                return "orientation_optimizer"
            if bool(_cfg.get("placement_optimizer", {}).get("enabled", False)):
                return "placement_optimizer"
            if bool(_cfg.get("turntable", {}).get("enabled", False)):
                return "turntable"
            if bool(_cfg.get("optimizer", {}).get("enabled", False)):
                return "optimizer"
            return "single"

        rel_files = sorted(
            p.relative_to(output_dir).as_posix()
            for p in output_dir.rglob("*")
            if p.is_file()
        )
        run_type = _run_type_from_cfg(cfg)
        caps: list[str] = ["manifest_refresh_v1"]
        rel_set = set(rel_files)
        if {"summary.json", "fields.npz"}.issubset(rel_set):
            caps.append("rf_summary_v5_v1")
        if "fields.npz" in rel_set:
            caps.append("core_reports_v1")
        if run_type == "orientation_optimizer" and ("orientation_candidates.json" in rel_set or "orientation_pareto.json" in rel_set):
            caps.append("orientation_diagnostics_v1")
        if run_type == "placement_optimizer" and "placement_topk_coupled.json" in rel_set:
            caps.append("placement_reports_v1")
        if run_type == "shell_sweep" and "shell_sweep_summary.json" in rel_set:
            caps.append("shell_sweep_report_v1")
        if "antennae_summary.json" in rel_set and {"summary.json", "fields.npz"}.issubset(rel_set):
            caps.append("antennae_reports_v1")
        reports = [
            r for r in rel_files
            if r.lower().endswith((".png", ".gif", ".svg", ".json", ".csv"))
            and ("report" in r.lower() or r.startswith("orientation_") or r.startswith("placement_") or r.startswith("shell_sweep_") or r.endswith("rf_summary_v5.png"))
        ]
        manifest_path = output_dir / "report_manifest.json"
        existing = {}
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
            except Exception:
                existing = {}
        manifest = {
            "schema_version": 1,
            "run_type": run_type,
            "created_at": existing.get("created_at", datetime.now().isoformat(timespec="seconds")),
            "last_backfill_at": existing.get("last_backfill_at", None),
            "available_inputs": rel_files,
            "available_reports": sorted(set(reports)),
            "backfill_capabilities": sorted(set(caps)),
            "module_history": existing.get("module_history", []),
            "latest_by_module": existing.get("latest_by_module", {}),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except Exception as _mf_err:
        print(f"  [report_manifest] Skipped: {_mf_err}")


def _set_exposure_seconds(cfg: dict, exposure_s: float) -> None:
    thermal = cfg.setdefault("thermal", {})
    dt = float(thermal.get("dt_s", 0.5))
    thermal["n_steps"] = int(max(1, round(float(exposure_s) / max(dt, 1e-9))))


def _get_exposure_seconds(cfg: dict) -> float:
    thermal = cfg.get("thermal", {}) if isinstance(cfg.get("thermal", {}), dict) else {}
    dt = float(thermal.get("dt_s", 0.5))
    n_steps = int(thermal.get("n_steps", 1))
    return float(max(dt, 0.0) * max(n_steps, 1))


def _dominates(a: dict, b: dict) -> bool:
    better_or_equal = (
        a["mean_rho"] >= b["mean_rho"]
        and a["min_rho"] >= b["min_rho"]
        and a["std_rho"] <= b["std_rho"]
        and a["temp_violation"] <= b["temp_violation"]
    )
    strictly_better = (
        a["mean_rho"] > b["mean_rho"]
        or a["min_rho"] > b["min_rho"]
        or a["std_rho"] < b["std_rho"]
        or a["temp_violation"] < b["temp_violation"]
    )
    return bool(better_or_equal and strictly_better)


def _pareto_front(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, r in enumerate(rows):
        dominated = False
        for j, s in enumerate(rows):
            if i == j:
                continue
            if _dominates(s, r):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return out


def _recommend_knee(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty pareto set")
    arr_mean = np.array([r["mean_rho"] for r in rows], dtype=float)
    arr_min = np.array([r["min_rho"] for r in rows], dtype=float)
    arr_std = np.array([r["std_rho"] for r in rows], dtype=float)
    arr_tv = np.array([r["temp_violation"] for r in rows], dtype=float)
    n_mean = (arr_mean - np.min(arr_mean)) / max(np.max(arr_mean) - np.min(arr_mean), 1e-9)
    n_min = (arr_min - np.min(arr_min)) / max(np.max(arr_min) - np.min(arr_min), 1e-9)
    n_std = 1.0 - (arr_std - np.min(arr_std)) / max(np.max(arr_std) - np.min(arr_std), 1e-9)
    n_tv = 1.0 - (arr_tv - np.min(arr_tv)) / max(np.max(arr_tv) - np.min(arr_tv), 1e-9)
    score = 0.35 * n_mean + 0.35 * n_min + 0.20 * n_std + 0.10 * n_tv
    return rows[int(np.argmax(score))]


def _knee_scores(rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    arr_mean = np.array([r["mean_rho"] for r in rows], dtype=float)
    arr_min = np.array([r["min_rho"] for r in rows], dtype=float)
    arr_std = np.array([r["std_rho"] for r in rows], dtype=float)
    arr_tv = np.array([r["temp_violation"] for r in rows], dtype=float)
    n_mean = (arr_mean - np.min(arr_mean)) / max(np.max(arr_mean) - np.min(arr_mean), 1e-9)
    n_min = (arr_min - np.min(arr_min)) / max(np.max(arr_min) - np.min(arr_min), 1e-9)
    n_std = 1.0 - (arr_std - np.min(arr_std)) / max(np.max(arr_std) - np.min(arr_std), 1e-9)
    n_tv = 1.0 - (arr_tv - np.min(arr_tv)) / max(np.max(arr_tv) - np.min(arr_tv), 1e-9)
    return 0.35 * n_mean + 0.35 * n_min + 0.20 * n_std + 0.10 * n_tv


def _candidate_metrics_from_summary(summary: dict, constraints: dict) -> dict:
    part_stats = summary.get("part_stats", []) if isinstance(summary.get("part_stats", []), list) else []
    per_part_mean = [float(s.get("mean_rho_rel", 0.0)) for s in part_stats if isinstance(s, dict)]
    per_part_min = [float(s.get("min_rho_rel", 0.0)) for s in part_stats if isinstance(s, dict)]
    mean_rho = float(summary.get("mean_rho_rel_part_final", 0.0))
    min_rho = float(min(per_part_min) if per_part_min else summary.get("mean_rho_rel_part_final", 0.0))
    std_rho = float(np.std(per_part_mean)) if per_part_mean else float(np.std([mean_rho]))
    max_t = float(summary.get("max_T_part_final_c", 0.0))
    temp_ceiling = float(constraints.get("temp_ceiling_c", 250.0))
    min_rho_floor = float(constraints.get("min_rho_floor", 0.0))
    return {
        "mean_rho": mean_rho,
        "min_rho": min_rho,
        "std_rho": std_rho,
        "max_temp_c": max_t,
        "temp_ceiling_c": temp_ceiling,
        "temp_violation": max(0.0, max_t - temp_ceiling),
        "min_rho_violation": max(0.0, min_rho_floor - min_rho),
    }


def _render_scatter_report(
    rows: list[dict],
    pareto: list[dict],
    best: dict,
    output_png: Path,
    title: str,
    context_text: str | None = None,
) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=180)
    means = [r["mean_rho"] for r in rows]
    mins = [r["min_rho"] for r in rows]
    stds = [r["std_rho"] for r in rows]
    tv = [r["temp_violation"] for r in rows]
    ax[0].scatter(means, mins, c=stds, cmap="viridis", s=20, alpha=0.35, label="all candidates")
    ax[0].set_xlabel("mean rho")
    ax[0].set_ylabel("min rho")
    ax[0].set_title("Densification tradeoff")
    ax[0].grid(alpha=0.25)
    sc_tv = ax[1].scatter(stds, means, c=tv, cmap="magma", s=20, alpha=0.35, label="all candidates")
    ax[1].set_xlabel("std rho")
    ax[1].set_ylabel("mean rho")
    ax[1].set_title("Uniformity vs mean")
    ax[1].grid(alpha=0.25)

    if pareto:
        p_means = [r["mean_rho"] for r in pareto]
        p_mins = [r["min_rho"] for r in pareto]
        p_stds = [r["std_rho"] for r in pareto]
        p_tvs = [r["temp_violation"] for r in pareto]
        ax[0].scatter(p_means, p_mins, marker="D", s=40, facecolors="none", edgecolors="black", linewidths=1.1, label="pareto front")
        ax[1].scatter(p_stds, p_means, marker="D", s=40, facecolors="none", edgecolors="black", linewidths=1.1, label="pareto front")

    ax[0].plot([best.get("mean_rho", 0.0)], [best.get("min_rho", 0.0)], "r*", ms=14)
    ax[1].plot([best.get("std_rho", 0.0)], [best.get("mean_rho", 0.0)], "r*", ms=14)
    if "rotation_deg" in best and "exposure_s" in best:
        ideal_txt = f"Ideal pareto angle: {float(best['rotation_deg']):.1f} deg @ {float(best['exposure_s']):.0f} s"
    else:
        ideal_txt = "Selected Pareto point"
    ax[0].annotate(
        ideal_txt,
        xy=(best.get("mean_rho", 0.0), best.get("min_rho", 0.0)),
        xytext=(0.03, 0.06),
        textcoords="axes fraction",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#444444", alpha=0.9),
    )
    ceil_c = None
    for r in rows:
        if "temp_ceiling_c" in r:
            ceil_c = float(r["temp_ceiling_c"])
            break
    cbar = plt.colorbar(sc_tv, ax=ax[1], shrink=0.88)
    if ceil_c is not None:
        cbar.set_label(f"temp ceiling violation [C], ceiling={ceil_c:.1f} C")
    else:
        cbar.set_label("temp ceiling violation [C]")
    if context_text:
        fig.text(0.01, 0.01, context_text, fontsize=8.2, color="#555555")
    for axi in ax:
        axi.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _render_orientation_angle_report(
    angle_rows: list[dict],
    best: dict,
    output_png: Path,
    color_metric: str = "mean_rho",
    temp_ceiling_c: float | None = None,
    exposure_text: str | None = None,
) -> None:
    if not angle_rows:
        return
    angles = [float(r["rotation_deg"]) for r in angle_rows]
    scores = [float(r["best_effectiveness_score"]) for r in angle_rows]
    temp_v = [float(r.get("best_temp_violation", 0.0)) for r in angle_rows]
    mean_r = [float(r.get("best_mean_rho", 0.0)) for r in angle_rows]
    min_r = [float(r.get("best_min_rho", 0.0)) for r in angle_rows]
    pareto_flag = [bool(r.get("is_pareto_angle", False)) for r in angle_rows]

    metric = str(color_metric or "mean_rho").strip().lower()
    if metric == "temp_violation":
        cvals = np.array(temp_v, dtype=float)
        cmap = "magma_r"
        if temp_ceiling_c is not None:
            cbar_label = f"temp ceiling violation [C], ceiling={temp_ceiling_c:.1f} C"
        else:
            cbar_label = "temp ceiling violation [C]"
    elif metric == "min_rho":
        cvals = np.array(min_r, dtype=float)
        cmap = "viridis"
        cbar_label = "best min rho"
    else:
        metric = "mean_rho"
        cvals = np.array(mean_r, dtype=float)
        cmap = "viridis"
        cbar_label = "best mean rho"

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.4), dpi=180)
    sc = ax.scatter(angles, scores, c=cvals, cmap=cmap, s=70, alpha=0.85, edgecolor="k", linewidth=0.35, label="angle candidates")
    # Always preserve temperature context: violating points are outlined in red.
    for a, s, tv in zip(angles, scores, temp_v):
        if float(tv) > 0.0:
            ax.plot([a], [s], marker="o", ms=10, markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.25, linestyle="None")
    for a, s, pr in zip(angles, scores, pareto_flag):
        if pr:
            ax.plot([a], [s], marker="D", ms=7, markerfacecolor="none", markeredgecolor="black", linestyle="None")
    b_ang = float(best.get("rotation_deg", 0.0))
    b_score = float(best.get("effectiveness_score", np.max(scores) if scores else 0.0))
    ax.plot([b_ang], [b_score], "r*", ms=15, label="selected ideal angle")
    ax.set_title("Orientation Angle Effectiveness (best exposure per angle)")
    ax.set_xlabel("rotation angle [deg]")
    ax.set_ylabel("effectiveness score")
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)
    for a, s, m in zip(angles, scores, mean_r):
        ax.text(a, s + 0.005, f"{m:.3f}", fontsize=7, ha="center", va="bottom", alpha=0.7)
    ax.text(
        0.995,
        0.015,
        "red ring = temp ceiling violated",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#b22222",
    )
    ax.legend(loc="best", fontsize=8)
    context_lines: list[str] = []
    if exposure_text:
        context_lines.append(str(exposure_text))
    if temp_ceiling_c is not None:
        exp_txt_l = str(exposure_text or "").lower()
        if "ceiling" not in exp_txt_l:
            context_lines.append(f"temperature ceiling: {float(temp_ceiling_c):.1f} C")
    if context_lines:
        ax.text(
            0.01,
            0.99,
            "\n".join(context_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666666", alpha=0.9),
        )
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _orientation_frame_from_state(state: SimState, row: dict) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    T = np.asarray(state.T, dtype=float)
    tmin = float(np.nanmin(T))
    tmax = float(np.nanmax(T))
    if math.isclose(tmin, tmax):
        tmax = tmin + 1e-6
    norm = np.clip((T - tmin) / max(tmax - tmin, 1e-9), 0.0, 1.0)
    cmap = plt.get_cmap("inferno")
    rgba = cmap(norm)
    rgb = (255.0 * np.asarray(rgba[..., :3], dtype=float)).astype(np.uint8)
    mask = np.asarray(state.part_mask, dtype=bool)
    rgb[~mask] = (0.30 * rgb[~mask]).astype(np.uint8)
    img = Image.fromarray(rgb).resize((280, 280), resample=Image.Resampling.BILINEAR)
    bar_h = 46
    canvas = Image.new("RGB", (280, 280 + bar_h), color=(15, 15, 18))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    txt1 = f"angle {float(row['rotation_deg']):.1f} deg | exposure {float(row['best_exposure_s']):.0f} s"
    txt2 = f"mean {float(row['best_mean_rho']):.3f} | min {float(row['best_min_rho']):.3f} | dTceil {float(row['best_temp_violation']):.1f} C"
    draw.text((8, 286), txt1, fill=(235, 235, 235))
    draw.text((8, 302), txt2, fill=(190, 190, 190))
    return np.asarray(canvas, dtype=np.uint8)


def _write_orientation_angle_gif(
    cfg_base: dict,
    angle_rows: list[dict],
    output_dir: Path,
    frame_duration_s: float = 2.0,
    output_name: str = "orientation_angle_gallery.gif",
) -> None:
    if not angle_rows:
        return
    try:
        import imageio.v2 as imageio
    except Exception:
        print("  [orientation gif] imageio not available; skipping.")
        return
    frames: list[np.ndarray] = []
    sorted_rows = sorted(angle_rows, key=lambda r: float(r["rotation_deg"]))
    for row in sorted_rows:
        c = copy.deepcopy(cfg_base)
        if isinstance(c.get("geometry", {}).get("part", {}), dict):
            c["geometry"]["part"]["rotation_deg"] = float(row["rotation_deg"])
        elif isinstance(c.get("geometry", {}).get("parts", []), list):
            for part in c["geometry"]["parts"]:
                if isinstance(part, dict):
                    part["rotation_deg"] = float(row["rotation_deg"])
        _set_exposure_seconds(c, float(row["best_exposure_s"]))
        state, _, _, _, _ = run_sim(c)
        frames.append(_orientation_frame_from_state(state, row))
    if frames:
        imageio.mimsave(
            output_dir / output_name,
            frames,
            duration=max(0.1, float(frame_duration_s)),
            loop=0,
        )


def run_orientation_optimizer(cfg: dict, output_dir: Path) -> dict:
    oo = cfg.get("orientation_optimizer", {}) if isinstance(cfg.get("orientation_optimizer", {}), dict) else {}
    color_metric = str(oo.get("color_metric", "mean_rho")).strip().lower()
    constraints = oo.get("constraints", {}) if isinstance(oo.get("constraints", {}), dict) else {}
    a_min = float(oo.get("angle_min_deg", 0.0))
    a_max = float(oo.get("angle_max_deg", 180.0))
    a_step = float(oo.get("angle_step_deg", 15.0))
    e_min = float(oo.get("exposure_min_s", 240.0))
    e_max = float(oo.get("exposure_max_s", 720.0))
    e_step = float(oo.get("exposure_step_s", 60.0))
    refine_window = float(oo.get("refine_window_deg", 10.0))
    refine_step = float(oo.get("refine_step_deg", 5.0))

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []

    base = copy.deepcopy(cfg)
    base.pop("orientation_optimizer", None)
    base.pop("optimizer", None)
    base.pop("turntable", None)
    base.pop("placement_optimizer", None)

    angles = np.arange(a_min, a_max + 1e-9, max(a_step, 1e-9), dtype=float)
    exposures = np.arange(e_min, e_max + 1e-9, max(e_step, 1e-9), dtype=float)
    for ang in angles:
        for exp_s in exposures:
            c = copy.deepcopy(base)
            if isinstance(c.get("geometry", {}).get("part", {}), dict):
                c["geometry"]["part"]["rotation_deg"] = float(ang)
            elif isinstance(c.get("geometry", {}).get("parts", []), list):
                for part in c["geometry"]["parts"]:
                    if isinstance(part, dict):
                        part["rotation_deg"] = float(ang)
            _set_exposure_seconds(c, exp_s)
            _, summary, _, _, _ = run_sim(c)
            m = _candidate_metrics_from_summary(summary, constraints)
            m.update({"rotation_deg": float(ang), "exposure_s": float(exp_s), "source": "coarse"})
            candidates.append(m)

    coarse_front = _pareto_front(candidates)
    coarse_best = _recommend_knee(coarse_front if coarse_front else candidates)
    ref_angles = np.arange(
        coarse_best["rotation_deg"] - refine_window,
        coarse_best["rotation_deg"] + refine_window + 1e-9,
        max(refine_step, 1e-9),
        dtype=float,
    )
    for ang in ref_angles:
        for exp_s in exposures:
            if any(abs(r["rotation_deg"] - float(ang)) < 1e-9 and abs(r["exposure_s"] - float(exp_s)) < 1e-9 for r in candidates):
                continue
            c = copy.deepcopy(base)
            if isinstance(c.get("geometry", {}).get("part", {}), dict):
                c["geometry"]["part"]["rotation_deg"] = float(ang)
            elif isinstance(c.get("geometry", {}).get("parts", []), list):
                for part in c["geometry"]["parts"]:
                    if isinstance(part, dict):
                        part["rotation_deg"] = float(ang)
            _set_exposure_seconds(c, exp_s)
            _, summary, _, _, _ = run_sim(c)
            m = _candidate_metrics_from_summary(summary, constraints)
            m.update({"rotation_deg": float(ang), "exposure_s": float(exp_s), "source": "refine"})
            candidates.append(m)

    pareto = _pareto_front(candidates)
    scores = _knee_scores(candidates)
    pareto_key = {
        (round(float(r["rotation_deg"]), 9), round(float(r["exposure_s"]), 9))
        for r in pareto
    }
    for idx, r in enumerate(candidates):
        r["effectiveness_score"] = float(scores[idx]) if idx < len(scores) else 0.0
        r["is_pareto"] = (
            round(float(r["rotation_deg"]), 9),
            round(float(r["exposure_s"]), 9),
        ) in pareto_key
    best = _recommend_knee(pareto if pareto else candidates)
    best = dict(best)
    if "effectiveness_score" not in best:
        best["effectiveness_score"] = float(max(scores)) if len(scores) else 0.0

    angle_map: dict[float, list[dict]] = {}
    for r in candidates:
        angle_map.setdefault(float(r["rotation_deg"]), []).append(r)
    angle_rows: list[dict] = []
    for ang in sorted(angle_map.keys()):
        rows_ang = angle_map[ang]
        best_ang = max(rows_ang, key=lambda rr: float(rr.get("effectiveness_score", 0.0)))
        pareto_exposures = sorted(
            float(rr["exposure_s"]) for rr in rows_ang if bool(rr.get("is_pareto", False))
        )
        angle_rows.append(
            {
                "rotation_deg": float(ang),
                "n_candidates": int(len(rows_ang)),
                "best_exposure_s": float(best_ang["exposure_s"]),
                "best_effectiveness_score": float(best_ang.get("effectiveness_score", 0.0)),
                "best_mean_rho": float(best_ang["mean_rho"]),
                "best_min_rho": float(best_ang["min_rho"]),
                "best_std_rho": float(best_ang["std_rho"]),
                "best_temp_violation": float(best_ang["temp_violation"]),
                "is_pareto_angle": bool(len(pareto_exposures) > 0),
                "pareto_exposures_s": pareto_exposures,
                "effective_reason": (
                    "on pareto front"
                    if pareto_exposures
                    else "dominated (lower score or higher violation)"
                ),
            }
        )

    (output_dir / "orientation_candidates.json").write_text(json.dumps(candidates, indent=2))
    with (output_dir / "orientation_candidates.csv").open("w", newline="", encoding="utf-8") as _f_csv:
        keys = [
            "rotation_deg", "exposure_s", "source", "effectiveness_score", "is_pareto",
            "mean_rho", "min_rho", "std_rho", "max_temp_c", "temp_violation", "min_rho_violation",
        ]
        w = csv.DictWriter(_f_csv, fieldnames=keys)
        w.writeheader()
        for r in candidates:
            w.writerow({k: r.get(k, "") for k in keys})
    (output_dir / "orientation_angle_summary.json").write_text(json.dumps(angle_rows, indent=2))
    with (output_dir / "orientation_angle_summary.csv").open("w", newline="", encoding="utf-8") as _f_ang:
        keys = [
            "rotation_deg", "n_candidates", "best_exposure_s", "best_effectiveness_score",
            "best_mean_rho", "best_min_rho", "best_std_rho", "best_temp_violation",
            "is_pareto_angle", "pareto_exposures_s", "effective_reason",
        ]
        w = csv.DictWriter(_f_ang, fieldnames=keys)
        w.writeheader()
        for r in angle_rows:
            row_out = dict(r)
            row_out["pareto_exposures_s"] = ",".join(f"{v:.1f}" for v in r.get("pareto_exposures_s", []))
            w.writerow({k: row_out.get(k, "") for k in keys})
    (output_dir / "orientation_pareto.json").write_text(json.dumps(pareto, indent=2))
    (output_dir / "orientation_best.json").write_text(json.dumps(best, indent=2))
    temp_ceiling_c = float(constraints.get("temp_ceiling_c", 250.0))
    orient_context = (
        f"exposure sweep: {e_min:.0f}-{e_max:.0f} s (step {e_step:.0f} s)\n"
        f"temperature ceiling: {temp_ceiling_c:.1f} C"
    )
    _render_scatter_report(
        candidates,
        pareto,
        best,
        output_dir / "orientation_report.png",
        "Orientation Optimizer",
        context_text=orient_context,
    )
    _render_orientation_angle_report(
        angle_rows,
        best,
        output_dir / "orientation_angle_effectiveness.png",
        color_metric=color_metric,
        temp_ceiling_c=temp_ceiling_c,
        exposure_text=orient_context,
    )
    if bool(oo.get("angle_gif_enabled", False)):
        _write_orientation_angle_gif(
            base,
            angle_rows,
            output_dir,
            frame_duration_s=float(oo.get("angle_gif_frame_duration_s", 2.0)),
        )

    final_cfg = copy.deepcopy(base)
    if isinstance(final_cfg.get("geometry", {}).get("part", {}), dict):
        final_cfg["geometry"]["part"]["rotation_deg"] = float(best["rotation_deg"])
    _set_exposure_seconds(final_cfg, float(best["exposure_s"]))
    state, summary, hist, tt_steps, opt_data = run_sim(final_cfg)
    summary["orientation_optimizer"] = {
        "best": best,
        "pareto_count": len(pareto),
        "angles_tried": int(len(angle_rows)),
        "color_metric": color_metric,
        "angle_gif_frame_duration_s": float(oo.get("angle_gif_frame_duration_s", 2.0)),
    }
    save_outputs(final_cfg, state, summary, hist, output_dir, tt_rotation_steps=tt_steps, opt_data=opt_data)
    return best


def _rankdata(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    ra = _rankdata(a)
    rb = _rankdata(b)
    c = np.corrcoef(ra, rb)
    return float(c[0, 1]) if c.shape == (2, 2) and np.isfinite(c[0, 1]) else 0.0


def _compute_eqs_qrf_for_cfg(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, _, part_mask, doped_mask, elec_hi, elec_lo, fill_frac, _pid, _ = make_domain(cfg)
    geom = cfg["geometry"]
    dx = float(geom["chamber_x"]) / max(int(geom["grid_nx"]) - 1, 1)
    dy = float(geom["chamber_y"]) / max(int(geom["grid_ny"]) - 1, 1)
    elec = cfg["electric"]
    mat = cfg["materials"]
    sigma_v = max(float(mat["virgin"]["sigma_s_per_m"]), 1e-8)
    eps_v = float(mat["virgin"]["eps_r"])
    sigma_d0 = float(mat["doped"]["sigma_s_per_m"])
    eps_d = float(mat["doped"]["eps_r"])
    sigma = sigma_v + fill_frac * (sigma_d0 - sigma_v)
    eps_r = eps_v + fill_frac * (eps_d - eps_v)
    omega = 2.0 * math.pi * float(elec["frequency_hz"])
    v0 = float(elec["voltage_v"])
    mode = str(elec.get("voltage_mode", "centered")).strip().lower()
    if mode in {"grounded", "single_ended"}:
        v_hi, v_lo = v0, 0.0
    else:
        v_hi, v_lo = 0.5 * v0, -0.5 * v0
    _, _, _, _, _, qrf = solve_electric_state(
        sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo, dx, dy, elec, x, y,
        float(elec.get("power_factor", 1.0)), float(elec.get("max_qrf_w_per_m3", 1.0e11)),
    )
    if bool(elec.get("zero_qrf_outside_doped", True)):
        qrf = np.where(doped_mask, qrf, 0.0)
    return x, y, part_mask, np.asarray(qrf, dtype=float)


def generate_antennae_preview(cfg: dict, output_png: Path) -> dict:
    if not _antennae_enabled(cfg):
        return {"ok": False, "error": "antennae disabled"}
    work = copy.deepcopy(cfg)
    ant = _resolve_antennae_instances(work)
    base = copy.deepcopy(cfg)
    base["antennae"] = dict(base.get("antennae", {})) if isinstance(base.get("antennae", {}), dict) else {}
    base["antennae"]["enabled"] = False
    if "spike" in base and isinstance(base["spike"], dict):
        base["spike"]["enabled"] = False
    for p in _parts_from_geometry(base.get("geometry", {})):
        if isinstance(p, dict):
            p.pop("antennae_instances", None)
    x, y, p_mask, qrf = _compute_eqs_qrf_for_cfg(base)
    qmw = qrf / 1.0e6
    qf = qmw[np.isfinite(qmw)]
    qmin = float(np.min(qf)) if qf.size else 0.0
    qmax = float(np.max(qf)) if qf.size else 1.0
    if math.isclose(qmin, qmax):
        qmax = qmin + 1e-6
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.8), dpi=180)
    im = ax.imshow(
        qmw,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap="magma",
        vmin=qmin,
        vmax=qmax,
        interpolation="bilinear",
    )
    ax.contour(x, y, p_mask.astype(float), levels=[0.5], colors=["w"], linewidths=0.9)
    rows = [r for r in ant.get("instances", []) if isinstance(r, dict)]
    for r in rows:
        cx = float(r.get("center_x", 0.0))
        cy = float(r.get("center_y", 0.0))
        ax.plot([cx], [cy], marker="o", ms=5, mfc="#22c55e", mec="black", mew=0.5)
        size_mm = float(r.get("size_mm", 0.0))
        ax.text(cx, cy, f"{size_mm:.2f}mm", fontsize=6.5, color="white", ha="left", va="bottom")
    cb = plt.colorbar(im, ax=ax, shrink=0.86)
    cb.set_label("Qrf [MW/m^3]")
    ax.set_title("Antennae Preview: EQS/Qrf + candidate anchors")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return {"ok": True, "image_path": str(output_png), "instances": rows, "n_instances": len(rows)}


def generate_antennae_quick_search(cfg: dict) -> dict:
    """Run EQS and return suggested antenna placements plus a Qrf heatmap as a base64 PNG data URL."""
    import base64
    import io as _io

    if not _antennae_enabled(cfg):
        return {"ok": False, "error": "antennae disabled"}

    # Resolve instances using current antennae config (auto-placement)
    work = copy.deepcopy(cfg)
    ant = _resolve_antennae_instances(work)
    instances = [r for r in ant.get("instances", []) if isinstance(r, dict)]

    # Compute Qrf on base geometry (no antennas) for visualization
    base = copy.deepcopy(cfg)
    base["antennae"] = dict(base.get("antennae", {})) if isinstance(base.get("antennae", {}), dict) else {}
    base["antennae"]["enabled"] = False
    if "spike" in base and isinstance(base["spike"], dict):
        base["spike"]["enabled"] = False
    for p in _parts_from_geometry(base.get("geometry", {})):
        if isinstance(p, dict):
            p.pop("antennae_instances", None)

    x, y, _p_mask, qrf = _compute_eqs_qrf_for_cfg(base)
    qmw = qrf / 1.0e6
    qf = qmw[np.isfinite(qmw) & (qmw > 0)]
    qmin = float(np.min(qf)) if qf.size else 0.0
    qmax = float(np.max(qf)) if qf.size else 1.0
    if math.isclose(qmin, qmax):
        qmax = qmin + 1e-6

    # Render heatmap only (no axes, no colorbar) for overlay use
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=120)
    ax.imshow(
        qmw,
        extent=[float(x[0]) * 1000, float(x[-1]) * 1000, float(y[0]) * 1000, float(y[-1]) * 1000],
        origin="lower",
        cmap="magma",
        vmin=qmin,
        vmax=qmax,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    heatmap_b64 = base64.b64encode(buf.read()).decode("ascii")

    geom = cfg.get("geometry", {})
    chamber_x = float(geom.get("chamber_x", 0.06))
    chamber_y = float(geom.get("chamber_y", 0.06))

    return {
        "ok": True,
        "instances": instances,
        "n_instances": len(instances),
        "heatmap_data_url": f"data:image/png;base64,{heatmap_b64}",
        "qrf_min_mw_m3": qmin,
        "qrf_max_mw_m3": qmax,
        "chamber_x_m": chamber_x,
        "chamber_y_m": chamber_y,
    }


def _compute_eqs_proxy(cfg: dict) -> dict:
    x, y, _, part_mask, doped_mask, elec_hi, elec_lo, fill_frac, part_id_mask, _ = make_domain(cfg)
    geom = cfg["geometry"]
    dx = float(geom["chamber_x"]) / max(int(geom["grid_nx"]) - 1, 1)
    dy = float(geom["chamber_y"]) / max(int(geom["grid_ny"]) - 1, 1)
    elec = cfg["electric"]
    mat = cfg["materials"]
    sigma_v = max(float(mat["virgin"]["sigma_s_per_m"]), 1e-8)
    eps_v = float(mat["virgin"]["eps_r"])
    sigma_d0 = float(mat["doped"]["sigma_s_per_m"])
    eps_d = float(mat["doped"]["eps_r"])
    sigma = sigma_v + fill_frac * (sigma_d0 - sigma_v)
    eps_r = eps_v + fill_frac * (eps_d - eps_v)
    omega = 2.0 * math.pi * float(elec["frequency_hz"])
    v0 = float(elec["voltage_v"])
    mode = str(elec.get("voltage_mode", "centered")).strip().lower()
    if mode == "grounded":
        v_hi, v_lo = v0, 0.0
    elif mode == "single_ended":
        v_hi, v_lo = v0, 0.0
    else:
        v_hi, v_lo = 0.5 * v0, -0.5 * v0
    _, _, _, _, _, Qrf = solve_electric_state(
        sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo, dx, dy, elec, x, y,
        float(elec.get("power_factor", 1.0)), float(elec.get("max_qrf_w_per_m3", 1.0e11))
    )
    dA = dx * dy
    if bool(elec.get("zero_qrf_outside_doped", True)):
        Qrf = np.where(doped_mask, Qrf, 0.0)
    means: list[float] = []
    p10s: list[float] = []
    for pid in sorted(int(v) for v in np.unique(part_id_mask) if int(v) > 0):
        m = part_id_mask == pid
        if not np.any(m):
            continue
        q = np.asarray(Qrf[m], dtype=float)
        means.append(float(np.mean(q)))
        p10s.append(float(np.percentile(q, 10.0)))
    mean_q = float(np.mean(means)) if means else 0.0
    std_q = float(np.std(means)) if means else 0.0
    min_p10 = float(min(p10s)) if p10s else 0.0
    # Fairness-first proxy: prioritize lower inter-part spread, then preserve low-tail and mean.
    score = (-1.0 * std_q) + (0.20 * min_p10) + (0.05 * mean_q)
    return {
        "proxy_score": score,
        "proxy_fairness_std_qrf": std_q,
        "proxy_mean_qrf": mean_q,
        "proxy_std_qrf": std_q,
        "proxy_min_p10_qrf": min_p10,
        "proxy_power_w_per_m": float(np.sum(Qrf[doped_mask]) * dA),
    }


def _generate_layouts(
    px_half: float,
    py_half: float,
    radius: float,
    clearance: float,
    n: int,
    seed: int = 7,
    n_parts: int = 4,
) -> list[list[tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    out: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    d_min = 2.0 * radius + clearance

    x_lo = -float(px_half)
    x_hi = float(px_half)
    y_lo = -float(py_half)
    y_hi = float(py_half)

    def _is_valid(points: list[tuple[float, float]], x: float, y: float) -> bool:
        if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            return False
        for px, py in points:
            if math.hypot(x - px, y - py) < d_min:
                return False
        return True

    def _signature(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        # Round and sort so permutations are considered the same layout.
        return tuple(sorted((round(float(x), 5), round(float(y), 5)) for x, y in points))

    tries = 0
    max_tries = max(2000, n * 600)
    while len(out) < n and tries < max_tries:
        tries += 1
        pts: list[tuple[float, float]] = []
        inner_tries = 0
        while len(pts) < n_parts and inner_tries < 400:
            inner_tries += 1
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))
            if _is_valid(pts, x, y):
                pts.append((x, y))
        if len(pts) != n_parts:
            continue
        sig = _signature(pts)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(pts)
    return out


def _layout_signature(layout: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple(sorted((round(float(x), 5), round(float(y), 5)) for x, y in layout))


def _is_layout_valid(
    layout: list[tuple[float, float]],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    d_min: float,
) -> bool:
    for i, (x, y) in enumerate(layout):
        if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            return False
        for j in range(i + 1, len(layout)):
            x2, y2 = layout[j]
            if math.hypot(x - x2, y - y2) < d_min:
                return False
    return True


def _sample_layout_from_gaussian(
    rng: np.random.Generator,
    mu: np.ndarray,
    sigma: np.ndarray,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    d_min: float,
    tries: int = 120,
) -> list[tuple[float, float]] | None:
    n_parts = int(mu.shape[0])
    for _ in range(max(1, tries)):
        pts = rng.normal(mu, sigma)
        pts[:, 0] = np.clip(pts[:, 0], x_lo, x_hi)
        pts[:, 1] = np.clip(pts[:, 1], y_lo, y_hi)
        layout = [(float(p[0]), float(p[1])) for p in pts]
        if _is_layout_valid(layout, x_lo, x_hi, y_lo, y_hi, d_min):
            return layout
    return None


def run_placement_optimizer(cfg: dict, output_dir: Path) -> dict:
    po = cfg.get("placement_optimizer", {}) if isinstance(cfg.get("placement_optimizer", {}), dict) else {}
    constraints = po.get("constraints", {}) if isinstance(po.get("constraints", {}), dict) else {}
    n_parts = int(po.get("n_parts", 4))
    if n_parts < 1:
        raise ValueError("placement_optimizer requires n_parts >= 1")
    ga_cfg = po.get("ga", {}) if isinstance(po.get("ga", {}), dict) else {}
    algorithm = str(po.get("algorithm", "ga")).strip().lower()
    if algorithm not in {"ga", "cem", "random"}:
        algorithm = "ga"

    allowed_shapes = {
        "square", "rectangle", "circle", "ellipse", "triangle", "equilateral_triangle",
        "diamond", "hexagon", "octagon", "pentagon", "star", "star6", "star8",
        "cross", "rounded_rect", "l_shape", "t_shape", "trapezoid",
    }
    use_turntable = bool(po.get("use_turntable", False))
    clearance = float(po.get("clearance_mm", 1.0)) * 1e-3
    margin = float(po.get("search_domain_margin_mm", 2.0)) * 1e-3
    proxy_top_k = int(po.get("proxy_top_k", 8))
    proxy_eval_budget = int(po.get("proxy_eval_budget", max(80, int(ga_cfg.get("population", 32)) * max(int(ga_cfg.get("generations", 8)), 1))))

    base = copy.deepcopy(cfg)
    base.pop("placement_optimizer", None)
    base.pop("orientation_optimizer", None)
    base.pop("optimizer", None)
    if not use_turntable:
        base.pop("turntable", None)

    part_src = base.get("geometry", {}).get("part", {})
    shape_inherited = str(part_src.get("shape", cfg.get("geometry", {}).get("part", {}).get("shape", "circle"))).strip().lower()
    if shape_inherited not in allowed_shapes:
        raise ValueError(f"unsupported placement shape: {shape_inherited}")
    width = float(po.get("part_width_mm", float(part_src.get("width", 0.020)) * 1e3)) * 1e-3
    height = float(po.get("part_height_mm", float(part_src.get("height", width)) * 1e3)) * 1e-3
    radius = 0.5 * math.hypot(width, height)
    chamber_x = float(base["geometry"]["chamber_x"])
    chamber_y = float(base["geometry"]["chamber_y"])
    px_half = 0.5 * chamber_x - radius - margin
    py_half = 0.5 * chamber_y - radius - margin
    x_lo = -float(px_half)
    x_hi = float(px_half)
    y_lo = -float(py_half)
    y_hi = float(py_half)
    d_min = 2.0 * radius + clearance
    if x_lo >= x_hi or y_lo >= y_hi:
        raise ValueError("No feasible placement domain after margins and part size")

    def _layout_to_cfg(layout: list[tuple[float, float]]) -> dict:
        c = copy.deepcopy(base)
        c["geometry"]["parts"] = []
        for cx, cy in layout:
            p_out = {
                "shape": str(shape_inherited),
                "width": width,
                "height": height,
                "center_x": float(cx),
                "center_y": float(cy),
                "rotation_deg": 0.0,
            }
            if shape_inherited in {"circle", "disk", "ellipse", "star", "star6", "star8"}:
                p_out["n_circle_pts"] = int(part_src.get("n_circle_pts", 220))
            c["geometry"]["parts"].append(p_out)
        c["geometry"].pop("part", None)
        return c

    seed = int(ga_cfg.get("seed", po.get("seed", 7)))
    rng = np.random.default_rng(seed)
    pop_n = int(ga_cfg.get("population", po.get("proxy_population", 32)))
    generations = int(ga_cfg.get("generations", po.get("proxy_iters", 8)))
    mutation_rate = float(ga_cfg.get("mutation_rate", 0.18))
    crossover_rate = float(ga_cfg.get("crossover_rate", 0.75))
    elitism = int(ga_cfg.get("elitism", 2))
    tournament_k = int(ga_cfg.get("tournament_k", 3))

    proxy_rows: list[dict] = []
    next_candidate_id = 0
    eval_cache: dict[tuple[tuple[float, float], ...], dict] = {}

    def _eval_layout(layout: list[tuple[float, float]]) -> dict:
        nonlocal next_candidate_id
        sig = _layout_signature(layout)
        if sig in eval_cache:
            return eval_cache[sig]
        c = _layout_to_cfg(layout)
        pxm = _compute_eqs_proxy(c)
        row = {"candidate_id": int(next_candidate_id), "layout": [(float(a), float(b)) for a, b in layout], **pxm}
        next_candidate_id += 1
        eval_cache[sig] = row
        proxy_rows.append(row)
        return row

    def _random_layout() -> list[tuple[float, float]] | None:
        lay = _generate_layouts(px_half, py_half, radius, clearance, n=1, seed=int(rng.integers(1, 10_000_000)), n_parts=n_parts)
        return lay[0] if lay else None

    def _repair_layout(layout: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
        if _is_layout_valid(layout, x_lo, x_hi, y_lo, y_hi, d_min):
            return layout
        pts = [tuple(p) for p in layout]
        for _ in range(80):
            i = int(rng.integers(0, len(pts)))
            x0, y0 = pts[i]
            xn = float(np.clip(x0 + rng.normal(0.0, 0.12 * (x_hi - x_lo)), x_lo, x_hi))
            yn = float(np.clip(y0 + rng.normal(0.0, 0.12 * (y_hi - y_lo)), y_lo, y_hi))
            pts[i] = (xn, yn)
            if _is_layout_valid(list(pts), x_lo, x_hi, y_lo, y_hi, d_min):
                return list(pts)
        rr = _random_layout()
        return rr

    def _tournament(pop_rows: list[dict]) -> dict:
        kk = min(max(2, tournament_k), len(pop_rows))
        idxs = rng.choice(len(pop_rows), size=kk, replace=False)
        cand = [pop_rows[int(i)] for i in idxs]
        return sorted(cand, key=lambda r: float(r.get("proxy_score", -1e30)), reverse=True)[0]

    if algorithm in {"ga", "cem"}:
        pop: list[list[tuple[float, float]]] = _generate_layouts(px_half, py_half, radius, clearance, n=max(pop_n, 8), seed=seed, n_parts=n_parts)
        while len(pop) < max(pop_n, 8):
            rr = _random_layout()
            if rr is None:
                break
            pop.append(rr)
        if not pop:
            raise RuntimeError("placement optimizer could not initialize feasible layouts")
        print(f"  [placement][ga] population={len(pop)} generations={max(1, generations)}")
        for gen in range(max(1, generations)):
            if len(proxy_rows) >= proxy_eval_budget:
                break
            pop_rows = []
            for lay in pop:
                if len(proxy_rows) >= proxy_eval_budget:
                    break
                pop_rows.append(_eval_layout(lay))
            if not pop_rows:
                break
            ranked = sorted(pop_rows, key=lambda r: float(r.get("proxy_score", -1e30)), reverse=True)
            next_pop: list[list[tuple[float, float]]] = [list(r["layout"]) for r in ranked[: max(1, elitism)]]
            mut_sigma = max(1e-6, 0.16 * (x_hi - x_lo) * (1.0 - (gen / max(1, generations))))
            while len(next_pop) < pop_n and len(proxy_rows) < proxy_eval_budget:
                p1 = _tournament(ranked)
                p2 = _tournament(ranked)
                a = np.asarray(p1["layout"], dtype=float)
                b = np.asarray(p2["layout"], dtype=float)
                if float(rng.random()) < crossover_rate:
                    alpha = float(rng.uniform(0.2, 0.8))
                    c = alpha * a + (1.0 - alpha) * b
                else:
                    c = np.array(a, copy=True)
                for i in range(c.shape[0]):
                    if float(rng.random()) < mutation_rate:
                        c[i, 0] += float(rng.normal(0.0, mut_sigma))
                        c[i, 1] += float(rng.normal(0.0, mut_sigma))
                c[:, 0] = np.clip(c[:, 0], x_lo, x_hi)
                c[:, 1] = np.clip(c[:, 1], y_lo, y_hi)
                child = [(float(v[0]), float(v[1])) for v in c]
                fixed = _repair_layout(child)
                if fixed is None:
                    continue
                next_pop.append(fixed)
            pop = next_pop
    else:
        n_candidates = int(po.get("n_candidates", 48))
        print(f"  [placement][proxy-random] evaluating {n_candidates} layouts")
        pop = _generate_layouts(px_half, py_half, radius, clearance, n=n_candidates, seed=seed, n_parts=n_parts)
        for lay in pop:
            if len(proxy_rows) >= proxy_eval_budget:
                break
            _eval_layout(lay)

    proxy_rows = sorted(proxy_rows, key=lambda r: r["proxy_score"], reverse=True)
    if not proxy_rows:
        raise RuntimeError("placement optimizer could not generate any feasible candidate layouts")
    top_rows = proxy_rows[: max(1, min(proxy_top_k, len(proxy_rows)))]

    coupled_rows: list[dict] = []
    for i, r in enumerate(top_rows):
        print(f"  [placement][coupled] {i+1}/{len(top_rows)} candidate_id={r['candidate_id']}")
        c = _layout_to_cfg(r["layout"])
        _, summary, _, _, _ = run_sim(c)
        m = _candidate_metrics_from_summary(summary, constraints)
        m.update({"candidate_id": int(r["candidate_id"]), "layout": r["layout"], "proxy_score": float(r["proxy_score"])})
        coupled_rows.append(m)

    if coupled_rows:
        rho_proxy = [float(r["proxy_score"]) for r in coupled_rows]
        rho_coupled = [float(r["mean_rho"]) for r in coupled_rows]
        sp = _spearman(rho_proxy, rho_coupled)
        top_proxy = [r["candidate_id"] for r in sorted(coupled_rows, key=lambda x: x["proxy_score"], reverse=True)[:3]]
        top_coupled = [r["candidate_id"] for r in sorted(coupled_rows, key=lambda x: x["mean_rho"], reverse=True)[:3]]
        overlap = len(set(top_proxy).intersection(set(top_coupled)))
    else:
        sp = 0.0
        overlap = 0

    gate_pass = bool(sp >= float(po.get("proxy_gate_spearman_min", 0.8)) and overlap >= int(po.get("proxy_gate_top3_overlap_min", 2)))
    pareto = _pareto_front(coupled_rows)
    rows_rank = pareto if pareto else coupled_rows
    eligible = [r for r in rows_rank if float(r.get("temp_violation", 0.0)) <= 0.0 and float(r.get("min_rho_violation", 0.0)) <= 0.0]
    if not eligible:
        eligible = rows_rank
    best = sorted(
        eligible,
        key=lambda r: (
            float(r.get("temp_violation", 0.0)),
            float(r.get("min_rho_violation", 0.0)),
            float(r.get("std_rho", 1e9)),
            -float(r.get("min_rho", 0.0)),
            -float(r.get("mean_rho", 0.0)),
        ),
    )[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "placement_candidates_proxy.json").write_text(json.dumps(proxy_rows, indent=2))
    (output_dir / "placement_topk_coupled.json").write_text(json.dumps(coupled_rows, indent=2))
    (output_dir / "placement_pareto.json").write_text(json.dumps(pareto, indent=2))
    placement_best = dict(best)
    placement_best["proxy_gate_pass"] = gate_pass
    placement_best["proxy_gate_spearman"] = sp
    placement_best["proxy_gate_top3_overlap"] = overlap
    placement_best["algorithm"] = algorithm
    placement_best["shape"] = str(shape_inherited)
    placement_best["part_width_mm"] = float(width * 1e3)
    placement_best["part_height_mm"] = float(height * 1e3)
    placement_best["n_parts"] = int(n_parts)
    placement_best["use_turntable"] = bool(use_turntable)
    (output_dir / "placement_best.json").write_text(json.dumps(placement_best, indent=2))
    temp_ceiling_c = float(constraints.get("temp_ceiling_c", 250.0))
    exposure_s = _get_exposure_seconds(base)
    placement_context = (
        f"fixed exposure: {exposure_s:.0f} s; "
        f"temperature ceiling: {temp_ceiling_c:.1f} C; "
        f"parts={n_parts}, size={width*1e3:.1f}x{height*1e3:.1f} mm, turntable={'on' if use_turntable else 'off'}"
    )
    _render_scatter_report(
        coupled_rows,
        pareto,
        best,
        output_dir / "placement_report.png",
        "Placement Optimizer",
        context_text=placement_context,
    )

    final_cfg = copy.deepcopy(base)
    final_cfg["geometry"]["parts"] = []
    for idx_part, (cx, cy) in enumerate(best["layout"]):
        p_out = {
            "shape": str(shape_inherited),
            "width": width,
            "height": height,
            "center_x": float(cx),
            "center_y": float(cy),
            "rotation_deg": 0.0,
        }
        if shape_inherited in {"circle", "disk", "ellipse", "star", "star6", "star8"}:
            p_out["n_circle_pts"] = int(part_src.get("n_circle_pts", 220))
        final_cfg["geometry"]["parts"].append(
            p_out
        )
    final_cfg["geometry"].pop("part", None)
    state, summary, hist, tt_steps, opt_data = run_sim(final_cfg)
    summary["placement_optimizer"] = placement_best
    save_outputs(final_cfg, state, summary, hist, output_dir, tt_rotation_steps=tt_steps, opt_data=opt_data)
    return placement_best


def main() -> None:
    p = argparse.ArgumentParser(description="RFAM EQS -> transient thermal/phase/densification simulator")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./outputs_eqs/run"))
    args = p.parse_args()

    cfg = load_config(args.config)
    if bool(cfg.get("orientation_optimizer", {}).get("enabled", False)):
        best = run_orientation_optimizer(cfg, args.output_dir)
        print(f"Wrote outputs to: {args.output_dir.resolve()}")
        print(json.dumps({"orientation_best": best}, indent=2))
        return
    if bool(cfg.get("placement_optimizer", {}).get("enabled", False)):
        best = run_placement_optimizer(cfg, args.output_dir)
        print(f"Wrote outputs to: {args.output_dir.resolve()}")
        print(json.dumps({"placement_best": best}, indent=2))
        return
    state, summary, hist, tt_rotation_steps, opt_data = run_sim(cfg)
    save_outputs(
        cfg, state, summary, hist, args.output_dir,
        tt_rotation_steps=tt_rotation_steps,
        opt_data=opt_data,
    )
    generate_optimizer_report(cfg, opt_data, hist, args.output_dir)

    print(f"Wrote outputs to: {args.output_dir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
