"""What the thermal march would cost in float32 -- measured, not asserted.

An Apple-silicon GPU offers no float64 (see ``march_metal.py``: Metal Shading
Language has no ``double`` type, and torch-MPS / MLX both refuse float64 on the
GPU). So the only Metal march anyone could actually ship is a float32 march.
This module measures what that would do to the numbers.

METHOD
------
A REDUCED march: the heatr3d thermal update with densification, powder loss,
heat sinks, power scheduling and in-march EQS re-solves all switched off, driven
by a fixed ``qrf_override`` field. Everything that remains -- harmonic face
conductivity, the six-face divergence, top-face convection, the enthalpy phase
inversion, the THM-01 dT cap and the THM-02 temperature clamp -- is written in
the same association order as ``heatr3d.run``, parameterised on dtype.

It is not a stand-in on trust. ``fidelity_vs_heatr3d`` marches the float64 arm
against ``heatr3d.run`` on the identical configuration and reports the measured
deviation; the test suite gates it at 1e-12.

WHY numpy float32 IS THE RIGHT PROXY FOR THE GPU
------------------------------------------------
Measured on this machine (probe recorded in metal_probe_results.json): on the
harmonic-face expression, torch-MPS float32 and MLX-GPU float32 are BIT-IDENTICAL
to numpy float32 (0 ULP difference on every element). Both GPU backends and numpy
are doing IEEE-754 binary32 with the same rounding, so the float32 arm here is
the arithmetic a Metal kernel would perform. If anything it FLATTERS the GPU
path, which would additionally be free to contract multiply-add pairs.

WHAT TO LOOK AT
---------------
Not only the field deviation. The standing parity gate compares
``clamp_bound``, ``n_substeps_used`` and ``cfl_violated`` for EXACT equality,
and those are driven by float comparisons against thresholds. The record below
counts the DISCRETE decisions that flip: enthalpy branch selection, dT-cap hits,
temperature-clamp hits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import heatr3d as h3

__all__ = [
    "SUITE",
    "ProbeConfig",
    "build_probe_config",
    "dtype_deviation",
    "fidelity_vs_heatr3d",
    "reduced_march",
    "run_suite",
]

# The probe suite. Deliberately mirrors the shape of engine_speed/cases.py: it
# is not a set of benign configurations. Two cases exist ONLY to make the
# guards fire, because a precision claim measured where every limiter is
# dormant says nothing about the fields the standing gate compares for EXACT
# equality.
SUITE: tuple[dict[str, Any], ...] = (
    {"name": "benign", "n": 32, "n_steps": 200, "power_gain": 1.0,
     "dt_s": None,
     "description": "production drive, sub-melt: the easy path"},
    {"name": "melt_window", "n": 32, "n_steps": 150, "power_gain": 20.0,
     "dt_s": None,
     "description": "melt front straddling the phase-change window, so the "
                    "enthalpy branch selection is live on ~1000 cells"},
    {"name": "melt_crossed", "n": 32, "n_steps": 200, "power_gain": 20.0,
     "dt_s": None,
     "description": "most of the part through the window, mixed branches"},
    {"name": "cfl_substep", "n": 32, "n_steps": 20, "power_gain": 20.0,
     "dt_s": 3.0,
     "description": "dt_s above the CFL bound: THM-03 substepping engages"},
    {"name": "clamp_temp", "n": 32, "n_steps": 400, "power_gain": 100.0,
     "dt_s": None,
     "description": "THM-02 temp_max clamp BINDS"},
    {"name": "clamp_dt", "n": 32, "n_steps": 60, "power_gain": 1000.0,
     "dt_s": None,
     "description": "THM-01 per-step dT cap BINDS"},
)


@dataclass(frozen=True)
class ProbeConfig:
    """One fully specified reduced-march configuration.

    ``Qrf`` is fixed for the whole march (supplied to heatr3d as
    ``qrf_override``), so both engines are driven by the identical source field
    and no EQS solve enters the comparison.
    """

    grid: Any
    part: np.ndarray
    params: Any
    Qrf: np.ndarray
    n_steps: int

    @property
    def max_time_s(self) -> float:
        return self.n_steps * self.params.dt_s


def build_probe_config(n: int = 32, n_steps: int = 60,
                       power_gain: float = 20.0,
                       dt_s: float | None = None) -> ProbeConfig:
    """A cube in the middle of the bed, driven hard enough to cross the melt
    window so the enthalpy branch selection is actually exercised.

    ``power_gain`` above 1 pushes part of the domain through the phase-change
    window within ``n_steps``; a march that never melts would measure float32
    on the easy path only. ``dt_s`` above the CFL bound engages the THM-03
    substepping, mirroring the ``cfl_substep`` gate case.
    """
    from dataclasses import replace

    grid = h3.Grid(n=n)
    part = h3.make_geometry(grid, "square", diam=0.020, zspan=0.020)
    params = h3.Params(phase_update="enthalpy")
    if dt_s is not None:
        params = replace(params, dt_s=float(dt_s))
    q0 = params.power_density_w_per_m3 * power_gain
    Qrf = np.zeros(part.shape, dtype=np.float64)
    Qrf[part] = q0
    return ProbeConfig(grid=grid, part=part, params=params, Qrf=Qrf,
                       n_steps=int(n_steps))


def _harmonic(a: np.ndarray, b: np.ndarray, d: Any) -> np.ndarray:
    den = a + b
    return np.where(np.abs(den) > d(1e-30),
                    d(2.0) * a * b / np.where(den == 0, d(1), den),
                    d(0.5) * (a + b))


def reduced_march(cfg: ProbeConfig, dtype=np.float64) -> dict[str, Any]:
    """March ``cfg`` in ``dtype`` and return the final state plus the discrete
    decision counters.

    Returns a dict with ``T`` and ``phi`` (float64 copies for comparison), the
    per-run clamp counters, and the per-cell enthalpy branch code of the LAST
    step (0 = below window, 1 = in window, 2 = above window).
    """
    d = np.dtype(dtype).type
    p = cfg.params
    grid = cfg.grid
    part = cfg.part
    h = d(grid.h)

    n_sub = h3.cfl_substeps(grid, p) if p.enforce_cfl else 1
    dt_sub = d(p.dt_s / n_sub)

    t_pc = d(p.t_pc_c)
    dt_pc = d(p.dt_pc_c)
    lo = d(p.t_pc_c - p.dt_pc_c / 2.0)
    max_dt = d(p.max_dt_step_c)
    tmin = d(p.temp_min_c)
    tmax = d(p.temp_max_c)
    latent = d(p.latent_j_per_kg)

    T = np.full(part.shape, d(p.preheat_c), dtype=dtype)
    Qrf = cfg.Qrf.astype(dtype)
    rho_rel = np.full(part.shape, d(p.rho_rel), dtype=dtype)

    rho_s_eff = d(p.rho_powder) + rho_rel * d(p.rho_solid - p.rho_powder)
    k_s_eff = d(p.k_powder) + rho_rel * d(p.k_solid - p.k_powder)
    rho_L_map = np.zeros(part.shape, dtype=dtype)
    rho_L_map[part] = rho_s_eff[part] * latent

    top = [slice(None)] * 3
    top[1] = -1
    top = tuple(top)

    n_dT_clip = 0
    n_temp_clip = 0
    branch = np.zeros(part.shape, dtype=np.int8)
    has_part = bool(part.any())
    phi_target = d(0.90)
    phi90_substep = -1          # -1 == never crossed within this march
    phi_hist: list[float] = []

    for _isub in range(cfg.n_steps * n_sub):
        arg = (T - t_pc) / dt_pc
        phi = np.clip(arg + d(0.5), d(0.0), d(1.0))

        rho = np.full(part.shape, d(p.rho_powder), dtype=dtype)
        k = np.full(part.shape, d(p.k_powder), dtype=dtype)
        cp = np.full(part.shape, d(p.cp_powder), dtype=dtype)
        rho[part] = (d(1) - phi[part]) * rho_s_eff[part] + phi[part] * d(p.rho_liquid)
        k[part] = (d(1) - phi[part]) * k_s_eff[part] + phi[part] * d(p.k_liquid)
        cp[part] = (d(1) - phi[part]) * d(p.cp_solid) + phi[part] * d(p.cp_liquid)

        div = np.zeros(part.shape, dtype=dtype)
        for ax in range(3):
            for s in (-1, +1):
                k_nb = np.roll(k, -s, axis=ax)
                T_nb = np.roll(T, -s, axis=ax)
                kf = _harmonic(k, k_nb, d)
                flux = kf * (T_nb - T) / (h * h)
                sl = [slice(None)] * 3
                sl[ax] = (-1 if s == +1 else 0)
                flux[tuple(sl)] = d(0.0)
                div += flux

        q_conv = np.zeros(part.shape, dtype=dtype)
        q_conv[top] = d(p.conv_h) * (np.array(T[top], copy=True) - d(p.preheat_c)) / h

        num = np.array(div, copy=True)
        num += Qrf
        num -= q_conv

        rho_cp = rho * cp
        frac = np.clip((T - lo) / dt_pc, d(0.0), d(1.0))
        H = rho_cp * T + rho_L_map * frac
        H = H + dt_sub * np.nan_to_num(num)
        H_lo = rho_cp * lo
        H_hi = rho_cp * (lo + dt_pc) + rho_L_map
        T_below = H / rho_cp
        T_window = (H + rho_L_map * lo / dt_pc) / (rho_cp + rho_L_map / dt_pc)
        T_above = (H - rho_L_map) / rho_cp
        T_new = np.where(H <= H_lo, T_below, np.where(H >= H_hi, T_above, T_window))
        branch = np.where(H <= H_lo, 0, np.where(H >= H_hi, 2, 1)).astype(np.int8)

        dT_raw = T_new - T
        dT = np.clip(dT_raw, -max_dt, max_dt)
        n_dT_clip += int(np.count_nonzero(np.abs(dT_raw) > max_dt))
        T_cand = np.array(T, copy=True) + dT
        n_temp_clip += int(np.count_nonzero((T_cand > tmax) | (T_cand < tmin)))
        T = np.clip(T_cand, tmin, tmax)

        # Melt-onset read, in the dtype of the march: heatr3d takes the mean
        # phase fraction over the PART and compares it to phi_target. It is a
        # threshold crossing, so it is where a small field shift turns into a
        # whole-substep shift in a reported time.
        phi_now = np.clip((T - t_pc) / dt_pc + d(0.5), d(0.0), d(1.0))
        mean_phi = phi_now[part].mean() if has_part else d(0.0)
        phi_hist.append(float(mean_phi))
        if phi90_substep < 0 and mean_phi >= phi_target:
            phi90_substep = _isub + 1

    phi_final = np.clip((T - t_pc) / dt_pc + d(0.5), d(0.0), d(1.0))
    return {
        "T": T.astype(np.float64),
        "phi": phi_final.astype(np.float64),
        "branch": branch,
        "n_dT_clip": n_dT_clip,
        "n_temp_clip": n_temp_clip,
        "n_substeps": int(n_sub),
        "phi90_substep": int(phi90_substep),
        "phi_hist": phi_hist,
        "dtype": np.dtype(dtype).name,
    }


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    """Same measure the parity gate uses: max |a-b| / max(|a|,|b|), skipping
    exactly equal pairs."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if np.all(same):
        return 0.0
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    return float(np.nanmax(np.where(same, 0.0, np.abs(a - b) / scale)))


def fidelity_vs_heatr3d(cfg: ProbeConfig) -> dict[str, Any]:
    """FIDELITY GATE: the float64 reduced march against heatr3d.run itself."""
    ref = h3.run(cfg.grid, cfg.part, cfg.params, sat=None,
                 max_time_s=cfg.max_time_s, phi_target=2.0, densify=False,
                 stop_mean_rho=None, qrf_override=cfg.Qrf)
    mine = reduced_march(cfg, dtype=np.float64)
    return {
        "grid_n": int(cfg.grid.n),
        "n_steps": int(cfg.n_steps),
        "n_substeps": mine["n_substeps"],
        "max_rel_dev_T": _max_rel(ref.T_final, mine["T"]),
        "max_rel_dev_phi": _max_rel(ref.phi_final, mine["phi"]),
        "bit_identical_T": bool(np.array_equal(ref.T_final, mine["T"])),
        "clamp_counts_match": bool(
            ref.clamp_bound == (mine["n_dT_clip"] > 0 or mine["n_temp_clip"] > 0)),
        "heatr3d_clamp_bound": bool(ref.clamp_bound),
        "reduced_n_dT_clip": mine["n_dT_clip"],
        "reduced_n_temp_clip": mine["n_temp_clip"],
    }


def dtype_deviation(cfg: ProbeConfig) -> dict[str, Any]:
    """Measure the float32 arm against the float64 arm on the same march."""
    a = reduced_march(cfg, dtype=np.float64)
    b = reduced_march(cfg, dtype=np.float32)
    n_cells = int(cfg.part.size)
    return {
        "grid_n": int(cfg.grid.n),
        "n_steps": int(cfg.n_steps),
        "n_substeps": a["n_substeps"],
        "n_cells": n_cells,
        "n_part_cells": int(cfg.part.sum()),
        "max_rel_dev_T": _max_rel(a["T"], b["T"]),
        "max_abs_dev_T_c": float(np.max(np.abs(a["T"] - b["T"]))),
        "max_rel_dev_phi": _max_rel(a["phi"], b["phi"]),
        "max_abs_dev_phi": float(np.max(np.abs(a["phi"] - b["phi"]))),
        "n_phi_cells_differing": int(np.count_nonzero(a["phi"] != b["phi"])),
        "n_enthalpy_branch_flips": int(np.count_nonzero(a["branch"] != b["branch"])),
        "n_dT_clip_f64": a["n_dT_clip"],
        "n_dT_clip_f32": b["n_dT_clip"],
        "n_temp_clip_f64": a["n_temp_clip"],
        "n_temp_clip_f32": b["n_temp_clip"],
        "clamp_bound_f64": bool(a["n_dT_clip"] > 0 or a["n_temp_clip"] > 0),
        "clamp_bound_f32": bool(b["n_dT_clip"] > 0 or b["n_temp_clip"] > 0),
        "mean_T_f64_c": float(a["T"].mean()),
        "mean_T_f32_c": float(b["T"].mean()),
        "mean_phi_part_f64": float(a["phi"][cfg.part].mean()),
        "mean_phi_part_f32": float(b["phi"][cfg.part].mean()),
        "phi90_substep_f64": a["phi90_substep"],
        "phi90_substep_f32": b["phi90_substep"],
        "phi90_substep_shift": (a["phi90_substep"] - b["phi90_substep"]),
        "max_rel_dev_phi_hist": _max_rel(np.array(a["phi_hist"]),
                                         np.array(b["phi_hist"])),
    }


def run_suite(with_fidelity: bool = True) -> dict[str, Any]:
    """Run every SUITE case and return the recorded records.

    ``with_fidelity`` also marches heatr3d.run itself on each configuration, so
    the report can quote the measured stand-in fidelity rather than assert it.
    """
    import platform
    import sys

    records = []
    for spec in SUITE:
        cfg = build_probe_config(n=spec["n"], n_steps=spec["n_steps"],
                                 power_gain=spec["power_gain"],
                                 dt_s=spec["dt_s"])
        rec = {"name": spec["name"], "description": spec["description"],
               "power_gain": spec["power_gain"], "dt_s": cfg.params.dt_s}
        rec.update(dtype_deviation(cfg))
        if with_fidelity:
            rec["fidelity"] = fidelity_vs_heatr3d(cfg)
        records.append(rec)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "cases": records,
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    from pathlib import Path

    out = run_suite()
    Path(__file__).with_name("fp32_cost_results.json").write_text(
        json.dumps(out, indent=2))
    for r in out["cases"]:
        print(f"{r['name']:14s} relT={r['max_rel_dev_T']:.3e} "
              f"absT={r['max_abs_dev_T_c']:.3e} C "
              f"branch_flips={r['n_enthalpy_branch_flips']} "
              f"phi_cells={r['n_phi_cells_differing']} "
              f"dTclip {r['n_dT_clip_f64']}/{r['n_dT_clip_f32']} "
              f"Tclip {r['n_temp_clip_f64']}/{r['n_temp_clip_f32']}")
