"""Honest, march-only benchmark: heatr3d.run vs engine_speed.march_fast.

WHAT IS MEASURED: the thermal/phase/densification march ONLY. Both engines are
driven with the SAME frozen Q_rf via ``qrf_override``, which bypasses the EQS
solve in both, so the reported ms/step is the march loop and nothing else. The
EQS LU factorisation and back-solves are a separate, one-time cost that this
prototype does not touch and does not claim to speed up.

Q_rf provenance: a real EQS solve at n <= 48 (affordable), and a synthetic
uniform in-part drive renormalised to the same absorbed-power target at n = 96
(a real n=96 EQS is ~322 s and would not change the per-step march cost). The
``qrf_sensitivity`` check runs n=48 both ways and reports the ms/step
difference, so the substitution is evidenced rather than assumed.

THREADS: single-threaded on both sides. The numba kernels contain no prange;
run this with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 as
the machine-wide compute convention requires. The measured thread count is
recorded in the JSON.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import heatr3d as h3

from .march_fast import march_fast

_P = h3.Params(phase_update="enthalpy")


def _synthetic_qrf(grid: h3.Grid, part: np.ndarray, p: h3.Params) -> np.ndarray:
    """Uniform in-part drive at the same fixed absorbed-power target that
    compute_qrf_3d enforces. Same total power, same support, flat pattern."""
    Q = np.zeros(part.shape, np.float64)
    Q[part] = 1.0
    p_target = p.power_density_w_per_m3 * (int(part.sum()) * grid.dV)
    Q *= p_target / (Q.sum() * grid.dV)
    return Q


def _real_qrf(grid: h3.Grid, part: np.ndarray, p: h3.Params) -> np.ndarray:
    gamma = h3.build_gamma(part, p, None)
    V = h3.solve_eqs_3d(gamma, grid, p)
    return h3.compute_qrf_3d(V, gamma, grid, p, part, premix=False,
                             qrf_gradient="masked")


def _time_march(fn, grid, part, p, Qrf, max_time_s) -> tuple[float, int, h3.Result]:
    t0 = time.perf_counter()
    res = fn(grid, part, p, sat=None, max_time_s=max_time_s, phi_target=0.90,
             densify=True, stop_mean_rho=None, qrf_override=Qrf)
    wall = time.perf_counter() - t0
    nsteps = int(max_time_s / p.dt_s)
    return wall, nsteps, res


def bench_one(n: int, max_time_s: float, qrf_mode: str = "auto") -> dict[str, Any]:
    grid = h3.Grid(n=n)
    part = h3.make_geometry(grid, "square", diam=0.020, zspan=0.020)
    p = _P
    if qrf_mode == "auto":
        qrf_mode = "eqs" if n <= 48 else "synthetic"
    t0 = time.perf_counter()
    Qrf = _real_qrf(grid, part, p) if qrf_mode == "eqs" else _synthetic_qrf(grid, part, p)
    t_qrf = time.perf_counter() - t0

    # warm the numba cache OUTSIDE the timed region (compile cost is one-time
    # per interpreter/cache generation and is reported separately)
    tiny = h3.Grid(n=8)
    tpart = h3.make_geometry(tiny, "square", diam=0.020, zspan=0.020)
    t0 = time.perf_counter()
    march_fast(tiny, tpart, p, max_time_s=2 * p.dt_s, densify=True,
               stop_mean_rho=None, qrf_override=_synthetic_qrf(tiny, tpart, p))
    t_warm = time.perf_counter() - t0

    w_ref, nsteps, r_ref = _time_march(h3.run, grid, part, p, Qrf, max_time_s)
    w_fast, _, r_fast = _time_march(march_fast, grid, part, p, Qrf, max_time_s)
    ident = bool(np.array_equal(r_ref.T_final, r_fast.T_final)
                 and np.array_equal(r_ref.rho_final, r_fast.rho_final))
    return {
        "n": n, "cells": int(np.prod(part.shape)), "nsteps": nsteps,
        "qrf_mode": qrf_mode, "qrf_build_s": t_qrf,
        "numba_warmup_s": t_warm,
        "heatr3d_wall_s": w_ref, "march_fast_wall_s": w_fast,
        "heatr3d_ms_per_step": 1e3 * w_ref / nsteps,
        "march_fast_ms_per_step": 1e3 * w_fast / nsteps,
        "speedup": w_ref / w_fast,
        "bit_identical_T_and_rho": ident,
    }


def project_30k(ms_per_step: float, nsteps: int = 30_000) -> float:
    return ms_per_step * nsteps / 1e3


def main(out_path: str | Path | None = None) -> dict[str, Any]:
    import numba
    recs = {
        "n48": bench_one(48, max_time_s=60.0),          # 1200 steps, real EQS Qrf
        "n96": bench_one(96, max_time_s=5.0),           # 100 steps, synthetic Qrf
        "n48_synthetic_qrf": bench_one(48, max_time_s=20.0, qrf_mode="synthetic"),
    }
    recs["qrf_sensitivity_pct"] = 100.0 * abs(
        recs["n48_synthetic_qrf"]["march_fast_ms_per_step"]
        - recs["n48"]["march_fast_ms_per_step"]) / recs["n48"]["march_fast_ms_per_step"]
    recs["projection_n96_30k_steps_s"] = {
        "heatr3d": project_30k(recs["n96"]["heatr3d_ms_per_step"]),
        "march_fast": project_30k(recs["n96"]["march_fast_ms_per_step"]),
    }
    recs["env"] = {
        "numba": numba.__version__, "numpy": np.__version__,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        "numba_config_num_threads": int(numba.config.NUMBA_NUM_THREADS),
        "uptime": os.popen("uptime").read().strip(),
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(recs, indent=2, default=float))
    return recs


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    r = main(here / "bench_results.json")
    for key in ("n48", "n96", "n48_synthetic_qrf"):
        b = r[key]
        print(f"{key:20s} n={b['n']:3d} steps={b['nsteps']:5d} qrf={b['qrf_mode']:9s} "
              f"heatr3d={b['heatr3d_ms_per_step']:8.3f} ms/step  "
              f"fast={b['march_fast_ms_per_step']:8.4f} ms/step  "
              f"speedup={b['speedup']:6.1f}x  bitident={b['bit_identical_T_and_rho']}")
    print("Qrf-pattern sensitivity of march ms/step at n=48: "
          f"{r['qrf_sensitivity_pct']:.2f} %")
    print("Projected n=96, 30k-step march wall time: "
          f"heatr3d {r['projection_n96_30k_steps_s']['heatr3d']:.0f} s  ->  "
          f"march_fast {r['projection_n96_30k_steps_s']['march_fast']:.0f} s")
