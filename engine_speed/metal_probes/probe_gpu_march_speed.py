"""How much headroom was actually on the table, had float32 been acceptable?

Times ONE reduced march substep -- the same six-face harmonic conduction,
convection, enthalpy inversion and clamps as engine_speed/fp32_cost.py -- in:
  * numpy float64 on the CPU (the shape of heatr3d's own vectorised march)
  * MLX float32 on the M2 Pro GPU
  * MLX float32 on the MLX CPU backend (isolates GPU dispatch from MLX itself)

The numba float64 kernel's measured ms/step is quoted from SPEED_REPORT.md, not
re-measured here (numba is not installed in this probe venv).
"""
from __future__ import annotations

import json
import time

import mlx.core as mx
import numpy as np

# Physical constants lifted from heatr3d.Params defaults.
K_POW, RHO_POW, CP_POW = 0.197, 490.0, 1072.0
K_SOL, RHO_SOL, CP_SOL = 0.10, 460.0, 2500.0
K_LIQ, RHO_LIQ, CP_LIQ = 0.26, 1010.0, 3279.0
LATENT, T_PC, DT_PC = 96700.0, 180.0, 10.0
CONV_H, PREHEAT = 5.0, 23.0
MAX_DT, TMIN, TMAX = 10.0, -50.0, 600.0
RHO_REL = 0.55


def make_case(n: int):
    L = 0.060
    h = L / n
    c = (np.arange(n) + 0.5) * h - L / 2.0
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    part = (np.abs(X) <= 0.010) & (np.abs(Y) <= 0.010) & (np.abs(Z) <= 0.010)
    Qrf = np.zeros(part.shape)
    Qrf[part] = 20.0 * 10.0 / (np.pi * 0.010 ** 2 * 0.020)
    T = np.full(part.shape, PREHEAT)
    return h, part, Qrf, T


# --------------------------------------------------------------------------- #
def step_numpy(T, part, Qrf, h, dt, d=np.float64):
    phi = np.clip((T - T_PC) / DT_PC + 0.5, 0.0, 1.0)
    rho_s = RHO_POW + RHO_REL * (RHO_SOL - RHO_POW)
    k_s = K_POW + RHO_REL * (K_SOL - K_POW)
    rho = np.where(part, (1 - phi) * rho_s + phi * RHO_LIQ, RHO_POW)
    k = np.where(part, (1 - phi) * k_s + phi * K_LIQ, K_POW)
    cp = np.where(part, (1 - phi) * CP_SOL + phi * CP_LIQ, CP_POW)
    div = np.zeros_like(T)
    for ax in range(3):
        for s in (-1, +1):
            k_nb = np.roll(k, -s, axis=ax)
            T_nb = np.roll(T, -s, axis=ax)
            den = k + k_nb
            kf = np.where(np.abs(den) > 1e-30,
                          2.0 * k * k_nb / np.where(den == 0, 1, den),
                          0.5 * den)
            flux = kf * (T_nb - T) / (h * h)
            sl = [slice(None)] * 3
            sl[ax] = (-1 if s == +1 else 0)
            flux[tuple(sl)] = 0.0
            div = div + flux
    q_conv = np.zeros_like(T)
    q_conv[:, -1, :] = CONV_H * (T[:, -1, :] - PREHEAT) / h
    num = div + Qrf - q_conv
    rho_cp = rho * cp
    rho_L = np.where(part, rho_s * LATENT, 0.0)
    lo = T_PC - DT_PC / 2.0
    frac = np.clip((T - lo) / DT_PC, 0.0, 1.0)
    H = rho_cp * T + rho_L * frac + dt * np.nan_to_num(num)
    T_new = np.where(H <= rho_cp * lo, H / rho_cp,
                     np.where(H >= rho_cp * (lo + DT_PC) + rho_L,
                              (H - rho_L) / rho_cp,
                              (H + rho_L * lo / DT_PC) / (rho_cp + rho_L / DT_PC)))
    dT = np.clip(T_new - T, -MAX_DT, MAX_DT)
    return np.clip(T + dT, TMIN, TMAX)


def step_mlx(T, part, Qrf, h, dt):
    phi = mx.clip((T - T_PC) / DT_PC + 0.5, 0.0, 1.0)
    rho_s = RHO_POW + RHO_REL * (RHO_SOL - RHO_POW)
    k_s = K_POW + RHO_REL * (K_SOL - K_POW)
    rho = mx.where(part, (1 - phi) * rho_s + phi * RHO_LIQ, RHO_POW)
    k = mx.where(part, (1 - phi) * k_s + phi * K_LIQ, K_POW)
    cp = mx.where(part, (1 - phi) * CP_SOL + phi * CP_LIQ, CP_POW)
    div = mx.zeros_like(T)
    for ax in range(3):
        for s in (-1, +1):
            k_nb = mx.roll(k, -s, axis=ax)
            T_nb = mx.roll(T, -s, axis=ax)
            den = k + k_nb
            kf = mx.where(mx.abs(den) > 1e-30, 2.0 * k * k_nb / den, 0.5 * den)
            flux = kf * (T_nb - T) / (h * h)
            # zero the wrap slice
            n = T.shape[ax]
            idx = (n - 1) if s == +1 else 0
            mask = (mx.arange(n) != idx).astype(T.dtype)
            shape = [1, 1, 1]
            shape[ax] = n
            flux = flux * mask.reshape(shape)
            div = div + flux
    ny = T.shape[1]
    topmask = (mx.arange(ny) == ny - 1).astype(T.dtype).reshape(1, ny, 1)
    q_conv = topmask * (CONV_H * (T - PREHEAT) / h)
    num = div + Qrf - q_conv
    rho_cp = rho * cp
    rho_L = mx.where(part, rho_s * LATENT, 0.0)
    lo = T_PC - DT_PC / 2.0
    frac = mx.clip((T - lo) / DT_PC, 0.0, 1.0)
    H = rho_cp * T + rho_L * frac + dt * num
    T_new = mx.where(H <= rho_cp * lo, H / rho_cp,
                     mx.where(H >= rho_cp * (lo + DT_PC) + rho_L,
                              (H - rho_L) / rho_cp,
                              (H + rho_L * lo / DT_PC) / (rho_cp + rho_L / DT_PC)))
    dT = mx.clip(T_new - T, -MAX_DT, MAX_DT)
    return mx.clip(T + dT, TMIN, TMAX)


def bench(n: int, reps: int = 20) -> dict:
    h, part, Qrf, T = make_case(n)
    dt = 0.05
    out = {"n": n, "cells": int(part.size)}

    for _ in range(2):
        step_numpy(T, part, Qrf, h, dt)
    t0 = time.perf_counter()
    for _ in range(reps):
        step_numpy(T, part, Qrf, h, dt)
    out["ms_numpy_f64"] = (time.perf_counter() - t0) / reps * 1e3

    for label, dev in (("gpu", mx.gpu), ("cpu", mx.cpu)):
        with mx.stream(dev):
            Tm = mx.array(T.astype(np.float32))
            pm = mx.array(part)
            Qm = mx.array(Qrf.astype(np.float32))
            for _ in range(3):
                r = step_mlx(Tm, pm, Qm, h, dt)
                mx.eval(r)
            t0 = time.perf_counter()
            for _ in range(reps):
                r = step_mlx(Tm, pm, Qm, h, dt)
                mx.eval(r)
            out[f"ms_mlx_f32_{label}"] = (time.perf_counter() - t0) / reps * 1e3
    # correctness sanity: the mlx gpu step must land near the numpy f64 step
    ref = step_numpy(T, part, Qrf, h, dt)
    with mx.stream(mx.gpu):
        got = np.array(step_mlx(mx.array(T.astype(np.float32)), mx.array(part),
                                mx.array(Qrf.astype(np.float32)), h, dt),
                       copy=False).astype(np.float64)
    out["max_rel_dev_one_step_gpu_f32_vs_numpy_f64"] = float(
        np.max(np.abs(got - ref) / np.abs(ref)))
    return out


if __name__ == "__main__":
    res = [bench(48), bench(96)]
    print(json.dumps({"mlx_version": "0.32.0", "numpy_version": np.__version__,
                      "benchmarks": res}, indent=2))
