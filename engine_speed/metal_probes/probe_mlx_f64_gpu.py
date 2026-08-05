"""Probe: MLX exposes mx.float64 -- but will it EXECUTE on the GPU stream?

The interesting failure mode is a silent CPU fallback: the call succeeds, the
dtype says float64, and the user believes the GPU ran it. Distinguish.
"""
from __future__ import annotations

import json
import time

import mlx.core as mx
import numpy as np

A = np.linspace(1.0, 2.0, 1 << 20, dtype=np.float64)
out: dict = {"mlx_default_device": str(mx.default_device())}


def kern(t):
    a, b = t[:-1], t[1:]
    return 2.0 * a * b / (a + b)


for label, dev in (("gpu", mx.gpu), ("cpu", mx.cpu)):
    rec: dict = {}
    try:
        with mx.stream(dev):
            t = mx.array(A, dtype=mx.float64)
            r = kern(t)
            mx.eval(r)
        rec["ok"] = True
        rec["dtype"] = str(r.dtype)
        got = np.array(r, copy=False).astype(np.float64)
        ref = kern(A)
        rec["max_rel_dev_vs_numpy_f64"] = float(
            np.max(np.abs(got - ref) / np.abs(ref)))
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    out[f"float64_on_{label}_stream"] = rec

# Timing contrast: if "gpu float64" is really CPU, it will not beat cpu float64,
# while gpu float32 will show the real GPU throughput.
def bench(dev, dtype, reps: int = 20) -> float:
    with mx.stream(dev):
        t = mx.array(A, dtype=dtype)
        mx.eval(t)
        r = kern(t)
        mx.eval(r)
        t0 = time.perf_counter()
        for _ in range(reps):
            r = kern(t)
            mx.eval(r)
        return (time.perf_counter() - t0) / reps * 1e3


for label, dev, dt in (("gpu_f32", mx.gpu, mx.float32),
                       ("cpu_f32", mx.cpu, mx.float32),
                       ("gpu_f64", mx.gpu, mx.float64),
                       ("cpu_f64", mx.cpu, mx.float64)):
    try:
        out[f"ms_{label}"] = bench(dev, dt)
    except Exception as exc:  # noqa: BLE001
        out[f"ms_{label}"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(out, indent=2))
