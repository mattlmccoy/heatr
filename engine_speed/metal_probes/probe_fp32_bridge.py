"""Bridge check: is GPU float32 bit-identical to numpy float32 on the hot kernel?

If yes, numpy float32 is a faithful stand-in for the Metal path's arithmetic, so
the float32 numerics cost can be measured in the project venv (which has heatr3d)
without shipping a GPU dependency there.
"""
from __future__ import annotations

import json

import numpy as np

rng = np.random.default_rng(12345)
A64 = (0.05 + rng.random((32, 32, 32))).astype(np.float64)
A32 = A64.astype(np.float32)


def face_np(a, b, two, half):
    den = a + b
    q = two * a * b / den
    return np.where(np.abs(den) > 1e-30, q, half * (a + b))


ref64 = face_np(A64[:-1], A64[1:], np.float64(2.0), np.float64(0.5))
np32 = face_np(A32[:-1], A32[1:], np.float32(2.0), np.float32(0.5))

out = {"numpy_f32_vs_f64_max_rel": float(
    np.max(np.abs(np32.astype(np.float64) - ref64) / np.abs(ref64)))}

try:
    import torch
    t = torch.from_numpy(A32).to("mps")
    a, b = t[:-1], t[1:]
    den = a + b
    q = 2.0 * a * b / den
    r = torch.where(den.abs() > 1e-30, q, 0.5 * (a + b))
    got = r.cpu().numpy()
    out["torch_mps_f32_bitidentical_to_numpy_f32"] = bool(
        np.array_equal(got.view(np.int32), np32.view(np.int32)))
    out["torch_mps_f32_max_ulp_diff_vs_numpy_f32"] = int(
        np.max(np.abs(got.view(np.int32).astype(np.int64)
                      - np32.view(np.int32).astype(np.int64))))
except Exception as exc:  # noqa: BLE001
    out["torch"] = f"{type(exc).__name__}: {exc}"

try:
    import mlx.core as mx
    with mx.stream(mx.gpu):
        t = mx.array(A32)
        a, b = t[:-1], t[1:]
        den = a + b
        q = 2.0 * a * b / den
        r = mx.where(mx.abs(den) > 1e-30, q, 0.5 * (a + b))
        mx.eval(r)
    got = np.array(r, copy=False)
    out["mlx_gpu_f32_bitidentical_to_numpy_f32"] = bool(
        np.array_equal(got.view(np.int32), np32.view(np.int32)))
    out["mlx_gpu_f32_max_ulp_diff_vs_numpy_f32"] = int(
        np.max(np.abs(got.view(np.int32).astype(np.int64)
                      - np32.view(np.int32).astype(np.int64))))
except Exception as exc:  # noqa: BLE001
    out["mlx"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(out, indent=2))
