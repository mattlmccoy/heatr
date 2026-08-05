"""Probe: do torch-MPS and MLX carry float64 end to end on this GPU?

Each framework gets the same three questions:
  1. can a float64 tensor be placed on the GPU device at all?
  2. does a stencil-shaped arithmetic op run there in float64?
  3. what dtype comes back, and does the result match the numpy float64 answer?
"""
from __future__ import annotations

import json

import numpy as np

A = np.linspace(1.0, 2.0, 4096, dtype=np.float64).reshape(16, 16, 16)


def _numpy_ref() -> np.ndarray:
    """A harmonic-mean face expression, the shape of the real hot kernel."""
    a = A[:-1]
    b = A[1:]
    return 2.0 * a * b / (a + b)


REF = _numpy_ref()


def probe_torch() -> dict:
    out: dict = {"framework": "torch"}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {**out, "importable": False, "error": f"{type(exc).__name__}: {exc}"}
    out["importable"] = True
    out["version"] = torch.__version__
    out["mps_available"] = bool(torch.backends.mps.is_available())
    out["mps_built"] = bool(torch.backends.mps.is_built())
    try:
        t = torch.from_numpy(A).to("mps")
        out["float64_to_mps"] = {"ok": True, "dtype": str(t.dtype),
                                 "device": str(t.device)}
    except Exception as exc:  # noqa: BLE001
        out["float64_to_mps"] = {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"}
    try:
        t = torch.from_numpy(A).to("mps", dtype=torch.float64)
        a, b = t[:-1], t[1:]
        r = 2.0 * a * b / (a + b)
        got = r.cpu().numpy()
        out["float64_kernel_on_mps"] = {
            "ok": True, "dtype": str(r.dtype),
            "max_rel_dev_vs_numpy_f64": float(
                np.max(np.abs(got - REF) / np.abs(REF))),
        }
    except Exception as exc:  # noqa: BLE001
        out["float64_kernel_on_mps"] = {"ok": False,
                                        "error": f"{type(exc).__name__}: {exc}"}
    try:
        t = torch.from_numpy(A).to("mps", dtype=torch.float32)
        a, b = t[:-1], t[1:]
        r = 2.0 * a * b / (a + b)
        got = r.cpu().numpy().astype(np.float64)
        out["float32_kernel_on_mps"] = {
            "ok": True, "dtype": str(r.dtype),
            "max_rel_dev_vs_numpy_f64": float(
                np.max(np.abs(got - REF) / np.abs(REF))),
        }
    except Exception as exc:  # noqa: BLE001
        out["float32_kernel_on_mps"] = {"ok": False,
                                        "error": f"{type(exc).__name__}: {exc}"}
    return out


def probe_mlx() -> dict:
    out: dict = {"framework": "mlx"}
    try:
        import mlx.core as mx
    except Exception as exc:  # noqa: BLE001
        return {**out, "importable": False, "error": f"{type(exc).__name__}: {exc}"}
    out["importable"] = True
    try:
        import importlib.metadata as md
        out["version"] = md.version("mlx")
    except Exception:  # noqa: BLE001
        out["version"] = "unknown"
    out["has_float64_dtype_attr"] = hasattr(mx, "float64")
    out["default_device"] = str(mx.default_device())
    try:
        t = mx.array(A, dtype=getattr(mx, "float64"))
        mx.eval(t)
        out["float64_array"] = {"ok": True, "dtype": str(t.dtype)}
    except Exception as exc:  # noqa: BLE001
        out["float64_array"] = {"ok": False,
                                "error": f"{type(exc).__name__}: {exc}"}
    try:
        t = mx.array(A)  # whatever mlx picks for a float64 numpy input
        mx.eval(t)
        out["float64_numpy_ingest_dtype"] = str(t.dtype)
    except Exception as exc:  # noqa: BLE001
        out["float64_numpy_ingest_dtype"] = f"{type(exc).__name__}: {exc}"
    try:
        with mx.stream(mx.gpu):
            t = mx.array(A.astype(np.float32))
            a, b = t[:-1], t[1:]
            r = 2.0 * a * b / (a + b)
            mx.eval(r)
        got = np.array(r, copy=False).astype(np.float64)
        out["float32_kernel_on_gpu"] = {
            "ok": True, "dtype": str(r.dtype),
            "max_rel_dev_vs_numpy_f64": float(
                np.max(np.abs(got - REF) / np.abs(REF))),
        }
    except Exception as exc:  # noqa: BLE001
        out["float32_kernel_on_gpu"] = {"ok": False,
                                        "error": f"{type(exc).__name__}: {exc}"}
    return out


if __name__ == "__main__":
    print(json.dumps({"numpy_version": np.__version__,
                      "probes": [probe_torch(), probe_mlx()]}, indent=2))
