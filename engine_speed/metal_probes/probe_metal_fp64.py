"""Probe: does the Metal runtime shader compiler on this machine accept fp64?

Evidence collector for the heatr3d Metal march port decision. Compiles three
MSL sources against the live MTLDevice and records the compiler's verdict.
"""
from __future__ import annotations

import json
import platform
import sys

import Metal

SRC_FLOAT = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    a[i] = a[i] * 2.0f + 1.0f;
}
"""

SRC_DOUBLE = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device double* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    a[i] = a[i] * 2.0 + 1.0;
}
"""

SRC_LONG_DOUBLE_SCALAR = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    double x = (double)a[i];
    a[i] = (float)(x * 2.0);
}
"""


def try_compile(dev, name: str, src: str) -> dict:
    lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
    return {
        "name": name,
        "compiled": lib is not None,
        "error": None if err is None else str(err),
    }


def main() -> int:
    dev = Metal.MTLCreateSystemDefaultDevice()
    if dev is None:
        print("NO METAL DEVICE")
        return 2
    fams = {}
    for label, val in [
        ("Apple7", 1007), ("Apple8", 1008), ("Apple9", 1009),
        ("Common3", 3003), ("Metal3", 5001),
    ]:
        try:
            fams[label] = bool(dev.supportsFamily_(val))
        except Exception as exc:  # noqa: BLE001
            fams[label] = f"query_failed:{type(exc).__name__}"
    out = {
        "machine": platform.platform(),
        "python": sys.version.split()[0],
        "device_name": str(dev.name()),
        "has_unified_memory": bool(dev.hasUnifiedMemory()),
        "max_threads_per_threadgroup": int(dev.maxThreadsPerThreadgroup().width),
        "recommended_max_working_set_bytes": int(dev.recommendedMaxWorkingSetSize()),
        "families": fams,
        "compiles": [
            try_compile(dev, "float32_kernel", SRC_FLOAT),
            try_compile(dev, "float64_buffer_kernel", SRC_DOUBLE),
            try_compile(dev, "float64_local_scalar", SRC_LONG_DOUBLE_SCALAR),
        ],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
