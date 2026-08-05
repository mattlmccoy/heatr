"""Metal/MPS backend gate for the heatr3d thermal march -- REFUSED, with evidence.

WHAT THIS MODULE IS
-------------------
Compute item (2) of the shrinkage-prewarp v2 spec asked for an Apple-silicon GPU
port of the march, gated the way the numba port in ``march_fast.py`` is gated but
at a MEASURED TOLERANCE FLOOR rather than bit-identity, because GPU floats
forfeit bit-identity.

The port was not written. The deciding question was asked FIRST, per the
data-contract rule, and the answer closes the door:

    Metal Shading Language has no ``double`` type. Not on the M2 Pro, not on any
    Apple silicon GPU. The runtime shader compiler on this machine rejects
    ``double`` in a buffer declaration AND as a local scalar:

        program_source:4:22: error: 'double' is not supported in Metal

    Every candidate framework compiles down to MSL, so none of them can offer
    float64 on this GPU. Measured on this machine, not inferred:
      * torch 2.13.0 MPS: TypeError -- "Cannot convert a MPS Tensor to float64
        dtype as the MPS framework doesn't support float64."
      * mlx 0.32.0: ``mx.float64`` exists but is CPU-only; on the GPU stream it
        raises ValueError -- "float64 is not supported on the GPU". Worse,
        ``mx.array(<float64 numpy array>)`` SILENTLY returns float32.

The alternative -- shipping the march in float32 -- is a numerics change this
project does not accept silently, and the measured cost is in ``fp32_cost.py``:
float32 does not merely miss the 1e-16 parity floor, it flips the DISCRETE
decisions the standing gate compares for EXACT equality (enthalpy branch
selection, THM-01/THM-02 clamp counters). See SPEED_REPORT.md section 7.

So this module ships the finding instead of the kernel: a live re-runnable probe,
a hard refusal, and provenance fields. It is OFF everywhere by construction --
nothing calls ``march_metal``, and if anything ever does it raises.

RE-RUNNING THE PROBE
--------------------
    pip install pyobjc-framework-Metal
    python -m engine_speed.march_metal

The probe degrades honestly: if the binding is missing it reports
``probe_error`` and ``metal_fp64_supported`` stays False. An unknown answer is
never reported as a healthy one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NoReturn

__all__ = [
    "MetalProbe",
    "MetalUnavailable",
    "march_metal",
    "metal_fp64_supported",
    "metal_provenance",
    "probe_dependency_available",
    "probe_metal_fp64",
]

# Minimal MSL sources. The float32 one is the control: if IT fails to compile,
# the probe is broken, not the GPU.
_SRC_FP32 = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    a[i] = a[i] * 2.0f + 1.0f;
}
"""

_SRC_FP64 = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device double* a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    a[i] = a[i] * 2.0 + 1.0;
}
"""


class MetalUnavailable(RuntimeError):
    """Raised whenever a Metal march is requested. There is no Metal march."""


@dataclass(frozen=True)
class MetalProbe:
    """One recorded answer from the live Metal runtime shader compiler.

    ``fp64_compiles is None`` means UNKNOWN (the probe could not run). It never
    means "fine".
    """

    device_reachable: bool
    device_name: str | None
    fp32_compiles: bool | None
    fp64_compiles: bool | None
    fp64_error: str | None
    probe_error: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_dependency_available() -> bool:
    """True iff the Metal binding needed by the live probe is importable."""
    try:
        import Metal  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def probe_metal_fp64() -> MetalProbe:
    """Ask the live Metal runtime compiler whether it accepts float64.

    Compiles two MSL sources against the default MTLDevice. Never raises: a
    failure to probe is recorded in ``probe_error`` and leaves the fp64 verdict
    UNKNOWN, which ``metal_fp64_supported`` reads as not supported.
    """
    try:
        import Metal
    except Exception as exc:  # noqa: BLE001
        return MetalProbe(False, None, None, None, None,
                          f"{type(exc).__name__}: {exc}")
    try:
        dev = Metal.MTLCreateSystemDefaultDevice()
    except Exception as exc:  # noqa: BLE001
        return MetalProbe(False, None, None, None, None,
                          f"{type(exc).__name__}: {exc}")
    if dev is None:
        return MetalProbe(False, None, None, None, None,
                          "MTLCreateSystemDefaultDevice returned None")

    def _compile(src: str) -> tuple[bool, str | None]:
        lib, err = dev.newLibraryWithSource_options_error_(src, None, None)
        return lib is not None, (None if err is None else str(err))

    try:
        fp32_ok, _ = _compile(_SRC_FP32)
        fp64_ok, fp64_err = _compile(_SRC_FP64)
    except Exception as exc:  # noqa: BLE001
        return MetalProbe(True, str(dev.name()), None, None, None,
                          f"{type(exc).__name__}: {exc}")
    return MetalProbe(device_reachable=True, device_name=str(dev.name()),
                      fp32_compiles=fp32_ok, fp64_compiles=fp64_ok,
                      fp64_error=fp64_err, probe_error=None)


def metal_fp64_supported(probe: MetalProbe | None = None) -> bool:
    """True ONLY on a positively confirmed float64 shader compile."""
    if probe is None:
        probe = probe_metal_fp64()
    return probe.probe_error is None and probe.fp64_compiles is True


def _refusal_reason(probe: MetalProbe) -> str:
    if probe.probe_error is not None:
        return (f"Metal float64 support could not be established "
                f"({probe.probe_error}); an unverified GPU is treated as "
                f"unsupported.")
    if probe.fp64_compiles is False:
        lines = [ln.strip() for ln in (probe.fp64_error or "").splitlines()
                 if ln.strip()]
        first = next((ln for ln in lines
                      if "error:" in ln or "not supported" in ln),
                     lines[0] if lines else "")
        return ("Metal Shading Language rejects float64 on this device"
                + (f": {first.strip()}" if first else "")
                + ". The heatr3d march is a float64 physics kernel and will not "
                  "be run in float32.")
    return "Metal float64 support is not confirmed."


def metal_provenance(probe: MetalProbe | None = None) -> dict[str, Any]:
    """Recorded-acceleration provenance fields for any run that TRIED the Metal
    path. Mirrors the numba path's provenance so a run record can always say
    which engine produced its numbers.
    """
    if probe is None:
        probe = probe_metal_fp64()
    return {
        "acceleration": ("metal" if metal_fp64_supported(probe)
                         else "metal_refused"),
        "metal_fp64_supported": bool(metal_fp64_supported(probe)),
        "metal_device": probe.device_name,
        "metal_refusal_reason": _refusal_reason(probe),
        "metal_probe": probe.as_dict(),
    }


def march_metal(grid, part, params, *args) -> NoReturn:
    """Placeholder for the Metal march. Always raises.

    Signature mirrors ``march_fast`` so a caller that wires this in fails loudly
    at the call site instead of quietly producing float32 physics. There is
    deliberately NO ``allow_float32`` option; passing one is a TypeError.
    """
    probe = probe_metal_fp64()
    raise MetalUnavailable(
        "No Metal thermal march exists. " + _refusal_reason(probe)
        + " See engine_speed/march_metal.py and SPEED_REPORT.md section 7 for "
          "the probe evidence and the measured float32 cost."
    )


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(metal_provenance(), indent=2))
