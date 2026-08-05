"""Tests for engine_speed.march_metal: the Metal/MPS march backend gate.

The module's whole job is to answer one question honestly -- can this GPU run
the heatr3d thermal march in float64? -- and to REFUSE rather than silently
downgrade the numerics when it cannot. These tests pin the refusal, the
no-false-green rule (a probe that could not run is never reported as
supported), and the provenance record.
"""
from __future__ import annotations

import pytest

from engine_speed import march_metal as mm


# --------------------------------------------------------------------------- #
# 1. no false green
# --------------------------------------------------------------------------- #
def test_unsupported_when_fp64_compile_fails():
    probe = mm.MetalProbe(device_reachable=True, device_name="Apple M2 Pro",
                          fp32_compiles=True, fp64_compiles=False,
                          fp64_error="'double' is not supported in Metal",
                          probe_error=None)
    assert mm.metal_fp64_supported(probe) is False


def test_unsupported_when_probe_could_not_run():
    """An unknown answer must read as NOT supported, never as healthy."""
    probe = mm.MetalProbe(device_reachable=False, device_name=None,
                          fp32_compiles=None, fp64_compiles=None,
                          fp64_error=None,
                          probe_error="pyobjc-framework-Metal not installed")
    assert mm.metal_fp64_supported(probe) is False


def test_supported_only_on_a_positive_fp64_compile():
    probe = mm.MetalProbe(device_reachable=True, device_name="Hypothetical",
                          fp32_compiles=True, fp64_compiles=True,
                          fp64_error=None, probe_error=None)
    assert mm.metal_fp64_supported(probe) is True


# --------------------------------------------------------------------------- #
# 2. refusal
# --------------------------------------------------------------------------- #
def test_march_metal_refuses_and_names_float64():
    with pytest.raises(mm.MetalUnavailable) as exc:
        mm.march_metal(None, None, None)
    msg = str(exc.value)
    assert "float64" in msg
    # A reason is always given: either the measured MSL rejection or the honest
    # "could not be established" when the probe binding is absent. Never silence.
    assert ("not supported" in msg or "rejects float64" in msg
            or "could not be established" in msg)


def test_march_metal_never_falls_back_to_float32():
    """There is no allow_float32 escape hatch. Shipping a float32 physics
    engine behind the same name is the failure mode this module exists to
    prevent."""
    with pytest.raises(TypeError):
        mm.march_metal(None, None, None, allow_float32=True)


# --------------------------------------------------------------------------- #
# 3. provenance
# --------------------------------------------------------------------------- #
def test_provenance_fields():
    probe = mm.MetalProbe(device_reachable=True, device_name="Apple M2 Pro",
                          fp32_compiles=True, fp64_compiles=False,
                          fp64_error="'double' is not supported in Metal",
                          probe_error=None)
    prov = mm.metal_provenance(probe)
    assert prov["acceleration"] == "metal_refused"
    assert prov["metal_fp64_supported"] is False
    assert prov["metal_device"] == "Apple M2 Pro"
    assert "double" in prov["metal_refusal_reason"]
    # JSON-quotable: every value survives a round trip.
    import json
    assert json.loads(json.dumps(prov)) == prov


def test_probe_is_jsonable():
    probe = mm.MetalProbe(device_reachable=True, device_name="Apple M2 Pro",
                          fp32_compiles=True, fp64_compiles=False,
                          fp64_error="err", probe_error=None)
    import json
    d = probe.as_dict()
    assert json.loads(json.dumps(d)) == d


# --------------------------------------------------------------------------- #
# 4. live probe (integration; needs pyobjc-framework-Metal)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not mm.probe_dependency_available(),
                    reason="pyobjc-framework-Metal not installed")
def test_live_probe_on_this_machine():
    probe = mm.probe_metal_fp64()
    assert probe.probe_error is None
    assert probe.device_reachable is True
    assert probe.fp32_compiles is True, "float32 shaders must compile"
    # The recorded finding on Apple silicon: MSL has no double type at all.
    assert probe.fp64_compiles is False
    assert "double" in (probe.fp64_error or "")
    assert mm.metal_fp64_supported(probe) is False
