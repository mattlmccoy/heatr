"""RED-first gate for the predicted-benefit check (TAMPER_DIAGNOSIS.md fix 2c).

ACCEPTANCE EVIDENCE. Every reject-side case here is a REAL shipped job, loaded
from fixtures captured by capture_tamper_fixtures.py -- not authored numbers.
The Tamper (feb850ec) is the case that shipped a map zeroing dopant in 79 % of
the part; if this file ever goes green without it rejecting, the harm class is
open again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio3d.correction_gate import (GUARD_KEYS, evaluate_correction,
                                      out_of_part_melt_frac)

FIX = Path(__file__).resolve().parent / "fixtures" / "tamper"
MANIFEST = json.loads((FIX / "manifest.json").read_text())

# Real harm cases: all three complete before/after pairs in the archive.
HARM_JOBS = ["feb850ec", "c474d787", "d4d50045"]


def _pair(job: str):
    """(before, after) results dicts + the field stats the gate needs."""
    rec = MANIFEST["jobs"][job]
    before = json.loads((FIX / f"{job}_uncorrected_results.json").read_text())
    after = json.loads((FIX / f"{job}_corrected_results.json").read_text())
    # the runner did not record out-of-part melt when these jobs ran, so the
    # captured field-derived value is injected exactly as the new runner will
    before["out_of_part_melt_frac"] = \
        rec["arms"]["uncorrected"]["fields"]["out_of_part_melt_frac"]
    after["out_of_part_melt_frac"] = \
        rec["arms"]["corrected"]["fields"]["out_of_part_melt_frac"]
    return before, after


# --------------------------------------------------------------------------- #
# THE RED FIXTURES: real jobs that must be REJECTED
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("job", HARM_JOBS)
def test_real_harm_jobs_are_rejected(job):
    before, after = _pair(job)
    v = evaluate_correction(before, after)
    assert v["verdict"] == "REJECTED", (
        f"{job} shipped a worse-than-uniform map and the gate accepted it: "
        f"{json.dumps(v, indent=2)}")
    assert v["failed_guards"], "a rejection must name which guards failed"


def test_tamper_rejection_names_the_bed_spill_and_the_ceiling():
    """The Tamper's two hard failures: out-of-part melt 0.032 -> 0.175 (bed
    spill) and T_max 288.0 -> 448.4 C. Both must be named, not just one."""
    before, after = _pair("feb850ec")
    v = evaluate_correction(before, after)
    assert "out_of_part_melt" in v["failed_guards"]
    assert "T_ceiling" in v["failed_guards"]


def test_ceiling_guard_catches_worsening_when_before_ALREADY_failed():
    """Data-contract trap: the Tamper's BEFORE arm is already over the 250 C
    ceiling (T_max 288.0, T_ceiling_ok False). A naive
    'before.ok and not after.ok' regression test would NOT fire. The guard
    must compare the VALUES, so 288.0 -> 448.4 is caught."""
    before, after = _pair("feb850ec")
    assert before["gates"]["T_ceiling_ok"] is False
    assert after["gates"]["T_ceiling_ok"] is False
    v = evaluate_correction(before, after)
    assert "T_ceiling" in v["failed_guards"]


def test_horizon_cap_guard_fires_on_the_tamper():
    """t90 blew to the horizon: before reached the density stop at 1075.35 s
    of a 1500 s budget; after ran the full 1500 s."""
    before, after = _pair("feb850ec")
    v = evaluate_correction(before, after)
    assert "horizon_cap" in v["failed_guards"]


def test_aircoil_rejection_names_the_clamp():
    """AIRCOIL: sigma_T 36.9 -> 111.5 and the march hit the 600 C clamp."""
    before, after = _pair("c474d787")
    v = evaluate_correction(before, after)
    assert v["verdict"] == "REJECTED"
    assert "clamp_bound" in v["failed_guards"]


# --------------------------------------------------------------------------- #
# ACCEPT side
# --------------------------------------------------------------------------- #
def test_identical_arms_are_accepted():
    """A null correction reproduces the before arm exactly; nothing regressed,
    so the gate must not reject. (The archive contains NO accept-side
    before/after pair -- all three complete pairs are harm cases -- so the
    accept-side control is constructed from a real before arm compared with
    itself, which is exactly what a null correction produces.)"""
    before, _ = _pair("feb850ec")
    v = evaluate_correction(before, dict(before))
    assert v["verdict"] == "ACCEPTED", json.dumps(v, indent=2)
    assert not v["failed_guards"]


def test_a_genuine_improvement_is_accepted():
    """Improve every guard from a real before arm: must accept."""
    before, _ = _pair("feb850ec")
    after = json.loads(json.dumps(before))
    after["sigma_T"] = before["sigma_T"] * 0.7
    after["rho_final_std"] = before["rho_final_std"] * 0.8
    after["warp_std_pct"] = before["warp_std_pct"] * 0.7
    after["out_of_part_melt_frac"] = before["out_of_part_melt_frac"] * 0.5
    after["gates"] = dict(before["gates"], T_max_C=before["gates"]["T_max_C"] - 20.0)
    v = evaluate_correction(before, after)
    assert v["verdict"] == "ACCEPTED", json.dumps(v, indent=2)


def test_diagnostic_tripwires_alone_do_not_reject():
    """sigma_T / rho_final_std / warp_std_pct are DIAGNOSTIC class: they are
    recorded and flagged, but a modest diagnostic drift with every hard guard
    green must not by itself revert the correction to uniform."""
    before, _ = _pair("feb850ec")
    after = json.loads(json.dumps(before))
    after["sigma_T"] = before["sigma_T"] * 1.05      # 5 % worse, small
    v = evaluate_correction(before, after)
    assert v["verdict"] == "ACCEPTED"
    assert "sigma_T" in v["tripwires_worse"]


def test_large_diagnostic_regression_does_reject():
    """A tripled sigma_T is not a 'diagnostic drift'; past the pre-registered
    factor it becomes a rejection on its own."""
    before, _ = _pair("feb850ec")
    after = json.loads(json.dumps(before))
    after["sigma_T"] = before["sigma_T"] * 3.0
    v = evaluate_correction(before, after)
    assert v["verdict"] == "REJECTED"


# --------------------------------------------------------------------------- #
# auditability of the read convention
# --------------------------------------------------------------------------- #
def test_verdict_states_the_read_convention_explicitly():
    before, after = _pair("feb850ec")
    v = evaluate_correction(before, after)
    conv = v["read_convention"]
    assert "own stop state" in conv["description"]
    assert conv["before_stop"]["sim_time_s"] == before["sim_time_s"]
    assert conv["after_stop"]["sim_time_s"] == after["sim_time_s"]
    assert conv["matched"]["grid_n"] is True
    assert conv["matched"]["max_time_s"] is True
    assert conv["matched"]["stop_mean_rho"] is True


def test_mismatched_conventions_are_refused_not_compared():
    """Comparing arms run at different grids or horizons is not a benefit
    measurement. Refuse loudly rather than emit a meaningless verdict."""
    before, after = _pair("feb850ec")
    after = json.loads(json.dumps(after))
    after["grid_n"] = 32
    v = evaluate_correction(before, after)
    assert v["verdict"] == "REJECTED"
    assert "convention_mismatch" in v["failed_guards"]
    assert v["read_convention"]["matched"]["grid_n"] is False


def test_every_guard_key_is_reported_even_when_passing():
    before, after = _pair("feb850ec")
    v = evaluate_correction(before, after)
    for k in GUARD_KEYS:
        assert k in v["guards"], f"{k} missing from the recorded guard set"
        assert "before" in v["guards"][k] and "after" in v["guards"][k]


def test_missing_out_of_part_melt_is_refused_not_assumed_fine():
    """False-green rule: absence of the bed-spill measurement must not read as
    'no bed spill'."""
    before, after = _pair("feb850ec")
    before = json.loads(json.dumps(before))
    after = json.loads(json.dumps(after))
    before.pop("out_of_part_melt_frac")
    after.pop("out_of_part_melt_frac")
    v = evaluate_correction(before, after)
    assert v["guards"]["out_of_part_melt"]["status"] == "UNKNOWN"
    assert v["verdict"] == "REJECTED"
    assert "out_of_part_melt" in v["failed_guards"]


# --------------------------------------------------------------------------- #
# the helper the runner will use
# --------------------------------------------------------------------------- #
def test_out_of_part_melt_frac_matches_the_captured_value():
    import numpy as np
    rec = MANIFEST["jobs"]["feb850ec"]
    src = Path(rec["source"]) / "uncorrected" / "fields.npz"
    if not src.exists():
        pytest.skip("original job artifacts not present on this machine")
    with np.load(src) as d:
        got = out_of_part_melt_frac(np.asarray(d["phi_final"], float),
                                    d["part"].astype(bool))
    assert got == pytest.approx(
        rec["arms"]["uncorrected"]["fields"]["out_of_part_melt_frac"], rel=1e-12)


def test_out_of_part_melt_frac_is_none_when_shapes_disagree():
    import numpy as np
    phi = np.zeros((4, 4, 4))
    part = np.zeros((8, 8, 8), bool)
    assert out_of_part_melt_frac(phi, part) is None
