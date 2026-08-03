"""RED-first parity gate: march_fast must reproduce heatr3d.run.

This test IS the gate. It is deliberately slow (it runs the reference engine and
the fast engine on five n=32 cases) and is the only acceptable evidence that the
fast march is usable. Never widen FLOOR_RTOL to make it pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_speed.cases import build_cases
from engine_speed.gate import FLOOR_RTOL, run_case

_HERE = Path(__file__).resolve().parent
_OUT = _HERE.parent / "gate_results_pytest_n32.json"


@pytest.fixture(scope="module")
def gate_records():
    records = {c.name: run_case(c) for c in build_cases(32)}
    _OUT.write_text(json.dumps(records, indent=2, default=float))
    return records


@pytest.mark.parametrize("case_name", [c.name for c in build_cases(32)])
def test_fields_match_reference(gate_records, case_name):
    rec = gate_records[case_name]
    assert rec["pass"], (
        f"{case_name}: fields above the {FLOOR_RTOL:g} floor: {rec['failures']}\n"
        + json.dumps(rec["fields"], indent=2, default=float))


def test_cfl_guard_actually_fires(gate_records):
    rec = gate_records["cfl_substep"]
    assert rec["guards"]["cfl_ref"] > 1, "THM-03 substepping never engaged in heatr3d"
    assert rec["guards"]["cfl_fast"] == rec["guards"]["cfl_ref"]


def test_temperature_clamp_actually_fires(gate_records):
    rec = gate_records["clamp_hot"]
    assert rec["guards"]["clamp_ref"] is True, "clamp never bound in heatr3d"
    assert rec["guards"]["clamp_fast"] is True


def test_energy_residual_matches(gate_records):
    for name, rec in gate_records.items():
        f = rec["fields"]["energy_residual_frac"]
        assert f["pass"], f"{name}: energy_residual_frac dev {f['max_rel_dev']:g}"


def test_fast_march_has_not_blown_up(gate_records):
    """Blow-up guard ONLY -- deliberately not a speed measurement.

    These wall times include the EQS solve, which is identical work on both
    sides and dominates a gate case (~17 s EQS vs ~2 s march at n=32), so the
    fast/ref margin here is a few percent and flips on machine noise. A strict
    'fast < ref' assertion was tried and proved flaky on a loaded machine, and
    a flaky assertion inside a correctness gate is worse than none: it trains
    people to ignore red.

    The honest, EQS-excluded, load-checked speed numbers come from bench.py
    (5.9x at n=48, 6.3x at n=96 -- SPEED_REPORT.md section 3). All that is
    asserted here is that nothing catastrophic happened.
    """
    blown = {n: (r["wall_fast_s"], r["wall_ref_s"])
             for n, r in gate_records.items()
             if r["wall_fast_s"] > 2.0 * r["wall_ref_s"]}
    assert not blown, f"march_fast wall time blew up (fast, ref): {blown}"
