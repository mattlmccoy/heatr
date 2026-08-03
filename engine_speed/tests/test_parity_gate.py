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


def test_fast_march_is_faster(gate_records):
    """Sanity only -- the honest speed numbers come from bench.py. This just
    catches a fast march that is accidentally slower than the reference."""
    slow = [n for n, r in gate_records.items()
            if r["wall_fast_s"] >= r["wall_ref_s"]]
    assert not slow, f"march_fast not faster on: {slow}"
