"""Stage B B2: uniform hold-out feasibility probe + penalty objective + verdicts.

The physics-bearing test (uniform hold-out peak) runs a REAL densify forward, so
the CONTRACT test here uses a small hold-out mesh for speed; the fine-mesh number
that bounds B2 feasibility is produced by the standalone probe run and recorded in
solve3d/results/stage_b_uniform_holdout.json (data-contract discipline: the test
checks the dict shape, the real run checks reality).
"""
from __future__ import annotations

import numpy as np


def test_uniform_holdout_peak_shape():
    from solve3d import stage_b
    # small hold-out keeps the contract check to ~20 s; the fine-mesh number is
    # the standalone probe run.
    out = stage_b.uniform_holdout_peak(holdout_nodes=800, holdout_lc0=0.060 / 24.0)
    assert set(out) >= {"true_peak_c", "ceiling_c", "feasible", "holdout_nodes_in_part"}
    assert out["ceiling_c"] == 250.0
    assert isinstance(out["feasible"], bool)


def test_is_shippable_reads_true_peak_not_ks():
    from solve3d import stage_b
    # KS (smooth) under ceiling but TRUE peak over -> NOT shippable
    v = stage_b.shippable_verdict(true_peak_c=251.0, ks_peak_c=249.0,
                                  ceiling_c=250.0, fd_gate_passed=True)
    assert v["is_shippable"] is False and v["reason"] == "over_ceiling_true_peak"


def test_is_shippable_requires_fd_gate():
    from solve3d import stage_b
    v = stage_b.shippable_verdict(true_peak_c=245.0, ks_peak_c=244.0,
                                  ceiling_c=250.0, fd_gate_passed=False)
    assert v["is_shippable"] is False and v["reason"] == "fd_gate_not_passed"


def test_is_shippable_true_when_fd_and_under_ceiling():
    from solve3d import stage_b
    v = stage_b.shippable_verdict(true_peak_c=245.0, ks_peak_c=244.0,
                                  ceiling_c=250.0, fd_gate_passed=True)
    assert v["is_shippable"] is True and v["reason"] == "shippable"


def test_honest_null_when_min_peak_over_ceiling():
    from solve3d import stage_b
    v = stage_b.null_verdict(best_true_peak_c=252.0, ceiling_c=250.0)
    assert v["verdict"] == "no_feasible_dopant_at_this_drive"
