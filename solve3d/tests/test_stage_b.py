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
