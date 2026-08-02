"""Phase B Task 5: Griewank-style checkpointing of the reverse march."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def tcase():
    from solve3d import transient_gate
    return transient_gate.build_case()


def test_gradient_identical(tcase):
    """Interval checkpoint/recompute must reproduce the store-everything
    gradient. Relative ~1e-12, never array_equal."""
    from solve3d import transient_gate
    doc = transient_gate.run_checkpoint_gate(tcase)
    for k, r in doc["intervals"].items():
        assert r["max_rel_diff"] < 1e-12, (k, r)


def test_memory_falls_and_cost_target_still_holds(tcase):
    from solve3d import transient_gate
    doc = transient_gate.run_checkpoint_gate(tcase)
    best = doc["chosen"]
    assert best["stored_state_bytes"] < doc["store_everything"]["stored_state_bytes"]
    assert best["forward_equivalents"] <= 2.0, best
