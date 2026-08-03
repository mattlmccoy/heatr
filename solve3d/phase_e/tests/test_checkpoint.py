"""Phase E: the solve checkpoint must make an interruption cost ONE iteration.

Tested against a synthetic quadratic so it runs in milliseconds -- the point is
the resume BOOKKEEPING, not the physics.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d.phase_e import checkpoint as ck


def _quad(n=12, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random(n) + 1.0
    target = 0.3 * rng.random(n)

    def fg(v):
        d = v - target
        return float(np.dot(A, d * d)), 2.0 * A * d
    return fg, n


def test_checkpoint_round_trips_every_field(tmp_path):
    p = tmp_path / "ck.npz"
    c = ck.SolveCheckpoint(p, budget=5)
    assert c.load() is None
    c.save(n=2, best_J=1.5, best_v=np.array([0.1, 0.2]), scale=3.0,
           hist=[{"eval": 1, "J": 2.0}, {"eval": 2, "J": 1.5}])
    st = c.load()
    assert st["n"] == 2 and st["best_J"] == pytest.approx(1.5)
    assert np.allclose(st["best_v"], [0.1, 0.2])
    assert st["scale"] == pytest.approx(3.0)
    assert len(st["hist"]) == 2 and st["hist"][1]["J"] == pytest.approx(1.5)


def test_done_is_true_only_once_the_budget_is_spent(tmp_path):
    p = tmp_path / "ck.npz"
    c = ck.SolveCheckpoint(p, budget=3)
    assert not c.done
    c.save(n=2, best_J=1.0, best_v=np.zeros(2), scale=1.0, hist=[])
    assert not c.done
    c.save(n=3, best_J=1.0, best_v=np.zeros(2), scale=1.0, hist=[])
    assert c.done


def test_an_interrupted_run_resumes_and_does_not_lose_its_best(tmp_path):
    """The whole point: kill it after k evaluations, resume, and the total
    budget is still honoured while the best-so-far is carried forward."""
    fg, n = _quad()
    p = tmp_path / "ck.npz"
    budget = 10

    # first leg: stop hard after 4 evaluations
    r1 = ck.run_with_checkpoint(fg, np.ones(n), budget=budget, path=p,
                                bounds=(0.0, 1.0), stop_after=4)
    st = ck.SolveCheckpoint(p, budget).load()
    assert st["n"] == 4 and r1["interrupted"] is True

    # second leg: resume
    r2 = ck.run_with_checkpoint(fg, np.ones(n), budget=budget, path=p,
                                bounds=(0.0, 1.0))
    assert r2["evals_used"] == budget
    assert r2["resumed_from_eval"] == 4
    assert r2["best_J"] <= st["best_J"] + 1e-12      # never worse than the checkpoint
    assert len(r2["hist"]) == budget                  # history is continuous


def test_resume_from_a_finished_checkpoint_does_no_work(tmp_path):
    fg, n = _quad()
    p = tmp_path / "ck.npz"
    ck.run_with_checkpoint(fg, np.ones(n), budget=4, path=p, bounds=(0.0, 1.0))
    calls = {"n": 0}

    def counting(v):
        calls["n"] += 1
        return fg(v)
    r = ck.run_with_checkpoint(counting, np.ones(n), budget=4, path=p,
                               bounds=(0.0, 1.0))
    assert calls["n"] == 0
    assert r["already_complete"] is True


def test_a_fresh_run_with_checkpointing_matches_one_without(tmp_path):
    """Flag-off equivalence: writing a checkpoint must not perturb the search."""
    fg, n = _quad()
    a = ck.run_with_checkpoint(fg, np.ones(n), budget=8,
                               path=tmp_path / "a.npz", bounds=(0.0, 1.0))
    b = ck.run_with_checkpoint(fg, np.ones(n), budget=8, path=None,
                               bounds=(0.0, 1.0))
    assert a["best_J"] == pytest.approx(b["best_J"], rel=1e-12)
    assert np.allclose(a["best_v"], b["best_v"], rtol=1e-12)
