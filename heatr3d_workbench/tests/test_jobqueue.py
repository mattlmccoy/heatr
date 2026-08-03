"""On-disk job queue: state machine + persistence (server-side, stdlib-only).

The queue survives server restarts: a running job whose PID is dead and whose
run dir has results.json is 'done'; dead PID without results is 'stale_failed'
(loud, never silently healthy).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from heatr3d_workbench import jobqueue as JQ


@pytest.fixture()
def q(tmp_path):
    return JQ.Queue(tmp_path / "store")


def test_enqueue_creates_run_dir_with_config(q):
    jid = q.enqueue({"source": "parametric", "shape": "sphere", "n": 32})
    d = q.run_dir(jid)
    assert d.is_dir()
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["shape"] == "sphere"
    assert cfg["out_dir"] == str(d)
    assert q.get(jid)["state"] == "queued"


def test_fifo_next_and_single_slot(q):
    a = q.enqueue({"shape": "a"}); b = q.enqueue({"shape": "b"})
    assert q.next_queued() == a
    q.mark_running(a, pid=os.getpid())
    assert q.claimable() is False              # default max_concurrent=1
    assert q.next_queued() == b                # still queued behind the slot


def test_done_when_results_exist(q):
    jid = q.enqueue({"shape": "s"})
    q.mark_running(jid, pid=os.getpid())
    (q.run_dir(jid) / "results.json").write_text("{}")
    q.refresh()
    assert q.get(jid)["state"] == "done"


def test_dead_pid_without_results_is_stale_failed(q):
    jid = q.enqueue({"shape": "s"})
    q.mark_running(jid, pid=99999999)          # certainly dead
    q.refresh()
    assert q.get(jid)["state"] == "stale_failed"


def test_cancel_marks_cancelled_distinct_from_failed(q):
    jid = q.enqueue({"shape": "s"})
    q.cancel(jid)
    assert q.get(jid)["state"] == "cancelled"


def test_persistence_across_instances(q, tmp_path):
    jid = q.enqueue({"shape": "s"})
    q2 = JQ.Queue(tmp_path / "store")
    assert q2.get(jid)["state"] == "queued"


def test_campaign_grouping(q):
    a = q.enqueue({"shape": "a"}, campaign="lib-sweep")
    b = q.enqueue({"shape": "b"}, campaign="lib-sweep")
    ids = [j["id"] for j in q.jobs() if j.get("campaign") == "lib-sweep"]
    assert ids == [a, b]
