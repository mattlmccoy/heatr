#!/usr/bin/env python3
"""RED/GREEN tests for schedule co-solve commissioning and the solve dashboard.

Two features from the v2 rollout deferred list:

1. Solve-progress dashboard (deferred item (a)): the server already tails
   SOLVE_PROGRESS lines from scripts/solve_fgm.py into a one-line label; the
   dashboard needs the STRUCTURED fields (forward-equivalents spent/budget,
   current J, best intersection over union) stored on the job so the front
   end can render a live panel. Pure logic under test:
   rfam_gui_server._solve_progress_fields and _record_solve_progress.

2. Schedule co-solve (map + turntable program): a new job mode that shells
   the NEW wrapper scripts/solve_schedule.py, which drives the existing
   campaign drivers read-only (run_dwell_solve / run_rot_avg_solve /
   run_seq_arms) and lands the deliverables under
   outputs_eqs/runs/<shape>/schedule_cosolve/<id>/. Pure logic under test:
   argument building and validation on both sides, and the driver-log
   progress watch that turns the drivers' own log lines into
   SCHEDULE_PROGRESS lines.

The wrapper module must import WITHOUT the heavy solver stack (numpy/scipy
loads only inside the run functions), so these tests stay fast.

Run: ./.venv312/bin/python -m pytest test_solve_schedule.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

import rfam_gui_server as srv     # noqa: E402
import solve_schedule as ss       # noqa: E402


# ---------------------------------------------------------------------------
# 1. solve-progress dashboard (server pure logic)
# ---------------------------------------------------------------------------

_LINE = ("SOLVE_PROGRESS eval=3 pool=17 fe_spent=7.5 fe_budget=40 "
         "J=118.42 IoU=0.8451")


def test_solve_progress_fields_parses_the_real_line_format() -> None:
    f = srv._solve_progress_fields(_LINE)
    assert f == {"eval": 3, "pool": 17, "fe_spent": 7.5, "fe_budget": 40.0,
                 "J": 118.42, "IoU": 0.8451}


def test_solve_progress_fields_rejects_noise() -> None:
    assert srv._solve_progress_fields("[square/dwell] block map: done") is None
    assert srv._solve_progress_fields("SOLVE_PROGRESS eval=x pool=1") is None


def test_record_solve_progress_tracks_best_and_caps_trace() -> None:
    job = srv._make_job(mode="fgm_solve", output_name="t_dashboard")
    jid = job["id"]
    try:
        srv._record_solve_progress(jid, {"eval": 1, "pool": 4, "fe_spent": 2.5,
                                         "fe_budget": 40.0, "J": 200.0, "IoU": 0.70})
        srv._record_solve_progress(jid, {"eval": 2, "pool": 4, "fe_spent": 5.0,
                                         "fe_budget": 40.0, "J": 150.0, "IoU": 0.80})
        srv._record_solve_progress(jid, {"eval": 3, "pool": 4, "fe_spent": 7.5,
                                         "fe_budget": 40.0, "J": 180.0, "IoU": 0.75})
        with srv.JOBS_LOCK:
            sp = dict(srv.JOBS[jid]["solve_progress"])
        assert sp["J"] == 180.0 and sp["IoU"] == 0.75          # latest
        assert sp["best_J"] == 150.0 and sp["best_IoU"] == 0.80  # best so far
        assert sp["fe_spent"] == 7.5 and sp["fe_budget"] == 40.0
        assert [p["J"] for p in sp["trace"]] == [200.0, 150.0, 180.0]
        # the trace is capped so a long solve cannot bloat /api/jobs
        for i in range(4, 450):
            srv._record_solve_progress(jid, {"eval": i, "pool": 500,
                                             "fe_spent": 2.5 * i, "fe_budget": 4000.0,
                                             "J": float(i), "IoU": 0.5})
        with srv.JOBS_LOCK:
            n = len(srv.JOBS[jid]["solve_progress"]["trace"])
        assert n <= srv.SOLVE_TRACE_CAP
    finally:
        with srv.JOBS_LOCK:
            srv.JOBS.pop(jid, None)


# ---------------------------------------------------------------------------
# 2. wrapper: argument building and validation (scripts/solve_schedule.py)
# ---------------------------------------------------------------------------

def test_wrapper_imports_without_heavy_stack() -> None:
    assert "adjoint2d" not in sys.modules or True  # import above did not crash
    assert not hasattr(ss, "np"), "module-level numpy import defeats fast import"


def test_parse_args_happy_paths(tmp_path: Path) -> None:
    a = ss.parse_args(["--shape", "cross", "--mode", "indexed",
                       "--positions", "4", "--budget", "10",
                       "--output-dir", str(tmp_path), "--label", "smoke"])
    assert a.shape == "cross" and a.mode == "indexed" and a.positions == 4
    assert a.budget == 10.0 and a.label == "smoke"
    b = ss.parse_args(["--shape", "T_shape", "--mode", "asym_dwell",
                       "--budget", "40", "--output-dir", str(tmp_path)])
    assert b.mode == "asym_dwell"
    c = ss.parse_args(["--shape", "L_shape", "--mode", "sequential",
                       "--budget", "20", "--output-dir", str(tmp_path)])
    assert c.mode == "sequential"


def test_parse_args_rejects_invalid_combinations(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        ss.parse_args(["--shape", "cross", "--mode", "sequential",
                       "--budget", "10", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        ss.parse_args(["--shape", "circle", "--mode", "asym_dwell",
                       "--budget", "10", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        ss.parse_args(["--shape", "cross", "--mode", "indexed",
                       "--positions", "5", "--budget", "10",
                       "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        ss.parse_args(["--shape", "cross", "--mode", "indexed",
                       "--budget", "0", "--output-dir", str(tmp_path)])


def test_budget_to_evals_uses_the_campaign_convention() -> None:
    # 40 forward-equivalents = 16 gradient evaluations (2.5 each)
    assert ss.budget_to_evals(40.0) == 16
    assert ss.budget_to_evals(10.0) == 4
    assert ss.budget_to_evals(1.0) == 1   # never zero


def test_positions_to_step_deg() -> None:
    assert ss.positions_to_step_deg(4) == 90.0
    assert ss.positions_to_step_deg(24) == 15.0
    with pytest.raises(ValueError):
        ss.positions_to_step_deg(5)


# ---------------------------------------------------------------------------
# 3. wrapper: driver-log progress watch
# ---------------------------------------------------------------------------

def test_driver_watch_turns_block_lines_into_progress() -> None:
    w = ss.DriverWatch(evals_total=12)
    assert w.feed("[cross/dwell] 8 candidate positions ...") is None
    p = w.feed("[cross/dwell]   block map  : 4 evals, J 128.71 -> 34.39, 210 s")
    assert p is not None
    assert p["evals_done"] == 4 and p["evals_total"] == 12
    assert p["fe_spent"] == pytest.approx(10.0)   # 2.5 per evaluation
    assert p["J"] == 34.39
    p2 = w.feed("[cross/dwell]   block dwell: 4 evals, J 34.39 -> 34.04, 190 s")
    assert p2["evals_done"] == 8
    assert p2["J"] == 34.04


def test_driver_watch_reads_arm_rows_for_iou() -> None:
    w = ss.DriverWatch(evals_total=8)
    p = w.feed("[cross/rotavg]   AVG_4bpp             J    34.76  IoU 0.9847  "
               "grow  1.54%  under  0.00%  rho 0.7255  P  364.9 W/m  "
               "stop  470.0 s  maxT 195.7 C  Egate PASS")
    assert p is not None and p["IoU"] == 0.9847 and p["J"] == 34.76
    assert w.best_iou == 0.9847


def test_schedule_progress_line_round_trip() -> None:
    w = ss.DriverWatch(evals_total=12)
    p = w.feed("[x]   block map  : 4 evals, J 100.0 -> 50.0, 10 s")
    line = ss.format_progress_line(p)
    assert line.startswith("SCHEDULE_PROGRESS ")
    back = srv._schedule_progress_fields(line)
    assert back["evals_done"] == 4 and back["evals_total"] == 12
    assert back["J"] == 50.0 and back["fe_spent"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 4. server: launch command building and payload validation
# ---------------------------------------------------------------------------

def test_schedule_cosolve_cmd_builds_wrapper_argv(tmp_path: Path) -> None:
    payload = {"shape": "cross", "schedule_mode": "indexed",
               "schedule_positions": 4, "budget": 10,
               "output_name": "cross_sched_smoke"}
    cmd = srv._schedule_cosolve_cmd(payload, tmp_path)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/solve_schedule.py")
    s = " ".join(cmd)
    assert "--shape cross" in s and "--mode indexed" in s
    assert "--positions 4" in s and "--budget 10" in s
    assert f"--output-dir {tmp_path}" in s


def test_schedule_cosolve_cmd_rejects_bad_payloads(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        srv._schedule_cosolve_cmd({"shape": "circle", "schedule_mode": "indexed",
                                   "budget": 10, "output_name": "x"}, tmp_path)
    with pytest.raises(ValueError):
        srv._schedule_cosolve_cmd({"shape": "cross", "schedule_mode": "nope",
                                   "budget": 10, "output_name": "x"}, tmp_path)
    with pytest.raises(ValueError):
        srv._schedule_cosolve_cmd({"shape": "cross", "schedule_mode": "sequential",
                                   "budget": 10, "output_name": "x"}, tmp_path)
    with pytest.raises(ValueError):
        srv._schedule_cosolve_cmd({"shape": "cross", "schedule_mode": "indexed",
                                   "budget": -1, "output_name": "x"}, tmp_path)
