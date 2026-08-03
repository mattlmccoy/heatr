#!/usr/bin/env python3
"""RED/GREEN tests for the pre-v2 dual read-state sigma_T backfill.

Deferred item (c) from HEATR_V2_ROLLOUT_NOTES.md: a one-shot maintenance
script/endpoint that walks historical run directories, computes the dual
read-state sigma_T from the stored time_series.json where recoverable, and
stamps summary.json with sigma_T_backfilled: true. NEVER overwrites existing
fields; skips and counts where unrecoverable.

Data contract (probed on the real tree before this was written):
time_series.json carries ui_rms_part, mean_T_part_c and mean_phi_part, which
are exactly the three history keys rfam_eqs_coupled.dual_read_state_from_hist
reads (verified on
outputs_eqs/runs/diamond/placement_optimizer/experimental/
diamond_place_20260303_rot/time_series.json; 854 run directories carry both
files).

Run: ./.venv312/bin/python -m pytest test_backfill_sigma_t.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

from rfam_eqs_coupled import dual_read_state_from_hist          # noqa: E402
from backfill_sigma_t import backfill_summary, walk_and_backfill  # noqa: E402

# A history that melts: phi crosses 0.90 at index 3.
_HIST_MELTS = {
    "ui_rms_part": [0.10, 0.20, 0.15, 0.12, 0.11],
    "mean_T_part_c": [30.0, 120.0, 180.0, 200.0, 210.0],
    "mean_phi_part": [0.0, 0.10, 0.60, 0.95, 1.0],
}


def _mk_run(root: Path, name: str, summary: dict, hist: dict | None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summary))
    if hist is not None:
        (d / "time_series.json").write_text(json.dumps(hist))
    return d


def test_backfill_computes_and_stamps() -> None:
    summary = {"grid": 120}
    out, status = backfill_summary(summary, _HIST_MELTS)
    assert status == "backfilled"
    pinned = dual_read_state_from_hist(_HIST_MELTS)
    for k, v in pinned.items():
        assert out[k] == v
    assert out["sigma_T_backfilled"] is True
    assert out["grid"] == 120
    # the input dict is not mutated
    assert "sigma_T_backfilled" not in summary


def test_never_overwrites_existing_fields() -> None:
    summary = {"sigma_T_heating_peak_c": 5.791, "grid": 120}
    out, status = backfill_summary(summary, _HIST_MELTS)
    assert status == "already_present"
    assert out == summary


def test_unrecoverable_history_is_skipped() -> None:
    out, status = backfill_summary({"grid": 120}, {"mean_T_part_c": [1.0]})
    assert status == "unrecoverable"
    assert "sigma_T_backfilled" not in out
    out2, status2 = backfill_summary({"grid": 120}, None)
    assert status2 == "unrecoverable"
    assert "sigma_T_backfilled" not in out2


def test_walk_dry_run_counts_and_writes_nothing(tmp_path: Path) -> None:
    a = _mk_run(tmp_path, "runs/square/single/a", {"grid": 120}, _HIST_MELTS)
    _mk_run(tmp_path, "runs/square/single/b",
            {"sigma_T_heating_peak_c": 1.0}, _HIST_MELTS)
    _mk_run(tmp_path, "runs/square/single/c", {"grid": 120}, None)
    before = (a / "summary.json").read_text()
    counts = walk_and_backfill(tmp_path, dry_run=True)
    assert counts["scanned"] == 3
    assert counts["backfillable"] == 1
    assert counts["already_present"] == 1
    assert counts["no_time_series"] == 1
    assert counts["written"] == 0
    assert (a / "summary.json").read_text() == before


def test_walk_real_run_writes_then_skips(tmp_path: Path) -> None:
    a = _mk_run(tmp_path, "runs/square/single/a", {"grid": 120}, _HIST_MELTS)
    counts = walk_and_backfill(tmp_path, dry_run=False)
    assert counts["written"] == 1
    stamped = json.loads((a / "summary.json").read_text())
    assert stamped["sigma_T_backfilled"] is True
    assert stamped["sigma_T_melt_reached"] is True
    # second pass finds nothing to do
    counts2 = walk_and_backfill(tmp_path, dry_run=False)
    assert counts2["written"] == 0
    assert counts2["already_present"] == 1


def test_walk_skips_archive_and_honors_limit(tmp_path: Path) -> None:
    _mk_run(tmp_path, "runs/_archive/old", {"grid": 120}, _HIST_MELTS)
    _mk_run(tmp_path, "runs/square/single/a", {"grid": 120}, _HIST_MELTS)
    _mk_run(tmp_path, "runs/square/single/b", {"grid": 120}, _HIST_MELTS)
    counts = walk_and_backfill(tmp_path, dry_run=True)
    assert counts["scanned"] == 2  # the archived run is never visited
    limited = walk_and_backfill(tmp_path, dry_run=False, limit=1)
    assert limited["written"] == 1
