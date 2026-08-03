#!/usr/bin/env python3
"""One-shot maintenance: backfill the dual read-state sigma_T on pre-v2 runs.

Deferred item (c) from HEATR_V2_ROLLOUT_NOTES.md. Walks historical run
directories under outputs_eqs/, computes the dual read-state sigma_T
(heating-peak and melt-onset, rfam_eqs_coupled.dual_read_state_from_hist)
from the stored time_series.json where recoverable, and stamps summary.json
with sigma_T_backfilled: true.

Rules, enforced and tested (test_backfill_sigma_t.py):
  * NEVER overwrites an existing sigma_T field; a summary that already
    carries any dual read-state key is counted and left byte-identical.
  * Unrecoverable runs (no time_series.json, or one missing the required
    history keys) are skipped and counted, never guessed.
  * --dry-run reports the counts without writing anything.
  * _archive subtrees are never visited.

Data contract: time_series.json carries ui_rms_part, mean_T_part_c and
mean_phi_part (probed on the real tree; 854 run directories carry both
summary.json and time_series.json as of 2026-08-03).

Run:
  ./.venv312/bin/python scripts/backfill_sigma_t.py --dry-run
  ./.venv312/bin/python scripts/backfill_sigma_t.py --subset runs/square --limit 5
  ./.venv312/bin/python scripts/backfill_sigma_t.py            # the real pass
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rfam_eqs_coupled import dual_read_state_from_hist  # noqa: E402

logger = logging.getLogger(__name__)

_SIGMA_KEYS = (
    "sigma_T_heating_peak_c", "sigma_T_heating_peak_idx",
    "sigma_T_melt_onset_c", "sigma_T_melt_onset_idx", "sigma_T_melt_reached",
)
_REQUIRED_HIST_KEYS = ("ui_rms_part", "mean_T_part_c", "mean_phi_part")


def backfill_summary(summary: Dict[str, Any],
                     hist: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """Pure backfill decision on one (summary, history) pair.

    Returns (new_summary, status); the input dict is never mutated.
    status: "already_present" | "unrecoverable" | "backfilled".
    """
    if any(k in summary for k in _SIGMA_KEYS):
        return summary, "already_present"
    if not isinstance(hist, dict):
        return summary, "unrecoverable"
    if any(not hist.get(k) for k in _REQUIRED_HIST_KEYS):
        return summary, "unrecoverable"
    dual = dual_read_state_from_hist(hist)
    out = dict(summary)
    out.update(dual)
    out["sigma_T_backfilled"] = True
    return out, "backfilled"


def walk_and_backfill(root: Path, *, dry_run: bool = True,
                      limit: Optional[int] = None,
                      subset: Optional[str] = None) -> Dict[str, Any]:
    """Walk run directories under `root` and backfill their summary.json.

    Counts every directory that has a summary.json (scanned); _archive
    subtrees are pruned at the walk level. `subset` restricts the walk to a
    path relative to `root`; `limit` caps the number of WRITES (dry runs are
    never limited, so the counts stay a full census).
    """
    root = Path(root)
    start = (root / subset) if subset else root
    counts: Dict[str, Any] = {
        "root": str(start), "dry_run": bool(dry_run),
        "scanned": 0, "already_present": 0, "no_time_series": 0,
        "unrecoverable": 0, "backfillable": 0, "written": 0,
        "read_errors": 0, "written_paths": [],
    }
    if not start.is_dir():
        return counts
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if not d.lower().startswith("_archive")]
        if "summary.json" not in filenames:
            continue
        d = Path(dirpath)
        counts["scanned"] += 1
        try:
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            counts["read_errors"] += 1
            continue
        ts_path = d / "time_series.json"
        if any(k in summary for k in _SIGMA_KEYS):
            counts["already_present"] += 1
            continue
        if not ts_path.exists():
            counts["no_time_series"] += 1
            continue
        try:
            hist = json.loads(ts_path.read_text(encoding="utf-8"))
        except Exception:
            counts["read_errors"] += 1
            continue
        new_summary, status = backfill_summary(summary, hist)
        if status != "backfilled":
            counts["unrecoverable"] += 1
            continue
        counts["backfillable"] += 1
        if dry_run:
            continue
        if limit is not None and counts["written"] >= int(limit):
            continue
        (d / "summary.json").write_text(
            json.dumps(new_summary, indent=2, default=float), encoding="utf-8")
        counts["written"] += 1
        if len(counts["written_paths"]) < 200:
            counts["written_paths"].append(str(d))
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO / "outputs_eqs"),
                    help="tree to walk (default outputs_eqs/)")
    ap.add_argument("--subset", default=None,
                    help="restrict to a path relative to --root (e.g. runs/square)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of summaries written (real runs only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts, write nothing")
    args = ap.parse_args(argv)
    counts = walk_and_backfill(Path(args.root), dry_run=args.dry_run,
                               limit=args.limit, subset=args.subset)
    print("SIGMA_T_BACKFILL " + json.dumps(
        {k: v for k, v in counts.items() if k != "written_paths"}))
    for p in counts["written_paths"]:
        logger.info("stamped %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
