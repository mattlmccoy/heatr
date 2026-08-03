#!/usr/bin/env python3
"""Proof-of-concept shape-fidelity scorer for existing HEATR run directories.

Computes, from a run's fields.npz:

    J = sum over the WHOLE domain of (phi - chi_part)^2

with chi_part = the binary part mask, matching the objective defined in the
shape-library solve report. Also reports intersection-over-union (IoU) of the
melt region (phi >= 0.5) against the part mask.

Limitation (by design of the stored outputs): fields.npz holds only the FINAL
phi field, so this scores J at the end of the exposure, not at the J-optimal
stop time. Selecting the optimal stop requires either re-running with a
shorter exposure or an in-loop J(t) recorder (see assessment report).

Usage:
    python scripts/analysis/score_J_from_run.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def score_run(run_dir: Path) -> dict:
    """Return J and IoU for one run directory containing fields.npz."""
    f = np.load(run_dir / "fields.npz")
    phi = np.asarray(f["phi"], dtype=float)
    chi = np.asarray(f["part_mask"], dtype=bool).astype(float)
    j = float(np.sum((phi - chi) ** 2))
    melt = phi >= 0.5
    part = chi.astype(bool)
    inter = float(np.logical_and(melt, part).sum())
    union = float(np.logical_or(melt, part).sum())
    iou = inter / union if union > 0 else float("nan")
    return {
        "run_dir": str(run_dir),
        "J_final": j,
        "IoU_melt_final": iou,
        "part_cells": int(part.sum()),
        "melt_cells": int(melt.sum()),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        r = score_run(Path(arg))
        print(f"{r['run_dir']}\n  J_final={r['J_final']:.2f}  "
              f"IoU_melt={r['IoU_melt_final']:.4f}  "
              f"part_cells={r['part_cells']}  melt_cells={r['melt_cells']}")


if __name__ == "__main__":
    main()
