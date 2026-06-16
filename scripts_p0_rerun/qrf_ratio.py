#!/usr/bin/env python3
"""Baseline Qrf concentration ratio (max/mean over the part) at iter-0 (uniform dopant)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_eqs"


def main() -> None:
    rel = sys.argv[1]  # runs/<shape>/fgm_iterate/<name>
    parent = OUT / rel
    name = parent.name
    iter0 = parent / f"{name}_iter0" / "fields.npz"
    if not iter0.exists():
        print(f"NO iter0 fields at {iter0}")
        sys.exit(2)
    d = np.load(iter0)
    qrf = d["Qrf"]
    pm = d["part_mask"].astype(bool)
    q = qrf[pm]
    q = q[q > 0]
    if q.size == 0:
        print("Qrf empty in part")
        sys.exit(2)
    qmax = float(q.max()); qmean = float(q.mean())
    print(f"{rel}  Qrf_max={qmax:.3e}  Qrf_mean={qmean:.3e}  ratio={qmax/qmean:.2f}")


if __name__ == "__main__":
    main()
