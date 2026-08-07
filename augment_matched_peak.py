"""Matched-peak augmentation of the premix sweep.

The fixed-500W metrics compare premix levels at DIFFERENT densification states
(premix runs heat less), which is not apples-to-apples. This re-runs each level at
ITS OWN drive-to-ceiling so every level reaches the SAME 250 C part peak, then
records part uniformity + densification completeness at that matched peak -- the
fair test of whether premix gives a more uniform / more complete part at the ceiling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from study_premix_sweep import _cfg_at, _run, OUTDIR
from premix_study_metrics import peak_to_mean


def main() -> None:
    import rfam_eqs_coupled as rc
    part_mask = rc.make_domain(_cfg_at(0.0))[3]
    for p in sorted(OUTDIR.glob("level_*.json"), key=lambda p: json.loads(p.read_text())["wtpct"]):
        r = json.loads(p.read_text())
        drive = r["drive_to_ceiling_W"]
        st = _run(_cfg_at(r["wtpct"], gen_power_w=drive))
        r["matched_peak_T_C"] = float(st.T[part_mask].max())
        r["matched_part_peak_to_mean"] = peak_to_mean(st.T, part_mask)
        r["matched_part_mean_phi"] = float(st.phi[part_mask].mean())
        r["matched_part_mean_T_C"] = float(st.T[part_mask].mean())
        p.write_text(json.dumps(r, indent=2))
        print(f"[wt%={r['wtpct']:g}] @drive={drive:.0f}W  matched_peakT={r['matched_peak_T_C']:.1f} "
              f"peak/mean={r['matched_part_peak_to_mean']:.3f} mean_phi={r['matched_part_mean_phi']:.3f}")


if __name__ == "__main__":
    main()
