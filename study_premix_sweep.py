"""Premix 0->15 wt% sweep on jared_exp1_40mm (2-D run_sim, proven premix path).

Answers, per premix level (wt% = frac*25):
  #2 RF absorption redistribution -- bed_absorption_fraction (unambiguous).
  part uniformity                 -- part peak-to-mean (premix's ceiling mechanism).
  peak relocation                 -- Δ(peak cell) vs the premix=0 case.
  #3 drive-to-ceiling             -- generator power to reach part-peak == 250 C,
                                     under the model's FIXED-TOTAL-ABSORBED-POWER
                                     convention (heatr3d-faithful). See the honest
                                     caveat in the writeup: this convention makes
                                     the premixed bed PARASITIC, so drive rises;
                                     under a voltage-drive convention it would fall.

NOT re-optimizing the printed map per level (#1): the cheap 2-D adjoint
(fgm_solve_campaign/adjoint2d) is a separate solver without premix; wiring premix
there is a documented follow-up. Here the printed map is held FIXED, which cleanly
isolates the premix forward-physics effect.

Run: python study_premix_sweep.py   (does a load check first; ~10-12 min)
Checkpoints each level to results/premix_sweep/level_<pct>.json so a crash resumes.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml

import rfam_eqs_coupled as rc
from premix_study_metrics import bed_absorption_fraction, peak_to_mean, peak_location

CONFIG = "configs/jared_exp1_40mm_premix.yaml"
OUTDIR = Path("results/premix_sweep")
WTPCTS = [0.0, 3.75, 7.5, 11.25, 15.0]     # dopant-to-nylon wt% (frac = wt%/25)
CEILING_C = 250.0                          # degradation ceiling (thermal-ceiling workstream)


def _cfg_at(wtpct: float, gen_power_w: float | None = None) -> dict:
    cfg = yaml.safe_load(open(CONFIG))
    cfg["premix"] = {"frac": wtpct / 25.0, "budget": "floor_added"}
    if gen_power_w is not None:
        cfg["electric"]["generator_power_w"] = float(gen_power_w)
        cfg["electric"]["target_power_w"] = float(gen_power_w)
    return cfg


def _run(cfg: dict):
    out = rc.run_sim(copy.deepcopy(cfg))
    return out[0]  # SimState


def _peak_part_T(state, part_mask) -> float:
    return float(state.T[part_mask].max())


def drive_to_ceiling(wtpct: float, part_mask, lo=300.0, hi=8000.0, iters=8) -> float:
    """Bisect generator power so end-state part-peak T == CEILING_C.

    Range is wide because under the fixed-total-absorbed-power convention the
    premixed bed is parasitic, so high-wt% levels need a much larger drive to bring
    the part peak to the ceiling (an honest consequence, not a bug)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        st = _run(_cfg_at(wtpct, gen_power_w=mid))
        if _peak_part_T(st, part_mask) < CEILING_C:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # Derive masks through the SAME cfg path the runs use (premix does not affect
    # geometry, so frac=0 gives the shared part/doped masks at the run grid).
    base = _cfg_at(0.0)
    part_mask = rc.make_domain(base)[3]
    doped_mask = rc.make_domain(base)[4]

    peak0 = None
    for wtpct in WTPCTS:
        tag = f"{wtpct:g}".replace(".", "p")
        rec_path = OUTDIR / f"level_{tag}.json"
        if rec_path.exists():
            print(f"[skip] {rec_path} exists")
            peak0 = peak0 or tuple(json.loads(rec_path.read_text())["peak_cell"])
            continue
        # fixed-500W forward: absorption + uniformity + peak location
        st = _run(_cfg_at(wtpct))
        bed_frac = bed_absorption_fraction(st.Qrf, doped_mask)
        p2m = peak_to_mean(st.T, part_mask)
        pk = peak_location(st.T, part_mask)
        if peak0 is None:
            peak0 = pk
        reloc = int(abs(pk[0] - peak0[0]) + abs(pk[1] - peak0[1]))  # manhattan cells
        peak_T = float(st.T[part_mask].max())
        mean_T = float(st.T[part_mask].mean())
        # drive-to-ceiling (fixed-total-power convention)
        drive = drive_to_ceiling(wtpct, part_mask)
        rec = dict(wtpct=wtpct, premix_frac=wtpct / 25.0,
                   bed_absorption_fraction=bed_frac, part_peak_to_mean=p2m,
                   part_peak_T_C=peak_T, part_mean_T_C=mean_T,
                   peak_cell=list(pk), peak_reloc_cells=reloc,
                   drive_to_ceiling_W=drive, ceiling_C=CEILING_C)
        rec_path.write_text(json.dumps(rec, indent=2))
        np.savez(OUTDIR / f"level_{tag}.npz", Qrf=st.Qrf, T=st.T)
        print(f"[level wt%={wtpct:g}] bed_absorb={bed_frac:.3f} peak/mean={p2m:.3f} "
              f"peakT={peak_T:.1f} meanT={mean_T:.1f} reloc={reloc} drive2ceil={drive:.0f}W")


if __name__ == "__main__":
    main()
