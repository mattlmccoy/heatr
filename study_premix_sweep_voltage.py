"""Premix 0->15 wt% sweep under VOLTAGE DRIVE (the fair counterpart to the
fixed-power sweep). Fixed applied voltage, absorbed power free to vary with sigma.

Contrast with study_premix_sweep.py (fixed total power, premix parasitic): here
raising sigma with premix raises absorbed power IN the part, so the voltage needed
to reach the 250 C ceiling should DROP with premix -> premix helps. Every level is
bisected to a ~250 C part peak, then compared at that matched ceiling.

Numerical note: at high drive the explicit thermal solve can hit the THM-01 dT cap /
THM-02 clamp (runaway). The bisection reads a clamped 600 C run as 'above ceiling'
(correct); the CONVERGED ceiling run must be checked stable -- run records the max
part T so a value pinned at the 600 C clamp is visible.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

import rfam_eqs_coupled as rc
from premix_study_metrics import bed_absorption_fraction, peak_to_mean

CONFIG = "configs/jared_exp1_40mm_premix.yaml"
OUTDIR = Path("results/premix_sweep_voltage")
WTPCTS = [0.0, 3.75, 7.5, 11.25, 15.0]
CEILING_C = 250.0
CLAMP_C = 600.0  # THM-02 temperature clamp; a converged peak at/above this is non-physical


def _cfg_at(wtpct: float, voltage_v: float) -> dict:
    cfg = yaml.safe_load(open(CONFIG))
    cfg["electric"]["enforce_generator_power"] = False   # VOLTAGE DRIVE
    cfg["electric"]["voltage_v"] = float(voltage_v)
    cfg["premix"] = {"frac": wtpct / 25.0, "budget": "floor_added"}
    return cfg


def _run(cfg):
    return rc.run_sim(copy.deepcopy(cfg))[0]


def voltage_to_ceiling(wtpct, part_mask, lo=30.0, hi=2600.0, iters=11):
    """Bisect applied voltage so end-state part-peak T == CEILING_C."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pk = float(_run(_cfg_at(wtpct, mid)).T[part_mask].max())
        if pk < CEILING_C:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    part_mask = rc.make_domain(_cfg_at(0.0, 1000.0))[3]
    doped_mask = rc.make_domain(_cfg_at(0.0, 1000.0))[4]
    for wtpct in WTPCTS:
        tag = f"{wtpct:g}".replace(".", "p")
        rec_path = OUTDIR / f"level_{tag}.json"
        if rec_path.exists():
            print(f"[skip] {rec_path}")
            continue
        v = voltage_to_ceiling(wtpct, part_mask)
        st = _run(_cfg_at(wtpct, v))
        rec = dict(
            wtpct=wtpct, premix_frac=wtpct / 25.0, voltage_to_ceiling_V=v,
            matched_peak_T_C=float(st.T[part_mask].max()),
            matched_part_peak_to_mean=peak_to_mean(st.T, part_mask),
            matched_part_mean_phi=float(st.phi[part_mask].mean()),
            matched_part_mean_T_C=float(st.T[part_mask].mean()),
            bed_absorption_fraction=bed_absorption_fraction(st.Qrf, doped_mask),
            clamp_hit=bool(float(st.T[part_mask].max()) >= CLAMP_C - 1.0),
        )
        rec_path.write_text(json.dumps(rec, indent=2))
        np.savez(OUTDIR / f"level_{tag}.npz", Qrf=st.Qrf, T=st.T)
        print(f"[wt%={wtpct:g}] V->ceiling={v:.0f}V  peakT={rec['matched_peak_T_C']:.1f} "
              f"phi={rec['matched_part_mean_phi']:.3f} peak/mean={rec['matched_part_peak_to_mean']:.3f} "
              f"bed={rec['bed_absorption_fraction']:.3f} clamp={rec['clamp_hit']}")


if __name__ == "__main__":
    main()
