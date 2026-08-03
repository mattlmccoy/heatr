#!/usr/bin/env python3
"""Capture a fixed-step turntable run as the backward-compatibility baseline.

Run this ONCE against the engine BEFORE the program-mode edit, then again
after; `max|diff|` on the final temperature, relative density and melt
fraction fields must be exactly 0.0.

The config is deliberately cheap (grid 60, 60 outer steps, 3 rotations of
90 degrees) so the regression can run inside the normal test suite.

Usage:
  ./.venv312/bin/python tests_fixtures/make_turntable_baseline.py <out.npz>
"""
from __future__ import annotations

import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE_CFG = REPO / "configs/diamond_tt_15deg_48rot_nearcont.yaml"


def baseline_cfg() -> dict:
    cfg = copy.deepcopy(yaml.safe_load(BASE_CFG.read_text()))
    cfg["geometry"]["grid_nx"] = 60
    cfg["geometry"]["grid_ny"] = 60
    cfg["thermal"]["n_steps"] = 260
    cfg["turntable"] = {
        "enabled": True,
        "rotation_deg": 90.0,
        "total_rotations": 6,
        "rotation_interval_s": 18.0,
    }
    return cfg


def run_baseline() -> dict:
    import rfam_eqs_coupled as rfam

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        state, summary, hist, tt_steps, _opt = rfam.run_sim(baseline_cfg())
    return {
        "T": np.asarray(state.T, dtype=float),
        "rho_rel": np.asarray(state.rho_rel, dtype=float),
        "phi": np.asarray(state.phi, dtype=float),
        "part_mask": np.asarray(state.part_mask, dtype=bool),
        "tt_rotation_steps": np.asarray(tt_steps, dtype=float),
        "energy_doped_J_per_m": np.asarray(hist["energy_doped_J_per_m"], dtype=float),
    }


def main(out_path: str) -> None:
    res = run_baseline()
    np.savez_compressed(out_path, **res)
    print(f"wrote {out_path}")
    print(f"  T range      {res['T'].min():.6f} .. {res['T'].max():.6f}")
    print(f"  rho range    {res['rho_rel'].min():.6f} .. {res['rho_rel'].max():.6f}")
    print(f"  phi mean     {res['phi'].mean():.9f}")
    print(f"  tt events    {res['tt_rotation_steps'].tolist()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else str(REPO / "tests_fixtures/turntable_baseline_pre.npz"))
