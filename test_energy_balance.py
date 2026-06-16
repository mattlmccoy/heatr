#!/usr/bin/env python3
"""
test_energy_balance.py — Physics-first unit tests for HEATR energy balance.

Tests A, B, C isolate each component of the energy balance:
  A. Zero Q_rf, no losses → T stays at ambient, residual = 0
  B. Q_rf only, no losses, no densification → residual ≈ 0 (FE truncation only)
  C. Full canonical run → residual < 5% of E_in at end (was ~20% before fix)

Run:
  python3 test_energy_balance.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from rfam_eqs_coupled import run_sim, load_config  # noqa: E402

CANONICAL_CONFIG = BASE / "configs" / "rfam_eqs_xz_uniform_500w.yaml"

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def _patch(base_cfg: dict, patches: dict) -> dict:
    """Deep-patch a config dict with nested dot-key updates."""
    cfg = copy.deepcopy(base_cfg)

    def _set(d: dict, keys: list[str], val):
        if len(keys) == 1:
            d[keys[0]] = val
        else:
            d.setdefault(keys[0], {})
            _set(d[keys[0]], keys[1:], val)

    for dotkey, val in patches.items():
        _set(cfg, dotkey.split("."), val)
    return cfg


def _load_base() -> dict:
    if not CANONICAL_CONFIG.exists():
        raise FileNotFoundError(f"Canonical config not found: {CANONICAL_CONFIG}")
    return load_config(CANONICAL_CONFIG)


# ---------------------------------------------------------------------------
# Test A — Zero Q_rf, no BC losses (adiabatic)
# ---------------------------------------------------------------------------
def test_a_zero_power_adiabatic():
    print("Test A: zero Q_rf, no losses (adiabatic) ...")
    base = _load_base()
    cfg = _patch(base, {
        "electric.generator_power_w": 0.0,
        "electric.voltage_v": 0.0,                 # zero voltage → E=0 → Q_rf=0 everywhere
        "thermal.convection.h_top": 0.0,
        "thermal.convection.h_bottom": 0.0,
        "thermal.convection.h_left": 0.0,
        "thermal.convection.h_right": 0.0,
        "thermal.effective_depth_m": 1e6,          # effectively infinite → no z-loss
        "thermal.n_steps": 120,                    # short run (2 min)
    })
    # Also try to disable all convection via the flat key used in some configs
    cfg.setdefault("thermal", {}).setdefault("convection", {})
    for face in ("h_top", "h_bottom", "h_left", "h_right", "h"):
        cfg["thermal"]["convection"][face] = 0.0

    _, _, hist, *_ = run_sim(cfg)
    residuals = hist["energy_balance_residual_J_per_m"]
    max_abs_residual = max(abs(r) for r in residuals)
    e_in_final = hist["energy_doped_J_per_m"][-1]

    # With zero power, E_in = 0 and stored energy should not grow → residual ≈ 0
    tol_J_per_m = 1.0  # 1 J/m absolute (essentially machine precision for this scale)
    ok = max_abs_residual < tol_J_per_m and abs(e_in_final) < tol_J_per_m
    print(f"  E_in_final = {e_in_final:.3f} J/m  |  max|residual| = {max_abs_residual:.3f} J/m")
    print(f"  {_PASS if ok else _FAIL}: residual < {tol_J_per_m} J/m")
    return ok


# ---------------------------------------------------------------------------
# Test B — Q_rf only, no losses, densification disabled
# ---------------------------------------------------------------------------
def test_b_qrf_only_no_losses():
    print("Test B: Q_rf only, no losses, densification disabled ...")
    base = _load_base()
    cfg = _patch(base, {
        "thermal.effective_depth_m": 1e6,
        "thermal.n_steps": 120,
        "densification.enabled": False,
    })
    for face in ("h_top", "h_bottom", "h_left", "h_right", "h"):
        cfg["thermal"].setdefault("convection", {})[face] = 0.0

    _, _, hist, *_ = run_sim(cfg)
    residuals = hist["energy_balance_residual_J_per_m"]
    e_in_final = hist["energy_doped_J_per_m"][-1]
    final_residual = residuals[-1] if residuals else 0.0
    rel_residual = abs(final_residual) / max(abs(e_in_final), 1.0)

    # Without densification, density is constant → variable and fixed stored energy are equal.
    # Residual should be near-zero (Forward Euler truncation only, typically < 1%).
    tol_frac = 0.01  # 1% relative tolerance
    ok = rel_residual < tol_frac
    print(f"  E_in_final = {e_in_final:.1f} J/m  |  final residual = {final_residual:.1f} J/m  "
          f"({rel_residual*100:.2f}%)")
    print(f"  {_PASS if ok else _FAIL}: relative residual < {tol_frac*100:.0f}%")
    return ok


# ---------------------------------------------------------------------------
# Test C — Full canonical run: residual < 5% (was ~20% before fix)
# ---------------------------------------------------------------------------
def test_c_full_run_residual():
    print("Test C: full canonical 6-min run — residual should be < 5% of E_in ...")
    base = _load_base()
    # Run full config as-is (all physics on)
    _, summary, hist, *_ = run_sim(base)
    e_in_final = hist["energy_doped_J_per_m"][-1]
    final_residual = hist["energy_balance_residual_J_per_m"][-1]
    dens_inject = (hist.get("energy_densification_injection_J_per_m") or [0.0])[-1]
    rel_residual = abs(final_residual) / max(abs(e_in_final), 1.0)

    tol_frac = 0.05  # 5% — realistic for coupled FE with densification
    ok = rel_residual < tol_frac
    print(f"  E_in_final           = {e_in_final/1000:.1f} kJ/m")
    print(f"  final residual       = {final_residual/1000:.2f} kJ/m  ({rel_residual*100:.1f}%)")
    print(f"  densification inject = {dens_inject/1000:.2f} kJ/m")
    print(f"  {_PASS if ok else _FAIL}: residual < {tol_frac*100:.0f}% of E_in")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("HEATR Energy Balance Unit Tests")
    print("=" * 60)
    results = []
    for fn in (test_a_zero_power_adiabatic, test_b_qrf_only_no_losses, test_c_full_run_residual):
        try:
            results.append(fn())
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append(False)
        print()
    print("=" * 60)
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} tests passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
