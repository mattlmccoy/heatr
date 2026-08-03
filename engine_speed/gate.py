"""Parity gate harness: heatr3d.run vs engine_speed.march_fast.march_fast.

Gate rule (task spec, S1 convention):
  a. PREFERRED verdict is BIT-IDENTITY (np.array_equal / exact float equality)
     on every field of the Result.
  b. If exact bit-identity is unreachable for a field, the fallback verdict is
     the S1 floor: the deviation must sit at the engine's own measured
     run-to-run nondeterminism, ~1e-16 relative. FLOOR_RTOL below is that floor
     and is NEVER widened to make a case pass -- a case above it is a FAIL and
     must be bisected.
  c. The energy audit (energy_residual_frac) is compared under the same rule.
  d. Standing-gate fields (reached, T_max_c, clamp_bound, n_substeps_used,
     cfl_violated) are compared for EXACT equality (they are bools/ints/scalars
     describing guard behaviour, so "at the floor" is not an acceptable answer).

Results are written to gate_results.json so the report quotes recorded numbers
rather than transcribed ones.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import heatr3d as h3

from .cases import Case, build_cases

# The S1 nondeterminism floor. Deviations at or below this are indistinguishable
# from the engine's own run-to-run spread. DO NOT WIDEN.
FLOOR_RTOL = 1e-16

ARRAY_FIELDS = ("T_final", "T_phi90", "phi_final", "Qrf", "rho_final", "part")
SCALAR_FLOAT_FIELDS = ("sigma_T", "T_max_c", "t_phi90_s", "exposure_s",
                       "energy_in_j", "energy_stored_j", "energy_loss_j",
                       "energy_residual_frac")
EXACT_FIELDS = ("reached", "clamp_bound", "n_substeps_used", "cfl_violated",
                "n_eqs_solves", "n_eqs_resolves_skipped")


def _rel_dev(a: np.ndarray, b: np.ndarray) -> float:
    """max |a-b| / max(|a|,|b|, tiny), ignoring pairs that are exactly equal
    (including both-NaN)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf")
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if np.all(same):
        return 0.0
    d = np.abs(a - b)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    r = np.where(same, 0.0, d / scale)
    return float(np.nanmax(r))


def compare_results(ref: h3.Result, fast: h3.Result) -> dict[str, Any]:
    out: dict[str, Any] = {"fields": {}, "bit_identical": True, "pass": True,
                           "failures": []}
    for name in ARRAY_FIELDS:
        a = getattr(ref, name)
        b = getattr(fast, name)
        if a is None and b is None:
            out["fields"][name] = {"status": "both_none"}
            continue
        if (a is None) != (b is None):
            out["fields"][name] = {"status": "one_none"}
            out["pass"] = False
            out["bit_identical"] = False
            out["failures"].append(name)
            continue
        a = np.asarray(a)
        b = np.asarray(b)
        exact = bool(a.shape == b.shape and np.array_equal(a, b))
        rel = 0.0 if exact else _rel_dev(a.astype(np.float64), b.astype(np.float64))
        ok = exact or rel <= FLOOR_RTOL
        out["fields"][name] = {"bit_identical": exact, "max_rel_dev": rel,
                               "pass": ok}
        out["bit_identical"] &= exact
        if not ok:
            out["pass"] = False
            out["failures"].append(name)
    for name in SCALAR_FLOAT_FIELDS:
        a = float(getattr(ref, name))
        b = float(getattr(fast, name))
        exact = (a == b) or (np.isnan(a) and np.isnan(b))
        rel = 0.0 if exact else _rel_dev(np.array([a]), np.array([b]))
        ok = exact or rel <= FLOOR_RTOL
        out["fields"][name] = {"bit_identical": exact, "max_rel_dev": rel,
                               "pass": ok, "ref": a, "fast": b}
        out["bit_identical"] &= exact
        if not ok:
            out["pass"] = False
            out["failures"].append(name)
    for name in EXACT_FIELDS:
        a = getattr(ref, name)
        b = getattr(fast, name)
        ok = (a == b)
        out["fields"][name] = {"bit_identical": bool(ok), "max_rel_dev": 0.0 if ok else float("inf"),
                               "pass": bool(ok), "ref": _jsonable(a), "fast": _jsonable(b)}
        out["bit_identical"] &= bool(ok)
        if not ok:
            out["pass"] = False
            out["failures"].append(name)
    # phi_hist: list of per-step mean melt fractions. Length AND values.
    ah, bh = list(ref.phi_hist), list(fast.phi_hist)
    len_ok = len(ah) == len(bh)
    if len_ok and ah:
        exact = all(x == y for x, y in zip(ah, bh))
        rel = 0.0 if exact else _rel_dev(np.array(ah), np.array(bh))
    else:
        exact = len_ok
        rel = 0.0 if len_ok else float("inf")
    ok = len_ok and (exact or rel <= FLOOR_RTOL)
    out["fields"]["phi_hist"] = {"bit_identical": bool(exact and len_ok),
                                 "max_rel_dev": rel, "pass": bool(ok),
                                 "len_ref": len(ah), "len_fast": len(bh)}
    out["bit_identical"] &= bool(exact and len_ok)
    if not ok:
        out["pass"] = False
        out["failures"].append("phi_hist")
    return out


def _jsonable(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def run_case(case: Case) -> dict[str, Any]:
    from .march_fast import march_fast
    t0 = time.perf_counter()
    ref = h3.run(case.grid, case.part, case.params, sat=case.sat, **case.run_kwargs)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    fast = march_fast(case.grid, case.part, case.params, sat=case.sat,
                      **case.run_kwargs)
    t_fast = time.perf_counter() - t0
    rec = compare_results(ref, fast)
    rec["name"] = case.name
    rec["description"] = case.description
    rec["wall_ref_s"] = t_ref
    rec["wall_fast_s"] = t_fast
    rec["n_steps_ref"] = len(ref.phi_hist)
    # ---- guard-firing assertions (must_fire) ----
    fired = {"cfl_ref": int(ref.n_substeps_used), "cfl_fast": int(fast.n_substeps_used),
             "clamp_ref": bool(ref.clamp_bound), "clamp_fast": bool(fast.clamp_bound)}
    rec["guards"] = fired
    if case.must_fire == "cfl":
        ok = fired["cfl_ref"] > 1 and fired["cfl_fast"] > 1
        rec["guard_fired"] = bool(ok)
        if not ok:
            rec["pass"] = False
            rec["failures"].append("must_fire:cfl")
    elif case.must_fire == "clamp":
        ok = fired["clamp_ref"] and fired["clamp_fast"]
        rec["guard_fired"] = bool(ok)
        if not ok:
            rec["pass"] = False
            rec["failures"].append("must_fire:clamp")
    else:
        rec["guard_fired"] = None
    return rec


def run_gate(n: int = 32, out_path: str | Path | None = None) -> dict[str, Any]:
    import numba

    cases = build_cases(n)
    records = [run_case(c) for c in cases]
    summary = {
        "floor_rtol": FLOOR_RTOL,
        "grid_n": n,
        "numba_version": numba.__version__,
        "numpy_version": np.__version__,
        "all_pass": all(r["pass"] for r in records),
        "all_bit_identical": all(r["bit_identical"] for r in records),
        "cases": records,
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(summary, indent=2, default=_jsonable))
    return summary


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    here = Path(__file__).resolve().parent
    s = run_gate(n, here / f"gate_results_n{n}.json")
    for r in s["cases"]:
        print(f"{r['name']:16s} pass={r['pass']} bitident={r['bit_identical']} "
              f"fired={r['guard_fired']} steps={r['n_steps_ref']} "
              f"ref={r['wall_ref_s']:.2f}s fast={r['wall_fast_s']:.2f}s "
              f"fail={r['failures']}")
    print("ALL PASS:", s["all_pass"], " ALL BIT-IDENTICAL:", s["all_bit_identical"])
