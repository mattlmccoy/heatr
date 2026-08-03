"""Express-relevant benchmark for the EQS cache.

Scenario, as specified by the engine lane:

  1. BEFORE arm  -- uncorrected part, uniform sat            -> cold MISS
  2. AFTER  arm  -- same geometry and n, DIFFERENT sat map   -> MISS (expected)
  3. package verify -- re-run of the BEFORE arm, sat identical -> HIT (expected)

Step 2 is the honesty check: a corrected arm MUST miss. If it ever hits, the
cache is returning a stale field for a changed dopant map, which is the
catastrophic failure mode.

Reported: wall time of every solve, time saved per hit, and a bit-identity
check of the hit against the original solve.

Load-check per the machine convention before running. This script performs
real EQS solves (n=96 is minutes), so it is not part of the pytest suite.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import heatr3d as h3

from .eqs_cache import EqsCache, solve_eqs_3d_cached

P = h3.Params()


def _sat_corrected(part: np.ndarray, grid: h3.Grid) -> np.ndarray:
    """A plausible AFTER-arm graded dopant map (the BEFORE arm is sat=None)."""
    X, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    g = 0.55 + 0.45 * np.cos(70.0 * X) * np.cos(70.0 * Y)
    sat = np.clip(g, 0.0, 1.0)
    sat[~part] = 0.0
    return sat


def _timed(fn) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def express_scenario(n: int) -> dict[str, Any]:
    grid = h3.Grid(n=n)
    part = h3.make_geometry(grid, "square", diam=0.020, zspan=0.020)
    gamma_before = h3.build_gamma(part, P, None)
    gamma_after = h3.build_gamma(part, P, _sat_corrected(part, grid))
    N = n ** 3
    cache = EqsCache()

    v_before, t_before = _timed(
        lambda: solve_eqs_3d_cached(gamma_before, grid, P, cache=cache))
    v_after, t_after = _timed(
        lambda: solve_eqs_3d_cached(gamma_after, grid, P, cache=cache))
    v_verify, t_verify = _timed(
        lambda: solve_eqs_3d_cached(gamma_before, grid, P, cache=cache))

    # the uncached reference cost of the verify solve, for an honest saving
    _, t_uncached = _timed(lambda: h3.solve_eqs_3d(gamma_before, grid, P))

    return {
        "n": n,
        "N_unknowns": N,
        "path": "iterative(ILU+BiCGSTAB)" if N > 50_000 else "direct(spsolve)",
        "t_before_miss_s": t_before,
        "t_after_miss_s": t_after,
        "t_verify_hit_s": t_verify,
        "t_verify_uncached_s": t_uncached,
        "saved_per_hit_s": t_uncached - t_verify,
        "after_arm_missed_as_required": bool(cache.stats["solution_hits"] == 1),
        "verify_hit_is_bit_identical": bool(np.array_equal(v_before, v_verify)),
        "after_field_differs_from_before": bool(not np.array_equal(v_before, v_after)),
        "stats": dict(cache.stats),
    }


def voltage_sweep(n: int, voltages=(860.0, 1800.0, 3600.0)) -> dict[str, Any]:
    """Same geometry, different drive voltage: the matrix is unchanged so the
    ILU is reused, but the RHS differs so the solution cache must MISS.

    Forced onto the iterative path so the factorization cache is exercised even
    at a test-affordable n (heatr3d's auto threshold is N > 50_000)."""
    from dataclasses import replace
    grid = h3.Grid(n=n)
    part = h3.make_geometry(grid, "square", diam=0.020, zspan=0.020)
    gamma = h3.build_gamma(part, P, None)
    cache = EqsCache()
    times, ok = [], []
    for v in voltages:
        p = replace(P, v_lo=v)
        V, dt = _timed(lambda p=p: solve_eqs_3d_cached(gamma, grid, p,
                                                       iterative=True, cache=cache))
        ref = h3.solve_eqs_3d(gamma, grid, p, iterative=True)
        times.append(dt)
        ok.append(bool(np.array_equal(V, ref)))
    return {
        "n": n, "voltages": list(voltages),
        "t_per_solve_s": times,
        "first_solve_s": times[0],
        "mean_reused_solve_s": float(np.mean(times[1:])) if len(times) > 1 else None,
        "all_bit_identical_vs_heatr3d": all(ok),
        "stats": dict(cache.stats),
    }


def renorm_survey() -> list[dict[str, Any]]:
    """Recorded evidence for THE RENORM QUESTION.

    Sweeps gamma -> c*gamma over grid sizes and scale factors and records
    whether V and the renormalised Qrf come back bit-identical. The answer is
    'sometimes', which is why the shortcut is rejected: a rule that is exact at
    n=20 and inexact at n=24 would pass a cheap gate and then corrupt
    production."""
    from .eqs_cache import scale_invariance_deviation
    out = []
    for n in (16, 20, 24, 28):
        grid = h3.Grid(n=n)
        part = h3.make_geometry(grid, "square", diam=0.020, zspan=0.020)
        gamma = h3.build_gamma(part, P, None)
        for c in (2.0, 10.0, 1000.0):
            d = scale_invariance_deviation(gamma, grid, P, part, c)
            d["n"] = n
            out.append(d)
    return out


def main(out_path: str | Path | None = None, big: bool = True) -> dict[str, Any]:
    recs: dict[str, Any] = {
        "express_n48": express_scenario(48),
        "voltage_sweep_n32": voltage_sweep(32),
        "renorm_survey": renorm_survey(),
    }
    if big:
        recs["express_n96"] = express_scenario(96)
    recs["env"] = {
        "numpy": np.__version__,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "uptime": os.popen("uptime").read().strip(),
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(recs, indent=2, default=float))
    return recs


if __name__ == "__main__":
    import sys
    big = "--no-n96" not in sys.argv
    here = Path(__file__).resolve().parent
    r = main(here / "eqs_cache_bench.json", big=big)
    for key in ("express_n48", "express_n96"):
        if key not in r:
            continue
        e = r[key]
        print(f"{key}  n={e['n']}  {e['path']}")
        print(f"   BEFORE (miss) {e['t_before_miss_s']:8.2f} s | "
              f"AFTER (miss) {e['t_after_miss_s']:8.2f} s | "
              f"VERIFY (hit) {e['t_verify_hit_s']:.6f} s")
        print(f"   uncached verify {e['t_verify_uncached_s']:8.2f} s -> "
              f"SAVED {e['saved_per_hit_s']:8.2f} s per hit")
        print(f"   after-arm missed as required: {e['after_arm_missed_as_required']} | "
              f"hit bit-identical: {e['verify_hit_is_bit_identical']} | "
              f"after != before: {e['after_field_differs_from_before']}")
    v = r["voltage_sweep_n32"]
    print(f"voltage_sweep n={v['n']} (ILU reuse): first {v['first_solve_s']:.2f} s -> "
          f"reused {v['mean_reused_solve_s']:.2f} s | "
          f"bit-identical vs heatr3d: {v['all_bit_identical_vs_heatr3d']} | "
          f"fac hits={v['stats']['factorization_hits']}")
