"""solve3d Stage B: the ceiling-coupled dopant solve (penalty + rho co-state).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -c "import json; from solve3d import stage_b; \
        print(json.dumps(stage_b.uniform_holdout_peak(), indent=2))"

WHY THIS EXISTS (spec 2026-08-07-stage-b-ceiling-coupled-dopant-design.md):
Stage A phase 2's unconstrained shape-optimal dopant RELOCATED the densify
end-state peak ~+11 C (dolfinx) / +13 C (heatr3d) above the uniform 0.40x map,
pushing the hold-out true peak to 251.15 C -- 1.15 C OVER the 250 C degradation
ceiling. "The ceiling is nearly dopant-independent" therefore has a real LIMIT
at this drive, so the dopant needs a GRADIENT of the end-state peak (the rho+T
density co-state adjoint, solve3d.density_adjoint). This module assembles the
penalty objective J_shape + mu*(KS_peak - ceiling)_+^2 at the FIXED 0.40x drive.

B1 (density co-state) is FD-gated in solve3d.density_adjoint BEFORE any solve.
The TRUE end-state peak (never the KS aggregate) is what is_shippable reads --
the false-green class is structurally unexpressible (shippable_verdict has no KS
path to a shippable verdict). rho_target and the ceiling are read from the shared
solve3d/thermal_config.json; the drive is read from the Phase 1 artifact via
stage_a_phase2. Nothing is restated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solve3d import stage_a, stage_a_phase2 as p2

RESULTS = Path(__file__).resolve().parent / "results"

# The FINE hold-out matching Stage A phase 2's arbiter mesh (run_solve used
# holdout_nodes=9000, holdout_lc0=0.060/64.0). Phase 1's 240.1 C uniform number
# was on the COARSER drive-sweep mesh, so the fine-mesh uniform peak is the
# unknown that bounds what any dopant can achieve at 0.40x (spec honest-null).
HOLDOUT_NODES = 9000
HOLDOUT_LC0_M = 0.060 / 64.0


def _rho_target() -> float:
    return float(stage_a.thermal_config()["rho_target"]["practical_ideal"])


def uniform_holdout_peak(holdout_nodes: int = HOLDOUT_NODES,
                         holdout_lc0: float = HOLDOUT_LC0_M,
                         rho_target: float | None = None,
                         max_time_s: float = 3000.0) -> dict:
    """The UNIFORM (s=1) dopant true end-state peak at 0.40x on the hold-out.

    Reuses stage_a_phase2.ceiling_end_state_gate with a uniform saturation map:
    a single-cell source map of value 1.0 transfers to every hold-out part cell
    as saturation 1.0 (the nearest-neighbour query lands on the one source cell).
    This is a handful of densify forwards, not a heavy solve.

    Bounds B2 feasibility: if this is already > 250 C, 0.40x is infeasible even
    for the uniform map and B2 will honest-null. Per the cross-engine table
    (uniform 240.1 dolfinx / 244.3 heatr3d on a coarser mesh) it should be under
    250, but that is CONFIRMED here on the fine mesh, never assumed.
    """
    rho_t = _rho_target() if rho_target is None else float(rho_target)
    gate = p2.ceiling_end_state_gate(
        np.ones(1), np.zeros((1, 3)), holdout_nodes=int(holdout_nodes),
        holdout_lc0=float(holdout_lc0), rho_target=rho_t, max_time_s=max_time_s)
    out = dict(gate)
    out["map"] = "uniform_s1"
    out["rho_target"] = rho_t
    out["drive_a"] = p2.chosen_drive_a()
    out["power_density_w_per_m3"] = p2.chosen_drive_power_density()
    return out


def shippable_verdict(true_peak_c: float, ks_peak_c: float, ceiling_c: float,
                      fd_gate_passed: bool) -> dict:
    """is_shippable = (the gradient was FD-verified) AND (the TRUE end-state peak
    is at/under the ceiling). The KS aggregate is INFORMATIONAL only -- it has no
    path to a shippable verdict, so the false-green class (KS under while the true
    peak is over) is structurally unexpressible (spec sec "False-green guard").

    reason precedence: an unverified gradient disqualifies before any physics is
    trusted; then the true-peak ceiling.
    """
    true_peak = float(true_peak_c)
    ceil = float(ceiling_c)
    over = bool(true_peak > ceil)
    if not fd_gate_passed:
        reason = "fd_gate_not_passed"
    elif over:
        reason = "over_ceiling_true_peak"
    else:
        reason = "shippable"
    return {
        "is_shippable": bool(fd_gate_passed and not over),
        "reason": reason,
        "true_peak_c": true_peak,
        "ceiling_c": ceil,
        "over_by_c": true_peak - ceil,
        "fd_gate_passed": bool(fd_gate_passed),
        "ks_peak_c_informational": float(ks_peak_c),
        "rule": "is_shippable = fd_gate_passed AND true_peak <= ceiling; the KS "
                "aggregate is never decisive",
    }


def null_verdict(best_true_peak_c: float, ceiling_c: float) -> dict:
    """The honest-null (spec B2 clause, mirrors Stage A select_from_sweep): if even
    the ceiling-optimal (peak-minimizing) dopant leaves the true hold-out peak over
    the ceiling, 0.40x is infeasible for EVERY dopant -- report it with the achieved
    min-peak as evidence rather than shipping an over-ceiling map. That means the
    drive is too high (escalate to B4 / drive backoff), NOT a machinery failure."""
    best = float(best_true_peak_c)
    ceil = float(ceiling_c)
    over = bool(best > ceil)
    return {
        "verdict": "no_feasible_dopant_at_this_drive" if over
        else "feasible_dopant_exists",
        "best_true_peak_c": best,
        "ceiling_c": ceil,
        "margin_c": ceil - best,
        "note": "the peak-minimizing dopant's true hold-out peak vs the ceiling",
    }


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float))
    tmp.replace(path)
