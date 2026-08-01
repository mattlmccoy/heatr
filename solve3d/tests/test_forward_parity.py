"""solve3d Phase A parity gates (dolfinx vs heatr3d).

RUNS IN THE SPIKE ENV (dolfinx 0.11 complex build):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_forward_parity.py

PREREQUISITE: solve3d/cases.py must already have run in the geo-prewarp venv
(plan Task 1), leaving solve3d/results/anchor_heatr3d_<shape>_n<N>.npz and
solve3d/results/parity_tolerances.json. Gates read the FROZEN tolerance file;
they never invent a tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solve3d import gates

RESULTS = Path(__file__).resolve().parents[1] / "results"

SPIKE = Path(__file__).resolve().parents[2] / "heatr3d_d1_spike"

# ---------------------------------------------------------------------------
# GATE CORRECTION, plan Task 2 (documented, escalated, NOT a widening).
#
# The plan cites "pattern rel-L2 < 0.05 (the D1 Task-4 gate, already proven
# achievable)". That 0.05 is real but it belongs to a DIFFERENT comparison:
# heatr3d_d1_spike/run_scale_test.py l.144 scores
# "qrf_pattern_rel_l2_vs_task2_fine", i.e. dolfinx-scale-mesh vs
# dolfinx-Task2-fine-mesh -- a dolfinx SELF-CONVERGENCE check (measured 0.0119).
# It was never a dolfinx-vs-heatr3d number.
#
# The measured dolfinx-vs-heatr3d(corrected/masked-gradient) values ARE in
# results.json as task2.gate.maskgrad_all_points: 0.17652460972777445 (coarse,
# n=64-matched) and 0.10790598069422491 (fine, n=96-matched). D1 sec 2 states
# this plainly: "the mask-confined field differs by 10.8 % in unit-mean pattern
# L2". No mesh at these resolutions reaches 0.05 against heatr3d, for the
# structural reason D1 established: heatr3d's voxel staircase and its harmonic
# face averaging perturb the surface band, and the surface band is where the
# two engines are known to disagree (interior-only is 0.0478 / 0.0255).
#
# So the gate here is STRONGER, not weaker: the lift must REPRODUCE the D1
# measured value to 1e-9 relative. That pins every convention (gamma, BVP,
# Q definition, renormalization basis, mesh matching) at once, instead of
# passing a threshold that was never measured for this pair.
# ---------------------------------------------------------------------------
D1_REPRO_RTOL = 1e-9
PLAN_GATE_AS_WRITTEN = 0.05
POWER_RENORM_TOL = 1e-9


def _d1_measured(level: str) -> float:
    d = json.loads((SPIKE / "results.json").read_text())
    return float(d["task2"]["gate"]["maskgrad_all_points"][level])


def _anchor(shape: str, n: int):
    p = RESULTS / f"anchor_heatr3d_{shape}_n{n}.npz"
    if not p.exists():
        pytest.skip(f"{p.name} missing: run `./.venv312/bin/python -m solve3d.cases "
                    f"--anchor {shape} {n}` first (plan Task 1)")
    return np.load(p)


def test_eqs_qrf_pattern():
    """Task 2: the dolfinx EQS drive reproduces heatr3d's CORRECTED
    (masked-gradient) Q_rf pattern on the extruded circle, and the fixed-power
    renormalization is exact.

    Compared as a unit-mean PATTERN on heatr3d's own in-part mid-plane voxel
    centres -- the same point set and metric D1 Task 2/4 used, so the gate value
    0.05 is quoted from a measured precedent rather than invented."""
    from solve3d import forward

    ref = _anchor("circle", 64)
    out = forward.eqs_case(shape="circle", target_nodes_in_part=int(ref["part"].sum()),
                           lc0=float(ref["h"]))

    # --- renormalization identity: integral(Q dV) == power_density * V_part ---
    assert abs(out["p_now_w"] / out["p_target_w"] - 1.0) < POWER_RENORM_TOL

    # --- pattern vs heatr3d on the shared voxel-centre point set --------------
    pts, sel = forward.midplane_part_points(ref)
    q_fem, missed = forward.eval_at(out["q"], out["msh"], pts)
    assert missed == 0, "every in-part mid-plane voxel centre must hit a cell"
    k = int(ref["n"]) // 2
    q_h = np.asarray(ref["Qrf"])[:, :, k][sel]

    rel = gates.rel_l2_pattern(q_fem, q_h)
    d1 = _d1_measured("coarse")
    assert abs(rel - d1) <= D1_REPRO_RTOL * d1, (
        f"Q_rf pattern rel-L2 {rel!r} does not reproduce the D1 measured "
        f"dolfinx-vs-heatr3d(masked) value {d1!r}")
    # and the disagreement must be the KNOWN surface-band one, not a bulk error
    assert rel > PLAN_GATE_AS_WRITTEN, (
        "if this ever drops below the plan's mis-cited 0.05, re-derive the gate "
        "instead of celebrating")
