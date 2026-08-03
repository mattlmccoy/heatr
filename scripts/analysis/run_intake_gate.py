#!/usr/bin/env python3
"""FINITE-DIFFERENCE GATE on the NOVEL imported geometry.

The generalization layer introduces no new gradient: the static arm uses
`adjoint.gradient` and the mode arms use `rot_kernel.AveragedKernel.gradient`,
both already gated by earlier passes. But those gates were run on library
shapes, and `CONTINUOUS_ROTATION_REPORT.md` limit 1 names "the gate was run on
one shape and the others are ASSUMED to inherit it" as an open limit. An
imported geometry is exactly the case where that assumption should not be
inherited for free, so the gate is re-run here on the geometry the pipeline was
actually asked to solve.

Three layers, so a failure localizes, reusing `gate_rot.gate_layer` unchanged:

  R0  one averaging angle at zero degrees, unfiltered. With a single angle at
      zero the averaged kernel reduces BIT FOR BIT to `forward.forward` and its
      gradient to `adjoint.gradient` (proven in `tests/test_rot_kernel.py`), so
      this layer IS the static solve's gradient on this geometry.
  R1  the full 24-angle set, unfiltered: adds both rotation transposes.
  R2  the full 24-angle set WITH the 1.0 mm physical-length design filter,
      which is the gradient the mode co-solve actually descends.

Standing protocol: central differences, epsilon swept over the campaign's eight
values, four to five probes per layer (maximum-sensitivity cell, a fixed
pseudo-random in-part cell, a rough random direction, a filter-smooth direction
on the filtered layer, and the gradient direction). Pass standard 1e-6
preferred, 1e-5 is the campaign's documented subgradient standard because the
objective reads a CLIPPED melt fraction and cells at the clip are kinks.

Run:
  ./.venv312/bin/python scripts/analysis/run_intake_gate.py gear8
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import design_filter as df, gradops, topopt        # noqa: E402
from adjoint2d import geometry_calibrate as gcal                  # noqa: E402
from adjoint2d import geometry_intake as gi                       # noqa: E402
from adjoint2d import shape_objective as so                       # noqa: E402
from adjoint2d.gate_rho import default_v                          # noqa: E402
from adjoint2d.gate_rot import (N_STEPS_GATE, gate_layer,         # noqa: E402
                                stop_index_stability)
from adjoint2d.rot_frame import averaging_angles                  # noqa: E402
from adjoint2d.rot_kernel import AveragedKernel                   # noqa: E402
from novel_shapes import NOVEL                                    # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
FILTER_RADIUS_M = 1.0e-3


def main(name: str, step_deg: float = 15.0) -> dict:
    t0 = time.perf_counter()
    it = gcal.calibrate_intake(gi.from_polygon(NOVEL[name](), grid=120, name=name))
    angles = averaging_angles(float(step_deg))
    kern1 = AveragedKernel.build(it.cfg, angles=np.array([0.0]))
    kernM = AveragedKernel.build(it.cfg, angles=angles)
    case = kernM.case0
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = topopt.sigma_cells_for(FILTER_RADIUS_M, case.dx)
    v0 = default_v(case)

    base = kernM.forward(np.where(case.part_mask, v0, 1.0), n_steps=N_STEPS_GATE)
    st = so.optimal_stop(base, case)
    res = {"shape": name, "source": "imported polygon (novel geometry)",
           "voltage_v": float(it.cfg["electric"]["voltage_v"]),
           "n_part_cells": int(case.part_mask.sum()),
           "objective": "J_phi = sum over the WHOLE domain of (phi - chi_part)^2, "
                        "raster chi, the gate's own convention",
           "angles_deg": [float(a) for a in angles],
           "sigma_cells": float(sigma_cells), "n_steps_gate": N_STEPS_GATE,
           "base": {"read_index": int(st.index), "J_phi": st.J,
                    "at_horizon": bool(st.at_horizon)},
           "layers": []}

    for lname, k, sig in (("R0_single_angle_zero_unfiltered", kern1, 0.0),
                          ("R1_full_angle_set_unfiltered", kernM, 0.0),
                          ("R2_full_angle_set_filtered", kernM, float(sigma_cells))):
        r = gate_layer(k, lname, ops, v0, st.index, sig)
        res["layers"].append(r)
        extra = (f"smoothdir={r['probes']['smooth_random_direction']['best_rel_err']:.3e} "
                 if "smooth_random_direction" in r["probes"] else "")
        print(f"{r['layer']:34s} M={r['n_angles']:3d} J0={r['J0']:.6f} "
              f"maxcell={r['probes']['max_sensitivity_cell']['best_rel_err']:.3e} "
              f"randcell={r['probes']['random_cell']['best_rel_err']:.3e} "
              f"randdir={r['probes']['random_direction']['best_rel_err']:.3e} "
              + extra
              + f"graddir={r['probes']['gradient_direction']['best_rel_err']:.3e} "
              + f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
                f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    res["stop_index_stability"] = stop_index_stability(kernM, v0, float(sigma_cells))
    res["ALL_GATES_PASS"] = all(l["PASS"] for l in res["layers"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = all(l["PASS_subgradient"] for l in res["layers"])
    res["wall_s"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gate_{name}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{name}] ALL_GATES_PASS at 1e-6 = {res['ALL_GATES_PASS']}, at the 1e-5 "
          f"subgradient standard = {res['ALL_GATES_PASS_SUBGRADIENT']}, "
          f"stop index moved = {res['stop_index_stability']['moved']}, "
          f"wall {res['wall_s']:.1f} s")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gear8")
