#!/usr/bin/env python3
"""SEQUENTIAL DWELL, step 0: which single orientation melts which limb?

The sequential hypothesis needs limb-aligned angles, and the campaign has never
measured the limb-resolved melt on a 15-degree grid. This is that measurement
and nothing else: one uniform-dopant static run per candidate orientation, in
the PART frame, scored at each orientation's own stop, with the part split into
its two limbs by row width.

Only the HALF-TURN-DISTINCT angles are run, because the part-frame heating at
theta and theta + 180 degrees is the same field to 4e-13 relative
(`DWELL_SCHEDULE_REPORT.md` Section 2).

Run:
  ./.venv312/bin/python scripts/analysis/run_seq_probe.py <shape>
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

from adjoint2d import energy_gate as eg, library_solve as lib    # noqa: E402
from adjoint2d import seq_dwell_march as sqm                     # noqa: E402
from adjoint2d import shape_objective as so                      # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                   # noqa: E402
from adjoint2d.pins import load_cfg                              # noqa: E402

ANGLES = np.arange(0.0, 180.0, 15.0)
N_STEPS = 1500
DT_S = 0.5
PATIENCE = lib.PATIENCE
OUT = REPO / "fgm_solve_campaign/out_seq"


def limb_masks(part_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a T or an L into its WIDE limb and its NARROW limb, by row width.

    Rows of the part mask have exactly two widths on these shapes (measured in
    `run_seq_probe`'s own output). The wide rows are the crossbar of the T and
    the long arm of the L; the narrow rows are the stem.
    """
    w = part_mask.sum(axis=1)
    rows = np.flatnonzero(w > 0)
    thresh = 0.5 * (w[rows].max() + w[rows].min())
    wide = np.zeros_like(part_mask)
    narrow = np.zeros_like(part_mask)
    for r in rows:
        (wide if w[r] > thresh else narrow)[r] = part_mask[r]
    return wide, narrow


def limb_report(T, case, wide, narrow) -> dict:
    phi, _ = so.phi_field(T, case)
    melted = phi >= so.MELT_LEVEL
    return {
        "wide_limb_cells": int(wide.sum()),
        "narrow_limb_cells": int(narrow.sum()),
        "wide_melted_pct": 100.0 * float(np.sum(melted & wide)) / max(int(wide.sum()), 1),
        "narrow_melted_pct": 100.0 * float(np.sum(melted & narrow)) / max(int(narrow.sum()), 1),
    }


def main(shape: str) -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(lib.shape_config(shape))
    kern = DwellKernel.build(cfg, angles=ANGLES)
    case = kern.case0
    pm = case.part_mask
    wide, narrow = limb_masks(pm)
    print(f"[{shape}] {int(pm.sum())} part cells, wide limb {int(wide.sum())}, "
          f"narrow limb {int(narrow.sum())}, {ANGLES.size} distinct angles",
          flush=True)

    s = np.ones(pm.shape)
    kern.set_weights(np.full(ANGLES.size, 1.0 / ANGLES.size))
    te = time.perf_counter()
    kern.averaged_Q(s)
    print(f"[{shape}] electro-quasi-static pass over all angles: "
          f"{time.perf_counter() - te:.1f} s", flush=True)

    rows = []
    for j, th in enumerate(ANGLES):
        tr = sqm.sequential_forward(kern, [j], [N_STEPS * DT_S], N_STEPS,
                                    shape_stop_patience=PATIENCE)
        m = so.full_metrics(tr, case)
        i = int(m["t_stop_index"])
        T = tr.T_at_end(i)
        m.update(limb_report(T, case, wide, narrow))
        m["angle_deg"] = float(th)
        m["max_T_at_stop_c"] = float(np.max(T))
        m["over_ceiling_250c"] = bool(m["max_T_at_stop_c"] > 250.0)
        m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
        m["P_abs_W_per_m"] = tr.P_abs_B
        m["energy_gate"] = eg.gate_from_trajectory(tr, i)
        rows.append(m)
        print(f"  {th:6.1f} deg  J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
              f"wide {m['wide_melted_pct']:6.2f}%  narrow {m['narrow_melted_pct']:6.2f}%  "
              f"stop {m['t_stop_s']:6.1f} s{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
              f"maxT {m['max_T_at_stop_c']:6.1f} C"
              f"{'  CEILING' if m['over_ceiling_250c'] else ''}", flush=True)
        del tr

    out = {"shape": shape, "angles_deg": [float(a) for a in ANGLES],
           "n_steps": N_STEPS, "dt_s": DT_S, "patience": PATIENCE,
           "map": "uniform saturation 1.0", "rows": rows,
           "wall_s": time.perf_counter() - t0}
    (OUT / f"{shape}_probe.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[{shape}] wrote {OUT / f'{shape}_probe.json'} in {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L_shape")
