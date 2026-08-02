#!/usr/bin/env python3
"""SEQUENTIAL DWELL: re-run the PREVIOUS pass's own deliverable, as the baseline.

The contrast arm has to be the arm actually on record, which for the T_shape is
a dopant map solved jointly with a CYCLED dwell vector and executed at the 20 s
cycle, not that map held at one angle. This script re-runs both shapes'
best-on-record arms from `out_dwell/` under this pass's scoring, and by doing so
also acts as a cross-pass reproduction check: the numbers should come back equal
to the stored ones.

Run:
  ./.venv312/bin/python scripts/analysis/run_seq_prior_baseline.py
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

from adjoint2d import dwell, dwell_march as dmarch                    # noqa: E402
from adjoint2d import library_solve as lib                            # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel                        # noqa: E402
from adjoint2d.pins import load_cfg                                   # noqa: E402
from run_seq_arms import CYCLE_ANGLES, CYCLE_TIME_S, log_row, score   # noqa: E402
from run_seq_probe import DT_S, N_STEPS, limb_masks                   # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_seq"
OUT_DWELL = REPO / "fgm_solve_campaign/out_dwell"

# The best arm on record for each shape, by J, from `DWELL_SCHEDULE_REPORT.md`
# Section 6 and the stored `out_dwell/<shape>_dwell.json`.
PRIOR = {
    "L_shape": {"arm": "D_refined_4bpp_discovered_lib", "stored_J": 396.20,
                "stored_IoU": 0.6484},
    "T_shape": {"arm": "D_refined_4bpp_discovered_lib", "stored_J": 426.21,
                "stored_IoU": 0.6415},
}


def main() -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for shape, meta in PRIOR.items():
        j = json.loads((OUT_DWELL / f"{shape}_dwell.json").read_text())
        a = j["arms"][meta["arm"]]
        w = np.asarray(a["dwell_weights"], dtype=float)
        sat = np.load(OUT_DWELL / f"{shape}_dwell_maps.npz")[f"sat_{meta['arm']}"]
        cfg = load_cfg(lib.shape_config(shape))
        kern = DwellKernel.build(cfg, angles=CYCLE_ANGLES)
        case = kern.case0
        pm = case.part_mask
        wide, narrow = limb_masks(pm)
        s = np.where(pm, np.clip(np.asarray(sat, dtype=float), 0.0, 1.0), 1.0)
        kern.set_weights(w)
        kern.averaged_Q(s)
        prog = dwell.cycle_program(w, CYCLE_ANGLES, cycle_time_s=CYCLE_TIME_S,
                                   total_s=N_STEPS * DT_S, dt_s=DT_S)
        idx = dmarch.program_step_positions(prog, CYCLE_ANGLES, DT_S, N_STEPS)
        tr = dmarch.program_forward(kern, idx, N_STEPS)
        m, _phi, _jc = score(tr, case, s, wide, narrow,
                             {"note": "the PREVIOUS pass's best arm on record, "
                                      "re-run time-resolved at the 20 s cycle",
                              "schedule": "cycled", "source_arm": meta["arm"],
                              "stored_J": meta["stored_J"],
                              "stored_IoU": meta["stored_IoU"],
                              "dwell_weights": [float(x) for x in w],
                              "n_moves": len(prog.moves)})
        m["arm"] = "S_prior_best_on_record"
        m["reproduction_delta_J_pct"] = 100.0 * (m["J"] - meta["stored_J"]) / meta["stored_J"]
        log_row(shape, m)
        print(f"[{shape}] stored {meta['stored_J']:.2f} / {meta['stored_IoU']:.4f} "
              f"against re-run {m['J']:.2f} / {m['IoU']:.4f} "
              f"({m['reproduction_delta_J_pct']:+.3f} %)", flush=True)
        out[shape] = m
        del kern, tr
    out["wall_s"] = time.perf_counter() - t0
    (OUT / "prior_baseline.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT / 'prior_baseline.json'} in {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main()
