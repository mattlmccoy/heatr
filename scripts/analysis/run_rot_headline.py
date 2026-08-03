#!/usr/bin/env python3
"""The four headline arms per shape, re-run with STOP-TIME field snapshots.

Every metric in this campaign is read at the arm's own J-stop, so a figure
drawn from the end-of-horizon field is not the figure the numbers describe.
`turntable_glue` now snapshots temperature, density and the part mask at the
running J minimum; this driver re-runs only the four arms the headline figure
shows and stores those snapshots.

Period per shape is the BEST GATE-PASSING rotation period measured by the speed
sweep, quoted in the report; the arms are otherwise identical to Level 2.

Run:
  ./.venv312/bin/python scripts/analysis/run_rot_headline.py <shape>
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

from scripts.analysis.run_rot_verify import (OUT_ROOT, base_cfg, joint_winner_map,  # noqa: E402
                                             realized_period_s, rotavg_maps,
                                             stored_zero_map, with_turntable)
from scripts.analysis.turntable_glue import run_rotating  # noqa: E402

# best gate-passing rotation period per shape, from the speed sweep
BEST_PERIOD_S = {"T_shape": 720.0, "L_shape": 360.0, "cross": 180.0,
                 "star": 180.0, "square": 60.0}


def one(name, cfg, sat, extra, log) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    r = run_rotating(cfg, sat_map=sat, corotate=True, corotate_eps=False,
                     record=True, quiet=True)
    m = r.at_stop()
    m["arm"] = name
    m["n_rotation_events"] = len(r.rotation_events)
    m["over_ceiling_250c"] = bool(m["max_T_part_c"] > 250.0)
    m["wall_s"] = time.perf_counter() - t0
    m.update(extra)
    log(f"  {name:12s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
        f"grow {m['bed_melt_pct_of_part']:6.2f}%  under {m['part_under_melt_pct']:6.2f}%  "
        f"stop {m['t_stop_s']:6.1f} s{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
        f"Egate {'PASS' if m['energy_gate_PASS'] else 'FAIL'} "
        f"({100 * m['energy_residual_rel']:.2f}%)  [{m['wall_s']:.0f} s]")
    return m, {f"phi_{name}": r.phi_at_stop.astype(np.float32),
               f"mask_{name}": r.mask_at_stop}


def main(shape: str) -> dict:
    def log(msg):
        print(f"[{shape}/headline] {msg}", flush=True)

    P = BEST_PERIOD_S[shape]
    Pr = realized_period_s(P)
    sat0, _m0 = stored_zero_map(shape)
    sat_j, ang_j, _mj = joint_winner_map(shape)
    sat_a, _sa4, meta_a = rotavg_maps(shape)
    cfgP = with_turntable(base_cfg(shape, 0.0), P)
    log(f"best gate-passing period {Pr:.0f} s")

    rows, fields = [], {}
    for name, cfg, sat, ex in (
            ("S_uniform", base_cfg(shape, 0.0), None, {"rotating": False}),
            ("S_best_static", base_cfg(shape, ang_j if sat_j is not None else 0.0),
             sat_j if sat_j is not None else sat0,
             {"rotating": False, "rotation_deg": ang_j if sat_j is not None else 0.0,
              "which": "joint winner" if sat_j is not None else "best zero-degree map"}),
            ("R_uniform", cfgP, None, {"rotating": True, "period_s": Pr}),
            ("R_avg", cfgP, sat_a, {"rotating": True, "period_s": Pr,
                                    "map_source": meta_a})):
        m, f = one(name, cfg, sat, ex, log)
        rows.append(m)
        fields.update(f)

    np.savez_compressed(OUT_ROOT / "verify_fields" / f"{shape}_headline.npz", **fields)
    out = {"shape": shape, "period_s": Pr, "rows": rows,
           "note": "fields are the melt state at each arm's OWN J-stop"}
    (OUT_ROOT / f"{shape}_headline.json").write_text(json.dumps(out, indent=1, default=float))
    log("DONE")
    return out


if __name__ == "__main__":
    main(sys.argv[1])
