#!/usr/bin/env python3
"""Rotation-speed sweep on the true engine: how fast must the turntable turn?

Two questions at once, and they pull in OPPOSITE directions in the rotation
period, which is what makes the sweep able to separate them:

  * the QUASI-STATIC approximation behind the Level-1 averaged kernel gets
    better as the period shortens;
  * the engine's own rotation-event REMAP error gets worse as the period
    shortens, because the number of bilinear remaps of temperature, density
    and melt fraction rises in proportion.

Both are reported against the same abscissa, the realized rotation period, with
the energy-residual gate on every row. Arms: uniform and the Level-1
averaged-kernel map, both co-rotated.

Run:
  ./.venv312/bin/python scripts/analysis/run_rot_speed_sweep.py <shape> <period_s> ...
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

from scripts.analysis.run_rot_verify import (OUT_ROOT, base_cfg, realized_period_s,  # noqa: E402
                                             rotavg_maps, run_arm, with_turntable)


def main(shape: str, periods: list[float]) -> dict:
    t0 = time.perf_counter()

    def log(msg):
        print(f"[{shape}/rotspeed] {msg}", flush=True)

    sat_a, _sat_a4, meta_a = rotavg_maps(shape)
    if sat_a is None:
        raise FileNotFoundError(f"Level 1 has not been run for {shape}")
    rows = []
    for P in periods:
        Pr = realized_period_s(P)
        cfgP = with_turntable(base_cfg(shape, 0.0), P)
        tag = f"P{int(round(Pr))}"
        log(f"period {Pr:.1f} s, event every "
            f"{cfgP['turntable']['rotation_interval_s']:.1f} s")
        ex = {"rotating": True, "period_s": Pr, "requested_period_s": float(P)}
        rows.append(run_arm(f"R_uniform_{tag}", cfgP, None, True, False, dict(ex), log)["metrics"])
        rows.append(run_arm(f"R_avg_{tag}", cfgP, sat_a, True, False,
                            dict(ex, map_source=meta_a), log)["metrics"])
    out = {"shape": shape, "mode": "speed sweep",
           "periods_realized_s": [realized_period_s(p) for p in periods],
           "rows": rows, "wall_s": time.perf_counter() - t0}
    p = OUT_ROOT / f"{shape}_speed.json"
    prev = json.loads(p.read_text())["rows"] if p.exists() else []
    seen = {r["arm"] for r in rows}
    out["rows"] = rows + [r for r in prev if r["arm"] not in seen]
    p.write_text(json.dumps(out, indent=1, default=float))
    log(f"DONE {len(rows)} arms, wall {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    main(sys.argv[1], [float(a) for a in sys.argv[2:]])
