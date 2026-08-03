"""Dose-matched control for the temporal power-scheduling campaign.

THE CONFOUND THIS EXISTS TO REMOVE. The continuous schedule box is [0, 1.5], so
an optimizer that simply raises every segment to the ceiling buys extra energy,
not temporal structure. Any improvement it reports would then be a dose result
wearing a scheduling label.

The control is one extra forward per arm. Take the arm's optimized schedule,
read its time-weighted duty cycle d over the schedule window, and re-score the
SAME dopant map with a CONSTANT schedule p == d. That constant schedule delivers
the same time-integrated power scale over the window and has no temporal
structure at all.

  * If the constant control matches the optimized schedule, the schedule bought
    DOSE. The honest statement is a null on temporal structure.
  * If the optimized schedule beats its own constant control, the difference is
    what the temporal STRUCTURE bought, and that difference is the result.

A second control at constant p == 1.0 is reported alongside, which is the
nominal-power reference with the same map.

Run:
  ./.venv312/bin/python -m adjoint2d.sched_dose_control <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .library_solve import shape_config
from .pins import build_case, load_cfg
from .sched_solve import score

CONTROLLED_ARMS = ("SCHED_only", "CO_cont", "CO_4bpp", "BIN_relax", "BIN_round",
                   "WARM_sched_only", "WARM_CO_cont", "WARM_CO_4bpp")


def main(shape: str, outdir: str, stem: str = "sched") -> dict:
    out = Path(outdir).resolve()
    res = json.loads((out / f"{shape}_{stem}.json").read_text())
    case = build_case(load_cfg(shape_config(shape)))
    mfile = out / f"{shape}_{stem}_maps.npz"
    if not mfile.exists():
        mfile = out / f"{shape}_{stem.replace('sched', 'sched')}_maps.npz"
    maps = np.load(mfile)
    n_seg = res["n_seg"]
    horizon = res["horizon_steps"]
    window = res["schedule_window_steps"]
    t0 = time.perf_counter()

    rows = []
    for arm in CONTROLLED_ARMS:
        if arm not in res["arms"]:
            continue
        m = res["arms"][arm]
        s = np.asarray(maps[f"map_{arm}"], dtype=float)
        duty = float(m["duty_cycle"])
        c_duty = score(case, s, np.full(n_seg, duty), n_seg, horizon, window=window)
        c_one = score(case, s, np.ones(n_seg), n_seg, horizon, window=window)
        for k in ("J_curve", "rho_curve"):
            c_duty.pop(k, None)
            c_one.pop(k, None)
        rows.append({
            "arm": arm,
            "duty_cycle": duty,
            "scheduled": {"J": m["J"], "IoU": m["IoU"],
                          "t_stop_s": m["t_stop_s"],
                          "mean_rho_part_at_stop": m["mean_rho_part_at_stop"],
                          "structure": m["structure"]["structure"],
                          "n_switches": m["n_switches"]},
            "constant_at_duty": {"J": c_duty["J"], "IoU": c_duty["IoU"],
                                 "t_stop_s": c_duty["t_stop_s"],
                                 "mean_rho_part_at_stop": c_duty["mean_rho_part_at_stop"]},
            "constant_at_one": {"J": c_one["J"], "IoU": c_one["IoU"],
                                "t_stop_s": c_one["t_stop_s"],
                                "mean_rho_part_at_stop": c_one["mean_rho_part_at_stop"]},
            "structure_gain_dJ": c_duty["J"] - m["J"],
            "structure_gain_dIoU": m["IoU"] - c_duty["IoU"],
            "dose_gain_dIoU_over_p1": c_duty["IoU"] - c_one["IoU"],
        })
        r = rows[-1]
        print(f"[{shape}] {arm:12s} duty {duty:.3f} | scheduled J {m['J']:8.2f} "
              f"IoU {m['IoU']:.4f} | constant@duty J {c_duty['J']:8.2f} "
              f"IoU {c_duty['IoU']:.4f} | structure dIoU {r['structure_gain_dIoU']:+.4f} "
              f"| dose dIoU {r['dose_gain_dIoU_over_p1']:+.4f}", flush=True)

    doc = {"shape": shape, "n_seg": n_seg, "horizon_steps": horizon,
           "schedule_window_steps": window,
           "note": ("structure_gain is the optimized schedule minus its own "
                    "constant-at-duty control; dose_gain is that control minus "
                    "constant nominal power on the same map"),
           "rows": rows, "wall_s": time.perf_counter() - t0}
    (out / f"{shape}_{stem}_dose_control.json").write_text(
        json.dumps(doc, indent=2, default=float))
    return doc


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "sched")
