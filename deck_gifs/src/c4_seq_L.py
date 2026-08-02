"""Compute stage for gif_sequential_L: the L_shape under its DELIVERABLE
sequential program (hold 90 degrees, quarter-turn to 0 degrees), capturing T
and rho snapshots to the recommended stop.

Read-only reuse of adjoint2d.seq_dwell_march (settled code). The march is the
campaign's own `sequential_forward` with checkpoints kept; nothing is
re-implemented.

Verification built in: the per-step J_phi is compared against the stored
campaign curve `J_curve_S_seq_cosolved_4bpp_interior_switch` from
out_seq/L_shape_seq_maps.npz, and J at the stop against the stored 276.3893.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import library_solve as lib               # noqa: E402
from adjoint2d import seq_dwell_march as sqm             # noqa: E402
from adjoint2d import shape_objective as so              # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel           # noqa: E402
from adjoint2d.pins import load_cfg                      # noqa: E402

OUT_SEQ = REPO / "fgm_solve_campaign/out_seq"
CACHE = REPO / "deck_gifs/cache/c4_seq_L.npz"
ARM = "S_seq_cosolved_4bpp_interior_switch"
SNAP_EVERY = 4


def main() -> None:
    t0 = time.perf_counter()
    r = json.loads((OUT_SEQ / "L_shape_arms.json").read_text())
    kink = json.loads((OUT_SEQ / "L_shape_kink.json").read_text())
    arms = dict(r["arms"])
    arms.update(kink["arms"])
    a = arms[ARM]
    maps = np.load(OUT_SEQ / "L_shape_seq_maps.npz")

    angles = np.asarray(r["angles_deg"], dtype=float)
    dt = float(r["dt_s"])
    seg = [int(np.argmin(np.abs(angles - x))) for x in a["segments_deg"]]
    dur = np.asarray(a["durations_s"], dtype=float)
    stop_index = int(a["t_stop_index"])
    n_steps = stop_index + 1
    sat = np.asarray(maps[f"sat_{ARM}"], dtype=float)
    jc_stored = np.asarray(maps[f"J_curve_{ARM}"], dtype=float)
    print(f"segments {a['segments_deg']} durations {dur} stop index "
          f"{stop_index} ({(stop_index + 1) * dt:.1f} s)", flush=True)

    kern = DwellKernel.build(load_cfg(lib.shape_config("L_shape")),
                             angles=angles)
    case = kern.case0
    p = case.pins
    s = np.where(case.part_mask, np.clip(sat, 0.0, 1.0), 1.0)
    kern.averaged_Q(s)
    print(f"kernel built, wall {time.perf_counter() - t0:.0f} s", flush=True)

    tr = sqm.sequential_forward(kern, seg, dur, n_steps, keep_checkpoints=True)
    j_curve = so.J_curve(tr, case)
    dj = np.abs(j_curve - jc_stored[:n_steps])
    print(f"J check against stored campaign curve: max abs diff {dj.max():.3e} "
          f"(J at stop {j_curve[stop_index]:.4f} vs stored arm J {a['J']:.4f})",
          flush=True)

    steps = list(range(0, n_steps, SNAP_EVERY))
    if steps[-1] != n_steps - 1:
        steps.append(n_steps - 1)
    snaps_T = np.asarray([tr.T_at_end(i) for i in steps], dtype=np.float32)
    snaps_rho = np.asarray([tr.rho_at_end(i) for i in steps], dtype=np.float32)

    np.savez_compressed(
        CACHE,
        snaps_T=snaps_T, snaps_rho=snaps_rho, snap_steps=np.asarray(steps),
        j_curve=j_curve, j_curve_stored=jc_stored[:n_steps],
        j_check_max_abs_diff=float(dj.max()),
        part_mask=case.part_mask, x=case.x, y=case.y, sat=s, dt_s=dt,
        stop_index=stop_index, switch_s=float(dur[0]),
        segments_deg=np.asarray(a["segments_deg"], dtype=float),
        J_at_stop=float(a["J"]), IoU_at_stop=float(a["IoU"]),
        t_pc_c=p.t_pc_c, dt_pc_c=p.dt_pc_c, ambient_c=p.ambient_c,
        wide_limb=np.asarray(maps["wide_limb"], dtype=bool),
        narrow_limb=np.asarray(maps["narrow_limb"], dtype=bool),
    )
    print(f"wrote {CACHE}  wall {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
