"""Compute stage for gif_dwell_cross: part-frame march of the cross under the
solved asymmetric dwell program, capturing T and rho snapshots.

Read-only reuse of fgm_solve_campaign/adjoint2d (forward physics, dwell kernel,
program step mapping). Nothing in adjoint2d is modified.

Verification built in: the per-step J_phi computed here is compared against the
stored campaign curve J_curve_D_refined_timeresolved_discovered_lib from
out_dwell/cross_dwell_maps.npz. Max abs difference is printed and saved.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import forward as fwd                     # noqa: E402
from adjoint2d import shape_objective as so              # noqa: E402
from adjoint2d.dwell_kernel import DwellKernel           # noqa: E402
from adjoint2d.pins import load_cfg                      # noqa: E402

OUT_DWELL = REPO / "fgm_solve_campaign/out_dwell"
CACHE = REPO / "deck_gifs/cache/c1_dwell_cross.npz"
SNAP_EVERY = 4          # outer steps between stored snapshots (2 s process time)
MAP_KEY = "sat_D_refined_4bpp_discovered_lib"
JC_KEY = "J_curve_D_refined_timeresolved_discovered_lib"


def main() -> None:
    t0 = time.perf_counter()
    deliv = json.loads((OUT_DWELL / "cross_turntable_deliverable.json").read_text())
    maps = np.load(OUT_DWELL / "cross_dwell_maps.npz")
    meta = json.loads((OUT_DWELL / "cross_dwell.json").read_text())

    cfg = load_cfg(meta["config"])
    angles = np.asarray(meta["candidate_angles_deg"], dtype=float)
    sat = np.asarray(maps[MAP_KEY], dtype=float)
    jc_stored = np.asarray(maps[JC_KEY], dtype=float)
    stop_index = int(np.argmin(jc_stored))
    n_steps = stop_index + 1
    dt = float(meta["dt_s"])
    print(f"stored stop index {stop_index} (t = {(stop_index + 1) * dt:.1f} s), "
          f"marching {n_steps} steps")

    kern = DwellKernel.build(cfg, angles=angles)
    case = kern.case0
    p = case.pins
    kern.averaged_Q(sat)     # populates the per-position part-frame fields

    # position index per outer step, straight from the deliverable's move list
    idx = np.zeros(n_steps, dtype=int)
    last = 0
    for m in deliv["moves"]:
        j = int(np.argmin(np.abs(angles - float(m["position_deg"]))))
        i0 = int(round(float(m["move_at_s"]) / dt))
        i1 = int(round((float(m["move_at_s"]) + float(m["dwell_s"])) / dt))
        if i0 >= n_steps:
            break
        idx[i0:min(i1, n_steps)] = j
        last = j
    end = int(round(sum(float(m["dwell_s"]) for m in deliv["moves"]) / dt))
    if end < n_steps:
        idx[end:] = last

    # the march, structurally dwell_march.program_forward with rho snapshots
    T = np.full(case.part_mask.shape, p.ambient_c, dtype=float)
    rho = np.zeros(case.part_mask.shape, dtype=float)
    rho[case.part_mask] = p.rho_rel_init
    ui = p.update_interval
    snaps_T, snaps_rho, snap_steps = [], [], []
    j_curve = np.zeros(n_steps)
    for it in range(n_steps):
        k = int(idx[it])
        use_b = ui > 0 and it >= ui
        Q = kern._Qk_b[k] if use_b else kern._Qk_a[k]
        for _ in range(p.n_substeps):
            T, rho, _phi, _c = fwd.substep(T, rho, Q, case, keep_cache=False)
        j_curve[it] = so.shape_J_and_seed(T, case)[0]
        if it % SNAP_EVERY == 0 or it == n_steps - 1:
            snaps_T.append(T.copy())
            snaps_rho.append(rho.copy())
            snap_steps.append(it)

    dj = np.abs(j_curve - jc_stored[:n_steps])
    print(f"J check against stored campaign curve: max abs diff {dj.max():.3e} "
          f"(J at stop {j_curve[-1]:.4f} vs stored {jc_stored[stop_index]:.4f})")

    np.savez_compressed(
        CACHE,
        snaps_T=np.asarray(snaps_T, dtype=np.float32),
        snaps_rho=np.asarray(snaps_rho, dtype=np.float32),
        snap_steps=np.asarray(snap_steps),
        pos_index=idx, angles_deg=angles, j_curve=j_curve,
        j_curve_stored=jc_stored[:n_steps],
        j_check_max_abs_diff=float(dj.max()),
        part_mask=case.part_mask, x=case.x, y=case.y,
        sat=sat, dt_s=dt, stop_index=stop_index,
        t_pc_c=p.t_pc_c, dt_pc_c=p.dt_pc_c, ambient_c=p.ambient_c,
        rho_rel_init=p.rho_rel_init,
    )
    print(f"wrote {CACHE}  wall {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
