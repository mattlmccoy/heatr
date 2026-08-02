"""Compute stage for the HEXAGON solve-vs-invert deliverables (round 2).

Re-runs, with snapshot capture, the three STORED library arms of
fgm_solve_campaign/out_lib/hexagon.json:

  HIST_best  best stored historical proportional-inverse mask
             (cal_map_m0p8327, outside1 convention, eps-co-varying channel)
  A1_4bpp    the solved single-pass 4 bits-per-pixel deliverable map
             (out_lib/hexagon_maps.npz key A1_4bpp, conductivity-only channel)
  U_uniform  uniform saturation s = 1 (final field only, for the static figure)

Each re-run is GATED against the stored arm's t_stop_index / J / IoU and the
deltas are printed. Additionally a FRESH small filtered adjoint solve (budget
15 gradient evaluations, cold start) is run purely as a process visualization
for the map-evolution phase of the GIF; its numbers are never captioned.

Read-only reuse of adjoint2d. Nothing in adjoint2d is modified. RESUMABLE:
the display solve appends each evaluation to c2h_partial.npz.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import adjoint, design_filter as df, forward as fwd  # noqa: E402
from adjoint2d import gradops, topopt                               # noqa: E402
from adjoint2d import shape_objective as so                         # noqa: E402
from adjoint2d.pins import build_case, load_cfg                     # noqa: E402
from adjoint2d.verify_hist import load_stored_map                   # noqa: E402

OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
CACHE = REPO / "deck_gifs/cache/c2h_hexagon.npz"
ARMS_CACHE = REPO / "deck_gifs/cache/c2h_arms.npz"
PARTIAL = REPO / "deck_gifs/cache/c2h_partial.npz"
N_EVALS = 15
PATIENCE = 250          # library_solve.py PATIENCE
SNAP_EVERY = 4


def melt_snaps(tr, stop_index: int) -> tuple[np.ndarray, np.ndarray]:
    steps = list(range(0, stop_index + 1, SNAP_EVERY))
    if steps[-1] != stop_index:
        steps.append(stop_index)
    T = np.asarray([tr.T_at_end(i) for i in steps], dtype=np.float32)
    return T, np.asarray(steps)


def rerun_arm(case, s: np.ndarray, eps_covary: bool, stored: dict,
              tag: str, keep_snaps: bool = True):
    """Forward-run a stored arm exactly as library_solve.score does; gate it."""
    tr = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                     shape_stop_patience=PATIENCE, eps_covary=eps_covary)
    st = so.optimal_stop(tr, case)
    m = so.full_metrics(tr, case)
    d_stop = st.index - int(stored["t_stop_index"])
    d_J = st.J - float(stored["J"])
    d_iou = m["IoU"] - float(stored["IoU"])
    print(f"GATE {tag}: stop {st.index} (stored {stored['t_stop_index']}, "
          f"delta {d_stop:+d})  J {st.J:.4f} (stored {stored['J']:.4f}, "
          f"delta {d_J:+.2e})  IoU {m['IoU']:.6f} (stored {stored['IoU']:.6f}, "
          f"delta {d_iou:+.2e})", flush=True)
    if keep_snaps:
        T, steps = melt_snaps(tr, st.index)
    else:
        T = np.asarray([tr.T_at_end(st.index)], dtype=np.float32)
        steps = np.asarray([st.index])
    del tr
    return T, steps, st.index, st.J, m["IoU"]


def main() -> None:
    t0 = time.perf_counter()
    r = json.loads((OUT_LIB / "hexagon.json").read_text())
    cfg = load_cfg(r["config"])
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    maps = np.load(OUT_LIB / "hexagon_maps.npz")

    hist = r["arms"]["HIST_best"]
    a1 = r["arms"]["A1_4bpp"]
    uni = r["arms"]["U_uniform"]
    assert hist["convention"] == "outside1" and hist["eps_covary"] is True

    if ARMS_CACHE.exists():
        ac = dict(np.load(ARMS_CACHE))
        print("stored-arm re-runs loaded from cache", flush=True)
    else:
        ac = {}
        # HIST_best: stored mask, outside1 convention, eps-co-varying channel
        s_raw = load_stored_map(case, Path(hist["map_npz"]), cfg)
        s_hist = np.where(pm, s_raw, 1.0)
        T_h, steps_h, i_h, J_h, iou_h = rerun_arm(
            case, s_hist, True, hist, "HIST_best")
        ac.update(sat_hist=s_hist.astype(np.float32), T_hist=T_h,
                  steps_hist=steps_h, hist_stop_index=i_h, hist_J=J_h,
                  hist_IoU=iou_h)
        # A1_4bpp: stored solved deliverable map, conductivity-only channel
        s_a1 = np.asarray(maps["A1_4bpp"], dtype=np.float64)
        T_a, steps_a, i_a, J_a, iou_a = rerun_arm(
            case, s_a1, False, a1, "A1_4bpp")
        ac.update(sat_a1=s_a1.astype(np.float32), T_a1=T_a, steps_a1=steps_a,
                  a1_stop_index=i_a, a1_J=J_a, a1_IoU=iou_a)
        # U_uniform: s = 1, final field only (static figure)
        s_u = np.ones(pm.shape)
        T_u, steps_u, i_u, J_u, iou_u = rerun_arm(
            case, s_u, False, uni, "U_uniform", keep_snaps=False)
        ac.update(T_uni_final=T_u[-1], uni_stop_index=i_u, uni_J=J_u,
                  uni_IoU=iou_u)
        np.savez_compressed(ARMS_CACHE, **ac)
        print(f"stored-arm re-runs done, wall {time.perf_counter()-t0:.0f} s",
              flush=True)

    # --- display solve: fresh, filtered, budget 15, process viz only -------
    sigma_cells = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)
    idx = np.flatnonzero(pm.ravel())
    rows: list[float] = []
    ev_maps: list[np.ndarray] = []
    v_last = [np.ones(len(idx))]
    if PARTIAL.exists():
        pp = np.load(PARTIAL)
        rows = [float(j) for j in pp["j_evals"]]
        ev_maps = [m for m in pp["maps"]]
        v_last = [pp["v_last"]]
        print(f"resuming display solve with {len(rows)} evaluations done",
              flush=True)

    def to_map(v):
        raw = np.ones(pm.shape)
        raw.ravel()[idx] = v
        return df.apply_filter(raw, pm, sigma_cells)

    def save_partial():
        np.savez_compressed(PARTIAL, j_evals=np.asarray(rows),
                            maps=np.asarray(ev_maps, dtype=np.float32),
                            v_last=v_last[0])

    def fun(v):
        if len(rows) >= N_EVALS:
            raise StopIteration
        v_last[0] = np.asarray(v, dtype=float).copy()
        s = to_map(v)
        tr = fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                         shape_stop_patience=PATIENCE)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        g_s = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops)
        del tr
        g = df.filter_vjp(g_s, pm, sigma_cells)
        rows.append(float(J))
        ev_maps.append(s.astype(np.float32))
        save_partial()
        print(f"  eval {len(rows):2d}  J {J:9.3f}  stop {st.index}  "
              f"wall {time.perf_counter() - t0:.0f} s", flush=True)
        return float(J), g.ravel()[idx].astype(float)

    while len(rows) < N_EVALS:
        try:
            minimize(fun, v_last[0], jac=True, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * len(idx),
                     options={"maxiter": 10_000, "maxfun": N_EVALS,
                              "ftol": 1e-16, "gtol": 1e-16})
        except StopIteration:
            break
        break  # optimizer converged early with budget left: stop honestly

    np.savez_compressed(
        CACHE,
        part_mask=pm, x=case.x, y=case.y, dt_s=case.pins.dt,
        t_pc_c=case.pins.t_pc_c, dt_pc_c=case.pins.dt_pc_c,
        # stored-arm re-runs (gated above)
        **ac,
        # stored library end-state numbers (captions quote THESE)
        hist_J_stored=hist["J"], hist_IoU_stored=hist["IoU"],
        hist_stop_s_stored=hist["t_stop_s"],
        a1_J_stored=a1["J"], a1_IoU_stored=a1["IoU"],
        a1_stop_s_stored=a1["t_stop_s"],
        uni_J_stored=uni["J"], uni_IoU_stored=uni["IoU"],
        uni_stop_s_stored=uni["t_stop_s"],
        dJ_rel_stored=r["verdict"]["dJ_rel"],
        # display solve (process visualization only)
        maps_evals=np.asarray(ev_maps, dtype=np.float32),
        j_evals=np.asarray(rows),
    )
    print(f"wrote {CACHE}  wall {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
