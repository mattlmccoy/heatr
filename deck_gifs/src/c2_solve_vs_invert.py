"""Compute stage for gif_solve_vs_inversion, on the SQUARE.

Left arm (stored): the best historical proportional-inverse 4-bits-per-pixel
mask (out_lib/square.json HIST_best, scored with the permittivity-co-varying
channel it was run in). Re-run its forward capturing T snapshots so the melt
front can be animated to its stored stop. Cached to c2_hist.npz.

Right arm (fresh, cheap): a small FILTERED adjoint solve (design filter at the
campaign's 1.0 mm radius, box [0, 1], budget 15 gradient evaluations total)
with the map snapshot kept at every evaluation, then the best map's forward
captured the same way. RESUMABLE: every evaluation is appended to
c2_partial.npz; on restart the optimizer warm-starts from the last iterate
with the remaining budget.

Read-only reuse of adjoint2d. Nothing in adjoint2d is modified.
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
CACHE = REPO / "deck_gifs/cache/c2_square.npz"
HIST_CACHE = REPO / "deck_gifs/cache/c2_hist.npz"
PARTIAL = REPO / "deck_gifs/cache/c2_partial.npz"
N_EVALS = 15
PATIENCE = 250
SNAP_EVERY = 4


def melt_snaps(tr, stop_index: int) -> tuple[np.ndarray, np.ndarray]:
    steps = list(range(0, stop_index + 1, SNAP_EVERY))
    if steps[-1] != stop_index:
        steps.append(stop_index)
    T = np.asarray([tr.T_at_end(i) for i in steps], dtype=np.float32)
    return T, np.asarray(steps)


def main() -> None:
    t0 = time.perf_counter()
    r = json.loads((OUT_LIB / "square.json").read_text())
    cfg = load_cfg(r["config"])
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    hist = r["arms"]["HIST_best"]

    # --- the invert arm: stored mask, stored channel, re-run forward -------
    s_hist = load_stored_map(case, Path(hist["map_npz"]), cfg)   # as stored
    if HIST_CACHE.exists():
        hc = np.load(HIST_CACHE)
        T_h, steps_h = hc["T_hist"], hc["steps_hist"]
        st_h_index, st_h_J, hist_iou = (int(hc["stop_index"]), float(hc["J"]),
                                        float(hc["IoU"]))
        print(f"invert arm loaded from cache: stop {st_h_index} J {st_h_J:.2f}")
    else:
        tr_h = fwd.forward(case, s_hist, keep_checkpoints=True,
                           stop_after_phi=None, shape_stop_patience=PATIENCE,
                           eps_covary=True)
        st_h = so.optimal_stop(tr_h, case)
        m_h = so.full_metrics(tr_h, case)
        print(f"invert re-run: stop {st_h.index} J {st_h.J:.2f} "
              f"(stored {hist['t_stop_index']} J {hist['J']:.2f})", flush=True)
        T_h, steps_h = melt_snaps(tr_h, st_h.index)
        st_h_index, st_h_J, hist_iou = st_h.index, st_h.J, m_h["IoU"]
        np.savez_compressed(HIST_CACHE, T_hist=T_h, steps_hist=steps_h,
                            stop_index=st_h_index, J=st_h_J, IoU=hist_iou)
        del tr_h

    # --- the solve arm: small filtered solve, map kept every eval ----------
    sigma_cells = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)
    idx = np.flatnonzero(pm.ravel())
    rows: list[float] = []
    maps: list[np.ndarray] = []
    v_last = [np.ones(len(idx))]
    if PARTIAL.exists():
        pp = np.load(PARTIAL)
        rows = [float(j) for j in pp["j_evals"]]
        maps = [m for m in pp["maps"]]
        v_last = [pp["v_last"]]
        print(f"resuming solve with {len(rows)} evaluations done", flush=True)

    def to_map(v):
        raw = np.ones(pm.shape)
        raw.ravel()[idx] = v
        return df.apply_filter(raw, pm, sigma_cells)

    def save_partial():
        np.savez_compressed(PARTIAL, j_evals=np.asarray(rows),
                            maps=np.asarray(maps, dtype=np.float32),
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
        maps.append(s.astype(np.float32))
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
        # optimizer converged early with budget left: stop honestly
        break

    j_evals = np.asarray(rows)
    best = int(np.argmin(j_evals))
    s_best = np.asarray(maps[best], dtype=float)
    tr_s = fwd.forward(case, s_best, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=PATIENCE)
    st_s = so.optimal_stop(tr_s, case)
    m_s = so.full_metrics(tr_s, case)
    print(f"solve best eval {best + 1}: J {st_s.J:.2f} IoU {m_s['IoU']:.4f} "
          f"stop {st_s.index}", flush=True)
    T_s, steps_s = melt_snaps(tr_s, st_s.index)

    np.savez_compressed(
        CACHE,
        part_mask=pm, x=case.x, y=case.y, dt_s=case.pins.dt,
        t_pc_c=case.pins.t_pc_c, dt_pc_c=case.pins.dt_pc_c,
        # invert arm
        sat_hist=s_hist.astype(np.float32), T_hist=T_h, steps_hist=steps_h,
        hist_stop_index=st_h_index, hist_J=st_h_J, hist_IoU=hist_iou,
        hist_J_stored=hist["J"], hist_IoU_stored=hist["IoU"],
        # solve arm
        maps_evals=np.asarray(maps, dtype=np.float32), j_evals=j_evals,
        best_eval=best, sat_solved=s_best.astype(np.float32),
        T_solve=T_s, steps_solve=steps_s,
        solve_stop_index=st_s.index, solve_J=st_s.J, solve_IoU=m_s["IoU"],
    )
    print(f"wrote {CACHE}  wall {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
