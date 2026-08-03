"""Density-region solve across the standardized shape library.

One shape per invocation. Everything is scored under BOTH objectives, each at
its own read state, plus the two cross reads:

    J_phi(s, t)  = sum over the WHOLE domain of (phi(x, t) - chi_part(x))^2
    J_rho(s, t)  = sum over the WHOLE domain of (rho_norm(x, t) - chi_part(x))^2

READ STATES, stated once and carried on every number.

  PHI-STOP  t = argmin over that arm's own stored trajectory of J_phi. The
            melt-region objective turns (melt spills into the bed), so its
            argmin is interior and the envelope argument applies.
  RHO-STOP  the FLAT ONSET of J_rho: the first stored step within `FLAT_TOL`
            of the terminal J_rho, as a fraction of the total decrease.
            J_rho is monotone non-increasing, so its argmin is ALWAYS the
            horizon and reading there would hand the optimizer a dead
            objective. See `density_objective` for the full statement.

THE SHAPE EARLY STOP IS DISABLED on every arm here, including the controls,
because it truncates the march on the melt objective long before the density
objective flattens. Every march runs the full horizon, so the J_phi numbers in
this campaign are re-scored and are NOT carried across from
`SHAPE_LIBRARY_SOLVE_REPORT.md` (a full-horizon argmin can only find a J_phi
lower than or equal to the truncated one).

THE DESIGN FILTER is on in the primary arm. `SOLVE_ROBUSTNESS_VALIDATION.md`
Task B measured that the previously solved maps are rim solutions with a
sensitivity length of about one cell; the filter makes that structure
inexpressible rather than merely discouraged. It filters the DESIGN VARIABLE,
s = F(v), so the optimizer never sees the unfiltered space.

Run:
  ./.venv312/bin/python -m adjoint2d.rho_solve <shape> <outdir> [budget] [sigma_cells]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import adjoint, control, density_objective as dobj, design_filter as df
from . import energy_gate as eg, forward as fwd, gradops
from . import library_solve as lib
from . import printability as pq
from . import shape_objective as so
from .pins import build_case, load_cfg
from .verify_hist import load_stored_map

BUDGET_FORWARD_EQUIVALENTS = 40.0    # combined over the two starts
SIGMA_CELLS = df.DEFAULT_SIGMA_CELLS
FLAT_TOL = dobj.FLAT_TOL
GRAD_DEATH_REL = 1e-6                # of the first iterate's max |gradient|
GRAD_DEATH_ABS = 1e-30
OUT_LIB = Path(__file__).resolve().parents[1] / "out_lib"
NOFILTER_CONTROL_SHAPES = ("square", "circle")


# ---------------------------------------------------------------------------
# pure logic (unit tested)
# ---------------------------------------------------------------------------

def gradient_death(grad_max_abs: float, grad_max_abs_first: float | None,
                   rel: float = GRAD_DEATH_REL, abs_floor: float = GRAD_DEATH_ABS) -> bool:
    """Has the density gradient died?

    Death is `|g|_inf` falling to or below `rel` times the first iterate's
    `|g|_inf`, with an absolute floor so that a run which starts numerically
    dead is not certified healthy by comparison with itself.
    """
    g = float(grad_max_abs)
    first = float(abs_floor) if grad_max_abs_first is None else float(grad_max_abs_first)
    floor = max(float(rel) * first, float(abs_floor))
    return bool(g <= floor)


def better_start(a: dict | None, b: dict | None) -> dict | None:
    """The lower-J_rho of two solved starts; either may be missing."""
    if a is None:
        return b
    if b is None:
        return a
    return a if float(a["J_rho"]) <= float(b["J_rho"]) else b


# ---------------------------------------------------------------------------
# forward and scoring
# ---------------------------------------------------------------------------

def run_forward(case, s, eps_covary: bool = False):
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=None, eps_covary=eps_covary)


def score_both(case, s, eps_covary: bool = False, tol: float = FLAT_TOL) -> dict:
    """One forward run, read at BOTH stops, with both cross reads."""
    pm = case.part_mask
    tr = run_forward(case, s, eps_covary=eps_covary)
    m = so.full_metrics(tr, case)                      # at the phi-stop
    m.update(dobj.full_metrics(tr, case, tol=tol))     # at the rho-stop

    i_phi = int(m["t_stop_index"])
    i_rho = int(m["rho_stop_index"])
    # cross reads: each objective evaluated at the OTHER objective's read state
    m["J_rho_at_phi_stop"] = float(dobj.rho_J_and_seed(tr.rho_at_end(i_phi), case)[0])
    m["J_phi_at_rho_stop"] = float(so.shape_J_and_seed(tr.T_at_end(i_rho), case)[0])
    for k, v in dobj.density_region_metrics(tr.rho_at_end(i_phi), case).items():
        m[f"{k}_at_phi_stop"] = v
    for k, v in so.region_metrics(tr.T_at_end(i_rho), case).items():
        m[f"{k}_at_rho_stop"] = v
    m["stop_gap_steps"] = i_rho - i_phi
    m["stop_gap_s"] = float(m["rho_stop_s"] - m["t_stop_s"])

    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["eps_covary"] = bool(eps_covary)
    m["sat_mean_in_part"] = float(np.mean(s[pm]))
    m["sat_std_in_part"] = float(np.std(s[pm]))
    m["sat_max_in_part"] = float(np.max(s[pm]))
    m["map_roughness"] = df.roughness_in_part(s, pm)
    m["energy_gate_at_phi_stop"] = eg.gate_from_trajectory(tr, i_phi)
    m["energy_gate_at_rho_stop"] = eg.gate_from_trajectory(tr, i_rho)
    del tr
    return m


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

def solve_rho(case, ops, n_evals: int, box, v_init: np.ndarray,
              sigma_cells: float, tol: float = FLAT_TOL) -> tuple[list[dict], dict, dict]:
    """L-BFGS-B on dJ_rho/dv with the design filter in the chain.

    The per-iterate gradient-death assertion is part of the contract: if the
    objective saturates, the density gradient goes to zero (every rate factor
    carries (1 - rho)^e and the rho clip subgradient closes), and continuing to
    optimize on a dead gradient is exactly how a spurious structure gets carved.
    The run stops and says so.
    """
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
    rows: list[dict] = []
    store: dict[str, np.ndarray] = {}
    info: dict = {"gradient_death": False, "stopped_reason": "budget",
                  "grad_max_abs_first": None}

    def unpack(vec):
        v = np.ones(pm.shape)
        v.ravel()[idx] = vec
        return v

    def fun(vec):
        if len(rows) >= n_evals:
            info["stopped_reason"] = "budget"
            raise StopIteration
        v = unpack(vec)
        s = df.apply_filter(v, pm, sigma_cells)
        tr = run_forward(case, s)
        st = dobj.rho_stop(tr, case, tol=tol)
        J, seed = dobj.rho_J_and_seed(tr.rho_at_end(st.index), case)
        g_s = adjoint.gradient(case, s, tr, {}, grad_ops=ops, seeds_rho={st.index: seed})
        g = df.filter_vjp(g_s, pm, sigma_cells)
        gmax = float(np.max(np.abs(g[pm])))
        if info["grad_max_abs_first"] is None:
            info["grad_max_abs_first"] = gmax
        rows.append({"eval_index": len(rows) + 1, "J_rho": float(J),
                     "rho_stop_index": st.index, "rho_stop_s": st.time_s,
                     "rho_stop_at_horizon": st.at_horizon,
                     "rho_stop_no_progress": st.no_progress,
                     "grad_max_abs": gmax,
                     "grad_norm": float(np.linalg.norm(g[pm])),
                     "mean_rho_rel_part": float(np.mean(tr.rho_at_end(st.index)[pm]))})
        store[str(len(rows))] = v.copy()
        del tr
        if gradient_death(gmax, info["grad_max_abs_first"]):
            info["gradient_death"] = True
            info["stopped_reason"] = "gradient death (objective saturated)"
            raise StopIteration
        return float(J), g.ravel()[idx].astype(float)

    v0 = np.clip(np.asarray(v_init, dtype=float).ravel()[idx], box[0], box[1])
    try:
        minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=[box] * len(idx),
                 options={"maxiter": 10_000, "maxfun": n_evals,
                          "ftol": 1e-16, "gtol": 1e-16})
    except StopIteration:
        pass
    info["n_evals_used"] = len(rows)
    return rows, store, info


# ---------------------------------------------------------------------------
# reference maps
# ---------------------------------------------------------------------------

def previous_phi_map(shape: str) -> np.ndarray | None:
    """The J_phi-solved printable 4-bits-per-pixel map of the library campaign."""
    p = OUT_LIB / f"{shape}_maps.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        if "A1_4bpp" not in d.files:
            return None
        return np.asarray(d["A1_4bpp"], dtype=float)


def best_hist_map(case, cfg, shape: str) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """The best stored historical dopant map, as the library campaign selected it."""
    p = OUT_LIB / f"{shape}.json"
    if not p.exists():
        return None, None
    j = json.loads(p.read_text())
    h = j.get("arms", {}).get("HIST_best")
    if not h or not h.get("map_npz"):
        return None, None
    s = load_stored_map(case, Path(h["map_npz"]), cfg)
    if h.get("convention") == "outside1":
        s = np.where(case.part_mask, s, 1.0)
    return np.asarray(s, dtype=float), {"map_npz": h["map_npz"],
                                        "convention": h.get("convention"),
                                        "source_arm": j.get("best_hist_arm"),
                                        "library_J_phi": h.get("J"),
                                        "library_IoU": h.get("IoU")}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _solve_pair(case, ops, n_each, v_warm, sigma, tol, tag, log):
    """Cold (uniform) and warm starts, both inside the budget; keep the better."""
    pm = case.part_mask
    out = {}
    best = None
    for start, v0 in (("cold", np.ones(pm.shape)),
                      ("warm", v_warm if v_warm is not None else None)):
        if v0 is None:
            log(f"  {tag}/{start}: SKIPPED, no previous solved map for this shape")
            continue
        t0 = time.perf_counter()
        rows, store, info = solve_rho(case, ops, n_each, (0.0, 1.0), v0, sigma, tol)
        info["wall_s"] = time.perf_counter() - t0
        if not rows:
            log(f"  {tag}/{start}: no evaluations completed")
            out[start] = {"rows": rows, "info": info}
            continue
        b = min(rows, key=lambda r: r["J_rho"])
        v_best = store[str(b["eval_index"])]
        out[start] = {"rows": rows, "info": info, "best_eval": b}
        log(f"  {tag}/{start}: {len(rows)} evals, J_rho {rows[0]['J_rho']:.2f} -> "
            f"{b['J_rho']:.2f}, grad_max {rows[0]['grad_max_abs']:.3e} -> "
            f"{rows[-1]['grad_max_abs']:.3e}, {info['stopped_reason']}, "
            f"{info['wall_s']:.0f} s")
        if better_start(None if best is None else best[1], b) is b:
            best = (start, b, v_best)
    return out, best


def main(shape: str, outdir: str, budget: float = BUDGET_FORWARD_EQUIVALENTS,
         sigma_cells: float = SIGMA_CELLS, tol: float = FLAT_TOL) -> dict:
    if shape not in lib.SHAPES:
        raise ValueError(f"{shape!r} is not in the standardized library")
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    def log(msg):
        print(f"[{shape}] {msg}", flush=True)

    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)

    res: dict = {
        "shape": shape, "config": str(cfg_path),
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "n_part_cells": case.n_part, "n_steps": case.pins.n_steps,
        "budget_forward_equivalents_total": float(budget),
        "sigma_cells": float(sigma_cells), "flat_tol": float(tol),
        "stop_convention": {
            "phi": "argmin over the arm's own trajectory of J_phi",
            "rho": "flat onset: first step within flat_tol of the terminal J_rho",
            "shape_early_stop": "DISABLED on every arm (full horizon)"},
        "arms": {}, "solves": {},
    }
    maps_store: dict[str, np.ndarray] = {"part_mask": pm.astype(np.uint8)}

    # --- cost model, measured on this shape --------------------------------
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = run_forward(case, s_u)
    t_b = time.perf_counter()
    st_u = dobj.rho_stop(tr_u, case, tol=tol)
    _J, seed = dobj.rho_J_and_seed(tr_u.rho_at_end(st_u.index), case)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {}, grad_ops=ops, seeds_rho={st_u.index: seed})
    t_d = time.perf_counter()
    del tr_u
    ratio = (t_d - t_c) / max(t_b - t_a, 1e-9)
    n_each = control.max_gradient_evals(float(budget) / 2.0, ratio)
    res["cost"] = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c, "ratio": ratio,
                   "n_gradient_evals_per_start": n_each,
                   "spent_forward_equivalents": 2.0 * control.forward_equivalents(
                       n_each, n_each, ratio)}
    log(f"cost: forward {t_b - t_a:.1f} s, adjoint {t_d - t_c:.1f} s, ratio {ratio:.2f}, "
        f"{n_each} gradient evaluations per start")

    # --- control arms -------------------------------------------------------
    m_u = score_both(case, s_u, tol=tol)
    m_u["arm"] = "U_uniform"
    res["arms"]["U_uniform"] = m_u
    maps_store["U_uniform"] = s_u
    log(f"U_uniform J_phi {m_u['J']:.2f} IoU {m_u['IoU']:.4f} | J_rho {m_u['J_rho']:.2f} "
        f"IoU_rho {m_u['IoU_rho']:.4f} rho {m_u['mean_rho_rel_part']:.4f} "
        f"stops {m_u['t_stop_s']:.1f}/{m_u['rho_stop_s']:.1f} s")

    s_hist, hist_info = best_hist_map(case, cfg, shape)
    if s_hist is not None:
        m_h = score_both(case, s_hist, eps_covary=True, tol=tol)
        m_h.update({"arm": "HIST_best", **hist_info})
        res["arms"]["HIST_best"] = m_h
        maps_store["HIST_best"] = s_hist
        log(f"HIST_best J_phi {m_h['J']:.2f} IoU {m_h['IoU']:.4f} | J_rho {m_h['J_rho']:.2f} "
            f"rho {m_h['mean_rho_rel_part']:.4f}")
    else:
        log("HIST_best: SKIPPED, no stored historical arm recorded in out_lib")

    s_phi4 = previous_phi_map(shape)
    if s_phi4 is not None:
        m_p = score_both(case, s_phi4, tol=tol)
        m_p["arm"] = "PHI4_prev"
        res["arms"]["PHI4_prev"] = m_p
        maps_store["PHI4_prev"] = s_phi4
        log(f"PHI4_prev J_phi {m_p['J']:.2f} IoU {m_p['IoU']:.4f} | J_rho {m_p['J_rho']:.2f} "
            f"rho {m_p['mean_rho_rel_part']:.4f}")
    else:
        log("PHI4_prev: SKIPPED, no library solved map for this shape")

    # --- the density solve, filtered ---------------------------------------
    solves, best = _solve_pair(case, ops, n_each, s_phi4, sigma_cells, tol, "RHO", log)
    res["solves"]["filtered"] = {k: {"info": v["info"], "rows": v["rows"]}
                                 for k, v in solves.items()}
    if best is None:
        raise RuntimeError(f"{shape}: the density solve produced no evaluations")
    res["best_start"] = best[0]
    v_best = best[2]
    s_cont = df.apply_filter(v_best, pm, sigma_cells)
    m_c = score_both(case, s_cont, tol=tol)
    m_c["arm"] = "RHO_cont"
    m_c["start"] = best[0]
    res["arms"]["RHO_cont"] = m_c
    maps_store["RHO_cont"] = s_cont
    maps_store["RHO_v"] = v_best

    s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)
    m_q = score_both(case, s_q, tol=tol)
    m_q["arm"] = "RHO_4bpp"
    m_q["start"] = best[0]
    m_q.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm, bpp=4).items()})
    res["arms"]["RHO_4bpp"] = m_q
    maps_store["RHO_4bpp"] = s_q
    log(f"RHO_4bpp ({best[0]}) J_rho {m_q['J_rho']:.2f} IoU_rho {m_q['IoU_rho']:.4f} "
        f"rho {m_q['mean_rho_rel_part']:.4f} | J_phi {m_q['J']:.2f} IoU {m_q['IoU']:.4f}")

    # --- unfiltered control, two shapes only --------------------------------
    if shape in NOFILTER_CONTROL_SHAPES:
        sv, bn = _solve_pair(case, ops, n_each, s_phi4, 0.0, tol, "RHO_NF", log)
        res["solves"]["unfiltered"] = {k: {"info": v["info"], "rows": v["rows"]}
                                       for k, v in sv.items()}
        if bn is not None:
            s_nf = df.apply_filter(bn[2], pm, 0.0)
            m_nf = score_both(case, s_nf, tol=tol)
            m_nf.update({"arm": "RHO_cont_nofilter", "start": bn[0]})
            res["arms"]["RHO_cont_nofilter"] = m_nf
            maps_store["RHO_cont_nofilter"] = s_nf
            s_nfq = pq.quantize_in_part(s_nf, pm, bpp=4, sat_max=1.0)
            m_nfq = score_both(case, s_nfq, tol=tol)
            m_nfq.update({"arm": "RHO_4bpp_nofilter", "start": bn[0]})
            res["arms"]["RHO_4bpp_nofilter"] = m_nfq
            maps_store["RHO_4bpp_nofilter"] = s_nfq
            log(f"RHO_4bpp_nofilter J_rho {m_nfq['J_rho']:.2f} "
                f"IoU_rho {m_nfq['IoU_rho']:.4f} | J_phi {m_nfq['J']:.2f} "
                f"IoU {m_nfq['IoU']:.4f} roughness {m_nfq['map_roughness']:.4f} "
                f"against filtered {m_q['map_roughness']:.4f}")

    # --- verdict ------------------------------------------------------------
    d = res["arms"]["RHO_4bpp"]
    refs = {k: res["arms"][k] for k in ("U_uniform", "HIST_best", "PHI4_prev")
            if k in res["arms"]}
    res["verdict"] = {
        "deliverable_arm": "RHO_4bpp",
        "beats_on_J_rho": {k: bool(d["J_rho"] < r["J_rho"]) for k, r in refs.items()},
        "beats_on_J_phi": {k: bool(d["J"] < r["J"]) for k, r in refs.items()},
        "dJ_rho_rel": {k: (r["J_rho"] - d["J_rho"]) / max(abs(r["J_rho"]), 1e-30)
                       for k, r in refs.items()},
        "dIoU_rho": {k: d["IoU_rho"] - r["IoU_rho"] for k, r in refs.items()},
        "dIoU_phi": {k: d["IoU"] - r["IoU"] for k, r in refs.items()},
        "mean_rho_at_rho_stop": d["mean_rho_rel_part"],
        "mean_rho_at_phi_stop": d["mean_rho_rel_part_at_phi_stop"],
        "gradient_death_any": any(
            v["info"].get("gradient_death", False)
            for grp in res["solves"].values() for v in grp.values()),
        "rho_stop_at_horizon_any": any(
            res["arms"][a].get("rho_stop_at_horizon", False) for a in res["arms"]),
    }
    res["energy_gate_violations"] = [
        a for a, m in res["arms"].items()
        if not (m["energy_gate_at_phi_stop"]["PASS"] and m["energy_gate_at_rho_stop"]["PASS"])]
    res["wall_s"] = time.perf_counter() - t_start

    np.savez_compressed(out / f"{shape}_maps.npz", x=case.x, y=case.y, **maps_store)
    (out / f"{shape}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done, wall {res['wall_s']:.0f} s, energy gate violations "
        f"{res['energy_gate_violations'] or 'none'}")
    return res


if __name__ == "__main__":
    _shape = sys.argv[1]
    _outdir = sys.argv[2]
    _budget = float(sys.argv[3]) if len(sys.argv) > 3 else BUDGET_FORWARD_EQUIVALENTS
    _sigma = float(sys.argv[4]) if len(sys.argv) > 4 else SIGMA_CELLS
    main(_shape, _outdir, _budget, _sigma)
