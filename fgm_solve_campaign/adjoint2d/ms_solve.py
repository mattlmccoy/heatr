"""Multi-start / warm-start melt-objective solve across the standardized library.

One shape per invocation. The objective is the melt-region shape-fidelity
objective, unchanged:

    J_phi(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

read at that arm's OWN J-stop, t_stop = argmin over its own stored trajectory.
`t_stop_at_horizon` is flagged whenever the minimum sits on the last stored
step, which makes that arm's J a bound. The melted region for intersection over
union, growth and under-melt is phi >= 0.5. GRID 120 x 120 throughout, and
`SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute fidelity at this grid
does not transfer to 160, so every intersection-over-union number here carries
the grid qualifier.

TWO THINGS CHANGE AT ONCE against the single-start library campaign
(`SHAPE_LIBRARY_SOLVE_REPORT.md`, results in `out_lib/`), and both are named
every time the comparison is quoted:

  1. FOUR STARTS instead of one, sharing the same 40 forward-equivalent budget.
  2. THE PHYSICAL-LENGTH DESIGN FILTER is applied to the design variable inside
     the solve, s = F(v) at sigma = 1.5 cells, mandated by
     `SOLVE_ROBUSTNESS_VALIDATION.md` Task B. The library solve had no filter.

A FILTERED SINGLE-START control (`--control cold`) is run on the two headline
shapes so the two changes can be separated on those shapes.

THE FOUR STARTS, in declaration order.

  cold   uniform saturation 1. The historical cold start.
  warm   the best stored historical 4-bits-per-pixel dopant map for this shape,
         as `out_lib/<shape>.json` selected it (`best_hist_arm`, in the boundary
         convention that won there), loaded through the PRODUCTION loader and
         clipped into the box. NOTE, and it matters: the historical map was
         SCORED in the permittivity-co-varying channel, and this solve actuates
         conductivity only. The start is the map, not the channel.
  prev   the previously J_phi-solved CONTINUOUS map of the library campaign,
         `out_lib/<shape>_maps.npz` key `A1_cont`.
  pert   the deterministic seedless perturbed start: the midpoint of uniform and
         the proportional-inverse control map at gain 0.5
         (`multistart.perturbed_start`, which documents why the midpoint and not
         the literal sum).

Every start is a DESIGN VARIABLE v; the injected map is s = F(v). The warm start
therefore injects a FILTERED historical mask, not the mask itself, so start
`warm`'s first evaluation is not the historical arm's number.

Budget, kill rule and the cost of the continuation restart: see `multistart`.

Run:
  ./.venv312/bin/python -m adjoint2d.ms_solve <shape> <outdir> [budget] [sigma_cells] [control]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import adjoint, control as ctl, design_filter as df
from . import energy_gate as eg, forward as fwd, gradops
from . import library_solve as lib
from . import multistart as ms
from . import printability as pq
from . import shape_objective as so
from .pins import build_case, load_cfg
from .shape_solve import heuristic_map
from .verify_hist import load_stored_map

BUDGET_FORWARD_EQUIVALENTS = ms.BUDGET_FORWARD_EQUIVALENTS
SIGMA_CELLS = df.DEFAULT_SIGMA_CELLS
PI_GAIN = 0.5
BOX = (0.0, 1.0)
OUT_LIB = Path(__file__).resolve().parents[1] / "out_lib"
START_ORDER = ("cold", "warm", "prev", "pert")
RATIO_SANE = (0.5, 4.0)


def budget_ratio(shape: str, measured: float, log) -> tuple[float, dict]:
    """The adjoint-to-forward cost ratio used to turn the budget into evaluations.

    MEASURED PROBLEM, and this is why the ratio is not simply the one timed in
    this run. Timing a single forward and a single adjoint on a loaded machine
    is unreliable: on the star shape, with one other solve job running, this
    pass timed 3.4 s forward against 43.4 s adjoint, a ratio of 12.91, against
    the 1.11 the library campaign timed for the same two calls on the same code
    path. A ratio that high converts a 40 forward-equivalent budget into ONE
    gradient evaluation, which would silently starve the solve and make the
    comparison against the library campaign meaningless.

    So the budget conversion uses the per-shape ratio RECORDED BY THE LIBRARY
    CAMPAIGN in `out_lib/<shape>.json`. Two consequences, both wanted: the
    evaluation pool is deterministic and independent of machine load, and the
    multi-start solve gets exactly the same number of gradient evaluations the
    single-start library solve got for the same nominal budget. The ratio timed
    in this run is still reported, labelled as contention-contaminated.
    """
    p = OUT_LIB / f"{shape}.json"
    if p.exists():
        r = float(json.loads(p.read_text())["cost"]["ratio"])
        if RATIO_SANE[0] <= r <= RATIO_SANE[1]:
            return r, {"source": "out_lib cost.ratio", "value": r,
                       "measured_here": float(measured)}
        log(f"library ratio {r:.2f} outside the sane band {RATIO_SANE}, using the "
            f"measured {measured:.2f}")
    m = float(np.clip(measured, *RATIO_SANE))
    return m, {"source": "measured in this run, clipped to the sane band",
               "value": m, "measured_here": float(measured)}


# ---------------------------------------------------------------------------
# forward and scoring
# ---------------------------------------------------------------------------

def run_forward(case, s, eps_covary: bool = False, checkpoints: bool = False):
    """The library campaign's forward, unchanged, so scores are comparable."""
    return fwd.forward(case, s, keep_checkpoints=checkpoints, stop_after_phi=None,
                       shape_stop_patience=lib.PATIENCE, eps_covary=eps_covary)


def score_trajectory(case, tr, s: np.ndarray, eps_covary: bool = False) -> dict:
    m = so.full_metrics(tr, case)
    i = int(m["t_stop_index"])
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["mean_rho_rel_part_at_end"] = float(tr.mean_rho_rel_part[tr.n_outer - 1])
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["eps_covary"] = bool(eps_covary)
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_std_in_part"] = float(np.std(s[case.part_mask]))
    m["sat_min_in_part"] = float(np.min(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    m["map_roughness"] = df.roughness_in_part(s, case.part_mask)
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    return m


def score(case, s: np.ndarray, eps_covary: bool = False) -> dict:
    t0 = time.perf_counter()
    tr = run_forward(case, s, eps_covary=eps_covary)
    m = score_trajectory(case, tr, s, eps_covary)
    m["wall_s"] = time.perf_counter() - t0
    del tr
    return m


# ---------------------------------------------------------------------------
# one start
# ---------------------------------------------------------------------------

class StartRun:
    """One start's rows, design points and its own evaluation cache."""

    def __init__(self, name: str, v0: np.ndarray) -> None:
        self.name = name
        self.v0 = np.asarray(v0, dtype=float)
        self.rows: list[dict] = []
        self.store: dict[int, np.ndarray] = {}
        self.cache = ms.EvalCache()
        self.killed = False
        self.kill_reason: str | None = None
        self.wall_s = 0.0
        self.phases: list[dict] = []

    def best(self) -> dict | None:
        return min(self.rows, key=lambda r: r["J"]) if self.rows else None

    def best_v(self) -> np.ndarray | None:
        b = self.best()
        return None if b is None else self.store[int(b["eval_index"])]

    def last_v(self) -> np.ndarray:
        return self.store[int(self.rows[-1]["eval_index"])] if self.rows else self.v0


def run_phase(case, ops, sr: StartRun, n_new: int, box, sigma_cells: float,
              v_start: np.ndarray, phase: str) -> dict:
    """L-BFGS-B from `v_start`, allowed `n_new` NEW (uncached) evaluations.

    A cache hit costs nothing and is not counted against the budget: the
    forward is deterministic, so the cached pair is exactly what a rerun would
    give. The only hits in practice are the continuation's first call, which
    lands on the probe's last iterate.
    """
    pm = case.part_mask
    idx = np.flatnonzero(pm.ravel())
    t0 = time.perf_counter()
    n_before = len(sr.rows)
    info = {"phase": phase, "n_new_allowed": int(n_new), "stopped_reason": "budget"}

    def unpack(vec):
        v = np.ones(pm.shape)
        v.ravel()[idx] = vec
        return v

    def fun(vec):
        hit = sr.cache.get(vec)
        if hit is not None:
            return hit[0], hit[1].copy()
        if len(sr.rows) - n_before >= int(n_new):
            raise StopIteration
        v = unpack(vec)
        s = df.apply_filter(v, pm, sigma_cells)
        tr = run_forward(case, s, checkpoints=True)
        st = so.optimal_stop(tr, case)
        J, seed = so.shape_J_and_seed(tr.T_at_end(st.index), case)
        g_s = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops)
        g = df.filter_vjp(g_s, pm, sigma_cells)
        gv = g.ravel()[idx].astype(float)
        row = {"eval_index": len(sr.rows) + 1, "phase": phase, "J": float(J),
               "t_stop_index": int(st.index), "t_stop_s": float(st.time_s),
               "t_stop_at_horizon": bool(st.at_horizon),
               "grad_max_abs": float(np.max(np.abs(g[pm]))),
               "grad_norm": float(np.linalg.norm(g[pm])),
               "IoU": float(so.region_metrics(tr.T_at_end(st.index), case)["IoU"])}
        sr.rows.append(row)
        sr.store[row["eval_index"]] = v.copy()
        sr.cache.put(vec, (float(J), gv))
        del tr
        return float(J), gv

    v0 = np.clip(np.asarray(v_start, dtype=float).ravel()[idx], box[0], box[1])
    if int(n_new) > 0:
        try:
            minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=[box] * len(idx),
                     options={"maxiter": 10_000, "maxfun": 10_000,
                              "ftol": 1e-16, "gtol": 1e-16})
            info["stopped_reason"] = "L-BFGS-B converged"
        except StopIteration:
            pass
    info["n_new_used"] = len(sr.rows) - n_before
    info["n_cache_hits"] = sr.cache.n_hit
    info["wall_s"] = time.perf_counter() - t0
    sr.wall_s += info["wall_s"]
    sr.phases.append(info)
    return info


# ---------------------------------------------------------------------------
# the starts
# ---------------------------------------------------------------------------

def build_starts(case, cfg, shape: str, log) -> tuple[dict[str, StartRun], dict]:
    pm = case.part_mask
    meta: dict = {}
    v: dict[str, np.ndarray | None] = {"cold": np.ones(pm.shape)}

    # warm: the best stored historical mask of the library census
    v["warm"] = None
    p = OUT_LIB / f"{shape}.json"
    if p.exists():
        j = json.loads(p.read_text())
        h = j.get("arms", {}).get("HIST_best")
        if h and h.get("map_npz"):
            s_h = load_stored_map(case, Path(h["map_npz"]), cfg)
            if h.get("convention") == "outside1":
                s_h = np.where(pm, s_h, 1.0)
            v["warm"] = ms.start_from_map(s_h, pm, BOX)
            meta["warm"] = {"map_npz": h["map_npz"], "convention": h.get("convention"),
                            "source_arm": j.get("best_hist_arm"),
                            "library_J_phi": h.get("J"), "library_IoU": h.get("IoU"),
                            "note": "scored historically in the permittivity-co-varying "
                                    "channel; used here only as a start point"}
    if v["warm"] is None:
        log("start warm: SKIPPED LOUDLY, no stored historical arm in out_lib")

    # prev: the library campaign's solved continuous map
    v["prev"] = None
    pm_npz = OUT_LIB / f"{shape}_maps.npz"
    if pm_npz.exists():
        with np.load(pm_npz) as d:
            if "A1_cont" in d.files:
                v["prev"] = ms.start_from_map(np.asarray(d["A1_cont"], dtype=float), pm, BOX)
                meta["prev"] = {"source": str(pm_npz), "key": "A1_cont"}
    if v["prev"] is None:
        log("start prev: SKIPPED LOUDLY, no A1_cont map in out_lib for this shape")

    # pert: deterministic, seedless
    s_pi, proxy_info = heuristic_map(case, PI_GAIN)
    v["pert"] = ms.perturbed_start(s_pi, pm)
    meta["pert"] = {"gain": PI_GAIN, "proxy": proxy_info,
                    "rule": "midpoint of uniform and the proportional-inverse map"}

    starts = {k: StartRun(k, v[k]) for k in START_ORDER if v[k] is not None}
    return starts, meta


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(shape: str, outdir: str, budget: float = BUDGET_FORWARD_EQUIVALENTS,
         sigma_cells: float = SIGMA_CELLS, control: str = "") -> dict:
    if shape not in lib.SHAPES:
        raise ValueError(f"{shape!r} is not in the standardized library; "
                         + (lib.GT_LOGO_SKIP_REASON if shape == "gt_logo" else ""))
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    tag = f"{shape}_control_cold" if control == "cold" else shape

    def log(msg):
        print(f"[{tag}] {msg}", flush=True)

    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)

    res: dict = {
        "shape": shape, "config": str(cfg_path), "mode": control or "multistart",
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "n_part_cells": case.n_part, "n_grid": int(pm.shape[0]),
        "budget_forward_equivalents_total": float(budget),
        "sigma_cells": float(sigma_cells), "box": list(BOX),
        "stop_convention": "t_stop = argmin over the arm's own trajectory of J_phi; "
                           "at_horizon flagged; melted region phi >= 0.5",
        "kill_rule": {"n_probe": ms.N_PROBE, "margin": ms.KILL_MARGIN,
                      "max_keep": ms.MAX_KEEP,
                      "statement": "after the probe a start survives when its best "
                                   "J_phi is within margin*|J*| of the leader J* and "
                                   "its rank is below max_keep; the leader always "
                                   "survives"},
        "arms": {}, "starts": {},
    }
    maps_store: dict[str, np.ndarray] = {"part_mask": pm.astype(np.uint8)}

    # --- cost model, measured on this shape; the uniform arm comes free -----
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = run_forward(case, s_u, checkpoints=True)
    t_b = time.perf_counter()
    st_u = so.optimal_stop(tr_u, case)
    _J, seed = so.shape_J_and_seed(tr_u.T_at_end(st_u.index), case)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed}, grad_ops=ops)
    t_d = time.perf_counter()
    ratio_measured = (t_d - t_c) / max(t_b - t_a, 1e-9)
    ratio, ratio_info = budget_ratio(shape, ratio_measured, log)
    n_total = ctl.max_gradient_evals(float(budget), ratio)
    m_u = score_trajectory(case, tr_u, s_u)
    m_u["arm"] = "U_uniform"
    res["arms"]["U_uniform"] = m_u
    maps_store["U_uniform"] = s_u
    del tr_u
    log(f"cost: forward {t_b - t_a:.1f} s, adjoint {t_d - t_c:.1f} s, ratio measured "
        f"{ratio_measured:.2f}, budget ratio {ratio:.2f} ({ratio_info['source']}), "
        f"pool {n_total} gradient evaluations")
    log(f"U_uniform J {m_u['J']:.2f} IoU {m_u['IoU']:.4f} "
        f"rho@stop {m_u['mean_rho_rel_part_at_stop']:.4f} P {m_u['P_abs_W_per_m']:.1f}")

    # --- starts -------------------------------------------------------------
    if control == "cold":
        starts = {"cold": StartRun("cold", np.ones(pm.shape))}
        start_meta: dict = {"control": "filtered SINGLE cold start at the full budget, "
                                       "to separate the filter from the multi-start"}
    else:
        starts, start_meta = build_starts(case, cfg, shape, log)
    res["start_meta"] = start_meta
    res["n_starts"] = len(starts)
    # Forward runs OUTSIDE the solve budget, counted and named rather than
    # absorbed: the cost-model probe (one forward, which doubles as the uniform
    # arm, plus one adjoint), the full-horizon forward the proportional-inverse
    # perturbed start needs to build its proxy field, and the two final scoring
    # runs of MS_cont and MS_4bpp.
    res["overhead"] = {
        "cost_probe_forwards": 1, "cost_probe_adjoints": 1,
        "perturbed_start_proxy_forwards": 0 if control == "cold" else 1,
        "final_scoring_forwards": 2,
        "note": "not charged against the 40 forward-equivalent solve budget; the "
                "single-start library campaign carried the same cost-probe and "
                "final-scoring overhead"}

    n_probe = ms.probe_evals(n_total, len(starts))
    res["cost"] = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c, "ratio": ratio,
                   "ratio_measured_here": ratio_measured, "ratio_info": ratio_info,
                   "n_total_gradient_evals": n_total, "n_probe_per_start": n_probe,
                   # `probe_evals` floors at one evaluation per start so that no
                   # start is silently dropped without reporting its own J. When
                   # the pool is smaller than the number of starts that floor
                   # overruns the budget, which is flagged rather than hidden.
                   "probe_overruns_pool": bool(len(starts) * n_probe > n_total)}
    log(f"{len(starts)} starts, probe {n_probe} evaluations each")

    # --- probe phase ---------------------------------------------------------
    for name, sr in starts.items():
        run_phase(case, ops, sr, n_probe, BOX, sigma_cells, sr.v0, "probe")
        b = sr.best()
        log(f"  probe/{name}: {len(sr.rows)} evals, J {sr.rows[0]['J']:.2f} -> "
            f"{b['J']:.2f}, IoU {b['IoU']:.4f}, {sr.wall_s:.0f} s")

    probe_best = {k: (None if starts[k].best() is None else starts[k].best()["J"])
                  for k in starts}
    keep = ms.survivors(probe_best) if control != "cold" else list(starts)
    n_spent = sum(len(sr.rows) for sr in starts.values())
    n_cont = ms.continuation_evals(n_total, n_spent, len(keep))
    for k, sr in starts.items():
        if k not in keep:
            sr.killed = True
            sr.kill_reason = (f"probe J_phi {probe_best[k]:.2f} against leader "
                              f"{min(v for v in probe_best.values() if v is not None):.2f}"
                              if probe_best[k] is not None else "no evaluations")
    res["probe"] = {"best_J": probe_best, "survivors": keep,
                    "killed": {k: starts[k].kill_reason for k in starts if starts[k].killed},
                    "n_spent": n_spent, "n_continuation_per_survivor": n_cont}
    log(f"kill rule: survivors {keep}, killed "
        f"{[k for k in starts if starts[k].killed] or 'none'}, "
        f"{n_cont} continuation evaluations each")

    # --- continuation phase ---------------------------------------------------
    for k in keep:
        sr = starts[k]
        run_phase(case, ops, sr, n_cont, BOX, sigma_cells, sr.last_v(), "continuation")
        b = sr.best()
        log(f"  cont/{k}: {len(sr.rows)} evals total, best J {b['J']:.2f}, "
            f"IoU {b['IoU']:.4f}, {sr.wall_s:.0f} s")

    # --- pick the winner ------------------------------------------------------
    finals = {k: starts[k].best() for k in START_ORDER if k in starts}
    winner = ms.best_final(finals)
    if winner is None:
        raise RuntimeError(f"{shape}: no start produced an evaluation")
    res["winner_start"] = winner
    res["starts"] = {
        k: {"rows": sr.rows, "phases": sr.phases, "killed": sr.killed,
            "kill_reason": sr.kill_reason, "wall_s": sr.wall_s,
            "n_evals": len(sr.rows), "n_cache_hits": sr.cache.n_hit,
            "best_J": None if sr.best() is None else sr.best()["J"]}
        for k, sr in starts.items()}
    res["n_evals_used_total"] = sum(len(sr.rows) for sr in starts.values())
    res["spent_forward_equivalents"] = ctl.forward_equivalents(
        res["n_evals_used_total"], res["n_evals_used_total"], ratio)

    v_best = starts[winner].best_v()
    s_cont = df.apply_filter(v_best, pm, sigma_cells)
    m_c = score(case, s_cont)
    m_c.update({"arm": "MS_cont", "start": winner})
    res["arms"]["MS_cont"] = m_c
    maps_store["MS_cont"] = s_cont
    maps_store["MS_v"] = v_best

    s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)
    m_q = score(case, s_q)
    m_q.update({"arm": "MS_4bpp", "start": winner})
    m_q.update({f"census_{k}": v for k, v in pq.level_census(s_q, pm, bpp=4).items()})
    res["arms"]["MS_4bpp"] = m_q
    maps_store["MS_4bpp"] = s_q
    log(f"MS_4bpp ({winner}) J {m_q['J']:.2f} IoU {m_q['IoU']:.4f} "
        f"grow {m_q['bed_melt_pct_of_part']:.2f}% under {m_q['part_under_melt_pct']:.2f}% "
        f"rho@stop {m_q['mean_rho_rel_part_at_stop']:.4f} P {m_q['P_abs_W_per_m']:.1f} "
        f"stop {m_q['t_stop_s']:.1f} s"
        f"{' HORIZON' if m_q['t_stop_at_horizon'] else ''}")

    # --- references, READ from the library campaign, not re-run --------------
    ref: dict = {}
    p = OUT_LIB / f"{shape}.json"
    if p.exists():
        j = json.loads(p.read_text())
        for a in ("U_uniform", "HIST_best", "A1_cont", "A1_4bpp"):
            if a in j.get("arms", {}):
                ref[a] = j["arms"][a]
        ref["_source"] = str(p)
        ref["_best_hist_arm"] = j.get("best_hist_arm")
        ref["_library_verdict"] = j.get("verdict")
    res["library_reference"] = ref

    # reproducibility check: this pass's uniform arm against the library's
    if "U_uniform" in ref:
        res["uniform_reproduction_check"] = {
            "J_here": m_u["J"], "J_library": ref["U_uniform"]["J"],
            "abs_diff": abs(m_u["J"] - float(ref["U_uniform"]["J"])),
            "IoU_here": m_u["IoU"], "IoU_library": ref["U_uniform"]["IoU"]}
        log(f"uniform reproduction against out_lib: J {m_u['J']:.6f} against "
            f"{float(ref['U_uniform']['J']):.6f}")

    # --- verdict --------------------------------------------------------------
    d = res["arms"]["MS_4bpp"]
    cmp: dict = {}
    for name, r in (("U_uniform_here", m_u), ("HIST_best", ref.get("HIST_best")),
                    ("LIB_A1_4bpp", ref.get("A1_4bpp"))):
        if r is None:
            continue
        cmp[name] = {"beats_on_J": bool(d["J"] < float(r["J"])),
                     "beats_on_IoU": bool(d["IoU"] > float(r["IoU"])),
                     "dJ_rel": (float(r["J"]) - d["J"]) / max(abs(float(r["J"])), 1e-30),
                     "dIoU": d["IoU"] - float(r["IoU"])}
    res["verdict"] = {
        "deliverable_arm": "MS_4bpp", "winner_start": winner,
        "class": (lib.classify(d["IoU"], d["J"], float(ref["HIST_best"]["IoU"]),
                               float(ref["HIST_best"]["J"]))
                  if "HIST_best" in ref else None),
        "reaches_nominal_at_grid_120": bool(d["IoU"] >= lib.SOLVED_IOU),
        "against": cmp,
    }
    res["energy_gate_violations"] = [a for a, m in res["arms"].items()
                                     if not m["energy_gate"]["PASS"]]
    res["stop_at_horizon_arms"] = [a for a, m in res["arms"].items()
                                   if m["t_stop_at_horizon"]]
    res["wall_s"] = time.perf_counter() - t_start

    np.savez_compressed(out / f"{tag}_maps.npz", x=case.x, y=case.y, **maps_store)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done: winner {winner}, class {res['verdict']['class']}, "
        f"{res['n_evals_used_total']} evaluations, "
        f"{res['spent_forward_equivalents']:.1f} forward-equivalents, "
        f"wall {res['wall_s']:.0f} s, energy gate violations "
        f"{res['energy_gate_violations'] or 'none'}")
    return res


if __name__ == "__main__":
    _shape = sys.argv[1]
    _outdir = sys.argv[2]
    _budget = float(sys.argv[3]) if len(sys.argv) > 3 else BUDGET_FORWARD_EQUIVALENTS
    _sigma = float(sys.argv[4]) if len(sys.argv) > 4 else SIGMA_CELLS
    _control = sys.argv[5] if len(sys.argv) > 5 else ""
    main(_shape, _outdir, _budget, _sigma, _control)
