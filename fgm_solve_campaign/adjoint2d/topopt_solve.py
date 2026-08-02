"""Topology-optimization solve of the dopant map: filter, projection, continuation.

One shape per invocation. The design variable is v; the injected saturation is

    s = P_beta(F(v))

with F the physical-radius filter and P the smoothed Heaviside (`topopt`). The
objective is the melt-region shape-fidelity objective against the
GRID-INDEPENDENT area-fill target (`chi_area`, `topopt_objective`), read at the
arm's own J-stop.

WHAT IS DIFFERENT FROM `ms_solve`, and all three are named every time a number
is quoted:

  1. the projection and the beta continuation are new;
  2. the filter radius is 1.0 mm, a PHYSICAL length, against the 1.5 cells
     (0.756 mm at grid 120) of the multi-start pass;
  3. the target is the sub-cell area fill, not the solve-grid binary raster, so
     J is NOT numerically comparable to `out_lib` or `out_ms` J. The
     intersection over union against the binary mask IS comparable and is
     reported alongside.

ACTUATOR. Conductivity only, the deployable channel: `eps_covary` is False on
every run, matching the adjoint arms of every previous pass. Every historical
arm co-varies permittivity, which is the standing actuator gap.

DRIVE. The per-shape calibrated voltage of the configuration, unchanged,
`enforce_generator_power` false. Arms are NOT dose matched; absorbed power is
reported per arm.

BUDGET. 40 forward-equivalents in total, converted to a gradient-evaluation
pool by the per-shape adjoint-to-forward ratio recorded by the library campaign
(the same convention and the same reason as `ms_solve.budget_ratio`), then split
over the five beta stages by `topopt.stage_split`. Single start, cold, from
uniform saturation 1.

Run:
  ./.venv312/bin/python -m adjoint2d.topopt_solve <shape> <outdir> [budget]
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, chi_area, control as ctl, energy_gate as eg
from . import forward as fwd, gradops
from . import library_solve as lib
from . import ms_solve as msv
from . import printability as pq
from . import mma as mma_mod
from . import topopt
from . import topopt_objective as tobj
from . import topopt_stage as tstage
from .pins import build_case, load_cfg

BUDGET_FORWARD_EQUIVALENTS = 40.0
DEFAULT_OPTIMIZER = "lbfgsb"
# Frozen BEFORE any solve of the retest and not tuned on any result. Svanberg's
# published defaults throughout except the move limit, which is the standard
# topology-optimization density move limit. See `mma.py`.
MMA_CFG = mma_mod.MMAConfig()
BOX = topopt.BOX
OUT_LIB = Path(__file__).resolve().parents[1] / "out_lib"


def output_tag(shape: str, control: str = "",
               optimizer: str = DEFAULT_OPTIMIZER,
               budget: float = BUDGET_FORWARD_EQUIVALENTS) -> str:
    """The stem of this run's result files.

    A pure function, and it is called AGAIN at write time rather than reusing a
    local. MEASURED reason: the first version held the stem in a local named
    `tag` which a later reference loop rebound, and all six shapes wrote to the
    same file. Recomputing from the arguments cannot be shadowed.

    The optimizer and the budget enter the stem only when they differ from the
    production recipe, so every stem written before this pass is unchanged and
    the four arms of the retest cannot overwrite one another.
    """
    if control not in ("", "filteronly"):
        raise ValueError(f"unknown control mode {control!r}")
    if optimizer not in tstage.OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    parts = [f"{shape}_control_{control}" if control else shape]
    if optimizer != DEFAULT_OPTIMIZER:
        parts.append(optimizer)
    if float(budget) != BUDGET_FORWARD_EQUIVALENTS:
        parts.append(f"b{float(budget):g}")
    return "_".join(parts)


def run_forward(case, s):
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=lib.PATIENCE, eps_covary=False)


def non_discreteness(s: np.ndarray, part_mask: np.ndarray) -> float:
    """M_nd = mean of 4 s (1 - s) over the part, 0 for a binary map, 1 at 0.5."""
    a = np.asarray(s, dtype=float)[np.asarray(part_mask, dtype=bool)]
    return float(np.mean(4.0 * a * (1.0 - a)))


def score(case, s: np.ndarray, chi: np.ndarray) -> dict:
    t0 = time.perf_counter()
    tr = run_forward(case, s)
    m = tobj.full_metrics(tr, case, chi)
    i = int(m["t_stop_index"])
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["mean_rho_rel_part_at_stop"] = float(tr.mean_rho_rel_part[i])
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_std_in_part"] = float(np.std(s[case.part_mask]))
    m["sat_min_in_part"] = float(np.min(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    m["non_discreteness"] = non_discreteness(s, case.part_mask)
    m["energy_gate"] = eg.gate_from_trajectory(tr, i)
    # The comparability reading: the SAME map scored against the binary raster
    # target, which is the objective every earlier report used.
    m["J_raster_chi"] = tobj.optimal_stop(tr, case, case.part_mask.astype(float)).J
    m["wall_s"] = time.perf_counter() - t0
    del tr
    return m


class Solver:
    """The continuation solve. Rows, design points, and the stage bookkeeping.

    The optimizer is a parameter (`topopt_stage.OPTIMIZERS`) and it is the ONLY
    thing that differs between the arms of `MMA_RETEST_REPORT.md`. Every
    optimizer receives the identical `evaluate` closure below, so the objective,
    the gradient, the stop rule and the budget accounting cannot drift between
    arms.
    """

    def __init__(self, case, chi, ops, log, optimizer: str = DEFAULT_OPTIMIZER):
        if optimizer not in tstage.OPTIMIZERS:
            raise ValueError(f"unknown optimizer {optimizer!r}")
        self.case = case
        self.chi = chi
        self.ops = ops
        self.log = log
        self.optimizer = optimizer
        self.rows: list[dict] = []
        self.store: dict[int, np.ndarray] = {}
        self.stages: list[dict] = []
        self.idx = np.flatnonzero(np.asarray(case.part_mask).ravel())
        self.mma_state = None
        self.beta_final_override = None

    def to_map(self, v, beta):
        return topopt.design_to_map(v, self.case.part_mask, dx=self.case.dx,
                                    radius_m=topopt.FILTER_RADIUS_M, beta=beta)

    def unpack(self, vec):
        v = np.ones(self.case.part_mask.shape)
        v.ravel()[self.idx] = vec
        return v

    def pack(self, v):
        return np.clip(np.asarray(v, dtype=float).ravel()[self.idx], BOX[0], BOX[1])

    # -- the one evaluation every optimizer sees -----------------------------

    def make_evaluate(self, beta_of_eval):
        """`beta_of_eval(n_done)` gives the beta the next evaluation runs at.

        It is a function of the global evaluation count, not a constant, so the
        carried-history L-BFGS-B arm can switch beta at a stage boundary
        WITHOUT ending the scipy call, which is the only way scipy's curvature
        memory can be carried across a continuation stage.
        """
        case, pm = self.case, self.case.part_mask

        def evaluate(vec):
            beta = float(beta_of_eval(len(self.rows)))
            v = self.unpack(vec)
            s = self.to_map(v, beta)
            tr = run_forward(case, s)
            st = tobj.optimal_stop(tr, case, self.chi)
            J, seed = tobj.J_and_seed(tr.T_at_end(st.index), case, self.chi)
            g_s = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=self.ops)
            g = topopt.design_vjp(g_s, v, pm, dx=case.dx,
                                  radius_m=topopt.FILTER_RADIUS_M, beta=beta)
            row = {"eval_index": len(self.rows) + 1, "beta": float(beta),
                   "J": float(J), "t_stop_index": int(st.index),
                   "t_stop_s": float(st.time_s),
                   "t_stop_at_horizon": bool(st.at_horizon),
                   "grad_norm": float(np.linalg.norm(g[pm])),
                   "grad_max_abs": float(np.max(np.abs(g[pm]))),
                   "non_discreteness": non_discreteness(s, pm),
                   "IoU": float(tobj.metrics(tr.T_at_end(st.index), case,
                                             self.chi)["IoU"])}
            self.rows.append(row)
            self.store[row["eval_index"]] = v.copy()
            del tr
            return float(J), g.ravel()[self.idx].astype(float)

        return evaluate

    def _best_of(self, rows, fallback):
        if not rows:
            return np.asarray(fallback, dtype=float), None
        b = min(rows, key=lambda r: r["J"])
        return self.store[b["eval_index"]], b

    @staticmethod
    def _num(info, key, spec):
        """Format a stage number, tolerating the None a starved stage writes.

        A stage can legitimately record None: the carried-history arm's scipy
        call can declare convergence before a later beta is reached. That is a
        result to be reported, not a crash, and the first version of this code
        crashed on it (`logs_mma/triangle_lbfgsb_carry_b80.log`, first attempt).
        """
        v = info.get(key)
        return "none" if v is None else format(float(v), spec)

    def _log_stage(self, info):
        self.log(f"  beta {info['beta']:5.1f}: {info['n_new_used']} evals, J "
                 f"{self._num(info, 'first_J', '.2f')} -> "
                 f"{self._num(info, 'best_J', '.2f')}, IoU "
                 f"{self._num(info, 'best_IoU', '.4f')}, M_nd "
                 f"{self._num(info, 'best_non_discreteness', '.3f')}, "
                 f"{info['wall_s']:.0f} s")

    # -- one stage, for the per-stage optimizers -----------------------------

    def stage(self, v_start: np.ndarray, beta: float, n_new: int) -> np.ndarray:
        """One beta stage. Returns the stage's best design point."""
        n_before = len(self.rows)
        if self.optimizer == "mma" and self.mma_state is None:
            self.mma_state = mma_mod.MMA(self.pack(v_start), lower=BOX[0],
                                         upper=BOX[1], cfg=MMA_CFG)
        runner = tstage.StageRunner(self.make_evaluate(lambda _n: beta), BOX,
                                    optimizer=self.optimizer,
                                    mma_state=self.mma_state)
        _v, rinfo = runner.run(self.pack(v_start), int(n_new))
        stage_rows = self.rows[n_before:]
        info = {"beta": float(beta), "n_new_allowed": int(n_new),
                "optimizer": self.optimizer,
                "n_new_used": len(stage_rows), "wall_s": rinfo["wall_s"],
                "n_optimizer_restarts": rinfo.get("n_optimizer_restarts", 0),
                "mma_state": rinfo.get("mma_state")}
        out, b = self._best_of(stage_rows, v_start)
        if b is not None:
            info.update({"best_J": b["J"], "best_eval_index": b["eval_index"],
                         "first_J": stage_rows[0]["J"], "best_IoU": b["IoU"],
                         "best_non_discreteness": b["non_discreteness"]})
        else:
            info.update({"best_J": None, "best_eval_index": None})
        self.stages.append(info)
        self._log_stage(info)
        return out

    # -- the whole continuation, for the carried-history arm ------------------

    def run_carried(self, v_start, schedule, split):
        """ONE L-BFGS-B call across every stage, beta switched inside it.

        scipy exposes no way to seed the curvature memory of a fresh
        `minimize` call, so the only way to carry it across a beta jump is
        never to end the call. The objective therefore changes underneath the
        optimizer at each stage boundary. That is a real and named property of
        this arm: the line search may be evaluating a different function from
        the one that produced its current direction.
        """
        bounds = np.cumsum([int(n) for n in split])
        total = int(bounds[-1]) if len(bounds) else 0

        def beta_of(n_done):
            for beta, cum in zip(schedule, bounds):
                if n_done < cum:
                    return float(beta)
            return float(schedule[-1])

        runner = tstage.StageRunner(self.make_evaluate(beta_of), BOX,
                                    optimizer="lbfgsb_carry")
        _v, rinfo = runner.run(self.pack(v_start), total)
        for beta, n_new in zip(schedule, split):
            rows = [r for r in self.rows if r["beta"] == float(beta)]
            info = {"beta": float(beta), "n_new_allowed": int(n_new),
                    "optimizer": self.optimizer, "n_new_used": len(rows),
                    "n_optimizer_restarts": (rinfo["n_optimizer_restarts"]
                                             if beta == schedule[0] else 0),
                    "wall_s": (rinfo["wall_s"] if beta == schedule[0] else 0.0)}
            _o, b = self._best_of(rows, v_start)
            if b is not None:
                info.update({"best_J": b["J"], "best_eval_index": b["eval_index"],
                             "first_J": rows[0]["J"], "best_IoU": b["IoU"],
                             "best_non_discreteness": b["non_discreteness"]})
            else:
                info.update({"best_J": None, "best_eval_index": None})
            self.stages.append(info)
            self._log_stage(info)
        # The deliverable beta is the HIGHEST beta actually evaluated, which is
        # not necessarily the last scheduled one: see the docstring.
        betas_seen = sorted({r["beta"] for r in self.rows})
        if not betas_seen:
            raise RuntimeError("the carried-history arm produced no evaluation")
        self.beta_final_override = float(betas_seen[-1])
        rows_f = [r for r in self.rows if r["beta"] == self.beta_final_override]
        out, _b = self._best_of(rows_f, v_start)
        return out


def main(shape: str, outdir: str,
         budget: float = BUDGET_FORWARD_EQUIVALENTS,
         control: str = "",
         optimizer: str = DEFAULT_OPTIMIZER) -> dict:
    """`control='filteronly'` spends the WHOLE pool at beta = 0.

    That control changes exactly one thing against the continuation arm, the
    projection, at the same radius, the same target and the same budget, so the
    two together separate the projection from the radius-and-target change. It
    is the missing arm of `MULTISTART_REPORT.md` Section 9 limit 2, which could
    not separate its filter from its multi-start.
    """
    if shape not in lib.SHAPES:
        raise ValueError(f"{shape!r} is not in the standardized library")
    if control not in ("", "filteronly"):
        raise ValueError(f"unknown control mode {control!r}")
    if optimizer not in tstage.OPTIMIZERS:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    tag = output_tag(shape, control, optimizer, budget)

    def log(msg):
        print(f"[{tag}] {msg}", flush=True)

    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    chi, chi_info = chi_area.chi_from_cfg(cfg, case.x, case.y)
    sig = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)

    res: dict = {
        "shape": shape, "config": str(cfg_path),
        "mode": control or "topopt_continuation",
        "optimizer": optimizer,
        "mma_config": (dataclasses.asdict(MMA_CFG)
                       if optimizer == "mma" else None),
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "enforce_generator_power": bool(
            cfg["electric"].get("enforce_generator_power", False)),
        "actuator": "conductivity only (eps_covary=False)",
        "n_part_cells": case.n_part, "n_grid": int(pm.shape[0]), "dx_m": case.dx,
        "filter_radius_m": topopt.FILTER_RADIUS_M, "sigma_cells": sig,
        "eta": topopt.ETA, "beta_schedule": list(topopt.BETA_SCHEDULE),
        "budget_forward_equivalents_total": float(budget), "box": list(BOX),
        "chi": chi_info,
        "chi_vs_raster": chi_area.raster_vs_area_delta(pm, chi, case.dx, case.dy),
        "stop_convention": "t_stop = argmin over the arm's own trajectory of J "
                           "against the area-fill chi; at_horizon flagged; the "
                           "melted region for IoU is phi >= 0.5",
        "arms": {},
    }
    maps: dict[str, np.ndarray] = {"part_mask": pm.astype(np.uint8),
                                   "chi_area": chi.astype(np.float32)}
    log(f"dx {case.dx*1e3:.4f} mm, filter radius {topopt.FILTER_RADIUS_M*1e3:.2f} mm = "
        f"{sig:.3f} cells; chi area {chi_info['area_m2']*1e6:.3f} mm2 against raster "
        f"{res['chi_vs_raster']['area_raster_m2']*1e6:.3f} mm2 "
        f"({res['chi_vs_raster']['area_rel_delta']*100:+.2f} percent)")

    # --- cost model; the uniform arm comes free ------------------------------
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = run_forward(case, s_u)
    t_b = time.perf_counter()
    st_u = tobj.optimal_stop(tr_u, case, chi)
    _J, seed = tobj.J_and_seed(tr_u.T_at_end(st_u.index), case, chi)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed}, grad_ops=ops)
    t_d = time.perf_counter()
    ratio, ratio_info = msv.budget_ratio(shape, (t_d - t_c) / max(t_b - t_a, 1e-9), log)
    pool = ctl.max_gradient_evals(float(budget), ratio)
    if control == "filteronly":
        schedule: tuple[float, ...] = (0.0,)
        split: tuple[int, ...] = (pool,)
    else:
        schedule = topopt.BETA_SCHEDULE
        split = topopt.stage_split(pool, len(schedule))
    res["beta_schedule"] = list(schedule)
    del tr_u

    m_u = score(case, s_u, chi)
    m_u["arm"] = "U_uniform"
    res["arms"]["U_uniform"] = m_u
    maps["U_uniform"] = s_u
    res["cost"] = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c, "ratio": ratio,
                   "ratio_info": ratio_info, "pool_gradient_evals": pool,
                   "stage_split": list(split)}
    log(f"pool {pool} gradient evaluations, stage split {list(split)} over beta "
        f"{list(schedule)}")
    log(f"U_uniform J {m_u['J']:.2f} IoU {m_u['IoU']:.4f} IoU_area {m_u['IoU_area']:.4f} "
        f"P {m_u['P_abs_W_per_m']:.1f}")

    # --- the continuation ----------------------------------------------------
    solver = Solver(case, chi, ops, log, optimizer=optimizer)
    v = np.ones(pm.shape)
    if optimizer == "lbfgsb_carry":
        v = solver.run_carried(v, schedule, split)
    else:
        for beta, n_new in zip(schedule, split):
            if int(n_new) <= 0:
                solver.stages.append({"beta": float(beta), "n_new_allowed": 0,
                                      "stopped_reason": "no budget left for this stage",
                                      "n_new_used": 0, "wall_s": 0.0, "best_J": None})
                log(f"  beta {beta:5.1f}: SKIPPED, no budget")
                continue
            v = solver.stage(v, float(beta), int(n_new))

    res["stages"] = solver.stages
    res["n_optimizer_restarts"] = sum(
        int(s.get("n_optimizer_restarts") or 0) for s in solver.stages)
    res["rows"] = solver.rows
    res["n_evals_used_total"] = len(solver.rows)
    res["spent_forward_equivalents"] = ctl.forward_equivalents(
        len(solver.rows), len(solver.rows), ratio)
    if not solver.rows:
        raise RuntimeError(f"{shape}: the continuation produced no evaluation")

    beta_final = float(schedule[max(i for i, n in enumerate(split) if n > 0)])
    if solver.beta_final_override is not None:
        if solver.beta_final_override != beta_final:
            log(f"NOTE: the scheduled final beta was {beta_final:g} but the "
                f"highest beta actually evaluated was "
                f"{solver.beta_final_override:g}; the deliverable is built at "
                f"the beta it was optimized at, and this is reported")
        beta_final = solver.beta_final_override
    res["beta_final_scheduled"] = float(schedule[max(
        i for i, n in enumerate(split) if n > 0)])
    res["beta_final_delivered"] = float(beta_final)
    s_cont = solver.to_map(v, beta_final)
    m_c = score(case, s_cont, chi)
    m_c.update({"arm": "TO_cont", "beta": beta_final})
    res["arms"]["TO_cont"] = m_c
    maps["TO_cont"] = s_cont
    maps["TO_v"] = v

    s_q = pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=1.0)
    m_q = score(case, s_q, chi)
    m_q.update({"arm": "TO_4bpp", "beta": beta_final})
    m_q.update({f"census_{k}": val for k, val in pq.level_census(s_q, pm, bpp=4).items()})
    res["arms"]["TO_4bpp"] = m_q
    maps["TO_4bpp"] = s_q
    log(f"TO_4bpp (beta {beta_final:g}) J {m_q['J']:.2f} IoU {m_q['IoU']:.4f} "
        f"IoU_area {m_q['IoU_area']:.4f} M_nd {m_q['non_discreteness']:.3f} "
        f"grow {m_q['bed_melt_pct_of_part']:.2f}% under {m_q['part_under_melt_pct']:.2f}% "
        f"P {m_q['P_abs_W_per_m']:.1f} stop {m_q['t_stop_s']:.1f} s"
        f"{' HORIZON' if m_q['t_stop_at_horizon'] else ''}")

    # The global argmin over ALL stages, reported as a diagnostic. It can be a
    # low-beta grey iterate; the deliverable is deliberately the crisp one.
    b_any = min(solver.rows, key=lambda r: r["J"])
    res["best_any_stage"] = {"eval_index": b_any["eval_index"], "beta": b_any["beta"],
                             "J": b_any["J"], "IoU": b_any["IoU"],
                             "is_final_stage": bool(b_any["beta"] == beta_final)}

    # --- references, READ from the earlier passes ----------------------------
    ref: dict = {}
    for ref_name, path in (
            ("library", OUT_LIB / f"{shape}.json"),
            ("multistart",
             Path(__file__).resolve().parents[1] / "out_ms" / f"{shape}.json")):
        if path.exists():
            j = json.loads(path.read_text())
            ref[ref_name] = {"source": str(path),
                        "arms": {a: {k: j["arms"][a].get(k) for k in ("J", "IoU",
                                                                      "P_abs_W_per_m",
                                                                      "t_stop_s")}
                                 for a in j.get("arms", {})}}
    res["reference_note"] = (
        "Reference J values were computed against the BINARY RASTER target and "
        "are NOT comparable to this pass's J. Compare on IoU, and on this pass's "
        "J_raster_chi, which is the same map scored under the old target.")
    res["references"] = ref

    res["energy_gate_violations"] = [a for a, m in res["arms"].items()
                                     if not m["energy_gate"]["PASS"]]
    res["stop_at_horizon_arms"] = [a for a, m in res["arms"].items()
                                   if m["t_stop_at_horizon"]]
    res["wall_s"] = time.perf_counter() - t_start
    stem = output_tag(shape, control, optimizer, budget)
    np.savez_compressed(out / f"{stem}_maps.npz", x=case.x, y=case.y, **maps)
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"done: {len(solver.rows)} evaluations, "
        f"{res['spent_forward_equivalents']:.1f} forward-equivalents, wall "
        f"{res['wall_s']:.0f} s, energy gate violations "
        f"{res['energy_gate_violations'] or 'none'}")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else BUDGET_FORWARD_EQUIVALENTS,
         sys.argv[4] if len(sys.argv) > 4 else "",
         sys.argv[5] if len(sys.argv) > 5 else DEFAULT_OPTIMIZER)
