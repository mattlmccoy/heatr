"""Arm H1eps: the SAME gain-calibrated proportional-inverse control, run in the
production permittivity-co-varying channel that the step-2 campaign used.

Needed because the two-sided per-node hook the adjoint arm uses pins
permittivity, and the step-2 control did not. Without this arm the head-to-head
would be against a weaker control than the one already published.

Also reproduces the step-2 stored square map end-to-end as an independent
anchor on the prototype.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import control, forward as fwd, objective as obj
from .pins import build_case, load_cfg
from .prod import rfam


def evaluate(case, s, eps_covary: bool) -> dict:
    tr = fwd.forward(case, s, stop_after_phi=0.90, stop_margin_steps=2,
                     eps_covary=eps_covary)
    m = obj.scored_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["feasible"] = m["status"] == "OK"
    m["fit"] = m["heating_peak_sigma_T_c"]
    m["holdout"] = m["melt_onset_sigma_T_c"]
    return m


def reproduce_step2_map(case, npz_path: str, magnitude: float, cfg) -> dict:
    """Load a stored step-2 map through the PRODUCTION loader and score it."""
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["fgm_feedback"] = {
        "enabled": True,
        "saturation_map_npz": str(npz_path),
        "magnitude": float(magnitude),
        "baseline_saturation": 0.5,
        "iterate": False,
    }
    fb = rfam._FgmFeedback.from_config(cfg2, case.x, case.y, case.part_mask)
    s = np.asarray(fb.sat_map, dtype=np.float64)
    m = evaluate(case, s, eps_covary=True)
    m["source_npz"] = str(npz_path)
    m["magnitude"] = float(magnitude)
    return m


def run(cfg_path: str, shape: str, outdir: str, n_gain: int = 39,
        outside: float = 1.0, tag: str = 'H1eps') -> dict:
    cfg = load_cfg(Path(cfg_path).resolve())
    case = build_case(cfg)
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    s_u = np.ones(case.part_mask.shape)
    tr_u = fwd.forward(case, s_u, stop_after_phi=0.90, stop_margin_steps=2)
    rs = obj.read_states(tr_u)
    if rs.melt_onset_index is None:
        raise obj.MeltNotReached("uniform arm never melts")
    proxy = tr_u.T_at_end(rs.melt_onset_index)

    rows: list[dict] = []
    maps: dict[str, np.ndarray] = {}

    def fit_of(m: float) -> float:
        sat = control.proportional_inverse_map(proxy, case.part_mask, magnitude=m,
                                               outside=outside)
        met = evaluate(case, sat, eps_covary=True)
        met["magnitude"] = float(m)
        met["eval_index"] = len(rows) + 1
        rows.append(met)
        maps[str(met["eval_index"])] = sat
        if met["feasible"]:
            return met["fit"]
        return control.infeasible_rank(met, t_pc_c=case.pins.t_pc_c)

    control.golden_section(fit_of, 0.05, 2.50, n_evals=n_gain)

    res = {"shape": shape, "arm": tag, "outside_saturation": float(outside),
           "config": str(cfg_path),
           "uniform": evaluate(case, s_u, eps_covary=True), "rows": rows, "budgets": {}}
    for B in (5, 15, 40):
        k = max(0, B - 1)
        sel = control.select_on_fit(rows[:k])
        selm = control.select_on_holdout(rows[:k])
        res["budgets"][str(B)] = {"n_gain_evals": k, f"{tag}_peak": sel, f"{tag}_melt": selm}
    best = control.select_on_fit(rows)
    if best is not None:
        np.savez_compressed(out / f"{shape}_{tag}_map.npz", sat=maps[str(best["eval_index"])])
    (out / f"{shape}_{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "reproduce":
        cfg = load_cfg(Path(sys.argv[2]).resolve())
        case = build_case(cfg)
        r = reproduce_step2_map(case, sys.argv[3], float(sys.argv[4]), cfg)
        print(json.dumps(r, indent=2, default=float))
    else:
        outside = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        tag = sys.argv[5] if len(sys.argv) > 5 else "H1eps"
        r = run(sys.argv[1], sys.argv[2], sys.argv[3], outside=outside, tag=tag)
        print(json.dumps(r["budgets"], indent=2, default=float))
