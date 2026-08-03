"""Emit the PHASE_C_REPORT.md tables STRAIGHT FROM solve3d/results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _l(n):
    return json.loads((RESULTS / n).read_text())


def arms_table() -> str:
    b = _l("phase_c_baselines.json")["arms"]
    s = _l("phase_c_solves.json")["arms"]
    pre = _l("phase_c_preregistration.json")
    o = ["### The arms (pre-registered priority order)", "",
         "| priority | arm | status | J_asymmetric (PRIMARY) | J_symmetric | "
         "IMPROVEMENT vs uniform (primary; + is better) |", "|---|---|---|---|---|---|"]
    uni = b["uniform_baseline"]
    ja_u = uni["J_asymmetric"]
    rows = [("1", "uniform_baseline", "scored", uni)]
    inv = b["inversion_map"]
    if inv.get("status") == "DROPPED":
        o.append(f"| 1 | uniform_baseline | scored | {uni['J_asymmetric']!r} | "
                 f"{uni['J_symmetric']!r} | reference |")
        o.append(f"| 3 | inversion_map | **DROPPED** | - | - | "
                 f"{inv['reason']} |")
    for name in ("solve_filter_only_symmetric", "solve_filter_only_asymmetric",
                 "solve_filter_only_asymmetric_scaled"):
        if name in s:
            a = s[name]
            d = (ja_u - a["J_asymmetric"]) / abs(ja_u)
            o.append(f"| {'2' if 'symmetric' in name and 'asym' not in name else '2/dev'} "
                     f"| {name} | {a['status'].split(':')[0]} "
                     f"({a['gradient_evaluations_used']}/{a['budget_gradient_evaluations']} evals) "
                     f"| {a['J_asymmetric']!r} | {a['J_symmetric']!r} | "
                     f"{100*d:+.2f} % |")
    for arm in pre["arms"]:
        if arm["name"] in ("solve_projection_beta_continuation",
                           "solve_filter_only_w3"):
            o.append(f"| {arm['priority']} | {arm['name']} | **NOT_RUN** | - | - | "
                     f"session compute exhausted; pre-registered "
                     f"unrun_arms_policy applies |")
    return "\n".join(o)


def stationarity_table() -> str:
    d = _l("phase_c_stationarity.json")["arms"]
    o = ["### Cold-start stationarity diagnostic (why the pre-registered arms "
         "stalled)", "",
         "| objective | \\|g\\| | \\|Pg\\| | \\|Pg\\|/\\|g\\| | best probe t | "
         "dJ at best | verdict |", "|---|---|---|---|---|---|---|"]
    for k, v in d.items():
        bp = v.get("best_probe") or {}
        o.append(f"| {k} | {v['grad_norm']!r} | {v['projected_grad_norm']!r} | "
                 f"{v['projected_over_full']!r} | {bp.get('t')!r} | "
                 f"{bp.get('rel_change_vs_v1')!r} | {v['verdict'].split(' (')[0]} |")
    o += ["", "Line probe along the projected steepest-descent direction:", "",
          "| objective | t | J | rel change vs v=1 |", "|---|---|---|---|"]
    for k, v in d.items():
        for p in v["line_probe"]:
            o.append(f"| {k} | {p['t']!r} | {p['J']!r} | {p['rel_change_vs_v1']!r} |")
    return "\n".join(o)


def trajectory_table() -> str:
    s = _l("phase_c_solves.json")["arms"]["solve_filter_only_asymmetric_scaled"]
    o = ["### The solve trajectory (scaled-start arm, asymmetric primary)", "",
         "| eval | J | t_stop [s] | at_horizon | \\|grad\\| | map mean | wall [s] |",
         "|---|---|---|---|---|---|---|"]
    for h in s["trajectory"]:
        o.append(f"| {h['eval']} | {h['J']!r} | {h['t_stop_s']!r} | "
                 f"{str(h['at_horizon']).lower()} | {h['grad_norm']!r} | "
                 f"{h['map_mean']!r} | {round(h['wall_s'])} |")
    o += ["", f"status `{s['status']}`, "
              f"{s['gradient_evaluations_used']} of "
              f"{s['budget_gradient_evaluations']} gradient evaluations, "
              f"{s['forward_equivalents_spent']!r} forward-equivalents spent, "
              f"wall {round(s['wall_total_s'])} s.",
          f"Delivered map: mean {s['map_stats']['mean']!r}, "
          f"min {s['map_stats']['min']!r}, max {s['map_stats']['max']!r}."]
    return "\n".join(o)


def decisive_table() -> str:
    b = _l("phase_c_baselines.json")["arms"]["uniform_baseline"]
    s = _l("phase_c_solves.json")["arms"]["solve_filter_only_asymmetric_scaled"]
    keys = [("J_asymmetric", "J asymmetric (PRIMARY)"),
            ("J_symmetric", "J symmetric (control, 2-D comparable)"),
            ("J_out_of_bounds", "J out-of-bounds (bed growth)"),
            ("J_in_bounds_deficit", "J in-bounds deficit"),
            ("out_of_part_melt_fraction_of_part", "out-of-part melt / part volume"),
            ("in_bounds_below_floor_fraction", "in-bounds fraction below the 0.85 floor"),
            ("part_mean_phi", "part mean melt fraction"),
            ("sigma_T_diagnostic_c", "sigma_T [C] (DIAGNOSTIC, never optimized)"),
            ("in_part_melt_frac_phi09", "in-part melt fraction at phi>=0.9"),
            ("bed_melt_frac_phi09", "bed melt fraction at phi>=0.9")]
    o = ["### The decisive comparison (solve mesh, same read rule)", "",
         "| quantity | uniform | solved | change |", "|---|---|---|---|"]
    for k, lab in keys:
        u, v = b[k], s[k]
        ch = ("n/a" if u == 0 else f"{100*(v-u)/abs(u):+.2f} %")
        o.append(f"| {lab} | {u!r} | {v!r} | {ch} |")
    return "\n".join(o)


def acceptance_table() -> str:
    g = _l("phase_c_gate.json")["arms"]["solve_filter_only_asymmetric_scaled"]
    h, s = g["mesh_holdout"], g["smoothing_robustness"]
    o = ["### Acceptance gates", "",
         f"**Mesh hold-out** ({h['solve_mesh']} -> {h['score_mesh']}), map "
         f"transfer moved total in-part dopant by "
         f"{h['transfer']['total_dopant_rel_move']!r}:", "",
         "| metric | measured move | band | uniform's OWN move | pass |",
         "|---|---|---|---|---|"]
    for k, c in h["checks"].items():
        o.append(f"| {k} | {c['measured']!r} | {c['band']!r} | "
                 f"{c['uniform_own_move']!r} | {'PASS' if c['pass'] else 'FAIL'} |")
    o += ["", "| J at each mesh | solve mesh | score mesh |", "|---|---|---|",
          f"| solved | {h['solved_at_solve_mesh']!r} | {h['solved_at_score_mesh']!r} |",
          f"| uniform | {h['uniform_at_solve_mesh']!r} | {h['uniform_at_score_mesh']!r} |",
          "",
          f"**Smoothing robustness**: blur at "
          f"{s['perturbation_radius_m']!r} m (filter radius "
          f"{s['filter_radius_m']!r} m), J {s['J_unblurred']!r} -> "
          f"{s['J_blurred']!r}, relative change {s['rel_change']!r} against a "
          f"tolerance of {s['tolerance']!r} -> "
          f"{'PASS' if s['pass'] else 'FAIL'}.", "",
          f"**SOLVED label**: `{str(g['solved_label']).lower()}` "
          f"(beats uniform {str(g['beats_uniform_in_grid']).lower()}, in-grid "
          f"margin {g['in_grid_margin_rel']!r}; hold-out "
          f"{str(h['pass']).lower()}; smoothing {str(s['pass']).lower()})."]
    return "\n".join(o)


def main() -> int:
    for f in (arms_table, stationarity_table, trajectory_table, decisive_table,
              acceptance_table):
        print(f())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def reread_table() -> str:
    d = _l("phase_c_reread.json")
    i1, i2 = d["item_1_stop_state_reread"], d["item_2_weight_sensitivity_by_rescoring"]
    o = ["#### Item 1 -- stop-state re-read", "",
         "| state | argmin (symmetric) | argmin (asymmetric) | shift [steps] | "
         "shift [s] | at_horizon |", "|---|---|---|---|---|---|"]
    for k, s in i1["stops"].items():
        o.append(f"| {k} | {s['argmin_symmetric']!r} | {s['argmin_asymmetric']!r} | "
                 f"{s['stop_shift_steps_asym_minus_sym']!r} | "
                 f"{s['stop_shift_s']!r} | "
                 f"{str(s['at_horizon_asymmetric']).lower()} |")
    o += ["",
          f"`already_scored_at_asymmetric_argmin` = "
          f"`{str(i1['already_scored_at_asymmetric_argmin']).lower()}`, "
          f"`bound_tightens_via_read_state` = "
          f"`{str(i1['bound_tightens_via_read_state']).lower()}`.", "",
          f"Objective moves the stop **{i1['objective_moves_the_stop_steps']} "
          f"steps**; the map moves it **{i1['map_moves_the_stop_steps']} steps**.",
          "", "#### Item 2 -- weight sensitivity by re-scoring", "",
          "| mesh | weighting | uniform | solved | margin (+ = solved wins) | "
          "solved wins |", "|---|---|---|---|---|---|"]
    for mesh, r in i2["rows"].items():
        for w in ("w10", "w3", "symmetric"):
            o.append(f"| {mesh} | {w} | {r[w]['uniform']!r} | {r[w]['solved']!r} | "
                     f"{100*r[w]['margin']:+.2f} % | "
                     f"{str(r[w]['solved_wins']).lower()} |")
    o += ["", f"`ranking_preserved_across_weightings` = "
              f"`{str(i2['ranking_preserved_across_weightings']).lower()}`."]
    return "\n".join(o)
