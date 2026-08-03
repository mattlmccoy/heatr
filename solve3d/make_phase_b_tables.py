"""Emit the PHASE_B_REPORT.md tables STRAIGHT FROM solve3d/results/*.json.

    heatr3d_d1_spike/env/bin/python -m solve3d.make_phase_b_tables

Same no-transcription rule as Phase A and D1: every number in the report is
printed by this script, so a stale or hand-copied figure is impossible.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _l(n):
    return json.loads((RESULTS / n).read_text())


def _probe_rows(gate, label):
    out = []
    for k, v in gate["probes"].items():
        out.append(f"| {label} | {k} | {v['best_rel_err']!r} | "
                   f"{v['best_abs_err']!r} | {v['analytic_directional']!r} | "
                   f"{v['best_eps']!r} | {v['evaluation_floor_estimate']!r} | "
                   f"{'yes' if v['pass_preferred'] else 'no'} / "
                   f"{'yes' if v['pass_subgradient'] else 'no'} |")
    return out


def protocol_table() -> str:
    d = _l("phase_b_protocol.json")
    o = ["### Pre-registered protocol (`phase_b_protocol.json`)", "",
         f"Thresholds source: {d['source_of_thresholds']}", "",
         "| item | value | citation |", "|---|---|---|",
         f"| epsilon sweep | {d['fd']['epsilons']} | {d['fd']['citation']} |"]
    t = d["thresholds"]
    for k in ("pass_rel_err", "subgradient_pass_rel_err", "transpose_rel_err"):
        o.append(f"| {k} | {t[k]!r} | {t['citations'][k]} |")
    o += ["", "| checklist item | status |", "|---|---|"]
    for it in d["checklist"]:
        o.append(f"| {it['n']}. {it['item'][:78]}... | {it['status']} |")
    m = d["fd_case"]["measured"]
    o += ["", "FD case, MEASURED (not asserted):", "",
          "| quantity | value |", "|---|---|"]
    for k in ("wall_forward_s", "n_dofs_total", "n_cells_total", "n_eqs_solves",
              "resolve_times_s", "n_march_steps", "part_mean_phi",
              "n_nodes_in_melt_window", "n_nodes_total", "n_substeps_used",
              "dt_stable_s", "energy_residual_frac", "clamp_bound"):
        o.append(f"| `{k}` | {m[k]!r} |")
    return "\n".join(o)


def gates_table() -> str:
    st = _l("phase_b_steady_gate.json")
    tr = _l("phase_b_transient_gate.json")
    o = ["### Every FD gate, all probes (the count at BOTH standards, "
         "per checklist item 5)", "",
         "| layer | probe | best rel err | best abs err | analytic "
         "directional | best eps | measured floor | pass 1e-6 / 1e-5 |",
         "|---|---|---|---|---|---|---|---|"]
    o += _probe_rows(st["gate"], "B1 steady")
    o += _probe_rows(tr["gate"], "B2 transient")
    o += _probe_rows(tr["design_gate"]["gate"], "B3 design")
    o += _probe_rows(tr["envelope_gate"]["gate"], "B4 envelope")
    o += ["", "| layer | probes passing 1e-6 | probes passing 1e-5 |",
          "|---|---|---|"]
    for lab, g in (("B1 steady", st["gate"]), ("B2 transient", tr["gate"]),
                   ("B3 design", tr["design_gate"]["gate"]),
                   ("B4 envelope", tr["envelope_gate"]["gate"])):
        o.append(f"| {lab} | {g['n_pass_preferred']} / {g['n_probes']} | "
                 f"{g['n_pass_subgradient']} / {g['n_probes']} |")
    return "\n".join(o)


def consistency_table() -> str:
    st = _l("phase_b_steady_gate.json")
    tr = _l("phase_b_transient_gate.json")
    o = ["### Assembly consistency and standing identities", "",
         "| quantity | value |", "|---|---|",
         f"| B1 assembly consistency (LU vs LU) | "
         f"{st['consistency']['assembly_consistency_gate']!r} |",
         f"| B1 same, vs Phase A's GMRES (recorded, NOT gated) | "
         f"{st['consistency']['vs_phase_a_iterative']['max_dQ_over_Qmax']!r} |",
         f"| B1 Qbar / power_density - 1 | "
         f"{st['forward']['qbar_over_power_density_minus_1']!r} |",
         f"| B1 Q clip active | {st['forward']['clip_active']!r} |",
         f"| B1 LU residual norm | {st['forward']['lu_residual_norm']!r} |"]
    return "\n".join(o)


def mutation_table() -> str:
    st = _l("phase_b_steady_gate.json")["mutations"]
    tr = _l("phase_b_transient_gate.json")["mutations"]
    o = ["### Mutation tests -- both mutants MUST fail the gate", "",
         "| layer | mutant | best rel err | passes 1e-5? | worst per-dof "
         "deviation vs the true gradient |", "|---|---|---|---|---|"]
    for lab, d in (("B1 steady", st), ("B2 transient", tr)):
        for name in ("renorm_frozen", "adjoint_dropped"):
            m = d[name]
            o.append(f"| {lab} | `{name}` | {m['worst_best_rel_err']!r} | "
                     f"{'YES (gate would be vacuous)' if m['all_pass_subgradient'] else 'no -- REJECTED'} | "
                     f"{m['max_rel_dev_vs_true_gradient']!r} |")
    return "\n".join(o)


def envelope_table() -> str:
    e = _l("phase_b_transient_gate.json")["envelope_gate"]
    r, a = e["read_state"], e["exact_agreement"]
    o = ["### B4 envelope stop time", "", "| quantity | value |", "|---|---|",
         f"| argmin step | {r['argmin_step']!r} of {r['n_steps']!r} |",
         f"| at_horizon | {r['at_horizon']!r} |",
         f"| argmin moves under probes (eps {r['probe_eps']!r}) | "
         f"{r['argmin_moves_under_probes']!r} |",
         f"| J at start / argmin / horizon | {r['J_at_start']!r} / {r['J']!r} "
         f"/ {r['J_at_horizon']!r} |",
         f"| envelope vs fixed-index gradient, max abs diff | "
         f"{a['max_abs_diff']!r} |",
         f"| envelope vs fixed-index gradient, rel diff | {a['rel_diff']!r} |",
         "", "Per-probe argmin under perturbation:", "",
         "| probe | argmin |", "|---|---|"]
    for k, v in r["probe_argmins"].items():
        o.append(f"| {k} | {v!r} |")
    return "\n".join(o)


def cost_table() -> str:
    c = _l("phase_b_cost.json")
    se, ck = c["store_everything"], c["checkpointing"]
    o = ["### Cost, in forward-equivalents", "",
         f"Accounting rule: {c['accounting_rule']}", "",
         f"Target: {c['target']} | 2-D reference: {c['reference_2d']}", "",
         "| scheme | wall gradient [s] | forward-equivalents | stored state "
         "[B] | recomputed steps | gradient vs store-everything |",
         "|---|---|---|---|---|---|",
         f"| store-everything | {ck['store_everything']['wall_gradient_s']!r} | "
         f"{ck['store_everything']['forward_equivalents']!r} | "
         f"{ck['store_everything']['stored_state_bytes']!r} | 0 | - |"]
    for k, v in ck["intervals"].items():
        o.append(f"| interval {k} | {v['wall_gradient_s']!r} | "
                 f"{v['forward_equivalents']!r} | {v['stored_state_bytes']!r} | "
                 f"{v['recomputed_steps']!r} | max rel diff "
                 f"{v['max_rel_diff']!r} |")
    o += ["", f"Chosen: interval {ck['chosen']['interval']!r}, "
              f"{ck['chosen']['forward_equivalents']!r} forward-equivalents, "
              f"{ck['chosen']['stored_state_bytes']!r} B "
              f"({ck['store_everything']['stored_state_bytes'] / ck['chosen']['stored_state_bytes']:.2f}x "
              f"less state than store-everything).", "",
          f"Scheme note: {ck['scheme']}", "",
          f"Wall forward on the FD case: {ck['wall_forward_s']!r} s over "
          f"{ck['n_steps']!r} march steps."]
    return "\n".join(o)


def main() -> int:
    for f in (protocol_table, consistency_table, gates_table, mutation_table,
              envelope_table, cost_table):
        print(f())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
