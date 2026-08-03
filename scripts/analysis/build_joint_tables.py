#!/usr/bin/env python3
"""Markdown tables for the JOINT per-angle dopant-map re-solve.

Writes `fgm_solve_campaign/out_joint/_tables.md`: the per-shape angle tables
with the fixed-map sweep columns alongside, the headline verdict table, the
symmetry-equivalent-angle reproducibility check (the campaign's own noise
floor), and the depth-check table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
JOINT = REPO / "fgm_solve_campaign/out_joint"
SWEEP = REPO / "outputs_eqs/orientation_optimization"

from analysis.joint_angle_lib import SYMMETRY_PERIOD_DEG  # noqa: E402

PAIRS = [("T_shape", "sigma"), ("L_shape", "sigma"), ("cross", "sigma"),
         ("star", "sigma"), ("cross", "eps")]

# Angle pairs the physics makes equivalent, VERIFIED on the fixed-map sweep's
# uniform arm (which carries no map asymmetry): the value in brackets is the
# uniform arm's own relative disagreement across the pair.
EQUIV_PAIRS = {
    "T_shape": [(0.0, 180.0), (22.5, 157.5), (45.0, 135.0), (67.5, 112.5)],
    "L_shape": [(0.0, 180.0)],
    "cross": [],
    "star": [(0.0, 36.0), (9.0, 27.0)],
}


def sweep_map(shape: str) -> dict:
    res = json.load(open(SWEEP / shape / "results.json"))
    out: dict = {}
    for r in res["rows"]:
        out.setdefault(float(r["angle_deg"]), {})[r["arm"]] = r
    return out


def load(shape: str, actuator: str, refine: bool = False) -> dict | None:
    p = (JOINT / f"{shape}_{actuator}_refine" / "results_refine.json" if refine
         else JOINT / f"{shape}_{actuator}" / "results.json")
    return json.loads(p.read_text()) if p.exists() else None


def angle_table(shape: str, actuator: str) -> list[str]:
    res = load(shape, actuator)
    sw = sweep_map(shape)
    best = res["joint_best_angle_deg"]
    lines = [
        f"### {shape}, actuator {actuator}",
        "",
        "| angle deg | J joint 4 bpp | J fixed map | J uniform | IoU joint | IoU fixed "
        "| grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(res["rows"], key=lambda x: x["angle_deg"]):
        a = float(r["angle_deg"])
        m = r["JOINT_4bpp"]
        g = sw.get(a, {}).get("graded", {})
        u = sw.get(a, {}).get("uniform", {})
        mark = "**" if a == best else ""
        hz = " H" if m["t_stop_at_horizon"] else ""
        ceil = " CEILING" if m["over_ceiling_250c"] else ""
        lines.append(
            f"| {mark}{a:g}{mark} | {mark}{m['J']:.2f}{mark} | {g.get('J', float('nan')):.2f} "
            f"| {u.get('J', float('nan')):.2f} | {m['IoU']:.4f} | {g.get('IoU', float('nan')):.4f} "
            f"| {m['bed_melt_pct_of_part']:.2f} | {m['part_under_melt_pct']:.2f} "
            f"| {m['mean_rho_rel_part_at_stop']:.4f} | {m['P_abs_W_per_m']:.1f} "
            f"| {m['t_stop_s']:.1f}{hz} | {m['max_T_at_stop_c']:.1f}{ceil} "
            f"| {r['winner_start']} |")
    lines.append("")
    return lines


def headline_table() -> list[str]:
    lines = ["## Headline: did the best angle move?", "",
             "| shape | actuator | fixed-map best deg | joint best deg | move deg "
             "| J fixed-map best | J joint best | J improvement % | IoU joint best "
             "| crosses 0.80 | crosses 0.95 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for shape, act in PAIRS:
        res = load(shape, act)
        if res is None:
            continue
        sw = sweep_map(shape)
        a_fix = res["fixed_map_sweep_best_angle_deg"]
        J_fix = sw[a_fix]["graded"]["J"]
        J_j = res["J_at_joint_best"]
        iou = res["IoU_at_joint_best"]
        lines.append(
            f"| {shape} | {act} | {a_fix:g} | {res['joint_best_angle_deg']:g} "
            f"| {res['angle_move_deg']:+.1f} | {J_fix:.2f} | {J_j:.2f} "
            f"| {100.0 * (J_fix - J_j) / J_fix:+.1f} | {iou:.4f} "
            f"| {'YES' if iou >= 0.80 else 'no'} | {'YES' if iou >= 0.95 else 'no'} |")
    lines.append("")
    return lines


def refine_table() -> list[str]:
    lines = ["## Depth check: the same angles re-solved at 40 forward-equivalents "
             "per start", "",
             "| shape | actuator | angles refined | scan best | refined best "
             "| argmin survives depth | J refined best | IoU refined best |",
             "|---|---|---|---|---|---|---|---|"]
    for shape, act in PAIRS:
        r = load(shape, act, refine=True)
        if r is None:
            lines.append(f"| {shape} | {act} | NOT RUN | | | | | |")
            continue
        lines.append(
            f"| {shape} | {act} | {', '.join(f'{a:g}' for a in r['angles_refined_deg'])} "
            f"| {r['scan_best_angle_deg']:g} | {r['refined_best_angle_deg']:g} "
            f"| {'YES' if r['argmin_survives_depth'] else 'NO'} "
            f"| {r['J_at_refined_best']:.2f} | {r['IoU_at_refined_best']:.4f} |")
    lines.append("")
    lines.append("Per-angle refined values:")
    lines.append("")
    lines.append("| shape | actuator | angle | J 4 bpp | IoU | start | max T C |")
    lines.append("|---|---|---|---|---|---|---|")
    for shape, act in PAIRS:
        r = load(shape, act, refine=True)
        if r is None:
            continue
        for row in sorted(r["rows"], key=lambda x: x["angle_deg"]):
            m = row["JOINT_4bpp"]
            lines.append(f"| {shape} | {act} | {row['angle_deg']:g} | {m['J']:.2f} "
                         f"| {m['IoU']:.4f} | {row['winner_start']} "
                         f"| {m['max_T_at_stop_c']:.1f} |")
    lines.append("")
    return lines


def symmetry_table() -> list[str]:
    lines = ["## The campaign's own noise floor: symmetry-equivalent angle pairs",
             "",
             "The forward physics is identical at these angle pairs, so any "
             "disagreement is the solve's angle-to-angle reproducibility, not a "
             "physical effect. The uniform column is the fixed-map sweep's uniform "
             "arm at the same pair and is the reference for how exactly the "
             "equivalence holds in the forward.",
             "",
             "| shape | actuator | pair deg | J joint a | J joint b | relative gap % "
             "| uniform relative gap % |",
             "|---|---|---|---|---|---|---|"]
    for shape, act in PAIRS:
        res = load(shape, act)
        if res is None:
            continue
        sw = sweep_map(shape)
        by = {float(r["angle_deg"]): r for r in res["rows"]}
        for a, b in EQUIV_PAIRS[shape]:
            if a not in by or b not in by:
                continue
            Ja, Jb = by[a]["J"], by[b]["J"]
            ua, ub = sw[a]["uniform"]["J"], sw[b]["uniform"]["J"]
            lines.append(
                f"| {shape} | {act} | {a:g} and {b:g} | {Ja:.2f} | {Jb:.2f} "
                f"| {100.0 * abs(Ja - Jb) / min(Ja, Jb):.2f} "
                f"| {100.0 * abs(ua - ub) / min(ua, ub):.3f} |")
    lines.append("")
    return lines


def budget_table() -> list[str]:
    lines = ["## Budget actually spent", "",
             "| shape | actuator | gradient evaluations per start | starts per angle "
             "| forward-equivalents per start | forward-equivalents per angle "
             "| angles | wall s |",
             "|---|---|---|---|---|---|---|---|"]
    for shape, act in PAIRS:
        res = load(shape, act)
        if res is None:
            continue
        n = res["n_gradient_evals_per_start"]
        ratio = res["ratio"]
        per_start = n * (1.0 + ratio)
        n_scored = 3    # two scoring forwards plus, at angle 0 only, the gate
        lines.append(
            f"| {shape} | {act} | {n} | 2 | {per_start:.1f} "
            f"| {2 * per_start + 2:.1f} (plus {n_scored - 1} scoring forwards) "
            f"| {len(res['angles_deg'])} | {res['wall_s']:.0f} |")
    lines.append("")
    return lines


def main() -> Path:
    out = ["# Joint per-angle re-solve, tables", ""]
    out += headline_table()
    out += refine_table()
    out += symmetry_table()
    out += budget_table()
    out += ["## Per-shape angle tables", ""]
    for shape, act in PAIRS:
        if load(shape, act) is not None:
            out += angle_table(shape, act)
    p = JOINT / "_tables.md"
    p.write_text("\n".join(out))
    print("wrote", p)
    print("\n".join(out))
    return p


if __name__ == "__main__":
    main()
