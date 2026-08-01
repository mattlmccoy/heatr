#!/usr/bin/env python3
"""Tables for the continuous-rotation campaign, written to out_rot/_tables.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "fgm_solve_campaign/out_rot"
SHAPES = ("T_shape", "L_shape", "cross", "star", "square")

H = ("| arm | J_phi | IoU | grow % | under % | rho at stop | stop s | max T C | "
     "events | energy resid % | gate |")
S = "|---|---|---|---|---|---|---|---|---|---|---|"


def row(m) -> str:
    hz = " (HORIZON, a bound)" if m.get("t_stop_at_horizon") else ""
    ce = " CEILING" if m.get("over_ceiling_250c") else ""
    return (f"| {m['arm']} | {m['J']:.2f}{hz} | {m['IoU']:.4f} | "
            f"{m['bed_melt_pct_of_part']:.2f} | {m['part_under_melt_pct']:.2f} | "
            f"{m['mean_rho_rel_part']:.4f} | {m['t_stop_s']:.1f} | "
            f"{m['max_T_part_c']:.1f}{ce} | {m.get('n_rotation_events', 0)} | "
            f"{100 * m['energy_residual_rel']:.2f} | "
            f"{'PASS' if m['energy_gate_PASS'] else '**FAIL**'} |")


def main() -> Path:
    out = ["# Continuous-rotation campaign, full tables",
           "",
           "Objective J_phi = sum over the WHOLE domain of (phi - chi_part)^2, read at "
           "each arm's OWN J-stop on a 1500-step horizon (dt 0.5 s, 750 s). The nominal "
           "target co-rotates with the part. Melted region phi >= 0.5. GRID 120.",
           ""]

    out += ["## Level 1: the rotationally averaged kernel (quasi-static)", "",
            "| shape | arm | J_phi | IoU | grow % | under % | rho at stop | "
            "P_abs W/m | stop s | max T C | energy gate |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        f = OUT / f"{sh}_rotavg.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        for name in ("AVG_uniform", "AVG_static0deg_map", "AVG_cont", "AVG_4bpp"):
            m = r["arms"].get(name)
            if not m:
                continue
            hz = " (HORIZON)" if m["t_stop_at_horizon"] else ""
            out.append(
                f"| {sh} | {name} | {m['J']:.2f}{hz} | {m['IoU']:.4f} | "
                f"{m['bed_melt_pct_of_part']:.2f} | {m['part_under_melt_pct']:.2f} | "
                f"{m['mean_rho_rel_part_at_stop']:.4f} | {m['P_abs_W_per_m']:.1f} | "
                f"{m['t_stop_s']:.1f} | {m['max_T_at_stop_c']:.1f} | "
                f"{'PASS' if m['energy_gate']['PASS'] else '**FAIL**'} |")
    out.append("")

    out += ["## Level 2: the TRUE rotating engine", ""]
    for sh in SHAPES:
        f = OUT / f"{sh}_verify.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows = list(r["rows"])
        sp = OUT / f"{sh}_speed.json"
        if sp.exists():
            seen = {m["arm"] for m in rows}
            rows += [m for m in json.loads(sp.read_text())["rows"]
                     if m["arm"] not in seen]
        rows.sort(key=lambda m: (m["arm"].startswith("C_"),
                                 not m["arm"].startswith("S_"),
                                 -m.get("period_s", 0.0), m["arm"]))
        out += [f"### {sh}", "",
                f"Joint campaign's winning static angle: {r['joint_best_angle_deg']} deg.",
                "", H, S]
        out += [row(m) for m in rows]
        out.append("")

    out += ["## Quasi-static approximation error", "",
            "The Level-1 averaged forward's PREDICTION for the averaged-kernel map "
            "against what the true rotating engine MEASURES for the same map.", "",
            "| shape | period s | J predicted (Level 1) | J measured (Level 2) | "
            "relative error % | IoU predicted | IoU measured |",
            "|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        f1, f2 = OUT / f"{sh}_rotavg.json", OUT / f"{sh}_verify.json"
        if not (f1.exists() and f2.exists()):
            continue
        r1 = json.loads(f1.read_text())
        r2 = json.loads(f2.read_text())
        rows2 = list(r2["rows"])
        sp = OUT / f"{sh}_speed.json"
        if sp.exists():
            seen = {m["arm"] for m in rows2}
            rows2 += [m for m in json.loads(sp.read_text())["rows"]
                      if m["arm"] not in seen]
        rows2.sort(key=lambda m: -m.get("period_s", 0.0))
        jp = float(r1["arms"]["AVG_cont"]["J"])
        ip = float(r1["arms"]["AVG_cont"]["IoU"])
        for m in rows2:
            if not m["arm"].startswith("R_avg_"):
                continue
            out.append(f"| {sh} | {m.get('period_s', float('nan')):.0f} | {jp:.2f} | "
                       f"{m['J']:.2f} | {100 * (m['J'] - jp) / max(jp, 1e-30):+.1f} | "
                       f"{ip:.4f} | {m['IoU']:.4f} |")
    out.append("")

    out += ["## The 90-degree indexing arm (exact permutation remap)", "",
            "Level 1 re-solved against the MATCHED four-angle averaged kernel "
            "{0, 90, 180, 270} degrees, then run on the true engine as a "
            "90-degree index every 2.0 s (375 events over the horizon). Every "
            "remap lands on grid points, so the bilinear interpolation error "
            "isolated by the C_null90 control is identically zero.", "",
            "| shape | Level-1 4-angle J | Level-1 4-angle IoU | engine arm | J_phi | "
            "IoU | grow % | under % | rho at stop | stop s | max T C | "
            "energy resid % | gate | quasi-static error % |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        f1 = OUT / f"{sh}_rotavg_step90.json"
        f2 = OUT / f"{sh}_index90.json"
        if not (f1.exists() and f2.exists()):
            continue
        a = json.loads(f1.read_text())["arms"]["AVG_cont"]
        for m in json.loads(f2.read_text())["rows"]:
            err = (100.0 * (m["J"] - a["J"]) / max(a["J"], 1e-30)
                   if m["arm"] != "I90_uniform" else float("nan"))
            hz = " (HORIZON, a bound)" if m.get("t_stop_at_horizon") else ""
            out.append(
                f"| {sh} | {a['J']:.2f} | {a['IoU']:.4f} | {m['arm']} | "
                f"{m['J']:.2f}{hz} | {m['IoU']:.4f} | "
                f"{m['bed_melt_pct_of_part']:.2f} | {m['part_under_melt_pct']:.2f} | "
                f"{m['mean_rho_rel_part']:.4f} | {m['t_stop_s']:.1f} | "
                f"{m['max_T_part_c']:.1f} | {100 * m['energy_residual_rel']:.2f} | "
                f"{'PASS' if m['energy_gate_PASS'] else '**FAIL**'} | "
                f"{'' if err != err else f'{err:+.1f}'} |")
    out.append("")

    p = OUT / "_tables.md"
    p.write_text("\n".join(out))
    print(p)
    return p


if __name__ == "__main__":
    main()
