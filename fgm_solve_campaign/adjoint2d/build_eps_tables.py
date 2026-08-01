"""Census tables for the permittivity-channel pass, de-confounded.

Every arm below is read at its OWN J-stop, t_stop = argmin over that arm's own
stored trajectory of J_phi; `t_stop_at_horizon` makes that arm's J a bound and
is marked. The melted region for intersection over union (IoU), growth and
under-melt is phi >= 0.5. GRID 120 x 120 unless a number says otherwise, and
`SOLVE_ROBUSTNESS_VALIDATION.md` established that absolute fidelity at grid 120
does not transfer to 160, so every IoU is a property of the method AT GRID 120
with an exactly reproduced dopant map.

The comparison this pass exists for: the historical arm `HIST_best` and the new
solved arms now share an ACTUATOR. Before this pass the historical maps moved
conductivity AND relative permittivity while the solve moved conductivity only.

Run:  ./.venv312/bin/python -m adjoint2d.build_eps_tables > out_eps/_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_EPS = ROOT / "out_eps"
OUT_LIB = ROOT / "out_lib"
OUT_MS = ROOT / "out_ms"
SHAPES = ("square", "circle", "hexagon", "triangle", "equilateral_triangle",
          "L_shape", "H_shape", "T_shape", "cross", "diamond", "ellipse",
          "octagon", "pentagon", "rectangle", "rounded_rect", "star", "star6",
          "trapezoid")
COMPACT = ("square", "rounded_rect", "star6", "rectangle")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def collect() -> list[dict]:
    rows = []
    for sh in SHAPES:
        e = _load(OUT_EPS / f"{sh}.json")
        if e is None:
            rows.append({"shape": sh, "missing": True})
            continue
        lib = _load(OUT_LIB / f"{sh}.json") or {"arms": {}}
        ms = _load(OUT_MS / f"{sh}.json") or {"arms": {}}
        ctl = _load(OUT_MS / f"{sh}_control_cold.json") or {"arms": {}}
        rows.append({
            "shape": sh, "missing": False,
            "winner": e["winner_start"],
            "eps_best": e["arms"]["EPS_best_4bpp"],
            "eps_cold": e["arms"].get("EPS_cold_4bpp"),
            "eps_warm": e["arms"].get("EPS_warm_4bpp"),
            "unif": e["arms"]["U_uniform"],
            "hist": lib["arms"].get("HIST_best"),
            "lib4": lib["arms"].get("A1_4bpp"),
            "ms4": ms["arms"].get("MS_4bpp"),
            "ctl4": ctl["arms"].get("MS_4bpp"),
            "class": e["verdict"]["class"],
            "class_budget_matched": e["verdict"]["class_budget_matched"],
            "invariance": e.get("uniform_channel_invariance_check"),
            "wall_s": e["wall_s"],
        })
    return rows


def _f(m, k, fmt="{:.2f}", miss="n/a"):
    if m is None or k not in m:
        return miss
    return fmt.format(float(m[k]))


def table_main(rows) -> str:
    out = ["| shape | win | J_eps_best | J_eps_cold | J_hist | J_ms | J_ctl | J_lib | J_unif "
           "| IoU_eps_best | IoU_eps_cold | IoU_hist | IoU_ms | grow % | under % "
           "| rho at stop | P_abs W/m | lvl | class |",
           "|" + "---|" * 19]
    for r in rows:
        if r["missing"]:
            out.append(f"| {r['shape']} | MISSING |" + " |" * 17)
            continue
        b, c = r["eps_best"], r["eps_cold"]
        hz = " (H)" if b.get("t_stop_at_horizon") else ""
        out.append(
            f"| {r['shape']} | {r['winner']} | {_f(b,'J')}{hz} | {_f(c,'J')} | "
            f"{_f(r['hist'],'J')} | {_f(r['ms4'],'J')} | {_f(r['ctl4'],'J')} | "
            f"{_f(r['lib4'],'J')} | {_f(r['unif'],'J')} | "
            f"{_f(b,'IoU','{:.4f}')} | {_f(c,'IoU','{:.4f}')} | "
            f"{_f(r['hist'],'IoU','{:.4f}')} | {_f(r['ms4'],'IoU','{:.4f}')} | "
            f"{_f(b,'bed_melt_pct_of_part')} | {_f(b,'part_under_melt_pct')} | "
            f"{_f(b,'mean_rho_rel_part_at_stop','{:.4f}')} | "
            f"{_f(b,'P_abs_W_per_m','{:.1f}')} | "
            f"{_f(b,'census_n_levels_used','{:.0f}')} | {r['class']} |")
    return "\n".join(out)


def _count(rows, arm_key, base_key, metric, better_high=False):
    n = ok = 0
    for r in rows:
        if r["missing"] or r[arm_key] is None or r[base_key] is None:
            continue
        n += 1
        a, b = float(r[arm_key][metric]), float(r[base_key][metric])
        ok += int(a > b) if better_high else int(a < b)
    return ok, n


def table_census(rows) -> str:
    arms = [("eps_best", "EPS best of two full-depth starts (80 fwd-equiv)"),
            ("eps_cold", "EPS cold full depth (40 fwd-equiv, budget matched)"),
            ("eps_warm", "EPS warm full depth (40 fwd-equiv)"),
            ("ms4", "conductivity-only multi-start (out_ms)"),
            ("ctl4", "conductivity-only filtered cold (out_ms control)"),
            ("lib4", "conductivity-only single start (out_lib)")]
    bases = [("hist", "best historical mask", ), ("unif", "uniform")]
    out = ["| arm | beats HIST on J_phi | beats HIST on IoU | beats UNIFORM on J_phi "
           "| IoU >= 0.95 at grid 120 | summed J_phi |", "|" + "---|" * 6]
    for key, label in arms:
        jh, nh = _count(rows, key, "hist", "J")
        ih, _ = _count(rows, key, "hist", "IoU", better_high=True)
        ju, nu = _count(rows, key, "unif", "J")
        solved = sum(1 for r in rows if not r["missing"] and r[key] is not None
                     and float(r[key]["IoU"]) >= 0.95)
        tot = sum(float(r[key]["J"]) for r in rows
                  if not r["missing"] and r[key] is not None)
        n = sum(1 for r in rows if not r["missing"] and r[key] is not None)
        out.append(f"| {label} | {jh} of {nh} | {ih} of {nh} | {ju} of {nu} | "
                   f"{solved} of {n} | {tot:.0f} |")
    del bases
    return "\n".join(out)


def table_compact(rows) -> str:
    """The four shapes the conductivity-only solve LOST to the historical mask."""
    out = ["| shape | J_hist | J_ms (sigma only) | J_eps_cold | J_eps_best | "
           "dJ vs hist % | IoU_hist | IoU_eps_best | verdict |", "|" + "---|" * 9]
    for r in rows:
        if r["missing"] or r["shape"] not in COMPACT or r["hist"] is None:
            continue
        b = r["eps_best"]
        d = 100.0 * (float(r["hist"]["J"]) - float(b["J"])) / max(abs(float(r["hist"]["J"])), 1e-30)
        v = "BEATS HIST" if float(b["J"]) < float(r["hist"]["J"]) else "still loses"
        out.append(f"| {r['shape']} | {_f(r['hist'],'J')} | {_f(r['ms4'],'J')} | "
                   f"{_f(r['eps_cold'],'J')} | {_f(b,'J')} | {d:+.1f} | "
                   f"{_f(r['hist'],'IoU','{:.4f}')} | {_f(b,'IoU','{:.4f}')} | {v} |")
    return "\n".join(out)


def table_channel_delta(rows) -> str:
    """What the permittivity channel bought, arm against matched-budget arm."""
    out = ["| shape | J_ctl (sigma only, 40) | J_eps_cold (eps, 40) | dJ % | "
           "IoU_ctl | IoU_eps_cold | dIoU |", "|" + "---|" * 7]
    for r in rows:
        if r["missing"] or r["ctl4"] is None or r["eps_cold"] is None:
            continue
        a, b = float(r["ctl4"]["J"]), float(r["eps_cold"]["J"])
        d = 100.0 * (a - b) / max(abs(a), 1e-30)
        out.append(f"| {r['shape']} | {a:.2f} | {b:.2f} | {d:+.1f} | "
                   f"{float(r['ctl4']['IoU']):.4f} | {float(r['eps_cold']['IoU']):.4f} | "
                   f"{float(r['eps_cold']['IoU']) - float(r['ctl4']['IoU']):+.4f} |")
    return "\n".join(out)


def table_gates(rows) -> str:
    out = ["| shape | energy gate violations | horizon arms | uniform channel "
           "invariance abs diff in J | wall s |", "|" + "---|" * 5]
    for r in rows:
        if r["missing"]:
            continue
        e = _load(OUT_EPS / f"{r['shape']}.json")
        inv = r["invariance"]
        out.append(f"| {r['shape']} | {e['energy_gate_violations'] or 'none'} | "
                   f"{e['stop_at_horizon_arms'] or 'none'} | "
                   f"{('%.3e' % float(inv['abs_diff'])) if inv else 'n/a'} | "
                   f"{r['wall_s']:.0f} |")
    return "\n".join(out)


def main() -> None:
    rows = collect()
    print("## 1. Per-shape census, grid 120, each arm at its own J-stop\n")
    print(table_main(rows))
    print("\n\n## 2. Census counts against the three baselines\n")
    print(table_census(rows))
    print("\n\n## 3. The four shapes the conductivity-only solve lost\n")
    print(table_compact(rows))
    print("\n\n## 4. What the permittivity channel bought at matched budget\n")
    print(table_channel_delta(rows))
    print("\n\n## 5. Gates and flags\n")
    print(table_gates(rows))


if __name__ == "__main__":
    main()
