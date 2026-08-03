"""Markdown tables for TOPOPT_REPORT.md, straight from the result JSONs.

Nothing here computes physics. It only reads `out_topopt/`, `out_ms/`,
`out_lib/` and `out_robust/` and formats, so every number in the report can be
traced to a file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_topopt"
OUT_MS = ROOT / "out_ms"
OUT_LIB = ROOT / "out_lib"
OUT_ROBUST = ROOT / "out_robust"
SHAPES = ("square", "circle", "trapezoid", "triangle", "diamond", "rectangle")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _f(v, fmt="{:.4f}"):
    return "" if v is None else fmt.format(v)


def t_solve() -> str:
    L = ["| shape | pool | uniform J | uniform IoU | TO_cont J | TO_4bpp J | "
         "TO_4bpp IoU | IoU_area | M_nd | P_abs W/m | stop s | horizon |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        j = _load(OUT / f"{sh}.json")
        if not j:
            L.append(f"| {sh} | MISSING | | | | | | | | | | |")
            continue
        u, c, q = j["arms"]["U_uniform"], j["arms"]["TO_cont"], j["arms"]["TO_4bpp"]
        L.append(f"| {sh} | {j['cost']['pool_gradient_evals']} | {_f(u['J'],'{:.2f}')} | "
                 f"{_f(u['IoU'])} | {_f(c['J'],'{:.2f}')} | {_f(q['J'],'{:.2f}')} | "
                 f"{_f(q['IoU'])} | {_f(q['IoU_area'])} | {_f(q['non_discreteness'],'{:.3f}')} | "
                 f"{_f(q['P_abs_W_per_m'],'{:.1f}')} | {_f(q['t_stop_s'],'{:.1f}')} | "
                 f"{'YES' if q['t_stop_at_horizon'] else 'no'} |")
    return "\n".join(L)


def t_control() -> str:
    L = ["| shape | TO_4bpp J (continuation) | control J (beta 0, same budget) | "
         "dJ percent | TO IoU | control IoU | TO M_nd | control M_nd |",
         "|---|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        j = _load(OUT / f"{sh}.json")
        c = _load(OUT / f"{sh}_control_filteronly.json")
        if not j or not c:
            L.append(f"| {sh} | | MISSING | | | | | |")
            continue
        a, b = j["arms"]["TO_4bpp"], c["arms"]["TO_4bpp"]
        d = 100.0 * (b["J"] - a["J"]) / max(abs(b["J"]), 1e-30)
        L.append(f"| {sh} | {_f(a['J'],'{:.2f}')} | {_f(b['J'],'{:.2f}')} | "
                 f"{d:+.1f} | {_f(a['IoU'])} | {_f(b['IoU'])} | "
                 f"{_f(a['non_discreteness'],'{:.3f}')} | "
                 f"{_f(b['non_discreteness'],'{:.3f}')} |")
    return "\n".join(L)


def t_iou_vs_history() -> str:
    """IoU is the only cross-pass comparable metric; J_raster_chi is the second."""
    L = ["| shape | uniform IoU | out_lib A1_4bpp IoU | out_ms MS_4bpp IoU | "
         "topopt TO_4bpp IoU | topopt J_raster_chi | out_ms J | out_lib J |",
         "|---|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        j = _load(OUT / f"{sh}.json")
        lb = _load(OUT_LIB / f"{sh}.json")
        ms = _load(OUT_MS / f"{sh}.json")
        if not j:
            continue
        a = j["arms"]["TO_4bpp"]
        lba = (lb or {}).get("arms", {}).get("A1_4bpp", {})
        msa = (ms or {}).get("arms", {}).get("MS_4bpp", {})
        L.append(f"| {sh} | {_f(j['arms']['U_uniform']['IoU'])} | "
                 f"{_f(lba.get('IoU'))} | {_f(msa.get('IoU'))} | {_f(a['IoU'])} | "
                 f"{_f(a['J_raster_chi'],'{:.2f}')} | {_f(msa.get('J'),'{:.2f}')} | "
                 f"{_f(lba.get('J'),'{:.2f}')} |")
    return "\n".join(L)


def t_gate_A() -> str:
    L = ["| shape | IoU 120 | IoU 160 map transfer | IoU 160 design transfer | "
         "uniform IoU 160 | J 160 best | uniform J 160 | drop | recal V | SOLVED at 160 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        r = _load(OUT / f"{sh}_robust.json")
        if not r:
            L.append(f"| {sh} | MISSING | | | | | | | | |")
            continue
        v = r["gate_A_verdict"]
        g = r["gate_A_grid"]
        best = g[v["best_transfer"]]
        L.append(f"| {sh} | {_f(v['IoU_at_120'])} | "
                 f"{_f(v['IoU_at_160_maptransfer_recal'])} | "
                 f"{_f(v['IoU_at_160_designtransfer_recal'])} | "
                 f"{_f(g['U_uniform_160_recal']['IoU'])} | {_f(best['J'],'{:.2f}')} | "
                 f"{_f(g['U_uniform_160_recal']['J'],'{:.2f}')} | "
                 f"{v['IoU_drop_best']:+.4f} | "
                 f"{_f(r['recal']['voltage_v_recalibrated'],'{:.1f}')} | "
                 f"{'YES' if v['reaches_SOLVED_at_160'] else 'no'} |")
    return "\n".join(L)


def t_gate_B() -> str:
    L = ["| shape | dJ at 0.25 mm | dJ at 0.50 mm | dJ at 0.76 mm | dJ at 1.01 mm "
         "(at the radius, not gated) | max sub-radius | PASS |",
         "|---|---|---|---|---|---|---|"]
    for sh in SHAPES:
        r = _load(OUT / f"{sh}_robust.json")
        if not r:
            L.append(f"| {sh} | MISSING | | | | | |")
            continue
        b = r["gate_B_sub_radius"]
        cells = sorted(b.values(), key=lambda m: m["gaussian_sigma_cells"])
        vals = [f"{100*m['dJ_rel']:+.2f} %" for m in cells]
        while len(vals) < 4:
            vals.append("")
        L.append(f"| {sh} | " + " | ".join(vals[:4]) +
                 f" | {100*r['gate_B_verdict']['max_abs_dJ_rel']:.2f} % | "
                 f"{'PASS' if r['gate_B_verdict']['PASS'] else 'FAIL'} |")
    return "\n".join(L)


def t_gate_B_history() -> str:
    """The same probe on the earlier arms, so the improvement is visible."""
    L = ["| shape | arm | dJ at 1 cell (0.50 mm) | dJ at 2 cells (1.01 mm) |",
         "|---|---|---|---|"]
    for sh in SHAPES:
        old = _load(OUT_ROBUST / f"{sh}_rim.json")
        if old:
            v = old["verdict"]
            L.append(f"| {sh} | unfiltered single start (out_robust) | "
                     f"{100*v['dJ_rel_r1.0']:+.1f} % | {100*v['dJ_rel_r2.0']:+.1f} % |")
        msr = _load(OUT_MS / f"{sh}_robust.json")
        if msr:
            b = msr["baseline_120"]["MS_4bpp"]["J"]
            L.append(f"| {sh} | filtered multi-start (out_ms) | "
                     f"{100*(msr['rim']['r1.0']['J']-b)/abs(b):+.1f} % | "
                     f"{100*(msr['rim']['r2.0']['J']-b)/abs(b):+.1f} % |")
        r = _load(OUT / f"{sh}_robust.json")
        if r:
            b = r["gate_B_sub_radius"]
            L.append(f"| {sh} | topopt (this pass) | "
                     f"{100*b['r1.0']['dJ_rel']:+.1f} % | "
                     f"{100*b['r2.0']['dJ_rel']:+.1f} % |")
    return "\n".join(L)


def t_cost() -> str:
    L = ["| shape | evaluations | forward-equivalents | solve wall s | "
         "control wall s | robust wall s |", "|---|---|---|---|---|---|"]
    tot = 0.0
    for sh in SHAPES:
        j = _load(OUT / f"{sh}.json")
        c = _load(OUT / f"{sh}_control_filteronly.json")
        r = _load(OUT / f"{sh}_robust.json")
        if not j:
            continue
        w = j["wall_s"] + (c or {}).get("wall_s", 0.0) + (r or {}).get("wall_s", 0.0)
        tot += w
        L.append(f"| {sh} | {j['n_evals_used_total']} | "
                 f"{j['spent_forward_equivalents']:.1f} | {j['wall_s']:.0f} | "
                 f"{(c or {}).get('wall_s', float('nan')):.0f} | "
                 f"{(r or {}).get('wall_s', float('nan')):.0f} |")
    L.append(f"| TOTAL | | | | | {tot:.0f} s of process time |")
    return "\n".join(L)


TABLES = {"solve": t_solve, "control": t_control, "history": t_iou_vs_history,
          "gateA": t_gate_A, "gateB": t_gate_B, "gateBhist": t_gate_B_history,
          "cost": t_cost}


def main(which: str = "all") -> None:
    for name, fn in TABLES.items():
        if which not in ("all", name):
            continue
        print(f"\n### {name}\n")
        print(fn())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
