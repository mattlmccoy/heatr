#!/usr/bin/env python3
"""Assemble outputs_eqs/dose_control_2d/DOSE_CONTROL_2D.md from run artifacts.

Every number in the report is read from a per-shape results JSON, which is
itself assembled only from summary.json / time_series.json / fields.npz of the
individual solves. Nothing is transcribed from prose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs_eqs" / "dose_control_2d"
GRID = 160
RES_DIR = OUT / f"results_g{GRID}"

ORDER = [
    "circle", "hexagon", "pentagon", "ellipse", "octagon",
    "star6", "diamond", "triangle", "equilateral_triangle", "square",
]


def _f(v, spec="{:.2f}", na="NOT REACHED"):
    return na if v is None else spec.format(v)


def load(grid: int = GRID) -> dict:
    res = {}
    res_dir = OUT / f"results_g{grid}"
    for s in ORDER:
        p = res_dir / f"{s}.json"
        if p.exists():
            res[s] = json.loads(p.read_text())
    return res


def design_table_md(res: dict) -> str:
    hdr = ("| shape | role | scale | w x h [mm] | realized A [mm^2] | area err [%] "
           "| gap x [mm] | gap y [mm] | thin feature [mm] (cells) | cells across x/y | part cells |")
    sep = "|---" * 11 + "|"
    rows = [hdr, sep]
    for s in ORDER:
        if s not in res:
            continue
        d = res[s].get("design", res[s].get("design_240"))
        rows.append(
            f"| {s} | {res[s]['role']} | {d['scale']:.4f} | "
            f"{d['new_w_mm']:.3f} x {d['new_h_mm']:.3f} | {d['new_A_mm2']:.2f} | "
            f"{d['err_pct']:+.3f} | {d['gap_mm']:.2f} | {d['gap_y_mm']:.2f} | "
            f"{d['thin_mm']:.4f} ({d['thin_cells']:.0f}) | "
            f"{d['cells_across_x']}/{d['cells_across_y']} | {d['n_cells']} |"
        )
    return "\n".join(rows)


def results_table_md(res: dict) -> str:
    hdr = ("| shape | role | V [V] | sat_B mean (range) | sigma_T A [C] | sigma_T B [C] "
           "| sigma_T C [C] | total A->B [%] | grading-only C->B [%] | verdict |")
    sep = "|---" * 10 + "|"
    rows = [hdr, sep]
    for s in ORDER:
        if s not in res:
            continue
        r = res[s]
        sb = r["arms"]["B"]["sat"]
        rows.append(
            f"| {s} | {r['role']} | {r['voltage_v']:.0f} | "
            f"{sb['mean']:.3f} ({sb['min']:.3f}-{sb['max']:.3f}) | "
            f"{_f(r['sigma_T_A_c'])} | {_f(r['sigma_T_B_c'])} | {_f(r['sigma_T_C_c'])} | "
            f"{_f(r['total_benefit_pct'], '{:+.1f}')} | "
            f"**{_f(r['grading_only_benefit_pct'], '{:+.1f}')}** | {r['verdict']} |"
        )
    return "\n".join(rows)


def transient_table_md(res: dict) -> str:
    hdr = ("| shape | arm | t_phi90 [s] | sigma_T @phi90 [C] | T_max @phi90 [C] "
           "| peak T over run [C] | P_abs [W/m] | dP vs A [%] | >20% flag | "
           "mean sigma [S/m] | rho_bar final | melt frac final |")
    sep = "|---" * 12 + "|"
    rows = [hdr, sep]
    for s in ORDER:
        if s not in res:
            continue
        for a in ("A", "B", "C"):
            d = res[s]["arms"][a]
            p = d["at_phi90"]
            rows.append(
                f"| {s} | {a} | {_f(p['t_s'], '{:.1f}')} | {_f(p['sigma_T_c'])} | "
                f"{_f(p['max_T_c'], '{:.1f}')} | {d['peak_T_over_run_c']:.1f} | "
                f"{d['P_abs_W_per_m']:.1f} | {d['P_dev_pct_vs_A']:+.1f} | "
                f"{'FLAG' if d['P_flag_gt20pct'] else '-'} | "
                f"{d['sigma_mean_S_per_m']:.4f} | {d['mean_rho_rel_part_final']:.3f} | "
                f"{d['frac_part_ge_melt_ref_final']:.3f} |"
            )
    return "\n".join(rows)


def gate_g2_table_md(res: dict) -> str:
    hdr = ("| shape | arm | frac part at Qrf cap | Qrf part max [W/m^3] | cap [W/m^3] "
           "| frac cells dT clipped (final / mean) | energy residual [J/m] | residual [% of dose] "
           "| max dT_raw [C] | max phi jump/step | G2a |")
    sep = "|---" * 11 + "|"
    rows = [hdr, sep]
    for s in ORDER:
        if s not in res:
            continue
        for a in ("A", "B", "C"):
            d = res[s]["arms"][a]
            rows.append(
                f"| {s} | {a} | {d['frac_part_at_qrf_cap_final']:.4f} | "
                f"{d['qrf_part_max_w_per_m3']:.3e} | {d['max_qrf_cap_w_per_m3']:.1e} | "
                f"{d['frac_cells_dT_clipped_final']:.4f} / {d['frac_cells_dT_clipped_mean']:.4f} | "
                f"{d['energy_balance_residual_J_per_m']:.1f} | "
                f"{d['energy_residual_pct_of_dose']:.2f} | "
                + (lambda g: f"{(g['max_dT_raw_c'] or 0.0):.2f} | "
                             f"{g['max_phi_jump_per_step']:.4f} | "
                             f"{'PASS' if g['passed'] else 'FAIL ' + ','.join(g['failures'])} |"
                   )(d["stability_gate"])
            )
    return "\n".join(rows)


def verdict_block(res: dict) -> str:
    tests = [res[s] for s in ORDER if s in res and res[s]["role"] == "test"]
    surv = [r for r in tests if r["grading_only_benefit_pct"] is not None
            and r["grading_only_benefit_pct"] <= -10.0]
    strong = [r for r in surv if r["grading_only_benefit_pct"] <= -25.0]
    marg = [r for r in tests if r["grading_only_benefit_pct"] is not None
            and -10.0 < r["grading_only_benefit_pct"] < 0.0]
    harm = [r for r in tests if r["grading_only_benefit_pct"] is not None
            and r["grading_only_benefit_pct"] >= 0.0]
    nr = [r for r in tests if r["grading_only_benefit_pct"] is None]
    tot_ok = [r for r in tests if r["total_benefit_pct"] is not None
              and r["total_benefit_pct"] <= -10.0]
    g_vals = [abs(r["grading_only_benefit_pct"]) for r in tests
              if r["grading_only_benefit_pct"] is not None]
    t_vals = [abs(r["total_benefit_pct"]) for r in tests
              if r["total_benefit_pct"] is not None]
    med_g = median(g_vals) if g_vals else float("nan")
    med_t = median(t_vals) if t_vals else float("nan")

    lines = [
        f"- test shapes evaluated: **{len(tests)}** of 9",
        f"- SURVIVE (grading-only <= -10 %): **{len(surv)}** "
        f"({', '.join(r['shape'] for r in surv) or 'none'})",
        f"- of those, STRONG (<= -25 %): {len(strong)} "
        f"({', '.join(r['shape'] for r in strong) or 'none'})",
        f"- MARGINAL (-10 % < b < 0 %): {len(marg)} "
        f"({', '.join(r['shape'] for r in marg) or 'none'})",
        f"- GRADING HARMFUL at matched dose (>= 0 %): {len(harm)} "
        f"({', '.join(r['shape'] for r in harm) or 'none'})",
        f"- NOT REACHED (excluded): {len(nr)} "
        f"({', '.join(r['shape'] for r in nr) or 'none'})",
        f"- clear -10 % on TOTAL benefit (A->B): {len(tot_ok)} "
        f"({', '.join(r['shape'] for r in tot_ok) or 'none'})",
        f"- median |grading-only| = {med_g:.1f} % ; median |total| = {med_t:.1f} % ; "
        f"ratio = {(med_g/med_t if med_t else float('nan')):.2f}",
    ]
    pure_dose = [r["shape"] for r in tests
                 if r["total_benefit_pct"] is not None
                 and r["total_benefit_pct"] <= -10.0
                 and r["grading_only_benefit_pct"] is not None
                 and r["grading_only_benefit_pct"] >= 0.0]
    lines.append(f"- pure dose effect (total <= -10 % but grading-only >= 0 %): "
                 f"{', '.join(pure_dose) or 'none'}")

    # pre-registered classification
    ratio = (med_g / med_t) if med_t else float("nan")
    if len(surv) < 3 and len(tot_ok) >= 6:
        cls = "CONFOUND PRESENT, SEVERE"
    elif len(surv) >= 6 and ratio >= 0.70:
        cls = "CONFOUND ABSENT"
    elif ratio < 0.50:
        cls = "CONFOUND PRESENT, PARTIAL"
    else:
        cls = ("INTERMEDIATE: does not match any pre-registered class cleanly "
               "(see the paragraph below)")
    lines.append(f"- **pre-registered classification: {cls}**")
    return "\n".join(lines)


def main() -> None:
    res = load()
    gates = json.loads((OUT / "gates.json").read_text())
    pre = (OUT / "PREREGISTRATION.md").read_text()

    md = []
    md.append("# Dose-controlled three-arm 2-D grading study, ten cross-sections\n")
    md.append(f"Generated from run artifacts by `scripts/analysis/dose_control_2d_report.py`. "
              f"Shapes present: {len(res)} of 10.\n")
    md.append("## 0. Pre-registration (verbatim, written before any arm solve)\n")
    md.append("```\n" + pre + "\n```\n")
    md.append("## 1. Gate results\n")
    md.append(gates["g1_text"] + "\n")
    md.append(gates["g3_text"] + "\n")
    md.append("### G2: clip and energy residual on every solve\n")
    md.append(gate_g2_table_md(res) + "\n")
    md.append("## 2. Per-shape design table, 240 x 240 grid, 0.2510 mm cells\n")
    md.append(design_table_md(res) + "\n")
    md.append("## 3. Three-arm results\n")
    md.append(results_table_md(res) + "\n")
    md.append("### Per-arm transient detail\n")
    md.append(transient_table_md(res) + "\n")
    md.append("## 4. Verdict against the pre-registered criteria\n")
    md.append(verdict_block(res) + "\n")

    (OUT / "DOSE_CONTROL_2D_autotables.md").write_text("\n".join(md))
    print(f"wrote {OUT / 'DOSE_CONTROL_2D_autotables.md'}")
    print(verdict_block(res))


if __name__ == "__main__":
    main()
