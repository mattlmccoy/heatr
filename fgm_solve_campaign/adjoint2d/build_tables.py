"""Assemble the head-to-head tables from the stored per-shape JSON artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SHAPES = ("square", "triangle", "cross")
BUDGETS = ("5", "15", "40")


def _f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _pct(v, ref):
    if v is None or ref is None:
        return "n/a"
    return f"{100.0 * (v - ref) / ref:+.1f} %"


def build(outdir: Path) -> str:
    lines: list[str] = []
    lines.append("### Table A. Cost, measured on this machine\n")
    lines.append("| shape | part cells | forward s | adjoint s, heating-peak fit | ratio | "
                 "adjoint s, melt-onset fit | ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for sh in SHAPES:
        d = json.loads((outdir / f"{sh}.json").read_text())
        c = d["cost"]
        lines.append(f"| {sh} | {d['n_part_cells']} | {_f(c['forward_s'], 2)} | "
                     f"{_f(c.get('adjoint_peak_s'), 2)} | {_f(c.get('ratio_peak'))} | "
                     f"{_f(c.get('adjoint_melt_s'), 2)} | {_f(c.get('ratio_melt'))} |")

    lines.append("\n### Table B. Hold-out melt-onset sigma_T, deg C, by budget\n")
    lines.append("| shape | budget | uniform | H1 sigma-only | H1eps co-varying | "
                 "H1eps0 step-2 boundary | A1 adjoint, hold-out-fair | A2 adjoint, melt-onset fit |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sh in SHAPES:
        d = json.loads((outdir / f"{sh}.json").read_text())
        de = json.loads((outdir / f"{sh}_H1eps.json").read_text())
        dz = json.loads((outdir / f"{sh}_H1eps0.json").read_text())
        u = d["uniform"]["holdout"]
        for B in BUDGETS:
            b = d["budgets"][B]
            be = de["budgets"][B]
            bz = dz["budgets"][B]

            def g(sel, key="holdout"):
                return None if sel is None else sel.get(key)
            lines.append(
                f"| {sh} | {B} | {_f(u)} | {_f(g(b['H1_peak']))} | {_f(g(be['H1eps_peak']))} | "
                f"{_f(g(bz['H1eps0_peak']))} | {_f(g(b['A1']))} | {_f(g(b['A2']))} |")

    lines.append("\n### Table C. Percent versus uniform at the hold-out read state "
                 "(negative is better)\n")
    lines.append("| shape | budget | H1 sigma-only | H1eps co-varying | H1eps0 step-2 boundary "
                 "| A1 | A2 |")
    lines.append("|---|---|---|---|---|---|---|")
    for sh in SHAPES:
        d = json.loads((outdir / f"{sh}.json").read_text())
        de = json.loads((outdir / f"{sh}_H1eps.json").read_text())
        dz = json.loads((outdir / f"{sh}_H1eps0.json").read_text())
        u = d["uniform"]["holdout"]
        for B in BUDGETS:
            b, be, bz = d["budgets"][B], de["budgets"][B], dz["budgets"][B]

            def g(sel):
                return None if sel is None else sel.get("holdout")
            lines.append(f"| {sh} | {B} | {_pct(g(b['H1_peak']), u)} | {_pct(g(be['H1eps_peak']), u)} | "
                         f"{_pct(g(bz['H1eps0_peak']), u)} | {_pct(g(b['A1']), u)} | {_pct(g(b['A2']), u)} |")

    lines.append("\n### Table D. Absorbed power at the melt-onset read state, W per metre depth\n")
    lines.append("| shape | uniform | H1 sigma-only (B=40) | H1eps (B=40) | A1 (B=40) | A2 (B=40) |")
    lines.append("|---|---|---|---|---|---|")
    for sh in SHAPES:
        d = json.loads((outdir / f"{sh}.json").read_text())
        de = json.loads((outdir / f"{sh}_H1eps.json").read_text())
        b, be = d["budgets"]["40"], de["budgets"]["40"]

        def p(sel):
            return None if sel is None else sel.get("P_abs_W_per_m")
        lines.append(f"| {sh} | {_f(d['uniform']['P_abs_W_per_m'], 1)} | {_f(p(b['H1_peak']), 1)} | "
                     f"{_f(p(be['H1eps_peak']), 1)} | {_f(p(b['A1']), 1)} | {_f(p(b['A2']), 1)} |")

    lines.append("\n### Table E. Standing gates on the selected arms (B = 40)\n")
    lines.append("| shape | arm | temperature-step clip fraction | Q_rf cap fraction | "
                 "melt-onset step | status |")
    lines.append("|---|---|---|---|---|---|")
    for sh in SHAPES:
        d = json.loads((outdir / f"{sh}.json").read_text())
        de = json.loads((outdir / f"{sh}_H1eps.json").read_text())
        arms = [("uniform", d["uniform"]),
                ("H1", d["budgets"]["40"]["H1_peak"]),
                ("H1eps", de["budgets"]["40"]["H1eps_peak"]),
                ("A1", d["budgets"]["40"]["A1"]),
                ("A2", d["budgets"]["40"]["A2"])]
        for name, sel in arms:
            if sel is None:
                lines.append(f"| {sh} | {name} | n/a | n/a | n/a | NOT_REACHED |")
                continue
            lines.append(f"| {sh} | {name} | {sel.get('frac_dT_clipped_max', 0.0):.4f} | "
                         f"{sel.get('frac_qrf_cap', 0.0):.4f} | {sel.get('melt_onset_index')} | "
                         f"{sel.get('status')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    txt = build(out)
    (out / "tables.md").write_text(txt)
    print(txt)
