"""Markdown tables for the shape-library report, built from `out_lib/<shape>.json`.

Every number is read from a stored solve artifact; nothing is retyped.

Run: ./.venv312/bin/python -m adjoint2d.build_library_table <out_lib>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .library_solve import SHAPES

ARMS = ("U_uniform", "HIST_best", "A1_cont", "A1_4bpp", "A1_2bpp",
        "A15_cont", "A15_4bpp", "A15_2bpp")
ARM_LABEL = {
    "U_uniform": "uniform s = 1",
    "HIST_best": "best stored historical mask",
    "A1_cont": "solved [0, 1] continuous",
    "A1_4bpp": "**solved [0, 1] 4 bpp**",
    "A1_2bpp": "solved [0, 1] 2 bpp",
    "A15_cont": "solved [0, 1.5] continuous",
    "A15_4bpp": "solved [0, 1.5] 4 bpp",
    "A15_2bpp": "solved [0, 1.5] 2 bpp",
}


def load(outdir: Path) -> dict:
    res = {}
    for sh in SHAPES:
        f = outdir / f"{sh}.json"
        if f.exists():
            res[sh] = json.loads(f.read_text())
    return res


def census_table(res: dict) -> str:
    hdr = ("| shape | best stored mask, J | its IoU | solved 4 bpp, J | its IoU | "
           "J change | IoU change | beats on J | beats on IoU | class |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for sh, r in res.items():
        d, h = r["arms"]["A1_4bpp"], r["arms"]["HIST_best"]
        v = r["verdict"]
        hz = " (H)" if h["t_stop_at_horizon"] else ""
        hzd = " (H)" if d["t_stop_at_horizon"] else ""
        lines.append(
            f"| {sh} | {h['J']:.2f}{hz} | {h['IoU']:.4f} | {d['J']:.2f}{hzd} | {d['IoU']:.4f} | "
            f"{v['dJ_rel']*100:+.1f} % | {v['dIoU']:+.4f} | "
            f"{'YES' if v['beats_hist_on_J'] else 'no'} | "
            f"{'YES' if v['beats_hist_on_IoU'] else 'no'} | {v['class']} |")
    return hdr + "\n".join(lines) + "\n"


def full_table(res: dict) -> str:
    hdr = ("| shape | arm | J | J per part cell | IoU | growth % | under % | stop idx | stop s | "
           "horizon | phi_bar | P_abs W/m | energy residual % of dose | gate |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for sh, r in res.items():
        for a in ARMS:
            m = r["arms"].get(a)
            if m is None:
                continue
            g = m["energy_gate"]
            lines.append(
                f"| {sh} | {ARM_LABEL[a]} | {m['J']:.2f} | {m['J_per_part_cell']:.4f} | "
                f"{m['IoU']:.4f} | {m['bed_melt_pct_of_part']:.2f} | "
                f"{m['part_under_melt_pct']:.2f} | {m['t_stop_index']} | {m['t_stop_s']:.1f} | "
                f"{'YES' if m['t_stop_at_horizon'] else 'no'} | "
                f"{m['phi_bar_part_at_stop']:.3f} | {m['P_abs_W_per_m']:.1f} | "
                f"{g['rel_residual_at_index']*100:.2f} | "
                f"{'PASS' if g['PASS'] else '**FAIL**'} |")
    return hdr + "\n".join(lines) + "\n"


def hist_winner_table(res: dict) -> str:
    hdr = ("| shape | stored masks scanned | winning mask | campaign | convention | J | IoU |\n"
           "|---|---|---|---|---|---|---|\n")
    lines = []
    for sh, r in res.items():
        name = r["best_hist_arm"]
        m = r["arms"]["HIST_best"]
        camp = "old {0.30 .. 0.85} grid" if "_oldgrid_" in name else "calibration campaign"
        tag = name.replace("hist_", "").replace("_asstored_eps", "").replace("_outside1_eps", "")
        lines.append(f"| {sh} | {r['n_stored_masks_scanned']} | `{tag}` | {camp} | "
                     f"{m['convention']} | {m['J']:.2f} | {m['IoU']:.4f} |")
    return hdr + "\n".join(lines) + "\n"


def cost_table(res: dict) -> str:
    hdr = ("| shape | part cells | forward s | adjoint s | ratio | gradient evaluations "
           "inside 40 forward-equivalents | double pass run | wall s |\n"
           "|---|---|---|---|---|---|---|---|\n")
    lines = []
    for sh, r in res.items():
        c = r["cost"]
        lines.append(f"| {sh} | {r['n_part_cells']} | {c['forward_s']:.2f} | "
                     f"{c['adjoint_s']:.2f} | {c['ratio']:.3f} | {c['n_gradient_evals']} | "
                     f"{'yes' if r['double_pass_triggered'] else 'no'} | {r['wall_s']:.0f} |")
    return hdr + "\n".join(lines) + "\n"


def summary(res: dict) -> dict:
    n = len(res)
    win_J = [s for s, r in res.items() if r["verdict"]["beats_hist_on_J"]]
    win_I = [s for s, r in res.items() if r["verdict"]["beats_hist_on_IoU"]]
    solved = [s for s, r in res.items() if r["arms"]["A1_4bpp"]["IoU"] >= 0.95]
    viol = {s: r["energy_gate_violations"] for s, r in res.items() if r["energy_gate_violations"]}
    horizon = {s: [a for a, m in r["arms"].items() if m["t_stop_at_horizon"]]
               for s, r in res.items()}
    return {"n": n, "beats_on_J": win_J, "beats_on_IoU": win_I, "solved": solved,
            "energy_gate_violations": viol,
            "horizon_flags": {k: v for k, v in horizon.items() if v},
            "classes": {s: r["verdict"]["class"] for s, r in res.items()},
            "max_energy_residual": max(
                (m["energy_gate"]["rel_residual_at_index"]
                 for r in res.values() for m in r["arms"].values()), default=0.0),
            "max_clip": max((max(m["frac_dT_clipped_max"], m["frac_temp_cap_max"],
                                 m["frac_qrf_cap"])
                             for r in res.values() for m in r["arms"].values()), default=0.0)}


if __name__ == "__main__":
    out = Path(sys.argv[1]).resolve()
    r = load(out)
    print("## census\n"); print(census_table(r))
    print("## historical winners\n"); print(hist_winner_table(r))
    print("## full table\n"); print(full_table(r))
    print("## cost\n"); print(cost_table(r))
    print("## summary\n"); print(json.dumps(summary(r), indent=2, default=float))
