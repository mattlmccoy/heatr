#!/usr/bin/env python3
"""Scan ALL iterations of an fgm_iterate run and report the best gate-valid iter.

The convergence.json best_iter scorer can pick a late diverged iteration when the
optimizer selects a long max_density t* (σ_T rises again). This evaluator instead
scans every iterN/summary.json, applies the validity gate
(melt>=0.95, maxT<250, eres<1.5%, dtclip~0), and among gate-passing iters returns
the one with the lowest ui_rms (best uniformity). If none pass, returns the
lowest-maxT iter for diagnosis.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_eqs"


def _eres_pct(s: dict) -> float | None:
    e = s.get("energy_balance_residual_final_J_per_m")
    st = s.get("energy_stored_domain_final_J_per_m") or s.get("energy_stored_part_final_J_per_m")
    if isinstance(e, (int, float)) and isinstance(st, (int, float)) and st:
        return abs(e) / abs(st) * 100.0
    return None


def main() -> None:
    rel = sys.argv[1]
    parent = OUT / rel
    name = parent.name
    rows = []
    for d in sorted(parent.glob(f"{name}_iter*")):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            it = int(d.name.rsplit("iter", 1)[1])
        except ValueError:
            continue
        s = json.loads(sp.read_text())
        maxT = s.get("max_T_part_final_c")
        melt = s.get("frac_part_ge_melt_ref")
        ui = s.get("ui_rms_part_final")
        rho = s.get("mean_rho_rel_part_final")
        dt = s.get("frac_cells_dT_clipped_final")
        ep = _eres_pct(s)
        g = (isinstance(melt, (int, float)) and melt >= 0.95
             and isinstance(maxT, (int, float)) and maxT < 250.0
             and (ep is None or ep < 1.5)
             and isinstance(dt, (int, float)) and dt <= 1e-4)
        rows.append({"iter": it, "maxT": maxT, "melt": melt, "ui_rms": ui,
                     "rho": rho, "eres_pct": ep, "dtclip": dt, "gate": g})
    if not rows:
        print(json.dumps({"rel": rel, "error": "no iter summaries"}))
        return
    passing = [r for r in rows if r["gate"]]
    if passing:
        best = min(passing, key=lambda r: r["ui_rms"])
        verdict = "PASS"
    else:
        best = min(rows, key=lambda r: (r["maxT"] if r["maxT"] else 1e9))
        verdict = "FAIL"
    conv = json.loads((parent / "convergence.json").read_text())
    print(json.dumps({
        "rel": rel,
        "tstar_min": conv.get("optimizer_chosen_time_min") or conv.get("exposure_minutes"),
        "opt_criterion": conv.get("optimizer_criterion"),
        "VERDICT": verdict,
        "best_valid_iter": best["iter"],
        "maxT_C": best["maxT"], "melt": best["melt"], "ui_rms": best["ui_rms"],
        "rho_bar": best["rho"], "eres_pct": best["eres_pct"], "dtclip": best["dtclip"],
        "n_passing_iters": len(passing),
    }, indent=2))


if __name__ == "__main__":
    main()
