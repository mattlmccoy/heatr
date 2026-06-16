#!/usr/bin/env python3
"""Extract gate metrics for one finished fgm_iterate run (P1 campaign).

Reads convergence.json (best_iter, optimizer t*, per-iter sigma_T/mean_rho),
then the best iteration's summary.json for maxT / melt / ui_rms / energy
residual / dT-clip, and computes PASS/FAIL on the validity gate.

Gate: melt >= 0.95 AND maxT < 250 C AND energy_residual < ~1% of stored
      AND dT-clip ~= 0.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_eqs"


def _f(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def main() -> None:
    rel = sys.argv[1]  # e.g. runs/ellipse/fgm_iterate/<name>
    parent = OUT / rel
    conv_p = parent / "convergence.json"
    if not conv_p.exists():
        print(f"NO convergence.json at {parent}")
        sys.exit(2)
    conv = json.loads(conv_p.read_text())

    iters = conv.get("iterations", [])
    best_iter = conv.get("best_iter")
    tstar_min = _f(conv, "optimizer_chosen_time_min", "exposure_minutes")
    opt_reason = _f(conv, "optimizer_reason")
    opt_crit = _f(conv, "optimizer_criterion")
    iter0_sigma = next((it.get("sigma_T") for it in iters
                        if str(it.get("iter")) == "0"), None)

    # find best-iter entry + its output dir
    best_entry = next((it for it in iters
                       if str(it.get("iter")) == str(best_iter)), None)
    best_dir = None
    if best_entry and best_entry.get("output_dir"):
        best_dir = OUT / best_entry["output_dir"]

    summ = {}
    if best_dir and (best_dir / "summary.json").exists():
        summ = json.loads((best_dir / "summary.json").read_text())

    # Prefer summary.json (authoritative); convergence per-iter as fallback.
    maxT = _f(summ, "max_T_part_final_c") or (best_entry.get("T_max_c") if best_entry else None)
    p95T = _f(summ, "p95_T_part_final_c") or (best_entry.get("T_p95_c") if best_entry else None)
    melt = _f(summ, "frac_part_ge_melt_ref")
    if melt is None and best_entry:
        melt = best_entry.get("frac_melt")
    rho = _f(summ, "mean_rho_rel_part_final") or (best_entry.get("mean_rho") if best_entry else None)
    ui = _f(summ, "ui_rms_part_final")
    eres = _f(summ, "energy_balance_residual_final_J_per_m")
    estored = _f(summ, "energy_stored_domain_final_J_per_m",
                 "energy_stored_part_final_J_per_m")
    dtclip = _f(summ, "frac_cells_dT_clipped_final")
    best_sigma = _f(conv, "best_sigma_T")
    min_sigma = _f(conv, "min_sigma_T")

    # energy residual % of stored (fall back: leave None)
    eres_pct = None
    if isinstance(eres, (int, float)) and isinstance(estored, (int, float)) and estored:
        eres_pct = abs(eres) / abs(estored) * 100.0

    # GATE
    g_melt = isinstance(melt, (int, float)) and melt >= 0.95
    g_maxT = isinstance(maxT, (int, float)) and maxT < 250.0
    g_eres = (eres_pct is None) or (eres_pct < 1.5)  # ~1%, small slack
    g_dtclip = isinstance(dtclip, (int, float)) and dtclip <= 1e-4
    passed = g_melt and g_maxT and g_eres and g_dtclip

    print(json.dumps({
        "rel": rel,
        "tstar_min": tstar_min,
        "opt_criterion": opt_crit,
        "opt_reason": opt_reason,
        "best_iter": best_iter,
        "iter0_sigma_T": iter0_sigma,
        "best_sigma_T": best_sigma,
        "min_sigma_T": min_sigma,
        "maxT_C": maxT, "p95T_C": p95T,
        "melt": melt, "rho_bar": rho, "ui_rms": ui,
        "eres_J_per_m": eres, "estored_J_per_m": estored,
        "eres_pct_stored": eres_pct,
        "dT_clip": dtclip,
        "gate": {"melt>=0.95": g_melt, "maxT<250": g_maxT,
                 "eres<1.5%": g_eres, "dtclip~0": g_dtclip},
        "PASS": passed,
        "best_dir": best_entry.get("output_dir") if best_entry else None,
    }, indent=2))


if __name__ == "__main__":
    main()
