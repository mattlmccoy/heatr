#!/usr/bin/env python3
"""Analyze multi-part interaction runs vs isolated single-part baseline."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "outputs_eqs" / "runs" / "circle" / "multipart_interaction"


def load(run: str, family: str) -> dict:
    p = RUN_ROOT / family / run / "summary.json"
    return json.loads(p.read_text())


def agg(s: dict) -> dict:
    ps = s["part_stats"]
    qrf = [p["mean_qrf_w_per_m3"] for p in ps]
    tmax = [p["max_T_c"] for p in ps]
    tmean = [p["mean_T_c"] for p in ps]
    rho = [p["mean_rho_rel"] for p in ps]
    return {
        "n": len(ps),
        "qrf_mean": sum(qrf) / len(qrf),
        "qrf_min": min(qrf),
        "qrf_max": max(qrf),
        "tmax_max": max(tmax),
        "tmean_mean": sum(tmean) / len(tmean),
        "rho_mean": sum(rho) / len(rho),
        "ui_rms": s.get("ui_rms_part_final", 0.0),
        "melt_frac": s.get("frac_part_ge_melt_ref", 0.0),
        "inter_rho_std": s.get("inter_part_mean_rho_std", 0.0),
        "ebal": s.get("energy_balance_residual_final_J_per_m", 0.0),
        "eds_tot": s.get("energy_doped_total_J_per_m", 1.0),
        "dtclip": s.get("frac_cells_dT_clipped_final", 0.0),
        "ps": ps,
    }


def main() -> None:
    base = agg(load("single_baseline", "baseline"))
    q0 = base["qrf_mean"]
    print(f"BASELINE (isolated 12mm circle): mean_Qrf={q0:.3e}  Tmax={base['tmax_max']:.1f}C  "
          f"rho={base['rho_mean']:.4f}  meltfrac={base['melt_frac']:.3f}  ui_rms={base['ui_rms']:.4f}")
    print(f"  energy_balance_resid/doped = {base['ebal']/base['eds_tot']*100:.3f}%  dTclip={base['dtclip']:.4f}\n")

    spacings = [14, 18, 24, 32]
    print("PAIR SWEEP (Qrf normalized to isolated baseline; ratio>1 = field enhancement, <1 = shielding)")
    hdr = f"{'arrange':8} {'s_mm':>5} {'gap_mm':>6} {'Qrf/Q0':>7} {'Tmax_C':>7} {'rho':>6} {'meltfr':>6} {'ui_rms':>7} {'dTclip':>7}"
    print(hdr)
    rows = []
    for arr in ("inline", "side"):
        for s in spacings:
            a = agg(load(f"pair_{arr}_s{s}mm", "experimental"))
            ratio = a["qrf_mean"] / q0
            gap = s - 12
            print(f"{arr:8} {s:5d} {gap:6d} {ratio:7.3f} {a['tmax_max']:7.1f} "
                  f"{a['rho_mean']:6.4f} {a['melt_frac']:6.3f} {a['ui_rms']:7.4f} {a['dtclip']:7.4f}")
            rows.append((arr, s, gap, ratio, a))
        print()

    print("MULTI-PART CLUSTERS at 18 mm pitch (per-part Qrf relative to baseline):")
    for run in ("triple_side_s18mm", "triple_inline_s18mm", "quad_cluster_s18mm"):
        a = agg(load(run, "experimental"))
        per = [p["mean_qrf_w_per_m3"] / q0 for p in a["ps"]]
        per_t = [p["max_T_c"] for p in a["ps"]]
        print(f"  {run:22} n={a['n']}  Qrf/Q0 per part = [{', '.join(f'{v:.3f}' for v in per)}]  "
              f"Tmax per part = [{', '.join(f'{v:.0f}' for v in per_t)}]C")
        print(f"  {'':22} mean rho={a['rho_mean']:.4f} meltfrac={a['melt_frac']:.3f} "
              f"inter_part_rho_std={a['inter_rho_std']:.4f} dTclip={a['dtclip']:.4f}")


if __name__ == "__main__":
    main()
