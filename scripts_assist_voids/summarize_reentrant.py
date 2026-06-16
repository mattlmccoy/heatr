#!/usr/bin/env python3
"""Build the reentrant-SRAF verdict summary + comparison figure."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_eqs" / "runs" / "reentrant" / "assist_voids"
R = json.load(open(OUT / "reentrant_results.json"))


def delta_table(shape):
    s = R[shape]; b = s["baseline"]
    rows = []
    for k, v in s.items():
        if k in ("baseline", "_hotcore_center_m") or "MHz" in k:
            continue
        rows.append({
            "variant": k,
            "dSigT_int_c": v["sigma_T_interior_c"] - b["sigma_T_interior_c"],
            "dUiRms": v["ui_rms_part"] - b["ui_rms_part"],
            "dIoU": v["iou"]["iou"] - b["iou"]["iou"],
            "dMeanRho": v["mean_rho_part"] - b["mean_rho_part"],
            "Tmax_c": v["max_T_part_c"],
            "void_mean_rho": v.get("void_mean_rho"),
            "void_min_rho": v.get("void_min_rho"),
            "void_mean_T_c": v.get("void_mean_T_c"),
        })
    return b, rows


def main():
    verdict = {
        "pilot": "SRAF assist-voids — REENTRANT / over-hot-interior regime",
        "solver": "HEATR 2.5D rfam_eqs_coupled.py",
        "operating_point": "n_steps=720 (6 min), 27.12 MHz, interior over-melts "
                           "(cross core meanT 194C, L core 181C; Tmax<250 ceiling)",
        "regime_confirmed": "cross +9.4C, L_shape +11.6C interior-minus-edge meanT "
                            "(probe_interior_overheat.json) -> interior IS the hot/error zone",
        "numerical_gates": {"dT_clip": "0 all runs",
                            "energy_residual_J_per_m": "1800-3800 (small vs ~1e5 stored)",
                            "phi90_local": "interior locally >melt; global phi_bar~0.65-0.84"},
        "by_shape": {},
    }
    for shape in ("cross", "L_shape"):
        b, rows = delta_table(shape)
        verdict["by_shape"][shape] = {
            "baseline": {"sigma_T_interior_c": b["sigma_T_interior_c"],
                         "ui_rms_part": b["ui_rms_part"], "IoU": b["iou"]["iou"],
                         "mean_rho_part": b["mean_rho_part"], "Tmax_c": b["max_T_part_c"]},
            "variants": rows,
        }
    # frequency cross-check (cross)
    c = R["cross"]
    verdict["wavelength_check_cross_d2mm"] = {
        f: {"void_mean_rho": c[k]["void_mean_rho"], "void_mean_T_c": c[k]["void_mean_T_c"],
            "IoU": c[k]["iou"]["iou"], "Tmax_c": c[k]["max_T_part_c"],
            "sigma_T_part_c": c[k]["sigma_T_part_c"]}
        for f, k in [("27.12MHz", "void_d2mm_core"),
                     ("13.56MHz", "void_d2mm_core_13p56MHz"),
                     ("40.68MHz", "void_d2mm_core_40p68MHz")]
    }
    verdict["verdict"] = (
        "MIXED -> SHAPE-DEPENDENT, and a REAL lever on ASYMMETRIC reentrant parts. "
        "L_shape: d2mm core void cuts interior sigma_T 40.9->37.9C (-3.0C), part ui_rms "
        "0.270->0.254 (-0.016), IoU +0.011, Tmax 233->221C, and the void OVER-CLOSES "
        "(void_rho 0.862 > part-mean 0.692; min 0.713) because it sits in the hottest "
        "spot (voidT 214C). Matched FGM core-knockdown does the OPPOSITE on the L "
        "(sigma_T +2.6C, ui_rms +0.015, Tmax +7C) -- uniform knockdown starves the "
        "asymmetric core. So here SRAF BEATS FGM. "
        "Cross (symmetric): hot junction == centroid, so void removal has near-zero "
        "shape authority (dIoU -0.007, dSigT -0.4C); FGM wins (dIoU +0.035). Voids still "
        "close (void_rho ~0.71-0.73). "
        "Closure-vs-authority tension from the square pilot is RESOLVED on the L (closure "
        "AND authority coincide in the thick leg) but PERSISTS on the symmetric cross."
    )
    verdict["wavelength_independence"] = (
        "WEAKER than the convex-square pilot. Total absorbed power is renormalized to 500W "
        "at all freqs, but the Qrf SHAPE shifts: maxQrf 8.7e7 (13.56) vs 5.1e7 (27.12) vs "
        "3.9e7 (40.68) W/m3 -> 13.56MHz over-peaks (Tmax 279C, void undercloses rho 0.589). "
        "Driven by the complex EQS admittance sigma+j*omega*eps0*eps_r interacting with the "
        "eps_r=2 virgin surroundings (displacement-current-only paths). Still quasi-static "
        "(part ~1000x<lambda), no resonance -- but NOT frequency-invariant in this regime."
    )
    verdict["escalation_recommendation"] = (
        "YES, escalate to inverse-design (adjoint over the void pattern) -- but ONLY for "
        "ASYMMETRIC reentrant / thick-section parts (L_shape class), where SRAF both heals "
        "and corrects shape and BEATS uniform FGM. The hand-placed 5-void cluster already "
        "recovers -3C sigma_T and +0.011 IoU with over-closure; an adjoint over void "
        "position/size/count should compound this. Do NOT pursue SRAF on symmetric/convex "
        "parts (square, cross) -- FGM dominates there. Best next experiment: adjoint void "
        "layout on L_shape + run it head-to-head against integral-FGM (magnitude 0.8, bpp4) "
        "at matched manufacturability, and fix the frequency at the hardware band before "
        "optimizing (13.56 vs 27.12 changes the Qrf shape materially)."
    )
    (OUT / "REENTRANT_SUMMARY.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))

    # ---- comparison figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, shape in enumerate(("cross", "L_shape")):
        b, rows = delta_table(shape)
        labels = [r["variant"].replace("_core", "").replace("void_", "v").replace("fgm_core_kd0.5", "FGM") for r in rows]
        x = np.arange(len(rows))
        ax = axes[col, 0]
        ax.bar(x, [r["dSigT_int_c"] for r in rows],
               color=["#3a7" if r["dSigT_int_c"] < 0 else "#c44" for r in rows])
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{shape}: d sigma_T interior (C)  [neg=better]"); ax.set_ylabel("delta C")
        ax = axes[col, 1]
        ax.bar(x, [r["dIoU"] for r in rows],
               color=["#3a7" if r["dIoU"] > 0 else "#c44" for r in rows])
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{shape}: d melt-IoU vs nominal  [pos=better]")
        ax = axes[col, 2]
        vr = [r["void_mean_rho"] for r in rows]
        ax.bar(x, [(v if v else np.nan) for v in vr], color="#48c")
        ax.axhline(b["mean_rho_part"], color="k", ls="--", lw=0.8, label=f"part-mean {b['mean_rho_part']:.2f}")
        ax.axhline(0.55, color="gray", ls=":", lw=0.8, label="initial 0.55")
        ax.set_ylim(0.5, 1.0); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{shape}: void closure (mean rho)"); ax.legend(fontsize=7)
    fig.suptitle("SRAF assist-voids — reentrant/over-hot-interior regime (n=720, 27.12 MHz)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "reentrant_comparison.png", dpi=130)
    print("wrote", OUT / "reentrant_comparison.png")


if __name__ == "__main__":
    main()
