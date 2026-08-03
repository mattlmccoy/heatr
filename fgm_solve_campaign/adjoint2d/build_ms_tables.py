"""Census tables for the multi-start pass. Reads only, computes nothing new."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_MS = Path(__file__).resolve().parents[1] / "out_ms"
OUT_LIB = Path(__file__).resolve().parents[1] / "out_lib"
SHAPES = ("square", "circle", "hexagon", "triangle", "equilateral_triangle",
          "L_shape", "H_shape", "T_shape", "cross", "diamond", "ellipse",
          "octagon", "pentagon", "rectangle", "rounded_rect", "star", "star6",
          "trapezoid")


def load(shape: str) -> dict:
    return json.loads((OUT_MS / f"{shape}.json").read_text())


def rows() -> list[dict]:
    out = []
    for sh in SHAPES:
        j = load(sh)
        d = j["arms"]["MS_4bpp"]
        u = j["arms"]["U_uniform"]
        ref = j["library_reference"]
        h = ref["HIST_best"]
        a1 = ref["A1_4bpp"]
        out.append({
            "shape": sh, "winner": j["winner_start"],
            "J": d["J"], "IoU": d["IoU"], "grow": d["bed_melt_pct_of_part"],
            "under": d["part_under_melt_pct"], "rho": d["mean_rho_rel_part_at_stop"],
            "P": d["P_abs_W_per_m"], "t_stop": d["t_stop_s"],
            "horizon": d["t_stop_at_horizon"], "rough": d["map_roughness"],
            "Eres": d["energy_gate"]["rel_residual_at_index"] * 100.0,
            "J_lib": a1["J"], "IoU_lib": a1["IoU"], "P_lib": a1["P_abs_W_per_m"],
            "J_hist": h["J"], "IoU_hist": h["IoU"], "P_hist": h["P_abs_W_per_m"],
            "J_unif": u["J"], "IoU_unif": u["IoU"],
            "rough_lib": None, "class": j["verdict"]["class"],
            "class_lib": ref["_library_verdict"]["class"],
            "n_evals": j["n_evals_used_total"], "fe": j["spent_forward_equivalents"],
            "wall": j["wall_s"], "starts": j["n_starts"],
            "survivors": j["probe"]["survivors"],
            "probe_best": j["probe"]["best_J"],
            "uniform_repro": j["uniform_reproduction_check"]["abs_diff"],
            "n_levels": d.get("census_n_levels_used"),
            "gate_viol": j["energy_gate_violations"],
        })
    return out


def main() -> None:
    rs = rows()
    hdr = (f"{'shape':22s}{'win':>6s}{'J_ms':>9s}{'J_lib':>9s}{'J_hist':>9s}"
           f"{'J_unif':>9s}{'IoU_ms':>8s}{'IoU_lib':>8s}{'IoU_h':>8s}"
           f"{'grow':>7s}{'under':>7s}{'rho':>7s}{'P':>7s}{'lvl':>4s}{'wall':>6s}")
    print(hdr)
    for r in rs:
        print(f"{r['shape']:22s}{r['winner']:>6s}{r['J']:9.2f}{r['J_lib']:9.2f}"
              f"{r['J_hist']:9.2f}{r['J_unif']:9.2f}{r['IoU']:8.4f}{r['IoU_lib']:8.4f}"
              f"{r['IoU_hist']:8.4f}{r['grow']:7.2f}{r['under']:7.2f}{r['rho']:7.4f}"
              f"{r['P']:7.1f}{r['n_levels']:4d}{r['wall']:6.0f}"
              + ("  HORIZON" if r["horizon"] else ""))

    def cnt(f):
        return sum(1 for r in rs if f(r))
    print()
    print(f"MS beats HIST oracle on J        {cnt(lambda r: r['J'] < r['J_hist'])} of 18")
    print(f"MS beats HIST oracle on IoU      {cnt(lambda r: r['IoU'] > r['IoU_hist'])} of 18")
    print(f"LIB beats HIST oracle on J       {cnt(lambda r: r['J_lib'] < r['J_hist'])} of 18")
    print(f"LIB beats HIST oracle on IoU     {cnt(lambda r: r['IoU_lib'] > r['IoU_hist'])} of 18")
    print(f"MS beats LIB single-start on J   {cnt(lambda r: r['J'] < r['J_lib'])} of 18")
    print(f"MS beats LIB single-start on IoU {cnt(lambda r: r['IoU'] > r['IoU_lib'])} of 18")
    print(f"MS beats uniform on J            {cnt(lambda r: r['J'] < r['J_unif'])} of 18")
    print(f"MS in the SOLVED class (IoU>=.95, grid 120) "
          f"{cnt(lambda r: r['IoU'] >= 0.95)} of 18   "
          f"(library {cnt(lambda r: r['IoU_lib'] >= 0.95)} of 18)")
    print()
    from collections import Counter
    print("winning start:", dict(Counter(r["winner"] for r in rs)))
    print("survivor sets:", dict(Counter(tuple(r["survivors"]) for r in rs)))
    print("energy gate violations:", {r["shape"]: r["gate_viol"] for r in rs if r["gate_viol"]})
    print("horizon stops:", [r["shape"] for r in rs if r["horizon"]])
    print("uniform reproduction max abs diff against out_lib:",
          max(r["uniform_repro"] for r in rs))
    print("forward-equivalents spent: min %.1f max %.1f"
          % (min(r["fe"] for r in rs), max(r["fe"] for r in rs)))
    print("wall s: min %.0f max %.0f total %.0f"
          % (min(r["wall"] for r in rs), max(r["wall"] for r in rs),
             sum(r["wall"] for r in rs)))


if __name__ == "__main__":
    main()
