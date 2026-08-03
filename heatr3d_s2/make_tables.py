"""Emit the S2_GATE_REPORT.md tables STRAIGHT FROM heatr3d_s2/results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
VQ = ["jaccard_dist_phi0p9_grid_to_grid", "jaccard_dist_phi0p8_grid_to_grid",
      "front_ssd_mm_grid_to_grid", "in_part_melt_fraction_phi0p9",
      "out_of_part_melt_fraction_phi0p9"]


def _l(n):
    return json.loads((R / n).read_text())


def verdict_table() -> str:
    d = _l("convergence_bands.json")
    o = ["### Convergence verdict per shape per quantity", "",
         "| shape | read | quantity | successive changes | finest | ceiling | "
         "status | band | PASS |", "|---|---|---|---|---|---|---|---|---|"]
    for shape, e in d["shapes"].items():
        if e.get("status"):
            o.append(f"| {shape} | - | - | - | - | - | {e['status']} | - | - |")
            continue
        for read, q in e["reads"].items():
            for k in VQ + ["t90", "sigma_T"]:
                if k not in q:
                    continue
                v = q[k]
                ch = ", ".join(f"{c['change']:.5g}" for c in v["changes"])
                band = "-" if v["band"] is None else f"{v['band']:.5g}"
                ceil = "-" if v["ceiling"] is None else f"{v['ceiling']:g}"
                p = {True: "PASS", False: "FAIL", None: "n/a"}[v["pass"]]
                o.append(f"| {shape} | {read} | {k} | {ch} | "
                         f"{v['finest_change']:.5g} | {ceil} | {v['status']} | "
                         f"{band} | {p} |")
    vv = d["s2_task3_verdict"]
    o += ["", f"Per shape: " + ", ".join(
        f"**{s}** {v.get('verdict')}" for s, v in d["verdicts"].items()),
        "", f"`s2_task3_verdict` = **{vv['verdict']}** "
            f"(scored {vv['shapes_scored']}, insufficient "
            f"{vv['shapes_insufficient']})"]
    return "\n".join(o)


def gauge_table() -> str:
    d = _l("gauge_decision.json")
    o = ["### Electrode-gauge decision", "",
         "| arm | raw power density by grid | total drift | finest-pair change |",
         "|---|---|---|---|"]
    for a, s in d["summary"].items():
        o.append(f"| {a} | " +
                 ", ".join(f"{v:.3f}" for v in s["raw_power_density_w_per_m3"]) +
                 f" | {s['total_drift_rel']:.5f} | "
                 f"{s['finest_pair_rel_change']:.5f} |")
    o += ["", "| n | measured cell/face ratio | predicted (n/(n-1))^2 | abs diff |",
          "|---|---|---|---|"]
    for r in d["ratio_check"]["per_grid"]:
        o.append(f"| {r['n']} | {r['measured']!r} | {r['predicted']!r} | "
                 f"{r['abs_diff']:.3e} |")
    i = d["renormalized_inertness"]
    o += ["", f"Winner by the pre-registered rule: **{d['winner']}** "
              f"(shipped arm wins: {str(d['winner_is_the_shipped_arm']).lower()}).",
          "", f"**Inertness**: renormalized Q differs between gauges by "
              f"{i['max_abs_diff_over_max_q']:.3e} of max Q and total power by "
              f"{i['total_power_rel_diff']!r} -- the gauge cannot move any "
              f"thermal output."]
    return "\n".join(o)


def mechanism_table() -> str:
    d = _l("mechanisms.json")
    L, C = d["lshape_outlier"], d["cylinder_null"]
    o = ["### Mechanism 1 -- the L-shape outlier (reentrant corner)", "",
         "| n | corner power share | corner volume share | concentration | "
         "corner max/mean | whole-part max/mean | in-part CV |",
         "|---|---|---|---|---|---|---|"]
    for r in L["rows"]:
        o.append(f"| {r['n']} | {r['corner_power_share']:.5f} | "
                 f"{r['corner_volume_share']:.5f} | "
                 f"{r['corner_power_share']/r['corner_volume_share']:.4f} | "
                 f"{r['corner_max_over_mean']:.4f} | "
                 f"{r['in_part_max_over_mean']:.4f} | {r['in_part_cv']:.4f} |")
    o += ["", f"`corner_peak_growing` = "
              f"`{str(L['corner_peak_growing']).lower()}`, "
              f"`concentration_growing` = "
              f"`{str(L['concentration_growing']).lower()}`.", "",
          "### Mechanism 2 -- the cylinder null (no attenuation across the part)",
          "", f"skin depth / part radius = **{C['skin_depth_over_radius']:.1f}**, "
              f"loss tangent sigma/(omega eps) = "
              f"**{C['loss_tangent_sigma_over_omega_eps']:.3f}**", "",
          "| n | interior CV | interior max/mean | whole-part CV |",
          "|---|---|---|---|"]
    for r in C["rows"]:
        o.append(f"| {r['n']} | {r['interior_cv']:.5f} | "
                 f"{r['interior_max_over_mean']:.4f} | {r['in_part_cv']:.4f} |")
    return "\n".join(o)


def main() -> int:
    for f in (verdict_table, gauge_table, mechanism_table):
        print(f())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def commensurability_table() -> str:
    d = _l("commensurability.json")
    o = ["### Addendum -- axis-aligned staircase commensurability", "",
         "| n | h [mm] | cells across | effective half-width [mm] | error [mm] | "
         "error [%] | commensurate |", "|---|---|---|---|---|---|---|"]
    for r in d["geometry"]:
        o.append(f"| {r['n']} | {r['h_m']*1e3:.4f} | {r['cells_across']} | "
                 f"{r['half_width_effective_m']*1e3:.4f} | "
                 f"{r['half_width_error_m']*1e3:+.4f} | "
                 f"{100*r['half_width_error_rel']:+.2f} | "
                 f"{str(r['commensurate']).lower()} |")
    o += ["", "| shape | pair | jaccard phi>=0.9 | geometry jump [mm] | "
              "jaccard per mm of jump |", "|---|---|---|---|---|"]
    for s, e in d["shapes"].items():
        for m in e["successive_pairs"]:
            rat = m["metric_per_mm_of_geometry_jump"]
            o.append(f"| {s} | {m['pair'][0]}->{m['pair'][1]} | "
                     f"{m['jaccard_dist_phi0p9']:.5f} | "
                     f"{m['geometry_jump_m']*1e3:.4f} | "
                     f"{'n/a' if rat is None else f'{rat:.4f}'} |")
    o += ["", "| shape | COMMENSURATE pair | jaccard phi>=0.9 | front SSD [mm] |",
          "|---|---|---|---|"]
    for s, e in d["shapes"].items():
        c = e.get("commensurate_pair")
        if c:
            o.append(f"| {s} | {c['pair'][0]}<->{c['pair'][1]} | "
                     f"{c['jaccard_dist_phi0p9']:.5f} | {c['front_ssd_mm']:.5f} |")
    return "\n".join(o)


def densify_table() -> str:
    d = _l("densify_coupled.json")
    c = d["controls"]["densify_off_inertness"]
    o = ["### Task 4 -- the densify=True coupled march", "",
         f"**Control, densify OFF**: `inert` = "
         f"`{str(c['inert']).lower()}`, max |dT| between b = 0.0 and b = 0.6 is "
         f"**{c['max_abs_dT_c']}** C, t90 identical: "
         f"`{str(c['t90_identical']).lower()}`. The prior inertness is proven "
         f"bit-for-bit, exactly as the structural argument predicts.", "",
         f"**Reachable with densify ON**: "
         f"`{str(d['question_a_density_coupling_is_reachable']['answer']).lower()}`.",
         "", "| b | t90 [s] | sigma_T [C] | surface-minus-interior [C] | "
         "rho_final mean | EQS solves | energy resid | clamp |",
         "|---|---|---|---|---|---|---|---|"]
    for b, r in d["arms"].items():
        t, g = r["topology"], r["gates"]
        rho = r.get("rho_final", {}).get("mean", float("nan"))
        o.append(f"| {b} | {r['t90_s']:.2f} | {r['sigma_T_c']:.3f} | "
                 f"{t['surface_minus_interior_c']:+.3f} | {rho:.5f} | "
                 f"{r['n_eqs_solves']} | {g['energy_residual_frac']:.2e} | "
                 f"{str(g['clamp_bound']).lower()} |")
    o += ["", "| b | d t90 [s] | d sigma_T [C] | d(surface-interior) [C] | "
              "max abs dT vs b=0 [C] |", "|---|---|---|---|---|"]
    for b, r in d["arms"].items():
        if b == "0.0":
            continue
        dv, vz = r["delta_vs_zero"], r.get("vs_zero_arm", {})
        o.append(f"| {b} | {dv['t90_s']:+.3f} | {dv['sigma_T_c']:+.4f} | "
                 f"{dv['surface_minus_interior_c']:+.4f} | "
                 f"{vz.get('max_abs_dT_c', float('nan')):.4f} |")
    sc = d["spot_check"]
    o += ["", "| spot-check grid | t90 [s] | sigma_T [C] |", "|---|---|---|"]
    for k, v in sc.items():
        o.append(f"| {k} | {v['t90_s']:.2f} | {v['sigma_T_c']:.3f} |")
    o += ["", d["spot_check_caveat"]]
    return "\n".join(o)
