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
