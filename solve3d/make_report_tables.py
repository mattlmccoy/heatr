"""Emit the PHASE_A_REPORT.md tables STRAIGHT FROM the results JSONs.

Same no-transcription rule the D1 report followed: every number in the report
is printed by this script from solve3d/results/*.json, so a stale or
hand-copied figure is impossible.

    heatr3d_d1_spike/env/bin/python -m solve3d.make_report_tables
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def tolerance_table() -> str:
    d = _load("parity_tolerances.json")
    raw, tol = d["raw_spread"], d["tolerances"]
    runs = d["measurement"]["runs"]
    out = ["### Task 1: measured heatr3d self-spread -> FROZEN tolerances", "",
           "| quantity | n=64 | n=96 (reference) | measured spread | "
           f"tolerance ({d['safety_factor']}x) |", "|---|---|---|---|---|"]
    a, b = runs["n64"], runs["n96"]
    out.append(f"| t90 [s] | {a['t90_s']!r} | {b['t90_s']!r} | "
               f"{raw['t90_rel_spread']!r} | {tol['t90_rel']!r} |")
    out.append(f"| part-mean heating curve (rel-L2) | - | - | "
               f"{raw['curve_rel_l2_spread']!r} | {tol['curve_rel_l2']!r} |")
    out.append(f"| sigma_T at melt onset [C] | {a['sigma_T_c']!r} | "
               f"{b['sigma_T_c']!r} | {raw['sigma_T_rel_spread']!r} | "
               f"{tol['sigma_T_rel']!r} |")
    out += ["", "Provenance of the two anchor runs (both `reached = true`, "
                "`clamp_bound = false`):", "",
            "| | n=64 | n=96 |", "|---|---|---|"]
    for k in ("n_voxels_in_part", "part_volume_m3", "p_total_w",
              "energy_residual_frac", "wall_eqs_s", "wall_march_s"):
        out.append(f"| `{k}` | {a[k]!r} | {b[k]!r} |")
    return "\n".join(out)


def gate_table() -> str:
    d = _load("phase_a_gate.json")
    tol, arms, v = d["tolerances"], d["arms"], d["verdict"]
    out = ["### Task 4: coupled-forward parity gate "
           "(`solve3d/results/phase_a_gate.json`)", "",
           "| arm | t90 rel diff | vs tol | curve rel-L2 | vs tol | "
           "sigma_T rel diff | vs tol | n_eqs_solves dolfinx / heatr3d | "
           "arm_ok |", "|---|---|---|---|---|---|---|---|---|"]
    for name, a in arms.items():
        c = v[name]
        out.append(
            f"| {name} | {a['t90_rel_diff']!r} | "
            f"{'PASS' if c['t90'] else 'FAIL'} ({tol['t90_rel']!r}) | "
            f"{a['curve_rel_l2']!r} | "
            f"{'PASS' if c['curve'] else 'FAIL'} ({tol['curve_rel_l2']!r}) | "
            f"{a['sigma_T_rel_diff']!r} | "
            f"{'PASS' if c['sigma_T'] else 'FAIL'} ({tol['sigma_T_rel']!r}) | "
            f"{a['n_eqs_solves_dolfinx']} / {a['n_eqs_solves_heatr3d']} "
            f"({'PASS' if c['n_eqs_solves'] else 'FAIL'}) | "
            f"{'PASS' if c['arm_ok'] else 'FAIL'} |")
    out += ["", f"`verdict.gate_ok` = `{str(v['gate_ok']).lower()}`", "",
            "Raw values behind the ratios:", "",
            "| arm | t90 dolfinx / heatr3d [s] | sigma_T dolfinx / heatr3d "
            "(mid-plane) [C] | heatr3d mid-plane / volumetric sigma_T |",
            "|---|---|---|---|"]
    for name, a in arms.items():
        out.append(f"| {name} | {a['t90_dolfinx_s']!r} / {a['t90_heatr3d_s']!r} "
                   f"| {a['sigma_T_dolfinx_midplane_c']!r} / "
                   f"{a['sigma_T_heatr3d_midplane_c']!r} | "
                   f"{a['heatr3d_midplane_over_volumetric']!r} |")
    return "\n".join(out)


def diagnostic_table() -> str:
    d = _load("phase_a_gate.json")
    out = ["### Where the two engines disagree (per-arm field diagnostic)", "",
           "| arm | T-rise rel-L2 all | interior | surface band | "
           "sigma_T interior rel diff | part volume rel diff | "
           "p_target rel diff |", "|---|---|---|---|---|---|---|"]
    for name, a in d["arms"].items():
        f = a["field_diagnostic"]
        out.append(f"| {name} | {f['T_rise_rel_l2_all']!r} | "
                   f"{f['T_rise_rel_l2_interior']!r} | "
                   f"{f['T_rise_rel_l2_surface_band']!r} | "
                   f"{f['sigma_T_interior_rel_diff']!r} | "
                   f"{a['part_volume_rel_diff']!r} | "
                   f"{a['p_target_rel_diff']!r} |")
    out += ["", "| arm | surface-minus-interior mean T, dolfinx [C] | "
                "heatr3d [C] | n interior / band points |",
            "|---|---|---|---|"]
    for name, a in d["arms"].items():
        f = a["field_diagnostic"]
        out.append(f"| {name} | {f['surface_minus_interior_dolfinx_c']!r} | "
                   f"{f['surface_minus_interior_heatr3d_c']!r} | "
                   f"{f['n_interior']} / {f['n_surface_band']} |")
    return "\n".join(out)


def cost_table() -> str:
    d = _load("phase_a_gate.json")
    out = ["### Cost (wall clock, serial, OMP_NUM_THREADS=1)", "",
           "| arm | dolfinx mesh [s] | dolfinx EQS [s] | dolfinx march [s] | "
           "heatr3d EQS [s] | heatr3d march [s] | EQS speedup |",
           "|---|---|---|---|---|---|---|"]
    for name, a in d["arms"].items():
        he, hm = a.get("heatr3d_wall_eqs_s"), a.get("heatr3d_wall_march_s")
        sp = (he / a["wall_eqs_s"]) if he else None
        out.append(f"| {name} | {a['wall_mesh_s']!r} | {a['wall_eqs_s']!r} | "
                   f"{a['wall_march_s']!r} | {he!r} | {hm!r} | {sp!r} |")
    out += ["", "| arm | dolfinx dofs / cells | in-part nodes | "
                "heatr3d in-part voxels | lc_part [m] | dt_stable [s] | "
                "n_substeps | k re-assemblies |",
            "|---|---|---|---|---|---|---|---|"]
    for name, a in d["arms"].items():
        out.append(f"| {name} | {a['n_dofs_total']} / {a['n_cells_total']} | "
                   f"{a['n_nodes_in_part']} | {a['heatr3d_voxels_in_part']} | "
                   f"{a['lc_part_m']!r} | {a['dt_stable_s']!r} | "
                   f"{a['n_substeps_used']} | {a['n_k_assemblies']} |")
    return "\n".join(out)


def health_table() -> str:
    d = _load("phase_a_gate.json")
    out = ["### Standing health checks (every arm)", "",
           "| arm | energy_residual_frac | clamp_bound | cfl_violated | "
           "power renorm residual | eval_missed |",
           "|---|---|---|---|---|---|"]
    for name, a in d["arms"].items():
        out.append(f"| {name} | {a['energy_residual_frac']!r} | "
                   f"{str(a['clamp_bound']).lower()} | "
                   f"{str(a['cfl_violated']).lower()} | "
                   f"{a['power_renorm_residual']!r} | {a['eval_missed']} |")
    return "\n".join(out)


def main() -> int:
    print(tolerance_table())
    print()
    print(gate_table())
    print()
    print(diagnostic_table())
    print()
    print(cost_table())
    print()
    print(health_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
