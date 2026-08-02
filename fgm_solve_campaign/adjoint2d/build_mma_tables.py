"""Tables for `MMA_RETEST_REPORT.md`: five arms, six shapes, one scoring rule.

THE ARMS. Each is a stem in `out_mma`, except the prior pass's projection arm,
which is read unchanged from `out_topopt` so the optimizer comparison at a
matched budget has its L-BFGS-B counterpart.

  a  filter only, L-BFGS-B, 40 forward-equivalents   the production recipe
  e  projection, L-BFGS-B, 40                        the prior pass, read only
  b  projection, MMA, 40                             the mechanics fix, matched
  c  projection, MMA, 80                             bigger budget
  d  projection, L-BFGS-B carried, 80                budget without the mechanics

STOP CONVENTION, repeated here because every number depends on it: each arm's
J is read at the argmin over that arm's OWN stored trajectory; `at_horizon`
makes the number an upper bound. Every fidelity number is at grid 120 unless
the column says 160.

Run: ./.venv312/bin/python -m adjoint2d.build_mma_tables
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_MMA = ROOT / "out_mma"
OUT_TOPOPT = ROOT / "out_topopt"

SHAPES = ("square", "circle", "trapezoid", "triangle", "diamond", "rectangle")

# label -> (directory, stem template, short description)
ARMS = {
    "a_filteronly_lbfgsb_40": (OUT_MMA, "{s}_control_filteronly",
                               "filter only, L-BFGS-B, 40"),
    "e_projection_lbfgsb_40": (OUT_TOPOPT, "{s}",
                               "projection, L-BFGS-B, 40 (prior pass)"),
    "b_projection_mma_40": (OUT_MMA, "{s}_mma", "projection, MMA, 40"),
    "c_projection_mma_80": (OUT_MMA, "{s}_mma_b80", "projection, MMA, 80"),
    "d_projection_lbfgsbcarry_80": (OUT_MMA, "{s}_lbfgsb_carry_b80",
                                    "projection, L-BFGS-B carried, 80"),
}

DELIVERABLE = "TO_4bpp"     # the deliverable arm is always the 4 bits-per-pixel map


def load(arm: str, shape: str) -> dict | None:
    d, stem, _ = ARMS[arm]
    p = d / f"{stem.format(s=shape)}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def row(arm: str, shape: str) -> dict | None:
    j = load(arm, shape)
    if j is None:
        return None
    m = j["arms"][DELIVERABLE]
    u = j["arms"]["U_uniform"]
    return {
        "arm": arm, "shape": shape, "source": str(ARMS[arm][1].format(s=shape)),
        "optimizer": j.get("optimizer", "lbfgsb"),
        "budget": j.get("budget_forward_equivalents_total"),
        "n_evals": j.get("n_evals_used_total"),
        "spent_fwd_eq": j.get("spent_forward_equivalents"),
        "J": m["J"], "J_raster_chi": m["J_raster_chi"],
        "IoU_120": m["IoU"], "IoU_area_120": m["IoU_area"],
        "growth_pct": m["bed_melt_pct_of_part"],
        "under_pct": m["part_under_melt_pct"],
        "M_nd": m["non_discreteness"],
        "P_abs_W_per_m": m["P_abs_W_per_m"],
        "t_stop_s": m["t_stop_s"], "at_horizon": bool(m["t_stop_at_horizon"]),
        "uniform_J": u["J"], "uniform_IoU_120": u["IoU"],
        "energy_gate_violations": j.get("energy_gate_violations", []),
        "wall_s": j.get("wall_s"),
    }


def collect() -> dict:
    out: dict = {"arms": {k: v[2] for k, v in ARMS.items()}, "rows": [],
                 "missing": []}
    for shape in SHAPES:
        for arm in ARMS:
            r = row(arm, shape)
            if r is None:
                out["missing"].append(f"{arm}/{shape}")
            else:
                out["rows"].append(r)
    return out


def _fmt(v, spec="8.4f"):
    return "n/a" if v is None else format(v, spec)


def markdown(data: dict) -> str:
    lines = []
    for metric, spec, title in (
            ("J", "9.2f", "J against the grid-independent area-fill target, grid 120"),
            ("J_raster_chi", "9.2f",
             "J_raster_chi, the same map under the old binary target, grid 120"),
            ("IoU_120", "7.4f", "IoU against the binary part mask, grid 120"),
            ("growth_pct", "7.2f", "growth: melted bed as a percent of part area, grid 120"),
            ("under_pct", "7.2f", "under: unmelted part as a percent of part area, grid 120"),
            ("M_nd", "6.3f", "non-discreteness M_nd = mean 4 s (1 - s) over the part"),
            ("n_evals", "3d", "gradient evaluations actually used"),
    ):
        lines.append(f"\n**{title}**\n")
        lines.append("| shape | " + " | ".join(ARMS) + " |")
        lines.append("|" + "---|" * (len(ARMS) + 1))
        for shape in SHAPES:
            cells = []
            for arm in ARMS:
                r = next((x for x in data["rows"]
                          if x["arm"] == arm and x["shape"] == shape), None)
                cells.append("n/a" if r is None else _fmt(r[metric], spec))
            lines.append(f"| {shape} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> dict:
    data = collect()
    (OUT_MMA / "tables.json").write_text(json.dumps(data, indent=2, default=float))
    (OUT_MMA / "tables.md").write_text(markdown(data))
    print(markdown(data))
    if data["missing"]:
        print("\nMISSING (not silently dropped):", ", ".join(data["missing"]))
    for r in data["rows"]:
        if r["at_horizon"]:
            print(f"HORIZON FLAG: {r['arm']}/{r['shape']} stops on the last stored "
                  f"step, so its J {r['J']:.2f} is an upper bound")
        if r["energy_gate_violations"]:
            print(f"ENERGY GATE VIOLATION: {r['arm']}/{r['shape']} "
                  f"{r['energy_gate_violations']}")
    return data


if __name__ == "__main__":
    main()
