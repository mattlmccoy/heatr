"""Phase A close-out STEP 1: the measured CROSS-FAMILY parity band.

Pure numpy; runs in either environment.

    heatr3d_d1_spike/env/bin/python -m solve3d.crossfamily

Combines the two engines' OWN same-method spreads into one band and re-judges
the four Task-4 arms against it, reading their recorded numbers from
solve3d/results/phase_a_gate.json. NOTHING is re-run and neither
solve3d/results/parity_tolerances.json nor the recorded Task-4 verdict is
modified -- this close-out is additive.

Combination rule (declared in solve3d/gates.py BEFORE any number existed):
SUM of the two spreads (triangle inequality), then the same 1.5x safety factor
Task 1 used. Root-sum-square was considered and rejected as not being a bound.

Provenance of each half:
  heatr3d spread -- parity_tolerances.json, n=64 vs n=96, extruded circle,
                    coupling off (a 1.5x LINEAR refinement)
  dolfinx spread -- dolfinx_refinement.json, MAX over its two pair spreads,
                    the primary pair being the SAME 1.5x linear ratio
"""
from __future__ import annotations

import json
from pathlib import Path

from solve3d import gates

RESULTS = Path(__file__).resolve().parent / "results"
OUT_JSON = RESULTS / "parity_tolerances_crossfamily.json"

QUANTITIES = ("t90_rel", "curve_rel_l2", "sigma_T_rel")
_SPREAD_KEY = {"t90_rel": "t90_rel_spread",
               "curve_rel_l2": "curve_rel_l2_spread",
               "sigma_T_rel": "sigma_T_rel_spread"}

# Which quantities carry the Phase A verdict, per Matt's recorded objective
# hierarchy (spec commit d298c6d): shape first, then the time-to-shape. sigma_T
# is a flatness diagnostic and the heating curve is a trajectory diagnostic --
# both are REPORTED with their bands but do not decide the verdict.
VERDICT_CARRYING = ("t90_rel",)
DIAGNOSTIC_ONLY = ("curve_rel_l2", "sigma_T_rel")


def build() -> dict:
    tol = json.loads((RESULTS / "parity_tolerances.json").read_text())
    ref = json.loads((RESULTS / "dolfinx_refinement.json").read_text())
    gate = json.loads((RESULTS / "phase_a_gate.json").read_text())
    h_spread = tol["raw_spread"]
    d_spread = ref["spreads"]

    # TWO bands are reported, and the STRICTER one carries the verdict.
    #
    # `band`        -- the rule declared before computing: dolfinx spread =
    #                  MAX(coarse_vs_mid, mid_vs_fine).
    # `band_strict` -- dolfinx spread = mid_vs_fine only.
    #
    # Why both, stated plainly: "MAX" was chosen as the conservative option, but
    # for a GATE, wider is not conservative -- it makes passing easier. The
    # measurement then showed the coarse level is genuinely under-resolved
    # (t90 348.25 s against 325.00 / 324.35 s at mid / fine, and its mesh landed
    # 14% under its node target), so MAX inflates the t90 band roughly 20x.
    # The pre-declared rule is honoured and reported, but the verdict is taken
    # on the STRICTER band -- tightening a gate is always allowed; widening is
    # not. mid_vs_fine is also the pair that brackets the mesh Task 4 actually
    # ran, so it is the relevant local convergence estimate.
    band, band_strict = {}, {}
    for q in QUANTITIES:
        k = _SPREAD_KEY[q]
        d_strict = float(ref["spreads"]["mid_vs_fine"][k])
        band_strict[q] = {
            "heatr3d_spread": float(h_spread[k]),
            "dolfinx_spread": d_strict,
            "dolfinx_spread_source": "mid_vs_fine",
            "combined_spread": float(h_spread[k]) + d_strict,
            "tolerance": gates.combine_spreads(h_spread[k], d_strict),
            "task1_same_method_tolerance": float(tol["tolerances"][q]),
        }
        band[q] = {
            "heatr3d_spread": float(h_spread[k]),
            "dolfinx_spread": float(d_spread[k]),
            "combined_spread": float(h_spread[k]) + float(d_spread[k]),
            "tolerance": gates.combine_spreads(h_spread[k], d_spread[k]),
            "task1_same_method_tolerance": float(tol["tolerances"][q]),
        }
        band[q]["dolfinx_spread_source"] = "max(coarse_vs_mid, mid_vs_fine)"
        band[q]["widening_vs_task1"] = (
            band[q]["tolerance"] / band[q]["task1_same_method_tolerance"])
        band_strict[q]["widening_vs_task1"] = (
            band_strict[q]["tolerance"] / float(tol["tolerances"][q]))

    arms = {}
    for name, a in gate["arms"].items():
        measured = {"t90_rel": a["t90_rel_diff"],
                    "curve_rel_l2": a["curve_rel_l2"],
                    "sigma_T_rel": a["sigma_T_rel_diff"]}
        res = {"n_eqs_solves_ok":
               a["n_eqs_solves_dolfinx"] == a["n_eqs_solves_heatr3d"]}
        for q in QUANTITIES:
            res[q] = {"measured": measured[q],
                      "tolerance_declared_max_rule": band[q]["tolerance"],
                      "pass_declared_max_rule":
                          bool(measured[q] <= band[q]["tolerance"]),
                      "margin_declared_max_rule":
                          measured[q] / band[q]["tolerance"],
                      "tolerance": band_strict[q]["tolerance"],
                      "pass": bool(measured[q] <= band_strict[q]["tolerance"]),
                      "margin": measured[q] / band_strict[q]["tolerance"],
                      "role": ("verdict" if q in VERDICT_CARRYING
                               else "diagnostic")}
        res["verdict_carrying_pass"] = bool(
            res["n_eqs_solves_ok"] and all(res[q]["pass"] for q in VERDICT_CARRYING))
        res["verdict_carrying_pass_declared_max_rule"] = bool(
            res["n_eqs_solves_ok"]
            and all(res[q]["pass_declared_max_rule"] for q in VERDICT_CARRYING))
        arms[name] = res

    doc = {
        "what": "Phase A close-out STEP 1: cross-family parity band, measured "
                "from BOTH engines' own same-method spreads. The four Task-4 "
                "arms are re-judged from their RECORDED numbers; nothing was "
                "re-run and parity_tolerances.json is untouched.",
        "combination_rule": gates.COMBINATION_RULE,
        "combination_rule_statement":
            "tolerance = 1.5 * (heatr3d_spread + dolfinx_spread); the sum is "
            "the triangle-inequality BOUND on |A-B| when both engines converge "
            "to the same continuum answer. Root-sum-square was considered and "
            "rejected: it is smaller than the sum, so it is not a bound.",
        "safety_factor": gates.CROSS_FAMILY_SAFETY,
        "provenance": {
            "heatr3d": "parity_tolerances.json (n=64 vs n=96, extruded circle, "
                       "coupling off; 1.5x linear refinement)",
            "dolfinx": ref["spreads"]["reporting_rule"] + " | primary pair: "
                       + ref["spreads"]["primary_pair_note"],
        },
        "verdict_carrying": list(VERDICT_CARRYING),
        "diagnostic_only": list(DIAGNOSTIC_ONLY),
        "band": band,
        "band_strict": band_strict,
        "verdict_band": "band_strict",
        "verdict_band_note":
            "the STRICTER of the two is verdict-carrying; the pre-declared MAX "
            "rule is reported alongside so the looser reading is visible, not "
            "silently benefited from",
        "arms": arms,
        "all_arms_verdict_carrying_pass":
            all(a["verdict_carrying_pass"] for a in arms.values()),
        "all_arms_verdict_carrying_pass_declared_max_rule":
            all(a["verdict_carrying_pass_declared_max_rule"]
                for a in arms.values()),
    }
    gates.write_json(OUT_JSON.name, doc)
    return doc


def main() -> int:
    doc = build()
    print(json.dumps({"band": doc["band"],
                      "arms": {k: {q: v[q]["pass"] for q in QUANTITIES}
                               | {"n_eqs_solves_ok": v["n_eqs_solves_ok"]}
                               for k, v in doc["arms"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
