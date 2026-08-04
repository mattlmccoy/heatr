"""Acceptance evidence: run the predicted-benefit gate over the REAL job pairs.

Prints, and records to gate_verdicts.json, what the gate says about every
complete before/after pair in the archive plus the accept-side controls. This
is the table quoted in the report; it is produced from the captured fixtures,
not typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

from studio3d.correction import proxy_degeneracy, choose_inversion_proxy
from studio3d.correction_gate import evaluate_correction

FIX = Path(__file__).resolve().parent / "tamper"
MANIFEST = json.loads((FIX / "manifest.json").read_text())
PAIRS = ["feb850ec", "c474d787", "d4d50045"]


def main() -> dict:
    out: dict = {"pairs": {}, "proxy": {}}
    for job in PAIRS:
        rec = MANIFEST["jobs"][job]
        before = json.loads((FIX / f"{job}_uncorrected_results.json").read_text())
        after = json.loads((FIX / f"{job}_corrected_results.json").read_text())
        before["out_of_part_melt_frac"] = \
            rec["arms"]["uncorrected"]["fields"]["out_of_part_melt_frac"]
        after["out_of_part_melt_frac"] = \
            rec["arms"]["corrected"]["fields"]["out_of_part_melt_frac"]
        v = evaluate_correction(before, after)
        out["pairs"][job] = {"label": rec["label"], "verdict": v["verdict"],
                             "failed_guards": v["failed_guards"],
                             "out_of_part_melt": [before["out_of_part_melt_frac"],
                                                  after["out_of_part_melt_frac"]],
                             "T_max_C": [before["gates"]["T_max_C"],
                                         after["gates"]["T_max_C"]],
                             "sigma_T": [before.get("sigma_T"),
                                         after.get("sigma_T")]}
    # accept-side control: a null correction reproduces the before arm
    rec = MANIFEST["jobs"]["feb850ec"]
    b = json.loads((FIX / "feb850ec_uncorrected_results.json").read_text())
    b["out_of_part_melt_frac"] = \
        rec["arms"]["uncorrected"]["fields"]["out_of_part_melt_frac"]
    out["pairs"]["null_correction_control"] = {
        "label": "null correction (before vs itself)",
        "verdict": evaluate_correction(b, dict(b))["verdict"],
        "failed_guards": [],
    }
    # proxy decisions per job
    for job in PAIRS + ["2ffdfc3c"]:
        rec = MANIFEST["jobs"][job]
        f = rec["arms"]["uncorrected"]["fields"]
        entry = {"label": rec["label"],
                 "rho_final": proxy_degeneracy(f["rho_final"], proxy_name="rho_final"),
                 "T_phi90": proxy_degeneracy(f["T_phi90"], proxy_name="T_phi90")}
        rp = FIX / f"{job}_uncorrected_results.json"
        if rp.exists():
            name, why = choose_inversion_proxy(json.loads(rp.read_text()))
            entry["chosen_proxy"] = name
            entry["why"] = why
        out["proxy"][job] = entry
    (FIX / "gate_verdicts.json").write_text(json.dumps(out, indent=2,
                                                        default=float))
    return out


if __name__ == "__main__":
    r = main()
    print("PREDICTED-BENEFIT GATE on the real job pairs")
    print(f"{'job':28s} {'verdict':10s} failed guards")
    for k, v in r["pairs"].items():
        print(f"{k + ' (' + v['label'] + ')':28.28s} {v['verdict']:10s} "
              f"{','.join(v['failed_guards'])}")
    print("\nPROXY DECISION / DEGENERACY")
    print(f"{'job':10s} {'chosen':9s} {'rho_final':>22s} {'T_phi90':>22s}")
    for k, v in r["proxy"].items():
        rd, td = v["rho_final"], v["T_phi90"]
        print(f"{k:10s} {v.get('chosen_proxy','-'):9s} "
              f"{('DEGEN(' + str(rd['reason']) + ') f=' + format(rd['frac_at_max'],'.4f')) if rd['degenerate'] else ('ok f=' + format(rd['frac_at_max'],'.4f')):>22s} "
              f"{('DEGEN(' + str(td['reason']) + ')') if td['degenerate'] else ('ok f=' + format(td['frac_at_max'],'.4f')):>22s}")
