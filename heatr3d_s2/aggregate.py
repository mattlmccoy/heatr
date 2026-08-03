"""S2 Task 3: turn the campaign into bands and verdicts.

    ./.venv312/bin/python -m heatr3d_s2.aggregate

A RESOLVED AMBIGUITY IN MY OWN PRE-REGISTRATION, recorded rather than quietly
picked. The pre-registration describes the IoU quantities as "IoU of the
phi>=0.9 melt region against the ANALYTIC nominal shape", but anchors their
ceiling on a precedent (0.0344 from phase_a_shape_gate.json) that is the
Jaccard distance between TWO GRIDS' melt regions, not between one grid and the
nominal. Those are different quantities and only one of them is what the
ceiling was calibrated against.

Resolution: BOTH are computed and reported, and the VERDICT is taken on the
grid-to-grid form, because
  (a) it is what the cited precedent measures, so the ceiling means something;
  (b) it is what "convergence" means -- does the melt region stop MOVING as the
      grid refines -- whereas agreement with the nominal is a physics outcome
      that a converged solver is free to get wrong.
The nominal-referenced form is reported with its own band as the physics
reading. Nothing is widened; an ambiguity is closed toward the calibrated side
and the other side is published beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from heatr3d_s2 import bands, harness
from solve3d import gates as sg
from solve3d import shape_metrics as sm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "convergence_bands.json"

# per-grid scalar quantities -> (campaign key, relative?, prereg ceiling key)
PER_GRID = [
    ("iou_vs_nominal_phi0p9", "iou_phi0p9", False, None),
    ("iou_vs_nominal_phi0p8", "iou_phi0p8", False, None),
    ("in_part_melt_fraction_phi0p9", "in_part_phi0p9", False,
     "in_part_melt_fraction"),
    ("out_of_part_melt_fraction_phi0p9", "out_of_part_phi0p9", False,
     "out_of_part_melt_fraction"),
    ("sigma_T", "sigma_T_c", True, "sigma_T"),
]


def _prereg() -> dict:
    return json.loads((RESULTS / "s2_preregistration.json").read_text())


def _ceiling(pr: dict, key: str | None):
    return None if key is None else pr["quantities"][key]["pass_ceiling"]


def _pairwise(shape: str, grids: list[int], read: str) -> dict:
    """Grid-to-grid melt-region disagreement: the calibrated form."""
    _, _, _, h = sg.eval_grid_axes()
    fields = {n: np.asarray(np.load(RESULTS / f"field_{shape}_n{n}.npz")[read])
              for n in grids}
    jac09, jac08, ssd = [], [], []
    for a, b in zip(grids, grids[1:]):
        ja, jb, sd = [], [], []
        for i in range(fields[a].shape[0]):
            pa = sg.phase_fraction_phi(fields[a][i])
            pb = sg.phase_fraction_phi(fields[b][i])
            ja.append(1.0 - sm.iou(pa >= 0.9, pb >= 0.9))
            jb.append(1.0 - sm.iou(pa >= 0.8, pb >= 0.8))
            sd.append(sm.symmetric_surface_distance_mm(pa >= 0.9, pb >= 0.9, h))
        jac09.append(float(np.nanmean(ja)))
        jac08.append(float(np.nanmean(jb)))
        ssd.append(float(np.nanmean(sd)))
    return {"jaccard_dist_phi0p9": jac09, "jaccard_dist_phi0p8": jac08,
            "front_ssd_mm": ssd}


def build() -> dict:
    pr = _prereg()
    camp = json.loads((RESULTS / "campaign.json").read_text())
    cases = camp["cases"]
    doc = {"what": "S2 Task 3 convergence bands and verdicts",
           "band_rule": pr["band_rule"], "pass_criteria": pr["pass_criteria"],
           "preregistration_ambiguity_resolved": __doc__.split("A RESOLVED")[1].strip(),
           "shapes": {}, "not_run": camp.get("not_run", {})}
    for shape in pr["shapes"]:
        grids = sorted(int(k.split("_n")[1]) for k in cases
                       if k.startswith(shape + "_n"))
        if len(grids) < pr["convergence"]["min_grids_for_a_claim"]:
            doc["shapes"][shape] = {"status": "INSUFFICIENT_GRIDS",
                                    "grids_completed": grids,
                                    "rule": pr["partial_completion_policy"]["rule"]}
            continue
        entry = {"grids": grids, "reads": {}}
        for read in ("melt_onset", "heating_fixed_time"):
            q = {}
            for name, key, rel, ck in PER_GRID:
                vals = [cases[f"{shape}_n{n}"]["reads"][read][key] for n in grids]
                q[name] = bands.analyse(grids, vals, relative=rel,
                                        ceiling=_ceiling(pr, ck))
            if read == "melt_onset":
                t90 = [cases[f"{shape}_n{n}"]["reads"][read]["t90_s"] for n in grids]
                q["t90"] = bands.analyse(grids, t90, relative=True,
                                         ceiling=_ceiling(pr, "t90"))
            pw = _pairwise(shape, grids, read)
            q["jaccard_dist_phi0p9_grid_to_grid"] = bands.analyse_from_changes(
                grids, pw["jaccard_dist_phi0p9"], ceiling=_ceiling(pr, "iou_phi0p9"))
            q["jaccard_dist_phi0p8_grid_to_grid"] = bands.analyse_from_changes(
                grids, pw["jaccard_dist_phi0p8"], ceiling=_ceiling(pr, "iou_phi0p8"))
            q["front_ssd_mm_grid_to_grid"] = bands.analyse_from_changes(
                grids, pw["front_ssd_mm"], ceiling=_ceiling(pr, "front_position_mm"))
            entry["reads"][read] = q
        doc["shapes"][shape] = entry

    # ---- verdicts ---------------------------------------------------- #
    verdict_quantities = ["jaccard_dist_phi0p9_grid_to_grid",
                          "jaccard_dist_phi0p8_grid_to_grid",
                          "front_ssd_mm_grid_to_grid",
                          "in_part_melt_fraction_phi0p9",
                          "out_of_part_melt_fraction_phi0p9"]
    verdicts = {}
    for shape, e in doc["shapes"].items():
        if e.get("status") == "INSUFFICIENT_GRIDS":
            verdicts[shape] = {"verdict": "INSUFFICIENT_GRIDS"}
            continue
        per = {}
        for read, q in e["reads"].items():
            per[read] = {
                "verdict_quantities": {k: q[k]["pass"] for k in verdict_quantities
                                       if k in q},
                "t90": q.get("t90", {}).get("pass"),
                "sigma_T_diagnostic_band": q["sigma_T"]["band"],
                "sigma_T_status": q["sigma_T"]["status"]}
            per[read]["all_verdict_quantities_pass"] = all(
                v for v in per[read]["verdict_quantities"].values())
        verdicts[shape] = {
            "per_read": per,
            "verdict": ("PASS" if all(p["all_verdict_quantities_pass"]
                                      for p in per.values()) else "FAIL")}
    doc["verdicts"] = verdicts
    completed = [s for s, v in verdicts.items()
                 if v.get("verdict") in ("PASS", "FAIL")]
    doc["s2_task3_verdict"] = {
        "shapes_scored": completed,
        "shapes_insufficient": [s for s, v in verdicts.items()
                                if v.get("verdict") == "INSUFFICIENT_GRIDS"],
        "verdict": ("PASS" if completed and all(
            verdicts[s]["verdict"] == "PASS" for s in completed) else
            ("FAIL" if completed else "NOT_ENOUGH_DATA")),
        "rule": pr["pass_criteria"]["verdict_rule"]}
    OUT.write_text(json.dumps(doc, indent=1))
    return doc


def main() -> int:
    d = build()
    print(json.dumps({"verdict": d["s2_task3_verdict"],
                      "per_shape": {s: v.get("verdict")
                                    for s, v in d["verdicts"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
