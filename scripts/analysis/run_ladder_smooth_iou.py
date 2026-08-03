#!/usr/bin/env python3
"""VARIANT B: re-score the rotating grid ladder with a smoothed melt indicator.

`ROTATING_GRID_LADDER_REPORT.md` Section 10 asks whether the cross's rotating
intersection-over-union scatter is manufactured by the NON-SMOOTH melt
threshold. The melted region is `phi >= 0.5`, a Heaviside on a field whose
phase-change window is 10 C wide, so a whole ring of cells can enter or leave
the melted set for a small change of the field.

WHAT THIS RUNS. NOTHING. No forward run, no solve, no gradient. Every melt
field this needs was already stored by the ladder at each arm's own stop, so
this is a pure RE-SCORING of stored fields. That is the strongest possible form
of the variant: the physics is bit-identical to the published ladder by
construction, and the only thing that changes is the metric. The recomputed
binary intersection over union is compared against the stored one as the
reproduction check.

THE REPLACEMENT METRIC, and why this one. `rot_ladder_variants.melt_area_fill`
gives the sub-cell AREA FRACTION of each cell that lies inside the melt front,
which is the same convention the target indicator chi already uses. Both sides
of the overlap are then the same kind of object. The Gaussian-smoothed variant
is computed too, over a shrinking width, as the regularizer-width diagnostic,
with sigma = 0 required to return the original metric.

Run:
  ./.venv312/bin/python scripts/analysis/run_ladder_smooth_iou.py \
      --tag cross --sources cross_ladder cross_align_probe_ladder
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from rot_ladder_variants import pearson_r, smoothed_scores, spread  # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_rot_ladder"
ARMS = ("ROT_uniform", "STATIC_uniform", "QS_uniform", "ROT_transfer")
N_SUB = 8
SIGMAS = (1.0, 0.5, 0.25, 0.0)
REPRO_TOL_IOU = 5e-4


def load_sources(names: list[str]) -> tuple[dict, dict]:
    """Merge several ladder result files and their melt-field archives."""
    grids: dict[str, dict] = {}
    fields: dict[str, np.ndarray] = {}
    prov = []
    for nm in names:
        j = OUT / f"{nm}.json"
        z = OUT / f"{nm}_maps.npz"
        d = json.loads(j.read_text())
        arr = np.load(z)
        for g, rec in d["grids"].items():
            if g in grids:
                raise ValueError(f"grid {g} appears in more than one source")
            grids[g] = rec
        for k in arr.files:
            fields[k] = np.asarray(arr[k])
        prov.append({"json": str(j), "npz": str(z),
                     "grids": sorted(d["grids"], key=int),
                     "reproduction_gate": d.get("reproduction_gate", {}).get("status")})
    return grids, {"fields": fields, "provenance": prov}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--n-sub", type=int, default=N_SUB)
    a = ap.parse_args()

    t0 = time.perf_counter()
    grids, bag = load_sources(a.sources)
    fields = bag["fields"]

    res = {
        "tag": a.tag,
        "task": "VARIANT B: re-score the stored ladder melt fields with a "
                "smoothed melt indicator. No forward run, no solve.",
        "metric_primary": "IoU_subcell_melt_vs_chi: the sub-cell area fill of "
                          "the melt front against the sub-cell area-fill "
                          "target chi, both by the same convention",
        "metric_original": "IoU_binary: the melted set phi >= 0.5 against the "
                           "BINARY part mask, the published reading",
        "regularizer_width_diagnostic": "IoU_gauss_sigma*: the melted set after "
                                        "smoothing phi by sigma cells; sigma = 0 "
                                        "must return the original metric",
        "n_sub": int(a.n_sub), "sigmas_cells": list(SIGMAS),
        "sources": bag["provenance"],
        "grids": {},
    }

    repro = []
    for g in sorted(grids, key=int):
        n = int(g)
        chi = np.asarray(fields[f"g{n}_chi"], dtype=float)
        pm = np.asarray(fields[f"g{n}_part_mask"], dtype=bool)
        row = {"n_grid": n, "n_part_cells": int(pm.sum()),
               "arms": {}}
        for arm in ARMS:
            key = f"g{n}_phi_{arm}"
            if key not in fields:
                continue
            phi = np.asarray(fields[key], dtype=float)
            sc = smoothed_scores(phi, chi, pm, n_sub=int(a.n_sub),
                                 sigmas=SIGMAS)
            stored = grids[g]["arms"][arm]
            sc["IoU_stored"] = float(stored["IoU"])
            sc["IoU_area_stored"] = float(stored["IoU_area"])
            sc["abs_dIoU_vs_stored"] = abs(sc["IoU_binary"] - sc["IoU_stored"])
            sc["rescoring_reproduces_stored"] = bool(
                sc["abs_dIoU_vs_stored"] <= REPRO_TOL_IOU)
            repro.append(sc["rescoring_reproduces_stored"])
            row["arms"][arm] = sc
            print(f"[{a.tag}@{n}] {arm:15s} binary {sc['IoU_binary']:.4f} "
                  f"(stored {sc['IoU_stored']:.4f}, d {sc['abs_dIoU_vs_stored']:.2e}) "
                  f" subcell-vs-chi {sc['IoU_subcell_melt_vs_chi']:.4f} "
                  f" phi-vs-chi {sc['IoU_phi_vs_chi_area']:.4f} "
                  f" gauss1.0 {sc['IoU_gauss_sigma1']:.4f}", flush=True)
        res["grids"][str(n)] = row

    res["rescoring_gate"] = {
        "n_comparisons_actually_made": len(repro),
        "ALL_PASS": (bool(all(repro)) if repro else None),
        "status": ("PASS" if (repro and all(repro))
                   else "FAIL" if repro else "NOT CHECKED, no stored melt "
                   "field overlapped a stored result row"),
        "tolerance_abs_IoU": REPRO_TOL_IOU,
        "note": "the recomputed binary reading against the number the ladder "
                "stored; the melt fields are archived as float32, so an exact "
                "bit match is not required, 5e-4 is",
    }

    # ladder-level readings, per arm and per metric
    ns = sorted((int(g) for g in res["grids"]), key=int)
    metrics = ["IoU_binary", "IoU_subcell_melt_vs_chi", "IoU_phi_vs_chi_area",
               "IoU_subcell_melt_vs_raster", "IoU_gauss_sigma1",
               "IoU_gauss_sigma0.5", "IoU_gauss_sigma0.25", "IoU_gauss_sigma0"]
    summary = {}
    for arm in ARMS:
        have = [n for n in ns if arm in res["grids"][str(n)]["arms"]]
        if len(have) < 2:
            continue
        s = {"grids": have}
        for m in metrics:
            v = [res["grids"][str(n)]["arms"][arm][m] for n in have]
            s[m] = {
                "values": v,
                "spread_all": spread(v),
                "spread_n_ge_160": spread(v, 160, have),
                "spread_n_ge_200": spread(v, 200, have),
                "last_step_abs": abs(v[-1] - v[-2]),
                "signs_of_steps": [float(np.sign(v[i + 1] - v[i]))
                                   for i in range(len(v) - 1)],
            }
        summary[arm] = s
    res["ladder_summary"] = summary
    res["wall_s"] = time.perf_counter() - t0

    (OUT / f"{a.tag}_smoothB.json").write_text(
        json.dumps(res, indent=2, default=float))
    print(f"[{a.tag}] RE-SCORING GATE {res['rescoring_gate']['status']} "
          f"({res['rescoring_gate']['n_comparisons_actually_made']} comparisons)",
          flush=True)
    for arm, s in summary.items():
        print(f"[{a.tag}] {arm:15s} spread n>=160: binary "
              f"{s['IoU_binary']['spread_n_ge_160']:.4f}  subcell "
              f"{s['IoU_subcell_melt_vs_chi']['spread_n_ge_160']:.4f}  "
              f"gauss1.0 {s['IoU_gauss_sigma1']['spread_n_ge_160']:.4f}",
              flush=True)
    print(f"[{a.tag}] wall {res['wall_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
