"""S2 Task 3: the convergence campaign driver.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ./.venv312/bin/python -m heatr3d_s2.run_campaign

GRID-MAJOR ordering, deliberately: every shape is run at 48, then every shape
at 64, then 80, then 96. The pre-registration's run order exists so that
partial completion still yields a 3-grid band for EVERY shape rather than a
complete ladder for one shape and nothing for the others. Each case is written
to results/campaign.json as it finishes, so an interrupted run loses at most
one case.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from heatr3d_s2 import harness

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "campaign.json"


def prereg() -> dict:
    return json.loads((RESULTS / "s2_preregistration.json").read_text())


def _load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"what": "S2 Task 3 convergence campaign, corrected defaults, "
                    "coupling OFF, densify OFF",
            "cases": {}, "not_run": {}}


def run(shapes=None, grids=None) -> dict:
    pr = prereg()
    shapes = shapes or pr["shapes"]
    grids = grids or pr["grids"]["run_order"]
    t_ref = pr["read_states"]["heating_fixed_time"]["t_ref_s"]
    doc = _load()
    for n in grids:                       # GRID-MAJOR: see the module docstring
        for shape in shapes:
            key = f"{shape}_n{n}"
            if key in doc["cases"]:
                continue
            print(f"[s2] {key} ...", flush=True)
            t0 = time.perf_counter()
            try:
                rec = harness.run_case(shape, n, float(t_ref[shape]))
                rec["wall_total_s"] = time.perf_counter() - t0
                doc["cases"][key] = rec
                doc["not_run"].pop(key, None)
                print(f"[s2] {key} done in {rec['wall_total_s']:.0f} s "
                      f"t90={rec['reads']['melt_onset']['t90_s']:.2f} "
                      f"iou09={rec['reads']['melt_onset']['iou_phi0p9']:.5f}",
                      flush=True)
            except Exception as exc:      # record, never silently skip
                doc["not_run"][key] = {"status": "NOT_RUN", "reason": repr(exc),
                                       "traceback": traceback.format_exc()[-800:]}
                print(f"[s2] {key} FAILED: {exc!r}", flush=True)
            OUT.write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    run()
