#!/usr/bin/env python3
"""CLASSIFIER VERSION 2 measurement pass: anisotropy WITH a dopant map injected.

WHAT THIS MEASURES AND WHY. `GEOMETRY_GENERALIZATION_REPORT.md` Section 6.2
named the version-1 classifier's failure mode: it predicts whether an actuator
beats NO actuator, while the pipeline's real competitor is a SOLVED DOPANT MAP.
On the novel eight-tooth gear that gap produced the pass's one honest miss.
Version 2 re-measures every mode's residual azimuthal anisotropy with a design
map injected into the averaged kernel, in two variants:

  solved        the static SOLVED map this pipeline (or the library campaign)
                already produced. Costs a filtered gradient solve first.
  prop_inverse  the FREE stand-in, the proportional inverse of this geometry's
                own static kernel, which costs zero additional gradient solves.

Both are reported for every geometry so the free variant can be accepted or
rejected on evidence rather than on convenience.

WHERE THE SOLVED MAP COMES FROM, stated because it is a provenance question.
  * library shapes: `out_lib/<shape>_maps.npz` key `A1_cont`, the shape-fidelity
    campaign's own solved static map. NAMED SIMPLIFICATION: that campaign solved
    against the BINARY RASTER target, while the intake convention is the sub-cell
    area fill, so it is a solved static map for a very slightly different target.
    It is the real artifact the pipeline would warm start from, which is why it
    is used rather than re-solving eighteen shapes.
  * novel shapes: `out_intake/<name>_maps.npz` key `static_cont`, solved in the
    generalization pass against the area-fill target, no such caveat.

Conventions carried: grid 120; uniform arm calibrated to 500 watts per metre;
electrical state B (the state the march runs in from the first update tick
onwards); conductivity channel only. The anisotropy metric is invariant to the
drive scale, so the calibration cannot move a recommendation.

Run:
  ./.venv312/bin/python scripts/analysis/run_intake_classifier_v2.py [name ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import geometry_actuator as ga            # noqa: E402
from adjoint2d import geometry_calibrate as gcal         # noqa: E402
from adjoint2d import geometry_intake as gi              # noqa: E402
from adjoint2d import geometry_symmetry as gs            # noqa: E402
from adjoint2d import library_solve as lib               # noqa: E402
from novel_shapes import NOVEL                           # noqa: E402
from run_intake_library import library_polygon           # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
GRID = 120

# The ground truth the version-2 bands are calibrated against: does the best
# ROTATING arm beat the best SOLVED STATIC arm? Seven measured points.
GROUND_TRUTH_V2 = {
    # CONTINUOUS_ROTATION_REPORT Section 1, whose "best static arm" column IS a
    # solved static map (the joint campaign's winning angle plus its winning
    # map, re-measured on the production engine).
    "square":  {"outcome": "rotation_wins",
                "source": "CONTINUOUS_ROTATION_REPORT S1: solved static 25.56 / "
                          "0.9804 -> rotating 13.93 / 1.0000, -45.5 pct J"},
    "cross":   {"outcome": "rotation_wins",
                "source": "CONTINUOUS_ROTATION_REPORT S1: 147.03 / 0.8514 -> "
                          "34.82 / 0.9866, -76.3 pct J"},
    "star":    {"outcome": "rotation_wins",
                "source": "CONTINUOUS_ROTATION_REPORT S1: 106.28 / 0.7870 -> "
                          "24.46 / 0.9527, -77.0 pct J"},
    "T_shape": {"outcome": "rotation_fails",
                "source": "CONTINUOUS_ROTATION_REPORT S1: 522.28 (HORIZON) / "
                          "0.5356 -> 467.33 / 0.5516, -10.5 pct J with 42.9 pct "
                          "of the part still unmelted; the penalty is NOT erased"},
    "L_shape": {"outcome": "rotation_fails",
                "source": "CONTINUOUS_ROTATION_REPORT S1: 376.94 / 0.6693 -> "
                          "352.97 / 0.6581, better J and worse IoU, a tie"},
    # GEOMETRY_GENERALIZATION_REPORT Section 6, the two novel end-to-end runs.
    "keyhole": {"outcome": "rotation_wins",
                "source": "GEOMETRY_GENERALIZATION_REPORT S6.1: solved static "
                          "4 bpp 168.35 / 0.8163 -> solved continuous 4 bpp "
                          "8.08 / 0.9753"},
    "gear8":   {"outcome": "rotation_fails",
                "source": "GEOMETRY_GENERALIZATION_REPORT S6.2: solved static "
                          "4 bpp 132.36 / 0.8458 BEATS solved continuous 4 bpp "
                          "138.59 / 0.8273; rotation improved the heating and "
                          "still lost to the map"},
}


def solved_static_map(name: str, part_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """The static solved map for this geometry, with its provenance."""
    if name in NOVEL:
        p = OUT / f"{name}_maps.npz"
        key, note = "static_cont", ("solved in the generalization pass against "
                                    "the sub-cell area-fill target")
    else:
        p = OUT_LIB / f"{name}_maps.npz"
        key, note = "A1_cont", ("the shape-fidelity campaign's solved static "
                                "map, solved against the BINARY RASTER target, "
                                "not the area fill")
    if not p.exists():
        raise FileNotFoundError(f"no stored solved static map at {p}")
    with np.load(p) as d:
        s = np.array(d[key], dtype=float)
    if s.shape != part_mask.shape:
        raise ValueError(f"{name}: stored map {s.shape} vs part {part_mask.shape}")
    return s, {"path": str(p), "key": key, "note": note}


def intake_for(name: str):
    if name in NOVEL:
        poly = NOVEL[name]()
    else:
        poly, _cfg = library_polygon(name)
    it = gi.from_polygon(poly, grid=GRID, name=name)
    return gcal.calibrate_intake(it)


def run_one(name: str, log, magnitudes: tuple[float, ...] = (1.0,)) -> dict:
    t0 = time.perf_counter()
    it = intake_for(name)
    rep = gs.analyze(it.chi)
    log(f"symmetry order {rep.rotational_order} ({rep.point_group}), "
        f"drive {it.cfg['electric']['voltage_v']:.2f} V")

    try:
        s_solved, prov = solved_static_map(name, it.part_mask)
    except (FileNotFoundError, ValueError) as exc:
        log(f"NO solved static map ({exc}); the solved variant is skipped")
        s_solved, prov = None, {"missing": str(exc)}

    spec = ga.spectrum_v2_from_cfg(it.cfg, rep.rotational_order,
                                   solved_static_map=s_solved, log=log)

    # magnitude sensitivity of the FREE variant, measured not assumed
    sens: dict = {}
    for m in magnitudes:
        if abs(m - ga.PROP_INVERSE_MAGNITUDE) < 1e-12:
            sens[f"{m:.2f}"] = dict(spec["residual"]["prop_inverse"])
            continue
        sp = ga.spectrum_v2_from_cfg(it.cfg, rep.rotational_order,
                                     solved_static_map=None, prop_magnitude=m)
        sens[f"{m:.2f}"] = dict(sp["residual"]["prop_inverse"])

    res = {
        "shape": name,
        "grid": GRID,
        "is_novel": name in NOVEL,
        "voltage_v": float(it.cfg["electric"]["voltage_v"]),
        "n_part_cells": int(it.part_mask.sum()),
        "symmetry": rep.as_json(),
        "modes": list(ga.mode_names(rep.rotational_order)),
        "residual": {k: dict(v) for k, v in spec["residual"].items()},
        "prop_inverse_magnitude": ga.PROP_INVERSE_MAGNITUDE,
        "prop_inverse_magnitude_sensitivity": sens,
        "solved_map_provenance": prov,
        "ground_truth": GROUND_TRUTH_V2.get(name),
        "wall_s": time.perf_counter() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_v2.json").write_text(json.dumps(res, indent=2, default=float))
    np.savez_compressed(OUT / f"{name}_v2.npz", chi=it.chi,
                        part_mask=it.part_mask, x=it.x, y=it.y,
                        prop_inverse_map=spec["prop_inverse_map"],
                        **({"solved_static_map": s_solved} if s_solved is not None
                           else {}),
                        **{f"kernel_{k}": v for k, v in spec["kernels"].items()})
    log(f"done in {res['wall_s']:.0f} s")
    return res


def main(names: list[str]) -> None:
    t0 = time.perf_counter()
    for n in names:
        def log(msg, _n=n):
            print(f"[{_n}] {msg}", flush=True)
        try:
            run_one(n, log, magnitudes=(0.50, 1.00, 1.50))
        except Exception as exc:                      # noqa: BLE001
            print(f"[{n}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            (OUT / f"{n}_v2_FAILED.json").write_text(
                json.dumps({"shape": n, "error": f"{type(exc).__name__}: {exc}"},
                           indent=2))
    print(f"[classifier-v2] {len(names)} geometries in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(list(sys.argv[1:]) or (list(lib.SHAPES) + ["gear8", "keyhole"]))
