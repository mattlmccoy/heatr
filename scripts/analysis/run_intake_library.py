#!/usr/bin/env python3
"""Run the GEOMETRY INTAKE on the eighteen standardized library shapes.

This is the validation pass for the intake and for the actuator classifier. It
does two separate things and reports them separately.

1. INTAKE FIDELITY, a real-data gate. Each library shape's polygon is taken
   from the production domain builder, pushed through `geometry_intake` as if
   it were an imported outline, and the result is compared against the shape's
   own stored calibrated configuration: the part mask must match cell for cell
   and the automatically calibrated drive voltage must match the stored one.
   Nothing about the shape's identity is used; the intake sees a vertex list.

2. THE CLASSIFIER. For each shape the symmetry analysis gives the rotational
   order and the mirror axes, the anisotropy spectrum gives the residual
   anisotropy of every applicable actuator mode, and the classifier emits an
   actuator recommendation. Those recommendations are then compared against the
   campaign's own MEASURED outcomes where they exist (the five shapes of
   `CONTINUOUS_ROTATION_REPORT.md`), and reported as predictions with the
   static-solve class from `out_lib` as context where they do not.

Run:
  ./.venv312/bin/python scripts/analysis/run_intake_library.py [shape ...]
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

from adjoint2d import geometry_actuator as ga            # noqa: E402
from adjoint2d import geometry_calibrate as gcal         # noqa: E402
from adjoint2d import geometry_intake as gi              # noqa: E402
from adjoint2d import geometry_symmetry as gs            # noqa: E402
from adjoint2d import library_solve as lib               # noqa: E402
from adjoint2d.pins import build_case, load_cfg          # noqa: E402
from adjoint2d.prod import rfam                          # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
GRID = 120

# The campaign's own measured rotation outcomes, quoted with their source.
KNOWN_ROTATION_OUTCOME = {
    "square": "rotation_wins", "cross": "rotation_wins", "star": "rotation_wins",
    "T_shape": "rotation_fails", "L_shape": "rotation_fails",
}
# The rotational order each shape is DEFINED with in shapes.py, for the
# symmetry-detector check. None where the definition does not fix one.
EXPECTED_ORDER = {
    "square": 4, "circle": None, "hexagon": 6, "triangle": 1,
    "equilateral_triangle": 3, "L_shape": 1, "H_shape": 2, "T_shape": 1,
    "cross": 4, "diamond": 4, "ellipse": 2, "octagon": 8, "pentagon": 5,
    "rectangle": 2, "rounded_rect": None, "star": 5, "star6": 6,
    "trapezoid": 1,
}


def library_polygon(shape: str) -> tuple[np.ndarray, dict]:
    """The shape's polygon, straight from the production domain builder."""
    cfg = load_cfg(lib.shape_config(shape))
    x, y = gi.grid_axes(GRID, float(cfg["geometry"]["chamber_x"]))
    part = dict(cfg["geometry"]["part"])
    poly, _mask, _fill = rfam._single_part_mask_and_fill(x, y, part)
    return np.asarray(poly, dtype=float), cfg


def run_shape(shape: str, log) -> dict:
    t0 = time.perf_counter()
    poly, cfg_stored = library_polygon(shape)
    it = gi.from_polygon(poly, grid=GRID, name=shape)

    case_stored = build_case(cfg_stored)
    mask_match = bool(np.array_equal(case_stored.part_mask, it.part_mask))
    n_diff = int(np.sum(case_stored.part_mask != it.part_mask))

    it = gcal.calibrate_intake(it)
    v_stored = float(cfg_stored["electric"]["voltage_v"])
    v_new = float(it.cfg["electric"]["voltage_v"])
    v_rel = abs(v_new - v_stored) / v_stored

    rep = gs.analyze(it.chi)
    spec = ga.spectrum_from_cfg(it.cfg, rep.rotational_order, log=log)
    rec = ga.recommend(spec["residual"], rep.rotational_order)

    log(f"  order {rep.rotational_order} ({rep.point_group}), "
        f"mirrors {len(rep.mirror_axes_deg)}, mask match {mask_match}, "
        f"voltage {v_new:.2f} vs stored {v_stored:.2f} ({v_rel*100:.4f} pct)")
    log(f"  RECOMMENDATION {rec.actuator_class} via {rec.mode} "
        f"(residual {rec.residual:.4f}, reduction {rec.reduction_factor:.2f})")

    res = {
        "shape": shape,
        "intake": {k: v for k, v in it.info.items() if k != "raster_vs_area"},
        "raster_vs_area": it.info["raster_vs_area"],
        "part_mask_matches_stored_config": mask_match,
        "n_cells_differing": n_diff,
        "voltage_stored_v": v_stored,
        "voltage_calibrated_v": v_new,
        "voltage_rel_error": v_rel,
        "symmetry": rep.as_json(),
        "expected_order": EXPECTED_ORDER.get(shape),
        "anisotropy": {k: float(v) for k, v in spec["residual"].items()},
        "recommendation": rec.as_json(),
        "known_rotation_outcome": KNOWN_ROTATION_OUTCOME.get(shape),
        "wall_s": time.perf_counter() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{shape}_intake.json").write_text(json.dumps(res, indent=2, default=float))
    np.savez_compressed(OUT / f"{shape}_intake.npz", chi=it.chi,
                        part_mask=it.part_mask, x=it.x, y=it.y,
                        **{f"kernel_{k}": v for k, v in spec["kernels"].items()})
    return res


def main(shapes: list[str]) -> None:
    t0 = time.perf_counter()
    for s in shapes:
        def log(msg, _s=s):
            print(f"[{_s}] {msg}", flush=True)
        try:
            run_shape(s, log)
        except Exception as exc:                      # noqa: BLE001
            print(f"[{s}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            (OUT / f"{s}_intake_FAILED.json").write_text(
                json.dumps({"shape": s, "error": f"{type(exc).__name__}: {exc}"},
                           indent=2))
    print(f"[intake-library] {len(shapes)} shapes in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(list(sys.argv[1:]) or list(lib.SHAPES))
