"""S2 addendum: is the cornered-shape non-convergence a SOLVER failure, or is
the voxelized part a different physical object at each grid?

    ./.venv312/bin/python -m heatr3d_s2.commensurability

heatr3d's part mask is `|x| <= half` evaluated at CELL CENTRES, which sit at
(i + 0.5)h - L/2. For an AXIS-ALIGNED boundary the whole edge therefore snaps
coherently to the nearest centre, and the voxelized half-width is
    half_eff = ceil_count * h / 2
which equals the nominal half-width only when (half/h - 0.5) is a half-integer.
A CURVED boundary has no such coherent snap: its staircase error averages out
around the perimeter, which is why the circle is immune.

If the metric changes track the GEOMETRY jump between grids, then the cornered
shapes' apparent non-convergence is not the solver failing to converge -- it is
a different part being solved at each grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import heatr3d
from heatr3d_s2 import harness
from solve3d import gates as sg
from solve3d import shape_metrics as sm

R = Path(__file__).resolve().parent / "results"
HALF_M = harness.PART_DIAM_M / 2.0


def geometry_table(grids=(48, 64, 80, 96)) -> list[dict]:
    out = []
    for n in grids:
        g = heatr3d.Grid(n=n)
        x = g.x
        ncov = int(np.sum(np.abs(x) <= HALF_M))
        half_eff = ncov * g.h / 2.0
        out.append({"n": n, "h_m": g.h,
                    "cells_across": ncov,
                    "half_width_effective_m": half_eff,
                    "half_width_error_m": half_eff - HALF_M,
                    "half_width_error_rel": (half_eff - HALF_M) / HALF_M,
                    "centre_offset_frac": float((HALF_M / g.h - 0.5) % 1.0),
                    "commensurate": bool(abs(half_eff - HALF_M) < 1e-12)})
    return out


def pair_metric(shape: str, a: int, b: int, read: str = "melt_onset") -> dict:
    fa = np.asarray(np.load(R / f"field_{shape}_n{a}.npz")[read])
    fb = np.asarray(np.load(R / f"field_{shape}_n{b}.npz")[read])
    _, _, _, h = sg.eval_grid_axes()
    j9, ssd = [], []
    for i in range(fa.shape[0]):
        pa, pb = sg.phase_fraction_phi(fa[i]), sg.phase_fraction_phi(fb[i])
        j9.append(1.0 - sm.iou(pa >= 0.9, pb >= 0.9))
        ssd.append(sm.symmetric_surface_distance_mm(pa >= 0.9, pb >= 0.9, h))
    return {"pair": [a, b], "jaccard_dist_phi0p9": float(np.nanmean(j9)),
            "front_ssd_mm": float(np.nanmean(ssd))}


def build(grids=(48, 64, 80, 96)) -> dict:
    geo = geometry_table(grids)
    err = {r["n"]: r["half_width_error_m"] for r in geo}
    doc = {"what": "S2 addendum: axis-aligned staircase commensurability",
           "geometry": geo,
           "commensurate_grids": [r["n"] for r in geo if r["commensurate"]],
           "shapes": {}}
    for shape in ("square", "lshape", "circle"):
        succ = []
        for a, b in zip(grids, grids[1:]):
            m = pair_metric(shape, a, b)
            jump = abs(err[b] - err[a])
            m["geometry_jump_m"] = jump
            m["metric_per_mm_of_geometry_jump"] = (
                m["jaccard_dist_phi0p9"] / (jump * 1e3) if jump > 0 else None)
            succ.append(m)
        entry = {"successive_pairs": succ}
        comm = [r["n"] for r in geo if r["commensurate"]]
        if len(comm) >= 2:
            entry["commensurate_pair"] = pair_metric(shape, comm[0], comm[-1])
            entry["commensurate_pair"]["geometry_jump_m"] = abs(
                err[comm[-1]] - err[comm[0]])
        doc["shapes"][shape] = entry
    (R / "commensurability.json").write_text(json.dumps(doc, indent=1))
    return doc


if __name__ == "__main__":
    d = build()
    print("commensurate grids:", d["commensurate_grids"])
    for s, e in d["shapes"].items():
        print(f"\n{s}:")
        for m in e["successive_pairs"]:
            r = m["metric_per_mm_of_geometry_jump"]
            print(f"   {m['pair']} jaccard={m['jaccard_dist_phi0p9']:.5f} "
                  f"geom_jump={m['geometry_jump_m']*1e3:.4f}mm "
                  f"ratio={'n/a' if r is None else round(r,4)}")
        c = e.get("commensurate_pair")
        if c:
            print(f"   COMMENSURATE {c['pair']} jaccard={c['jaccard_dist_phi0p9']:.5f} "
                  f"front={c['front_ssd_mm']:.5f}mm geom_jump={c['geometry_jump_m']*1e3:.4f}mm")
