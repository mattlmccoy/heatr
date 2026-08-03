"""Phase E: recompute the shape metrics on SHAPE-RELATIVE evaluation planes.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.rescore --shape pyramid

WHY THIS EXISTS. The shared evaluation grid (solve3d/gates.EVAL_Z_M) fixes its
z-planes at -20, -10, 0, +10, +20 mm, which was right for the Phase A/C anchors
-- full-height extrusions spanning the whole 60 mm chamber. The Phase E shapes
are compact: the pyramid spans only +-11.62 mm and the cube +-8.06 mm. So most
of those planes fall OUTSIDE the part, their analytic nominal is empty, and the
first scoring pass silently produced no IoU at all (the key list was derived
from an empty plane).

Fixed two ways, both necessary:
  * planes are placed SHAPE-RELATIVE, at fractions of the part's own z-extent,
    so every plane cuts real material;
  * the aggregation derives its keys from the first NON-EMPTY plane and averages
    only over planes whose nominal is non-empty, so an empty plane can never
    again delete a verdict-carrying metric by accident.

No forward is re-run. score_arm stores the nodal read state T_read, and the
mesh is deterministic (fixed gmsh seed), so the metrics are recomputed exactly
by re-evaluating that stored state on the new planes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solve3d import forward as fwd, gates as sg, shape_metrics as sm
from solve3d.phase_e import geometry as geo, run as R

RESULTS = Path(__file__).resolve().parent / "results"
Z_FRACTIONS = (-0.40, -0.20, 0.0, 0.20, 0.40)   # of the part's own half-height


def z_planes(shape: str) -> list[float]:
    half = geo.PYR_H_M / 2.0 if shape == "pyramid" else geo.CUBE_A_M / 2.0
    return [f * 2.0 * half for f in Z_FRACTIONS]


def eval_points(shape: str):
    x, y, _, h = sg.eval_grid_axes()
    zs = z_planes(shape)
    Z, X, Y = np.meshgrid(np.asarray(zs), x, y, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts, (len(zs), x.size, y.size), h, zs


def metrics_from_T(shape: str, Te: np.ndarray, zs, h: float) -> dict:
    per, used = [], []
    for i, zc in enumerate(zs):
        nom = geo.nominal_mask_2d(shape, z=zc)
        if not nom.any():
            continue
        p2 = sg.phase_fraction_phi(Te[i])
        row = {}
        for t in (0.8, 0.9):
            m = p2 >= t
            k = f"phi{t:g}".replace(".", "p")
            row[f"iou_{k}"] = sm.iou(m, nom)
            row[f"in_part_{k}"] = sm.in_part_melt_fraction(m, nom)
            row[f"out_of_part_{k}"] = sm.out_of_part_fraction(m, nom)
        row["front_ssd_mm"] = sm.symmetric_surface_distance_mm(p2 >= 0.9, nom, h)
        per.append(row)
        used.append(float(zc))
    if not per:
        raise ValueError(f"{shape}: no evaluation plane cut the part")
    keys = list(per[0])
    agg = {k: float(np.nanmean([r[k] for r in per])) for k in keys}
    agg.update({k + "__plane_spread":
                float(np.nanmax([r[k] for r in per]) - np.nanmin([r[k] for r in per]))
                for k in keys})
    agg["z_planes_used_m"] = used
    agg["n_planes_used"] = len(per)
    return agg


def rescore(shape: str) -> dict:
    tc = R.build_case(shape)
    pts, shp, h, zs = eval_points(shape)
    W = fwd.functionspace(tc.msh, ("Lagrange", 1))
    p = RESULTS / f"phase_e_{shape}.json"
    doc = json.loads(p.read_text())
    for name, rec in doc["arms"].items():
        if name == "_mesh" or rec.get("status") == "DROPPED":
            continue
        f = RESULTS / f"field_{shape}_{name}.npz"
        if not f.exists():
            continue
        T_read = np.asarray(np.load(f)["T_read"], dtype=float)
        Tf = fwd.fem.Function(W)
        Tf.x.array[:] = T_read.astype(fwd.dolfinx.default_scalar_type)
        Te, missed = fwd.eval_at(Tf, tc.msh, pts)
        rec.update(metrics_from_T(shape, Te.reshape(shp), zs, h))
        rec["eval_missed_rescore"] = int(missed)
        rec["rescored_on_shape_relative_planes"] = True
        np.savez_compressed(RESULTS / f"fieldz_{shape}_{name}.npz",
                            T_eval=Te.reshape(shp), z_planes=np.asarray(zs))
        print(f"  rescored {name}: iou0.9={rec['iou_phi0p9']:.5f} "
              f"front={rec['front_ssd_mm']:.4f}mm "
              f"bed={rec['out_of_part_phi0p9']:.5f}", flush=True)
    doc["eval_planes"] = {"z_fractions_of_extent": list(Z_FRACTIONS),
                          "z_planes_m": zs,
                          "why": "the shared grid's fixed +-20/10/0 mm planes "
                                 "mostly miss these compact shapes"}
    p.write_text(json.dumps(doc, indent=1, default=float))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    rescore(ap.parse_args().shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
