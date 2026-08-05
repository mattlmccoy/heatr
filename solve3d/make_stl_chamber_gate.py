"""Emit solve3d/results/stl_chamber_gate.json: the chamber-embedding gates.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.make_stl_chamber_gate

Every number in TRANCHE1 follow-up reporting is read from this file rather than
retyped, and the equivalence band is read from dolfinx_refinement.json rather
than chosen here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solve3d import forward as fwd, precomp, stl_mesh
from solve3d.phase_e import geometry as geo
from solve3d.phase_e import run_tamper as rt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "solve3d" / "results" / "stl_chamber_gate.json"
SAFETY = 1.5
LC = 1.5e-3


def band() -> float:
    d = json.loads((ROOT / "solve3d" / "results"
                    / "dolfinx_refinement.json").read_text())["spreads"]
    return SAFETY * float(d["t90_rel_spread"])


def feature_angle_sweep(path, angles=(40.0, 5.0, 2.0, 1.0)) -> list:
    rows = []
    for a in angles:
        try:
            msh, i = stl_mesh.build_mesh_from_stl(
                path, lc_part=LC, with_chamber=True, L=fwd.L_DOMAIN,
                precomp_coeffs=None, feature_angle_deg=a, volume_rel_tol=1e9)
            h = (6.0 * stl_mesh._cell_volumes(msh)) ** (1.0 / 3.0)
            rows.append({"feature_angle_deg": a,
                         "part_volume_rel_err_vs_stl":
                             float(i.part_volume_rel_err_vs_stl),
                         "n_cells": int(i.n_cells_total),
                         "h_min_mm": float(h.min() * 1e3),
                         "h_median_mm": float(np.median(h) * 1e3),
                         "refused_at_shipped_tol": bool(
                             abs(i.part_volume_rel_err_vs_stl)
                             > stl_mesh.VOLUME_REL_TOL)})
        except stl_mesh.MeshRefusal as e:
            rows.append({"feature_angle_deg": a, "refused": str(e)[:200]})
    return rows


def main() -> int:
    b = band()
    doc = {
        "what": ("chamber (bed) embedding for arbitrary STL parts -- the "
                 "TRANCHE1_REPORT.md named blocker, and its gates"),
        "root_cause": ("_add_box_surface_loop created each of the box's 12 "
                       "edges twice (gmsh.model.geo.addLine does not "
                       "deduplicate), so the two faces meeting at an edge "
                       "meshed independent copies and the shell was not "
                       "conforming with itself. The EMPTY box failed "
                       "identically with no STL part present, which is what "
                       "shows the Tranche 1 attribution to the discrete "
                       "entity was wrong."),
        "equivalence_band": b,
        "band_rule": ("1.5 x MAX of the dolfinx own-refinement pair spreads "
                      "(t90_rel_spread); the Phase A same-engine band"),
        "band_source": "solve3d/results/dolfinx_refinement.json",
        "lc_part_m": LC,
        "L_chamber_m": fwd.L_DOMAIN,
        "feature_angle_deg_default": stl_mesh.FEATURE_ANGLE_DEG,
        "volume_rel_tol": stl_mesh.VOLUME_REL_TOL,
    }

    # 1. the box shell, the actual root cause
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("box")
        loop = stl_mesh._add_box_surface_loop(gmsh, fwd.L_DOMAIN)
        v = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [v], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.008)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.012)
        gmsh.model.mesh.generate(3)
        doc["box_shell"] = {
            "n_points": len(gmsh.model.getEntities(0)),
            "n_curves": len(gmsh.model.getEntities(1)),
            "n_curves_before_fix": 24,
            "n_surfaces": len(gmsh.model.getEntities(2)),
            "empty_box_tets": len(gmsh.model.mesh.getElements(3)[1][0])}
    finally:
        gmsh.finalize()

    # 2. OCC vs STL, both chamber-embedded, same lc and same chamber
    _mo, io = geo.build_mesh("pyramid", lc_part=LC, L=fwd.L_DOMAIN,
                             precomp_coeffs=None)
    _ms, isl = stl_mesh.build_mesh_from_stl(
        ROOT / "shape_library_3d" / "stl" / "pyramid.stl", lc_part=LC,
        with_chamber=True, L=fwd.L_DOMAIN, precomp_coeffs=None)
    rel = isl.part_volume_m3 / io.part_volume_m3 - 1.0
    doc["occ_vs_stl_chamber"] = {
        "shape": "pyramid",
        "occ_part_volume_m3": float(io.part_volume_m3),
        "stl_part_volume_m3": float(isl.part_volume_m3),
        "rel_diff": float(rel),
        "band": b,
        "pass": bool(abs(rel) <= b),
        "stl_part_volume_rel_err_vs_stl":
            float(isl.part_volume_rel_err_vs_stl),
        "occ_n_cells": int(io.n_cells_total),
        "stl_n_cells": int(isl.n_cells_total),
        "stl_n_part_cells": int(isl.n_part_cells),
        "stl_n_bed_cells": int(isl.n_bed_cells)}

    # 3. L0 reaches the chamber mesh
    c = precomp.load_defaults()
    _m0, i0 = stl_mesh.build_mesh_from_stl(
        ROOT / "shape_library_3d" / "stl" / "pyramid.stl", lc_part=2.0e-3,
        with_chamber=True, L=fwd.L_DOMAIN,
        precomp_coeffs=precomp.ShrinkageL0(0.0, 0.0))
    _m1, i1 = stl_mesh.build_mesh_from_stl(
        ROOT / "shape_library_3d" / "stl" / "pyramid.stl", lc_part=2.0e-3,
        with_chamber=True, L=fwd.L_DOMAIN, precomp_coeffs=c)
    doc["level0_in_chamber"] = {
        "expected_volume_growth": float(c.xy_scale ** 2 * c.z_scale),
        "measured_stl_volume_growth":
            float(i1.stl_volume_m3 / i0.stl_volume_m3),
        "measured_mesh_volume_growth":
            float(i1.part_volume_m3 / i0.part_volume_m3),
        "coefficients": i1.precomp}

    # 4. the Tamper, a genuinely arbitrary non-library part
    v, f = stl_mesh.load_stl(rt.TAMPER_STL)
    e = np.concatenate([v[f[:, 0]] - v[f[:, 1]], v[f[:, 1]] - v[f[:, 2]],
                        v[f[:, 2]] - v[f[:, 0]]])
    el = np.linalg.norm(e, axis=1)
    vp, fp = stl_mesh.load_stl(ROOT / "shape_library_3d" / "stl"
                               / "pyramid.stl")
    ep = np.concatenate([vp[fp[:, 0]] - vp[fp[:, 1]],
                         vp[fp[:, 1]] - vp[fp[:, 2]],
                         vp[fp[:, 2]] - vp[fp[:, 0]]])
    _mt, it = stl_mesh.build_mesh_from_stl(
        rt.TAMPER_STL, lc_part=LC, with_chamber=True, L=fwd.L_DOMAIN,
        precomp_coeffs=None)
    ht = (6.0 * stl_mesh._cell_volumes(_mt)) ** (1.0 / 3.0)
    doc["tamper"] = {
        "stl": str(rt.TAMPER_STL),
        "n_facets": int(f.shape[0]),
        "bbox_extent_mm": [float(x) for x in (v.max(axis=0) - v.min(axis=0))],
        "stl_min_edge_mm": float(el.min()),
        "stl_median_edge_mm": float(np.median(el)),
        "pyramid_min_edge_mm": float(np.linalg.norm(ep, axis=1).min()),
        "part_volume_rel_err_vs_stl": float(it.part_volume_rel_err_vs_stl),
        "n_cells": int(it.n_cells_total),
        "n_part_cells": int(it.n_part_cells),
        "n_bed_cells": int(it.n_bed_cells),
        "h_min_mm": float(ht.min() * 1e3),
        "h_median_mm": float(np.median(ht) * 1e3),
        "note": ("the STL's OWN tessellation carries 0.1 mm edges, so any "
                 "conforming mesh of it inherits ~0.1 mm elements and the "
                 "explicit march's CFL step follows, independently of the "
                 "requested lc_part. Phase E's analytic primitives have no "
                 "such feature (pyramid min edge 23 mm).")}
    doc["tamper_feature_angle_sweep"] = feature_angle_sweep(rt.TAMPER_STL)
    doc["pyramid_feature_angle_sweep"] = feature_angle_sweep(
        ROOT / "shape_library_3d" / "stl" / "pyramid.stl")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
