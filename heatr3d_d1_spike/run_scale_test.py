"""D1 Task 4: the n=200-equivalent EQS solve heatr3d cannot do.

RUNS IN THE SPIKE ENV:
    heatr3d_d1_spike/env/bin/python heatr3d_d1_spike/run_scale_test.py

The case: the SAME extruded circle (d = 20 mm, full height, 60 mm bed) as Task
2, meshed at an in-part element size of 0.3 mm = L/200 -- the element size that
matches a heatr3d n=200 voxel. heatr3d cannot run that grid: the direct complex
LU at N = 8.0e6 SIGSEGVs the interpreter and is estimated at ~29.8 GB
(heatr3d.py EQS-01 / _direct_lu_memory_estimate_gb).

Why the FEM needs far fewer unknowns for the same in-part resolution: only the
part carries the 0.3 mm size field; the powder bed is graded to
lc_bed_factor * lc_part, so the 8.0e6 uniform voxels collapse to a mesh whose
dof count is recorded below.

Gates (plan Task 4):
  * completes on this machine, peak RSS < 34 GB   -> recorded, not assumed
  * mid-plane fields consistent with Task 2's FINE dolfinx solution within 5 %
    (unit-mean pattern relative L2), evaluated at the IDENTICAL points Task 2
    used (the heatr3d n=96 in-part mid-plane voxel centres, replayed from
    d1_circle_fine.npz) -- so this is an FEM-vs-FEM self-convergence check, not
    a cross-engine comparison, and the voxel-engine artifacts play no part.
  * wall time (mesh generation and solve separately) recorded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

import eqs_common as ec           # applies jit_fix before dolfinx
import femutils as fu
import mesh_gmsh as mg
import metrics as M

HERE = Path(__file__).resolve().parent
R_PART = 0.010
SURFACE_BAND_H = 1.5

N_EQUIV = 200
LC_PART_M = ec.L_DOMAIN / N_EQUIV          # 0.3 mm
RSS_GATE_GB = 34.0
PATTERN_GATE = 0.05

KSP_ITER = {"ksp_type": "gmres", "pc_type": "gamg",
            "ksp_rtol": "1e-10", "ksp_max_it": "500",
            "ksp_error_if_not_converged": "true"}


def in_cylinder(mp: np.ndarray) -> np.ndarray:
    return np.sqrt(mp[0] ** 2 + mp[1] ** 2) <= R_PART


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc-part-m", type=float, default=LC_PART_M)
    ap.add_argument("--lc-bed-factor", type=float, default=4.0)
    args = ap.parse_args()

    rec: dict = {
        "engine": "dolfinx (P1 CG, conforming tets)",
        "scalar_path": ec.SCALAR_PATH,
        "shape": "extruded circle d=20mm, full height",
        "target": {"n_equiv": N_EQUIV,
                   "voxel_h_m": ec.L_DOMAIN / N_EQUIV,
                   "lc_part_m": args.lc_part_m,
                   "lc_bed_factor": args.lc_bed_factor,
                   "heatr3d_uniform_unknowns_at_n200": N_EQUIV ** 3},
        "petsc_options": KSP_ITER,
        "gate": {"rss_lt_gb": RSS_GATE_GB, "pattern_rel_l2_lt": PATTERN_GATE},
    }

    print(f"[task4] meshing lc_part={args.lc_part_m*1e3:.4f} mm ...", flush=True)
    t0 = time.perf_counter()
    msh, info = mg.build("cylinder", args.lc_part_m,
                         lc_bed_factor=args.lc_bed_factor)
    t_mesh = time.perf_counter() - t0
    print(f"[task4] mesh: {info.n_cells_total} cells, {info.n_nodes_total} nodes, "
          f"{info.n_nodes_in_part} in-part nodes, {t_mesh:.1f} s "
          f"(RSS {fu.peak_rss_gb():.2f} GB)", flush=True)

    mats = ec.materials(msh, in_part=in_cylinder)
    import ufl
    from dolfinx import fem
    v_part = float(np.real(MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(mats.doped * ufl.dx)), op=MPI.SUM)))

    print("[task4] solving (GMRES + GAMG) ...", flush=True)
    t0 = time.perf_counter()
    Vr, Vi = ec.solve_eqs(msh, mats, petsc_options=KSP_ITER)
    t_solve = time.perf_counter() - t0
    print(f"[task4] solve: {t_solve:.1f} s (RSS {fu.peak_rss_gb():.2f} GB)",
          flush=True)

    t0 = time.perf_counter()
    q, scale, p_target = ec.qrf_dg0(msh, Vr, Vi, mats, premix=False)
    e = fu.emag_dg0(Vr, Vi, mats.dg0)
    t_post = time.perf_counter() - t0

    rec.update({
        "lc_part_m": info.lc_part, "lc_bed_m": info.lc_bed,
        "n_dofs_total": info.n_nodes_total,
        "n_cells_total": info.n_cells_total,
        "n_nodes_in_part": info.n_nodes_in_part,
        "dof_reduction_vs_n200_voxels": N_EQUIV ** 3 / info.n_nodes_total,
        "part_volume_m3": info.part_volume_m3,
        "part_volume_rel_err_vs_analytic":
            info.part_volume_m3 / (np.pi * R_PART ** 2 * ec.L_DOMAIN) - 1.0,
        "doped_volume_assembled_m3": v_part,
        "wall_mesh_s": t_mesh, "wall_solve_s": t_solve, "wall_post_s": t_post,
        "peak_rss_gb": fu.peak_rss_gb(),
        "p_target_w": p_target,
        "v_range": {"vr_min": float(np.real(Vr.x.array).min()),
                    "vr_max": float(np.real(Vr.x.array).max())},
    })

    # ---- pattern comparison against Task 2's FINE dolfinx solution ---------
    fine = np.load(HERE / "d1_circle_fine.npz")
    pts = fine["pts"]
    q_ref, e_ref, interior = fine["q_fem"], fine["e_fem"], fine["interior"]
    band = ~interior
    q_s, miss_q = fu.eval_points(q, msh, pts)
    e_s, miss_e = fu.eval_points(e, msh, pts)

    def three_ways(a, b):
        return {"all": M.rel_l2_pattern(a, b),
                "interior": M.rel_l2_pattern(a[interior], b[interior]),
                "surface_band": M.rel_l2_pattern(a[band], b[band])}

    cmp = {
        "reference": "d1_circle_fine.npz (Task 2 dolfinx fine, lc_part = "
                     "0.58 mm, matched to heatr3d n=96 in-part node count)",
        "points": "heatr3d n=96 in-part mid-plane voxel centres (identical to "
                  "the points Task 2 compared on)",
        "n_points": int(pts.shape[0]),
        "n_points_interior": int(interior.sum()),
        "n_points_surface_band": int(band.sum()),
        "eval_missed_q": miss_q, "eval_missed_e": miss_e,
        "qrf_pattern_rel_l2_vs_task2_fine": three_ways(q_s, q_ref),
        "emag_pattern_rel_l2_vs_task2_fine": three_ways(e_s, e_ref),
        "qrf_abs_mean_ratio_scale_over_fine": float(np.nanmean(q_s) / np.nanmean(q_ref)),
        "uniformity_cv": {"scale": float(np.nanstd(q_s) / np.nanmean(q_s)),
                          "task2_fine": float(np.nanstd(q_ref) / np.nanmean(q_ref)),
                          "note": "the infinite-cylinder limit is a UNIFORM "
                                  "interior field; a silently failed solve "
                                  "would not reproduce this CV"},
        "power_fraction_in_surface_band": {
            "scale": float(np.nansum(q_s[band]) / np.nansum(q_s)),
            "task2_fine": float(np.nansum(q_ref[band]) / np.nansum(q_ref)),
            "area_fraction": float(band.sum() / band.size)},
    }
    rec["comparison"] = cmp

    l2_all = cmp["qrf_pattern_rel_l2_vs_task2_fine"]["all"]
    rec["gate"].update({
        "completed": True,
        "rss_ok": bool(rec["peak_rss_gb"] < RSS_GATE_GB),
        "peak_rss_gb": rec["peak_rss_gb"],
        "qrf_pattern_rel_l2_all": l2_all,
        "pattern_ok": bool(l2_all < PATTERN_GATE),
        "gate_ok": bool(rec["peak_rss_gb"] < RSS_GATE_GB and l2_all < PATTERN_GATE),
    })

    np.savez_compressed(HERE / "d1_circle_scale.npz",
                        pts=pts, q_fem=q_s, e_fem=e_s, interior=interior)

    p = HERE / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d["task4"] = rec
    p.write_text(json.dumps(d, indent=1))
    print(json.dumps(rec["gate"], indent=1), flush=True)
    print(json.dumps(cmp["qrf_pattern_rel_l2_vs_task2_fine"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
