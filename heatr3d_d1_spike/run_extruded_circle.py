"""D1 Task 2: extruded circle (d = 20 mm, full height) in dolfinx vs heatr3d.

RUNS IN THE SPIKE ENV:
    heatr3d_d1_spike/env/bin/python heatr3d_d1_spike/run_extruded_circle.py

Prerequisite: run_heatr3d_reference.py --shape cylinder --n 64 96 (geo-prewarp
venv) has written ref_heatr3d_cylinder_n{64,96}.npz and results.json
["task2_ref"].

Comparison discipline (Task 0/1 caveats baked in):
  * Q_rf is compared as a UNIT-MEAN PATTERN over the in-part mid-plane voxel
    centres. This is scale-free, so the electrode-gauge difference (heatr3d's
    cell-centred electrodes span L-h, the FEM's span L) cannot contaminate it.
  * The gauge itself is then CONFIRMED separately from the |E| mean ratio,
    which must equal (1 - 1/n) to within the discretization noise if the
    offset is pure gauge and nothing else.
  * heatr3d harmonic face-averaging vs FEM DG0 gamma differs only in a
    boundary layer at the part surface, so every pattern metric is reported
    THREE ways: all in-part points, interior points (> 1.5 h from the
    surface), and the surface band -- the disagreement is localized, not
    averaged away.
"""
from __future__ import annotations

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
SURFACE_BAND_H = 1.5              # a "surface" point is within 1.5 h of r = R


def in_cylinder(mp: np.ndarray) -> np.ndarray:
    """mp is (3, ncell) midpoints."""
    return np.sqrt(mp[0] ** 2 + mp[1] ** 2) <= R_PART


# PETSc options. A direct complex LU is 300x slower here than AMG-preconditioned
# GMRES at the same accuracy (benchmarked: 334 s vs 1.0 s at 57 k dofs, agreeing
# to 5.6e-8 relative), so the iterative path is the default and the LU is kept
# as an explicit cross-check on the coarse level.
KSP_ITER = {"ksp_type": "gmres", "pc_type": "gamg",
            "ksp_rtol": "1e-10", "ksp_max_it": "500"}
KSP_LU = {"ksp_type": "preonly", "pc_type": "lu"}


def solve_case(target_nodes: int, lc0: float, verify_lu: bool = False) -> dict:
    t0 = time.perf_counter()
    msh, info, hist = mg.match_lc("cylinder", target_nodes, lc0)
    t_mesh = time.perf_counter() - t0

    mats = ec.materials(msh, in_part=in_cylinder)
    import ufl
    from dolfinx import fem
    v_part = float(np.real(MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(mats.doped * ufl.dx)), op=MPI.SUM)))

    t0 = time.perf_counter()
    Vr, Vi = ec.solve_eqs(msh, mats, petsc_options=KSP_ITER)
    t_solve = time.perf_counter() - t0

    lu_check = None
    if verify_lu:
        t0 = time.perf_counter()
        Vr2, Vi2 = ec.solve_eqs(msh, mats, petsc_options=KSP_LU)
        lu_check = {
            "wall_lu_s": time.perf_counter() - t0,
            "max_dV_over_860": float(max(
                np.abs(np.real(Vr.x.array - Vr2.x.array)).max(),
                np.abs(np.real(Vi.x.array - Vi2.x.array)).max()) / ec.V_LO),
            "speedup_lu_over_iterative": None}
        lu_check["speedup_lu_over_iterative"] = lu_check["wall_lu_s"] / t_solve

    t0 = time.perf_counter()
    q, scale, p_target = ec.qrf_dg0(msh, Vr, Vi, mats, premix=False)
    e = fu.emag_dg0(Vr, Vi, mats.dg0)
    t_post = time.perf_counter() - t0

    return {"msh": msh, "info": info, "hist": hist, "Vr": Vr, "Vi": Vi,
            "q": q, "e": e, "mats": mats, "v_part": v_part,
            "p_target": p_target, "scale": scale, "lu_check": lu_check,
            "wall_mesh_s": t_mesh, "wall_solve_s": t_solve,
            "wall_post_s": t_post}


def compare(case: dict, ref_npz: Path, n_ref: int) -> dict:
    z = np.load(ref_npz)
    x, y, z_mid = z["x"], z["y"], float(z["z_mid"])
    part_mid, Q_ref, E_ref = z["part_mid"], z["Q_mid"], z["Emag_mid"]
    h = float(z["h"])

    pts, sel = fu.voxel_plane_points(x, y, z_mid, part_mid)
    msh = case["msh"]
    q_d, miss_q = fu.eval_points(case["q"], msh, pts)
    e_d, miss_e = fu.eval_points(case["e"], msh, pts)
    q_h = Q_ref[sel]
    e_h = E_ref[sel]

    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    interior = (R_PART - r) > SURFACE_BAND_H * h
    band = ~interior

    def three_ways(a_d, a_h):
        return {"all": M.rel_l2_pattern(a_d, a_h),
                "interior": M.rel_l2_pattern(a_d[interior], a_h[interior]),
                "surface_band": M.rel_l2_pattern(a_d[band], a_h[band])}

    # --- DIAGNOSTIC: is the heatr3d surface Q spike a cross-interface stencil
    # artifact? Recompute heatr3d's Q from its OWN saved V with a stencil that
    # never differences across the part boundary (Ez = 0: the case is z-invariant,
    # verified in the reference run).
    Ex, Ey = M.masked_grad_2d(z["V_mid"], part_mid, h)
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey))
    q_mg_full = 0.5 * ec.SIGMA_DOPED * np.clip(e2, 0.0, None)
    q_mg = q_mg_full[sel]
    e_mg = np.sqrt(np.clip(e2, 0.0, None))[sel]

    def cv(a):
        return float(np.nanstd(a) / np.nanmean(a))

    gauge_expected = 1.0 - 1.0 / n_ref            # FEM/heatr3d |E| ratio if pure gauge
    e_ratio = float(e_d.mean() / e_h.mean())
    return {
        "diagnostic_maskgrad": {
            "what": "heatr3d V re-post-processed with a stencil that never "
                    "crosses the part boundary (metrics.masked_grad_2d); "
                    "isolates compute_qrf_3d's whole-domain np.gradient from "
                    "the FV solve itself",
            "qrf_pattern_rel_l2_vs_fem": three_ways(q_d, q_mg),
            "emag_pattern_rel_l2_vs_fem": three_ways(e_d, e_mg),
            "q_maskgrad_p99_over_mean": float(np.percentile(q_mg, 99) / q_mg.mean()),
            "q_maskgrad_max_over_mean": float(q_mg.max() / q_mg.mean()),
            "emag_mean_ratio_fem_over_maskgrad": float(e_d.mean() / e_mg.mean()),
            "emag_maskgrad_gauge_residual":
                float((e_d.mean() / e_mg.mean()) / gauge_expected - 1.0),
        },
        "analytic_anchor": {
            "what": "an infinitely long cylinder of gamma_i in an unbounded "
                    "uniform transverse field E0 through gamma_e has a UNIFORM "
                    "interior field E_in = 2*gamma_e/(gamma_i+gamma_e) * E0",
            "E_in_over_E0": float(abs(2 * ec.gamma_of(ec.SIGMA_VIRGIN, ec.EPS_VIRGIN)
                                      / (ec.gamma_of(ec.SIGMA_DOPED, ec.EPS_DOPED)
                                         + ec.gamma_of(ec.SIGMA_VIRGIN, ec.EPS_VIRGIN)))),
            "E0_plate_v_per_m": ec.V_LO / ec.L_DOMAIN,
            "E_in_analytic_v_per_m": float(
                abs(2 * ec.gamma_of(ec.SIGMA_VIRGIN, ec.EPS_VIRGIN)
                    / (ec.gamma_of(ec.SIGMA_DOPED, ec.EPS_DOPED)
                       + ec.gamma_of(ec.SIGMA_VIRGIN, ec.EPS_VIRGIN)))
                * ec.V_LO / ec.L_DOMAIN),
            "E_in_fem_v_per_m": float(np.nanmean(e_d)),
            "caveat": "the 60 mm box with Neumann side walls is NOT unbounded "
                      "(R/L = 1/6), so this anchors the UNIFORMITY and the "
                      "order of magnitude, not the exact value",
        },
        "uniformity_cv": {
            "fem": cv(q_d), "heatr3d": cv(q_h), "heatr3d_maskgrad": cv(q_mg),
            "note": "an infinite cylinder in a uniform transverse field has a "
                    "UNIFORM interior field, so CV -> 0 is the physically "
                    "expected answer (finite 60 mm box + Neumann walls perturb "
                    "it slightly)"},
        "power_fraction_in_surface_band": {
            "fem": float(np.nansum(q_d[band]) / np.nansum(q_d)),
            "heatr3d": float(q_h[band].sum() / q_h.sum()),
            "heatr3d_maskgrad": float(q_mg[band].sum() / q_mg.sum()),
            "area_fraction": float(band.sum() / band.size)},
        "interior_mean_ratio_fem_over_heatr3d":
            float(np.nanmean(q_d[interior]) / q_h[interior].mean()),
        "ref_npz": ref_npz.name,
        "n_points_compared": int(pts.shape[0]),
        "n_points_interior": int(interior.sum()),
        "n_points_surface_band": int(band.sum()),
        "eval_missed_q": miss_q, "eval_missed_e": miss_e,
        "qrf_pattern_rel_l2": three_ways(q_d, q_h),
        "emag_pattern_rel_l2": three_ways(e_d, e_h),
        "qrf_abs_mean_ratio_fem_over_heatr3d": float(q_d.mean() / q_h.mean()),
        "emag_mean_ratio_fem_over_heatr3d": e_ratio,
        "emag_gauge_expected_ratio": gauge_expected,
        "emag_gauge_residual": float(e_ratio / gauge_expected - 1.0),
        "qrf_max_ratio_fem_over_heatr3d": float(np.nanmax(q_d) / np.nanmax(q_h)),
        "q_fem": {"mean": float(np.nanmean(q_d)), "max": float(np.nanmax(q_d)),
                  "p99_over_mean": float(np.nanpercentile(q_d, 99) / np.nanmean(q_d))},
        "q_heatr3d": {"mean": float(q_h.mean()), "max": float(q_h.max()),
                      "p99_over_mean": float(np.percentile(q_h, 99) / q_h.mean())},
        "_arrays": (pts, q_d, q_h, q_mg, e_d, e_h, e_mg, interior),
    }


def main() -> int:
    p = HERE / "results.json"
    res = json.loads(p.read_text())
    ref = res["task2_ref"]["runs"]

    out = {"engine": "dolfinx (P1 CG, conforming tets)",
           "scalar_path": ec.SCALAR_PATH,
           "shape": "extruded circle d=20mm, full height",
           "gate": {"qrf_pattern_rel_l2_fine_lt": 0.10,
                    "improves_or_holds_under_refinement": None},
           "levels": {}}

    labels = {"n64": "coarse", "n96": "fine"}
    for key, label in labels.items():
        if key not in ref or not ref[key].get("ok"):
            print(f"skip {key}: no heatr3d reference")
            continue
        target = int(ref[key]["n_voxels_in_part"])
        h = float(ref[key]["h_m"])
        print(f"[dolfinx] {label} target in-part nodes={target} (heatr3d h={h*1e3:.4f} mm)",
              flush=True)
        case = solve_case(target, lc0=h, verify_lu=(label == "coarse"))
        cmp = compare(case, HERE / ref[key]["npz"], int(ref[key]["n"]))
        pts, q_d, q_h, q_mg, e_d, e_h, e_mg, interior = cmp.pop("_arrays")
        info = case["info"]
        np.savez_compressed(HERE / f"d1_circle_{label}.npz",
                            pts=pts, q_fem=q_d, q_heatr3d=q_h,
                            q_heatr3d_maskgrad=q_mg,
                            e_fem=e_d, e_heatr3d=e_h, e_heatr3d_maskgrad=e_mg,
                            interior=interior)
        rec = {
            "matched_to": key,
            "target_nodes_in_part": target,
            "lc_part_m": info.lc_part, "lc_bed_m": info.lc_bed,
            "lc_over_voxel_h": info.lc_part / h,
            "n_nodes_in_part": info.n_nodes_in_part,
            "node_count_ratio_vs_voxels": info.n_nodes_in_part / target,
            "n_dofs_total": info.n_nodes_total,
            "n_cells_total": info.n_cells_total,
            "part_volume_m3": info.part_volume_m3,
            "part_volume_rel_err_vs_analytic":
                info.part_volume_m3 / (np.pi * R_PART ** 2 * ec.L_DOMAIN) - 1.0,
            "doped_volume_assembled_m3": case["v_part"],
            "match_history": case["hist"],
            "wall_mesh_s": case["wall_mesh_s"],
            "wall_solve_s": case["wall_solve_s"],
            "wall_post_s": case["wall_post_s"],
            "petsc_options": KSP_ITER,
            "direct_lu_cross_check": case["lu_check"],
            "peak_rss_gb": fu.peak_rss_gb(),
            "p_target_w": case["p_target"],
            "heatr3d_wall_eqs_s": ref[key]["wall_eqs_s"],
            "heatr3d_peak_rss_gb": ref[key]["peak_rss_gb"],
            "eqs_solve_speedup_vs_heatr3d":
                ref[key]["wall_eqs_s"] / case["wall_solve_s"],
            "comparison": cmp,
        }
        out["levels"][label] = rec
        print(json.dumps({k: v for k, v in rec.items() if k != "match_history"},
                         indent=1)[:2000], flush=True)
        prev = cmp

    lv = out["levels"]
    if "coarse" in lv and "fine" in lv:
        def pair(path):
            def get(d):
                for k in path:
                    d = d[k]
                return float(d)
            return get(lv["coarse"]["comparison"]), get(lv["fine"]["comparison"])

        variants = {
            # the plan's metric, verbatim: all in-part mid-plane points
            "as_written_all_points": ("qrf_pattern_rel_l2", "all"),
            # Task 0/1 caveat 2: the harmonic-vs-DG0 boundary layer, excluded
            "interior_only": ("qrf_pattern_rel_l2", "interior"),
            "surface_band_only": ("qrf_pattern_rel_l2", "surface_band"),
            # heatr3d re-post-processed WITHOUT the cross-interface stencil
            "maskgrad_all_points": ("diagnostic_maskgrad",
                                    "qrf_pattern_rel_l2_vs_fem", "all"),
            "maskgrad_interior": ("diagnostic_maskgrad",
                                  "qrf_pattern_rel_l2_vs_fem", "interior"),
        }
        g = out["gate"]
        for name, path in variants.items():
            c, f = pair(path)
            g[name] = {"coarse": c, "fine": f,
                       "improves_or_holds": bool(f <= c * 1.05),
                       "fine_under_10pct": bool(f < 0.10),
                       "gate_ok": bool(f < 0.10 and f <= c * 1.05)}
        g["gate_ok"] = g["as_written_all_points"]["gate_ok"]
        g["gate_ok_interior"] = g["interior_only"]["gate_ok"]
        g["gate_ok_maskgrad"] = g["maskgrad_all_points"]["gate_ok"]
    res["task2"] = out
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(out["gate"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
