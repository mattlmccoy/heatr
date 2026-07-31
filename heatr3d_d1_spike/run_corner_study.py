"""D1 Task 3: corner behaviour of an extruded square under refinement.

RUNS IN THE SPIKE ENV:
    heatr3d_d1_spike/env/bin/python heatr3d_d1_spike/run_corner_study.py

Prerequisite: run_heatr3d_reference.py --shape square (geo-prewarp venv) has
written ref_heatr3d_square_n{64,96,128}.npz and results.json["task3_ref"]
(n=128 lands in "task3_ref_n128" so that hitting the documented EQS-01 memory
ceiling cannot destroy the n=64/96 records).

Physics being measured: a 90-degree material corner carries a genuine,
INTEGRABLE field singularity, so BOTH engines must grow max|Q| under
refinement. The plan's question is CONTROL, i.e. is the growth a clean power
law you can extrapolate and quote, or an erratic jump you cannot.

Two heatr3d Q variants are reported, because Task 2 established that
compute_qrf_3d's whole-domain np.gradient differences ACROSS the part boundary
and inflates every surface voxel:
  * "raw"      -- heatr3d exactly as it ships (what its published metrics use)
  * "maskgrad" -- the same V re-post-processed with a stencil confined to the
                  part (metrics.masked_grad_2d), which removes the
                  cross-interface artifact but keeps the real corner physics
Only "maskgrad" is a like-for-like comparison against the FEM corner growth.

Geometry: the FULL-HEIGHT square prism (zspan = L), so the case is exactly
z-invariant and the mid-plane slice carries every in-part statistic.
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
HALF = 0.010                      # 20 mm square
CORNER_BAND_M = 0.001             # "within 1 mm of a vertical corner edge"
CORNER_FIELD_RADIUS_M = 0.002     # locally refined region around each edge
KSP_ITER = {"ksp_type": "gmres", "pc_type": "gamg",
            "ksp_rtol": "1e-10", "ksp_max_it": "500"}


def in_square(mp: np.ndarray) -> np.ndarray:
    return (np.abs(mp[0]) <= HALF) & (np.abs(mp[1]) <= HALF)


# --------------------------------------------------------------------------- #
# heatr3d side (from the saved mid-plane; the prism is z-invariant)
# --------------------------------------------------------------------------- #
def heatr3d_metrics(npz_path: Path) -> dict:
    z = np.load(npz_path)
    x, y, h = z["x"], z["y"], float(z["h"])
    part, Q = z["part_mid"], z["Q_mid"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    d = M.corner_edge_distance(X.ravel(), Y.ravel(), HALF).reshape(X.shape)

    Ex, Ey = M.masked_grad_2d(z["V_mid"], part, h)
    e2 = np.real(Ex * np.conj(Ex) + Ey * np.conj(Ey))
    Q_mg = np.where(part, 0.5 * ec.SIGMA_DOPED * np.clip(e2, 0.0, None), 0.0)

    band = part & (d <= CORNER_BAND_M)
    bulk = part & (d > CORNER_BAND_M)
    out = {"h_m": h, "n_voxels_in_part": int(part.sum()),
           "n_voxels_in_band": int(band.sum())}
    for name, QQ in (("raw", Q), ("maskgrad", Q_mg)):
        q = QQ[part]
        w = np.ones_like(q)
        out[name] = {
            "mean": float(q.mean()),
            "corner_max": float(QQ[band].max()),
            "corner_max_over_mean": float(QQ[band].max() / q.mean()),
            "p99_over_mean": float(M.weighted_percentile(q, w, 99.0) / q.mean()),
            "bulk_p99_over_mean": float(
                M.weighted_percentile(QQ[bulk], np.ones(int(bulk.sum())), 99.0)
                / q.mean()),
            "bulk_max_over_mean": float(QQ[bulk].max() / q.mean()),
        }
    return out


# --------------------------------------------------------------------------- #
# dolfinx side
# --------------------------------------------------------------------------- #
def dolfinx_case(target_nodes: int, lc0: float, corner_lc: float | None,
                 label: str) -> dict:
    t0 = time.perf_counter()
    if corner_lc is None:
        msh, info, hist = mg.match_lc("square", target_nodes, lc0)
    else:
        msh, info = mg.build("square", lc0, corner_lc=corner_lc,
                             corner_radius_m=CORNER_FIELD_RADIUS_M)
        hist = [{"lc_part_m": lc0, "n_nodes_in_part": info.n_nodes_in_part,
                 "ratio_vs_target": info.n_nodes_in_part / target_nodes}]
    t_mesh = time.perf_counter() - t0

    mats = ec.materials(msh, in_part=in_square)
    t0 = time.perf_counter()
    Vr, Vi = ec.solve_eqs(msh, mats, petsc_options=KSP_ITER)
    t_solve = time.perf_counter() - t0
    q, scale, p_target = ec.qrf_dg0(msh, Vr, Vi, mats, premix=False)

    vol = fu.cell_volumes(msh, mats.dg0)
    mid = fu.cell_midpoints(msh)           # (ncell, 3)
    qa = np.real(q.x.array)
    inpart = np.real(mats.doped.x.array) > 0.5
    d = M.corner_edge_distance(mid[:, 0], mid[:, 1], HALF)
    band = inpart & (d <= CORNER_BAND_M)
    bulk = inpart & (d > CORNER_BAND_M)

    v_part = float(vol[inpart].sum())
    mean = float((qa[inpart] * vol[inpart]).sum() / v_part)
    h_corner = float((vol[band].mean()) ** (1.0 / 3.0)) if band.any() else float("nan")
    return {
        "label": label,
        "lc_part_m": info.lc_part, "corner_lc_m": corner_lc,
        "n_nodes_in_part": info.n_nodes_in_part,
        "node_count_ratio_vs_voxels": info.n_nodes_in_part / target_nodes,
        "n_dofs_total": info.n_nodes_total,
        "n_cells_total": info.n_cells_total,
        "n_cells_in_part": int(inpart.sum()),
        "n_cells_in_band": int(band.sum()),
        "part_volume_m3": v_part,
        "part_volume_rel_err": v_part / (4 * HALF ** 2 * ec.L_DOMAIN) - 1.0,
        "h_corner_m": h_corner,
        "h_corner_over_lc": h_corner / info.lc_part,
        "mean": mean,
        "corner_max": float(qa[band].max()),
        "corner_max_over_mean": float(qa[band].max() / mean),
        "p99_over_mean": float(
            M.weighted_percentile(qa[inpart], vol[inpart], 99.0) / mean),
        "bulk_p99_over_mean": float(
            M.weighted_percentile(qa[bulk], vol[bulk], 99.0) / mean),
        "bulk_max_over_mean": float(qa[bulk].max() / mean),
        "wall_mesh_s": t_mesh, "wall_solve_s": t_solve,
        "peak_rss_gb": fu.peak_rss_gb(),
        "match_history": hist,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    p = HERE / "results.json"
    res = json.loads(p.read_text())
    refs = dict(res.get("task3_ref", {}).get("runs", {}))
    extra = res.get("task3_ref_n128", {}).get("runs", {})
    refs.update(extra)

    out = {"engines": {"heatr3d": "voxel FV, EQS only",
                       "dolfinx": f"P1 CG conforming tets, {ec.SCALAR_PATH} scalars"},
           "shape": "extruded square 20 mm, full height (zspan = L)",
           "corner_band_m": CORNER_BAND_M,
           "petsc_options": KSP_ITER,
           "heatr3d": {}, "dolfinx": {}, "fits": {}, "gate": {}}

    # ---- heatr3d refinement sequence -------------------------------------
    for key in ("n64", "n96", "n128"):
        rec = refs.get(key)
        if rec is None:
            continue
        if not rec.get("ok"):
            out["heatr3d"][key] = {"ok": False,
                                   "failure_mode": rec.get("failure_mode"),
                                   "failure_message": rec.get("failure_message"),
                                   "wall_eqs_s": rec.get("wall_eqs_s"),
                                   "peak_rss_gb": rec.get("peak_rss_gb")}
            continue
        m = heatr3d_metrics(HERE / rec["npz"])
        m.update({"ok": True, "n": rec["n"], "wall_eqs_s": rec["wall_eqs_s"],
                  "peak_rss_gb": rec["peak_rss_gb"],
                  "n_unknowns": rec["n_unknowns"]})
        out["heatr3d"][key] = m
        print(f"[heatr3d] {key}: corner_max/mean raw="
              f"{m['raw']['corner_max_over_mean']:.3f} maskgrad="
              f"{m['maskgrad']['corner_max_over_mean']:.3f}", flush=True)

    # ---- dolfinx refinement sequence -------------------------------------
    plan = []
    for key in ("n64", "n96", "n128"):
        rec = refs.get(key)
        if rec is None or not rec.get("ok"):
            continue
        plan.append((f"uniform_{key}", int(rec["n_voxels_in_part"]),
                     float(rec["h_m"]), None))

    def append_corner_refined():
        """One locally corner-refined mesh whose BULK element size is exactly
        the matched size of the finest uniform level, so any change in the
        away-from-edge statistics is attributable to the corner band alone."""
        if not plan:
            return
        last = out["dolfinx"][plan[-1][0]]
        lcf = float(last["lc_part_m"])
        plan.append(("corner_refined", plan[-1][1], lcf, lcf / 4.0))

    i = 0
    while i < len(plan):
        label, tgt, lc0, clc = plan[i]
        i += 1
        print(f"[dolfinx] {label} lc0={lc0*1e3:.4f} mm corner_lc="
              f"{'-' if clc is None else f'{clc*1e3:.4f} mm'}", flush=True)
        rec = dolfinx_case(tgt, lc0, clc, label)
        out["dolfinx"][label] = rec
        print(f"    dofs={rec['n_dofs_total']} h_corner={rec['h_corner_m']*1e3:.4f} mm "
              f"corner_max/mean={rec['corner_max_over_mean']:.3f} "
              f"bulk_p99/mean={rec['bulk_p99_over_mean']:.3f} "
              f"solve={rec['wall_solve_s']:.1f}s", flush=True)
        if i == len(plan) and not any(l == "corner_refined" for l, *_ in plan):
            append_corner_refined()

    # ---- growth-law fits --------------------------------------------------
    def fit(hs, qs, name):
        hs, qs = np.asarray(hs, float), np.asarray(qs, float)
        if hs.size < 3:
            out["fits"][name] = {"n_points": int(hs.size),
                                 "note": "fewer than 3 refinement levels"}
            return None
        f = M.power_law_fit(hs, qs)
        f["h_m"] = hs.tolist()
        f["corner_max_over_mean"] = qs.tolist()
        out["fits"][name] = f
        return f

    hd = [r for r in out["dolfinx"].values()]
    uni = [r for r in hd if r["corner_lc_m"] is None]
    fit([r["h_corner_m"] for r in uni], [r["corner_max_over_mean"] for r in uni],
        "dolfinx_uniform")
    f_all = fit([r["h_corner_m"] for r in hd],
                [r["corner_max_over_mean"] for r in hd],
                "dolfinx_all_incl_corner_refined")
    hh = [r for r in out["heatr3d"].values() if r.get("ok")]
    fit([r["h_m"] for r in hh], [r["raw"]["corner_max_over_mean"] for r in hh],
        "heatr3d_raw")
    fit([r["h_m"] for r in hh], [r["maskgrad"]["corner_max_over_mean"] for r in hh],
        "heatr3d_maskgrad")

    g = out["gate"]
    g["criterion"] = "dolfinx growth-law fit R^2 > 0.98 across its refinements"
    for k, v in out["fits"].items():
        if "r2" in v:
            g[f"r2_{k}"] = v["r2"]
    prim = out["fits"].get("dolfinx_all_incl_corner_refined", {})
    g["dolfinx_r2"] = prim.get("r2")
    g["dolfinx_exponent"] = prim.get("exponent")
    g["gate_ok"] = bool(prim.get("r2", 0.0) > 0.98)
    # stability of the away-from-edge statistic (the plan's control criterion)
    bp = [r["bulk_p99_over_mean"] for r in hd]
    g["dolfinx_bulk_p99_over_mean_spread"] = (
        float(max(bp) - min(bp)) if bp else None)
    bph = [r["maskgrad"]["bulk_p99_over_mean"] for r in hh]
    g["heatr3d_maskgrad_bulk_p99_over_mean_spread"] = (
        float(max(bph) - min(bph)) if bph else None)
    bpr = [r["raw"]["bulk_p99_over_mean"] for r in hh]
    g["heatr3d_raw_bulk_p99_over_mean_spread"] = (
        float(max(bpr) - min(bpr)) if bpr else None)

    res["task3"] = out
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({"fits": out["fits"], "gate": g}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
