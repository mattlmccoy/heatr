"""Phase A close-out compute (dolfinx side).

RUNS IN THE SPIKE ENV:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.closeout --refine
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.closeout --export-fields

--refine        STEP 1. The extruded-circle anchor (coupling OFF) at three
                dolfinx in-part element sizes, to measure dolfinx's OWN
                convergence spread of t90, the part-mean heating curve and
                sigma_T -- exactly as Task 1 did for heatr3d.

                LEVELS, and why. Task 1's heatr3d spread came from a 1.5x
                LINEAR refinement (n=64 -> n=96). To combine two spreads they
                must mean the same thing, so the primary dolfinx pair uses the
                SAME 1.5x linear ratio, with the finer member being the mesh
                Task 4 actually ran:
                  coarse -- 23040 in-part nodes (heatr3d n=64 unknown count)
                  mid    -- 77952 in-part nodes (heatr3d n=96; the Task-4 mesh)
                  fine   -- 175392 in-part nodes (1.31x linear finer than mid)
                `fine` satisfies the close-out requirement for at least one
                level finer than the Task-4 mesh and shows whether the
                coarse->mid spread is a convergence trend or an artifact. The
                reported dolfinx spread is the MAX of the two pair spreads --
                the conservative reading, stated rather than chosen quietly.

--export-fields STEP 2 input. Re-runs the four Task-4 arms ONLY to export their
                melt-onset field on the shared evaluation grid, and CHECKS that
                each re-run reproduces the recorded phase_a_gate.json numbers.
                `phase_a_gate.json` itself is never rewritten: the Task-4
                verdict stands as recorded, and the close-out is additive.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import forward, gates

RESULTS = Path(__file__).resolve().parent / "results"
REFINE_JSON = RESULTS / "dolfinx_refinement.json"
EXPORT_JSON = RESULTS / "field_export.json"
GATE_JSON = RESULTS / "phase_a_gate.json"
REFS_JSON = RESULTS / "task4_heatr3d_refs.json"

# heatr3d in-part voxel counts at the two certified grids, and the extra level.
LEVELS = (("coarse", 23040, 0.0009375),
          ("mid", 77952, 0.000625),
          ("fine", 175392, 0.0004767))

REPRO_RTOL = 1e-9                   # re-run determinism check (never array_equal)


def _midplane_reader(ref_npz_name: str):
    """The fixed point set every sigma_T read uses: heatr3d's in-part mid-plane
    voxel centres at n=96. Holding the point set fixed across dolfinx mesh
    levels is what makes the level-to-level spread a MESH effect and not a
    sampling effect."""
    npz = np.load(RESULTS / ref_npz_name)
    pts, sel = forward.midplane_part_points(npz)
    k = int(npz["n"]) // 2
    return npz, pts, sel, k


def _read_fields(out: dict, pts_mid: np.ndarray, tag: str) -> dict:
    """sigma_T on the fixed mid-plane point set + the melt-onset field on the
    shared evaluation grid (saved for the shape gate)."""
    W = forward.functionspace(out["msh"], ("Lagrange", 1))
    T_fn = forward.fem.Function(W)
    T_fn.x.array[:] = out["T_phi90"].astype(forward.dolfinx.default_scalar_type)
    T_mid, missed_mid = forward.eval_at(T_fn, out["msh"], pts_mid)
    pts_e, shp_e, h_e = gates.eval_grid_points()
    T_e, missed_e = forward.eval_at(T_fn, out["msh"], pts_e)
    T_e = T_e.reshape(shp_e)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / f"eval_dolfinx_{tag}.npz",
                        T=T_e, h_xy_m=h_e, z_m=np.asarray(gates.EVAL_Z_M),
                        t90_s=out["t90_s"], missed=missed_e)
    return {"sigma_T_midplane_c": float(np.std(T_mid)),
            "eval_missed_midplane": int(missed_mid),
            "eval_missed_grid": int(missed_e),
            "eval_npz": f"eval_dolfinx_{tag}.npz",
            # measured z-invariance of the exported field (extrusion premise)
            "eval_z_plane_mean_spread_c":
                float(np.nanmax(np.nanmean(T_e, axis=(1, 2)))
                      - np.nanmin(np.nanmean(T_e, axis=(1, 2))))}


def refine() -> dict:
    p = forward.ForwardParams()
    npz, pts_mid, sel, k = _midplane_reader("anchor_heatr3d_circle_n96.npz")
    ref = json.loads(REFS_JSON.read_text())["runs"]["circle_off"]
    doc = {"what": "dolfinx OWN mesh-refinement spread, extruded circle, "
                   "coupling OFF -- the dolfinx half of the cross-family band",
           "shape": "circle", "arm": "off",
           "sigma_T_read": "heatr3d n=96 in-part mid-plane voxel centres "
                           "(point set held FIXED across levels)",
           "levels": {}}
    if REFINE_JSON.exists():
        doc = json.loads(REFINE_JSON.read_text())
    for name, target, lc0 in LEVELS:
        if name in doc.get("levels", {}):
            continue
        print(f"[refine] dolfinx circle/off level={name} target={target}",
              flush=True)
        t0 = time.perf_counter()
        out = forward.run_forward("circle", target, lc0, p=p,
                                  max_time_s=1500.0, phi_target=0.90,
                                  sample_dt_s=float(ref["sample_dt_s"]))
        wall = time.perf_counter() - t0
        if not out["reached"]:
            raise RuntimeError(f"refine/{name}: never reached melt onset")
        rec = {
            "target_nodes_in_part": target, "lc0_m": lc0,
            "lc_part_m": float(out["lc_part_m"]),
            "n_nodes_in_part": int(out["n_nodes_in_part"]),
            "n_dofs_total": int(out["n_dofs_total"]),
            "n_cells_total": int(out["n_cells_total"]),
            "t90_s": float(out["t90_s"]),
            "part_volume_m3": float(out["part_volume_m3"]),
            "curve_t_s": list(map(float, out["curve_t_s"])),
            "curve_part_mean_T_c": list(map(float, out["curve_part_mean_T_c"])),
            "energy_residual_frac": float(out["energy_residual_frac"]),
            "clamp_bound": bool(out["clamp_bound"]),
            "n_substeps_used": int(out["n_substeps_used"]),
            "dt_stable_s": float(out["dt_stable_s"]),
            "wall_mesh_s": float(out["wall_mesh_s"]),
            "wall_eqs_s": float(out["wall_eqs_s"]),
            "wall_march_s": float(out["wall_march_s"]),
            "wall_total_s": wall,
        }
        rec.update(_read_fields(out, pts_mid, f"circle_off_{name}"))
        doc.setdefault("levels", {})[name] = rec
        gates.write_json(REFINE_JSON.name, doc)
    doc["spreads"] = _refine_spreads(doc["levels"])
    gates.write_json(REFINE_JSON.name, doc)
    print(json.dumps(doc["spreads"], indent=1))
    return doc


def _pair_spread(a: dict, b: dict) -> dict:
    """Spread of the coarser run `a` against the finer reference `b`."""
    c = gates.curve_rel_l2(a["curve_t_s"], a["curve_part_mean_T_c"],
                           b["curve_t_s"], b["curve_part_mean_T_c"])
    return {"t90_rel_spread": gates.rel_spread(a["t90_s"], b["t90_s"]),
            "curve_rel_l2_spread": c["rel_l2"],
            "sigma_T_rel_spread": gates.rel_spread(a["sigma_T_midplane_c"],
                                                   b["sigma_T_midplane_c"]),
            "curve_detail": c}


def _refine_spreads(levels: dict) -> dict:
    out = {
        "coarse_vs_mid": _pair_spread(levels["coarse"], levels["mid"]),
        "mid_vs_fine": _pair_spread(levels["mid"], levels["fine"]),
        "primary_pair": "coarse_vs_mid",
        "primary_pair_note":
            "coarse->mid is a 1.5x LINEAR refinement, the SAME ratio Task 1 "
            "used for heatr3d (n=64->n=96), so the two spreads are "
            "commensurable. mid->fine (1.31x) is the required level finer than "
            "the Task-4 mesh and is a convergence-trend check.",
        "reporting_rule": "dolfinx spread = MAX of the two pair spreads "
                          "(conservative; declared before computing)",
    }
    for k in ("t90_rel_spread", "curve_rel_l2_spread", "sigma_T_rel_spread"):
        out[k] = float(max(out["coarse_vs_mid"][k], out["mid_vs_fine"][k]))
    return out


def refine_level(shape: str, level: str, target: int, lc0: float,
                 ref_key: str) -> dict:
    """One extra mesh level for a shape that needs its OWN self-spread.

    Added because the shape-metric band cannot be borrowed across shapes: the
    circle has ZERO out-of-part (bed) melt on every grid of both engines, so a
    circle-derived bed-melt band is identically zero and cannot bound the
    square, which does spill at its corners. Measuring the square's own spread
    is the fix; widening the circle band would not be."""
    p = forward.ForwardParams()
    ref = json.loads(REFS_JSON.read_text())["runs"][ref_key]
    npz96, pts_mid, sel, k = _midplane_reader(ref["npz"])
    print(f"[refine] dolfinx {shape}/off level={level} target={target}",
          flush=True)
    t0 = time.perf_counter()
    out = forward.run_forward(shape, target, lc0, p=p, max_time_s=1500.0,
                              phi_target=0.90,
                              sample_dt_s=float(ref["sample_dt_s"]))
    if not out["reached"]:
        raise RuntimeError(f"refine_level {shape}/{level}: no melt onset")
    rec = {"shape": shape, "level": level, "target_nodes_in_part": target,
           "lc0_m": lc0, "lc_part_m": float(out["lc_part_m"]),
           "n_nodes_in_part": int(out["n_nodes_in_part"]),
           "n_cells_total": int(out["n_cells_total"]),
           "t90_s": float(out["t90_s"]),
           "energy_residual_frac": float(out["energy_residual_frac"]),
           "clamp_bound": bool(out["clamp_bound"]),
           "wall_total_s": time.perf_counter() - t0}
    rec.update(_read_fields(out, pts_mid, f"{shape}_off_{level}"))
    doc = json.loads(REFINE_JSON.read_text())
    doc.setdefault("extra_levels", {})[f"{shape}_{level}"] = rec
    gates.write_json(REFINE_JSON.name, doc)
    print(json.dumps({k2: v for k2, v in rec.items()
                      if k2 not in ("eval_npz",)}, indent=1))
    return rec


def export_fields() -> dict:
    """Re-run the four Task-4 arms to export their melt-onset field on the
    shared evaluation grid, checking each reproduces its recorded numbers."""
    gate = json.loads(GATE_JSON.read_text())
    refs = json.loads(REFS_JSON.read_text())["runs"]
    npz96, pts_mid, sel, k = _midplane_reader("anchor_heatr3d_circle_n96.npz")
    doc = {"what": "melt-onset fields for the four Task-4 arms, exported on "
                   "the shared evaluation grid; phase_a_gate.json is NOT "
                   "rewritten -- the recorded Task-4 verdict stands",
           "repro_rtol": REPRO_RTOL, "arms": {}}
    if EXPORT_JSON.exists():
        doc = json.loads(EXPORT_JSON.read_text())
    for name, arm_rec in gate["arms"].items():
        if name in doc.get("arms", {}):
            continue
        shape, arm = name.rsplit("_", 1)
        ref = refs[name]
        ref_npz = np.load(RESULTS / ref["npz"])
        print(f"[export] dolfinx {name}", flush=True)
        from solve3d.phase_a_gate import _params_for
        out = forward.run_forward(shape, int(np.asarray(ref_npz["part"]).sum()),
                                  float(ref_npz["h"]), p=_params_for(arm),
                                  max_time_s=1500.0, phi_target=0.90,
                                  sample_dt_s=float(ref["sample_dt_s"]))
        # the exported field must belong to the SAME run the gate recorded
        t90_rel = gates.rel_spread(out["t90_s"], arm_rec["t90_dolfinx_s"])
        pts_arm, sel_arm = forward.midplane_part_points(ref_npz)
        rec = _read_fields(out, pts_arm, name)
        sig_rel = gates.rel_spread(rec["sigma_T_midplane_c"],
                                   arm_rec["sigma_T_dolfinx_midplane_c"])
        rec.update({"shape": shape, "arm": arm,
                    "t90_s": float(out["t90_s"]),
                    "t90_rel_vs_recorded": t90_rel,
                    "sigma_T_rel_vs_recorded": sig_rel,
                    "n_eqs_solves": int(out["n_eqs_solves"]),
                    "reproduces_recorded_run":
                        bool(t90_rel <= REPRO_RTOL and sig_rel <= REPRO_RTOL)})
        doc.setdefault("arms", {})[name] = rec
        gates.write_json(EXPORT_JSON.name, doc)
    doc["all_reproduce"] = all(a["reproduces_recorded_run"]
                               for a in doc["arms"].values())
    gates.write_json(EXPORT_JSON.name, doc)
    print(json.dumps({k: {kk: v[kk] for kk in
                          ("t90_rel_vs_recorded", "sigma_T_rel_vs_recorded",
                           "reproduces_recorded_run")}
                      for k, v in doc["arms"].items()}, indent=1))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--export-fields", action="store_true")
    ap.add_argument("--refine-level", nargs=5,
                    metavar=("SHAPE", "LEVEL", "TARGET", "LC0", "REFKEY"))
    args = ap.parse_args()
    if args.refine:
        refine()
    if args.refine_level:
        sh, lv, tg, lc, rk = args.refine_level
        refine_level(sh, lv, int(tg), float(lc), rk)
    if args.export_fields:
        export_fields()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
