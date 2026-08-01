"""solve3d Phase A Task 4: coupled-forward parity, dolfinx vs heatr3d.

RUNS IN THE SPIKE ENV (dolfinx):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_a_gate

PREREQUISITES (geo-prewarp venv, plan Tasks 1 and 4):
    ./.venv312/bin/python -m solve3d.cases --measure
    ./.venv312/bin/python -m solve3d.cases --task4-refs

Four arms: {extruded circle, extruded square} x {coupling defaults-OFF,
coupling ARMED}. The armed arm uses eqs_update_interval_s = 60 s and
sigma_temp_coeff_per_K = -0.002 /K so the in-march re-solve semantics are
actually exercised (spec/plan Task 4). Every number lands in
solve3d/results/phase_a_gate.json; the report quotes that file only.

LIKE-FOR-LIKE READING RULES (each one is a decision, stated once here):
  * t90 -- both engines stop at the first time the PART-MEAN melt fraction
    reaches 0.90. heatr3d averages over its in-part voxels; dolfinx averages
    with the nodal part-volume weights (sum_i V_i m_i == the exact part volume).
  * heating curve -- part-mean T(t), compared as RISE on the common time span.
    The dolfinx curve is sampled at the SAME interval as the heatr3d one for
    that arm (10 s off-arm, 60 s coupled-arm), so interpolation error enters
    both sides identically.
  * melt-onset std(T) -- read for BOTH engines at heatr3d's own in-part
    MID-PLANE voxel centres. The anchors are full-height extrusions, so the
    heatr3d field is exactly z-invariant and the mid-plane read equals its
    volumetric sigma_T (the ratio is recorded per arm as a check, not assumed).
  * n_eqs_solves -- exact integer equality. Semantics, not tolerance.

WHAT PARITY CANNOT ABSORB (recorded per arm, never silently averaged away):
the two engines do not discretize the SAME part. heatr3d's part is a voxel
staircase; the FEM part is the exact cylinder / prism. The absolute absorbed
power is power_density * V_part, so a part-volume difference is a real drive
difference. `part_volume_rel_diff` and `p_target_rel_diff` are reported for
every arm.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import forward, gates

RESULTS = Path(__file__).resolve().parent / "results"
GATE_JSON = RESULTS / "phase_a_gate.json"
REFS_JSON = RESULTS / "task4_heatr3d_refs.json"

COUPLED_INTERVAL_S = 60.0            # must match solve3d.cases
COUPLED_SIGMA_TEMP_COEFF = -0.002


def _refs() -> dict:
    if not REFS_JSON.exists():
        raise FileNotFoundError(
            f"{REFS_JSON} not found. Run "
            "`./.venv312/bin/python -m solve3d.cases --task4-refs` "
            "in the geo-prewarp venv first (plan Task 4).")
    return json.loads(REFS_JSON.read_text())


def _params_for(arm: str) -> forward.ForwardParams:
    if arm == "off":
        return forward.ForwardParams()
    return forward.ForwardParams(
        eqs_update_interval_s=COUPLED_INTERVAL_S,
        sigma_temp_coeff_per_K=COUPLED_SIGMA_TEMP_COEFF)


def run_arm(shape: str, arm: str, ref: dict) -> dict:
    """One dolfinx arm, matched to the heatr3d reference run it is scored on."""
    npz = np.load(RESULTS / ref["npz"])
    n = int(npz["n"])
    part = np.asarray(npz["part"])
    h = float(npz["h"])
    p = _params_for(arm)
    sample_dt = float(ref["sample_dt_s"])

    t0 = time.perf_counter()
    out = forward.run_forward(shape, int(part.sum()), h, p=p,
                              max_time_s=1500.0, phi_target=0.90,
                              sample_dt_s=sample_dt)
    wall_total = time.perf_counter() - t0
    if not out["reached"]:
        raise RuntimeError(f"{shape}/{arm}: dolfinx never reached melt onset")

    # ---- melt-onset std(T): both engines at heatr3d's mid-plane centres ---- #
    pts, sel = forward.midplane_part_points(npz)
    T_fn = forward.fem.Function(
        forward.functionspace(out["msh"], ("Lagrange", 1)))
    T_fn.x.array[:] = out["T_phi90"].astype(
        forward.dolfinx.default_scalar_type)
    T_fem, missed = forward.eval_at(T_fn, out["msh"], pts)
    k = n // 2
    T_h = np.asarray(npz["T_phi90"])[:, :, k][sel]
    sigma_fem = float(np.std(T_fem))
    sigma_h_mid = float(np.std(T_h))
    sigma_h_vol = float(ref["sigma_T_c"])

    # ---- DIAGNOSTIC: where does the melt-onset field actually disagree? ---- #
    # D1 established that the two engines agree in the part INTERIOR and
    # disagree in a one-to-two-voxel SURFACE BAND (Q pattern rel-L2 0.0255
    # interior vs 0.108 overall at this resolution). If the thermal parity gap
    # has the same signature, the cause is the known drive difference, not a
    # new defect in the march. Measured, not assumed.
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    half = forward.PART_DIAM_M / 2.0
    d_edge = (half - r) if shape == "circle" else \
        (half - np.maximum(np.abs(pts[:, 0]), np.abs(pts[:, 1])))
    interior = d_edge > 1.5 * h
    band = ~interior
    rise_f, rise_h = T_fem - p.preheat_c, T_h - p.preheat_c

    def _rl2(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    field_diag = {
        "n_interior": int(interior.sum()), "n_surface_band": int(band.sum()),
        "T_rise_rel_l2_all": _rl2(rise_f, rise_h),
        "T_rise_rel_l2_interior": _rl2(rise_f[interior], rise_h[interior]),
        "T_rise_rel_l2_surface_band": _rl2(rise_f[band], rise_h[band]),
        "mean_T_dolfinx_c": float(T_fem.mean()),
        "mean_T_heatr3d_c": float(T_h.mean()),
        "surface_minus_interior_dolfinx_c":
            float(T_fem[band].mean() - T_fem[interior].mean()),
        "surface_minus_interior_heatr3d_c":
            float(T_h[band].mean() - T_h[interior].mean()),
        "sigma_T_interior_dolfinx_c": float(np.std(T_fem[interior])),
        "sigma_T_interior_heatr3d_c": float(np.std(T_h[interior])),
        "sigma_T_interior_rel_diff":
            gates.rel_spread(np.std(T_fem[interior]), np.std(T_h[interior])),
    }

    curve = gates.curve_rel_l2(out["curve_t_s"], out["curve_part_mean_T_c"],
                               ref["curve_t_s"], ref["curve_part_mean_T_c"])
    v_part_h = float(ref["part_volume_m3"])
    v_part_f = float(out["part_volume_m3"])
    p_target_h = float(forward.ForwardParams().power_density_w_per_m3) * v_part_h
    return {
        "shape": shape, "arm": arm, "heatr3d_grid_n": n,
        "heatr3d_ref_npz": ref["npz"],
        # --- gate quantities ---------------------------------------------- #
        "t90_dolfinx_s": float(out["t90_s"]),
        "t90_heatr3d_s": float(ref["t90_s"]),
        "t90_rel_diff": gates.rel_spread(out["t90_s"], ref["t90_s"]),
        "curve_rel_l2": curve["rel_l2"],
        "curve_detail": curve,
        "field_diagnostic": field_diag,
        "sigma_T_dolfinx_midplane_c": sigma_fem,
        "sigma_T_heatr3d_midplane_c": sigma_h_mid,
        "sigma_T_rel_diff": gates.rel_spread(sigma_fem, sigma_h_mid),
        "n_eqs_solves_dolfinx": int(out["n_eqs_solves"]),
        "n_eqs_solves_heatr3d": int(ref.get("n_eqs_solves", 1)),
        "n_eqs_resolves_skipped_dolfinx": int(out["n_eqs_resolves_skipped"]),
        # --- context that parity cannot absorb ----------------------------- #
        "sigma_T_heatr3d_volumetric_c": sigma_h_vol,
        "heatr3d_midplane_over_volumetric": sigma_h_mid / sigma_h_vol,
        "part_volume_dolfinx_m3": v_part_f,
        "part_volume_heatr3d_m3": v_part_h,
        "part_volume_rel_diff": gates.rel_spread(v_part_f, v_part_h),
        "p_target_dolfinx_w": float(out["p_target_w"]),
        "p_target_heatr3d_w": p_target_h,
        "p_target_rel_diff": gates.rel_spread(out["p_target_w"], p_target_h),
        "power_renorm_residual": abs(out["p_now_w"] / out["p_target_w"] - 1.0),
        "n_points_compared": int(pts.shape[0]),
        "eval_missed": int(missed),
        # --- provenance / cost --------------------------------------------- #
        "energy_residual_frac": float(out["energy_residual_frac"]),
        "clamp_bound": bool(out["clamp_bound"]),
        "n_substeps_used": int(out["n_substeps_used"]),
        "cfl_violated": bool(out["cfl_violated"]),
        "dt_stable_s": float(out["dt_stable_s"]),
        "n_k_assemblies": int(out["n_k_assemblies"]),
        "n_dofs_total": int(out["n_dofs_total"]),
        "n_cells_total": int(out["n_cells_total"]),
        "n_nodes_in_part": int(out["n_nodes_in_part"]),
        "lc_part_m": float(out["lc_part_m"]),
        "heatr3d_voxels_in_part": int(ref["n_voxels_in_part"]),
        "wall_mesh_s": float(out["wall_mesh_s"]),
        "wall_eqs_s": float(out["wall_eqs_s"]),
        "wall_march_s": float(out["wall_march_s"]),
        "wall_total_s": wall_total,
        "heatr3d_wall_eqs_s": ref.get("wall_eqs_s"),
        "heatr3d_wall_march_s": ref.get("wall_march_s"),
        "heatr3d_wall_total_s": ref.get("wall_total_s"),
    }


def build(reuse: bool = True, only: tuple[str, ...] | None = None) -> dict:
    """Run every arm not already recorded and (re)write phase_a_gate.json."""
    tol_doc = gates.load_tolerances()
    refs = _refs()
    doc: dict
    if reuse and GATE_JSON.exists():
        doc = json.loads(GATE_JSON.read_text())
    else:
        doc = {"arms": {}}
    doc.update({
        "what": "Phase A Task 4 coupled-forward parity, dolfinx vs heatr3d, "
                "scored against the FROZEN Task-1 tolerances.",
        "engine_dolfinx": {"version": forward.DOLFINX_VERSION,
                           "scalar_path": forward.SCALAR_PATH},
        "tolerances": tol_doc["tolerances"],
        "tolerance_provenance": tol_doc["raw_spread"],
        "coupled_arm": {"eqs_update_interval_s": COUPLED_INTERVAL_S,
                        "sigma_temp_coeff_per_K": COUPLED_SIGMA_TEMP_COEFF,
                        "label": "EXPLORATORY: no validated nonzero "
                                 "coefficient exists in this repo"},
    })
    for shape in ("circle", "square"):
        for arm in ("off", "coupled"):
            name = f"{shape}_{arm}"
            if reuse and name in doc.get("arms", {}):
                continue
            if only is not None and name not in only:
                continue
            ref = refs["runs"].get(name)
            if ref is None:
                raise KeyError(f"heatr3d reference {name!r} missing from "
                               f"{REFS_JSON.name}")
            print(f"[phase-a] dolfinx {name} ...", flush=True)
            doc.setdefault("arms", {})[name] = run_arm(shape, arm, ref)
            gates.write_json(GATE_JSON.name, doc)
    # ---- verdict ---------------------------------------------------------- #
    tol = doc["tolerances"]
    verdict = {"arms_present": sorted(doc["arms"])}
    verdict["all_four_arms_present"] = len(doc["arms"]) == 4
    for name, a in doc["arms"].items():
        checks = {
            "t90": a["t90_rel_diff"] <= tol["t90_rel"],
            "curve": a["curve_rel_l2"] <= tol["curve_rel_l2"],
            "sigma_T": a["sigma_T_rel_diff"] <= tol["sigma_T_rel"],
            "n_eqs_solves": a["n_eqs_solves_dolfinx"] == a["n_eqs_solves_heatr3d"],
        }
        checks["arm_ok"] = all(checks.values())
        verdict[name] = checks
    verdict["gate_ok"] = (verdict["all_four_arms_present"]
                          and all(v["arm_ok"] for v in verdict.values()
                                  if isinstance(v, dict)))
    doc["verdict"] = verdict
    gates.write_json(GATE_JSON.name, doc)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="ignore cached arms and re-run all four")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these arm names (e.g. circle_off)")
    args = ap.parse_args()
    doc = build(reuse=not args.fresh,
                only=tuple(args.only) if args.only else None)
    print(json.dumps(doc["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
