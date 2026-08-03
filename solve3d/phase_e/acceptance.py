"""Phase E acceptance gates: mesh hold-out + smoothing robustness.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.acceptance --shape pyramid

Phase C rules, with bands MEASURED PER SHAPE in this campaign (registration
acceptance.bands_are_per_shape) rather than inherited from the circle/square
anchors -- these are different geometries and an inherited band would mean
nothing.

THE S2 HONESTY CONSTRAINT APPLIES HERE. Both shapes are cornered and the
pyramid has an apex. If the hold-out fails, that is a finding about solving on
singular geometry -- the delivered map is tuned to a mesh-dependent feature --
and NOT a campaign bug (registration honesty_constraint_from_s2).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import design_chain as dc
from solve3d.phase_e import geometry as geo, rescore as rs, run as R

RESULTS = Path(__file__).resolve().parent / "results"
TRUNC = 3.0


def transfer_map(src_c, src_v, src_map, dst_c, radius_m: float) -> dict:
    """Cross-mesh transfer by the SAME normalized convolution the filter uses
    (Phase C's rule): the delivered map is already a filtered field of this
    kernel, so re-evaluating it with the same kernel invents no structure.
    Dopant conservation is REPORTED -- it is what the Phase C inversion arm was
    dropped for."""
    from scipy.spatial import cKDTree
    sig = float(radius_m)
    tree = cKDTree(src_c)
    nb = tree.query_ball_point(dst_c, r=TRUNC * sig)
    out = np.empty(dst_c.shape[0], dtype=float)
    for i, idx in enumerate(nb):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            idx = np.array([tree.query(dst_c[i])[1]], dtype=np.int64)
        d2 = np.sum((src_c[idx] - dst_c[i]) ** 2, axis=1)
        w = src_v[idx] * np.exp(-d2 / (2.0 * sig * sig))
        out[i] = float(np.dot(w, src_map[idx]) / w.sum())
    return {"map": np.clip(out, 0.0, 1.0), "radius_m": sig,
            "method": "normalized-convolution re-evaluation (design filter kernel)"}


def _centroids(tc):
    import dolfinx
    return np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]


def run(shape: str, arm: str = "solve_filter_only") -> dict:
    pr = R.prereg()["acceptance"]
    doc = json.loads((RESULTS / f"phase_e_{shape}.json").read_text())
    solved = doc["arms"][arm]
    uni_c = doc["arms"]["uniform_baseline"]
    z = np.load(RESULTS / f"map_{shape}_{arm}.npz")
    s_src = np.asarray(z["s_map"], float)
    src_c, src_v = np.asarray(z["centroids"], float), np.asarray(z["volumes"], float)

    # ---------------- mesh hold-out ---------------- #
    t0 = time.perf_counter()
    tc_f = R.build_case(shape, lc_part=R.LC_FINE_M)
    dst_c = _centroids(tc_f)
    tr = transfer_map(src_c, src_v, s_src, dst_c, 1.0e-3)
    s_fine = tr.pop("map")
    dop_src = float(np.dot(s_src, src_v))
    dop_dst = float(np.dot(s_fine, tc_f.eqs.vol[tc_f.eqs.part]))
    tr.update({"total_dopant_src_m3": dop_src, "total_dopant_dst_m3": dop_dst,
               "total_dopant_rel_move": abs(dop_dst - dop_src) / dop_src})
    uni_f = R.score_arm(tc_f, np.ones(tc_f.eqs.part.size), shape,
                        "uniform_at_score_mesh")
    sol_f = R.score_arm(tc_f, s_fine, shape, f"{arm}_at_score_mesh")
    # shape-relative rescoring for both fine-mesh reads
    pts, shp, h, zs = rs.eval_points(shape)
    from solve3d import forward as fwd
    W = fwd.functionspace(tc_f.msh, ("Lagrange", 1))
    for rec, nm in ((uni_f, "uniform_at_score_mesh"), (sol_f, f"{arm}_at_score_mesh")):
        T = np.asarray(np.load(RESULTS / f"field_{shape}_{nm}.npz")["T_read"], float)
        Tf = fwd.fem.Function(W)
        Tf.x.array[:] = T.astype(fwd.dolfinx.default_scalar_type)
        Te, _m = fwd.eval_at(Tf, tc_f.msh, pts)
        rec.update(rs.metrics_from_T(shape, Te.reshape(shp), zs, h))
    wall_hold = time.perf_counter() - t0

    def J(r):
        return r["J_asymmetric"]

    band_J = 1.5 * abs(J(uni_f) - J(uni_c)) / abs(J(uni_c))
    move_J = abs(J(sol_f) - J(solved)) / abs(J(solved))
    checks = {"J_rel": {"measured": move_J, "band": band_J,
                        "pass": bool(move_J <= band_J),
                        "uniform_own_move": abs(J(uni_f) - J(uni_c)) / abs(J(uni_c))}}
    for k in ("iou_phi0p9", "iou_phi0p8", "in_part_phi0p9", "out_of_part_phi0p9",
              "front_ssd_mm"):
        u_move = abs(uni_f[k] - uni_c[k])
        s_move = abs(sol_f[k] - solved[k])
        checks[k] = {"measured": s_move, "band": 1.5 * u_move,
                     "pass": bool(s_move <= 1.5 * u_move),
                     "uniform_own_move": u_move}
    holdout = {"solve_lc_m": R.LC_PART_M, "score_lc_m": R.LC_FINE_M,
               "transfer": tr, "checks": checks,
               "pass": all(c["pass"] for c in checks.values()),
               "J_solved_coarse": J(solved), "J_solved_fine": J(sol_f),
               "J_uniform_coarse": J(uni_c), "J_uniform_fine": J(uni_f),
               "solved_still_beats_uniform_at_score_mesh":
                   bool(J(sol_f) < J(uni_f)),
               "margin_at_score_mesh": (J(uni_f) - J(sol_f)) / abs(J(uni_f)),
               "wall_s": wall_hold,
               "band_rule": pr["mesh_holdout"]["band_rule"],
               "scores": {"uniform_at_score_mesh": uni_f,
                          "solved_at_score_mesh": sol_f}}

    # ---------------- smoothing robustness ---------------- #
    tc_c = R.build_case(shape)
    tc_c.set_objective("asymmetric")
    r_sub = float(pr["smoothing"]["perturbation_radius_m"])
    blur = dc.DesignChain(src_c, src_v, r_sub, [0.0])
    s_blur = blur.filter_apply(s_src)
    rec_b = R.score_arm(tc_c, s_blur, shape, f"{arm}_blurred")
    dJ = abs(J(rec_b) - J(solved)) / abs(J(solved))
    tol = float(pr["smoothing"]["tolerance_rel_J"])
    smooth = {"perturbation_radius_m": r_sub, "filter_radius_m": 1.0e-3,
              "J_unblurred": J(solved), "J_blurred": J(rec_b),
              "rel_change": dJ, "tolerance": tol, "pass": bool(dJ <= tol),
              "scores": {"blurred": rec_b}}

    beats = J(solved) < J(uni_c)
    out = {"arm": arm, "shape": shape, "mesh_holdout": holdout,
           "smoothing_robustness": smooth,
           "beats_uniform_in_grid": bool(beats),
           "in_grid_margin_rel": (J(uni_c) - J(solved)) / abs(J(uni_c)),
           "solved_label": bool(beats and holdout["pass"] and smooth["pass"]),
           "solved_label_rule": pr["solved_label_rule"],
           "s2_honesty_note": R.prereg()["honesty_constraint_from_s2"]["statement"]}
    p = RESULTS / f"phase_e_gate_{shape}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    d = run(ap.parse_args().shape)
    print(json.dumps({"holdout": d["mesh_holdout"]["pass"],
                      "smoothing": d["smoothing_robustness"]["pass"],
                      "beats_uniform": d["beats_uniform_in_grid"],
                      "margin": d["in_grid_margin_rel"],
                      "SOLVED": d["solved_label"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
