"""Stage A: score the mirror-symmetrized Phase C cylinder map.

CAMPAIGN SCRIPT. No solve3d solver module is modified. It calls the EXISTING
Phase C machinery so the numbers are produced by the same code path as every
other arm:

  * phase_c_run.score_arm       -- the read rule (envelope argmin of the
                                   asymmetric objective) and every recorded
                                   scalar,
  * phase_c_run.run_acceptance  -- the mesh hold-out (transfer by the design
                                   filter's own kernel onto phase_a_mid) and
                                   the 0.5 mm smoothing-robustness gate.

The arm is recorded as an ADDITIONAL recorded-deviation arm; it never displaces
the pre-registered solve arm. The deviation is stated in the record: the
delivered map is the solved map projected onto the mirror-symmetric subspace
and re-filtered once, not a new solve.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
      heatr3d_d1_spike/env/bin/python \
      scripts/analysis/score_symmetrized_cylinder_map.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from solve3d import gates, phase_c_run as pcr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "solve3d" / "results"
ARM = "symmetrized_filter_only_asymmetric_scaled"
SRC_ARM = "solve_filter_only_asymmetric_scaled"
CKPT = ROOT / "scripts" / "analysis" / "score_symmetrized_checkpoint.json"


def _ckpt(stage: str, payload) -> None:
    doc = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    doc[stage] = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "payload": payload}
    CKPT.write_text(json.dumps(doc, indent=1, default=str))
    print(f"[ckpt] {stage}", flush=True)


def main() -> int:
    z = np.load(RESULTS / f"phase_c_map_{ARM}.npz")
    s_sym = np.asarray(z["s_map"], float)
    c_src = np.asarray(z["centroids"], float)

    tc = pcr.build_solve_case()
    import dolfinx
    mp = np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]
    dev = float(np.abs(mp - c_src).max())
    if dev > 1e-12:
        raise SystemExit(f"cell ordering does not match the stored map "
                         f"(max centroid deviation {dev:.3e} m)")
    _ckpt("mesh_match", {"max_centroid_dev_m": dev, "n_design": int(s_sym.size)})

    src = json.loads((RESULTS / "phase_c_solves.json").read_text())["arms"][SRC_ARM]
    rec = pcr.score_arm(tc, s_sym, ARM, extra={
        "status": "not_a_solve: projection of the solved map onto the "
                  "mirror-symmetric subspace",
        "objective_optimized": "asymmetric", "w_ratio": None, "beta": 0.0,
        "budget_gradient_evaluations": 0, "gradient_evaluations_used": 0,
        "forward_equivalents_spent": 1.0,
        "derived_from_arm": SRC_ARM,
        "construction": (
            "s_sym_i = mean over the 4-element group {identity, x-mirror, "
            "z-mirror, xz-mirror} of the source map at the nearest centroid "
            "to the reflected point, then the 1.0 mm pre-registered design "
            "filter applied once to clean the nearest-centroid matching "
            "artefacts, then clipped to [0, 1]."),
        "symmetry_group": ["identity", "x_mirror", "z_mirror", "xz_mirror"],
        "symmetry_group_justification": (
            "part INTERSECT chamber INTERSECT electrode-field INTERSECT "
            "objective INTERSECT convection-boundary symmetry. The cylinder "
            "axis is z (mesh_gmsh occ.addCylinder(0,0,-L/2, 0,0,L, half); "
            "in_part_predicate('circle') has no z condition) and the part "
            "spans the full chamber, so the part admits all three mirrors. "
            "The electrodes sit at y = +/- L/2 (forward.py l.274-278) and the "
            "deposited power goes as |E|^2, which is even in y, but they fix "
            "the field direction so NO rotation about the cylinder axis "
            "survives. Convection acts on the top face y = +L/2 ONLY "
            "(forward.py l.111, l.563-577), which BREAKS the y-mirror in the "
            "thermal field and therefore in the objective. The y-mirror is "
            "excluded: the map's odd-in-y content is physical, not residue."),
        "deviation_note": (
            "RECORDED DEVIATION. The continuum optimum of a group-invariant "
            "objective over a group-invariant admissible set lies in the "
            "group-invariant subspace, so any content of a solved map outside "
            "that subspace is discretization noise. The 12-evaluation solved "
            "map retains only 49.9 percent of its mid-height-slab variance "
            "under this projection (42.6 percent over the full part), so "
            "about half of its structure is mesh-frame fitting. This arm "
            "delivers the invariant part. It is an ADDITIONAL arm and does "
            "not displace the pre-registered solve arm."),
        "source_arm_J_asymmetric": src["J_asymmetric"],
    })
    p = RESULTS / "phase_c_solves.json"
    doc = json.loads(p.read_text())
    doc["arms"][ARM] = rec
    gates.write_json(p.name, doc)
    _ckpt("score_coarse", {k: v for k, v in rec.items()
                           if k in ("J_asymmetric", "J_symmetric",
                                    "argmin_asymmetric", "J_out_of_bounds",
                                    "J_in_bounds_deficit", "map_stats")})

    gate = pcr.run_acceptance(ARM)
    _ckpt("acceptance", {"in_grid_margin_rel": gate["in_grid_margin_rel"],
                         "holdout_pass": gate["mesh_holdout"]["pass"],
                         "smoothing_pass": gate["smoothing_robustness"]["pass"],
                         "solved_label": gate["solved_label"],
                         "uniform_at_score_mesh":
                             gate["mesh_holdout"]["uniform_at_score_mesh"],
                         "solved_at_score_mesh":
                             gate["mesh_holdout"]["solved_at_score_mesh"]})
    print(json.dumps({"in_grid_margin_rel": gate["in_grid_margin_rel"],
                      "solved_label": gate["solved_label"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
